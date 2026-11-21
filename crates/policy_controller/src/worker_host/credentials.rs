//! The credentials a host keeps for the worker objects it builds.

use std::{
    collections::HashMap,
    error::Error,
    fmt, mem,
    sync::{
        atomic::{AtomicBool, Ordering::Relaxed},
        Arc,
    },
};

use parking_lot::RwLock;

use super::object::dial_address;
use crate::{engine::view::WorkerId, store::WorkerFacts};

/// The credentials of a host's workers, one entry per address, each bound
/// to the rows its registrations wrote.
///
/// The keeper lives in the host's process and is lost when the process
/// exits. The membership store, its log, snapshots, watches and operator
/// views carry no credential.
///
/// A registration makes three calls. It calls [`Self::keep`] before its
/// write, with the facts it writes and `None` where it has no credential.
/// It writes the row. Then it answers the token `keep` returned:
/// [`KeptCredential::accepted`] with the row the door answered where the
/// write was taken, whether or not it moved the row
/// ([`KeptCredential::accepted_rows`] with every row, where it states the
/// ranks of one engine), and
/// [`KeptCredential::put_back`] where it was refused. It answers at the
/// door's answer, with no await between the two. A token dropped with no
/// answer is put back. Where either answer returns `true`, the host has
/// the holder lay the address's objects again
/// ([`Objects::lay_again`](super::Objects::lay_again)).
///
/// One registration is in flight at an address. A keep at an address whose
/// last keep has no answer yet takes nothing and returns
/// [`RegistrationInFlight`]. The host waits, or tells its caller to ask
/// again.
///
/// A credential is bound to rows. An accepted registration binds the row
/// its door answered, and an object takes the address's credential only
/// where its row is bound: a row no registration here stated takes none.
/// So a credential never reaches a worker that came to the address later
/// through another host. That worker's row is another row, and its object
/// is built with no key, which its engine refuses in the open.
///
/// While a registration is in flight its row is not answered yet, so an
/// object built at the address takes what the registration stated. The
/// answer settles it: an acceptance binds the row, a put-back restores
/// what stood, and each returns `true` where an object may hold a
/// credential that is not its row's any more.
///
/// The holder forgets by itself. Each of its reads drops the rows that
/// left from the keeper, and an entry goes with its last row. The holder
/// reads the keeper's clock ahead of the store: a row bound behind that
/// read may have been written behind it, and stays. Once the clock is
/// past the last forgetting, the next read forgets again whatever it
/// finds: an answer may bind a row that left before it. A forgetting
/// that misses a row leaves a binding no later worker matches.
///
/// The address is the one the row's object dials. The keeper works it out
/// of the facts by the holder's own rule, so a host states no address.
#[derive(Default)]
pub struct Credentials {
    // Shared with the token of each registration in flight.
    held: Arc<RwLock<Held>>,
}

#[derive(Default)]
struct Held {
    entries: HashMap<String, Entry>,
    /// How many registrations were answered as taken: the keeper's clock.
    taken: u64,
}

/// What the keeper holds at one address.
struct Entry {
    /// What the object of a bound row takes now. While a registration is
    /// in flight it is what that registration stated, for any row.
    credential: Option<String>,
    /// The rows the accepted registrations at the address wrote.
    rows: Vec<Bound>,
    /// The registration in flight at the address.
    flight: Option<Flight>,
}

/// A row an accepted registration wrote.
struct Bound {
    row: WorkerId,
    /// The keeper's clock when its registration was answered as taken.
    taken_at: u64,
}

impl Entry {
    /// Whether the address holds a credential: the one an object takes now,
    /// or the one a registration in flight displaced.
    fn holds_credential(&self) -> bool {
        self.credential.is_some()
            || self
                .flight
                .as_ref()
                .and_then(|flight| flight.stood.as_ref())
                .is_some_and(|stood| stood.credential.is_some())
    }

    /// Whether an accepted registration bound `row` here.
    fn binds(&self, row: WorkerId) -> bool {
        self.rows.iter().any(|bound| bound.row == row)
    }
}

/// A registration whose write has no answer yet.
struct Flight {
    /// What its keep displaced: `None` where the address held no entry.
    stood: Option<Stood>,
    /// Whether an object was built with the credential it stated. The
    /// holder marks the flight while it reads the keeper.
    built: AtomicBool,
}

impl Flight {
    /// A flight no object was built in yet, over what its keep displaced.
    fn over(stood: Option<Stood>) -> Self {
        Self {
            stood,
            built: AtomicBool::new(false),
        }
    }
}

/// The credential of an address's entry as a registration found it.
struct Stood {
    credential: Option<String>,
}

/// A registration in flight at an address: what [`Credentials::keep`]
/// returns. The registration answers it once its write is answered, with
/// [`Self::accepted`] or [`Self::put_back`]. Dropped with no answer, it is
/// put back: a registration that does not say its write was taken changes
/// nothing the keeper held.
#[must_use = "a keep with no answer is put back: call `accepted` where the write was taken"]
pub struct KeptCredential {
    held: Arc<RwLock<Held>>,
    address: String,
    in_flight: bool,
    /// Whether the stated credential is another than the one that stood.
    differs: bool,
}

/// A keep found a registration in flight at its address and took nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegistrationInFlight;

impl fmt::Display for RegistrationInFlight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a registration is in flight at the address")
    }
}

impl Error for RegistrationInFlight {}

// No field: an address as a registration writes it can carry a credential
// in its userinfo.
impl fmt::Debug for KeptCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeptCredential").finish_non_exhaustive()
    }
}

impl KeptCredential {
    /// The write was taken: the stated credential stands at the address,
    /// bound to `row`, the row the door answered (the `new` row of a
    /// [`MembershipWriteOutcome::Another`](super::MembershipWriteOutcome::Another)),
    /// beside the rows bound there before. Call it whether or not the write
    /// moved the row: a worker stated again binds its row on a host that had
    /// not bound it. Call it at the door's answer, with no await between the
    /// two: a token dropped there puts back a credential the row's object was
    /// built with.
    ///
    /// Return `true` where an object at the address may hold another
    /// credential than its row takes now: the credential changed, the row
    /// was not bound here before, or an object was built while the
    /// registration was in flight. Have the holder lay the address's
    /// objects again then
    /// ([`Objects::lay_again`](super::Objects::lay_again)).
    pub fn accepted(mut self, row: WorkerId) -> bool {
        self.end(Some(&[row]))
    }

    /// The write of several rows at the one address was taken:
    /// [`Self::accepted`] for a registration that states the ranks of one
    /// engine, with every row the door answered. It is one flight, so its
    /// one answer binds them all.
    pub fn accepted_rows(mut self, rows: &[WorkerId]) -> bool {
        self.end(Some(rows))
    }

    /// The write was refused: restore what stood at the address when the
    /// registration kept. Return `true` if an object was built with the
    /// refused credential meanwhile. Have the holder lay the address's
    /// objects again then
    /// ([`Objects::lay_again`](super::Objects::lay_again)): each takes
    /// what stood for its row.
    pub fn put_back(mut self) -> bool {
        self.end(None)
    }

    /// End the flight, once: accepted with the rows the door answered, or
    /// put back. Return whether the address is to be laid again.
    fn end(&mut self, accepted: Option<&[WorkerId]>) -> bool {
        if !mem::take(&mut self.in_flight) {
            return false;
        }
        let mut held = self.held.write();
        let Held { entries, taken } = &mut *held;
        // The keeper ends no flight itself, so the entry holds this one.
        let Some((entry, flight)) = entries.get_mut(&self.address).and_then(|entry| {
            let flight = entry.flight.take()?;
            Some((entry, flight))
        }) else {
            return false;
        };
        let built = flight.built.into_inner();
        if let Some(rows) = accepted {
            *taken += 1;
            let mut again = built || self.differs;
            for &row in rows {
                match entry.rows.iter_mut().find(|bound| bound.row == row) {
                    Some(bound) => bound.taken_at = *taken,
                    None => {
                        entry.rows.push(Bound {
                            row,
                            taken_at: *taken,
                        });
                        // A row not bound here before: its object took
                        // no credential.
                        again = true;
                    }
                }
            }
            return again;
        }
        if let Some(stood) = flight.stood {
            entry.credential = stood.credential;
        }
        // No row is bound where no registration was accepted, and where
        // the rows one bound left meanwhile: nothing takes what stood.
        if entry.rows.is_empty() {
            entries.remove(&self.address);
        }
        built
    }
}

impl Drop for KeptCredential {
    fn drop(&mut self) {
        self.end(None);
    }
}

impl fmt::Debug for Credentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Credentials")
            .field("held", &self.len())
            .finish()
    }
}

impl Credentials {
    /// A keeper with no credentials.
    pub fn new() -> Self {
        Self::default()
    }

    /// State the credential of the registration that writes `facts`,
    /// before its write. Pass `None` for a registration without one. The
    /// keeper holds it at the address the row's object dials. An object
    /// built there while the registration is in flight takes it. Where the
    /// address's last keep has no answer yet, take nothing and return
    /// [`RegistrationInFlight`].
    pub fn keep(
        &self,
        facts: &WorkerFacts,
        credential: Option<String>,
    ) -> Result<KeptCredential, RegistrationInFlight> {
        let address = dial_address(facts);
        let mut held = self.held.write();
        let differs = match held.entries.get_mut(address) {
            Some(entry) if entry.flight.is_some() => return Err(RegistrationInFlight),
            Some(entry) => {
                let stood = Stood {
                    credential: mem::replace(&mut entry.credential, credential),
                };
                let differs = stood.credential != entry.credential;
                entry.flight = Some(Flight::over(Some(stood)));
                differs
            }
            None => {
                let differs = credential.is_some();
                let entry = Entry {
                    credential,
                    rows: Vec::new(),
                    flight: Some(Flight::over(None)),
                };
                held.entries.insert(address.to_owned(), entry);
                differs
            }
        };
        Ok(KeptCredential {
            held: Arc::clone(&self.held),
            address: address.to_owned(),
            in_flight: true,
            differs,
        })
    }

    /// How many addresses hold a credential. An address counts once where
    /// a registration in flight displaced one.
    pub fn len(&self) -> usize {
        self.held
            .read()
            .entries
            .values()
            .filter(|entry| entry.holds_credential())
            .count()
    }

    /// Whether the keeper holds no credentials.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Give the object of `row`, built at `address`, its credential. A
    /// row an accepted registration bound takes the address's. While a
    /// registration is in flight there, any row takes what it stated, and
    /// the flight is marked as one an object was built in. Any other row
    /// takes none.
    pub(super) fn of(&self, address: &str, row: WorkerId) -> Option<String> {
        let held = self.held.read();
        let entry = held.entries.get(address)?;
        match &entry.flight {
            Some(flight) => flight.built.store(true, Relaxed),
            None if !entry.binds(row) => return None,
            None => {}
        }
        entry.credential.clone()
    }

    /// The keeper's clock: how many registrations were answered as taken.
    /// The holder reads it ahead of its read of the store.
    pub(super) fn taken(&self) -> u64 {
        self.held.read().taken
    }

    /// Forget the rows that left. The holder calls this behind a read of
    /// the store, with the clock it read ahead of that read and the rows
    /// that stand in it. A bound row that does not stand and was bound no
    /// later than `read` is dropped: it left the store the holder read.
    /// One bound later stays: its row may have been written behind that
    /// read. An entry with no row and no registration in flight goes.
    /// The holder calls again behind its next read once the clock is past
    /// `read`: a registration answered since bound a row this call did
    /// not ask about.
    pub(super) fn forget_left(&self, read: u64, stands: impl Fn(WorkerId) -> bool) {
        let mut held = self.held.write();
        held.entries.retain(|_, entry| {
            entry
                .rows
                .retain(|bound| stands(bound.row) || bound.taken_at > read);
            !entry.rows.is_empty() || entry.flight.is_some()
        });
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use super::{Credentials, KeptCredential, RegistrationInFlight};
    use crate::{engine::view::WorkerId, store::WorkerFacts};

    const ADDRESS: &str = "http://a:8000";
    const OTHER_ADDRESS: &str = "http://b:8000";
    const KEY_A: &str = "key-a";
    const KEY_B: &str = "key-b";
    const KEY_C: &str = "key-c";
    const ROW: WorkerId = WorkerId(1);
    const OTHER_ROW: WorkerId = WorkerId(2);

    /// The facts of a row that dials `address`.
    fn facts(address: &str) -> WorkerFacts {
        WorkerFacts {
            key: address.to_owned(),
            ..WorkerFacts::default()
        }
    }

    /// A registration's keep at an address where none is in flight.
    fn keep(credentials: &Credentials, address: &str, credential: Option<&str>) -> KeptCredential {
        credentials
            .keep(&facts(address), credential.map(str::to_owned))
            .expect("no registration in flight at the address")
    }

    /// A registration whose write was taken, with the row its door
    /// answered. Return whether the address is to be laid again.
    fn register(
        credentials: &Credentials,
        address: &str,
        row: WorkerId,
        credential: Option<&str>,
    ) -> bool {
        keep(credentials, address, credential).accepted(row)
    }

    /// Whether the object of `row`, built at the address now, takes
    /// `credential`. The read marks a flight, as a build does. No failure
    /// prints a credential.
    fn builds_with(
        credentials: &Credentials,
        address: &str,
        row: WorkerId,
        credential: Option<&str>,
    ) -> bool {
        credentials.of(address, row).as_deref() == credential
    }

    /// A forgetting whose read of the store begins now and finds no row.
    fn forget(credentials: &Credentials) {
        credentials.forget_left(credentials.taken(), |_| false);
    }

    fn has_entry(credentials: &Credentials, address: &str) -> bool {
        credentials.held.read().entries.contains_key(address)
    }

    /// A registration of several rows at one address is one flight. Its
    /// acceptance binds every row its door answered, and the object of
    /// each takes the credential.
    #[test]
    fn an_acceptance_of_several_rows_binds_each_of_them() {
        let credentials = Credentials::new();
        let rows = [ROW, OTHER_ROW];
        assert!(keep(&credentials, ADDRESS, Some(KEY_A)).accepted_rows(&rows));
        for row in rows {
            assert!(builds_with(&credentials, ADDRESS, row, Some(KEY_A)));
        }
        // Stated again with no change, nothing is to be laid again.
        assert!(!keep(&credentials, ADDRESS, Some(KEY_A)).accepted_rows(&rows));
        // A row beside them that no registration bound takes nothing.
        assert!(builds_with(&credentials, ADDRESS, WorkerId(3), None));
    }

    #[test]
    fn neither_the_keeper_nor_a_token_prints_an_address_or_a_credential() {
        let credentials = Credentials::new();
        // An address as a registration writes it can carry a userinfo.
        let token = keep(&credentials, "http://pebble:cedar@a:8000", Some(KEY_A));
        for text in [
            format!("{token:?}"),
            format!("{token:#?}"),
            format!("{credentials:?}"),
            format!("{credentials:#?}"),
        ] {
            for secret in ["pebble", "cedar", KEY_A] {
                assert!(!text.contains(secret), "{text}");
            }
        }
        assert!(!token.put_back());
    }

    #[test]
    fn the_keeper_holds_a_credential_at_the_address_the_rows_object_dials() {
        let credentials = Credentials::new();
        // A row that names an engine dials it. One that names none dials
        // its key.
        let named = WorkerFacts {
            engine: ADDRESS.to_owned(),
            ..facts("http://a:8000@0")
        };
        let token = credentials
            .keep(&named, Some(KEY_A.to_owned()))
            .expect("no registration in flight at the address");
        assert!(token.accepted(ROW));
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
        assert!(!has_entry(&credentials, "http://a:8000@0"));
    }

    #[test]
    fn an_acceptance_says_whether_the_address_is_to_be_laid_again() {
        let credentials = Credentials::new();
        // A row bound for the first time, with or without a credential.
        assert!(register(&credentials, ADDRESS, ROW, None));
        assert!(register(
            &credentials,
            OTHER_ADDRESS,
            OTHER_ROW,
            Some(KEY_A)
        ));

        // The same row again: only where the credential changed.
        assert!(!register(&credentials, ADDRESS, ROW, None));
        assert!(register(&credentials, ADDRESS, ROW, Some(KEY_A)));
        assert!(!register(&credentials, ADDRESS, ROW, Some(KEY_A)));
        assert!(register(&credentials, ADDRESS, ROW, Some(KEY_B)));
        assert!(register(&credentials, ADDRESS, ROW, None));

        // Another row of the address is bound beside the first.
        assert!(register(&credentials, ADDRESS, OTHER_ROW, None));
        assert!(!register(&credentials, ADDRESS, ROW, None));

        // An object built while the registration was in flight.
        let token = keep(&credentials, ADDRESS, None);
        assert!(builds_with(&credentials, ADDRESS, ROW, None));
        assert!(token.accepted(ROW));

        // A put-back restores what stood: the next keep is held against it.
        assert!(register(&credentials, ADDRESS, ROW, Some(KEY_B)));
        assert!(!keep(&credentials, ADDRESS, None).put_back());
        assert!(!register(&credentials, ADDRESS, ROW, Some(KEY_B)));
    }

    #[test]
    fn a_row_no_registration_here_stated_takes_no_credential() {
        let credentials = Credentials::new();
        assert!(register(&credentials, ADDRESS, ROW, Some(KEY_A)));
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
        assert!(builds_with(&credentials, ADDRESS, OTHER_ROW, None));
        assert!(builds_with(&credentials, OTHER_ADDRESS, ROW, None));

        // A registration in flight states its credential for the address:
        // its row is not answered yet. Its answer binds its row alone.
        let flight = keep(&credentials, ADDRESS, Some(KEY_B));
        assert!(builds_with(&credentials, ADDRESS, OTHER_ROW, Some(KEY_B)));
        assert!(flight.accepted(ROW));
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_B)));
        assert!(builds_with(&credentials, ADDRESS, OTHER_ROW, None));
    }

    #[test]
    fn a_registration_taken_leaves_its_credential_standing() {
        let credentials = Credentials::new();
        assert!(credentials.is_empty());
        register(&credentials, ADDRESS, ROW, Some(KEY_A));
        assert_eq!(credentials.len(), 1);
        assert!(!credentials.is_empty());

        register(&credentials, ADDRESS, ROW, Some(KEY_B));
        register(&credentials, OTHER_ADDRESS, OTHER_ROW, None);
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_B)));
        assert_eq!(credentials.len(), 1);

        register(&credentials, ADDRESS, ROW, None);
        assert!(credentials.is_empty());
        assert!(builds_with(&credentials, ADDRESS, ROW, None));
        forget(&credentials);
        assert!(!has_entry(&credentials, ADDRESS));
        assert!(!has_entry(&credentials, OTHER_ADDRESS));
    }

    #[test]
    fn the_keeper_prints_how_many_it_keeps_and_no_credential() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, ROW, Some(KEY_A));
        let shown = format!("{credentials:?}");
        assert!(!shown.contains(KEY_A));
        assert_eq!(shown, "Credentials { held: 1 }");
    }

    #[test]
    fn a_keep_at_an_address_in_flight_takes_nothing() {
        let credentials = Credentials::new();
        let first = keep(&credentials, ADDRESS, Some(KEY_A));
        let second = credentials.keep(&facts(ADDRESS), Some(KEY_B.to_owned()));
        assert_eq!(second.err(), Some(RegistrationInFlight));
        assert_eq!(
            RegistrationInFlight.to_string(),
            "a registration is in flight at the address"
        );
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
        // Another address is its own turn.
        register(&credentials, OTHER_ADDRESS, OTHER_ROW, Some(KEY_C));

        assert!(first.accepted(ROW));
        register(&credentials, ADDRESS, ROW, Some(KEY_B));
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_B)));
    }

    #[test]
    fn two_refused_registrations_at_once_leave_what_stood() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, ROW, Some(KEY_A));
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));

        let first = keep(&credentials, ADDRESS, Some(KEY_B));
        let second = credentials.keep(&facts(ADDRESS), Some(KEY_C.to_owned()));
        assert!(second.is_err());
        assert!(!first.put_back());
        // The second asks again and is refused too.
        assert!(!keep(&credentials, ADDRESS, Some(KEY_C)).put_back());
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
        assert_eq!(credentials.len(), 1);
    }

    #[test]
    fn a_put_back_says_that_an_object_was_built_with_the_refused_credential() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, ROW, Some(KEY_A));

        let refused = keep(&credentials, ADDRESS, Some(KEY_B));
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_B)));
        assert!(refused.put_back());
        // The next flight starts with no object built in it.
        assert!(!keep(&credentials, ADDRESS, Some(KEY_C)).put_back());
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
    }

    #[test]
    fn a_put_back_of_a_first_registration_leaves_no_entry() {
        let credentials = Credentials::new();
        assert!(!keep(&credentials, ADDRESS, Some(KEY_A)).put_back());
        assert!(!has_entry(&credentials, ADDRESS));

        let refused = keep(&credentials, ADDRESS, Some(KEY_A));
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
        assert!(refused.put_back());
        assert!(!has_entry(&credentials, ADDRESS));
        assert!(credentials.is_empty());
    }

    #[test]
    fn a_registration_that_states_none_can_be_put_back() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, ROW, Some(KEY_A));

        let refused = keep(&credentials, ADDRESS, None);
        // What it displaced is still held.
        assert_eq!(credentials.len(), 1);
        assert!(builds_with(&credentials, ADDRESS, ROW, None));
        assert!(refused.put_back());
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
    }

    #[test]
    fn a_token_dropped_with_no_answer_is_put_back() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, ROW, Some(KEY_A));

        drop(keep(&credentials, ADDRESS, Some(KEY_B)));
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));
        // The address takes the next registration.
        register(&credentials, ADDRESS, ROW, Some(KEY_B));
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_B)));
    }

    #[test]
    fn a_forgetting_drops_the_rows_that_left_and_an_entry_with_its_last() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, ROW, Some(KEY_A));
        register(&credentials, ADDRESS, OTHER_ROW, Some(KEY_A));
        let read = credentials.taken();

        // One row left: the other keeps the address's credential.
        credentials.forget_left(read, |row| row == OTHER_ROW);
        assert!(builds_with(&credentials, ADDRESS, ROW, None));
        assert!(builds_with(&credentials, ADDRESS, OTHER_ROW, Some(KEY_A)));
        assert_eq!(credentials.len(), 1);

        // The last row left: the entry goes.
        credentials.forget_left(read, |_| false);
        assert!(!has_entry(&credentials, ADDRESS));
        assert!(credentials.is_empty());
    }

    #[test]
    fn a_forgetting_keeps_a_row_bound_behind_its_read() {
        let credentials = Credentials::new();
        let read = credentials.taken();
        // Its row may have been written behind the holder's read.
        register(&credentials, ADDRESS, ROW, Some(KEY_A));
        credentials.forget_left(read, |_| false);
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));

        // A put-back leaves the binding as it was taken.
        assert!(!keep(&credentials, ADDRESS, Some(KEY_B)).put_back());
        credentials.forget_left(read, |_| false);
        assert!(builds_with(&credentials, ADDRESS, ROW, Some(KEY_A)));

        // A read that begins behind the answer forgets it.
        forget(&credentials);
        assert!(!has_entry(&credentials, ADDRESS));
    }

    #[test]
    fn a_forgetting_in_flight_drops_the_rows_and_keeps_what_was_stated() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, ROW, Some(KEY_A));

        // The row left while the next registration is in flight.
        let next = keep(&credentials, ADDRESS, Some(KEY_B));
        forget(&credentials);
        assert_eq!(credentials.len(), 1);
        assert!(next.accepted(OTHER_ROW));
        assert!(builds_with(&credentials, ADDRESS, OTHER_ROW, Some(KEY_B)));
        assert!(builds_with(&credentials, ADDRESS, ROW, None));

        // Refused instead: no row is bound any more, so nothing is left.
        let refused = keep(&credentials, ADDRESS, Some(KEY_C));
        forget(&credentials);
        assert!(!refused.put_back());
        assert!(!has_entry(&credentials, ADDRESS));
    }

    #[test]
    fn a_build_reads_beside_another_reader_of_the_keeper() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, ROW, Some(KEY_A));

        let keeper = &credentials;
        let held = keeper.held.read();
        let (send, receive) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(move || {
                let _ = send.send(keeper.of(ADDRESS, ROW).as_deref() == Some(KEY_A));
            });
            let read = receive.recv_timeout(Duration::from_secs(5));
            drop(held);
            assert_eq!(read, Ok(true));
        });
    }

    #[test]
    fn the_token_of_a_keep_prints_no_credential() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, ROW, Some(KEY_A));
        let kept = keep(&credentials, ADDRESS, Some(KEY_B));

        let shown = format!("{kept:?}");
        assert!(!shown.contains(KEY_A));
        assert!(!shown.contains(KEY_B));
        assert_eq!(shown, "KeptCredential { .. }");
        assert!(kept.accepted(ROW));
    }
}
