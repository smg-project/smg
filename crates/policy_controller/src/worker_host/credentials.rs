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

/// The credentials of a host's workers, one entry per address.
///
/// The keeper lives in the host's process and is lost when the process
/// exits. The membership store, its log, snapshots, watches and operator
/// views carry no credential. It holds one entry for each address a
/// registration was accepted at and no forgetting dropped, with or
/// without a credential: as many as the host's rows dial, where each
/// removal is followed by
/// [`Objects::forget_credential`](super::Objects::forget_credential).
///
/// A registration makes three calls. It calls [`Self::keep`] before its
/// write, with `None` where it has no credential. It writes the row. Then
/// it answers the token `keep` returned: [`KeptCredential::accepted`] where
/// the write moved the row or the credential alone changed,
/// [`KeptCredential::put_back`] where it was refused or changed nothing.
/// It answers at the door's answer, with no await between the two. A token
/// dropped with no answer is put back.
///
/// One registration is in flight at an address. A keep at an address whose
/// last keep has no answer yet takes nothing and returns
/// [`RegistrationInFlight`]. The host waits, or tells its caller to ask
/// again.
///
/// An object built at the address while a registration is in flight takes
/// the credential it stated. A write that moves the row's revision makes
/// the holder build the object with it. A write that moves nothing builds
/// nothing: a door answers [`Wrote::Unmoved`](super::Wrote::Unmoved) for
/// equal facts. Where the token says the stated credential differs
/// ([`KeptCredential::differs`]), accept it and have the holder lay the
/// address's objects again
/// ([`Objects::lay_again`](super::Objects::lay_again)). Where it does not,
/// put the keep back: nothing changed.
///
/// A put-back restores what stood at the address when the registration
/// kept. It says whether an object was built with the refused credential
/// meanwhile. Have the holder lay the address's objects again then: each
/// takes what stood.
///
/// Write the removal before calling
/// [`Objects::forget_credential`](super::Objects::forget_credential). The
/// holder reads the store first and forgets an entry once no row dials the
/// address. A registration answered behind the start of that read may have
/// written behind it: its entry stays for the next call, and so does the
/// credential of a registration in flight.
///
/// Use the address as the registration writes it. The keeper compares
/// addresses as written.
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
    /// What an object built at the address takes now.
    credential: Option<String>,
    /// The keeper's clock when the registration that stated it was
    /// answered as taken.
    taken_at: u64,
    /// The registration in flight at the address.
    flight: Option<Flight>,
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

/// An address's entry as a registration found it.
struct Stood {
    credential: Option<String>,
    taken_at: u64,
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

// No address: as a registration writes it, its userinfo can carry a
// credential.
impl fmt::Debug for KeptCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeptCredential")
            .field("in_flight", &self.in_flight)
            .finish_non_exhaustive()
    }
}

impl KeptCredential {
    /// Whether the registration stated another credential than the one
    /// that stood at the address when it kept: another key, a key where
    /// none stood, or none where one stood. A write that moves no row
    /// builds no object. Where this is `true` behind such a write, accept
    /// the token and have the holder lay the address's objects again
    /// ([`Objects::lay_again`](super::Objects::lay_again)).
    pub fn differs(&self) -> bool {
        self.differs
    }

    /// The write moved the row, or moved nothing and the stated credential
    /// differs: the stated credential stands at the address. Call it at
    /// the door's answer, with no await between the two: a token dropped
    /// there puts back a credential the row's object was built with.
    pub fn accepted(mut self) {
        self.end(true);
    }

    /// The write was refused, or changed nothing: restore what stood at
    /// the address when the registration kept. Return `true` if an object
    /// was built with the credential put back meanwhile. That object holds
    /// it until the holder lays it again
    /// ([`Objects::lay_again`](super::Objects::lay_again)) or its row is
    /// written again.
    pub fn put_back(mut self) -> bool {
        self.end(false)
    }

    /// End the flight, once. Return whether an object was built with a
    /// refused credential.
    fn end(&mut self, accepted: bool) -> bool {
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
        if accepted {
            *taken += 1;
            entry.taken_at = *taken;
            return false;
        }
        let built = flight.built.into_inner();
        let Some(stood) = flight.stood else {
            entries.remove(&self.address);
            return built;
        };
        entry.credential = stood.credential;
        entry.taken_at = stood.taken_at;
        built
    }
}

impl Drop for KeptCredential {
    fn drop(&mut self) {
        self.end(false);
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

    /// State the credential of the registration at `address`, before its
    /// write. Pass `None` for a registration without one. An object built
    /// at the address from here on takes it. The token says whether it is
    /// another than the one that stood ([`KeptCredential::differs`]). Where
    /// the address's last keep has no answer yet, take nothing and return
    /// [`RegistrationInFlight`].
    pub fn keep(
        &self,
        address: &str,
        credential: Option<String>,
    ) -> Result<KeptCredential, RegistrationInFlight> {
        let mut held = self.held.write();
        let differs = match held.entries.get_mut(address) {
            Some(entry) if entry.flight.is_some() => return Err(RegistrationInFlight),
            Some(entry) => {
                let stood = Stood {
                    credential: mem::replace(&mut entry.credential, credential),
                    taken_at: entry.taken_at,
                };
                let differs = stood.credential != entry.credential;
                entry.flight = Some(Flight::over(Some(stood)));
                differs
            }
            None => {
                let differs = credential.is_some();
                let entry = Entry {
                    credential,
                    taken_at: 0,
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

    /// Give an object built at `address` its credential, and mark the
    /// registration in flight there as one an object was built in.
    pub(super) fn of(&self, address: &str) -> Option<String> {
        let held = self.held.read();
        let entry = held.entries.get(address)?;
        if let Some(flight) = &entry.flight {
            flight.built.store(true, Relaxed);
        }
        entry.credential.clone()
    }

    /// Whether an object built at `address` now would take `credential`:
    /// what the keeper holds there. The read marks no flight.
    pub(super) fn holds(&self, address: &str, credential: Option<&str>) -> bool {
        let held = self.held.read();
        held.entries
            .get(address)
            .and_then(|entry| entry.credential.as_deref())
            == credential
    }

    /// The keeper's clock: how many registrations were answered as taken.
    /// The holder reads it ahead of its read of the store.
    pub(super) fn taken(&self) -> u64 {
        self.held.read().taken
    }

    /// Forget what the rows that left held at `address`. The holder calls
    /// this once no row dials the address, with the clock it read ahead of
    /// its read of the store. An entry taken no later than `read` is
    /// dropped: its row was in the store the holder read, and is gone. One
    /// taken later stays: its row may have been written behind that read.
    /// Where a registration is in flight, the same rule drops what it
    /// displaced, and what it stated stays. Return whether a credential
    /// was dropped.
    pub(super) fn forget(&self, address: &str, read: u64) -> bool {
        let mut held = self.held.write();
        let Some(entry) = held.entries.get_mut(address) else {
            return false;
        };
        if let Some(flight) = &mut entry.flight {
            // No row dials the address, so no object holds what was stated.
            *flight.built.get_mut() = false;
            if flight
                .stood
                .as_ref()
                .is_some_and(|stood| stood.taken_at > read)
            {
                return false;
            }
            return flight
                .stood
                .take()
                .is_some_and(|stood| stood.credential.is_some());
        }
        if entry.taken_at > read {
            return false;
        }
        let had_credential = entry.credential.is_some();
        held.entries.remove(address);
        had_credential
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use super::{Credentials, KeptCredential, RegistrationInFlight};

    const ADDRESS: &str = "http://a:8000";
    const OTHER_ADDRESS: &str = "http://b:8000";
    const KEY_A: &str = "key-a";
    const KEY_B: &str = "key-b";
    const KEY_C: &str = "key-c";

    /// A registration's keep at an address where none is in flight.
    fn keep(credentials: &Credentials, address: &str, credential: Option<&str>) -> KeptCredential {
        credentials
            .keep(address, credential.map(str::to_owned))
            .expect("no registration in flight at the address")
    }

    /// A registration whose write moved its row.
    fn register(credentials: &Credentials, address: &str, credential: Option<&str>) {
        keep(credentials, address, credential).accepted();
    }

    /// Whether an object built at the address now takes `credential`. The
    /// read marks a flight, as a build does. No failure prints a credential.
    fn builds_with(credentials: &Credentials, address: &str, credential: Option<&str>) -> bool {
        credentials.of(address).as_deref() == credential
    }

    /// A forgetting whose read of the store begins now.
    fn forget(credentials: &Credentials, address: &str) -> bool {
        credentials.forget(address, credentials.taken())
    }

    fn has_entry(credentials: &Credentials, address: &str) -> bool {
        credentials.held.read().entries.contains_key(address)
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
    fn a_keep_says_whether_its_credential_differs_from_what_stood() {
        let credentials = Credentials::new();
        // At an address that held nothing, a credential differs and none
        // does not.
        let none = keep(&credentials, ADDRESS, None);
        assert!(!none.differs());
        none.accepted();
        let first = keep(&credentials, ADDRESS, Some(KEY_A));
        assert!(first.differs());
        first.accepted();
        assert!(keep(&credentials, OTHER_ADDRESS, Some(KEY_A)).differs());

        // The credential that stands differs in nothing. Another does, and
        // so does none.
        let same = keep(&credentials, ADDRESS, Some(KEY_A));
        assert!(!same.differs());
        assert!(!same.put_back());
        let other = keep(&credentials, ADDRESS, Some(KEY_B));
        assert!(other.differs());
        other.accepted();
        let cleared = keep(&credentials, ADDRESS, None);
        assert!(cleared.differs());

        // A put-back restores what stood: the next keep is held against it.
        assert!(!cleared.put_back());
        assert!(!keep(&credentials, ADDRESS, Some(KEY_B)).differs());
    }

    #[test]
    fn what_the_keeper_holds_is_read_with_no_mark_on_a_flight() {
        let credentials = Credentials::new();
        assert!(credentials.holds(ADDRESS, None));
        register(&credentials, ADDRESS, Some(KEY_A));
        assert!(credentials.holds(ADDRESS, Some(KEY_A)));
        assert!(!credentials.holds(ADDRESS, Some(KEY_B)));
        assert!(!credentials.holds(ADDRESS, None));

        // A flight's stated credential is what an object would take, and
        // the read is no build: the put-back says no object was built.
        let flight = keep(&credentials, ADDRESS, Some(KEY_C));
        assert!(credentials.holds(ADDRESS, Some(KEY_C)));
        assert!(!flight.put_back());
        assert!(credentials.holds(ADDRESS, Some(KEY_A)));
    }

    #[test]
    fn a_registration_taken_leaves_its_credential_standing() {
        let credentials = Credentials::new();
        assert!(credentials.is_empty());
        register(&credentials, ADDRESS, Some(KEY_A));
        assert_eq!(credentials.len(), 1);
        assert!(!credentials.is_empty());

        register(&credentials, ADDRESS, Some(KEY_B));
        register(&credentials, OTHER_ADDRESS, None);
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_B)));
        assert_eq!(credentials.len(), 1);

        register(&credentials, ADDRESS, None);
        assert!(credentials.is_empty());
        assert!(builds_with(&credentials, ADDRESS, None));
        assert!(!forget(&credentials, ADDRESS));
        assert!(!has_entry(&credentials, ADDRESS));
    }

    #[test]
    fn the_keeper_prints_how_many_it_keeps_and_no_credential() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, Some(KEY_A));
        let shown = format!("{credentials:?}");
        assert!(!shown.contains(KEY_A));
        assert_eq!(shown, "Credentials { held: 1 }");
    }

    #[test]
    fn a_keep_at_an_address_in_flight_takes_nothing() {
        let credentials = Credentials::new();
        let first = keep(&credentials, ADDRESS, Some(KEY_A));
        let second = credentials.keep(ADDRESS, Some(KEY_B.to_owned()));
        assert_eq!(second.err(), Some(RegistrationInFlight));
        assert_eq!(
            RegistrationInFlight.to_string(),
            "a registration is in flight at the address"
        );
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_A)));
        // Another address is its own turn.
        register(&credentials, OTHER_ADDRESS, Some(KEY_C));

        first.accepted();
        register(&credentials, ADDRESS, Some(KEY_B));
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_B)));
    }

    #[test]
    fn two_registrations_at_once_leave_an_entry_a_removal_forgets() {
        let credentials = Credentials::new();
        let first = keep(&credentials, ADDRESS, Some(KEY_A));
        assert!(credentials.keep(ADDRESS, Some(KEY_B.to_owned())).is_err());
        first.accepted();
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_A)));

        assert!(forget(&credentials, ADDRESS));
        assert!(!has_entry(&credentials, ADDRESS));
    }

    #[test]
    fn two_refused_registrations_at_once_leave_what_stood() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, Some(KEY_A));
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_A)));

        let first = keep(&credentials, ADDRESS, Some(KEY_B));
        assert!(credentials.keep(ADDRESS, Some(KEY_C.to_owned())).is_err());
        assert!(!first.put_back());
        // The second asks again and is refused too.
        assert!(!keep(&credentials, ADDRESS, Some(KEY_C)).put_back());
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_A)));
        assert_eq!(credentials.len(), 1);
    }

    #[test]
    fn a_put_back_restores_what_stood_as_it_was_taken() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, Some(KEY_A));
        let read = credentials.taken();

        assert!(!keep(&credentials, ADDRESS, Some(KEY_B)).put_back());
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_A)));
        // It was taken ahead of the read, and the put-back left that so.
        assert!(credentials.forget(ADDRESS, read));
        assert!(!has_entry(&credentials, ADDRESS));
    }

    #[test]
    fn a_put_back_says_that_an_object_was_built_with_the_refused_credential() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, Some(KEY_A));

        let refused = keep(&credentials, ADDRESS, Some(KEY_B));
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_B)));
        assert!(refused.put_back());
        // The next flight starts with no object built in it.
        assert!(!keep(&credentials, ADDRESS, Some(KEY_C)).put_back());
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_A)));
    }

    #[test]
    fn a_put_back_of_a_first_registration_leaves_no_entry() {
        let credentials = Credentials::new();
        assert!(!keep(&credentials, ADDRESS, Some(KEY_A)).put_back());
        assert!(!has_entry(&credentials, ADDRESS));

        let refused = keep(&credentials, ADDRESS, Some(KEY_A));
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_A)));
        assert!(refused.put_back());
        assert!(!has_entry(&credentials, ADDRESS));
        assert!(credentials.is_empty());
    }

    #[test]
    fn a_registration_that_states_none_can_be_put_back() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, Some(KEY_A));

        let refused = keep(&credentials, ADDRESS, None);
        // What it displaced is still held.
        assert_eq!(credentials.len(), 1);
        assert!(builds_with(&credentials, ADDRESS, None));
        assert!(refused.put_back());
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_A)));
    }

    #[test]
    fn a_token_dropped_with_no_answer_is_put_back() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, Some(KEY_A));

        drop(keep(&credentials, ADDRESS, Some(KEY_B)));
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_A)));
        // The address takes the next registration.
        register(&credentials, ADDRESS, Some(KEY_B));
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_B)));
    }

    #[test]
    fn forget_keeps_a_registration_answered_behind_its_read() {
        let credentials = Credentials::new();
        let read = credentials.taken();
        // Its row may have been written behind the holder's read.
        register(&credentials, ADDRESS, Some(KEY_A));
        assert!(!credentials.forget(ADDRESS, read));

        // So may the row of what a later flight displaced.
        let refused = keep(&credentials, ADDRESS, Some(KEY_B));
        assert!(!credentials.forget(ADDRESS, read));
        assert!(!refused.put_back());
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_A)));

        // A read that begins behind the answer forgets it.
        assert!(forget(&credentials, ADDRESS));
        assert!(!has_entry(&credentials, ADDRESS));
    }

    #[test]
    fn a_build_reads_beside_another_reader_of_the_keeper() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, Some(KEY_A));

        let keeper = &credentials;
        let held = keeper.held.read();
        let (send, receive) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(move || {
                let _ = send.send(keeper.of(ADDRESS).as_deref() == Some(KEY_A));
            });
            let read = receive.recv_timeout(Duration::from_secs(5));
            drop(held);
            assert_eq!(read, Ok(true));
        });
    }

    #[test]
    fn forget_in_flight_drops_what_stood_and_keeps_what_was_stated() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, Some(KEY_A));

        // The row left while the next registration is in flight.
        let next = keep(&credentials, ADDRESS, Some(KEY_B));
        assert!(forget(&credentials, ADDRESS));
        assert_eq!(credentials.len(), 1);
        assert!(!forget(&credentials, ADDRESS));
        next.accepted();
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_B)));

        // Refused instead: nothing stands any more, so nothing is left.
        let refused = keep(&credentials, ADDRESS, Some(KEY_C));
        assert!(forget(&credentials, ADDRESS));
        assert!(!refused.put_back());
        assert!(!has_entry(&credentials, ADDRESS));

        // The object built in flight left with its row: none holds the
        // refused credential.
        let refused = keep(&credentials, ADDRESS, Some(KEY_A));
        assert!(builds_with(&credentials, ADDRESS, Some(KEY_A)));
        assert!(!forget(&credentials, ADDRESS));
        assert!(!refused.put_back());
        assert!(!has_entry(&credentials, ADDRESS));
    }

    #[test]
    fn the_token_of_a_keep_prints_no_credential() {
        let credentials = Credentials::new();
        register(&credentials, ADDRESS, Some(KEY_A));
        let kept = keep(&credentials, ADDRESS, Some(KEY_B));

        let shown = format!("{kept:?}");
        assert!(!shown.contains(KEY_A));
        assert!(!shown.contains(KEY_B));
        assert_eq!(shown, "KeptCredential { in_flight: true, .. }");
        kept.accepted();
    }
}
