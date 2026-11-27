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
/// The holder forgets by itself. A cut laid whole drops the rows it
/// finds gone from the keeper, and an entry goes with its last row.
/// A write of the holder's own that takes a row out drops that row, and
/// the lay-again an answer asks for drops each bound row that holds no
/// object. Neither reads the store.
/// The holder reads the keeper's clock ahead of each pass: a row bound
/// behind that read may have been written behind it, and stays. A
/// forgetting that misses a row leaves a binding no later worker matches.
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
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
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
    /// The holder reads it ahead of each pass of its forgetting.
    pub(super) fn taken(&self) -> u64 {
        self.held.read().taken
    }

    /// Forget the rows that left. The holder calls this with the clock it
    /// read ahead and the rows that stand. It calls behind a cut laid whole,
    /// at a lay-again, and for the one row a write took out.
    /// A bound row that does not stand and was
    /// bound no later than `read` is dropped.
    /// One bound later stays: its row may have been written behind that
    /// read. A call about one record asks about that record's row alone,
    /// and says of every other row that it stands.
    /// An entry with no row and no registration in flight goes.
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
#[path = "../../tests/unit/worker_host/credentials/tests.rs"]
mod tests;
