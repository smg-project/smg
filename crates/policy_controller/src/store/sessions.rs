//! The sessions store: a session key's pin to its target (design 5.1, the
//! sessions row).

use std::time::Instant;

use crate::view::WorkerId;

/// A session's pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pin {
    pub worker: WorkerId,
    /// The last decision that used it, on the host's clock: a merge reads
    /// it.
    pub last_used: Instant,
    /// The instant the pin ends, by the TTL of the rule that placed its
    /// key: at or past it the sweep drops the pin and a decision's read
    /// takes none. `None` for a key its rule keeps with no TTL, and for a
    /// key whose rule the writer could not name where no standing rule had
    /// a TTL: such a pin stays until the key's next dispatch writes it
    /// anew, or its worker's row goes.
    pub ends: Option<Instant>,
}

/// Where session pins are written: at the commit of a keyed decision, never
/// at selection. Keys are fingerprints (never `0`). A removed worker takes its
/// pins with it; the gateways keep a bounded table of the pins they received
/// and replay it in `Hello`, merged here by last use.
pub trait SessionsStore: Send + Sync {
    /// Pin `key` to `worker` at `at`, to end at `ends` (`None`: at no
    /// instant); a re-pin moves it and its end.
    fn pin(&self, key: u64, worker: WorkerId, at: Instant, ends: Option<Instant>);

    fn lookup(&self, key: u64) -> Option<Pin>;

    fn unpin(&self, key: u64);

    /// A worker left: its pins go with it. Returns how many.
    fn drop_worker(&self, worker: WorkerId) -> usize;

    /// Drop the pins whose own end is at or before `now`. Returns how
    /// many.
    fn expire(&self, now: Instant) -> usize;

    /// Merge a pin a gateway replayed: a key already pinned keeps the later
    /// use.
    fn merge(&self, key: u64, pin: Pin);

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
