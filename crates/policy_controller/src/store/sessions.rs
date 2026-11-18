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
    /// takes none. It stands on the decision clock, in microseconds since
    /// the store's origin, as the engine's table holds the instant it
    /// forgets the key: a dispatch writes it as read, or as its lease's
    /// own rule counts it from the lease's issue, and converts no clock.
    /// `None` for a key its rule keeps with no TTL: such a pin stays until
    /// its worker's row goes.
    pub ends: Option<u64>,
}

/// Where session pins are written: at the commit of a keyed decision, never
/// at selection. Keys are fingerprints (never `0`). A removed worker takes its
/// pins with it. By the design the gateways keep a bounded table of the pins
/// they received and replay it in `Hello`, merged here by last use: no
/// gateway replays its pins in this build, and the merge door has no caller.
pub trait SessionsStore: Send + Sync {
    /// Pin `key` to `worker` at `at`, to end at `ends` on the decision
    /// clock (`None`: at no instant); a re-pin moves it and its end.
    fn pin(&self, key: u64, worker: WorkerId, at: Instant, ends: Option<u64>);

    fn lookup(&self, key: u64) -> Option<Pin>;

    fn unpin(&self, key: u64);

    /// A worker left: its pins go with it. Returns how many.
    fn drop_worker(&self, worker: WorkerId) -> usize;

    /// Drop the pins whose own end is at or before `now`, an instant of
    /// the decision clock. Returns how many.
    fn expire(&self, now: u64) -> usize;

    /// Merge a pin a gateway replayed: a key already pinned keeps the later
    /// use. The pin's end is a number of the clock of the store that wrote
    /// it: a caller turns it into this store's before it merges.
    fn merge(&self, key: u64, pin: Pin);

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
