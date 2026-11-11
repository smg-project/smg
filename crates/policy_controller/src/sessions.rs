//! Sticky sessions (design 4.3, 4.4 and 4.6): a binding's `sessions` rule and
//! the session table it reads. A keyed request stays on the worker its key
//! was placed on; the table is plugin state the lifecycle feeds (the pin is
//! written at the commit, `on_dispatched`), keyed by the session key's
//! fingerprint as the host scopes it (one key names one session per model and
//! phase). A decision reads the table once and carries what it did with the
//! key as its own verdict (on the pick, then the lease), so its commit moves
//! the pins by that verdict alone: a hit leaves them, a respill fronts the
//! worker taken. The one write before a commit is a fresh key's claim of the
//! worker its decision chose, so concurrent first requests agree on one.

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

pub use crate::spec::{Assignment, Recovery};
use crate::view::WorkerId;

/// A binding's sessions rule as the engine runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionRule {
    pub assignment: Assignment,
    /// Requests of one key a worker carries before the next is placed
    /// elsewhere; `0` for no cap.
    pub inflight_cap: u16,
    /// Idle time after which a key's pins are forgotten; `None` for never.
    pub ttl_micros: Option<u64>,
    /// How often idle keys are swept, on the decision clock; `0` for never
    /// (a rule without a TTL).
    pub sweep_micros: u64,
    pub recovery: Recovery,
}

/// What a decision under a sessions rule did with its key. It travels on the
/// pick and the lease, so a commit moves the pins by its own decision's
/// verdict and no other request's reading of the key can erase it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SessionVerdict {
    /// No sessions rule governed the decision: its commit pins nothing.
    #[default]
    None,
    /// The pin was taken, or the key was placed by the rule: the commit
    /// records the worker by the key's recovery.
    Pinned,
    /// The routable pin was over the in-flight cap and the stages placed the
    /// request: the commit fronts the worker it lands on.
    Respill,
}

/// The pins of one key: the preferred worker first, the failover second;
/// `WorkerId(0)` is an empty slot.
pub type Pins = [WorkerId; 2];

#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    pins: Pins,
    /// The instant the key's pins expire on the decision clock; `u64::MAX`
    /// under a rule without a TTL.
    expires_at_micros: u64,
    recovery: Recovery,
}

impl Entry {
    fn touch(&mut self, now_micros: u64, rule: &SessionRule) {
        self.expires_at_micros = rule
            .ttl_micros
            .map_or(u64::MAX, |ttl| now_micros.saturating_add(ttl));
        self.recovery = rule.recovery;
    }

    /// `target` becomes the preferred worker; the preferred one before it the
    /// failover.
    fn front(&mut self, target: WorkerId) {
        if self.pins[0] != target {
            self.pins = [target, self.pins[0]];
        }
    }

    /// `target` joins the pins behind the ones there, the oldest dropped when
    /// both slots are taken.
    fn append(&mut self, target: WorkerId) {
        if self.pins.contains(&target) {
            return;
        }
        if self.pins[0] == WorkerId(0) {
            self.pins[0] = target;
        } else if self.pins[1] == WorkerId(0) {
            self.pins[1] = target;
        } else {
            self.pins = [self.pins[1], target];
        }
    }
}

/// Shards of the table, by the key's low bits: keyed decisions on different
/// keys take different locks.
const SHARDS: usize = 16;

/// The session table: every key's pins, swept on the decision clock.
#[derive(Debug)]
pub struct SessionTable {
    shards: Vec<Mutex<FxHashMap<u64, Entry>>>,
    /// The decision clock at the next sweep.
    next_sweep_micros: AtomicU64,
}

impl Default for SessionTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionTable {
    pub fn new() -> Self {
        Self {
            shards: (0..SHARDS)
                .map(|_| Mutex::new(FxHashMap::default()))
                .collect(),
            next_sweep_micros: AtomicU64::new(0),
        }
    }

    fn shard(&self, key: u64) -> &Mutex<FxHashMap<u64, Entry>> {
        &self.shards[(key % SHARDS as u64) as usize]
    }

    /// The pins of `key` at a decision taken at `now_micros` under `rule`,
    /// the key touched (its TTL renewed, its recovery rule recorded) and the
    /// table swept when a sweep is due. A key never seen has empty pins.
    pub(crate) fn lookup(&self, key: u64, now_micros: u64, rule: &SessionRule) -> Pins {
        self.sweep_if_due(now_micros, rule);
        let mut shard = self.shard(key).lock();
        let entry = shard.entry(key).or_default();
        entry.touch(now_micros, rule);
        entry.pins
    }

    /// A decision that found no routable pin (a fresh key, or every pin
    /// down) claims the worker it chose, under the entry's lock, by the key's
    /// recovery: the pins move now, and the commit finds them in place. When
    /// another decision moved the pins since this one read them (`seen`),
    /// the newcomer among them is the claim to follow instead, when it is
    /// among this decision's survivors; the pin of record is returned. The
    /// claim is the one pin made before a commit; one whose commit never
    /// comes stands until a later dispatch moves it or the TTL sweeps it.
    pub(crate) fn claim(&self, key: u64, seen: Pins, target: WorkerId) -> WorkerId {
        if key == 0 || target == WorkerId(0) {
            return target;
        }
        let mut shard = self.shard(key).lock();
        let entry = shard.entry(key).or_default();
        if entry.pins == seen {
            match entry.recovery {
                Recovery::Latest => entry.front(target),
                Recovery::Original => entry.append(target),
            }
            return target;
        }
        entry
            .pins
            .iter()
            .copied()
            .find(|pin| *pin != WorkerId(0) && !seen.contains(pin))
            .unwrap_or(target)
    }

    /// The commit dispatched a decision for `key` to `target`, `respill` when
    /// the decision placed the request past its saturated pin: the pin
    /// follows the rule. The preferred pin taken, or the failover taken while
    /// the preferred one was unroutable, leaves the pins as they are; a
    /// respill, and a worker outside the pins, move them by the key's
    /// recovery rule.
    pub(crate) fn dispatched(&self, key: u64, respill: bool, target: WorkerId) {
        if key == 0 || target == WorkerId(0) {
            return;
        }
        let mut shard = self.shard(key).lock();
        let entry = shard.entry(key).or_default();
        if entry.pins[0] == target {
            return;
        }
        if !respill && entry.pins.contains(&target) {
            return;
        }
        match (respill, entry.recovery) {
            (true, _) | (false, Recovery::Latest) => entry.front(target),
            (false, Recovery::Original) => entry.append(target),
        }
    }

    /// Drops a worker from every key's pins when the membership feed removes
    /// or replaces it: one walk of the table on a rare event. A key whose owner
    /// went keeps its failover as the owner; a key with no pin left leaves the
    /// table and is placed afresh. A worker that comes back under a new id
    /// (the same address re-registered) does not get its keys back.
    pub(crate) fn forget_worker(&self, worker: WorkerId) {
        if worker == WorkerId(0) {
            return;
        }
        for shard in &self.shards {
            shard.lock().retain(|_, entry| {
                if entry.pins[1] == worker {
                    entry.pins[1] = WorkerId(0);
                }
                if entry.pins[0] == worker {
                    entry.pins = [entry.pins[1], WorkerId(0)];
                }
                entry.pins[0] != WorkerId(0)
            });
        }
    }

    /// Forget every key whose TTL passed, once per sweep interval; a rule
    /// without a sweep (no TTL) never walks the table.
    fn sweep_if_due(&self, now_micros: u64, rule: &SessionRule) {
        if rule.sweep_micros == 0 {
            return;
        }
        let due = self.next_sweep_micros.load(Ordering::Relaxed);
        if now_micros < due {
            return;
        }
        let next = now_micros.saturating_add(rule.sweep_micros);
        if self
            .next_sweep_micros
            .compare_exchange(due, next, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        for shard in &self.shards {
            shard
                .lock()
                .retain(|_, entry| entry.expires_at_micros > now_micros);
        }
    }

    /// The pins of `key`, for tests and dumps.
    pub fn pins(&self, key: u64) -> Option<Pins> {
        self.shard(key).lock().get(&key).map(|e| e.pins)
    }

    /// Keys the table holds.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.lock().len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULE: SessionRule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 2,
        ttl_micros: Some(1_000),
        sweep_micros: 100,
        recovery: Recovery::Latest,
    };

    const ORIGINAL: SessionRule = SessionRule {
        recovery: Recovery::Original,
        ..RULE
    };

    #[test]
    fn the_latest_dispatch_leads_and_the_one_before_it_is_the_failover() {
        let table = SessionTable::new();
        assert_eq!(table.lookup(7, 10, &RULE), [WorkerId(0); 2]);
        table.dispatched(7, false, WorkerId(1));
        assert_eq!(table.pins(7), Some([WorkerId(1), WorkerId(0)]));
        // A hit on the preferred pin changes nothing.
        table.lookup(7, 20, &RULE);
        table.dispatched(7, false, WorkerId(1));
        assert_eq!(table.pins(7), Some([WorkerId(1), WorkerId(0)]));
        // A worker outside the pins (the pins were unroutable) leads.
        table.lookup(7, 30, &RULE);
        table.dispatched(7, false, WorkerId(2));
        assert_eq!(table.pins(7), Some([WorkerId(2), WorkerId(1)]));
        // The failover taken while the preferred one is unroutable keeps the
        // order; a respill to it makes it the preferred one.
        table.lookup(7, 40, &RULE);
        table.dispatched(7, false, WorkerId(1));
        assert_eq!(table.pins(7), Some([WorkerId(2), WorkerId(1)]));
        table.lookup(7, 50, &RULE);
        table.dispatched(7, true, WorkerId(1));
        assert_eq!(table.pins(7), Some([WorkerId(1), WorkerId(2)]));
    }

    #[test]
    fn under_original_recovery_the_first_worker_keeps_the_lead() {
        let table = SessionTable::new();
        table.lookup(3, 10, &ORIGINAL);
        table.dispatched(3, false, WorkerId(1));
        table.lookup(3, 20, &ORIGINAL);
        table.dispatched(3, false, WorkerId(2));
        assert_eq!(table.pins(3), Some([WorkerId(1), WorkerId(2)]));
        // A third worker evicts the oldest.
        table.lookup(3, 30, &ORIGINAL);
        table.dispatched(3, false, WorkerId(3));
        assert_eq!(table.pins(3), Some([WorkerId(2), WorkerId(3)]));
    }

    #[test]
    fn idle_keys_are_swept_once_the_interval_passes() {
        let table = SessionTable::new();
        table.lookup(1, 0, &RULE);
        table.dispatched(1, false, WorkerId(1));
        table.lookup(2, 900, &RULE);
        table.dispatched(2, false, WorkerId(2));
        assert_eq!(table.len(), 2);
        // Within the sweep interval nothing is swept.
        table.lookup(5, 950, &RULE);
        assert_eq!(table.pins(1), Some([WorkerId(1), WorkerId(0)]));
        // Past the interval and the first key's TTL: the first key is gone,
        // the second (touched at 900) stays.
        table.lookup(6, 1_100, &RULE);
        assert_eq!(table.pins(1), None);
        assert_eq!(table.pins(2), Some([WorkerId(2), WorkerId(0)]));
        // Without a TTL nothing expires.
        let forever = SessionTable::new();
        let no_ttl = SessionRule {
            ttl_micros: None,
            ..RULE
        };
        forever.lookup(1, 0, &no_ttl);
        forever.dispatched(1, false, WorkerId(1));
        forever.lookup(2, 1_000_000, &no_ttl);
        assert_eq!(forever.pins(1), Some([WorkerId(1), WorkerId(0)]));
    }

    #[test]
    fn an_empty_key_or_target_writes_nothing() {
        let table = SessionTable::new();
        table.dispatched(0, false, WorkerId(1));
        table.dispatched(4, false, WorkerId(0));
        assert!(table.is_empty());
    }

    /// Two decisions on one key interleave: A reads, B reads, A notes its
    /// respill, B dispatches onto the failover pin (a hit, its instant is
    /// not the note's), then A dispatches onto it (the respill). The note is
    /// keyed by the decision instant and consumed by its own dispatch alone.
    #[test]
    fn a_respill_verdict_travels_with_its_decision_past_another_decisions_reading() {
        let table = SessionTable::new();
        table.lookup(9, 10, &RULE);
        table.dispatched(9, false, WorkerId(1));
        table.lookup(9, 20, &RULE);
        table.dispatched(9, false, WorkerId(2));
        assert_eq!(table.pins(9), Some([WorkerId(2), WorkerId(1)]));
        // A decides a respill onto the failover, B reads the key and takes
        // the failover as a plain hit, B commits first: B's hit leaves the
        // pins, A's respill fronts the failover whatever came between.
        table.lookup(9, 30, &RULE);
        table.lookup(9, 31, &RULE);
        table.dispatched(9, false, WorkerId(1));
        assert_eq!(
            table.pins(9),
            Some([WorkerId(2), WorkerId(1)]),
            "B's dispatch onto the failover is a hit"
        );
        table.dispatched(9, true, WorkerId(1));
        assert_eq!(
            table.pins(9),
            Some([WorkerId(1), WorkerId(2)]),
            "A's dispatch is the respill"
        );
    }

    #[test]
    fn the_first_claim_stands_for_a_fresh_key_and_for_a_key_whose_pins_are_down() {
        let table = SessionTable::new();
        let none = [WorkerId(0); 2];
        let seen = table.lookup(5, 10, &ORIGINAL);
        assert_eq!(table.claim(5, seen, WorkerId(3)), WorkerId(3), "the first");
        assert_eq!(table.claim(5, none, WorkerId(4)), WorkerId(3), "follows it");
        assert_eq!(table.pins(5), Some([WorkerId(3), WorkerId(0)]));
        table.dispatched(5, false, WorkerId(3));
        table.dispatched(5, false, WorkerId(4));
        assert_eq!(
            table.pins(5),
            Some([WorkerId(3), WorkerId(4)]),
            "a loser dispatched elsewhere is the failover"
        );
        // Both pins down: two decisions read [3, 4]; the first claim appends
        // its worker by the recovery (the oldest slot goes), the second
        // follows the newcomer.
        let seen = table.lookup(5, 20, &ORIGINAL);
        assert_eq!(table.claim(5, seen, WorkerId(7)), WorkerId(7));
        assert_eq!(table.pins(5), Some([WorkerId(4), WorkerId(7)]));
        assert_eq!(
            table.claim(5, seen, WorkerId(8)),
            WorkerId(7),
            "the newcomer"
        );
        // Under the latest-first recovery the claim leads at once.
        let seen = table.lookup(6, 30, &RULE);
        assert_eq!(table.claim(6, seen, WorkerId(1)), WorkerId(1));
        assert_eq!(table.pins(6), Some([WorkerId(1), WorkerId(0)]));
        assert_eq!(
            table.claim(0, none, WorkerId(9)),
            WorkerId(9),
            "no key, no claim"
        );
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn a_forgotten_worker_leaves_every_pin_and_a_key_with_no_pin_left_leaves_the_table() {
        let table = SessionTable::new();
        for (key, first, second) in [(1, 1, 2), (2, 2, 0), (3, 2, 1), (4, 3, 0)] {
            table.lookup(key, 10, &ORIGINAL);
            table.dispatched(key, false, WorkerId(first));
            if second != 0 {
                table.lookup(key, 20, &ORIGINAL);
                table.dispatched(key, false, WorkerId(second));
            }
        }
        assert_eq!(table.pins(3), Some([WorkerId(2), WorkerId(1)]));
        table.forget_worker(WorkerId(2));
        assert_eq!(
            table.pins(1),
            Some([WorkerId(1), WorkerId(0)]),
            "the failover slot went"
        );
        assert_eq!(table.pins(2), None, "its only pin went");
        assert_eq!(
            table.pins(3),
            Some([WorkerId(1), WorkerId(0)]),
            "the failover is the owner"
        );
        assert_eq!(table.pins(4), Some([WorkerId(3), WorkerId(0)]), "untouched");
        assert_eq!(table.len(), 3);
        table.forget_worker(WorkerId(0));
        assert_eq!(table.len(), 3, "the empty id is never a pin");
    }
}
