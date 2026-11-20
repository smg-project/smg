//! Sticky sessions (design 4.3, 4.4 and 4.6): a binding's `sessions` rule and
//! the session table it reads. A keyed request stays on the worker its key
//! was placed on; the table is plugin state the lifecycle feeds (the pin is
//! written at the commit, `on_dispatched`), keyed by the session's value
//! (`RequestFacts::session`: one key names one session per model and phase,
//! by one formula on every host). A decision reads the table once and
//! carries what it did with the key as its own verdict (on the pick, then
//! the lease), so its commit moves the pins by that verdict alone: a hit
//! leaves them, a respill fronts the worker taken. The one write before a
//! commit is a fresh key's claim of the worker its decision chose, so
//! concurrent first requests agree on one. A key the table holds no pin of
//! asks the host's pins of record first, where the host lends them
//! ([`crate::PinSource`]): a pin a dispatch made under an earlier
//! generation is written back, and no claim is made.

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

pub use crate::spec::{Assignment, Recovery};
use crate::view::{Session, WorkerId};

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
    /// (a rule without a TTL). The table sweeps at the shortest interval
    /// among its rules ([`SessionTable::new`]).
    pub sweep_micros: u64,
    pub recovery: Recovery,
}

/// What a decision under a sessions rule did with its key. It travels on the
/// pick and the lease, so a commit moves the pins by its own decision's
/// verdict and no other request's reading of the key can erase it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SessionVerdict {
    /// No sessions rule governed the decision, or its rule placed a request
    /// of a key at its cap on another worker at the cap: its commit pins
    /// nothing.
    #[default]
    None,
    /// The pin was taken, or the key was placed by the rule: the commit
    /// records the worker by the key's recovery.
    Pinned,
    /// The routable pin was at the in-flight cap and the stages placed the
    /// request on another worker under the cap: the commit fronts the
    /// worker it lands on.
    Respill,
    /// The key had no routable pin and the decision claimed the worker it
    /// placed the request on: that decision's commit confirms the claim on
    /// the worker, or withdraws it when the dispatch landed elsewhere.
    Claimed(WorkerId),
}

/// The pins of one key: the preferred worker first, the failover second;
/// `WorkerId(0)` is an empty slot.
pub type Pins = [WorkerId; 2];

/// When a table forgets a key it holds, on the decision clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Forgets {
    /// At this instant: the key's rule has a TTL.
    At(u64),
    /// At no instant: the key's rule has no TTL.
    Never,
    /// No decision has read the key since its entry was made: a dispatch
    /// made the entry anew, behind the end of an absence, a forget or the
    /// end of the key's TTL, and the next decision's read starts the key
    /// over.
    Unread,
}

/// What [`SessionTable::claim`] did for a decision that found no routable
/// pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Claim {
    /// The decision's own worker is the claim of record; that decision's
    /// commit settles it.
    Made(WorkerId),
    /// The claim was not taken: the key's pins are not the ones the
    /// decision read, or nothing was recorded (an empty key or target).
    /// It names the worker the decision follows ([`Claim::standing`]) and
    /// the pins the key held: a shadow reads those against its own pins
    /// and its own pick.
    Standing(WorkerId, Pins),
}

impl Claim {
    /// The worker a decision follows where the key holds `held` and the
    /// decision read `seen` and chose `target`. None where the pins stand
    /// as it read them: its claim is taken. Else the newcomer among them,
    /// or its own `target` where no pin came in: the pins moved, a claim
    /// was withdrawn, or a worker left. That last answer is the
    /// decision's own pick, and no pin for another decision to follow.
    /// The pins are read in place: a claim that is taken loads no pin it
    /// does not test.
    #[inline(always)]
    pub(crate) fn standing(held: &Pins, seen: &Pins, target: WorkerId) -> Option<WorkerId> {
        if held == seen {
            return None;
        }
        Some(
            held.iter()
                .copied()
                .find(|pin| *pin != WorkerId(0) && !seen.contains(pin))
                .unwrap_or(target),
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    pins: Pins,
    /// The instant the key's pins expire on the decision clock; `u64::MAX`
    /// under a rule without a TTL, and `0` until a decision reads the key.
    expires_at_micros: u64,
    /// The newest decision instant that set the expiry and recovery rule.
    touched_at_micros: u64,
    recovery: Recovery,
    /// The pin a decision's claim wrote before its commit; `WorkerId(0)`
    /// when none stands. Any dispatch to it confirms it (the key was served
    /// there), the claiming decision's dispatch elsewhere withdraws it while
    /// it stands unconfirmed, and no other dispatch elsewhere touches it; a
    /// claim whose commit never comes stands until the TTL sweeps the key.
    claim: WorkerId,
    /// The id the claim's lease names: the claimed worker's id when the
    /// claim was made, read while a claim stands. It differs from `claim`
    /// once the worker is registered again under the claim
    /// ([`SessionTable::rename_worker`]).
    leased: WorkerId,
}

impl Entry {
    fn touch(&mut self, now_micros: u64, rule: &SessionRule) {
        if now_micros < self.touched_at_micros {
            return;
        }
        self.expires_at_micros = rule
            .ttl_micros
            .map_or(u64::MAX, |ttl| now_micros.saturating_add(ttl));
        self.touched_at_micros = now_micros;
        self.recovery = rule.recovery;
    }

    /// `target` becomes the preferred worker; the preferred one before it the
    /// failover.
    fn front(&mut self, target: WorkerId) {
        if self.pins[0] != target {
            self.pins = [target, self.pins[0]];
        }
    }

    /// Withdraws `claim` from the pins (the pin behind it moves up): its
    /// commit never came and another worker served the key.
    fn withdraw(&mut self, claim: WorkerId) {
        if self.pins[0] == claim {
            self.pins = [self.pins[1], WorkerId(0)];
        } else if self.pins[1] == claim {
            self.pins[1] = WorkerId(0);
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
pub(crate) const SHARDS: usize = 16;

/// The shard of `key`, in this table and in the store's pins of record.
pub(crate) fn shard_of(key: u64) -> usize {
    (key % SHARDS as u64) as usize
}

/// How long a worker that left keeps its pins where no rule says a TTL:
/// four hours, the TTL a rule takes where its document names none.
pub const ABSENCE_MICROS: u64 = 14_400 * 1_000_000;

/// The most workers a table holds as absent ([`SessionTable::left`]): past
/// it the one that left first is forgotten at once, so the list is bounded
/// where no decision stamps or ends an absence.
pub(crate) const ABSENT_WORKERS: usize = 4_096;

/// The session table: every key's pins, swept on the decision clock.
#[derive(Debug)]
pub struct SessionTable {
    shards: Vec<Mutex<FxHashMap<u64, Entry>>>,
    /// The decision clock at the next sweep, of the idle keys or of a
    /// worker that left: `u64::MAX` while neither is owed.
    next_sweep_micros: AtomicU64,
    /// The decision clock at the next pass over the idle keys, read by the
    /// sweep alone: a sweep a leave asked for walks no shard for the keys
    /// before their pass is due. `u64::MAX` where no rule sweeps.
    next_pass_micros: AtomicU64,
    /// The sweep interval: the shortest among the rules of the engine's
    /// bindings, so no rule's interval defers another rule's expiry; `0`
    /// never sweeps (no rule has a TTL).
    sweep_micros: u64,
    /// The workers that left membership while their pins stand
    /// ([`Self::left`]), each with the decision clock at which it is
    /// forgotten: `0` until a decision stamps it, the word of a leave
    /// carrying no clock. One id and one instant a worker, in the order
    /// they left, [`ABSENT_WORKERS`] at most.
    ///
    /// A host tells a leave and a join under its own membership lock, and
    /// the sweep that ends an absence holds this lock across its walk:
    /// the host's lock first, then this one, then a shard's. Nothing
    /// takes this lock under a shard's. A membership write that lands in
    /// that sweep waits for one walk of the table: the price of a worker
    /// that joins again behind the end losing no pin it is given after.
    absent: Mutex<Vec<(WorkerId, u64)>>,
    /// How long a worker that left keeps its pins.
    absence_micros: u64,
}

impl SessionTable {
    /// A table swept every `sweep_micros` on the decision clock: the
    /// shortest sweep interval among the rules that look keys up in it, `0`
    /// when none sweeps.
    pub fn new(sweep_micros: u64) -> Self {
        Self {
            shards: (0..SHARDS)
                .map(|_| Mutex::new(FxHashMap::default()))
                .collect(),
            next_sweep_micros: AtomicU64::new(if sweep_micros == 0 { u64::MAX } else { 0 }),
            next_pass_micros: AtomicU64::new(if sweep_micros == 0 { u64::MAX } else { 0 }),
            sweep_micros,
            absent: Mutex::new(Vec::new()),
            absence_micros: ABSENCE_MICROS,
        }
    }

    /// The table with `micros` as the time a worker that left keeps its
    /// pins: the longest TTL among the rules that look
    /// keys up in it, so a key loses no pin to an absence before its own
    /// rule would forget it.
    #[must_use]
    pub fn with_absence(mut self, micros: u64) -> Self {
        self.absence_micros = micros;
        self
    }

    /// How long a worker that left keeps its pins, in microseconds.
    pub fn absence_micros(&self) -> u64 {
        self.absence_micros
    }

    fn shard(&self, key: u64) -> &Mutex<FxHashMap<u64, Entry>> {
        &self.shards[shard_of(key)]
    }

    /// The pins of `key` at a decision taken at `now_micros` under `rule`,
    /// the key touched when its decision instant is not older (its TTL
    /// renewed, its recovery rule recorded) and the table swept when a sweep
    /// is due. A key never seen, or one whose TTL passed, has empty pins:
    /// expiry is the key's own compare, the sweep only reclaims.
    /// Always inlined: the keyed decision has two forms, and a second caller
    /// would otherwise move this out of line under the first.
    #[inline(always)]
    pub(crate) fn lookup(&self, key: Session, now_micros: u64, rule: &SessionRule) -> Pins {
        let key = key.get();
        self.sweep_if_due(now_micros);
        let mut shard = self.shard(key).lock();
        let entry = shard.entry(key).or_default();
        if entry.expires_at_micros <= now_micros {
            // Expired and not yet swept: the key starts over here, under the
            // lock the sweep takes; the walk only reclaims what expired.
            *entry = Entry::default();
        }
        entry.touch(now_micros, rule);
        entry.pins
    }

    /// A decision that found no routable pin (a fresh key, or every pin
    /// down) claims the worker it chose, under the entry's lock, by the key's
    /// recovery: the pins move now, and the commit finds them in place. When
    /// another decision moved the pins since this one read them (`seen`),
    /// the newcomer among them is the claim to follow instead, when it is
    /// among this decision's survivors; the pin of record is returned with
    /// the pins the key held ([`Claim::standing`] is the rule). The
    /// claim is the one pin made before a commit, and it is marked as such:
    /// any dispatch to it confirms it, the claiming decision's own dispatch
    /// elsewhere withdraws it (`dispatched`), the TTL sweeps it otherwise.
    /// Always inlined: the keyed decision has two forms, and a second caller
    /// would otherwise move this out of line under the first.
    #[inline(always)]
    pub(crate) fn claim(&self, key: Session, seen: Pins, target: WorkerId) -> Claim {
        let key = key.get();
        if key == 0 || target == WorkerId(0) {
            return Claim::Standing(target, seen);
        }
        let mut shard = self.shard(key).lock();
        let entry = shard.entry(key).or_default();
        let Some(owner) = Claim::standing(&entry.pins, &seen, target) else {
            match entry.recovery {
                Recovery::Latest => entry.front(target),
                Recovery::Original => entry.append(target),
            }
            entry.claim = target;
            entry.leased = target;
            return Claim::Made(target);
        };
        Claim::Standing(owner, entry.pins)
    }

    /// A pin of record read for a key the table held no pin of (a key it
    /// never saw, or one a new generation's table starts without): `worker`
    /// becomes the key's preferred pin where the key still has none, under
    /// the entry's lock, and the pins that stand are returned. It is a pin
    /// a dispatch made, never a claim: no commit withdraws it. A key that
    /// took a pin since the decision read it keeps that pin.
    pub(crate) fn restore(
        &self,
        key: Session,
        now_micros: u64,
        rule: &SessionRule,
        worker: WorkerId,
    ) -> Pins {
        let key = key.get();
        if key == 0 || worker == WorkerId(0) {
            return [WorkerId(0); 2];
        }
        let mut shard = self.shard(key).lock();
        let entry = shard.entry(key).or_default();
        if entry.pins[0] == WorkerId(0) {
            entry.pins[0] = worker;
            // The entry may be new here: between this decision's read and
            // this write another decision pinned the key, and the end of
            // that worker's absence, or a forget, took its last pin.
            entry.touch(now_micros, rule);
        }
        entry.pins
    }

    /// The commit dispatched a decision for `key` to `target` under the
    /// decision's `verdict`: the pin follows the rule. The preferred pin
    /// taken, or the failover taken while the preferred one was unroutable,
    /// leaves the pins as they are; a respill, and a worker outside the pins,
    /// move them by the key's recovery rule. A dispatch to the claimed worker
    /// confirms the claim first; the claiming decision's commit elsewhere
    /// withdraws it.
    pub(crate) fn dispatched(&self, key: Session, verdict: SessionVerdict, target: WorkerId) {
        let key = key.get();
        if key == 0 || target == WorkerId(0) {
            return;
        }
        let mut shard = self.shard(key).lock();
        let entry = shard.entry(key).or_default();
        // A dispatch to the claimed worker confirms the claim, whichever
        // decision commits there: the key was served on it, so a pin another
        // request already used is never withdrawn. Only the claiming
        // decision's commit elsewhere withdraws a claim still unconfirmed; an
        // older request's commit on the owner leaves a retry's claim standing
        // for the retries after it.
        // A claim's lease names the id its worker had when it claimed: a
        // dispatch on that id is one on the claimed worker, registered
        // again since or not.
        let claimed_here = entry.claim != WorkerId(0) && entry.leased == target;
        let target = if claimed_here { entry.claim } else { target };
        if entry.claim != WorkerId(0) && entry.claim == target {
            entry.claim = WorkerId(0);
        }
        if let SessionVerdict::Claimed(claimed) = verdict {
            if claimed != WorkerId(0) && entry.claim != WorkerId(0) && entry.leased == claimed {
                let pin = entry.claim;
                entry.claim = WorkerId(0);
                entry.withdraw(pin);
            }
        }
        if entry.pins[0] == target {
            return;
        }
        let respill = verdict == SessionVerdict::Respill;
        if !respill && entry.pins.contains(&target) {
            return;
        }
        match (respill, entry.recovery) {
            (true, _) | (false, Recovery::Latest) => entry.front(target),
            (false, Recovery::Original) => entry.append(target),
        }
    }

    /// A worker left membership: its pins stand, and read as not routable
    /// while no row holds its id, as an unhealthy member's do. A request
    /// of one of its keys is placed by the rule meanwhile, and its commit
    /// moves the pins by the rule's recovery. The worker is forgotten
    /// ([`Self::forget_workers`]) once the absence's time passes on the
    /// decision clock, counted from the next decision of any key. One
    /// registered again before that takes its pins under its new id
    /// ([`Self::rename_worker`]), and one that joins under the id it had
    /// keeps them ([`Self::returned`]).
    ///
    /// A table that holds no key holds no pin of the worker and files
    /// nothing: a host that only follows a membership, or one whose
    /// document keeps no key, has no absence to end. At
    /// [`ABSENT_WORKERS`] the worker that left first is forgotten now.
    pub(crate) fn left(&self, worker: WorkerId) {
        if worker == WorkerId(0) || self.is_empty() {
            return;
        }
        let mut absent = self.absent.lock();
        let mut first = None;
        if !absent.iter().any(|(id, _)| *id == worker) {
            if absent.len() >= ABSENT_WORKERS {
                first = Some(absent.remove(0).0);
            }
            absent.push((worker, 0));
        }
        // Under the lock: a sweep in flight writes its next instant under
        // it too, and stamps this leave or is asked again by this store.
        self.next_sweep_micros.store(0, Ordering::Relaxed);
        drop(absent);
        if let Some(first) = first {
            self.drop_pins(&[first]);
        }
    }

    /// A worker joined under `worker`. Where that id had left, its absence
    /// ends and its pins stand: the worker is back as it was, and the end
    /// of its absence would take the pins of a live worker.
    pub(crate) fn returned(&self, worker: WorkerId) {
        let mut absent = self.absent.lock();
        if !absent.is_empty() {
            absent.retain(|(id, _)| *id != worker);
        }
    }

    /// [`Self::returned`] for each of `workers`, the members a snapshot
    /// lists.
    pub(crate) fn returned_all(&self, workers: &[WorkerId]) {
        let mut absent = self.absent.lock();
        if !absent.is_empty() {
            absent.retain(|(id, _)| !workers.contains(id));
        }
    }

    /// How many workers the table holds as absent, for the cases.
    #[cfg(test)]
    pub(crate) fn absent(&self) -> usize {
        self.absent.lock().len()
    }

    /// The pins of `old` are `new`'s: a worker that left was registered
    /// again, and a registration makes a new id. One walk of the table on
    /// a rare event. The rule's recovery then rules as for two routable
    /// pins: under `original` a key goes back to the worker it was first
    /// placed on, under `latest` it stays where it was served last. A
    /// claim no dispatch confirmed follows its worker and keeps the id its
    /// lease names: the claiming decision's dispatch on the old id
    /// confirms it, and its commit elsewhere withdraws it.
    pub(crate) fn rename_worker(&self, old: WorkerId, new: WorkerId) {
        if old == WorkerId(0) || new == WorkerId(0) {
            return;
        }
        self.absent.lock().retain(|(id, _)| *id != old);
        // The same life registered again holds its pins as they stand.
        if old == new {
            return;
        }
        for shard in &self.shards {
            for entry in shard.lock().values_mut() {
                for pin in &mut entry.pins {
                    if *pin == old {
                        *pin = new;
                    }
                }
                if entry.pins[0] == entry.pins[1] {
                    entry.pins[1] = WorkerId(0);
                }
                // The lease's id stays beside it ([`Entry::leased`]).
                if entry.claim == old {
                    entry.claim = new;
                }
            }
        }
    }

    /// Drops `workers` from every key's pins: the ones that left and did
    /// not return inside the absence's time, or that a host knows return
    /// as other workers. One walk of the table for them all, on a rare
    /// event. A key whose owner went keeps its failover as the owner; a
    /// key with no pin left leaves the table and is placed afresh.
    pub(crate) fn forget_workers(&self, workers: &[WorkerId]) {
        // No pin names the empty id: a list of it alone walks nothing.
        if workers.iter().all(|&worker| worker == WorkerId(0)) {
            return;
        }
        self.absent.lock().retain(|(id, _)| !workers.contains(id));
        self.drop_pins(workers);
    }

    /// The pins of `workers` leave every key, in one walk.
    fn drop_pins(&self, workers: &[WorkerId]) {
        for shard in &self.shards {
            shard
                .lock()
                .retain(|_, entry| Self::unpinned(entry, workers));
        }
    }

    /// Takes the pins of `workers` off `entry`. False where that takes the
    /// key's last pin: it leaves the table and is placed afresh. A key
    /// that holds no pin stays: a decision read it and its claim follows
    /// ([`Self::lookup`]), and the idle keys' pass alone reclaims it.
    fn unpinned(entry: &mut Entry, workers: &[WorkerId]) -> bool {
        let held = entry.pins[0] != WorkerId(0);
        if workers.contains(&entry.pins[1]) {
            entry.pins[1] = WorkerId(0);
        }
        if workers.contains(&entry.pins[0]) {
            entry.pins = [entry.pins[1], WorkerId(0)];
        }
        !held || entry.pins[0] != WorkerId(0)
    }

    /// Whether a sweep is due: of the keys whose TTL passed, once per the
    /// table's sweep interval, or of a worker that left. A table that owes
    /// neither never walks its shards. Always inlined with its one caller
    /// ([`Self::lookup`]): the check is one compare on the pin's path.
    #[inline(always)]
    fn sweep_if_due(&self, now_micros: u64) {
        let due = self.next_sweep_micros.load(Ordering::Relaxed);
        if now_micros < due {
            return;
        }
        self.sweep(due, now_micros);
    }

    /// The sweep that is due at `due`, out of line. A leave not yet stamped
    /// takes its end, the absence's time from this decision. A worker whose
    /// absence ended is forgotten, and every key whose TTL passed where the
    /// pass over the idle keys is due: one walk of the table for both, and
    /// none where neither is owed. The pass keeps its own instant, so a
    /// sweep a leave asked for walks no shard for the keys ahead of it.
    #[cold]
    #[inline(never)]
    fn sweep(&self, due: u64, now_micros: u64) {
        let pass = self.next_pass_micros.load(Ordering::Relaxed);
        let idle = now_micros >= pass;
        let next_pass = if idle {
            now_micros.saturating_add(self.sweep_micros)
        } else {
            pass
        };
        let claimed = self.next_sweep_micros.compare_exchange(
            due,
            next_pass,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        if claimed.is_err() {
            return;
        }
        if idle {
            self.next_pass_micros.store(next_pass, Ordering::Relaxed);
        }
        let mut ended: Vec<WorkerId> = Vec::new();
        let mut absent = self.absent.lock();
        let mut next = next_pass;
        absent.retain_mut(|(worker, ends)| {
            if *ends == 0 {
                *ends = now_micros.saturating_add(self.absence_micros).max(1);
            }
            if *ends <= now_micros {
                ended.push(*worker);
                return false;
            }
            next = next.min(*ends);
            true
        });
        // Under the lock, as a leave's own store is.
        self.next_sweep_micros.store(next, Ordering::Relaxed);
        // Where an absence ended the walk runs under the list's lock: a
        // worker that joins again behind its end waits for the walk, and
        // loses no pin it is given after. A pass over the idle keys alone
        // holds nothing.
        let held = (!ended.is_empty()).then_some(absent);
        if held.is_none() && !idle {
            return;
        }
        for shard in &self.shards {
            shard.lock().retain(|_, entry| {
                let stays = !idle || entry.expires_at_micros > now_micros;
                stays && Self::unpinned(entry, &ended)
            });
        }
        drop(held);
    }

    /// The pins of `key`, for tests and dumps.
    pub fn pins(&self, key: Session) -> Option<Pins> {
        let key = key.get();
        self.shard(key).lock().get(&key).map(|e| e.pins)
    }

    /// What a host keeps of `key` beside its own pin of record: the pins,
    /// and when the table forgets the key.
    pub fn held(&self, key: Session) -> Option<(Pins, Forgets)> {
        let key = key.get();
        self.shard(key).lock().get(&key).map(|entry| {
            let forgets = match entry.expires_at_micros {
                0 => Forgets::Unread,
                u64::MAX => Forgets::Never,
                at => Forgets::At(at),
            };
            (entry.pins, forgets)
        })
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

    /// A session of the table's own cases, by its value.
    fn s(value: u64) -> Session {
        Session::from_raw(value)
    }

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
        let table = SessionTable::new(RULE.sweep_micros);
        assert_eq!(table.lookup(s(7), 10, &RULE), [WorkerId(0); 2]);
        table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(table.pins(s(7)), Some([WorkerId(1), WorkerId(0)]));
        // A hit on the preferred pin changes nothing.
        table.lookup(s(7), 20, &RULE);
        table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(table.pins(s(7)), Some([WorkerId(1), WorkerId(0)]));
        // A worker outside the pins (the pins were unroutable) leads.
        table.lookup(s(7), 30, &RULE);
        table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(2));
        assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
        // The failover taken while the preferred one is unroutable keeps the
        // order; a respill to it makes it the preferred one.
        table.lookup(s(7), 40, &RULE);
        table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
        table.lookup(s(7), 50, &RULE);
        table.dispatched(s(7), SessionVerdict::Respill, WorkerId(1));
        assert_eq!(table.pins(s(7)), Some([WorkerId(1), WorkerId(2)]));
    }

    #[test]
    fn under_original_recovery_the_first_worker_keeps_the_lead() {
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(3), 10, &ORIGINAL);
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(1));
        table.lookup(s(3), 20, &ORIGINAL);
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(2));
        assert_eq!(table.pins(s(3)), Some([WorkerId(1), WorkerId(2)]));
        // A third worker evicts the oldest.
        table.lookup(s(3), 30, &ORIGINAL);
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(3));
        assert_eq!(table.pins(s(3)), Some([WorkerId(2), WorkerId(3)]));
    }

    #[test]
    fn idle_keys_are_swept_once_the_interval_passes() {
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(1), 0, &RULE);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        table.lookup(s(2), 900, &RULE);
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(2));
        assert_eq!(table.len(), 2);
        // Within the sweep interval nothing is swept.
        table.lookup(s(5), 950, &RULE);
        assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
        // Past the interval and the first key's TTL: the first key is gone,
        // the second (touched at 900) stays.
        table.lookup(s(6), 1_100, &RULE);
        assert_eq!(table.pins(s(1)), None);
        assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(0)]));
        // Without a TTL nothing expires.
        let forever = SessionTable::new(0);
        let no_ttl = SessionRule {
            ttl_micros: None,
            ..RULE
        };
        forever.lookup(s(1), 0, &no_ttl);
        forever.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        forever.lookup(s(2), 1_000_000, &no_ttl);
        assert_eq!(forever.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
    }

    /// Two rules share one table: the sweep runs at the shortest interval
    /// among them, so a rule with an hour between sweeps never defers the
    /// expiry of another rule's keys.
    #[test]
    fn a_long_sweep_rule_does_not_defer_another_rules_expiry() {
        let hourly = SessionRule {
            ttl_micros: Some(3_600_000_000),
            sweep_micros: 3_600_000_000,
            ..RULE
        };
        let table = SessionTable::new(RULE.sweep_micros.min(hourly.sweep_micros));
        table.lookup(s(1), 0, &hourly);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        table.lookup(s(2), 0, &RULE);
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(2));
        // Past the short rule's TTL the next decision sweeps its key; the
        // hourly key stays.
        assert_eq!(table.lookup(s(2), 2_000, &RULE), [WorkerId(0); 2]);
        assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
    }

    /// A host reads a key's pins with the instant its rule forgets it, and
    /// no instant under a rule without a TTL. A key a dispatch alone wrote
    /// is held unread.
    #[test]
    fn a_key_is_held_with_the_instant_its_rule_forgets_it() {
        let table = SessionTable::new(RULE.sweep_micros);
        assert_eq!(table.held(s(1)), None);
        table.lookup(s(1), 50, &RULE);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(
            table.held(s(1)),
            Some(([WorkerId(1), WorkerId(0)], Forgets::At(1_050)))
        );
        let for_ever = SessionRule {
            ttl_micros: None,
            sweep_micros: 0,
            ..RULE
        };
        table.lookup(s(2), 50, &for_ever);
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(2));
        assert_eq!(
            table.held(s(2)),
            Some(([WorkerId(2), WorkerId(0)], Forgets::Never))
        );
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(3));
        assert_eq!(
            table.held(s(3)),
            Some(([WorkerId(3), WorkerId(0)], Forgets::Unread))
        );
    }

    /// A key whose TTL passed is forgotten at its next lookup even when no
    /// sweep ran since: the compare is the key's own, the walk only reclaims.
    #[test]
    fn an_expired_key_is_forgotten_at_its_lookup_before_any_sweep() {
        let rarely = SessionTable::new(1_000_000);
        rarely.lookup(s(1), 0, &RULE);
        rarely.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(rarely.lookup(s(1), 500, &RULE), [WorkerId(1), WorkerId(0)]);
        assert_eq!(rarely.lookup(s(1), 2_000, &RULE), [WorkerId(0); 2]);
    }

    #[test]
    fn an_older_lookup_cannot_shorten_a_newer_touch() {
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(7), 500, &RULE);
        table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));

        table.lookup(s(7), 1_000, &RULE);
        table.lookup(s(7), 600, &ORIGINAL);

        assert_eq!(
            table.held(s(7)),
            Some(([WorkerId(1), WorkerId(0)], Forgets::At(2_000)))
        );
        assert_eq!(
            table.claim(s(7), [WorkerId(1), WorkerId(0)], WorkerId(2)),
            Claim::Made(WorkerId(2))
        );
        assert_eq!(table.lookup(s(7), 1_600, &RULE), [WorkerId(2), WorkerId(1)]);
    }

    /// A retry's claim survives the commit of an older request on the owner:
    /// only the claiming decision settles its claim, so the retries after it
    /// read the claim as the pin of record instead of claiming elsewhere, and
    /// the claim is confirmed when its own commit lands.
    #[test]
    fn an_older_requests_commit_on_the_owner_leaves_a_retrys_claim_standing() {
        let table = SessionTable::new(RULE.sweep_micros);
        let seen = table.lookup(s(1), 0, &ORIGINAL);
        assert_eq!(
            table.claim(s(1), seen, WorkerId(1)),
            Claim::Made(WorkerId(1))
        );
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(1)), WorkerId(1));
        // An older request decided on the owner is still in flight when a
        // retry that cannot use the owner claims B.
        let seen = table.lookup(s(1), 10, &ORIGINAL);
        assert_eq!(
            table.claim(s(1), seen, WorkerId(2)),
            Claim::Made(WorkerId(2))
        );
        assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(
            table.pins(s(1)),
            Some([WorkerId(1), WorkerId(2)]),
            "the older request's commit on the owner leaves the claim standing"
        );
        // The next retry reads B as the pin of record; the claiming retry's
        // own commit confirms it.
        assert_eq!(
            table.lookup(s(1), 20, &ORIGINAL),
            [WorkerId(1), WorkerId(2)]
        );
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(2)), WorkerId(2));
        assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
    }

    #[test]
    fn an_empty_key_or_target_writes_nothing() {
        let table = SessionTable::new(RULE.sweep_micros);
        table.dispatched(s(0), SessionVerdict::Pinned, WorkerId(1));
        table.dispatched(s(4), SessionVerdict::Pinned, WorkerId(0));
        assert!(table.is_empty());
    }

    /// Two decisions on one key interleave: A reads, B reads, A notes its
    /// respill, B dispatches onto the failover pin (a hit, its instant is
    /// not the note's), then A dispatches onto it (the respill). The note is
    /// keyed by the decision instant and consumed by its own dispatch alone.
    #[test]
    fn a_respill_verdict_travels_with_its_decision_past_another_decisions_reading() {
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(9), 10, &RULE);
        table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(1));
        table.lookup(s(9), 20, &RULE);
        table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(2));
        assert_eq!(table.pins(s(9)), Some([WorkerId(2), WorkerId(1)]));
        // A decides a respill onto the failover, B reads the key and takes
        // the failover as a plain hit, B commits first: B's hit leaves the
        // pins, A's respill fronts the failover whatever came between.
        table.lookup(s(9), 30, &RULE);
        table.lookup(s(9), 31, &RULE);
        table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(
            table.pins(s(9)),
            Some([WorkerId(2), WorkerId(1)]),
            "B's dispatch onto the failover is a hit"
        );
        table.dispatched(s(9), SessionVerdict::Respill, WorkerId(1));
        assert_eq!(
            table.pins(s(9)),
            Some([WorkerId(1), WorkerId(2)]),
            "A's dispatch is the respill"
        );
    }

    #[test]
    fn the_first_claim_stands_for_a_fresh_key_and_for_a_key_whose_pins_are_down() {
        let table = SessionTable::new(RULE.sweep_micros);
        let none = [WorkerId(0); 2];
        let seen = table.lookup(s(5), 10, &ORIGINAL);
        assert_eq!(
            table.claim(s(5), seen, WorkerId(3)),
            Claim::Made(WorkerId(3)),
            "the first"
        );
        assert_eq!(
            table.claim(s(5), none, WorkerId(4)),
            Claim::Standing(WorkerId(3), [WorkerId(3), WorkerId(0)]),
            "follows it"
        );
        assert_eq!(table.pins(s(5)), Some([WorkerId(3), WorkerId(0)]));
        table.dispatched(s(5), SessionVerdict::Claimed(WorkerId(3)), WorkerId(3));
        table.dispatched(s(5), SessionVerdict::Pinned, WorkerId(4));
        assert_eq!(
            table.pins(s(5)),
            Some([WorkerId(3), WorkerId(4)]),
            "a loser dispatched elsewhere is the failover"
        );
        // Both pins down: two decisions read [3, 4]; the first claim appends
        // its worker by the recovery (the oldest slot goes), the second
        // follows the newcomer.
        let seen = table.lookup(s(5), 20, &ORIGINAL);
        assert_eq!(
            table.claim(s(5), seen, WorkerId(7)),
            Claim::Made(WorkerId(7))
        );
        assert_eq!(table.pins(s(5)), Some([WorkerId(4), WorkerId(7)]));
        assert_eq!(
            table.claim(s(5), seen, WorkerId(8)),
            Claim::Standing(WorkerId(7), [WorkerId(4), WorkerId(7)]),
            "the newcomer"
        );
        // Under the latest-first recovery the claim leads at once.
        let seen = table.lookup(s(6), 30, &RULE);
        assert_eq!(
            table.claim(s(6), seen, WorkerId(1)),
            Claim::Made(WorkerId(1))
        );
        assert_eq!(table.pins(s(6)), Some([WorkerId(1), WorkerId(0)]));
        assert_eq!(
            table.claim(s(0), none, WorkerId(9)),
            Claim::Standing(WorkerId(9), none),
            "no key, no claim"
        );
        assert_eq!(table.len(), 2);
    }

    /// Another request's respill turns the pins behind a decision's read:
    /// the claim is not taken and no pin came in. The answer is the
    /// caller's own worker with the pins the key held, and nothing is
    /// recorded. Another decision that reads the same pins gets its own
    /// worker: the answer is no pin to follow.
    #[test]
    fn a_claim_that_finds_no_new_pin_names_the_callers_own_worker() {
        let table = SessionTable::new(RULE.sweep_micros);
        let _ = table.restore(s(3), 10, &RULE, WorkerId(1));
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(2));
        let seen = table.lookup(s(3), 20, &RULE);
        assert_eq!(seen, [WorkerId(2), WorkerId(1)]);
        table.dispatched(s(3), SessionVerdict::Respill, WorkerId(1));
        let held = [WorkerId(1), WorkerId(2)];
        assert_eq!(
            table.claim(s(3), seen, WorkerId(7)),
            Claim::Standing(WorkerId(7), held)
        );
        assert_eq!(table.pins(s(3)), Some(held), "nothing is recorded");
        assert_eq!(
            Claim::standing(&held, &seen, WorkerId(8)),
            Some(WorkerId(8))
        );
        assert_eq!(Claim::standing(&seen, &seen, WorkerId(8)), None);
    }

    /// The owner is down, the decision claims B, the commit refuses B and
    /// dispatches the alternate C: the claim is withdrawn and the owner keeps
    /// its slot (B served nothing); a dispatch to the claimed worker confirms
    /// the claim instead, and the normal rule follows after it.
    #[test]
    fn a_claim_whose_commit_landed_elsewhere_is_withdrawn_before_the_pins_move() {
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(1), 10, &ORIGINAL);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        let seen = table.lookup(s(1), 20, &ORIGINAL);
        assert_eq!(
            table.claim(s(1), seen, WorkerId(2)),
            Claim::Made(WorkerId(2))
        );
        assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(2)), WorkerId(3));
        assert_eq!(
            table.pins(s(1)),
            Some([WorkerId(1), WorkerId(3)]),
            "the owner keeps the lead, the alternate is the failover"
        );
        // Under the latest-first recovery the claim led; the alternate leads
        // instead and the owner stays the failover.
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(2), 10, &RULE);
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(1));
        let seen = table.lookup(s(2), 20, &RULE);
        assert_eq!(
            table.claim(s(2), seen, WorkerId(2)),
            Claim::Made(WorkerId(2))
        );
        assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(1)]));
        table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(2)), WorkerId(3));
        assert_eq!(table.pins(s(2)), Some([WorkerId(3), WorkerId(1)]));
        // A dispatch to the claimed worker confirms the claim.
        let seen = table.lookup(s(2), 30, &RULE);
        assert_eq!(
            table.claim(s(2), seen, WorkerId(4)),
            Claim::Made(WorkerId(4))
        );
        table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(4)), WorkerId(4));
        assert_eq!(table.pins(s(2)), Some([WorkerId(4), WorkerId(3)]));
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(5));
        assert_eq!(
            table.pins(s(2)),
            Some([WorkerId(5), WorkerId(4)]),
            "a confirmed claim served: the normal rule moves it"
        );
    }

    #[test]
    fn a_pin_another_request_already_used_survives_the_claimants_commit_elsewhere() {
        // X claims A for a fresh key; Y finds A pinned and commits there; A
        // goes unroutable and X's own commit lands on B. A was served, so it
        // stays: [A, B], and the recovered A keeps the key.
        let table = SessionTable::new(RULE.sweep_micros);
        let seen = table.lookup(s(1), 0, &ORIGINAL);
        assert_eq!(
            table.claim(s(1), seen, WorkerId(1)),
            Claim::Made(WorkerId(1))
        );
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
        assert_eq!(
            table.pins(s(1)),
            Some([WorkerId(1), WorkerId(2)]),
            "a pin another request already used survives the claimant's commit elsewhere"
        );
        assert_eq!(
            table.lookup(s(1), 10, &ORIGINAL),
            [WorkerId(1), WorkerId(2)]
        );
        // Under the latest-first recovery the alternate leads and A stays the
        // failover instead of vanishing.
        let table = SessionTable::new(RULE.sweep_micros);
        let seen = table.lookup(s(2), 0, &RULE);
        assert_eq!(
            table.claim(s(2), seen, WorkerId(1)),
            Claim::Made(WorkerId(1))
        );
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(1));
        table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
        assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(1)]));
    }

    #[test]
    fn a_commit_elsewhere_by_another_request_leaves_the_claim_unconfirmed() {
        // The owner is pinned; a retry claims C; an older request commits on
        // the owner (not on C): the claim stands unconfirmed, so the retry's
        // own commit on D still withdraws C.
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(1), 10, &ORIGINAL);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        let seen = table.lookup(s(1), 20, &ORIGINAL);
        assert_eq!(
            table.claim(s(1), seen, WorkerId(2)),
            Claim::Made(WorkerId(2))
        );
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(2)), WorkerId(3));
        assert_eq!(
            table.pins(s(1)),
            Some([WorkerId(1), WorkerId(3)]),
            "a commit elsewhere by another request confirms nothing"
        );
    }

    #[test]
    fn a_pin_of_record_is_restored_where_the_key_has_none_and_is_no_claim() {
        let table = SessionTable::new(RULE.sweep_micros);
        assert_eq!(table.lookup(s(7), 10, &RULE), [WorkerId(0); 2]);
        assert_eq!(
            table.restore(s(7), 10, &RULE, WorkerId(3)),
            [WorkerId(3), WorkerId(0)]
        );
        // A second restore, of another worker, keeps the first: two first
        // requests of one key agree.
        assert_eq!(
            table.restore(s(7), 11, &RULE, WorkerId(4)),
            [WorkerId(3), WorkerId(0)]
        );
        // It is a pin a dispatch made, no claim: a commit elsewhere under a
        // claim's verdict withdraws nothing, and the pins move by the rule.
        table.dispatched(s(7), SessionVerdict::Claimed(WorkerId(3)), WorkerId(5));
        assert_eq!(table.pins(s(7)), Some([WorkerId(5), WorkerId(3)]));
        // An empty key or worker writes nothing.
        assert_eq!(
            table.restore(s(8), 10, &RULE, WorkerId(0)),
            [WorkerId(0); 2]
        );
        assert_eq!(table.pins(s(8)), None);
    }

    #[test]
    fn a_restored_pin_on_a_key_the_table_dropped_lives_by_the_rules_ttl() {
        // The key's entry left the table between the decision's read and
        // the write (the end of an absence took its last pin): the pin
        // restored is the key's from that instant, for one TTL.
        let table = SessionTable::new(RULE.sweep_micros);
        assert_eq!(
            table.restore(s(9), 10, &RULE, WorkerId(3)),
            [WorkerId(3), WorkerId(0)]
        );
        assert_eq!(table.lookup(s(9), 11, &RULE), [WorkerId(3), WorkerId(0)]);
        assert_eq!(table.lookup(s(9), 1_011, &RULE), [WorkerId(0); 2]);
    }

    #[test]
    fn a_forgotten_worker_leaves_every_pin_and_a_key_with_no_pin_left_leaves_the_table() {
        let table = SessionTable::new(RULE.sweep_micros);
        for (key, first, second) in [(1, 1, 2), (2, 2, 0), (3, 2, 1), (4, 3, 0)] {
            table.lookup(s(key), 10, &ORIGINAL);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(first));
            if second != 0 {
                table.lookup(s(key), 20, &ORIGINAL);
                table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(second));
            }
        }
        assert_eq!(table.pins(s(3)), Some([WorkerId(2), WorkerId(1)]));
        table.forget_workers(&[WorkerId(2)]);
        assert_eq!(
            table.pins(s(1)),
            Some([WorkerId(1), WorkerId(0)]),
            "the failover slot went"
        );
        assert_eq!(table.pins(s(2)), None, "its only pin went");
        assert_eq!(
            table.pins(s(3)),
            Some([WorkerId(1), WorkerId(0)]),
            "the failover is the owner"
        );
        assert_eq!(
            table.pins(s(4)),
            Some([WorkerId(3), WorkerId(0)]),
            "untouched"
        );
        assert_eq!(table.len(), 3);
        table.forget_workers(&[WorkerId(0)]);
        assert_eq!(table.len(), 3, "the empty id is never a pin");
    }

    /// The pass over the idle keys reclaims what expired alone. A key a
    /// decision read and has not claimed yet holds no pin, and stays.
    #[test]
    fn the_pass_keeps_a_key_read_and_not_yet_claimed() {
        let rule = SessionRule {
            ttl_micros: Some(100),
            sweep_micros: 1_000,
            ..ORIGINAL
        };
        let table = SessionTable::new(rule.sweep_micros);
        // The first pass: the next is due at 1,010.
        table.lookup(s(1), 10, &rule);
        assert_eq!(table.lookup(s(1), 1_005, &rule), [WorkerId(0); 2]);
        // Another key's decision runs the pass.
        table.lookup(s(2), 1_010, &rule);
        let claim = table.claim(s(1), [WorkerId(0); 2], WorkerId(3));
        assert_eq!(claim, Claim::Made(WorkerId(3)));
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(3)), WorkerId(3));
        assert_eq!(table.lookup(s(1), 1_020, &rule), [WorkerId(3), WorkerId(0)]);
    }

    /// A walk that forgets a worker takes a key with no pin left out of
    /// the table, and leaves the key a decision read and has not claimed.
    #[test]
    fn a_forgotten_workers_walk_keeps_a_key_read_and_not_yet_claimed() {
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(1), 10, &RULE);
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(4));
        table.forget_workers(&[WorkerId(0)]);
        table.forget_workers(&[WorkerId(4)]);
        assert_eq!(table.pins(s(1)), Some([WorkerId(0); 2]));
        assert_eq!(table.pins(s(2)), None);
    }

    /// A claim follows its worker to its new id and keeps the id its lease
    /// names. The claiming decision's own dispatch, which names the old
    /// id, confirms the renamed claim. Its commit elsewhere withdraws it,
    /// and the worker that served the key is pinned alone.
    #[test]
    fn a_claim_renamed_with_its_worker_is_confirmed_or_withdrawn_by_its_lease() {
        let table = SessionTable::new(ORIGINAL.sweep_micros);
        for key in [7, 8] {
            assert_eq!(table.lookup(s(key), 10, &ORIGINAL), [WorkerId(0); 2]);
            let claim = table.claim(s(key), [WorkerId(0); 2], WorkerId(1));
            assert_eq!(claim, Claim::Made(WorkerId(1)));
        }
        table.rename_worker(WorkerId(1), WorkerId(11));
        let renamed = Some([WorkerId(11), WorkerId(0)]);
        assert_eq!((table.pins(s(7)), table.pins(s(8))), (renamed, renamed));
        // Its own dispatch names the id the worker had: confirmed.
        table.dispatched(s(7), SessionVerdict::Claimed(WorkerId(1)), WorkerId(1));
        assert_eq!(table.pins(s(7)), renamed);
        // A pin a dispatch confirmed is withdrawn by no later commit.
        table.dispatched(s(7), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
        assert_eq!(table.pins(s(7)), Some([WorkerId(11), WorkerId(2)]));
        // The other key's commit lands elsewhere: withdrawn.
        table.dispatched(s(8), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
        assert_eq!(table.pins(s(8)), Some([WorkerId(2), WorkerId(0)]));
    }

    /// A worker that left keeps its pins. Its key is served elsewhere
    /// meanwhile, and the worker registered again under a new id leads the
    /// key again under `original`.
    #[test]
    fn a_worker_that_left_keeps_its_pins_and_takes_them_back_under_its_new_id() {
        let table = SessionTable::new(RULE.sweep_micros);
        for (key, worker) in [(1, 1), (2, 2)] {
            table.lookup(s(key), 10, &ORIGINAL);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
        }
        table.left(WorkerId(1));
        assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
        table.lookup(s(1), 20, &ORIGINAL);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
        assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(2)]));
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(2)]));
        assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(0)]));
    }

    /// Under `latest` the worker that served the key in the absence keeps
    /// the lead, and the one registered again is its failover.
    #[test]
    fn under_latest_recovery_a_worker_registered_again_is_the_failover() {
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(1), 10, &RULE);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        table.left(WorkerId(1));
        table.lookup(s(1), 20, &RULE);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(11)]));
    }

    /// A worker that stays away is forgotten once the absence's time
    /// passes on the decision clock, counted from the first decision
    /// behind the leave. A table no rule sweeps counts it too.
    #[test]
    fn a_worker_that_stays_away_is_forgotten_at_the_absences_end() {
        for sweep in [RULE.sweep_micros, 0] {
            let rule = SessionRule {
                ttl_micros: None,
                sweep_micros: sweep,
                ..ORIGINAL
            };
            let table = SessionTable::new(sweep).with_absence(500);
            for (key, worker) in [(1, 1), (2, 2)] {
                table.lookup(s(key), 10, &rule);
                table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
            }
            table.left(WorkerId(1));
            // The decision at 100 stamps the end: 600.
            table.lookup(s(2), 100, &rule);
            table.lookup(s(2), 599, &rule);
            let stands = Some([WorkerId(1), WorkerId(0)]);
            assert_eq!(table.pins(s(1)), stands, "inside its time, sweep {sweep}");
            table.lookup(s(2), 600, &rule);
            assert_eq!(table.pins(s(1)), None, "its only pin went, sweep {sweep}");
            assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(0)]));
        }
    }

    /// A worker registered again inside its time is not forgotten at its
    /// end, under a new id or under the one it had.
    #[test]
    fn a_worker_registered_again_in_time_is_not_forgotten_at_the_absences_end() {
        let rule = SessionRule {
            ttl_micros: None,
            sweep_micros: 0,
            ..ORIGINAL
        };
        let table = SessionTable::new(0).with_absence(500);
        table.lookup(s(1), 10, &rule);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        table.left(WorkerId(1));
        table.lookup(s(1), 100, &rule);
        table.rename_worker(WorkerId(1), WorkerId(11));
        table.lookup(s(1), 10_000, &rule);
        assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(0)]));
        table.left(WorkerId(11));
        table.rename_worker(WorkerId(11), WorkerId(11));
        table.lookup(s(1), 20_000, &rule);
        assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(0)]));
    }

    /// A join under the id that left ends its absence, by its own word or
    /// by a snapshot that lists it: no end of the absence takes the pins of
    /// the worker that is back.
    #[test]
    fn a_join_under_the_id_that_left_ends_its_absence() {
        let rule = SessionRule {
            ttl_micros: None,
            sweep_micros: 0,
            ..ORIGINAL
        };
        let table = SessionTable::new(0).with_absence(500);
        for (key, worker) in [(1, 1), (2, 2)] {
            table.lookup(s(key), 10, &rule);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
        }
        table.left(WorkerId(1));
        table.returned(WorkerId(1));
        assert_eq!(table.absent(), 0);
        table.lookup(s(2), 100, &rule);
        table.lookup(s(2), 10_000, &rule);
        assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
        table.left(WorkerId(1));
        table.left(WorkerId(2));
        table.returned_all(&[WorkerId(2)]);
        assert_eq!(table.absent(), 1, "the one the snapshot does not list");
        table.lookup(s(2), 20_000, &rule);
        table.lookup(s(2), 30_000, &rule);
        assert_eq!(table.pins(s(2)), Some([WorkerId(2), WorkerId(0)]));
        assert_eq!(table.pins(s(1)), None, "the worker that stayed away");
    }

    /// A leave asks for a sweep, and that sweep walks no idle key ahead of
    /// the pass over them: the pass keeps its own instant. The key whose
    /// TTL passed stays until the pass is due, as with no leave.
    #[test]
    fn a_leave_walks_no_idle_key_before_their_pass_is_due() {
        let rule = SessionRule {
            ttl_micros: Some(100),
            sweep_micros: 1_000,
            ..ORIGINAL
        };
        let table = SessionTable::new(rule.sweep_micros).with_absence(5_000);
        // The first lookup runs the pass: the next is due at 1,010.
        for (key, worker) in [(1, 1), (2, 2)] {
            table.lookup(s(key), 10, &rule);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
        }
        table.left(WorkerId(9));
        // Key 1 is idle past its TTL at 300, and the leave's sweep runs.
        table.lookup(s(2), 300, &rule);
        assert_eq!(table.absent(), 1, "the leave is stamped");
        assert!(table.pins(s(1)).is_some(), "no pass before 1,010");
        table.lookup(s(2), 1_010, &rule);
        assert_eq!(table.pins(s(1)), None, "the pass at its own instant");
    }

    /// A table that holds no key files no absence: no pin names the
    /// worker, and no decision would end the entry.
    #[test]
    fn a_table_that_holds_no_key_files_no_absence() {
        let table = SessionTable::new(0).with_absence(500);
        table.left(WorkerId(1));
        assert_eq!(table.absent(), 0);
        table.lookup(s(1), 10, &ORIGINAL);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
        table.left(WorkerId(1));
        assert_eq!(table.absent(), 1);
    }

    /// The absent workers are bounded: at the bound the one that left
    /// first is forgotten at once, its pins with it.
    #[test]
    fn at_the_bound_of_absent_workers_the_first_to_leave_is_forgotten() {
        let rule = SessionRule {
            ttl_micros: None,
            sweep_micros: 0,
            ..ORIGINAL
        };
        let table = SessionTable::new(0).with_absence(500);
        for (key, worker) in [(1, 1), (2, 9_999)] {
            table.lookup(s(key), 10, &rule);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
        }
        let bound = ABSENT_WORKERS as u64;
        for worker in 1..=bound {
            table.left(WorkerId(worker));
        }
        assert_eq!(table.absent(), ABSENT_WORKERS);
        assert_eq!(table.pins(s(1)), Some([WorkerId(1), WorkerId(0)]));
        table.left(WorkerId(bound + 1));
        assert_eq!(table.absent(), ABSENT_WORKERS);
        assert_eq!(table.pins(s(1)), None, "the first to leave");
        assert_eq!(table.pins(s(2)), Some([WorkerId(9_999), WorkerId(0)]));
    }
}
