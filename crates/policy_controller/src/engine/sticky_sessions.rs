//! Sticky sessions (design 4.3, 4.4 and 4.6): a binding's `sessions` rule and
//! the session table it reads. A keyed request stays on the worker its key
//! was placed on; the table is plugin state the lifecycle feeds (the pin is
//! written at the commit, `on_dispatched`), keyed by the session's value
//! (`RequestFacts::session`: one key names one session per model and phase,
//! by one formula on every host). A decision reads the table once and
//! carries what it did with the key as its own verdict (on the pick, then
//! the lease), so its commit moves the pins by that verdict alone: a hit
//! leaves them, a respill fronts the worker taken. Two writes come before a
//! commit. A fresh key's decision claims the worker it chose, so concurrent
//! first requests agree on one. A decision that met its key's pin at the cap
//! fronts the worker it placed the request on, so the key's next request
//! follows the move ahead of that dispatch, as a gateway's sticky map moves
//! a pin where it selects the worker.

use std::{
    collections::hash_map,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use parking_lot::{Mutex, MutexGuard, RwLock};
use rustc_hash::FxHashMap;

use crate::{
    document::spec::{Assignment, Recovery, RulePlaced},
    engine::view::{Session, WorkerId},
};

/// A binding's sessions rule as the engine runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionRule {
    pub assignment: Assignment,
    /// Requests of one key a worker carries before the next is placed
    /// elsewhere; `0` for no cap.
    pub inflight_cap: u16,
    /// Idle time after which a key's pins are forgotten, at the rule's next
    /// sweep; `None` for never.
    pub ttl_micros: Option<u64>,
    /// How often idle keys are swept, on the decision clock: a key is
    /// forgotten at the first multiple of this interval at or after the end
    /// of its idle time. `0` for never (a rule without a TTL; under a TTL
    /// the key ends with its idle time). The table walks its shards at the
    /// shortest interval among its rules ([`SessionTable::new`]).
    pub sweep_micros: u64,
    pub recovery: Recovery,
    /// Whether a request the rule placed alone is the profile's own: a
    /// pick the rule placed carries the word to the engine that tells the
    /// stages of it.
    pub rule_placed: RulePlaced,
}

impl SessionRule {
    /// The first sweep at or after `idle_ends`, the end of a key's idle time,
    /// under the interval `sweep_micros`: the instant the key is forgotten.
    /// Sweeps fall on its multiples on the decision clock, whichever pass
    /// walks the table. No interval ends the key with its idle time, and so
    /// does a sweep at the clock's last instant or past it: that instant reads
    /// as no end ([`SessionExpiry::Never`]). Out of line: under an interval a
    /// key comes here once a sweep at most, under none at every read.
    #[cold]
    #[inline(never)]
    fn sweep_at_or_after(sweep_micros: u64, idle_ends: u64) -> u64 {
        // No multiple of an interval of zero, and none past the clock.
        match idle_ends.checked_next_multiple_of(sweep_micros) {
            Some(sweep) if sweep != u64::MAX => sweep,
            _ => idle_ends,
        }
    }
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
    /// request on another worker under the cap. The decision fronts that
    /// worker where it is made, and the commit fronts the worker it lands
    /// on.
    Respill,
    /// The key had no routable pin and the decision claimed the worker it
    /// placed the request on: that decision's commit confirms the claim on
    /// the worker, or withdraws it when the dispatch landed elsewhere.
    Claimed(WorkerId),
}

/// A [`SessionVerdict`] without the worker a claim names: what a pair's
/// decode side keeps of its own verdict, in a byte its pair had spare
/// ([`Pair::decode_verdict`]). A claim names the decode the pair names, so
/// the whole verdict is read back with that worker ([`Self::on`]).
///
/// It is a byte with a name for each verdict, and no enum: every value of
/// the byte is a verdict's or reads as none, so the byte tells nothing of
/// whether a target names a pair, and a pick is laid out as it was.
///
/// [`Pair::decode_verdict`]: crate::engine::decision::Pair::decode_verdict
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeVerdict(u8);

impl DecodeVerdict {
    /// [`SessionVerdict::None`].
    pub const NONE: Self = Self(0);
    /// [`SessionVerdict::Pinned`].
    pub const PINNED: Self = Self(1);
    /// [`SessionVerdict::Respill`].
    pub const RESPILL: Self = Self(2);
    /// [`SessionVerdict::Claimed`], of the decode the pair names.
    pub const CLAIMED: Self = Self(3);

    /// The whole verdict, where a claim names `worker`.
    #[must_use]
    pub const fn on(self, worker: WorkerId) -> SessionVerdict {
        match self {
            Self::PINNED => SessionVerdict::Pinned,
            Self::RESPILL => SessionVerdict::Respill,
            Self::CLAIMED => SessionVerdict::Claimed(worker),
            _ => SessionVerdict::None,
        }
    }

    /// What a pair keeps of its decode side's `verdict`: a claim's worker
    /// is the decode the pair names, the first target of that side's
    /// decision.
    #[must_use]
    pub(crate) const fn of(verdict: SessionVerdict) -> Self {
        match verdict {
            SessionVerdict::None => Self::NONE,
            SessionVerdict::Pinned => Self::PINNED,
            SessionVerdict::Respill => Self::RESPILL,
            SessionVerdict::Claimed(_) => Self::CLAIMED,
        }
    }
}

/// The pins of one key: the preferred worker first, the failover second;
/// `WorkerId(0)` is an empty slot.
pub type SessionPins = [WorkerId; 2];

/// When a table forgets a key it holds, on the decision clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionExpiry {
    /// At this instant: the key's rule has a TTL.
    At(u64),
    /// At no instant: the key's rule has no TTL.
    Never,
    /// No decision has read the key since its entry was made: a dispatch
    /// made the entry anew, behind the end of an absence, a forget or the
    /// key's own end, and the next decision's read starts the key over.
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
    /// decision read, the worker is a pin of the key already under another
    /// id, or nothing was recorded (an empty key or target).
    /// It names the worker the decision follows ([`Claim::standing`]) and
    /// the pins the key held: a shadow reads those against its own pins
    /// and its own pick.
    Standing(WorkerId, SessionPins),
    /// The table is full and does not hold the key: nothing is recorded of
    /// it, and its decision keeps the placement it made.
    NoRoom,
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
    pub(crate) fn standing(
        held: &SessionPins,
        seen: &SessionPins,
        target: WorkerId,
    ) -> Option<WorkerId> {
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
    pins: SessionPins,
    /// The instant the key's pins expire on the decision clock: its rule's
    /// first sweep at or after the end of its idle time, or an end another
    /// interval wrote less than one interval past that ([`Entry::touch`]).
    /// `u64::MAX` under a rule without a TTL, and `0` until a decision
    /// reads the key.
    expires_at_micros: u64,
    /// The newest decision instant that set the expiry and recovery rule.
    touched_at_micros: u64,
    recovery: Recovery,
    /// The pin a decision's claim wrote before its commit; `WorkerId(0)`
    /// when none stands. Any dispatch to it confirms it (the key was served
    /// there), the claiming decision's dispatch elsewhere withdraws it while
    /// it stands unconfirmed, and no other dispatch elsewhere touches it; a
    /// claim whose commit never comes stands until the TTL sweeps the key.
    /// A pair's decision moves its own claim to the prefill its pair formed
    /// on, and withdraws it where none formed ([`SessionTable::move_claim`]).
    /// A rename's step takes the mark where the key held the claim's
    /// worker under another id as well ([`Entry::follow`]).
    claim: WorkerId,
    /// The id the claim's lease names: the id the claiming decision read
    /// for its worker, read while a claim stands. It differs from `claim`
    /// where the worker was registered again, under the claim
    /// ([`SessionTable::rename_worker`]) or between the decision's read
    /// of its rows and its claim ([`SessionTable::stands`]).
    leased: WorkerId,
}

impl Entry {
    fn touch(&mut self, now_micros: u64, rule: &SessionRule) {
        if now_micros < self.touched_at_micros {
            return;
        }
        match rule.ttl_micros {
            Some(ttl) => {
                let idle_ends = now_micros.saturating_add(ttl);
                // A key's end is its rule's first sweep at or after the end
                // of its idle time. An end this interval wrote that is less
                // than one interval above this one is that sweep still: it
                // is kept, and no division runs. An end another interval
                // wrote is kept inside the same span, another rule's or the
                // one a document before set: the key's end then stands at
                // or after the end of its idle time and less than one
                // interval past it, and is no sweep of this rule's. Past
                // the span this rule's sweep replaces the end.
                let ahead = self.expires_at_micros.checked_sub(idle_ends);
                if ahead.is_none_or(|ahead| ahead >= rule.sweep_micros) {
                    self.expires_at_micros =
                        SessionRule::sweep_at_or_after(rule.sweep_micros, idle_ends);
                }
            }
            None => self.expires_at_micros = u64::MAX,
        }
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

    /// The step a rename takes for one key, by its walk of the shards or
    /// by a write of the key that asks ahead of the walk
    /// ([`SessionTable::standing`]): each pin and the claim's mark are
    /// read by the id that stands, as `stands` says it, and two pins of one
    /// worker are made one. The mark goes where the key held the claim's
    /// worker under another id as well: that pin is no claim, so the key
    /// holds the worker as a pin, and the claiming decision's commit
    /// elsewhere withdraws nothing. A claim written ahead of the table's
    /// word of a rename meets the worker's pin so: it found no rename to
    /// ask. So does a mark whose own pin a later write dropped. The
    /// lease's id stays beside the mark ([`Entry::leased`]).
    fn follow(&mut self, stands: impl Fn(WorkerId) -> WorkerId) {
        let claim = self.claim;
        let claimed = stands(claim);
        for pin in &mut self.pins {
            let id = stands(*pin);
            if id == claimed && *pin != claim {
                self.claim = WorkerId(0);
            }
            if id != *pin {
                *pin = id;
            }
        }
        if self.pins[0] == self.pins[1] {
            self.pins[1] = WorkerId(0);
        }
        if self.claim != WorkerId(0) {
            self.claim = claimed;
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

/// Shards of a table no host sized ([`SessionTable::new`]): keyed
/// decisions on different keys take different locks.
const SHARDS: usize = 16;

/// The shards of a table whose keyed decisions run on every core of the
/// host at once ([`SessionTable::with_shards`]): four a core, rounded up to
/// a power of two, so two decisions of different keys seldom meet on one
/// lock.
pub(crate) fn shards_for_this_host() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    cores.saturating_mul(4).next_power_of_two()
}

/// One shard of the table, on cache lines of its own: a decision whose key
/// falls in the next shard writes neither this shard's lock nor its map's
/// header.
#[derive(Debug, Default)]
#[repr(align(128))]
struct Shard(Mutex<FxHashMap<u64, Entry>>);

/// `count` empty shards.
fn empty_shards(count: usize) -> Box<[Shard]> {
    (0..count).map(|_| Shard::default()).collect()
}

/// The most keys a table holds. A full table records no further key
/// ([`SessionTable::slot`]): a request of such a key is placed with no pin
/// until keys leave the table.
const KEYS: usize = 4_194_304;

/// How many keys a table holds, on cache lines of its own: a key that enters
/// or leaves writes it, and a decision of a key the table holds reads none of
/// its lines.
#[derive(Debug, Default)]
#[repr(align(128))]
struct Held(AtomicUsize);

/// How long a worker that left keeps its pins where no rule says a TTL:
/// four hours, the TTL a rule takes where its document names none.
pub const ABSENT_WORKER_PIN_TTL_MICROS: u64 = 14_400 * 1_000_000;

/// The most workers a table holds as absent ([`SessionTable::left`]): past
/// it the one that left first is forgotten at once, so the list is bounded
/// where no decision stamps or ends an absence.
pub(crate) const ABSENT_WORKERS: usize = 4_096;

/// The most ids a table remembers workers had before they were registered
/// again ([`SessionTable::rename_worker`]): a fleet's worth, as the absent
/// list holds. At the bound the list starts over.
pub(crate) const RENAMED_WORKERS: usize = 4_096;

/// The session table: the pins of the keys it holds, swept on the decision
/// clock.
///
/// A decision of a key takes one lock, the mutex of its key's shard. The
/// read of the pins is a write: it renews the key's idle time, and starts a
/// key over that is past its end, so a shard is behind a mutex and not a
/// lock readers share. The claim of a key no pin takes is under the same
/// mutex, and so is a dispatch. Two decisions meet on a mutex only where
/// their keys fall in one shard.
#[derive(Debug)]
pub struct SessionTable {
    /// The shards, a power of two of them: a key's shard is named by the
    /// key's bits above its low word ([`Self::shard`]).
    shards: Box<[Shard]>,
    /// The keys the table holds: a key is counted before its entry is made
    /// and given back behind the walk that takes it, each under its shard's
    /// mutex, so the count is never under the entries.
    held: Held,
    /// The decision clock at the next sweep, of the idle keys or of a
    /// worker that left: `u64::MAX` while neither is owed.
    next_sweep_micros: AtomicU64,
    /// The decision clock at the next pass over the idle keys, read by the
    /// sweep alone: a sweep a leave asked for walks no shard for the keys
    /// before their pass is due. `u64::MAX` where no rule sweeps.
    next_pass_micros: AtomicU64,
    /// The sweep interval: the shortest among the rules of the document in
    /// force, so no rule's interval defers another rule's expiry; `0`
    /// never sweeps (no rule has had a TTL). A word a document re-sets
    /// ([`Self::set_times`]), read by the sweep, and by an engine that
    /// hands its document's times to the table a host keeps
    /// (`Engine::share_sessions`).
    sweep_micros: AtomicU64,
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
    /// How long a worker that left keeps its pins: the document's in force
    /// ([`Self::set_times`]), read where a sweep stamps a leave.
    absence_micros: AtomicU64,
    /// The ids workers had before they were registered again, each with
    /// the id that stands for it ([`Self::rename_worker`]). A dispatch
    /// names the id its lease was issued on, and one that lands behind
    /// the rename is the standing worker's. [`RENAMED_WORKERS`] at most.
    /// A write of a worker's id reads this lock under its shard's
    /// ([`Self::stands`]). A rename writes it with no shard held.
    renamed: RwLock<FxHashMap<WorkerId, WorkerId>>,
    /// How many ids the list holds: what a write of a worker's id reads
    /// ahead of the lock, so a table that renamed no worker asks nothing.
    renamed_len: AtomicUsize,
    /// Whether a pass of the table's host sweeps it ([`Self::sweep_due`]):
    /// set at the first such pass, for good. A read of a key then runs no
    /// sweep. A host that runs no pass leaves it unset, and its reads
    /// sweep where one is due and the list's lock is free
    /// ([`Self::sweep_where_free`]).
    swept_by_pass: AtomicBool,
}

impl SessionTable {
    /// A table swept every `sweep_micros` on the decision clock: the
    /// shortest sweep interval among the rules that look keys up in it, `0`
    /// when none sweeps.
    pub fn new(sweep_micros: u64) -> Self {
        Self {
            shards: empty_shards(SHARDS),
            held: Held::default(),
            next_sweep_micros: AtomicU64::new(if sweep_micros == 0 { u64::MAX } else { 0 }),
            next_pass_micros: AtomicU64::new(if sweep_micros == 0 { u64::MAX } else { 0 }),
            sweep_micros: AtomicU64::new(sweep_micros),
            absent: Mutex::new(Vec::new()),
            absence_micros: AtomicU64::new(ABSENT_WORKER_PIN_TTL_MICROS),
            renamed: RwLock::new(FxHashMap::default()),
            renamed_len: AtomicUsize::new(0),
            swept_by_pass: AtomicBool::new(false),
        }
    }

    /// The table with `micros` as the time a worker that left keeps its
    /// pins: the longest TTL among the rules that look
    /// keys up in it, so a key loses no pin to an absence before its idle
    /// time ends.
    #[must_use]
    pub fn with_absence(mut self, micros: u64) -> Self {
        *self.absence_micros.get_mut() = micros;
        self
    }

    /// How long a worker that left keeps its pins, in microseconds.
    pub fn absence_micros(&self) -> u64 {
        self.absence_micros.load(Ordering::Relaxed)
    }

    /// The table's sweep interval, in microseconds: `0` where it never
    /// sweeps.
    pub(crate) fn sweep_micros(&self) -> u64 {
        self.sweep_micros.load(Ordering::Relaxed)
    }

    /// The table takes the two times of a new document: the shortest sweep
    /// interval among its rules and the time a worker that left keeps its
    /// pins. The keys stand as they are: a table outlives a document where
    /// its host keeps one engine and gives it documents.
    ///
    /// A new interval makes a pass due at once, so a table that never
    /// swept starts to, and a shorter interval waits for no pass of the
    /// longer one. A document with no interval (`0`: none of its rules
    /// has a TTL) leaves the interval that stands: the keys the document
    /// before wrote still leave at a pass.
    ///
    /// The times are written under the list's lock, as a leave's own store
    /// is. A sweep holds that lock from its claim to the store of its two
    /// instants ([`Self::sweep`]), so the new times land ahead of a sweep
    /// or behind it: no sweep reads one interval and stores under another.
    /// A request's sweep takes its turn only where the interval and the
    /// instant it read both stand under the lock
    /// ([`Self::sweep_where_free`]): one that read the interval ahead of a
    /// new one leaves the pass due.
    #[cfg(any(test, feature = "in-process"))]
    pub(crate) fn set_times(&self, sweep_micros: u64, absence_micros: u64) {
        let _absent = self.absent.lock();
        self.absence_micros.store(absence_micros, Ordering::Relaxed);
        if sweep_micros == 0 {
            return;
        }
        if self.sweep_micros.swap(sweep_micros, Ordering::Relaxed) != sweep_micros {
            self.next_pass_micros.store(0, Ordering::Relaxed);
            self.next_sweep_micros.store(0, Ordering::Relaxed);
        }
    }

    /// The table with `shards` shards, rounded up to a power of two and one
    /// at the least: what an engine asks for the table its decisions read,
    /// four shards a core of the host, and one shard for a table no rule
    /// looks a key up in. Called where the table is built, while it holds
    /// no key.
    #[must_use]
    pub fn with_shards(mut self, shards: usize) -> Self {
        debug_assert!(self.is_empty(), "a table is sized before it holds a key");
        self.shards = empty_shards(shards.max(1).next_power_of_two());
        self
    }

    /// How many shards the table holds.
    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    /// The shard of `key`, by the key's bits above its low word: a session's
    /// value is mixed over all its bits ([`RequestFacts::session`]), and the
    /// shard's own map buckets a key by other bits than these.
    ///
    /// [`RequestFacts::session`]: crate::engine::view::RequestFacts::session
    #[inline]
    fn shard(&self, key: u64) -> &Mutex<FxHashMap<u64, Entry>> {
        // A power of two of shards, so the mask names one of them.
        let at = (key >> 32) as usize & (self.shards.len() - 1);
        &self.shards[at].0
    }

    /// The entry of `key` in its `shard`: the one the shard holds, or a new
    /// one while the table holds fewer than [`KEYS`]. `None` for a key a
    /// full table does not hold: nothing is recorded of it, and its request
    /// is placed with no pin. Room comes back as keys leave
    /// ([`Self::retain`]). An entry a claim or a dispatch makes where its
    /// decision's read found no room is one no decision has read: the
    /// key's next read starts it over ([`SessionExpiry::Unread`]).
    /// Writers of different shards that race for the last room each take
    /// it: the table can stand past [`KEYS`] by a key a shard at the most.
    /// Always inlined: it stands on the pin's path of every keyed decision.
    #[inline(always)]
    fn slot<'a>(&self, shard: &'a mut FxHashMap<u64, Entry>, key: u64) -> Option<&'a mut Entry> {
        match shard.entry(key) {
            hash_map::Entry::Occupied(held) => Some(held.into_mut()),
            hash_map::Entry::Vacant(free) => {
                // A full table refuses by a read alone, and a key that
                // enters writes the count once: no writer tries twice.
                if self.held.0.load(Ordering::Relaxed) >= KEYS {
                    return None;
                }
                self.held.0.fetch_add(1, Ordering::Relaxed);
                Some(free.insert(Entry::default()))
            }
        }
    }

    /// The pins of `key` at a decision taken at `now_micros` under `rule`,
    /// the key touched when its decision instant is not older (its TTL
    /// renewed, its recovery rule recorded) and the table swept when a sweep
    /// is due and the list's lock is free, where no pass of the table's
    /// host sweeps it ([`Self::sweep_due`]): under such a host a read walks
    /// no shard but its key's. A key never seen, or one past its end, has empty pins:
    /// expiry is the key's own compare, the sweep only reclaims. So has a
    /// key a full table does not hold, of which nothing is written
    /// ([`Self::slot`]).
    /// Always inlined: the keyed decision has two forms, and a second caller
    /// would otherwise move this out of line under the first.
    #[inline(always)]
    pub(crate) fn lookup(&self, key: Session, now_micros: u64, rule: &SessionRule) -> SessionPins {
        let key = key.get();
        if !self.swept_by_pass.load(Ordering::Relaxed) {
            self.sweep_if_due(now_micros);
        }
        let mut shard = self.shard(key).lock();
        let Some(entry) = self.slot(&mut shard, key) else {
            return [WorkerId(0); 2];
        };
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
    /// among this decision's survivors; the pin that stands is returned with
    /// the pins the key held ([`Claim::standing`] is the rule). The
    /// claim is a pin made before a commit, and it is marked as such:
    /// any dispatch to it confirms it, the claiming decision's own dispatch
    /// elsewhere withdraws it (`dispatched`), the TTL sweeps it otherwise.
    /// A pair's decision moves its own claim before its dispatch
    /// ([`Self::move_claim`]).
    ///
    /// The pin and the claim of record take the id that stands for the
    /// worker ([`Self::stands`]): the decision read its rows ahead of this
    /// write, and the worker may have been registered again between the
    /// two. The lease is issued on the id the decision read, so that id is
    /// the lease's and the answer's, as behind a rename that finds a
    /// claim made ahead of it ([`Self::rename_worker`]). A worker that is
    /// a pin of the key already, under the id that stands or under one it
    /// had that a rename's walk has yet to reach, is no claim: the answer
    /// names the decision's own pick, and no pin is written beside it.
    /// Always inlined: the keyed decision has two forms, and a second caller
    /// would otherwise move this out of line under the first.
    #[inline(always)]
    pub(crate) fn claim(&self, key: Session, seen: SessionPins, target: WorkerId) -> Claim {
        let key = key.get();
        if key == 0 || target == WorkerId(0) {
            return Claim::Standing(target, seen);
        }
        let mut shard = self.shard(key).lock();
        let Some(entry) = self.slot(&mut shard, key) else {
            return Claim::NoRoom;
        };
        let Some(owner) = Claim::standing(&entry.pins, &seen, target) else {
            let stands = self.stands(entry, target);
            if !seen.contains(&target) && entry.pins.contains(&stands) {
                // No pin as the decision read them, and a pin under the
                // id that stands: the worker is one the key was served
                // on, or another decision's claim. A claim of this
                // decision's would mark it again, and this decision's
                // commit on an alternate would withdraw it.
                return Claim::Standing(target, entry.pins);
            }
            match entry.recovery {
                Recovery::Latest => entry.front(stands),
                Recovery::Original => entry.append(stands),
            }
            entry.claim = stands;
            entry.leased = target;
            return Claim::Made(target);
        };
        Claim::Standing(owner, entry.pins)
    }

    /// A decision that met `key`'s routable pin at the cap and placed the
    /// request on `target`, another worker under the cap, fronts that worker
    /// where it is made. The key's next request follows the move ahead of
    /// this one's dispatch, as a gateway's sticky map moves a pin where it
    /// selects the worker. The move is no claim, and nothing takes it back:
    /// the commit fronts the worker the request lands on, this one or an
    /// alternate of its pick, and a request never dispatched leaves the pin
    /// moved. Two decisions that read one pin at its cap both front, the
    /// later one last, as their two commits do. No entry is made: a key
    /// the table holds no more, as an empty key, is left to the commit.
    /// A move on an id its worker had before it was registered again
    /// fronts the id that stands ([`Self::stands`]), as a dispatch on
    /// that id does.
    pub(crate) fn respilled(&self, key: Session, target: WorkerId) {
        let key = key.get();
        if target == WorkerId(0) {
            return;
        }
        if let Some(entry) = self.shard(key).lock().get_mut(&key) {
            // The decision read its rows ahead of this write. In neither
            // pin, its worker may have been registered again between the
            // two: the dispatch behind this move finds the worker pinned
            // and asks the renames no more, so the move asks them here.
            let target = self.stands(entry, target);
            entry.front(target);
        }
    }

    /// The decision that claimed `claimed` for `key` does not go there: a
    /// pair's decision formed its pair on another prefill, `target`, or on
    /// none (`WorkerId(0)`). The claim is withdrawn where it still stands
    /// unconfirmed, as the claiming decision's commit elsewhere withdraws
    /// it ([`Self::dispatched`]), and `target` is claimed in its place by
    /// the key's recovery: the key's next decision follows a worker the
    /// request goes to, before any dispatch. The answer is the verdict the
    /// decision states of `target`: its claim where `target` is the claim
    /// of record, else a pin's. A claim a dispatch confirmed is a pin the
    /// key was served on: it stands, and nothing is written. Another
    /// decision's claim is not this one's to move. A `target` that is a pin
    /// of the key already, under any id of its worker, is no claim either.
    /// The new claim is written as a first one is ([`Self::claim`]): the
    /// pin and the claim of record under the id that stands for `target`,
    /// the lease's id and the answer under the id the decision read.
    pub(crate) fn move_claim(
        &self,
        key: Session,
        claimed: WorkerId,
        target: WorkerId,
    ) -> SessionVerdict {
        let key = key.get();
        let mut shard = self.shard(key).lock();
        let Some(entry) = shard.get_mut(&key) else {
            return SessionVerdict::Pinned;
        };
        if entry.claim == WorkerId(0) || entry.leased != claimed {
            return SessionVerdict::Pinned;
        }
        let pin = std::mem::replace(&mut entry.claim, WorkerId(0));
        entry.withdraw(pin);
        if target == WorkerId(0) {
            return SessionVerdict::Pinned;
        }
        let stands = self.stands(entry, target);
        if entry.pins.contains(&stands) {
            return SessionVerdict::Pinned;
        }
        match entry.recovery {
            Recovery::Latest => entry.front(stands),
            Recovery::Original => entry.append(stands),
        }
        entry.claim = stands;
        entry.leased = target;
        SessionVerdict::Claimed(target)
    }

    /// The commit dispatched a decision for `key` to `target` under the
    /// decision's `verdict`: the pin follows the rule. The preferred pin
    /// taken, or the failover taken while the preferred one was unroutable,
    /// leaves the pins as they are; a respill, and a worker outside the pins,
    /// move them by the key's recovery rule. A dispatch to the claimed worker
    /// confirms the claim first; the claiming decision's commit elsewhere
    /// withdraws it. A dispatch on an id its worker had before it was
    /// registered again is one on that worker, under the id that stands
    /// ([`Self::stands`]).
    pub(crate) fn dispatched(&self, key: Session, verdict: SessionVerdict, target: WorkerId) {
        let key = key.get();
        if key == 0 || target == WorkerId(0) {
            return;
        }
        let mut shard = self.shard(key).lock();
        let Some(entry) = self.slot(&mut shard, key) else {
            return;
        };
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
        let target = if claimed_here {
            entry.claim
        } else {
            // A dispatch names the id its lease was issued on. In neither
            // pin, its worker may have been registered again since: the
            // table's renames say which id stands for it.
            self.stands(entry, target)
        };
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
    /// confirms it, and its commit elsewhere withdraws it. Where the key
    /// held the claim's worker under the rename's other id as well, the
    /// claim's mark goes ([`Entry::follow`], the walk's step for a key).
    /// A write of a key that asks the renames ahead of the walk takes
    /// that step for its key first ([`Self::standing`]).
    pub(crate) fn rename_worker(&self, old: WorkerId, new: WorkerId) {
        if old == WorkerId(0) || new == WorkerId(0) {
            return;
        }
        self.absent.lock().retain(|(id, _)| *id != old);
        // The same life registered again holds its pins as they stand.
        if old == new {
            return;
        }
        // Ahead of the walk: a dispatch on the old id that lands behind
        // the walk of its shard finds the id that stands, and one that
        // lands ahead of it reads a pin the walk renames.
        self.remember(old, new);
        for shard in &self.shards {
            for entry in shard.0.lock().values_mut() {
                entry.follow(|id| if id == old { new } else { id });
            }
        }
    }

    /// `new` stands for `old`, and for every id `old` stood for. A worker
    /// back under an id it had stands under it again.
    fn remember(&self, old: WorkerId, new: WorkerId) {
        let mut renamed = self.renamed.write();
        if renamed.len() >= RENAMED_WORKERS {
            renamed.clear();
        }
        for standing in renamed.values_mut() {
            if *standing == old {
                *standing = new;
            }
        }
        renamed.insert(old, new);
        renamed.remove(&new);
        self.renamed_len.store(renamed.len(), Ordering::Relaxed);
    }

    /// The id a write names `target` by in `entry`, its key's: `target`
    /// where it is one of the key's pins as written or the table renamed
    /// no worker, else the id that stands for its worker
    /// ([`Self::standing`]). A decision reads its rows ahead of its writes
    /// and a dispatch names the id its lease was issued on, so the worker
    /// may have been registered again since. Each write of a worker's id
    /// asks, under its shard's lock: a claim, a pair's moved claim, a
    /// respill's move and the commit. The pins are read first: a worker
    /// among them costs no read of the renames' count. A write whose
    /// target is a pin as written asks nothing, whatever a rename's walk
    /// has yet to reach: it puts the pins in another order and drops
    /// none, so the walk, or the key's next write that asks, finds what
    /// it would have found ahead of it.
    #[inline]
    fn stands(&self, entry: &mut Entry, target: WorkerId) -> WorkerId {
        if !entry.pins.contains(&target) && self.renamed_len.load(Ordering::Relaxed) != 0 {
            self.standing(entry, target)
        } else {
            target
        }
    }

    /// The id that stands for `target`, an id a decision read or a
    /// dispatch's lease was issued on: its worker's own where the worker
    /// was registered again since. The key's entry takes the walk's step
    /// first ([`Entry::follow`]), by every rename the table remembers. A
    /// rename says the id that stands ahead of its walk of the shards
    /// ([`Self::rename_worker`]), so an entry the walk has yet to reach
    /// holds ids its workers had, and may hold one worker under two.
    /// Behind the step every write that asks runs on the walked entry:
    /// it meets a worker's pin under the id that stands, and no claim's
    /// mark stands on a pin the key was served on. Out of line: a write
    /// asks only where its target is in neither pin and the table renamed
    /// a worker ([`Self::stands`]).
    #[cold]
    #[inline(never)]
    fn standing(&self, entry: &mut Entry, target: WorkerId) -> WorkerId {
        let renamed = self.renamed.read();
        let stands = |id: WorkerId| renamed.get(&id).copied().unwrap_or(id);
        entry.follow(stands);
        stands(target)
    }

    /// How many ids the table remembers workers had.
    #[cfg(test)]
    pub(crate) fn renamed(&self) -> usize {
        self.renamed.read().len()
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
            self.retain(shard, |entry| Self::unpinned(entry, workers));
        }
    }

    /// Keeps the entries of `shard` that `keep` says stay, and gives the
    /// room of the keys that left back to the table's count. A shard the
    /// walk leaves under a quarter of what its map can hold gives the
    /// spare storage back too, to twice its keys.
    fn retain(&self, shard: &Shard, mut keep: impl FnMut(&mut Entry) -> bool) {
        let mut keys = shard.0.lock();
        let before = keys.len();
        keys.retain(|_, entry| keep(entry));
        let left = before - keys.len();
        if left != 0 {
            self.held.0.fetch_sub(left, Ordering::Relaxed);
            let kept = keys.len();
            if kept < keys.capacity() / 4 {
                keys.shrink_to(2 * kept);
            }
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

    /// The sweep that is due, for a host that calls it on a timer of its
    /// own, at `now_micros` on the decision clock: the keys past their end
    /// leave the table and a worker whose absence ended is forgotten, off
    /// the path of a request. The list's lock claims a sweep. A call that
    /// reads a sweep due waits for the lock, and what is due is read under
    /// it. So what a leave or a document's new times stored while the call
    /// waited is its to take, and a pass another call took meanwhile is
    /// not walked twice. The call counts by its own `now_micros`, which is
    /// older than such a store by the time it waited. The walk takes each
    /// shard's mutex in turn.
    ///
    /// From the first call every sweep is this pass's: a read of a key runs
    /// none, however long one has been due. So a host that calls this goes
    /// on calling it. What a sweep does waits for the pass then: a key past
    /// its end is held until it, and reads as no pins meanwhile, and the
    /// absence of a worker that left is timed from it.
    pub fn sweep_due(&self, now_micros: u64) {
        if !self.swept_by_pass.load(Ordering::Relaxed) {
            self.swept_by_pass.store(true, Ordering::Relaxed);
        }
        if now_micros < self.next_sweep_micros.load(Ordering::Relaxed) {
            return;
        }
        self.sweep(self.absent.lock(), now_micros);
    }

    /// Whether a sweep is due: of the keys past their end, once per the
    /// table's sweep interval, or of a worker that left. A table that owes
    /// neither never walks its shards. Always inlined with its caller
    /// ([`Self::lookup`]): the check is one compare on the pin's path.
    #[inline(always)]
    fn sweep_if_due(&self, now_micros: u64) {
        if now_micros < self.next_sweep_micros.load(Ordering::Relaxed) {
            return;
        }
        self.sweep_as_read(now_micros);
    }

    /// A request's sweep, out of line. It reads the interval, then the
    /// instant once more, and takes its turn by the two
    /// ([`Self::sweep_where_free`]). The interval is read first: new times
    /// that land between the two reads then show as their instant beside
    /// the interval before, which no longer stands under the lock.
    #[cold]
    #[inline(never)]
    fn sweep_as_read(&self, now_micros: u64) {
        let interval = self.sweep_micros();
        let due = self.next_sweep_micros.load(Ordering::Relaxed);
        self.sweep_where_free((due, interval), now_micros);
    }

    /// A request's sweep by what it `read`: the sweep's instant, and the
    /// interval it read ahead of that instant. It takes its turn where
    /// the list's lock is free, and never waits for that lock. Held by
    /// another sweep, the turn is that sweep's. Held by a leave, a return
    /// or a document's new times, the lock leaves the sweep due for a
    /// later call.
    ///
    /// Both values are read again under the lock, and the turn is taken
    /// only where both stand. Where another call took the turn, or a leave
    /// or a new interval moved it, this call counts by an instant that no
    /// longer stands: it leaves the turn to a call that reads the new one.
    /// A new interval stores the pass as due at once, which a call that
    /// read a leave's store cannot tell by the instant alone: so the
    /// interval is compared too. A call that read both behind such a store
    /// takes the turn and counts by its own `now_micros`.
    fn sweep_where_free(&self, read: (u64, u64), now_micros: u64) {
        let Some(absent) = self.absent.try_lock() else {
            return;
        };
        let stands = (
            self.next_sweep_micros.load(Ordering::Relaxed),
            self.sweep_micros(),
        );
        if stands == read {
            self.sweep(absent, now_micros);
        }
    }

    /// The sweep that is due, under the list's lock. A leave not yet
    /// stamped takes its end, the absence's time from this decision. A
    /// worker whose absence ended is forgotten, and every key past its end
    /// where the pass over the idle keys is due: one walk of the table for
    /// both, and none where neither is owed. The pass keeps its own
    /// instant, so a sweep a leave asked for walks no shard for the keys
    /// ahead of it.
    ///
    /// The lock is the sweep's claim. A leave's store and a document's new
    /// times stand under the same lock ([`Self::left`], `set_times`), so
    /// each lands ahead of a sweep or behind it: the sweep reads the pass's
    /// instant and the interval under it, and stores under it what it
    /// stores of the two instants. What is due is read there too: the pass
    /// by its instant, a leave by its missing stamp and an absence by its
    /// end. A call that finds none of the three walks nothing.
    fn sweep(&self, mut absent: MutexGuard<'_, Vec<(WorkerId, u64)>>, now_micros: u64) {
        let pass = self.next_pass_micros.load(Ordering::Relaxed);
        let idle = now_micros >= pass;
        let mut next = pass;
        if idle {
            next = now_micros.saturating_add(self.sweep_micros());
            self.next_pass_micros.store(next, Ordering::Relaxed);
        }
        let mut ended: Vec<WorkerId> = Vec::new();
        absent.retain_mut(|(worker, ends)| {
            if *ends == 0 {
                *ends = now_micros.saturating_add(self.absence_micros()).max(1);
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
            self.retain(shard, |entry| {
                let stays = !idle || entry.expires_at_micros > now_micros;
                stays && Self::unpinned(entry, &ended)
            });
        }
        drop(held);
    }

    /// The pins of `key` as a decision at `now_micros` finds them, with
    /// nothing written: a key the table never read, or one past its end,
    /// has none, as at [`Self::lookup`]. The key's idle time is not
    /// renewed and no sweep runs: a decision that goes on to follow one of
    /// these pins renews it by that read.
    pub(crate) fn pins_at(&self, key: Session, now_micros: u64) -> SessionPins {
        let key = key.get();
        match self.shard(key).lock().get(&key) {
            Some(entry) if entry.expires_at_micros > now_micros => entry.pins,
            _ => [WorkerId(0); 2],
        }
    }

    /// The pins of `key`, for tests and dumps.
    pub fn pins(&self, key: Session) -> Option<SessionPins> {
        let key = key.get();
        self.shard(key).lock().get(&key).map(|e| e.pins)
    }

    /// The pins of `key` and when the table forgets the key, for tests and
    /// dumps.
    pub fn held(&self, key: Session) -> Option<(SessionPins, SessionExpiry)> {
        let key = key.get();
        self.shard(key).lock().get(&key).map(|entry| {
            let forgets = match entry.expires_at_micros {
                0 => SessionExpiry::Unread,
                u64::MAX => SessionExpiry::Never,
                at => SessionExpiry::At(at),
            };
            (entry.pins, forgets)
        })
    }

    /// Keys the table holds, by its count: no shard is locked.
    pub fn len(&self) -> usize {
        self.held.0.load(Ordering::Relaxed)
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
        rule_placed: RulePlaced::Told,
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

    /// A table that outlives a document takes the next one's two times and
    /// keeps its keys. A table that never swept takes an interval, and a
    /// pass is due at once. A document with no interval leaves the one
    /// that stands, so the keys the document before wrote still leave at a
    /// pass. A shorter interval waits for no pass of the longer one.
    #[test]
    fn a_table_takes_a_new_documents_two_times_and_keeps_its_keys() {
        let pin = |table: &SessionTable, key: u64, at: u64| {
            table.lookup(mixed(key), at, &RULE);
            table.dispatched(mixed(key), SessionVerdict::Pinned, WorkerId(key));
        };
        // A table made with no interval runs no pass over its idle keys
        // ahead of the clock's last instant: a key past its own rule's end
        // stands at a timer's call.
        let table = SessionTable::new(0).with_shards(8);
        pin(&table, 1, 0);
        table.sweep_due(5_000);
        assert_eq!(table.len(), 1, "a table with no interval sweeps nothing");
        // The next document's rule sweeps: the key past its end leaves at
        // the pass that is due now.
        table.set_times(RULE.sweep_micros, 7_000);
        assert_eq!((table.sweep_micros(), table.absence_micros()), (100, 7_000));
        assert_eq!(table.len(), 1, "the new times keep the keys");
        table.sweep_due(5_000);
        assert_eq!(table.len(), 0, "the first pass under the new interval");
        // A key of that document, then a document with no interval: the
        // interval stands, and the key leaves at its pass.
        pin(&table, 2, 5_000);
        table.set_times(0, 9_000);
        assert_eq!((table.sweep_micros(), table.absence_micros()), (100, 9_000));
        assert_eq!(table.len(), 1, "the new times keep the keys");
        let pins = table.pins(mixed(2));
        assert_eq!(pins, Some([WorkerId(2), WorkerId(0)]), "and their pins");
        table.sweep_due(6_200);
        assert_eq!(table.len(), 0, "a pass under the interval that stood");
        // New times that name the interval that stands move no instant: the
        // pass just taken is not due again.
        assert_eq!(instants(&table), (6_300, 6_300));
        table.set_times(RULE.sweep_micros, 9_500);
        assert_eq!(table.absence_micros(), 9_500);
        assert_eq!(instants(&table), (6_300, 6_300), "the same interval");

        // A long interval, its first pass behind it, and a key past its end.
        let table = SessionTable::new(1_000_000).with_shards(8);
        table.sweep_due(10);
        pin(&table, 3, 20);
        table.sweep_due(2_000);
        assert_eq!(table.len(), 1, "the long interval's pass is not due");
        table.set_times(RULE.sweep_micros, 7_000);
        assert_eq!(table.len(), 1, "the new times keep the keys");
        table.sweep_due(2_000);
        assert_eq!(table.len(), 0, "the shorter interval's pass is due at once");
    }

    /// The table's two instants: its next sweep, and its next pass over the
    /// idle keys.
    fn instants(table: &SessionTable) -> (u64, u64) {
        let read = |word: &AtomicU64| word.load(Ordering::Relaxed);
        (
            read(&table.next_sweep_micros),
            read(&table.next_pass_micros),
        )
    }

    /// A request's sweep claims its turn by taking the list's lock, and
    /// where the lock is held it takes no turn and waits for none: the two
    /// instants stand, the keys stand, and the sweep stays due. A later
    /// call takes the turn the first one left.
    ///
    /// Another thread holds the lock until the case has looked, or until it
    /// has waited two minutes for that word. A sweep that waited for the
    /// lock goes on behind that and takes its turn, so the case fails where
    /// it would hang.
    #[test]
    fn a_requests_sweep_takes_no_turn_and_waits_for_none_while_the_lists_lock_is_held() {
        use std::sync::{mpsc, Arc};
        let table = Arc::new(SessionTable::new(1_000_000).with_shards(8));
        // The table's first request sweeps: the next pass is one interval
        // on. Its key ends at 1,100.
        table.lookup(mixed(4), 10, &RULE);
        assert_eq!(instants(&table), (1_000_010, 1_000_010));
        let (held, is_held) = mpsc::channel();
        let (looked, has_looked) = mpsc::channel::<()>();
        let holder = {
            let table = Arc::clone(&table);
            std::thread::spawn(move || {
                let absent = table.absent.lock();
                held.send(()).expect("the case waits for the holder");
                let _ = has_looked.recv_timeout(std::time::Duration::from_secs(120));
                drop(absent);
            })
        };
        is_held.recv().expect("the holder took the lock");
        table.lookup(mixed(5), 1_000_020, &RULE);
        assert_eq!(instants(&table), (1_000_010, 1_000_010), "no turn taken");
        assert_eq!(table.len(), 2, "no turn taken: the key past its end stands");
        looked.send(()).expect("the holder waits for the case");
        holder.join().expect("the holder ends");
        table.lookup(mixed(6), 1_000_020, &RULE);
        assert_eq!(instants(&table), (2_000_020, 2_000_020), "the turn left");
        assert_eq!(table.len(), 2, "the key past its end left");
    }

    /// What a request read ahead of the lock is read again under it. A call
    /// whose turn moved meanwhile counts nothing by its own instant. A
    /// document's pass stays due, also behind a call that read a leave's
    /// store, whose instant is the document's own. A leave is stamped by
    /// the call that reads the leave's store, and a turn is taken once. A
    /// request reads the interval ahead of the instant, so new times that
    /// land between its two reads show as the interval before.
    #[test]
    fn a_requests_sweep_whose_turn_moved_ahead_of_the_lock_leaves_it() {
        const OLD: u64 = 1_000_000;
        let table = SessionTable::new(OLD).with_shards(8);
        table.lookup(mixed(4), 10, &RULE);
        table.dispatched(mixed(4), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(instants(&table), (1_000_010, 1_000_010));
        // A request read that turn as due and stalled. New times then made
        // a pass due at once.
        let new = RULE.sweep_micros;
        table.set_times(new, 7_000);
        assert_eq!(instants(&table), (0, 0));
        table.sweep_where_free((1_000_010, OLD), 1_000_015);
        assert_eq!(instants(&table), (0, 0), "the document's pass stays due");
        assert_eq!(table.len(), 1, "and no key was walked");
        // A request that read a leave's store ahead of the new times read
        // the instant that stands, and the interval before.
        table.sweep_where_free((0, OLD), 400);
        assert_eq!(instants(&table), (0, 0), "the document's pass stays due");
        // The request that reads both as they stand takes the turn.
        table.sweep_where_free((0, new), 500);
        assert_eq!(instants(&table), (600, 600));
        // A leave stores its own instant. A request that read the one
        // before stamps nothing: the absence is timed from the call that
        // reads the leave's store.
        table.left(WorkerId(1));
        assert_eq!(instants(&table), (0, 600));
        table.sweep_where_free((600, new), 650);
        assert_eq!(*table.absent.lock(), [(WorkerId(1), 0)], "not stamped");
        table.sweep_where_free((0, new), 700);
        assert_eq!(*table.absent.lock(), [(WorkerId(1), 7_700)]);
        assert_eq!(instants(&table), (800, 800));
        // A second request read the leave's store too, and runs behind the
        // one that took the turn.
        table.sweep_where_free((0, new), 900);
        assert_eq!(instants(&table), (800, 800), "a turn is taken once");
        // A request's own two reads, taken as they stand: a call dated
        // ahead of the turn finds nothing due under the lock, and the call
        // at 900 takes the turn.
        table.sweep_as_read(750);
        assert_eq!(instants(&table), (800, 800), "nothing is due at 750");
        table.sweep_as_read(900);
        assert_eq!(instants(&table), (1_000, 1_000));
    }

    /// A sweep asked off the path of a request waits for the list's lock,
    /// and what is due is read under it. A leave that stores its instant
    /// while the sweep waits is stamped by that sweep, and a pass another
    /// call took while it waited is not walked twice. A document's new
    /// times wait for the lock too, with both of them. Each call runs on a
    /// thread that says it started, and the holder then yields to it. A
    /// thread that has not reached the lock by then hides a call that did
    /// not wait: it fails no call that does.
    #[test]
    fn a_timers_sweep_and_new_times_wait_for_the_lists_lock() {
        use std::sync::Arc;
        const LONG: u64 = 1_000_000;
        // What the lock's holder writes before it lets go.
        type Writes = fn(&SessionTable, &mut Vec<(WorkerId, u64)>);
        let table = Arc::new(SessionTable::new(LONG).with_shards(8));
        table.sweep_due(10);
        table.lookup(mixed(4), 20, &RULE);
        assert_eq!(instants(&table), (1_000_010, 1_000_010));
        // `call` on a thread of its own while this thread holds the lock:
        // whether it was still waiting, and the instants and the two times
        // that stood, all read ahead of the lock's release. The holder
        // `writes` under the lock before it lets go.
        let behind_the_lock = |call: fn(&SessionTable), writes: Writes| {
            let started = Arc::new(AtomicBool::new(false));
            let mut absent = table.absent.lock();
            let thread = {
                let (table, started) = (Arc::clone(&table), Arc::clone(&started));
                std::thread::spawn(move || {
                    started.store(true, Ordering::Relaxed);
                    call(&table);
                })
            };
            while !started.load(Ordering::Relaxed) {
                std::thread::yield_now();
            }
            for _ in 0..1_000 {
                std::thread::yield_now();
            }
            let stood = (
                !thread.is_finished(),
                instants(&table),
                (table.sweep_micros(), table.absence_micros()),
            );
            writes(&table, &mut absent);
            drop(absent);
            thread.join().expect("the call ends");
            stood
        };
        let times = (LONG, ABSENT_WORKER_PIN_TTL_MICROS);
        // A leave's two stores while the call waited: the call takes the
        // turn and stamps the leave.
        let leaves: Writes = |table, absent| {
            absent.push((WorkerId(9), 0));
            table.next_sweep_micros.store(0, Ordering::Relaxed);
        };
        let stood = behind_the_lock(|table| table.sweep_due(1_000_020), leaves);
        assert_eq!(stood, (true, (1_000_010, 1_000_010), times));
        assert_eq!(table.len(), 0, "the turn is taken behind the holder");
        let ends = 1_000_020 + ABSENT_WORKER_PIN_TTL_MICROS;
        assert_eq!(*table.absent.lock(), [(WorkerId(9), ends)], "the leave too");
        assert_eq!(instants(&table), (2_000_020, 2_000_020));
        // The pass another call took while this one waited, as that call's
        // two stores: this one walks nothing, and a key past its end stands.
        table.lookup(mixed(5), 2_000_030, &RULE);
        let taken: Writes = |table, _| {
            table.next_pass_micros.store(4_000_000, Ordering::Relaxed);
            table.next_sweep_micros.store(4_000_000, Ordering::Relaxed);
        };
        let stood = behind_the_lock(|table| table.sweep_due(3_000_000), taken);
        assert_eq!(stood, (true, (2_000_020, 2_000_020), times));
        assert_eq!(table.len(), 1, "a pass is not walked twice");
        assert_eq!(instants(&table), (4_000_000, 4_000_000));
        // New times wait for the lock with both of them.
        let stood = behind_the_lock(|table| table.set_times(RULE.sweep_micros, 7_000), |_, _| {});
        assert_eq!(stood, (true, (4_000_000, 4_000_000), times));
        assert_eq!(instants(&table), (0, 0), "the new interval's pass is due");
        let times = (table.sweep_micros(), table.absence_micros());
        assert_eq!(times, (RULE.sweep_micros, 7_000));
    }

    /// A key renewed under a rule of another sweep interval keeps the end
    /// the interval before gave it, while that end is less than one new
    /// interval past the end of the key's idle time: the end stays at or
    /// after the end of that idle time. Past the span the new interval's
    /// own sweep replaces the end.
    #[test]
    fn a_key_renewed_under_a_new_interval_keeps_an_end_inside_it() {
        const SECOND: u64 = 1_000_000;
        let rule = |sweep_s: u64| SessionRule {
            ttl_micros: Some(10 * SECOND),
            sweep_micros: sweep_s * SECOND,
            ..RULE
        };
        let ends = |table: &SessionTable| table.held(s(1)).map(|(_, ends)| ends);
        let table = SessionTable::new(60 * SECOND);
        // Idle until 45 s: the first sweep of 60 s at or after it is at 60 s.
        table.lookup(s(1), 35 * SECOND, &rule(60));
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(ends(&table), Some(SessionExpiry::At(60 * SECOND)));
        // The next document's rule sweeps every 40 s. A use at 45 s is idle
        // until 55 s: the end at 60 s is less than 40 s past it and stands,
        // where the new interval alone gives 80 s.
        table.lookup(s(1), 45 * SECOND, &rule(40));
        assert_eq!(ends(&table), Some(SessionExpiry::At(60 * SECOND)));
        // A use at 58 s is idle until 68 s, past the old end: the new
        // interval's sweep at 80 s replaces it.
        table.lookup(s(1), 58 * SECOND, &rule(40));
        assert_eq!(ends(&table), Some(SessionExpiry::At(80 * SECOND)));
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
        // Idle until 1 050: the rule's next sweep is at 1 100.
        assert_eq!(
            table.held(s(1)),
            Some(([WorkerId(1), WorkerId(0)], SessionExpiry::At(1_100)))
        );
        // A read past the idle time and ahead of that sweep renews the key.
        assert_eq!(table.lookup(s(1), 1_060, &RULE), [WorkerId(1), WorkerId(0)]);
        let renewed = table.held(s(1)).map(|(_, ends)| ends);
        assert_eq!(renewed, Some(SessionExpiry::At(2_100)));
        let for_ever = SessionRule {
            ttl_micros: None,
            sweep_micros: 0,
            ..RULE
        };
        table.lookup(s(2), 50, &for_ever);
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(2));
        assert_eq!(
            table.held(s(2)),
            Some(([WorkerId(2), WorkerId(0)], SessionExpiry::Never))
        );
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(3));
        assert_eq!(
            table.held(s(3)),
            Some(([WorkerId(3), WorkerId(0)], SessionExpiry::Unread))
        );
    }

    /// The read with nothing written gives a key's pins as a decision at
    /// that instant finds them: none for a key the table never read, the
    /// pins until the key's end, and none from that instant. It makes no
    /// key, renews nothing and forgets nothing.
    #[test]
    fn a_keys_pins_are_read_as_they_stand_with_nothing_written() {
        let table = SessionTable::new(RULE.sweep_micros);
        assert_eq!(table.pins_at(s(1), 50), [WorkerId(0); 2]);
        assert!(table.is_empty(), "the read makes no key");
        table.lookup(s(1), 50, &RULE);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(7));
        let held = table.held(s(1));
        assert_eq!(table.pins_at(s(1), 1_099), [WorkerId(7), WorkerId(0)]);
        assert_eq!(table.pins_at(s(1), 1_100), [WorkerId(0); 2], "its end");
        assert_eq!(table.pins_at(s(1), u64::MAX), [WorkerId(0); 2]);
        assert_eq!(table.held(s(1)), held, "nothing renewed or forgotten");
        // Under a rule with no idle time the pins stand at every instant
        // but the clock's last, and a key dispatched to and never read has
        // none.
        let for_ever = SessionRule {
            ttl_micros: None,
            sweep_micros: 0,
            ..RULE
        };
        table.lookup(s(2), 50, &for_ever);
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(9));
        assert_eq!(
            table.pins_at(s(2), u64::MAX - 1),
            [WorkerId(9), WorkerId(0)]
        );
        assert_eq!(table.pins_at(s(2), u64::MAX), [WorkerId(0); 2]);
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(3));
        assert_eq!(table.pins_at(s(3), 50), [WorkerId(0); 2]);
    }

    /// A key past its end is forgotten at its next lookup even when no
    /// sweep ran since: the compare is the key's own, the walk only reclaims.
    #[test]
    fn an_expired_key_is_forgotten_at_its_lookup_before_any_sweep() {
        let rarely = SessionTable::new(1_000_000);
        rarely.lookup(s(1), 0, &RULE);
        rarely.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(rarely.lookup(s(1), 500, &RULE), [WorkerId(1), WorkerId(0)]);
        assert_eq!(rarely.lookup(s(1), 2_000, &RULE), [WorkerId(0); 2]);
    }

    /// The rounding writes no end a host would misread. A rule with an idle
    /// time and no sweep interval ends a key with its idle time. A sweep at
    /// the clock's last instant, which reads as no end, is not taken.
    #[test]
    fn a_keys_end_is_never_rounded_to_no_end() {
        let table = SessionTable::new(RULE.sweep_micros);
        let ends = |key, now, sweep_micros| {
            let rule = SessionRule {
                ttl_micros: Some(60),
                sweep_micros,
                ..RULE
            };
            table.lookup(s(key), now, &rule);
            table.held(s(key)).map(|(_, ends)| ends)
        };
        assert_eq!(ends(1, 10, 0), Some(SessionExpiry::At(70)));
        // The clock's last instant is a multiple of five.
        assert_eq!(
            ends(2, u64::MAX - 63, 5),
            Some(SessionExpiry::At(u64::MAX - 3))
        );
    }

    /// One key read under two rules, as no engine reads one: a key meets one
    /// rule while its table lives. An end another rule wrote stays where it
    /// stands less than one interval above the new idle end, and one a whole
    /// interval above is replaced by this rule's sweep.
    #[test]
    fn an_end_another_rule_wrote_stays_inside_one_interval_alone() {
        let table = SessionTable::new(RULE.sweep_micros);
        let ends = |key, now, ttl, sweep_micros| {
            let rule = SessionRule {
                ttl_micros: Some(ttl),
                sweep_micros,
                ..RULE
            };
            table.lookup(s(key), now, &rule);
            table.held(s(key)).map(|(_, ends)| ends)
        };
        // A sweep every 70 ends a read at 30 at 1 050. RULE's read at 40
        // ends the idle time at 1 040, and 1 050 stands inside one interval
        // of 100 above it: RULE alone gives 1 100.
        assert_eq!(ends(1, 30, 1_000, 70), Some(SessionExpiry::At(1_050)));
        assert_eq!(ends(1, 40, 1_000, 100), Some(SessionExpiry::At(1_050)));
        // An end exactly one interval above the new idle end is replaced.
        assert_eq!(ends(2, 0, 1_100, 0), Some(SessionExpiry::At(1_100)));
        assert_eq!(ends(2, 0, 1_000, 100), Some(SessionExpiry::At(1_000)));
    }

    /// The table against a model of the gateway's sticky map, over seeded
    /// histories of one key. The map ends an entry at an eviction tick
    /// alone, one tick an interval, where the entry's last use is the idle
    /// time old, and every use renews the entry with no compare
    /// (`model_gateway/src/policies/manual.rs:168-179`, `:395`, `:234`).
    /// Each request finds its key kept or started over as the map does,
    /// under reads that sweep and under a host's own pass. A model, not a
    /// recorded line: no recorded row moves time.
    #[test]
    fn a_keys_pins_end_as_an_entry_of_the_gateways_sticky_map_does() {
        let mut seed = 7_u64;
        let mut draw = |bound: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % bound
        };
        let (mut requests, mut differ) = (0_u32, 0_u32);
        for history in 0..20_000 {
            let idle = [1, 7, 30, 60, 61, 100, 240, 1_000][draw(8) as usize];
            let interval = [1, 7, 60, 100, 3_600][draw(5) as usize];
            let rule = SessionRule {
                ttl_micros: Some(idle),
                sweep_micros: interval,
                ..RULE
            };
            let table = SessionTable::new(interval).with_shards(1);
            // The map's one entry: the instant of its last use while it is
            // kept, and the map's next tick.
            let (mut used, mut tick) = (None, interval);
            let mut now = 0;
            for _ in 0..2 + draw(39) {
                let long = 1 + draw(2 * idle + 2 * interval);
                now += [0, 1, long, 1 + draw(idle + 1)][draw(4) as usize];
                if history % 2 == 1 {
                    table.sweep_due(now);
                }
                let kept = table.lookup(s(1), now, &rule) != [WorkerId(0); 2];
                table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
                // A tick at a use's instant runs ahead of the use.
                while tick <= now {
                    if used.is_some_and(|used| tick - used >= idle) {
                        used = None;
                    }
                    tick += interval;
                }
                differ += u32::from(kept != used.replace(now).is_some());
                requests += 1;
            }
        }
        assert_eq!((differ, requests), (0, 420_215), "answers that differ");
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
            Some(([WorkerId(1), WorkerId(0)], SessionExpiry::At(2_000)))
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

    /// A respill's move, written ahead of its commit, fronts the worker
    /// taken under either recovery, as the commit does. It makes no entry:
    /// an empty key, an empty worker and a key the table holds no more
    /// write nothing.
    #[test]
    fn a_respills_move_fronts_the_worker_taken_and_makes_no_entry() {
        for rule in [RULE, ORIGINAL] {
            let table = SessionTable::new(rule.sweep_micros);
            table.lookup(s(7), 10, &rule);
            table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));
            table.respilled(s(7), WorkerId(2));
            assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
            // The worker that leads already: nothing moves, and the commit
            // finds the pins in place.
            table.respilled(s(7), WorkerId(2));
            table.dispatched(s(7), SessionVerdict::Respill, WorkerId(2));
            assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
            table.respilled(s(0), WorkerId(3));
            table.respilled(s(7), WorkerId(0));
            table.respilled(s(8), WorkerId(3));
            assert_eq!(table.pins(s(7)), Some([WorkerId(2), WorkerId(1)]));
            assert_eq!((table.pins(s(8)), table.len()), (None, 1));
        }
    }

    /// Two decisions read one pin at its cap, and each takes another
    /// worker. Both front, the later one last, and their commits front in
    /// their own order: the pins end where the two commits alone leave
    /// them.
    #[test]
    fn two_respills_of_one_capped_pin_both_front_the_later_one_last() {
        let ends = [true, false].map(|moved| {
            let table = SessionTable::new(RULE.sweep_micros);
            table.lookup(s(7), 10, &RULE);
            table.dispatched(s(7), SessionVerdict::Pinned, WorkerId(1));
            // Both decisions read the pin ahead of either's move.
            let read = [20, 21].map(|now| table.lookup(s(7), now, &RULE));
            assert_eq!(read, [[WorkerId(1), WorkerId(0)]; 2]);
            if moved {
                table.respilled(s(7), WorkerId(2));
                table.respilled(s(7), WorkerId(3));
                assert_eq!(table.pins(s(7)), Some([WorkerId(3), WorkerId(2)]));
            }
            // The later decision's request is sent first.
            table.dispatched(s(7), SessionVerdict::Respill, WorkerId(3));
            table.dispatched(s(7), SessionVerdict::Respill, WorkerId(2));
            table.pins(s(7))
        });
        assert_eq!(ends, [Some([WorkerId(2), WorkerId(3)]); 2]);
    }

    /// A respill's move beside a claim no commit has confirmed leaves the
    /// claim to its own commit. On the claimed worker that commit confirms
    /// it, and elsewhere it withdraws it. Whichever of the two commits
    /// comes first, under either recovery, the pins end where the two
    /// commits alone leave them.
    #[test]
    fn a_respills_move_beside_an_open_claim_ends_where_the_commits_alone_end() {
        let (claimed, taken) = (WorkerId(1), WorkerId(2));
        for rule in [RULE, ORIGINAL] {
            for lands in [claimed, WorkerId(3)] {
                for claim_first in [true, false] {
                    let ends = [true, false].map(|moved| {
                        let table = SessionTable::new(rule.sweep_micros);
                        let seen = table.lookup(s(1), 10, &rule);
                        assert_eq!(table.claim(s(1), seen, claimed), Claim::Made(claimed));
                        if moved {
                            table.respilled(s(1), taken);
                            assert_eq!(table.pins(s(1)), Some([taken, claimed]));
                        }
                        let claims = SessionVerdict::Claimed(claimed);
                        if claim_first {
                            table.dispatched(s(1), claims, lands);
                        }
                        table.dispatched(s(1), SessionVerdict::Respill, taken);
                        if !claim_first {
                            table.dispatched(s(1), claims, lands);
                        }
                        table.pins(s(1))
                    });
                    assert_eq!(
                        ends[0], ends[1],
                        "{:?}: the claim's commit on {lands:?}, first: {claim_first}",
                        rule.recovery
                    );
                    assert_eq!(ends[0].map(|pins| pins.contains(&taken)), Some(true));
                }
            }
        }
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
        table.lookup(s(3), 10, &RULE);
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(1));
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

    /// A decision that does not go to the worker it claimed moves the
    /// claim while it stands unconfirmed: the worker claimed leaves the
    /// pins, and the worker the request goes to is the claim in its place,
    /// by the key's recovery. With no worker to go to, the claim is
    /// withdrawn alone and the key is claimed afresh. A claim a dispatch
    /// confirmed stands, and a worker that is a pin already takes no claim.
    #[test]
    fn a_claim_moves_to_the_worker_its_decision_goes_to_while_it_stands_unconfirmed() {
        let (id, pinned) = (WorkerId, SessionVerdict::Pinned);
        let cases = [
            (&RULE, [2, 1], [3, 1], [4, 1]),
            (&ORIGINAL, [1, 2], [1, 3], [1, 4]),
        ];
        for (rule, claimed, moved, committed) in cases {
            // The owner is down: the decision claims 2, then goes to 3.
            let table = SessionTable::new(RULE.sweep_micros);
            table.lookup(s(1), 10, rule);
            table.dispatched(s(1), pinned, id(1));
            let seen = table.lookup(s(1), 20, rule);
            assert_eq!(table.claim(s(1), seen, id(2)), Claim::Made(id(2)));
            // A decision whose claim is not the one of record moves none.
            assert_eq!(table.move_claim(s(1), id(9), id(3)), pinned);
            assert_eq!(table.pins(s(1)), Some(claimed.map(id)));
            let moved_to = SessionVerdict::Claimed(id(3));
            assert_eq!(table.move_claim(s(1), id(2), id(3)), moved_to);
            assert_eq!(table.pins(s(1)), Some(moved.map(id)));
            // The worker it moved to is the claim: the decision's commit
            // elsewhere withdraws that one.
            table.dispatched(s(1), SessionVerdict::Claimed(id(3)), id(4));
            assert_eq!(table.pins(s(1)), Some(committed.map(id)));
            // No worker to go to: the claim is withdrawn alone, and the
            // key's next decision claims afresh.
            let seen = table.lookup(s(2), 10, rule);
            assert_eq!(table.claim(s(2), seen, id(2)), Claim::Made(id(2)));
            assert_eq!(table.move_claim(s(2), id(2), id(0)), pinned);
            assert_eq!(table.pins(s(2)), Some([id(0); 2]));
            assert_eq!(table.claim(s(2), seen, id(5)), Claim::Made(id(5)));
            // Another request was served on the claim: it stands.
            table.dispatched(s(2), pinned, id(5));
            assert_eq!(table.move_claim(s(2), id(5), id(6)), pinned);
            assert_eq!(table.pins(s(2)), Some([id(5), id(0)]));
            // A key the table does not hold is not written.
            assert_eq!(table.move_claim(s(3), id(1), id(2)), pinned);
            assert_eq!(table.len(), 2);
        }
        // Another request's commit made 3 a pin behind the claim on 2: the
        // claim leaves, and 3 stands as the pin it is, marked no claim. A
        // claimant's commit elsewhere then withdraws nothing.
        let table = SessionTable::new(RULE.sweep_micros);
        let seen = table.lookup(s(1), 10, &RULE);
        assert_eq!(table.claim(s(1), seen, id(2)), Claim::Made(id(2)));
        table.dispatched(s(1), pinned, id(3));
        assert_eq!(table.pins(s(1)), Some([id(3), id(2)]));
        assert_eq!(table.move_claim(s(1), id(2), id(3)), pinned);
        assert_eq!(table.pins(s(1)), Some([id(3), id(0)]));
        table.dispatched(s(1), SessionVerdict::Claimed(id(3)), id(9));
        assert_eq!(table.pins(s(1)), Some([id(9), id(3)]));
        // Two other requests' commits pushed the claim on 2 out of the
        // pins: withdrawn to no worker, it leaves them as they stand.
        let table = SessionTable::new(RULE.sweep_micros);
        let seen = table.lookup(s(1), 10, &ORIGINAL);
        assert_eq!(table.claim(s(1), seen, id(2)), Claim::Made(id(2)));
        table.dispatched(s(1), pinned, id(3));
        table.dispatched(s(1), pinned, id(4));
        assert_eq!(table.pins(s(1)), Some([id(3), id(4)]));
        assert_eq!(table.move_claim(s(1), id(2), id(0)), pinned);
        assert_eq!(table.pins(s(1)), Some([id(3), id(4)]));
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

    /// A dispatch that lands behind its worker's rename names the id the
    /// worker had. The pins move under the id that stands, through a
    /// second rename too, and no pin names an id no row holds.
    #[test]
    fn a_dispatch_on_the_id_a_worker_had_moves_the_pins_under_the_id_that_stands() {
        let table = SessionTable::new(RULE.sweep_micros);
        for (key, worker) in [(1, 1), (2, 2)] {
            table.lookup(s(key), 10, &RULE);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
        }
        table.rename_worker(WorkerId(1), WorkerId(11));
        // The key's own late dispatch: its pin stands as renamed.
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(0)]));
        // Another key's respill landed there: the standing id leads.
        table.dispatched(s(2), SessionVerdict::Respill, WorkerId(1));
        assert_eq!(table.pins(s(2)), Some([WorkerId(11), WorkerId(2)]));
        table.rename_worker(WorkerId(11), WorkerId(21));
        table.dispatched(s(1), SessionVerdict::Respill, WorkerId(1));
        assert_eq!(table.pins(s(1)), Some([WorkerId(21), WorkerId(0)]));
        // Back under the id it had first: that id stands again.
        table.rename_worker(WorkerId(21), WorkerId(1));
        table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(11));
        assert_eq!(table.pins(s(2)), Some([WorkerId(1), WorkerId(2)]));
        assert_eq!(table.renamed(), 2, "the two ids it left behind");
    }

    /// A decision reads its rows ahead of its move, and a worker may be
    /// registered again between the two. The move fronts the id that
    /// stands where the rename is over and where its walk has not reached
    /// the key. A pin the walk has yet to rename leads under the id it
    /// had, and the walk renames it there, as it renames a move made
    /// ahead of the whole rename. The dispatch on the id the worker had
    /// finds the pins in place, and no pin names an id no row holds.
    #[test]
    fn a_respills_move_on_the_id_a_worker_had_fronts_the_id_that_stands() {
        let table = SessionTable::new(RULE.sweep_micros);
        for key in [1, 2, 4] {
            table.lookup(s(key), 10, &RULE);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(2));
        }
        table.lookup(s(3), 10, &RULE);
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(4));
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(2));
        assert_eq!(table.pins(s(3)), Some([WorkerId(2), WorkerId(4)]));
        // The whole rename is ahead of the move.
        table.rename_worker(WorkerId(1), WorkerId(11));
        table.respilled(s(1), WorkerId(1));
        assert_eq!(table.pins(s(1)), Some([WorkerId(11), WorkerId(2)]));
        // The rename's first half: remembered, the shards not yet walked.
        table.remember(WorkerId(3), WorkerId(13));
        table.respilled(s(2), WorkerId(3));
        assert_eq!(table.pins(s(2)), Some([WorkerId(13), WorkerId(2)]));
        // The same half where the worker is the key's failover: its pin
        // leads as it stands, and the walk renames it.
        table.remember(WorkerId(4), WorkerId(14));
        table.respilled(s(3), WorkerId(4));
        assert_eq!(table.pins(s(3)), Some([WorkerId(4), WorkerId(2)]));
        // The move is ahead of the whole rename.
        table.respilled(s(4), WorkerId(5));
        assert_eq!(table.pins(s(4)), Some([WorkerId(5), WorkerId(2)]));
        for (had, stands) in [(3, 13), (4, 14), (5, 15)] {
            table.rename_worker(WorkerId(had), WorkerId(stands));
        }
        for (key, had, stands) in [(1, 1, 11), (2, 3, 13), (3, 4, 14), (4, 5, 15)] {
            let led = Some([WorkerId(stands), WorkerId(2)]);
            assert_eq!(table.pins(s(key)), led, "key {key} behind the renames");
            table.dispatched(s(key), SessionVerdict::Respill, WorkerId(had));
            assert_eq!(table.pins(s(key)), led, "key {key} behind its dispatch");
        }
    }

    /// A decision reads its rows ahead of its claim, and its worker may be
    /// registered again between the two. The pin and the claim of record
    /// take the id that stands, behind the whole rename and behind its
    /// first half, and the answer keeps the id the decision read. The
    /// lease names that id: the claim's own dispatch on it confirms the
    /// claim, a later commit elsewhere withdraws no confirmed pin, and the
    /// claiming decision's commit elsewhere withdraws its claim.
    #[test]
    fn a_claim_on_the_id_a_worker_had_pins_the_id_that_stands() {
        for rule in [RULE, ORIGINAL] {
            let table = SessionTable::new(rule.sweep_micros);
            // Worker 1 is registered again whole, worker 3 by the rename's
            // first half: remembered, the shards not yet walked.
            table.rename_worker(WorkerId(1), WorkerId(11));
            table.remember(WorkerId(3), WorkerId(13));
            let claims = [(1, 1, 11), (2, 1, 11), (3, 3, 13), (4, 3, 13)];
            for (key, had, stands) in claims {
                assert_eq!(table.lookup(s(key), 10, &rule), [WorkerId(0); 2]);
                let claim = table.claim(s(key), [WorkerId(0); 2], WorkerId(had));
                assert_eq!(
                    claim,
                    Claim::Made(WorkerId(had)),
                    "the id the decision read"
                );
                let led = Some([WorkerId(stands), WorkerId(0)]);
                assert_eq!(
                    table.pins(s(key)),
                    led,
                    "key {key} claims the id that stands"
                );
            }
            table.rename_worker(WorkerId(3), WorkerId(13));
            for (key, had, stands) in claims {
                let claimed = SessionVerdict::Claimed(WorkerId(had));
                if key % 2 == 1 {
                    // Its own dispatch names the id the decision read.
                    table.dispatched(s(key), claimed, WorkerId(had));
                    let led = Some([WorkerId(stands), WorkerId(0)]);
                    assert_eq!(table.pins(s(key)), led, "key {key} is confirmed");
                    // A pin a dispatch confirmed is withdrawn by no commit.
                    table.dispatched(s(key), claimed, WorkerId(2));
                    let both = match rule.recovery {
                        Recovery::Latest => [WorkerId(2), WorkerId(stands)],
                        Recovery::Original => [WorkerId(stands), WorkerId(2)],
                    };
                    assert_eq!(table.pins(s(key)), Some(both), "key {key} keeps its pin");
                } else {
                    // The claiming decision's commit lands elsewhere.
                    table.dispatched(s(key), claimed, WorkerId(2));
                    let alone = Some([WorkerId(2), WorkerId(0)]);
                    assert_eq!(table.pins(s(key)), alone, "key {key} is withdrawn");
                }
            }
        }
    }

    /// A key pinned on a worker is claimed by a decision that read the
    /// worker's row under the id it had: the key's pin, which names the id
    /// that stands, read as not routable there, and the decision's pick is
    /// the same worker. The table writes nothing and answers the
    /// decision's own pick. A pin the key was served on is marked no
    /// claim, so the request's commit on an alternate withdraws nothing.
    #[test]
    fn a_claim_on_a_worker_pinned_under_the_id_that_stands_writes_nothing() {
        for rule in [RULE, ORIGINAL] {
            let table = SessionTable::new(rule.sweep_micros);
            table.lookup(s(1), 10, &rule);
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
            table.rename_worker(WorkerId(1), WorkerId(11));
            let pins = [WorkerId(11), WorkerId(0)];
            assert_eq!(table.lookup(s(1), 20, &rule), pins);
            let claim = table.claim(s(1), pins, WorkerId(1));
            assert_eq!(claim, Claim::Standing(WorkerId(1), pins));
            assert_eq!(table.pins(s(1)), Some(pins), "nothing is written");
            // The request's dispatch on its pick is one on the pin.
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
            assert_eq!(table.pins(s(1)), Some(pins));
            // Its commit on an alternate moves the pins as a pin's does.
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
            let both = match rule.recovery {
                Recovery::Latest => [WorkerId(2), WorkerId(11)],
                Recovery::Original => [WorkerId(11), WorkerId(2)],
            };
            assert_eq!(table.pins(s(1)), Some(both));
        }
    }

    /// A pair's decision moves its claim to the prefill its pair formed
    /// on, which it read ahead of that write: a prefill registered again
    /// between the two is claimed under the id that stands, behind the
    /// whole rename and behind its first half, and the answer keeps the
    /// id the decision read. Its dispatch on that id confirms the claim,
    /// and its commit elsewhere withdraws it. A prefill that is a pin of
    /// the key under the id that stands is no claim.
    #[test]
    fn a_moved_claim_on_the_id_a_worker_had_pins_the_id_that_stands() {
        for rule in [RULE, ORIGINAL] {
            let table = SessionTable::new(rule.sweep_micros);
            for key in [1, 2, 3] {
                table.lookup(s(key), 10, &rule);
                let claim = table.claim(s(key), [WorkerId(0); 2], WorkerId(5));
                assert_eq!(claim, Claim::Made(WorkerId(5)));
            }
            table.rename_worker(WorkerId(1), WorkerId(11));
            table.remember(WorkerId(3), WorkerId(13));
            for (key, had, stands) in [(1, 1, 11), (2, 1, 11), (3, 3, 13)] {
                let verdict = table.move_claim(s(key), WorkerId(5), WorkerId(had));
                let claimed = SessionVerdict::Claimed(WorkerId(had));
                assert_eq!(verdict, claimed, "the id the decision read");
                let led = Some([WorkerId(stands), WorkerId(0)]);
                assert_eq!(
                    table.pins(s(key)),
                    led,
                    "key {key} claims the id that stands"
                );
            }
            table.rename_worker(WorkerId(3), WorkerId(13));
            for (key, had, stands) in [(1, 1, 11), (3, 3, 13)] {
                table.dispatched(
                    s(key),
                    SessionVerdict::Claimed(WorkerId(had)),
                    WorkerId(had),
                );
                let led = Some([WorkerId(stands), WorkerId(0)]);
                assert_eq!(table.pins(s(key)), led, "key {key} is confirmed");
            }
            table.dispatched(s(2), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
            assert_eq!(
                table.pins(s(2)),
                Some([WorkerId(2), WorkerId(0)]),
                "withdrawn"
            );
            // The key was served on the prefill under the id that stands.
            table.lookup(s(4), 10, &rule);
            table.dispatched(s(4), SessionVerdict::Pinned, WorkerId(11));
            let pins = [WorkerId(11), WorkerId(0)];
            assert_eq!(
                table.claim(s(4), pins, WorkerId(5)),
                Claim::Made(WorkerId(5))
            );
            let verdict = table.move_claim(s(4), WorkerId(5), WorkerId(1));
            assert_eq!(verdict, SessionVerdict::Pinned);
            assert_eq!(
                table.pins(s(4)),
                Some(pins),
                "the pin stands alone, and no claim"
            );
            table.dispatched(s(4), SessionVerdict::Pinned, WorkerId(1));
            assert_eq!(table.pins(s(4)), Some(pins));
        }
    }

    /// The race the table leaves as it stands: the key's own pin is
    /// renamed between a decision's read of the key and its claim. The
    /// pins are not the ones the decision read, and the renamed id answers
    /// as their newcomer. No row of that decision holds it, so the request
    /// keeps its own pick, and its commit moves the pins by the key's
    /// recovery beside the renamed pin.
    #[test]
    fn a_claim_behind_the_rename_of_the_keys_own_pin_answers_the_renamed_pin() {
        for rule in [RULE, ORIGINAL] {
            let table = SessionTable::new(rule.sweep_micros);
            table.lookup(s(1), 10, &rule);
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(5));
            // The decision read the pin, found no row for it and chose 2.
            let seen = table.lookup(s(1), 20, &rule);
            assert_eq!(seen, [WorkerId(5), WorkerId(0)]);
            table.rename_worker(WorkerId(5), WorkerId(15));
            let renamed = [WorkerId(15), WorkerId(0)];
            let claim = table.claim(s(1), seen, WorkerId(2));
            assert_eq!(claim, Claim::Standing(WorkerId(15), renamed));
            assert_eq!(table.pins(s(1)), Some(renamed), "nothing is claimed");
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
            let moved = match rule.recovery {
                Recovery::Latest => [WorkerId(2), WorkerId(15)],
                Recovery::Original => [WorkerId(15), WorkerId(2)],
            };
            assert_eq!(table.pins(s(1)), Some(moved));
        }
    }

    /// A rename says the id that stands ahead of its walk of the shards,
    /// so a pin the walk has yet to reach holds an id its worker had. A
    /// claim of that worker takes the walk's step for the key first: the
    /// pin is read by the id that stands, the claim finds the worker
    /// pinned and answers the decision's own pick, and no pin is written
    /// beside it. In both orders: a worker registered twice and claimed
    /// under its first id while the second rename is half walked, and a
    /// worker registered once and claimed under its new id while that
    /// rename is half walked. Written beside the pin, the claim would be
    /// made one with it by the walk, and the claiming decision's commit
    /// on an alternate would withdraw a pin the key was served on.
    #[test]
    fn a_claim_beside_a_pin_the_renames_walk_has_yet_to_reach_takes_the_walks_step() {
        for rule in [RULE, ORIGINAL] {
            let table = SessionTable::new(rule.sweep_micros);
            // Key 1 is served on a worker registered once since, key 2 on
            // a worker under its first id.
            for (key, worker) in [(1, 1), (2, 3)] {
                table.lookup(s(key), 10, &rule);
                table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(worker));
            }
            table.rename_worker(WorkerId(1), WorkerId(11));
            // Each worker's next rename by its first half: remembered,
            // the shards not yet walked.
            table.remember(WorkerId(11), WorkerId(21));
            table.remember(WorkerId(3), WorkerId(13));
            // Key 1's decision read the worker's first id, key 2's the
            // rows that stand: (key, the pin as held, the id read, the id
            // that stands).
            let orders = [(1, 11, 1, 21), (2, 3, 13, 13)];
            for (key, held, read, stands) in orders {
                let pins = [WorkerId(held), WorkerId(0)];
                assert_eq!(table.lookup(s(key), 20, &rule), pins);
                let claim = table.claim(s(key), pins, WorkerId(read));
                let led = [WorkerId(stands), WorkerId(0)];
                assert_eq!(claim, Claim::Standing(WorkerId(read), led), "key {key}");
                assert_eq!(table.pins(s(key)), Some(led), "key {key}: the step alone");
            }
            table.rename_worker(WorkerId(11), WorkerId(21));
            table.rename_worker(WorkerId(3), WorkerId(13));
            for (key, _, read, stands) in orders {
                let led = [WorkerId(stands), WorkerId(0)];
                assert_eq!(table.pins(s(key)), Some(led), "key {key} behind the walk");
                // The decision's dispatch on its pick is one on the pin.
                table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(read));
                assert_eq!(table.pins(s(key)), Some(led), "key {key} on its pick");
                // Its commit on an alternate withdraws nothing.
                table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(2));
                let both = match rule.recovery {
                    Recovery::Latest => [WorkerId(2), WorkerId(stands)],
                    Recovery::Original => [WorkerId(stands), WorkerId(2)],
                };
                assert_eq!(table.pins(s(key)), Some(both), "key {key} keeps its pin");
            }
        }
    }

    /// A pair's moved claim beside such a pin takes the walk's step too:
    /// the prefill its pair formed on is the key's pin, under an id the
    /// second rename's walk has yet to reach, and the decision read the
    /// first. The pin takes the id that stands, and nothing is claimed.
    #[test]
    fn a_moved_claim_beside_a_pin_the_renames_walk_has_yet_to_reach_takes_the_walks_step() {
        for rule in [RULE, ORIGINAL] {
            let table = SessionTable::new(rule.sweep_micros);
            table.lookup(s(1), 10, &rule);
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
            table.rename_worker(WorkerId(1), WorkerId(11));
            let pins = [WorkerId(11), WorkerId(0)];
            // The decision claims another prefill, then pairs on the
            // worker under its first id, the second rename half walked.
            let claim = table.claim(s(1), pins, WorkerId(5));
            assert_eq!(claim, Claim::Made(WorkerId(5)));
            table.remember(WorkerId(11), WorkerId(21));
            let verdict = table.move_claim(s(1), WorkerId(5), WorkerId(1));
            assert_eq!(verdict, SessionVerdict::Pinned);
            let led = [WorkerId(21), WorkerId(0)];
            assert_eq!(table.pins(s(1)), Some(led), "the pin alone, no claim");
            table.rename_worker(WorkerId(11), WorkerId(21));
            // The pair's commit on an alternate withdraws nothing.
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(2));
            let both = match rule.recovery {
                Recovery::Latest => [WorkerId(2), WorkerId(21)],
                Recovery::Original => [WorkerId(21), WorkerId(2)],
            };
            assert_eq!(table.pins(s(1)), Some(both));
        }
    }

    /// A commit and a respill's move on an id a worker had meet the
    /// worker's own pin where a rename's walk has yet to reach it: the
    /// pin takes the id that stands, the worker is pinned once, and the
    /// key's other pin stays.
    #[test]
    fn a_commit_and_a_move_beside_a_pin_the_walk_has_yet_to_reach_pin_the_worker_once() {
        let table = SessionTable::new(RULE.sweep_micros);
        for key in [1, 2] {
            table.lookup(s(key), 10, &RULE);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(1));
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(4));
        }
        table.rename_worker(WorkerId(1), WorkerId(11));
        table.remember(WorkerId(11), WorkerId(21));
        assert_eq!(table.pins(s(1)), Some([WorkerId(4), WorkerId(11)]));
        // A commit on the worker's first id is the failover taken.
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        let taken = Some([WorkerId(4), WorkerId(21)]);
        assert_eq!(table.pins(s(1)), taken, "the pins keep their order");
        // A respill's move there fronts the worker's own pin.
        table.respilled(s(2), WorkerId(1));
        let fronted = Some([WorkerId(21), WorkerId(4)]);
        assert_eq!(table.pins(s(2)), fronted);
        table.rename_worker(WorkerId(11), WorkerId(21));
        assert_eq!(table.pins(s(1)), taken);
        assert_eq!(table.pins(s(2)), fronted);
        // A commit on a worker's new id inside its first rename's walk:
        // the key's leading pin is the worker's, and the failover stays.
        table.lookup(s(3), 10, &RULE);
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(4));
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(5));
        table.remember(WorkerId(5), WorkerId(15));
        table.dispatched(s(3), SessionVerdict::Pinned, WorkerId(15));
        let pins = Some([WorkerId(15), WorkerId(4)]);
        assert_eq!(table.pins(s(3)), pins, "the failover stays");
        table.rename_worker(WorkerId(5), WorkerId(15));
        assert_eq!(table.pins(s(3)), pins);
    }

    /// A key that holds one worker under two ids, half walked: its pin
    /// under the id the worker had, and a claim written under the new id
    /// ahead of the table's word of the rename. Inside that rename's walk
    /// a dispatch on an id the worker had is one on the claimed worker,
    /// and a respill fronts another worker. The first write that asks
    /// takes the walk's step for the key: the two pins are one, and the
    /// claim's mark is gone. The claiming decision's commit on the
    /// respill's worker then withdraws nothing, with the dispatch between
    /// the two and without it, as behind the whole walk.
    #[test]
    fn a_key_that_holds_one_worker_twice_is_walked_by_its_first_write_that_asks() {
        for rule in [RULE, ORIGINAL] {
            for dispatch in [true, false] {
                let said = (rule.recovery, dispatch);
                let table = SessionTable::new(rule.sweep_micros);
                table.lookup(s(1), 10, &rule);
                table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
                table.rename_worker(WorkerId(1), WorkerId(11));
                let served = [WorkerId(11), WorkerId(0)];
                // The decision read the row that replaced the worker's,
                // ahead of the table's word of that rename.
                let claim = table.claim(s(1), served, WorkerId(21));
                assert_eq!(claim, Claim::Made(WorkerId(21)), "{said:?}");
                // The rename's first half: remembered, the shards not
                // yet walked.
                table.remember(WorkerId(11), WorkerId(21));
                if dispatch {
                    table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
                    let one = [WorkerId(21), WorkerId(0)];
                    assert_eq!(table.pins(s(1)), Some(one), "{said:?}: one pin");
                }
                table.respilled(s(1), WorkerId(2));
                let moved = [WorkerId(2), WorkerId(21)];
                assert_eq!(table.pins(s(1)), Some(moved), "{said:?}: the move");
                table.rename_worker(WorkerId(11), WorkerId(21));
                let claimed = SessionVerdict::Claimed(WorkerId(21));
                table.dispatched(s(1), claimed, WorkerId(2));
                assert_eq!(table.pins(s(1)), Some(moved), "{said:?}: the pin stands");
            }
        }
    }

    /// A claim's mark outlives its pin where a respill drops the pin: the
    /// mark then names an id no pin of the key holds. Under `original` a
    /// claim written ahead of the table's word of a rename stands behind
    /// the key's served pin, and a respill ahead of that word drops it.
    /// The rename brings the served pin under the mark's id, and the mark
    /// goes there: that pin is no claim, and the claiming decision's
    /// commit on the respill's worker withdraws nothing.
    #[test]
    fn a_mark_whose_pin_a_respill_dropped_goes_where_a_rename_brings_a_pin_under_it() {
        let table = SessionTable::new(ORIGINAL.sweep_micros);
        table.lookup(s(1), 10, &ORIGINAL);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        let served = [WorkerId(1), WorkerId(0)];
        let claim = table.claim(s(1), served, WorkerId(11));
        assert_eq!(claim, Claim::Made(WorkerId(11)));
        table.respilled(s(1), WorkerId(2));
        assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(1)]));
        table.rename_worker(WorkerId(1), WorkerId(11));
        let moved = [WorkerId(2), WorkerId(11)];
        assert_eq!(table.pins(s(1)), Some(moved));
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(11)), WorkerId(2));
        assert_eq!(table.pins(s(1)), Some(moved), "the served pin stands");
    }

    /// The limit of a claim written ahead of the table's word of a
    /// rename, held as it stands. Under `latest` the claim leads the
    /// key's served pin, and a respill ahead of that word drops the
    /// served pin as it drops any failover: nothing in the table says the
    /// two ids are one worker yet. The claim stands unconfirmed behind
    /// the rename, and the claiming decision's commit on the respill's
    /// worker withdraws it: the key holds no pin on that worker, where a
    /// respill behind the walk keeps one.
    #[test]
    fn a_respill_ahead_of_the_tables_word_drops_the_served_pin_under_latest() {
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(1), 10, &RULE);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        let served = [WorkerId(1), WorkerId(0)];
        let claim = table.claim(s(1), served, WorkerId(11));
        assert_eq!(claim, Claim::Made(WorkerId(11)));
        table.respilled(s(1), WorkerId(2));
        assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(11)]));
        table.rename_worker(WorkerId(1), WorkerId(11));
        table.dispatched(s(1), SessionVerdict::Claimed(WorkerId(11)), WorkerId(2));
        assert_eq!(table.pins(s(1)), Some([WorkerId(2), WorkerId(0)]));
    }

    /// A claim written ahead of the table's word of a rename finds no
    /// rename to ask: a host replaces a worker's row before it tells the
    /// table, and a decision between the two reads a pin no pool holds
    /// and is placed by the rule. The key then holds the worker twice, as
    /// a claim and as a pin, in both orders: a claim on the new row beside
    /// the pin under the id the worker had, and a commit on the new row
    /// beside a claim under that id. The rename's walk makes the two pins
    /// one and the claim's mark goes there: the key holds the worker as a
    /// pin, and the claiming decision's commit on an alternate withdraws
    /// nothing.
    #[test]
    fn a_claim_made_one_with_the_keys_pin_by_a_rename_is_confirmed() {
        for rule in [RULE, ORIGINAL] {
            let table = SessionTable::new(rule.sweep_micros);
            // Key 1 was served on the worker and is claimed on the row
            // that replaced it.
            table.lookup(s(1), 10, &rule);
            table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
            let served = [WorkerId(1), WorkerId(0)];
            let claim = table.claim(s(1), served, WorkerId(11));
            assert_eq!(claim, Claim::Made(WorkerId(11)));
            // Key 2 is claimed on the worker and served on the row that
            // replaced it.
            table.lookup(s(2), 10, &rule);
            let claim = table.claim(s(2), [WorkerId(0); 2], WorkerId(1));
            assert_eq!(claim, Claim::Made(WorkerId(1)));
            table.dispatched(s(2), SessionVerdict::Pinned, WorkerId(11));
            table.rename_worker(WorkerId(1), WorkerId(11));
            // (key, the id its claim's lease names)
            for (key, leased) in [(1, 11), (2, 1)] {
                let led = [WorkerId(11), WorkerId(0)];
                assert_eq!(table.pins(s(key)), Some(led), "key {key} holds one pin");
                // The claiming decision's commit on an alternate.
                let claimed = SessionVerdict::Claimed(WorkerId(leased));
                table.dispatched(s(key), claimed, WorkerId(2));
                let both = match rule.recovery {
                    Recovery::Latest => [WorkerId(2), WorkerId(11)],
                    Recovery::Original => [WorkerId(11), WorkerId(2)],
                };
                assert_eq!(table.pins(s(key)), Some(both), "key {key} keeps its pin");
            }
        }
    }

    /// A claim's commit elsewhere withdraws it inside a rename's walk too:
    /// the renames already say the new id, the entry still holds the old
    /// one, and the claim is matched by the id its lease names.
    #[test]
    fn a_claims_commit_elsewhere_inside_the_renames_walk_withdraws_it() {
        let table = SessionTable::new(ORIGINAL.sweep_micros);
        table.lookup(s(9), 10, &ORIGINAL);
        let claim = table.claim(s(9), [WorkerId(0); 2], WorkerId(1));
        assert_eq!(claim, Claim::Made(WorkerId(1)));
        // The rename's first half: remembered, the shard not yet walked.
        table.remember(WorkerId(1), WorkerId(11));
        table.dispatched(s(9), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
        table.rename_worker(WorkerId(1), WorkerId(11));
        assert_eq!(table.pins(s(9)), Some([WorkerId(2), WorkerId(0)]));
    }

    /// A claim outlives the renames the table forgets: the id its lease
    /// names is on its entry, so its commit elsewhere withdraws it behind
    /// a list that started over.
    #[test]
    fn a_claim_renamed_ahead_of_the_bound_is_withdrawn_by_its_commit() {
        let table = SessionTable::new(ORIGINAL.sweep_micros);
        table.lookup(s(9), 10, &ORIGINAL);
        let claim = table.claim(s(9), [WorkerId(0); 2], WorkerId(1));
        assert_eq!(claim, Claim::Made(WorkerId(1)));
        table.rename_worker(WorkerId(1), WorkerId(11));
        let bound = RENAMED_WORKERS as u64;
        for life in 0..bound {
            table.rename_worker(WorkerId(100_000 + life), WorkerId(200_000 + life));
        }
        assert_eq!(table.renamed(), 1, "the list started over");
        table.dispatched(s(9), SessionVerdict::Claimed(WorkerId(1)), WorkerId(2));
        assert_eq!(table.pins(s(9)), Some([WorkerId(2), WorkerId(0)]));
    }

    /// The table remembers a bounded count of renames: at the bound the
    /// list starts over, and the rename that follows is remembered.
    #[test]
    fn the_renames_the_table_remembers_are_bounded() {
        let table = SessionTable::new(RULE.sweep_micros);
        table.lookup(s(1), 10, &RULE);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(1));
        let bound = RENAMED_WORKERS as u64;
        for life in 1..=bound {
            table.rename_worker(WorkerId(life), WorkerId(life + 1));
        }
        assert_eq!(table.renamed(), RENAMED_WORKERS);
        table.rename_worker(WorkerId(bound + 1), WorkerId(bound + 2));
        assert_eq!(table.renamed(), 1);
        table.dispatched(s(1), SessionVerdict::Pinned, WorkerId(bound + 1));
        assert_eq!(table.pins(s(1)), Some([WorkerId(bound + 2), WorkerId(0)]));
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
    /// the pass over them: the pass keeps its own instant. The key past its
    /// end stays until the pass is due, as with no leave.
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
        // Key 1 is past its end, the sweep at 1,000, and the leave's sweep
        // runs.
        table.lookup(s(2), 1_005, &rule);
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

    /// A session of the sizing cases: a counter mixed over every bit, as a
    /// session's value is.
    fn mixed(count: u64) -> Session {
        s(count.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    #[test]
    fn a_table_of_any_size_keeps_every_key_and_is_sized_by_a_power_of_two() {
        for (asked, shards) in [(0, 1), (1, 1), (3, 4), (16, 16), (1_000, 1_024)] {
            let table = SessionTable::new(RULE.sweep_micros).with_shards(asked);
            assert_eq!(table.shards(), shards, "{asked} shards asked");
            for count in 1..=200 {
                let key = mixed(count);
                assert_eq!(table.lookup(key, 10, &RULE), [WorkerId(0); 2]);
                table.dispatched(key, SessionVerdict::Pinned, WorkerId(count));
                assert_eq!(table.lookup(key, 11, &RULE)[0], WorkerId(count));
            }
            assert_eq!(table.len(), 200);
            // A walk of the table reaches every shard.
            table.forget_workers(&[WorkerId(200)]);
            assert_eq!(table.pins(mixed(200)), None);
            assert_eq!(table.len(), 199);
        }
        // A table no host sized holds the count it always held.
        assert_eq!(SessionTable::new(0).shards(), SHARDS);
        assert!(shards_for_this_host().is_power_of_two());
        assert!(shards_for_this_host() >= 4);
    }

    /// The keys of a sized table spread over its shards: no shard holds
    /// many times its share, so two decisions of different keys seldom
    /// take one mutex.
    #[test]
    fn a_sized_tables_keys_spread_over_its_shards() {
        let table = SessionTable::new(0).with_shards(64);
        let rule = SessionRule {
            ttl_micros: None,
            sweep_micros: 0,
            ..RULE
        };
        let keys = 64 * 100;
        for count in 1..=keys {
            table.lookup(mixed(count), 10, &rule);
        }
        let held: Vec<usize> = table.shards.iter().map(|s| s.0.lock().len()).collect();
        assert_eq!(held.iter().sum::<usize>(), keys as usize);
        let (least, most) = (held.iter().min(), held.iter().max());
        assert!(
            least >= Some(&50) && most <= Some(&200),
            "a share of 100 a shard: {least:?} to {most:?}"
        );
    }

    /// A full table records no further key, in whichever shard it falls:
    /// such a key reads no pin, is written none and claims none, while the
    /// keys the table holds are answered. Keys that leave at their end give
    /// their room back, and their shard its spare storage.
    #[test]
    fn a_full_table_records_no_further_key_and_answers_the_keys_it_holds() {
        let table = SessionTable::new(RULE.sweep_micros);
        for key in 1..=8 {
            table.lookup(s(key), 10, &RULE);
            table.dispatched(s(key), SessionVerdict::Pinned, WorkerId(1));
        }
        // One key under the 4,194,304 a table holds: the count stands for
        // the keys no test inserts, and key 9 takes the last room.
        table.held.0.store(4_194_303, Ordering::Relaxed);
        table.lookup(s(9), 10, &RULE);
        table.dispatched(s(9), SessionVerdict::Pinned, WorkerId(1));
        assert_eq!(table.held.0.load(Ordering::Relaxed), KEYS);
        // A key of another shard finds the table full.
        let fresh = s(1 << 32);
        let none = [WorkerId(0); 2];
        assert_eq!(table.lookup(fresh, 20, &RULE), none);
        assert_eq!(table.claim(fresh, none, WorkerId(2)), Claim::NoRoom);
        table.dispatched(fresh, SessionVerdict::Pinned, WorkerId(2));
        assert_eq!(table.pins(fresh), None);
        // A key the table holds is read and renewed as before.
        assert_eq!(table.lookup(s(1), 500, &RULE), [WorkerId(1), WorkerId(0)]);
        // The other eight end at 1 100 and leave at the pass.
        let room = table.shards[0].0.lock().capacity();
        table.sweep_due(1_200);
        assert_eq!(table.held.0.load(Ordering::Relaxed), KEYS - 8);
        assert!(table.shards[0].0.lock().capacity() < room);
        assert_eq!(table.lookup(fresh, 1_200, &RULE), none);
        assert_eq!(
            table.claim(fresh, none, WorkerId(2)),
            Claim::Made(WorkerId(2))
        );
        assert_eq!(table.held.0.load(Ordering::Relaxed), KEYS - 7);
    }

    #[test]
    fn a_shard_stands_on_cache_lines_of_its_own() {
        assert_eq!(align_of::<Shard>(), 128);
        assert_eq!(size_of::<Shard>(), 128);
    }

    /// A host that sweeps on a timer of its own drops the keys past their
    /// end with no decision's read, and a sweep that is not due walks
    /// nothing.
    #[test]
    fn a_hosts_sweep_drops_the_keys_past_their_end_with_no_read() {
        let table = SessionTable::new(RULE.sweep_micros).with_shards(8);
        table.lookup(mixed(1), 0, &RULE);
        table.dispatched(mixed(1), SessionVerdict::Pinned, WorkerId(1));
        table.lookup(mixed(2), 900, &RULE);
        table.dispatched(mixed(2), SessionVerdict::Pinned, WorkerId(2));
        // The read at 900 swept and set the next pass a sweep interval on:
        // a host's call ahead of it walks nothing.
        table.sweep_due(950);
        assert_eq!(table.len(), 2);
        // Past the interval and the first key's end.
        table.sweep_due(1_100);
        assert_eq!(table.pins(mixed(1)), None);
        assert_eq!(table.pins(mixed(2)), Some([WorkerId(2), WorkerId(0)]));
    }

    /// A read sweeps where a sweep is due until a pass of the table's host
    /// has run. From the pass's first run a read walks no shard but its
    /// key's, however long a sweep has been due: a key past its end reads
    /// as no pins and is held until the pass, which drops it.
    #[test]
    fn a_read_leaves_every_sweep_to_the_hosts_pass_once_one_has_run() {
        let table = SessionTable::new(RULE.sweep_micros).with_shards(8);
        let pin = |key: u64, at: u64| {
            table.lookup(mixed(key), at, &RULE);
            table.dispatched(mixed(key), SessionVerdict::Pinned, WorkerId(key));
        };
        // No pass has run: the read at 1 100 sweeps the key past its end.
        pin(1, 0);
        pin(2, 1_100);
        assert_eq!(table.pins(mixed(1)), None, "a read swept");
        assert_eq!(table.len(), 1);
        // The pass runs, with no sweep due. Every sweep is its own now.
        table.sweep_due(1_150);
        assert_eq!(table.len(), 1);
        // Far past the second key's end and many sweep intervals on: reads
        // of other keys sweep nothing, and the key's own read finds none.
        pin(3, 9_000);
        pin(4, 9_500);
        assert_eq!(table.pins_at(mixed(2), 9_500), [WorkerId(0); 2]);
        assert_eq!(table.len(), 3, "the key past its end is held for the pass");
        table.sweep_due(9_600);
        assert_eq!(table.pins(mixed(2)), None, "the pass dropped it");
        assert_eq!(table.len(), 2);
    }
}
