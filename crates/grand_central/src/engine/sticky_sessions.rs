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

/// A binding's sessions rule as the engine runs it. A document's validation
/// makes it, and code outside this crate states a sessions rule by a
/// document: it builds none, so a binding it makes in code holds none, or
/// one it read from a document's binding. The record gains fields: a pattern
/// of it there ends with `..`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// the worker, or withdraws it when the dispatch landed elsewhere or it
    /// refused every target.
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
#[non_exhaustive]
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
    /// decision read, the worker is a pin of the key already, one the
    /// decision read or one under another id, or nothing was recorded (an
    /// empty key or target).
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
    /// when none stands. Any dispatch of the key's current life to it
    /// confirms it (the key was served there), the claiming decision's
    /// dispatch elsewhere withdraws it while
    /// it stands unconfirmed, and no other dispatch elsewhere touches it; a
    /// claim whose commit never comes stands until the TTL sweeps the key.
    /// A pair's decision moves its own claim to the prefill its pair formed
    /// on, and withdraws it where none formed; a commit that refused every
    /// target withdraws its decision's ([`SessionTable::move_claim`]).
    /// A rename's step takes the mark where the key held the claim's
    /// worker under another id as well ([`Entry::follow`]).
    claim: WorkerId,
    /// The id the claim's lease names: the id the claiming decision read
    /// for its worker, read while a claim stands. It differs from `claim`
    /// where the worker was registered again, under the claim
    /// ([`SessionTable::rename_worker`]) or between the decision's read
    /// of its rows and its claim ([`SessionTable::stands`]).
    leased: WorkerId,
    /// The instant the claim's lease is issued under, by its low 32 bits,
    /// read while a claim stands. The word of a lease reaches the claim
    /// where the lease names the claim's id and this instant
    /// ([`Entry::reached_by`]): a later decision's claim on the same worker
    /// is dated later. Two claims of one key on one worker a whole multiple
    /// of 2^32 microseconds apart read as one, the answer by the worker
    /// alone.
    claimed_at: u32,
    /// The least date under which a decision that read this life of the
    /// key tells its words, on the decision clock. It never rises inside
    /// a life.
    ///
    /// Four calls write it. The read that begins the life writes its
    /// instant ([`SessionTable::lookup`]). A reader with an older instant
    /// lowers it ([`Entry::touch`]). So does a claim call, which reads
    /// the pins as they stand ([`SessionTable::claim`]). So does a decode
    /// call that read this life, by the date of its pair's lease
    /// (`SessionTable::told_under`, with the in-process engine).
    ///
    /// Three doors read it, by one test ([`Entry::outlived`]). A call
    /// dated before it comes from a decision that read no pin of this
    /// life, and writes nothing. The three are a dispatch
    /// ([`SessionTable::dispatched`]), a move at the cap
    /// ([`SessionTable::respilled`]) and a decode call's date. `0` in an
    /// entry no decision has read: every word is written there.
    oldest_reader_micros: u64,
}

impl Entry {
    fn touch(&mut self, now_micros: u64, rule: &SessionRule) {
        if now_micros < self.touched_at_micros {
            // A decision that read its clock ahead of another reads this
            // life behind it, and tells its words under its own instant.
            self.read_under(now_micros);
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

    /// Whether `id` is one of the key's two slots as written. The empty id
    /// is one where a slot is free: a write that names no worker asks so
    /// ([`SessionTable::move_claim`]).
    fn holds(&self, id: WorkerId) -> bool {
        self.pins.contains(&id)
    }

    /// The worker a decision follows where it read `seen` and chose
    /// `target`, by the claim's rule ([`Claim::standing`]). None where the
    /// pins stand as it read them. The answer can be the decision's own
    /// `target`: no pin came in.
    fn newcomer(&self, seen: &SessionPins, target: WorkerId) -> Option<WorkerId> {
        Claim::standing(&self.pins, seen, target)
    }

    /// `target` becomes the preferred worker; the preferred one before it the
    /// failover.
    fn front(&mut self, target: WorkerId) {
        if self.pins[0] != target {
            self.pins = [target, self.pins[0]];
        }
    }

    /// A decision's claim: `stands` joins the pins by the key's recovery
    /// ([`Entry::place`]) and is written as the claim of record. `leased`
    /// is the id the claim's lease names, and `at` the instant that lease
    /// is issued under.
    fn claimed(&mut self, stands: WorkerId, leased: WorkerId, at: u64) {
        self.place(stands);
        self.claim = stands;
        self.leased = leased;
        // The low 32 bits: the map's slot keeps its size.
        self.claimed_at = at as u32;
    }

    /// Whether the word of a lease reaches the claim of record: a claim
    /// stands, its lease names `leased`, and that lease was issued at `at`.
    /// A word that states no instant (`0`) reaches a claim by its worker
    /// alone.
    fn reached_by(&self, leased: WorkerId, at: u64) -> bool {
        self.claim != WorkerId(0)
            && self.leased == leased
            && (at == 0 || self.claimed_at == at as u32)
    }

    /// Dates the claim of record by `issued`, the instant of the lease its
    /// words are told under, where that claim is the one `leased` and `at`
    /// name ([`Self::reached_by`]). Its caller hands the date of a lease
    /// that states one ([`SessionTable::told_under`]).
    #[cfg(any(test, feature = "in-process"))]
    fn dated(&mut self, leased: WorkerId, at: u64, issued: u64) {
        if self.reached_by(leased, at) {
            self.claimed_at = issued as u32;
        }
    }

    /// A key's life as the decision at `now_micros` begins it: no pin, and
    /// that decision its oldest reader.
    fn begun(now_micros: u64) -> Self {
        Self {
            oldest_reader_micros: now_micros,
            ..Self::default()
        }
    }

    /// A decision that tells its words under `date` read this life of the
    /// key: the oldest reader's date falls to it where it is older.
    fn read_under(&mut self, date: u64) {
        if date < self.oldest_reader_micros {
            self.oldest_reader_micros = date;
        }
    }

    /// Whether a word dated `at` comes from an older life of the key: it
    /// is dated before the oldest date a decision read this one under. A
    /// pair's decode call reads under its lease's date
    /// ([`SessionTable::told_under`]). A word that states no instant (`0`)
    /// does not. Neither does any word in an entry no decision has read.
    fn outlived(&self, at: u64) -> bool {
        at != 0 && at < self.oldest_reader_micros
    }

    /// The id a dispatch on `id` asks the table by: the claim of record
    /// where a claim stands and its lease names `id`, else `id` itself.
    /// A claim's lease names the id its worker had when it claimed: a
    /// dispatch on that id is one on the claimed worker, registered
    /// again since or not.
    fn of_record(&self, id: WorkerId) -> WorkerId {
        if self.claim != WorkerId(0) && self.leased == id {
            self.claim
        } else {
            id
        }
    }

    /// A dispatch to `worker` confirms the claim on it: the mark goes and
    /// its pin stands, a pin the key was served on.
    fn confirm_claim(&mut self, worker: WorkerId) {
        if self.claim != WorkerId(0) && self.claim == worker {
            self.claim = WorkerId(0);
        }
    }

    /// Withdraws the claim of record: the mark goes, and the claim's pin
    /// leaves the pins (the pin behind it moves up). Its commit never came:
    /// its decision moved the claim or gave it up, or the commit went to
    /// another worker.
    fn withdraw_claim(&mut self) {
        let claim = std::mem::replace(&mut self.claim, WorkerId(0));
        if self.pins[0] == claim {
            self.pins = [self.pins[1], WorkerId(0)];
        } else if self.pins[1] == claim {
            self.pins[1] = WorkerId(0);
        }
    }

    /// The key was served on `worker` under its decision's `verdict`. The
    /// preferred pin, or the failover taken under no respill, leaves the
    /// pins as they are. A respill fronts the worker; under no respill a
    /// worker outside the pins joins them by the key's recovery
    /// ([`Entry::place`]). The verdict's compare with a respill stands
    /// behind the preferred pin's test, as the table held it: a request
    /// that follows its pin does not make it.
    fn served(&mut self, worker: WorkerId, verdict: SessionVerdict) {
        if self.pins[0] == worker {
            return;
        }
        let respill = verdict == SessionVerdict::Respill;
        if !respill && self.pins.contains(&worker) {
            return;
        }
        if respill {
            self.front(worker);
        } else {
            self.place(worker);
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

    /// `worker` joins the pins by the key's recovery: in front under
    /// `latest`, behind the pins there under `original`.
    fn place(&mut self, worker: WorkerId) {
        match self.recovery {
            Recovery::Latest => self.front(worker),
            Recovery::Original => self.append(worker),
        }
    }

    /// Takes the pins of `workers` off the key. False where that takes the
    /// key's last pin: it leaves the table and is placed afresh. A key
    /// that holds no pin stays: a decision read it and its claim follows
    /// ([`SessionTable::lookup`]), and the idle keys' pass alone reclaims it.
    fn unpin(&mut self, workers: &[WorkerId]) -> bool {
        let held = self.pins[0] != WorkerId(0);
        if workers.contains(&self.pins[1]) {
            self.pins[1] = WorkerId(0);
        }
        if workers.contains(&self.pins[0]) {
            self.pins = [self.pins[1], WorkerId(0)];
        }
        !held || self.pins[0] != WorkerId(0)
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
            // This decision is the new life's first reader.
            *entry = Entry::begun(now_micros);
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
    /// any dispatch of the key's current life to it confirms it, the
    /// claiming decision's own dispatch elsewhere withdraws it
    /// (`dispatched`), the TTL sweeps it otherwise.
    /// A pair's decision moves its own claim before its dispatch, and a
    /// commit that refused every target withdraws its decision's
    /// ([`Self::move_claim`]).
    ///
    /// The claim keeps `at`, the instant its decision's lease is issued
    /// under. The table knows the claiming decision by it: a word of a
    /// lease moves or withdraws the claim where the lease names the
    /// claim's id and that instant ([`Entry::reached_by`]). A decode call
    /// claims at its own instant, and its claim is dated again by its
    /// pair's lease (`Self::told_under`, with the in-process engine).
    ///
    /// The call reads the key's pins as they stand and answers by them:
    /// it is a read of the key's life. Where the key started over between
    /// the decision's first read and this call, the new life takes `at`
    /// as a reader's date ([`Entry::read_under`]). The decision's
    /// dispatch is then written into the life its call read. A call that
    /// states no instant (`0`) lowers nothing.
    ///
    /// The pin and the claim of record take the id that stands for the
    /// worker ([`Self::stands`]): the decision read its rows ahead of this
    /// write, and the worker may have been registered again between the
    /// two. The lease is issued on the id the decision read, so that id is
    /// the lease's and the answer's, as behind a rename that finds a
    /// claim made ahead of it ([`Self::rename_worker`]). A worker that is
    /// a pin of the key already, under the id that stands or under one it
    /// had that a rename's walk has yet to reach, is no claim: the answer
    /// names the decision's own pick, and no pin is written beside it. A
    /// pin the decision read is such a worker too. A claim would mark a
    /// worker the key was served on, and the decision's commit on an
    /// alternate would withdraw it.
    ///
    /// The ask comes ahead of the read of the pins against `seen`. Where
    /// it takes a rename's step for the key ([`Self::standing`]), that
    /// read runs on the walked entry: a pin the step renamed is the
    /// newcomer to follow, as behind the walk, and nothing is claimed.
    /// Pins that moved under the decision take the step whatever its
    /// worker, where the table remembers a rename: the answer names a
    /// pin, and a newcomer under an id its worker had is one no row of
    /// the decision holds. Pins the step makes the ones the decision
    /// read again stand as read: a worker that is a pin already is no
    /// claim there either.
    /// Always inlined: the keyed decision has two forms, and a second caller
    /// would otherwise move this out of line under the first.
    #[inline(always)]
    pub(crate) fn claim(
        &self,
        key: Session,
        seen: SessionPins,
        target: WorkerId,
        at: u64,
    ) -> Claim {
        let key = key.get();
        if key == 0 || target == WorkerId(0) {
            return Claim::Standing(target, seen);
        }
        let mut shard = self.shard(key).lock();
        let Some(entry) = self.slot(&mut shard, key) else {
            return Claim::NoRoom;
        };
        // A read of the key's life as it stands, whatever the answer. A
        // call that states no instant lowers nothing: a zero date would
        // open the life to every older word.
        if at != 0 {
            entry.read_under(at);
        }
        // The pins as written, read ahead of the ask. Where they are
        // the ones the decision read, the ask is a write's
        // (`stands`). Where they moved, the answer names their
        // newcomer: the key takes a pending rename's step whatever
        // the target, a pin as written too.
        let stood = entry.pins == seen;
        let stands = if stood {
            self.stands(entry, target)
        } else if self.renamed_len.load(Ordering::Relaxed) != 0 {
            self.standing(entry, target)
        } else {
            target
        };
        if (stood || entry.pins == seen) && entry.holds(stands) {
            // The pins stand as the decision read them, as written or
            // behind the step, and the worker is a pin among them: one
            // the decision read, or one under the id that stands. The
            // key was served on it, or another decision claimed it. A
            // claim of this decision's would mark it again, and this
            // decision's commit on an alternate would withdraw it.
            return Claim::Standing(target, entry.pins);
        }
        // The pins are not the ones the decision read: another
        // decision moved them, or the step renamed one. Their
        // newcomer is the pin to follow, by the id that stands.
        if let Some(owner) = entry.newcomer(&seen, target) {
            return Claim::Standing(owner, entry.pins);
        }
        entry.claimed(stands, target, at);
        Claim::Made(target)
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
    /// A pair's decision moves its prefill side's pin where the pair
    /// forms, to the prefill the pair formed on. A prefill its side placed
    /// first and passed over is fronted at no instant, and a decision
    /// that forms no pair moves no pin.
    ///
    /// `at` is the moving decision's instant. A move dated before every
    /// decision that read the key's current life writes nothing
    /// ([`Entry::outlived`]). The key started over since the moving
    /// decision read its pin, and the move would front a worker over
    /// pins that decision never read. A caller that states no instant
    /// (`0`) moves the pin in any life.
    pub(crate) fn respilled(&self, key: Session, target: WorkerId, at: u64) {
        let key = key.get();
        if target == WorkerId(0) {
            return;
        }
        if let Some(entry) = self.shard(key).lock().get_mut(&key) {
            if entry.outlived(at) {
                return;
            }
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
    /// none (`WorkerId(0)`), or its commit refused every target
    /// (`WorkerId(0)` too). The claim is withdrawn where it still stands
    /// unconfirmed, as the claiming decision's commit elsewhere withdraws
    /// it ([`Self::dispatched`]), and `target` is claimed in its place by
    /// the key's recovery: the key's next decision follows a worker the
    /// request goes to, before any dispatch. The answer is the verdict the
    /// decision states of `target`: its claim where `target` is the claim
    /// of record, else a pin's. A claim a dispatch confirmed is a pin the
    /// key was served on: it stands, and nothing is written. Another
    /// decision's claim is not this one's to move. A `target` that is a pin
    /// of the key already, under any id of its worker, is no claim either.
    /// The key takes a pending rename's step first ([`Self::stands`]): a
    /// claim that step takes the mark of is a pin, and it stands too.
    /// The new claim is written as a first one is ([`Self::claim`]): the
    /// pin and the claim of record under the id that stands for `target`,
    /// the lease's id and the answer under the id the decision read.
    ///
    /// `at` is the instant the caller's lease is issued under. The claim of
    /// record is the caller's where its lease names `claimed` and was
    /// issued at `at` ([`Entry::reached_by`]). A later decision's claim on
    /// the same worker is dated later, so a refusal told late leaves it.
    /// The claim written for `target` keeps `at`. A caller that states no
    /// instant (`0`) is known by its worker alone. The call reads no date
    /// of the key's life and writes none: it reaches its caller's own
    /// claim alone.
    pub(crate) fn move_claim(
        &self,
        key: Session,
        claimed: WorkerId,
        target: WorkerId,
        at: u64,
    ) -> SessionVerdict {
        let key = key.get();
        let mut shard = self.shard(key).lock();
        let Some(entry) = shard.get_mut(&key) else {
            return SessionVerdict::Pinned;
        };
        // The ask comes ahead of the claim's own lines: where the key
        // takes a pending rename's step ([`Self::standing`]), the claim
        // is read on the walked entry. A claim the step took the mark of
        // is a pin the key was served on, and nothing moves. An empty
        // `target`, a pair that formed on none or a commit every target
        // refused, asks where both of the key's slots are taken: an
        // empty slot reads as its pin. A key that holds one worker twice
        // has both taken. A key with a slot free holds one pin at most,
        // and its withdrawal needs no step: it takes the claim's own pin
        // as written, or none where that pin is gone, and the mark goes.
        let stands = self.stands(entry, target);
        if !entry.reached_by(claimed, at) {
            return SessionVerdict::Pinned;
        }
        entry.withdraw_claim();
        if target == WorkerId(0) || entry.holds(stands) {
            return SessionVerdict::Pinned;
        }
        entry.claimed(stands, target, at);
        SessionVerdict::Claimed(target)
    }

    /// A decode call read `key` at `at`, its own instant, under the lease
    /// its pair's first decision issued at `issued`. Every word of the side
    /// that call issues is told under that lease. The key's life takes the
    /// lease's date as a reader's, so the call's dispatch is of this life
    /// ([`Entry::outlived`]). Where the call claimed `claimed`, the claim
    /// takes the lease's date too, unless the claim of record is no longer
    /// that call's.
    ///
    /// A life whose every reader is dated after `at` began behind the
    /// call's read. The call read no pin of it, and nothing is written:
    /// the call's dispatch is an older life's there.
    ///
    /// A lease that states no instant (`issued` is 0) writes nothing, and
    /// neither does a call with no key: no lock is taken for either.
    #[cfg(any(test, feature = "in-process"))]
    pub(crate) fn told_under(&self, key: Session, claimed: Option<WorkerId>, at: u64, issued: u64) {
        let key = key.get();
        if key == 0 || issued == 0 {
            return;
        }
        if let Some(entry) = self.shard(key).lock().get_mut(&key) {
            if entry.outlived(at) {
                return;
            }
            entry.read_under(issued);
            if let Some(leased) = claimed {
                entry.dated(leased, at, issued);
            }
        }
    }

    /// The commit dispatched a decision for `key` to `target` under the
    /// decision's `verdict`: the pin follows the rule. The preferred pin
    /// taken, or the failover taken while the preferred one was unroutable,
    /// leaves the pins as they are; a respill fronts the worker, and under no
    /// respill a worker outside the pins joins them by the key's recovery
    /// rule. A dispatch to the claimed worker confirms the claim first; the
    /// claiming decision's commit elsewhere withdraws it. A dispatch on an
    /// id its worker had before it was
    /// registered again is one on that worker, under the id that stands
    /// ([`Self::stands`]). A dispatch on a claim's lease asks by the
    /// claim of record.
    ///
    /// `at` is the instant the dispatched lease is issued under. The
    /// claiming decision is the one whose lease names the claim's id and
    /// that instant ([`Entry::reached_by`]): the commit elsewhere of an
    /// older decision that claimed the same worker withdraws nothing. A
    /// lease that states no instant (`0`) is known by its worker alone.
    ///
    /// A dispatch dated before every decision that read the key's current
    /// life writes nothing ([`Entry::outlived`]): the key started over
    /// between that dispatch's decision and its send. One that states no
    /// instant is written.
    pub(crate) fn dispatched(
        &self,
        key: Session,
        verdict: SessionVerdict,
        target: WorkerId,
        at: u64,
    ) {
        let key = key.get();
        if key == 0 || target == WorkerId(0) {
            return;
        }
        let mut shard = self.shard(key).lock();
        let Some(entry) = self.slot(&mut shard, key) else {
            return;
        };
        // A dispatch dated before every decision that read the key's
        // current life comes from an older one. The new life keeps its
        // own pins: a gateway's sticky map pins where it selects, and a
        // send that comes late writes nothing there.
        if entry.outlived(at) {
            return;
        }
        // A dispatch to the claimed worker confirms the claim, whichever
        // decision commits there: the key was served on it, so a pin another
        // request already used is never withdrawn. Only the claiming
        // decision's commit elsewhere withdraws a claim still unconfirmed; an
        // older request's commit on the owner leaves a retry's claim standing
        // for the retries after it.
        // A dispatch names the id its lease was issued on. In neither
        // pin, its worker may have been registered again since: the
        // table's renames say which id stands for it. A dispatch on a
        // claim's lease asks by the claim of record, the id the table
        // holds for that worker, so it asks too where a respill dropped
        // the claim's own pin: the key may hold the worker as a pin a
        // rename's walk has yet to reach.
        let named = entry.of_record(target);
        let target = self.stands(entry, named);
        entry.confirm_claim(target);
        if let SessionVerdict::Claimed(claimed) = verdict {
            if claimed != WorkerId(0) && entry.reached_by(claimed, at) {
                entry.withdraw_claim();
            }
        }
        entry.served(target, verdict);
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
    /// respill's move and the commit. A moved claim, a respill's move and
    /// the commit ask ahead of every line of their own, and a claim asks
    /// ahead of its read of the pins against the ones its decision read
    /// ([`Self::claim`]). The pins are read first: a worker among them
    /// costs no read of the renames' count. A write whose target is a pin
    /// as written asks nothing here, whatever a rename's walk has yet to
    /// reach: it adds no pin. It puts the pins in another order or
    /// withdraws its own decision's claim, so the walk, or the key's next
    /// write that asks, finds what it would have found ahead of it. One
    /// caller asks past this test: a claim on pins that moved under its
    /// decision takes the step whatever its target, a pin as written
    /// too, because its answer names a pin ([`Self::claim`]).
    #[inline]
    fn stands(&self, entry: &mut Entry, target: WorkerId) -> WorkerId {
        if !entry.holds(target) && self.renamed_len.load(Ordering::Relaxed) != 0 {
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
    /// a worker ([`Self::stands`]), and a claim where the pins moved
    /// under its decision and the table renamed a worker
    /// ([`Self::claim`]).
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
            self.retain(shard, |entry| entry.unpin(workers));
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
                stays && entry.unpin(&ended)
            });
        }
        drop(held);
    }

    /// The pins of `key` as a decision at `now_micros` finds them, with
    /// nothing written: a key the table never read, or one past its end,
    /// has none, as at [`Self::lookup`]. The key's idle time is not
    /// renewed and no sweep runs: a decision that goes on to follow one of
    /// these pins reads them by [`Self::pins_renewed`].
    pub(crate) fn pins_at(&self, key: Session, now_micros: u64) -> SessionPins {
        let key = key.get();
        match self.shard(key).lock().get(&key) {
            Some(entry) if entry.expires_at_micros > now_micros => entry.pins,
            _ => [WorkerId(0); 2],
        }
    }

    /// The pins of `key` for a decision at `now_micros` that follows one of
    /// them, under one lock: a key that holds a pin and is not past its end
    /// is touched as at [`Self::lookup`]. Of any other key nothing is
    /// written, as at [`Self::pins_at`]: it reads as no pins and no entry is
    /// made for it. No sweep runs here: the decision that goes on to follow
    /// one of these pins runs the one this read owes ([`Self::sweep_owed`]).
    pub(crate) fn pins_renewed(
        &self,
        key: Session,
        now_micros: u64,
        rule: &SessionRule,
    ) -> SessionPins {
        let key = key.get();
        match self.shard(key).lock().get_mut(&key) {
            Some(entry)
                if entry.expires_at_micros > now_micros && entry.pins != [WorkerId(0); 2] =>
            {
                entry.touch(now_micros, rule);
                entry.pins
            }
            _ => [WorkerId(0); 2],
        }
    }

    /// The sweep a renewing read owes where no pass of the table's host
    /// sweeps it ([`Self::lookup`]), for a decision that follows a pin
    /// [`Self::pins_renewed`] read: it runs behind that read, with no
    /// shard's lock held, and a request that follows no pin runs none.
    /// Always inlined: the check is one compare on a followed pin's path.
    #[inline(always)]
    pub(crate) fn sweep_owed(&self, now_micros: u64) {
        if !self.swept_by_pass.load(Ordering::Relaxed) {
            self.sweep_if_due(now_micros);
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
#[path = "../../tests/unit/engine/sticky_sessions/tests.rs"]
mod tests;
