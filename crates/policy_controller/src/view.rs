//! The bounded view: the per-decision, handle-free picture of one request and
//! its candidates (design 4.1). Built once per decision by the host into a
//! reused buffer; every row is `Copy`, and nothing here reaches back into the
//! host. The names are the wire's: a candidate on the wire is this row with
//! its affinity facts and without its position and sample instant, a request
//! this request without its host clock and its affinity column, and positions
//! into the spec's declared lists (zones, engine kinds, connectors, labels,
//! headers) are the same positions here.

/// The stable identity of a candidate: a hash of the registration key and the
/// incarnation, never a URL. A same-URL replacement is a new id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub u64);

/// A worker's role in the fleet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Role {
    #[default]
    Single,
    Prefill,
    Decode,
    Encode,
}

/// Which phase of a request a decision is for.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    #[default]
    Single,
    Prefill,
    Decode,
    Encode,
}

/// What the request's block hashes were computed over.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HashKind {
    /// Token blocks of the model's block size of record.
    #[default]
    Token,
    /// Fixed character windows over the text, for a host without a tokenizer.
    Text,
}

/// Where a candidate's load row came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SignalSource {
    #[default]
    None,
    /// The load records riding the worker's event stream.
    Events,
    /// The scrape of an engine without a servicer.
    Scrape,
    /// A gateway relaying a host-local worker's reports.
    Relay,
    /// The poll a servicer falls back to when its stream stops.
    Poll,
}

/// The pool's overload regime, as the host reads it per decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OverloadRegime {
    #[default]
    None,
    Partial,
    Saturated,
}

/// The wires a worker serves, or the wires a deciding gateway speaks, as
/// bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WireModes(u8);

impl WireModes {
    /// No wire.
    pub const NONE: Self = Self(0);
    pub const HTTP: Self = Self(1);
    pub const GRPC: Self = Self(2);
    pub const ZMQ: Self = Self(4);

    /// The bits as the wire carries them.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// The set of the three known bits in `bits`; unknown bits are dropped.
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & 0b111)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Every wire of `other` is in `self`.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Some wire is in both: the compatibility read of a worker against the
    /// gateway deciding for it.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Health as the pool reads it. Each flag is one bit of the host's routing
/// state, read once per decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Health {
    /// The worker's status is ready.
    pub healthy: bool,
    /// The pool's circuit breaker refuses execution.
    pub breaker_open: bool,
    /// The liveness veto: unreachable or wedged.
    pub stalled: bool,
    /// Being drained; nothing new goes there.
    pub draining: bool,
    /// The latched overload verdict of the last load report.
    pub overloaded: bool,
}

impl Health {
    /// Every routing veto but the overload one.
    pub const fn routable_ignoring_overload(self) -> bool {
        self.healthy && !self.breaker_open && !self.stalled && !self.draining
    }

    /// Every routing veto.
    pub const fn routable(self) -> bool {
        self.routable_ignoring_overload() && !self.overloaded
    }
}

/// One load row as the view carries it: the engine load record's routing
/// core, in the record's own names, rounded to narrow fields by the host.
/// Permille fields are the record's ratios times 1000.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadRow {
    /// Requests running on the engine.
    pub running: u32,
    /// Requests queued on the engine.
    pub queued: u32,
    /// Token-work of the queue, when the engine reports it (`0` when it
    /// reports a depth alone).
    pub waiting_uncached_tokens: u32,
    /// KV-cache usage, in permille.
    pub kv_usage_permille: u16,
    /// The share of the KV cache the running requests hold, in permille.
    pub active_kv_permille: u16,
    /// Tokens per second the worker drains; `0` when unknown.
    pub live_rate: u32,
    /// Context tokens resident for the running requests; `0` too when the
    /// engine does not count them.
    pub resident_tokens: u32,
    /// Whether the report counts resident tokens: an engine that does not
    /// count them sends none, which is not a count of zero. The host's own
    /// reading of the report, in no plugin's view.
    pub resident_counted: bool,
    /// Age of the report's sample at the decision, in milliseconds.
    pub report_age_ms: u32,
    /// Where the row came from.
    pub source: SignalSource,
    /// The engine's admission window (maximum running requests); `0` when
    /// unknown.
    pub max_running: u32,
}

impl LoadRow {
    /// KV-cache usage as a ratio, `0.0` to `1.0`.
    pub fn kv_usage(&self) -> f32 {
        f32::from(self.kv_usage_permille) / 1000.0
    }

    /// The running requests' share of the KV cache as a ratio, `0.0` to
    /// `1.0`: the occupancy a request waits on, the reclaimable prefix
    /// cache left out (a report without the split carries its usage here).
    pub fn active_kv(&self) -> f32 {
        f32::from(self.active_kv_permille) / 1000.0
    }
}

/// A tenant or a session key as the view carries it: its fingerprint, and the
/// plain form while the host has it. Native stages read the fingerprint; a
/// plugin sees one form or the other by its exposure. [`Identity::NONE`] is
/// the absent key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Identity<'a> {
    /// The fingerprint of the key; `0` when absent.
    pub hash: u64,
    /// The key as the host received it, when it keeps it beside the hash.
    pub plain: Option<&'a str>,
}

impl<'a> Identity<'a> {
    /// The absent key.
    pub const NONE: Self = Self {
        hash: 0,
        plain: None,
    };

    /// A key the host holds in the clear, fingerprinted here.
    pub fn plain(key: &'a str) -> Self {
        Self {
            hash: fingerprint(key.as_bytes()),
            plain: Some(key),
        }
    }

    /// A key the host holds as its fingerprint alone.
    pub const fn hashed(hash: u64) -> Self {
        Self { hash, plain: None }
    }

    pub const fn is_none(&self) -> bool {
        self.hash == 0
    }
}

/// The fingerprint of a key: FNV-1a over its bytes, never `0` for a present
/// key, so `0` reads as absent everywhere a fingerprint travels (the lease,
/// the trace, a session table).
pub fn fingerprint(key: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in key {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if h == 0 {
        1
    } else {
        h
    }
}

/// The fingerprint of a model's name, the model's part of a session's value.
/// A host that keeps its models makes it once a model and carries it in the
/// request's facts, so a keyed decision does not read the name's bytes
/// again. Only [`ModelFingerprint::of`] makes one: a number of the host's
/// own is no model's fingerprint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelFingerprint(u64);

impl ModelFingerprint {
    /// Not carried: a read of the request's session fingerprints the name.
    pub const NONE: Self = Self(0);

    /// The fingerprint of `model`, the name as a request carries it.
    pub fn of(model: &str) -> Self {
        Self(fingerprint(model.as_bytes()))
    }
}

/// A session: a key within its model and phase, as the one value a lease
/// carries and the session table is keyed by. Only
/// [`RequestFacts::session`] makes one, so a lease or a pin cannot carry a
/// key's own fingerprint in its place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Session(u64);

impl Session {
    /// No session: the request carried no key. `0` is no session's and no
    /// key's: a present key never makes it ([`RequestFacts::session`]).
    pub const NONE: Self = Self(0);

    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    /// The value as the wire carries it; `0` for no session.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// A session from its value, for the session table's own cases.
    #[cfg(test)]
    pub(crate) const fn from_raw(value: u64) -> Self {
        Self(value)
    }
}

/// What the affinity stages read of one candidate: a column beside the rows,
/// by the candidate's index ([`RequestFacts::affinity`]). A host fills it for
/// a profile whose stages declare it
/// ([`Needs::affinity`](crate::Needs::affinity)) and for no other, so a
/// decision that reads none of it copies none of it. `Default` is nothing
/// held by a worker just admitted, which is what every candidate reads as
/// when the column is empty. On a wire the five are fields of the
/// candidate's own record, at the numbers they were given.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AffinityFacts {
    /// Prefix blocks of the request this worker holds, by the event index.
    pub overlap_blocks: u32,
    /// The same by the sent-prefix predictor.
    pub predicted_blocks: u32,
    /// Milliseconds since the worker was admitted to the pool, saturating at
    /// `u32::MAX` as `report_age_ms` does; never wrapping.
    pub admitted_age_ms: u32,
    /// Blocks the index learned about the worker since admission.
    pub indexed_growth: u32,
    /// Blocks the index holds for the worker: what a warm-up stage reads
    /// thinness from, against the fleet's level.
    pub indexed_blocks: u32,
}

/// What a host that names rows feeds a decision in place of the affinity
/// column: the rows that hold a leading part of the request and the pool's
/// warming rows, as indices into the candidate slice the decision is made
/// over. A host that derives them where it writes hands these and no
/// column, so nothing of the pool's length is filled for a decision. Both
/// lists are the host's to keep ascending by row and under the candidates'
/// count: a debug build asserts it where a decision takes its inputs, and a
/// release build reads them as they stand. In-process only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NamedAffinity<'a> {
    /// The rows that hold a leading part of the request, ascending by row,
    /// each with what it holds. A row not named reads as holding nothing.
    pub holders: &'a [NamedHolder],
    /// The pool's warming rows, ascending, by the warm-up stage's rule as
    /// the host applied it: the thin whatever the fleet's age, else the
    /// young unless every row is young.
    pub warming: &'a [u32],
    /// The warming rows are the thin.
    pub thin: bool,
}

/// One row a host names as a holder: what the row's entry of a column says
/// of what it holds, by both readings. Its age and its index size stay with
/// the host, which ruled the warming rows from them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NamedHolder {
    /// The row's index in the candidate slice.
    pub row: u32,
    /// As [`AffinityFacts::overlap_blocks`].
    pub overlap_blocks: u32,
    /// As [`AffinityFacts::predicted_blocks`].
    pub predicted_blocks: u32,
}

impl<'a> NamedAffinity<'a> {
    /// Row `index`'s entry: what the host said the row holds when it is
    /// named, nothing held otherwise.
    #[inline]
    pub(crate) fn entry(&self, index: u32) -> AffinityFacts {
        let Ok(at) = self
            .holders
            .binary_search_by_key(&index, |holder| holder.row)
        else {
            return AffinityFacts::default();
        };
        self.holders[at].facts()
    }

    /// One pass's reads of the rows, walked beside the rows the pass asks.
    #[inline]
    pub(crate) fn walk(self) -> NamedWalk<'a> {
        NamedWalk(Walk::new(self.holders))
    }
}

impl NamedHolder {
    /// The row's entry of a column: what it holds by both readings, and
    /// nothing said of its age or its index.
    #[inline]
    fn facts(self) -> AffinityFacts {
        AffinityFacts {
            overlap_blocks: self.overlap_blocks,
            predicted_blocks: self.predicted_blocks,
            ..AffinityFacts::default()
        }
    }
}

/// A forward cursor over a list its host keeps ascending by row, for one
/// pass that asks its rows in order: a read is a compare, and each entry is
/// passed once. A row before the cursor starts it over by a search, so rows
/// asked in any order read as a search reads them.
#[derive(Debug)]
pub(crate) struct Walk<'a, T> {
    list: &'a [T],
    at: usize,
    last: u32,
}

impl<'a, T: Copy> Walk<'a, T> {
    pub(crate) fn new(list: &'a [T]) -> Self {
        Self {
            list,
            at: 0,
            last: 0,
        }
    }

    /// The entry of row `index`, where the list names it; `row` reads an
    /// entry's row.
    #[inline]
    pub(crate) fn find(&mut self, index: u32, row: impl Fn(&T) -> u32) -> Option<T> {
        // The cursor stands at the first entry at or past the last row
        // asked, so a row before that one is looked for again.
        if index < self.last {
            self.at = self.behind(index, &row);
        }
        self.last = index;
        while let Some(entry) = self.list.get(self.at) {
            let at = row(entry);
            if at >= index {
                return (at == index).then_some(*entry);
            }
            self.at += 1;
        }
        None
    }

    /// Where the entries at or past row `index` start: the search a row
    /// behind the cursor costs, out of the walk's own line.
    #[cold]
    #[inline(never)]
    fn behind(&self, index: u32, row: &impl Fn(&T) -> u32) -> usize {
        self.list.partition_point(|entry| row(entry) < index)
    }
}

/// One pass's reads of the rows a host named, in the pass's own order.
#[derive(Debug)]
pub(crate) struct NamedWalk<'a>(Walk<'a, NamedHolder>);

impl NamedWalk<'_> {
    /// Row `index`'s entry, as [`NamedAffinity::entry`] reads it.
    #[inline]
    pub(crate) fn entry(&mut self, index: u32) -> AffinityFacts {
        self.0
            .find(index, |holder| holder.row)
            .map_or_else(AffinityFacts::default, NamedHolder::facts)
    }
}

/// What a pass reads of what the candidates hold, taken once
/// ([`RequestFacts::feed`]): the host's column, or the rows it named in the
/// column's place.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Feed<'a> {
    Column(&'a [AffinityFacts]),
    Named(NamedAffinity<'a>),
}

impl Feed<'_> {
    /// Nobody holds anything: an empty column, or no holder named.
    pub(crate) fn is_empty(self) -> bool {
        match self {
            Self::Column(column) => column.is_empty(),
            Self::Named(named) => named.holders.is_empty(),
        }
    }

    /// Row `index`'s entry: a column's own, where an index past the end
    /// fails, or a named row's ([`NamedAffinity::entry`]).
    #[inline]
    pub(crate) fn entry(self, index: u32) -> AffinityFacts {
        match self {
            Self::Column(column) => column[index as usize],
            Self::Named(named) => named.entry(index),
        }
    }
}

impl<'a> Feed<'a> {
    /// One pass's reads of the feed: the column by index, or a walk of the
    /// rows named. Taken where a pass starts, so no two passes share a
    /// cursor.
    pub(crate) fn reads(self) -> Reads<'a> {
        match self {
            Self::Column(column) => Reads::Column(column),
            Self::Named(named) => Reads::Named(named.walk()),
        }
    }
}

/// What one pass reads of what the candidates hold: see [`Feed::reads`].
#[derive(Debug)]
pub(crate) enum Reads<'a> {
    Column(&'a [AffinityFacts]),
    Named(NamedWalk<'a>),
}

impl Reads<'_> {
    /// Row `index`'s entry, as [`Feed::entry`] reads it.
    #[inline]
    pub(crate) fn entry(&mut self, index: u32) -> AffinityFacts {
        match self {
            Self::Column(column) => column[index as usize],
            Self::Named(walk) => walk.entry(index),
        }
    }
}

/// One candidate row: a worker, or one DP rank of a data-parallel worker:
/// what every decision reads. Its fields take 103 bytes, and the row is
/// aligned to a 64-byte cache line: 128 bytes in memory, every row on a
/// line's start.
///
/// The alignment is measured. The filters walk the rows once a filter, and
/// where the pull-request bench gate runs, such a pass over 512 rows read 5
/// to 11 percent slower at a stride of 96, 104 or 112 bytes than at 128,
/// wherever the flags sat in the row. A build host read the narrower
/// strides faster, so a change of the row's size or alignment carries the
/// decision bench (`benches/decision.rs`) on both.
///
/// The wire form drops the position and the sample instant, carries the
/// candidate's [`AffinityFacts`] in the same record and encodes to about 76
/// bytes with its load row.
#[derive(Clone, Copy, Debug, Default)]
#[repr(align(64))]
pub struct Candidate {
    /// The row's position in the host's candidate slice; targets name it.
    /// In-process only: the wire names rows by their order. It is also the
    /// row's index in the slice the decision is made over (its own side's
    /// pool on a pair): the plugin's filter adapter asks for its row by it
    /// and the pair plan looks a chosen prefill up by it. A filter's `keep`
    /// takes one row and no index, so a filter that read the affinity
    /// column there would have to read it at the position; no filter of
    /// this crate does. A debug build asserts that the two agree where a
    /// decision takes its inputs; a release build does not check.
    pub position: u32,
    pub id: WorkerId,
    pub role: Role,
    pub dp_rank: u16,
    pub dp_size: u16,
    pub health: Health,
    /// The wires the worker serves.
    pub wire_modes: WireModes,
    /// The engine family, as a position into the spec's declared list; `0`
    /// when unspecified.
    pub engine_kind: u16,
    /// The pairing descriptor's connector, as a position into the spec's
    /// declared list; `0` when none.
    pub connector: u16,
    /// Requests the deciding host holds on the worker right now.
    pub in_flight: u32,
    /// The latest load row, when the host has a report for the worker.
    pub load: Option<LoadRow>,
    /// The sample instant of `load` on the host's clock, in microseconds since
    /// the host's origin: dispatches made at or before it are the engine's by
    /// now and leave a load scorer's credit. In-process only; the wire carries
    /// the row's age.
    pub load_sampled_at_micros: u64,
    /// The session key of the request is pinned to this worker.
    pub session_hit: bool,
    /// Requests of the request's session key in flight on this worker.
    pub session_in_flight: u16,
    /// The pairing class: `0` for a worker outside any pairing. Which decode
    /// classes a prefill class pairs with is the pair table's to say
    /// ([`PairTable`](crate::PairTable)); with no table a decode pairs with a
    /// prefill of the same class. Until a store derives classes, a host keys
    /// it by the open prefill a decode partners with.
    pub pair_class: u32,
    /// Host-supplied admission bit: the prefill capacity window has a slot.
    pub has_capacity: bool,
    /// Host-supplied admission bit: the decode's running window has a room
    /// nobody waiting at its gate is in line for.
    pub has_room: bool,
    /// The zone, as a position into the spec's declared list; `0` when
    /// unzoned.
    pub zone: u16,
    /// The worker's labels, as bits over the spec's declared list (at most
    /// 64).
    pub labels: u64,
    /// Encode-side compatibility for an encode-prefill-decode dispatch.
    pub ec_compatible: bool,
    /// The worker takes images.
    pub vision: bool,
    /// Reachable from one gateway only (a host-local wire).
    pub host_local: bool,
    /// Session keys with a request in flight on this worker. Last because the
    /// wire numbers its fields in the record's order and this one came after
    /// the contract's first cut.
    pub active_sessions: u16,
    /// The worker's stable identity for placement: its registration key's
    /// fingerprint (the low 32 bits), equal across the handles of one worker
    /// and across gateway replicas, where the id is the handle's. A ring
    /// positions a member by it, so a replaced handle keeps its
    /// predecessor's positions and every replica agrees on a key's owner.
    /// Two members with equal keys share positions and the lower member
    /// index owns first: by the birthday bound, n(n - 1) / 2 over 2^32, about
    /// one chance in 8,600 over a pool of a thousand and one in 870,000 over
    /// a hundred; the 64-bit form waits for the candidate's byte budget.
    /// Zero when the host carries none. After the contract's first cut (the
    /// wire numbers its fields in the record's order).
    pub key: u32,
}

/// Media on the request: counts and bytes, for workers that take media.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MediaCounts {
    pub images: u16,
    pub audio: u16,
    pub video: u16,
    pub bytes: u32,
}

/// An allow-listed header: its name as a position into the spec's declared
/// header list, and its value. A value never pairs with a name the spec did
/// not allow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Header<'a> {
    pub name: u8,
    pub value: &'a str,
}

/// One request as a decision sees it. Borrowed from the host for the
/// duration of the decision.
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestFacts<'a> {
    /// The resolved model id, after aliasing.
    pub model: &'a str,
    /// The cache namespace folded into a seed (salt, adapter, extra key).
    pub namespace_seed: u64,
    pub tenant: Identity<'a>,
    /// Priority as a signed band; `0` without a scheduler; negative is
    /// sheddable.
    pub priority: i32,
    pub streaming: bool,
    pub max_output_tokens: u32,
    /// The host's token count or estimate.
    pub prompt_tokens: u32,
    /// Prefix block hashes, capped by the host.
    pub block_hashes: &'a [u64],
    pub hash_kind: HashKind,
    /// The routing key as the host derives it, lineage-stripped: the key
    /// alone, in the form the host holds. The session it names within the
    /// request's model and phase is [`RequestFacts::session`].
    pub session_key: Identity<'a>,
    pub phase: Phase,
    /// The retry number, `0` for the first attempt.
    pub attempt: u8,
    /// Candidates earlier attempts of this request already tried.
    pub tried: &'a [WorkerId],
    /// Time left for the whole decision, relative to the host's receipt; a
    /// plugin's own share of it rides its frame, clamped to at most this.
    pub budget_micros: u32,
    /// Allow-listed headers only.
    pub headers: &'a [Header<'a>],
    pub media: MediaCounts,
    /// The per-decision seed, recorded in the trace: the only randomness a
    /// stage gets.
    pub seed: u64,
    /// The size in tokens of one of the request's blocks as the host
    /// counted them: an index's learned block size on an event-fed request,
    /// the configured one on a tree-fed request; `0` reads the view's. Last
    /// on the wire (the numbering follows the record's order).
    pub block_size: u32,
    /// The host's clock at the decision, in microseconds since its origin.
    /// In-process only; no absolute instant crosses the wire.
    pub now_micros: u64,
    /// The candidates' affinity facts, by index in the candidate slice the
    /// decision is made over: the one column the affinity stages read, in
    /// the candidates' own order. It is EMPTY or holds exactly one entry a
    /// candidate; the engine refuses any other before a stage runs. An
    /// empty column is the host's word that it holds nothing, unless the
    /// host names rows in its place ([`RequestFacts::named`]): it reads as
    /// nothing held for every candidate, under a profile that declared
    /// [`Needs::affinity`](crate::Needs::affinity) as under one that did
    /// not, and the engine neither refuses it nor counts it. A host that
    /// owes the column and hands none is routed by its other stages alone,
    /// with no signal. A pair decision takes each side's column from
    /// its pool ([`PairPools`](crate::PairPools)) and none from here: a
    /// non-empty column on the request of a pair decision is refused, as a
    /// column of another length is. The host keeps it readable, as it keeps
    /// the candidates, until the decision is committed. In-process only: a
    /// wire carries each candidate's facts in its record.
    pub affinity: &'a [AffinityFacts],
    /// The rows the host names in that column's place ([`NamedAffinity`]):
    /// beside an empty column, the affinity stages read them. A pair
    /// decision takes none: its sides read their pools' columns. In-process
    /// only.
    pub named: Option<NamedAffinity<'a>>,
    /// The fingerprint of `model` when the host holds it, as a host that
    /// keeps its models makes it once a model; [`ModelFingerprint::NONE`]
    /// when it does not, and each read of [`RequestFacts::session`] then
    /// fingerprints the name. A debug build holds a carried fingerprint to
    /// the name. A release build trusts it: a request that carries another
    /// name's fingerprint is that model's session for its key, and reads
    /// and moves that model's pins. The trust ends when the facts carry
    /// the model as one value, its name with its fingerprint: the change
    /// that hands a decision its facts from the host's own store makes
    /// them so. In-process only: the wire carries the name.
    pub model_fingerprint: ModelFingerprint,
}

/// The multiplier that spreads a session's model and phase over its key.
const SESSION_MIX: u64 = 0x9e37_79b9_7f4a_7c15;

/// A phase's number in a session's value: pinned here, never the enum's
/// order, which a later phase may change.
const fn session_phase(phase: Phase) -> u64 {
    match phase {
        Phase::Single => 1,
        Phase::Prefill => 2,
        Phase::Decode => 3,
        Phase::Encode => 4,
    }
}

impl<'a> RequestFacts<'a> {
    /// The rows the host named, where they answer: beside an empty column.
    /// The column is asked first, so a host that fills one pays nothing for
    /// the rows it does not name.
    #[inline(always)]
    pub(crate) fn named_alone(&self) -> Option<NamedAffinity<'a>> {
        if self.affinity.is_empty() {
            self.named
        } else {
            None
        }
    }

    /// What the candidates hold as the host fed this decision: the rows it
    /// named where they answer, else its column. A pass takes it once.
    pub(crate) fn feed(&self) -> Feed<'a> {
        match self.named_alone() {
            Some(named) => Feed::Named(named),
            None => Feed::Column(self.affinity),
        }
    }

    /// The affinity facts of the candidate at `index` of the slice the
    /// decision is made over. An empty column reads as nothing held for
    /// every candidate. Any other column holds the candidate's own entry,
    /// and the read is that entry: an index past its end is a host's
    /// mistake and fails as an index past the candidates does, never as a
    /// default in the entry's place. Where the host named rows in the
    /// column's place ([`NamedAffinity`]), a named row reads what the host
    /// said it holds and every other row nothing held. Each read tests the
    /// feed and its bound: a pass over a pool takes the column once
    /// ([`RequestFacts::affinity`]) and walks it beside the rows, as this
    /// crate's stages and its packed plugin fill do.
    pub fn affinity_at(&self, index: u32) -> AffinityFacts {
        let feed = self.feed();
        if feed.is_empty() {
            AffinityFacts::default()
        } else {
            feed.entry(index)
        }
    }

    /// The session the request belongs to: its key within its model and
    /// phase, as one value; [`Session::NONE`] when it carries no key. One key names one
    /// session per model, and each side of a pair keeps its own. The value
    /// is a wire fact: a lease carries it and a session table is keyed by
    /// it on every replica, so this formula alone makes it, from the key's
    /// fingerprint, the fingerprint of the model's name as the request
    /// carries it and the phase's pinned number. Two hosts that hold one
    /// key in different forms (the plain key, its fingerprint) get one
    /// value, and so do a host that carries the model's fingerprint and
    /// one that leaves it to this read. The value is never `0` for a
    /// present key: a result of `0` is carried as `1`, a term of the
    /// formula as the plugin protocol states it.
    pub fn session(&self) -> Session {
        if self.session_key.is_none() {
            return Session::NONE;
        }
        let model = if self.model_fingerprint == ModelFingerprint::NONE {
            fingerprint(self.model.as_bytes())
        } else {
            debug_assert_eq!(
                self.model_fingerprint,
                ModelFingerprint::of(self.model),
                "the carried fingerprint is not the model name's"
            );
            self.model_fingerprint.0
        };
        let scope = model ^ session_phase(self.phase).wrapping_mul(SESSION_MIX);
        let mut h = self.session_key.hash ^ scope.wrapping_mul(SESSION_MIX);
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        Session(if h == 0 { 1 } else { h })
    }
}

/// What only the deciding host knows now: its standing vetoes, the wires it
/// speaks and the request-level lifts around the profile. The admission bits
/// of a prefill window and a decode room ride the candidate rows.
#[derive(Clone, Copy, Debug, Default)]
pub struct Constraints<'a> {
    /// Candidates this host just failed to reach.
    pub excluded: &'a [WorkerId],
    /// The wires this host speaks.
    pub wire_modes: WireModes,
    /// The overload veto does not apply to this decision (a pool whose
    /// policy balances load against affinity itself, a session owner).
    pub overload_lifted: bool,
    /// A subset hint: decide over at most this many candidates.
    pub subset_hint: Option<u32>,
}

/// Healthy capacity of one zone, by the zone's position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZoneCapacity {
    pub zone: u16,
    pub healthy: u32,
    pub total: u32,
}

/// Pool-wide facts beside the candidates.
#[derive(Clone, Copy, Debug, Default)]
pub struct PoolFacts<'a> {
    /// Pool size before any narrowing.
    pub pool_size: u32,
    pub membership_epoch: u64,
    /// The epoch of the pool's hash ring; `0` without one.
    pub ring_epoch: u64,
    pub regime: OverloadRegime,
    /// The spec generation the decision runs under.
    pub generation: u64,
    /// Healthy capacity per zone; empty for an unzoned fleet.
    pub zones: &'a [ZoneCapacity],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_flags_compose_into_the_two_routability_reads() {
        let ready = Health {
            healthy: true,
            ..Health::default()
        };
        assert!(ready.routable());
        let overloaded = Health {
            overloaded: true,
            ..ready
        };
        assert!(!overloaded.routable());
        assert!(overloaded.routable_ignoring_overload());
        let open = Health {
            breaker_open: true,
            ..ready
        };
        assert!(!open.routable_ignoring_overload());
    }

    #[test]
    fn a_candidate_row_is_two_lines_and_its_affinity_facts_20_bytes() {
        // The filters walk the rows once a filter, so the rows' stride is a
        // cost every policy pays: the pin is exact, 128 bytes at a 64-byte
        // alignment, and a change of any number here carries the crate's
        // decision bench (`benches/decision.rs`) where the pull-request gate
        // runs as well as on a build host: the two read a narrower stride in
        // opposite directions.
        assert_eq!(size_of::<Candidate>(), 128);
        assert_eq!(align_of::<Candidate>(), 64);
        assert_eq!(size_of::<AffinityFacts>(), 20);
    }

    #[test]
    fn a_candidate_rows_fields_take_103_bytes() {
        // The row is written once per candidate per decision, so what its
        // fields take is a cost every policy pays. The alignment leaves
        // spare bytes, so the row's size no longer says it, and a field
        // added to the row would move no size. This case names every field
        // (a new one does not compile until it is named here) and pins
        // where the last one ends. A fact one stage family reads belongs in
        // a column beside the rows, as the affinity facts are. The plugin
        // view's packed row is 104 bytes WITH the five facts in it: nothing
        // ties that number to this one, and each is pinned in its own file.
        fn end<T>(at: usize, _field: &T) -> usize {
            at + size_of::<T>()
        }
        macro_rules! ends {
            ($($field:ident),* $(,)?) => {{
                let Candidate { $($field),* } = Candidate::default();
                [$(end(std::mem::offset_of!(Candidate, $field), &$field)),*]
            }};
        }
        let ends = ends!(
            position,
            id,
            role,
            dp_rank,
            dp_size,
            health,
            wire_modes,
            engine_kind,
            connector,
            in_flight,
            load,
            load_sampled_at_micros,
            session_hit,
            session_in_flight,
            pair_class,
            has_capacity,
            has_room,
            zone,
            labels,
            ec_compatible,
            vision,
            host_local,
            active_sessions,
            key,
        );
        assert_eq!(ends.into_iter().max(), Some(103));
    }

    #[test]
    fn an_empty_affinity_column_reads_as_nothing_held_for_every_candidate() {
        let empty = RequestFacts::default();
        for index in [0, 1, 511] {
            assert_eq!(empty.affinity_at(index), AffinityFacts::default());
        }
    }

    #[test]
    fn a_filled_affinity_column_reads_the_entry_at_the_candidates_index() {
        let column = [3, 5, 8].map(|overlap_blocks| AffinityFacts {
            overlap_blocks,
            ..AffinityFacts::default()
        });
        let facts = RequestFacts {
            affinity: &column,
            ..RequestFacts::default()
        };
        let read = [0, 1, 2].map(|index| facts.affinity_at(index).overlap_blocks);
        assert_eq!(read, [3, 5, 8]);
    }

    /// A column that holds an entry is never read as a default past its
    /// end: zero is a real answer (a cold worker), so a forgiving read would
    /// turn a column filled for another pool into a routing change.
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn a_read_past_a_filled_affinity_column_is_not_a_default() {
        let column = [AffinityFacts::default()];
        let facts = RequestFacts {
            affinity: &column,
            ..RequestFacts::default()
        };
        let _ = facts.affinity_at(1);
    }

    /// A pass's walk of the rows named reads each row as the search does, in
    /// whatever order the pass asks: ascending, a row twice, a row behind
    /// the cursor.
    #[test]
    fn a_walk_of_the_rows_named_reads_each_row_as_the_search_does() {
        let holders = [(1, 4, 0), (4, 9, 2), (5, 1, 7), (9, 3, 0)].map(
            |(row, overlap_blocks, predicted_blocks)| NamedHolder {
                row,
                overlap_blocks,
                predicted_blocks,
            },
        );
        let named = NamedAffinity {
            holders: &holders,
            ..NamedAffinity::default()
        };
        let orders: [&[u32]; 4] = [
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[1, 1, 4, 4, 9, 9],
            &[10, 9, 5, 4, 1, 0],
            &[5, 0, 9, 4, 4, 1, 7, 10, 5],
        ];
        for order in orders {
            let mut walk = named.walk();
            for &row in order {
                assert_eq!(walk.entry(row), named.entry(row), "row {row} of {order:?}");
            }
        }
        // No row named: nothing held at any row.
        let mut walk = NamedAffinity::default().walk();
        assert_eq!(walk.entry(0), AffinityFacts::default());
        assert_eq!(walk.entry(3), AffinityFacts::default());
    }

    /// Rows named in the column's place read through the same accessor: a
    /// named row reads what the host said it holds, by both readings, and
    /// every other row nothing held. Beside a column the column is read.
    #[test]
    fn rows_named_in_the_columns_place_read_what_each_holds() {
        let holders = [
            NamedHolder {
                row: 1,
                overlap_blocks: 9,
                predicted_blocks: 0,
            },
            NamedHolder {
                row: 4,
                overlap_blocks: 3,
                predicted_blocks: 7,
            },
        ];
        let facts = RequestFacts {
            named: Some(NamedAffinity {
                holders: &holders,
                ..NamedAffinity::default()
            }),
            ..RequestFacts::default()
        };
        let read = [0, 1, 2, 4, 511].map(|index| {
            let held = facts.affinity_at(index);
            (held.overlap_blocks, held.predicted_blocks)
        });
        assert_eq!(read, [(0, 0), (9, 0), (0, 0), (3, 7), (0, 0)]);
        let nobody = RequestFacts {
            named: Some(NamedAffinity::default()),
            ..RequestFacts::default()
        };
        assert_eq!(nobody.affinity_at(3), AffinityFacts::default());
        assert_eq!(
            RequestFacts::default().affinity_at(1),
            AffinityFacts::default()
        );
        // Beside a column the column is read.
        let column = [
            AffinityFacts::default(),
            AffinityFacts {
                overlap_blocks: 3,
                ..AffinityFacts::default()
            },
        ];
        let beside = RequestFacts {
            affinity: &column,
            ..facts
        };
        assert_eq!(beside.affinity_at(1).overlap_blocks, 3);
    }

    #[test]
    fn the_load_row_holds_its_resident_count_flag_in_its_padding() {
        // 34 bytes of fields in a row aligned to four: the flag adds no
        // byte, and an optional row needs no tag beside it.
        assert_eq!(size_of::<LoadRow>(), 36);
        assert_eq!(size_of::<Option<LoadRow>>(), 36);
    }

    #[test]
    fn an_identity_fingerprint_is_stable_and_never_zero() {
        let a = Identity::plain("session-1");
        let b = Identity::plain("session-1");
        assert_eq!(a.hash, b.hash);
        assert_ne!(a.hash, Identity::plain("session-2").hash);
        assert!(!a.is_none());
        assert!(Identity::NONE.is_none());
        assert_eq!(Identity::hashed(a.hash).hash, a.hash);
        // The empty key is present too: a fingerprint is never the absent
        // marker.
        assert_ne!(fingerprint(b""), 0);
    }

    /// The session's value is a wire fact: these literals stop an edit of
    /// the formula that keeps every relative case green.
    #[test]
    fn a_sessions_value_is_pinned_for_a_fixed_key_model_and_each_phase() {
        let of = |phase| {
            RequestFacts {
                model: "model-a",
                session_key: Identity::plain("conversation-7"),
                phase,
                ..RequestFacts::default()
            }
            .session()
            .get()
        };
        assert_eq!(fingerprint(b"conversation-7"), 0xd449_9ecc_c740_2a12);
        assert_eq!(of(Phase::Single), 0x6e88_ff71_cc98_2a42);
        assert_eq!(of(Phase::Prefill), 0x619f_dbb0_c262_9a55);
        assert_eq!(of(Phase::Decode), 0x3e92_fbbd_183d_f4b6);
        assert_eq!(of(Phase::Encode), 0x3780_ba54_336a_57d0);
        // The lift of a zero result is a term of the formula. The mix sends
        // 0 to 0 and no other value there, so one key of a model and phase
        // would read as no session: it reads 1, and is a session.
        let lifted = RequestFacts {
            model: "model-a",
            session_key: Identity::hashed(0x2bb0_c344_f139_5abd),
            phase: Phase::Single,
            ..RequestFacts::default()
        }
        .session();
        assert_eq!(lifted.get(), 1);
        assert!(!lifted.is_none());
    }

    #[test]
    fn a_session_is_one_key_in_one_model_and_phase_whatever_form_the_key_takes() {
        let plain = RequestFacts {
            model: "model-a",
            session_key: Identity::plain("conversation-7"),
            ..RequestFacts::default()
        };
        // A host that holds the key's fingerprint alone names the same
        // session.
        let hashed = RequestFacts {
            session_key: Identity::hashed(fingerprint(b"conversation-7")),
            ..plain
        };
        assert_eq!(plain.session(), hashed.session());
        // So does a host that carries the model's fingerprint, made once a
        // model, where this one fingerprints the name at the read.
        let carried = RequestFacts {
            model_fingerprint: ModelFingerprint::of("model-a"),
            ..plain
        };
        assert_eq!(plain.session(), carried.session());
        let other_model = RequestFacts {
            model: "model-b",
            ..plain
        };
        let other_phase = RequestFacts {
            phase: Phase::Decode,
            ..plain
        };
        let other_key = RequestFacts {
            session_key: Identity::plain("conversation-8"),
            ..plain
        };
        for other in [other_model, other_phase, other_key] {
            assert_ne!(plain.session(), other.session());
        }
        // The model's part is its name as the request carries it: no name
        // is a name too.
        let unnamed = RequestFacts { model: "", ..plain };
        assert!(!unnamed.session().is_none());
        assert_ne!(unnamed.session(), plain.session());
        // No key, no session.
        let keyless = RequestFacts {
            session_key: Identity::NONE,
            ..plain
        };
        assert!(keyless.session().is_none());
    }

    /// A carried fingerprint is the host's word that it is the name's. A
    /// debug build stops the read of one that is another name's; a release
    /// build trusts it, and the request is then that other model's session.
    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "the carried fingerprint is not the model name's")
    )]
    fn a_carried_fingerprint_of_another_name_stops_a_debug_build_and_is_trusted_by_a_release_one() {
        let other = RequestFacts {
            model: "model-b",
            session_key: Identity::plain("conversation-7"),
            ..RequestFacts::default()
        };
        let carried = RequestFacts {
            model: "model-a",
            model_fingerprint: ModelFingerprint::of("model-b"),
            ..other
        };
        assert_eq!(carried.session(), other.session());
    }

    #[test]
    fn wire_modes_compose_as_bits() {
        let both = WireModes::HTTP.union(WireModes::GRPC);
        assert!(both.contains(WireModes::HTTP));
        assert!(!both.contains(WireModes::ZMQ));
        assert!(both.intersects(WireModes::GRPC.union(WireModes::ZMQ)));
        assert!(!WireModes::ZMQ.intersects(both));
        assert_eq!(WireModes::from_bits(0xff).bits(), 0b111);
        assert!(WireModes::NONE.is_empty());
    }

    #[test]
    fn a_load_row_reads_its_permille_as_a_ratio() {
        let row = LoadRow {
            kv_usage_permille: 250,
            ..LoadRow::default()
        };
        assert!((row.kv_usage() - 0.25).abs() < 1e-6);
    }
}
