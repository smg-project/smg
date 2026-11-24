//! The bounded view: the per-decision, handle-free picture of one request and
//! its candidates (design 4.1). Built once per decision by the host into a
//! reused buffer; every row is `Copy`, and nothing here reaches back into the
//! host. The names are the wire's: a candidate on the wire is this row with
//! its affinity facts and without its position and sample instant, a request
//! this request without its host clock and its affinity column, and positions
//! into the spec's declared lists (zones, engine kinds, connectors, labels,
//! headers) are the same positions here.

use std::sync::atomic::AtomicU64;

use super::demands::DemandFacts;
use crate::{load::signals, policies::consistent_hashing::LentRing, tracking::in_flight};

/// The identity of a candidate, as its host states it. A controller's
/// membership states a hash of the registration key and the incarnation
/// ([`identity`](crate::store::identity)): a same-URL replacement is a new
/// id. A gateway's worker list states the fingerprint of the worker's
/// address ([`WorkerId::from_address`]), one id for a worker and the one
/// that replaces it there, unless its program states ids of its own
/// ([`FixedFacts::id`]). A hash ring a caller lends finds a worker's row
/// by the address's fingerprint alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub u64);

impl WorkerId {
    /// The id a gateway's worker list states for the worker registered at
    /// `address`: the address's fingerprint, the same for every worker
    /// object registered there. A hash ring a caller lends names its
    /// workers by address ([`LentRing`]), and a decision finds a named
    /// worker's row by this: the ring serves rows whose ids are stated
    /// this way.
    #[must_use]
    pub fn from_address(address: &str) -> Self {
        Self(fingerprint(address.as_bytes()))
    }
}

/// A worker's role in the fleet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Role {
    #[default]
    Single,
    Prefill,
    Decode,
    Encode,
}

/// The role of a worker object of this type.
#[cfg(any(feature = "in-process", feature = "worker-host"))]
impl From<smg_worker::worker::WorkerType> for Role {
    fn from(worker_type: smg_worker::worker::WorkerType) -> Self {
        use smg_worker::worker::WorkerType;
        match worker_type {
            WorkerType::Regular => Self::Single,
            WorkerType::Prefill => Self::Prefill,
            WorkerType::Decode => Self::Decode,
            WorkerType::Encode => Self::Encode,
        }
    }
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

impl Phase {
    /// The phase's name, as a document writes it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Encode => "encode",
        }
    }

    /// The role whose list decides a request of this phase that stands
    /// alone: the role its demands are held by
    /// ([`Demands::held_by`](super::demands::Demands::held_by)), and the
    /// list a controller lends it.
    #[cfg(feature = "in-process")]
    pub(crate) const fn role(self) -> Role {
        match self {
            Self::Single => Role::Single,
            Self::Prefill => Role::Prefill,
            Self::Decode => Role::Decode,
            Self::Encode => Role::Encode,
        }
    }
}

/// What the request's block hashes were computed over.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HashKind {
    /// Token blocks of the model's block size of record.
    #[default]
    Token,
    /// Fixed windows of bytes over the text, for a host without a tokenizer.
    Text,
}

/// Bytes of a request's text a token, for a host with no tokenizer. A text
/// window is this many times the model's block size in bytes, and a text's
/// size in tokens is its bytes over this (`tokens_or_text`).
pub const TEXT_BYTES_PER_TOKEN: u32 = 4;

/// A request's size in tokens for a reader that prices a request of text
/// by its length: `tokens`, the count of its token ids, and where it states
/// none its text's `length` in bytes at [`TEXT_BYTES_PER_TOKEN`] a token.
/// The one statement of the rule: a request's facts read it
/// ([`RequestFacts::prompt_size`]), and so does a held decision's record,
/// by the bytes its lease keeps
/// ([`Lease::prompt_size`](crate::tracking::lease::Lease::prompt_size)).
pub(crate) const fn tokens_or_text(tokens: u32, length: u32) -> u32 {
    if tokens == 0 {
        length / TEXT_BYTES_PER_TOKEN
    } else {
        tokens
    }
}

/// The first `tokens` tokens of a request hashed in `kind`, in the unit its
/// blocks are sized in: tokens, and under the text kind their bytes.
pub(crate) fn units_within(tokens: u32, kind: HashKind) -> u32 {
    match kind {
        HashKind::Token => tokens,
        HashKind::Text => tokens.saturating_mul(TEXT_BYTES_PER_TOKEN),
    }
}

/// How many whole blocks of `block_size` lie within the first `tokens`
/// tokens of a request hashed in `kind`: the tokens over the block size,
/// under the text kind their bytes over the window's size in bytes. A
/// block size is one at least.
pub(crate) fn blocks_within(tokens: u32, kind: HashKind, block_size: u32) -> u32 {
    units_within(tokens, kind) / block_size
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

/// The one wire a worker object of this connection mode serves.
#[cfg(any(feature = "in-process", feature = "worker-host"))]
impl From<smg_worker::worker::ConnectionMode> for WireModes {
    fn from(mode: smg_worker::worker::ConnectionMode) -> Self {
        use smg_worker::worker::ConnectionMode;
        match mode {
            ConnectionMode::Http => Self::HTTP,
            ConnectionMode::Grpc => Self::GRPC,
            ConnectionMode::Zmq => Self::ZMQ,
        }
    }
}

/// Health as the pool reads it. Each flag is one bit of the host's routing
/// state, read once per decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Health {
    /// The worker's status is ready.
    pub healthy: bool,
    /// The pool's circuit breaker refuses execution, and no rule lifts it.
    /// A host writes it into a row it copies for the decision, for a veto
    /// of its own that holds, beside the stalled bit every veto of its own
    /// writes. Over a publication's own rows such a row stands among the
    /// host's held rows instead ([`Constraints::held`]).
    pub breaker_open: bool,
    /// A standing veto of the asking host masks the row: the liveness
    /// veto, unreachable or wedged, or a veto that holds, which writes the
    /// breaker bit too. A host writes it into a row it copies for the
    /// decision. Over a publication's own rows the asking host's vetoes
    /// stand beside the rows instead, and a reader of this bit asks both
    /// ([`Constraints::masks`]). The all-vetoed rule lifts it as its last
    /// resort, on a row whose breaker bit is clear.
    pub stalled: bool,
    /// Being drained; nothing new goes there.
    pub draining: bool,
    /// The latched overload verdict of the last load report: the engine
    /// is at a threshold, or an answer of it named another of its ranks
    /// and not this one. That mark ends when an answer or a record names
    /// the rank again, when its member's rows go, or with the build that
    /// follows its member's registration under another rank number. The
    /// bit reads the mark only while a rank the answer named has a record
    /// on file.
    pub overloaded: bool,
}

impl Health {
    /// Every routing veto but the overload one, as the row carries them: a
    /// veto that stands beside lent rows is asked of the decision's
    /// constraints ([`Constraints::masks`]).
    pub const fn routable_ignoring_overload(self) -> bool {
        self.healthy && !self.breaker_open && !self.stalled && !self.draining
    }

    /// Every routing veto the row carries
    /// ([`Self::routable_ignoring_overload`]).
    pub const fn routable(self) -> bool {
        self.routable_ignoring_overload() && !self.overloaded
    }
}

/// A ratio as a permille field: the one rounding a report's ratio and a
/// usage threshold both take, so one rank's usage at or above a threshold
/// stated as a ratio is never read as under it. A mean of
/// several ranks' rounded usages within a permille of the threshold can
/// read on either side of it.
pub fn permille(ratio: f64) -> u16 {
    (ratio * 1000.0).round().clamp(0.0, 1000.0) as u16
}

/// A usage threshold as a permille field: [`permille`]'s rounding with a
/// floor of one, the one conversion every threshold stated as a ratio
/// takes. At no permille the store reads every engine with a rank on file
/// as over the threshold, an idle one too, so a valid threshold that would
/// round to none is held as one.
pub(crate) fn threshold_permille(usage: f64) -> u16 {
    permille(usage).max(1)
}

/// One load row as the view carries it: the engine load record's routing
/// core, in the record's own names, rounded to narrow fields by the host.
/// Permille fields are the record's ratios times 1000.
///
/// The record gains fields: code outside this crate that builds one writes the
/// rest as `..LoadRow::default()`, and a pattern of it ends with `..`.
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
    /// Tokens per second the worker drains; `0` when unknown, and a known
    /// rate under one reads as `1`.
    pub live_rate: u32,
    /// Context tokens resident for the running requests; `0` too when the
    /// engine does not count them.
    pub resident_tokens: u32,
    /// Whether the report counts resident tokens: an engine that does not
    /// count them sends none, which is not a count of zero. The host's own
    /// reading of the report, in no plugin's view.
    pub resident_counted: bool,
    /// Age of the report's sample at the decision, in milliseconds. On a
    /// row a store lends, it is the age the report carried when it was
    /// recorded. The request's `now_micros` less the row's
    /// `load_sampled_at_micros` gives the age on either form of the row.
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

/// The key of a prefix one block longer: the key of the prefix before it
/// folded with the block's hash. A host hashes each block by its own
/// content ([`RequestFacts::block_hashes`]), so one hash names one block
/// wherever it stands. A prefix is named by this fold over its blocks'
/// hashes in order from the first, started at the request's cache
/// namespace ([`RequestFacts::namespace_seed`]): the ring's key of a prefix
/// and the sent-prefix predictor's are that one number. The key's top bits
/// take every bit of both.
pub(crate) const fn fold_block(prefix: u64, hash: u64) -> u64 {
    (prefix.rotate_left(23) ^ hash).wrapping_mul(0x9E37_79B9_7F4A_7C15)
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

    /// The fingerprint's value, for a key the crate derives from a model.
    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

/// A session: a key within its model and phase, as the one value a lease
/// carries and the session table is keyed by. [`RequestFacts::session`]
/// derives the value from a request. A lease read from the wire already
/// carries that value.
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

    /// The bytes [`Self::text`] writes: a `u64` in sixteen hexadecimal
    /// digits.
    pub const TEXT_LEN: usize = 16;

    /// The session's number as text, written into `buffer`: sixteen
    /// lower-case hexadecimal digits, the highest first. A worker object
    /// counts a key's requests under a text, and where held decisions on
    /// file count a session's requests there, this is that text
    /// ([`HostCount::session_booked`](crate::tracking::HostCount::session_booked)).
    /// It is the one writer of it: what moves a held decision's count onto
    /// a worker object and what reads a record's count both call it, so
    /// the two name one entry. Nothing is allocated: the buffer is the
    /// caller's, on its stack.
    pub fn text(self, buffer: &mut [u8; Self::TEXT_LEN]) -> &str {
        for (at, digit) in buffer.iter_mut().enumerate() {
            let nibble = (self.0 >> (60 - 4 * at)) & 0xf;
            *digit = b"0123456789abcdef"[nibble as usize];
        }
        // Sixteen ASCII digits: the conversion cannot fail.
        std::str::from_utf8(buffer).unwrap_or_default()
    }

    /// A session from the scoped value a lease carries on the wire.
    #[cfg(any(test, feature = "client"))]
    pub(crate) const fn from_raw(value: u64) -> Self {
        Self(value)
    }
}

/// What the affinity stages read of one candidate: a column beside the rows,
/// by the candidate's index ([`RequestFacts::affinity`]). A host fills it for
/// a profile whose stages declare it
/// ([`Needs::affinity`](crate::document::Needs::affinity)) and for no other, so a
/// decision that reads none of it copies none of it. `Default` is nothing
/// held by a worker just admitted, which is what every candidate reads as
/// when the column is empty. On a wire the five are fields of the
/// candidate's own record, at the numbers they were given.
///
/// The record gains fields: code outside this crate that builds one writes the
/// rest as `..AffinityFacts::default()`, and a pattern of it ends with `..`.
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
/// release build reads them as they stand, a row past the count naming
/// nobody. In-process only.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AffinityRows<'a> {
    /// The rows that hold a leading part of the request, ascending by row,
    /// each with what it holds. A row not named reads as holding nothing.
    pub holders: &'a [AffinityHolder],
    /// The pool's warming rows, ascending, by the warm-up stage's rule as
    /// the host applied it: the thin whatever the fleet's age, else the
    /// young unless every row is young.
    pub warming: &'a [u32],
    /// The warming rows are the thin.
    pub thin: bool,
    /// The mean in-flight count the spill gate reads behind a draw: the
    /// mean of the rows the picker drew, so a row the host named does not
    /// move the mean it is judged against. The engine sets it; a host
    /// leaves `None`, which reads as the mean of the rows a stage is
    /// handed, and a debug build asserts that it did.
    pub mean_in_flight: Option<f32>,
}

/// One row a host names as a holder: what the row's entry of a column says
/// of what it holds, by both readings. Its age and its index size stay with
/// the host, which ruled the warming rows from them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AffinityHolder {
    /// The row's index in the candidate slice.
    pub row: u32,
    /// As [`AffinityFacts::overlap_blocks`].
    pub overlap_blocks: u32,
    /// As [`AffinityFacts::predicted_blocks`].
    pub predicted_blocks: u32,
}

impl<'a> AffinityRows<'a> {
    /// Row `index`'s entry: what the host said the row holds when it is
    /// named, nothing held otherwise.
    ///
    /// The search ends on the first holder at or past the row, and the
    /// entry is read from that holder or from a holder of nothing, by
    /// whether it is the row: a choice between two places, where a branch
    /// would tell a named row from another. Behind a draw the rows asked
    /// come in the draw's order, and such a branch is one nothing predicts.
    #[inline]
    pub(crate) fn entry(&self, index: u32) -> AffinityFacts {
        const NOBODY: AffinityHolder = AffinityHolder {
            row: 0,
            overlap_blocks: 0,
            predicted_blocks: 0,
        };
        let Some(last) = self.holders.len().checked_sub(1) else {
            return AffinityFacts::default();
        };
        let at = self.holders.partition_point(|holder| holder.row < index);
        let found = &self.holders[at.min(last)];
        let holder = if found.row == index { found } else { &NOBODY };
        holder.facts()
    }

    /// One pass's reads of the rows, walked beside the rows the pass asks.
    #[inline]
    pub(crate) fn walk(self) -> NamedWalk<'a> {
        NamedWalk(Walk::new(self.holders))
    }

    /// Whether the engine drew the rows a stage is handed: it set their
    /// mean ([`AffinityRows::mean_in_flight`]). They stand in the draw's
    /// order, so a pass reads each row by a search and keeps no cursor; a
    /// walk is the read of a pool read whole, whose rows stand ascending.
    /// The sign chooses a cost and never an answer: both reads answer
    /// alike in whatever order the rows are asked.
    #[inline]
    pub(crate) fn drawn(&self) -> bool {
        self.mean_in_flight.is_some()
    }
}

impl AffinityHolder {
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

#[cfg(test)]
thread_local! {
    /// How many times a walk on this thread looked behind its cursor: a
    /// case reads it to tell a walk from a search.
    pub(crate) static LOOKS_BEHIND: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
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
        #[cfg(test)]
        LOOKS_BEHIND.with(|looks| looks.set(looks.get() + 1));
        self.list.partition_point(|entry| row(entry) < index)
    }
}

/// One pass's reads of the rows a host named, in the pass's own order.
#[derive(Debug)]
pub(crate) struct NamedWalk<'a>(Walk<'a, AffinityHolder>);

impl NamedWalk<'_> {
    /// Row `index`'s entry, as [`AffinityRows::entry`] reads it.
    #[inline]
    pub(crate) fn entry(&mut self, index: u32) -> AffinityFacts {
        self.0
            .find(index, |holder| holder.row)
            .map_or_else(AffinityFacts::default, AffinityHolder::facts)
    }
}

/// What a pass reads of what the candidates hold, taken once
/// ([`RequestFacts::feed`]): the host's column, or the rows it named in the
/// column's place.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Feed<'a> {
    Column(&'a [AffinityFacts]),
    Named(AffinityRows<'a>),
}

impl Feed<'_> {
    /// Nobody holds anything: an empty column, or no holder named.
    pub(crate) fn is_empty(self) -> bool {
        match self {
            Self::Column(column) => column.is_empty(),
            Self::Named(named) => named.holders.is_empty(),
        }
    }

    /// The mean in flight the engine set behind a draw
    /// ([`AffinityRows::mean_in_flight`]): the rows named carry it, a
    /// column carries none.
    pub(crate) fn drawn_mean(self) -> Option<f32> {
        match self {
            Self::Column(_) => None,
            Self::Named(named) => named.mean_in_flight,
        }
    }

    /// Row `index`'s entry: a column's own, where an index past the end
    /// fails, or a named row's ([`AffinityRows::entry`]).
    #[inline]
    pub(crate) fn entry(self, index: u32) -> AffinityFacts {
        match self {
            Self::Column(column) => column[index as usize],
            Self::Named(named) => named.entry(index),
        }
    }

    /// Row `index`'s entry behind a decision: a row past a column holds
    /// nothing, as a row no host named. Only what is stated of a pick reads
    /// this way: it orders nothing, and the decision may have read no entry
    /// of that row. A pass reads [`Self::entry`].
    #[inline]
    pub(crate) fn entry_or_default(self, index: u32) -> AffinityFacts {
        match self {
            Self::Column(column) => column.get(index as usize).copied().unwrap_or_default(),
            Self::Named(named) => named.entry(index),
        }
    }

    /// Row `index`'s entry as one pass reads it: by the pass's own `reads`
    /// where the pool is read whole, and by a search behind a draw
    /// (`DRAWN`), where the rows stand in the draw's order and a pass
    /// keeps no cursor. The choice is a constant of the body that reads.
    #[inline(always)]
    pub(crate) fn read<const DRAWN: bool>(
        self,
        reads: &mut Reads<'_>,
        index: u32,
    ) -> AffinityFacts {
        if DRAWN {
            self.entry(index)
        } else {
            reads.entry(index)
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
/// what every decision reads. Its fields take 107 bytes, and the row is
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
///
/// The record gains fields: code outside this crate that builds one writes the
/// rest as `..Candidate::default()`, and a pattern of it ends with `..`.
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
    /// decision takes its inputs; a release build does not check. A count
    /// the host lent beside its rows is read at the position too
    /// ([`RequestFacts::in_flight`]): the row's place in the host's pool,
    /// whatever its index.
    pub position: u32,
    /// The worker's id ([`WorkerId`]). Each rank of a data-parallel engine
    /// is a row of its own with an id of its own: a rank's worker object
    /// stands at an address of its own, and its id is that address's
    /// fingerprint. The rows of one engine are joined by
    /// [`Self::engine`], and never by this id.
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
    /// Requests the deciding host holds on the worker right now. A host
    /// that lends its counts beside the rows
    /// ([`RequestFacts::lent_in_flight`]) leaves this zero, so a reader
    /// asks [`RequestFacts::in_flight`].
    ///
    /// The in-process front reads it off the worker object at every
    /// decision: the object's own counter, which a host raises where it
    /// makes the worker's load guard and lowers where the guard is
    /// dropped. A guard can be made ahead of its send. The decode of a
    /// pair named at the first decision is so counted on its row before
    /// it is sent, and the classic policies count it the same way.
    pub in_flight: u32,
    /// The latest load row, when the host has a report for the worker.
    pub load: Option<LoadRow>,
    /// The sample instant of `load` on the host's clock, in microseconds since
    /// the host's origin: dispatches made at or before it are the engine's by
    /// now and leave a load scorer's credit, or those made a profile's
    /// `sent_report_delay_ms` before it where the profile sets one.
    /// In-process only; the wire carries the row's age.
    pub load_sampled_at_micros: u64,
    /// The session key of the request is pinned to this worker. A host
    /// writes it into a row it copies for the decision. Over a publication's
    /// own rows the pin stands beside the rows instead
    /// ([`RequestFacts::pinned_at`]).
    pub session_hit: bool,
    /// Requests of the request's session key in flight on this worker. A
    /// host writes it into a row it copies for the decision.
    pub session_in_flight: u16,
    /// The pairing class: `0` for a worker outside any pairing. Which decode
    /// classes a prefill class pairs with is the pair table's to say
    /// ([`PairTable`](crate::engine::PairTable)); with no table a decode pairs with a
    /// prefill of the same class. The store derives the classes and their
    /// table where a membership write moves a member or an offer, and again
    /// under a generation of another pairing mode
    /// ([`store::pairing`](crate::store::pairing)). It numbers each side by
    /// itself, so a store's class pairs through its table alone: a prefill
    /// and a decode of one number are not a pair for that. A host that
    /// builds its own rows keys it by the open prefill a decode partners
    /// with.
    pub pair_class: u32,
    /// Host-supplied admission bit: the prefill capacity window has a slot.
    /// The store knows no window, so a row it builds reads the bit open.
    pub has_capacity: bool,
    /// Host-supplied admission bit: the decode's running window has a room
    /// nobody waiting at its gate is in line for. Open on a row the store
    /// builds, as [`Self::has_capacity`] is.
    pub has_room: bool,
    /// The zone, as a position into the spec's declared list; `0` when
    /// unzoned.
    pub zone: u16,
    /// The worker's labels, as bits over the spec's declared list (at most
    /// 64).
    pub labels: u64,
    /// Encode-side compatibility for an encode-prefill-decode dispatch.
    pub ec_compatible: bool,
    /// The worker takes pixel media: an image or a video
    /// ([`MediaCounts::holds_pixels`]). The crate's docs say of such a
    /// worker that it takes images.
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
    /// The place of the row's engine among the engines of its list, in the
    /// order each engine's first row stands in the list: the rows that
    /// state one place are the ranks of one data-parallel engine, and a
    /// worker that stands alone states a place no other row states. Where
    /// the list lists its engines ([`Engines`]) it is the place
    /// [`Engines::rows`] takes. The number holds within one generation of
    /// one list and is no id beyond it.
    ///
    /// The engine in a gateway's own process writes it, once a list, where
    /// it finds the list's engines. The engine over the policy store and a
    /// default record leave `0` on every row, which would read as one
    /// engine: a stage reads the field under the engine in a gateway's own
    /// process alone, until that engine is the one engine. In-process
    /// only: no wire form carries it. After the contract's first cut (the
    /// wire numbers its fields in the record's order).
    pub engine: u32,
}

impl Candidate {
    /// A record of a worker that can be sent to, for a caller that makes
    /// records by hand: its status is ready and no other health bit is set,
    /// both admission bits are open, and it is an engine of its own
    /// ([`Self::engine`] is its position). Every other field is the
    /// default's, and a caller writes what its stage reads over it:
    /// `Candidate { in_flight: 3, ..Candidate::ready(0, id, Role::Single) }`.
    ///
    /// An engine states its place among the engines of its list, so the
    /// position is that place only while every record before it is an
    /// engine of its own. A caller that makes several records the ranks of
    /// one engine writes the engine of each record behind them too.
    ///
    /// A default record is no such record. It is not ready and both of its
    /// admission bits are closed, so the health and admission filters drop
    /// it, and every default record states engine `0`.
    #[must_use]
    pub fn ready(position: u32, id: WorkerId, role: Role) -> Self {
        Self {
            position,
            id,
            role,
            health: Health {
                healthy: true,
                ..Health::default()
            },
            has_capacity: true,
            has_room: true,
            engine: position,
            ..Self::default()
        }
    }
}

/// What a program says of a worker for the half of its record that no
/// request changes, where the worker object holds no such fact: whether
/// the worker takes images, the id the program knows the worker by where
/// it keeps ids of its own, and what the worker states for a request's
/// hard demands. A host that keeps such a fact beside its workers tells it
/// to what writes the workers' records, once a list: a gateway of its own
/// registrations, a controller of the stored worker an object stands for.
/// The default says nothing. Further facts add their fields, so a host
/// starts from the default and sets what it knows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FixedFacts {
    /// Whether the worker takes pixel media, an image or a video
    /// ([`Candidate::vision`]).
    pub vision: bool,
    /// The id the program knows the worker by ([`Candidate::id`]): a
    /// controller's id of the stored worker an object stands for. A
    /// decision's targets name the worker by it, and so does every state
    /// the engine writes from a decision. `None` for the fingerprint of
    /// the worker's address, which a gateway's own process names a worker
    /// by. A program states an id for every worker of its lists or for
    /// none: one id space a program. A stated id is never `WorkerId(0)`:
    /// zero reads as absent wherever an id travels (a lease, a trace, a
    /// session's pins), so a zero stated here states none. A program that
    /// states ids lends no hash ring ([`RequestFacts::ring`]): a lent ring
    /// finds a worker's row by the fingerprint of its address, so it would
    /// find none and every key would go by its position alone.
    pub id: Option<WorkerId>,
    /// What the worker states for a request's hard demands: its entry of
    /// the column beside the records.
    pub demands: DemandFacts,
}

impl FixedFacts {
    /// The id the engine knows the worker by: the one stated
    /// ([`Self::id`]), or `address`, the fingerprint of the worker's
    /// address, where none is stated or the one stated is zero. A list's
    /// records are written by it, so no record carries zero for an id.
    #[must_use]
    pub fn id_or(&self, address: WorkerId) -> WorkerId {
        match self.id {
            Some(id) if id != WorkerId(0) => id,
            _ => address,
        }
    }
}

/// How far back the store's decision clock starts.
pub(crate) const LOAD_REPORT_CLOCK_HISTORY_MS: u32 = 3_600_000;

/// A sample clipped to the clock's origin must already be past the limit.
pub(crate) const MAX_LOAD_REPORT_AGE_MS: u32 = LOAD_REPORT_CLOCK_HISTORY_MS - 1;

/// The oldest sample a decision can use under a report age limit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LoadReportCutoff(u64);

impl LoadReportCutoff {
    /// The cutoff of a view that sets no limit: no sample is older than
    /// the clock's origin, so every report is read.
    pub(crate) const ANY_AGE: Self = Self(0);

    pub(crate) fn new(now_micros: u64, max_age_ms: u64) -> Self {
        Self(now_micros.saturating_sub(max_age_ms * 1_000))
    }

    pub(crate) fn load(self, row: &Candidate) -> Option<&LoadRow> {
        row.load
            .as_ref()
            .filter(|_| row.load_sampled_at_micros >= self.0)
    }
}

/// Media on the request: counts and bytes, for workers that take media.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MediaCounts {
    pub images: u16,
    pub audio: u16,
    pub video: u16,
    pub bytes: u32,
}

impl MediaCounts {
    /// No media: every count at zero, and no bytes.
    pub const NONE: Self = Self {
        images: 0,
        audio: 0,
        video: 0,
        bytes: 0,
    };

    /// Whether the request holds pixels: an image or a video. Only a
    /// worker with a vision encoder takes such media, so a test of vision
    /// by these counts reads this, never the image count alone. A host
    /// states each count as it is.
    #[must_use]
    pub const fn holds_pixels(&self) -> bool {
        (self.images | self.video) != 0
    }
}

/// An allow-listed header: its name as a position into the spec's declared
/// header list, and its value. A value never pairs with a name the spec did
/// not allow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Header<'a> {
    pub name: u8,
    pub value: &'a str,
}

/// The mean of the in-flight counts of `rows` of `candidates`; zero over no
/// row.
// Forced into its callers: the walk is two small loops, one a kind of rows.
// Out of line they cost an affinity pass a call and one instruction a row
// more than the one loop that stood inside the pass.
#[inline(always)]
pub(crate) fn mean_in_flight(
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    rows: &[u32],
) -> f32 {
    if rows.is_empty() {
        return 0.0;
    }
    let sum: u64 = crate::tracking::in_flight::with_counts!(facts, |counts| rows
        .iter()
        .map(|&i| u64::from(counts(&candidates[i as usize])))
        .sum());
    sum as f32 / rows.len() as f32
}

/// Where a worker's object counts the requests of a request's session
/// key: the name the count stands under there. A request's maker states it
/// ([`RequestMaker::session_counted_under`](super::decision_engine::RequestMaker::session_counted_under)),
/// and the reader of a record's count asks under that name.
///
/// One worker object stands in one process, which counts by its own guards
/// or by the decisions it holds on file and never by both, so the two
/// names do not meet on one object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SessionCountedUnder {
    /// The key's own text: the caller's own guard counts each request on
    /// the worker it sends it to, for as long as the request runs. What a
    /// process that serves its requests itself does.
    #[default]
    KeyText,
    /// The session's number written as text ([`Session::text`]): the
    /// decisions held on file count them, one a lease that books the
    /// worker
    /// ([`HostCount::session_booked`](crate::tracking::HostCount::session_booked)).
    /// What a controller does, which serves no request. Its frame carries
    /// the key in the clear or hashed: the lease carries the number alone.
    SessionNumber,
}

impl SessionCountedUnder {
    /// The name the count of `facts`' session key stands under on a
    /// worker's object: the key's text as the request carries it, or the
    /// session's number as [`Session::text`] writes it into `buffer`.
    /// `None` for a request with no such name: no key in the clear under
    /// the first, no key at all under the second. The one place that turns
    /// the statement into the name.
    pub fn name<'a: 'b, 'b>(
        self,
        facts: &RequestFacts<'a>,
        buffer: &'b mut [u8; Session::TEXT_LEN],
    ) -> Option<&'b str> {
        match self {
            Self::KeyText => facts.session_key.plain,
            Self::SessionNumber => {
                let session = facts.session();
                (!session.is_none()).then(|| session.text(buffer))
            }
        }
    }
}

/// One request as a decision sees it. Borrowed from the host for the
/// duration of the decision.
///
/// The record gains fields: code outside this crate that builds one writes the
/// rest as `..RequestFacts::default()`, and a pattern of it ends with `..`.
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
    /// The count of the request's token ids, or the count its host states,
    /// and zero where the request gives no ids: a request of text alone
    /// states its text's length ([`Self::prompt_bytes`],
    /// [`Self::prompt_chars`]) and no count, so
    /// a stage that prices a request of no stated size prices it by its
    /// own rule. A stage that reads the size declares
    /// [`Needs::prompt_size`](crate::document::Needs::prompt_size): a host
    /// that sends a request on without reading its prompt states zero
    /// where no stage does. Nothing masks the field for a stage of a
    /// host's own build, which reads it here: undeclared, it reads zero on
    /// such a request and the true size on every other.
    pub prompt_tokens: u32,
    /// One hash a whole block of the prompt, in order from the first,
    /// capped by the host. A hash is of its own block's content and of
    /// nothing ahead of it: what names a prefix is the fold of its
    /// blocks' hashes, which the stages that key by a prefix make.
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
    /// A host that asks in process states no budget by `0`
    /// ([`crate::engine::Request`]); on the wire a frame's `0` says none is
    /// left.
    pub budget_micros: u32,
    /// Allow-listed headers only.
    pub headers: &'a [Header<'a>],
    pub media: MediaCounts,
    /// The per-decision seed, recorded in the trace: the only randomness a
    /// stage gets. The seed's own stream is the one the profile's picker
    /// reads. A filter or a scorer its host registers is to draw from the
    /// stream of its own name
    /// ([`Draw::for_stage`](crate::Draw::for_stage)), whose values are not
    /// the picker's.
    pub seed: u64,
    /// The size of one of the request's blocks as the host counted them.
    /// Under the token kind it is in tokens: an index's learned block size
    /// on an event-fed request, the configured one on a tree-fed request;
    /// `0` reads the view's. Under the text kind it is the window's size in
    /// bytes, and a text request always states it: the wire refuses one
    /// that carries hashes and states none. Last on the wire (the
    /// numbering follows the record's order).
    pub block_size: u32,
    /// The length of the request's text in characters, where the request
    /// gives text and no token ids and a stage of the document reads a
    /// request's size ([`Needs::prompt_size`](crate::document::Needs::prompt_size));
    /// `0` otherwise. Its reader at a decision is a stage that sizes a
    /// request of text by its characters, as `bucket` does, and a lease
    /// keeps it ([`Lease::chars`](crate::tracking::lease::Lease::chars)).
    /// Counting it is one pass over the text, made once a request.
    /// In-process only: a frame states a count.
    pub prompt_chars: u32,
    /// The length of the request's text in bytes, where the request gives
    /// text and no token ids; `0` otherwise. It is written wherever the
    /// text is at hand and costs no pass. [`Self::prompt_size`] makes a
    /// count of it, which every reader that prices a request of text by
    /// its length takes: the room rule, a plugin's view of a request, the
    /// prefix stages, whose windows of text are bytes, and a frame. A
    /// lease keeps it ([`Lease::bytes`](crate::tracking::lease::Lease::bytes)),
    /// so a held decision reads the same count.
    /// In-process only: a frame states a count.
    pub prompt_bytes: u32,
    /// The host's clock at the decision, in microseconds since its origin.
    /// In-process only; no absolute instant crosses the wire.
    pub now_micros: u64,
    /// One more than the position of the row the request's session key is
    /// pinned to, where the host did not write the session bit into the
    /// row: a view of a publication's own rows hands the pin here. A reader
    /// of one row's session bit asks both ([`RequestFacts::pins`]), and a
    /// pass over the rows looks for the pinned position once, as the
    /// session scorer does. Zero for no pin, and where the rows were written
    /// for the decision. A pair takes none: a pin is a position in one
    /// pool, and neither side reads it. In-process only.
    pub pinned_at: u32,
    /// What the leases on file book on each row, where the host did not
    /// write the count into the row: a view of a publication's own rows
    /// lends the table here. A reader of a row's requests in flight asks
    /// both ([`RequestFacts::in_flight`]). `None` where the rows carry
    /// their counts: rows written for the decision, and a host that keeps
    /// the count itself. A pair takes none: a position is a place in one
    /// pool, and a pair's rows carry what counts their host wrote.
    /// In-process only.
    pub lent_in_flight: Option<&'a in_flight::PoolInFlightCounts<'a>>,
    /// The host holds each row's count of this request's session key, the
    /// count a capped sessions rule asks for
    /// ([`Needs::session_in_flight`](crate::document::Needs::session_in_flight)),
    /// written into the rows ([`Candidate::session_in_flight`]): a sessions
    /// rule's in-flight cap reads it there. A key kept on its
    /// worker over the overload veto is kept under a capped rule only
    /// where this is set: with no count the cap could not hand the key on,
    /// so the veto stands for it as for any request. Where the count
    /// stands on a worker's object is the request's maker's to say
    /// ([`SessionCountedUnder`]). In-process only.
    pub session_counted: bool,
    /// The candidates' affinity facts, by index in the candidate slice the
    /// decision is made over: the one column the affinity stages read, in
    /// the candidates' own order. It is EMPTY or holds exactly one entry a
    /// candidate; the engine refuses any other before a stage runs. An
    /// empty column is the host's word that it holds nothing, unless the
    /// host names rows in its place ([`RequestFacts::named`]): it reads as
    /// nothing held for every candidate, under a profile that declared
    /// [`Needs::affinity`](crate::document::Needs::affinity) as under one that did
    /// not, and the engine neither refuses it nor counts it. A host that
    /// owes the column and hands none is routed by its other stages alone,
    /// with no signal. A pair decision takes each side's column from
    /// its pool ([`PairPools`](crate::engine::PairPools)) and none from here: a
    /// non-empty column on the request of a pair decision is refused, as a
    /// column of another length is. The host keeps it readable, as it keeps
    /// the candidates, until the decision is committed. In-process only: a
    /// wire carries each candidate's facts in its record.
    pub affinity: &'a [AffinityFacts],
    /// The rows the host names in that column's place ([`AffinityRows`]):
    /// beside an empty column, the affinity stages read them. A pair
    /// decision takes none: its sides read their pools' columns. In-process
    /// only.
    pub named: Option<AffinityRows<'a>>,
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
    /// The signal cells beside the rows of this decision, by the row's
    /// position: handed by a host for a profile whose stages read a signal
    /// ([`Needs::signal_cells`](crate::document::Needs::signal_cells)), and for no
    /// other. The engine over worker lists takes none from its host: it
    /// makes a list's cells from the load snapshot its decision holds, and
    /// cells a caller put here are not read by a single request's decision.
    /// `None` reads as no value for every row. A pair decision takes
    /// each side's cells from its pool ([`PairPools`](crate::engine::PairPools)),
    /// and an encode pair its encoders' from theirs: cells here are not
    /// read there. One word for every signal the document declares: a
    /// second signal adds none. In-process only: the contract's facts gain
    /// nothing.
    pub signal_cells: Option<&'a signals::SignalCells>,
    /// The fingerprint of the request's client key, `0` where it has none
    /// ([`KeySources::derive_client`](crate::document::KeySources::derive_client)):
    /// what a ring places by where the request carries no session key. It
    /// lifts no overload veto, pins no session and is counted under no key.
    /// The wire's request facts carry it to a controller, and a plugin's
    /// frame keeps `0` under every exposure.
    pub client_key: u64,
    /// The hash ring the request's caller keeps of the model's workers,
    /// lent to this decision ([`LentRing`]): the ring stage places the
    /// request where this ring's walk reaches, and its own ring learns
    /// nothing of the decision. The rows of such a decision carry the ids
    /// of their workers' addresses ([`WorkerId::from_address`]): a program
    /// that states ids of its own lends none ([`FixedFacts::id`]). A caller
    /// that keeps rings lends one with every such request, an empty one
    /// for a model whose ring it has not built: a key its walk hands to no
    /// kept worker goes to the kept worker at the key's position modulo
    /// their count, and under the prefix form to the worker the stages
    /// behind the ring take. `None` from a caller that keeps no ring, as a
    /// controller: the stage places by its own. Both sides of a pair take
    /// the one ring, each over its own list: a side takes a worker of the
    /// ring only where its list keeps that worker, and a key no kept
    /// worker of the side takes goes by the same rule over that side's
    /// kept workers. An encode request's items take none: the place stated
    /// is the request's routing value's, and an item is placed by its own
    /// key on the stage's ring. In-process only.
    pub ring: Option<&'a dyn LentRing>,
    /// Where the request stands on the lent ring, as its caller states it:
    /// the position of the value it is placed by ([`LentRing::position`]).
    /// A routing value gives one entry. Empty where the request states
    /// nothing to place by, and the lent ring is then not asked. A caller
    /// states a key's position with every ring it lends, an empty ring
    /// too. The list stands as long as the request does: a caller that
    /// finds a position while it makes the request, with the request's
    /// keys ([`KeySources::derive_keys`](crate::document::KeySources::derive_keys)),
    /// keeps it beside the request. A prompt gives one entry for each of
    /// its heads the caller's ring places it by, the deepest first: a stage
    /// under a bounded-load rule tries the entries in that order, and any
    /// other reads the first. The one call states them by the profile's
    /// word: a routing value's own entry under the plain ring, and under
    /// the prefix form what the lender answers for each head
    /// ([`LentRing::head_position`]). In-process only: no frame carries it.
    pub ring_positions: &'a [u64],
    /// The request's own id handed [`Self::session_key`]: a `rid` source
    /// of the binding's `routing_key` block, and no header. The prefix
    /// form of a ring places no request by such a key. It names the
    /// session alone: a sessions rule pins it, and the ring places the
    /// request by its prompt, as one that carries no routing key. The
    /// plain ring places by it as by any session key. The one call sets it
    /// where it derives the key
    /// ([`DerivedKey::source`](crate::document::DerivedKey)). An encode
    /// item's facts state it as false: the key they carry is the item's
    /// own. In-process only: no frame states it, so under a controller's
    /// address the prefix form places by any session key.
    pub session_key_from_rid: bool,
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

/// The fingerprint of the key whose session within a model and a phase is
/// `session`: the formula of [`RequestFacts::session`] run backwards, each
/// step of which is one-to-one. A host that is sent a session key as the
/// session's value reads the key's fingerprint by it, so its record is the
/// one the plain key gives: the session is `session` again, and a stage
/// that reads the key itself reads the same key. `model` is the
/// fingerprint of the model's name.
///
/// `0` for no session, and for the one value a model and a phase give the
/// fingerprint no key has: neither is a key's session. The value `1` reads
/// as the key the formula sends there by itself, not the one it lifts
/// there.
pub const fn key_of_session(session: u64, model: u64, phase: Phase) -> u64 {
    if session == 0 {
        return 0;
    }
    let mut h = session;
    h ^= (h >> 31) ^ (h >> 62);
    h = h.wrapping_mul(0x3196_42b2_d24d_8ec3);
    h ^= (h >> 27) ^ (h >> 54);
    h = h.wrapping_mul(0x96de_1b17_3f11_9089);
    h ^= (h >> 30) ^ (h >> 60);
    let scope = model ^ session_phase(phase).wrapping_mul(SESSION_MIX);
    h ^ scope.wrapping_mul(SESSION_MIX)
}

impl<'a> RequestFacts<'a> {
    /// The request's size in tokens for a reader that prices a request of
    /// text by its length (`tokens_or_text`): the count of its token ids
    /// ([`Self::prompt_tokens`]), and where it states none its text's
    /// bytes at [`TEXT_BYTES_PER_TOKEN`] a token ([`Self::prompt_bytes`]).
    /// Zero where it states neither. A window of text is bytes, so a
    /// prefix stage divides one unit by the same unit.
    #[inline]
    #[must_use]
    pub const fn prompt_size(&self) -> u32 {
        tokens_or_text(self.prompt_tokens, self.prompt_bytes)
    }

    /// The rows the host named, where they answer: beside an empty column.
    /// The column is asked first, so a host that fills one pays nothing for
    /// the rows it does not name.
    #[inline(always)]
    pub(crate) fn named_alone(&self) -> Option<AffinityRows<'a>> {
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

    /// The request's size in blocks: its token count over the block size
    /// (the text kind's bytes over the window's size in bytes), never under
    /// one and never under the block hashes the host sent. The host caps
    /// that list, so a long request's hashes alone would understate it and
    /// overstate every holder's share. The block size is the request's
    /// ([`Self::block_size_or`]), and the count is the request's size
    /// ([`Self::prompt_size`]): a request of text states its length and no
    /// count of tokens. The prefix stage's gates and the overlap a pick
    /// states both count here.
    pub(crate) fn blocks(&self, view_block: u32) -> u32 {
        let block = self.block_size_or(view_block);
        let from_tokens = blocks_within(self.prompt_size(), self.hash_kind, block).max(1);
        let from_hashes = u32::try_from(self.block_hashes.len()).unwrap_or(u32::MAX);
        from_tokens.max(from_hashes)
    }

    /// The size of one of the request's blocks: the request's own where its
    /// host counted with one (an index's learned size), else `view_block`,
    /// the view's.
    pub(crate) fn block_size_or(&self, view_block: u32) -> u32 {
        if self.block_size > 0 {
            self.block_size
        } else {
            view_block
        }
    }

    /// Whether the request's session key is pinned to the row at
    /// `position`, beside the row's own session bit.
    #[inline]
    pub fn pins(&self, position: u32) -> bool {
        self.pinned_at != 0 && self.pinned_at - 1 == position
    }

    /// The requests in flight of `row`: what the host lent for the row's
    /// position where it lent a table ([`RequestFacts::lent_in_flight`]),
    /// the row's own count otherwise. Every reader of a row's count asks
    /// here, and none learns which rows it was given. A decision reads one
    /// count a row: beside lent rows the first read is kept, and every
    /// later read returns it.
    #[inline]
    pub fn in_flight(&self, row: &Candidate) -> u32 {
        match self.lent_in_flight {
            Some(lent) => lent.at(row.position),
            None => row.in_flight,
        }
    }

    /// The affinity facts of the candidate at `index` of the slice the
    /// decision is made over. An empty column reads as nothing held for
    /// every candidate. Any other column holds the candidate's own entry,
    /// and the read is that entry: an index past its end is a host's
    /// mistake and fails as an index past the candidates does, never as a
    /// default in the entry's place. Where the host named rows in the
    /// column's place ([`AffinityRows`]), a named row reads what the host
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

    /// The key a ring places the request by: its session key's fingerprint,
    /// else its client's ([`RequestFacts::client_key`]), `0` where it has
    /// neither.
    #[must_use]
    pub fn ring_key(&self) -> u64 {
        match self.session_key.hash {
            0 => self.client_key,
            session => session,
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
///
/// The record gains fields: code outside this crate that builds one writes the
/// rest as `..Constraints::default()`, and a pattern of it ends with `..`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Constraints<'a> {
    /// Candidates this host just failed to reach.
    pub excluded: &'a [WorkerId],
    /// The wires this host speaks.
    pub wire_modes: WireModes,
    /// The overload veto does not apply to this decision: the host's
    /// override, for a pool whose policy balances load against affinity
    /// itself. An engine built from a document lifts the veto itself for a
    /// keyed request of a profile that places by the session key, where the
    /// binding steers; under a sessions rule with an in-flight cap, only
    /// where the host holds the key's counts
    /// ([`RequestFacts::session_counted`]). An engine built by hand does
    /// not: its host says so here.
    pub overload_lifted: bool,
    /// This pair's decode is sent after its prefill has run. Default: false.
    /// A door in process reads it. A link does not carry it yet: it sends
    /// the field as false, the controller issues both sides at the first
    /// ask, and a decode call over the link answers that the side was not
    /// chosen.
    pub decode_later: bool,
    /// A subset hint: decide over at most this many candidates.
    pub subset_hint: Option<u32>,
    /// What the request demands of the worker that takes it
    /// ([`Demands`](crate::engine::Demands)). A door keeps the demands before any
    /// stage runs, where it keeps its gateway's wires, and no stage reads
    /// them: an engine handed rows by a host of its own is handed rows
    /// that meet them. Over a pair's two worker lists each side is held
    /// to the workers that meet them as its role is held to them
    /// ([`Demands::held_by`](crate::engine::Demands::held_by)). An encode
    /// request's three lists keep none yet: such a request that states a
    /// demand is shed under the stage `wires`, whatever its lists state.
    /// In-process only: this protocol version
    /// states no demand, and a handle refuses a request that states one
    /// before it sends it over a link (`DecisionGuard::asked`).
    pub demands: crate::engine::Demands,
    /// The rows a standing veto of this host masks, by position, in order,
    /// where the host did not write them into the rows: a view of a
    /// publication's own rows hands them here, and a reader of a row's
    /// stalled bit asks both ([`Constraints::masks`]). Empty where the rows
    /// were written for the decision. In-process only.
    ///
    /// They are positions of the one list a set is opened over, as
    /// [`Self::places`] are. A pair's host states each side's beside its
    /// two lists, with the rows a hold stands on. A pair's request that
    /// states either of its own, and an encode request's, is declined by
    /// the engine over worker lists, by name, as one that states places
    /// of its own is.
    pub stalled: &'a [u32],
    /// The rows of [`Self::stalled`] a hold of this host stands on, in
    /// order, each once: a veto no rule lifts, as an open breaker's is. The
    /// all-vetoed rule lifts the mask as its last resort and leaves these
    /// rows masked. Empty where the rows were written for the decision:
    /// there a held row carries the breaker bit beside the stalled one.
    /// In-process only.
    pub held: &'a [u32],
    /// The places of the pool's list the request may take, ascending, each
    /// once, where its host holds it to a part of the pool: the workers a
    /// retry may still take, the worker a request names, the workers that
    /// meet its route's rule. `None`: every place.
    ///
    /// The request's own fact and no veto of a worker: no stage reads it,
    /// and a profile that lists no filter is held to it like any other. A
    /// decision's staged pass opens its set with these places and hands no
    /// target outside them. A pass that draws from the pool leaves such a
    /// request to it, and so does a keyed request's read of a row by its
    /// place. They are places of the one list a set is opened over: a
    /// single request's, or one side's of a pair, whose entry takes one
    /// set of constraints a side. A pair's host states each side's places
    /// beside its two lists. A pair's request that states places of its
    /// own is declined by the engine over worker lists, by name, and the
    /// served controller states none.
    ///
    /// A request held to places whose every worker it already tried is
    /// handed one of them again, not refused: the filter of tried
    /// workers is soft, so the answer is the one a list of those workers
    /// alone gives. In-process only.
    pub places: Option<&'a [u32]>,
}

impl Constraints<'_> {
    /// Whether a standing veto of this host masks the row at `position`,
    /// beside the row's own stalled bit. A decision under no such veto pays
    /// the first test alone.
    #[inline]
    pub fn masks(&self, position: u32) -> bool {
        !self.stalled.is_empty() && self.stalled.binary_search(&position).is_ok()
    }
}

/// Healthy capacity of one zone, by the zone's position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZoneCapacity {
    pub zone: u16,
    pub healthy: u32,
    pub total: u32,
}

/// The engines of a pool, as its host lists them beside the rows: each
/// engine's rows by position, the ranks of a data-parallel engine together
/// and in their ranks' order, a single-rank worker alone. A stage that
/// names the engine as its unit ([`Picker::unit`](crate::engine::Picker::unit))
/// draws over the list and reads the rows of the engine it takes, so a
/// decision it makes does not grow with the ranks of the engines it passes
/// by. Empty where the host lists none: a pool with no engine of several
/// rows, or rows whose positions are not the list's. In-process only.
///
/// A host that lists a pool's engines hands the pool's own rows with the
/// list, in the order of their ids and under a membership epoch
/// ([`PoolFacts::membership_epoch`]), and numbers the pool
/// ([`Engines::pool`]). A stage that finds a member's row by its id, with
/// no pool read whole ([`Scorer::own`](crate::engine::Scorer::own)), takes the
/// list as that word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Engines<'a> {
    pool: u32,
    rows: &'a [u32],
    ends: &'a [u32],
}

impl<'a> Engines<'a> {
    /// The list of the pool its host numbers `pool`, over `rows`, the
    /// positions engine after engine, and `ends`, each engine's end in
    /// `rows` in rising order. A list whose last end is not the count of
    /// its rows is no list, and reads as empty.
    pub fn new(pool: u32, rows: &'a [u32], ends: &'a [u32]) -> Self {
        if ends.last().is_some_and(|&end| end as usize == rows.len()) {
            Self { pool, rows, ends }
        } else {
            Self::default()
        }
    }

    /// The host's number for the pool: one pool's while a membership epoch
    /// stands, and no other pool's under that epoch. A stage that keeps
    /// what it learned of a pool keeps it by this number and the epoch.
    pub const fn pool(&self) -> u32 {
        self.pool
    }

    /// How many engines the pool holds.
    pub const fn len(&self) -> usize {
        self.ends.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// The rows of the engine at `engine`, by position; none for a place
    /// past the list.
    #[inline]
    pub fn rows(&self, engine: usize) -> &'a [u32] {
        let from = match engine.checked_sub(1) {
            Some(before) => self.ends.get(before).copied(),
            None => Some(0),
        };
        let to = self.ends.get(engine).copied();
        match (from, to) {
            (Some(from), Some(to)) => self
                .rows
                .get(from as usize..to as usize)
                .unwrap_or_default(),
            _ => &[],
        }
    }
}

/// Pool-wide facts beside the candidates.
///
/// The record gains fields: code outside this crate that builds one writes the
/// rest as `..PoolFacts::default()`, and a pattern of it ends with `..`.
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
    /// Which of the pool's lists the candidates are, where the host hands
    /// more than one of a pool under one membership epoch: two that differ
    /// by a row, or by a bit the host writes on a row for one asker alone,
    /// carry different values, and a host that hands one leaves `0`. A
    /// stage that keeps state by the pool and reads no row for it, as a
    /// rotation does, keys it by this too. In-process only: a plugin reads
    /// the rows it is handed.
    pub reach: u64,
    /// Whether every row of the pool holds a report. A host says so where
    /// it knows it of the rows it wrote, and a decision reads it once: a
    /// profile decided by rank ([`Scorer::spread`](crate::engine::Scorer::spread))
    /// is asked by rank only where it is on, since the load stage costs a
    /// row without a report from every row that holds one. By rank the
    /// targets are the whole pass's wherever its sums tell the rows in
    /// front apart: rows it rounds to one sum it draws among, and by rank
    /// they are ordered. Off by default: a host that does not say it gets
    /// the whole pass. A report without a rate is a report: among the rows
    /// in front it drains as in the whole pass, at the mean rate of the
    /// rows that pass costs, which the load stage folds over them, and the
    /// decision is the one by rank. A host that says the fact of a pool
    /// that holds a row without a report is answered by what the row
    /// lacks: in front it makes the stage say that it must be asked whole
    /// ([`ScoreFront::Whole`](crate::engine::ScoreFront::Whole)), and the
    /// decision is the whole pass's, which paid for its ranks' reading. Behind
    /// the rows in front it is not costed where they decide alone, and the
    /// decision is the one by rank. Where they cannot, their band being wider
    /// than the ranks leave room for, every row is asked, that row with them,
    /// and the decision is the whole pass's. In-process only.
    pub reported: bool,
    /// The pool's engines, where the host lists them ([`Engines`]): the
    /// candidates are then the rows the list's positions name. Empty by
    /// default, and every decision then reads rows alone.
    pub engines: Engines<'a>,
    /// The pool's own word, where the host keeps one beside the list of
    /// workers the candidates are and lends it to the decision: the word a
    /// pick over that list alone counts its turns in
    /// ([`PoolWord::word`](crate::engine::PoolWord::word)). A picker that
    /// takes turns counts a draw's turn in it too, and looks no rotation up
    /// by the pool's name. That pick and that draw over one list then turn
    /// one rotation, and two lists turn two, whatever else they share. `None`:
    /// such a picker finds its rotation by the pool's name, with
    /// [`Self::reach`] in it. In-process only.
    pub word: Option<&'a AtomicU64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::in_flight::tests::Lent;

    /// One reader, two feeds: a row's requests in flight are the lent count
    /// at the row's position where the host lends a table, whatever the
    /// row's own field holds, and the row's own field where it lends none.
    #[test]
    fn a_rows_requests_in_flight_are_the_lent_count_or_the_rows_own() {
        let row = |position, in_flight| Candidate {
            position,
            in_flight,
            ..Candidate::default()
        };
        let mut rows = [row(0, 5), row(1, 2)];
        let own = RequestFacts::default();
        let read = |facts: &RequestFacts<'_>, rows: &[Candidate]| -> Vec<u32> {
            rows.iter().map(|row| facts.in_flight(row)).collect()
        };
        assert_eq!(read(&own, &rows), [5, 2]);
        let lent = Lent::take(&mut rows);
        assert_eq!(read(&own, &rows), [0, 0]);
        // A count a host left on a row does not speak over its table.
        rows[0].in_flight = 9;
        assert_eq!(read(&lent.beside(&own), &rows), [5, 2]);
    }

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
    #[expect(clippy::print_stdout, reason = "the case reports the candidate layout")]
    fn a_candidate_row_is_two_lines_and_its_affinity_facts_20_bytes() {
        // The filters walk the rows once a filter, so the rows' stride is a
        // cost every policy pays: the pin is exact, 128 bytes at a 64-byte
        // alignment, and a change of any number here carries the crate's
        // decision bench (`benches/decision.rs`) where the pull-request gate
        // runs as well as on a build host: the two read a narrower stride in
        // opposite directions.
        let size = size_of::<Candidate>();
        let alignment = align_of::<Candidate>();
        println!("Candidate size {size} bytes, alignment {alignment} bytes");
        assert_eq!((size, alignment), (128, 64));
        assert_eq!(size_of::<AffinityFacts>(), 20);
    }

    /// A record made ready is one the health and admission filters keep
    /// and an engine of its own, and every field it does not write is the
    /// default's. A default record is dropped by three of those filters.
    #[test]
    fn a_ready_record_can_be_sent_to_and_is_an_engine_of_its_own() {
        use crate::{
            engine::framework::Filter,
            stages::{HasCapacity, HasRoom, Healthy, NotOverloaded},
        };

        let made = Candidate::ready(3, WorkerId(9), Role::Decode);
        assert_eq!(
            (made.position, made.id, made.role, made.engine),
            (3, WorkerId(9), Role::Decode, 3)
        );
        let (facts, cx) = (RequestFacts::default(), Constraints::default());
        let stock: [&dyn Filter; 4] = [&Healthy, &NotOverloaded, &HasCapacity, &HasRoom];
        let dropped = |record: &Candidate| {
            let refuses = stock
                .iter()
                .filter(|stage| !stage.keep(&facts, record, &cx));
            refuses.map(|stage| stage.name()).collect::<Vec<_>>()
        };
        assert!(dropped(&made).is_empty(), "{:?}", dropped(&made));
        let unset = Candidate::default();
        assert_eq!(dropped(&unset), ["healthy", "has-capacity", "has-room"]);

        let rest = Candidate {
            position: 0,
            id: WorkerId::default(),
            role: Role::default(),
            health: Health::default(),
            has_capacity: false,
            has_room: false,
            engine: 0,
            ..made
        };
        assert_eq!(format!("{rest:?}"), format!("{unset:?}"));
    }

    /// A request's facts are copied where a decision narrows or resumes
    /// them, so their size is a cost of those decisions: the pin is exact.
    /// The signal cells are one word, for every signal, and the client's
    /// key is one word more. The ring a caller lends is two words, and
    /// where the request stands on it two more. A text's two lengths are
    /// one word: its characters fill what was spare, and its bytes take
    /// the word. The word that says the request's own id handed its
    /// session key is one byte of what is spare. A field the facts gain
    /// beyond the spare bytes moves this number and carries the decision
    /// bench's cells that copy the facts (`benches/decision.rs`).
    #[test]
    fn a_requests_facts_take_the_pinned_bytes() {
        assert_eq!(size_of::<RequestFacts<'_>>(), 320);
    }

    #[test]
    fn a_candidate_rows_fields_take_107_bytes() {
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
            engine,
        );
        assert_eq!(ends.into_iter().max(), Some(107));
    }

    /// A request's size in blocks is its tokens over its block size: the size
    /// its host counted with where the request states one, the view's where
    /// it states none. It is never under one, and never under the block
    /// hashes the host sent. A request of text states its length and no
    /// count of tokens: its windows are bytes.
    #[test]
    fn a_requests_blocks_are_its_tokens_over_its_block_size_and_never_under_its_hashes() {
        let hashes = [1u64, 2, 3];
        let sized = |prompt_tokens: u32, block_size: u32, hashed: usize| RequestFacts {
            prompt_tokens,
            block_size,
            block_hashes: &hashes[..hashed],
            ..RequestFacts::default()
        };
        // 100 tokens at the view's 16 are six whole blocks.
        assert_eq!(sized(100, 0, 0).blocks(16), 6);
        assert_eq!(sized(100, 0, 0).block_size_or(16), 16);
        // At the size its host counted with, the view's is not read.
        assert_eq!(sized(100, 32, 0).blocks(16), 3);
        assert_eq!(sized(100, 32, 0).block_size_or(16), 32);
        // Under one block, and with no token counted: one block.
        assert_eq!(sized(5, 0, 0).blocks(16), 1);
        assert_eq!(sized(0, 0, 0).blocks(16), 1);
        // Never under the hashes the host sent, and over them by its tokens.
        assert_eq!(sized(20, 0, 3).blocks(16), 3);
        assert_eq!(sized(100, 0, 3).blocks(16), 6);
        // A text of 200 two-byte characters is 400 bytes: six windows of 64.
        // Read by its characters it would be three.
        let text = RequestFacts {
            hash_kind: HashKind::Text,
            prompt_chars: 200,
            prompt_bytes: 400,
            ..sized(0, 64, 0)
        };
        assert_eq!(text.blocks(16), 6);
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
            |(row, overlap_blocks, predicted_blocks)| AffinityHolder {
                row,
                overlap_blocks,
                predicted_blocks,
            },
        );
        let named = AffinityRows {
            holders: &holders,
            ..AffinityRows::default()
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
        let mut walk = AffinityRows::default().walk();
        assert_eq!(walk.entry(0), AffinityFacts::default());
        assert_eq!(walk.entry(3), AffinityFacts::default());
    }

    /// Behind a draw a pass reads the rows named by a search: each row reads
    /// as the walk of a whole read reads it, and no read looks behind a
    /// cursor, where the walk asked in that order does.
    #[test]
    fn a_read_behind_a_draw_searches_the_rows_named_and_keeps_no_cursor() {
        let holders = [(1, 4), (4, 9), (9, 3)].map(|(row, overlap_blocks)| AffinityHolder {
            row,
            overlap_blocks,
            predicted_blocks: 0,
        });
        let whole = AffinityRows {
            holders: &holders,
            ..AffinityRows::default()
        };
        let drawn = AffinityRows {
            mean_in_flight: Some(1.0),
            ..whole
        };
        assert!(drawn.drawn() && !whole.drawn());
        let feed = Feed::Named(drawn);
        let order = [9, 0, 4, 4, 1, 10, 9];
        LOOKS_BEHIND.with(|looks| looks.set(0));
        let mut reads = feed.reads();
        for row in order {
            assert_eq!(
                feed.read::<true>(&mut reads, row),
                whole.entry(row),
                "row {row}"
            );
        }
        assert_eq!(
            LOOKS_BEHIND.with(std::cell::Cell::get),
            0,
            "a search looked behind"
        );
        for row in order {
            assert_eq!(
                feed.read::<false>(&mut reads, row),
                whole.entry(row),
                "row {row}"
            );
        }
        assert!(
            LOOKS_BEHIND.with(std::cell::Cell::get) > 0,
            "the walk never looked behind"
        );
    }

    /// Rows named in the column's place read through the same accessor: a
    /// named row reads what the host said it holds, by both readings, and
    /// every other row nothing held. Beside a column the column is read.
    #[test]
    fn rows_named_in_the_columns_place_read_what_each_holds() {
        let holders = [
            AffinityHolder {
                row: 1,
                overlap_blocks: 9,
                predicted_blocks: 0,
            },
            AffinityHolder {
                row: 4,
                overlap_blocks: 3,
                predicted_blocks: 7,
            },
        ];
        let facts = RequestFacts {
            named: Some(AffinityRows {
                holders: &holders,
                ..AffinityRows::default()
            }),
            ..RequestFacts::default()
        };
        let read = [0, 1, 2, 4, 511].map(|index| {
            let held = facts.affinity_at(index);
            (held.overlap_blocks, held.predicted_blocks)
        });
        assert_eq!(read, [(0, 0), (9, 0), (0, 0), (3, 7), (0, 0)]);
        let nobody = RequestFacts {
            named: Some(AffinityRows::default()),
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

    /// The formula runs backwards: a session's value gives its key's
    /// fingerprint within the same model and phase, on the contract's own
    /// numbers and across keys, models and phases.
    #[test]
    fn a_sessions_value_gives_its_keys_fingerprint_back_within_its_model_and_phase() {
        let phases = [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode];
        let contract: [u64; 4] = [
            0x6e88_ff71_cc98_2a42,
            0x619f_dbb0_c262_9a55,
            0x3e92_fbbd_183d_f4b6,
            0x3780_ba54_336a_57d0,
        ];
        let model = fingerprint(b"model-a");
        for (phase, session) in phases.into_iter().zip(contract) {
            let key = key_of_session(session, model, phase);
            assert_eq!(key, 0xd449_9ecc_c740_2a12, "{phase:?}");
        }
        for (index, phase) in (0..4_000_u32).zip(phases.into_iter().cycle()) {
            let name = format!("model-{}", index % 7);
            let key = fingerprint(format!("key-{index}").as_bytes());
            let session = RequestFacts {
                model: &name,
                session_key: Identity::hashed(key),
                phase,
                ..RequestFacts::default()
            }
            .session()
            .get();
            let read = key_of_session(session, fingerprint(name.as_bytes()), phase);
            assert_eq!(read, key, "key-{index} under {name}, {phase:?}");
        }
        // No session has no key, and neither has the one value a model and
        // a phase give the fingerprint no key has.
        assert_eq!(key_of_session(0, model, Phase::Single), 0);
        assert_eq!(
            key_of_session(0xc771_96d2_b9cf_9497, model, Phase::Single),
            0
        );
        // The value the formula lifts a zero result to reads as the key it
        // sends there by itself, whose session it is too.
        let sent = key_of_session(1, model, Phase::Single);
        let lifted = RequestFacts {
            model: "model-a",
            session_key: Identity::hashed(sent),
            ..RequestFacts::default()
        };
        assert_eq!((sent, lifted.session().get()), (0xbddd_7c85_982a_9f6f, 1));
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

    /// A session key's count stands on a worker's object under one of two
    /// names. Under the key's text the name is the text the request
    /// carries, and a key carried as its fingerprint alone has none. Under
    /// the session's number it is the number's sixteen digits: one name for
    /// the key in either form, another for another model. A request with
    /// no key has no name under either.
    #[test]
    fn a_sessions_count_is_named_by_where_it_stands() {
        let plain = RequestFacts {
            model: "model-a",
            session_key: Identity::plain("conversation-7"),
            ..RequestFacts::default()
        };
        let hashed = RequestFacts {
            session_key: Identity::hashed(plain.session_key.hash),
            ..plain
        };
        let named = |under: SessionCountedUnder, facts: &RequestFacts<'_>| {
            let mut buffer = [0; Session::TEXT_LEN];
            under.name(facts, &mut buffer).map(str::to_owned)
        };
        let text = SessionCountedUnder::KeyText;
        let number = SessionCountedUnder::SessionNumber;
        assert_eq!(SessionCountedUnder::default(), text);
        assert_eq!(named(text, &plain).as_deref(), Some("conversation-7"));
        assert_eq!(named(text, &hashed), None);
        let digits = format!("{:016x}", plain.session().get());
        assert_eq!(named(number, &plain).as_deref(), Some(digits.as_str()));
        assert_eq!(named(number, &hashed).as_deref(), Some(digits.as_str()));
        let other = RequestFacts {
            model: "model-b",
            ..plain
        };
        assert_ne!(named(number, &other), Some(digits));
        let keyless = RequestFacts::default();
        assert_eq!(named(text, &keyless), None);
        assert_eq!(named(number, &keyless), None);
        // The number's text: every digit, the highest first.
        let text_of = |value| {
            let mut buffer = [0; Session::TEXT_LEN];
            Session::from_raw(value).text(&mut buffer).to_owned()
        };
        assert_eq!(text_of(0x1f), "000000000000001f");
        assert_eq!(text_of(u64::MAX), "ffffffffffffffff");
        assert_eq!(text_of(0x0123_4567_89ab_cdef), "0123456789abcdef");
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
    fn no_media_is_the_default_of_the_counts() {
        assert_eq!(MediaCounts::NONE, MediaCounts::default());
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
