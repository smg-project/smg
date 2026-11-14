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
    /// Context tokens resident for the running requests.
    pub resident_tokens: u32,
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

    /// A key within a scope (a model and a phase, fingerprinted by the host):
    /// the fingerprint mixes the scope in, so one key names one session per
    /// scope, and the plain form is kept. Never `0` for a present key.
    pub fn scoped(key: &'a str, scope: u64) -> Self {
        let mut h = fingerprint(key.as_bytes()) ^ scope.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        Self {
            hash: if h == 0 { 1 } else { h },
            plain: Some(key),
        }
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

/// One candidate row: a worker, or one DP rank of a data-parallel worker:
/// what every decision reads. 104 bytes in memory; the wire form drops the
/// position and the sample instant, carries the candidate's
/// [`AffinityFacts`] in the same record and encodes to about 76 bytes with
/// its load row.
#[derive(Clone, Copy, Debug, Default)]
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
    /// The pairing class: a decode may pair with a prefill of the same class;
    /// `0` for a worker outside any pairing. Until a store derives classes, a
    /// host keys it by the open prefill a decode partners with.
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
    /// The routing key as the host derives it, lineage-stripped.
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
    /// the candidates' own order. It is EMPTY (no stage of the decision's
    /// profile declared [`Needs::affinity`](crate::Needs::affinity)) or
    /// holds exactly one entry a candidate; the engine refuses any other
    /// before a stage runs. A pair decision takes each side's column from
    /// its pool ([`PairPools`](crate::PairPools)) and none from here: a
    /// non-empty column on the request of a pair decision is refused, as a
    /// column of another length is. The host keeps it readable, as it keeps
    /// the candidates, until the decision is committed. In-process only: a
    /// wire carries each candidate's facts in its record.
    pub affinity: &'a [AffinityFacts],
}

impl RequestFacts<'_> {
    /// The affinity facts of the candidate at `index` of the slice the
    /// decision is made over. An empty column reads as nothing held for
    /// every candidate. Any other column holds the candidate's own entry,
    /// and the read is that entry: an index past its end is a host's
    /// mistake and fails as an index past the candidates does, never as a
    /// default in the entry's place. Each read tests the column's emptiness
    /// and its bound: a pass over a pool takes the column once
    /// ([`RequestFacts::affinity`]) and walks it beside the rows, as this
    /// crate's stages and its packed plugin fill do.
    pub fn affinity_at(&self, index: u32) -> AffinityFacts {
        if self.affinity.is_empty() {
            AffinityFacts::default()
        } else {
            self.affinity[index as usize]
        }
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
    fn a_candidate_row_is_104_bytes_and_its_affinity_facts_20() {
        // The row is written once per candidate per decision, so its size is
        // a cost every policy pays: the pin is exact, and a change of either
        // number carries the crate's decision bench (`benches/decision.rs`).
        // A fact one stage family reads belongs in a column beside the rows,
        // as the affinity facts are. The plugin view's packed row is 104
        // bytes too, WITH the five facts in it: nothing ties that number to
        // this one, and each is pinned in its own file.
        assert_eq!(size_of::<Candidate>(), 104);
        assert_eq!(size_of::<AffinityFacts>(), 20);
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
