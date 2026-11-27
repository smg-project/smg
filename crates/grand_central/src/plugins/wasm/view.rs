//! The view and the lifecycle as a guest sees them: the record form through
//! the generated types, the packed form as the little-endian blocks the
//! WIT's layout table names, and the lifecycle as the versioned variant. The
//! exposure decides what of the identities and headers crosses, and the
//! plugin's declared needs whether the request's block hashes do.

use super::{
    bindings::wit,
    config::{Exposure, Reveal},
};
use crate::{
    document::spec::Needs,
    engine::{
        decision::{Decision, Target, MAX_TARGETS},
        view::{
            AffinityFacts, Candidate, Feed, HashKind, Health, Identity, LoadRow, OverloadRegime,
            Phase, PoolFacts, RequestFacts, Role, SignalSource, WireModes, ZoneCapacity,
        },
    },
    plugins::shared::{shown_hashes, shown_size},
    tracking::lease::{
        Lease, LeaseKind, Membership, Outcome, ReleaseReason, Side, Signal, TransportClass,
    },
};

/// One health flag of a candidate: whether it is set, its bit in the packed
/// row, its `types.health` flag.
struct HealthFlag {
    on: bool,
    bit: u8,
    flag: wit::Health,
}

/// The health numbering, one table for the `types.health` flags and the
/// packed row's byte: each flag beside its bit, read by both forms. Built
/// from the fields per call, so it folds to bit tests.
fn health_flags(h: Health) -> [HealthFlag; 5] {
    [
        HealthFlag {
            on: h.healthy,
            bit: 1,
            flag: wit::Health::HEALTHY,
        },
        HealthFlag {
            on: h.breaker_open,
            bit: 1 << 1,
            flag: wit::Health::BREAKER_OPEN,
        },
        HealthFlag {
            on: h.stalled,
            bit: 1 << 2,
            flag: wit::Health::STALLED,
        },
        HealthFlag {
            on: h.draining,
            bit: 1 << 3,
            flag: wit::Health::DRAINING,
        },
        HealthFlag {
            on: h.overloaded,
            bit: 1 << 4,
            flag: wit::Health::OVERLOADED,
        },
    ]
}

pub(crate) fn health(h: Health) -> wit::Health {
    health_flags(h)
        .into_iter()
        .filter(|f| f.on)
        .fold(wit::Health::empty(), |acc, f| acc | f.flag)
}

/// The health bits of the packed row: the same table as the flags.
fn health_bits(h: Health) -> u8 {
    health_flags(h)
        .into_iter()
        .filter(|f| f.on)
        .fold(0, |acc, f| acc | f.bit)
}

pub(crate) fn wire_modes(w: WireModes) -> wit::WireModes {
    let mut flags = wit::WireModes::empty();
    if w.contains(WireModes::HTTP) {
        flags |= wit::WireModes::HTTP;
    }
    if w.contains(WireModes::GRPC) {
        flags |= wit::WireModes::GRPC;
    }
    if w.contains(WireModes::ZMQ) {
        flags |= wit::WireModes::ZMQ;
    }
    flags
}

pub(crate) fn role(r: Role) -> wit::Role {
    match r {
        Role::Single => wit::Role::Single,
        Role::Prefill => wit::Role::Prefill,
        Role::Decode => wit::Role::Decode,
        Role::Encode => wit::Role::Encode,
    }
}

pub(crate) fn phase(p: Phase) -> wit::Phase {
    match p {
        Phase::Single => wit::Phase::Single,
        Phase::Prefill => wit::Phase::Prefill,
        Phase::Decode => wit::Phase::Decode,
        Phase::Encode => wit::Phase::Encode,
    }
}

pub(crate) fn side(s: Side) -> wit::Side {
    match s {
        Side::Single => wit::Side::Single,
        Side::Prefill => wit::Side::Prefill,
        Side::Decode => wit::Side::Decode,
        Side::Encode => wit::Side::Encode,
    }
}

pub(crate) fn hash_kind(k: HashKind) -> wit::HashKind {
    match k {
        HashKind::Token => wit::HashKind::Token,
        HashKind::Text => wit::HashKind::Text,
    }
}

pub(crate) fn source(s: SignalSource) -> wit::SignalSource {
    match s {
        SignalSource::None => wit::SignalSource::None,
        SignalSource::Events => wit::SignalSource::Events,
        SignalSource::Scrape => wit::SignalSource::Scrape,
        SignalSource::Relay => wit::SignalSource::Relay,
        SignalSource::Poll => wit::SignalSource::Poll,
    }
}

pub(crate) fn regime(r: OverloadRegime) -> wit::OverloadRegime {
    match r {
        OverloadRegime::None => wit::OverloadRegime::None,
        OverloadRegime::Partial => wit::OverloadRegime::Partial,
        OverloadRegime::Saturated => wit::OverloadRegime::Saturated,
    }
}

pub(crate) fn load(row: &LoadRow) -> wit::Load {
    wit::Load {
        running: row.running,
        queued: row.queued,
        waiting_uncached_tokens: row.waiting_uncached_tokens,
        kv_usage_permille: row.kv_usage_permille,
        active_kv_permille: row.active_kv_permille,
        live_rate: row.live_rate,
        resident_tokens: row.resident_tokens,
        report_age_ms: row.report_age_ms,
        source: source(row.source),
        max_running: row.max_running,
    }
}

/// One candidate as the record form carries it: the row, and the affinity
/// facts the host holds for it beside the row (`held`), which the record
/// keeps as fields of its own. Its requests in flight are what the
/// request's facts read for it (`in_flight`).
pub(crate) fn candidate(c: &Candidate, held: AffinityFacts, in_flight: u32) -> wit::Candidate {
    wit::Candidate {
        id: c.id.0,
        role: role(c.role),
        dp_rank: c.dp_rank,
        dp_size: c.dp_size,
        health: health(c.health),
        wire_modes: wire_modes(c.wire_modes),
        engine_kind: c.engine_kind,
        connector: c.connector,
        in_flight,
        load: c.load.as_ref().map(load),
        overlap_blocks: held.overlap_blocks,
        predicted_blocks: held.predicted_blocks,
        session_hit: c.session_hit,
        session_in_flight: c.session_in_flight,
        pair_class: c.pair_class,
        has_capacity: c.has_capacity,
        has_room: c.has_room,
        admitted_age_ms: held.admitted_age_ms,
        indexed_growth: held.indexed_growth,
        zone: c.zone,
        labels: c.labels,
        ec_compatible: c.ec_compatible,
        vision: c.vision,
        host_local: c.host_local,
        active_sessions: c.active_sessions,
        key: c.key,
        indexed_blocks: held.indexed_blocks,
    }
}

/// An identity under the plugin's exposure: withheld, its fingerprint, or
/// the plain form when the host holds it (its fingerprint otherwise).
pub(crate) fn identity<'a>(id: &Identity<'a>, reveal: Reveal) -> wit::Identity<'a> {
    exposed(id, reveal, || id.hash)
}

/// [`identity`] with the hash form's value made on demand: the exposure is
/// asked first, and `hash` runs only where that form crosses.
fn exposed<'a>(id: &Identity<'a>, reveal: Reveal, hash: impl FnOnce() -> u64) -> wit::Identity<'a> {
    if id.is_none() {
        return wit::Identity::None;
    }
    match (reveal, id.plain) {
        (Reveal::None, _) => wit::Identity::None,
        (Reveal::Plain, Some(plain)) => wit::Identity::Plain(plain),
        (Reveal::Hash | Reveal::Plain, _) => wit::Identity::Hash(hash()),
    }
}

/// The request's session key as a plugin sees it under its exposure: the
/// session's value in the hash form, which is what a lease carries, so a
/// plugin matches the two; the key itself in the plain form. A value the
/// plugin is not shown is never made.
fn session_identity<'a>(facts: &RequestFacts<'a>, reveal: Reveal) -> wit::Identity<'a> {
    exposed(&facts.session_key, reveal, || facts.session().get())
}

/// The allow-listed headers the exposure lets through, as the record form
/// lists them. Allocates only when the request carries an exposed header.
pub(crate) fn headers<'a>(facts: &RequestFacts<'a>, exposure: &Exposure) -> Vec<(u8, &'a str)> {
    facts
        .headers
        .iter()
        .filter(|h| exposure.headers.contains(h.name))
        .map(|h| (h.name, h.value))
        .collect()
}

/// The request under the plugin's exposure and its declared needs.
pub(crate) fn request<'a>(
    facts: &RequestFacts<'a>,
    exposure: &Exposure,
    needs: Needs,
    tried: &'a [u64],
    headers: &'a [(u8, &'a str)],
) -> wit::Request<'a> {
    wit::Request {
        model: facts.model,
        namespace_seed: facts.namespace_seed,
        tenant: identity(&facts.tenant, exposure.tenant),
        priority: facts.priority,
        streaming: facts.streaming,
        max_output_tokens: facts.max_output_tokens,
        prompt_tokens: shown_size(facts, needs),
        block_hashes: shown_hashes(facts, needs),
        hash_kind: hash_kind(facts.hash_kind),
        session_key: session_identity(facts, exposure.session_key),
        phase: phase(facts.phase),
        attempt: facts.attempt,
        tried,
        budget_micros: facts.budget_micros,
        headers,
        media: wit::Media {
            images: facts.media.images,
            audio: facts.media.audio,
            video: facts.media.video,
            bytes: facts.media.bytes,
        },
        seed: facts.seed,
        block_size: facts.block_size,
    }
}

pub(crate) fn zone(z: &ZoneCapacity) -> wit::ZoneCapacity {
    wit::ZoneCapacity {
        zone: z.zone,
        healthy: z.healthy,
        total: z.total,
    }
}

pub(crate) fn pool<'a>(pool: &PoolFacts<'_>, zones: &'a [wit::ZoneCapacity]) -> wit::Pool<'a> {
    wit::Pool {
        pool_size: pool.pool_size,
        membership_epoch: pool.membership_epoch,
        ring_epoch: pool.ring_epoch,
        regime: regime(pool.regime),
        generation: pool.generation,
        zones,
    }
}

/// The record form's reused buffers: the candidate rows, the zones and the
/// tried ids of one decision, filled in place.
#[derive(Debug, Default)]
pub(crate) struct Records {
    pub(crate) candidates: Vec<wit::Candidate>,
    pub(crate) zones: Vec<wit::ZoneCapacity>,
    pub(crate) tried: Vec<u64>,
}

impl Records {
    pub(crate) fn fill(
        &mut self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        pool: &PoolFacts<'_>,
    ) {
        self.candidates.clear();
        // A candidate's affinity facts ride the request's column, or the
        // rows the host names in the column's place, by its index: the same
        // index reads the row and what it holds. The packed fill walks
        // either feed instead; this one keeps the read by index, whose
        // tests are lost in what a record a candidate costs.
        let records = (0u32..)
            .zip(candidates)
            .map(|(index, c)| candidate(c, facts.affinity_at(index), facts.in_flight(c)));
        self.candidates.extend(records);
        self.zones.clear();
        self.zones.extend(pool.zones.iter().map(zone));
        self.tried.clear();
        self.tried.extend(facts.tried.iter().map(|w| w.0));
    }
}

/// The packed form's layout: the committed WIT's table, byte for byte. A
/// change to the table changes these numbers and the test that pins them.
pub mod layout {
    /// One candidate row, with the five affinity facts in it. The fields of
    /// the engine's own row take 107 bytes WITHOUT them: the two numbers
    /// are near by accident, each pinned in its own file, and nothing ties
    /// one to the other.
    pub const ROW: usize = 104;
    /// The request block's fixed header; the variable sections follow.
    pub const HEADER: usize = 80;
    /// The pool block before its zone rows.
    pub const POOL: usize = 40;
    /// One zone row.
    pub const ZONE: usize = 12;

    pub const ROW_ID: usize = 0;
    pub const ROW_IN_FLIGHT: usize = 8;
    pub const ROW_RUNNING: usize = 12;
    pub const ROW_QUEUED: usize = 16;
    pub const ROW_WAITING_UNCACHED: usize = 20;
    pub const ROW_OVERLAP: usize = 24;
    pub const ROW_PREDICTED: usize = 28;
    pub const ROW_RESIDENT: usize = 32;
    pub const ROW_PAIR_CLASS: usize = 36;
    pub const ROW_ADMITTED_AGE: usize = 40;
    pub const ROW_REPORT_AGE: usize = 44;
    pub const ROW_INDEXED_GROWTH: usize = 48;
    pub const ROW_LIVE_RATE: usize = 52;
    pub const ROW_MAX_RUNNING: usize = 56;
    pub const ROW_KV_USAGE: usize = 60;
    pub const ROW_ACTIVE_KV: usize = 62;
    pub const ROW_DP_RANK: usize = 64;
    pub const ROW_DP_SIZE: usize = 66;
    pub const ROW_ZONE: usize = 68;
    pub const ROW_ENGINE_KIND: usize = 70;
    pub const ROW_CONNECTOR: usize = 72;
    pub const ROW_SESSION_IN_FLIGHT: usize = 74;
    pub const ROW_HEALTH: usize = 76;
    pub const ROW_WIRE_MODES: usize = 77;
    pub const ROW_ROLE: usize = 78;
    pub const ROW_SOURCE: usize = 79;
    pub const ROW_BITS: usize = 80;
    pub const ROW_ACTIVE_SESSIONS: usize = 82;
    pub const ROW_KEY: usize = 84;
    pub const ROW_LABELS: usize = 88;
    pub const ROW_INDEXED_BLOCKS: usize = 96;

    pub const BIT_SESSION_HIT: u8 = 1;
    pub const BIT_HAS_CAPACITY: u8 = 1 << 1;
    pub const BIT_HAS_ROOM: u8 = 1 << 2;
    pub const BIT_EC_COMPATIBLE: u8 = 1 << 3;
    pub const BIT_VISION: u8 = 1 << 4;
    pub const BIT_HOST_LOCAL: u8 = 1 << 5;
    pub const BIT_LOAD_PRESENT: u8 = 1 << 6;

    pub const REQ_NAMESPACE_SEED: usize = 0;
    pub const REQ_SEED: usize = 8;
    pub const REQ_TENANT: usize = 16;
    pub const REQ_SESSION_KEY: usize = 24;
    pub const REQ_BUDGET: usize = 32;
    pub const REQ_PROMPT_TOKENS: usize = 36;
    pub const REQ_MAX_OUTPUT: usize = 40;
    pub const REQ_PRIORITY: usize = 44;
    pub const REQ_HASH_COUNT: usize = 48;
    pub const REQ_TRIED_COUNT: usize = 50;
    pub const REQ_HEADER_COUNT: usize = 52;
    pub const REQ_MODEL_LEN: usize = 54;
    pub const REQ_PHASE: usize = 56;
    pub const REQ_HASH_KIND: usize = 57;
    pub const REQ_ATTEMPT: usize = 58;
    pub const REQ_BITS: usize = 59;
    pub const REQ_MEDIA_IMAGES: usize = 60;
    pub const REQ_MEDIA_AUDIO: usize = 62;
    pub const REQ_MEDIA_VIDEO: usize = 64;
    pub const REQ_MEDIA_BYTES: usize = 68;
    pub const REQ_BLOCK_SIZE: usize = 72;

    pub const REQ_BIT_STREAMING: u8 = 1;
    pub const REQ_BIT_TENANT_PLAIN: u8 = 1 << 1;
    pub const REQ_BIT_SESSION_PLAIN: u8 = 1 << 2;

    pub const POOL_SIZE: usize = 0;
    pub const POOL_REGIME: usize = 4;
    pub const POOL_MEMBERSHIP_EPOCH: usize = 8;
    pub const POOL_RING_EPOCH: usize = 16;
    pub const POOL_GENERATION: usize = 24;
    pub const POOL_ZONE_COUNT: usize = 32;
    pub const ZONE_ZONE: usize = 0;
    pub const ZONE_HEALTHY: usize = 4;
    pub const ZONE_TOTAL: usize = 8;
}

/// The packed layout held to the records the bindings generate from the
/// contract (`contracts/smg-policy.wit`), at compile time. Each record's
/// fields are named once: a field the contract adds or drops fails its
/// pattern, and a scalar takes its width from the record's own type, so a
/// width the contract changes moves its span. The spans of a block, with
/// the zero bytes the layout table names, must then cover it exactly: every
/// byte once. A contract change the packed form does not follow fails the
/// build here, not a guest's first read.
mod pinned {
    use super::{layout::*, wit};

    /// The bytes one field occupies in a packed block.
    #[derive(Clone, Copy)]
    struct Span {
        at: usize,
        width: usize,
    }

    const fn span(at: usize, width: usize) -> Span {
        Span { at, width }
    }

    /// The width of a record's field: the size of its type.
    const fn width<R, F>(_: fn(&R) -> &F) -> usize {
        size_of::<F>()
    }

    /// Whether the spans cover a block of `size` bytes exactly: each inside
    /// the block, no two overlapping, no byte left out.
    const fn tiles(parts: &[&[Span]], size: usize) -> bool {
        let mut covered = 0;
        let mut p = 0;
        while p < parts.len() {
            let mut i = 0;
            while i < parts[p].len() {
                let a = parts[p][i];
                if a.at + a.width > size {
                    return false;
                }
                covered += a.width;
                // Against every span after it.
                let (mut q, mut j) = (p, i + 1);
                while q < parts.len() {
                    while j < parts[q].len() {
                        let b = parts[q][j];
                        if a.at < b.at + b.width && b.at < a.at + a.width {
                            return false;
                        }
                        j += 1;
                    }
                    q += 1;
                    j = 0;
                }
                i += 1;
            }
            p += 1;
        }
        covered == size
    }

    /// Names every field of a generated record, once: the scalars with
    /// where the block carries them (their spans, by the record's own
    /// widths), then the fields the block codes another way (a byte, a bit,
    /// a count before a section, a record of its own). The pattern is
    /// exhaustive: a field it does not name fails here, and the record
    /// form's own literal above names the field.
    macro_rules! record {
        ($spans:ident = $record:path { $($scalar:ident @ $at:expr,)* } coded { $($coded:ident,)* }) => {
            const _: fn(&$record) = |record| {
                let $record { $($scalar: _,)* $($coded: _,)* } = record;
            };
            const $spans: &[Span] = &[$(span($at, width(|record: &$record| &record.$scalar)),)*];
        };
    }

    record!(CANDIDATE = wit::Candidate {
        id @ ROW_ID,
        in_flight @ ROW_IN_FLIGHT,
        overlap_blocks @ ROW_OVERLAP,
        predicted_blocks @ ROW_PREDICTED,
        pair_class @ ROW_PAIR_CLASS,
        admitted_age_ms @ ROW_ADMITTED_AGE,
        indexed_growth @ ROW_INDEXED_GROWTH,
        dp_rank @ ROW_DP_RANK,
        dp_size @ ROW_DP_SIZE,
        zone @ ROW_ZONE,
        engine_kind @ ROW_ENGINE_KIND,
        connector @ ROW_CONNECTOR,
        session_in_flight @ ROW_SESSION_IN_FLIGHT,
        active_sessions @ ROW_ACTIVE_SESSIONS,
        key @ ROW_KEY,
        labels @ ROW_LABELS,
        indexed_blocks @ ROW_INDEXED_BLOCKS,
    } coded {
        // A byte each.
        health,
        wire_modes,
        role,
        // A bit each of the bits byte.
        session_hit,
        has_capacity,
        has_room,
        ec_compatible,
        vision,
        host_local,
        // The next record, in the same row, present by its bit.
        load,
    });
    record!(LOAD = wit::Load {
        running @ ROW_RUNNING,
        queued @ ROW_QUEUED,
        waiting_uncached_tokens @ ROW_WAITING_UNCACHED,
        kv_usage_permille @ ROW_KV_USAGE,
        active_kv_permille @ ROW_ACTIVE_KV,
        live_rate @ ROW_LIVE_RATE,
        resident_tokens @ ROW_RESIDENT,
        report_age_ms @ ROW_REPORT_AGE,
        max_running @ ROW_MAX_RUNNING,
    } coded {
        source,
    });
    /// The row's one-byte codes: health, wire modes, role, the load's
    /// source, and the bits.
    const ROW_CODES: &[Span] = &[
        span(ROW_HEALTH, 1),
        span(ROW_WIRE_MODES, 1),
        span(ROW_ROLE, 1),
        span(ROW_SOURCE, 1),
        span(ROW_BITS, 1),
    ];
    /// The table's zero bytes: 81, and 100 to 103.
    const ROW_ZERO: &[Span] = &[span(81, 1), span(100, 4)];
    const _: () = assert!(
        tiles(&[CANDIDATE, LOAD, ROW_CODES, ROW_ZERO], ROW),
        "the packed candidate row does not tile the candidate and load records of contracts/smg-policy.wit"
    );

    record!(REQUEST = wit::Request {
        namespace_seed @ REQ_NAMESPACE_SEED,
        seed @ REQ_SEED,
        budget_micros @ REQ_BUDGET,
        prompt_tokens @ REQ_PROMPT_TOKENS,
        max_output_tokens @ REQ_MAX_OUTPUT,
        priority @ REQ_PRIORITY,
        attempt @ REQ_ATTEMPT,
        block_size @ REQ_BLOCK_SIZE,
    } coded {
        // A fingerprint in the header, the plain form in its section.
        tenant,
        session_key,
        // A count or a length in the header, the bytes in a section.
        model,
        block_hashes,
        tried,
        headers,
        // A byte each, and a bit of the bits byte.
        phase,
        hash_kind,
        streaming,
        // The next record, in the header.
        media,
    });
    record!(MEDIA = wit::Media {
        images @ REQ_MEDIA_IMAGES,
        audio @ REQ_MEDIA_AUDIO,
        video @ REQ_MEDIA_VIDEO,
        bytes @ REQ_MEDIA_BYTES,
    } coded {});
    /// The header's fingerprints, counts and one-byte codes.
    const HEADER_CODES: &[Span] = &[
        span(REQ_TENANT, 8),
        span(REQ_SESSION_KEY, 8),
        span(REQ_HASH_COUNT, 2),
        span(REQ_TRIED_COUNT, 2),
        span(REQ_HEADER_COUNT, 2),
        span(REQ_MODEL_LEN, 2),
        span(REQ_PHASE, 1),
        span(REQ_HASH_KIND, 1),
        span(REQ_BITS, 1),
    ];
    /// The table's zero bytes: 66 to 67, and 76 to 79.
    const HEADER_ZERO: &[Span] = &[span(66, 2), span(76, 4)];
    const _: () = assert!(
        tiles(&[REQUEST, MEDIA, HEADER_CODES, HEADER_ZERO], HEADER),
        "the packed request header does not tile the request and media records of contracts/smg-policy.wit"
    );

    record!(POOL_FIELDS = wit::Pool {
        pool_size @ POOL_SIZE,
        membership_epoch @ POOL_MEMBERSHIP_EPOCH,
        ring_epoch @ POOL_RING_EPOCH,
        generation @ POOL_GENERATION,
    } coded {
        // A byte, and a count before the zone rows.
        regime,
        zones,
    });
    const POOL_CODES: &[Span] = &[span(POOL_REGIME, 1), span(POOL_ZONE_COUNT, 2)];
    /// The table's zero bytes: 5 to 7, and 34 to 39.
    const POOL_ZERO: &[Span] = &[span(5, 3), span(34, 6)];
    const _: () = assert!(
        tiles(&[POOL_FIELDS, POOL_CODES, POOL_ZERO], POOL),
        "the packed pool block does not tile the pool record of contracts/smg-policy.wit"
    );

    record!(ZONE_FIELDS = wit::ZoneCapacity {
        zone @ ZONE_ZONE,
        healthy @ ZONE_HEALTHY,
        total @ ZONE_TOTAL,
    } coded {});
    /// The table's zero bytes: 2 to 3.
    const ZONE_ZERO: &[Span] = &[span(2, 2)];
    const _: () = assert!(
        tiles(&[ZONE_FIELDS, ZONE_ZERO], ZONE),
        "the packed zone row does not tile the zone-capacity record of contracts/smg-policy.wit"
    );
}

/// A request the packed form cannot carry: a model name, a header value or a
/// plain identity past `u16::MAX` bytes. The host skips the packed plugin
/// for the decision, as for a `not-applicable` decline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TooLong;

/// The packed form's three blocks, reused across decisions.
#[derive(Debug, Default)]
pub(crate) struct Packed {
    pub(crate) request: Vec<u8>,
    pub(crate) candidates: Vec<u8>,
    pub(crate) pool: Vec<u8>,
}

fn put(buf: &mut [u8], at: usize, bytes: &[u8]) {
    buf[at..at + bytes.len()].copy_from_slice(bytes);
}

/// Appends a section of u64 values, eight little-endian bytes each: the
/// block grows once, by the section's length, and one pass extends it from
/// the values' bytes.
fn put_u64s(buf: &mut Vec<u8>, values: impl ExactSizeIterator<Item = u64>) {
    buf.reserve(8 * values.len());
    buf.extend(values.flat_map(u64::to_le_bytes));
}

/// The bits whose flag is set.
fn bits<const N: usize>(flags: [(bool, u8); N]) -> u8 {
    flags
        .iter()
        .fold(0, |acc, &(on, bit)| if on { acc | bit } else { acc })
}

fn section_len(s: &str) -> Result<u16, TooLong> {
    u16::try_from(s.len()).map_err(|_| TooLong)
}

impl Packed {
    /// Fills the three blocks for one decision.
    pub(crate) fn fill(
        &mut self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        pool: &PoolFacts<'_>,
        exposure: &Exposure,
        needs: Needs,
    ) -> Result<(), TooLong> {
        self.fill_request(facts, exposure, needs)?;
        self.fill_candidates(facts, candidates);
        self.fill_pool(pool);
        Ok(())
    }

    fn fill_request(
        &mut self,
        facts: &RequestFacts<'_>,
        exposure: &Exposure,
        needs: Needs,
    ) -> Result<(), TooLong> {
        use layout::*;
        let buf = &mut self.request;
        buf.clear();
        buf.resize(HEADER, 0);
        let tenant = identity(&facts.tenant, exposure.tenant);
        let session = session_identity(facts, exposure.session_key);
        let (tenant_hash, tenant_plain) = packed_identity(&tenant, || facts.tenant.hash);
        let (session_hash, session_plain) = packed_identity(&session, || facts.session().get());
        let model_len = section_len(facts.model)?;
        let exposed = facts
            .headers
            .iter()
            .filter(|h| exposure.headers.contains(h.name));
        let header_count = u16::try_from(exposed.clone().count()).map_err(|_| TooLong)?;
        let hashes = shown_hashes(facts, needs);
        let hash_count = u16::try_from(hashes.len()).map_err(|_| TooLong)?;
        let tried_count = u16::try_from(facts.tried.len()).map_err(|_| TooLong)?;
        put(buf, REQ_NAMESPACE_SEED, &facts.namespace_seed.to_le_bytes());
        put(buf, REQ_SEED, &facts.seed.to_le_bytes());
        put(buf, REQ_TENANT, &tenant_hash.to_le_bytes());
        put(buf, REQ_SESSION_KEY, &session_hash.to_le_bytes());
        put(buf, REQ_BUDGET, &facts.budget_micros.to_le_bytes());
        let size = shown_size(facts, needs);
        put(buf, REQ_PROMPT_TOKENS, &size.to_le_bytes());
        put(buf, REQ_MAX_OUTPUT, &facts.max_output_tokens.to_le_bytes());
        put(buf, REQ_PRIORITY, &facts.priority.to_le_bytes());
        put(buf, REQ_HASH_COUNT, &hash_count.to_le_bytes());
        put(buf, REQ_TRIED_COUNT, &tried_count.to_le_bytes());
        put(buf, REQ_HEADER_COUNT, &header_count.to_le_bytes());
        put(buf, REQ_MODEL_LEN, &model_len.to_le_bytes());
        buf[REQ_PHASE] = facts.phase as u8;
        buf[REQ_HASH_KIND] = facts.hash_kind as u8;
        buf[REQ_ATTEMPT] = facts.attempt;
        buf[REQ_BITS] = bits([
            (facts.streaming, REQ_BIT_STREAMING),
            (tenant_plain.is_some(), REQ_BIT_TENANT_PLAIN),
            (session_plain.is_some(), REQ_BIT_SESSION_PLAIN),
        ]);
        put(buf, REQ_MEDIA_IMAGES, &facts.media.images.to_le_bytes());
        put(buf, REQ_MEDIA_AUDIO, &facts.media.audio.to_le_bytes());
        put(buf, REQ_MEDIA_VIDEO, &facts.media.video.to_le_bytes());
        put(buf, REQ_MEDIA_BYTES, &facts.media.bytes.to_le_bytes());
        put(buf, REQ_BLOCK_SIZE, &facts.block_size.to_le_bytes());
        buf.extend_from_slice(facts.model.as_bytes());
        buf.resize(buf.len().next_multiple_of(8), 0);
        put_u64s(buf, hashes.iter().copied());
        put_u64s(buf, facts.tried.iter().map(|t| t.0));
        for h in exposed {
            let len = section_len(h.value)?;
            buf.push(h.name);
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(h.value.as_bytes());
        }
        for plain in [tenant_plain, session_plain].into_iter().flatten() {
            let len = section_len(plain)?;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(plain.as_bytes());
        }
        Ok(())
    }

    fn fill_candidates(&mut self, facts: &RequestFacts<'_>, candidates: &[Candidate]) {
        use layout::ROW;
        let buf = &mut self.candidates;
        buf.clear();
        buf.resize(candidates.len() * ROW, 0);
        let rows = buf.as_chunks_mut::<ROW>().0;
        // The candidates' affinity facts ride the request's column; the
        // table keeps them in the row. The column is walked beside the
        // candidates and its emptiness decided once: read by index, the
        // emptiness and the bound are tested for every row, which cost the
        // fill 2.5 ns a row at 512 candidates. An empty column is nothing
        // held by any candidate. Any other holds one entry a candidate, and
        // one that ends before the candidates fails here, as an index past
        // its end does. Rows the host named in the column's place are
        // walked beside the candidates: a holder's row carries its two
        // depths, and every row reads its age, growth and index size as
        // zero, so a host fills the column for a profile whose guest reads
        // those.
        match facts.feed() {
            feed if feed.is_empty() => {
                let nothing = AffinityFacts::default();
                for (c, row) in candidates.iter().zip(rows) {
                    Self::fill_row(row, c, nothing, facts.in_flight(c));
                }
            }
            Feed::Column(column) => {
                let column = &column[..candidates.len()];
                for ((c, held), row) in candidates.iter().zip(column).zip(rows) {
                    Self::fill_row(row, c, *held, facts.in_flight(c));
                }
            }
            Feed::Named(named) => {
                let mut walk = named.walk();
                for ((index, c), row) in (0u32..).zip(candidates).zip(rows) {
                    Self::fill_row(row, c, walk.entry(index), facts.in_flight(c));
                }
            }
        }
    }

    /// One candidate's row of the table, its affinity facts and the
    /// requests in flight the request's facts read for it in the row.
    fn fill_row(row: &mut [u8; layout::ROW], c: &Candidate, held: AffinityFacts, in_flight: u32) {
        use layout::*;
        put(row, ROW_ID, &c.id.0.to_le_bytes());
        put(row, ROW_IN_FLIGHT, &in_flight.to_le_bytes());
        let load = c.load.unwrap_or_default();
        put(row, ROW_RUNNING, &load.running.to_le_bytes());
        put(row, ROW_QUEUED, &load.queued.to_le_bytes());
        put(
            row,
            ROW_WAITING_UNCACHED,
            &load.waiting_uncached_tokens.to_le_bytes(),
        );
        put(row, ROW_OVERLAP, &held.overlap_blocks.to_le_bytes());
        put(row, ROW_PREDICTED, &held.predicted_blocks.to_le_bytes());
        put(row, ROW_RESIDENT, &load.resident_tokens.to_le_bytes());
        put(row, ROW_PAIR_CLASS, &c.pair_class.to_le_bytes());
        put(row, ROW_ADMITTED_AGE, &held.admitted_age_ms.to_le_bytes());
        put(row, ROW_REPORT_AGE, &load.report_age_ms.to_le_bytes());
        put(row, ROW_INDEXED_GROWTH, &held.indexed_growth.to_le_bytes());
        put(row, ROW_LIVE_RATE, &load.live_rate.to_le_bytes());
        put(row, ROW_MAX_RUNNING, &load.max_running.to_le_bytes());
        put(row, ROW_KV_USAGE, &load.kv_usage_permille.to_le_bytes());
        put(row, ROW_ACTIVE_KV, &load.active_kv_permille.to_le_bytes());
        put(row, ROW_DP_RANK, &c.dp_rank.to_le_bytes());
        put(row, ROW_DP_SIZE, &c.dp_size.to_le_bytes());
        put(row, ROW_ZONE, &c.zone.to_le_bytes());
        put(row, ROW_ENGINE_KIND, &c.engine_kind.to_le_bytes());
        put(row, ROW_CONNECTOR, &c.connector.to_le_bytes());
        put(
            row,
            ROW_SESSION_IN_FLIGHT,
            &c.session_in_flight.to_le_bytes(),
        );
        row[ROW_HEALTH] = health_bits(c.health);
        row[ROW_WIRE_MODES] = c.wire_modes.bits();
        row[ROW_ROLE] = c.role as u8;
        row[ROW_SOURCE] = load.source as u8;
        row[ROW_BITS] = bits([
            (c.session_hit, BIT_SESSION_HIT),
            (c.has_capacity, BIT_HAS_CAPACITY),
            (c.has_room, BIT_HAS_ROOM),
            (c.ec_compatible, BIT_EC_COMPATIBLE),
            (c.vision, BIT_VISION),
            (c.host_local, BIT_HOST_LOCAL),
            (c.load.is_some(), BIT_LOAD_PRESENT),
        ]);
        put(row, ROW_ACTIVE_SESSIONS, &c.active_sessions.to_le_bytes());
        put(row, ROW_KEY, &c.key.to_le_bytes());
        put(row, ROW_LABELS, &c.labels.to_le_bytes());
        put(row, ROW_INDEXED_BLOCKS, &held.indexed_blocks.to_le_bytes());
    }

    fn fill_pool(&mut self, pool: &PoolFacts<'_>) {
        use layout::*;
        let buf = &mut self.pool;
        buf.clear();
        buf.resize(POOL + pool.zones.len() * ZONE, 0);
        put(buf, POOL_SIZE, &pool.pool_size.to_le_bytes());
        buf[POOL_REGIME] = pool.regime as u8;
        put(
            buf,
            POOL_MEMBERSHIP_EPOCH,
            &pool.membership_epoch.to_le_bytes(),
        );
        put(buf, POOL_RING_EPOCH, &pool.ring_epoch.to_le_bytes());
        put(buf, POOL_GENERATION, &pool.generation.to_le_bytes());
        // The zone count is bounded by the declared zone list, far below u16.
        put(
            buf,
            POOL_ZONE_COUNT,
            &(pool.zones.len().min(usize::from(u16::MAX)) as u16).to_le_bytes(),
        );
        for (z, row) in pool.zones.iter().zip(buf[POOL..].as_chunks_mut::<ZONE>().0) {
            put(row, ZONE_ZONE, &z.zone.to_le_bytes());
            put(row, ZONE_HEALTHY, &z.healthy.to_le_bytes());
            put(row, ZONE_TOTAL, &z.total.to_le_bytes());
        }
    }
}

/// The packed block's identity fields: the fingerprint under both the hash
/// and the plain form (`0` withheld), and the plain string when it crosses;
/// `hash` makes the fingerprint the plain form carries beside its string.
fn packed_identity<'a>(
    id: &wit::Identity<'a>,
    hash: impl FnOnce() -> u64,
) -> (u64, Option<&'a str>) {
    match id {
        wit::Identity::None => (0, None),
        wit::Identity::Hash(h) => (*h, None),
        wit::Identity::Plain(plain) => (hash(), Some(plain)),
    }
}

/// The lease as the plugin may see it: the session key under the plugin's
/// exposure, as in the request view (withheld reads `0`).
pub(crate) fn lease(l: &Lease, exposure: &Exposure) -> wit::Lease {
    wit::Lease {
        id: l.id.0,
        kind: match l.kind {
            LeaseKind::Credit => wit::LeaseKind::Credit,
            LeaseKind::Reservation => wit::LeaseKind::Reservation,
        },
        side: side(l.side),
        target_index: l.target_index,
        generation: l.generation,
        session_key: match exposure.session_key {
            Reveal::None => 0,
            Reveal::Hash | Reveal::Plain => l.session.get(),
        },
        tokens: l.prompt_size(),
    }
}

pub(crate) fn target(t: &Target) -> wit::Target {
    wit::Target {
        id: t.id.0,
        dp_rank: t.dp_rank,
        pair: t.pair.map(|p| wit::Pair {
            decode_id: p.decode_id.0,
            decode_dp_rank: p.decode_dp_rank,
        }),
    }
}

/// The decision's targets into `out`; a shed or a queue has none.
pub(crate) fn targets(decision: &Decision, out: &mut [wit::Target; MAX_TARGETS]) -> usize {
    match decision {
        Decision::Pick(pick) => {
            let mut n = 0;
            for (slot, t) in out.iter_mut().zip(pick.targets.iter()) {
                *slot = target(t);
                n += 1;
            }
            n
        }
        Decision::Shed(_) | Decision::Queue(_) => 0,
    }
}

fn transport_class(c: TransportClass) -> wit::TransportClass {
    match c {
        TransportClass::Ok => wit::TransportClass::Ok,
        TransportClass::ClientError => wit::TransportClass::ClientError,
        TransportClass::ServerError => wit::TransportClass::ServerError,
        TransportClass::ConnectFailed => wit::TransportClass::ConnectFailed,
        TransportClass::Timeout => wit::TransportClass::Timeout,
        TransportClass::Cancelled => wit::TransportClass::Cancelled,
    }
}

pub(crate) fn completed(l: &Lease, o: &Outcome, exposure: &Exposure) -> wit::Completed {
    wit::Completed {
        lease: lease(l, exposure),
        side: side(o.side),
        served: o.served.0,
        status: o.status,
        class: transport_class(o.class),
        ttft_micros: o.ttft_micros,
        tokens_in: o.tokens_in,
        tokens_out: o.tokens_out,
        duration_micros: o.duration_micros,
    }
}

/// Every release cause maps to exactly one member of the WIT enum: a cause
/// added to the crate's enum fails to compile here until the WIT names it.
pub(crate) fn release_reason(r: ReleaseReason) -> wit::ReleaseReason {
    match r {
        ReleaseReason::Abandoned => wit::ReleaseReason::Abandoned,
        ReleaseReason::LateDecision => wit::ReleaseReason::LateDecision,
        ReleaseReason::ClientGone => wit::ReleaseReason::ClientGone,
        ReleaseReason::CommitFailed => wit::ReleaseReason::CommitFailed,
        ReleaseReason::Expired => wit::ReleaseReason::Expired,
        ReleaseReason::WorkerRemoved => wit::ReleaseReason::WorkerRemoved,
    }
}

/// A membership word as the guest's world carries it. A snapshot's ids are
/// written into `snapshot` and a left word's names into `names`: the word
/// borrows them for the call.
pub(crate) fn membership<'a>(
    m: &Membership<'a>,
    snapshot: &'a mut Vec<u64>,
    names: &'a mut Vec<&'a str>,
) -> wit::Membership<'a> {
    match *m {
        Membership::Left { id, models } => {
            names.clear();
            names.extend(models.iter().map(String::as_str));
            wit::Membership::Left((id.0, names.as_slice()))
        }
        Membership::Added(w) => wit::Membership::Added(w.0),
        Membership::Removed(w) => wit::Membership::Removed(w.0),
        Membership::Replaced { old, new } => wit::Membership::Replaced((old.0, new.0)),
        Membership::Relabelled(w) => wit::Membership::Relabelled(w.0),
        Membership::Resized(w) => wit::Membership::Resized(w.0),
        Membership::Snapshot(workers) => {
            snapshot.clear();
            snapshot.extend(workers.iter().map(|w| w.0));
            wit::Membership::Snapshot(snapshot)
        }
    }
}

/// A signal; a veto's instant on the host's clock becomes the microseconds
/// it has left at `now_micros` on that clock (`0` when none).
pub(crate) fn signal(s: &Signal<'_>, now_micros: u64) -> wit::Signal {
    match s {
        Signal::Load {
            worker,
            dp_rank,
            row,
            ..
        } => wit::Signal::Load((worker.0, *dp_rank, load(row))),
        Signal::Health { worker, health: h } => wit::Signal::Health((worker.0, health(*h))),
        Signal::GatewayVeto {
            worker,
            until_micros,
        } => wit::Signal::GatewayVeto((
            worker.0,
            u32::try_from(until_micros.saturating_sub(now_micros)).unwrap_or(u32::MAX),
        )),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/plugins/wasm/view/tests.rs"]
mod tests;
