//! MEASUREMENT PROBE, NOT FOR MERGE. What a decision's passes and the view
//! build cost over several layouts of the candidate row, side by side in one
//! binary, read on the pull-request gate's runner.
//!
//! The gate builds both of its sides from this file over the crate as it
//! stands, so the real row cannot differ a side. The layouts are therefore
//! MODELS: row types local to this file, their offsets pinned, walked by a
//! copy of the engine's filter stage (one call a filter a decision, the test
//! inlined, the survivors kept by position), of its passthrough pick and of
//! its trace pass, and written field by field in place with what the bench's
//! view build writes. The model of the row as it stands is checked against
//! the real row at start, and the engine's own ids for it are kept beside the
//! model as its test: `policy_engine/view_build/<n>` and
//! `policy_engine/decide/<preset>/<n>` for three presets.
//!
//! Probe ids, every row buffer on a page's start:
//! - `policy_engine/probe_build/<layout>/<n>`: the view build;
//! - `policy_engine/probe_pass/<layout>/<n>`: the decision of a preset
//!   without a scorer (three filters, the first survivors picked);
//! - `policy_engine/probe_score/<layout>/512`: the same with one scorer that
//!   reads each survivor's load block, in-flight count and sample instant;
//! - `policy_engine/probe_raw/<stride>_at_<first>_<second>/512`: the filter
//!   pass with the stride and the two passes' flag offsets as run-time
//!   values, one machine code for every place, so a place differs from
//!   another by where its bytes sit and by nothing the compiler chose.
//!
//! Each run prints the runner's processor line and data cache sizes, the
//! table of layouts and the lines each one's two flag passes touch to its
//! log.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

mod common;

use std::{
    hint::black_box,
    mem::{offset_of, size_of},
    time::Duration,
};

use common::{build_view, engine_of, fleet, WorkerState};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};
use smg_policy_controller::{
    Candidate, Constraints, Health, LoadRow, PoolFacts, RequestFacts, Scratch, SignalSource,
    WorkerId,
};

const SIZES: [usize; 3] = [8, 64, 512];

/// The engine's own ids kept beside the model: two presets without a scorer
/// and one with the load scorer, at the two larger sizes.
const KEPT_PRESETS: [&str; 3] = ["passthrough", "manual", "least_load"];
const KEPT_SIZES: [usize; 2] = [64, 512];

fn view_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/view_build");
    for &n in &SIZES {
        let fleet = fleet(n);
        let mut out = Vec::with_capacity(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                build_view(black_box(&fleet), &mut out);
                black_box(&out);
                out.len()
            });
        });
    }
    group.finish();
}

fn decide(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/decide");
    for preset in KEPT_PRESETS {
        let engine = engine_of(preset);
        for &n in &KEPT_SIZES {
            let fleet = fleet(n);
            let mut view = Vec::with_capacity(n);
            build_view(&fleet, &mut view);
            let mut scratch = Scratch::new();
            let pool = PoolFacts {
                pool_size: n as u32,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            group.bench_with_input(BenchmarkId::new(preset, n), &n, |b, _| {
                let mut seed = 0u64;
                b.iter(|| {
                    seed += 1;
                    let facts = RequestFacts {
                        model: "m",
                        prompt_tokens: 512,
                        seed,
                        now_micros: 1_000_000,
                        ..RequestFacts::default()
                    };
                    engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
                });
            });
        }
    }
    group.finish();
}

// ------------------------------------------------------------ the layouts

/// The engine's bounds, copied: the targets of a pick and the trace's top.
const MAX_TARGETS: usize = 4;
const TRACE_TOP: usize = 3;

/// A model of the candidate row: the real row's fields under the real names,
/// in an order and at a size of its own.
trait Row: Copy + Send + Sync + 'static {
    /// The layout's name in the ids.
    const NAME: &'static str;
    /// Where the five health flags, the id, the sample instant, the in-flight
    /// count, the load block and the position start.
    const OFFSETS: [usize; 6];

    /// The row the bench's view build writes for one worker.
    fn fill(position: u32, w: &WorkerState) -> Self;
    /// The same row written in place, field by field.
    fn write(&mut self, position: u32, w: &WorkerState);
    fn health(&self) -> &Health;
    fn id(&self) -> WorkerId;
    fn position(&self) -> u32;
    fn dp_rank(&self) -> u16;
    fn in_flight(&self) -> u32;
    fn load(&self) -> Option<&LoadRow>;
    fn sampled_at(&self) -> u64;
}

/// What the bench's view build writes into one field of a row: the worker's
/// state for seven of them, the default for the rest. Every field is written
/// by name, the load block's and the flags' too, so no layout is built
/// through a default value first.
macro_rules! filled {
    (position, $position:ident, $w:ident) => {
        $position
    };
    (id, $position:ident, $w:ident) => {
        WorkerId($w.id)
    };
    (key, $position:ident, $w:ident) => {
        $w.id as u32
    };
    (health, $position:ident, $w:ident) => {
        Health {
            healthy: $w.healthy,
            breaker_open: false,
            stalled: false,
            draining: false,
            overloaded: false,
        }
    };
    (in_flight, $position:ident, $w:ident) => {
        $w.in_flight
    };
    (load, $position:ident, $w:ident) => {
        Some(LoadRow {
            running: $w.running,
            queued: $w.queued,
            waiting_uncached_tokens: 0,
            kv_usage_permille: $w.kv_permille,
            active_kv_permille: 0,
            live_rate: $w.live_rate,
            resident_tokens: 0,
            report_age_ms: 5,
            source: SignalSource::Events,
            max_running: 0,
        })
    };
    (load_sampled_at_micros, $position:ident, $w:ident) => {
        $w.sampled_at_micros
    };
    ($other:ident, $position:ident, $w:ident) => {
        Default::default()
    };
}

macro_rules! row_layout {
    ($(#[$attr:meta])* $name:ident, $label:literal, { $($field:ident: $ty:ty),* $(,)? }) => {
        $(#[$attr])*
        #[derive(Clone, Copy)]
        #[repr(C)]
        struct $name {
            $($field: $ty),*
        }

        impl Row for $name {
            const NAME: &'static str = $label;
            const OFFSETS: [usize; 6] = [
                offset_of!($name, health),
                offset_of!($name, id),
                offset_of!($name, load_sampled_at_micros),
                offset_of!($name, in_flight),
                offset_of!($name, load),
                offset_of!($name, position),
            ];

            #[inline]
            fn fill(position: u32, w: &WorkerState) -> Self {
                Self {
                    $($field: filled!($field, position, w)),*
                }
            }

            #[inline]
            fn write(&mut self, position: u32, w: &WorkerState) {
                $(self.$field = filled!($field, position, w);)*
            }

            #[inline]
            fn health(&self) -> &Health {
                &self.health
            }

            #[inline]
            fn id(&self) -> WorkerId {
                self.id
            }

            #[inline]
            fn position(&self) -> u32 {
                self.position
            }

            #[inline]
            fn dp_rank(&self) -> u16 {
                self.dp_rank
            }

            #[inline]
            fn in_flight(&self) -> u32 {
                self.in_flight
            }

            #[inline]
            fn load(&self) -> Option<&LoadRow> {
                self.load.as_ref()
            }

            #[inline]
            fn sampled_at(&self) -> u64 {
                self.load_sampled_at_micros
            }
        }
    };
}

// The row as it stands, 128 bytes, in the order the compiler gives it (the
// start check holds this model to the real row): the flags at 112.
row_layout!(M128, "m128", {
    id: WorkerId,
    load_sampled_at_micros: u64,
    labels: u64,
    position: u32,
    in_flight: u32,
    overlap_blocks: u32,
    predicted_blocks: u32,
    pair_class: u32,
    admitted_age_ms: u32,
    indexed_growth: u32,
    key: u32,
    indexed_blocks: u32,
    load: Option<LoadRow>,
    dp_rank: u16,
    dp_size: u16,
    engine_kind: u16,
    connector: u16,
    session_in_flight: u16,
    zone: u16,
    active_sessions: u16,
    wire_modes: u8,
    role: u8,
    health: Health,
    session_hit: bool,
    has_capacity: bool,
    has_room: bool,
    ec_compatible: bool,
    vision: bool,
    host_local: bool,
});

// The row less its five affinity words, 104 bytes, in the order the compiler
// gives that field set: the flags at 92.
row_layout!(H104, "h104", {
    id: WorkerId,
    load_sampled_at_micros: u64,
    labels: u64,
    position: u32,
    in_flight: u32,
    pair_class: u32,
    key: u32,
    load: Option<LoadRow>,
    dp_rank: u16,
    dp_size: u16,
    engine_kind: u16,
    connector: u16,
    session_in_flight: u16,
    zone: u16,
    active_sessions: u16,
    wire_modes: u8,
    role: u8,
    health: Health,
    session_hit: bool,
    has_capacity: bool,
    has_room: bool,
    ec_compatible: bool,
    vision: bool,
    host_local: bool,
});

// The same 104 bytes with what a filter and a load scorer read first: the
// flags in the first word, then the id, the sample instant, the in-flight
// count and the load block, 64 bytes in all.
row_layout!(H104Hot, "h104hot", {
    health: Health,
    has_capacity: bool,
    has_room: bool,
    session_hit: bool,
    id: WorkerId,
    load_sampled_at_micros: u64,
    in_flight: u32,
    load: Option<LoadRow>,
    labels: u64,
    position: u32,
    pair_class: u32,
    key: u32,
    dp_rank: u16,
    dp_size: u16,
    engine_kind: u16,
    connector: u16,
    session_in_flight: u16,
    zone: u16,
    active_sessions: u16,
    wire_modes: u8,
    role: u8,
    ec_compatible: bool,
    vision: bool,
    host_local: bool,
});

// 96 bytes in the order of `h104`: the model drops one cold word (the label
// set) to get there; the real row would have to find its eight bytes.
row_layout!(H96, "h96", {
    id: WorkerId,
    load_sampled_at_micros: u64,
    position: u32,
    in_flight: u32,
    pair_class: u32,
    key: u32,
    load: Option<LoadRow>,
    dp_rank: u16,
    dp_size: u16,
    engine_kind: u16,
    connector: u16,
    session_in_flight: u16,
    zone: u16,
    active_sessions: u16,
    wire_modes: u8,
    role: u8,
    health: Health,
    session_hit: bool,
    has_capacity: bool,
    has_room: bool,
    ec_compatible: bool,
    vision: bool,
    host_local: bool,
});

// 96 bytes in the order of `h104hot`.
row_layout!(H96Hot, "h96hot", {
    health: Health,
    has_capacity: bool,
    has_room: bool,
    session_hit: bool,
    id: WorkerId,
    load_sampled_at_micros: u64,
    in_flight: u32,
    load: Option<LoadRow>,
    position: u32,
    pair_class: u32,
    key: u32,
    dp_rank: u16,
    dp_size: u16,
    engine_kind: u16,
    connector: u16,
    session_in_flight: u16,
    zone: u16,
    active_sessions: u16,
    wire_modes: u8,
    role: u8,
    ec_compatible: bool,
    vision: bool,
    host_local: bool,
});

// 112 bytes: `h104hot` and one spare word.
row_layout!(H112Hot, "h112hot", {
    health: Health,
    has_capacity: bool,
    has_room: bool,
    session_hit: bool,
    id: WorkerId,
    load_sampled_at_micros: u64,
    in_flight: u32,
    load: Option<LoadRow>,
    labels: u64,
    reserve: u64,
    position: u32,
    pair_class: u32,
    key: u32,
    dp_rank: u16,
    dp_size: u16,
    engine_kind: u16,
    connector: u16,
    session_in_flight: u16,
    zone: u16,
    active_sessions: u16,
    wire_modes: u8,
    role: u8,
    ec_compatible: bool,
    vision: bool,
    host_local: bool,
});

// `h104` aligned to a 64-byte line: a stride of 128 again, every row on a
// line's start, the flags at 92.
row_layout!(
    #[repr(align(64))]
    H128Line,
    "h128line",
    {
        id: WorkerId,
        load_sampled_at_micros: u64,
        labels: u64,
        position: u32,
        in_flight: u32,
        pair_class: u32,
        key: u32,
        load: Option<LoadRow>,
        dp_rank: u16,
        dp_size: u16,
        engine_kind: u16,
        connector: u16,
        session_in_flight: u16,
        zone: u16,
        active_sessions: u16,
        wire_modes: u8,
        role: u8,
        health: Health,
        session_hit: bool,
        has_capacity: bool,
        has_room: bool,
        ec_compatible: bool,
        vision: bool,
        host_local: bool,
    }
);

// `h104hot` aligned to a 64-byte line: a stride of 128, what a filter and a
// load scorer read on the row's first line.
row_layout!(
    #[repr(align(64))]
    H128Hot,
    "h128hot",
    {
        health: Health,
        has_capacity: bool,
        has_room: bool,
        session_hit: bool,
        id: WorkerId,
        load_sampled_at_micros: u64,
        in_flight: u32,
        load: Option<LoadRow>,
        labels: u64,
        position: u32,
        pair_class: u32,
        key: u32,
        dp_rank: u16,
        dp_size: u16,
        engine_kind: u16,
        connector: u16,
        session_in_flight: u16,
        zone: u16,
        active_sessions: u16,
        wire_modes: u8,
        role: u8,
        ec_compatible: bool,
        vision: bool,
        host_local: bool,
    }
);

/// The model of the row as it stands is the real row: the same size, and the
/// same place for every field a walk reads. A change of the real row stops
/// the probe here.
fn the_model_is_the_real_row() {
    assert_eq!(size_of::<M128>(), size_of::<Candidate>());
    assert_eq!(
        [
            offset_of!(M128, health),
            offset_of!(M128, id),
            offset_of!(M128, load_sampled_at_micros),
            offset_of!(M128, in_flight),
            offset_of!(M128, load),
            offset_of!(M128, position),
            offset_of!(M128, dp_rank),
            offset_of!(M128, key),
        ],
        [
            offset_of!(Candidate, health),
            offset_of!(Candidate, id),
            offset_of!(Candidate, load_sampled_at_micros),
            offset_of!(Candidate, in_flight),
            offset_of!(Candidate, load),
            offset_of!(Candidate, position),
            offset_of!(Candidate, dp_rank),
            offset_of!(Candidate, key),
        ],
        "the model of the row as it stands has the real row's offsets"
    );
}

// ------------------------------------------------------------ placement

/// The page the row buffers are placed against.
const PAGE: usize = 4096;

/// How often the allocator is asked: it hands out every 16th byte of a page
/// in turn, and a later placement of one size is handed the earlier one's
/// rejects first, so a size placed four times needs a thousand asks.
const ASKS: usize = 4_096;

/// A buffer of capacity `n` that starts on a page's start. The allocator is
/// asked until it hands such an address; a reject stays allocated until the
/// end, so each ask gets another address. Nothing is timed in a buffer that
/// is not placed: when no ask of the bound lands on a page's start the probe
/// stops here.
fn page_start_buffer<T>(n: usize) -> Vec<T> {
    let mut rejects = Vec::new();
    let mut buffer = Vec::<T>::with_capacity(n);
    for _ in 1..ASKS {
        if buffer.as_ptr().addr().is_multiple_of(PAGE) {
            break;
        }
        rejects.push(std::mem::replace(&mut buffer, Vec::with_capacity(n)));
    }
    assert!(
        buffer.as_ptr().addr().is_multiple_of(PAGE),
        "no buffer of {n} rows on a page's start in {ASKS} asks: a placed id is not timed elsewhere"
    );
    buffer
}

// ------------------------------------------------------------ the walks

/// What a filter reads of the request and of the constraints.
struct Facts<'a> {
    tried: &'a [WorkerId],
    prompt_tokens: u32,
    now_micros: u64,
}

struct Limits<'a> {
    excluded: &'a [WorkerId],
    overload_lifted: bool,
}

/// The engine's filter stage, over a model row: one `retain` a decision with
/// the test inlined into its loop.
trait Stage<R: Row>: Send + Sync {
    fn keep(&self, facts: &Facts<'_>, candidate: &R, cx: &Limits<'_>) -> bool;

    fn soft(&self) -> bool {
        false
    }

    fn retain(&self, facts: &Facts<'_>, candidates: &[R], cx: &Limits<'_>, kept: &mut Vec<u32>) {
        kept.retain(|&i| self.keep(facts, &candidates[i as usize], cx));
    }
}

struct Healthy;

impl<R: Row> Stage<R> for Healthy {
    fn keep(&self, _: &Facts<'_>, c: &R, _: &Limits<'_>) -> bool {
        c.health().routable_ignoring_overload()
    }
}

struct NotOverloaded;

impl<R: Row> Stage<R> for NotOverloaded {
    fn keep(&self, _: &Facts<'_>, c: &R, cx: &Limits<'_>) -> bool {
        cx.overload_lifted || !c.health().overloaded
    }
}

struct NotTried;

impl<R: Row> Stage<R> for NotTried {
    fn keep(&self, facts: &Facts<'_>, c: &R, cx: &Limits<'_>) -> bool {
        !facts.tried.contains(&c.id()) && !cx.excluded.contains(&c.id())
    }

    fn soft(&self) -> bool {
        true
    }
}

/// The presets' filters, behind the same kind of pointer the engine holds.
fn standard_filters<R: Row>() -> Vec<Box<dyn Stage<R>>> {
    black_box(vec![
        Box::new(Healthy) as Box<dyn Stage<R>>,
        Box::new(NotOverloaded),
        Box::new(NotTried),
    ])
}

/// The decision's buffers, kept across decisions as the engine's scratch.
#[derive(Default)]
struct Buffers {
    kept: Vec<u32>,
    before_soft: Vec<u32>,
    scores: Vec<f32>,
    stage: Vec<f32>,
}

/// What a decision hands back: its targets, its survivors and the trace's top.
#[derive(Clone, Copy, Default)]
struct Picked {
    targets: [(u32, WorkerId, u16); MAX_TARGETS],
    len: u32,
    survivors: u32,
    top: [(WorkerId, f32); TRACE_TOP],
    top_len: u32,
}

/// The engine's filter pass: the hard filters, then each soft filter as long
/// as it leaves something. False when a stage emptied the pool.
#[inline]
fn filter<R: Row>(
    filters: &[Box<dyn Stage<R>>],
    facts: &Facts<'_>,
    cx: &Limits<'_>,
    candidates: &[R],
    buf: &mut Buffers,
) -> bool {
    let kept = &mut buf.kept;
    kept.clear();
    kept.extend(0..candidates.len() as u32);
    if kept.is_empty() {
        return false;
    }
    for hard in filters.iter().filter(|f| !f.soft()) {
        hard.retain(facts, candidates, cx, kept);
        if kept.is_empty() {
            return false;
        }
    }
    for soft in filters.iter().filter(|f| f.soft()) {
        buf.before_soft.clear();
        buf.before_soft.extend_from_slice(kept);
        soft.retain(facts, candidates, cx, kept);
        if kept.is_empty() {
            kept.extend_from_slice(&buf.before_soft);
        }
    }
    true
}

fn target_of<R: Row>(c: &R) -> (u32, WorkerId, u16) {
    (c.position(), c.id(), c.dp_rank())
}

/// The engine's trace pass: the three best scores in scan order, then their
/// rows' ids.
#[inline]
fn trace<R: Row>(candidates: &[R], kept: &[u32], scores: &[f32], picked: &mut Picked) {
    let mut top: [(usize, f32); TRACE_TOP] = [(0, f32::NEG_INFINITY); TRACE_TOP];
    let mut len = 0usize;
    for (i, &score) in scores.iter().enumerate() {
        if len == TRACE_TOP && score <= top[TRACE_TOP - 1].1 {
            continue;
        }
        let mut slot = if len < TRACE_TOP {
            len += 1;
            len - 1
        } else {
            TRACE_TOP - 1
        };
        while slot > 0 && score > top[slot - 1].1 {
            top[slot] = top[slot - 1];
            slot -= 1;
        }
        top[slot] = (i, score);
    }
    for (slot, &(i, score)) in top.iter().enumerate().take(len) {
        picked.top[slot] = (candidates[kept[i] as usize].id(), score);
    }
    picked.top_len = len as u32;
    picked.survivors = kept.len() as u32;
}

/// A decision of a preset without a scorer: the filters, the scores at zero,
/// the first survivors as the targets, the trace.
#[inline(never)]
fn decide_unscored<R: Row>(
    filters: &[Box<dyn Stage<R>>],
    facts: &Facts<'_>,
    cx: &Limits<'_>,
    candidates: &[R],
    buf: &mut Buffers,
) -> Picked {
    let mut picked = Picked::default();
    if !filter(filters, facts, cx, candidates, buf) {
        return picked;
    }
    buf.scores.clear();
    buf.scores.resize(buf.kept.len(), 0.0);
    for &i in buf.kept.iter().take(MAX_TARGETS) {
        picked.targets[picked.len as usize] = target_of(&candidates[i as usize]);
        picked.len += 1;
    }
    trace(candidates, &buf.kept, &buf.scores, &mut picked);
    picked
}

/// A decision with one scorer that reads each survivor's load block,
/// in-flight count and sample instant: a wait by the row's load, normalised
/// by the worst, the best score picked.
#[inline(never)]
fn decide_scored<R: Row>(
    filters: &[Box<dyn Stage<R>>],
    facts: &Facts<'_>,
    cx: &Limits<'_>,
    candidates: &[R],
    buf: &mut Buffers,
) -> Picked {
    let mut picked = Picked::default();
    if !filter(filters, facts, cx, candidates, buf) {
        return picked;
    }
    buf.scores.clear();
    buf.scores.resize(buf.kept.len(), 0.0);
    buf.stage.clear();
    buf.stage.resize(buf.kept.len(), 0.0);
    let mut worst = 0.0f32;
    for (wait, &i) in buf.stage.iter_mut().zip(buf.kept.iter()) {
        let c = &candidates[i as usize];
        let ahead = match c.load() {
            Some(load) => {
                let rate = load.live_rate.max(1) as f32;
                let age_ms = (facts.now_micros.saturating_sub(c.sampled_at()) / 1_000) as f32;
                let queue = (load.running + load.queued + c.in_flight()) as f32;
                queue * facts.prompt_tokens as f32 / rate
                    * (1.0 + f32::from(load.kv_usage_permille) * 0.001)
                    + age_ms * 0.001
            }
            None => c.in_flight() as f32,
        };
        *wait = ahead;
        worst = worst.max(ahead);
    }
    let norm = if worst > 0.0 { 1.0 / worst } else { 1.0 };
    for (sum, wait) in buf.scores.iter_mut().zip(buf.stage.iter()) {
        *sum += (1.0 - wait * norm).clamp(0.0, 1.0);
    }
    let mut best = 0usize;
    for (i, &score) in buf.scores.iter().enumerate() {
        if score > buf.scores[best] {
            best = i;
        }
    }
    picked.targets[0] = target_of(&candidates[buf.kept[best] as usize]);
    picked.len = 1;
    trace(candidates, &buf.kept, &buf.scores, &mut picked);
    picked
}

/// Where an untyped pass reads a row: the stride, the offset of the first
/// filter's four flags and the offset of the second filter's flag.
#[derive(Clone, Copy)]
struct Place {
    stride: usize,
    first: usize,
    second: usize,
}

/// The 64-byte lines the two flag passes touch over a buffer on a line's
/// start: the first pass reads four bytes of every row, the second one byte
/// of every row the first kept.
fn lines_touched(place: Place, healthy: &[bool]) -> usize {
    let mut lines: Vec<usize> = Vec::with_capacity(healthy.len() * 3);
    for (i, &kept) in healthy.iter().enumerate() {
        let row = i * place.stride;
        lines.push((row + place.first) / 64);
        lines.push((row + place.first + 3) / 64);
        if kept {
            lines.push((row + place.second) / 64);
        }
    }
    lines.sort_unstable();
    lines.dedup();
    lines.len()
}

/// The filter pass over bytes: the stride and the two flag offsets are
/// run-time values, so every place runs one machine code. The four flags of
/// the first filter and the flag of the second are read where the place puts
/// them; the soft filter reads no row while no worker was tried, as in the
/// engine.
#[inline(never)]
fn decide_raw(
    rows: &[u8],
    place: Place,
    len: usize,
    facts: &Facts<'_>,
    cx: &Limits<'_>,
    buf: &mut Buffers,
) -> (u32, u32) {
    let Place {
        stride,
        first,
        second,
    } = place;
    let kept = &mut buf.kept;
    kept.clear();
    kept.extend(0..len as u32);
    kept.retain(|&i| {
        let at = i as usize * stride + first;
        let flags = &rows[at..at + 4];
        flags[0] != 0 && flags[1] == 0 && flags[2] == 0 && flags[3] == 0
    });
    kept.retain(|&i| cx.overload_lifted || rows[i as usize * stride + second] == 0);
    buf.before_soft.clear();
    buf.before_soft.extend_from_slice(kept);
    kept.retain(|&i| {
        let id = WorkerId(u64::from(i));
        !facts.tried.contains(&id) && !cx.excluded.contains(&id)
    });
    if kept.is_empty() {
        kept.extend_from_slice(&buf.before_soft);
    }
    buf.scores.clear();
    buf.scores.resize(kept.len(), 0.0);
    (kept.len() as u32, kept.first().copied().unwrap_or(u32::MAX))
}

/// The bench's view build, over a model row: every field of every row
/// written in place. The engine's build pushes a row built by value; a model
/// row built that way went through the stack in five of the eight layouts
/// (a copy the engine's own build does not have), so a build cell would have
/// read the compiler's choice and not the layout.
fn build<R: Row>(fleet: &[WorkerState], out: &mut [R]) {
    for (i, (row, w)) in out.iter_mut().zip(fleet).enumerate() {
        row.write(i as u32, w);
    }
}

// ------------------------------------------------------------ the probe

/// The probe's ids are many: each is timed for two seconds, so three runs a
/// side stay inside the gate's budget.
fn short(group: &mut BenchmarkGroup<'_, WallTime>) {
    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_millis(1_500))
        .nresamples(10_000);
}

/// The runner's processor line and data cache sizes, into the run's log: a
/// cell of this probe is read beside them.
#[expect(
    clippy::print_stderr,
    reason = "the probe's log is where its cells are read from"
)]
fn runner_facts() {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    for key in [
        "vendor_id",
        "cpu family",
        "model",
        "model name",
        "stepping",
        "microcode",
        "cache size",
        "clflush size",
        "cache_alignment",
    ] {
        let line = cpuinfo
            .lines()
            .find(|line| line.split(':').next().is_some_and(|k| k.trim() == key));
        if let Some(line) = line {
            let words: Vec<&str> = line.split_whitespace().collect();
            eprintln!("PROBE runner: {}", words.join(" "));
        }
    }
    let processors = cpuinfo
        .lines()
        .filter(|line| line.starts_with("processor"))
        .count();
    eprintln!("PROBE runner: processors {processors}");
    for index in 0..6 {
        let dir = format!("/sys/devices/system/cpu/cpu0/cache/index{index}");
        let read = |name: &str| {
            std::fs::read_to_string(format!("{dir}/{name}"))
                .map(|text| text.trim().to_string())
                .unwrap_or_default()
        };
        let level = read("level");
        if level.is_empty() {
            continue;
        }
        eprintln!(
            "PROBE runner: cache level {level} {} size {} ways {} sets {} line {} shared by {}",
            read("type"),
            read("size"),
            read("ways_of_associativity"),
            read("number_of_sets"),
            read("coherency_line_size"),
            read("shared_cpu_list"),
        );
    }
}

#[expect(
    clippy::print_stderr,
    reason = "the probe's log is where its cells are read from"
)]
fn layout_line<R: Row>() {
    let [flags, id, sampled, in_flight, load, position] = R::OFFSETS;
    let healthy: Vec<bool> = fleet(512).iter().map(|w| w.healthy).collect();
    let place = Place {
        stride: size_of::<R>(),
        first: flags,
        second: flags + 4,
    };
    eprintln!(
        "PROBE layout {}: {} bytes, flags at {flags}, id at {id}, sample instant at {sampled}, \
         in-flight at {in_flight}, load at {load}, position at {position}; \
         the two flag passes touch {} lines at 512 rows",
        R::NAME,
        size_of::<R>(),
        lines_touched(place, &healthy),
    );
}

fn typed<R: Row>(c: &mut Criterion) {
    layout_line::<R>();
    let filters = standard_filters::<R>();
    for &n in &SIZES {
        let fleet = fleet(n);
        let healthy = fleet.iter().filter(|w| w.healthy).count() as u32;
        let first = fleet.iter().position(|w| w.healthy).map(|i| i as u32);
        let mut rows: Vec<R> = page_start_buffer(n);
        for (i, w) in fleet.iter().enumerate() {
            rows.push(R::fill(i as u32, w));
        }
        build(&fleet, &mut rows);
        let tried: &[WorkerId] = black_box(&[]);
        let excluded: &[WorkerId] = black_box(&[]);
        let facts = Facts {
            tried,
            prompt_tokens: 512,
            now_micros: 1_000_000,
        };
        let cx = Limits {
            excluded,
            overload_lifted: black_box(false),
        };
        let mut buf = Buffers::default();

        // The walks do what the engine's do: every healthy row survives, the
        // first of them leads an unscored pick, the trace names three.
        let unscored = decide_unscored(&filters, &facts, &cx, &rows, &mut buf);
        assert_eq!(unscored.survivors, healthy, "{}/{n}", R::NAME);
        assert_eq!(
            (unscored.len > 0).then(|| unscored.targets[0].0),
            first,
            "{}/{n}",
            R::NAME
        );
        assert_eq!(
            unscored.top_len,
            healthy.min(TRACE_TOP as u32),
            "{}/{n}",
            R::NAME
        );
        let scored = decide_scored(&filters, &facts, &cx, &rows, &mut buf);
        assert_eq!(scored.survivors, healthy, "{}/{n}", R::NAME);
        assert_eq!(
            scored.top[0].0,
            WorkerId(1_000 + u64::from(scored.targets[0].0)),
            "{}/{n}: the best score leads the pick and the trace",
            R::NAME
        );

        let mut group = c.benchmark_group("policy_engine/probe_build");
        short(&mut group);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new(R::NAME, n), &n, |b, _| {
            b.iter(|| {
                build(black_box(&fleet), &mut rows);
                black_box(&rows);
                rows.len()
            });
        });
        group.finish();
        assert!(
            rows.as_ptr().addr().is_multiple_of(PAGE),
            "{}/{n}: the build never moved its buffer",
            R::NAME
        );

        let mut group = c.benchmark_group("policy_engine/probe_pass");
        short(&mut group);
        group.bench_with_input(BenchmarkId::new(R::NAME, n), &n, |b, _| {
            b.iter(|| decide_unscored(&filters, &facts, &cx, black_box(&rows), &mut buf));
        });
        group.finish();

        if n == 512 {
            let mut group = c.benchmark_group("policy_engine/probe_score");
            short(&mut group);
            group.bench_with_input(BenchmarkId::new(R::NAME, n), &n, |b, _| {
                b.iter(|| decide_scored(&filters, &facts, &cx, black_box(&rows), &mut buf));
            });
            group.finish();
        }
    }
}

/// The untyped places. The first eight are the typed layouts' own (the
/// second filter's flag four bytes past the first's). The ninth is `h104`
/// moved four bytes down: the same stride and the same multiply, the two
/// passes' flags in one aligned 8-byte word, where at 92 and 96 the second
/// pass's flag is on the next line in one row of eight. The tenth is a
/// column of flag words of its own, 64 lines at 512 rows.
const RAW: [(usize, usize, usize); 10] = [
    (128, 112, 116),
    (128, 92, 96),
    (128, 0, 4),
    (112, 0, 4),
    (104, 92, 96),
    (104, 88, 92),
    (104, 0, 4),
    (96, 84, 88),
    (96, 0, 4),
    (8, 0, 4),
];

#[expect(
    clippy::print_stderr,
    reason = "the probe's log is where its cells are read from"
)]
fn raw(c: &mut Criterion) {
    let n = 512usize;
    let fleet = fleet(n);
    let flags: Vec<bool> = fleet.iter().map(|w| w.healthy).collect();
    let healthy = fleet.iter().filter(|w| w.healthy).count() as u32;
    let first = fleet.iter().position(|w| w.healthy).map(|i| i as u32);
    let tried: &[WorkerId] = black_box(&[]);
    let excluded: &[WorkerId] = black_box(&[]);
    let facts = Facts {
        tried,
        prompt_tokens: 512,
        now_micros: 1_000_000,
    };
    let cx = Limits {
        excluded,
        overload_lifted: black_box(false),
    };
    let mut buf = Buffers::default();
    let mut group = c.benchmark_group("policy_engine/probe_raw");
    short(&mut group);
    for stride in [128usize, 112, 104, 96, 8] {
        let mut rows: Vec<u8> = page_start_buffer(stride * n);
        for &(_, at, second) in RAW.iter().filter(|(s, _, _)| *s == stride) {
            let place = Place {
                stride,
                first: at,
                second,
            };
            rows.clear();
            rows.resize(stride * n, 0);
            for (i, w) in fleet.iter().enumerate() {
                rows[i * stride + at] = u8::from(w.healthy);
            }
            assert!(rows.as_ptr().addr().is_multiple_of(PAGE));
            let (survivors, lead) = decide_raw(&rows, place, n, &facts, &cx, &mut buf);
            assert_eq!(
                (survivors, Some(lead)),
                (healthy, first),
                "{stride} at {at} and {second}"
            );
            eprintln!(
                "PROBE raw {stride}_at_{at}_{second}: the two flag passes touch {} lines at 512 rows",
                lines_touched(place, &flags),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{stride}_at_{at}_{second}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        decide_raw(black_box(&rows), black_box(place), n, &facts, &cx, &mut buf)
                    });
                },
            );
        }
    }
    group.finish();
}

fn probe(c: &mut Criterion) {
    runner_facts();
    the_model_is_the_real_row();
    typed::<M128>(c);
    typed::<H104>(c);
    typed::<H104Hot>(c);
    typed::<H96>(c);
    typed::<H96Hot>(c);
    typed::<H112Hot>(c);
    typed::<H128Line>(c);
    typed::<H128Hot>(c);
    raw(c);
}

criterion_group!(benches, view_build, decide, probe);
criterion_main!(benches);
