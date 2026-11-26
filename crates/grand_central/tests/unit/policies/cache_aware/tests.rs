use super::*;
use crate::{
    document::spec::{PolicySpec, SpillParams},
    engine::{
        decision::OverlapSource,
        framework::{Engine, Scratch},
        prefix_pass::forget_prefix_pass,
        view::{AffinityHolder, Constraints, Health, LoadRow, PoolFacts},
    },
};

/// A row and its entry of the affinity column, as a case writes a pool.
type Held = (Candidate, AffinityFacts);

fn candidate(position: u32, in_flight: u32, overlap_blocks: u32) -> Held {
    (
        Candidate {
            position,
            id: WorkerId(u64::from(position) + 1),
            health: Health {
                healthy: true,
                ..Health::default()
            },
            in_flight,
            ..Candidate::default()
        },
        AffinityFacts {
            overlap_blocks,
            admitted_age_ms: u32::MAX,
            ..AffinityFacts::default()
        },
    )
}

fn rows(pool: &[Held]) -> Vec<Candidate> {
    pool.iter().map(|(row, _)| *row).collect()
}

fn column(pool: &[Held]) -> Vec<AffinityFacts> {
    pool.iter().map(|(_, held)| *held).collect()
}

fn kept(n: u32) -> Vec<u32> {
    (0..n).collect()
}

fn facts(blocks: usize) -> RequestFacts<'static> {
    static HASHES: [u64; 64] = [0; 64];
    RequestFacts {
        block_hashes: &HASHES[..blocks],
        prompt_tokens: 16 * blocks as u32,
        seed: 7,
        ..RequestFacts::default()
    }
}

fn affinity(spill: Option<SpillParams>) -> PrefixAffinity {
    let params = PrefixAffinityParams {
        spill,
        ..PrefixAffinityParams::default()
    };
    PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16)))
}

/// The scores of `pool`'s rows, the facts carrying the pool's column.
fn scores(scorer: &dyn Scorer, facts: &RequestFacts<'_>, pool: &[Held]) -> Vec<f32> {
    let column = column(pool);
    scores_beside(scorer, facts, &rows(pool), &column)
}

/// The scores of `rows` beside a column of the case's own length.
fn scores_beside(
    scorer: &dyn Scorer,
    facts: &RequestFacts<'_>,
    rows: &[Candidate],
    column: &[AffinityFacts],
) -> Vec<f32> {
    let facts = RequestFacts {
        affinity: column,
        ..*facts
    };
    let kept = kept(rows.len() as u32);
    let mut out = vec![0.0; rows.len()];
    scorer.score(&facts, rows, &kept, &mut out);
    out
}

#[test]
fn the_deepest_holders_rank_first_and_a_miss_leaves_every_row_alike() {
    let pool = [candidate(0, 1, 2), candidate(1, 1, 8), candidate(2, 1, 8)];
    assert_eq!(
        scores(&affinity(None), &facts(16), &pool),
        vec![0.5, 1.0, 1.0]
    );
    let miss = [candidate(0, 1, 0), candidate(1, 1, 0)];
    assert_eq!(scores(&affinity(None), &facts(16), &miss), vec![0.5, 0.5]);
}

#[test]
fn a_holder_over_the_spill_gate_yields_to_the_rows_under_it() {
    // Mean 10 in flight; the holder at 40 is over 1.5 times the mean and
    // over the mean by the margin (min(64, max(ceil(2 * sqrt(10)), 4))
    // = 7), so it spills; the other rows are under the gate.
    let pool = [candidate(0, 40, 8), candidate(1, 0, 0), candidate(2, 0, 2)];
    let gated = affinity(Some(SpillParams {
        abs: Some(64),
        rel: Some(1.5),
        ..SpillParams::default()
    }));
    let mean = mean_in_flight(&RequestFacts::default(), &rows(&pool), &kept(3));
    assert!((mean - 40.0 / 3.0).abs() < 1e-6);
    assert_eq!(scores(&gated, &facts(16), &pool), vec![0.0, 0.5, 0.5]);
    // Under a lifted relative part the holder keeps the request.
    let lifted = affinity(Some(SpillParams {
        rel: Some(4.0),
        ..SpillParams::default()
    }));
    assert_eq!(scores(&lifted, &facts(16), &pool), vec![1.0, 0.5, 0.5]);
}

#[test]
fn a_priced_prefix_keeps_its_holder_through_a_skew_the_swing_would_spill() {
    // Mean 10: the swing margin is 7 and the holder at 20 is over it. A
    // prefix of 8 blocks of 16 tokens priced at 8 tokens a request is
    // worth 16 requests, so the margin is 16 and the holder stays.
    let pool = [candidate(0, 20, 8), candidate(1, 0, 0)];
    let unpriced = affinity(Some(SpillParams {
        rel: Some(1.1),
        ..SpillParams::default()
    }));
    assert_eq!(scores(&unpriced, &facts(16), &pool), vec![0.0, 0.5]);
    let priced = affinity(Some(SpillParams {
        rel: Some(1.1),
        prefix_margin_tokens: Some(8),
        ..SpillParams::default()
    }));
    assert_eq!(scores(&priced, &facts(16), &pool), vec![1.0, 0.5]);
    // The price reads the block size the request was counted with: at
    // one token a block the eight blocks are one request, the margin
    // falls back to the swing and the holder spills; at 128 a block
    // they are 128 requests, capped at the absolute 64, and it stays.
    let narrow = RequestFacts {
        block_size: 1,
        ..facts(16)
    };
    assert_eq!(scores(&priced, &narrow, &pool), vec![0.0, 0.5]);
    let wide = RequestFacts {
        block_size: 128,
        ..facts(16)
    };
    assert_eq!(scores(&priced, &wide, &pool), vec![1.0, 0.5]);
    // A decode decision prices nothing.
    let decode = RequestFacts {
        phase: Phase::Decode,
        ..facts(16)
    };
    assert_eq!(scores(&priced, &decode, &pool), vec![0.0, 0.5]);
}

#[test]
fn a_predicted_placement_counts_only_above_the_threshold_and_the_index_at_any_depth() {
    let params = PrefixAffinityParams {
        threshold: Some(0.5),
        source: Some(AffinitySource::Merged),
        ..PrefixAffinityParams::default()
    };
    let merged = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16)));
    let predicted = |(row, held): Held, predicted_blocks: u32| {
        (
            row,
            AffinityFacts {
                predicted_blocks,
                ..held
            },
        )
    };
    let shallow = predicted(candidate(0, 0, 0), 4);
    let deep = predicted(candidate(1, 0, 0), 9);
    let indexed = candidate(2, 0, 1);
    // 4 of 16 is under the threshold, 9 of 16 over it; the index's one
    // block is a match whatever the threshold, and the deepest wins.
    assert_eq!(
        scores(&merged, &facts(16), &[shallow, deep, indexed]),
        vec![0.5, 1.0, 0.5]
    );
    assert_eq!(
        scores(&merged, &facts(16), &[shallow, indexed]),
        vec![0.5, 1.0]
    );
    // Without hashes the request is sized by its tokens.
    let by_tokens = RequestFacts {
        block_hashes: &[],
        prompt_tokens: 256,
        ..RequestFacts::default()
    };
    assert_eq!(
        scores(&merged, &by_tokens, &[shallow, deep]),
        vec![0.5, 1.0]
    );
}

/// A sent prefix's share is taken of the whole prompt, as the classic
/// policy takes it: its matched tokens over the prompt's tokens, above
/// the threshold (`model_gateway/src/policies/cache_aware.rs:3561-3569`).
/// A prompt that ends inside a block is larger than its whole blocks, so
/// a share of those would let a prefix under the threshold count. The
/// share is one division of the two counts: at the threshold itself it
/// does not clear it.
#[test]
fn a_sent_prefix_counts_by_its_share_of_the_whole_prompt() {
    static HASHES: [u64; 64] = [0; 64];
    // Whether `held` blocks of `block` tokens sent to a worker count
    // for a prompt of `tokens` tokens, hashed by its whole blocks.
    let counts = |threshold: Option<f32>, block: u32, held: u32, tokens: u32| {
        let params = PrefixAffinityParams {
            threshold,
            source: Some(AffinitySource::Predictor),
            ..PrefixAffinityParams::default()
        };
        let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(block)));
        let (row, nothing) = candidate(0, 0, 0);
        let sent = AffinityFacts {
            predicted_blocks: held,
            ..nothing
        };
        let facts = RequestFacts {
            block_hashes: &HASHES[..(tokens / block) as usize],
            prompt_tokens: tokens,
            seed: 7,
            ..RequestFacts::default()
        };
        let scored = scores(&stage, &facts, &[(row, sent), candidate(1, 0, 0)]);
        assert!(scored == [1.0, 0.5] || scored == [0.5, 0.5], "{scored:?}");
        scored[0] == 1.0
    };
    // One block of 16 tokens at the default share of 0.3: it is 0.302 of
    // 53 tokens and 0.296 of 54, and a third of either prompt's three
    // whole blocks.
    let counted = [48, 53, 54, 63, 64].map(|tokens| counts(None, 16, 1, tokens));
    assert_eq!(counted, [true, true, false, false, false]);
    // Four blocks: 0.3005 of 213 tokens, 0.2991 of 214.
    assert!(counts(None, 16, 4, 213));
    assert!(!counts(None, 16, 4, 214));
    // 72 tokens of 160 are 0.45 to the bit, and do not clear 0.45. Three
    // blocks over 160 / 24 blocks would read one bit above it.
    assert!(!counts(Some(0.45), 24, 3, 160));
    assert!(counts(Some(0.45), 24, 3, 159));
    // The whole keeps what the request's block count guards: a size
    // stated under the hashes the host sent reads as the hashes, a
    // prompt of no size as one block, and the text kind's as its bytes
    // beside a window of bytes. A request of text states its length and
    // no count of tokens: 100 two-byte characters are 200 bytes, and
    // the share is taken of the bytes.
    let whole = |facts: RequestFacts<'_>| {
        let whole = Gates::of(None, None, Some(16)).whole(&facts);
        (whole.block, whole.size)
    };
    let hashed = RequestFacts {
        block_hashes: &HASHES[..3],
        ..RequestFacts::default()
    };
    let sized = |prompt_tokens| RequestFacts {
        prompt_tokens,
        ..hashed
    };
    assert_eq!(whole(sized(54)), (16, 54));
    assert_eq!(whole(sized(20)), (16, 48));
    assert_eq!(whole(RequestFacts::default()), (16, 16));
    let text = RequestFacts {
        hash_kind: HashKind::Text,
        block_size: 64,
        prompt_chars: 100,
        prompt_bytes: 200,
        ..hashed
    };
    assert_eq!(whole(text), (64, 200));
}

/// A request the engine makes of a prompt's token ids with no block
/// hashed is sized as the one it makes with its list. The stage reads
/// the list's length alone, under a `max` with the count the prompt's
/// size gives ([`Gates::whole`], [`RequestFacts::blocks`]), and the
/// list the engine makes is never longer than that count: one hash for
/// each whole block, as deep as the document reads at most, one for a
/// shorter prompt, none for an empty one. So an empty prompt, one
/// under one block, one that ends on a block's edge, one that ends
/// inside a block and one longer than the document reads each state
/// one size both ways: the prompt's own.
#[cfg(feature = "in-process")]
#[test]
fn a_request_made_with_no_block_hashed_is_sized_as_the_one_hashed() {
    use crate::{
        document::spec::AppliedSpec,
        engine::prompt_hashes::{HashSeeds, PromptHashes},
    };
    let document = PolicySpec::preset("cache_aware").expect("a preset");
    let spec = AppliedSpec::validate(&document).expect("the preset validates");
    let gates = Gates::of(None, None, Some(16));
    // A prompt's tokens, the hashes of its list and its size in blocks.
    let prompts = [
        (0_u32, 0, 1),
        (7, 1, 1),
        (16, 1, 1),
        (54, 3, 3),
        (64, 4, 4),
        (1_600, 100, 100),
        (100_000, 256, 6_250),
    ];
    for (tokens, hashes, blocks) in prompts {
        let prompt: Vec<u32> = (0..tokens).collect();
        let listed =
            PromptHashes::under_document(&spec, Some(16), Some(&prompt), None, HashSeeds::PLAIN);
        let unhashed = PromptHashes::unhashed(&spec, Some(16), &prompt, HashSeeds::PLAIN);
        let mut with_list = RequestFacts::default();
        listed.write_to(&mut with_list);
        let mut without = RequestFacts::default();
        unhashed.write_to(&mut without);
        assert_eq!(with_list.block_hashes.len(), hashes, "{tokens}");
        assert!(without.block_hashes.is_empty(), "{tokens}");
        let counted = (with_list.blocks(16), without.blocks(16));
        assert_eq!(counted, (blocks, blocks), "{tokens}");
        let size = (16, tokens.max(16));
        let (full, bare) = (gates.whole(&with_list), gates.whole(&without));
        assert_eq!((full.block, full.size), size, "{tokens}");
        assert_eq!((bare.block, bare.size), size, "{tokens}");
    }
}

/// The engine of the built-in `cache_aware` document.
fn built_in() -> Engine {
    PolicySpec::preset("cache_aware")
        .expect("a preset")
        .validate()
        .expect("the preset validates")
}

/// The built-in document's decision over `pool`, a pool of more than
/// twice its sample of sixteen rows, for a prompt of `tokens` tokens in
/// three whole blocks whose first block was sent to row 7: the row
/// picked. The host names row 7, and the decision runs behind a draw.
fn decided_past_the_sample(engine: &Engine, pool: &[Candidate], tokens: u32, seed: u64) -> u32 {
    static HASHES: [u64; 3] = [0; 3];
    let sent = [AffinityHolder {
        row: 7,
        overlap_blocks: 0,
        predicted_blocks: 1,
    }];
    let facts = RequestFacts {
        model: "m",
        block_hashes: &HASHES,
        prompt_tokens: tokens,
        seed: Draw::new(seed).next_u64(),
        named: Some(named(&sent, &[], false)),
        ..RequestFacts::default()
    };
    let decision = engine.decide(
        &facts,
        &PoolFacts::default(),
        pool,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(decision.trace().drew_first(), "seed {seed}");
    let pick = decision.pick().expect("a pick");
    pick.targets.first().expect("a target").position
}

/// Behind a draw the share is the same one. A pool of more than twice
/// the built-in document's sample is decided over the rows drawn and the
/// rows its host named, and 16 tokens sent of a prompt of 54 hold
/// nothing there: they are under 0.3 of the prompt, and a third of its
/// three whole blocks. The classic policy divides the same two counts,
/// and under its threshold it selects with no holder named
/// (`model_gateway/src/policies/cache_aware.rs:3561-3569`, `:3579-3588`).
#[test]
fn behind_a_draw_a_sent_prefix_counts_by_its_share_of_the_whole_prompt() {
    let engine = built_in();
    // Sixty-four rows. Row 7 was sent the prefix and is the one busy
    // row: it takes a request it holds, and an idle row takes any other.
    let pool: Vec<Candidate> = (0..64)
        .map(|row| candidate(row, if row == 7 { 2 } else { 0 }, 0).0)
        .collect();
    for seed in 0..32 {
        // 0.302 of 53 tokens, 0.296 of 54.
        let held = decided_past_the_sample(&engine, &pool, 53, seed);
        assert_eq!(held, 7, "53 tokens, seed {seed}");
        let missed = decided_past_the_sample(&engine, &pool, 54, seed);
        assert_ne!(missed, 7, "54 tokens, seed {seed}");
    }
}

fn reporting(position: u32, in_flight: u32, overlap_blocks: u32, active_kv: u16) -> Held {
    let (row, held) = candidate(position, in_flight, overlap_blocks);
    (
        Candidate {
            load: Some(LoadRow {
                active_kv_permille: active_kv,
                kv_usage_permille: active_kv,
                live_rate: 1000,
                ..LoadRow::default()
            }),
            ..row
        },
        held,
    )
}

#[test]
fn under_the_fleet_kv_gate_affinity_yields_to_load() {
    let kv = KvPressureParams {
        spread: Some(0.3),
        ceiling: Some(0.9),
        ..KvPressureParams::default()
    };
    let params = PrefixAffinityParams::default();
    let gated = PrefixAffinity::new(&params, Gates::of(Some(&params), Some(&kv), Some(16)));
    let balanced = [reporting(0, 0, 8, 500), reporting(1, 0, 0, 450)];
    assert_eq!(scores(&gated, &facts(16), &balanced), vec![1.0, 0.5]);
    let spread = [reporting(0, 0, 8, 800), reporting(1, 0, 0, 300)];
    assert_eq!(scores(&gated, &facts(16), &spread), vec![1.0, 1.0]);
    let ceiling = [reporting(0, 0, 8, 950), reporting(1, 0, 0, 900)];
    assert_eq!(scores(&gated, &facts(16), &ceiling), vec![1.0, 1.0]);
    // The gate at its defaults never fires, whatever the usage.
    let off = affinity(None);
    assert_eq!(scores(&off, &facts(16), &ceiling), vec![1.0, 0.5]);
    assert!(KvPressure.needs().loads);
    assert_eq!(scores(&KvPressure, &facts(16), &ceiling), vec![1.0, 1.0]);
}

fn warmup(share: f32, divert_every: u64) -> WarmupSlice {
    let params = WarmupSliceParams {
        share: Some(share),
        divert_every: Some(divert_every),
        blocks: Some(8),
        ..WarmupSliceParams::default()
    };
    WarmupSlice::new(&params, Gates::of(None, None, Some(16)))
}

fn sized(position: u32, in_flight: u32, overlap_blocks: u32, indexed: u32) -> Held {
    let (row, held) = candidate(position, in_flight, overlap_blocks);
    (
        row,
        AffinityFacts {
            indexed_blocks: indexed,
            indexed_growth: indexed,
            ..held
        },
    )
}

#[test]
fn one_thin_miss_in_a_period_goes_to_the_least_loaded_thin_worker() {
    let slice = warmup(0.25, 0);
    // Level 64 (the upper median of 64, 64, 0): the empty worker is thin;
    // nobody holds the request.
    let pool = [sized(0, 3, 0, 64), sized(1, 2, 0, 64), sized(2, 5, 0, 0)];
    let taken: Vec<bool> = (0..8)
        .map(|_| scores(&slice, &facts(16), &pool) == vec![0.0, 0.0, 1.0])
        .collect();
    assert_eq!(
        taken,
        vec![true, false, false, false, true, false, false, false]
    );
    // A settled fleet moves no counter and scores alike.
    let settled = [sized(0, 3, 0, 64), sized(1, 2, 0, 64)];
    assert_eq!(scores(&slice, &facts(16), &settled), vec![1.0, 1.0]);
    assert_eq!(slice.misses.load(Ordering::Relaxed), 8);
}

/// Two thin workers part by their requests in flight: a sliced miss
/// goes to the thin worker that holds the fewest.
#[test]
fn a_thin_miss_goes_to_the_thin_worker_with_the_fewest_requests() {
    let slice = warmup(0.25, 0);
    // Level 64 (the median of 64, 64, 64, 0, 0): rows 3 and 4 are thin,
    // and row 4 has the fewer requests.
    let pool = [
        sized(0, 3, 0, 64),
        sized(1, 2, 0, 64),
        sized(2, 4, 0, 64),
        sized(3, 5, 0, 0),
        sized(4, 1, 0, 0),
    ];
    let column = column(&pool);
    let rows = rows(&pool);
    let facts = RequestFacts {
        affinity: &column,
        ..facts(16)
    };
    let kept = kept(rows.len() as u32);
    let mut out = vec![0.0; rows.len()];
    slice.score(&facts, &rows, &kept, &mut out);
    assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0, 1.0]);
}

/// A few-block hit on a long request is a miss for the slice: at most
/// four blocks held that cover less than half the request. A fifth
/// block, or half the request, is a hit the slice leaves alone.
#[test]
fn a_few_block_hit_on_a_long_request_is_sliced_as_a_miss() {
    let params = WarmupSliceParams {
        share: Some(1.0),
        divert_every: Some(0),
        blocks: Some(8),
        miss_share: Some(0.0),
        ..WarmupSliceParams::default()
    };
    let slice = WarmupSlice::new(&params, Gates::of(None, None, Some(16)));
    let held = |blocks: u32| {
        [
            sized(0, 3, blocks, 64),
            sized(1, 2, 0, 64),
            sized(2, 5, 0, 0),
        ]
    };
    // Three of sixteen blocks: a miss, the thin worker takes it.
    assert_eq!(scores(&slice, &facts(16), &held(3)), vec![0.0, 0.0, 1.0]);
    // Five blocks: over the few-block bound, a hit.
    assert_eq!(scores(&slice, &facts(16), &held(5)), vec![1.0, 1.0, 1.0]);
    // Four of eight blocks cover half the request: a hit; three of
    // eight do not: a miss.
    assert_eq!(scores(&slice, &facts(8), &held(4)), vec![1.0, 1.0, 1.0]);
    assert_eq!(scores(&slice, &facts(8), &held(3)), vec![0.0, 0.0, 1.0]);
}

/// The host caps the hash list it sends: a request past the cap keeps
/// its token size, so a few-block match on it is not read as half of it.
#[test]
fn a_request_past_the_hash_cap_keeps_its_token_size() {
    let params = WarmupSliceParams {
        share: Some(1.0),
        divert_every: Some(0),
        blocks: Some(8),
        miss_share: Some(0.0),
        ..WarmupSliceParams::default()
    };
    let slice = WarmupSlice::new(&params, Gates::of(None, None, Some(16)));
    let held = |blocks: u32| {
        [
            sized(0, 3, blocks, 64),
            sized(1, 2, 0, 64),
            sized(2, 5, 0, 0),
        ]
    };
    // Eight hashes of a sixteen-block request: four blocks are a
    // quarter of it, a miss; of the hashes alone they would read half.
    let mut capped = facts(8);
    capped.prompt_tokens = 16 * 16;
    assert_eq!(scores(&slice, &capped, &held(4)), vec![0.0, 0.0, 1.0]);
    assert_eq!(scores(&slice, &capped, &held(8)), vec![1.0, 1.0, 1.0]);
    // A text-kind request states its length and no count of tokens.
    // Its windows are bytes: 128 two-byte characters are 256 bytes,
    // sixteen windows of sixteen, the same quarter. Read by its
    // characters the request would be eight windows, and four of them
    // half of it.
    let mut text = facts(8);
    text.hash_kind = HashKind::Text;
    (text.prompt_tokens, text.prompt_chars, text.prompt_bytes) = (0, 128, 256);
    assert_eq!(scores(&slice, &text, &held(4)), vec![0.0, 0.0, 1.0]);
}

/// A diverting hit reserves its thin worker on the lock turn that found
/// it eligible: the window is on the slot before the lock is let go.
#[test]
fn a_diverting_hit_reserves_its_thin_worker_on_the_same_turn() {
    let slice = warmup(0.25, 4);
    let pool = [sized(0, 2, 12, 64), sized(1, 1, 0, 64), sized(2, 0, 0, 0)];
    let mut now = facts(16);
    now.now_micros = 1_000;
    for _ in 0..3 {
        assert_eq!(scores(&slice, &now, &pool), vec![1.0, 1.0, 1.0]);
        assert!(slice.divert_until.lock().is_empty());
    }
    assert_eq!(scores(&slice, &now, &pool), vec![0.0, 0.0, 1.0]);
    assert_eq!(
        slice.divert_until.lock().get(&pool[2].0.id).copied(),
        Some(1_000 + DIVERT_WINDOW_MICROS),
        "the diverted worker holds its window from the turn that picked it"
    );
}

#[test]
fn a_young_fleet_slices_nothing_and_a_young_worker_among_warm_ones_takes_its_share() {
    let slice = warmup(1.0, 0);
    let (row, held) = sized(2, 0, 0, 2);
    let young = (
        row,
        AffinityFacts {
            admitted_age_ms: 1_000,
            indexed_growth: 2,
            ..held
        },
    );
    let all_young = [young, young, young];
    assert_eq!(scores(&slice, &facts(16), &all_young), vec![1.0, 1.0, 1.0]);
    // No size fed (a fleet below the level that makes anyone thin), one
    // young worker: the age rule alone picks it on every miss at share 1.
    let mixed = [sized(0, 1, 0, 0), sized(1, 1, 0, 0), young];
    assert_eq!(scores(&slice, &facts(16), &mixed), vec![0.0, 0.0, 1.0]);
    // A hit (a deep match on another worker) is not sliced.
    let hit = [sized(0, 1, 12, 0), sized(1, 1, 0, 0), young];
    assert_eq!(scores(&slice, &facts(16), &hit), vec![1.0, 1.0, 1.0]);
}

#[test]
fn the_median_sets_the_thin_boundary_when_one_index_is_large() {
    let slice = warmup(1.0, 0);
    let pool = [sized(0, 0, 0, 32), sized(1, 0, 0, 64), sized(2, 0, 0, 1024)];
    assert_eq!(scores(&slice, &facts(16), &pool), vec![1.0; 3]);
    assert_eq!(slice.misses.load(Ordering::Relaxed), 0);
}

#[test]
fn one_hit_in_divert_every_goes_to_a_thin_worker_within_its_window() {
    let slice = warmup(0.25, 4);
    // A deep hit on worker 0 (12 of 16 blocks); worker 2 is thin and idle.
    let pool = [sized(0, 2, 12, 64), sized(1, 1, 0, 64), sized(2, 0, 0, 0)];
    let mut now = facts(16);
    now.now_micros = 1_000;
    let diverted: Vec<bool> = (0..8)
        .map(|_| scores(&slice, &now, &pool) == vec![0.0, 0.0, 1.0])
        .collect();
    // A deep hit is due once a window of divert_every hits passed
    // without a shallow one: the fourth hit diverts, and the fourth
    // after it.
    assert_eq!(
        diverted,
        vec![false, false, false, true, false, false, false, true]
    );
    // Within the window a thin worker with a request in flight takes no
    // second hit; past the window it does.
    let busy = [sized(0, 2, 12, 64), sized(1, 1, 0, 64), sized(2, 1, 0, 0)];
    for _ in 0..8 {
        assert_eq!(scores(&slice, &now, &busy), vec![1.0, 1.0, 1.0]);
    }
    now.now_micros = 1_000 + DIVERT_WINDOW_MICROS;
    let later: Vec<bool> = (0..8)
        .map(|_| scores(&slice, &now, &busy) == vec![0.0, 0.0, 1.0])
        .collect();
    assert!(later.iter().any(|&d| d), "{later:?}");
    // A shallow hit (7 of 16: over the miss share, at most half) diverts
    // at divert_every hits.
    let fresh = warmup(0.25, 4);
    let shallow = [sized(0, 2, 7, 64), sized(1, 1, 0, 64), sized(2, 0, 0, 0)];
    let hits: Vec<bool> = (0..4)
        .map(|_| scores(&fresh, &now, &shallow) == vec![0.0, 0.0, 1.0])
        .collect();
    assert_eq!(hits, vec![false, false, false, true]);
    // The worker's window leaves with it.
    fresh.on_membership(&Membership::Removed(WorkerId(3)));
    assert!(fresh.divert_until.lock().is_empty());
}

#[test]
fn a_maximal_diversion_period_does_not_overflow() {
    let slice = warmup(0.25, (1_u64 << 63) + 1);
    let shallow = [sized(0, 2, 7, 64), sized(1, 1, 0, 64), sized(2, 0, 0, 0)];
    for _ in 0..2 {
        assert_eq!(scores(&slice, &facts(16), &shallow), vec![1.0, 1.0, 1.0]);
    }
}

#[test]
fn a_deep_hit_is_not_diverted_to_a_thin_worker_busier_than_its_holder() {
    let slice = warmup(0.25, 1);
    // The holder has nothing in flight; the thin worker has one: a deep
    // hit stays, a shallow one moves.
    let deep = [sized(0, 0, 12, 64), sized(1, 0, 0, 64), sized(2, 1, 0, 0)];
    let mut now = facts(16);
    now.now_micros = 5_000_000;
    for _ in 0..3 {
        assert_eq!(scores(&slice, &now, &deep), vec![1.0, 1.0, 1.0]);
    }
    let shallow = [sized(0, 0, 7, 64), sized(1, 0, 0, 64), sized(2, 1, 0, 0)];
    assert_eq!(scores(&slice, &now, &shallow), vec![0.0, 0.0, 1.0]);
    // A holder with more in flight than the thin worker gives its due
    // deep hit up: the holder's count is read as the thin worker's is.
    let busier = [sized(0, 4, 12, 64), sized(1, 0, 0, 64), sized(2, 1, 0, 0)];
    assert_eq!(scores(&warmup(0.25, 1), &now, &busier), vec![0.0, 0.0, 1.0]);
}

/// A thin worker among a deep hit's deepest holders recomputes nothing:
/// the due hit goes to it however busy it stands beside the other
/// holders, on either feed.
#[test]
fn a_thin_worker_among_the_deepest_holders_takes_a_due_hit_however_busy() {
    // Rows 0 and 2 hold the deepest match; row 2 is thin and the busier.
    let pool = [sized(0, 0, 12, 64), sized(1, 0, 0, 64), sized(2, 3, 12, 12)];
    let sizes = column(&pool);
    let holders = [holder(0, 12), holder(2, 12)];
    let by_column = RequestFacts {
        affinity: &sizes,
        now_micros: 5_000_000,
        ..facts(16)
    };
    let by_named = RequestFacts {
        named: Some(named(&holders, &[2], true)),
        now_micros: 5_000_000,
        ..facts(16)
    };
    for (facts, feed) in [(&by_column, "the column"), (&by_named, "the rows named")] {
        let mut out = [0.0; 3];
        warmup(0.25, 1).score(facts, &rows(&pool), &[0, 1, 2], &mut out);
        assert_eq!(out, [0.0, 0.0, 1.0], "{feed}");
    }
}

/// The stages read the column and never the row: a decision fed no
/// column reads as a miss for every row, and leaves the warm-up stage
/// neutral before any counter moves, the answer its pass gives over rows
/// that all read as just admitted.
#[test]
fn no_column_is_a_miss_for_every_row_and_slices_nothing() {
    let pool = [candidate(0, 1, 8), candidate(1, 1, 8), candidate(2, 1, 8)];
    let (held_rows, held_column) = (rows(&pool), column(&pool));
    let stage = affinity(None);
    assert_eq!(
        scores_beside(&stage, &facts(16), &held_rows, &held_column),
        vec![1.0, 1.0, 1.0]
    );
    assert_eq!(
        scores_beside(&stage, &facts(16), &held_rows, &[]),
        vec![0.5, 0.5, 0.5],
        "no column: a miss for every row, the spill gate alone orders"
    );
    // The warm-up stage: a thin worker takes a miss beside its column,
    // and nothing is sliced without one.
    let slice = warmup(1.0, 0);
    let thin = [sized(0, 3, 0, 64), sized(1, 2, 0, 64), sized(2, 5, 0, 0)];
    let (thin_rows, thin_column) = (rows(&thin), column(&thin));
    assert_eq!(
        scores_beside(&slice, &facts(16), &thin_rows, &[]),
        vec![1.0, 1.0, 1.0]
    );
    assert_eq!(
        scores_beside(
            &slice,
            &facts(16),
            &thin_rows,
            &[AffinityFacts::default(); 3]
        ),
        vec![1.0, 1.0, 1.0],
        "rows that all read as just admitted are a young fleet"
    );
    assert_eq!(slice.misses.load(Ordering::Relaxed), 0);
    assert_eq!(
        scores_beside(&slice, &facts(16), &thin_rows, &thin_column),
        vec![0.0, 0.0, 1.0]
    );
    assert_eq!(slice.misses.load(Ordering::Relaxed), 1);
}

#[test]
fn the_gates_read_their_parameters_and_their_defaults() {
    let gates = Gates::of(None, None, None);
    assert_eq!(gates, Gates::default());
    assert!(!gates.kv_configured());
    assert_eq!(gates.margin(0.0, 0.0), 4.0);
    assert_eq!(gates.margin(100.0, 0.0), 20.0);
    assert_eq!(gates.margin(10_000.0, 0.0), 64.0);
    let params = PrefixAffinityParams {
        spill: Some(SpillParams {
            abs: Some(8),
            rel: Some(2.0),
            prefix_margin_tokens: Some(16),
            uncached_credit: None,
        }),
        ..PrefixAffinityParams::default()
    };
    let kv = KvPressureParams {
        spread: Some(0.25),
        ..KvPressureParams::default()
    };
    let tuned = Gates::of(Some(&params), Some(&kv), Some(32));
    assert_eq!(tuned.abs, 8);
    assert_eq!(tuned.rel, 2.0);
    assert_eq!(tuned.prefix_margin_tokens, 16);
    assert_eq!(tuned.kv_spread, 0.25);
    assert_eq!(tuned.kv_ceiling, 1.0);
    assert_eq!(tuned.block_size, 32);
    assert!(tuned.kv_configured());
    // Eight blocks of 32 tokens at 16 a request price at 16 requests,
    // capped at the configured 8.
    let tokens = RequestFacts::default();
    assert_eq!(tuned.margin(0.0, tuned.tokens_of(8, &tokens)), 8.0);
    let text = RequestFacts {
        hash_kind: HashKind::Text,
        ..RequestFacts::default()
    };
    assert_eq!(tuned.tokens_of(8, &text), 64.0);
    // The block size the host counted with (an index's learned 128)
    // overrides the view's: eight blocks are 1024 tokens, and a hashless
    // request of 1024 tokens is eight blocks, not the view's 32.
    let learned = RequestFacts {
        block_size: 128,
        prompt_tokens: 1024,
        ..RequestFacts::default()
    };
    assert_eq!(tuned.tokens_of(8, &learned), 1024.0);
    assert_eq!(
        tuned.request_blocks(&learned),
        8,
        "1024 tokens at the learned 128 a block, not 32 at the view's 32"
    );
}

/// One index, one worker: with the pool's FIRST rows dropped, each kept
/// row scores by its own entry of the column, not by the entry at its
/// place among the kept.
#[test]
fn a_narrowed_pool_scores_each_kept_row_by_its_own_entry() {
    // The dropped rows hold the deepest matches; of the two kept, the
    // last holds more.
    let pool = [
        candidate(0, 1, 8),
        candidate(1, 1, 6),
        candidate(2, 1, 2),
        candidate(3, 1, 4),
    ];
    let column = column(&pool);
    let facts = RequestFacts {
        affinity: &column,
        ..facts(16)
    };
    let mut out = [0.0; 2];
    affinity(None).score(&facts, &rows(&pool), &[2, 3], &mut out);
    assert_eq!(out, [0.5, 1.0]);
}

/// The same for the slice: with the pool's FIRST row dropped, the thin
/// worker among the kept is found by its own entry.
#[test]
fn a_narrowed_pool_finds_the_thin_worker_by_its_own_entry() {
    let params = WarmupSliceParams {
        share: Some(1.0),
        divert_every: Some(0),
        blocks: Some(8),
        miss_share: Some(0.0),
        ..WarmupSliceParams::default()
    };
    let slice = WarmupSlice::new(&params, Gates::of(None, None, Some(16)));
    // Thin at the first and at the last index; the first is dropped.
    let pool = [
        sized(0, 5, 0, 0),
        sized(1, 3, 0, 64),
        sized(2, 2, 0, 64),
        sized(3, 4, 0, 0),
    ];
    let column = column(&pool);
    let facts = RequestFacts {
        affinity: &column,
        ..facts(16)
    };
    let mut out = [0.0; 3];
    slice.score(&facts, &rows(&pool), &[1, 2, 3], &mut out);
    assert_eq!(out, [0.0, 0.0, 1.0]);
}

/// A decision fed no column reads as one whose column holds nothing for
/// every candidate: the whole score vector of each stage is the same,
/// and it is the vector the stage wrote when the five facts were fields
/// of the row and every one of them was zero.
#[test]
fn an_empty_column_scores_as_a_column_of_nothing_held() {
    let rows: Vec<Candidate> = (0..)
        .zip([3, 0, 7, 1, 4])
        .map(|(position, in_flight)| Candidate {
            position,
            id: WorkerId(u64::from(position) + 1),
            health: Health {
                healthy: true,
                ..Health::default()
            },
            in_flight,
            ..Candidate::default()
        })
        .collect();
    let nothing = vec![AffinityFacts::default(); rows.len()];
    let prefix = affinity(None);
    let slice = warmup(1.0, 0);
    let stages: [(&dyn Scorer, [f32; 5]); 3] = [
        (&prefix, [0.5; 5]),
        (&KvPressure, [1.0; 5]),
        (&slice, [1.0; 5]),
    ];
    for (stage, over_zero_fields) in stages {
        let empty = scores_beside(stage, &facts(16), &rows, &[]);
        let zeros = scores_beside(stage, &facts(16), &rows, &nothing);
        assert_eq!(empty, zeros, "{}", stage.name());
        assert_eq!(empty, over_zero_fields, "{}", stage.name());
    }
}

/// The warm-up stage takes one entry a candidate. A column that ends
/// before the candidates is a host's mistake and fails the stage, as an
/// index past its end does: walked as it stands it would read as a
/// smaller fleet.
#[test]
#[should_panic(expected = "out of range")]
fn the_warmup_stage_fails_on_a_column_that_ends_before_the_candidates() {
    let pool = [sized(0, 0, 0, 4096), sized(1, 0, 0, 4096)];
    let column = column(&pool[..1]);
    scores_beside(&warmup(1.0, 0), &facts(16), &rows(&pool), &column);
}

/// What a stage reads of the host it declares under its own name. The
/// affinity stage reads the column on every source: the predictor's
/// placements are entries of it, and that source asks for no events. The
/// warm-up stage reads it too. The KV gate reads loads alone.
#[test]
fn every_reader_of_the_column_declares_it_whatever_feeds_it() {
    for source in [
        AffinitySource::Index,
        AffinitySource::Predictor,
        AffinitySource::Merged,
    ] {
        let params = PrefixAffinityParams {
            source: Some(source),
            ..PrefixAffinityParams::default()
        };
        let needs = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16))).needs();
        assert!(needs.affinity, "{source:?} reads the column");
        assert_eq!(
            needs.events,
            source != AffinitySource::Predictor,
            "{source:?}: events feed the index, they do not name the column"
        );
        let placements = source != AffinitySource::Index;
        assert_eq!(
            (needs.sent_prefixes, needs.text_hashes, needs.text),
            (placements, placements, false),
            "{source:?}: text windows are read by the record, and none asks for the text"
        );
    }
    assert!(warmup(1.0, 0).needs().affinity);
    assert!(!KvPressure.needs().affinity);
}

/// The two stages that read the prompt's size declare it, under every
/// source: a host may state zero where no stage declares the size
/// ([`Needs::prompt_size`]), and a zero size reads every sent prefix as
/// the whole prompt. The classic policy is handed the prompt's length
/// always (`model_gateway/src/policies/cache_aware.rs:3474-3482`).
#[test]
fn the_stages_that_read_the_prompts_size_declare_it() {
    for source in [
        AffinitySource::Index,
        AffinitySource::Predictor,
        AffinitySource::Merged,
    ] {
        let params = PrefixAffinityParams {
            source: Some(source),
            ..PrefixAffinityParams::default()
        };
        let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16)));
        assert!(stage.needs().prompt_size, "{source:?}");
    }
    assert!(warmup(1.0, 0).needs().prompt_size);
    assert!(!KvPressure.needs().prompt_size);
}

/// `row` named as holding `blocks` by the index.
fn holder(row: u32, blocks: u32) -> AffinityHolder {
    AffinityHolder {
        row,
        overlap_blocks: blocks,
        predicted_blocks: 0,
    }
}

/// The rows a host names: the `holders`, and the `warming` rows, thin
/// or young.
fn named<'a>(holders: &'a [AffinityHolder], warming: &'a [u32], thin: bool) -> AffinityRows<'a> {
    AffinityRows {
        holders,
        warming,
        thin,
        mean_in_flight: None,
        tree: None,
    }
}

/// The scores of `rows` beside the rows a host named and no column.
fn scores_named(
    scorer: &dyn Scorer,
    facts: &RequestFacts<'_>,
    rows: &[Candidate],
    named: AffinityRows<'_>,
) -> Vec<f32> {
    let facts = RequestFacts {
        named: Some(named),
        ..*facts
    };
    let kept = kept(rows.len() as u32);
    let mut out = vec![0.0; rows.len()];
    scorer.score(&facts, rows, &kept, &mut out);
    out
}

/// Under no cap the two feeds are still one reading: a profile that
/// takes no sample is named every holder and every warming row. Over a
/// pool where thirty-three of forty rows hold the request, more than
/// any sample names, and two rows are thin, each stage scores the rows
/// named as the column they were derived from, turn by turn.
#[test]
fn every_holder_named_scores_as_the_column_past_any_cap() {
    // Rows 0 to 23 hold the deepest match and rows 24 to 31 a shallower
    // one; rows 38 and 39 are thin, and row 38 holds the deepest too.
    let pool: Vec<Held> = (0..40u32)
        .map(|row| match row {
            0..=23 => sized(row, row % 3, 12, 64),
            24..=31 => sized(row, row % 3, 4, 64),
            38 => sized(row, 1, 12, 12),
            39 => sized(row, 0, 0, 12),
            _ => sized(row, row % 3, 0, 64),
        })
        .collect();
    let each: Vec<AffinityHolder> = pool
        .iter()
        .filter(|(_, held)| held.overlap_blocks > 0)
        .map(|(row, held)| holder(row.position, held.overlap_blocks))
        .collect();
    assert_eq!(each.len(), 33);
    let stage = affinity(None);
    assert_eq!(
        scores_named(
            &stage,
            &facts(16),
            &rows(&pool),
            named(&each, &[38, 39], true)
        ),
        scores(&stage, &facts(16), &pool)
    );
    let (by_column, by_named) = (warmup(0.25, 1), warmup(0.25, 1));
    for turn in 0..8 {
        assert_eq!(
            scores_named(
                &by_named,
                &facts(16),
                &rows(&pool),
                named(&each, &[38, 39], true)
            ),
            scores(&by_column, &facts(16), &pool),
            "turn {turn}"
        );
    }
}

/// The two feeds are one reading: over one pool, the rows a host named
/// (each holder with what it holds, the warming rows) score as the
/// column they were derived from, turn by turn and whichever rows a
/// filter kept.
#[test]
fn rows_a_host_named_score_as_the_column_they_were_derived_from() {
    // Two holders of the deepest match, a shallower holder, a miss.
    let pool = [
        candidate(0, 1, 8),
        candidate(1, 1, 2),
        candidate(2, 1, 8),
        candidate(3, 1, 0),
    ];
    let each = [holder(0, 8), holder(1, 2), holder(2, 8)];
    let stage = affinity(None);
    let by_column = scores(&stage, &facts(16), &pool);
    assert_eq!(by_column, vec![1.0, 0.5, 1.0, 0.5]);
    assert_eq!(
        scores_named(&stage, &facts(16), &rows(&pool), named(&each, &[], false)),
        by_column
    );
    // A filter drops the deepest holders (a retry's tried rows): the
    // shallower holder is the deepest kept and leads on either feed.
    let over = |facts: RequestFacts<'_>| {
        let mut out = vec![0.0; 2];
        stage.score(&facts, &rows(&pool), &[1, 3], &mut out);
        out
    };
    let entries = column(&pool);
    let by_column = over(RequestFacts {
        affinity: &entries,
        ..facts(16)
    });
    assert_eq!(by_column, vec![1.0, 0.5]);
    assert_eq!(
        over(RequestFacts {
            named: Some(named(&each, &[], false)),
            ..facts(16)
        }),
        by_column
    );
    // A miss names no holder.
    let miss = [candidate(0, 1, 0), candidate(1, 1, 0)];
    assert_eq!(
        scores_named(&stage, &facts(16), &rows(&miss), named(&[], &[], false)),
        scores(&stage, &facts(16), &miss)
    );
    // A placement the predictor made rides a named row as it rides a
    // column's entry, for a source that reads it.
    let params = PrefixAffinityParams {
        threshold: Some(0.5),
        source: Some(AffinitySource::Merged),
        ..PrefixAffinityParams::default()
    };
    let merged = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16)));
    let placed = |(row, held): Held, predicted_blocks: u32| {
        (
            row,
            AffinityFacts {
                predicted_blocks,
                ..held
            },
        )
    };
    let pool = [
        placed(candidate(0, 0, 0), 4),
        placed(candidate(1, 0, 0), 9),
        candidate(2, 0, 1),
    ];
    let each = [
        AffinityHolder {
            row: 0,
            overlap_blocks: 0,
            predicted_blocks: 4,
        },
        AffinityHolder {
            row: 1,
            overlap_blocks: 0,
            predicted_blocks: 9,
        },
        holder(2, 1),
    ];
    let by_column = scores(&merged, &facts(16), &pool);
    assert_eq!(by_column, vec![0.5, 1.0, 0.5]);
    assert_eq!(
        scores_named(&merged, &facts(16), &rows(&pool), named(&each, &[], false)),
        by_column
    );
    // One thin worker: a miss in a period goes to it on either feed.
    let thin = [sized(0, 3, 0, 64), sized(1, 2, 0, 64), sized(2, 5, 0, 0)];
    let (by_column, by_named) = (warmup(0.25, 0), warmup(0.25, 0));
    for turn in 0..8 {
        assert_eq!(
            scores_named(&by_named, &facts(16), &rows(&thin), named(&[], &[2], true)),
            scores(&by_column, &facts(16), &thin),
            "a miss, turn {turn}"
        );
    }
    assert_eq!(by_named.misses.load(Ordering::Relaxed), 8);
    // Three thin workers: the miss goes to the least loaded one, the
    // middle of the warming rows named.
    let thin = [
        sized(0, 3, 0, 64),
        sized(1, 2, 0, 64),
        sized(2, 2, 0, 64),
        sized(3, 4, 0, 64),
        sized(4, 6, 0, 0),
        sized(5, 1, 0, 0),
        sized(6, 7, 0, 0),
    ];
    let (by_column, by_named) = (warmup(0.25, 0), warmup(0.25, 0));
    let mut taken = 0;
    for turn in 0..8 {
        let scored = scores_named(
            &by_named,
            &facts(16),
            &rows(&thin),
            named(&[], &[4, 5, 6], true),
        );
        assert_eq!(
            scored,
            scores(&by_column, &facts(16), &thin),
            "three thin rows, turn {turn}"
        );
        taken += usize::from(scored == vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }
    assert!(taken > 0, "the middle warming row takes a miss");
    // A deep hit on worker 0 is diverted to the thin worker on the same
    // turns.
    let hit = [sized(0, 2, 12, 64), sized(1, 1, 0, 64), sized(2, 0, 0, 0)];
    let (by_column, by_named) = (warmup(0.25, 4), warmup(0.25, 4));
    let mut now = facts(16);
    now.now_micros = 1_000;
    let mut diverted = 0;
    for turn in 0..8 {
        let scored = scores_named(
            &by_named,
            &now,
            &rows(&hit),
            named(&[holder(0, 12)], &[2], true),
        );
        assert_eq!(scored, scores(&by_column, &now, &hit), "a hit, turn {turn}");
        diverted += usize::from(scored == vec![0.0, 0.0, 1.0]);
    }
    assert_eq!(diverted, 2);
    // Young rows among warm ones take the misses and no diverted hit.
    let young = [sized(0, 1, 0, 0), sized(1, 1, 0, 0), sized(2, 0, 0, 2)];
    let slice = warmup(1.0, 1);
    assert_eq!(
        scores_named(&slice, &facts(16), &rows(&young), named(&[], &[2], false)),
        vec![0.0, 0.0, 1.0]
    );
    assert_eq!(
        scores_named(
            &slice,
            &facts(16),
            &rows(&young),
            named(&[holder(0, 12)], &[2], false)
        ),
        vec![1.0, 1.0, 1.0]
    );
}

/// Kept rows a narrowing stage swapped: a pass's walk starts over at a
/// row behind its cursor, so each kept row scores by its own entry on
/// either feed, at the prefix stage and at the slice.
#[test]
fn kept_rows_out_of_order_score_alike_on_either_feed() {
    let pool = [
        candidate(0, 1, 8),
        candidate(1, 1, 6),
        candidate(2, 1, 2),
        candidate(3, 1, 4),
    ];
    let depths = column(&pool);
    let holders = [holder(0, 8), holder(1, 6), holder(2, 2), holder(3, 4)];
    let by_column = RequestFacts {
        affinity: &depths,
        ..facts(16)
    };
    let by_named = RequestFacts {
        named: Some(named(&holders, &[], false)),
        ..facts(16)
    };
    for (facts, feed) in [(&by_column, "the column"), (&by_named, "the rows named")] {
        let mut out = [0.0; 3];
        affinity(None).score(facts, &rows(&pool), &[3, 1, 2], &mut out);
        assert_eq!(out, [0.5, 1.0, 0.5], "{feed}");
    }
    // Two thin workers, the less loaded behind the other among the kept.
    let pool = [
        sized(0, 5, 0, 64),
        sized(1, 1, 0, 0),
        sized(2, 2, 0, 64),
        sized(3, 4, 0, 0),
    ];
    let sizes = column(&pool);
    let by_column = RequestFacts {
        affinity: &sizes,
        ..facts(16)
    };
    let by_named = RequestFacts {
        named: Some(named(&[], &[1, 3], true)),
        ..facts(16)
    };
    for (facts, feed) in [(&by_column, "the column"), (&by_named, "the rows named")] {
        let mut out = [0.0; 3];
        warmup(1.0, 0).score(facts, &rows(&pool), &[3, 1, 2], &mut out);
        assert_eq!(out, [0.0, 1.0, 0.0], "{feed}");
    }
}

/// Two arms of the rule that tells who warms and who takes a miss: a
/// fleet fed no index size still warms its young worker, and a warming
/// worker over the load gate takes no miss.
#[test]
fn a_young_worker_warms_with_no_size_fed_and_none_over_the_load_gate_takes_a_miss() {
    let (row, held) = sized(2, 0, 0, 0);
    let young = (
        row,
        AffinityFacts {
            admitted_age_ms: 1_000,
            ..held
        },
    );
    let unfed = [sized(0, 1, 0, 0), sized(1, 1, 0, 0), young];
    assert_eq!(
        scores(&warmup(1.0, 0), &facts(16), &unfed),
        vec![0.0, 0.0, 1.0]
    );
    // One thin worker beside nine warm ones: level with them it takes
    // the miss alone, far over the load gate it takes none.
    let fleet = |load: u32| {
        let mut pool: Vec<Held> = (0..9).map(|at| sized(at, 10, 0, 64)).collect();
        pool.push(sized(9, load, 0, 0));
        pool
    };
    let mut alone = vec![0.0; 10];
    alone[9] = 1.0;
    assert_eq!(scores(&warmup(1.0, 0), &facts(16), &fleet(10)), alone);
    assert_eq!(
        scores(&warmup(1.0, 0), &facts(16), &fleet(1_000)),
        vec![1.0; 10]
    );
}

/// A host that names no warming row ruled that nobody warms: the stage
/// is neutral before any counter moves, hit or miss.
#[test]
fn no_warming_row_named_slices_nothing_and_moves_no_counter() {
    let slice = warmup(1.0, 1);
    let pool = [candidate(0, 3, 0), candidate(1, 2, 0), candidate(2, 5, 0)];
    for holders in [&[][..], &[holder(1, 9)][..]] {
        assert_eq!(
            scores_named(&slice, &facts(16), &rows(&pool), named(holders, &[], true)),
            vec![1.0, 1.0, 1.0]
        );
    }
    assert_eq!(
        (
            slice.misses.load(Ordering::Relaxed),
            slice.hits.load(Ordering::Relaxed)
        ),
        (0, 0)
    );
}

/// Behind a draw the rows handed are the drawn rows and the rows a host
/// named. The gate's mean is the drawn rows' alone: a busy holder lifts
/// a mean taken over the rows handed, and passes the gate it is judged
/// by.
#[test]
fn the_spill_gate_reads_the_drawn_rows_mean_where_the_engine_set_one() {
    // The holder has 8 in flight and the two drawn rows 2 each. Over the
    // rows handed the mean is 4: 8 is not past 4 + 4, the holder stays.
    // Over the drawn rows it is 2: 8 is past 1.5 times 2 and past 2 + 4,
    // the holder spills.
    let pool = rows(&[candidate(0, 8, 0), candidate(1, 2, 0), candidate(2, 2, 0)]);
    let stage = affinity(None);
    let holders = [holder(0, 8)];
    let held = named(&holders, &[], false);
    assert_eq!(
        scores_named(&stage, &facts(16), &pool, held),
        vec![1.0, 0.5, 0.5]
    );
    let behind_a_draw = AffinityRows {
        mean_in_flight: Some(2.0),
        ..held
    };
    assert_eq!(
        scores_named(&stage, &facts(16), &pool, behind_a_draw),
        vec![0.0, 0.5, 0.5]
    );
    // The slice's gate reads the same mean: a warming row that busy
    // takes a miss by the mean of the rows handed, and none by the
    // drawn rows'.
    let warming = named(&[], &[0], true);
    assert_eq!(
        scores_named(&warmup(1.0, 0), &facts(16), &pool, warming),
        vec![1.0, 0.0, 0.0]
    );
    let behind_a_draw = AffinityRows {
        mean_in_flight: Some(2.0),
        ..warming
    };
    let slice = warmup(1.0, 0);
    assert_eq!(
        scores_named(&slice, &facts(16), &pool, behind_a_draw),
        vec![1.0, 1.0, 1.0]
    );
    assert_eq!(slice.misses.load(Ordering::Relaxed), 0);
}

/// Behind a draw the rows handed stand in the draw's order, and both
/// stages read the rows a host named by a search: no pass looks behind
/// a cursor, where the walk of a whole read asked in that order does,
/// and the two read the same scores.
#[test]
fn behind_a_draw_the_stages_search_the_rows_named_and_keep_no_cursor() {
    use crate::engine::view::LOOKS_BEHIND;
    let pool: Vec<Held> = (0..6).map(|row| candidate(row, 2, 0)).collect();
    let pool = rows(&pool);
    // A draw's order. The holders are shallow, so the slice takes the
    // miss and asks its warming rows.
    let order = [4u32, 1, 5, 0, 3, 2];
    let holders = [holder(1, 4), holder(4, 4), holder(5, 3)];
    let whole = named(&holders, &[0, 3], true);
    // The mean of the rows handed, so both reads judge by one mean.
    let drawn = AffinityRows {
        mean_in_flight: Some(2.0),
        ..whole
    };
    let (prefix, slice) = (affinity(None), warmup(1.0, 0));
    for stage in [&prefix as &dyn Scorer, &slice] {
        let [by_walk, by_search] = [whole, drawn].map(|named| {
            LOOKS_BEHIND.with(|looks| looks.set(0));
            let facts = RequestFacts {
                named: Some(named),
                ..facts(16)
            };
            let mut out = vec![0.0; order.len()];
            stage.score(&facts, &pool, &order, &mut out);
            (out, LOOKS_BEHIND.with(std::cell::Cell::get))
        });
        assert_eq!(by_search.0, by_walk.0, "{}", stage.name());
        assert!(by_walk.1 > 0, "{}: a walk looks behind", stage.name());
        assert_eq!(by_search.1, 0, "{}: a search keeps no cursor", stage.name());
    }
}

/// The stage's word of a row: its match by the stage's source, of the
/// request's blocks, and where the match was read.
#[test]
fn the_stage_says_what_a_row_holds_by_its_source() {
    use AffinitySource::{Index, Merged, Predictor};
    let said = |source, overlap_blocks, predicted_blocks| {
        let params = PrefixAffinityParams {
            source: Some(source),
            ..PrefixAffinityParams::default()
        };
        let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16)));
        let column = [AffinityFacts {
            overlap_blocks,
            predicted_blocks,
            ..AffinityFacts::default()
        }];
        let facts = RequestFacts {
            affinity: &column,
            ..facts(16)
        };
        let said = stage.said(&facts, &rows(&[candidate(0, 0, 0)]), 0);
        assert!(!said.margin_held);
        (said.blocks, said.of, said.source)
    };
    // The index's blocks, at any depth.
    assert_eq!(said(Index, 3, 12), (3, 16, OverlapSource::Index));
    // The predictor's, past the stage's threshold alone.
    assert_eq!(said(Predictor, 9, 12), (12, 16, OverlapSource::Predictor));
    assert_eq!(said(Predictor, 9, 4), (0, 16, OverlapSource::None));
    // Merged: the larger of the two, the index on a tie.
    assert_eq!(said(Merged, 3, 12), (12, 16, OverlapSource::Predictor));
    assert_eq!(said(Merged, 13, 12), (13, 16, OverlapSource::Index));
    assert_eq!(said(Merged, 12, 12), (12, 16, OverlapSource::Index));
    assert_eq!(said(Merged, 0, 0), (0, 16, OverlapSource::None));
}

/// A row no host named and a row past a column hold nothing. A request
/// past sixteen bits of blocks halves both counts until its own fits,
/// and their share stands.
#[test]
fn a_row_nobody_named_holds_nothing_and_a_long_request_keeps_its_share() {
    let stage = affinity(None);
    let pool = rows(&[candidate(0, 0, 0), candidate(1, 0, 0)]);
    let holders = [holder(1, 5)];
    let named = RequestFacts {
        named: Some(named(&holders, &[], false)),
        ..facts(16)
    };
    let nothing = Overlap {
        of: 16,
        ..Overlap::default()
    };
    assert_eq!(stage.said(&named, &pool, 1).blocks, 5);
    assert_eq!(stage.said(&named, &pool, 0), nothing);
    let column = [AffinityFacts {
        overlap_blocks: 70_000,
        ..AffinityFacts::default()
    }];
    let long = RequestFacts {
        affinity: &column,
        prompt_tokens: 16 * 80_000,
        ..facts(16)
    };
    let said = stage.said(&long, &pool, 0);
    assert_eq!((said.blocks, said.of), (35_000, 40_000));
    assert_eq!(stage.said(&long, &pool, 9).blocks, 0);
    let longer = RequestFacts {
        affinity: &column,
        prompt_tokens: 16 * 300_000,
        ..facts(16)
    };
    let said = stage.said(&longer, &pool, 0);
    assert_eq!((said.blocks, said.of), (8_750, 37_500));
}

/// The rows a lent tree's match named: `holders`, and the match's two
/// numbers.
fn in_tree(holders: &[AffinityHolder], matched: u32, of: u32) -> AffinityRows<'_> {
    AffinityRows {
        tree: Some(TreeMatch::new(matched, of)),
        ..named(holders, &[], false)
    }
}

/// Rows a lent tree's match named are scored by the match's own share:
/// its size over the request's, nothing rounded to a block. Where the
/// share is over the threshold every row named is a holder, though its
/// whole blocks would not clear, or are none. At the threshold, or
/// with nobody named, nobody is. The same rows with no tree's match are
/// read by their whole blocks, as before. Behind a draw the scores are
/// the same, and a source that reads the index alone follows no tree.
///
/// The classic policy follows a tree's match where its share is over
/// `cache_threshold` and makes every tenant of it a candidate
/// (`select_worker_with_tokens` and `select_matched_candidate` in
/// `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn rows_a_trees_match_named_are_scored_by_the_matchs_share() {
    let stage = affinity(None);
    let pool = rows(&[candidate(0, 0, 0), candidate(1, 0, 0), candidate(2, 0, 0)]);
    // One whole block of the request's sixteen, and no whole block.
    let sent = |row, predicted_blocks| AffinityHolder {
        row,
        overlap_blocks: 0,
        predicted_blocks,
    };
    let holders = [sent(0, 1), sent(2, 0)];
    let facts = facts(16);
    let alike = [0.5, 0.5, 0.5];
    let over = scores_named(&stage, &facts, &pool, in_tree(&holders, 31, 100));
    assert_eq!(over, [1.0, 0.5, 1.0]);
    // Three of ten is the default threshold, and not over it.
    let at = scores_named(&stage, &facts, &pool, in_tree(&holders, 3, 10));
    assert_eq!(at, alike);
    let nobody = scores_named(&stage, &facts, &pool, in_tree(&[], 31, 100));
    assert_eq!(nobody, alike);
    // No tree's match: one block of sixteen is under the threshold.
    let blocks = scores_named(&stage, &facts, &pool, named(&holders, &[], false));
    assert_eq!(blocks, alike);
    let drawn = AffinityRows {
        mean_in_flight: Some(0.0),
        ..in_tree(&holders, 31, 100)
    };
    assert_eq!(scores_named(&stage, &facts, &pool, drawn), over);
    let params = PrefixAffinityParams {
        source: Some(AffinitySource::Index),
        ..PrefixAffinityParams::default()
    };
    let indexed = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16)));
    let unread = scores_named(&indexed, &facts, &pool, in_tree(&holders, 31, 100));
    assert_eq!(unread, alike);
}

/// A tree's match is priced at its whole blocks of the tree's unit, a
/// text's chars at four a token: the one count of a tree's match that
/// is rounded, as the classic policy prices it
/// (`select_matched_candidate` and `tree_request` in
/// `model_gateway/src/policies/cache_aware.rs`). So a loaded holder of
/// eight blocks is kept by its price, and one of a match under one
/// block is not. The stage says the match's two numbers of a row the
/// match named, where it follows the match, and a miss of the request's
/// blocks of any other row.
#[test]
fn a_trees_match_is_priced_in_whole_blocks_and_said_in_its_own_unit() {
    let gates = Gates::of(None, None, Some(16));
    assert_eq!(gates.tokens_in_tree(40, HashKind::Token), 32.0);
    assert_eq!(gates.tokens_in_tree(40, HashKind::Text), 8.0);
    assert_eq!(gates.tokens_in_tree(15, HashKind::Token), 0.0);

    let stage = affinity(Some(SpillParams {
        rel: Some(1.1),
        prefix_margin_tokens: Some(8),
        ..SpillParams::default()
    }));
    let pool = rows(&[
        candidate(0, 10, 0),
        candidate(1, 0, 0),
        candidate(2, 0, 0),
        candidate(3, 0, 0),
    ]);
    let holders = [AffinityHolder {
        row: 0,
        overlap_blocks: 0,
        predicted_blocks: 0,
    }];
    let said = |matched: u32, of: u32, row: u32| {
        let facts = RequestFacts {
            named: Some(in_tree(&holders, matched, of)),
            ..facts(16)
        };
        let mut out = vec![0.0; 4];
        stage.score(&facts, &pool, &kept(4), &mut out);
        (out, stage.said(&facts, &pool, row))
    };
    let tree = |blocks: u16, of: u16, margin_held: bool| Overlap {
        blocks,
        of,
        source: OverlapSource::Tree,
        margin_held,
    };
    // Eight whole blocks are sixteen requests' worth: over the swing's
    // margin of four, and under the priced one.
    let (scores, word) = said(128, 256, 0);
    assert_eq!(scores, [1.0, 0.5, 0.5, 0.5]);
    assert_eq!(word, tree(128, 256, true));
    // Fifteen tokens of forty clear the threshold and are no whole
    // block: nothing is priced, and the loaded holder spills.
    let (scores, word) = said(15, 40, 0);
    assert_eq!(scores, [0.0, 0.5, 0.5, 0.5]);
    assert_eq!(word, tree(15, 40, false));
    // A row the match did not name, and a match the stage does not
    // follow: a miss of the request's sixteen blocks.
    let miss = Overlap {
        of: 16,
        ..Overlap::default()
    };
    assert_eq!(said(128, 256, 1).1, miss);
    assert_eq!(said(30, 100, 0).1, miss);
}

/// One holder over the swing's margin and under the priced one: the
/// stage says the margin held. It says so once, of its own pass, of a
/// deepest holder, and not behind [`forget_prefix_pass`].
#[test]
fn the_word_of_a_pass_is_one_decisions_and_one_stages() {
    let gate = || {
        affinity(Some(SpillParams {
            rel: Some(1.1),
            prefix_margin_tokens: Some(8),
            ..SpillParams::default()
        }))
    };
    let (stage, other) = (gate(), gate());
    let pool = [
        candidate(0, 10, 8),
        candidate(1, 0, 0),
        candidate(2, 0, 0),
        candidate(3, 0, 0),
    ];
    let (rows, column) = (rows(&pool), column(&pool));
    let facts = RequestFacts {
        affinity: &column,
        ..facts(16)
    };
    let pass = |stage: &PrefixAffinity| {
        let mut out = vec![0.0; 4];
        stage.score(&facts, &rows, &kept(4), &mut out);
        out
    };
    assert_eq!(pass(&stage), vec![1.0, 0.5, 0.5, 0.5]);
    let held = Overlap {
        blocks: 8,
        of: 16,
        source: OverlapSource::Index,
        margin_held: true,
    };
    assert_eq!(stage.said(&facts, &rows, 0), held);
    // Asked again, the word is spent.
    assert!(!stage.said(&facts, &rows, 0).margin_held);
    // A row that is no deepest holder was kept by no price.
    pass(&stage);
    assert!(!stage.said(&facts, &rows, 1).margin_held);
    // The first stage of a decision to judge keeps the slot: a later
    // stage's pass does not take its word.
    pass(&stage);
    pass(&other);
    assert!(stage.said(&facts, &rows, 0).margin_held);
    // One stage's pass is no word of another's.
    pass(&other);
    assert!(!stage.said(&facts, &rows, 0).margin_held);
    // A decision whose overlap the engine states begins with no word.
    pass(&stage);
    forget_prefix_pass();
    assert!(!stage.said(&facts, &rows, 0).margin_held);
}

/// A later pass of the same stage writes over its own word: the word
/// left is of the stage's last pass. Over level loads the swing's
/// margin spills nobody, so no price kept the holder.
#[test]
fn a_later_pass_of_a_stage_writes_over_its_own_word() {
    let stage = affinity(Some(SpillParams {
        rel: Some(1.1),
        prefix_margin_tokens: Some(8),
        ..SpillParams::default()
    }));
    let pool = |others| {
        [
            candidate(0, 10, 8),
            candidate(1, others, 0),
            candidate(2, others, 0),
            candidate(3, others, 0),
        ]
    };
    let (busy, level) = (pool(0), pool(10));
    let pass = |pool: &[Held]| scores(&stage, &facts(16), pool);
    let held = |pool: &[Held]| {
        let column = column(pool);
        let facts = RequestFacts {
            affinity: &column,
            ..facts(16)
        };
        stage.said(&facts, &rows(pool), 0).margin_held
    };
    forget_prefix_pass();
    assert_eq!(pass(&busy), vec![1.0, 0.5, 0.5, 0.5]);
    assert!(held(&busy));
    assert_eq!(pass(&busy), vec![1.0, 0.5, 0.5, 0.5]);
    assert_eq!(pass(&level), vec![1.0, 0.5, 0.5, 0.5]);
    assert!(!held(&level));
}

/// A pass that orders nothing under the fleet's KV gate takes back the
/// word an earlier pass of its stage left: the price did not keep the
/// pick behind it.
#[test]
fn a_pass_under_the_kv_gate_takes_the_stages_word_back() {
    let kv = KvPressureParams {
        ceiling: Some(0.9),
        ..KvPressureParams::default()
    };
    let params = PrefixAffinityParams {
        spill: Some(SpillParams {
            rel: Some(1.1),
            prefix_margin_tokens: Some(8),
            ..SpillParams::default()
        }),
        ..PrefixAffinityParams::default()
    };
    let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), Some(&kv), Some(16)));
    let pool = |usage| {
        [
            reporting(0, 10, 8, usage),
            reporting(1, 0, 0, 500),
            reporting(2, 0, 0, 500),
            reporting(3, 0, 0, 500),
        ]
    };
    let (calm, hot) = (pool(500), pool(950));
    let pass = |pool: &[Held]| scores(&stage, &facts(16), pool);
    let held = || {
        let column = column(&calm);
        let facts = RequestFacts {
            affinity: &column,
            ..facts(16)
        };
        stage.said(&facts, &rows(&calm), 0).margin_held
    };
    forget_prefix_pass();
    assert_eq!(pass(&calm), vec![1.0, 0.5, 0.5, 0.5]);
    assert!(held());
    assert_eq!(pass(&calm), vec![1.0, 0.5, 0.5, 0.5]);
    assert_eq!(pass(&hot), vec![1.0; 4]);
    assert!(!held());
}

/// The front pass takes the word back as the whole pass does: under the
/// fleet's KV gate it says every row alike, and the price an earlier
/// pass judged by is not said.
#[test]
fn a_front_pass_under_the_kv_gate_takes_the_stages_word_back() {
    let kv = KvPressureParams {
        ceiling: Some(0.9),
        ..KvPressureParams::default()
    };
    let params = PrefixAffinityParams {
        spill: Some(SpillParams {
            rel: Some(1.1),
            prefix_margin_tokens: Some(8),
            ..SpillParams::default()
        }),
        ..PrefixAffinityParams::default()
    };
    let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), Some(&kv), Some(16)));
    let pool = |usage| {
        [
            reporting(0, 10, 8, usage),
            reporting(1, 0, 0, 500),
            reporting(2, 0, 0, 500),
            reporting(3, 0, 0, 500),
        ]
    };
    let (calm, hot) = (pool(500), pool(950));
    // The front pass over a pool: the scores and what the pass said.
    let front = |pool: &[Held]| {
        let column = column(pool);
        let rows = rows(pool);
        let facts = RequestFacts {
            affinity: &column,
            ..facts(16)
        };
        let kept = kept(rows.len() as u32);
        let mut out = vec![0.0; rows.len()];
        let (_, said) = stage.score_front(&facts, &rows, &kept, &kept, &mut out);
        (out, matches!(said, ScoreFront::Alike))
    };
    let held = || {
        let column = column(&calm);
        let facts = RequestFacts {
            affinity: &column,
            ..facts(16)
        };
        stage.said(&facts, &rows(&calm), 0).margin_held
    };
    forget_prefix_pass();
    assert!(!front(&calm).1);
    assert!(held());
    assert!(!front(&calm).1);
    assert_eq!(front(&hot), (vec![1.0; 4], true));
    assert!(!held());
}

/// The margin's bit is the difference of two scores: the stage that
/// prices the held prefix puts the row in front, and the same gate
/// with no price puts it behind. Over pools, holders, loads and
/// orders drawn at random, on a column, on named rows and behind a
/// draw with the engine's mean, a decode phase among them. The word
/// is one decision's: asked a second time it says no margin.
#[test]
fn the_margin_held_is_the_price_alone_keeping_a_row_in_front() {
    let gate = |prefix_margin_tokens| {
        affinity(Some(SpillParams {
            rel: Some(1.1),
            prefix_margin_tokens,
            ..SpillParams::default()
        }))
    };
    let (priced, plain) = (gate(Some(8)), gate(None));
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |below: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % below
    };
    let mut held = [0u32; 3];
    for round in 0..3_000u32 {
        let n = 1 + next(24) as u32;
        let pool: Vec<Candidate> = (0..n)
            .map(|row| candidate(row, next(40) as u32, 0).0)
            .collect();
        // Every fourth round names nobody.
        let mut holders = Vec::new();
        let mut column = vec![AffinityFacts::default(); n as usize];
        for row in 0..n {
            if round % 4 != 0 && next(3) == 0 {
                let blocks = 1 + next(16) as u32;
                holders.push(holder(row, blocks));
                column[row as usize].overlap_blocks = blocks;
            }
        }
        // A draw's order: some of the rows, each once.
        let mut order: Vec<u32> = (0..n).filter(|&row| row == 0 || next(4) != 0).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, next(i as u64 + 1) as usize);
        }
        let whole = named(&holders, &[], false);
        let drawn = AffinityRows {
            mean_in_flight: Some(mean_in_flight(&RequestFacts::default(), &pool, &order)),
            ..whole
        };
        // Named rows, named rows behind a draw, a column.
        let feed = round as usize % 3;
        let facts = RequestFacts {
            affinity: if feed == 2 { &column } else { &[] },
            named: [Some(whole), Some(drawn), None][feed],
            phase: if round % 5 == 0 {
                Phase::Decode
            } else {
                Phase::Single
            },
            ..facts(16)
        };
        let scored = |stage: &PrefixAffinity| {
            let mut out = vec![0.0; order.len()];
            stage.score(&facts, &pool, &order, &mut out);
            out
        };
        let unpriced = scored(&plain);
        for (at, &row) in order.iter().enumerate() {
            // The pass that decided is the last that ran.
            let by_price = scored(&priced);
            let kept = by_price[at] >= 1.0 && unpriced[at] <= 0.0;
            assert_eq!(
                priced.said(&facts, &pool, row).margin_held,
                kept,
                "round {round}, row {row}: {holders:?}, order {order:?}"
            );
            assert!(
                !priced.said(&facts, &pool, row).margin_held,
                "round {round}, row {row}: asked twice"
            );
            held[feed] += u32::from(kept);
        }
    }
    assert!(
        held.iter().all(|&rows| rows > 30),
        "rows the price alone kept, a feed: {held:?}"
    );
}

/// The prefix stage's body behind a draw scores as the whole pass's
/// does: over pools, holders, loads and orders drawn at random, under
/// a plain gate, a relative one and one that prices the held prefix,
/// a decode phase among them. Both reads judge by one mean.
#[test]
fn behind_a_draw_the_prefix_stage_scores_as_the_whole_pass_does() {
    let gate = |prefix_margin_tokens| {
        affinity(Some(SpillParams {
            rel: Some(1.1),
            prefix_margin_tokens,
            ..SpillParams::default()
        }))
    };
    let stages = [affinity(None), gate(None), gate(Some(8))];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |below: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % below
    };
    let mut reached = [0u32; 3];
    for round in 0..3_000u32 {
        let n = 1 + next(24) as u32;
        let pool: Vec<Candidate> = (0..n)
            .map(|row| candidate(row, next(40) as u32, 0).0)
            .collect();
        // Every fourth round names nobody.
        let mut holders = Vec::new();
        for row in 0..n {
            if round % 4 != 0 && next(3) == 0 {
                holders.push(holder(row, 1 + next(16) as u32));
            }
        }
        // A draw's order: some of the rows, each once.
        let mut order: Vec<u32> = (0..n).filter(|&row| row == 0 || next(4) != 0).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, next(i as u64 + 1) as usize);
        }
        let whole = named(&holders, &[], false);
        let drawn = AffinityRows {
            mean_in_flight: Some(mean_in_flight(&RequestFacts::default(), &pool, &order)),
            ..whole
        };
        let stage = &stages[round as usize % stages.len()];
        let [by_pass, by_draw] = [whole, drawn].map(|named| {
            let facts = RequestFacts {
                named: Some(named),
                phase: if round % 5 == 0 {
                    Phase::Decode
                } else {
                    Phase::Single
                },
                ..facts(16)
            };
            let mut out = vec![0.0; order.len()];
            stage.score(&facts, &pool, &order, &mut out);
            out
        });
        assert_eq!(
            by_draw, by_pass,
            "round {round}: {holders:?}, order {order:?}"
        );
        for (score, count) in [0.0, 0.5, 1.0].into_iter().zip(&mut reached) {
            *count += u32::from(by_pass.contains(&score));
        }
    }
    // The rounds reach every score.
    assert!(reached.iter().all(|&rounds| rounds > 300), "{reached:?}");
}

/// A hit the slice diverts behind a draw reads the rows named by a
/// search too: its holders' load, its warming rows and a busy warming
/// row's entry look behind no cursor, where a whole read asked in that
/// order does, and the two divert to the same row.
#[test]
fn behind_a_draw_a_diverted_hit_searches_the_rows_named_and_keeps_no_cursor() {
    use crate::engine::view::LOOKS_BEHIND;
    // Rows 1 and 4 hold the request deep. Of the thin rows 3 and 0 are
    // busier than the holders, so the entry of each is read, and row 2
    // takes the hit.
    let pool: Vec<Held> = (0..)
        .zip([6, 5, 1, 6, 5, 1])
        .map(|(row, busy)| candidate(row, busy, 0))
        .collect();
    let pool = rows(&pool);
    // A draw's order: each of the three reads meets a row behind the
    // one before it.
    let order = [4u32, 1, 5, 3, 0, 2];
    let holders = [holder(1, 12), holder(4, 12)];
    let whole = named(&holders, &[0, 2, 3], true);
    let drawn = AffinityRows {
        mean_in_flight: Some(4.0),
        ..whole
    };
    let [by_walk, by_search] = [whole, drawn].map(|named| {
        LOOKS_BEHIND.with(|looks| looks.set(0));
        let facts = RequestFacts {
            named: Some(named),
            ..facts(16)
        };
        let mut out = vec![0.0; order.len()];
        // A stage of its own a read: a diversion reserves its row's
        // window. Its first hit is due.
        warmup(0.25, 1).score(&facts, &pool, &order, &mut out);
        (out, LOOKS_BEHIND.with(std::cell::Cell::get))
    });
    assert_eq!(by_walk.0, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(by_search.0, by_walk.0);
    assert!(by_walk.1 > 0, "the walk looks behind");
    assert_eq!(by_search.1, 0, "a search keeps no cursor");
}

/// A host that names a pool's warming rows enters the stage's own
/// ruling: for fleets of every kind the rows it is handed, and the bit,
/// are what the stage's pass rules over a column of the same sizes and
/// ages; a young row leaves the rows named with its window; and a
/// parameter of the stage moves the rows named.
#[test]
fn a_host_that_names_warming_rows_enters_the_stages_own_ruling() {
    const NOW: u64 = 600_000_000;
    let stage = warmup(0.25, 8);
    let rule = stage.warm_up_rule().expect("the stage rules who warms");
    // A row: its index size and how many seconds ago it was admitted.
    let fleets: [&[(u32, u64)]; 8] = [
        &[(40, 500), (40, 500), (4, 500), (40, 500)],
        &[(40, 500), (40, 10), (0, 10), (40, 500)],
        &[(6, 500), (2, 10), (7, 500), (3, 20)],
        &[(0, 10), (0, 20), (0, 30)],
        &[(0, 500), (0, 500)],
        &[(50, 500), (50, 500)],
        &[(40, 10), (40, 500), (5, 10)],
        &[(9, 10), (3, 10), (6, 500), (7, 500)],
    ];
    // The rows a host names at `now` from the ruling's entry at `read`.
    let read_at = |rule: &WarmUpRule, fleet: &[(u32, u64)], read: u64, now: u64| {
        let sizes: Vec<u32> = fleet.iter().map(|row| row.0).collect();
        let admitted: Vec<u64> = fleet.iter().map(|row| NOW - row.1 * 1_000_000).collect();
        let (mut scratch, mut rows) = (Vec::new(), vec![(99, 0)]);
        let warming = rule.list_warming_rows(&sizes, &admitted, read, &mut scratch, &mut rows);
        let named = rows
            .iter()
            .filter(|&&(_, until)| warming.from <= now && now < until);
        (
            named.map(|&(row, _)| row).collect::<Vec<u32>>(),
            warming.thin,
        )
    };
    let named_by =
        |rule: &WarmUpRule, fleet: &[(u32, u64)], now: u64| read_at(rule, fleet, now, now);
    let mut seen = Vec::new();
    for fleet in fleets {
        let column: Vec<AffinityFacts> = fleet
            .iter()
            .map(|&(size, age)| AffinityFacts {
                admitted_age_ms: (age * 1_000) as u32,
                indexed_growth: size,
                indexed_blocks: size,
                ..AffinityFacts::default()
            })
            .collect();
        let by_stage = stage.ruled(&column).map_or((Vec::new(), false), |warm| {
            let mut warming = stage.warming::<false>(warm, Feed::Column(&column));
            let rows: Vec<u32> = (0..column.len() as u32)
                .filter(|&row| warming(row))
                .collect();
            (rows, warm.thin())
        });
        let named = named_by(&rule, fleet, NOW);
        assert_eq!(named, by_stage, "{fleet:?}");
        seen.push(named);
    }
    // The fleets hold each kind: the thin, the young, and nobody.
    assert_eq!(seen[0], (vec![2], true));
    assert_eq!(seen[2], (vec![1, 3], false));
    assert_eq!(seen[3], (vec![], false));
    // A young row is named until its window ends: sixty seconds from its
    // admission.
    assert_eq!(named_by(&rule, fleets[2], 639_999_999).0, vec![1, 3]);
    assert_eq!(named_by(&rule, fleets[2], 640_000_000).0, vec![1]);
    // The rows of one entry stand for every later instant while the
    // sizes do: read then, they are the ruling's own of that instant.
    let laters = [
        1, 29_999_999, 30_000_000, 39_999_999, 40_000_000, 49_999_999, 50_000_000, 70_000_000,
    ];
    for fleet in fleets {
        for later in laters {
            let now = NOW + later;
            let stood = read_at(&rule, fleet, NOW, now);
            assert_eq!(stood, named_by(&rule, fleet, now), "{fleet:?} +{later}");
        }
    }
    // A fleet that is all young names nobody until its first row is
    // young no more, then the rest, each until its own end.
    let all_young = |later: u64| read_at(&rule, fleets[3], NOW, NOW + later).0;
    assert_eq!(all_young(29_999_999), Vec::<u32>::new());
    assert_eq!(all_young(30_000_000), vec![0, 1]);
    assert_eq!(all_young(40_000_000), vec![0]);
    assert_eq!(all_young(50_000_000), Vec::<u32>::new());
    // A parameter of the stage moves the rows named.
    let fleet = [(40, 500), (40, 500), (15, 500), (40, 500)];
    assert_eq!(named_by(&rule, &fleet, NOW).0, vec![2]);
    let narrow = WarmupSliceParams {
        thin_ratio: Some(0.25),
        blocks: Some(8),
        ..WarmupSliceParams::default()
    };
    let narrow = WarmupSlice::new(&narrow, Gates::of(None, None, Some(16)));
    let narrow = narrow.warm_up_rule().expect("the stage rules who warms");
    assert_eq!(named_by(&narrow, &fleet, NOW).0, Vec::<u32>::new());
}

#[test]
fn the_affinity_stages_answer_row_by_row_unless_the_fleet_kv_gate_is_configured() {
    let kv = KvPressureParams {
        spread: Some(0.3),
        ..KvPressureParams::default()
    };
    let plain = Gates::of(None, None, Some(16));
    let gated = Gates::of(None, Some(&kv), Some(16));
    for (gates, row_wise) in [(plain, true), (gated, false)] {
        let affinity = PrefixAffinity::new(&PrefixAffinityParams::default(), gates);
        let slice = WarmupSlice::new(&WarmupSliceParams::default(), gates);
        assert_eq!(
            (affinity.row_wise(), slice.row_wise()),
            (row_wise, row_wise)
        );
    }
    assert!(KvPressure.row_wise());
}

/// The two ranks say the word their scores read as ([`ScoreFront::of`]),
/// and score a row alike by either entry. Seeded pools of one to forty
/// rows, a part of them kept: nobody, some or every row holds the request,
/// rows stand over and under the spill gate, and some rows warm. Two builds
/// of each stage are asked side by side, one by each entry, so the slice's
/// counters turn alike.
#[test]
fn the_ranks_say_the_word_their_scores_read_as() {
    let spill = || {
        Some(SpillParams {
            abs: Some(64),
            rel: Some(1.5),
            ..SpillParams::default()
        })
    };
    let plain: [Box<dyn Scorer>; 2] = [Box::new(affinity(spill())), Box::new(warmup(0.25, 3))];
    let said: [Box<dyn Scorer>; 2] = [Box::new(affinity(spill())), Box::new(warmup(0.25, 3))];
    let mut draw = Draw::new(0x51ce);
    // The words said, a stage: alike, no row at the ceiling, one, more.
    let mut words = [[0u32; 4]; 2];
    for round in 0..4_000u64 {
        let n = 1 + draw.below(40) as u32;
        let odds = [0, 1, 2, 8][draw.below(4)];
        let pool: Vec<Held> = (0..n)
            .map(|i| {
                let holds = odds != 0 && draw.below(odds) == 0;
                let indexed = [0, 64][draw.below(2)];
                sized(i, draw.below(12) as u32, if holds { 8 } else { 0 }, indexed)
            })
            .collect();
        let (rows, column) = (rows(&pool), column(&pool));
        // One row in four is left out, and never every row.
        let mut kept: Vec<u32> = (0..n).filter(|_| draw.below(4) != 0).collect();
        if kept.is_empty() {
            kept.push(draw.below(n as usize) as u32);
        }
        let facts = RequestFacts {
            affinity: &column,
            seed: round,
            ..facts(16)
        };
        for (stage, (plain, said)) in plain.iter().zip(&said).enumerate() {
            let mut by_score = vec![0.0; kept.len()];
            let mut by_front = vec![0.0; kept.len()];
            let band = plain.score(&facts, &rows, &kept, &mut by_score);
            let (front_band, word) = said.score_front(&facts, &rows, &kept, &kept, &mut by_front);
            let name = plain.name();
            assert_eq!(by_front, by_score, "{name}, round {round}");
            assert_eq!(front_band, band, "{name}, round {round}");
            assert_eq!(
                word,
                ScoreFront::of(plain.spread(), &by_score),
                "{name}, round {round}: {by_score:?}"
            );
            let which = match word {
                ScoreFront::Alike => 0,
                ScoreFront::At(0) => 1,
                ScoreFront::At(1) => 2,
                _ => 3,
            };
            words[stage][which] += 1;
        }
    }
    // Each stage said each of its words in some rounds: the holders'
    // stage every one, the slice alike and its one row.
    assert!(words[0].iter().all(|&rounds| rounds > 50), "{words:?}");
    assert!(words[1][0] > 50 && words[1][2] > 50, "{words:?}");
    assert_eq!((words[1][1], words[1][3]), (0, 0), "{words:?}");
    // A lone kept row that takes the request stands at the ceiling,
    // which reads alike.
    let pool = [sized(0, 3, 0, 64), sized(1, 2, 0, 64), sized(2, 5, 0, 0)];
    let (three, held) = (rows(&pool), column(&pool));
    let fed = RequestFacts {
        affinity: &held,
        ..facts(16)
    };
    let mut lone = [0.0];
    let (_, word) = warmup(0.25, 0).score_front(&fed, &three, &[2], &[2], &mut lone);
    assert_eq!((lone, word), ([1.0], ScoreFront::Alike));
    // A decision fed no column: the slice stands alike, and no row
    // holds the request.
    let unfed = facts(16);
    let (mut slices, mut holds) = ([0.0; 3], [0.0; 3]);
    let (_, word) = warmup(0.25, 0).score_front(&unfed, &three, &kept(3), &kept(3), &mut slices);
    assert_eq!((slices, word), ([1.0; 3], ScoreFront::Alike));
    let (_, word) = affinity(None).score_front(&unfed, &three, &kept(3), &kept(3), &mut holds);
    assert_eq!((holds, word), ([0.5; 3], ScoreFront::At(0)));
    // The rows a host named say nothing, and score as by the plain
    // entry: the second row holds the request and the third warms.
    let holders = [holder(1, 8)];
    let naming = RequestFacts {
        named: Some(named(&holders, &[2], true)),
        ..facts(16)
    };
    let pairs: [[Box<dyn Scorer>; 2]; 2] = [
        [Box::new(affinity(None)), Box::new(affinity(None))],
        [Box::new(warmup(0.25, 0)), Box::new(warmup(0.25, 0))],
    ];
    for [plain, said] in &pairs {
        let (mut by_score, mut by_front) = ([0.0; 3], [0.0; 3]);
        let band = plain.score(&naming, &three, &kept(3), &mut by_score);
        assert_eq!(
            said.score_front(&naming, &three, &kept(3), &kept(3), &mut by_front),
            (band, ScoreFront::Unsaid),
            "{}",
            plain.name()
        );
        assert_eq!(by_front, by_score, "{}", plain.name());
    }
    // Under the fleet's KV gate both ranks stand alike.
    let kv = KvPressureParams {
        spread: Some(0.3),
        ceiling: Some(0.9),
        ..KvPressureParams::default()
    };
    let params = PrefixAffinityParams::default();
    let gates = Gates::of(Some(&params), Some(&kv), Some(16));
    let slicing = WarmupSliceParams {
        share: Some(0.25),
        ..WarmupSliceParams::default()
    };
    let gated: [Box<dyn Scorer>; 2] = [
        Box::new(PrefixAffinity::new(&params, gates)),
        Box::new(WarmupSlice::new(&slicing, gates)),
    ];
    let ceiling = [reporting(0, 0, 8, 950), reporting(1, 0, 0, 900)];
    let (two, usage) = (rows(&ceiling), column(&ceiling));
    let pressed = RequestFacts {
        affinity: &usage,
        ..facts(16)
    };
    for stage in &gated {
        let mut out = [0.0; 2];
        let (_, word) = stage.score_front(&pressed, &two, &kept(2), &kept(2), &mut out);
        assert_eq!(
            (out, word),
            ([1.0, 1.0], ScoreFront::Alike),
            "{}",
            stage.name()
        );
    }
}
