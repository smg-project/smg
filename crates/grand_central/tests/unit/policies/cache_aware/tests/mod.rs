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

mod prefix_affinity;
mod warmup_slice;

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
/// always (`select_worker_with_tokens`,
/// `model_gateway/src/policies/cache_aware.rs`).
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
