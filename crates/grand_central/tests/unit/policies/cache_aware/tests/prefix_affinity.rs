//! The `prefix-affinity` stage: which rows it ranks first for a request's
//! prefix, what it counts as a match, and the word it leaves of a pass.

use super::*;

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

/// The pass reads a kept row's entry once: the row's match rides its score
/// slot to the scoring, which reads no entry again.
#[test]
fn the_affinity_pass_reads_a_kept_rows_entry_once() {
    let pool = [candidate(0, 1, 2), candidate(1, 1, 8)];
    let (rows, column) = (rows(&pool), column(&pool));
    let facts = RequestFacts {
        affinity: &column,
        ..facts(16)
    };
    let reads = std::cell::Cell::new(0u32);
    let mut out = [0.0; 2];
    let feed = Feed::Column(&column);
    affinity(None).rank(&facts, &rows, &kept(2), &mut out, feed, || {
        |i: u32| {
            reads.set(reads.get() + 1);
            column[i as usize]
        }
    });
    assert_eq!(out, [0.5, 1.0], "the deeper holder leads");
    assert_eq!(reads.get(), 2, "one read a kept row");
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
/// the threshold (`select_worker_with_tokens`,
/// `model_gateway/src/policies/cache_aware.rs`).
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
/// (`select_worker_with_tokens`, `model_gateway/src/policies/cache_aware.rs`).
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

/// A head a placement map holds is priced by its whole count, where a
/// tree's match of as many tokens is priced by its whole blocks
/// ([`TreeMatch::counted`]). Forty-seven tokens at eight a request are six
/// requests' worth, and their two whole blocks are four. A holder seven
/// requests over three idle workers stands under the first margin and
/// over the second: it keeps the request as a map's holder, and spills it
/// as a tree's. A head is in tokens whatever the request's hash kind: under
/// the text kind nothing divides it by four, so the holder keeps the
/// request there too.
///
/// The classic policy prices a hash index's holder by its boundary's
/// tokens: `select_worker_hash` hands `select_final_from_affinity` the
/// boundary, and `held_prefix_margin` divides it
/// (`model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_counted_head_is_priced_by_its_tokens() {
    let stage = affinity(Some(SpillParams {
        rel: Some(1.1),
        prefix_margin_tokens: Some(8),
        ..SpillParams::default()
    }));
    let pool = rows(&[
        candidate(0, 7, 0),
        candidate(1, 0, 0),
        candidate(2, 0, 0),
        candidate(3, 0, 0),
    ]);
    let holders = [AffinityHolder {
        row: 0,
        overlap_blocks: 0,
        predicted_blocks: 2,
    }];
    let scored = |tree: TreeMatch, hash_kind: HashKind| {
        let rows = AffinityRows {
            tree: Some(tree),
            ..named(&holders, &[], false)
        };
        let facts = RequestFacts {
            hash_kind,
            ..facts(6)
        };
        scores_named(&stage, &facts, &pool, rows)
    };
    let held = [1.0, 0.5, 0.5, 0.5];
    assert_eq!(scored(TreeMatch::head(47, 96), HashKind::Token), held);
    assert_eq!(scored(TreeMatch::head(47, 96), HashKind::Text), held);
    let spilt = [0.0, 0.5, 0.5, 0.5];
    assert_eq!(scored(TreeMatch::new(47, 96), HashKind::Token), spilt);
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
