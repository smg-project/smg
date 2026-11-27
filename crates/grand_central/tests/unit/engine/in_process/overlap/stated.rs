//! What a pick states of its first target: its worker's entry of a column its
//! host filled or of the rows its host named, a tree's match in tokens or in
//! chars and a miss of it, a count by the document's block, and a profile that
//! reads no affinity, whose pick states nothing.

use super::*;

/// A pick states its worker's entry of a column the request's host filled,
/// read as an entry of the engine's own rows is: the larger of its two
/// readings, the index's on a tie, a miss where the entry holds nothing or
/// the column ends before the row. The engine names no row for such a
/// request and reads no table.
#[test]
fn a_pick_states_its_workers_entry_of_a_column_its_host_filled() {
    let (affinity, host, _) = hosted(true);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    let entry = |index: u32, predicted: u32| AffinityFacts {
        overlap_blocks: index,
        predicted_blocks: predicted,
        ..AffinityFacts::default()
    };
    let column = [entry(0, 0), entry(3, 0), entry(1, 4), entry(2, 2)];
    let facts = RequestFacts {
        affinity: &column,
        ..request(&asked, 7, NOW)
    };
    assert_eq!(bed.states(&facts, 1), holds(3, 4, OverlapSource::Index));
    assert_eq!(bed.states(&facts, 2), holds(4, 4, OverlapSource::Predictor));
    assert_eq!(bed.states(&facts, 3), holds(2, 4, OverlapSource::Index));
    assert_eq!(bed.states(&facts, 0), holds(0, 4, OverlapSource::None));
    // A row past a shorter column holds nothing, as a row no host named.
    let short = RequestFacts {
        affinity: &column[..3],
        ..facts
    };
    assert_eq!(bed.states(&short, 3), holds(0, 4, OverlapSource::None));
    assert_eq!(host.asked.load(Ordering::Relaxed), 0, "no index is asked");
    assert!(bed.pool.table.load().is_none(), "no table is read");
}

/// A pick states its worker's entry of the rows the request's host named in
/// a column's place: a row named by what it holds, a row not named as a
/// miss of the request's blocks.
#[test]
fn a_pick_states_its_workers_entry_of_the_rows_its_host_named() {
    let (affinity, host, _) = hosted(true);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    let holders = [
        AffinityHolder {
            row: 1,
            overlap_blocks: 3,
            predicted_blocks: 0,
        },
        AffinityHolder {
            row: 3,
            overlap_blocks: 0,
            predicted_blocks: 2,
        },
    ];
    let facts = RequestFacts {
        named: Some(AffinityRows {
            holders: &holders,
            warming: &[],
            thin: false,
            mean_in_flight: None,
            tree: None,
        }),
        ..request(&asked, 7, NOW)
    };
    assert_eq!(bed.states(&facts, 1), holds(3, 4, OverlapSource::Index));
    assert_eq!(bed.states(&facts, 3), holds(2, 4, OverlapSource::Predictor));
    assert_eq!(bed.states(&facts, 0), holds(0, 4, OverlapSource::None));
    assert_eq!(host.asked.load(Ordering::Relaxed), 0, "no index is asked");
}

/// A pick that is none of the match's holders states a miss of the
/// request's blocks, whatever the match holds for another worker: the pick
/// states what its own worker holds. A gateway's classic policy observes
/// the match's share of such a request, whoever it chose
/// (`log_tree_decision` in `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_pick_that_is_no_holder_of_a_trees_match_states_a_miss() {
    let (affinity, _tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 4);
    let miss = holds(0, 4, OverlapSource::None);
    assert_eq!(stated_in_tree(&bed, &asked, 1), miss);
    assert_eq!(bed.keyed(&asked, 0, None), [(1, 0, 4)]);
    // The match names the first worker alone, and the pick is the second.
    assert_eq!(stated_in_tree(&bed, &asked, 2), miss);
    // The descent wrote the prompt for the second too: both hold it now.
    assert_eq!(bed.keyed(&asked, 0, None), [(1, 0, 4), (2, 0, 4)]);
    let sent = holds(64, 64, OverlapSource::Tree);
    assert_eq!(stated_in_tree(&bed, &asked, 2), sent);
    assert_eq!(stated_in_tree(&bed, &asked, 1), sent);
}

/// A match under one block names its workers and states the match: the
/// tree's own two numbers are rounded to no block. Under a tree of
/// four-token pages, a prompt that shares its first eight tokens with one
/// the tree holds is named that worker with no block held, and a pick of it
/// states eight tokens of sixty-four. A prompt shorter than one block
/// states a miss of its one block while the tree holds it for another
/// worker, and every token of it once the tree holds it for the worker
/// picked. A gateway's classic policy divides the tokens matched by the
/// prompt's tokens (`select_worker_with_tokens` in
/// `model_gateway/src/policies/cache_aware.rs`), makes every tenant of the
/// match a candidate whatever its blocks (`select_matched_candidate`), and
/// observes the same division (`log_tree_decision`).
#[test]
fn a_trees_match_under_one_block_names_its_workers_and_states_the_match() {
    const PAGE: usize = 4;
    let (affinity, tree) = lending_by(PAGE, false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 4);
    assert_eq!(
        stated_in_tree(&bed, &asked, 1),
        holds(0, 4, OverlapSource::None)
    );
    // Eight tokens of the prompt held, and then another prompt's.
    let mut branch = asked[..2 * PAGE].to_vec();
    branch.extend(tokens(5_000, 4).into_iter().skip(2 * PAGE));
    let named = bed.keyed_whole(&branch, 0, None);
    assert_eq!(named.holders, [(1, 0, 0)], "named, with no block held");
    assert_eq!(named.tree, Some(TreeMatch::new(8, 64)));
    assert_eq!(
        stated_in_tree(&bed, &branch, 1),
        holds(8, 64, OverlapSource::Tree)
    );
    // Three whole pages, and under one block. The tree holds them for the
    // first worker, and the pick is another.
    let short = &asked[..3 * PAGE];
    let none = holds(0, 1, OverlapSource::None);
    assert_eq!(stated_in_tree(&bed, short, 3), none);
    let held = tree.get_tenant_token_counts();
    assert!(held
        .get(&address(3))
        .is_some_and(|&held| held >= short.len()));
    let whole = holds(12, 12, OverlapSource::Tree);
    assert_eq!(stated_in_tree(&bed, short, 3), whole);
    // A prompt the tree holds nothing of carries a match of nothing, and
    // names nobody.
    let named = bed.keyed_whole(&tokens(9_000, 4), 0, None);
    assert!(named.holders.is_empty());
    assert_eq!(named.tree, Some(TreeMatch::new(0, 64)));
}

/// A string tree's match is carried in chars, a partition's marker in
/// neither count, and no count is rounded to a window. A text of forty
/// chars is under one window: once the tree holds it for a worker, its rows
/// name that worker with no window held and carry forty of forty, and a
/// pick of it states them. A text that shares its first hundred chars with
/// one the tree holds, a window and a part of the next, carries a hundred
/// of its own three hundred behind a marker of eighteen chars, where its
/// holder's row counts one whole window. A text of three bytes a char
/// carries its chars, where its holder's row counts its bytes. The classic
/// policy divides the chars its tree matched by the text's chars, both
/// behind the marker (`select_worker_with_text` in
/// `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_string_trees_match_is_carried_in_chars_behind_the_marker() {
    let (affinity, _tree) = lending_text(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let short = "a".repeat(40);
    assert!(bed.worded(&short, 0, Some(2)).is_empty());
    let named = bed.worded_whole(&short, 0, None);
    assert_eq!(named.holders, [(2, 0, 0)], "named, with no window held");
    assert_eq!(named.tree, Some(TreeMatch::new(40, 40)));
    let stated = |row: u32| {
        let hashes = worded(&short);
        let entry = TextKey {
            text: Cow::Borrowed(short.as_str()),
            marker: 0,
        };
        let answer = picked(row, bed.ids[row as usize]);
        bed.stated_entered(&request(&hashes, 7, NOW), (None, Some(entry)), answer)
    };
    assert_eq!(stated(2), holds(40, 40, OverlapSource::Tree));
    assert_eq!(stated(1), holds(0, 1, OverlapSource::None));

    // Behind a marker of eighteen chars, one byte each.
    let under = |prompt: &str| format!("\u{1}{:016x}\u{1}{prompt}", 1);
    let marker = 18;
    let stored = "b".repeat(100);
    assert!(bed.worded(&under(&stored), marker, Some(1)).is_empty());
    let asked = format!("{stored}{}", "c".repeat(200));
    let named = bed.worded_whole(&under(&asked), marker, None);
    assert_eq!(
        named.holders,
        [(1, 0, 1)],
        "one whole window of a hundred bytes"
    );
    assert_eq!(named.tree, Some(TreeMatch::new(100, 300)));

    // Thirty chars of three bytes: ninety bytes are one whole window.
    let stored = "語".repeat(30);
    assert!(bed.worded(&stored, 0, Some(3)).is_empty());
    let asked = format!("{stored}{}", "d".repeat(30));
    let named = bed.worded_whole(&asked, 0, None);
    assert_eq!(named.holders, [(3, 0, 1)]);
    assert_eq!(named.tree, Some(TreeMatch::new(30, 60)));
}

/// A holder the sample's draw leaves out states a miss. Eight workers hold
/// the prompt in the tree, and under a sample of four a decision is named
/// four of them. A pick of one of the four states the match's tokens. A
/// pick of one the draw left out states a miss of the request's blocks,
/// though the tree holds the prompt for it. A gateway's classic policy
/// counts every tenant of the match (`log_tree_decision` in
/// `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_holder_the_samples_draw_leaves_out_states_a_miss() {
    let (affinity, _tree) = lending(false);
    let bed = Bed::new(affinity, sampled(4), 20);
    let asked = tokens(1_000, 4);
    for at in 0..8 {
        bed.keyed(&asked, 0, Some(at));
    }
    let hashes = hashed(&asked);
    let (mut named_picks, mut left_out) = (0, 0);
    for seed in 0..32 {
        for row in 0..8 {
            let key = TokenKey {
                tokens: Cow::Borrowed(asked.as_slice()),
                marker: 0,
            };
            let mut named = Vec::new();
            let returned = decide(
                &bed.affinity,
                &bed.pool,
                &bed.spec,
                &bed.workers,
                &bed.ids,
                &mut request(&hashes, seed, NOW),
                None,
                |_| Some(key),
                || None,
                |_| true,
                |facts| {
                    let rows = facts.named.expect("rows named");
                    named = rows.holders.iter().map(|holder| holder.row).collect();
                    picked(row, bed.ids[row as usize])
                },
            );
            let stated = returned.pick().map(Pick::overlap).unwrap_or_default();
            assert_eq!(named.len(), 4, "seed {seed}: the sample");
            if named.contains(&row) {
                let sent = holds(64, 64, OverlapSource::Tree);
                assert_eq!(stated, sent, "seed {seed}, row {row}: named");
                named_picks += 1;
            } else {
                let miss = holds(0, 4, OverlapSource::None);
                assert_eq!(stated, miss, "seed {seed}, row {row}: left out");
                left_out += 1;
            }
        }
    }
    assert!(named_picks > 0 && left_out > 0, "both kinds were picked");
}

/// A miss inside a tree is counted by the document's block size where the
/// request states its tokens and no hash of them: sixty-four tokens are
/// four blocks of sixteen. Once the tree holds the prompt for a worker, a
/// pick of it states the match's tokens, which no block size counts.
#[test]
fn a_tree_pick_of_a_request_with_no_hashes_is_counted_by_the_documents_block() {
    let (affinity, _tree) = lending(false);
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = tokens(1_000, 4);
    let facts = RequestFacts {
        model: MODEL,
        seed: 7,
        now_micros: NOW,
        prompt_tokens: 4 * BLOCK,
        ..RequestFacts::default()
    };
    let key = || TokenKey {
        tokens: Cow::Borrowed(asked.as_slice()),
        marker: 0,
    };
    let answer = picked(1, bed.ids[1]);
    let miss = holds(0, 4, OverlapSource::None);
    assert_eq!(bed.stated(&facts, Some(key()), answer), miss);
    let sent = holds(64, 64, OverlapSource::Tree);
    assert_eq!(bed.stated(&facts, Some(key()), answer), sent);
}

/// A decision whose profile reads no affinity states nothing, whatever its
/// host fed: under a document none of whose bindings reads affinity, and
/// under one where another model's profile does. The reader's pick states.
/// Nor does it state where its host lends a token tree and names the
/// request's entry: such a profile is decided inside no tree.
#[test]
fn a_profile_that_reads_no_affinity_states_nothing() {
    let asked = prompt(1_000, 4);
    let column = [AffinityFacts {
        overlap_blocks: 4,
        ..AffinityFacts::default()
    }; 4];
    let plain = request(&asked, 7, NOW);
    let fed = RequestFacts {
        affinity: &column,
        ..plain
    };
    let (affinity, _, _) = hosted(true);
    let bed = Bed::new(affinity, preset("least_load"), 4);
    assert_eq!(bed.states(&plain, 1), Overlap::NONE);
    assert_eq!(bed.states(&fed, 1), Overlap::NONE);
    let (affinity, tree) = lending(false);
    let bed = Bed::new(affinity, preset("least_load"), 4);
    let key = tokens(1_000, 4);
    assert_eq!(stated_in_tree(&bed, &key, 1), Overlap::NONE);
    assert!(
        tree.get_tenant_token_counts().is_empty(),
        "nothing is written"
    );
    // One model's profile reads affinity in the single phase. Another
    // model's reads none there.
    let mut document = PolicySpec::preset("least_load").expect("the preset");
    let reader = PolicySpec::preset("cache_aware").expect("the preset");
    let mut binding = reader.bindings[0].clone();
    binding.matcher.model = MODEL.into();
    binding.matcher.phase = vec![Phase::Single];
    document.bindings.insert(0, binding);
    document.profiles.extend(reader.profiles);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let (affinity, _, _) = hosted(true);
    let bed = Bed::new(affinity, spec, 4);
    let other = |facts: &RequestFacts<'_>| {
        bed.states(
            &RequestFacts {
                model: "model-b",
                ..*facts
            },
            1,
        )
    };
    assert_eq!(other(&plain), Overlap::NONE);
    assert_eq!(other(&fed), Overlap::NONE);
    assert_eq!(bed.states(&fed, 1), holds(4, 4, OverlapSource::Index));
}
