//! The rows a decision is named from an index of cache events: the block
//! hashes a request is found by, its holders by their depth, the cap's draw
//! among them, the list a lookup is kept within, the lookup within groups on
//! a fleet that keeps none, and the warming rows: a thin worker, a young one,
//! their own draw and the stage that rules them.

use super::*;

/// The request's token blocks hash as an engine's stored blocks do: the
/// index finds a request by the hashes its facts carry.
#[test]
fn a_prompts_block_hashes_are_the_hashes_an_engine_stores() {
    let tokens: Vec<u32> = (100..100 + 5 * BLOCK + 3).collect();
    let spec = preset("cache_aware");
    let hashes =
        PromptHashes::under_document(&spec, Some(BLOCK), Some(&tokens), None, HashSeeds::PLAIN);
    let stored: Vec<u64> = compute_request_content_hashes(&tokens, BLOCK as usize)
        .iter()
        .map(|hash| hash.0)
        .collect();
    assert_eq!(stored.len(), 5);
    assert_eq!(hashes_of(&hashes), stored);
}

/// The index names the workers that hold a leading part of the request,
/// each with its depth, ascending by row. A worker that holds another
/// prompt is not named, and a fleet just admitted names no warming row.
#[test]
fn the_index_names_a_requests_holders_by_their_depth() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    let hashes = hashes_of(&asked);
    store(&index, 5, &hashes[..2]);
    store(&index, 2, &hashes);
    store(&index, 6, &hashes_of(&prompt(9_000, 4)));
    let bed = Bed::new(affinity, preset("cache_aware"), 8);
    let named = bed.decide(&request(&asked, 7, NOW), None).expect("rows");
    assert_eq!(named.holders, [(2, 4, 0), (5, 2, 0)]);
    assert_eq!(named.warming, [] as [u32; 0]);
    // A request no worker holds a block of names nobody.
    assert!(bed.holders(&prompt(50_000, 4), NOW).is_empty());
}

/// Past the profile's sample the deepest depth is cut by a draw from the
/// decision's seed: sixteen rows, ascending, none twice, and another seed
/// draws another sixteen.
#[test]
fn the_cap_cuts_the_holders_of_one_depth_by_a_draw() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 2);
    for at in 0..24 {
        store(&index, at, &hashes_of(&asked));
    }
    let bed = Bed::new(affinity, preset("cache_aware"), 24);
    let rows = |seed| {
        let named = bed.decide(&request(&asked, seed, NOW), None).expect("rows");
        let rows: Vec<u32> = named.holders.iter().map(|&(row, _, _)| row).collect();
        assert_eq!(rows.len(), 16);
        assert!(rows.is_sorted_by(|a, b| a < b), "{rows:?}");
        assert!(rows.iter().all(|&row| row < 24));
        assert!(named.holders.iter().all(|&(_, held, _)| held == 2));
        rows
    };
    let first = rows(1);
    assert_eq!(first, rows(1), "one seed draws one set");
    assert!((2..40).any(|seed| rows(seed) != first));
}

/// The lookup is kept within the list's own workers. A worker outside the
/// list holds the request deeper than any worker of it, under a sample of
/// one: the one place goes to the list's deepest holder. A lookup over
/// every worker of the index would spend it on the worker outside, which
/// is no row of the list, and name nobody.
#[test]
fn a_holder_outside_the_list_takes_no_place_of_the_cap() {
    use crate::document::spec::{BestParams, PickStage};

    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    let hashes = hashes_of(&asked);
    store(&index, 2, &hashes[..2]);
    store(&index, 9, &hashes);
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    let profile = document.profiles.get_mut("cache_aware");
    let profile = profile.expect("the preset's profile");
    profile.pick = PickStage::Best(BestParams { sample: 1 }).into();
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let bed = Bed::new(affinity, spec, 4);
    assert_eq!(bed.holders(&asked, NOW), [(2, 2, 0)]);
}

/// A worker whose index is empty beside a fleet that holds a level is thin
/// inside its window: it is named as the list's one warming row.
#[test]
fn a_worker_with_nothing_indexed_beside_a_warm_fleet_is_named_thin() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let held = hashes_of(&prompt(1_000, 8));
    for at in 0..3 {
        store(&index, at, &held);
    }
    let bed = Bed::new(affinity, warming_by(60), 4);
    let named = bed.decide(&request(&prompt(50_000, 2), 7, NOW), None);
    let named = named.expect("rows");
    assert_eq!((named.warming, named.thin), (vec![3], true));
}

/// A worker whose index is empty is thin for its window alone: past it the
/// list names it no more. Blocks taken away under its number start the
/// window again, on the clock of the emptying and not of the admission.
#[test]
fn a_worker_with_nothing_indexed_is_named_thin_for_its_window_alone() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let held = hashes_of(&prompt(1_000, 8));
    for at in 0..3 {
        store(&index, at, &held);
    }
    let bed = Bed::new(affinity, warming_by(1), 4);
    let other = prompt(50_000, 2);
    let warming = |now| {
        let named = bed.decide(&request(&other, 7, now), None).expect("rows");
        (named.warming, named.thin)
    };
    // Admitted a moment ago, the empty row is the list's one thin row.
    assert_eq!(warming(NOW), (vec![3], true));
    // A row's age is real time: its window of a second over, the table
    // read again names nobody.
    thread::sleep(Duration::from_millis(1_050));
    assert_eq!(warming(NOW + SECOND), (vec![], false));
    // It gains the fleet's blocks and is no thin row.
    let (number, mut map) = stored(&index, 3, &held);
    assert_eq!(warming(NOW + 2 * SECOND), (vec![], false));
    // It loses them under its number: emptied underneath, its window starts
    // again at that read, however old its admission.
    let contents: Vec<ContentHash> = held.iter().map(|&hash| ContentHash(hash)).collect();
    index.apply_removed(number, &request_prefix_hashes(&contents), &mut map);
    assert_eq!(warming(NOW + 3 * SECOND), (vec![3], true));
}

/// Under a window of 0 a worker whose index is empty is never thin, and one
/// that holds a block under the ratio still is.
#[test]
fn a_window_of_none_names_no_worker_with_nothing_indexed() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let held = hashes_of(&prompt(1_000, 64));
    for at in 0..3 {
        store(&index, at, &held);
    }
    let bed = Bed::new(affinity, warming_by(0), 4);
    let other = prompt(50_000, 2);
    let warming = |now| {
        let named = bed.decide(&request(&other, 7, now), None).expect("rows");
        (named.warming, named.thin)
    };
    assert_eq!(warming(NOW), (vec![], false));
    store(&index, 3, &hashes_of(&prompt(1_000, 1)));
    assert_eq!(warming(NOW + SECOND), (vec![3], true));
}

/// A fleet that is all young names nobody. A worker still short of its
/// first blocks beside workers that grew past them is named until its
/// window ends, and no longer.
#[test]
fn a_young_worker_is_named_until_its_window_ends() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let first = hashes_of(&prompt(1_000, 8));
    for at in 0..4 {
        store(&index, at, &first);
    }
    let bed = Bed::new(affinity, warming_by(1), 4);
    let other = prompt(50_000, 2);
    let warming = |now| {
        let named = bed.decide(&request(&other, 7, now), None).expect("rows");
        (named.warming, named.thin)
    };
    // Every worker was admitted a moment ago and none grew: nobody to
    // catch up to.
    assert_eq!(warming(NOW), (vec![], false));
    // Three workers gain their first blocks. The table is read again a
    // second later, when each worker's admission is some milliseconds old.
    let second = hashes_of(&prompt(2_000, 8));
    for at in 0..3 {
        store(&index, at, &second);
    }
    thread::sleep(Duration::from_millis(3));
    assert_eq!(warming(NOW + SECOND), (vec![3], false));
    // By the table that stands, past the end of the row's own window.
    assert_eq!(warming(NOW + 2 * SECOND - 1), (vec![], false));
}

/// Past the picker's sample the warming rows a decision is named are a
/// draw of the sample from its seed, each row as likely as another and
/// none twice. Within the sample every one is named.
#[test]
fn warming_rows_past_the_sample_are_a_draw_each_row_as_likely() {
    // Fourteen workers hold a level, and the ten behind them nothing: thin
    // beside a fleet whose median holds one.
    let bed = |sample| {
        let (affinity, _, index) = hosted(true);
        let index = index.expect("the index");
        let level = hashes_of(&prompt(1_000, 8));
        for at in 0..14 {
            store(&index, at, &level);
        }
        Bed::new(affinity, sampled(sample), 24)
    };
    let other = prompt(50_000, 2);
    let thin: Vec<u32> = (14..24).collect();
    let named = bed(16).decide(&request(&other, 1, NOW), None);
    let named = named.expect("rows");
    assert_eq!((named.warming, named.thin), (thin.clone(), true));
    let cut = bed(4);
    let mut drawn = std::collections::BTreeMap::new();
    for seed in 0..1_000 {
        let named = cut.decide(&request(&other, seed, NOW), None);
        let rows = named.expect("rows").warming;
        assert_eq!(rows.len(), 4, "seed {seed}");
        assert!(rows.is_sorted_by(|a, b| a < b), "{rows:?}");
        for row in rows {
            *drawn.entry(row).or_insert(0_u32) += 1;
        }
    }
    // 1,000 draws of 4 in 10: about 400 a row.
    assert_eq!(drawn.keys().copied().collect::<Vec<_>>(), thin);
    assert!(
        drawn.values().all(|count| (300..=500).contains(count)),
        "{drawn:?}"
    );
}

/// The holders' draw and the warming rows' draw are on streams of their
/// own. Two workers are the request's only holders and the list's only
/// warming rows, under a sample of one: over the seeds each list names
/// either worker beside either worker of the other. On one stream the two
/// draws would read one value, and one list's row would give the other's.
#[test]
fn the_holders_and_the_warming_rows_are_drawn_apart() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    // Two workers hold a level of blocks of their own.
    let level = hashes_of(&prompt(1_000, 8));
    for at in 0..2 {
        store(&index, at, &level);
    }
    // Two hold the request's one block and nothing more: thin.
    let asked = prompt(9_000, 1);
    for at in 2..4 {
        store(&index, at, &hashes_of(&asked));
    }
    let bed = Bed::new(affinity, sampled(1), 4);
    let mut pairs = std::collections::BTreeSet::new();
    for seed in 0..64 {
        let named = bed.decide(&request(&asked, seed, NOW), None);
        let named = named.expect("rows");
        assert_eq!(
            (named.holders.len(), named.warming.len()),
            (1, 1),
            "seed {seed}"
        );
        pairs.insert((named.holders[0].0, named.warming[0]));
    }
    assert_eq!(pairs.len(), 4, "{pairs:?}");
}

/// A document rules the warming rows by its own stage. One whose profile
/// holds no warm-up stage names nobody. One whose stage holds another
/// number rules by it: a worker that holds a quarter of the fleet's median
/// is thin under half of it, and is not under a fifth.
#[test]
fn a_document_rules_the_warming_rows_by_its_own_stage() {
    // Three workers hold a level, the fourth a quarter of what they hold,
    // and the fifth nothing.
    let warming = |spec| {
        let (affinity, _, index) = hosted(true);
        let index = index.expect("the index");
        let level = hashes_of(&prompt(1_000, 8));
        for at in 0..3 {
            store(&index, at, &level);
        }
        store(&index, 3, &level[..2]);
        let bed = Bed::new(affinity, spec, 5);
        let named = bed.decide(&request(&prompt(50_000, 2), 7, NOW), None);
        let named = named.expect("rows");
        (named.warming, named.thin)
    };
    assert_eq!(warming(warming_by(60)), (vec![3, 4], true));
    let unruled = ruled_by(warm_up(60), |profile| {
        profile.score.remove(0);
    });
    assert_eq!(warming(unruled), (vec![], false));
    let fifth = WarmupSliceParams {
        thin_ratio: Some(0.2),
        ..warm_up(60)
    };
    assert_eq!(warming(ruled_by(fifth, |_| {})), (vec![4], true));
}

/// On a fleet none of whose workers keeps cache groups, the lookup within
/// groups is the chain index's own capped lookup, holder for holder, under
/// every mask, cap and draw. Holders a host scored itself are kept by the
/// same rule: as many, of the same depths.
#[test]
fn the_lookup_within_groups_is_the_plain_lookup_where_no_worker_keeps_groups() {
    let index = ChainIndex::new();
    let request = [1u64, 2, 3];
    // Three blocks, two, two and one.
    for (at, depth) in [3usize, 2, 2, 1].into_iter().enumerate() {
        store(&index, at, &request[..depth]);
    }
    let all = numbers(&index, 4);
    let groups = GroupViews::new();
    let contents: Vec<ContentHash> = request.iter().map(|&hash| ContentHash(hash)).collect();
    let depths = |kept: &[(u32, u32)]| {
        let mut depths: Vec<u32> = kept.iter().map(|&(_, depth)| depth).collect();
        depths.sort_unstable();
        depths
    };
    let none: &[u32] = &[];
    for among in [&all[..], &all[1..], none] {
        let within = naming(among);
        for cap in [0usize, 1, 2, 3, 16] {
            for drawn in 0..2u32 {
                let said = format!("{among:?}, cap {cap}, draw {drawn}");
                let (mut plain, mut grouped) = (vec![(9, 9)], vec![(9, 9)]);
                index.deepest_into(&request, |hash| *hash, &within, cap, |_| drawn, &mut plain);
                let mut below = |_: u32| drawn;
                deepest_within_groups(
                    &index,
                    &groups,
                    &request,
                    BLOCK,
                    &within,
                    cap,
                    &mut below,
                    &mut grouped,
                );
                assert_eq!(grouped, plain, "{said}");
                let scores = index.find_matches(&contents, false).scores;
                let mut scored = vec![(9, 9)];
                deepest_of_within_groups(
                    scores,
                    &groups,
                    &request,
                    BLOCK,
                    &within,
                    cap,
                    &mut below,
                    &mut scored,
                );
                assert_eq!(depths(&scored), depths(&plain), "{said}");
                let holds = |&(number, depth): &(u32, u32)| {
                    let at = all.iter().position(|&held| held == number);
                    at.is_some_and(|at| [3, 2, 2, 1][at] == depth) && among.contains(&number)
                };
                assert!(scored.iter().all(holds), "{said}: {scored:?}");
            }
        }
    }
}
