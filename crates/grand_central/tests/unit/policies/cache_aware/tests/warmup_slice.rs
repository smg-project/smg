//! The `warmup-slice` stage: the misses it slices to a worker with a thin
//! or young cache, and the hits it diverts to a thin one.

use super::*;

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
    // sizes do: read then, they are the ruling's own of that instant. One
    // ruling ends: thin rows that all hold nothing, once the last of them
    // ended, where a host rules again.
    let ends_at = |rule: &WarmUpRule, fleet: &[(u32, u64)]| {
        let sizes: Vec<u32> = fleet.iter().map(|row| row.0).collect();
        let admitted: Vec<u64> = fleet.iter().map(|row| NOW - row.1 * 1_000_000).collect();
        let (mut scratch, mut rows) = (Vec::new(), Vec::new());
        let warming = rule.list_warming_rows(&sizes, &admitted, NOW, &mut scratch, &mut rows);
        let last = rows.iter().map(|&(_, until)| until).max();
        last.filter(|_| warming.thin).unwrap_or(u64::MAX)
    };
    let laters = [
        1, 29_999_999, 30_000_000, 39_999_999, 40_000_000, 49_999_999, 50_000_000, 70_000_000,
    ];
    let mut ruled_again = 0;
    for fleet in fleets {
        let end = ends_at(&rule, fleet);
        for later in laters {
            let now = NOW + later;
            let read = if now < end { NOW } else { end };
            ruled_again += usize::from(now >= end);
            let stood = read_at(&rule, fleet, read, now);
            assert_eq!(stood, named_by(&rule, fleet, now), "{fleet:?} +{later}");
        }
    }
    // The second fleet's one thin row holds nothing, ten seconds into its
    // window: its ruling ends fifty seconds on, and no other ruling ends.
    assert_eq!(ends_at(&rule, fleets[1]), NOW + 50_000_000);
    assert_eq!(ruled_again, 2);
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

/// A row that holds nothing is thin for the window of its warm-up clock
/// alone: a fleet half warm where no worker joined and no index was emptied
/// names nobody, a row whose clock started inside the window is named until
/// the window is over, and from its first block a row is named with no end
/// until it crosses the ratio. A column says no more of a worker's age than
/// its admission, so the stage's own ruling of one keeps a row that holds
/// nothing thin whatever its age.
#[test]
fn a_row_that_holds_nothing_is_thin_for_its_window_alone() {
    const NOW: u64 = 600_000_000;
    const WINDOW: u64 = 60_000_000;
    let stage = warmup(0.25, 8);
    let rule = stage.warm_up_rule().expect("the stage rules who warms");
    // A row: its index size and how many seconds ago its clock started.
    let named = |fleet: &[(u32, u64)]| {
        let sizes: Vec<u32> = fleet.iter().map(|row| row.0).collect();
        let started: Vec<u64> = fleet.iter().map(|row| NOW - row.1 * 1_000_000).collect();
        let (mut scratch, mut rows) = (Vec::new(), Vec::new());
        let warming = rule.list_warming_rows(&sizes, &started, NOW, &mut scratch, &mut rows);
        (rows, warming.thin)
    };
    // Half the fleet warm, nobody new, nothing emptied: nobody warms.
    let half_warm = [(40, 500), (40, 500), (0, 500), (0, 500)];
    assert_eq!(named(&half_warm), (vec![], false));
    assert_eq!(
        named(&[(40, 500), (40, 500), (40, 500), (0, 500)]),
        (vec![], false)
    );
    // A row whose clock started ten seconds ago, by a join or by an
    // emptying, is thin until its window is over; the old one beside it is
    // not.
    assert_eq!(
        named(&[(40, 500), (40, 500), (0, 10), (0, 500)]),
        (vec![(2, NOW - 10_000_000 + WINDOW)], true)
    );
    // From its first block a row is thin with no end, whatever its age.
    assert_eq!(
        named(&[(40, 500), (40, 500), (1, 500), (0, 500)]),
        (vec![(2, u64::MAX)], true)
    );
    // The stage's own ruling of a column: the rows that hold nothing stay
    // thin whatever their age.
    let column: Vec<AffinityFacts> = half_warm
        .iter()
        .map(|&(size, age)| AffinityFacts {
            admitted_age_ms: (age * 1_000) as u32,
            indexed_growth: size,
            indexed_blocks: size,
            ..AffinityFacts::default()
        })
        .collect();
    let warm = stage.ruled(&column).expect("a thin row is ruled");
    let mut warming = stage.warming::<false>(warm, Feed::Column(&column));
    let rows: Vec<u32> = (0..4).filter(|&row| warming(row)).collect();
    assert_eq!((rows, warm.thin()), (vec![2, 3], true));
}

/// An emptied row is thin for a window while it holds nothing and is not
/// young again: the young rule reads a row's admission, and the instant its
/// index was emptied is read for the thin row that holds nothing alone. So
/// a row emptied ten seconds ago that regrew past the ratio is named no
/// more, though it gained less than a young row's first blocks since. A
/// host that hands the emptying as the row's admission names it young.
#[test]
fn an_emptied_row_regrown_past_the_ratio_is_not_young() {
    const NOW: u64 = 600_000_000;
    const WINDOW: u64 = 60_000_000;
    let stage = warmup(0.25, 8);
    let rule = stage.warm_up_rule().expect("the stage rules who warms");
    let old = NOW - 500_000_000;
    let admitted = [old; 4];
    let emptied = [old, old, NOW - 10_000_000, old];
    let named = |sizes: [u32; 4], admitted: &[u64], emptied: &[u64]| {
        let (mut scratch, mut rows) = (Vec::new(), Vec::new());
        let warming = rule.list_warming_rows_emptied(
            &sizes,
            &sizes,
            admitted,
            emptied,
            NOW,
            &mut scratch,
            &mut rows,
        );
        (rows, warming.thin)
    };
    // Emptied, it holds nothing: thin until its window is over.
    assert_eq!(
        named([12, 12, 0, 12], &admitted, &emptied),
        (vec![(2, NOW - 10_000_000 + WINDOW)], true)
    );
    // Regrown past the ratio of the level, short of a young row's first
    // blocks: neither thin nor young.
    assert_eq!(named([12, 12, 7, 12], &admitted, &emptied), (vec![], false));
    // The emptying handed as the admission: young until the window is over.
    assert_eq!(
        named([12, 12, 7, 12], &emptied, &emptied),
        (vec![(2, NOW - 10_000_000 + WINDOW)], false)
    );
}
