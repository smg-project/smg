use std::sync::mpsc;

use super::*;
use crate::stages::scorers::tests::{candidate, kept};

fn first_pick(picker: &dyn Picker, facts: &RequestFacts<'_>, pool: &[Candidate]) -> u32 {
    let mut out = Targets::new();
    let kept = kept(pool.len() as u32);
    let scores = vec![0.0; pool.len()];
    picker.pick(facts, pool, &kept, &scores, TieBand::NORMALIZED, &mut out);
    out.first().map(|t| t.position).unwrap_or(u32::MAX)
}

#[test]
fn round_robin_walks_a_set_in_turn_and_keeps_sets_apart() {
    let rr = RoundRobin::new();
    let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
    let facts = RequestFacts::default();
    let seen: Vec<u32> = (0..6).map(|_| first_pick(&rr, &facts, &pool)).collect();
    let start = seen[0];
    for (k, &p) in seen.iter().enumerate() {
        assert_eq!(p, (start as usize + k) as u32 % 3, "{seen:?}");
    }
    // Another set has its own rotation.
    let mut out = Targets::new();
    rr.pick(
        &facts,
        &pool,
        &[0, 1],
        &[0.0; 2],
        TieBand::NORMALIZED,
        &mut out,
    );
    assert!(out.first().is_some());
}

/// A draw counts its turn in the pool's own word where the host lends
/// one, and claims no rotation by the pool's name. Where none is lent
/// the named rotation turns and the word stands.
#[test]
fn a_draw_turns_the_lent_word_and_names_no_rotation() {
    let rr = RoundRobin::new();
    let pool: Vec<Candidate> = (0..16).map(|row| candidate(row, 0)).collect();
    let facts = RequestFacts::default();
    let word = AtomicU64::new(21);
    let lent = PoolFacts {
        word: Some(&word),
        ..PoolFacts::default()
    };
    let drawn = |pool_facts: &PoolFacts<'_>| {
        let mut kept = Vec::new();
        let mut draw = Draw::new(1);
        let mut all = |_: &Candidate| true;
        assert!(rr.draw_first(&facts, pool_facts, &pool, &mut all, &mut kept, &mut draw));
        kept
    };
    // The word's turn, 21, lands on row 5 of 16, and the next on row 6.
    assert_eq!(drawn(&lent), [5, 6, 7, 8]);
    assert_eq!(drawn(&lent), [6, 7, 8, 9]);
    let read = |word: &AtomicU64| word.load(Ordering::Relaxed);
    assert_eq!((read(&word), read(&rr.claims)), (23, 0));
    drawn(&PoolFacts::default());
    assert_eq!((read(&word), read(&rr.claims)), (23, 1));
}

/// Picks made from two threads follow one order: each pick takes the
/// turn behind the pick before it, whichever thread made that one, so
/// no worker is picked again while another of the pool waits for its
/// turn. A thread's thousandth pick of a pool steps the pool's word as
/// its first did. The pool's own turn and a draw's turn alike.
#[test]
fn picks_from_two_threads_follow_one_order() {
    const ROWS: usize = 8;
    const PICKS: usize = 2_048;
    let rr = RoundRobin::new();
    let pool: Vec<Candidate> = (0..ROWS as u32).map(|row| candidate(row, 0)).collect();
    // The rows `pick` lands on, asked of two threads by turns: each ask
    // is answered before the other thread is asked.
    let by_turns = |pick: &(dyn Fn() -> usize + Sync)| {
        std::thread::scope(|scope| {
            let (rows, picked) = mpsc::channel();
            let asks: Vec<_> = (0..2)
                .map(|_| {
                    let (ask, asked) = mpsc::channel::<()>();
                    let rows = rows.clone();
                    scope.spawn(move || {
                        for () in asked {
                            rows.send(pick()).expect("the test listens");
                        }
                    });
                    ask
                })
                .collect();
            let turn = |k: usize| {
                asks[k % 2].send(()).expect("the thread listens");
                picked.recv().expect("its pick")
            };
            (0..PICKS).map(turn).collect::<Vec<usize>>()
        })
    };
    let word = AtomicU64::new(0);
    let alone = || {
        let pool = PoolWord {
            rows: ROWS,
            word: &word,
        };
        let mut all = |_: usize| true;
        rr.pick_alone(&pool, &mut all).expect("a kept turn")
    };
    let lent = AtomicU64::new(0);
    let drawn = || {
        let facts = RequestFacts::default();
        let pool_facts = PoolFacts {
            word: Some(&lent),
            ..PoolFacts::default()
        };
        let (mut kept, mut draw) = (Vec::new(), Draw::new(1));
        let mut all = |_: &Candidate| true;
        assert!(rr.draw_first(&facts, &pool_facts, &pool, &mut all, &mut kept, &mut draw));
        kept[0] as usize
    };
    let entries: [(&str, &(dyn Fn() -> usize + Sync)); 2] =
        [("the pool's own turn", &alone), ("a draw's turn", &drawn)];
    for (entry, pick) in entries {
        let rows = by_turns(pick);
        for (k, pair) in rows.windows(2).enumerate() {
            let next = (pair[0] + 1) % ROWS;
            assert_eq!(pair[1], next, "{entry}: the pick behind pick {k}");
        }
    }
}

#[test]
fn two_sets_of_one_bucket_keep_their_own_rotations() {
    // Keys a and b share a bucket under different tags. A fresh table's
    // counter runs 0, 1, 2, ... per claim: a starts at the row drawn from
    // 0 and b from 1, and seen in turn for a thousand rounds each walks
    // on by one, neither taking the other's word.
    let rr = RoundRobin::new();
    let a = 0x0123_4567_89ab_cdefu64;
    let (bucket_a, tag_a) = RoundRobin::bucket_of(a);
    let b = (1u64..)
        .find(|&k| k != a && RoundRobin::bucket_of(k).0 == bucket_a)
        .expect("a key sharing a's bucket");
    assert_ne!(RoundRobin::bucket_of(b).1, tag_a, "b's tag differs");
    let (start_a, start_b) = (RoundRobin::start_of(0), RoundRobin::start_of(1));
    for n in 0..1_000 {
        assert_eq!(
            [rr.advance(a), rr.advance(b)],
            [start_a + n, start_b + n],
            "round {n}"
        );
    }
}

#[test]
fn a_ninth_set_of_a_full_bucket_takes_the_word_the_counter_names() {
    // Nine keys share a bucket of eight words. The ninth, claim 8, takes
    // word 0 from the first set; the first, seen again, claims at 9 and
    // takes word 1 from the second; a set whose word nobody took walks
    // on. Each claim starts at the row drawn from its counter value.
    let rr = RoundRobin::new();
    let bucket = RoundRobin::bucket_of(0x0123_4567_89ab_cdef).0;
    let keys: Vec<u64> = (1u64..)
        .filter(|&k| RoundRobin::bucket_of(k).0 == bucket)
        .take(9)
        .collect();
    let start = RoundRobin::start_of;
    for (n, &k) in keys[..8].iter().enumerate() {
        assert_eq!(
            rr.advance(k),
            start(n as u64),
            "each claims a word of its own"
        );
    }
    assert_eq!(rr.advance(keys[8]), start(8), "the ninth takes word 0");
    assert_eq!(rr.advance(keys[2]), start(2) + 1, "the third walks on");
    assert_eq!(
        rr.advance(keys[0]),
        start(9),
        "the first draws anew, at word 1"
    );
    assert_eq!(rr.advance(keys[8]), start(8) + 1, "the ninth walks on");
    assert_eq!(rr.advance(keys[1]), start(10), "the second draws anew");
}

#[test]
fn nine_sets_cycling_one_bucket_take_every_row() {
    // Nine keys of one bucket visited in a fixed cycle: eight words cannot
    // hold them, so every visit past the first eight draws anew, at the
    // row its counter value draws. Reduced to 2, 3, 4, 9, 12 or 14 rows,
    // every set takes every row within a third of its even share over a
    // thousand rounds, where a start that advanced by the cycle's length
    // fixed one row, and one multiply left rows sharing a factor with
    // the cycle's stride unvisited.
    let rr = RoundRobin::new();
    let bucket = RoundRobin::bucket_of(0x0123_4567_89ab_cdef).0;
    let keys: Vec<u64> = (1u64..)
        .filter(|&k| RoundRobin::bucket_of(k).0 == bucket)
        .take(9)
        .collect();
    const ROWS: [usize; 6] = [2, 3, 4, 9, 12, 14];
    let rounds = 1_000u64;
    let mut taken = vec![[[0u64; 14]; ROWS.len()]; 9];
    for round in 0..rounds {
        for (n, &k) in keys.iter().enumerate() {
            let position = rr.advance(k);
            assert_eq!(
                position,
                RoundRobin::start_of(9 * round + n as u64),
                "round {round}, set {n}"
            );
            for (r, &rows) in ROWS.iter().enumerate() {
                taken[n][r][(position % rows as u64) as usize] += 1;
            }
        }
    }
    for (n, set) in taken.iter().enumerate() {
        for (r, &rows) in ROWS.iter().enumerate() {
            for (row, &count) in set[r][..rows].iter().enumerate() {
                assert!(
                    count * rows as u64 * 3 >= rounds,
                    "set {n} over {rows} rows: row {row} taken {count} of {rounds}"
                );
            }
        }
    }
}

#[test]
fn survivor_sets_that_differ_by_rank_alone_rotate_apart() {
    // A is a two-rank worker whose rows share its id; a per-rank bit drops
    // one rank at a time, so the survivors alternate between [A1, B] and
    // [A0, B]. Each set keeps its own rotation and walks both of its rows.
    let rr = RoundRobin::new();
    let mut pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
    pool[1].id = pool[0].id;
    pool[1].dp_rank = 1;
    let x = [1u32, 2];
    let y = [0u32, 2];
    assert_ne!(
        RoundRobin::set_key(&pool, &x),
        RoundRobin::set_key(&pool, &y)
    );
    let facts = RequestFacts::default();
    let mut seen_x = [0usize; 3];
    let mut seen_y = [0usize; 3];
    for _ in 0..4 {
        for (kept, seen) in [(&x[..], &mut seen_x), (&y[..], &mut seen_y)] {
            let mut out = Targets::new();
            rr.pick(
                &facts,
                &pool,
                kept,
                &[0.0; 2],
                TieBand::NORMALIZED,
                &mut out,
            );
            seen[out.first().expect("a pick").position as usize] += 1;
        }
    }
    assert_eq!(seen_x, [0, 2, 2], "X walks A1 and B in turn: {seen_x:?}");
    assert_eq!(seen_y, [2, 0, 2], "Y walks A0 and B in turn: {seen_y:?}");
}
