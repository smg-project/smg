//! The scratch rule, pinned on the `epd` plan: after the first decision on a
//! thread (which sizes the thread's scratch and lets the ring learn the pool),
//! an encode item's decision allocates nothing, over 8, 64 and 512 encoders
//! and under each encode preset. The count is the test thread's own
//! (`common::counting_alloc`), so what runs beside it in the process, the
//! harness included, is not charged to the decisions: a second thread
//! allocates inside every measured window.

mod common;

use std::{
    hint::black_box,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use common::counting_alloc::{calls, Counting};
use smg_policy_controller::{
    Candidate, Constraints, Health, Identity, Phase, PolicySpec, PoolFacts, RequestFacts, Role,
    Scratch, WorkerId,
};

#[global_allocator]
static GLOBAL: Counting = Counting;

fn encoders(n: u32) -> Vec<Candidate> {
    (0..n)
        .map(|i| Candidate {
            position: i,
            id: WorkerId(1_000 + u64::from(i)),
            key: 1_000 + i,
            role: Role::Encode,
            health: Health {
                healthy: true,
                ..Health::default()
            },
            ec_compatible: true,
            vision: true,
            ..Candidate::default()
        })
        .collect()
}

#[test]
fn an_encode_items_decision_allocates_nothing_after_the_first() {
    // A second thread that allocates until the case ends, and how many
    // allocations it has made. That number moves after its allocation, so
    // every window below is opened by the deciding thread's count first
    // and stays open until the number moved twice: the allocation between
    // the two moves is made inside the window, and a count that is not the
    // deciding thread's own reads it.
    let stop = Arc::new(AtomicBool::new(false));
    let beside = Arc::new(AtomicUsize::new(0));
    let allocating = {
        let (stop, beside) = (Arc::clone(&stop), Arc::clone(&beside));
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                drop(black_box(Box::new([0u8; 64])));
                beside.fetch_add(1, Ordering::Relaxed);
            }
        })
    };
    for preset in ["consistent_hashing", "least_load", "round_robin"] {
        let engine = PolicySpec::epd_presets(preset, "round_robin", "round_robin")
            .expect("an epd spec")
            .validate()
            .expect("validates");
        for n in [8u32, 64, 512] {
            let pool = encoders(n);
            let facts_of = |key: u64| RequestFacts {
                model: "m",
                phase: Phase::Encode,
                session_key: Identity::hashed(key),
                seed: key,
                now_micros: 1_000_000,
                ..RequestFacts::default()
            };
            let pool_facts = PoolFacts {
                pool_size: n,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            let mut scratch = Scratch::new();
            // The first decision sizes the scratch and teaches the ring.
            engine
                .decide(&facts_of(1), &pool_facts, &pool, &cx, &mut scratch)
                .pick()
                .expect("a pick");
            let before = calls();
            let beside_before = beside.load(Ordering::Relaxed);
            let mut picked = 0usize;
            for key in 2..=200u64 {
                let decision = engine.decide(&facts_of(key), &pool_facts, &pool, &cx, &mut scratch);
                picked += decision.pick().map_or(0, |p| p.targets.len());
            }
            // The window closes only once the second thread allocated in it:
            // its number's second move follows an allocation made after the
            // first.
            let waiting = Instant::now();
            while beside.load(Ordering::Relaxed) < beside_before + 2 {
                assert!(
                    waiting.elapsed() < Duration::from_secs(10),
                    "the second thread made no allocation in 10 s"
                );
                thread::yield_now();
            }
            let allocated = calls() - before;
            assert!(picked > 0);
            assert_eq!(
                allocated, 0,
                "{preset} over {n} encoders: {allocated} allocations in 199 decisions"
            );
        }
    }
    stop.store(true, Ordering::Relaxed);
    allocating.join().expect("the allocating thread");
}
