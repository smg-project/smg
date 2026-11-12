//! The scratch rule, pinned on the `epd` plan: after the first decision on a
//! thread (which sizes the thread's scratch and lets the ring learn the pool),
//! an encode item's decision allocates nothing, over 8, 64 and 512 encoders
//! and under each encode preset.
#![expect(
    unsafe_code,
    reason = "a counting allocator is the one proof of an allocation count; it forwards to the system allocator"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

use smg_policy_controller::{
    Candidate, Constraints, Health, Identity, Phase, PolicySpec, PoolFacts, RequestFacts, Role,
    Scratch, WorkerId,
};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

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
            let before = ALLOCATIONS.load(Ordering::Relaxed);
            let mut picked = 0usize;
            for key in 2..=200u64 {
                let decision = engine.decide(&facts_of(key), &pool_facts, &pool, &cx, &mut scratch);
                picked += decision.pick().map_or(0, |p| p.targets.len());
            }
            let after = ALLOCATIONS.load(Ordering::Relaxed);
            assert!(picked > 0);
            assert_eq!(
                after - before,
                0,
                "{preset} over {n} encoders: {} allocations in 199 decisions",
                after - before
            );
        }
    }
}
