//! The scratch rule, pinned on the `epd` plan: after the first decision on a
//! thread (which sizes the thread's scratch and lets the ring learn the pool),
//! an encode item's decision allocates nothing, over 8, 64 and 512 encoders
//! and under each encode preset; and neither does the plan's one decision
//! after its first 32, the pair with its items, with every row admitted or
//! with rows left out, and under a mask on every side that those first
//! decisions did not meet.
//! The count is the test thread's own
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
    document::PolicySpec,
    engine::{
        Candidate, Constraints, EncodeItems, EncodePlaced, EncodePool, EncoderFacts, Health,
        Identity, MediaCounts, PairPools, Phase, PoolFacts, RequestFacts, Role, Scratch, WorkerId,
    },
};

#[global_allocator]
static GLOBAL: Counting = Counting(std::alloc::System);

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

/// A second thread that allocates until it is stopped, and how many
/// allocations it has made. That number moves after its allocation, so a
/// window is opened by the deciding thread's count first and stays open
/// until the number moved twice: the allocation between the two moves is
/// made inside the window, and a count that is not the deciding thread's
/// own reads it. Each move is released and each read acquires it. The
/// thread is stopped and joined where its owner ends, by a failed
/// assertion too.
struct Beside {
    stop: Arc<AtomicBool>,
    made: Arc<AtomicUsize>,
    allocating: Option<thread::JoinHandle<()>>,
}

impl Beside {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let made = Arc::new(AtomicUsize::new(0));
        let allocating = {
            let (stop, made) = (Arc::clone(&stop), Arc::clone(&made));
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    drop(black_box(Box::new([0u8; 64])));
                    made.fetch_add(1, Ordering::Release);
                }
            })
        };
        Self {
            stop,
            made,
            allocating: Some(allocating),
        }
    }

    /// The deciding thread's allocations while `run` ran, in a window the
    /// second thread allocated in.
    fn allocations_of(&self, run: impl FnOnce()) -> usize {
        let before = calls();
        let beside_before = self.made.load(Ordering::Acquire);
        run();
        // The window closes only once the second thread allocated in it:
        // its number's second move follows an allocation made after the
        // first.
        let waiting = Instant::now();
        while self.made.load(Ordering::Acquire) < beside_before + 2 {
            assert!(
                waiting.elapsed() < Duration::from_secs(10),
                "the second thread made no allocation in 10 s"
            );
            thread::yield_now();
        }
        calls() - before
    }
}

impl Drop for Beside {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let ended = self.allocating.take().map(thread::JoinHandle::join);
        // A case that is failing already keeps its own failure.
        if !thread::panicking() {
            assert!(
                matches!(ended, Some(Ok(()))),
                "the allocating thread ended by a panic"
            );
        }
    }
}

#[test]
fn an_encode_items_decision_allocates_nothing_after_the_first() {
    let beside = Beside::start();
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
            let mut picked = 0usize;
            let allocated = beside.allocations_of(|| {
                for key in 2..=200u64 {
                    let decision =
                        engine.decide(&facts_of(key), &pool_facts, &pool, &cx, &mut scratch);
                    picked += decision.pick().map_or(0, |p| p.targets.len());
                }
            });
            assert!(picked > 0);
            assert_eq!(
                allocated, 0,
                "{preset} over {n} encoders: {allocated} allocations in 199 decisions"
            );
        }
    }
}

/// The rows of one side of a pair: ready, of pair class 1, every admission
/// bit set.
fn side(role: Role, first: u32, n: u32) -> Vec<Candidate> {
    (0..n)
        .map(|i| Candidate {
            position: i,
            id: WorkerId(u64::from(first + i)),
            key: first + i,
            role,
            health: Health {
                healthy: true,
                ..Health::default()
            },
            pair_class: 1,
            has_capacity: true,
            has_room: true,
            vision: true,
            ..Candidate::default()
        })
        .collect()
}

/// The plan's one decision, the pair with three items, over two encode
/// classes that the prefills and the encoders take in turn. With rows left
/// out, the second class's encoders are not ready and every other decode
/// takes no image while the request asks one of its decode: the pair is
/// picked over pools of admitted rows. The first decisions size the
/// scratch, the admitted pools and the placed list, and let the ring learn
/// the encoders of both classes. They meet no mask: the measured decisions
/// meet one on every side from the hundredth on, and allocate nothing for
/// the lists it is numbered anew into. Under one more engine the encode
/// profile holds no hard filter that reads the mask: the masked encoder is
/// offered, and the mask is numbered with the encoders offered.
#[test]
fn an_encode_pair_decision_allocates_nothing_after_its_first_32() {
    let beside = Beside::start();
    for (preset, unfiltered) in [
        ("consistent_hashing", false),
        ("least_load", false),
        ("round_robin", false),
        ("round_robin", true),
    ] {
        let mut spec =
            PolicySpec::epd_presets(preset, "round_robin", "round_robin").expect("an epd spec");
        if unfiltered {
            let encode = spec.profiles.get_mut("encode").expect("the encode profile");
            encode.filter.retain(|entry| entry.name() != "healthy");
        }
        let engine = spec.validate().expect("validates");
        for n in [8u32, 64, 512] {
            for left_out in [false, true] {
                let prefills = side(Role::Prefill, 10_000, n);
                let mut decodes = side(Role::Decode, 20_000, n);
                let mut pool = side(Role::Encode, 30_000, n);
                if left_out {
                    for row in decodes.iter_mut().skip(1).step_by(2) {
                        row.vision = false;
                    }
                    for row in pool.iter_mut().skip(1).step_by(2) {
                        row.health.healthy = false;
                    }
                }
                let of_encoders: Vec<EncoderFacts> = (0..n)
                    .map(|i| EncoderFacts {
                        class: 1 + i % 2,
                        pushes: true,
                        references: false,
                    })
                    .collect();
                let prefill_classes: Vec<u32> = (0..n).map(|i| 1 + i % 2).collect();
                let pools = PairPools {
                    prefills: &prefills,
                    decodes: &decodes,
                    ..PairPools::default()
                };
                let encode = EncodePool {
                    encoders: &pool,
                    facts: &of_encoders,
                    prefill_classes: &prefill_classes,
                    affinity: &[],
                    ..EncodePool::default()
                };
                let pool_facts = PoolFacts {
                    pool_size: n,
                    generation: 1,
                    ..PoolFacts::default()
                };
                let cx = Constraints::default();
                // The first row of each pool, masked.
                let stalled = [0u32];
                let masked = Constraints {
                    stalled: &stalled,
                    ..Constraints::default()
                };
                let mut scratch = Scratch::new();
                let mut placed = EncodePlaced::new();
                let mut decide = |seed: u64, cx: &Constraints<'_>| {
                    let keys = [1u64, 2, 3].map(|item| Identity::hashed(seed * 16 + item));
                    let facts = RequestFacts {
                        model: "m",
                        media: MediaCounts {
                            images: 1,
                            ..MediaCounts::default()
                        },
                        seed,
                        now_micros: 1_000_000,
                        ..RequestFacts::default()
                    };
                    let items = EncodeItems {
                        keys: &keys,
                        one: None,
                        references: false,
                        decode_images: left_out,
                    };
                    let decision = engine.decide_encode_pair(
                        &facts,
                        &pool_facts,
                        pools,
                        encode,
                        items,
                        cx,
                        cx,
                        cx,
                        &mut scratch,
                        &mut placed,
                    );
                    decision.pick().expect("a pick");
                    placed.encoders().len()
                };
                for seed in 1..=32u64 {
                    assert_eq!(decide(seed, &cx), 3);
                }
                let mut encoders_placed = 0usize;
                let allocated = beside.allocations_of(|| {
                    for seed in 33..=231u64 {
                        let cx = if seed < 132 { &cx } else { &masked };
                        encoders_placed += decide(seed, cx);
                    }
                });
                assert_eq!(encoders_placed, 3 * 199);
                assert_eq!(
                    allocated, 0,
                    "{preset} over {n} of each side, rows left out: {left_out}, no hard filter on the mask: {unfiltered}: {allocated} allocations in 199 decisions"
                );
            }
        }
    }
}
