//! The cost of the encode classes over one model of `n` prefill and decode
//! workers, a quarter of them prefills, and `n / 8` encoders, at `n` of 8,
//! 64 and 512:
//!
//! - `policy_encodes/derive/<fleet>/<n>`: one derivation over the encoders
//!   and the prefills;
//! - `policy_encodes/membership_write/<fleet>/<n>`: one membership write of
//!   a store that holds the fleet: a registration refreshed with the facts
//!   it had, every pool derived again, the classes carried, and published;
//! - `policy_encodes/moved_write/<fleet>/<n>`: the same write where the
//!   registration states another runtime each time: the model's classes,
//!   the pair classes and the encode classes, are derived;
//! - `policy_encodes/build/pair/<fleet>/<n>` and
//!   `policy_encodes/build/encode_pair/<fleet>/<n>`: one decision's build
//!   through the pair's door, and through the encode pair's, which adds
//!   the encode pool and its two columns.
//!
//! The fleets, each of one runtime and one wire:
//!
//! - `none`: no encoder at all, what a model with no encode stage pays;
//! - `silent`: nobody states an end of the encode handoff, as a store holds
//!   its workers until a handshake fills one: no class;
//! - `meeting`: every encoder and prefill meets at a rendezvous: one class;
//! - `connectors`: every encoder and prefill pushes on one of eight
//!   connectors: a class a connector that has an encoder;
//! - `apart`: every encoder pushes on a connector of its own and the
//!   prefills take from them in turn: a class an encoder.
//!
//! Before the timed cells the bench prints, per fleet and size, what the
//! derivation holds: its classes and the bytes of their tables.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

use std::{hint::black_box, time::Instant};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use smg_policy_controller::{
    store::{
        encoding::EncodeMember, Ask, EncodeClasses, EncodeHandoff, EncodeOffer, EncodePairBuffers,
        GatewayId, MemoryStore, PairBuffers, WorkerFacts, WorkersStore,
    },
    Role, WireModes,
};

const SIZES: [usize; 3] = [8, 64, 512];
const MODEL: &str = "m";
const FLEETS: [&str; 5] = ["none", "silent", "meeting", "connectors", "apart"];

/// What worker `index` of `role` states in `fleet`, a fleet of `stage`
/// encoders.
fn offer(fleet: &str, role: Role, index: usize, stage: usize) -> EncodeOffer {
    let push = |connector: usize| EncodeOffer {
        handoff: EncodeHandoff::Push,
        producer: role == Role::Encode,
        consumer: role == Role::Prefill,
        connector: Some(format!("c{connector}")),
        ..EncodeOffer::default()
    };
    match (fleet, role) {
        ("none" | "silent", _) | (_, Role::Decode | Role::Single) => EncodeOffer::default(),
        ("meeting", _) => EncodeOffer {
            handoff: EncodeHandoff::Rendezvous,
            ..EncodeOffer::default()
        },
        ("connectors", _) => push(index % 8),
        _ => push(index % stage),
    }
}

/// The fleet at `n`: `n / 4` prefills, the rest of `n` decodes, then
/// `n / 8` encoders, or none of them in the fleet `none`.
fn fleet(fleet: &str, n: usize) -> Vec<WorkerFacts> {
    let stage = if fleet == "none" { 0 } else { n / 8 };
    let worker = |role: Role, side: &str, index: usize| {
        let key = format!("grpc://{side}{index}:9000");
        WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            role,
            wire_modes: WireModes::GRPC,
            engine_kind: "rt-a".to_string(),
            models: vec![MODEL.to_string()],
            dp_size: 1,
            encode: offer(fleet, role, index, stage),
            ..WorkerFacts::default()
        }
    };
    let prefills = (0..n / 4).map(|index| worker(Role::Prefill, "p", index));
    let decodes = (0..n - n / 4).map(|index| worker(Role::Decode, "d", index));
    let encoders = (0..stage).map(|index| worker(Role::Encode, "e", index));
    prefills.chain(decodes).chain(encoders).collect()
}

/// One role's workers as the encode handoff's rule reads them.
fn members(fleet: &[WorkerFacts], role: Role) -> Vec<EncodeMember<'_>> {
    let of_role = fleet.iter().filter(|facts| facts.role == role);
    of_role
        .map(|facts| EncodeMember {
            runtime: &facts.engine_kind,
            offer: &facts.encode,
        })
        .collect()
}

/// A store that holds `fleet`.
fn store_of(fleet: &[WorkerFacts]) -> MemoryStore {
    let store = MemoryStore::new();
    for facts in fleet {
        store.upsert(facts.clone());
    }
    store
}

#[expect(clippy::print_stdout, reason = "the sizes are the bench's report")]
fn sizes() {
    for name in FLEETS {
        for n in SIZES {
            let fleet = fleet(name, n);
            let encoders = members(&fleet, Role::Encode);
            let classes = EncodeClasses::derive(encoders, members(&fleet, Role::Prefill));
            println!(
                "policy_encodes/sizes/{name}/{n}: classes={} bytes={}",
                classes.classes(),
                classes.bytes(),
            );
        }
    }
}

fn derivation(c: &mut Criterion) {
    sizes();
    let mut group = c.benchmark_group("policy_encodes/derive");
    for name in FLEETS {
        for n in SIZES {
            let fleet = fleet(name, n);
            let encoders = members(&fleet, Role::Encode);
            let prefills = members(&fleet, Role::Prefill);
            group.bench_function(BenchmarkId::new(name, n), |b| {
                b.iter(|| {
                    EncodeClasses::derive(
                        black_box(&encoders).iter().copied(),
                        black_box(&prefills).iter().copied(),
                    )
                });
            });
        }
    }
    group.finish();
}

/// A membership write that moves no offer: the classes are carried.
fn membership_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_encodes/membership_write");
    for name in FLEETS {
        for n in SIZES {
            let fleet = fleet(name, n);
            let store = store_of(&fleet);
            let refreshed = fleet.first().expect("a prefill").clone();
            group.bench_function(BenchmarkId::new(name, n), |b| {
                b.iter_batched(
                    || refreshed.clone(),
                    |facts| store.upsert(black_box(facts)),
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

/// A membership write that moves an offer: the first prefill registers with
/// another runtime each time, so every write derives the model's classes.
fn moved_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_encodes/moved_write");
    for name in FLEETS {
        for n in SIZES {
            let fleet = fleet(name, n);
            let store = store_of(&fleet);
            let first = fleet.first().expect("a prefill").clone();
            let other = WorkerFacts {
                engine_kind: "rt-b".to_string(),
                ..first.clone()
            };
            let mut turn = false;
            group.bench_function(BenchmarkId::new(name, n), |b| {
                b.iter_batched(
                    || {
                        turn = !turn;
                        if turn {
                            other.clone()
                        } else {
                            first.clone()
                        }
                    },
                    |facts| store.upsert(black_box(facts)),
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

/// One decision's build through each pair door of a store that holds the
/// fleet: the pair's door, and the encode pair's, which adds the encode
/// pool and its two columns.
fn doors(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_encodes/build");
    for name in FLEETS {
        for n in SIZES {
            let store = store_of(&fleet(name, n));
            let ask = Ask {
                gateway: GatewayId(1),
                now: Instant::now(),
                pinned: None,
                affinity: false,
            };
            let mut pair = PairBuffers::default();
            assert!(
                store.build_pair(MODEL, &ask, &mut pair).is_some(),
                "the fleet holds both sides of a pair"
            );
            group.bench_function(BenchmarkId::new(format!("pair/{name}"), n), |b| {
                b.iter(|| {
                    store
                        .build_pair(MODEL, black_box(&ask), &mut pair)
                        .is_some()
                });
            });
            let mut out = EncodePairBuffers::default();
            group.bench_function(BenchmarkId::new(format!("encode_pair/{name}"), n), |b| {
                b.iter(|| {
                    store
                        .build_encode_pair(MODEL, black_box(&ask), &mut out)
                        .is_some()
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, derivation, membership_write, moved_write, doors);
criterion_main!(benches);
