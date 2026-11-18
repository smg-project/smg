//! The cost of a decision over a publication's own rows, beside the same
//! decision over rows written for it, at 8, 64 and 512 members of one pool.
//! Both doors are timed in this binary over one store, so a row and its
//! twin are one build:
//!
//! - `policy_store_lent/view/<door>/<n>`: one decision's view alone, under
//!   no veto and no pin. `built` copies the pool's rows and writes them
//!   (`World::build`); `lent` hands the published rows where they stand
//!   (`World::view`);
//! - `policy_store_lent/view_veto/<door>/<n>` and `view_pin/<door>/<n>`:
//!   the same under one standing veto of the asking gateway, and under a
//!   pin;
//! - `policy_store_lent/view_local/<door>/<n>`: the same in a pool that
//!   holds another gateway's host-local row, where both doors copy;
//! - `policy_store_lent/decide/<preset>/<door>/<n>`: the whole decision,
//!   the view and the engine's answer over it, under each preset of the
//!   build that places a request over this fleet.
//!
//! A bench target of its own: the store bench is built on a pull request's
//! base with the pull request's bench sources, and the base's store has no
//! door that lends. No gate reads these rows yet.

#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

#[path = "common/filled.rs"]
mod filled;

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    store::{
        Ask, GatewayId, HealthSink, LoadSink, LoadSource, MemoryStore, Reachability, ViewBuffers,
        WorkerFacts, WorkerStatus, WorkersStore, PUBLISH_WINDOW,
    },
    Constraints, Engine, LoadRow, PolicySpec, RequestFacts, Role, Scratch, SignalSource, WorkerId,
    PRESETS,
};

const SIZES: [usize; 3] = [8, 64, 512];
const MODEL: &str = "m";
const ASKER: GatewayId = GatewayId(1);

/// Member `index`'s record: figures that differ per member, so no two rows
/// read alike.
fn record_of(index: usize) -> LoadRow {
    let x = (index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    LoadRow {
        running: (x >> 16) as u32 % 12,
        queued: (x >> 24) as u32 % 4,
        kv_usage_permille: ((x >> 32) % 900) as u16,
        live_rate: 1_000 + ((x >> 40) % 3_000) as u32,
        report_age_ms: 5,
        source: SignalSource::Events,
        ..LoadRow::default()
    }
}

/// What stands in a pool beside its members.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Beside {
    Nothing,
    /// One standing veto of the asking gateway.
    Veto,
    /// One more member, host-local to another gateway.
    Local,
}

/// A store with one pool of `n` members, every one registered, ready, with
/// its record on file and published: the store, its members by index and
/// the instant its last window opened.
fn pool_of(n: usize, beside: Beside) -> (MemoryStore, Vec<WorkerId>, Instant) {
    let store = MemoryStore::new();
    let opened = Instant::now();
    let register = |key: String, reachable_from: Reachability| {
        let id = store
            .upsert(WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                models: vec![MODEL.to_string()],
                dp_size: 1,
                max_running: Some(4),
                reachable_from,
                ..WorkerFacts::default()
            })
            .id();
        let revision = WorkersStore::get(&store, id).expect("the row").revision;
        store
            .set_status(id, WorkerStatus::Ready, revision)
            .expect("ready");
        id
    };
    let ids: Vec<WorkerId> = (0..n)
        .map(|index| {
            let id = register(format!("http://w{index}:8000"), Reachability::Any);
            store.record(id, 0, record_of(index), opened, LoadSource::Stream);
            id
        })
        .collect();
    match beside {
        Beside::Nothing => {}
        Beside::Veto => {
            let until = Some(opened + Duration::from_secs(3_600));
            store.gateway_veto(ASKER, ids[1], until);
        }
        Beside::Local => {
            let theirs = Reachability::Gateway(GatewayId(2));
            register("unix:///local".to_string(), theirs);
        }
    }
    let opened = opened + PUBLISH_WINDOW;
    store.publish(opened);
    assert_eq!(store.publish_due(), None);
    let rows = n + usize::from(beside == Beside::Local);
    filled::filled(&store, MODEL, Role::Single, rows);
    (store, ids, opened)
}

fn ask_at(opened: Instant, pinned: Option<WorkerId>, affinity: bool) -> Ask {
    Ask {
        gateway: ASKER,
        now: opened + Duration::from_secs(1),
        pinned,
        affinity,
    }
}

/// One view through each door in turn: `built` and `lent`.
fn view_cells(c: &mut Criterion, name: &str, beside: Beside, pin: bool) {
    let mut group = c.benchmark_group(format!("policy_store_lent/{name}"));
    for n in SIZES {
        let (store, ids, opened) = pool_of(n, beside);
        let ask = ask_at(opened, pin.then(|| ids[n / 2]), false);
        let mut out = ViewBuffers::default();
        group.bench_with_input(BenchmarkId::new("built", n), &n, |b, _| {
            b.iter(|| {
                let built = store.build_view(MODEL, Role::Single, black_box(&ask), &mut out);
                black_box(&out.candidates);
                black_box(built)
            });
        });
        group.bench_with_input(BenchmarkId::new("lent", n), &n, |b, _| {
            b.iter(|| {
                store.with_world(|world| {
                    let view = world.view(MODEL, Role::Single, black_box(&ask), false, &mut out);
                    let view = view.expect("the pool");
                    black_box(view.candidates);
                    black_box((view.stalled.len(), view.pinned_at, view.built))
                })
            });
        });
    }
    group.finish();
}

fn engine_of(preset: &str) -> Engine {
    PolicySpec::preset(preset)
        .expect("a preset of this build")
        .validate()
        .expect("a preset validates")
}

fn facts<'a>(seed: u64, now_micros: u64) -> RequestFacts<'a> {
    RequestFacts {
        model: MODEL,
        prompt_tokens: 512,
        seed,
        now_micros,
        ..RequestFacts::default()
    }
}

/// The whole decision through each door in turn, under every preset that
/// places a request over this fleet.
fn decide_cells(c: &mut Criterion) {
    for preset in PRESETS {
        let mut group = c.benchmark_group(format!("policy_store_lent/decide/{preset}"));
        for n in SIZES {
            let (store, _, opened) = pool_of(n, Beside::Nothing);
            let (built_engine, lent_engine) = (engine_of(preset), engine_of(preset));
            let ask = ask_at(opened, None, built_engine.needs().affinity);
            let mut out = ViewBuffers::default();
            let mut scratch = Scratch::new();
            let over_built = |seed: u64, out: &mut ViewBuffers, scratch: &mut Scratch| {
                let built = store
                    .build_view(MODEL, Role::Single, black_box(&ask), out)
                    .expect("the pool");
                let facts = RequestFacts {
                    affinity: &out.affinity,
                    ..facts(seed, built.now_micros)
                };
                let cx = Constraints::default();
                built_engine.decide(&facts, &built.pool_facts(), &out.candidates, &cx, scratch)
            };
            let over_lent = |seed: u64, out: &mut ViewBuffers, scratch: &mut Scratch| {
                store.with_world(|world| {
                    let view = world.view(MODEL, Role::Single, black_box(&ask), false, out);
                    let view = view.expect("the pool");
                    let facts = RequestFacts {
                        pinned_at: view.pinned_at,
                        affinity: view.affinity,
                        ..facts(seed, view.built.now_micros)
                    };
                    let cx = Constraints {
                        stalled: view.stalled,
                        ..Constraints::default()
                    };
                    let pool = view.built.pool_facts();
                    lent_engine.decide(&facts, &pool, view.candidates, &cx, scratch)
                })
            };
            // A preset that places nobody over this fleet times nothing.
            if over_built(1, &mut out, &mut scratch).pick().is_none() {
                continue;
            }
            let mut seed = 1u64;
            group.bench_with_input(BenchmarkId::new("built", n), &n, |b, _| {
                b.iter(|| {
                    seed = seed.wrapping_add(1);
                    black_box(over_built(seed, &mut out, &mut scratch))
                });
            });
            group.bench_with_input(BenchmarkId::new("lent", n), &n, |b, _| {
                b.iter(|| {
                    seed = seed.wrapping_add(1);
                    black_box(over_lent(seed, &mut out, &mut scratch))
                });
            });
        }
        group.finish();
    }
}

fn store_lent(c: &mut Criterion) {
    view_cells(c, "view", Beside::Nothing, false);
    view_cells(c, "view_veto", Beside::Veto, false);
    view_cells(c, "view_pin", Beside::Nothing, true);
    view_cells(c, "view_local", Beside::Local, false);
    decide_cells(c);
}

criterion_group!(benches, store_lent);
criterion_main!(benches);
