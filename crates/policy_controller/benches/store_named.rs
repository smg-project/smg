//! One cache-aware decision through the store, as one job: the request's
//! tokens hashed into blocks, the pool's rows read where the published world
//! holds them (lent, the store's door for a pool every gateway reaches), the
//! rows named in the affinity column's place (the request's deepest holders
//! by one lookup of the pool's index under the cap of the picker's sample)
//! and the `cache_aware` preset's decision over them. The wire's decider
//! lends rows to a profile that declares no count of requests in flight and
//! writes them for one that does, as the preset's affinity stage does: these
//! cells are the naming and the decision over rows already in hand. Ids:
//! `policy_store_named/<hit|shared|miss>/<n>` at 8 to 1,024 workers, and
//! `policy_store_named/column/<n>` for the same miss decided over the
//! column a profile named no row is handed. Every worker holds one prompt
//! of four blocks of its own and one prompt the whole fleet holds, stored
//! through the KV sink: a hit asks for the workers' own prompts in turn (one
//! holder each), `shared` for the prompt every worker holds (the cap cuts
//! one depth of the whole pool), a miss for prompts nobody holds.
//!
//! `policy_store_named/best_hit/<n>`, `best_shared/<n>` and `best_miss/<n>`
//! are the three decisions over a fleet of the same making under the
//! preset's stages and a picker that takes no sample: such a profile is
//! named every holder and every warming row, under no cap, and its pass
//! reads every row. `best_column/<n>` is that miss decided over the column
//! instead: the two feeds of one decision, side by side.
//!
//! `policy_store_named/viewed_hit/<n>` and `viewed_shared/<n>` are the
//! first cells' hit and shared decision over a fleet each of whose workers
//! also holds a view of its cache groups: one state group that retains the
//! last block of each prompt. Every holder of the request is scored and
//! read at what its groups serve, which is all it holds, before the cap
//! cuts: the price of the views on one decision.
//!
//! `policy_store_named/publish/<n>` is one window's publication of that
//! fleet: under the preset's warm-up ruling it reads each row's index size
//! and rules the pool's warming rows again.
//!
//! A bench target of its own: the store's gated rows are the `store_view`
//! binary's.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use kv_index::{compute_content_hash, request_prefix_hashes, ContentHash, GroupKind};
use smg_policy_controller::{
    store::{
        Ask, BatchFacts, BatchOutcome, GatewayId, KvEvent, KvEventSink, LoadSink, LoadSource,
        MemoryStore, Named, Naming, Signals, StoredBlock, ViewBuffers, WorkerFacts, WorkerStatus,
        WorkersStore, PUBLISH_WINDOW,
    },
    AppliedSpec, Constraints, LoadRow, Phase, PolicySpec, RequestFacts, Role, Scratch, WorkerId,
};

const MODEL: &str = "m";
const GATEWAY: GatewayId = GatewayId(1);
const BLOCK: usize = 16;
/// Blocks of a request's prompt.
const PROMPT_BLOCKS: usize = 4;
const SIZES: [usize; 7] = [8, 16, 32, 64, 128, 512, 1_024];
/// The prompt every worker holds.
const SHARED: usize = 1_000_000;

/// Prompt `p`: tokens of its own.
fn prompt(p: usize) -> Vec<u32> {
    (0..BLOCK * PROMPT_BLOCKS)
        .map(|t| (p * 1_000 + t + 1) as u32)
        .collect()
}

/// One batch: `id` stores the blocks of `tokens`.
fn stores(store: &MemoryStore, id: WorkerId, sequence: u64, tokens: &[u32]) {
    let contents: Vec<ContentHash> = tokens.chunks(BLOCK).map(compute_content_hash).collect();
    let blocks = contents
        .iter()
        .zip(request_prefix_hashes(&contents))
        .map(|(&content_hash, seq_hash)| StoredBlock {
            seq_hash,
            content_hash,
        })
        .collect();
    let stored = [KvEvent::Stored {
        parent: None,
        blocks,
    }];
    let facts = BatchFacts {
        sequence,
        ..BatchFacts::default()
    };
    assert_eq!(store.admit(id, 0, facts, &stored), BatchOutcome::Applied);
}

/// A store under the `cache_aware` preset with `n` ready workers that
/// publish events, worker `p` holding prompt `p` and the shared prompt;
/// their ids in order.
fn fleet(n: usize) -> (MemoryStore, Vec<WorkerId>) {
    fleet_picking(n, None)
}

/// [`fleet`] under the preset's stages and the picker `pick`, where one is
/// named: `best` takes no sample.
fn fleet_picking(n: usize, pick: Option<&str>) -> (MemoryStore, Vec<WorkerId>) {
    let store = MemoryStore::new();
    let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
    document.generation = 1;
    if let Some(pick) = pick {
        for profile in document.profiles.values_mut() {
            profile.pick = pick.into();
        }
    }
    let spec = AppliedSpec::validate(&document).expect("a preset validates");
    store.apply(spec).expect("above the unbound start");
    let ids: Vec<WorkerId> = (0..n)
        .map(|p| {
            let key = format!("grpc://regular-{p:04}:9000");
            let id = store
                .upsert(WorkerFacts {
                    engine: key.clone(),
                    key,
                    incarnation: 1,
                    models: vec![MODEL.to_string()],
                    dp_size: 1,
                    signals: Signals {
                        events: true,
                        ..Signals::default()
                    },
                    ..WorkerFacts::default()
                })
                .id();
            let revision = WorkersStore::get(&store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("the status");
            id
        })
        .collect();
    // The build the registrations owe, asked once: a decision reads the
    // fleet whole.
    store.publish_now();
    for (p, id) in ids.iter().enumerate() {
        stores(&store, *id, 1, &prompt(p));
        stores(&store, *id, 2, &prompt(SHARED));
    }
    (store, ids)
}

/// The index of the one group a worker's view holds beside main attention.
const GROUP: u32 = 1;

/// [`fleet`] each of whose workers holds a view of its cache groups: one
/// state group that retains the last block of the worker's own prompt and
/// of the shared one, written through the worker's lent lane. Its groups
/// serve every block a worker holds.
fn fleet_viewed(n: usize) -> (MemoryStore, Vec<WorkerId>) {
    let (store, ids) = fleet(n);
    for (p, id) in ids.iter().enumerate() {
        let mut view = None;
        let facts = BatchFacts {
            sequence: 3,
            ..BatchFacts::default()
        };
        let outcome = store.admit_through(*id, 0, facts, &mut |lane| {
            view = Some(lane.groups());
        });
        assert_eq!(outcome, BatchOutcome::Applied);
        let view = view.expect("the body ran");
        view.learn(GROUP, Some(GroupKind::State), 0);
        for tokens in [prompt(p), prompt(SHARED)] {
            let contents: Vec<ContentHash> =
                tokens.chunks(BLOCK).map(compute_content_hash).collect();
            let main: Vec<_> = request_prefix_hashes(&contents)
                .into_iter()
                .zip(contents)
                .collect();
            view.main_stored(None, &main);
            let last = main.last().expect("a prompt has blocks").0;
            view.group_stored(GROUP, &[last]);
        }
    }
    (store, ids)
}

/// What a deciding thread keeps across decisions.
struct Host {
    hashes: Vec<u64>,
    view: ViewBuffers,
    named: Named,
    scratch: Scratch,
}

/// One decision for a request of `tokens` over lent rows: the worker it
/// picks. With `named` the decision is handed the rows the store names;
/// without, the column, as under a profile that is named no row.
/// Out of line, so a count reads it by name.
#[inline(never)]
fn decide(
    store: &MemoryStore,
    host: &mut Host,
    tokens: &[u32],
    now: Instant,
    seed: u64,
    named: bool,
) -> Option<WorkerId> {
    host.hashes.clear();
    host.hashes.extend(
        tokens
            .as_chunks::<BLOCK>()
            .0
            .iter()
            .map(|block| compute_content_hash(block).0),
    );
    store.deciding(|world| {
        let engine = world.spec().engine();
        let ask = Ask {
            gateway: GATEWAY,
            now,
            pinned: None,
            affinity: !named,
            signal_cells: false,
        };
        // The rows where their publication left them: lent, the door of a
        // pool every gateway reaches whole.
        let read = world.view(MODEL, Role::Single, &ask, false, &mut host.view)?;
        let built = read.built;
        if named {
            let naming = Naming {
                gateway: GATEWAY,
                block_hashes: &host.hashes,
                namespace_seed: 0,
                seed,
                cap: engine.named_for(MODEL, Phase::Single),
                wires: None,
                now_micros: built.now_micros,
            };
            world.name(
                MODEL,
                Role::Single,
                &naming,
                read.candidates,
                &mut host.named,
            );
        }
        let facts = RequestFacts {
            model: MODEL,
            prompt_tokens: tokens.len() as u32,
            block_hashes: &host.hashes,
            block_size: BLOCK as u32,
            seed,
            now_micros: built.now_micros,
            affinity: read.affinity,
            named: named.then(|| host.named.affinity()),
            ..RequestFacts::default()
        };
        let constraints = Constraints {
            stalled: read.stalled,
            ..Constraints::default()
        };
        let decision = engine.decide(
            &facts,
            &built.pool_facts(),
            read.candidates,
            &constraints,
            &mut host.scratch,
        );
        decision
            .pick()
            .and_then(|pick| pick.targets.first())
            .map(|target| target.id)
    })
}

fn decisions(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_named");
    let now = Instant::now();
    for &n in &SIZES {
        let (store, ids) = fleet(n);
        let held: Vec<Vec<u32>> = (0..n).map(prompt).collect();
        let unheld: Vec<Vec<u32>> = (0..n).map(|p| prompt(n + p)).collect();
        let shared = prompt(SHARED);
        let mut host = Host {
            hashes: Vec::new(),
            view: ViewBuffers::default(),
            named: Named::default(),
            scratch: Scratch::new(),
        };
        // A hit is its one holder's, whatever the seed.
        for (p, id) in ids.iter().enumerate() {
            let picked = decide(&store, &mut host, &held[p], now, p as u64, true);
            assert_eq!(picked, Some(*id), "{n}: prompt {p} goes to its holder");
        }
        let cells: [(&str, &[Vec<u32>], bool); 3] = [
            ("hit", &held, true),
            ("miss", &unheld, true),
            ("column", &unheld, false),
        ];
        for (name, prompts, named) in cells {
            let mut next = 0usize;
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                b.iter(|| {
                    let tokens = &prompts[next % n];
                    next += 1;
                    decide(
                        &store,
                        &mut host,
                        black_box(tokens),
                        now,
                        next as u64,
                        named,
                    )
                });
            });
        }
        let mut seed = 0u64;
        group.bench_with_input(BenchmarkId::new("shared", n), &n, |b, _| {
            b.iter(|| {
                seed += 1;
                decide(&store, &mut host, black_box(&shared), now, seed, true)
            });
        });
        // A fleet of the same making with a view of its cache groups a
        // worker: every holder is read at what its groups serve, and a hit
        // is still its one holder's.
        let (viewed, views) = fleet_viewed(n);
        for (p, id) in views.iter().enumerate() {
            let picked = decide(&viewed, &mut host, &held[p], now, p as u64, true);
            assert_eq!(picked, Some(*id), "{n}: prompt {p} goes to its holder");
        }
        let mut next = 0usize;
        group.bench_with_input(BenchmarkId::new("viewed_hit", n), &n, |b, _| {
            b.iter(|| {
                let tokens = &held[next % n];
                next += 1;
                decide(
                    &viewed,
                    &mut host,
                    black_box(tokens),
                    now,
                    next as u64,
                    true,
                )
            });
        });
        let mut seed = 0u64;
        group.bench_with_input(BenchmarkId::new("viewed_shared", n), &n, |b, _| {
            b.iter(|| {
                seed += 1;
                decide(&viewed, &mut host, black_box(&shared), now, seed, true)
            });
        });
        // The same fleet under a picker that takes no sample: named every
        // holder, and a hit is still its one holder's.
        let (whole, wholes) = fleet_picking(n, Some("best"));
        for (p, id) in wholes.iter().enumerate() {
            let picked = decide(&whole, &mut host, &held[p], now, p as u64, true);
            assert_eq!(picked, Some(*id), "{n}: prompt {p} goes to its holder");
        }
        let cells: [(&str, &[Vec<u32>], bool); 3] = [
            ("best_hit", &held, true),
            ("best_miss", &unheld, true),
            ("best_column", &unheld, false),
        ];
        for (name, prompts, named) in cells {
            let mut next = 0usize;
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                b.iter(|| {
                    let tokens = &prompts[next % n];
                    next += 1;
                    decide(
                        &whole,
                        &mut host,
                        black_box(tokens),
                        now,
                        next as u64,
                        named,
                    )
                });
            });
        }
        let mut seed = 0u64;
        group.bench_with_input(BenchmarkId::new("best_shared", n), &n, |b, _| {
            b.iter(|| {
                seed += 1;
                decide(&whole, &mut host, black_box(&shared), now, seed, true)
            });
        });
        // One window's publication: a record folded inside the window, then
        // the publication, which rules the pool's warming rows again. The
        // store's first record opens a window, and a record inside it waits
        // for the window's end.
        let (mut window, mut turn) = (Instant::now(), 0u32);
        store.record(ids[0], 0, LoadRow::default(), window, LoadSource::Stream);
        let waits = LoadRow {
            running: 1,
            ..LoadRow::default()
        };
        let inside = window + Duration::from_millis(1);
        store.record(ids[0], 0, waits, inside, LoadSource::Stream);
        assert!(store.publish_due().is_some(), "{n}: the record waits");
        window += PUBLISH_WINDOW;
        store.publish(window);
        assert_eq!(store.publish_due(), None, "{n}: its window is published");
        group.bench_with_input(BenchmarkId::new("publish", n), &n, |b, _| {
            b.iter_batched(
                || {
                    turn += 1;
                    let row = LoadRow {
                        running: turn % 12,
                        live_rate: turn,
                        ..LoadRow::default()
                    };
                    let at = window + Duration::from_millis(1);
                    store.record(ids[turn as usize * 67 % n], 0, row, at, LoadSource::Stream);
                    window += PUBLISH_WINDOW;
                    window
                },
                |due| store.publish(black_box(due)),
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, decisions);
criterion_main!(benches);
