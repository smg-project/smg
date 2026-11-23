//! The cases of the engine's start: each has its engine by
//! [`PolicyEngine::start`] alone, over the worker crate's own registry, and
//! spawns the task the call answered, on a paused clock. The registry, the
//! load monitor's stand-in and the requests are the neighbouring cases' own:
//! the follow's and the task's.

use std::{sync::Arc, thread, time::Instant};

use kv_index::{request_prefix_hashes, ChainBlockMap, ChainIndex, ContentHash, StoredBlock};
use smg_worker::identity::basic::Worker;
use tokio::{runtime::Builder, task::yield_now};

use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision_engine::DecisionEngine,
        list_cell::{tests::PublishedList, ModelLists},
        policy_engine::{
            membership::tests::{
                fills::{answered, Program},
                handed, id, in_id_order, register, registry, remove, told, Registry,
            },
            overlap::{CacheIndex, CacheIndexes},
            single::tests::hearing_spec,
            task::tests::{asks, monitor, picked, reports, settle, spawned, url, MODEL},
            PolicyEngine,
        },
        prompt_hashes::HashSeeds,
        view::RequestFacts,
        WorkerData,
    },
    store::GatewayId,
    tracking::{
        lease::{Outcome, Side, TransportClass},
        DecisionGuard,
    },
};

/// Tokens a block in the program's index of cache events: twice the size a
/// stock document states.
const INDEX_BLOCK: u32 = 32;

/// The model's worker list as `registry` publishes it now.
fn list(registry: &Registry) -> Arc<[Arc<dyn Worker>]> {
    registry.get_by_model(MODEL)
}

/// The place in `list` of the worker at `url(at)`.
fn place_of(list: &[Arc<dyn Worker>], at: usize) -> u32 {
    let address = url(at);
    let place = list.iter().position(|worker| worker.url() == address);
    place.expect("the worker is in the list") as u32
}

/// The document of a built-in preset.
fn preset(name: &str) -> AppliedSpec {
    let document = PolicySpec::preset(name).expect("a built-in preset");
    AppliedSpec::validate(&document).expect("the document validates")
}

/// A program's indexes of cache events: one model's chain index, of blocks
/// of a size of its own.
struct Indexes(Arc<ChainIndex>);

impl CacheIndexes for Indexes {
    fn index(&self, model: &str) -> Option<Arc<dyn CacheIndex>> {
        (model == MODEL).then(|| Arc::clone(&self.0) as Arc<dyn CacheIndex>)
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        (model == MODEL).then_some(INDEX_BLOCK)
    }
}

/// `index` as a program hands its indexes of cache events to the engine.
fn indexes(index: &Arc<ChainIndex>) -> Arc<dyn CacheIndexes> {
    Arc::new(Indexes(Arc::clone(index)))
}

/// Stores `hashes`, a prompt's blocks from its first, for the worker at
/// `url(at)`, as the worker's stream of cache events would.
fn store(index: &ChainIndex, at: usize, hashes: &[u64]) {
    let number = index.intern_worker(&url(at)).expect("a number");
    let contents: Vec<ContentHash> = hashes.iter().map(|&hash| ContentHash(hash)).collect();
    let blocks: Vec<StoredBlock> = contents
        .iter()
        .zip(request_prefix_hashes(&contents))
        .map(|(&content_hash, seq_hash)| StoredBlock {
            seq_hash,
            content_hash,
        })
        .collect();
    let stored = index.apply_stored(number, &blocks, None, &mut ChainBlockMap::new());
    stored.expect("the blocks are stored");
}

/// How many of 64 requests of as many seeds each worker of `members` is
/// named for. No request gives a load snapshot of its own.
fn spread(engine: &PolicyEngine, members: &PublishedList) -> Vec<u32> {
    let mut named = vec![0_u32; members.workers().len()];
    for seed in 0..64 {
        named[picked(engine, &asks(seed, members, None)) as usize] += 1;
    }
    named
}

/// The load channel a program hands at the start is the one the engine's
/// task reads, and the start spawns nothing: with the runtime turned and no
/// task spawned, the engine holds no snapshot and reads no report. Behind
/// the task's first step a request that gives no snapshot goes to the
/// worker the channel reports the shortest queue for, and to the next one
/// once the task has kept the monitor's next publication. The indexes of
/// cache events handed beside the channel, as a gateway hands both, stand
/// with it.
#[tokio::test(start_paused = true)]
async fn a_started_engine_reads_the_load_channel_it_was_handed() {
    let registry = registry(4);
    let (feed, channel) = monitor(&reports(4, 2));
    let data = WorkerData {
        registry: handed(&registry),
        loads: Some(channel),
        cache_indexes: Some(indexes(&Arc::new(ChainIndex::new()))),
    };
    let (engine, task) = PolicyEngine::start(preset("least_load"), data);
    settle().await;
    let list = list(&registry);
    let members = PublishedList::new(Arc::clone(&list));
    let unread = spread(engine, &members);
    assert!(unread.iter().all(|&picks| picks > 0), "{unread:?}");

    let task = spawned(task).await;
    for seed in 0..32 {
        let place = picked(engine, &asks(seed, &members, None));
        assert_eq!(place, place_of(&list, 2), "seed {seed}: the queue handed");
    }
    feed.publish(&reports(4, 0));
    settle().await;
    for seed in 0..32 {
        let place = picked(engine, &asks(seed, &members, None));
        assert_eq!(place, place_of(&list, 0), "seed {seed}: the next one");
    }
    assert_eq!(engine.affinity().block_size(MODEL), Some(INDEX_BLOCK));
    task.abort();
}

/// The registry a program hands at the start is the one the engine's task
/// follows, with no call of the program's: the workers that stand are told
/// to the document's stages as joined at the task's first step, a worker
/// registered later as joined and a worker removed as left. The start
/// spawns nothing: with the runtime turned and no task spawned, nothing is
/// told.
#[tokio::test(start_paused = true)]
async fn a_started_engine_follows_the_registry_it_was_handed() {
    let registry = registry(2);
    let (spec, heard) = hearing_spec();
    let data = WorkerData {
        registry: handed(&registry),
        loads: None,
        cache_indexes: None,
    };
    let (_engine, task) = PolicyEngine::start(spec, data);
    settle().await;
    assert_eq!(told(&heard), [], "nothing is told with no task spawned");

    let task = spawned(task).await;
    let stand = vec![("joined", id(0)), ("joined", id(1))];
    assert_eq!(in_id_order(told(&heard)), in_id_order(stand));
    register(&registry, 2);
    remove(&registry, 0);
    settle().await;
    assert_eq!(told(&heard), [("joined", id(2)), ("left", id(0))]);
    task.abort();
}

/// The lists a program hands beside the start
/// ([`PolicyEngine::set_lists`]) are the ones the engine's task fills, with
/// no further call of the program's. The start spawns nothing: with the
/// runtime turned and no task spawned, the program is asked for no model
/// and no value is made. At its first step the task fills the cell of the
/// list that stands, and behind a registration the cell of the model's
/// newer list, each ahead of a request: a request over a list filled so
/// makes no value.
#[tokio::test(start_paused = true)]
async fn a_started_engine_fills_the_cells_of_the_lists_it_was_handed() {
    let program = Program::of(2);
    let data = WorkerData {
        registry: handed(&program.registry),
        loads: None,
        cache_indexes: None,
    };
    let (engine, task) = PolicyEngine::start(preset("least_load"), data);
    engine.set_lists(Arc::clone(&program) as Arc<dyn ModelLists>);
    settle().await;
    assert!(program.asked().is_empty(), "nothing is asked with no task");
    assert_eq!(engine.values_made(), 0);

    let task = spawned(task).await;
    assert_eq!(program.asked(), [MODEL]);
    assert_eq!(engine.values_made(), 1, "the first step filled the cell");
    let standing = program.published();
    assert_eq!(answered(engine, &program, &standing), [true; 2]);
    assert_eq!(engine.values_made(), 1, "the requests read the cell");

    program.register(2);
    settle().await;
    assert_eq!(program.asked(), [MODEL]);
    assert_eq!(engine.values_made(), 2, "the task filled the new cell");
    let newer = program.published();
    assert_eq!(answered(engine, &program, &newer), [true; 2]);
    assert_eq!(engine.values_made(), 2, "the requests read the cell");
    assert_eq!(engine.upkeep_faults(), 0);
    task.abort();
}

/// The indexes of cache events a program hands at the start are the ones a
/// decision reads. A prompt is hashed by the block size the index states,
/// not the document's, and a request of that prompt goes to the worker the
/// index holds its blocks for, whatever its seed.
#[tokio::test(start_paused = true)]
async fn a_started_engine_reads_the_cache_indexes_it_was_handed() {
    let registry = registry(8);
    let index = Arc::new(ChainIndex::new());
    let data = WorkerData {
        registry: handed(&registry),
        loads: None,
        cache_indexes: Some(indexes(&index)),
    };
    let (engine, task) = PolicyEngine::start(preset("cache_aware"), data);
    let task = spawned(task).await;

    let tokens: Vec<u32> = (0..1_024).collect();
    let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
    let mut stated = RequestFacts::default();
    hashes.write_to(&mut stated);
    let blocks = stated.block_hashes;
    assert_eq!(blocks.len(), 1_024 / INDEX_BLOCK as usize);
    store(&index, 5, blocks);

    let list = list(&registry);
    let members = PublishedList::new(Arc::clone(&list));
    for seed in 0..32 {
        let mut asked = asks(seed, &members, None);
        hashes.write_to(&mut asked.request.facts);
        let place = picked(engine, &asked);
        assert_eq!(place, place_of(&list, 5), "seed {seed}: the holder");
    }
    task.abort();
}

/// A program with no load monitor and no index of cache events says so at
/// the start, and its engine decides without them: behind the task's first
/// step no decision reads a load report, and the engine holds no index's
/// block size.
#[tokio::test(start_paused = true)]
async fn an_engine_started_with_no_channel_and_no_index_decides_without_them() {
    let registry = registry(4);
    let data = WorkerData {
        registry: handed(&registry),
        loads: None,
        cache_indexes: None,
    };
    let (engine, task) = PolicyEngine::start(preset("least_load"), data);
    let task = spawned(task).await;
    let list = list(&registry);
    let members = PublishedList::new(Arc::clone(&list));
    let unread = spread(engine, &members);
    assert!(unread.iter().all(|&picks| picks > 0), "{unread:?}");
    assert_eq!(engine.affinity().block_size(MODEL), None);
    task.abort();
}

/// The engine a program starts stands for the life of the process: a
/// request's guard holds it on another thread, behind the frame that
/// started it, and the document's stages hear the request's words there.
#[tokio::test(start_paused = true)]
async fn a_started_engine_outlives_the_frame_that_started_it() {
    let registry = registry(2);
    let (engine, heard, task) = {
        let (spec, heard) = hearing_spec();
        let data = WorkerData {
            registry: handed(&registry),
            loads: None,
            cache_indexes: None,
        };
        let (engine, task) = PolicyEngine::start(spec, data);
        (engine, heard, task)
    };
    let task = spawned(task).await;
    let _joined = told(&heard);

    let list = list(&registry);
    let request = thread::spawn(move || {
        let members = PublishedList::new(Arc::clone(&list));
        let asks = asks(7, &members, None);
        let now = Instant::now();
        let decided = DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now);
        let mut guard = decided.expect("a decision");
        let target = guard.take(0).expect("the first target");
        guard.dispatched(now);
        let outcome = Outcome {
            side: Side::Single,
            served: target.id,
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        };
        guard.completed(&outcome, now);
        target.id
    });
    let served = request.join().expect("the request's thread ends");
    let words = [
        ("issued", served),
        ("dispatched", served),
        ("completed", served),
    ];
    assert_eq!(told(&heard), words);
    task.abort();
}

/// The task needs its runtime's timer. Spawned on a runtime built with
/// none, it ends at its first step with a panic, which its handle holds:
/// one turn of the runtime, and the task has ended. The engine goes on
/// deciding with none of the task's work.
#[test]
fn a_task_spawned_on_a_runtime_with_no_timer_ends_with_a_panic() {
    let registry = registry(4);
    let data = WorkerData {
        registry: handed(&registry),
        loads: None,
        cache_indexes: None,
    };
    let (engine, task) = PolicyEngine::start(preset("least_load"), data);
    let untimed = Builder::new_current_thread().build();
    let untimed = untimed.expect("a runtime");
    let task = untimed.spawn(task);
    untimed.block_on(yield_now());
    assert!(task.is_finished(), "the task ended at its first step");
    let ended = untimed.block_on(task);
    assert!(ended.expect_err("the task ended").is_panic());

    let list = list(&registry);
    let members = PublishedList::new(Arc::clone(&list));
    let named = spread(engine, &members);
    assert!(named.iter().all(|&picks| picks > 0), "{named:?}");
}
