//! Test support of the policy engine: the store a start makes with one ready
//! row a URL, a registry with one worker a URL, and the policy engine over
//! both. A router's cases build their rows this way, so one fixture stands.

use std::{sync::Arc, time::Instant};

use kv_index::{chain_prefix_hash, ContentHash, SequenceHash, StoredBlock};
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus, UNKNOWN_MODEL_ID};
use smg_policy_controller::{
    document::{AppliedSpec, DecodePick, PolicySpec},
    engine::{LoadRow, Role, WireModes, WorkerId},
    store::{
        Ask as StoreAsk, BatchFacts, BatchOutcome, KvEvent, KvEventSink, LoadSink, LoadSource,
        MemoryStore, PairBuffers, Signals, WorkerFacts, WorkerStatus as RowStatus, WorkersStore,
    },
    tracking::LeaseId,
};
use smg_worker::identity::basic::{Worker, WorkerType};

use super::{Placed, PolicyEngine, GATEWAY};
use crate::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    policy_controller::start,
    routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder},
};

/// The model every row and worker of [`bed`] serves.
pub(crate) const MODEL: &str = "model-a";

/// A policy engine, its store, and the registry's workers in the URLs' order.
type Bed = (PolicyEngine, Arc<MemoryStore>, Vec<Arc<dyn Worker>>);

/// The id of the lease `placed` holds: a router's case reads the lease's
/// sides in the store by it.
pub(crate) fn lease_of(placed: &Placed) -> LeaseId {
    placed.handle.lease().expect("a lease").lease.id
}

/// The store a start makes under the controller with one ready row a URL,
/// a registry with one worker a URL, and the policy engine over the two.
pub(crate) fn bed(urls: &[&str]) -> Bed {
    bed_under(RouterConfig::default().policy, urls)
}

/// [`bed`] where the gateway routes by `policy`.
pub(crate) fn bed_under(policy: PolicyConfig, urls: &[&str]) -> Bed {
    let rows = urls.iter().map(|&url| (url, Some(MODEL)));
    started(policy, Arc::new(WorkerRegistry::new()), rows, None)
}

/// The same store and policy engine over workers registered in `registry`.
pub(crate) fn bed_over(registry: Arc<WorkerRegistry>, urls: &[&str]) -> Bed {
    bed_over_models(registry, urls.iter().map(|&url| (url, Some(MODEL))), None)
}

/// One worker and one ready row per URL, with each worker's model or the
/// no-model id in its row. The worker at `ranked_url` has rank zero.
pub(crate) fn bed_over_models<'a>(
    registry: Arc<WorkerRegistry>,
    urls_and_models: impl IntoIterator<Item = (&'a str, Option<&'a str>)>,
    ranked_url: Option<&str>,
) -> Bed {
    started(
        RouterConfig::default().policy,
        registry,
        urls_and_models,
        ranked_url,
    )
}

/// The store a start makes for `policy`, with its rows and its workers in
/// `registry`. The worker at `ranked_url` has rank zero.
fn started<'a>(
    policy: PolicyConfig,
    registry: Arc<WorkerRegistry>,
    urls_and_models: impl IntoIterator<Item = (&'a str, Option<&'a str>)>,
    ranked_url: Option<&str>,
) -> Bed {
    let config = RouterConfig {
        policy_controller: true,
        policy,
        ..RouterConfig::default()
    };
    let store = start(&config)
        .expect("the start takes the routing")
        .expect("the controller was picked");
    bed_with(store, registry, urls_and_models, ranked_url)
}

/// The URLs of a preset bed's three rows.
const URLS: [&str; 3] = ["http://w0:8000", "http://w1:8000", "http://w2:8000"];

/// One ready row a URL under the cache-aware preset, over workers
/// registered in `registry`, each publishing cache events or none.
fn preset_bed(registry: Arc<WorkerRegistry>, urls: &[&str], events: bool) -> Bed {
    let store = Arc::new(MemoryStore::new());
    let document = PolicySpec::preset("cache_aware").expect("the preset");
    let applied = AppliedSpec::validate(&document).expect("the preset validates");
    store.apply(applied).expect("the store takes the preset");
    let rows = urls.iter().map(|&url| (url, Some(MODEL)));
    bed_publishing(store, registry, rows, None, events)
}

/// [`preset_bed`] of three rows that publish cache events, with the second
/// row's id.
pub(crate) fn cache_bed() -> (PolicyEngine, Arc<MemoryStore>, WorkerId) {
    let (policy_engine, store, _) = preset_bed(Arc::new(WorkerRegistry::new()), &URLS, true);
    let holder = store.by_key(URLS[1]).expect("the second worker's row").id;
    (policy_engine, store, holder)
}

/// [`preset_bed`] of three rows that publish none: the preset's prefix
/// stage reads the record of sent prefixes for them.
pub(crate) fn record_bed() -> (PolicyEngine, Arc<MemoryStore>) {
    record_bed_over(Arc::new(WorkerRegistry::new()), &URLS)
}

/// [`record_bed`] with one row a URL, over workers registered in `registry`.
pub(crate) fn record_bed_over(
    registry: Arc<WorkerRegistry>,
    urls: &[&str],
) -> (PolicyEngine, Arc<MemoryStore>) {
    let (policy_engine, store, _) = preset_bed(registry, urls, false);
    (policy_engine, store)
}

/// [`bed`] over `store` and `registry`: a row publishes cache events where
/// the store's generation reads them.
pub(crate) fn bed_with<'a>(
    store: Arc<MemoryStore>,
    registry: Arc<WorkerRegistry>,
    urls_and_models: impl IntoIterator<Item = (&'a str, Option<&'a str>)>,
    ranked_url: Option<&str>,
) -> Bed {
    let events = store.deciding(|world| world.spec().needs().events);
    bed_publishing(store, registry, urls_and_models, ranked_url, events)
}

/// [`bed_with`], each row publishing cache events or none.
fn bed_publishing<'a>(
    store: Arc<MemoryStore>,
    registry: Arc<WorkerRegistry>,
    urls_and_models: impl IntoIterator<Item = (&'a str, Option<&'a str>)>,
    ranked_url: Option<&str>,
    events: bool,
) -> Bed {
    let mut workers = Vec::new();
    for (url, model) in urls_and_models {
        let builder = BasicWorkerBuilder::new(url).status(WorkerStatus::Ready);
        let builder = match model {
            Some(model) => builder.model(ModelCard::new(model)),
            None => builder,
        };
        let ranked = ranked_url == Some(url);
        let dp_size: u16 = if ranked { 2 } else { 1 };
        let builder = if ranked {
            builder.dp_config(0, usize::from(dp_size))
        } else {
            builder
        };
        let worker: Arc<dyn Worker> = Arc::new(builder.build());
        let facts = WorkerFacts {
            key: worker.url().to_string(),
            engine: url.to_string(),
            incarnation: 1,
            role: Role::Single,
            models: vec![model.unwrap_or(UNKNOWN_MODEL_ID).to_string()],
            dp_size,
            max_running: Some(64),
            wire_modes: wire_of(url),
            signals: Signals {
                events,
                ..Signals::default()
            },
            ..WorkerFacts::default()
        };
        let id = store.upsert(facts).id();
        store.publish_now();
        let row = WorkersStore::get(&*store, id).expect("the row");
        if row.status != RowStatus::Ready {
            store
                .set_status(id, RowStatus::Ready, row.revision)
                .expect("the row stands");
        }
        registry
            .register(Arc::clone(&worker))
            .expect("the worker registers");
        workers.push(worker);
    }
    store.publish_now();
    (
        PolicyEngine::at_keys(Arc::clone(&store), registry),
        store,
        workers,
    )
}

/// A store under the stock pair presets with one ready row a URL, the
/// prefills' then the decodes', a registry with one worker a row, and the
/// policy engine over the two. The workers come back in the rows' order.
pub(crate) fn pair_bed(prefills: &[&str], decodes: &[&str]) -> Bed {
    let store = Arc::new(MemoryStore::new());
    let document = PolicySpec::pair_presets("least_load", "least_load").expect("the pair presets");
    let applied = AppliedSpec::validate(&document).expect("the presets validate");
    store.apply(applied).expect("the store takes the presets");
    pair_rows(store, MODEL, prefills, decodes)
}

/// [`pair_bed`] under a document that says when a pair's decode is picked
/// (`plans.pd-pair.decode`).
pub(crate) fn pair_bed_picking(decode: DecodePick, prefills: &[&str], decodes: &[&str]) -> Bed {
    pair_bed_shaped(decode, |_, worker| worker, prefills, decodes)
}

/// [`pair_bed_picking`] with each registry worker as `shaped` leaves it, by
/// its URL. The rows are written the same for every shape.
pub(crate) fn pair_bed_shaped(
    decode: DecodePick,
    shaped: impl Fn(&str, BasicWorkerBuilder) -> BasicWorkerBuilder,
    prefills: &[&str],
    decodes: &[&str],
) -> Bed {
    let store = Arc::new(MemoryStore::new());
    let mut document =
        PolicySpec::pair_presets("least_load", "least_load").expect("the pair presets");
    document.plans.pd_pair.decode = decode;
    let applied = AppliedSpec::validate(&document).expect("the presets validate");
    store.apply(applied).expect("the store takes the presets");
    pair_rows_shaped(store, MODEL, shaped, prefills, decodes)
}

/// The engines report the worker at `busy` deep in its queue and the one at
/// `idle` empty, and the store publishes the two loads. The publication's
/// instant comes back: a decision at it reads them.
pub(crate) fn report_loads(store: &MemoryStore, busy: &str, idle: &str) -> Instant {
    let at = Instant::now();
    for (url, queued) in [(busy, 80), (idle, 0)] {
        let row = store.by_key(url).expect("the worker's row");
        let load = LoadRow {
            queued,
            live_rate: 2_000,
            ..LoadRow::default()
        };
        store.record(row.id, 0, load, at, LoadSource::Stream);
    }
    // A publication shows what waits once it is due.
    let published = store.publish_due().map_or(at, |due| due.max(at));
    store.publish(published);
    published
}

/// The requests the store counts in flight on `row`, a row of [`MODEL`]'s
/// pair pools.
pub(crate) fn pair_in_flight(store: &MemoryStore, row: WorkerId) -> u32 {
    store.with_snapshot(|world| {
        let ask = StoreAsk {
            gateway: GATEWAY,
            now: Instant::now(),
            pinned: None,
            affinity: false,
            signal_cells: false,
        };
        let mut rows = PairBuffers::default();
        let built = world
            .build_pair(MODEL, &ask, &mut rows)
            .expect("a pair's pools");
        world.count_pair_in_flight(&built, &ask, &mut rows);
        let pools = rows.pools();
        let row = pools
            .prefills
            .iter()
            .chain(pools.decodes)
            .find(|listed| listed.id == row)
            .expect("the row is listed");
        row.in_flight
    })
}

/// The store a pair gateway starts, with one ready row a URL as
/// [`pair_bed`] writes them, each serving `model`. The document is the one
/// the gateway's configuration makes, with every filter it names, and each
/// row is as the store builds it: no case sets a filter or a row's bit.
pub(crate) fn pair_bed_started(model: &str, prefills: &[&str], decodes: &[&str]) -> Bed {
    let config = RouterConfig {
        policy_controller: true,
        mode: RoutingMode::PrefillDecode {
            prefill_urls: Vec::new(),
            decode_urls: Vec::new(),
            prefill_policy: None,
            decode_policy: None,
        },
        ..RouterConfig::default()
    };
    let store = start(&config)
        .expect("the start takes the pair routing")
        .expect("the controller was picked");
    pair_rows(store, model, prefills, decodes)
}

/// The wire a bed's row at `url` states, as a store's writer states it for
/// a worker: gRPC under its scheme, else HTTP. A case that types its worker
/// otherwise writes the row's wire itself.
fn wire_of(url: &str) -> WireModes {
    if url.starts_with("grpc://") {
        WireModes::GRPC
    } else {
        WireModes::HTTP
    }
}

/// One ready row a URL in `store`, the prefills' then the decodes', each
/// serving `model`, a registry with one worker a row, and the policy engine
/// over the two.
fn pair_rows(store: Arc<MemoryStore>, model: &str, prefills: &[&str], decodes: &[&str]) -> Bed {
    pair_rows_shaped(store, model, |_, worker| worker, prefills, decodes)
}

/// [`pair_rows`] with each registry worker as `shaped` leaves it, by its
/// URL.
fn pair_rows_shaped(
    store: Arc<MemoryStore>,
    model: &str,
    shaped: impl Fn(&str, BasicWorkerBuilder) -> BasicWorkerBuilder,
    prefills: &[&str],
    decodes: &[&str],
) -> Bed {
    let registry = Arc::new(WorkerRegistry::new());
    let mut workers = Vec::new();
    let sides = [
        (prefills, Role::Prefill, WorkerType::Prefill),
        (decodes, Role::Decode, WorkerType::Decode),
    ];
    for (urls, role, worker_type) in sides {
        for &url in urls {
            let facts = WorkerFacts {
                key: url.to_string(),
                engine: url.to_string(),
                incarnation: 1,
                role,
                models: vec![model.to_string()],
                dp_size: 1,
                max_running: Some(64),
                wire_modes: wire_of(url),
                ..WorkerFacts::default()
            };
            let id = store.upsert(facts).id();
            store.publish_now();
            let row = WorkersStore::get(&*store, id).expect("the row");
            if row.status != RowStatus::Ready {
                store
                    .set_status(id, RowStatus::Ready, row.revision)
                    .expect("the row stands");
            }
            let builder = BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model))
                .worker_type(worker_type)
                .status(WorkerStatus::Ready);
            let worker: Arc<dyn Worker> = Arc::new(shaped(url, builder).build());
            registry
                .register(Arc::clone(&worker))
                .expect("the worker registers");
            workers.push(worker);
        }
    }
    store.publish_now();
    (
        PolicyEngine::at_keys(Arc::clone(&store), registry),
        store,
        workers,
    )
}

/// Stores a chain of content hashes in one batch on `id`, through the
/// store's own sink.
pub(crate) fn store_blocks(
    store: &MemoryStore,
    id: WorkerId,
    contents: &[ContentHash],
) -> BatchOutcome {
    let mut blocks = Vec::with_capacity(contents.len());
    let mut parent = None;
    for &content_hash in contents {
        let seq_hash = parent.map_or(SequenceHash(content_hash.0), |previous| {
            chain_prefix_hash(previous, content_hash)
        });
        blocks.push(StoredBlock {
            seq_hash,
            content_hash,
        });
        parent = Some(seq_hash);
    }
    let facts = BatchFacts {
        sequence: 1,
        ..BatchFacts::default()
    };
    store.admit(
        id,
        0,
        facts,
        &[KvEvent::Stored {
            parent: None,
            blocks,
        }],
    )
}
