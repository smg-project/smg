//! What the gateway's start hands the policy engine it starts: the worker
//! registry, whose events the engine's task follows, and the monitor of the
//! workers' cache events. And what a start under a supplied document builds
//! its worker objects with: the overload thresholds of its own settings,
//! under the switch that document's bindings say, with the routes' shed word
//! theirs too.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use smg_grpc_client::common_proto::{
    kv_cache_event::Data, KvBlock, KvBlocksStored, KvCacheEvent, KvEventBatch,
};
use smg_policy_controller::{
    document::{OverloadSpec, PolicySpec},
    engine::{Candidate, RequestFacts, Scorer, TieBand, WorkerId},
    stages::{Lifecycle, StageRegistry},
    tracking::{EventKind, EventSet, Membership},
};
use smg_worker::{
    health::overload::OverloadThresholds,
    identity::basic::{ConnectionMode, WorkerType},
};

use super::super::{build_app_context, Extensions, ServerConfig};
use crate::{
    app_context::AppContext,
    config::{PolicyConfig, RouterConfig},
    policies::kv_index::{event_monitor::bench_support::IndexFeed, index_backend::KvIndex},
    routers::{
        common::{
            placement::{select_single, PlacementInputs, Stated},
            snapshot::RoutingPool,
        },
        grpc::backend_client::BasicWorkerBuilder,
    },
};

const MODEL: &str = "sample-model";
const WORKERS: [&str; 3] = [
    "grpc://worker-1.invalid:9000",
    "grpc://worker-2.invalid:9000",
    "grpc://worker-3.invalid:9000",
];
/// The worker whose cache events stored the prompts' blocks.
const HOLDER: &str = WORKERS[1];
/// The index's block size in tokens. The stock document states another.
const BLOCK: usize = 32;
/// The blocks of one prompt.
const BLOCKS: usize = 32;
/// The prompts, each of tokens no other prompt holds.
const PROMPTS: u32 = 32;
/// The host's stage, as its table and its document name it.
const STAGE: &str = "hears-workers";
/// How long the engine's task may take to tell a stage a change of the
/// registry: far more than a busy machine needs.
const TOLD_WITHIN: Duration = Duration::from_secs(10);

fn register(app: &AppContext, url: &str) {
    let worker = BasicWorkerBuilder::new(url)
        .model(ModelCard::new(MODEL))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    app.worker_registry
        .register(Arc::new(worker))
        .expect("the worker registers");
}

/// The workers a stage was told joined and left, in order.
type Told = Arc<Mutex<Vec<(&'static str, WorkerId)>>>;

/// A host's own stage that takes the word of a worker that joins or leaves.
/// It ranks every worker alike.
#[derive(Debug)]
struct HearsWorkers(Told);

impl Lifecycle for HearsWorkers {
    fn hears(&self) -> EventSet {
        EventSet::NONE.with(EventKind::Membership)
    }

    fn on_membership(&self, change: &Membership<'_>) {
        let word = match change {
            Membership::Added(id) => ("joined", *id),
            Membership::Removed(id) => ("left", *id),
            _ => return,
        };
        self.0.lock().expect("the stage was told").push(word);
    }
}

impl Scorer for HearsWorkers {
    fn name(&self) -> &'static str {
        STAGE
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        TieBand::NORMALIZED
    }
}

/// What the stage was told once it holds `words` words, or what it holds
/// when [`TOLD_WITHIN`] has passed: the engine's task is a task of its own.
async fn told_of(told: &Told, words: usize) -> Vec<(&'static str, WorkerId)> {
    let asked = Instant::now();
    loop {
        let said = told.lock().expect("the stage was told").clone();
        if said.len() >= words || asked.elapsed() > TOLD_WITHIN {
            return said;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// The gateway hands the engine it started its worker registry, and the
/// engine's task follows the registry's events with no call of the gateway's:
/// a host's stage is told a worker registered behind the start as joined, and
/// the same worker as left once the registry removed it.
#[tokio::test]
async fn the_started_engine_tells_a_stage_a_worker_that_joins_and_one_that_leaves() {
    let told = Told::default();
    let kept = Arc::clone(&told);
    let mut stages = StageRegistry::builtin();
    stages
        .scorer(STAGE, move || Box::new(HearsWorkers(Arc::clone(&kept))))
        .expect("a free name");
    let document: PolicySpec = serde_json::from_value(serde_json::json!({
        "apiVersion": "smg.policy/v1",
        "generation": 1,
        "profiles": { "default": { "score": [STAGE], "pick": "best" } },
    }))
    .expect("a policy document");
    let config = RouterConfig {
        policy_controller: true,
        policy_spec: Some(document),
        ..RouterConfig::default()
    };
    let host = Extensions::new().policy_stages(stages);
    let app = build_app_context(&ServerConfig::minimal(config), host)
        .await
        .expect("the gateway starts");
    let worker = WorkerId::from_address(WORKERS[0]);

    register(&app, WORKERS[0]);
    assert_eq!(told_of(&told, 1).await, [("joined", worker)]);

    app.worker_registry
        .remove_by_url(WORKERS[0])
        .expect("the worker stood");
    assert_eq!(
        told_of(&told, 2).await,
        [("joined", worker), ("left", worker)]
    );
}

/// No document names an overload threshold: a gateway started under a
/// supplied document builds its worker objects with the two of its own
/// settings. A threshold off its default beside the document does not end
/// the start, and the registration settings every worker object is built
/// from hold the two.
#[tokio::test]
async fn the_started_gateway_builds_its_workers_with_its_own_overload_thresholds() {
    let config = RouterConfig {
        policy_controller: true,
        policy_spec: Some(PolicySpec::preset("round_robin").expect("a stock policy")),
        worker_overload_waiting_requests: Some(32),
        worker_overload_token_usage: Some(0.9),
        ..RouterConfig::default()
    };
    let app = build_app_context(&ServerConfig::minimal(config), Extensions::new())
        .await
        .expect("the gateway starts");
    let built = app.registration_settings.overload_defaults;
    assert_eq!(built.waiting_requests, Some(32));
    assert_eq!(built.token_usage, Some(0.9));
}

/// The stock document with one binding for each of `blocks`, under that
/// `overload` block: the last binding is the stock one, and each binding
/// ahead of it matches a model of its own.
fn bound(blocks: &[OverloadSpec]) -> PolicySpec {
    let mut document = PolicySpec::preset("round_robin").expect("a stock policy");
    let stock = document.bindings.remove(0);
    let ahead = blocks.len().saturating_sub(1);
    document.bindings = blocks
        .iter()
        .enumerate()
        .map(|(at, block)| {
            let mut binding = stock.clone();
            if at < ahead {
                binding.matcher.model = format!("model-{at}");
            }
            binding.overload = Some(block.clone());
            binding
        })
        .collect();
    document
}

/// A gateway that runs `document` under the policy controller with these
/// two overload thresholds of its own.
fn under(document: PolicySpec, waiting: Option<usize>, usage: Option<f64>) -> ServerConfig {
    ServerConfig::minimal(RouterConfig {
        policy_controller: true,
        policy_spec: Some(document),
        worker_overload_waiting_requests: waiting,
        worker_overload_token_usage: usage,
        ..RouterConfig::default()
    })
}

/// An `overload` block that says `enabled` and `shed`.
fn said(enabled: bool, shed: bool) -> OverloadSpec {
    OverloadSpec {
        enabled,
        shed_when_all_vetoed: shed,
        ..OverloadSpec::default()
    }
}

/// A gateway started under a supplied document reads the overload veto's
/// two words from that document, once. Bindings that switch the block off
/// leave the registration settings every worker object is built from no
/// default threshold, whatever the two threshold settings say, and the
/// registry the routes ask sheds by the document's word. Both thresholds at
/// none agree with such a document.
#[tokio::test]
async fn the_started_gateway_reads_its_documents_overload_switch_and_shed_word() {
    for (waiting, usage) in [(Some(32), Some(0.9)), (None, None)] {
        let off_and_shed = bound(&[said(false, true)]);
        let app = build_app_context(&under(off_and_shed, waiting, usage), Extensions::new())
            .await
            .expect("the gateway starts");
        assert_eq!(
            app.registration_settings.overload_defaults,
            OverloadThresholds::default()
        );
        assert!(app.worker_registry.overload_shed_enabled());
    }

    // A block that is on leaves the thresholds standing, one of them at none
    // too, and a document that says no shed leaves the shed word off.
    for (waiting, usage) in [(Some(32), Some(0.9)), (Some(32), None)] {
        let on = bound(&[said(true, false)]);
        let app = build_app_context(&under(on, waiting, usage), Extensions::new())
            .await
            .expect("the gateway starts");
        assert_eq!(
            app.registration_settings.overload_defaults,
            OverloadThresholds {
                waiting_requests: waiting,
                token_usage: usage,
            }
        );
        assert!(!app.worker_registry.overload_shed_enabled());
    }
}

/// A worker object has one default set of thresholds. A supplied document
/// whose bindings differ on the overload switch ends the start, in the
/// refusal's own words, and so does one whose bindings differ on the shed
/// word. So does a document whose block is on over a gateway whose two
/// thresholds are both none: the switch would switch nothing on, and the
/// refusal names the two settings and the two ways out.
#[tokio::test]
async fn a_document_whose_overload_words_this_gateway_refuses_ends_the_start() {
    let stated = (Some(32), Some(0.9));
    for (blocks, (waiting, usage), refusal) in [
        (
            vec![said(true, false), said(false, false)],
            stated,
            "policy controller: bindings[1].overload.enabled = false is not the true bindings[0] says, by its block or the block's default: a gateway builds its workers with one default set of overload thresholds, so the bindings of its document say one word",
        ),
        (
            vec![said(true, false), said(true, true)],
            stated,
            "policy controller: bindings[1].overload.shed_when_all_vetoed = true is not the false bindings[0] says, by its block or the block's default: a gateway's routes read one shed word for the process, so the bindings of its document say one word",
        ),
        (
            vec![said(true, false)],
            (None, None),
            "policy controller: worker_overload_waiting_requests and worker_overload_token_usage are both none beside a policy document whose overload block is on: a gateway with no default threshold vetoes by none; state a threshold, or switch the block off in the document",
        ),
    ] {
        let refused = build_app_context(&under(bound(&blocks), waiting, usage), Extensions::new())
            .await
            .err();
        assert_eq!(refused.as_deref(), Some(refusal));
    }
}

/// The token ids of prompt `number`: as many as [`BLOCKS`] blocks hold.
fn prompt(number: u32) -> Vec<u32> {
    let length = u32::try_from(BLOCK * BLOCKS).expect("a prompt's length");
    (0..length)
        .map(|token| 1 + number * 100_000 + token)
        .collect()
}

/// The event of a worker that stored `tokens` whole: one chain of blocks
/// from the prompt's first token on.
fn stored(event_id: u64, tokens: &[u32]) -> KvCacheEvent {
    let first = i64::try_from(event_id).expect("an event's number") * 1_000;
    let blocks = tokens
        .chunks(BLOCK)
        .zip(first..)
        .map(|(block, block_hash)| KvBlock {
            block_hash,
            token_ids: block.to_vec(),
            block_size: i32::try_from(BLOCK).expect("a block's size"),
            ..Default::default()
        })
        .collect();
    KvCacheEvent {
        event_id,
        data: Some(Data::Stored(KvBlocksStored {
            blocks,
            ..Default::default()
        })),
    }
}

/// The gateway hands the engine it started the monitor of its workers'
/// cache events. Under the cache-aware policy at its flags' defaults the
/// engine reads the model's block size from the monitor, and a prompt whose
/// blocks one worker's events stored is placed on that worker: 32 prompts of
/// tokens of their own all land on the holder, where an engine that read no
/// index would have no reason to.
#[tokio::test]
async fn the_started_engine_places_by_the_cache_events_the_monitor_applied() {
    let config = RouterConfig {
        policy_controller: true,
        policy: PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 67_108_864),
        ..RouterConfig::default()
    };
    let app = build_app_context(&ServerConfig::minimal(config), Extensions::new())
        .await
        .expect("the gateway starts");
    let monitor = app
        .kv_event_monitor
        .as_ref()
        .expect("the cache-event monitor");
    let index = Arc::new(KvIndex::chain());
    monitor.set_index(MODEL, Arc::clone(&index));
    monitor.set_block_size(MODEL, BLOCK);
    for url in WORKERS {
        register(&app, url);
    }

    let prompts: Vec<Vec<u32>> = (0..PROMPTS).map(prompt).collect();
    let batch = KvEventBatch {
        events: prompts
            .iter()
            .zip(1..)
            .map(|(tokens, event_id)| stored(event_id, tokens))
            .collect(),
        ..Default::default()
    };
    let mut feed = IndexFeed::new(&index, HOLDER).expect("a free worker number");
    feed.apply(&index, &batch);

    let engine = app
        .policy_registry
        .engine()
        .expect("the gateway started the engine");
    assert_eq!(
        engine.affinity().block_size(MODEL),
        u32::try_from(BLOCK).ok(),
        "the engine reads the monitor's block size"
    );
    for tokens in &prompts {
        let placed = select_single(
            &app.worker_registry,
            &app.policy_registry,
            MODEL,
            RoutingPool::GrpcPipelineRegular,
            None,
            PlacementInputs {
                tokens: Some(tokens),
                ..Default::default()
            },
            Stated::nothing,
        )
        .expect("a worker is selectable");
        assert_eq!(placed.worker().url(), HOLDER);
    }
}
