//! Cases of the feed's cache events: the crate's follow under the gateway's
//! feed, over a mock engine that stores cache blocks.

use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};

use kv_index::compute_content_hash;
use mock_worker::{config::Config as MockConfig, engine::EngineParams, grpc, overrides};
use smg_grpc_client::{
    tokenspeed_proto::{GenerateRequest, SamplingParams, TokenizedInput},
    TokenSpeedSchedulerClient as SchedulerClient,
};
use smg_policy_controller::{
    store::{
        Ask as ViewAsk, GatewayId, HealthPolicy, Maker, MemoryStore, Named, Naming, Signals,
        ViewBuffers, WorkerFacts,
    },
    view::{Role, WireModes, WorkerId},
    worker_host::{Expect, Wrote},
    AppliedSpec, PolicySpec,
};
use tokio::{
    net::TcpListener,
    time::{sleep, timeout},
};
use tokio_stream::StreamExt as _;

use super::{feed, Feed};
use crate::routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BackendConnector};

const MODEL: &str = "model-a";
/// A prompt of these tokens fills one block of the engine's cache.
const TOKENS: [u32; 4] = [1, 2, 3, 4];
const BLOCK: u32 = 4;
const WAIT: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_millis(1);

/// A mock engine with a prefix cache, serving for the test process.
struct Engine {
    url: String,
    kind: String,
}

impl Engine {
    async fn serve() -> Result<Self, Box<dyn Error>> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let addr = listener.local_addr()?;
        let mock = MockConfig {
            model_id: MODEL.to_owned(),
            realistic: true,
            engine: EngineParams {
                block_size: BLOCK,
                prefix_cache: true,
                ..EngineParams::default()
            },
            ..MockConfig::default()
        };
        let kind = mock.grpc_wire.as_str().to_owned();
        // A port an earlier engine served on keeps that engine's overrides.
        overrides::fresh_for_port(addr.port());
        #[expect(
            clippy::disallowed_methods,
            reason = "test engine: the server task lives for the test process"
        )]
        tokio::spawn(grpc::serve_with_listener(Arc::new(mock), listener));
        Ok(Self {
            url: format!("grpc://{addr}"),
            kind,
        })
    }

    /// The row of a gRPC worker that asks for its cache events.
    fn row(&self) -> WorkerFacts {
        WorkerFacts {
            key: self.url.clone(),
            engine: self.url.clone(),
            role: Role::Single,
            wire_modes: WireModes::GRPC,
            models: vec![MODEL.to_owned()],
            dp_size: 1,
            engine_kind: self.kind.clone(),
            kv_block_size: Some(BLOCK),
            signals: Signals {
                events: true,
                loads: true,
                ..Signals::default()
            },
            health: HealthPolicy {
                probes: false,
                ..HealthPolicy::default()
            },
            ..WorkerFacts::default()
        }
    }

    /// Write the engine's row at the feed's door.
    fn register(&self, feed: &Feed) -> Result<WorkerId, String> {
        let facts = self.row();
        match feed
            .door
            .found(&Maker::Control, Expect::Nothing, &self.url, &|_| {
                facts.clone()
            }) {
            Ok(Wrote::Registered(id)) => Ok(id),
            other => Err(format!("expected a registration, got {other:?}")),
        }
    }

    /// One request whose prompt fills one block of the engine's cache.
    async fn store_block(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let client = SchedulerClient::connect(&self.url).await?;
        let mut generated = client
            .generate(GenerateRequest {
                request_id: "cache-a-block".to_owned(),
                tokenized: Some(TokenizedInput {
                    input_ids: TOKENS.to_vec(),
                    ..TokenizedInput::default()
                }),
                sampling_params: Some(SamplingParams {
                    max_new_tokens: Some(1),
                    ..SamplingParams::default()
                }),
                ..GenerateRequest::default()
            })
            .await?;
        while let Some(frame) = generated.next().await {
            frame?;
        }
        generated.mark_completed();
        Ok(())
    }

    /// Wait until a stream dialed the engine through the worker's connector.
    async fn dialed(&self, registry: &WorkerRegistry) -> bool {
        let dialed = async {
            while !is_dialed(registry, &self.url) {
                sleep(POLL).await;
            }
        };
        timeout(WAIT, dialed).await.is_ok()
    }
}

fn is_dialed(registry: &WorkerRegistry, url: &str) -> bool {
    registry
        .get_by_url(url)
        .and_then(|worker| worker.engine_connection_if_initialized())
        .is_some()
}

/// A store whose one generation is the cache-aware preset, which names the
/// holders of a request's blocks.
fn cache_store() -> Result<Arc<MemoryStore>, Box<dyn Error>> {
    let mut spec = PolicySpec::preset("cache_aware")?;
    spec.generation = 1;
    let store = MemoryStore::new();
    store.apply(AppliedSpec::validate(&spec)?)?;
    Ok(Arc::new(store))
}

fn bed(store: &Arc<MemoryStore>) -> (Arc<WorkerRegistry>, Feed) {
    let registry = Arc::new(WorkerRegistry::new());
    let feed = feed(store, &registry, BackendConnector::shared());
    (registry, feed)
}

/// The rows a decision names for the prompt's block, each with the blocks
/// it holds.
fn holders(store: &MemoryStore) -> Vec<(WorkerId, u32)> {
    let hashes = [compute_content_hash(&TOKENS).0];
    store.deciding(|world| {
        let ask = ViewAsk {
            gateway: GatewayId::LOCAL,
            now: Instant::now(),
            pinned: None,
            affinity: false,
        };
        let mut view = ViewBuffers::default();
        let Some(built) = world.build(MODEL, Role::Single, &ask, &mut view) else {
            return Vec::new();
        };
        let naming = Naming {
            gateway: GatewayId::LOCAL,
            block_hashes: &hashes,
            namespace_seed: 0,
            seed: 0,
            cap: 0,
            wires: None,
            now_micros: built.now_micros,
        };
        let mut named = Named::default();
        world.name(MODEL, Role::Single, &naming, &view.candidates, &mut named);
        named
            .holders
            .iter()
            .map(|holder| {
                (
                    view.candidates[holder.row as usize].id,
                    holder.overlap_blocks,
                )
            })
            .collect()
    })
}

/// Wait until a decision names a holder of the prompt's block.
async fn named_holders(store: &MemoryStore) -> Option<Vec<(WorkerId, u32)>> {
    let named = async {
        loop {
            let named = holders(store);
            if !named.is_empty() {
                return named;
            }
            sleep(POLL).await;
        }
    };
    timeout(WAIT, named).await.ok()
}

#[tokio::test]
async fn a_stored_block_reaches_the_stores_index_under_the_feed() {
    let engine = Engine::serve().await.expect("the mock engine serves");
    let store = cache_store().expect("the store takes the preset");
    let (registry, feed) = bed(&store);
    let _events = feed.cache_events();
    let row = engine.register(&feed).expect("the worker registers");
    assert!(engine.dialed(&registry).await, "the follow asks the stream");
    assert!(holders(&store).is_empty());

    engine
        .store_block()
        .await
        .expect("the engine stores a block");

    let named = named_holders(&store)
        .await
        .expect("the block reaches the index");
    assert_eq!(named, [(row, 1)]);
}

#[tokio::test]
async fn the_follow_ends_when_the_answer_is_dropped() {
    let engine = Engine::serve().await.expect("the mock engine serves");
    let store = cache_store().expect("the store takes the preset");
    let (registry, feed) = bed(&store);
    engine.register(&feed).expect("the worker registers");
    let held = Arc::strong_count(&store);
    // The follow's first read holds the worker laid before it.
    let events = feed.cache_events();
    assert!(engine.dialed(&registry).await, "the follow asks the stream");
    assert!(
        Arc::strong_count(&store) > held,
        "the follow holds the store"
    );

    drop(events);

    // The follow and each of its streams let go of the store as they end.
    let ended = async {
        while Arc::strong_count(&store) != held {
            sleep(POLL).await;
        }
    };
    timeout(WAIT, ended)
        .await
        .expect("the follow and its streams end");
}
