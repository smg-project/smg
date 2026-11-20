//! What a benchmark needs of the policy controller in this process: the
//! store, its membership feed and its host built as the context builds them
//! under the setting, workers registered through the feed's door as the
//! registration line writes them, and one request's routing as a regular
//! router runs it (the ask, the commit, the send's report, the load guard
//! and the end's report).

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use futures::FutureExt;
use openai_protocol::worker::WorkerLoadResponse;
use smg_policy_controller::{
    engine::WireModes,
    store::{LoadSink as StoreLoadSink, Maker},
    tracking::TransportClass,
    worker_host::Expect,
};
use smg_worker::identity::basic::{Worker, WorkerLoadGuard};

use super::{
    engine::{self, Ask, End, PolicyEngine, Prompt},
    feed::Feed,
    feeds, start_with, Started,
};
use crate::{
    config::RouterConfig,
    routers::{
        common::snapshot::{RoutingPool, WorkerRegistry},
        grpc::backend_client::BackendConnector,
    },
};

/// The route a request is placed on: what its router states of itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// The regular HTTP router: it states no wire and keeps to its own pool
    /// at the commit.
    Http,
    /// The gRPC pipeline's regular selection stage: it states the wires it
    /// dials and keeps to its own pool at the commit.
    Grpc,
}

/// One model's fleet placed by the policy controller in this process.
pub struct ControllerPath {
    host: Arc<PolicyEngine>,
    feed: Feed,
    registry: Arc<WorkerRegistry>,
    model: String,
}

impl ControllerPath {
    /// The store a gateway starts under `config`, which holds the setting,
    /// with its feed over a new registry and the host over both.
    ///
    /// # Errors
    ///
    /// The start's refusal, in its words, and a configuration under which
    /// the classic policies decide.
    pub fn start(config: &RouterConfig, model: &str) -> Result<Self, String> {
        let started = start_with(config, None).map_err(|refusal| refusal.to_string())?;
        let Some(Started { store, door: _ }) = started else {
            return Err("the classic policies decide under this configuration".to_owned());
        };
        let registry = Arc::new(WorkerRegistry::new());
        let (feed, _) = feeds(&store, &registry, BackendConnector::shared(), None);
        let host = Arc::new(PolicyEngine::new(
            Arc::clone(&store),
            Arc::clone(&feed.objects),
        ));
        Ok(Self {
            host,
            feed,
            registry,
            model: model.to_owned(),
        })
    }

    /// The registry the feed's holder writes.
    pub fn registry(&self) -> &Arc<WorkerRegistry> {
        &self.registry
    }

    /// Registers `workers`, one registration a worker, through the feed's
    /// door: the write the registration line makes for a local worker.
    ///
    /// # Errors
    ///
    /// The door's refusal of a write.
    pub fn register(&self, workers: &[Arc<dyn Worker>]) -> Result<(), String> {
        for worker in workers {
            self.feed
                .door
                .found_rows(
                    &Maker::Control,
                    Expect::Nothing,
                    std::slice::from_ref(worker),
                    None,
                )
                .map_err(|refused| format!("{refused:?}"))?;
        }
        Ok(())
    }

    /// Says one load report a worker through the feed's load sink, as the
    /// load collector does, and shows them to the decisions.
    pub fn report_loads(&self, loads: &[(Arc<dyn Worker>, WorkerLoadResponse)]) {
        let sink = self.feed.load_sink(Duration::from_secs(1));
        let at = Instant::now();
        for (worker, report) in loads {
            sink.record(worker, 0, report, at, at);
        }
        let store = self.feed.store();
        let due = StoreLoadSink::publish_due(&**store).map_or(at, |due| due.max(at));
        StoreLoadSink::publish(&**store, due);
    }

    /// One pass of the host's clock, as its timer makes about once a second.
    pub fn upkeep(&self) {
        self.host.sweep(Instant::now());
    }

    /// One request's routing on `route`: the prompt's hashes, the ask and
    /// the commit, the worker's load guard, the send's report and the end's
    /// report. `false` where the request had no worker.
    pub fn route(&self, route: Route, tokens: Option<&[u32]>) -> bool {
        let host = &self.host;
        let model = self.model.as_str();
        let hashes = host.hashes(model, &Prompt::of(tokens, None));
        let id = engine::mark();
        let (wires, pool) = match route {
            Route::Http => (WireModes::NONE, RoutingPool::HttpRegular),
            Route::Grpc => (
                WireModes::GRPC.union(WireModes::ZMQ),
                RoutingPool::GrpcPipelineRegular,
            ),
        };
        let ask = Ask {
            wires,
            hashes: Some(&hashes),
            ..Ask::of(&id, model)
        };
        let placing = host.place_where(&ask, |worker| pool.matches(worker), Instant::now());
        // The door in this process decides on the calling thread.
        let Some(Ok(mut placed)) = placing.now_or_never() else {
            return false;
        };
        let guard = WorkerLoadGuard::with_key(Arc::clone(placed.worker()), None);
        placed.dispatched(Instant::now());
        let end = End {
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        };
        placed.completed(end, Instant::now());
        drop(guard);
        true
    }
}
