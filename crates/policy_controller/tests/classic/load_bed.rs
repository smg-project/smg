//! The bed the replays of the two load sheets share: one worker among a
//! host's worker objects, the engine its objects dial and the host's load
//! monitor over them. The holder is built with the calls a controller
//! builds its own with, and the worker is written into the store the holder
//! follows. A replay scripts the engine's answer to a load poll, polls the
//! worker's group once through the worker crate's own monitor, and has the
//! monitor's own loop drop the worker's loads when its row leaves the ready
//! set.

use std::{
    collections::HashMap,
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use parking_lot::Mutex;
use smg_policy_controller::{
    engine::WireModes,
    store::{MemoryStore, Overload, WorkerFacts, WorkerStatus, WorkersStore},
    worker_host::{load_monitor, Members, Objects},
};
use smg_worker::{
    common_proto::KvEventBatch,
    connection::engine::{AdminAnswer, EngineConnection, EngineConnector, HealthAnswer, ZmqLink},
    health::overload::OverloadThresholds,
    identity::{basic::Worker, builder::rank_url},
    load::monitor::{poll_group_once_for_test, WorkerMonitor},
    registry::WorkerRegistry,
    worker::{
        ConnectionMode, ProfileOptions, RuntimeType, WorkerGroupKey, WorkerLoadResponse, WorkerType,
    },
};

const MODEL: &str = "m";

/// The worker's engine: the address of a worker of one rank, and what the
/// objects of a worker's ranks dial.
pub const ENGINE: &str = "grpc://w0:9000";

/// The time between two steps of a row, as the recorders set it.
pub const STEP: Duration = Duration::from_secs(1);

/// The poll interval a step's poll states: shorter than a step, so a rank
/// that pushed a step ago is asked again, as the recorder asks it.
const INTERVAL: Duration = Duration::from_millis(1);

/// The monitor's own poll interval: a day, so every poll of a row is a
/// step's.
const IDLE: Duration = Duration::from_secs(86_400);

/// Lets the monitor's tasks run until each waits again.
async fn settle() {
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
}

/// Runs a row's replay on a runtime of its own whose clock stands still.
pub fn paused<T>(replay: impl Future<Output = Result<T, String>>) -> Result<T, String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .build()
        .map_err(|err| err.to_string())?;
    runtime.block_on(replay)
}

/// How a row's worker stands among the host's objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Laid {
    /// One object under the engine's own address, which states the width
    /// and no rank: the engine whole.
    Whole,
    /// One object for each rank, as a registration of several ranks lays
    /// them.
    Ranks,
    /// One object, for the one rank of its engine the worker stands for.
    Rank(u16),
}

/// The engine a row's objects dial: it answers a load poll with the answer
/// the step scripted, or with a failure where the step scripted none.
#[derive(Debug, Default)]
struct Engine {
    answer: Mutex<Option<WorkerLoadResponse>>,
}

fn not_scripted<T>() -> Result<T, tonic::Status> {
    Err(tonic::Status::unimplemented(
        "the replay scripts no such call",
    ))
}

#[async_trait]
impl EngineConnection for Engine {
    fn carries_admin_calls(&self) -> bool {
        false
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, tonic::Status> {
        not_scripted()
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        match self.answer.lock().clone() {
            Some(answer) => Ok(answer),
            None => Err(tonic::Status::unavailable("the step scripts no report")),
        }
    }

    async fn flush_cache(&self, _timeout_s: f32) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn start_profile(&self, _options: &ProfileOptions) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        not_scripted()
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        not_scripted()
    }

    async fn subscribe_kv_events(
        &self,
        _start_seq: u64,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        not_scripted()
    }

    async fn subscribe_load_records(
        &self,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        not_scripted()
    }
}

/// A connector whose every gRPC connect is answered by the one engine.
#[derive(Debug)]
struct Dials(Arc<Engine>);

#[async_trait]
impl EngineConnector for Dials {
    async fn connect_grpc(
        &self,
        _base_url: &str,
        _runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        Ok(self.0.clone())
    }

    async fn connect_zmq(&self, _link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        Err("the replay scripts no such connect".to_owned())
    }

    async fn dial_grpc(&self, _base_url: &str, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}

/// One worker among a host's objects, its engine and the host's load
/// monitor over them.
pub struct Bed {
    store: MemoryStore,
    objects: Arc<Objects<Members>>,
    engine: Arc<Engine>,
    monitor: Arc<WorkerMonitor<Members>>,
    /// The worker's objects: one, or one a rank in rank order.
    workers: Vec<Arc<dyn Worker>>,
    laid: Laid,
}

impl Bed {
    /// A worker of `ranks` laid as `laid`, under a holder that states
    /// `defaults` for every worker, with `own` on each of the worker's
    /// rows. The monitor is built stopped ([`Self::start`]).
    pub fn new(
        ranks: u16,
        laid: Laid,
        defaults: OverloadThresholds,
        own: Overload,
    ) -> Result<Self, String> {
        let engine = Arc::new(Engine::default());
        let dials: Arc<dyn EngineConnector> = Arc::new(Dials(Arc::clone(&engine)));
        // The holder as a controller builds its own, with the row's
        // thresholds where a controller states its program's defaults: a
        // row's own values reach its object under any defaults, none too.
        let objects = Arc::new(
            Objects::new(Arc::new(WorkerRegistry::new()))
                .with_connector(dials)
                .with_host_local_unchecked()
                .with_status_followed()
                .with_overload_thresholds(defaults),
        );
        let store = MemoryStore::with_publish_window(Duration::ZERO);
        // Each row's key and the rank it states.
        let keys: Vec<(String, u16)> = match laid {
            Laid::Ranks if ranks > 1 => (0..ranks)
                .map(|rank| (rank_url(ENGINE, usize::from(rank)), rank))
                .collect(),
            Laid::Rank(rank) => vec![(rank_url(ENGINE, usize::from(rank)), rank)],
            Laid::Whole | Laid::Ranks => vec![(ENGINE.to_string(), 0)],
        };
        for (key, rank) in &keys {
            let written = store.upsert(WorkerFacts {
                key: key.clone(),
                engine: ENGINE.to_string(),
                incarnation: 1,
                models: vec![MODEL.to_string()],
                dp_rank: *rank,
                dp_size: ranks,
                wire_modes: WireModes::GRPC,
                overload: own,
                ..WorkerFacts::default()
            });
            let id = written.id();
            let stored = WorkersStore::get(&store, id).ok_or("no row behind its write")?;
            store
                .set_status(id, WorkerStatus::Ready, stored.revision)
                .ok_or("a status the store did not take")?;
        }
        store.publish_now();
        objects.follow(&store);
        let mut workers = Vec::with_capacity(keys.len());
        for (key, _) in &keys {
            let object = objects.registry().get_by_url(key);
            workers.push(object.ok_or_else(|| format!("no object under {key}"))?);
        }
        let monitor = load_monitor(&objects, IDLE);
        Ok(Self {
            store,
            objects,
            engine,
            monitor,
            workers,
            laid,
        })
    }

    /// Starts the monitor's own loop, which hears a worker leave the ready
    /// set. A group's loop polls once as it starts, ahead of the row's
    /// first step: the engine scripts no answer yet, so that poll brings no
    /// report to a monitor that holds none.
    pub async fn start(&self) {
        self.monitor.start_event_loop();
        settle().await;
    }

    /// The host's load monitor over the worker's objects.
    pub fn monitor(&self) -> &Arc<WorkerMonitor<Members>> {
        &self.monitor
    }

    /// The worker's objects: one, or one a rank in rank order.
    pub fn workers(&self) -> &[Arc<dyn Worker>] {
        &self.workers
    }

    /// One poll of the worker's group at `now`, through the monitor's own
    /// poll: the engine answers `answer`, or fails the call where the step
    /// scripts none.
    pub async fn poll(&self, answer: Option<WorkerLoadResponse>, now: Instant) {
        *self.engine.answer.lock() = answer;
        let group = WorkerGroupKey {
            model_id: MODEL.to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Grpc,
        };
        poll_group_once_for_test(&self.monitor, &group, INTERVAL, now).await;
    }

    /// The worker leaves the ready set: its rows are written not ready, the
    /// holder moves each object's status, and the monitor's loop drops the
    /// worker's loads at that word. The recorders' worker is polled again
    /// behind the drop, so its rows are ready again.
    pub async fn evict(&self) -> Result<(), String> {
        self.status(WorkerStatus::NotReady)?;
        settle().await;
        self.status(WorkerStatus::Ready)?;
        settle().await;
        Ok(())
    }

    /// Writes `status` on each row of the worker and has the holder read
    /// the change.
    fn status(&self, status: WorkerStatus) -> Result<(), String> {
        for worker in &self.workers {
            let row = self
                .store
                .by_key(worker.url())
                .ok_or("a worker object with no row")?;
            self.store
                .set_status(row.id, status, row.revision)
                .ok_or("a status the store did not take")?;
        }
        self.store.publish_now();
        self.objects.follow(&self.store);
        Ok(())
    }

    /// The object a rank's pushed record lands on: the rank's own where the
    /// worker stands as several objects, and `None` for a rank it has none
    /// for.
    pub fn pusher(&self, rank: u16) -> Option<&Arc<dyn Worker>> {
        match self.laid {
            Laid::Ranks if self.workers.len() > 1 => self.workers.get(usize::from(rank)),
            Laid::Whole | Laid::Ranks | Laid::Rank(_) => self.workers.first(),
        }
    }

    /// Whether each object of the worker reads as overloaded, in the
    /// objects' order.
    pub fn verdicts(&self) -> Vec<bool> {
        self.workers
            .iter()
            .map(|worker| worker.is_overloaded())
            .collect()
    }
}
