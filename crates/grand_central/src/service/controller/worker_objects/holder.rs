//! The holder of the replica's worker objects over the controller's store.

use std::sync::Arc;

use smg_grpc_client::engine_port::GrpcConnector;
use smg_worker::{
    connection::engine::EngineConnector, health::overload::OverloadThresholds,
    registry::WorkerRegistry,
};

#[cfg(test)]
use crate::service::controller::Config;
use crate::{
    engine::CacheIndexes,
    store::MemoryStore,
    worker_host::{EventIndexes, KvTables, Members, Objects},
};

/// The replica's worker objects: their holder and the store their rows
/// stand in. The registry is built here, and the holder alone
/// writes it. Each object connects to its engine through the engine clients'
/// gRPC connector, so a worker of the host-local wire has no connection
/// here. The replica cannot dial a row that one gateway alone reaches: such
/// a row's object is built with its checks off, and no probe is sent to it.
///
/// The object of an engine's row states its row's overload thresholds over
/// the program's default, which the controller's start states
/// ([`Config::worker_overload`](crate::service::controller::Config::worker_overload)),
/// and an external row's object its row's alone: what a load monitor over
/// the registry judges the worker by
/// ([`load_monitor`](crate::worker_host::load_monitor)). One set a worker:
/// a registration's own values, each over the start's for an engine's
/// worker. The start's are stated while the document in force leaves its
/// overload switch on, and an engine's row states its own alone where
/// every binding switches it off: the document door states the word
/// ([`Objects::state_overload_switch`]).
/// The service runs that monitor while the replica leads, and it writes
/// each verdict on the worker's object.
///
/// An object's status leads its row's. A decision reads a worker's status
/// on its object. While the replica leads, a probe's outcome, a connect
/// signal and a drain are written on the object, and the store is told.
/// While it follows, the holder lays each status its leader's log tells.
/// So the holder follows no status of its store's own tail.
pub(crate) struct WorkerObjects {
    store: Arc<MemoryStore>,
    objects: Arc<Objects<Members>>,
    /// The KV tables the holder was handed, kept beside it: a reader of the
    /// replica's tables is handed these, the ones its feed writes.
    tables: Arc<KvTables>,
}

impl WorkerObjects {
    /// The objects of `store`'s rows, none built yet: the write that makes
    /// a row lays its object. `overload` is the thresholds the object of an
    /// engine's row states for each signal its row names none for: the
    /// start's.
    pub(crate) fn stating(store: Arc<MemoryStore>, overload: OverloadThresholds) -> Self {
        Self::with_connector(store, Arc::new(GrpcConnector), overload)
    }

    /// [`Self::stating`] the default thresholds of a start, for a case that
    /// sets no others.
    #[cfg(test)]
    pub(crate) fn new(store: Arc<MemoryStore>) -> Self {
        Self::stating(store, Config::DEFAULT_WORKER_OVERLOAD)
    }

    /// [`Self::stating`] with `connector` making each object's engine
    /// connection.
    pub(super) fn with_connector(
        store: Arc<MemoryStore>,
        connector: Arc<dyn EngineConnector>,
        overload: OverloadThresholds,
    ) -> Self {
        // The KV tables are the replica's own, made here once: the holder
        // states its rows to them and the feed writes them.
        let tables = Arc::new(KvTables::default());
        Self {
            store,
            objects: Arc::new(
                Objects::new(Arc::new(WorkerRegistry::new()))
                    .with_connector(connector)
                    .with_host_local_unchecked()
                    .with_overload_thresholds(overload)
                    .with_kv_tables(Arc::clone(&tables)),
            ),
            tables,
        }
    }

    /// The store the objects' rows stand in.
    pub(crate) fn store(&self) -> Arc<MemoryStore> {
        Arc::clone(&self.store)
    }

    /// The replica's KV tables: the ones the holder was handed at its build.
    pub(crate) fn kv_tables(&self) -> Arc<KvTables> {
        Arc::clone(&self.tables)
    }

    /// The indexes of cache events the replica's engine reads: its KV
    /// tables' own, a worker found by its address among the worker objects.
    /// The store is lent them once, at the controller's build.
    pub(crate) fn cache_indexes(&self) -> Arc<dyn CacheIndexes> {
        Arc::new(EventIndexes::new(Arc::clone(&self.tables), &self.objects))
    }

    /// The holder shared with readers of the replica's worker objects.
    pub(crate) fn holder(&self) -> Arc<Objects<Members>> {
        Arc::clone(&self.objects)
    }

    /// The holder, for a case that reads what stands.
    #[cfg(test)]
    pub(crate) fn objects(&self) -> &Objects<Members> {
        &self.objects
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use smg_worker::{
        health::port::{HealthSink, ProbeAnswer},
        worker::WorkerStatus as ObjectStatus,
    };
    use tokio::{net::TcpListener, sync::oneshot, task::JoinSet};

    use super::*;
    use crate::{
        document::spec::{OVERLOAD_TOKEN_USAGE, OVERLOAD_WAITING_REQUESTS},
        engine::view::WireModes,
        store::{
            HealthPolicy, HealthWrite, Overload, Registrant, WorkerFacts, WorkerStatus,
            WorkersStore,
        },
    };

    const URL: &str = "http://w:8000";

    fn facts(incarnation: u64, dp_size: u16) -> WorkerFacts {
        WorkerFacts {
            key: URL.to_string(),
            engine: URL.to_string(),
            incarnation,
            models: vec!["m".to_string()],
            dp_size,
            ..WorkerFacts::default()
        }
    }

    /// A replica's store and its worker objects.
    fn replica() -> (Arc<MemoryStore>, WorkerObjects) {
        let store = Arc::new(MemoryStore::default());
        let workers = WorkerObjects::new(Arc::clone(&store));
        (store, workers)
    }

    /// Lay `facts` through the holder, as a registration call lays them.
    fn register(store: &MemoryStore, workers: &WorkerObjects, facts: WorkerFacts) {
        let change = workers.objects().upsert(store, facts, Registrant::Control);
        assert!(change.is_some(), "the store took the registration");
    }

    /// A replica's object states its row's overload thresholds, each over
    /// the program's default, which its start states. The start's own
    /// default is the worker crate's two numbers, which are the document's.
    /// Its usage is the 64-bit 0.8 an engine reports: the document's 32-bit
    /// constant is the same number, and widened it reads above it. A start
    /// that states its own two has them stated by the object of a row that
    /// names none, and a row's own count stands over the start's as over
    /// the default. A signal a start states none for has no default: the
    /// object of a row that names none for it states none.
    #[test]
    fn an_object_states_its_rows_thresholds_over_the_programs_default() {
        const STRICT: &str = "http://strict:8000";
        let defaults = OverloadThresholds {
            waiting_requests: Some(8),
            token_usage: Some(0.8),
        };
        assert_eq!(Config::DEFAULT_WORKER_OVERLOAD, defaults);
        assert_eq!(OVERLOAD_WAITING_REQUESTS, 8);
        assert_eq!(0.8_f64 as f32, OVERLOAD_TOKEN_USAGE);
        assert!(f64::from(OVERLOAD_TOKEN_USAGE) > 0.8, "widened, it is over");

        let started = OverloadThresholds {
            waiting_requests: Some(4),
            token_usage: Some(0.625),
        };
        let count_alone = OverloadThresholds {
            token_usage: None,
            ..started
        };
        let usage_alone = OverloadThresholds {
            waiting_requests: None,
            ..started
        };
        let neither = OverloadThresholds::default();
        assert_eq!(
            (neither.waiting_requests, neither.token_usage),
            (None, None)
        );
        for start in [defaults, started, count_alone, usage_alone, neither] {
            let store = Arc::new(MemoryStore::default());
            let workers = WorkerObjects::stating(Arc::clone(&store), start);
            register(&store, &workers, facts(1, 1));
            let strict = WorkerFacts {
                key: STRICT.to_string(),
                engine: STRICT.to_string(),
                overload: Overload {
                    kv_usage_permille: None,
                    waiting_requests: Some(2),
                },
                ..facts(1, 1)
            };
            register(&store, &workers, strict);
            let stated = |url: &str| {
                let registry = workers.objects().registry();
                let object = registry.get_by_url(url).expect("the row's object");
                object.metadata().overload
            };
            assert_eq!(stated(URL), start);
            let own = OverloadThresholds {
                waiting_requests: Some(2),
                ..start
            };
            assert_eq!(stated(STRICT), own);
        }
    }

    /// Two moves in a row end with the table at the object's newest
    /// status. The follow reads the first move's event between the second
    /// move and its telling, and leaves the object alone: the replica's
    /// holder copies no status of its store's tail back onto an object.
    /// With that copy the second move is lost: the object goes back to the
    /// first status, and its tell finds nothing to tell.
    #[test]
    fn two_moves_in_a_row_end_with_the_table_at_the_objects_newest_status() {
        let (store, workers) = replica();
        let holder = workers.objects();
        // A row each probe moves.
        let stated = WorkerFacts {
            health: HealthPolicy {
                failure_threshold: 1,
                success_threshold: 1,
                ..HealthPolicy::default()
            },
            ..facts(1, 1)
        };
        let change = holder.upsert(&store, stated, Registrant::Control);
        let row = change.expect("registered").id();
        let id = holder.object(row).expect("the object");
        let registry = holder.registry().as_ref();
        let object = || registry.get(&id).expect("it stands");
        let revision = object().revision();
        // One probe, judged by the registry's own state machine as the
        // health adapter hands it on: the object alone moves.
        let probe = |passed: bool| {
            HealthSink::probe_outcome(registry, &id, passed, revision, object().status())
        };
        let tell = |moved: ProbeAnswer| {
            let ProbeAnswer::Moved { old, new } = moved else {
                panic!("the probe moved nothing: {moved:?}");
            };
            holder.tell_status(&store, &id, revision, (old, new), &|| true)
        };
        let table = || WorkersStore::get(&*store, row).map(|row| row.status);
        let statuses = || (table(), object().status());
        // The first move, told.
        assert!(tell(probe(true)).moved().is_some());
        let ready = (Some(WorkerStatus::Ready), ObjectStatus::Ready);
        assert_eq!(statuses(), ready);
        // The second move stands on the object, and its tell has not run.
        let second = probe(false);
        holder.follow(&*store);
        let trails = (Some(WorkerStatus::Ready), ObjectStatus::NotReady);
        assert_eq!(statuses(), trails, "the follow left the object alone");
        let told = HealthWrite::Moved {
            from: WorkerStatus::Ready,
            to: WorkerStatus::NotReady,
        };
        assert_eq!(tell(second), told);
        let down = (Some(WorkerStatus::NotReady), ObjectStatus::NotReady);
        assert_eq!(statuses(), down);
    }

    /// A mock engine that serves `model` behind a port of its own: its
    /// address, what stops it and its task.
    pub(crate) async fn engine(model: &str) -> (String, oneshot::Sender<()>, JoinSet<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let addr = listener.local_addr().expect("the port's address");
        let config = mock_worker::config::Config {
            grpc_base_port: addr.port(),
            grpc_count: 1,
            grpc_dp_size: 1,
            grpc_wire: mock_worker::config::SchedulerWire::Vllm,
            loads_like: mock_worker::engine::LoadsLike::Vllm,
            realistic: true,
            model_id: model.to_string(),
            ..mock_worker::config::Config::default()
        };
        let (stop, stopped) = oneshot::channel::<()>();
        let mut task = JoinSet::new();
        task.spawn(mock_worker::grpc::serve_with_listener_until(
            Arc::new(config),
            listener,
            async move {
                let _ = stopped.await;
            },
        ));
        (format!("http://{addr}"), stop, task)
    }

    /// The facts of a gRPC worker under `key` whose engine answers at
    /// `engine`.
    fn grpc_worker(key: &str, engine: &str) -> WorkerFacts {
        WorkerFacts {
            key: key.to_string(),
            engine: engine.to_string(),
            incarnation: 1,
            models: vec!["m".to_string()],
            dp_size: 1,
            wire_modes: WireModes::GRPC,
            engine_kind: "vllm".to_string(),
            ..WorkerFacts::default()
        }
    }

    /// A row of a gRPC worker builds an object that reaches its engine. The
    /// holder hands each object the gRPC connector, and the object's own
    /// connection reads the engine's health, its model's labels and its load.
    #[tokio::test]
    async fn a_rows_object_reaches_its_engine_over_the_controllers_connector() {
        let (url, stop, _engine) = engine("served").await;
        let (store, workers) = replica();
        register(&store, &workers, grpc_worker(&url, &url));
        let registry = workers.objects().registry();
        let object = registry.get_by_url(&url).expect("the row's object");
        let connection = object
            .engine_connection()
            .await
            .expect("a connect")
            .expect("a gRPC worker's connection");
        assert!(connection.health_check().await.expect("a verdict"));
        let labels = connection.model_labels().await.expect("the labels");
        assert!(labels.values().any(|value| value == "served"), "{labels:?}");
        let loads = connection.get_loads().await.expect("a load answer");
        assert_eq!(loads.dp_rank_count, 1);

        drop(stop);
    }

    /// A row whose engine's address moved reads the new engine. The row
    /// keeps its key, so its object is replaced in place, and the replacement
    /// connects for itself: the standing object's connection is the old
    /// engine's.
    #[tokio::test]
    async fn a_row_whose_engine_moved_reads_the_new_engine() {
        const KEY: &str = "http://worker:8000";
        let (first, _stop_first, _first_engine) = engine("first").await;
        let (second, _stop_second, _second_engine) = engine("second").await;
        let (store, workers) = replica();
        let registry = workers.objects().registry();
        // The model the engine behind the row's object says it serves.
        let served = || async {
            let object = registry.get_by_url(KEY).expect("the row's object");
            let connection = object
                .engine_connection()
                .await
                .expect("a connect")
                .expect("a gRPC worker's connection");
            connection.model_labels().await.expect("the labels")
        };

        register(&store, &workers, grpc_worker(KEY, &first));
        let labels = served().await;
        assert!(labels.values().any(|value| value == "first"), "{labels:?}");

        register(&store, &workers, grpc_worker(KEY, &second));
        let labels = served().await;
        assert!(labels.values().any(|value| value == "second"), "{labels:?}");
    }
}
