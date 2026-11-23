//! The holder of the replica's worker objects over the controller's store,
//! and the one task that follows it.

use std::sync::Arc;

use smg_grpc_client::engine_port::GrpcConnector;
use smg_worker::{
    connection::engine::EngineConnector,
    health::overload::{OverloadThresholds, DEFAULT_TOKEN_USAGE_CEILING, DEFAULT_WAITING_REQUESTS},
    registry::WorkerRegistry,
};
use tokio::sync::watch;

use super::ObjectsWake;
use crate::{
    store::MemoryStore,
    worker_host::{Members, Objects},
};

/// The overload thresholds an object states for each signal its row names
/// none for: the worker crate's two default numbers, which a gateway hands
/// each worker of an engine too. An external row's object takes none, as a
/// gateway's external worker takes none at its registration. They are this
/// crate's own two defaults, by which the store judges under a binding
/// whose block is on
/// ([`OVERLOAD_WAITING_REQUESTS`](crate::document::spec::OVERLOAD_WAITING_REQUESTS),
/// [`OVERLOAD_TOKEN_USAGE`](crate::document::spec::OVERLOAD_TOKEN_USAGE)):
/// no document names a threshold.
/// One set a worker: a registration's own values, each over these for an
/// engine's worker.
///
/// The usage is the worker crate's 64-bit number, as an engine reports it.
/// This crate's constant is 32 bits wide: widened, it reads just above
/// the 0.8 an engine reports, and a worker at the threshold would not be
/// over it.
const OVERLOAD_DEFAULTS: OverloadThresholds = OverloadThresholds {
    waiting_requests: Some(DEFAULT_WAITING_REQUESTS),
    token_usage: Some(DEFAULT_TOKEN_USAGE_CEILING),
};

/// The replica's worker objects: their holder, the store it follows and the
/// wake of its one task. The registry is built here, and the holder alone
/// writes it. Each object connects to its engine through the engine clients'
/// gRPC connector, so a worker of the host-local wire has no connection
/// here. The replica cannot dial a row that one gateway alone reaches: such
/// a row's object is built with its checks off, and no probe is sent to it.
///
/// The object of an engine's row states its row's overload thresholds over
/// the program's default ([`OVERLOAD_DEFAULTS`]), and an external row's
/// object its row's alone: what a load monitor over the registry judges
/// the worker by ([`load_monitor`](crate::worker_host::load_monitor)).
/// The service runs that monitor while the replica leads, and it writes
/// each verdict on the worker's object.
///
/// An object's status follows its row's
/// ([`Objects::with_status_followed`]). A decision reads a worker's status
/// on its object, and a probe's outcome is not the only write of a row's
/// status: a drain marks the row, and a standby applies the statuses its
/// leader wrote.
pub(crate) struct WorkerObjects {
    store: Arc<MemoryStore>,
    objects: Arc<Objects<Members>>,
    wake: ObjectsWake,
}

impl WorkerObjects {
    /// The objects of `store`'s rows, none built yet: [`Self::run`] builds
    /// them. `wake` is the copy of the one the store's log tells.
    pub(crate) fn new(store: Arc<MemoryStore>, wake: ObjectsWake) -> Self {
        Self::with_connector(store, wake, Arc::new(GrpcConnector))
    }

    /// [`Self::new`] with `connector` making each object's engine
    /// connection.
    pub(super) fn with_connector(
        store: Arc<MemoryStore>,
        wake: ObjectsWake,
        connector: Arc<dyn EngineConnector>,
    ) -> Self {
        Self {
            store,
            objects: Arc::new(
                Objects::new(Arc::new(WorkerRegistry::new()))
                    .with_connector(connector)
                    .with_host_local_unchecked()
                    .with_status_followed()
                    .with_overload_thresholds(OVERLOAD_DEFAULTS),
            ),
            wake,
        }
    }

    /// The store the objects' rows stand in.
    pub(crate) fn store(&self) -> Arc<MemoryStore> {
        Arc::clone(&self.store)
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

    /// Follows the store until `stop` reads true: once at the start, then
    /// once a wake. A wake says only that the store's watched kinds moved,
    /// and writes that land together are told once, so each turn reads the
    /// store to its end ([`Objects::follow`]). A wake that comes while a
    /// turn works is kept, once, and starts the next: no write is slept
    /// through.
    ///
    /// The task waits on its own wake and on no timer. It hangs on no
    /// watch: a replica no gateway watches follows as any other. A leader
    /// and a standby run it alike, the standby over the rows it applies.
    pub(crate) async fn run(&self, mut stop: watch::Receiver<bool>) {
        loop {
            self.objects.follow(&*self.store);
            tokio::select! {
                () = self.wake.moved.notified() => {}
                _ = stop.wait_for(|stop| *stop) => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, time::Instant};

    use tokio::{
        net::TcpListener,
        sync::oneshot,
        task::{yield_now, JoinSet},
    };

    use super::*;
    use crate::{
        document::spec::{OVERLOAD_TOKEN_USAGE, OVERLOAD_WAITING_REQUESTS},
        engine::view::{WireModes, WorkerId},
        service::{
            controller::{
                election::LeasePlace, replication::journaled_at, standby::Follower,
                watch_workers::Feed, worker_wire,
            },
            proto,
            replication::Log,
        },
        store::{
            LeasesStore, Overload, Position, WorkerFacts, WorkerPatch, WorkerStatus, WorkersStore,
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

    /// A replica: its store over its log, its worker objects, and their
    /// one task on the case's runtime.
    struct Replica {
        store: Arc<MemoryStore>,
        feed: Feed,
        workers: Arc<WorkerObjects>,
        stop: watch::Sender<bool>,
        task: JoinSet<()>,
    }

    /// One write of a store, under the name a failure says.
    type Write<'a> = (&'a str, &'a dyn Fn(&MemoryStore));

    impl Replica {
        fn new() -> Self {
            let wake = ObjectsWake::default();
            let place = LeasePlace::File("shared".into());
            let (store, feed) = journaled_at(&place, 0, 1, wake.clone());
            let workers = Arc::new(WorkerObjects::new(Arc::clone(&store), wake));
            let (stop, stopped) = watch::channel(false);
            let mut task = JoinSet::new();
            task.spawn({
                let workers = Arc::clone(&workers);
                async move { workers.run(stopped).await }
            });
            Self {
                store,
                feed,
                workers,
                stop,
                task,
            }
        }

        /// Each object with its row's id and the revision it was built
        /// from, by URL.
        fn held(&self) -> Vec<(String, WorkerId, u64)> {
            let objects = &self.workers.objects;
            let mut held: Vec<_> = objects
                .registry()
                .get_all_with_ids()
                .into_iter()
                .map(|(id, object)| {
                    let standing = objects.standing(&id).expect("an object's row");
                    (object.url().to_string(), standing.row, standing.revision)
                })
                .collect();
            held.sort();
            held
        }

        /// Each row of the store with its id and its revision, by key.
        fn rows(&self) -> Vec<(String, WorkerId, u64)> {
            let (_, rows) = self.store.all_rows();
            rows.into_iter()
                .map(|row| (row.facts.key, row.id, row.revision))
                .collect()
        }

        /// Lets the follower run until it has read the store to its end
        /// and holds its rows' objects. The case's clock stands still and
        /// nothing advances it: a follower that waited on a timer would
        /// never get here.
        async fn caught_up(&self, after: &str) {
            for _ in 0..64 {
                let read = self.workers.objects.read_to() == Some(self.store.epoch());
                if read && self.held() == self.rows() {
                    return;
                }
                yield_now().await;
            }
            panic!("the follower did not catch up after {after}");
        }

        async fn stopped(mut self) {
            self.stop.send_replace(true);
            let ended = self.task.join_next().await.expect("the follower's task");
            ended.expect("the task ends at the stop");
        }
    }

    /// The standby applies what the leader's log holds after its place, as
    /// the batches a leader sends: a standby that applied nothing from the
    /// leader's first write.
    fn ship(leader: &Log, follower: &Follower) {
        let now = Instant::now();
        let mut place = match follower.place() {
            nothing if nothing == Position::default() => {
                leader.first().expect("every write is retained")
            }
            place => place,
        };
        let mut entries = Vec::new();
        loop {
            leader
                .read(place, 64, &mut entries)
                .expect("the place is held");
            let Some(last) = entries.last() else {
                return;
            };
            place = last.at;
            for entry in &entries {
                let batch = proto::ReplicationBatch {
                    first_seq: entry.at.seq,
                    term: entry.at.term,
                    events: vec![worker_wire::replication(entry)],
                    resync_required: false,
                    snapshot: None,
                    head_seq: entry.at.seq,
                };
                assert_eq!(follower.apply(&batch, now), Ok(()), "{entry:?}");
            }
        }
    }

    /// A replica's object states its row's overload thresholds, each over
    /// the program's default: the worker crate's two default numbers, which
    /// are the document's. The default's usage is the 64-bit 0.8 an engine
    /// reports: the document's 32-bit constant is the same number, and
    /// widened it reads above it.
    #[tokio::test(start_paused = true)]
    async fn an_object_states_its_rows_thresholds_over_the_programs_default() {
        const STRICT: &str = "http://strict:8000";
        let defaults = OverloadThresholds {
            waiting_requests: Some(8),
            token_usage: Some(0.8),
        };
        assert_eq!(OVERLOAD_DEFAULTS, defaults);
        assert_eq!(OVERLOAD_WAITING_REQUESTS, 8);
        assert_eq!(0.8_f64 as f32, OVERLOAD_TOKEN_USAGE);
        assert!(f64::from(OVERLOAD_TOKEN_USAGE) > 0.8, "widened, it is over");

        let replica = Replica::new();
        replica.store.upsert(facts(1, 1));
        replica.store.upsert(WorkerFacts {
            key: STRICT.to_string(),
            engine: STRICT.to_string(),
            overload: Overload {
                kv_usage_permille: None,
                waiting_requests: Some(2),
            },
            ..facts(1, 1)
        });
        replica.caught_up("two registrations").await;
        let stated = |url: &str| {
            let registry = replica.workers.objects.registry();
            let object = registry.get_by_url(url).expect("the row's object");
            object.metadata().overload
        };
        assert_eq!(stated(URL), defaults);
        let own = OverloadThresholds {
            waiting_requests: Some(2),
            ..defaults
        };
        assert_eq!(stated(STRICT), own);
        replica.stopped().await;
    }

    /// The six membership changes, one write each on the leader. Each
    /// wakes the leader's follower, and applied on the standby wakes the
    /// standby's: both hold the objects of the same rows after each.
    #[tokio::test(start_paused = true)]
    async fn each_membership_change_wakes_the_follower_on_a_leader_and_on_a_standby() {
        let began = tokio::time::Instant::now();
        let leader = Replica::new();
        leader.store.set_term(1);
        let standby = Replica::new();
        let follower = Follower::new(
            Arc::clone(&standby.store),
            Arc::clone(&standby.feed.log),
            standby.workers.holder(),
        );
        let row = |store: &MemoryStore| store.by_key(URL).expect("a row");
        let writes: [Write<'_>; 6] = [
            ("a registration", &|store| {
                store.upsert(facts(1, 1));
            }),
            ("a relabel", &|store| {
                let (row, labels) = (row(store), BTreeMap::from([("a".into(), "b".into())]));
                let patch = WorkerPatch {
                    labels: Some(labels),
                    ..WorkerPatch::default()
                };
                store.patch(row.id, patch, row.revision).expect("a patch");
            }),
            ("a resize", &|store| {
                store.upsert(facts(1, 2));
            }),
            ("a status change", &|store| {
                let row = row(store);
                store
                    .set_status(row.id, WorkerStatus::Ready, row.revision)
                    .expect("a status");
            }),
            ("a new incarnation", &|store| {
                store.upsert(facts(2, 1));
            }),
            ("a removal", &|store| {
                let row = row(store);
                store.remove(row.id, row.revision).expect("a removal");
            }),
        ];
        let counts = [1, 1, 1, 1, 1, 0];
        for ((name, write), count) in writes.into_iter().zip(counts) {
            write(&leader.store);
            leader.caught_up(name).await;
            ship(&leader.feed.log, &follower);
            standby.caught_up(name).await;
            assert_eq!(leader.held().len(), count, "{name}");
            assert_eq!(standby.held(), leader.held(), "{name}");
        }
        assert_eq!(tokio::time::Instant::now(), began, "no timer ran");
        leader.stopped().await;
        standby.stopped().await;
    }

    /// A store that takes another's rows whole with no turn of the holder
    /// wakes the follower by the log's restart, and the follower lays the
    /// objects of those rows. A standby's applier restores in the holder's
    /// turn: this is the restore that went around it.
    #[tokio::test(start_paused = true)]
    async fn a_restore_around_the_holder_is_laid_at_the_logs_restart() {
        let leader = Replica::new();
        leader.store.set_term(1);
        leader.store.upsert(facts(1, 1));
        leader.caught_up("the leader's registration").await;
        let standby = Replica::new();
        standby.caught_up("the standby's start").await;
        assert!(standby.held().is_empty());
        let cut = leader.store.checkpoint();
        assert!(standby.store.restore_from(cut));
        assert!(standby.held().is_empty(), "the follower's to lay");
        standby.caught_up("the restore").await;
        assert_eq!(standby.held(), leader.held());
        assert_eq!(standby.held().len(), 1);
        leader.stopped().await;
        standby.stopped().await;
    }

    /// Bursts of writes from one thread while the follower works on
    /// another: a wake that comes while a turn works starts the next, so
    /// the last write of a burst is never slept through.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_wake_that_comes_while_the_follower_works_is_kept() {
        let replica = Replica::new();
        replica.store.set_term(1);
        for round in 0..200_u64 {
            for step in 0..8 {
                let width = 1 + u16::from((round + step) % 2 == 1);
                replica.store.upsert(facts(1, width));
            }
            let mut spins = 0_u64;
            while replica.workers.objects.read_to() != Some(replica.store.epoch()) {
                spins += 1;
                assert!(
                    spins < 50_000_000,
                    "round {round}: a write was slept through"
                );
                std::thread::yield_now();
            }
        }
        assert_eq!(replica.held(), replica.rows());
        replica.stopped().await;
    }

    /// A mock engine that serves `model` behind a port of its own: its
    /// address, what stops it and its task.
    async fn engine(model: &str) -> (String, oneshot::Sender<()>, JoinSet<()>) {
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
        let replica = Replica::new();
        replica.store.set_term(1);
        replica.store.upsert(grpc_worker(&url, &url));
        replica.caught_up("the registration").await;
        let registry = replica.workers.objects.registry();
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
        replica.stopped().await;
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
        let replica = Replica::new();
        replica.store.set_term(1);
        let registry = replica.workers.objects.registry();
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

        replica.store.upsert(grpc_worker(KEY, &first));
        replica.caught_up("the registration").await;
        let labels = served().await;
        assert!(labels.values().any(|value| value == "first"), "{labels:?}");

        replica.store.upsert(grpc_worker(KEY, &second));
        replica.caught_up("the engine's move").await;
        let labels = served().await;
        assert!(labels.values().any(|value| value == "second"), "{labels:?}");

        replica.stopped().await;
    }
}
