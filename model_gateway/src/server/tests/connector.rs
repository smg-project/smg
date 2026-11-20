//! Worker connections through the context built at startup.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use smg_policy_controller::{
    store::{HealthPolicy, LoadSink, Maker, WorkerFacts},
    view::{Role, WireModes},
    worker_host::{Expect, Membership, Wrote},
};
use smg_worker::{
    common_proto::KvEventBatch,
    connection::engine::{AdminAnswer, EngineConnection, EngineConnector, HealthAnswer, ZmqLink},
    identity::{
        basic::{ConnectionMode, Worker},
        error::WorkerError,
    },
    jobs::WorkerJobAsker,
    load::monitor::poll_group_once_for_test,
    registry::RemoteWorkerState,
    test_support::NoEngine,
    worker::{
        NamedValue, ProfileOptions, RuntimeType, SchedulerLoadSnapshot, WorkerGroupKey,
        WorkerLoadResponse, WorkerSpec, WorkerUpdateRequest,
    },
    workflow::{
        CreateLocalWorkerStep, DetectConnectionModeStep, DetectRuntimeStep, DiscoverMetadataStep,
        WorkerKind, WorkflowHost,
    },
};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowInstanceId};

use super::super::{build_app_context, Extensions, ServerConfig};
use crate::{
    app_context::AppContext,
    config::RouterConfig,
    policy_controller::{
        feed::signal_tests::{stages, SINGLE, SLOTS},
        host::Ask,
    },
    routers::grpc::backend_client::{get_backend_client, BackendConnector, BasicWorkerBuilder},
    workflow::{
        data::{WorkerRegistrationMode, WorkerUpdateWorkflowData, WorkerWorkflowData},
        steps::{
            create_worker_update_workflow_data, create_worker_workflow_data,
            local::RediscoverLabelsStep, util::mock_grpc_engine, CreateExternalWorkersStep,
            FindWorkerToUpdateStep, RegisterWorkersStep, UpdateWorkerPropertiesStep,
        },
    },
};

const HOST_URL: &str = "grpc://worker.invalid:53177";
const MODEL_ID: &str = "sample-model";

/// A host's connection to one engine. Its load answer states the
/// engine's free slots by name where the host knows the count.
#[derive(Debug, Default)]
struct HostConnection {
    free_slots: Option<f64>,
}

#[async_trait]
impl EngineConnection for HostConnection {
    fn carries_admin_calls(&self) -> bool {
        false
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, tonic::Status> {
        Ok(HealthAnswer {
            healthy: true,
            message: String::new(),
        })
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        Ok(HashMap::from([(
            "served_model_name".into(),
            MODEL_ID.into(),
        )]))
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        Ok(HashMap::new())
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        let stated = self.free_slots.map(|free| NamedValue {
            name: SLOTS.into(),
            unit: "slots".into(),
            value: free,
            age_ms: None,
        });
        let rank = SchedulerLoadSnapshot {
            num_running_reqs: 37,
            num_waiting_reqs: 29,
            values: stated.into_iter().collect(),
            ..Default::default()
        };
        Ok(WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![rank],
            ..Default::default()
        })
    }

    async fn flush_cache(&self, timeout_s: f32) -> Result<AdminAnswer, tonic::Status> {
        NoEngine.flush_cache(timeout_s).await
    }

    async fn start_profile(&self, options: &ProfileOptions) -> Result<AdminAnswer, tonic::Status> {
        NoEngine.start_profile(options).await
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status> {
        NoEngine.stop_profile().await
    }

    async fn subscribe_kv_events(
        &self,
        start_seq: u64,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        NoEngine.subscribe_kv_events(start_seq).await
    }

    async fn subscribe_load_records(
        &self,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        NoEngine.subscribe_load_records().await
    }
}

/// The host owns one address with no listener and delegates other addresses.
#[derive(Debug)]
struct HostConnector;

#[async_trait]
impl EngineConnector for HostConnector {
    async fn connect_grpc(
        &self,
        url: &str,
        runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        if url == HOST_URL {
            return Ok(Arc::new(HostConnection::default()));
        }
        BackendConnector::shared().connect_grpc(url, runtime).await
    }

    async fn connect_zmq(&self, link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        BackendConnector::shared().connect_zmq(link).await
    }

    async fn dial_grpc(&self, url: &str, timeout: Duration) -> Result<(), String> {
        if url == HOST_URL {
            return Ok(());
        }
        BackendConnector::shared().dial_grpc(url, timeout).await
    }
}

/// The host's engines that state their free slots, each at its address.
const STATING: [(&str, f64); 3] = [
    ("grpc://first.invalid:53177", 10.0),
    ("grpc://second.invalid:53177", 70.0),
    ("grpc://third.invalid:53177", 40.0),
];

/// The host owns the [`STATING`] addresses, with no listener at any.
#[derive(Debug)]
struct StatingConnector;

#[async_trait]
impl EngineConnector for StatingConnector {
    async fn connect_grpc(
        &self,
        url: &str,
        _: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        let engine = STATING.iter().find(|(address, _)| *address == url);
        let &(_, free) = engine.ok_or_else(|| format!("the host has no engine at {url}"))?;
        Ok(Arc::new(HostConnection {
            free_slots: Some(free),
        }))
    }

    async fn connect_zmq(&self, link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        BackendConnector::shared().connect_zmq(link).await
    }

    async fn dial_grpc(&self, _: &str, _: Duration) -> Result<(), String> {
        Ok(())
    }
}

async fn gateway(extensions: Extensions) -> Arc<AppContext> {
    let config = ServerConfig::minimal(RouterConfig::default());
    build_app_context(&config, extensions).await.unwrap()
}

fn registration(app: &Arc<AppContext>, url: &str) -> WorkflowContext<WorkerWorkflowData> {
    let mut spec = WorkerSpec::new(url);
    spec.health.disable_health_check = Some(true);
    let data = create_worker_workflow_data(
        spec,
        WorkerRegistrationMode::Upsert,
        Arc::clone(app),
        WorkerJobAsker::Api,
    );
    WorkflowContext::new(WorkflowInstanceId::new(), data)
}

async fn register_grpc(app: &Arc<AppContext>, url: &str) -> Arc<dyn Worker> {
    let mut context = registration(app, url);
    context.data.worker_kind = Some(WorkerKind::Local);
    DetectConnectionModeStep
        .execute(&mut context)
        .await
        .unwrap();
    DetectRuntimeStep.execute(&mut context).await.unwrap();
    DiscoverMetadataStep.execute(&mut context).await.unwrap();
    CreateLocalWorkerStep.execute(&mut context).await.unwrap();
    RegisterWorkersStep.execute(&mut context).await.unwrap();
    app.worker_registry.get_by_url(url).unwrap()
}

async fn build_local(app: &Arc<AppContext>) -> Arc<dyn Worker> {
    let mut context = registration(app, HOST_URL);
    context.data.worker_kind = Some(WorkerKind::Local);
    context.data.connection_mode = Some(ConnectionMode::Grpc);
    context.data.config.runtime_type = RuntimeType::Generic;
    CreateLocalWorkerStep.execute(&mut context).await.unwrap();
    context.data.actual_workers.unwrap().pop().unwrap()
}

async fn build_external(app: &Arc<AppContext>) -> Arc<dyn Worker> {
    let mut context = registration(app, "https://external.invalid");
    context.data.worker_kind = Some(WorkerKind::External);
    CreateExternalWorkersStep
        .execute(&mut context)
        .await
        .unwrap();
    context.data.actual_workers.unwrap().pop().unwrap()
}

fn import_worker(app: &AppContext) -> Arc<dyn Worker> {
    let mut spec = WorkerSpec::new(HOST_URL);
    spec.connection_mode = ConnectionMode::Grpc;
    spec.runtime_type = RuntimeType::Generic;
    app.worker_registry
        .on_remote_worker_state(&RemoteWorkerState {
            worker_id: "peer-worker".into(),
            model_id: MODEL_ID.into(),
            url: HOST_URL.into(),
            health: true,
            version: 1,
            spec: serde_json::to_vec(&spec).unwrap(),
        });
    app.worker_registry.get_by_url(HOST_URL).unwrap()
}

fn update_context(app: &Arc<AppContext>) -> WorkflowContext<WorkerUpdateWorkflowData> {
    let request = WorkerUpdateRequest {
        priority: None,
        cost: None,
        labels: None,
        api_key: None,
        health: None,
    };
    let data = create_worker_update_workflow_data(
        HOST_URL.into(),
        request,
        Arc::clone(app),
        WorkerJobAsker::Api,
    );
    WorkflowContext::new(WorkflowInstanceId::new(), data)
}

async fn assert_host_connection(worker: &dyn Worker) {
    let connection = worker.engine_connection().await.unwrap().unwrap();
    assert!(connection.downcast_arc::<HostConnection>().is_some());
}

#[tokio::test]
async fn registration_probes_and_connects_through_the_host() {
    let connector: Arc<dyn EngineConnector> = Arc::new(HostConnector);
    let app = gateway(Extensions::new().connector(Arc::clone(&connector))).await;
    assert!(Arc::ptr_eq(&app.engine_connector(), &connector));
    let worker = register_grpc(&app, HOST_URL).await;
    assert_eq!(worker.model_id(), MODEL_ID);
    assert_host_connection(worker.as_ref()).await;
}

#[tokio::test]
async fn an_imported_worker_connects_through_the_host() {
    let app = gateway(Extensions::new().connector(Arc::new(HostConnector))).await;
    let worker = import_worker(&app);
    assert_host_connection(worker.as_ref()).await;
}

#[tokio::test]
async fn workers_built_or_rebuilt_by_the_gateway_carry_the_host_connector() {
    let connector: Arc<dyn EngineConnector> = Arc::new(HostConnector);
    let app = gateway(Extensions::new().connector(Arc::clone(&connector))).await;
    let local = build_local(&app).await;
    let external = build_external(&app).await;
    assert_host_connection(local.as_ref()).await;
    assert!(Arc::ptr_eq(
        &external.engine_connector().unwrap(),
        &connector
    ));

    app.worker_registry.register(Arc::clone(&local)).unwrap();
    let mut context = update_context(&app);
    FindWorkerToUpdateStep.execute(&mut context).await.unwrap();
    UpdateWorkerPropertiesStep
        .execute(&mut context)
        .await
        .unwrap();
    let rebuilt = app.worker_registry.get_by_url(HOST_URL).unwrap();
    assert!(!Arc::ptr_eq(&local, &rebuilt));
    assert!(Arc::ptr_eq(
        &rebuilt.engine_connector().unwrap(),
        &connector
    ));
    assert_host_connection(rebuilt.as_ref()).await;
    assert!(matches!(
        get_backend_client(rebuilt.as_ref()).await,
        Err(WorkerError::ConnectionFailed { reason, .. })
            if reason == "its engine connection is not the gateway's backend client"
    ));
}

#[tokio::test]
async fn label_rediscovery_reads_through_the_gateway_context() {
    let app = gateway(Extensions::new().connector(Arc::new(HostConnector))).await;
    // The worker holds the stock connector, so the labels below come through
    // the context's connector or not at all.
    let worker = BasicWorkerBuilder::new(HOST_URL)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::Generic)
        .engine_connector(BackendConnector::shared())
        .build();
    app.worker_registry.register(Arc::new(worker)).unwrap();
    let mut context = update_context(&app);
    context.data.refresh_discovered_labels = true;
    FindWorkerToUpdateStep.execute(&mut context).await.unwrap();
    assert_eq!(
        RediscoverLabelsStep.execute(&mut context).await.unwrap(),
        StepResult::Success
    );
    let labels = context.data.fresh_engine_labels.unwrap();
    assert_eq!(
        labels.get("served_model_name").map(String::as_str),
        Some(MODEL_ID)
    );
}

#[tokio::test]
async fn stock_startup_uses_the_stock_connector_on_each_path() {
    let app = gateway(Extensions::default()).await;
    let stock = BackendConnector::shared();
    assert!(Arc::ptr_eq(&app.engine_connector(), &stock));
    for worker in [
        build_local(&app).await,
        build_external(&app).await,
        import_worker(&app),
    ] {
        assert!(Arc::ptr_eq(&worker.engine_connector().unwrap(), &stock));
    }
}

#[tokio::test]
async fn the_host_can_delegate_a_stock_engine_to_the_stock_connector() {
    let (url, _) = mock_grpc_engine().await;
    let connector: Arc<dyn EngineConnector> = Arc::new(HostConnector);
    let app = gateway(Extensions::new().connector(Arc::clone(&connector))).await;
    let worker = register_grpc(&app, &url).await;
    assert!(Arc::ptr_eq(&worker.engine_connector().unwrap(), &connector));
    let client = get_backend_client(worker.as_ref()).await.unwrap().unwrap();
    assert!(EngineConnection::health_check(client.as_ref())
        .await
        .unwrap());
}

#[tokio::test]
async fn the_load_port_publishes_the_host_connections_report() {
    let app = gateway(Extensions::new().connector(Arc::new(HostConnector))).await;
    let worker = register_grpc(&app, HOST_URL).await;
    let monitor = app.worker_monitor.as_ref().unwrap();
    let mut reader = monitor.subscribe();
    let key = WorkerGroupKey {
        model_id: worker.model_id().into(),
        worker_type: *worker.worker_type(),
        connection_mode: *worker.connection_mode(),
    };

    assert!(reader.borrow_and_update().get(worker.url()).is_none());
    poll_group_once_for_test(monitor, &key, Duration::from_secs(11), Instant::now()).await;
    assert!(reader.has_changed().unwrap());
    let snapshot = reader.borrow_and_update();
    let report = snapshot.get(worker.url()).unwrap();
    assert_eq!(report.loads.len(), 1);
    assert_eq!(report.loads[0].num_running_reqs, 37);
    assert_eq!(report.loads[0].num_waiting_reqs, 29);
}

/// Under a document that declares a signal, the values a host's connections
/// state in their polled load answers decide a placement. The gateway's
/// context is built with the host's connector and the host's stage. The
/// load monitor polls each engine through its connection, the feed's sink
/// hands the store the named values, and the host places by them.
#[tokio::test]
async fn a_placement_follows_the_values_a_host_connections_polled_answer_states() {
    let router = RouterConfig {
        policy_controller: true,
        policy_spec: Some(serde_yaml::from_str(SINGLE).unwrap()),
        ..RouterConfig::default()
    };
    let extensions = Extensions::new()
        .connector(Arc::new(StatingConnector))
        .stages(stages().unwrap());
    let app = build_app_context(&ServerConfig::minimal(router), extensions)
        .await
        .unwrap();
    let door: &dyn Membership = &*app.policy_feed.as_ref().unwrap().door;
    for (url, _) in STATING {
        let facts = WorkerFacts {
            key: url.to_owned(),
            engine: url.to_owned(),
            role: Role::Single,
            wire_modes: WireModes::GRPC,
            engine_kind: "generic".to_owned(),
            models: vec![MODEL_ID.to_owned()],
            dp_size: 1,
            max_running: Some(64),
            health: HealthPolicy {
                probes: false,
                ..HealthPolicy::default()
            },
            ..WorkerFacts::default()
        };
        let wrote = door.found(&Maker::Control, Expect::Nothing, url, &|_| facts.clone());
        assert!(matches!(wrote, Ok(Wrote::Registered(_))), "{wrote:?}");
    }
    let (second, _) = STATING[1];
    let worker = app.worker_registry.get_by_url(second).unwrap();
    assert_host_connection(worker.as_ref()).await;
    let key = WorkerGroupKey {
        model_id: worker.model_id().into(),
        worker_type: *worker.worker_type(),
        connection_mode: *worker.connection_mode(),
    };
    assert_eq!(key.connection_mode, ConnectionMode::Grpc);

    let monitor = app.worker_monitor.as_ref().unwrap();
    let now = Instant::now();
    poll_group_once_for_test(monitor, &key, Duration::from_secs(11), now).await;
    let store = app.policy_store.as_ref().unwrap();
    // The case runs no publication loop: it shows what waits at its instant.
    if let Some(due) = store.publish_due() {
        store.publish(due);
    }
    // One value an engine was taken, and none dropped.
    assert_eq!(store.counts().signal_writes, [3, 0, 0, 0, 0, 0]);

    let host = app.policy_host.as_ref().unwrap();
    let ask = Ask::of(b"polled", MODEL_ID);
    let placed = host.place(&ask, now + Duration::from_millis(10)).await;
    assert_eq!(placed.unwrap().worker().url(), second);
}
