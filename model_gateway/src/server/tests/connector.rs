//! Worker connections through the context built at startup.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use mock_worker::config::SchedulerWire;
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
        ProfileOptions, RuntimeType, SchedulerLoadSnapshot, WorkerGroupKey, WorkerLoadResponse,
        WorkerSpec, WorkerUpdateRequest,
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
    config::{PolicyConfig, RouterConfig},
    routers::{
        common::{
            placement::{select_single, PlacementInputs, Stated},
            snapshot::RoutingPool,
        },
        grpc::backend_client::{get_backend_client, BackendConnector, BasicWorkerBuilder, Fronted},
    },
    workflow::{
        data::{WorkerRegistrationMode, WorkerUpdateWorkflowData, WorkerWorkflowData},
        steps::{
            create_worker_update_workflow_data, create_worker_workflow_data,
            local::RediscoverLabelsStep,
            util::{mock_grpc_engine, mock_grpc_engine_stating},
            CreateExternalWorkersStep, FindWorkerToUpdateStep, RegisterWorkersStep,
            UpdateWorkerPropertiesStep,
        },
    },
};

const HOST_URL: &str = "grpc://worker.invalid:53177";
const MODEL_ID: &str = "sample-model";

/// A host's connection to one engine.
#[derive(Debug)]
struct HostConnection;

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
        let rank = SchedulerLoadSnapshot {
            num_running_reqs: 37,
            num_waiting_reqs: 29,
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
            return Ok(Arc::new(HostConnection));
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
    register_grpc_from(app, url, registration(app, url)).await
}

/// [`register_grpc`] over `context`, a registration of `url` its case set
/// up.
async fn register_grpc_from(
    app: &Arc<AppContext>,
    url: &str,
    mut context: WorkflowContext<WorkerWorkflowData>,
) -> Arc<dyn Worker> {
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

/// The worker's connection is the gateway's front, and the host's own
/// stands in it.
async fn assert_host_connection(worker: &dyn Worker) {
    let connection = worker.engine_connection().await.unwrap().unwrap();
    let fronted = connection.downcast_arc::<Fronted>().unwrap();
    let front = Arc::clone(fronted.front());
    assert!(front.downcast_arc::<HostConnection>().is_some());
}

#[tokio::test]
async fn registration_probes_and_connects_through_the_host() {
    let app = gateway(Extensions::new().connector(Arc::new(HostConnector))).await;
    // The context's connector is the gateway's wrap of the host's.
    let wrap = format!("{:?}", app.engine_connector());
    assert_eq!(wrap, "Fronting(HostConnector)");
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
    let app = gateway(Extensions::new().connector(Arc::new(HostConnector))).await;
    let connector = app.engine_connector();
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
    // The gateway's client has no wire for this worker's runtime: the
    // request path refuses it by name, and dials nothing.
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
    let app = gateway(Extensions::new().connector(Arc::new(HostConnector))).await;
    let worker = register_grpc(&app, &url).await;
    let connector = app.engine_connector();
    assert!(Arc::ptr_eq(&worker.engine_connector().unwrap(), &connector));
    // The gateway's client the host delegated to passes the wrap as it is.
    let connection = worker.engine_connection().await.unwrap().unwrap();
    assert!(connection.downcast_arc::<Fronted>().is_none());
    let client = get_backend_client(worker.as_ref()).await.unwrap().unwrap();
    assert!(EngineConnection::health_check(client.as_ref())
        .await
        .unwrap());
}

/// A host's own connection that holds the gateway's client: it answers the
/// load call itself and hands every other call to that client.
struct HostLoads(Arc<dyn EngineConnection>);

#[async_trait]
impl EngineConnection for HostLoads {
    fn carries_admin_calls(&self) -> bool {
        self.0.carries_admin_calls()
    }

    fn is_alive(&self) -> bool {
        self.0.is_alive()
    }

    async fn health_answer(&self) -> Result<HealthAnswer, tonic::Status> {
        self.0.health_answer().await
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        self.0.model_labels().await
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        self.0.server_labels().await
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        HostConnection.get_loads().await
    }

    async fn flush_cache(&self, timeout_s: f32) -> Result<AdminAnswer, tonic::Status> {
        self.0.flush_cache(timeout_s).await
    }

    async fn start_profile(&self, options: &ProfileOptions) -> Result<AdminAnswer, tonic::Status> {
        self.0.start_profile(options).await
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status> {
        self.0.stop_profile().await
    }

    async fn subscribe_kv_events(
        &self,
        start_seq: u64,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        self.0.subscribe_kv_events(start_seq).await
    }

    async fn subscribe_load_records(
        &self,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        self.0.subscribe_load_records().await
    }
}

/// A host whose connector returns its own connection for each gRPC engine
/// and knows nothing of a front.
#[derive(Debug)]
struct HostLoadsConnector;

#[async_trait]
impl EngineConnector for HostLoadsConnector {
    async fn connect_grpc(
        &self,
        url: &str,
        runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        let stock = BackendConnector::shared()
            .connect_grpc(url, runtime)
            .await?;
        Ok(Arc::new(HostLoads(stock)))
    }

    async fn connect_zmq(&self, link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        BackendConnector::shared().connect_zmq(link).await
    }

    async fn dial_grpc(&self, url: &str, timeout: Duration) -> Result<(), String> {
        BackendConnector::shared().dial_grpc(url, timeout).await
    }
}

/// A gRPC worker on a host's own connection, with the host's connector as
/// it is: the worker registers through it, is polled through the host's
/// connection and is sent to through the gateway's client behind the
/// gateway's front. The load port publishes the host's report, and the
/// engine answers through the client the request path reads.
#[tokio::test]
async fn a_worker_on_a_hosts_connection_is_polled_through_it_and_sent_to_through_the_gateways_client(
) {
    let (url, _) = mock_grpc_engine().await;
    let app = gateway(Extensions::new().connector(Arc::new(HostLoadsConnector))).await;
    // The engine listens and the registration's probes dial it: they get
    // time enough on a loaded host.
    let mut registering = registration(&app, &url);
    registering.data.config.health.timeout_secs = Some(30);
    let worker = register_grpc_from(&app, &url, registering).await;
    let connection = worker.engine_connection().await.unwrap().unwrap();
    let fronted = connection.downcast_arc::<Fronted>().unwrap();
    let front = Arc::clone(fronted.front());
    assert!(front.downcast_arc::<HostLoads>().is_some());

    let client = get_backend_client(worker.as_ref()).await.unwrap().unwrap();
    assert!(EngineConnection::health_check(client.as_ref())
        .await
        .unwrap());

    let monitor = app.worker_monitor.as_ref().unwrap();
    let mut reader = monitor.subscribe();
    let key = WorkerGroupKey {
        model_id: worker.model_id().into(),
        worker_type: *worker.worker_type(),
        connection_mode: *worker.connection_mode(),
    };
    poll_group_once_for_test(monitor, &key, Duration::from_secs(11), Instant::now()).await;
    let snapshot = reader.borrow_and_update();
    let report = snapshot.get(worker.url()).unwrap();
    assert_eq!(report.loads.len(), 1);
    assert_eq!(report.loads[0].num_running_reqs, 37);
    assert_eq!(report.loads[0].num_waiting_reqs, 29);
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

/// One poll of `worker`'s group by the gateway's load monitor, each of the
/// group's engines through its own connection.
async fn poll_once(app: &AppContext, worker: &dyn Worker) {
    let key = WorkerGroupKey {
        model_id: worker.model_id().into(),
        worker_type: *worker.worker_type(),
        connection_mode: *worker.connection_mode(),
    };
    assert_eq!(key.connection_mode, ConnectionMode::Grpc);
    let monitor = app.worker_monitor.as_ref().unwrap();
    poll_group_once_for_test(monitor, &key, Duration::from_secs(11), Instant::now()).await;
}

/// On each stock gRPC wire an engine states a value by name in its answer
/// to a load poll. The gateway's own client reads the list and the load
/// monitor publishes it: the feed the gateway's policies read holds each
/// engine's own value under its name.
#[tokio::test]
async fn the_load_feed_holds_the_value_a_stock_grpc_polls_answer_states_by_name() {
    const NAME: &str = "free_slots";
    const UNIT: &str = "slots";
    for wire in [
        SchedulerWire::TokenSpeed,
        SchedulerWire::Vllm,
        SchedulerWire::Sglang,
    ] {
        let router = RouterConfig {
            policy: PolicyConfig::RoundRobin,
            ..RouterConfig::default()
        };
        let app = build_app_context(&ServerConfig::minimal(router), Extensions::new())
            .await
            .unwrap();
        let mut engines = Vec::new();
        for free in [10.0, 40.0, 70.0] {
            let url = mock_grpc_engine_stating(wire, NAME, UNIT, free).await;
            // The engine listens and the registration's probes dial it:
            // they get time enough on a loaded host.
            let mut registering = registration(&app, &url);
            registering.data.config.health.timeout_secs = Some(30);
            let worker = register_grpc_from(&app, &url, registering).await;
            engines.push((worker, free));
        }
        let (first, _) = &engines[0];
        poll_once(&app, first.as_ref()).await;

        let feed = app.worker_monitor.as_ref().unwrap().subscribe();
        let published = feed.borrow();
        for (worker, free) in &engines {
            let report = published.get(worker.url()).unwrap();
            let stated: Vec<_> = report
                .loads
                .iter()
                .flat_map(|rank| &rank.values)
                .map(|value| (&*value.name, &*value.unit, value.value))
                .collect();
            assert_eq!(stated, [(NAME, UNIT, *free)], "{wire:?}");
        }
    }
}

/// A worker learned from a peer registers through no workflow. The policy
/// registry builds a model's own policy at a local registration and keeps
/// it while a local worker serves the model, so a model that only peers'
/// workers serve has none: the placement reads the default policy for it.
#[tokio::test]
async fn a_model_only_a_peers_worker_serves_is_placed_by_the_default_policy() {
    let router = RouterConfig {
        policy: PolicyConfig::RoundRobin,
        ..RouterConfig::default()
    };
    let host = Extensions::new().connector(Arc::new(HostConnector));
    let app = build_app_context(&ServerConfig::minimal(router), host)
        .await
        .unwrap();
    let worker = import_worker(&app);
    // The model the peer's worker serves, as its specification names it.
    let model = worker.model_id().to_owned();
    assert!(!app.worker_registry.serves_model_locally(&model));

    let policies = &app.policy_registry;
    assert!(policies.get_policy(&model).is_none());
    let read = policies.get_policy_or_default(&model);
    assert!(Arc::ptr_eq(&read, &policies.get_default_policy()));
    let placed = select_single(
        &app.worker_registry,
        policies,
        &model,
        RoutingPool::GrpcPipelineRegular,
        None,
        PlacementInputs::default(),
        Stated::nothing,
    )
    .unwrap();
    assert!(Arc::ptr_eq(placed.worker(), &worker));
}
