//! An HTTP worker's load through the load source the host's connector keeps.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use smg_worker::{
    connection::engine::{EngineConnection, EngineConnector, LoadSource, ZmqLink},
    identity::basic::{ConnectionMode, Worker},
    jobs::WorkerJobAsker,
    load::monitor::poll_group_once_for_test,
    worker::{
        NamedValue, RuntimeType, SchedulerLoadSnapshot, WorkerGroupKey, WorkerLoadResponse,
        WorkerSpec,
    },
    workflow::{CreateLocalWorkerStep, WorkerKind},
};
use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

use super::super::{build_app_context, Extensions, ServerConfig};
use crate::{
    app_context::AppContext,
    config::RouterConfig,
    routers::grpc::backend_client::BackendConnector,
    workflow::{
        data::WorkerRegistrationMode,
        steps::{create_worker_workflow_data, RegisterWorkersStep},
    },
};

/// An address with no listener: a load request sent there fails.
const HTTP_URL: &str = "http://worker.invalid:53178";

fn queue_age() -> NamedValue {
    NamedValue {
        name: "queue_age".into(),
        unit: "ms".into(),
        value: 6.5,
        age_ms: Some(37),
    }
}

/// The host's own service: the load of the one worker it knows.
struct HostLoads;

#[async_trait]
impl LoadSource for HostLoads {
    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        let rank = SchedulerLoadSnapshot {
            num_running_reqs: 41,
            num_waiting_reqs: 23,
            values: vec![queue_age()],
            ..Default::default()
        };
        Ok(WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![rank],
            ..Default::default()
        })
    }
}

/// The host keeps the load of one HTTP address and delegates every engine
/// to the stock connector.
#[derive(Debug, Default)]
struct HostConnector {
    asked: AtomicUsize,
}

#[async_trait]
impl EngineConnector for HostConnector {
    async fn connect_grpc(
        &self,
        url: &str,
        runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        BackendConnector::shared().connect_grpc(url, runtime).await
    }

    async fn connect_zmq(&self, link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        BackendConnector::shared().connect_zmq(link).await
    }

    async fn dial_grpc(&self, url: &str, timeout: Duration) -> Result<(), String> {
        BackendConnector::shared().dial_grpc(url, timeout).await
    }

    async fn http_load_source(
        &self,
        base_url: &str,
        _runtime: RuntimeType,
    ) -> Option<Arc<dyn LoadSource>> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        (base_url == HTTP_URL).then(|| Arc::new(HostLoads) as Arc<dyn LoadSource>)
    }
}

/// An HTTP worker the gateway's own steps build and register.
async fn register_http(app: &Arc<AppContext>) -> Arc<dyn Worker> {
    let mut spec = WorkerSpec::new(HTTP_URL);
    spec.health.disable_health_check = Some(true);
    let data = create_worker_workflow_data(
        spec,
        WorkerRegistrationMode::Upsert,
        Arc::clone(app),
        WorkerJobAsker::Api,
    );
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    context.data.worker_kind = Some(WorkerKind::Local);
    context.data.connection_mode = Some(ConnectionMode::Http);
    context.data.config.runtime_type = RuntimeType::Generic;
    CreateLocalWorkerStep.execute(&mut context).await.unwrap();
    RegisterWorkersStep.execute(&mut context).await.unwrap();
    app.worker_registry.get_by_url(HTTP_URL).unwrap()
}

/// The load port publishes an HTTP worker's report from the host's load
/// source, named values and all. The worker's address has no listener, so
/// a load request sent to it would have published nothing. The connector
/// is asked for the source once.
#[tokio::test]
async fn the_load_port_publishes_an_http_workers_report_from_the_hosts_source() {
    let connector = Arc::new(HostConnector::default());
    let config = ServerConfig::minimal(RouterConfig::default());
    let extensions =
        Extensions::new().connector(Arc::clone(&connector) as Arc<dyn EngineConnector>);
    let app = build_app_context(&config, extensions).await.unwrap();
    let worker = register_http(&app).await;
    assert_eq!(*worker.connection_mode(), ConnectionMode::Http);
    let monitor = app.worker_monitor.as_ref().unwrap();
    let mut reader = monitor.subscribe();
    let key = WorkerGroupKey {
        model_id: worker.model_id().into(),
        worker_type: *worker.worker_type(),
        connection_mode: *worker.connection_mode(),
    };

    assert!(reader.borrow_and_update().get(worker.url()).is_none());
    for _ in 0..2 {
        poll_group_once_for_test(monitor, &key, Duration::from_secs(11), Instant::now()).await;
    }
    assert!(reader.has_changed().unwrap());
    let snapshot = reader.borrow_and_update();
    let report = snapshot.get(worker.url()).unwrap();
    assert_eq!(report.loads.len(), 1);
    assert_eq!(report.loads[0].num_running_reqs, 41);
    assert_eq!(report.loads[0].num_waiting_reqs, 23);
    assert_eq!(report.loads[0].values, [queue_age()]);
    assert_eq!(connector.asked.load(Ordering::Relaxed), 1);
}
