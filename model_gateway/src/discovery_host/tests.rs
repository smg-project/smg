use std::{
    net::{SocketAddr, TcpStream},
    pin::pin,
    sync::Arc,
    task::Poll,
    time::Duration,
};

use openai_protocol::worker::{RuntimeType, WorkerSpec, WorkerType};
use smg_worker::{
    identity::basic::{ConnectionMode, DEFAULT_BOOTSTRAP_PORT},
    jobs::WorkerRegistrationMode,
};
use tokio::time;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowInstanceId};

use crate::{
    app_context::testing::{app_context_for_tests, create_test_app_context},
    config::RouterConfig,
    routers::common::snapshot::WorkerRegistry,
    server::http_client,
    service_discovery::DiscoveryHost,
    workflow::{
        data::WorkerKind,
        steps::{create_worker_workflow_data, CreateLocalWorkerStep, RegisterWorkersStep},
        Job, JobQueue, JobQueueConfig,
    },
};

/// The context stamps a discovered worker with the router's credential and
/// with the discovered budget of connection attempts: the default success
/// threshold of 2, twenty attempts per success.
#[test]
fn the_context_stamps_its_credential_and_the_discovered_budget() {
    let router_config = RouterConfig::builder()
        .api_key("router-key")
        .build_unchecked();
    let context = app_context_for_tests(router_config, Arc::new(WorkerRegistry::new()), None);
    let mut spec = WorkerSpec::new("http://10.0.0.1:8080");
    context.stamp(&mut spec);

    assert_eq!(spec.api_key.as_deref(), Some("router-key"));
    assert_eq!(
        spec.max_connection_attempts,
        context
            .router_config
            .connection_attempt_budget(RouterConfig::DISCOVERED_WORKER_ATTEMPTS_PER_SUCCESS)
    );
    assert_eq!(spec.max_connection_attempts, 40);
}

/// An addition a pass hands the queue holds its URL's claim while it waits
/// for a queue slot: the address reads as in flight, and a `POST` of the URL
/// during the wait is told a registration is in flight. A pass used to read
/// the URL's status first and submit without a claim, so such a `POST` was
/// admitted and the pass then queued a second registration of the URL beside
/// it.
#[tokio::test]
async fn an_addition_waiting_for_a_queue_slot_holds_its_urls_claim() {
    let context = create_test_app_context();
    assert!(context.jobs().is_none(), "no queue stands yet");
    // One slot and no runner: the dispatcher takes one job out and waits for
    // a permit that never comes, and the next job fills the slot.
    let queue = JobQueue::new(
        JobQueueConfig {
            queue_capacity: 1,
            max_concurrent_jobs: 0,
        },
        Arc::downgrade(&context),
    );
    context.worker_job_queue.set(Arc::clone(&queue)).ok();
    let registration = |url: &str, registration_mode| Job::AddWorker {
        config: Box::new(WorkerSpec::new(url)),
        registration_mode,
    };
    for other in 1.. {
        tokio::task::yield_now().await;
        if queue.get_load_info().0 == 1 {
            break;
        }
        let url = format!("http://10.0.0.{other}:8080");
        queue
            .submit(registration(&url, WorkerRegistrationMode::Upsert))
            .await
            .expect("a slot");
    }

    let jobs = context.jobs().expect("the queue stands");
    let url = "http://10.0.9.1:8080";
    assert!(!jobs.job_in_flight(url), "no job of the address yet");
    let mut addition = pin!(jobs.add_worker(WorkerSpec::new(url), WorkerRegistrationMode::Upsert));
    assert!(
        matches!(futures::poll!(addition.as_mut()), Poll::Pending),
        "the addition waits for a queue slot"
    );

    assert!(jobs.job_in_flight(url), "the waiting addition is in flight");
    assert!(
        !queue.claim_if_idle(&registration(url, WorkerRegistrationMode::CreateOnly)),
        "the waiting addition holds its URL: a POST of it is told to poll"
    );
}

/// A prefill worker discovered without a bootstrap annotation carries no
/// bootstrap port into its spec and is handed the default port on the
/// request path (here a vLLM worker on the Mooncake connector, as the
/// provider reports it). Registered, it is probed on that default port:
/// with nothing listening there, the failed probe is counted like a
/// registered port's.
#[tokio::test]
async fn a_discovered_prefill_worker_without_a_bootstrap_annotation_is_probed_on_the_default_port()
{
    // A listener on the default port would make a passed probe and an
    // absent one look alike, so the test stands down then.
    let default_port = SocketAddr::from(([127, 0, 0, 1], DEFAULT_BOOTSTRAP_PORT));
    if TcpStream::connect_timeout(&default_port, Duration::from_millis(500)).is_ok() {
        return;
    }

    let context = create_test_app_context();
    // What a pass hands the queue for such a worker: no bootstrap port, the
    // KV settings the provider reported, and the context's stamp.
    let mut spec = WorkerSpec::new("127.0.0.1:18081");
    spec.worker_type = WorkerType::Prefill;
    spec.kv_connector = Some("MooncakeConnector".to_string());
    spec.kv_role = Some("kv_producer".to_string());
    spec.kv_engine_id = Some("engine-1".to_string());
    context.stamp(&mut spec);
    assert_eq!(spec.bootstrap_port, None, "no annotation, no port");
    // The runtime the worker's engine reports when the registration detects it.
    spec.runtime_type = RuntimeType::Vllm;

    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mut data =
        create_worker_workflow_data(spec, WorkerRegistrationMode::Upsert, Arc::clone(&context));
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(ConnectionMode::Http);
    data.http_client_handle = Some(Arc::new(http_client()));
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);
    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    assert_eq!(
        RegisterWorkersStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let registered = &ctx.data.actual_workers.as_ref().unwrap()[0];
    assert_eq!(
        registered.bootstrap_port(),
        None,
        "the spec keeps what was discovered"
    );

    // The probe runs off the path, within its own timeout.
    let deadline = time::Instant::now() + Duration::from_secs(5);
    let counted = loop {
        let rendered = handle.render();
        if let Some(line) = rendered
            .lines()
            .find(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
        {
            break line.to_string();
        }
        assert!(
            time::Instant::now() < deadline,
            "the failed probe of the default port is counted within its timeout:\n{rendered}"
        );
        time::sleep(Duration::from_millis(20)).await;
    };
    assert!(
        counted.contains(r#"leg="prefill""#)
            && counted.contains(r#"phase="probe""#)
            && counted.ends_with(" 1"),
        "{counted}"
    );
}
