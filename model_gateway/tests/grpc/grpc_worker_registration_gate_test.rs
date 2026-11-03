//! A gRPC worker registers only once its engine reports itself healthy. A
//! servicer binds its port before its engine has loaded the model and answers
//! the health RPC with `healthy: false` until then; a worker admitted on the
//! reachable port alone is routed to and fails every request until the
//! engine is up, with the readiness route saying ready meanwhile. The
//! startup detection keeps polling at its cadence instead, and the worker
//! registers within a poll of the engine's first healthy answer.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

use std::time::{Duration, Instant};

use common::grpc_harness::{grpc_router_config, grpc_test_context, serve_canned_worker, Canned};
use smg::workflow::Job;

use crate::common;

const MODEL: &str = "registration-gate-model";
/// The startup detection's cadence in the test.
const STARTUP_POLL: Duration = Duration::from_secs(1);
/// Several startup polls' worth of the engine reporting not healthy.
const NOT_HEALTHY_FOR: Duration = Duration::from_secs(4);
/// How long after the engine's first healthy answer the worker must be in
/// the registry: a poll, the detection steps, and headroom for a busy host.
const REGISTRATION_BOUND: Duration = Duration::from_secs(30);

#[tokio::test]
async fn a_grpc_worker_registers_once_its_engine_reports_healthy() {
    let port = serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: 3,
    })
    .await;
    let engine = mock_worker::overrides::for_port(port);
    engine.set_serving(false);

    let config = grpc_router_config()
        .regular_mode(vec![format!("grpc://127.0.0.1:{port}")])
        .random_policy()
        .worker_startup_timeout_secs(60)
        .worker_startup_check_interval_secs(STARTUP_POLL.as_secs())
        .build_unchecked();
    let app_context = grpc_test_context(config.clone(), MODEL).await;
    let registry = app_context.worker_registry.clone();
    app_context
        .worker_job_queue
        .get()
        .expect("the job queue")
        .submit(Job::InitializeWorkersFromConfig {
            router_config: Box::new(config),
        })
        .await
        .expect("submit the startup registration");

    // Poll after poll finds the engine not healthy: nothing registers.
    let started = Instant::now();
    while started.elapsed() < NOT_HEALTHY_FOR {
        assert_eq!(
            registry.get_all().len(),
            0,
            "a worker registered while its engine reported not healthy"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // The engine's first healthy answer lets the worker in.
    engine.set_serving(true);
    let healthy_since = Instant::now();
    while !registry.get_all().iter().any(|worker| worker.is_healthy()) {
        assert!(
            healthy_since.elapsed() < REGISTRATION_BOUND,
            "the worker did not register within {REGISTRATION_BOUND:?} of its engine reporting healthy"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
