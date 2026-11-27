use axum::http::StatusCode;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use smg_worker::identity::basic::{ConnectionMode, WorkerType};

use super::*;
use crate::routers::{
    common::retry::{is_retryable_response, is_retryable_status},
    error::extract_error_code_from_response,
    grpc::backend_client::BasicWorkerBuilder,
};

fn worker(url: &str, model_id: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model_id))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    )
}

#[test]
fn unready_unvetoed_worker_does_not_block_the_ready_pool_verdict() {
    let dead = worker("http://127.0.0.1:9850", "m");
    let a = worker("http://127.0.0.1:9851", "m");
    let b = worker("http://127.0.0.1:9852", "m");
    let c = worker("http://127.0.0.1:9853", "m");
    dead.set_status(openai_protocol::worker::WorkerStatus::NotReady);
    assert!(!dead.is_overloaded());
    assert!(dead.stall_reason().is_none());
    for w in [&a, &b, &c] {
        w.set_overloaded(true);
        w.increment_load();
    }
    a.increment_load();
    c.increment_load();
    let pool = vec![dead, Arc::clone(&a), Arc::clone(&b), c];
    assert!(all_vetoed(&pool));
    assert!(all_overloaded(&pool));
    let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false)
        .expect("the least-loaded healthy overloaded worker is still routable");
    assert!(Arc::ptr_eq(&picked, &b));
    let shed = shed_if_all_overloaded(&pool, "m", true).expect("explicit overload shed");
    assert_eq!(shed.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        extract_error_code_from_response(&shed),
        "worker_overload_protection_shed"
    );
    assert!(fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, true).is_none());

    a.set_overloaded(false);
    assert!(!all_vetoed(&pool));
    assert!(!all_overloaded(&pool));
    assert!(fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none());
    assert!(shed_if_all_overloaded(&pool, "m", true).is_none());
}

#[test]
fn an_unready_pool_is_neither_a_fallback_nor_an_overload_shed() {
    let a = worker("http://127.0.0.1:9861", "m");
    let b = worker("http://127.0.0.1:9862", "m");
    let pool = vec![a, b];
    for w in &pool {
        w.set_status(openai_protocol::worker::WorkerStatus::NotReady);
        w.set_overloaded(true);
    }
    assert!(!all_vetoed(&pool));
    assert!(!all_overloaded(&pool));
    assert!(fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none());
    assert!(shed_if_all_overloaded(&pool, "m", true).is_none());
}

#[test]
fn fallback_respects_open_and_half_open_circuit_execution() {
    use smg_worker::health::circuit_breaker::{CircuitBreakerConfig, CircuitState};

    let w: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:9871")
            .model(ModelCard::new("m"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 1,
                timeout_duration: std::time::Duration::from_secs(3600),
                ..Default::default()
            })
            .build(),
    );
    w.set_overloaded(true);
    w.record_circuit_breaker_outcome(false);
    let pool = vec![Arc::clone(&w)];
    assert_eq!(w.circuit_breaker_state(), CircuitState::Open);
    assert!(!all_vetoed(&pool));
    assert!(!all_overloaded(&pool));
    assert!(fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none());
    assert!(shed_if_all_overloaded(&pool, "m", true).is_none());

    assert!(w.readmit_circuit_breaker());
    assert_eq!(w.circuit_breaker_state(), CircuitState::HalfOpen);
    assert!(w.circuit_breaker_can_execute());
    assert!(all_vetoed(&pool));
    let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false)
        .expect("half-open uses the existing circuit execution permission");
    assert!(Arc::ptr_eq(&picked, &w));
    assert_eq!(w.circuit_breaker_state(), CircuitState::HalfOpen);
    assert!(fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, true).is_none());
    assert!(shed_if_all_overloaded(&pool, "m", true).is_some());
    w.record_circuit_breaker_outcome(false);
    assert_eq!(w.circuit_breaker_state(), CircuitState::Open);
    assert!(fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none());
}

/// The shed is a distinct 503, taken from the candidate pool rather than
/// the model index.
#[test]
fn all_overloaded_sheds_with_distinct_503_code() {
    let a = worker("http://127.0.0.1:9801", "m");
    let b = worker("http://127.0.0.1:9802", "m");
    let pool = vec![Arc::clone(&a), Arc::clone(&b)];

    a.set_overloaded(true);
    assert!(
        shed_if_all_overloaded(&pool, "m", true).is_none(),
        "one eligible worker left is not a shed"
    );

    b.set_overloaded(true);
    assert!(
        shed_if_all_overloaded(&pool, "m", false).is_none(),
        "without shedding an all-overloaded pool is steered, not refused"
    );
    let response = shed_if_all_overloaded(&pool, "m", true).expect("shed");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        extract_error_code_from_response(&response),
        "worker_overload_protection_shed"
    );
    assert_eq!(
        response
            .headers()
            .get(error::HEADER_X_SMG_ERROR_CODE)
            .expect("gateway error code header"),
        "worker_overload_protection_shed"
    );
}

/// A shed must be terminal for the retry layer, or "shed immediately"
/// becomes six pipeline runs and ~400 ms of backoff per request.
#[test]
fn shed_responses_are_not_retried() {
    let a = worker("http://127.0.0.1:9811", "m");
    a.set_overloaded(true);

    let selection = shed_if_all_overloaded(std::slice::from_ref(&a), "m", true).expect("shed");
    assert!(
        is_retryable_status(selection.status()),
        "the status stays the retryable 503 clients already understand"
    );
    assert!(
        !is_retryable_response(&selection),
        "but the retry layer must decline it"
    );

    let dispatch = shed_if_worker_overloaded(a.as_ref(), "m", true).expect("shed");
    assert!(!is_retryable_response(&dispatch));
}

/// Both shed kinds advertise the poll interval as Retry-After — the veto
/// cannot clear faster — while staying terminal for the retry layer.
#[test]
fn shed_responses_carry_retry_after() {
    let a = worker("http://127.0.0.1:9821", "m");
    a.set_overloaded(true);

    let selection = shed_if_all_overloaded(std::slice::from_ref(&a), "m", true).expect("shed");
    let dispatch = shed_if_worker_overloaded(a.as_ref(), "m", true).expect("shed");
    for response in [&selection, &dispatch] {
        let value = response
            .headers()
            .get(RETRY_AFTER)
            .expect("Retry-After present")
            .to_str()
            .expect("ascii");
        assert!(
            value.parse::<u64>().is_ok_and(|secs| secs >= 1),
            "Retry-After must be whole seconds >= 1, got {value}"
        );
        assert!(!is_retryable_response(response));
    }
}

/// An empty pool is a 404/unavailable question for the caller, not a shed
/// and not a fallback.
#[test]
fn empty_pool_is_not_a_shed() {
    assert!(shed_if_all_overloaded(&[], "nobody", true).is_none());
    assert!(fallback_if_all_vetoed(&[], "nobody", STAGE_SELECTION, false).is_none());
    assert!(!all_overloaded(&[]));
}

/// The steering default: an all-overloaded pool routes to its least-loaded
/// routable worker; a pool with an eligible worker left is not touched,
/// and a worker that is also unhealthy is never the fallback.
#[test]
fn equally_loaded_workers_share_the_fallback_and_a_lighter_one_wins() {
    use rand::{rngs::StdRng, SeedableRng};
    let pool: Vec<Arc<dyn Worker>> = (0..128)
        .map(|i| worker(&format!("http://127.0.0.1:{}", 20000 + i), "m"))
        .collect();
    let mut rng = StdRng::seed_from_u64(7);
    let mut hits = vec![0usize; pool.len()];
    for _ in 0..1000 {
        let picked = least_loaded_uniform(pool.iter(), &mut rng).expect("a worker");
        hits[pool.iter().position(|w| Arc::ptr_eq(w, picked)).unwrap()] += 1;
    }
    assert!(hits.iter().all(|&h| h >= 1), "{hits:?}");
    assert!(*hits.iter().max().unwrap() <= 24, "{hits:?}");
    for (i, w) in pool.iter().enumerate() {
        if i != 5 {
            w.increment_load();
        }
    }
    for _ in 0..100 {
        let picked = least_loaded_uniform(pool.iter(), &mut rng).expect("a worker");
        assert!(Arc::ptr_eq(picked, &pool[5]));
    }
}

#[test]
fn all_overloaded_falls_back_to_the_least_loaded_routable_worker() {
    let a = worker("http://127.0.0.1:9831", "m");
    let b = worker("http://127.0.0.1:9832", "m");
    let c = worker("http://127.0.0.1:9833", "m");
    let pool = vec![Arc::clone(&a), Arc::clone(&b), Arc::clone(&c)];
    for _ in 0..3 {
        a.increment_load();
    }
    b.increment_load();

    a.set_overloaded(true);
    b.set_overloaded(true);
    assert!(
        fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none(),
        "an eligible worker left means selection handles it"
    );

    c.set_overloaded(true);
    let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).expect("fallback");
    assert_eq!(
        picked.url(),
        "http://127.0.0.1:9833",
        "the least-loaded wins"
    );

    c.set_status(openai_protocol::worker::WorkerStatus::NotReady);
    let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).expect("fallback");
    assert_eq!(
        picked.url(),
        "http://127.0.0.1:9832",
        "an unhealthy worker is not routable even as the fallback"
    );

    a.set_status(openai_protocol::worker::WorkerStatus::NotReady);
    b.set_status(openai_protocol::worker::WorkerStatus::NotReady);
    assert!(
        fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none(),
        "nothing routable: the caller's unavailable answer stands"
    );
}

/// A liveness veto steers but never refuses: when every ready worker is
/// vetoed by the tracker the request goes to the least-loaded of them,
/// after any worker vetoed by overload alone, and under shedding never
/// to an overloaded one.
#[test]
fn all_stalled_falls_back_to_the_least_loaded_ready_worker() {
    use smg_worker::identity::basic::StallReason;

    let a = worker("http://127.0.0.1:9841", "m");
    let b = worker("http://127.0.0.1:9842", "m");
    let pool = vec![Arc::clone(&a), Arc::clone(&b)];
    for _ in 0..2 {
        a.increment_load();
    }

    assert!(a.set_stall(Some(StallReason::Wedged)));
    assert!(
        fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none(),
        "a worker free of both vetoes means selection handles it"
    );

    assert!(b.set_stall(Some(StallReason::Unreachable)));
    let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).expect("fallback");
    assert_eq!(
        picked.url(),
        "http://127.0.0.1:9842",
        "the least-loaded of the vetoed is taken"
    );
    let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, true).expect("under shedding");
    assert_eq!(
        picked.url(),
        "http://127.0.0.1:9842",
        "a liveness veto steers under shedding too"
    );

    assert!(b.set_stall(None));
    b.set_overloaded(true);
    let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).expect("fallback");
    assert_eq!(
        picked.url(),
        "http://127.0.0.1:9842",
        "overload alone outranks a liveness veto"
    );
    let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, true).expect("under shedding");
    assert_eq!(
        picked.url(),
        "http://127.0.0.1:9841",
        "under shedding the overloaded worker is never taken"
    );

    a.set_overloaded(true);
    assert!(
        fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, true).is_none(),
        "all overloaded under shedding: the caller sheds"
    );

    b.set_status(openai_protocol::worker::WorkerStatus::NotReady);
    a.set_overloaded(false);
    let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).expect("fallback");
    assert_eq!(
        picked.url(),
        "http://127.0.0.1:9841",
        "an unhealthy worker is never the fallback; the stalled ready one is"
    );

    a.set_status(openai_protocol::worker::WorkerStatus::NotReady);
    assert!(
        fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none(),
        "nothing ready: the caller's unavailable answer stands"
    );
}

#[test]
fn dispatch_recheck_sheds_only_for_a_flagged_worker() {
    let w = worker("http://127.0.0.1:9803", "m");
    assert!(shed_if_worker_overloaded(w.as_ref(), "m", true).is_none());

    w.set_overloaded(true);
    let response = shed_if_worker_overloaded(w.as_ref(), "m", true).expect("shed");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        extract_error_code_from_response(&response),
        "worker_overload_protection_shed"
    );
    assert_eq!(
        response
            .headers()
            .get(error::HEADER_X_SMG_ERROR_CODE)
            .expect("gateway error code header"),
        "worker_overload_protection_shed"
    );
}
