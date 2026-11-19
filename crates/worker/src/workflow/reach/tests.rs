//! The reach helpers' cases.

use std::{
    future::pending,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use super::*;
use crate::{
    test_support::{RefusedAddress, ScriptedAnswers, ScriptedConnector},
    workflow::host::testing::worker_client,
};

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[test]
fn http_health_url_accepts_http_https_and_bare_urls() {
    assert_eq!(
        http_health_url("http://localhost:30000").unwrap(),
        "http://localhost:30000/health"
    );
    assert_eq!(
        http_health_url("https://example.com/").unwrap(),
        "https://example.com/health"
    );
    assert_eq!(
        http_health_url("localhost:30000").unwrap(),
        "http://localhost:30000/health"
    );
}

#[test]
fn http_health_url_rejects_grpc_schemes() {
    assert!(http_health_url("grpc://localhost:30001").is_err());
    assert!(http_health_url("grpcs://localhost:30001").is_err());
}

/// An address registered with a query, at a port nobody listens on: the
/// failure does not repeat the query. The HTTP client's own error text
/// ends with the request URL, the query kept.
#[tokio::test]
async fn a_failed_http_probe_goes_without_the_clients_url() {
    let refused = RefusedAddress::reserve();
    let url = format!("{}?token=s3cret", refused.http_url());

    let err = try_http_reachable(&url, 1, &worker_client())
        .await
        .expect_err("nothing listens at the address");

    assert!(!err.contains("s3cret"), "{err}");
}

#[test]
fn grpc_reachable_url_accepts_grpc_grpcs_and_bare_urls() {
    assert_eq!(
        grpc_reachable_url("grpc://localhost:30001").unwrap(),
        "grpc://localhost:30001"
    );
    assert_eq!(
        grpc_reachable_url("grpcs://localhost:30001/").unwrap(),
        "grpcs://localhost:30001"
    );
    assert_eq!(
        grpc_reachable_url("localhost:30001").unwrap(),
        "grpc://localhost:30001"
    );
}

/// A scheme refusal names the address masked: a secret in the URL's
/// userinfo reaches no failure message through it (the detection step's
/// auto-detect arm runs both probes and appends their refusals to its
/// failure message).
#[test]
fn a_scheme_refusal_names_the_address_masked() {
    for refusal in [
        grpc_reachable_url("http://user:secret@worker:3000").unwrap_err(),
        grpc_reachable_url("https://user:secret@worker:3000/").unwrap_err(),
        http_health_url("grpc://user:secret@worker:3000").unwrap_err(),
        http_health_url("grpcs://user:secret@worker:3000/").unwrap_err(),
    ] {
        assert!(!refusal.contains("secret"), "{refusal}");
        assert!(refusal.contains("://***@worker:3000"), "{refusal}");
    }
    let unknown = grpc_reachable_url("ftp://user:secret@worker:3000").unwrap_err();
    assert!(!unknown.contains("secret"), "{unknown}");
    assert!(unknown.contains("'ftp'"), "{unknown}");
}

/// A scheme is read at the start of an address alone. An address with
/// none may hold a `://` further in, in a path or a query: both probes
/// dial it as they dial any address with no scheme. Read as a scheme,
/// what stood before that `://` was refused by name, a userinfo in it,
/// in the detection step's failure message, which a log line carries.
/// A word of scheme characters is a scheme, known or not, and one the
/// probes do not know is still refused by name.
#[test]
fn a_scheme_is_read_at_the_start_of_a_probed_address_alone() {
    for url in [
        "user:secret@10.0.0.11:8000/a://b",
        "10.0.0.11:8000/a://b",
        "10.0.0.11:8000/proxy/http://backend:9000",
        "decode.internal:8000/v1?next=http://x",
    ] {
        assert_eq!(
            http_health_url(url).as_deref(),
            Ok(format!("http://{url}/health").as_str()),
            "{url}"
        );
        assert_eq!(
            grpc_reachable_url(url).as_deref(),
            Ok(format!("grpc://{url}").as_str()),
            "{url}"
        );
    }

    let unknown = http_health_url("X-tls+v1.2://user:secret@worker:3000").unwrap_err();
    assert!(!unknown.contains("secret"), "{unknown}");
    assert!(unknown.contains("'x-tls+v1.2'"), "{unknown}");
}

#[test]
fn grpc_reachable_url_rejects_http_schemes() {
    assert!(grpc_reachable_url("http://localhost:30000").is_err());
    assert!(grpc_reachable_url("https://localhost:30000").is_err());
}

#[tokio::test]
async fn runtime_fanout_returns_on_first_success_and_cancels_stalled_checks() {
    let stalled_started = Arc::new(AtomicBool::new(false));
    let stalled_cancelled = Arc::new(AtomicBool::new(false));
    let checks = FuturesUnordered::new();

    for (index, succeeds) in [false, true].into_iter().enumerate() {
        let stalled_started = Arc::clone(&stalled_started);
        let stalled_cancelled = Arc::clone(&stalled_cancelled);
        checks.push(async move {
            let result = if succeeds {
                tokio::task::yield_now().await;
                Ok(())
            } else {
                stalled_started.store(true, Ordering::SeqCst);
                let _drop_flag = DropFlag(stalled_cancelled);
                pending::<Result<(), String>>().await
            };
            (index, result)
        });
    }

    let result = tokio::time::timeout(
        Duration::from_secs(1),
        first_success_or_all_errors(checks, &GRPC_RUNTIMES[..2]),
    )
    .await;

    assert!(matches!(result, Ok(Ok(()))));
    assert!(stalled_started.load(Ordering::SeqCst));
    assert!(stalled_cancelled.load(Ordering::SeqCst));
}

#[test]
fn the_not_healthy_error_carries_the_engines_reason_when_it_gives_one() {
    assert_eq!(
        not_healthy_error("Engine is starting"),
        "gRPC health check reports the engine not healthy: Engine is starting"
    );
    assert_eq!(
        not_healthy_error("  "),
        "gRPC health check reports the engine not healthy"
    );
}

/// An endpoint whose dial fails is asked no health call. The reach ends at
/// the dial with its reason, so an unreachable worker costs one dial and no
/// connect a runtime.
#[tokio::test]
async fn a_dial_that_fails_asks_no_connect() {
    let connector = ScriptedConnector::default().refusing_the_dial("nothing listens there");

    let err = try_grpc_reachable(&connector, "grpc://worker-a:9000", 1)
        .await
        .expect_err("the dial fails");

    assert_eq!(err, "gRPC connection failed: nothing listens there");
    assert_eq!(connector.connects(), 0);
}

/// An engine that answers its health call but reports itself not healthy
/// (its model still loading, say) is not reachable yet. The single check
/// fails with the engine's reason, the fan-out names the runtime before that
/// reason, and both pass once the engine reports healthy.
#[tokio::test]
async fn an_engine_that_reports_not_healthy_is_not_reachable_yet() {
    let (runtime, another) = (GRPC_RUNTIMES[4], GRPC_RUNTIMES[0]);
    let engine = Arc::new(ScriptedAnswers::answering_health(Ok(HealthAnswer {
        healthy: false,
        message: "not serving".to_string(),
    })));
    let connector = ScriptedConnector::default().with_engine(runtime, Arc::clone(&engine));
    let url = "grpc://worker-a:9000";

    let answer = grpc_health_answer(&connector, url, 5, runtime)
        .await
        .expect("the runtime answers its health call whether or not the engine is healthy");
    assert!(!answer.healthy);
    assert_eq!(answer.message, "not serving");
    grpc_health_answer(&connector, url, 5, another)
        .await
        .expect_err("another runtime's service does not answer there");
    let err = do_grpc_health_check(&connector, url, 5, runtime)
        .await
        .expect_err("an engine reporting not healthy must not pass the probe");
    assert_eq!(
        err,
        "gRPC health check reports the engine not healthy: not serving"
    );
    let err = try_grpc_reachable(&connector, url, 5)
        .await
        .expect_err("the fan-out must not count a not-healthy answer as reachable");
    assert!(
        err.contains(&format!(
            "{runtime}=gRPC health check reports the engine not healthy: not serving"
        )),
        "the fan-out error does not carry the engine's reason: {err}"
    );

    engine.set_health(Ok(HealthAnswer {
        healthy: true,
        message: String::new(),
    }));
    do_grpc_health_check(&connector, url, 5, runtime)
        .await
        .expect("a healthy engine passes the probe");
    try_grpc_reachable(&connector, url, 5)
        .await
        .expect("a healthy engine is reachable");
}
