mod max_concurrent_requests;

use std::sync::atomic::{AtomicUsize, Ordering};

use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

use super::*;
use crate::{
    config::types::PolicyConfig,
    worker::{BasicWorkerBuilder, PrefillAdmissionAttempt, Worker, WorkerType},
};

/// Loopback echo server; axum::serve accepts HTTP/1.1 and prior-knowledge
/// h2c on the same listener, mirroring a dual-protocol engine.
async fn spawn_echo_server() -> String {
    let app = axum::Router::new().route("/probe", axum::routing::get(|| async { "ok" }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind echo server");
    let addr = listener.local_addr().expect("echo server address");
    #[expect(
        clippy::disallowed_methods,
        reason = "test server lives for the duration of the test process"
    )]
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("echo serve");
    });
    format!("http://{addr}/probe")
}

/// Child-process body of the root-store test below (see
/// `worker::http_client::no_root_store`); a no-op unless the case
/// variable is set.
#[test]
fn child_with_client_without_native_roots() {
    use crate::worker::http_client::no_root_store;

    if std::env::var(no_root_store::CASE).is_err() {
        return;
    }
    AppContextBuilder::new()
        .with_client(&RouterConfig::default(), 5)
        .expect("a gateway without TLS configuration starts without a root store");
}

#[test]
#[cfg_attr(
    any(target_vendor = "apple", windows, target_os = "android"),
    ignore = "the native verifier does not use SSL_CERT_FILE/SSL_CERT_DIR"
)]
fn the_gateway_client_builds_without_a_native_root_store() {
    use crate::worker::http_client::no_root_store;

    no_root_store::run(
        &no_root_store::test_name(module_path!(), "child_with_client_without_native_roots"),
        "plaintext",
    );
}

#[test]
fn https_worker_urls_make_tls_required() {
    use crate::config::RoutingMode;

    let with_worker = |url: &str| RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec![url.to_string()],
        },
        ..RouterConfig::default()
    };
    assert!(!has_https_worker(&with_worker("http://worker:8000")));
    assert!(has_https_worker(&with_worker("https://worker:8443")));
}

fn built_client(upstream_http2: bool) -> Client {
    let config = RouterConfig {
        upstream_http2,
        ..RouterConfig::default()
    };
    AppContextBuilder::new()
        .with_client(&config, 5)
        .expect("client builds")
        .client
        .expect("client set")
}

/// `--upstream-http2` is a worker-client concern; the shared client keeps
/// negotiating normally (HTTP/1.1 on cleartext, ALPN on TLS).
#[tokio::test]
async fn shared_client_ignores_upstream_http2() {
    let url = spawn_echo_server().await;
    let resp = built_client(true)
        .get(&url)
        .send()
        .await
        .expect("h1 request");
    assert_eq!(resp.version(), http::Version::HTTP_11);
    assert_eq!(resp.text().await.expect("body"), "ok");
}

#[tokio::test]
async fn default_client_stays_http1() {
    let url = spawn_echo_server().await;
    let resp = built_client(false)
        .get(&url)
        .send()
        .await
        .expect("h1 request");
    assert_eq!(resp.version(), http::Version::HTTP_11);
    assert_eq!(resp.text().await.expect("body"), "ok");
}

/// The admission bucket of a ten-request cap under
/// `rate_limit_tokens_per_second` hands out its ten tokens and refills
/// none over 150 ms; a returned token is handed out again.
async fn ten_tokens_and_no_refill(rate_limit_tokens_per_second: Option<i32>) {
    let config = RouterConfig {
        max_concurrent_requests: 10,
        rate_limit_tokens_per_second,
        ..RouterConfig::default()
    };
    let bucket = AppContextBuilder::new()
        .maybe_rate_limiter(&config)
        .rate_limiter
        .expect("rate limiter should be enabled");

    assert!(bucket.try_acquire(10.0).is_ok());
    // An earlier fallback refilled at max_concurrent_requests per second,
    // which would restore a token during this wait.
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(bucket.try_acquire(1.0).is_err());

    bucket.return_tokens_sync(1.0);
    assert!(bucket.try_acquire(1.0).is_ok());
}

#[tokio::test]
async fn unset_rate_limit_defaults_to_no_refill() {
    ten_tokens_and_no_refill(None).await;
}

/// With no explicit cap the admission bucket still exists, sized by the
/// host-derived default: the request past the bound is rejected instead
/// of queued without limit, and a completion frees a slot again.
#[test]
fn unset_concurrency_cap_gets_a_host_derived_bound() {
    let bucket = AppContextBuilder::new()
        .maybe_rate_limiter(&RouterConfig::default())
        .rate_limiter
        .expect("the unset cap must still bound in-flight requests");
    let bound = default_max_concurrent_requests();
    assert!(bound >= DEFAULT_INFLIGHT_FLOOR);

    for admitted in 0..bound {
        assert!(
            bucket.try_acquire(1.0).is_ok(),
            "request {admitted} is within the bound of {bound}"
        );
    }
    assert!(
        bucket.try_acquire(1.0).is_err(),
        "the request past the bound must be rejected, not admitted"
    );

    bucket.return_tokens_sync(1.0);
    assert!(bucket.try_acquire(1.0).is_ok());
}

#[test]
fn zero_concurrency_cap_disables_admission_control() {
    let config = RouterConfig {
        max_concurrent_requests: 0,
        ..RouterConfig::default()
    };
    assert!(AppContextBuilder::new()
        .maybe_rate_limiter(&config)
        .rate_limiter
        .is_none());
}

#[tokio::test]
async fn explicit_zero_rate_limit_disables_refill() {
    ten_tokens_and_no_refill(Some(0)).await;
}

fn config_with_policy(policy: PolicyConfig) -> RouterConfig {
    RouterConfig {
        policy,
        ..Default::default()
    }
}

/// Whether `with_kv_event_monitor` creates the monitor under `policy`.
/// This run of the builder needs no storage or network.
fn kv_monitor_created_for(policy: PolicyConfig) -> bool {
    let config = config_with_policy(policy);
    AppContextBuilder::new()
        .with_policy_registry(&config)
        .with_kv_event_monitor()
        .kv_event_monitor
        .is_some()
}

/// The load-snapshot feed must be wired whenever the worker monitor is
/// built — without a KV-event monitor in the chain — so HTTP-only
/// cache-aware deployments get waiting-prefill and KV-usage data.
#[test]
fn worker_monitor_wires_load_receiver_into_policies() {
    use crate::policies::CacheAwarePolicy;

    let config = config_with_policy(PolicyConfig::CacheAware {
        cache_threshold: 0.5,
        balance_abs_threshold: 32,
        balance_rel_threshold: 1.1,
        eviction_interval_secs: 0,
        max_tree_size: 1000,
        block_size: 16,
        balance_token_usage_threshold: 1.0,
        overload_token_usage_threshold: 1.0,
        overlap_decay: 1.0,
        selection_temperature: 0.0,
        cache_index: Default::default(),
        cache_ttl_secs: 180,
        cache_boundaries: Vec::new(),
        selection_policy: None,
        selection_accounting_ttl_ms: 0,
        balance_prefix_margin_tokens: 0,
        balance_uncached_credit: false,
    });
    let builder = AppContextBuilder::new()
        .with_client(&config, 5)
        .expect("client builds")
        .with_worker_registry()
        .with_policy_registry(&config)
        .with_worker_monitor(&config)
        .expect("worker monitor builds");

    let policy = builder
        .policy_registry
        .as_ref()
        .expect("policy registry set")
        .get_default_policy();
    let cache_aware = policy
        .as_any()
        .downcast_ref::<CacheAwarePolicy>()
        .expect("default policy is cache-aware");
    assert!(cache_aware.has_load_receiver_for_test());
}

/// Every policy gets the monitor: a cache-aware one for its indexes, any
/// other for the load stream of its gRPC workers, which the policies'
/// load feed reads in place of the poll.
#[test]
fn every_policy_creates_the_kv_event_monitor() {
    assert!(kv_monitor_created_for(PolicyConfig::Passthrough));
    assert!(kv_monitor_created_for(PolicyConfig::RoundRobin));
    assert!(kv_monitor_created_for(PolicyConfig::PowerOfTwo {
        load_metric: Default::default(),
    }));
    assert!(kv_monitor_created_for(PolicyConfig::cache_aware(
        0.5, 32, 1.1, 30, 1000
    )));
}

/// The monitor reaches a cache-aware policy created later from a
/// worker's policy label under a non-cache-aware default: its model's
/// workers get the event-driven index rather than the approximate tree.
#[test]
fn a_cache_aware_policy_from_a_worker_label_gets_the_kv_event_monitor() {
    use crate::policies::CacheAwarePolicy;

    let config = config_with_policy(PolicyConfig::Random);
    let builder = AppContextBuilder::new()
        .with_policy_registry(&config)
        .with_kv_event_monitor();
    assert!(builder.kv_event_monitor.is_some());
    let registry = builder
        .policy_registry
        .as_ref()
        .expect("policy registry set");
    let policy = registry.on_worker_added("affine", Some("cache_aware"));
    let cache_aware = policy
        .as_any()
        .downcast_ref::<CacheAwarePolicy>()
        .expect("the label's policy is cache-aware");
    assert!(cache_aware.kv_event_monitor_is_set_for_test());
}

#[test]
fn maybe_rate_limit_manager_disabled_is_ok_none() {
    let config = RouterConfig::default();
    let result = AppContextBuilder::new().maybe_rate_limit_manager(&config);
    assert!(result.is_ok());
    assert!(result.unwrap().rate_limit_manager.is_none());
}

#[test]
fn maybe_rate_limit_manager_enabled_with_missing_file_fails_startup() {
    let config = RouterConfig::builder()
        .tenant_rate_limit_enabled(true)
        .tenant_rate_limit_config(Some("/nonexistent/rate_limit.yaml".to_string()))
        .build_unchecked();
    assert!(AppContextBuilder::new()
        .maybe_rate_limit_manager(&config)
        .is_err());
}

#[tokio::test]
async fn maybe_rate_limit_manager_enabled_with_valid_policy_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rate_limit.yaml");
    std::fs::write(
        &path,
        "default_policy:\n  tokens_per_minute: 1000\n  requests_per_minute: 60\n",
    )
    .unwrap();
    let config = RouterConfig::builder()
        .tenant_rate_limit_enabled(true)
        .tenant_rate_limit_config(Some(path.to_str().unwrap().to_string()))
        .build_unchecked();
    let result = AppContextBuilder::new().maybe_rate_limit_manager(&config);
    assert!(result.is_ok());
    assert!(result.unwrap().rate_limit_manager.is_some());
}

#[test]
fn mcp_bootstrap_keeps_everything_but_the_server_list() {
    let file: smg_mcp::McpConfig = serde_yaml::from_str(
        r#"
servers:
  - name: "docs"
    protocol: sse
    url: "https://mcp.example.com/sse"
pool:
  max_connections: 7
proxy:
  https: "http://proxy.example:3128"
policy:
  default: deny
"#,
    )
    .unwrap();
    assert_eq!(file.servers.len(), 1);

    let config = mcp_bootstrap_config(Some(&file));
    assert!(
        config.servers.is_empty(),
        "servers register through the job queue"
    );
    assert_eq!(config.pool.max_connections, 7);
    assert_eq!(
        config
            .proxy
            .as_ref()
            .and_then(|proxy| proxy.https.as_deref()),
        Some("http://proxy.example:3128")
    );
    assert!(matches!(
        config.policy.default,
        smg_mcp::PolicyDecisionConfig::Deny
    ));

    assert!(mcp_bootstrap_config(None).servers.is_empty());
}

#[test]
fn mcp_bootstrap_reads_the_proxy_from_the_environment_when_the_file_has_none() {
    std::env::set_var("MCP_HTTPS_PROXY", "http://env-proxy.example:3128");
    let config = mcp_bootstrap_config(None);
    std::env::remove_var("MCP_HTTPS_PROXY");
    assert_eq!(
        config
            .proxy
            .as_ref()
            .and_then(|proxy| proxy.https.as_deref()),
        Some("http://env-proxy.example:3128")
    );
}

fn prefill_worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Prefill)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    )
}

#[tokio::test]
async fn worker_status_change_wakes_prefill_admission() {
    let registry = Arc::new(WorkerRegistry::new());
    let admission = Arc::new(PrefillAdmission::new(1, 1, Duration::from_secs(1)));
    start_prefill_admission_notifier(&registry, &admission).unwrap();

    let first = prefill_worker("http://prefill-1:8000");
    registry.register(Arc::clone(&first)).unwrap();
    let occupied = admission
        .admit(None, {
            let first = Arc::clone(&first);
            move |capacity| capacity.select(Arc::clone(&first), ())
        })
        .await
        .unwrap();

    let attempts = Arc::new(AtomicUsize::new(0));
    #[expect(
        clippy::disallowed_methods,
        reason = "test waiter must run concurrently with worker status changes"
    )]
    let waiting = tokio::spawn({
        let admission = Arc::clone(&admission);
        let registry = Arc::clone(&registry);
        let attempts = Arc::clone(&attempts);
        async move {
            admission
                .admit(None, |capacity| {
                    attempts.fetch_add(1, Ordering::Relaxed);
                    let workers = registry.get_by_type(WorkerType::Prefill);
                    if workers.is_empty() {
                        return PrefillAdmissionAttempt::Unavailable;
                    }
                    workers
                        .iter()
                        .find(|worker| worker.is_available() && capacity.has_capacity(worker))
                        .map_or(PrefillAdmissionAttempt::AtCapacity, |worker| {
                            capacity.select(Arc::clone(worker), Arc::clone(worker))
                        })
                })
                .await
        }
    });

    tokio::time::timeout(Duration::from_secs(1), async {
        while admission.queued_requests() != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    let second: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://prefill-2:8000")
            .worker_type(WorkerType::Prefill)
            .build(),
    );
    let second_id = registry.register(Arc::clone(&second)).unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while attempts.load(Ordering::Relaxed) < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(admission.queued_requests(), 1);

    registry.transition_status(&second_id, WorkerStatus::Ready);
    let selected = tokio::time::timeout(Duration::from_secs(1), waiting)
        .await
        .unwrap()
        .unwrap()
        .unwrap();

    assert_eq!(selected.selected.url(), second.url());
    drop(selected);
    drop(occupied);
}
