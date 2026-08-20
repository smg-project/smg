//! HTTP-level integration tests for the priority-admission scheduler.
//!
//! These complement the engine's in-process unit tests by driving the whole
//! axum stack (tenant resolution → `priority_admission_middleware` → handler →
//! `SchedulerGuardBody`) against mock workers, asserting the documented HTTP
//! contract:
//!
//! | Outcome              | Status | Extra                                   |
//! |----------------------|--------|-----------------------------------------|
//! | queue full           | 429    | `X-SMG-Error-Code: scheduler_queue_full`, `Retry-After: 2` |
//! | queue timeout        | 503    | `X-SMG-Error-Code: scheduler_queue_timeout`, `Retry-After: 2` |
//! | preempted (pre-TTFT) | 503    | `X-SMG-Preempted: true`, `Retry-After: 1` |
//!
//! ## How these are made deterministic
//!
//! * **Capacity.** The scheduler sizes itself from `WorkerCapacity`, which sums
//!   each healthy worker's reported `max_running_requests`. The mock worker now
//!   reports a test-chosen value, so a single worker reporting `1` pins backend
//!   capacity to exactly one slot — no reliance on the legacy
//!   `--max-concurrent-requests` fallback (which only applies when zero workers
//!   are healthy).
//! * **Reservations.** Every test ships a YAML config that sets `reserved_floor: 0`
//!   on *all four* classes. This matters: with the built-in defaults
//!   (Interactive 128 + System 32 = 160) `PriorityScheduler::new` would reject
//!   the tiny capacities here and `AdmissionMode::from_config` would silently
//!   fall back to the legacy limiter — so the test would no longer exercise the
//!   scheduler at all. Zeroing reservations keeps the priority path active.
//! * **Slot occupancy & the pre-TTFT window.** Instead of racing `sleep`s, the
//!   mock worker's `/generate` endpoint can be gated by a [`HoldGate`]: each
//!   request signals arrival (so the test can confirm it now occupies a slot)
//!   then parks until the test releases it. A parked request has emitted no
//!   byte, so it sits squarely in the pre-TTFT window a preemption victim must
//!   be in.

#[path = "common/mod.rs"]
mod common;

use std::{io::Write, sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use portpicker::pick_unused_port;
use serde_json::{json, Value};
use serial_test::serial;
use smg::config::RouterConfig;
use tempfile::NamedTempFile;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{
        set_scheduler_controls, HealthStatus, HoldGate, MockWorkerConfig, SchedulerControls,
        WorkerType,
    },
    AppTestContext,
};

// ── Per-class YAML knobs ────────────────────────────────────────────────────

/// One class's tunables for the generated YAML.
///
/// [`ClassYaml::base`] zeroes `reserved_floor` (see the module docs for why
/// every class must read 0 to keep small-capacity tests on the priority path);
/// the starvation test is the sole place that deliberately sets it non-zero.
#[derive(Clone, Copy)]
struct ClassYaml {
    queue_size: u32,
    queue_timeout_secs: u64,
    starvation_threshold_secs: u64,
    can_preempt: bool,
    reserved_floor: u16,
}

impl ClassYaml {
    /// Sensible non-reserving default: a roomy queue, a long timeout, default
    /// preemption off, and a high starvation threshold so it never fires
    /// unless a test opts in.
    const fn base() -> Self {
        Self {
            queue_size: 64,
            queue_timeout_secs: 30,
            starvation_threshold_secs: 3600,
            can_preempt: false,
            reserved_floor: 0,
        }
    }
}

/// Per-class YAML for all four classes. Defaults to non-reserving `base()`
/// everywhere; tests mutate the specific classes they care about.
#[derive(Clone, Copy)]
struct SchedulerYaml {
    bulk: ClassYaml,
    default: ClassYaml,
    interactive: ClassYaml,
    system: ClassYaml,
}

impl SchedulerYaml {
    const fn base() -> Self {
        Self {
            bulk: ClassYaml::base(),
            default: ClassYaml::base(),
            interactive: ClassYaml::base(),
            system: ClassYaml::base(),
        }
    }
}

/// Render the YAML and write it to a temp file. The returned [`NamedTempFile`]
/// must outlive `create_app()` (the scheduler reads the path once at build
/// time); dropping it removes the file.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn write_scheduler_yaml(cfg: &SchedulerYaml) -> NamedTempFile {
    fn class_block(name: &str, c: &ClassYaml) -> String {
        format!(
            "  {name}:\n    reserved_floor: {}\n    queue_size: {}\n    queue_timeout_secs: {}\n    \
             starvation_threshold_secs: {}\n    can_preempt: {}\n",
            c.reserved_floor,
            c.queue_size,
            c.queue_timeout_secs,
            c.starvation_threshold_secs,
            c.can_preempt,
        )
    }

    let mut yaml = String::from("classes:\n");
    yaml.push_str(&class_block("bulk", &cfg.bulk));
    yaml.push_str(&class_block("default", &cfg.default));
    yaml.push_str(&class_block("interactive", &cfg.interactive));
    yaml.push_str(&class_block("system", &cfg.system));

    let mut file = NamedTempFile::new().expect("create temp scheduler YAML");
    file.write_all(yaml.as_bytes())
        .expect("write scheduler YAML");
    file.flush().expect("flush scheduler YAML");
    file
}

// ── Config / worker builders ────────────────────────────────────────────────

/// Build a scheduler-enabled `RouterConfig` pointing at the given YAML path,
/// with `default_max_class` controlling the anonymous-tenant clamp.
fn scheduler_config(port: u16, default_max_class: &str, yaml_path: &str) -> RouterConfig {
    let mut config = RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(port)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        // Legacy fallback only; capacity actually comes from the worker's
        // reported max_running_requests below. Kept > 0 so the rate limiter /
        // legacy path stay sane if a future change re-routes through them.
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .priority_scheduler_enabled(true)
        .priority_scheduler_default_max_class(default_max_class)
        .priority_scheduler_config(Some(yaml_path.to_string()))
        .build_unchecked();
    config.health_check.disable_health_check = true;
    config
}

/// Build a mock worker that reports `capacity` as its `max_running_requests`
/// (so a single such worker pins scheduler capacity to `capacity`) and,
/// optionally, gates its `/generate` on `gate`.
///
/// A free port is picked up front so the scheduler controls can be registered
/// *before* the worker binds — the worker's handlers look them up by port.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn scheduler_worker(capacity: u16, gate: Option<&Arc<HoldGate>>) -> MockWorkerConfig {
    let port = pick_unused_port().expect("a free port for the mock worker");
    set_scheduler_controls(
        port,
        SchedulerControls {
            max_running_requests: Some(capacity),
            hold_gate: gate.map(Arc::clone),
        },
    );
    MockWorkerConfig {
        port,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }
}

// ── Request helpers ─────────────────────────────────────────────────────────

/// A minimal non-streaming `/generate` request, optionally tagged with a
/// priority header.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn generate_request(text: &str, priority: Option<&str>) -> Request<Body> {
    let payload = json!({ "text": text, "stream": false });
    let mut builder = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json");
    if let Some(p) = priority {
        builder = builder.header("x-smg-priority", p);
    }
    builder
        .body(Body::from(
            serde_json::to_string(&payload).expect("serialize body"),
        ))
        .expect("build request")
}

/// Read the `X-SMG-Error-Code` response header, if present.
fn error_code(resp: &axum::response::Response) -> Option<String> {
    resp.headers()
        .get("x-smg-error-code")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

/// Drain a response body to JSON (also releases the scheduler slot the body
/// was holding).
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn body_json(resp: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read response body");
    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
}

/// Drive a request to a response, bounded by a timeout.
///
/// The timeout is a *diagnostic guard*, not a race: every probe in these tests
/// is expected to return promptly (an immediate rejection, or a bounded queue
/// wait). The only way to exceed it is the pathological case where backend
/// capacity came out larger than the intended `1` — then the probe would be
/// admitted and park on the hold gate forever. Surfacing that as a clear
/// failure beats hanging the whole test binary.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn send(app: &axum::Router, req: Request<Body>) -> axum::response::Response {
    tokio::time::timeout(Duration::from_secs(8), app.clone().oneshot(req))
        .await
        .expect("request did not complete in time (scheduler capacity likely exceeded 1)")
        .expect("router service is infallible")
}

/// Wait for `n` handler arrivals, bounded by a timeout so a capacity/config
/// mismatch fails loudly instead of hanging.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn wait_arrivals(gate: &Arc<HoldGate>, n: u32) {
    tokio::time::timeout(Duration::from_secs(8), gate.wait_for_arrivals(n))
        .await
        .expect("expected handler arrivals never reached the worker");
}

/// Await a spawned request's `JoinHandle`, bounded by a timeout.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn join(
    handle: tokio::task::JoinHandle<axum::response::Response>,
) -> axum::response::Response {
    tokio::time::timeout(Duration::from_secs(8), handle)
        .await
        .expect("spawned request did not complete in time")
        .expect("spawned request task panicked")
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// Queue full → 429. Capacity 1, occupied by a held request; the Default
/// queue has size 0 so the next admission can't even enqueue.
#[expect(
    clippy::disallowed_methods,
    reason = "test infra: holds a request open in a spawned task"
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn queue_full_returns_429() {
    let mut yaml = SchedulerYaml::base();
    yaml.default.queue_size = 0; // nowhere to wait → immediate reject
    let yaml_file = write_scheduler_yaml(&yaml);
    let config = scheduler_config(3601, "default", yaml_file.path().to_str().unwrap());

    let gate = HoldGate::new();
    let ctx = AppTestContext::new_with_config(config, vec![scheduler_worker(1, Some(&gate))]).await;
    let app = ctx.create_app();

    // Occupy the single slot with a held (pre-TTFT) request.
    let held_app = app.clone();
    let held = tokio::spawn(async move {
        held_app
            .oneshot(generate_request("held", None))
            .await
            .unwrap()
    });
    wait_arrivals(&gate, 1).await; // slot is now occupied

    // Probe: capacity is full and the queue can't hold anyone → 429.
    let resp = send(&app, generate_request("probe", None)).await;
    assert_eq!(
        resp.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "queue full must map to 429"
    );
    assert_eq!(error_code(&resp).as_deref(), Some("scheduler_queue_full"));
    assert_eq!(
        resp.headers()
            .get("retry-after")
            .map(|v| v.to_str().unwrap()),
        Some("2"),
        "queue-full 429 must carry Retry-After: 2"
    );
    let body = body_json(resp).await;
    assert_eq!(body["error"]["code"], "scheduler_queue_full");

    // Release the held request and confirm it then succeeds.
    gate.release();
    let held_resp = join(held).await;
    assert_eq!(held_resp.status(), StatusCode::OK);
    let _ = body_json(held_resp).await;

    ctx.shutdown().await;
}

/// Queue timeout → 503 + Retry-After. Capacity 1, occupied by a held
/// request; the probe enqueues (queue_size 1) but the slot never frees, so
/// it ages out.
#[expect(
    clippy::disallowed_methods,
    reason = "test infra: holds a request open in a spawned task"
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn queue_timeout_returns_503() {
    let mut yaml = SchedulerYaml::base();
    yaml.default.queue_size = 4;
    yaml.default.queue_timeout_secs = 1; // short, but the mechanism — not a race
    let yaml_file = write_scheduler_yaml(&yaml);
    let config = scheduler_config(3602, "default", yaml_file.path().to_str().unwrap());

    let gate = HoldGate::new();
    let ctx = AppTestContext::new_with_config(config, vec![scheduler_worker(1, Some(&gate))]).await;
    let app = ctx.create_app();

    let held_app = app.clone();
    let held = tokio::spawn(async move {
        held_app
            .oneshot(generate_request("held", None))
            .await
            .unwrap()
    });
    wait_arrivals(&gate, 1).await;

    // Probe enqueues, waits ~1s, then times out. The held request keeps the
    // slot the whole time, so the outcome is forced — not timing-dependent.
    let resp = send(&app, generate_request("probe", None)).await;
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "queue timeout must map to 503"
    );
    assert_eq!(
        resp.headers()
            .get("retry-after")
            .map(|v| v.to_str().unwrap()),
        Some("2"),
        "queue-timeout 503 must carry Retry-After: 2"
    );
    assert_eq!(
        error_code(&resp).as_deref(),
        Some("scheduler_queue_timeout")
    );

    gate.release();
    let held_resp = join(held).await;
    assert_eq!(held_resp.status(), StatusCode::OK);
    let _ = body_json(held_resp).await;

    ctx.shutdown().await;
}

/// Preemption end-to-end. Capacity 1 held by a pre-TTFT **bulk** request; an
/// **interactive** admission preempts it. The bulk request returns 503 +
/// `X-SMG-Preempted: true`; the interactive request is then served (200).
#[expect(
    clippy::disallowed_methods,
    reason = "test infra: holds requests open in spawned tasks"
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn preemption_evicts_bulk_and_serves_high_priority() {
    let mut yaml = SchedulerYaml::base();
    // Interactive may preempt (also the built-in default); give the preemptor
    // a real queue + generous timeout so that if the victim's slot is reclaimed
    // via the enqueue fallback (rather than the 50ms preempt budget) it still
    // gets admitted once the victim's body drains.
    yaml.interactive.can_preempt = true;
    yaml.interactive.queue_size = 4;
    yaml.interactive.queue_timeout_secs = 30;
    yaml.bulk.queue_size = 4;
    yaml.bulk.queue_timeout_secs = 30;
    let yaml_file = write_scheduler_yaml(&yaml);
    // default_max_class = system so the interactive header is *not* clamped.
    let config = scheduler_config(3603, "system", yaml_file.path().to_str().unwrap());

    let gate = HoldGate::new();
    let ctx = AppTestContext::new_with_config(config, vec![scheduler_worker(1, Some(&gate))]).await;
    let app = ctx.create_app();

    // Bulk request occupies the only slot, parked pre-TTFT.
    let bulk_app = app.clone();
    let bulk = tokio::spawn(async move {
        bulk_app
            .oneshot(generate_request("bulk", Some("bulk")))
            .await
            .unwrap()
    });
    wait_arrivals(&gate, 1).await;

    // Interactive admission: preempts the bulk victim (fires its cancel).
    let vip_app = app.clone();
    let vip = tokio::spawn(async move {
        vip_app
            .oneshot(generate_request("vip", Some("interactive")))
            .await
            .unwrap()
    });

    // The victim's handler is unwound by PreemptionGuard → 503 + header.
    let bulk_resp = join(bulk).await;
    assert_eq!(
        bulk_resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "preempted request must map to 503"
    );
    assert_eq!(
        bulk_resp
            .headers()
            .get("x-smg-preempted")
            .and_then(|v| v.to_str().ok()),
        Some("true"),
        "preemption 503 must carry X-SMG-Preempted: true"
    );
    assert_eq!(
        bulk_resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("1"),
        "preemption 503 must carry Retry-After: 1"
    );
    // Draining the victim's 503 body drops its SchedulerGuardBody and frees the
    // slot, so the (now-enqueued) interactive admission can proceed.
    let _ = body_json(bulk_resp).await;

    // Once the interactive request reaches the worker it holds the slot too;
    // release the gate so it can produce its 200.
    wait_arrivals(&gate, 1).await;
    gate.release();

    let vip_resp = join(vip).await;
    assert_eq!(
        vip_resp.status(),
        StatusCode::OK,
        "the preempting interactive request must be served"
    );
    let _ = body_json(vip_resp).await;

    ctx.shutdown().await;
}

/// Tenant clamp. With `default_max_class = default`, an anonymous request that
/// asks for `system` is clamped down to Default — so it gains *no* preemption
/// power and cannot evict a pre-TTFT bulk request. Contrast with
/// [`preemption_evicts_bulk_and_serves_high_priority`], whose only material
/// difference is `default_max_class = system`.
#[expect(
    clippy::disallowed_methods,
    reason = "test infra: holds a request open in a spawned task"
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn tenant_clamp_strips_system_header_of_preemption_power() {
    let mut yaml = SchedulerYaml::base();
    // System *would* be allowed to preempt — but the clamp must demote the
    // request to Default before it ever gets here.
    yaml.system.can_preempt = true;
    yaml.interactive.can_preempt = true;
    yaml.default.queue_size = 0; // a non-preempting probe has nowhere to wait
    yaml.bulk.queue_size = 4;
    yaml.bulk.queue_timeout_secs = 30;
    let yaml_file = write_scheduler_yaml(&yaml);
    // Clamp ceiling is Default for the anonymous tenant.
    let config = scheduler_config(3604, "default", yaml_file.path().to_str().unwrap());

    let gate = HoldGate::new();
    let ctx = AppTestContext::new_with_config(config, vec![scheduler_worker(1, Some(&gate))]).await;
    let app = ctx.create_app();

    // Bulk request occupies the slot, parked pre-TTFT.
    let bulk_app = app.clone();
    let bulk = tokio::spawn(async move {
        bulk_app
            .oneshot(generate_request("bulk", Some("bulk")))
            .await
            .unwrap()
    });
    wait_arrivals(&gate, 1).await;

    // Probe asks for `system` but is clamped to Default → cannot preempt →
    // hits a zero-size queue → 429 (and the bulk victim is untouched).
    let resp = send(&app, generate_request("probe", Some("system"))).await;
    assert_eq!(
        resp.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "clamped system→default request must not preempt; it should be queue-rejected"
    );
    assert_eq!(error_code(&resp).as_deref(), Some("scheduler_queue_full"));
    let _ = body_json(resp).await;

    // The bulk request must still be in flight (never preempted).
    assert!(
        !bulk.is_finished(),
        "the bulk request must not have been preempted by a clamped request"
    );

    // Clean up: release and confirm bulk completes normally (200, no preempt).
    gate.release();
    let bulk_resp = join(bulk).await;
    assert_eq!(
        bulk_resp.status(),
        StatusCode::OK,
        "bulk completes normally once released"
    );
    assert!(
        bulk_resp.headers().get("x-smg-preempted").is_none(),
        "bulk was never preempted, so it must not carry the preemption header"
    );
    let _ = body_json(bulk_resp).await;

    ctx.shutdown().await;
}

/// Slot release. Capacity 1: a first request completes and its body is fully
/// drained (freeing the slot); a second request then succeeds via the fast
/// path. If the slot leaked, the second request would block/queue.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn slot_frees_after_response_drains() {
    let mut yaml = SchedulerYaml::base();
    // A tiny queue + short timeout means a *leaked* slot would surface as a
    // fast 503 rather than hanging the test.
    yaml.default.queue_size = 1;
    yaml.default.queue_timeout_secs = 2;
    let yaml_file = write_scheduler_yaml(&yaml);
    let config = scheduler_config(3605, "default", yaml_file.path().to_str().unwrap());

    // No gate: requests complete immediately.
    let ctx = AppTestContext::new_with_config(config, vec![scheduler_worker(1, None)]).await;
    let app = ctx.create_app();

    // First request: succeed and fully drain the body (releases the slot).
    let first = send(&app, generate_request("first", None)).await;
    assert_eq!(first.status(), StatusCode::OK);
    let _ = body_json(first).await; // drain → slot released

    // Second request must now be admitted via the freed slot.
    let second = send(&app, generate_request("second", None)).await;
    assert_eq!(
        second.status(),
        StatusCode::OK,
        "slot must be reusable after the prior response drained"
    );
    let _ = body_json(second).await;

    ctx.shutdown().await;
}

/// Starvation promotion. Capacity 1, entirely reserved by Interactive, with a
/// 1s Bulk starvation threshold. No Interactive traffic ever arrives and no
/// slot is ever released, so the only way a queued Bulk request can be admitted
/// is the dispatcher's periodic starvation override.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn starvation_override_promotes_aged_bulk_request() {
    let mut yaml = SchedulerYaml::base();
    // Interactive reserves the whole (capacity-1) pool, locking Bulk out of the
    // normal admission path. Reservations sum to 1 == capacity, so the
    // scheduler still builds (no fallback to legacy).
    yaml.interactive.reserved_floor = 1;
    // Bulk ages out of starvation quickly; keep a real queue to wait in.
    yaml.bulk.starvation_threshold_secs = 1;
    yaml.bulk.queue_size = 4;
    yaml.bulk.queue_timeout_secs = 30; // must outlast the ~1–2s promotion delay
    let yaml_file = write_scheduler_yaml(&yaml);
    let config = scheduler_config(3606, "bulk", yaml_file.path().to_str().unwrap());

    // No gate: once promoted, the Bulk request completes immediately.
    let ctx = AppTestContext::new_with_config(config, vec![scheduler_worker(1, None)]).await;
    let app = ctx.create_app();

    // A lone Bulk request can't admit normally (Interactive's reservation
    // blocks it) and nothing releases a slot — only the starvation override,
    // firing on the dispatcher's periodic tick, can let it through.
    let resp = tokio::time::timeout(
        Duration::from_secs(10),
        app.clone()
            .oneshot(generate_request("starved-bulk", Some("bulk"))),
    )
    .await
    .expect("starvation override should admit the bulk request well within 10s")
    .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "aged bulk request must be promoted past the unused reservation"
    );
    let _ = body_json(resp).await;

    ctx.shutdown().await;
}
