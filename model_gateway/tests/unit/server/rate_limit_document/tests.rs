use std::{
    sync::{mpsc, OnceLock},
    task::{Context, Waker},
    time::Duration,
};

use axum::{body::Body, http::header::CONTENT_TYPE};
use llm_tokenizer::registry::TokenizerRegistry;
use smg_auth::{AuthMethod, Principal, Role};
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use tracing_test::traced_test;

use super::*;
use crate::{
    app_context::AppContext,
    config::RouterConfig,
    policies::PolicyRegistry,
    routers::{common::snapshot::WorkerRegistry, gateway::Gateway},
    server::{health::ProbeState, http_client},
};

const START: &str = "default_policy:\n  tokens_per_minute: 1000\n  requests_per_minute: 60\ntenants:\n  - tenant_key: auth:team-red\n    tokens_per_minute: 500\n    requests_per_minute: 30\n";
const NEXT: &str = r#"{"default_policy":{"tokens_per_minute":1000,"requests_per_minute":60},"tenants":[{"tenant_key":"auth:team-red","tokens_per_minute":600,"requests_per_minute":30}]}"#;

/// A gateway state over a tenant rate limiter read from a start file in
/// `dir`, the one tenant beside the default.
fn documented_state(dir: &std::path::Path) -> Arc<AppState> {
    let path = dir.join("rate_limit.yaml");
    std::fs::write(&path, START).expect("the start file written");
    let router_config = RouterConfig::builder()
        .tenant_rate_limit_enabled(true)
        .tenant_rate_limit_config(Some(path.to_string_lossy().into_owned()))
        .build_unchecked();
    let manager = RateLimitManager::from_config(&router_config)
        .expect("a start file the manager takes")
        .expect("a tenant rate limiter");
    let context = Arc::new(
        AppContext::builder()
            .client(http_client())
            .rate_limit_manager(Some(manager))
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
            .reasoning_parser_factory(None)
            .tool_parser_factory(None)
            .worker_registry(Arc::new(WorkerRegistry::new()))
            .policy_registry(Arc::new(PolicyRegistry::new(router_config.policy.clone())))
            .router_config(router_config)
            .response_storage(Arc::new(MemoryResponseStorage::new()))
            .conversation_storage(Arc::new(MemoryConversationStorage::new()))
            .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
            .worker_monitor(None)
            .worker_job_queue(Arc::new(OnceLock::new()))
            .workflow_engines(Arc::new(OnceLock::new()))
            .mcp_orchestrator(Arc::new(OnceLock::new()))
            .build()
            .expect("an app context"),
    );
    Arc::new(AppState {
        router: Arc::new(Gateway::new(context.worker_registry.clone())),
        probe_state: ProbeState::new(context.inflight_tracker.clone()),
        context,
        admission_queue: None,
        gateway: None,
        mesh_handler: None,
        mesh_adapters: None,
        secret_sources: Default::default(),
        secrets: None,
    })
}

/// The JSON body of `response`.
async fn body_json(response: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body read");
    serde_json::from_slice(&bytes).expect("a JSON body")
}

/// A write of `document` against `etag` by the admin principal `ops`,
/// as the control-plane gate hands it to the handler.
fn write(path: &str, etag: &str, document: &str) -> Request {
    let mut request = Request::builder()
        .method("PUT")
        .uri(path)
        .header(IF_MATCH, etag)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(document.to_string()))
        .expect("a well-formed request");
    request.extensions_mut().insert(Principal {
        id: "ops".into(),
        name: None,
        auth_method: AuthMethod::ApiKey {
            key_id: "ops".into(),
        },
        role: Role::Admin,
    });
    request
}

/// A client that leaves while its write is applied: the handler's future
/// is dropped at its await of the blocking task, the task applies the
/// revision all the same, and the applied write's log line is written
/// under the request's span. The runtime's one blocking thread is held
/// until the drop, so the first poll is pending by construction, and the
/// line is awaited before the revision is read. A dry run, read to its
/// end, logs nothing.
#[traced_test]
#[test]
fn a_write_whose_client_leaves_still_logs_its_application() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .expect("a runtime");
    runtime.block_on(async {
        let dir = tempfile::tempdir().expect("a temp dir");
        let state = documented_state(dir.path());
        let manager = Arc::clone(
            state
                .context
                .rate_limit_manager
                .as_ref()
                .expect("a tenant rate limiter"),
        );
        let live = manager.document().expect("the start's document");

        let tried = put(
            State(Arc::clone(&state)),
            write("/rate-limits/tenants?dry_run=true", &live.etag, NEXT),
        )
        .await;
        assert_eq!(tried.status(), StatusCode::OK);
        assert!(!logs_contain("the tenant rate-limit policy was replaced"));

        let (release, held) = mpsc::channel::<()>();
        let _hold = tokio::task::spawn_blocking(move || held.recv());
        let mut applied = Box::pin(put(
            State(Arc::clone(&state)),
            write("/rate-limits/tenants", &live.etag, NEXT),
        ));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(
            std::future::Future::poll(applied.as_mut(), &mut cx).is_pending(),
            "the handler yields at its await of the queued replacement task"
        );
        drop(applied);
        release.send(()).expect("the hold waits for its release");

        let logged = tokio::time::timeout(Duration::from_secs(10), async {
            while !logs_contain("the tenant rate-limit policy was replaced") {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(
            logged.is_ok(),
            "the replacement task logs the applied write without its handler"
        );
        assert_eq!(manager.document().map(|live| live.revision), Some(2));
        assert!(logs_contain("principal=ops"));
    });
}

/// A dry-run key spelt any other way is refused by name with nothing
/// applied: a current `If-Match` and a valid document ride along, and the
/// revision stays where it was.
#[tokio::test]
async fn a_misspelt_dry_run_key_is_refused_by_name_and_applies_nothing() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let state = documented_state(dir.path());
    let manager = Arc::clone(
        state
            .context
            .rate_limit_manager
            .as_ref()
            .expect("a tenant rate limiter"),
    );
    let live = manager.document().expect("the start's document");
    for key in ["dryRun", "dry-run", "dryrun"] {
        let refused = put(
            State(Arc::clone(&state)),
            write(
                &format!("/rate-limits/tenants?{key}=true"),
                &live.etag,
                NEXT,
            ),
        )
        .await;
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST, "{key}");
        let body = body_json(refused).await;
        assert_eq!(body["code"], "BAD_REQUEST", "{key}: {body}");
        let error = body["error"].as_str().expect("an error message");
        assert!(error.contains(key), "{key}: {error}");
        assert_eq!(
            manager.document().expect("the document").revision,
            1,
            "{key} applied the document"
        );
    }
}
