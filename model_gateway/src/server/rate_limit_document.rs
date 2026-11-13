//! `GET` and `PUT /rate-limits/tenants`: the tenant rate-limit policy as a
//! control-plane document. The `admin` role reads the revision in force
//! with its ETag and replaces it against that ETag: a write without one is
//! asked for it, a write against another revision is refused with the live
//! one, and a dry run answers what would change and applies nothing.

use std::sync::Arc;

use axum::{
    extract::{FromRequest, Query, Request, State},
    http::{
        header::{ETAG, IF_MATCH},
        HeaderName, HeaderValue, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use smg_auth::{AuditResource, PrincipalExt};
use tracing::{error, info};

use super::AppState;
use crate::{
    admin::document::Applied,
    rate_limit::{
        DocumentOutcome, DocumentRefusal, PolicyChanges, PolicyReplacement, PolicySummary,
        RateLimitManager, RateLimitYaml,
    },
    worker::admin_json::{refusal, AdminJson},
};

/// The resource in audit lines: the document's path under the control-plane
/// root.
const RESOURCE: &str = "rate-limits/tenants";

/// The revision in force, beside `ETag` on every answer about the document;
/// the named-origin CORS policy exposes both to browser clients.
pub(super) const REVISION_HEADER: HeaderName = HeaderName::from_static("x-smg-config-revision");

/// `?dry_run=true` on a write. Any other key is refused by name, so a
/// misspelt dry run never applies the document it meant to preview.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteQuery {
    #[serde(default)]
    dry_run: bool,
}

/// The answer of a write: the revision in force after it and what the
/// document given means against the one it was checked against.
#[derive(Serialize)]
struct WriteAnswer<'a> {
    applied: bool,
    revision: u64,
    etag: &'a str,
    applied_at: DateTime<Utc>,
    principal: &'a str,
    summary: &'a PolicySummary,
    changes: &'a PolicyChanges,
    #[serde(skip_serializing_if = "Option::is_none")]
    buckets: Option<PolicyReplacement>,
}

/// `response` with the `ETag` and revision headers of `revision`.
fn with_revision(mut response: Response, revision: &Applied<RateLimitYaml>) -> Response {
    let headers = response.headers_mut();
    if let Ok(etag) = HeaderValue::from_str(&revision.etag) {
        headers.insert(ETAG, etag);
    }
    headers.insert(REVISION_HEADER, HeaderValue::from(revision.revision));
    response
}

/// The tenant rate limiter, or the 409 of a gateway without one.
fn limiter(state: &AppState) -> Result<&Arc<RateLimitManager>, Response> {
    state.context.rate_limit_manager.as_ref().ok_or_else(|| {
        refusal(
            StatusCode::CONFLICT,
            "RATE_LIMITER_DISABLED",
            "No tenant rate limiter is configured; there is no document".to_string(),
        )
    })
}

/// The 409 of a manager over a backend of its own, which took no document.
fn no_document() -> Response {
    refusal(
        StatusCode::CONFLICT,
        "POLICY_NOT_REPLACEABLE",
        "The rate-limit backend in use holds no document".to_string(),
    )
}

/// `GET /rate-limits/tenants`: the document in force with its revision,
/// ETag, application time and principal. The `admin` role alone, by the
/// gate's admin-only reads (`ADMIN_ONLY_READS`); 409 without a tenant rate
/// limiter.
pub(super) async fn get(State(state): State<Arc<AppState>>) -> Response {
    let manager = match limiter(&state) {
        Ok(manager) => manager,
        Err(refused) => return refused,
    };
    match manager.document() {
        Some(revision) => with_revision(Json(&*revision).into_response(), &revision),
        None => no_document(),
    }
}

/// `PUT /rate-limits/tenants`: a document checked as the start checks its
/// file and put in force as the next revision, the live buckets keeping
/// their fill where their limits stand. `If-Match` carries the ETag of the
/// revision the caller read: 428 without one, 412 against another.
/// `?dry_run=true` answers what would change and applies nothing. The
/// `admin` role alone. The revision lives in this gateway's memory: a
/// restart reads the start's file again, and another gateway holds its
/// own.
pub(super) async fn put(State(state): State<Arc<AppState>>, request: Request) -> Response {
    // The role is the control-plane gate's: it admits `admin` alone to a
    // PUT before any handler, and the role matrix test pins this route. The
    // principal names the writer in the revision.
    let Some(principal) = request.principal().map(|p| p.id.clone()) else {
        return (
            StatusCode::UNAUTHORIZED,
            "A principal is required to replace the rate-limit document",
        )
            .into_response();
    };
    let manager = match limiter(&state) {
        Ok(manager) => Arc::clone(manager),
        Err(refused) => return refused,
    };
    let dry_run = match Query::<WriteQuery>::try_from_uri(request.uri()) {
        Ok(Query(query)) => query.dry_run,
        Err(error) => {
            return refusal(
                StatusCode::BAD_REQUEST,
                "BAD_REQUEST",
                format!("Invalid query: {error}"),
            )
        }
    };
    // `If-Match: *` matches any revision, which is no protection against a
    // stale write: it counts as absent.
    let expected = match request.headers().get(IF_MATCH).map(HeaderValue::to_str) {
        Some(Ok(value)) if value.trim() != "*" => value.trim().to_string(),
        _ => {
            let refused = refusal(
                StatusCode::PRECONDITION_REQUIRED,
                "PRECONDITION_REQUIRED",
                "If-Match must carry the ETag of the revision in force; read the document first"
                    .to_string(),
            );
            return match manager.document() {
                Some(live) => with_revision(refused, &live),
                None => refused,
            };
        }
    };
    let document = match AdminJson::<RateLimitYaml>::from_request(request, &()).await {
        Ok(AdminJson(document)) => document,
        Err(refused) => return refused,
    };
    // The replacement walks every live bucket in the writer's turn, and a
    // second writer waits for that turn: both off the runtime's workers. The
    // applied write is logged by that task, under this request's span: a
    // client that leaves mid-write drops this handler's future, not the task,
    // and the line is written whenever the swap is.
    let span = tracing::Span::current();
    let replaced = tokio::task::spawn_blocking(move || {
        span.in_scope(|| replace_logged(&manager, &expected, document, &principal, dry_run))
    })
    .await;
    let replaced = match replaced {
        Ok(replaced) => replaced,
        Err(join) => {
            error!(%join, "the tenant rate-limit document's replacement task failed");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };
    match replaced {
        Ok(outcome) => {
            let revision = &outcome.revision;
            let answer = Json(WriteAnswer {
                applied: outcome.applied,
                revision: revision.revision,
                etag: &revision.etag,
                applied_at: revision.applied_at,
                principal: &revision.principal,
                summary: &outcome.summary,
                changes: &outcome.changes,
                buckets: outcome.buckets,
            });
            let response = with_revision(answer.into_response(), revision);
            if !outcome.applied {
                return response;
            }
            AuditResource::name(response, RESOURCE)
        }
        Err(DocumentRefusal::NoDocument) => no_document(),
        Err(DocumentRefusal::Invalid(error)) => {
            refusal(StatusCode::BAD_REQUEST, "INVALID_DOCUMENT", error)
        }
        Err(DocumentRefusal::Stale(live)) => with_revision(
            refusal(
                StatusCode::PRECONDITION_FAILED,
                "PRECONDITION_FAILED",
                format!(
                    "If-Match names another revision; revision {} is in force",
                    live.revision
                ),
            ),
            &live,
        ),
        Err(DocumentRefusal::BackendFixed(error)) => {
            refusal(StatusCode::CONFLICT, "POLICY_NOT_REPLACEABLE", error)
        }
        Err(DocumentRefusal::Record(error)) => {
            error!(%error, "the tenant rate-limit document could not be recorded; nothing was applied");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// `manager.replace_document(..)` with the applied write's log line written
/// by the same call. The handler awaits this call off the runtime; a client
/// that leaves meanwhile drops the handler, not the call, so the line is
/// never lost to a response nobody reads.
fn replace_logged(
    manager: &RateLimitManager,
    expected_etag: &str,
    document: RateLimitYaml,
    principal: &str,
    dry_run: bool,
) -> Result<DocumentOutcome, DocumentRefusal> {
    let replaced = manager.replace_document(expected_etag, document, principal, dry_run);
    if let Ok(DocumentOutcome {
        applied: true,
        revision,
        buckets,
        ..
    }) = &replaced
    {
        let buckets = buckets.unwrap_or_default();
        info!(
            resource = RESOURCE,
            revision_before = revision.revision - 1,
            revision = revision.revision,
            etag = %revision.etag,
            principal = %revision.principal,
            kept = buckets.kept,
            replaced = buckets.replaced,
            dropped = buckets.dropped,
            "the tenant rate-limit policy was replaced"
        );
    }
    replaced
}

#[cfg(test)]
mod tests {
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
        app_context::AppContext, config::RouterConfig, health::ProbeState,
        policies::PolicyRegistry, routers::gateway::Gateway, worker::WorkerRegistry,
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
                .client(crate::server::http_client())
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
}
