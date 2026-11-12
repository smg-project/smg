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
        DocumentRefusal, PolicyChanges, PolicyReplacement, PolicySummary, RateLimitManager,
        RateLimitYaml,
    },
    worker::admin_json::{refusal, AdminJson},
};

/// The resource in audit lines: the document's path under the control-plane
/// root.
const RESOURCE: &str = "rate-limits/tenants";

/// The revision in force, beside `ETag` on every answer about the document;
/// the named-origin CORS policy exposes both to browser clients.
pub(super) const REVISION_HEADER: HeaderName = HeaderName::from_static("x-smg-config-revision");

/// `?dry_run=true` on a write.
#[derive(Deserialize)]
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
    // second writer waits for that turn: both off the runtime's workers.
    let replaced = tokio::task::spawn_blocking(move || {
        manager.replace_document(&expected, document, &principal, dry_run)
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
            let buckets = outcome.buckets.unwrap_or_default();
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
