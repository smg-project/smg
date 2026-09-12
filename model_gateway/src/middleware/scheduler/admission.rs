//! `priority_admission_middleware`: the axum layer that runs the priority
//! scheduler on protected routes when it is enabled.
//!
//! Pipeline position (route_layer order): runs after tenant resolution
//! (so `RouteRequestMeta.tenant_key` is in extensions for the clamp) and
//! before the handler. On admission it inserts the permit's cancel token
//! into request extensions (so long-running handlers can `select!` against
//! it for preemption) and wraps the response body in `SchedulerGuardBody`
//! (TTFT marking + slot release).

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use axum::{
    body::Body,
    extract::State,
    http::Request,
    middleware::Next,
    response::{IntoResponse, Response},
};
use smg_auth::RequestId;
use tokio_util::sync::CancellationToken;
use tracing::trace;

use super::{
    metrics as sched_metrics, state::SchedulerState, AdmitOutcome, Class, RejectionReason,
    SchedulerError, SchedulerGuardBody, HEADER_X_SMG_PREEMPTED, PRIORITY_HEADER,
};
use crate::{
    middleware::{concurrency::is_capacity_release_request, RouteRequestMeta},
    observability::metrics::{metrics_labels, Metrics},
    tenant::TenantKey,
};

/// Monotonic source of registry keys for admitted requests. Each admission
/// gets a unique key, so the inflight registry can never be clobbered by a
/// duplicate client-supplied `x-request-id` (the collision hazard flagged
/// in earlier review). The client request id is still used for logging.
static NEXT_ADMISSION_ID: AtomicU64 = AtomicU64::new(0);

fn next_registry_id() -> RequestId {
    let n = NEXT_ADMISSION_ID.fetch_add(1, Ordering::Relaxed);
    RequestId(format!("sched-{n}"))
}

/// The class a request resolves to, plus what it asked for.
struct ResolvedPriority {
    /// Post-clamp class the request is admitted under.
    effective: Class,
    /// Class parsed from the header before the tenant clamp.
    requested: Class,
    /// Header carried a non-empty value that didn't name a known class.
    unknown: bool,
}

/// Resolve the effective class: parse the priority header, then clamp it
/// down to the tenant's configured `max_class` (a low-tier tenant cannot
/// self-promote by setting the header). `min` is the clamp because of the
/// `Ord` derive on `Class`.
fn resolve_priority(
    req: &Request<Body>,
    state: &SchedulerState,
    tenant: &TenantKey,
) -> ResolvedPriority {
    let raw = req
        .headers()
        .get(PRIORITY_HEADER)
        .and_then(|h| h.to_str().ok());
    let requested = raw.map(Class::parse_header).unwrap_or(Class::Default);
    // Unknown = present, non-empty, not "default", yet still parsed to
    // Default (i.e. an unrecognized value silently downgraded).
    let unknown = raw.map(str::trim).is_some_and(|v| {
        !v.is_empty() && !v.eq_ignore_ascii_case("default") && requested == Class::Default
    });
    let max_class = state.resolver.policy(tenant).max_class;
    ResolvedPriority {
        effective: requested.min(max_class),
        requested,
        unknown,
    }
}

/// Map an admit rejection to an `admit_total` outcome label.
fn rejection_outcome(reason: RejectionReason) -> &'static str {
    match reason {
        RejectionReason::QueueFull => sched_metrics::outcome::REJECTED_QUEUE_FULL,
        RejectionReason::QueueTimeout => sched_metrics::outcome::REJECTED_QUEUE_TIMEOUT,
        RejectionReason::Preempted => sched_metrics::outcome::PREEMPTED,
        RejectionReason::ClientCancelled => sched_metrics::outcome::CLIENT_CANCELLED,
    }
}

pub async fn priority_admission_middleware(
    State(state): State<Arc<SchedulerState>>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    if is_capacity_release_request(&req) {
        return next.run(req).await;
    }

    let tenant = req
        .extensions()
        .get::<RouteRequestMeta>()
        .map(|m| m.tenant_key().clone())
        .unwrap_or_else(|| TenantKey::new("anonymous"));
    let resolved = resolve_priority(&req, &state, &tenant);
    let class = resolved.effective;

    if resolved.unknown {
        sched_metrics::record_unknown_priority(tenant.as_str());
    }
    if class < resolved.requested {
        sched_metrics::record_clamp(resolved.requested, class, tenant.as_str());
    }

    // RPS sibling check (only set when an explicit per-second limit is
    // configured). Checked before admission so a rejected request never
    // consumes a slot. Tokens are not returned — refill is time-based.
    // Tracked by smg_http_rate_limit_total, so not double-counted here.
    if let Some(bucket) = &state.rate_limiter {
        if bucket.try_acquire(1.0).is_err() {
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            return SchedulerError::QueueFull.into_response();
        }
    }

    let request_id = next_registry_id();

    // NOTE: client-disconnect detection during the queue wait is not yet
    // wired (axum does not surface it to a middleware pre-`next.run`), so we
    // pass a fresh token; queued waits are bounded by `queue_timeout`. A
    // real disconnect drops the response future (and the SchedulerGuardBody)
    // once admitted, releasing the slot.
    let cancel = CancellationToken::new();

    match state.scheduler.admit(class, request_id, cancel).await {
        AdmitOutcome::Admitted(permit) => {
            // Hand the handler the cancel token (for preemption select!).
            req.extensions_mut().insert(permit.cancel_token());
            let response = next.run(req).await;
            // Best-effort: the handler's PreemptionGuard tags a *pre-response*
            // preemption (a 503 carrying this header), which we count as
            // `preempted`. A preemption that fires after the handler produced
            // its 200 headers but before the first body byte is truncated by
            // SchedulerGuardBody and shows here as `admitted` — the response
            // headers are already flushed, so the marker cannot be added. The
            // authoritative preemption count is `smg_scheduler_preemption_total`
            // (recorded at the preemptor side), not this bucket.
            let outcome = if response.headers().contains_key(HEADER_X_SMG_PREEMPTED) {
                sched_metrics::outcome::PREEMPTED
            } else {
                sched_metrics::outcome::ADMITTED
            };
            sched_metrics::record_admit(class, outcome);
            trace!(
                scheduler.class = class.as_str(),
                scheduler.requested_class = resolved.requested.as_str(),
                scheduler.tenant = %tenant,
                scheduler.admit_outcome = outcome,
                "scheduler admission decision"
            );
            let (parts, body) = response.into_parts();
            Response::from_parts(parts, Body::new(SchedulerGuardBody::new(body, permit)))
        }
        AdmitOutcome::Rejected(reason) => {
            let outcome = rejection_outcome(reason);
            sched_metrics::record_admit(class, outcome);
            trace!(
                scheduler.class = class.as_str(),
                scheduler.requested_class = resolved.requested.as_str(),
                scheduler.tenant = %tenant,
                scheduler.admit_outcome = outcome,
                "scheduler admission decision"
            );
            SchedulerError::from(reason).into_response()
        }
    }
}
