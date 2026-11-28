//! `priority_admission_middleware`: the axum layer that runs the priority
//! scheduler on protected routes when it is enabled.
//!
//! Pipeline position (route_layer order): runs after tenant resolution
//! (so `RouteRequestMeta.tenant_key` is in extensions for the clamp) and
//! before the handler. On admission it inserts the permit's cancel token
//! into request extensions (so long-running handlers can `select!` against
//! it for preemption) and hands the permit to the edge's one body wrapper
//! (TTFT marking on the first data frame, slot release when the body ends).

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
    SchedulerError, SchedulerPermit, HEADER_X_SMG_PREEMPTED, PRIORITY_HEADER,
};
use crate::{
    middleware::{edge_body::BodyDuties, tenant_resolution::restamp_accepted_at, RouteRequestMeta},
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
    let tenant = req
        .extensions()
        .get::<RouteRequestMeta>()
        .map(|m| m.tenant_key().clone())
        .unwrap_or_else(|| TenantKey::new("anonymous"));
    let resolved = resolve_priority(&req, &state, &tenant);
    let class = resolved.effective;

    if resolved.unknown {
        sched_metrics::record_unknown_priority(state.tenant_labels.intern(tenant.as_str()));
    }
    if class < resolved.requested {
        sched_metrics::record_clamp(
            resolved.requested,
            class,
            state.tenant_labels.intern(tenant.as_str()),
        );
    }

    // Per-second check (only set when an explicit rate is configured),
    // run before admission so a rejected request never consumes a slot.
    // `try_take` spends rate budget without a concurrency slot: the
    // scheduler owns concurrency. Tracked by smg_http_rate_limit_total, so
    // not double-counted here.
    if let Some(bucket) = &state.rate_limiter {
        if bucket.try_take(1.0).is_err() {
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            return SchedulerError::QueueFull.into_response();
        }
    }

    let request_id = next_registry_id();

    // NOTE: client-disconnect detection during the queue wait is not yet
    // wired (axum does not surface it to a middleware pre-`next.run`), so we
    // pass a fresh token; queued waits are bounded by `queue_timeout`. A
    // real disconnect drops the response future: while queued, the scheduler
    // records that exit as `client_cancelled` and reaps the waiter; once
    // admitted, the permit rides on the response body and is dropped with
    // it, releasing the slot.
    let cancel = CancellationToken::new();

    match state.scheduler.admit(class, request_id, cancel).await {
        AdmitOutcome::Admitted(permit) => {
            // Hand the handler the cancel token (for preemption select!).
            req.extensions_mut().insert(permit.cancel_token());
            // The class's queue wait was the scheduler's, bounded by its own
            // timeout; the bound ahead of worker selection counts from here.
            restamp_accepted_at(req.extensions_mut());
            let mut response = next.run(req).await;
            // Best-effort: the handler's PreemptionGuard tags a *pre-response*
            // preemption (a 503 carrying this header), which we count as
            // `preempted`. A preemption that fires after the handler produced
            // its 200 headers but before the first body byte is truncated by
            // the edge's body wrapper and shows here as `admitted` — the
            // response headers are already flushed, so the marker cannot be added. The
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
            carry_permit(&mut response, permit);
            response
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

/// The permit rides on the response body, held until the body ends or is
/// dropped by the edge's one body wrapper, which marks TTFT on the first data
/// frame; unless the response is the preemption 503 the handler's guard
/// wrote. That request is over: the permit goes now, the slot with it, and the
/// 503's body goes out whole. The wrapper's first-frame check could not tell
/// that 503 from a 200 whose first byte a preemption beat (the scheduler won
/// the preempt CAS either way), and a layer between here and the edge that
/// replaces the body while keeping the extensions (the Messages error
/// envelope) would have it end the rebuilt body before its first byte.
///
/// The header alone does not say so: a worker's response carries its own
/// headers through the relays, this one included (a gateway one tier down
/// writes it). The scheduler's own mark does: nothing has asked the permit
/// about a first frame yet, so a handle that is no longer preemptible was
/// preempted by this scheduler, which no worker can fake. A relayed header
/// on an admitted stream leaves the permit on the body, the slot held and
/// TTFT marked, as before.
fn carry_permit(response: &mut Response, permit: SchedulerPermit) {
    if response.headers().contains_key(HEADER_X_SMG_PREEMPTED) && !permit.handle().is_preemptible()
    {
        drop(permit);
        return;
    }
    BodyDuties::hand(response, |duties| {
        duties.permit = Some(Box::new(permit));
    });
}

#[cfg(test)]
#[path = "../../../tests/unit/middleware/scheduler/admission/permit_tests.rs"]
mod permit_tests;
