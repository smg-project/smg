//! HTTP mapping for priority-scheduler admission outcomes.
//!
//! [`SchedulerError`] is the response-facing form of a rejected admission
//! (and of a mid-flight preemption). The admission middleware converts a
//! [`RejectionReason`] into one of these; long-running handlers return
//! [`SchedulerError::Preempted`] when their cancel token fires.

use axum::{
    http::{header::RETRY_AFTER, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};

use super::engine::RejectionReason;
use crate::{middleware::SHED_RETRY_AFTER_SECS, routers::error::create_error};

/// Response header set on a preempted request so clients/proxies can tell a
/// preemption 503 apart from an ordinary overload 503.
pub const HEADER_X_SMG_PREEMPTED: &str = "X-SMG-Preempted";

/// Client-Closed-Request (nginx convention). Used for `ClientCancelled`,
/// which is essentially never read — the client has already disconnected.
const STATUS_CLIENT_CLOSED_REQUEST: u16 = 499;

/// How the scheduler's admission decision surfaces to the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerError {
    /// Per-class queue at its configured limit. → 429 + `Retry-After: 2`.
    QueueFull,
    /// Queued waiter aged past `queue_timeout`. → 503 + `Retry-After: 2`.
    QueueTimeout,
    /// Cancelled in-flight to admit a higher-priority waiter, before TTFT.
    /// → 503 + `Retry-After: 1` + `X-SMG-Preempted: true`.
    Preempted,
    /// The client disconnected before admission completed. → 499 (never
    /// actually read; the connection is gone).
    ClientCancelled,
}

impl SchedulerError {
    fn status(self) -> StatusCode {
        match self {
            Self::QueueFull => StatusCode::TOO_MANY_REQUESTS,
            Self::QueueTimeout => StatusCode::SERVICE_UNAVAILABLE,
            Self::Preempted => StatusCode::SERVICE_UNAVAILABLE,
            // `unwrap_or` (not `unwrap`/`expect`, both denied) — 499 is a
            // valid code so the fallback is never taken.
            Self::ClientCancelled => StatusCode::from_u16(STATUS_CLIENT_CLOSED_REQUEST)
                .unwrap_or(StatusCode::REQUEST_TIMEOUT),
        }
    }

    fn code(self) -> &'static str {
        match self {
            Self::QueueFull => "scheduler_queue_full",
            Self::QueueTimeout => "scheduler_queue_timeout",
            Self::Preempted => "scheduler_preempted",
            Self::ClientCancelled => "scheduler_client_cancelled",
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::QueueFull => "request queue is full for this priority class",
            Self::QueueTimeout => "timed out waiting for an admission slot",
            Self::Preempted => "request preempted by higher-priority traffic",
            Self::ClientCancelled => "client closed the request before admission",
        }
    }
}

impl From<RejectionReason> for SchedulerError {
    fn from(reason: RejectionReason) -> Self {
        match reason {
            RejectionReason::QueueFull => Self::QueueFull,
            RejectionReason::QueueTimeout => Self::QueueTimeout,
            RejectionReason::Preempted => Self::Preempted,
            RejectionReason::ClientCancelled => Self::ClientCancelled,
        }
    }
}

impl IntoResponse for SchedulerError {
    fn into_response(self) -> Response {
        // Reuse the gateway's standard error shape (JSON body + the
        // X-SMG-Error-Code header), then layer the shed/preemption headers
        // on top.
        let mut resp = create_error(self.status(), self.code(), self.message());
        let headers = resp.headers_mut();
        match self {
            Self::QueueFull | Self::QueueTimeout => {
                headers.insert(RETRY_AFTER, HeaderValue::from(SHED_RETRY_AFTER_SECS));
            }
            Self::Preempted => {
                headers.insert(RETRY_AFTER, HeaderValue::from_static("1"));
                headers.insert(HEADER_X_SMG_PREEMPTED, HeaderValue::from_static("true"));
            }
            Self::ClientCancelled => {}
        }
        resp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_full_maps_to_429_with_retry_after() {
        let resp = SchedulerError::QueueFull.into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            resp.headers().get(RETRY_AFTER),
            Some(&HeaderValue::from(SHED_RETRY_AFTER_SECS))
        );
    }

    #[test]
    fn test_queue_timeout_maps_to_503_with_retry_after() {
        let resp = SchedulerError::QueueTimeout.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get(RETRY_AFTER),
            Some(&HeaderValue::from(SHED_RETRY_AFTER_SECS))
        );
    }

    #[test]
    fn test_preempted_maps_to_503_with_headers() {
        let resp = SchedulerError::Preempted.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get(RETRY_AFTER).map(|v| v.to_str().unwrap()),
            Some("1")
        );
        assert_eq!(
            resp.headers()
                .get(HEADER_X_SMG_PREEMPTED)
                .map(|v| v.to_str().unwrap()),
            Some("true")
        );
    }

    #[test]
    fn test_client_cancelled_maps_to_499() {
        let resp = SchedulerError::ClientCancelled.into_response();
        assert_eq!(resp.status().as_u16(), 499);
    }

    #[test]
    fn test_non_preempt_has_no_preempt_header() {
        let resp = SchedulerError::QueueFull.into_response();
        assert!(resp.headers().get(HEADER_X_SMG_PREEMPTED).is_none());
    }

    #[test]
    fn test_client_cancelled_has_no_retry_after() {
        let resp = SchedulerError::ClientCancelled.into_response();
        assert!(resp.headers().get(RETRY_AFTER).is_none());
    }

    #[test]
    fn test_from_rejection_reason_is_one_to_one() {
        assert_eq!(
            SchedulerError::from(RejectionReason::QueueFull),
            SchedulerError::QueueFull
        );
        assert_eq!(
            SchedulerError::from(RejectionReason::QueueTimeout),
            SchedulerError::QueueTimeout
        );
        assert_eq!(
            SchedulerError::from(RejectionReason::Preempted),
            SchedulerError::Preempted
        );
        assert_eq!(
            SchedulerError::from(RejectionReason::ClientCancelled),
            SchedulerError::ClientCancelled
        );
    }
}
