//! Error type for the RL control plane. Every variant maps to one HTTP status
//! and a JSON body `{"error": <code>, "message": <text>, ...context}`.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RlError {
    #[error("invalid engine path: {0}")]
    InvalidEnginePath(String),
    #[error("a non-empty `selector` query parameter is required")]
    SelectorRequired,
    #[error("invalid selector at byte {offset}: {message}")]
    InvalidSelector { offset: usize, message: String },
    #[error("no workers match selector `{0}`")]
    NoWorkersMatch(String),
    #[error("worker `{0}` not found")]
    WorkerNotFound(String),
    #[error("worker `{worker_id}` uses connection mode `{mode}`, which cannot be proxied")]
    UnsupportedConnectionMode {
        worker_id: String,
        url: String,
        mode: String,
    },
    #[error("upstream `{url}` unreachable: {message}")]
    UpstreamUnreachable {
        worker_id: String,
        url: String,
        message: String,
    },
    #[error("upstream `{url}` timed out after {timeout_secs}s")]
    UpstreamTimeout {
        worker_id: String,
        url: String,
        timeout_secs: u64,
    },
    #[error("failed to build the RL control-plane HTTP client: {0}")]
    Client(String),
}

impl RlError {
    /// Stable machine-readable code for the JSON body.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidEnginePath(_) => "invalid_engine_path",
            Self::SelectorRequired => "selector_required",
            Self::InvalidSelector { .. } => "invalid_selector",
            Self::NoWorkersMatch(_) => "no_workers_match",
            Self::WorkerNotFound(_) => "worker_not_found",
            Self::UnsupportedConnectionMode { .. } => "unsupported_connection_mode",
            Self::UpstreamUnreachable { .. } => "upstream_unreachable",
            Self::UpstreamTimeout { .. } => "upstream_timeout",
            Self::Client(_) => "client_init_failed",
        }
    }

    pub fn status(&self) -> StatusCode {
        match self {
            Self::InvalidEnginePath(_)
            | Self::SelectorRequired
            | Self::InvalidSelector { .. }
            | Self::NoWorkersMatch(_) => StatusCode::BAD_REQUEST,
            Self::WorkerNotFound(_) => StatusCode::NOT_FOUND,
            Self::UnsupportedConnectionMode { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            Self::UpstreamUnreachable { .. } => StatusCode::BAD_GATEWAY,
            Self::UpstreamTimeout { .. } => StatusCode::GATEWAY_TIMEOUT,
            Self::Client(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// JSON body with the code, message, and per-variant context fields.
    pub fn to_json(&self) -> Value {
        let mut body = json!({ "error": self.code(), "message": self.to_string() });
        match self {
            Self::InvalidSelector { offset, .. } => body["offset"] = json!(offset),
            Self::NoWorkersMatch(selector) => body["selector"] = json!(selector),
            Self::WorkerNotFound(id) => body["id"] = json!(id),
            Self::UnsupportedConnectionMode {
                worker_id,
                url,
                mode,
            } => {
                body["worker_id"] = json!(worker_id);
                body["url"] = json!(url);
                body["connection_mode"] = json!(mode);
            }
            Self::UpstreamUnreachable { worker_id, url, .. }
            | Self::UpstreamTimeout { worker_id, url, .. } => {
                body["worker_id"] = json!(worker_id);
                body["url"] = json!(url);
            }
            _ => {}
        }
        body
    }
}

impl IntoResponse for RlError {
    fn into_response(self) -> Response {
        (self.status(), Json(self.to_json())).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_and_statuses() {
        let e = RlError::WorkerNotFound("abc".into());
        assert_eq!(e.status(), StatusCode::NOT_FOUND);
        assert_eq!(e.to_json()["error"], "worker_not_found");
        assert_eq!(e.to_json()["id"], "abc");

        let e = RlError::InvalidSelector {
            offset: 7,
            message: "expected operator".into(),
        };
        assert_eq!(e.status(), StatusCode::BAD_REQUEST);
        assert_eq!(e.to_json()["offset"], 7);

        let e = RlError::UpstreamTimeout {
            worker_id: "w".into(),
            url: "http://x".into(),
            timeout_secs: 5,
        };
        assert_eq!(e.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(e.to_json()["url"], "http://x");
    }
}
