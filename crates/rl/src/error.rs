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
    #[error("invalid request body: {0}")]
    InvalidBody(String),
    #[error("a non-empty `selector` query parameter is required")]
    SelectorRequired,
    #[error("invalid selector at byte {offset}: {message}")]
    InvalidSelector { offset: usize, message: String },
    #[error("no workers match selector `{0}`")]
    NoWorkersMatch(String),
    #[error("worker `{0}` not found")]
    WorkerNotFound(String),
    #[error(
        "worker `{worker_id}` has no control endpoint ({connection_mode} transport): {message}"
    )]
    NoControlEndpoint {
        worker_id: String,
        url: String,
        connection_mode: String,
        message: String,
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
}

impl RlError {
    /// Stable machine-readable code for the JSON body.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidEnginePath(_) => "invalid_engine_path",
            Self::InvalidBody(_) => "invalid_body",
            Self::SelectorRequired => "selector_required",
            Self::InvalidSelector { .. } => "invalid_selector",
            Self::NoWorkersMatch(_) => "no_workers_match",
            Self::WorkerNotFound(_) => "worker_not_found",
            Self::NoControlEndpoint { .. } => "no_control_endpoint",
            Self::UpstreamUnreachable { .. } => "upstream_unreachable",
            Self::UpstreamTimeout { .. } => "upstream_timeout",
        }
    }

    pub fn status(&self) -> StatusCode {
        match self {
            Self::InvalidEnginePath(_)
            | Self::InvalidBody(_)
            | Self::SelectorRequired
            | Self::InvalidSelector { .. }
            | Self::NoWorkersMatch(_) => StatusCode::BAD_REQUEST,
            Self::WorkerNotFound(_) => StatusCode::NOT_FOUND,
            Self::NoControlEndpoint { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            Self::UpstreamUnreachable { .. } => StatusCode::BAD_GATEWAY,
            Self::UpstreamTimeout { .. } => StatusCode::GATEWAY_TIMEOUT,
        }
    }

    /// JSON body with the code, message, and per-variant context fields.
    pub fn to_json(&self) -> Value {
        let mut body = json!({ "error": self.code(), "message": self.to_string() });
        match self {
            Self::InvalidSelector { offset, .. } => body["offset"] = json!(offset),
            Self::NoWorkersMatch(selector) => body["selector"] = json!(selector),
            Self::WorkerNotFound(id) => body["id"] = json!(id),
            Self::NoControlEndpoint {
                worker_id,
                url,
                connection_mode,
                ..
            } => {
                body["worker_id"] = json!(worker_id);
                body["url"] = json!(url);
                body["connection_mode"] = json!(connection_mode);
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
#[path = "../tests/unit/error/tests.rs"]
mod tests;
