// model_gateway/src/router/topology/error.rs

use thiserror::Error;

/// Errors that can occur during topology‑aware routing.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TopologyError {
    /// No text content was found in the request JSON.
    #[error("No text content found in request")]
    MissingContent,

    /// The JSON structure is invalid.
    #[error("Invalid JSON structure: {0}")]
    InvalidJson(String),

    /// The hash ring is empty (no workers available).
    #[error("Consistent hash ring is empty")]
    EmptyRing,

    /// The requested worker ID was not found.
    #[error("Worker not found: {0}")]
    WorkerNotFound(String),

    /// The request contains invalid UTF‑8.
    #[error("Invalid UTF‑8 sequence")]
    InvalidUtf8,

    /// The request is empty.
    #[error("Request is empty")]
    EmptyRequest,

    /// The cache capacity has been exceeded.
    #[error("Cache capacity exceeded")]
    CacheFull,
}