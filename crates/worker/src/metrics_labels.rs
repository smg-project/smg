//! Metric label values the worker's writers share with a host's.
//!
//! The `worker_type` and `connection_mode` values are the string forms of the
//! types that carry them, [`WorkerType::as_str`] and
//! [`ConnectionMode::as_str`]; the values no type spells are here.
//! `smg-external-router` re-exports each at the path it had there.
//!
//! [`WorkerType::as_str`]: crate::worker::WorkerType::as_str
//! [`ConnectionMode::as_str`]: crate::worker::ConnectionMode::as_str

/// The `model` label of a request that names no served model: an unknown
/// name refused with 404, or a value past the model-label cap. One fixed
/// value, as the `path` label is `other` for an unmatched route, so a
/// client cannot mint label values.
pub const MODEL_OTHER: &str = "other";

/// The `outcome` of a request the circuit breaker counted as a success, and
/// the `result` of a health check that passed.
pub const CB_SUCCESS: &str = "success";

/// The `outcome` of a request the circuit breaker counted as a failure, and
/// the `result` of a health check that failed.
pub const CB_FAILURE: &str = "failure";
