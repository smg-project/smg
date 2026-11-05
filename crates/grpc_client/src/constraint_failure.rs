//! A worker's report that the engine could not continue a request under the
//! request's own constraint (a JSON schema, a regular expression, a grammar,
//! a structural tag), carried on the status that ends its `Generate` stream.
//!
//! An engine ends such a request the way it ends any it cannot serve (an
//! abort with no status of its own, an error finish), which a router reads as
//! a server fault it retries (the same tokens fail the same way on every
//! replay, each one a prefill) or as a completion with nothing in it. Where
//! the engine itself attributes the failure to the constraint, the wire ends
//! the request with this one status instead: the code `FAILED_PRECONDITION`
//! (a router that predates the trailer answers a 400 for it and retries
//! nothing), a message that names the constraint and keeps the engine's own
//! words, and the trailer [`CONSTRAINT_FAILED_METADATA_KEY`], which a router
//! reads to answer the client for the constraint (a 422 under the code
//! `constraint_failed`), terminal for its retry layer, with no circuit-breaker
//! sample against a worker that is healthy. A failure the engine does not
//! attribute is never inferred from the request carrying a constraint: it
//! keeps the status it has on any request.

use std::fmt;

use tonic::{metadata::MetadataValue, Status};

/// The gRPC trailer that carries the report; its value names the side that
/// reports it (`engine`: the engine's own account, relayed by the wire).
pub const CONSTRAINT_FAILED_METADATA_KEY: &str = "smg-constraint-failed";

/// The trailer's value: the engine's own report.
const REPORTED_BY_ENGINE: &str = "engine";

/// The status a wire ends a constrained request with when the engine could
/// not continue it under the constraint; `detail` is the engine's own account
/// (its abort message, its finish reason).
pub fn constraint_failed_status(detail: impl fmt::Display) -> Status {
    let mut status = Status::failed_precondition(format!(
        "the engine could not continue the request under its constraint: {detail}"
    ));
    status.metadata_mut().insert(
        CONSTRAINT_FAILED_METADATA_KEY,
        MetadataValue::from_static(REPORTED_BY_ENGINE),
    );
    status
}

/// Whether `status` is a worker's report that the engine could not continue
/// the request under its constraint (see [`constraint_failed_status`]).
pub fn is_constraint_failure(status: &Status) -> bool {
    status
        .metadata()
        .get(CONSTRAINT_FAILED_METADATA_KEY)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == REPORTED_BY_ENGINE)
}

#[cfg(test)]
mod tests {
    use tonic::Code;

    use super::*;

    #[test]
    fn the_report_names_the_constraint_and_keeps_the_engines_words() {
        let status = constraint_failed_status("the scheduler aborted the request: Aborted");
        assert_eq!(status.code(), Code::FailedPrecondition);
        assert_eq!(
            status.message(),
            "the engine could not continue the request under its constraint: the scheduler \
             aborted the request: Aborted"
        );
        assert!(is_constraint_failure(&status));
    }

    #[test]
    fn a_status_without_the_report_is_not_one() {
        assert!(!is_constraint_failure(&Status::failed_precondition(
            "string `stop` sequences need the model tokenizer"
        )));
        assert!(!is_constraint_failure(&Status::internal(
            "the scheduler aborted the request: Aborted"
        )));
        let mut odd = Status::failed_precondition("odd");
        odd.metadata_mut().insert(
            CONSTRAINT_FAILED_METADATA_KEY,
            MetadataValue::from_static("elsewhere"),
        );
        assert!(!is_constraint_failure(&odd));
    }
}
