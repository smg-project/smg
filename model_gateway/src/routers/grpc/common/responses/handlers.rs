//! Shared response handlers for both regular and harmony implementations
//!
//! These handlers are used by both pipelines for retrieving and cancelling responses.

use axum::response::Response;

use super::{background, ResponsesContext};

/// Implementation for POST /v1/responses/{response_id}/cancel
///
/// A background response this gateway runs is stopped and answered as
/// `cancelled`; see [`background::cancel`] for the other cases.
pub(crate) async fn cancel_response_impl(ctx: &ResponsesContext, response_id: &str) -> Response {
    background::cancel(ctx, response_id).await
}
