//! PreviousInteractionLoading step.
//!
//! Transition: LoadPreviousInteraction → BuildRequest

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::gemini::{
    context::RequestContext,
    state::{RequestState, StepResult},
};

/// Load previous interactions if `previous_interaction_id` is set.
///
/// If the request does not specify a `previous_interaction_id`, this step is
/// a no-op and simply advances the state. Otherwise, returns 501 until
/// interaction storage is implemented. Planned behavior: load the stored
/// interaction chain and prepend the conversation history so that the model
/// sees the full context.
///
/// ## Reads
/// - `ctx.input.original_request.previous_interaction_id` — the link to prior interaction.
///
/// ## Writes
/// - `ctx.state` → `BuildRequest`.
pub(crate) async fn previous_interaction_loading(
    ctx: &mut RequestContext,
) -> Result<StepResult, Response> {
    // TODO: implement previous interaction loading
    //
    // 1. Check ctx.input.original_request.previous_interaction_id.
    // 2. If None: no-op, advance state.
    // 3. If Some(id):
    //    a. Retrieve the stored interaction by id from the interaction store.
    //    b. Follow the previous_interaction_id chain to collect the full
    //       conversation history (or rely on the store returning the full chain).
    //    c. Prepend the loaded turns to the request input so that the upstream
    //       worker receives the complete conversation context.
    //    d. On retrieval error: return Err with a suitable error response.

    if ctx.input.original_request.previous_interaction_id.is_some() {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            "previous interaction loading not yet implemented",
        )
            .into_response());
    }

    ctx.state = RequestState::BuildRequest;
    Ok(StepResult::Continue)
}
