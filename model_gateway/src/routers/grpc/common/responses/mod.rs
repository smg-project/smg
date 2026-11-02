//! Shared response functionality used by both regular and harmony implementations

pub(crate) mod background;
pub(crate) mod context;
pub(crate) mod handlers;
pub(crate) mod history;
pub(crate) mod streaming;
pub(crate) mod utils;

// Re-export commonly used items
pub(crate) use background::{spawn_unstored_sweeper, BackgroundResponses, BackgroundStream};
pub(crate) use context::ResponsesContext;
pub(crate) use history::load_history;
pub(crate) use streaming::{
    await_stream_startup, body_guards, build_sse_response, build_sse_response_from_stream,
    build_sse_response_guarded, signal_stream_startup, stream_startup_channel, StreamStartupSender,
};
pub(crate) use utils::{
    ensure_mcp_connection, persist_response_if_needed, previous_response_chain, stored_foreground,
    NotStored,
};

pub(crate) use crate::routers::common::mcp_utils::collect_user_function_names;

#[cfg(test)]
pub(crate) mod test_support {
    //! What the responses modules' tests share.

    use openai_protocol::responses::{ResponseStatus, ResponsesRequest, ResponsesResponse};
    use serde_json::json;

    /// The terminal object of `request` with one output message.
    pub(crate) fn completed_response(id: &str, request: &ResponsesRequest) -> ResponsesResponse {
        let message = serde_json::from_value(json!({
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}]
        }))
        .unwrap();
        ResponsesResponse::builder(id, &request.model)
            .copy_from_request(request)
            .created_at(chrono::Utc::now().timestamp())
            .completed_at(chrono::Utc::now().timestamp())
            .status(ResponseStatus::Completed)
            .add_output(message)
            .build()
    }
}
