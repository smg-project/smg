//! Utility functions for FFI

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::chat::ChatCompletionRequest;
use smg::routers::grpc::utils::chat_reasoning_starts_in_prefill;
use uuid::Uuid;

/// Helper function to generate tool call ID (matches router implementation)
pub fn generate_tool_call_id(
    model: &str,
    function_name: &str,
    index: usize,
    history_tool_calls_count: usize,
) -> String {
    if model.to_lowercase().contains("kimi") {
        // KimiK2 format: functions.{name}:{global_index}
        format!(
            "functions.{}:{}",
            function_name,
            history_tool_calls_count + index
        )
    } else {
        // Standard OpenAI format: call_{24-char-uuid}
        format!("call_{}", &Uuid::now_v7().simple().to_string()[..24])
    }
}

/// Determine whether the SGLang gRPC request should ask the backend to count
/// reasoning tokens for this chat request, from the request alone: the guess
/// of the entry point that renders nothing
/// (`sgl_chat_requires_reasoning_with_tokenizer`). The streaming path renders
/// and decides from its rendered prompt (`chat_stream::start_chat_stream`).
pub(crate) fn chat_requires_reasoning(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> bool {
    chat_reasoning_starts_in_prefill(request, tokenizer)
}
