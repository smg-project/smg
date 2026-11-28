//! Utility functions for FFI

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::chat::ChatCompletionRequest;
use smg::routers::grpc::utils::chat_reasoning_starts_in_prefill;

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
