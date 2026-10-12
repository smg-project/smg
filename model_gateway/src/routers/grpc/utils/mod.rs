//! Shared utilities for gRPC routers.

mod chat_utils;
mod logprobs;
pub(crate) mod message_utils;
mod metrics;
mod parsers;
mod pd_bootstrap;
pub(crate) mod stream_failure;
pub(crate) mod tonic_ext;

// Re-export all public items so consumer imports stay unchanged.
pub use chat_utils::{create_stop_decoder, process_chat_messages};
pub(crate) use chat_utils::{
    encode_blocking, encode_prompt_blocking, filter_chat_request_by_tool_choice,
    filter_tools_by_tool_choice, generate_tool_call_id, get_history_tool_calls_count,
    missing_tokenizer_response, parse_finish_reason, parse_json_schema_response,
    process_chat_messages_with_placeholders, request_as_rendered, resolve_tokenizer,
    send_error_sse, validate_chat_content_parts,
};
pub(crate) use logprobs::{
    convert_generate_input_logprobs, convert_generate_output_logprobs, convert_proto_logprobs,
    convert_proto_to_openai_logprobs,
};
pub(crate) use metrics::{error_type_from_status, route_to_endpoint};
// `pub` (not `pub(crate)`) so the Go bindings can reuse the gateway's arming
// predicate instead of duplicating it.
pub use parsers::chat_reasoning_starts_in_prefill;
pub(crate) use parsers::{
    check_reasoning_parser_availability, check_tool_parser_availability, continues_final_assistant,
    create_reasoning_parser, create_tool_parser, get_tool_parser,
    messages_reasoning_starts_in_prefill, reasoning_parser_requires_special_tokens,
    reasoning_starts_in_prefill, rendered_prompt_reasoning_in_prefill,
    should_mark_reasoning_started, tool_constraints_for_worker, ParserResolver,
};
pub(crate) use pd_bootstrap::account_pd_leg_failure;
pub(crate) use stream_failure::{send_stream_failure, StreamFailure, StreamSource};

/// One WARN per choice whose stream recovered from a rewritten decode prefix
/// (see `smg_detokenize_prefix_mismatch_total`): the client got the text as
/// streamed plus the remainder, not the batch text.
pub(crate) fn warn_detokenize_recoveries(
    request_id: &str,
    index: u32,
    decoder: &llm_tokenizer::StopSequenceDecoder,
) {
    let recoveries = decoder.prefix_mismatches();
    if recoveries > 0 {
        tracing::warn!(
            request_id,
            index,
            recoveries,
            "the decoder rewrote text already streamed; the stream kept its text and sent the remainder"
        );
    }
}
