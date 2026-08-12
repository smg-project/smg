//! Shared utilities for gRPC routers.

mod chat_utils;
mod logprobs;
pub(crate) mod message_utils;
mod metrics;
mod parsers;
pub(crate) mod tonic_ext;

// Re-export all public items so consumer imports stay unchanged.
pub use chat_utils::{create_stop_decoder, process_chat_messages};
pub(crate) use chat_utils::{
    encode_blocking, filter_chat_request_by_tool_choice, filter_tools_by_tool_choice,
    generate_tool_call_id, get_history_tool_calls_count, parse_finish_reason,
    parse_json_schema_response, process_chat_messages_with_placeholders, resolve_tokenizer,
    send_error_sse,
};
pub(crate) use logprobs::{
    convert_generate_input_logprobs, convert_generate_output_logprobs, convert_proto_logprobs,
    convert_proto_to_openai_logprobs,
};
pub(crate) use metrics::{error_type_from_status, route_to_endpoint};
pub(crate) use parsers::{
    check_reasoning_parser_availability, check_tool_parser_availability, create_reasoning_parser,
    create_tool_parser, get_tool_parser, reasoning_parser_requires_special_tokens, ParserResolver,
};
// `pub` (not `pub(crate)`) so the Go bindings can reuse the gateway's reasoning
// detection instead of duplicating it.
pub use parsers::{resolve_user_thinking, should_mark_reasoning_started};
