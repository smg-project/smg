//! OpenAI Responses API bridge.
//!
//! Conversion logic between MCP protocol types and OpenAI Responses API
//! shapes. Gateway-internal — `smg-mcp` does not depend on OpenAI vocabulary.

pub mod connect;
pub mod format_descriptor;
pub mod format_registry;
pub mod overrides;
pub mod response_format;
pub mod tool_descriptors;
pub mod transformer;

pub use connect::{connect_dynamic_server, connect_static_server};
pub use format_descriptor::{
    descriptor, format_from_type_str, is_hosted_tool_call_item_type, FormatDescriptor,
};
pub use format_registry::{lookup_tool_format, FormatRegistry};
pub use overrides::{apply_hosted_tool_overrides, extract_hosted_tool_overrides};
pub use response_format::ResponseFormat;
pub use tool_descriptors::{
    build_mcp_tool_infos, builtin_type_for_response_tool, chat_function_tools,
    configure_response_tools_approval, function_tools_json, inject_client_visible_mcp_output_items,
    mcp_list_tools_item, mcp_list_tools_json, response_tools, should_hide_output_item_json,
    should_hide_tool_json,
};
pub use transformer::{
    compact_image_generation_output, compact_image_generation_outputs_json,
    extract_embedded_openai_responses, mcp_response_item_id, transform_tool_output,
    ResponseTransformer,
};
