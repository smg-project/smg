//! Per-format dispatch table — one row per `ResponseFormat`.
//!
//! Routers read fields off [`FormatDescriptor`] instead of matching on the
//! enum directly, so adding a new format is one row here and zero edits in
//! router code.

use openai_protocol::event_types::{
    CodeInterpreterCallEvent, FileSearchCallEvent, ImageGenerationCallEvent, ItemType, McpEvent,
    WebSearchCallEvent,
};

use super::ResponseFormat;

#[derive(Debug, Clone, Copy)]
pub struct FormatDescriptor {
    pub type_str: &'static str,
    /// Item-id kind without trailing `_` (e.g. `"ws"`, `"mcp"`). Joined to a
    /// suffix at construction sites: `format!("{kind}_{rest}")`. Stored
    /// without the underscore so callers that just need the discriminator
    /// (allocator prefix, log labels) don't have to trim.
    pub id_prefix: &'static str,
    pub in_progress_event: &'static str,
    /// `None` for formats with no intermediate phase (e.g. Passthrough).
    pub searching_event: Option<&'static str>,
    pub completed_event: &'static str,
    pub streams_arguments: bool,
    /// `None` for formats without a partial-image-style intermediate frame.
    pub partial_image_event: Option<&'static str>,
}

/// Reverse lookup for routers that have already received an item-type
/// string (`mcp_call`, `web_search_call`, …) from the upstream wire and
/// need to recover the matching `ResponseFormat` to consult the
/// descriptor. Returns `None` for non-format item types like
/// `function_call` or `message`.
pub fn format_from_type_str(type_str: &str) -> Option<ResponseFormat> {
    match type_str {
        ItemType::MCP_CALL => Some(ResponseFormat::Passthrough),
        ItemType::WEB_SEARCH_CALL => Some(ResponseFormat::WebSearchCall),
        ItemType::CODE_INTERPRETER_CALL => Some(ResponseFormat::CodeInterpreterCall),
        ItemType::FILE_SEARCH_CALL => Some(ResponseFormat::FileSearchCall),
        ItemType::IMAGE_GENERATION_CALL => Some(ResponseFormat::ImageGenerationCall),
        _ => None,
    }
}

/// True iff `item_type` maps to a non-Passthrough `ResponseFormat`.
pub fn is_hosted_tool_call_item_type(item_type: &str) -> bool {
    format_from_type_str(item_type).is_some_and(|f| f.to_builtin_tool_type().is_some())
}

pub const fn descriptor(format: ResponseFormat) -> FormatDescriptor {
    match format {
        ResponseFormat::WebSearchCall => FormatDescriptor {
            type_str: ItemType::WEB_SEARCH_CALL,
            id_prefix: "ws",
            in_progress_event: WebSearchCallEvent::IN_PROGRESS,
            searching_event: Some(WebSearchCallEvent::SEARCHING),
            completed_event: WebSearchCallEvent::COMPLETED,
            streams_arguments: false,
            partial_image_event: None,
        },
        ResponseFormat::CodeInterpreterCall => FormatDescriptor {
            type_str: ItemType::CODE_INTERPRETER_CALL,
            id_prefix: "ci",
            in_progress_event: CodeInterpreterCallEvent::IN_PROGRESS,
            searching_event: Some(CodeInterpreterCallEvent::INTERPRETING),
            completed_event: CodeInterpreterCallEvent::COMPLETED,
            streams_arguments: false,
            partial_image_event: None,
        },
        ResponseFormat::FileSearchCall => FormatDescriptor {
            type_str: ItemType::FILE_SEARCH_CALL,
            id_prefix: "fs",
            in_progress_event: FileSearchCallEvent::IN_PROGRESS,
            searching_event: Some(FileSearchCallEvent::SEARCHING),
            completed_event: FileSearchCallEvent::COMPLETED,
            streams_arguments: false,
            partial_image_event: None,
        },
        ResponseFormat::ImageGenerationCall => FormatDescriptor {
            type_str: ItemType::IMAGE_GENERATION_CALL,
            id_prefix: "ig",
            in_progress_event: ImageGenerationCallEvent::IN_PROGRESS,
            searching_event: Some(ImageGenerationCallEvent::GENERATING),
            completed_event: ImageGenerationCallEvent::COMPLETED,
            streams_arguments: false,
            partial_image_event: Some(ImageGenerationCallEvent::PARTIAL_IMAGE),
        },
        ResponseFormat::Passthrough => FormatDescriptor {
            type_str: ItemType::MCP_CALL,
            id_prefix: "mcp",
            in_progress_event: McpEvent::CALL_IN_PROGRESS,
            searching_event: None,
            completed_event: McpEvent::CALL_COMPLETED,
            streams_arguments: true,
            partial_image_event: None,
        },
    }
}
