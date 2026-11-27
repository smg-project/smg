//! Multimodal content detection and extraction.
//!
//! Both the chat completion pipeline (`ChatMessage`) and the Messages API
//! pipeline (`InputMessage`) funnel into the shared processing core; only the
//! detection and extraction differ, because the input message types differ.

use std::sync::Arc;

use llm_multimodal::{ImageDetail, MediaContentPart};
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::ContentPart,
    messages::{ImageBlock, ImageSource, InputContent, InputContentBlock, InputMessage, Role},
};

use super::plan::MediaPlan;
use crate::routers::grpc::utils::message_utils::tool_result_image_blocks;

/// Extract media parts from OpenAI chat messages, converting protocol
/// `ContentPart` to multimodal crate `MediaContentPart`; a part's text (a URL,
/// a data URL, base64 audio) is shared with the request, not copied.
fn extract_media_parts(messages: &[ChatMessage]) -> Vec<MediaContentPart> {
    let mut parts = Vec::new();

    for msg in messages {
        let content = match msg {
            ChatMessage::User { content, .. } => Some(content),
            ChatMessage::System { content, .. } => Some(content),
            ChatMessage::Developer { content, .. } => Some(content),
            ChatMessage::Tool { content, .. } => Some(content),
            ChatMessage::Root { content, .. } => Some(content),
            ChatMessage::Assistant { .. } | ChatMessage::Function { .. } => None,
        };

        if let Some(MessageContent::Parts(message_parts)) = content {
            for part in message_parts {
                match part {
                    ContentPart::ImageUrl { image_url } => {
                        let detail = image_url.detail.as_deref().and_then(parse_detail);
                        parts.push(MediaContentPart::ImageUrl {
                            url: Arc::clone(&image_url.url),
                            detail,
                            uuid: None,
                            max_long_side_pixel: image_url.max_long_side_pixel,
                        });
                    }
                    ContentPart::AudioUrl { audio_url } => {
                        parts.push(MediaContentPart::AudioUrl {
                            url: Arc::clone(&audio_url.url),
                            uuid: None,
                        });
                    }
                    ContentPart::InputAudio { input_audio }
                        if media_type_is_plain(&input_audio.format) =>
                    {
                        parts.push(MediaContentPart::AudioBase64 {
                            media_type: format!("audio/{}", input_audio.format),
                            data: Arc::clone(&input_audio.data),
                            uuid: None,
                        });
                    }
                    ContentPart::InputAudio { input_audio } => {
                        parts.push(MediaContentPart::AudioUrl {
                            url: Arc::new(format!(
                                "data:audio/{};base64,{}",
                                input_audio.format, input_audio.data
                            )),
                            uuid: None,
                        });
                    }
                    ContentPart::VideoUrl { video_url } => {
                        parts.push(MediaContentPart::VideoUrl {
                            url: Arc::clone(&video_url.url),
                            uuid: None,
                            fps: video_url.fps,
                            max_long_side_pixel: video_url.max_long_side_pixel,
                        });
                    }
                    ContentPart::Text { .. } | ContentPart::Refusal { .. } => {}
                    // Chat preparation rejects unknown parts before building the media plan.
                    ContentPart::Unknown(_) => {}
                }
            }
        }
    }

    parts
}

/// Build the canonical ordered media plan for Chat Completions input.
pub(crate) fn media_plan_chat(messages: &[ChatMessage]) -> MediaPlan {
    MediaPlan::new(extract_media_parts(messages))
}

/// Parse OpenAI detail string to multimodal ImageDetail enum.
fn parse_detail(detail: &str) -> Option<ImageDetail> {
    match detail.to_ascii_lowercase().as_str() {
        "auto" => Some(ImageDetail::Auto),
        "low" => Some(ImageDetail::Low),
        "high" => Some(ImageDetail::High),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Messages API multimodal detection and extraction
// ---------------------------------------------------------------------------

/// Extract media parts from Messages API input messages,
/// converting `InputContentBlock::Image` to multimodal crate `MediaContentPart`.
fn extract_media_parts_messages(messages: &[InputMessage]) -> Vec<MediaContentPart> {
    let mut parts = Vec::new();

    for msg in messages {
        if msg.role != Role::User {
            continue;
        }
        let blocks = match &msg.content {
            InputContent::Blocks(blocks) => blocks,
            InputContent::String(_) => continue,
        };

        for block in blocks {
            match block {
                InputContentBlock::Image(image_block) => {
                    parts.extend(image_media_part(image_block));
                }
                // Images inside a tool result enter the plan where the tool
                // result stands, in block order: `convert_user_message` renders
                // them into the user content at that same position.
                InputContentBlock::ToolResult(tool_result) => {
                    parts
                        .extend(tool_result_image_blocks(tool_result).filter_map(image_media_part));
                }
                InputContentBlock::Text(_) => {}
                _ => {}
            }
        }
    }

    parts
}

/// Whether a base64 part may stand for the data URL `data:<media_type>;base64,<data>`:
/// a `,` in the media type would move the delimiter a data URL parser splits
/// on, so such a type keeps the data URL form and its parsing.
fn media_type_is_plain(media_type: &str) -> bool {
    !media_type.contains(',')
}

/// Convert a Messages API image block to a media part for the connector: a
/// base64 source travels as its media type and its text, shared with the
/// request; a URL is shared the same way. A file source names a file of the
/// public API's Files API, which this gateway cannot fetch, so it makes no
/// part: the message conversion refuses the block by name.
fn image_media_part(image_block: &ImageBlock) -> Option<MediaContentPart> {
    Some(match &image_block.source {
        ImageSource::Base64 { media_type, data } if media_type_is_plain(media_type) => {
            MediaContentPart::ImageBase64 {
                media_type: media_type.clone(),
                data: Arc::clone(data),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            }
        }
        ImageSource::Base64 { media_type, data } => MediaContentPart::ImageUrl {
            url: Arc::new(format!("data:{media_type};base64,{data}")),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        },
        ImageSource::Url { url } => MediaContentPart::ImageUrl {
            url: Arc::clone(url),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        },
        ImageSource::File { .. } => return None,
    })
}

/// Build the canonical ordered media plan for Messages API input.
pub(crate) fn media_plan_messages(messages: &[InputMessage]) -> MediaPlan {
    MediaPlan::new(extract_media_parts_messages(messages))
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/multimodal/detect/tests.rs"]
mod tests;
