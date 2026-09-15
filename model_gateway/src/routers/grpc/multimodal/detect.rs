//! Multimodal content detection and extraction.
//!
//! Both the chat completion pipeline (`ChatMessage`) and the Messages API
//! pipeline (`InputMessage`) funnel into the shared processing core; only the
//! detection and extraction differ, because the input message types differ.

use std::collections::HashMap;

use llm_multimodal::{ImageDetail, MediaContentPart, ToolResultOrder};
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::ContentPart,
    messages::{ImageSource, InputContent, InputContentBlock, InputMessage, Role},
};

use super::plan::MediaPlan;

/// Extract media parts from OpenAI chat messages,
/// converting protocol `ContentPart` to multimodal crate `MediaContentPart`.
#[derive(Clone)]
enum RenderedChatMedia {
    Media(MediaContentPart),
    ToolResult {
        tool_call_id: String,
        media: Vec<MediaContentPart>,
    },
}

fn extract_media_parts(
    messages: &[ChatMessage],
    tool_result_order: ToolResultOrder,
) -> Vec<MediaContentPart> {
    let mut parts = Vec::new();
    let mut user_run = Vec::new();
    let mut last_tool_call_order = HashMap::new();

    for msg in messages {
        match msg {
            ChatMessage::User { content, .. } => {
                user_run.extend(
                    chat_content_media(content)
                        .into_iter()
                        .map(RenderedChatMedia::Media),
                );
            }
            ChatMessage::Tool {
                content,
                tool_call_id,
            } => {
                user_run.push(RenderedChatMedia::ToolResult {
                    tool_call_id: tool_call_id.clone(),
                    media: chat_content_media(content),
                });
            }
            ChatMessage::Assistant { tool_calls, .. } => {
                flush_chat_user_run(
                    &mut parts,
                    &mut user_run,
                    &last_tool_call_order,
                    tool_result_order,
                );
                // Match the DeepSeek encoder: `Some`, including an empty
                // tool-call list, replaces the active ordering; `None` does not.
                if let Some(tool_calls) = tool_calls {
                    last_tool_call_order.clear();
                    for (index, call) in tool_calls.iter().enumerate() {
                        last_tool_call_order.insert(call.id.clone(), index);
                    }
                }
            }
            ChatMessage::System { content, .. }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::Root { content, .. } => {
                flush_chat_user_run(
                    &mut parts,
                    &mut user_run,
                    &last_tool_call_order,
                    tool_result_order,
                );
                parts.extend(chat_content_media(content));
            }
            ChatMessage::Function { .. } => {
                flush_chat_user_run(
                    &mut parts,
                    &mut user_run,
                    &last_tool_call_order,
                    tool_result_order,
                );
            }
        }
    }
    flush_chat_user_run(
        &mut parts,
        &mut user_run,
        &last_tool_call_order,
        tool_result_order,
    );

    parts
}

fn flush_chat_user_run(
    parts: &mut Vec<MediaContentPart>,
    user_run: &mut Vec<RenderedChatMedia>,
    last_tool_call_order: &HashMap<String, usize>,
    tool_result_order: ToolResultOrder,
) {
    let tool_indices = user_run
        .iter()
        .enumerate()
        .filter_map(|(index, block)| {
            matches!(block, RenderedChatMedia::ToolResult { .. }).then_some(index)
        })
        .collect::<Vec<_>>();
    if tool_result_order == ToolResultOrder::AssistantCall
        && tool_indices.len() > 1
        && !last_tool_call_order.is_empty()
    {
        let mut sorted = tool_indices
            .iter()
            .map(|&index| user_run[index].clone())
            .collect::<Vec<_>>();
        sorted.sort_by_key(|block| match block {
            RenderedChatMedia::ToolResult { tool_call_id, .. } => {
                last_tool_call_order.get(tool_call_id).copied().unwrap_or(0)
            }
            RenderedChatMedia::Media(_) => usize::MAX,
        });
        for (index, block) in tool_indices.into_iter().zip(sorted) {
            user_run[index] = block;
        }
    }
    for block in user_run.drain(..) {
        match block {
            RenderedChatMedia::Media(media) => parts.push(media),
            RenderedChatMedia::ToolResult { media, .. } => parts.extend(media),
        }
    }
}

fn chat_content_media(content: &MessageContent) -> Vec<MediaContentPart> {
    let MessageContent::Parts(message_parts) = content else {
        return Vec::new();
    };
    message_parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::ImageUrl { image_url } => {
                let detail = image_url.detail.as_deref().and_then(parse_detail);
                Some(MediaContentPart::ImageUrl {
                    url: image_url.url.clone(),
                    detail,
                    uuid: None,
                    max_long_side_pixel: image_url.max_long_side_pixel,
                })
            }
            ContentPart::Text { .. } => None,
            // Chat preparation rejects unknown parts before building the media plan.
            ContentPart::Unknown(_) => None,
            ContentPart::AudioUrl { audio_url } => Some(MediaContentPart::AudioUrl {
                url: audio_url.url.clone(),
                uuid: None,
            }),
            ContentPart::InputAudio { input_audio } => Some(MediaContentPart::AudioUrl {
                url: format!(
                    "data:audio/{};base64,{}",
                    input_audio.format, input_audio.data
                ),
                uuid: None,
            }),
            ContentPart::VideoUrl { video_url } => Some(MediaContentPart::VideoUrl {
                url: video_url.url.clone(),
                uuid: None,
                fps: video_url.fps,
                max_long_side_pixel: video_url.max_long_side_pixel,
            }),
        })
        .collect()
}

/// Build the canonical ordered media plan for Chat Completions input.
pub(crate) fn media_plan_chat(
    messages: &[ChatMessage],
    tool_result_order: ToolResultOrder,
) -> MediaPlan {
    MediaPlan::new(extract_media_parts(messages, tool_result_order))
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
#[derive(Clone)]
enum RenderedMessageMedia {
    Image(MediaContentPart),
    ToolResult {
        tool_use_id: String,
        images: Vec<MediaContentPart>,
    },
}

fn extract_media_parts_messages(messages: &[InputMessage]) -> Vec<MediaContentPart> {
    let mut parts = Vec::new();
    let mut user_run = Vec::new();
    let mut last_tool_call_order = HashMap::new();

    for msg in messages {
        if msg.role != Role::User {
            flush_messages_user_run(&mut parts, &mut user_run, &last_tool_call_order);
            if msg.role == Role::Assistant {
                let InputContent::Blocks(blocks) = &msg.content else {
                    continue;
                };
                let tool_use_ids = blocks
                    .iter()
                    .filter_map(|block| match block {
                        InputContentBlock::ToolUse(tool_use) => Some(tool_use.id.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                // Match the DeepSeek encoder: an assistant turn only replaces
                // the active ordering when it actually carries tool calls.
                if !tool_use_ids.is_empty() {
                    last_tool_call_order.clear();
                    for (index, id) in tool_use_ids.into_iter().enumerate() {
                        last_tool_call_order.insert(id.to_string(), index);
                    }
                }
            }
            continue;
        }
        let blocks = match &msg.content {
            InputContent::Blocks(blocks) => blocks,
            InputContent::String(_) => continue,
        };

        // The Messages adapter renders ordinary user parts first, followed by
        // tool-result messages. Build the transport plan in that same order so
        // the nth decoded image always backs the nth rendered placeholder.
        for block in blocks {
            if let InputContentBlock::Image(image) = block {
                user_run.push(RenderedMessageMedia::Image(messages_image(&image.source)));
            }
        }

        for result in blocks.iter().filter_map(|block| match block {
            InputContentBlock::ToolResult(result) => Some(result),
            _ => None,
        }) {
            let mut images = Vec::new();
            if let Some(openai_protocol::messages::ToolResultContent::Blocks(blocks)) =
                &result.content
            {
                for block in blocks {
                    if let openai_protocol::messages::ToolResultContentBlock::Image(image) = block {
                        images.push(messages_image(&image.source));
                    }
                }
            }
            user_run.push(RenderedMessageMedia::ToolResult {
                tool_use_id: result.tool_use_id.clone(),
                images,
            });
        }
    }
    flush_messages_user_run(&mut parts, &mut user_run, &last_tool_call_order);

    parts
}

fn flush_messages_user_run(
    parts: &mut Vec<MediaContentPart>,
    user_run: &mut Vec<RenderedMessageMedia>,
    last_tool_call_order: &HashMap<String, usize>,
) {
    let tool_indices = user_run
        .iter()
        .enumerate()
        .filter_map(|(index, block)| {
            matches!(block, RenderedMessageMedia::ToolResult { .. }).then_some(index)
        })
        .collect::<Vec<_>>();
    if tool_indices.len() > 1 && !last_tool_call_order.is_empty() {
        let mut sorted = tool_indices
            .iter()
            .map(|&index| user_run[index].clone())
            .collect::<Vec<_>>();
        sorted.sort_by_key(|block| match block {
            RenderedMessageMedia::ToolResult { tool_use_id, .. } => {
                last_tool_call_order.get(tool_use_id).copied().unwrap_or(0)
            }
            RenderedMessageMedia::Image(_) => usize::MAX,
        });
        for (index, block) in tool_indices.into_iter().zip(sorted) {
            user_run[index] = block;
        }
    }
    for block in user_run.drain(..) {
        match block {
            RenderedMessageMedia::Image(image) => parts.push(image),
            RenderedMessageMedia::ToolResult { images, .. } => parts.extend(images),
        }
    }
}

fn messages_image(source: &ImageSource) -> MediaContentPart {
    let url = match source {
        ImageSource::Base64 { media_type, data } => {
            format!("data:{media_type};base64,{data}")
        }
        ImageSource::Url { url } => url.clone(),
    };
    MediaContentPart::ImageUrl {
        url,
        detail: None,
        uuid: None,
        max_long_side_pixel: None,
    }
}

/// Build the canonical ordered media plan for Messages API input.
pub(crate) fn media_plan_messages(messages: &[InputMessage]) -> MediaPlan {
    MediaPlan::new(extract_media_parts_messages(messages))
}

#[cfg(test)]
mod tests {
    use llm_multimodal::Modality;
    use openai_protocol::common::{AudioUrl, ImageUrl, InputAudio, VideoUrl};

    use super::*;

    #[test]
    fn messages_tool_result_first_images_follow_rendered_order() {
        let messages: Vec<InputMessage> = serde_json::from_value(serde_json::json!([{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
                        {"type": "image", "source": {"type": "url", "url": "https://example.test/nested.png"}}
                    ]
                },
                {"type": "image", "source": {"type": "url", "url": "https://example.test/top.png"}},
                {"type": "image", "source": {"type": "url", "url": "https://example.test/last.png"}}
            ]
        }]))
        .unwrap();
        let parts = media_plan_messages(&messages).into_parts();
        let urls = parts
            .iter()
            .map(|part| match part {
                MediaContentPart::ImageUrl { url, .. } => url.as_str(),
                other => panic!("expected image URL, got {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            urls,
            [
                "https://example.test/top.png",
                "https://example.test/last.png",
                "data:image/png;base64,AA==",
                "https://example.test/nested.png",
            ]
        );
    }

    #[test]
    fn messages_reordered_tool_results_follow_rendered_call_order() {
        let messages: Vec<InputMessage> = serde_json::from_value(serde_json::json!([
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tool-1", "name": "first", "input": {}},
                    {"type": "tool_use", "id": "tool-2", "name": "second", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-2",
                        "content": [{"type": "image", "source": {"type": "url", "url": "https://example.test/tool-2.png"}}]
                    },
                    {"type": "image", "source": {"type": "url", "url": "https://example.test/top.png"}},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": [{"type": "image", "source": {"type": "url", "url": "https://example.test/tool-1.png"}}]
                    }
                ]
            }
        ]))
        .unwrap();
        let urls = media_plan_messages(&messages)
            .into_parts()
            .into_iter()
            .map(|part| match part {
                MediaContentPart::ImageUrl { url, .. } => url,
                other => panic!("expected image URL, got {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            urls,
            [
                "https://example.test/top.png",
                "https://example.test/tool-1.png",
                "https://example.test/tool-2.png",
            ]
        );
    }

    #[test]
    fn messages_reordered_tool_results_across_user_run_follow_rendered_order() {
        let messages: Vec<InputMessage> = serde_json::from_value(serde_json::json!([
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tool-1", "name": "first", "input": {}},
                    {"type": "tool_use", "id": "tool-2", "name": "second", "input": {}}
                ]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tool-2",
                    "content": [{"type": "image", "source": {"type": "url", "url": "https://example.test/tool-2.png"}}]
                }]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.test/top.png"}},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": [{"type": "image", "source": {"type": "url", "url": "https://example.test/tool-1.png"}}]
                    }
                ]
            }
        ]))
        .unwrap();
        let urls = media_plan_messages(&messages)
            .into_parts()
            .into_iter()
            .map(|part| match part {
                MediaContentPart::ImageUrl { url, .. } => url,
                other => panic!("expected image URL, got {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            urls,
            [
                "https://example.test/tool-1.png",
                "https://example.test/top.png",
                "https://example.test/tool-2.png",
            ]
        );
    }

    #[test]
    fn chat_tool_result_media_order_follows_rendering_policy() {
        let messages: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "first", "arguments": "{}"}
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "second", "arguments": "{}"}
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call-2",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/call-2.png"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/call-1.png"}
                }]
            }
        ]))
        .unwrap();

        for (policy, expected) in [
            (
                ToolResultOrder::AssistantCall,
                [
                    "https://example.test/call-1.png",
                    "https://example.test/call-2.png",
                ],
            ),
            (
                ToolResultOrder::default(),
                [
                    "https://example.test/call-2.png",
                    "https://example.test/call-1.png",
                ],
            ),
        ] {
            let urls = media_plan_chat(&messages, policy)
                .into_parts()
                .into_iter()
                .map(|part| match part {
                    MediaContentPart::ImageUrl { url, .. } => url,
                    other => panic!("expected image URL, got {other:?}"),
                })
                .collect::<Vec<_>>();
            assert_eq!(urls, expected, "policy: {policy:?}");
        }
    }

    #[test]
    fn media_plan_detects_image() {
        let messages = vec![ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "What is this?".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/cat.jpg".to_string(),
                        detail: None,
                        max_long_side_pixel: None,
                    },
                },
            ]),
            name: None,
        }];

        assert_eq!(
            media_plan_chat(&messages, ToolResultOrder::Authored).modalities(),
            &[Modality::Image]
        );
    }

    #[test]
    fn media_plan_detects_video() {
        let messages = vec![ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Parts(vec![ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "https://example.com/clip.mp4".to_string(),
                    fps: None,
                    max_long_side_pixel: None,
                },
            }]),
            name: None,
        }];

        assert_eq!(
            media_plan_chat(&messages, ToolResultOrder::Authored).modalities(),
            &[Modality::Video]
        );
    }

    #[test]
    fn media_plan_detects_audio() {
        let messages = vec![ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Parts(vec![ContentPart::AudioUrl {
                audio_url: AudioUrl {
                    url: "https://example.com/clip.wav".to_string(),
                },
            }]),
            name: None,
        }];

        assert_eq!(
            media_plan_chat(&messages, ToolResultOrder::Authored).modalities(),
            &[Modality::Audio]
        );
    }

    #[test]
    fn media_plan_is_empty_for_string_text() {
        let messages = vec![ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Text("Hello".to_string()),
            name: None,
        }];

        assert!(media_plan_chat(&messages, ToolResultOrder::Authored).is_empty());
    }

    #[test]
    fn media_plan_is_empty_for_text_parts() {
        let messages = vec![ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Parts(vec![ContentPart::Text {
                text: "Just text".to_string(),
            }]),
            name: None,
        }];

        assert!(media_plan_chat(&messages, ToolResultOrder::Authored).is_empty());
    }

    #[test]
    fn extracts_image_media_part() {
        let messages = vec![
            ChatMessage::System {
                ext: Default::default(),
                content: MessageContent::Text("You are helpful".to_string()),
                name: None,
            },
            ChatMessage::User {
                ext: Default::default(),
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Describe this:".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: Some("high".to_string()),
                            max_long_side_pixel: None,
                        },
                    },
                ]),
                name: None,
            },
        ];

        let parts = extract_media_parts(&messages, ToolResultOrder::Authored);
        assert_eq!(parts.len(), 1);

        match &parts[0] {
            MediaContentPart::ImageUrl { url, detail, .. } => {
                assert_eq!(url, "https://example.com/image.jpg");
                assert_eq!(*detail, Some(ImageDetail::High));
            }
            _ => panic!("Expected ImageUrl part"),
        }
    }

    #[test]
    fn extracts_video_media_part() {
        let messages = vec![ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Parts(vec![ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "https://example.com/video.mp4".to_string(),
                    fps: None,
                    max_long_side_pixel: None,
                },
            }]),
            name: None,
        }];

        let parts = extract_media_parts(&messages, ToolResultOrder::Authored);
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            MediaContentPart::VideoUrl { url, .. } => {
                assert_eq!(url, "https://example.com/video.mp4");
            }
            _ => panic!("Expected VideoUrl part"),
        }
    }

    #[test]
    fn extracts_audio_url_media_part() {
        let messages = vec![ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Parts(vec![ContentPart::AudioUrl {
                audio_url: AudioUrl {
                    url: "https://example.com/audio.wav".to_string(),
                },
            }]),
            name: None,
        }];

        let parts = extract_media_parts(&messages, ToolResultOrder::Authored);
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            MediaContentPart::AudioUrl { url, .. } => {
                assert_eq!(url, "https://example.com/audio.wav");
            }
            _ => panic!("Expected AudioUrl part"),
        }
    }

    #[test]
    fn extracts_inline_audio_as_data_url() {
        let messages = vec![ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Parts(vec![ContentPart::InputAudio {
                input_audio: InputAudio {
                    data: "UklGRg==".to_string(),
                    format: "wav".to_string(),
                },
            }]),
            name: None,
        }];

        assert_eq!(
            media_plan_chat(&messages, ToolResultOrder::Authored).modalities(),
            &[Modality::Audio]
        );

        let parts = extract_media_parts(&messages, ToolResultOrder::Authored);
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            MediaContentPart::AudioUrl { url, .. } => {
                assert_eq!(url, "data:audio/wav;base64,UklGRg==");
            }
            _ => panic!("Expected AudioUrl part"),
        }
    }

    #[test]
    fn test_parse_detail() {
        assert_eq!(parse_detail("auto"), Some(ImageDetail::Auto));
        assert_eq!(parse_detail("Auto"), Some(ImageDetail::Auto));
        assert_eq!(parse_detail("LOW"), Some(ImageDetail::Low));
        assert_eq!(parse_detail("high"), Some(ImageDetail::High));
        assert_eq!(parse_detail("unknown"), None);
    }
}
