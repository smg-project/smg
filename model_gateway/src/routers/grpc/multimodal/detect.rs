//! Multimodal content detection and extraction.
//!
//! Both the chat completion pipeline (`ChatMessage`) and the Messages API
//! pipeline (`InputMessage`) funnel into the shared processing core; only the
//! detection and extraction differ, because the input message types differ.

use llm_multimodal::{ImageDetail, MediaContentPart};
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::ContentPart,
    messages::{ImageBlock, ImageSource, InputContent, InputContentBlock, InputMessage, Role},
};

use super::plan::MediaPlan;
use crate::routers::grpc::utils::message_utils::tool_result_image_blocks;

/// Extract media parts from OpenAI chat messages,
/// converting protocol `ContentPart` to multimodal crate `MediaContentPart`.
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
                            url: image_url.url.clone(),
                            detail,
                            uuid: None,
                            max_long_side_pixel: image_url.max_long_side_pixel,
                        });
                    }
                    ContentPart::AudioUrl { audio_url } => {
                        parts.push(MediaContentPart::AudioUrl {
                            url: audio_url.url.clone(),
                            uuid: None,
                        });
                    }
                    ContentPart::InputAudio { input_audio } => {
                        parts.push(MediaContentPart::AudioUrl {
                            url: format!(
                                "data:audio/{};base64,{}",
                                input_audio.format, input_audio.data
                            ),
                            uuid: None,
                        });
                    }
                    ContentPart::VideoUrl { video_url } => {
                        parts.push(MediaContentPart::VideoUrl {
                            url: video_url.url.clone(),
                            uuid: None,
                            fps: video_url.fps,
                            max_long_side_pixel: video_url.max_long_side_pixel,
                        });
                    }
                    ContentPart::Text { .. } => {}
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
                InputContentBlock::Image(image_block) => parts.push(image_media_part(image_block)),
                // Images inside a tool result enter the plan where the tool
                // result stands, in block order: `convert_user_message` renders
                // them into the user content at that same position.
                InputContentBlock::ToolResult(tool_result) => {
                    parts.extend(tool_result_image_blocks(tool_result).map(image_media_part));
                }
                InputContentBlock::Text(_) => {}
                _ => {}
            }
        }
    }

    parts
}

/// Convert a Messages API image block to a media part for the connector,
/// turning a base64 source into a data URL.
fn image_media_part(image_block: &ImageBlock) -> MediaContentPart {
    let url = match &image_block.source {
        ImageSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
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

        assert_eq!(media_plan_chat(&messages).modalities(), &[Modality::Image]);
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

        assert_eq!(media_plan_chat(&messages).modalities(), &[Modality::Video]);
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

        assert_eq!(media_plan_chat(&messages).modalities(), &[Modality::Audio]);
    }

    #[test]
    fn media_plan_is_empty_for_string_text() {
        let messages = vec![ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Text("Hello".to_string()),
            name: None,
        }];

        assert!(media_plan_chat(&messages).is_empty());
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

        assert!(media_plan_chat(&messages).is_empty());
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

        let parts = extract_media_parts(&messages);
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

        let parts = extract_media_parts(&messages);
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

        let parts = extract_media_parts(&messages);
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

        assert_eq!(media_plan_chat(&messages).modalities(), &[Modality::Audio]);

        let parts = extract_media_parts(&messages);
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

    #[test]
    fn media_plan_messages_counts_images_inside_tool_results() {
        use openai_protocol::messages::{
            TextBlock, ToolResultBlock, ToolResultContent, ToolResultContentBlock,
        };

        let image = |data: &str| ImageBlock {
            source: ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: data.to_string(),
            },
            cache_control: None,
        };
        let messages = vec![InputMessage {
            role: Role::User,
            content: InputContent::Blocks(vec![
                InputContentBlock::ToolResult(ToolResultBlock {
                    tool_use_id: "tu_1".to_string(),
                    content: Some(ToolResultContent::Blocks(vec![
                        ToolResultContentBlock::Text(TextBlock {
                            text: "screenshot taken".to_string(),
                            cache_control: None,
                            citations: None,
                        }),
                        ToolResultContentBlock::Image(image("AAAA")),
                    ])),
                    is_error: None,
                    cache_control: None,
                }),
                InputContentBlock::Image(image("BBBB")),
            ]),
        }];

        let plan = media_plan_messages(&messages);
        assert_eq!(plan.modalities(), &[Modality::Image]);
        assert_eq!(plan.count(Modality::Image), 2);
        // Block order: the tool result's image comes before the top-level one.
        let urls: Vec<&str> = plan
            .parts()
            .iter()
            .map(|part| match part {
                MediaContentPart::ImageUrl { url, .. } => url.as_str(),
                _ => panic!("tool result images are image parts"),
            })
            .collect();
        assert_eq!(
            urls,
            ["data:image/png;base64,AAAA", "data:image/png;base64,BBBB"]
        );
    }

    #[test]
    fn media_plan_messages_ignores_text_only_tool_results() {
        use openai_protocol::messages::{ToolResultBlock, ToolResultContent};

        let messages = vec![InputMessage {
            role: Role::User,
            content: InputContent::Blocks(vec![InputContentBlock::ToolResult(ToolResultBlock {
                tool_use_id: "tu_1".to_string(),
                content: Some(ToolResultContent::String("4".to_string())),
                is_error: None,
                cache_control: None,
            })]),
        }];

        assert!(media_plan_messages(&messages).is_empty());
    }
}
