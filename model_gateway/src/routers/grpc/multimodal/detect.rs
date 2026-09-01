//! Multimodal content detection and extraction.
//!
//! Both the chat completion pipeline (`ChatMessage`) and the Messages API
//! pipeline (`InputMessage`) funnel into the shared processing core; only the
//! detection and extraction differ, because the input message types differ.

use llm_multimodal::{ImageDetail, MediaContentPart};
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::ContentPart,
    messages::{ImageSource, InputContent, InputContentBlock, InputMessage, Role},
};

use super::plan::MediaPlan;

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
            _ => None,
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
                    ContentPart::Text { .. } => {}
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
                InputContentBlock::Image(image_block) => match &image_block.source {
                    ImageSource::Base64 { media_type, data } => {
                        // Convert base64 to data URL for the media connector
                        let data_url = format!("data:{media_type};base64,{data}");
                        parts.push(MediaContentPart::ImageUrl {
                            url: data_url,
                            detail: None,
                            uuid: None,
                            max_long_side_pixel: None,
                        });
                    }
                    ImageSource::Url { url } => {
                        parts.push(MediaContentPart::ImageUrl {
                            url: url.clone(),
                            detail: None,
                            uuid: None,
                            max_long_side_pixel: None,
                        });
                    }
                },
                InputContentBlock::Text(_) => {}
                _ => {}
            }
        }
    }

    parts
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
            content: MessageContent::Text("Hello".to_string()),
            name: None,
        }];

        assert!(media_plan_chat(&messages).is_empty());
    }

    #[test]
    fn media_plan_is_empty_for_text_parts() {
        let messages = vec![ChatMessage::User {
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
                content: MessageContent::Text("You are helpful".to_string()),
                name: None,
            },
            ChatMessage::User {
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
}
