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
                    url: "https://example.com/cat.jpg".to_string().into(),
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
                url: "https://example.com/clip.mp4".to_string().into(),
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
                url: "https://example.com/clip.wav".to_string().into(),
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
                        url: "https://example.com/image.jpg".to_string().into(),
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
            assert_eq!(url.as_str(), "https://example.com/image.jpg");
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
                url: "https://example.com/video.mp4".to_string().into(),
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
            assert_eq!(url.as_str(), "https://example.com/video.mp4");
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
                url: "https://example.com/audio.wav".to_string().into(),
            },
        }]),
        name: None,
    }];

    let parts = extract_media_parts(&messages);
    assert_eq!(parts.len(), 1);
    match &parts[0] {
        MediaContentPart::AudioUrl { url, .. } => {
            assert_eq!(url.as_str(), "https://example.com/audio.wav");
        }
        _ => panic!("Expected AudioUrl part"),
    }
}

#[test]
fn extracts_inline_audio_as_a_shared_base64_source() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![ContentPart::InputAudio {
            input_audio: InputAudio {
                data: "UklGRg==".to_string().into(),
                format: "wav".to_string(),
            },
        }]),
        name: None,
    }];
    let request_data = request_text(&messages);

    assert_eq!(media_plan_chat(&messages).modalities(), &[Modality::Audio]);

    let parts = extract_media_parts(&messages);
    assert_eq!(parts.len(), 1);
    match &parts[0] {
        MediaContentPart::AudioBase64 {
            media_type, data, ..
        } => {
            assert_eq!(media_type, "audio/wav");
            assert_eq!(data.as_str(), "UklGRg==");
            assert!(
                std::ptr::eq(data.as_ptr(), request_data),
                "the plan copied the audio text"
            );
        }
        _ => panic!("Expected AudioBase64 part"),
    }
}

/// The start of the text the first part of the first message carries:
/// its url, or its base64 data.
fn request_text(messages: &[ChatMessage]) -> *const u8 {
    let ChatMessage::User {
        content: MessageContent::Parts(parts),
        ..
    } = &messages[0]
    else {
        panic!("a user message with parts")
    };
    match &parts[0] {
        ContentPart::ImageUrl { image_url } => image_url.url.as_ptr(),
        ContentPart::InputAudio { input_audio } => input_audio.data.as_ptr(),
        _ => panic!("an image or audio part"),
    }
}

#[test]
fn the_plan_shares_the_request_image_text() {
    let data_url = format!("data:image/png;base64,{}", "A".repeat(4096));
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: data_url.into(),
                detail: None,
                max_long_side_pixel: None,
            },
        }]),
        name: None,
    }];
    let request_url = request_text(&messages);

    let plan = media_plan_chat(&messages);
    match &plan.parts()[0] {
        MediaContentPart::ImageUrl { url, .. } => assert!(
            std::ptr::eq(url.as_ptr(), request_url),
            "the plan copied the payload"
        ),
        _ => panic!("Expected ImageUrl part"),
    }
}

#[test]
fn the_plan_shares_a_messages_base64_image_text() {
    let payload = "A".repeat(4096);
    let messages = vec![InputMessage {
        role: Role::User,
        content: InputContent::Blocks(vec![InputContentBlock::Image(ImageBlock {
            source: ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: payload.into(),
            },
            cache_control: None,
        })]),
    }];
    let InputContent::Blocks(blocks) = &messages[0].content else {
        panic!("blocks")
    };
    let InputContentBlock::Image(ImageBlock {
        source: ImageSource::Base64 { data, .. },
        ..
    }) = &blocks[0]
    else {
        panic!("a base64 image block")
    };
    let request_data = data.as_ptr();

    let plan = media_plan_messages(&messages);
    match &plan.parts()[0] {
        MediaContentPart::ImageBase64 {
            media_type, data, ..
        } => {
            assert_eq!(media_type, "image/png");
            assert!(
                std::ptr::eq(data.as_ptr(), request_data),
                "the plan re-encoded the payload"
            );
        }
        _ => panic!("Expected ImageBase64 part"),
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

/// A file image makes no media part, as this gateway cannot fetch the
/// file: the plan holds the other images only, and the message
/// conversion refuses the file image by name.
#[test]
fn media_plan_messages_leaves_a_file_image_out() {
    let messages = vec![InputMessage {
        role: Role::User,
        content: InputContent::Blocks(vec![
            InputContentBlock::Image(ImageBlock {
                source: ImageSource::File {
                    file_id: "file_1".to_string(),
                },
                cache_control: None,
            }),
            InputContentBlock::Image(ImageBlock {
                source: ImageSource::Url {
                    url: "https://example.com/chart.png".to_string().into(),
                },
                cache_control: None,
            }),
        ]),
    }];

    let plan = media_plan_messages(&messages);
    assert_eq!(plan.count(Modality::Image), 1);
}

#[test]
fn media_plan_messages_counts_images_inside_tool_results() {
    use openai_protocol::messages::{
        TextBlock, ToolResultBlock, ToolResultContent, ToolResultContentBlock,
    };

    let image = |data: &str| ImageBlock {
        source: ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: data.to_string().into(),
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
    let payloads: Vec<&str> = plan
        .parts()
        .iter()
        .map(|part| match part {
            MediaContentPart::ImageBase64 { data, .. } => data.as_str(),
            _ => panic!("tool result images are base64 image parts"),
        })
        .collect();
    assert_eq!(payloads, ["AAAA", "BBBB"]);
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

#[test]
fn a_media_type_with_a_comma_keeps_the_data_url_form() {
    let messages = vec![InputMessage {
        role: Role::User,
        content: InputContent::Blocks(vec![InputContentBlock::Image(ImageBlock {
            source: ImageSource::Base64 {
                media_type: "image/png,x".to_string(),
                data: "AAAA".to_string().into(),
            },
            cache_control: None,
        })]),
    }];
    match &media_plan_messages(&messages).parts()[0] {
        MediaContentPart::ImageUrl { url, .. } => {
            assert_eq!(url.as_str(), "data:image/png,x;base64,AAAA");
        }
        _ => panic!("expected the data URL form"),
    }
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![ContentPart::InputAudio {
            input_audio: InputAudio {
                data: "UklGRg==".to_string().into(),
                format: "wav,x".to_string(),
            },
        }]),
        name: None,
    }];
    match &extract_media_parts(&messages)[0] {
        MediaContentPart::AudioUrl { url, .. } => {
            assert_eq!(url.as_str(), "data:audio/wav,x;base64,UklGRg==");
        }
        _ => panic!("expected the data URL form"),
    }
}
