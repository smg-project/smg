use llm_tokenizer::{chat_template::ChatTemplateContentFormat, traits::RendererCapabilities};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{AudioUrl, ContentPart, ImageUrl, InputAudio, VideoUrl},
};
use serde_json::json;

use super::*;

/// The error frame is `data: {"error":{"message":..,"type":..}}` with
/// the message escaped as JSON, whatever it carries.
#[tokio::test]
async fn an_sse_error_frame_carries_the_escaped_message_and_its_type() {
    let (tx, mut rx) = sse::sse_channel();
    send_error_sse(&tx, "a \"quoted\"\nline \u{1F600}", "stream_error").await;
    send_error_sse(&tx, format!("built {}", 42), "internal_error").await;
    drop(tx);
    let mut frames = Vec::new();
    while let Some(Ok(bytes)) = rx.recv().await {
        frames.push(String::from_utf8(bytes.to_vec()).unwrap());
    }
    assert_eq!(
        frames,
        [
            "data: {\"error\":{\"message\":\"a \\\"quoted\\\"\\nline \u{1F600}\",\
                 \"type\":\"stream_error\"}}\n\n",
            "data: {\"error\":{\"message\":\"built 42\",\"type\":\"internal_error\"}}\n\n",
        ]
    );
}

fn placeholders(entries: &[(Modality, &str)]) -> PlaceholderTokens {
    let mut tokens = PlaceholderTokens::default();
    for (modality, token) in entries {
        tokens.insert(*modality, (*token).to_string());
    }
    tokens
}

#[test]
fn unknown_chat_content_parts_are_rejected_in_every_role() {
    for role in ["system", "user", "assistant", "tool", "developer"] {
        let message: ChatMessage = serde_json::from_value(json!({
            "role": role,
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "vendor_media", "payload": {"id": "media_1"}}
            ]
        }))
        .unwrap();
        let error = validate_chat_content_parts(&[message]).unwrap_err();
        assert!(error.contains("vendor_media"), "{role}: {error}");
        assert!(error.contains("gRPC"), "{role}: {error}");
    }
}

#[test]
fn process_chat_messages_rejects_unknown_content_parts() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this attachment"},
            {"type": "vendor_media", "payload": "media_1"}
        ]}]
    }))
    .unwrap();
    let tokenizer = llm_tokenizer::MockTokenizer::new();
    let error = process_chat_messages(&request, &tokenizer, None).unwrap_err();
    assert!(error.contains("vendor_media"));
}

#[test]
fn known_chat_content_parts_are_accepted() {
    let messages: Vec<ChatMessage> = serde_json::from_value(json!([
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.wav"}},
            {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]},
        {"role": "assistant", "content": null},
        {"role": "function", "name": "lookup", "content": "result"}
    ]))
    .unwrap();
    assert!(validate_chat_content_parts(&messages).is_ok());
}

#[test]
fn test_transform_messages_string_format() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Hello".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: None,
                    max_long_side_pixel: None,
                },
            },
            ContentPart::Text {
                text: "World".to_string(),
            },
        ]),
        name: None,
    }];

    let tokens = placeholders(&[(Modality::Image, "<|image|>")]);
    let result =
        process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
            .unwrap();

    assert_eq!(result.len(), 1);
    let transformed_message = &result[0];

    // Media is hoisted and uses the modality-specific placeholder.
    assert_eq!(
        transformed_message["content"].as_str().unwrap(),
        "<|image|>\nHello\nWorld"
    );
    assert_eq!(transformed_message["role"].as_str().unwrap(), "user");
}

#[test]
fn test_transform_messages_string_format_without_placeholders_omits_media() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Describe this".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.png".to_string(),
                    detail: None,
                    max_long_side_pixel: None,
                },
            },
        ]),
        name: None,
    }];

    let result =
        process_content_format(&messages, ChatTemplateContentFormat::String, None).unwrap();

    assert_eq!(result[0]["content"], "Describe this");
}

#[test]
fn test_transform_messages_string_format_with_video_placeholder() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Watch this".to_string(),
            },
            ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "https://example.com/video.mp4".to_string(),
                    fps: None,
                    max_long_side_pixel: None,
                },
            },
        ]),
        name: None,
    }];

    let tokens = placeholders(&[(Modality::Video, "<|video|>")]);
    let result =
        process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
            .unwrap();

    // Media placeholder is emitted before the text (vLLM front placement).
    assert_eq!(
        result[0]["content"].as_str().unwrap(),
        "<|video|>\nWatch this"
    );
}

#[test]
fn test_transform_messages_string_format_uses_per_modality_placeholders() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Describe and transcribe".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "image".to_string(),
                    detail: None,
                    max_long_side_pixel: None,
                },
            },
            ContentPart::AudioUrl {
                audio_url: AudioUrl {
                    url: "audio".to_string(),
                },
            },
        ]),
        name: None,
    }];
    let tokens = placeholders(&[(Modality::Image, "<image>"), (Modality::Audio, "<audio>")]);

    let result =
        process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
            .unwrap();

    assert_eq!(
        result[0]["content"],
        "<image>\n<audio>\nDescribe and transcribe"
    );

    let openai =
        process_content_format(&messages, ChatTemplateContentFormat::OpenAI, Some(&tokens))
            .unwrap();
    assert_eq!(
        openai[0]["content"],
        json!([
            {"type": "text", "text": "Describe and transcribe"},
            {"type": "image"},
            {"type": "audio"}
        ])
    );
}

#[test]
fn test_transform_messages_input_audio_uses_audio_placeholder() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Transcribe this".to_string(),
            },
            ContentPart::InputAudio {
                input_audio: InputAudio {
                    data: "UklGRg==".to_string(),
                    format: "wav".to_string(),
                },
            },
        ]),
        name: None,
    }];
    let tokens = placeholders(&[(Modality::Audio, "<audio>")]);

    let string =
        process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
            .unwrap();
    assert_eq!(string[0]["content"], "<audio>\nTranscribe this");

    let openai =
        process_content_format(&messages, ChatTemplateContentFormat::OpenAI, Some(&tokens))
            .unwrap();
    assert_eq!(
        openai[0]["content"],
        json!([
            {"type": "text", "text": "Transcribe this"},
            {"type": "audio"}
        ])
    );
}

#[test]
fn test_transform_messages_string_format_rejects_missing_modality_placeholder() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![ContentPart::AudioUrl {
            audio_url: AudioUrl {
                url: "audio".to_string(),
            },
        }]),
        name: None,
    }];
    let tokens = placeholders(&[(Modality::Image, "<image>")]);

    let error = process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
        .unwrap_err();

    assert!(error.contains("missing audio placeholder"));
}

#[test]
fn test_transform_messages_openai_format() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Describe this image:".to_string(),
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
    }];

    let result =
        process_content_format(&messages, ChatTemplateContentFormat::OpenAI, None).unwrap();

    assert_eq!(result.len(), 1);
    let transformed_message = &result[0];

    // Media URLs replaced with simple type placeholders; the parts stay
    // where the request put them (the template writes the anchor there).
    let content_array = transformed_message["content"].as_array().unwrap();
    assert_eq!(content_array.len(), 2);
    assert_eq!(content_array[0]["type"], "text");
    assert_eq!(content_array[0]["text"], "Describe this image:");
    assert_eq!(content_array[1], json!({"type": "image"}));
}

#[test]
fn test_transform_messages_simple_string_content() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Text("Simple text message".to_string()),
        name: None,
    }];

    let tokens = placeholders(&[(Modality::Image, "<|image|>")]);
    let result =
        process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
            .unwrap();

    assert_eq!(result.len(), 1);
    let transformed_message = &result[0];

    // Simple string content should remain unchanged
    assert_eq!(
        transformed_message["content"].as_str().unwrap(),
        "Simple text message"
    );
}

#[test]
fn test_transform_messages_multiple_messages() {
    let messages = vec![
        ChatMessage::System {
            ext: Default::default(),
            content: MessageContent::Text("System prompt".to_string()),
            name: None,
        },
        ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "User message".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: None,
                        max_long_side_pixel: None,
                    },
                },
            ]),
            name: None,
        },
    ];

    let tokens = placeholders(&[(Modality::Image, "<|image|>")]);
    let result =
        process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
            .unwrap();

    assert_eq!(result.len(), 2);

    // System message should remain unchanged
    assert_eq!(result[0]["role"].as_str().unwrap(), "system");
    assert_eq!(result[0]["content"].as_str().unwrap(), "System prompt");

    // User message retains the media anchor.
    assert_eq!(result[1]["role"].as_str().unwrap(), "user");
    assert_eq!(
        result[1]["content"].as_str().unwrap(),
        "<|image|>\nUser message"
    );
}

#[test]
fn test_transform_messages_empty_text_parts() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/image.jpg".to_string(),
                detail: None,
                max_long_side_pixel: None,
            },
        }]),
        name: None,
    }];

    let tokens = placeholders(&[(Modality::Image, "<|image|>")]);
    let result =
        process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
            .unwrap();

    assert_eq!(result.len(), 1);
    let transformed_message = &result[0];

    assert_eq!(transformed_message["content"], "<|image|>");
}

#[test]
fn test_transform_messages_mixed_content_types() {
    let messages = vec![
        ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Text("Plain text".to_string()),
            name: None,
        },
        ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "With image".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: Some("low".to_string()),
                        max_long_side_pixel: None,
                    },
                },
            ]),
            name: None,
        },
    ];

    let tokens = placeholders(&[(Modality::Image, "<|image|>")]);
    let result_string =
        process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
            .unwrap();

    assert_eq!(result_string.len(), 2);
    assert_eq!(result_string[0]["content"].as_str().unwrap(), "Plain text");
    assert_eq!(
        result_string[1]["content"].as_str().unwrap(),
        "<|image|>\nWith image"
    );

    let result_openai =
        process_content_format(&messages, ChatTemplateContentFormat::OpenAI, None).unwrap();

    assert_eq!(result_openai.len(), 2);
    assert_eq!(result_openai[0]["content"].as_str().unwrap(), "Plain text");

    let content_array = result_openai[1]["content"].as_array().unwrap();
    assert_eq!(content_array.len(), 2);
    // The request's order: the text, then the image.
    assert_eq!(content_array[0]["type"], "text");
    assert_eq!(content_array[1], json!({"type": "image"}));
}

#[test]
fn parts_reach_a_parts_template_in_the_requests_order_whatever_the_media_order() {
    // A template that takes parts places them itself: a text part before
    // the image stays before it, under the media-first default too (the
    // engine's own server hands such a template the parts as given).
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Question: ...\nAnswer with only the option letter.".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/jpeg;base64,XXX".to_string(),
                    detail: None,
                    max_long_side_pixel: None,
                },
            },
        ]),
        name: None,
    }];

    let result =
        process_content_format(&messages, ChatTemplateContentFormat::OpenAI, None).unwrap();
    let arr = result[0]["content"].as_array().unwrap();
    assert_eq!(arr[0]["type"], "text");
    assert_eq!(arr[1], json!({"type": "image"}));
}

#[test]
fn test_media_hoisted_before_text_string() {
    // String-format template: placeholder prepended, matching vLLM exactly.
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Question?".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/jpeg;base64,XXX".to_string(),
                    detail: None,
                    max_long_side_pixel: None,
                },
            },
        ]),
        name: None,
    }];

    let tokens = placeholders(&[(Modality::Image, "<|image_pad|>")]);
    let result =
        process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
            .unwrap();
    assert_eq!(
        result[0]["content"].as_str().unwrap(),
        "<|image_pad|>\nQuestion?"
    );
}

#[test]
fn test_media_first_stable_and_multi() {
    // String template: every media part first, each group in its own
    // order, joined by newlines as the engine's own server joins them;
    // parts template: the request's order, text and media interleaved.
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "a".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "i1".to_string(),
                    detail: None,
                    max_long_side_pixel: None,
                },
            },
            ContentPart::Text {
                text: "b".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "i2".to_string(),
                    detail: None,
                    max_long_side_pixel: None,
                },
            },
        ]),
        name: None,
    }];

    let tokens = placeholders(&[(Modality::Image, "<image>")]);
    let result =
        process_content_format(&messages, ChatTemplateContentFormat::String, Some(&tokens))
            .unwrap();
    assert_eq!(result[0]["content"], "<image>\n<image>\na\nb");

    let result =
        process_content_format(&messages, ChatTemplateContentFormat::OpenAI, None).unwrap();
    assert_eq!(
        result[0]["content"],
        json!([
            {"type": "text", "text": "a"},
            {"type": "image"},
            {"type": "text", "text": "b"},
            {"type": "image"}
        ])
    );
}

/// A chat template that walks a message's parts (the shape of the
/// vision families' templates: one anchor written where each image part
/// sits, the text where it sits).
const PARTS_TEMPLATE: &str = r"
{%- for m in messages -%}
<|turn|>{{ m.role }}<|body|>
{%- if m.content is string -%}{{ m.content }}
{%- else -%}{%- for part in m.content -%}
{%- if part.type == 'image' -%}<|image|>{%- elif part.type == 'text' -%}{{ part.text }}{%- endif -%}
{%- endfor -%}{%- endif -%}<|end|>
{%- endfor -%}
{%- if add_generation_prompt -%}<|turn|>assistant<|body|>{%- endif -%}";

/// A user message of `parts` (`"<img>"` for an image, any other string
/// for a text part) rendered through `PARTS_TEMPLATE` under the
/// media-first default, as the pipeline renders a chat request.
fn render_parts(parts: &[&str]) -> String {
    let content: Vec<Value> = parts
        .iter()
        .map(|part| match *part {
            "<img>" => {
                json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}})
            }
            text => json!({"type": "text", "text": text}),
        })
        .collect();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": content}]
    }))
    .unwrap();
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase).unwrap();
    tokenizer
        .set_chat_template(PARTS_TEMPLATE.to_string())
        .unwrap();
    assert_eq!(
        tokenizer.chat_template_content_format(),
        ChatTemplateContentFormat::OpenAI
    );
    process_chat_messages_with_placeholders(&request, &tokenizer, None, MediaPartOrder::MediaFirst)
        .unwrap()
        .0
        .text
}

/// The rendered prompt follows the request's content-part order on a
/// template that walks the parts: text then image, image then text,
/// images interleaved with text, images then text, and text alone; each
/// equal, byte for byte, to what the template itself renders for the
/// parts in that order.
#[test]
fn a_parts_template_renders_the_parts_where_the_request_put_them() {
    let turn = |body: &str| format!("<|turn|>user<|body|>{body}<|end|><|turn|>assistant<|body|>");
    assert_eq!(
        render_parts(&["What is in this picture?", "<img>"]),
        turn("What is in this picture?<|image|>")
    );
    assert_eq!(
        render_parts(&["<img>", "What is in this picture?"]),
        turn("<|image|>What is in this picture?")
    );
    assert_eq!(
        render_parts(&[
            "Figure 1: ",
            "<img>",
            " Figure 2: ",
            "<img>",
            " Which figure shows a cat?"
        ]),
        turn("Figure 1: <|image|> Figure 2: <|image|> Which figure shows a cat?")
    );
    assert_eq!(
        render_parts(&["<img>", "<img>", "Which picture is brighter?"]),
        turn("<|image|><|image|>Which picture is brighter?")
    );
    assert_eq!(
        render_parts(&["No picture here.", " Just text."]),
        turn("No picture here. Just text.")
    );
}

#[test]
fn test_tml_preserves_authored_multipart_order_openai() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "question".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "image".to_string(),
                    detail: None,
                    max_long_side_pixel: None,
                },
            },
        ]),
        name: None,
    }];

    let expected = json!([
        {"type": "text", "text": "question"},
        {"type": "image"}
    ]);
    let result = process_content_format_with_order(
        &messages,
        ChatTemplateContentFormat::OpenAI,
        None,
        MediaPartOrder::Authored,
    )
    .unwrap();
    assert_eq!(result[0]["content"], expected);
}

// --- an assistant turn without content on the render path ----------------

/// A template of the string format that tests and splits every assistant
/// message's content (a reasoning model's template strips the thought it
/// wrote before its answer this way) and prints the content after the role.
const CONTENT_TESTING_STRING_TEMPLATE: &str = "{%- for message in messages -%}\
{%- if message.role == 'assistant' and '</think>' in message.content -%}\
{%- set content = message.content.split('</think>')[-1].lstrip() -%}\
{%- else -%}{%- set content = message.content -%}{%- endif -%}\
<|{{ message.role }}|>{{ content }}\
{%- if message.tool_calls -%}{%- for call in message.tool_calls -%}<call:{{ call.function.name }}>{%- endfor -%}{%- endif -%}\
<|end|>{%- endfor -%}\
{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

/// A template of the OpenAI format: it walks a message's content parts, and
/// tests an assistant message's content as the string template does.
const CONTENT_TESTING_PARTS_TEMPLATE: &str = "{%- for message in messages -%}\
{%- if message.role == 'assistant' and '</think>' in message.content -%}\
{%- set content = message.content.split('</think>')[-1].lstrip() -%}\
{%- else -%}{%- set content = message.content -%}{%- endif -%}\
<|{{ message.role }}|>\
{%- if content is string -%}{{ content }}\
{%- else -%}{%- for part in content -%}{%- if part.type == 'text' -%}{{ part.text }}{%- endif -%}{%- endfor -%}{%- endif -%}\
{%- if message.tool_calls -%}{%- for call in message.tool_calls -%}<call:{{ call.function.name }}>{%- endfor -%}{%- endif -%}\
<|end|>{%- endfor -%}\
{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

/// A chat request whose history holds a tool-call-only assistant turn (the
/// OpenAI shape: `content` null beside `tool_calls`) and its tool result.
fn request_with_a_tool_call_only_turn() -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "18 celsius"}
        ]
    }))
    .unwrap()
}

/// `request` rendered through `template`, whose detected content format
/// must be `format`.
fn render_through(
    template: &str,
    format: ChatTemplateContentFormat,
    request: &ChatCompletionRequest,
) -> Result<String, String> {
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase).unwrap();
    tokenizer.set_chat_template(template.to_string()).unwrap();
    assert_eq!(tokenizer.chat_template_content_format(), format);
    process_chat_messages_with_placeholders(request, &tokenizer, None, MediaPartOrder::MediaFirst)
        .map(|(processed, _)| processed.text)
}

const TOOL_CALL_ONLY_TURN_RENDERED: &str = "<|user|>What is the weather in Paris?<|end|>\
<|assistant|><call:get_weather><|end|><|tool|>18 celsius<|end|><|assistant|>";

/// A tool-call-only assistant turn in the history renders through a
/// string-format template that tests, splits and prints `message.content`:
/// the turn's content reaches the template as the empty string the engine's
/// chat utils hand it, so the check is false and nothing is printed for it
/// (not the text `None`, and not a render failure).
#[test]
fn a_tool_call_only_assistant_turn_renders_through_a_content_testing_string_template() {
    let text = render_through(
        CONTENT_TESTING_STRING_TEMPLATE,
        ChatTemplateContentFormat::String,
        &request_with_a_tool_call_only_turn(),
    )
    .unwrap();
    assert_eq!(text, TOOL_CALL_ONLY_TURN_RENDERED);
}

/// The same turn through a template that takes content parts: the content
/// reaches it as the empty parts list the engine's chat utils hand such a
/// template, so the check is false and the parts loop writes nothing.
#[test]
fn a_tool_call_only_assistant_turn_renders_through_a_content_testing_parts_template() {
    let text = render_through(
        CONTENT_TESTING_PARTS_TEMPLATE,
        ChatTemplateContentFormat::OpenAI,
        &request_with_a_tool_call_only_turn(),
    )
    .unwrap();
    assert_eq!(text, TOOL_CALL_ONLY_TURN_RENDERED);
}

/// `skip_serializing_none` drops an assistant message's absent content; the
/// render path restores it as the empty content the engine's chat utils
/// hand a template: `""` under the string format, `[]` under the OpenAI
/// format, with or without tool calls and reasoning content beside it. A
/// message with content is left as it is.
#[test]
fn an_absent_assistant_content_is_the_engines_empty_content_in_the_templates_format() {
    let messages: Vec<ChatMessage> = serde_json::from_value(json!([
        {"role": "assistant", "content": null},
        {
            "role": "assistant",
            "content": null,
            "reasoning_content": "a thought",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"}
            }]
        },
        {"role": "assistant", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}
    ]))
    .unwrap();
    // Text parts under the string format are joined into the text, as the
    // engine joins them; a parts template keeps them.
    for (format, empty, parts) in [
        (ChatTemplateContentFormat::String, json!(""), json!("hello")),
        (
            ChatTemplateContentFormat::OpenAI,
            json!([]),
            json!([{"type": "text", "text": "hello"}]),
        ),
    ] {
        let result =
            process_content_format_with_order(&messages, format, None, MediaPartOrder::MediaFirst)
                .unwrap();
        assert_eq!(result[0]["content"], empty, "{format:?}");
        assert_eq!(result[1]["content"], empty, "{format:?}");
        assert_eq!(
            result[1]["reasoning_content"],
            json!("a thought"),
            "{format:?}"
        );
        assert_eq!(
            result[1]["tool_calls"][0]["function"]["name"],
            json!("f"),
            "{format:?}"
        );
        assert_eq!(result[2]["content"], json!("hello"), "{format:?}");
        assert_eq!(result[3]["content"], parts, "{format:?}");
    }
}

#[test]
fn test_tml_preserves_authored_multipart_order_string() {
    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "question".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "image".to_string(),
                    detail: None,
                    max_long_side_pixel: None,
                },
            },
        ]),
        name: None,
    }];
    let tokens = placeholders(&[(Modality::Image, "<image>")]);

    let result = process_content_format_with_order(
        &messages,
        ChatTemplateContentFormat::String,
        Some(&tokens),
        MediaPartOrder::Authored,
    )
    .unwrap();

    assert_eq!(result[0]["content"], "question\n<image>");
}

/// A named `tool_choice` renders every tool (the engine's own server
/// renders them all and forces the call with its grammar), while the
/// constraint still targets the named tool; an `allowed_tools` subset
/// still narrows the rendered list.
#[test]
fn a_named_tool_choice_renders_every_tool() {
    let request = |tool_choice: Value| -> ChatCompletionRequest {
        serde_json::from_value(json!({
                "model": "m",
                "messages": [{"role": "user", "content": "q"}],
                "tools": [
                    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}},
                    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
                ],
                "tool_choice": tool_choice,
            }))
            .expect("chat request")
    };
    let names = |request: &ChatCompletionRequest| -> Vec<String> {
        request
            .tools
            .as_deref()
            .into_iter()
            .flatten()
            .map(|tool| tool.function.name.clone())
            .collect()
    };

    let named = request(json!({"type": "function", "function": {"name": "get_weather"}}));
    assert_eq!(
        names(&request_as_rendered(&named)),
        ["get_weather", "get_time"]
    );
    assert_eq!(
        named
            .callable_tools()
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<Vec<_>>(),
        ["get_weather"]
    );

    let allowed = request(json!({
        "type": "allowed_tools", "mode": "auto",
        "tools": [{"type": "function", "name": "get_time"}]
    }));
    assert_eq!(names(&request_as_rendered(&allowed)), ["get_time"]);

    let auto = request(json!("auto"));
    assert_eq!(
        names(&request_as_rendered(&auto)),
        ["get_weather", "get_time"]
    );
}

fn tool_choice_none_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "q"}],
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}
            ],
            "tool_choice": "none",
        }))
        .expect("chat request")
}

/// `tool_choice: none` renders the tools as `auto` renders them: the request
/// renders as sent (borrowed, nothing narrowed), so the prompt is the one the
/// engine's own server renders for it and the one the session's other turns
/// rendered; the choice itself reaches the template as a kwarg.
#[test]
fn tool_choice_none_renders_the_tools_as_auto_does() {
    let request = tool_choice_none_request();
    let rendered = request_as_rendered(&request);
    assert!(matches!(rendered, std::borrow::Cow::Borrowed(_)));
    assert_eq!(
        rendered.tools.as_deref().map(<[_]>::len),
        Some(1),
        "the tool declaration stays for the template to write"
    );
    let mut auto = request.clone();
    auto.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));
    let under_auto = request_as_rendered(&auto);
    assert_eq!(
        serde_json::to_value(&under_auto.tools).expect("tools"),
        serde_json::to_value(&rendered.tools).expect("tools"),
        "the same tools render under none and under auto"
    );
    assert_eq!(
        build_chat_template_kwargs(&request)[TOOL_CHOICE_KEY],
        json!("none")
    );
}

fn effort_request(reasoning_effort: Option<&str>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "inkling-chat".to_string(),
        messages: vec![ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Text("hello".to_string()),
            name: None,
        }],
        reasoning_effort: reasoning_effort.map(str::to_string),
        ..Default::default()
    }
}

#[test]
fn reasoning_effort_is_forwarded_verbatim() {
    // The chat template owns level->value mapping, defaulting, and validation;
    // the router forwards the string and omits it when absent so the template
    // applies its own default.
    let kwargs = build_chat_template_kwargs(&effort_request(Some("high")));
    assert_eq!(kwargs.get(REASONING_EFFORT_KEY), Some(&json!("high")));

    let kwargs = build_chat_template_kwargs(&effort_request(None));
    assert!(!kwargs.contains_key(REASONING_EFFORT_KEY));
}

/// The request-level `reasoning_effort` is written over a
/// `chat_template_kwargs` entry, as the engine's own server renders it;
/// the other entries pass through, and the entry stands when the request
/// carries no field.
#[test]
fn top_level_effort_overrides_a_chat_template_kwargs_entry() {
    let mut request = effort_request(Some("high"));
    request.chat_template_kwargs = Some(HashMap::from([
        (REASONING_EFFORT_KEY.to_string(), json!("low")),
        ("custom".to_string(), json!(true)),
    ]));
    let kwargs = build_chat_template_kwargs(&request);
    assert_eq!(kwargs.get(REASONING_EFFORT_KEY), Some(&json!("high")));
    assert_eq!(kwargs.get("custom"), Some(&Value::Bool(true)));

    let mut request = effort_request(None);
    request.chat_template_kwargs = Some(HashMap::from([(
        REASONING_EFFORT_KEY.to_string(),
        json!("low"),
    )]));
    let kwargs = build_chat_template_kwargs(&request);
    assert_eq!(kwargs.get(REASONING_EFFORT_KEY), Some(&json!("low")));
}

fn thinking_request(thinking: Value, reasoning_effort: Option<&str>) -> ChatCompletionRequest {
    let mut request = effort_request(reasoning_effort);
    request.thinking = Some(serde_json::from_value(thinking).expect("thinking param"));
    request
}

#[test]
fn thinking_effort_overrides_top_level_effort_in_kwargs() {
    // KVV test_reasoning_effort_ignored_when_effort_present: effort=low beats max.
    let request = thinking_request(json!({"type": "enabled", "effort": "low"}), Some("max"));
    let kwargs = build_chat_template_kwargs(&request);
    assert_eq!(kwargs.get(REASONING_EFFORT_KEY), Some(&json!("low")));

    // Absent `thinking.effort`, the top-level field still applies.
    let request = thinking_request(json!({"type": "enabled"}), Some("max"));
    let kwargs = build_chat_template_kwargs(&request);
    assert_eq!(kwargs.get(REASONING_EFFORT_KEY), Some(&json!("max")));

    // The request-level effort outranks an explicit chat_template_kwargs entry too.
    let mut request = thinking_request(json!({"effort": "low"}), Some("max"));
    request.chat_template_kwargs = Some(HashMap::from([(
        REASONING_EFFORT_KEY.to_string(),
        json!("high"),
    )]));
    let kwargs = build_chat_template_kwargs(&request);
    assert_eq!(kwargs.get(REASONING_EFFORT_KEY), Some(&json!("low")));
}

#[test]
fn thinking_disabled_projects_params_thinking_false() {
    let request = thinking_request(json!({"type": "disabled"}), Some("high"));
    assert_eq!(resolve_template_thinking(&request), Some(false));
    // `adaptive` and an omitted `type` leave the template default in charge.
    for thinking in [json!({"type": "adaptive"}), json!({"keep": "all"})] {
        assert_eq!(
            resolve_template_thinking(&thinking_request(thinking, None)),
            None
        );
    }
    assert_eq!(resolve_template_thinking(&effort_request(None)), None);
    assert_eq!(
        resolve_template_thinking(&effort_request(Some("none"))),
        Some(false)
    );
}

/// Under the MiniMax profile no effort value switches thinking off: the
/// request renders the template's default (thinking on, the model's own
/// depth). An explicit `thinking.disabled` wins over the effort there too.
#[test]
fn minimax_effort_keeps_thinking_on() {
    let minimax = |thinking: Option<Value>, reasoning_effort: Option<&str>| {
        let mut request = effort_request(reasoning_effort);
        request.model = "MiniMax-M3".to_string();
        request.thinking = thinking.map(|value| serde_json::from_value(value).unwrap());
        request
    };
    for effort in ["none", "minimal", "high"] {
        assert_eq!(
            resolve_template_thinking(&minimax(None, Some(effort))),
            None,
            "{effort}"
        );
        assert_eq!(
            resolve_template_thinking(&minimax(Some(json!({"type": "disabled"})), Some(effort))),
            Some(false),
            "{effort}: thinking.disabled wins"
        );
    }
    assert_eq!(
        resolve_template_thinking(&minimax(Some(json!({"type": "disabled"})), None)),
        Some(false)
    );
    assert_eq!(
        resolve_template_thinking(&minimax(Some(json!({"type": "enabled"})), Some("none"))),
        Some(true)
    );
}

#[test]
fn thinking_enabled_beats_reasoning_effort_none() {
    let request = thinking_request(json!({"type": "enabled"}), Some("none"));
    assert_eq!(resolve_template_thinking(&request), Some(true));
    // A `none` inside `thinking.effort` is the switch too when no type is given.
    let request = thinking_request(json!({"effort": "none"}), Some("high"));
    assert_eq!(resolve_template_thinking(&request), Some(false));
}

#[test]
fn tool_and_output_controls_are_forwarded_to_renderer() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hello"}],
        "tool_choice": "required",
        "response_format": {"type": "json_object"}
    }))
    .unwrap();
    let kwargs = build_chat_template_kwargs(&request);
    assert_eq!(kwargs.get(TOOL_CHOICE_KEY), Some(&json!("required")));
    assert_eq!(
        kwargs.get(RESPONSE_FORMAT_KEY),
        Some(&json!({"type": "json_object"}))
    );

    // Absent both, neither key is forwarded.
    let bare: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();
    let kwargs = build_chat_template_kwargs(&bare);
    assert!(!kwargs.contains_key(TOOL_CHOICE_KEY));
    assert!(!kwargs.contains_key(RESPONSE_FORMAT_KEY));
}

/// End-to-end: run a real MMBench-shaped `[text, image]` message through the
/// full SMG pipeline (`process_content_format` + the actual model chat
/// template) and assert the rendered prompt places the image BEFORE the
/// question. Proves the fix end-to-end without needing the GPU worker.
/// Ignored by default (needs the real template); run with:
///   QWEN35_CHAT_TEMPLATE=/path/chat_template.jinja cargo test -p smg \
///     render_image_before_question -- --ignored --nocapture
#[test]
#[ignore = "needs real chat_template.jinja via QWEN35_CHAT_TEMPLATE"]
fn test_render_image_before_question_real_template() {
    use llm_tokenizer::chat_template::{
        detect_chat_template_content_format, ChatTemplateParams, ChatTemplateProcessor,
    };

    let Ok(path) = std::env::var("QWEN35_CHAT_TEMPLATE") else {
        return; // skip when not provided
    };
    let template = std::fs::read_to_string(&path).expect("read template");
    let format = detect_chat_template_content_format(&template);

    let messages = vec![ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Question: Which description is correct?\n\
                           Answer with only the option letter (A/B/C/D)."
                    .to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/jpeg;base64,XXX".to_string(),
                    detail: None,
                    max_long_side_pixel: None,
                },
            },
        ]),
        name: None,
    }];

    let tokens = placeholders(&[(Modality::Image, "<|image_pad|>")]);
    let transformed = process_content_format(&messages, format, Some(&tokens)).unwrap();

    let mut kwargs = HashMap::new();
    kwargs.insert("enable_thinking".to_string(), json!(false));
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        template_kwargs: Some(&kwargs),
        ..Default::default()
    };
    let rendered = ChatTemplateProcessor::new(template)
        .unwrap()
        .apply_chat_template(&transformed, params)
        .unwrap();

    let vstart = rendered
        .find("<|vision_start|>")
        .expect("rendered prompt has <|vision_start|>");
    let qpos = rendered
        .find("Question:")
        .expect("rendered prompt has the question");
    assert!(
            vstart < qpos,
            "image must precede the question (vstart={vstart}, qpos={qpos}).\n--- rendered ---\n{rendered}"
        );
}

#[test]
fn test_generate_tool_call_id_kimi_k3_opaque_format() {
    // K3: `{name}_{history + index}` — no `functions.` prefix, `_` (not K2's
    // `:`) as the separator, and a history offset that keeps ids unique
    // across turns.
    assert_eq!(
        generate_tool_call_id("moonshotai/Kimi-K3", "get_weather", 0, 0),
        "get_weather_0"
    );
    assert_eq!(
        generate_tool_call_id("kimi_k3", "get_weather", 1, 5),
        "get_weather_6"
    );
}

#[test]
fn test_generate_tool_call_id_kimi_k3_continues_across_turns() {
    // A first turn emitting three parallel calls yields 0,1,2; the next
    // turn's three calls continue at 3,4,5 instead of colliding with them.
    // K3 matches tool results back to calls by opaque id equality, so a
    // repeated id would be ambiguous to any caller correlating by id.
    let first: Vec<String> = (0..3)
        .map(|i| generate_tool_call_id("Kimi-K3", "get_weather", i, 0))
        .collect();
    let second: Vec<String> = (0..3)
        .map(|i| generate_tool_call_id("Kimi-K3", "get_weather", i, 3))
        .collect();

    assert_eq!(first, ["get_weather_0", "get_weather_1", "get_weather_2"]);
    assert_eq!(second, ["get_weather_3", "get_weather_4", "get_weather_5"]);
    assert!(
        first.iter().all(|id| !second.contains(id)),
        "ids must not repeat across turns: {first:?} vs {second:?}"
    );
}

#[test]
fn test_generate_tool_call_id_kimi_k2_unchanged() {
    // K2 keeps the globally unique `functions.{name}:{history + index}` form.
    assert_eq!(
        generate_tool_call_id("moonshotai/Kimi-K2-Instruct", "get_weather", 0, 0),
        "functions.get_weather:0"
    );
    assert_eq!(
        generate_tool_call_id("Kimi-K2", "get_weather", 1, 2),
        "functions.get_weather:3"
    );
}

#[test]
fn test_generate_tool_call_id_non_kimi_uses_uuid() {
    let id = generate_tool_call_id("gpt-4o", "get_weather", 0, 0);
    assert!(id.starts_with("call_"), "got: {id}");
    assert!(!id.contains("get_weather"), "got: {id}");
}

// --- render -> encode contract -------------------------------------------

fn prefill_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "m",
        "continue_final_message": true,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Sure"}
        ]
    }))
    .unwrap()
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

#[test]
fn flat_renderer_joins_the_prefill_and_reports_from_text() {
    let tokenizer = llm_tokenizer::MockTokenizer::new();
    let (processed, encoding) = process_chat_messages_with_placeholders(
        &prefill_request(),
        &tokenizer,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    assert_eq!(processed.text, "user: Hello\nassistant: Sure");
    assert!(matches!(encoding, PromptEncoding::FromText));

    // The tokenize step encodes the text exactly as before.
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(tokenizer);
    let ids = block_on(encode_prompt_blocking(
        tokenizer.clone(),
        &processed.text,
        encoding,
    ))
    .unwrap();
    assert_eq!(
        ids.token_ids(),
        tokenizer
            .encode(&processed.text, false)
            .unwrap()
            .token_ids()
    );
}

// --- renderer-capability gates on the render path ------------------------

/// Render `request` through a mock that declares `capabilities` and
/// hands back the message list it received as JSON, so both gates are
/// observable without a checkpoint: which messages reach the template,
/// whether a generation prompt was requested, and whether tool-call
/// `arguments` arrive as written.
fn render_with(capabilities: RendererCapabilities, request: &ChatCompletionRequest) -> Value {
    let tokenizer = llm_tokenizer::MockTokenizer::new()
        .with_renderer_capabilities(capabilities)
        .with_json_chat_template();
    let (processed, _) = process_chat_messages_with_placeholders(
        request,
        &tokenizer,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    // The default `apply_chat_template_with_encoding` appends the
    // assistant prefix after the rendered text, so a prefill that was
    // popped shows up as a suffix on the JSON document.
    let split = processed.text.rfind('}').unwrap() + 1;
    let (rendered, suffix) = processed.text.split_at(split);
    let mut value: Value = serde_json::from_str(rendered).unwrap();
    value["assistant_prefix"] = json!(suffix);
    value
}

const NATIVE_CONTINUATION: RendererCapabilities = RendererCapabilities {
    media_handling: llm_tokenizer::traits::MediaHandling::Preprocess,
    enable_thinking_alias: false,
    native_assistant_continuation: true,
    raw_tool_call_arguments: false,
    leading_system_only: false,
};
const RAW_ARGUMENTS: RendererCapabilities = RendererCapabilities {
    media_handling: llm_tokenizer::traits::MediaHandling::Preprocess,
    enable_thinking_alias: false,
    native_assistant_continuation: false,
    raw_tool_call_arguments: true,
    leading_system_only: false,
};

/// Without the capability the trailing assistant message is popped and its
/// content is appended after a generation prompt; with it the message
/// stays in the list and no generation prompt is requested (the renderer
/// continues the turn itself).
#[test]
fn native_continuation_keeps_the_trailing_assistant_message_instead_of_prefixing_it() {
    let request = prefill_request();

    let default = render_with(Default::default(), &request);
    assert_eq!(default["messages"].as_array().unwrap().len(), 1);
    assert_eq!(default["messages"][0]["role"], "user");
    assert_eq!(default["add_generation_prompt"], json!(true));
    assert_eq!(default["assistant_prefix"], json!("Sure"));

    let native = render_with(NATIVE_CONTINUATION, &request);
    assert_eq!(native["messages"].as_array().unwrap().len(), 2);
    assert_eq!(native["messages"][1]["role"], "assistant");
    assert_eq!(native["messages"][1]["content"], "Sure");
    assert_eq!(native["add_generation_prompt"], json!(false));
    assert_eq!(native["assistant_prefix"], json!(""));
}

/// `add_generation_prompt: false` renders the conversation without a
/// generation header; absent, the header is rendered as before. A continued
/// final message keeps its own rule: no header natively, the emulation's
/// header and prefix otherwise, whatever the field says.
#[test]
fn add_generation_prompt_false_renders_the_conversation_without_a_header() {
    let request = |fields: Value| -> ChatCompletionRequest {
        let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "Hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(body).unwrap()
    };
    let closed = render_with(
        Default::default(),
        &request(json!({"add_generation_prompt": false})),
    );
    assert_eq!(closed["add_generation_prompt"], json!(false));
    assert_eq!(closed["messages"].as_array().unwrap().len(), 1);
    let open = render_with(Default::default(), &request(json!({})));
    assert_eq!(open["add_generation_prompt"], json!(true));

    let pair = request(json!({
        "continue_final_message": true,
        "add_generation_prompt": false,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Sure"}
        ]
    }));
    let emulated = render_with(Default::default(), &pair);
    assert_eq!(emulated["add_generation_prompt"], json!(true));
    assert_eq!(emulated["assistant_prefix"], json!("Sure"));
    let native = render_with(NATIVE_CONTINUATION, &pair);
    assert_eq!(native["add_generation_prompt"], json!(false));
    assert_eq!(native["messages"].as_array().unwrap().len(), 2);
}

/// Through a Jinja template: without the header the prompt ends at the last
/// turn's end marker; with it (the field absent) the assistant header follows.
/// The rendered prompt reports which, for the reasoning decisions. A
/// `chat_template_kwargs` entry of that name does not override the request's
/// field (the render context keeps the field's value), so the header stays
/// and is reported as such.
#[test]
fn jinja_template_renders_no_header_when_add_generation_prompt_is_false() {
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase).unwrap();
    tokenizer
        .set_chat_template(HEADER_TEMPLATE.to_string())
        .unwrap();
    let render = |add_generation_prompt: Value| {
        let mut body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Hello."}],
            "add_generation_prompt": add_generation_prompt
        });
        if let Some(kwargs) = add_generation_prompt.get("chat_template_kwargs") {
            body["chat_template_kwargs"] = kwargs.clone();
            body["add_generation_prompt"] = Value::Null;
        }
        let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        process_chat_messages_with_placeholders(
            &request,
            &tokenizer,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap()
        .0
    };
    let closed = render(json!(false));
    assert_eq!(closed.text, "<|turn|>user<|body|>Hello.<|end|>");
    assert!(!closed.generation_prompt);
    let open = render(Value::Null);
    assert_eq!(
        open.text,
        "<|turn|>user<|body|>Hello.<|end|><|turn|>assistant"
    );
    assert!(open.generation_prompt);
    let by_kwargs = render(json!({"chat_template_kwargs": {"add_generation_prompt": false}}));
    assert_eq!(
        by_kwargs.text,
        "<|turn|>user<|body|>Hello.<|end|><|turn|>assistant"
    );
    assert!(by_kwargs.generation_prompt);
}

/// The rendered prompt reports whether it carries the generation header, as
/// the reasoning decisions read it: the header the request asked for, none
/// under `add_generation_prompt: false`, none for a natively continued
/// message, and the emulation's header (the message's text after it)
/// otherwise.
#[test]
fn the_rendered_prompt_reports_its_generation_header() {
    let processed = |capabilities: RendererCapabilities, body: Value| -> ProcessedMessages {
        let tokenizer = llm_tokenizer::MockTokenizer::new()
            .with_renderer_capabilities(capabilities)
            .with_json_chat_template();
        let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        process_chat_messages_with_placeholders(
            &request,
            &tokenizer,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap()
        .0
    };
    let hello = json!({"model": "m", "messages": [{"role": "user", "content": "Hello"}]});
    assert!(processed(Default::default(), hello.clone()).generation_prompt);
    let mut closed = hello;
    closed["add_generation_prompt"] = json!(false);
    assert!(!processed(Default::default(), closed).generation_prompt);
    let pair = json!({
        "model": "m",
        "continue_final_message": true,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Sure"}
        ]
    });
    let native = processed(NATIVE_CONTINUATION, pair.clone());
    assert!(native.continued_final_message);
    assert!(!native.generation_prompt);
    let emulated = processed(Default::default(), pair);
    assert!(!emulated.continued_final_message);
    assert!(emulated.generation_prompt);
}

/// A text-rendered prompt reaches the tokenizer with each rendered marker
/// already rewritten to its anchor; without placeholders it is left alone.
/// The mock's JSON template writes no marker, so the one here is typed.
#[test]
fn rendered_markers_are_anchored_before_tokenizing() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "https://a/v.mp4"}},
            {"type": "text", "text": "What happens in <video>?"}
        ]}]
    }))
    .unwrap();
    let tokenizer = llm_tokenizer::MockTokenizer::new()
        .with_content_format(ChatTemplateContentFormat::OpenAI)
        .with_json_chat_template();
    let mut placeholders = PlaceholderTokens::default();
    placeholders.insert(Modality::Video, "<SPECIAL_30>".to_string());
    placeholders.add_rendered_marker("<video>".to_string(), "<SPECIAL_30>".to_string());
    let render = |placeholders| {
        process_chat_messages_with_placeholders(
            &request,
            &tokenizer,
            placeholders,
            MediaPartOrder::MediaFirst,
        )
        .unwrap()
        .0
        .text
    };
    let anchored = render(Some(&placeholders));
    assert!(anchored.contains("<SPECIAL_30>") && !anchored.contains("<video>"));
    assert!(render(None).contains("<video>"));
}

/// An assistant turn that opens with a header the generation prompt
/// lacks: the continued text must follow the header.
const HEADER_TEMPLATE: &str = r"
{%- for m in messages -%}
{%- if m.role == 'assistant' -%}{{- '<|turn|>assistant<|to|>user<|body|>' + m.content + '<|end|>' -}}
{%- else -%}{{- '<|turn|>' + m.role + '<|body|>' + m.content + '<|end|>' -}}{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|turn|>assistant' -}}{%- endif -%}";

/// A Jinja template continues the trailing assistant message inside the
/// turn it renders for it, as transformers does.
#[test]
fn jinja_template_continues_inside_the_assistant_turn() {
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase).unwrap();
    tokenizer
        .set_chat_template(HEADER_TEMPLATE.to_string())
        .unwrap();
    let (processed, _) = process_chat_messages_with_placeholders(
        &prefill_request(),
        &tokenizer,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    assert_eq!(
        processed.text,
        "<|turn|>user<|body|>Hello<|end|><|turn|>assistant<|to|>user<|body|>Sure"
    );
    assert!(processed.continued_final_message);
}

/// Under the OpenAI content format the popped assistant message keeps its
/// parts, so the prefill is their text.
#[test]
fn parts_valued_prefill_reaches_the_render_call_under_openai_content_format() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "continue_final_message": true,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Sure"},
                {"type": "text", "text": "!"}
            ]}
        ]
    }))
    .unwrap();
    let tokenizer = llm_tokenizer::MockTokenizer::new()
        .with_content_format(ChatTemplateContentFormat::OpenAI)
        .with_json_chat_template();
    let (processed, _) = process_chat_messages_with_placeholders(
        &request,
        &tokenizer,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    let (rendered, prefix) = processed
        .text
        .split_at(processed.text.rfind('}').unwrap() + 1);
    let rendered: Value = serde_json::from_str(rendered).unwrap();
    assert_eq!(rendered["messages"].as_array().unwrap().len(), 1);
    assert_eq!(rendered["add_generation_prompt"], json!(true));
    assert_eq!(prefix, "Sure!");
}

/// A tool definition carries whatever the client wrote beyond name,
/// description, parameters and strict (the `response` schema of a BFCL
/// tool); the template renders the tool object as given, so the typed
/// request hands it on whole and in the client's order.
#[test]
fn tool_fields_beyond_the_schema_reach_the_template_in_order() {
    let function = json!({
        "name": "authenticate",
        "description": "Authenticate a user.",
        "parameters": {"type": "object", "properties": {"user": {"type": "string"}}},
        "response": {"type": "dict", "properties": {"ok": {"type": "boolean"}}}
    });
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "Log me in"}],
        "tools": [{"type": "function", "function": function.clone()}]
    }))
    .unwrap();

    let rendered = render_with(Default::default(), &request);
    assert_eq!(
        rendered["tools"][0]["function"].to_string(),
        function.to_string()
    );
}

/// Without the capability a tool call's `arguments` string is parsed into
/// an object before rendering (what Transformers templates expect); with
/// it the string reaches the renderer as written, so a native renderer can
/// apply the reference's own tolerance.
#[test]
fn raw_tool_call_arguments_reach_the_renderer_as_written() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{\"city\": \"Hangzhou\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
        ]
    }))
    .unwrap();

    let parsed = render_with(Default::default(), &request);
    assert_eq!(
        parsed["messages"][1]["tool_calls"][0]["function"]["arguments"],
        json!({"city": "Hangzhou"})
    );

    let raw = render_with(RAW_ARGUMENTS, &request);
    assert_eq!(
        raw["messages"][1]["tool_calls"][0]["function"]["arguments"],
        json!("{\"city\": \"Hangzhou\"}")
    );
}

/// A Jinja template that writes a history call's arguments by string
/// concatenation (the DeepSeek V3 line's own templates do) tells the
/// gateway through its renderer capabilities, and the arguments reach it
/// as the request sent them, spelling and all; a template that serializes
/// the object gets the object, as before.
#[test]
fn a_template_that_writes_the_arguments_as_text_gets_them_as_the_request_sent_them() {
    let concatenating = "{% for m in messages %}{% if m.tool_calls %}{% for t in m.tool_calls %}\
                             {{ '<call>' + t.function.name + ':' + t.function.arguments + '</call>' }}\
                             {% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}";
    let serializing = "{% for m in messages %}{% if m.tool_calls %}{% for t in m.tool_calls %}\
                           <call>{{ t.function.name }}:{{ t.function.arguments | tojson }}</call>\
                           {% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}";
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "Weather in Rome?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Rome\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "24"},
            {"role": "user", "content": "And Paris?"}
        ]
    }))
    .unwrap();

    let rendered = |template: &str| {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer.json"),
            r#"{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
                    "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
                    "post_processor": null, "decoder": null,
                    "model": {"type": "BPE", "vocab": {"hello": 0}, "merges": []}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            json!({"chat_template": template}).to_string(),
        )
        .unwrap();
        let tok = llm_tokenizer::create_tokenizer(dir.path().to_str().unwrap()).unwrap();
        let raw = tok.renderer_capabilities().raw_tool_call_arguments;
        (
            raw,
            process_chat_messages(&request, tok.as_ref(), None)
                .unwrap()
                .text,
        )
    };

    let (raw, text) = rendered(concatenating);
    assert!(raw);
    assert_eq!(
        text,
        "Weather in Rome?<call>get_weather:{\"city\":\"Rome\"}</call>24And Paris?"
    );
    let (raw, text) = rendered(serializing);
    assert!(!raw);
    assert_eq!(
        text,
        "Weather in Rome?<call>get_weather:{\"city\": \"Rome\"}</call>24And Paris?"
    );
}

/// Tool-call `arguments` reach the template as the object the engines hand
/// theirs: a JSON text is parsed in its key order; an empty text (a call cut
/// before it wrote any argument may carry one back in history) is a call
/// without arguments; a text that is not JSON, or not an object, is an empty
/// object as well instead of a refusal that would fail every later turn of
/// the conversation; absent arguments are an empty object; an object stays.
#[test]
fn arguments_text_in_history_reaches_the_template_as_the_engines_hand_it() {
    let rendered = |arguments: Value| {
        let mut messages = vec![json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "apply_patch", "arguments": arguments}
            }]
        })];
        process_tool_call_arguments(&mut messages);
        messages[0]["tool_calls"][0]["function"]["arguments"].take()
    };
    let parsed = rendered(json!("{\"patch\": \"x\", \"at\": 1}"));
    assert_eq!(parsed.to_string(), "{\"patch\":\"x\",\"at\":1}");
    for empty in ["", "  \n"] {
        assert_eq!(rendered(json!(empty)), json!({}), "{empty:?}");
    }
    for not_an_object in ["{", "{\"patch\": \"x\"", "[]", "42", "\"x\""] {
        assert_eq!(
            rendered(json!(not_an_object)),
            json!({}),
            "{not_an_object:?}"
        );
    }
    assert_eq!(rendered(Value::Null), json!({}));
    assert_eq!(rendered(json!({"patch": "y"})), json!({"patch": "y"}));
}

/// A history tool call whose `arguments` is the JSON object itself (a client
/// replaying the parsed call) is accepted and renders exactly as the same
/// call with the arguments as text, the shape the API documents: a template
/// gets the object either way, in the request's key order, and a renderer
/// that reads the text itself gets the object's JSON text.
#[test]
fn tool_call_arguments_sent_as_an_object_render_like_the_text() {
    let request = |arguments: Value| -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": arguments}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature\": 18}"}
            ]
        }))
        .expect("a history tool call's arguments may be text or an object")
    };
    let as_object = request(json!({"unit": "celsius", "city": "Paris"}));
    let as_text = request(json!("{\"unit\": \"celsius\", \"city\": \"Paris\"}"));
    let arguments =
        |rendered: Value| rendered["messages"][1]["tool_calls"][0]["function"]["arguments"].clone();

    let object = arguments(render_with(Default::default(), &as_object));
    assert_eq!(
        object.to_string(),
        "{\"unit\":\"celsius\",\"city\":\"Paris\"}"
    );
    assert_eq!(object, arguments(render_with(Default::default(), &as_text)));
    assert_eq!(
        arguments(render_with(RAW_ARGUMENTS, &as_object)),
        json!("{\"unit\":\"celsius\",\"city\":\"Paris\"}")
    );
}

/// A malformed `arguments` text in history: a renderer that parses
/// `arguments` itself (K3) gets it as written and applies its own tolerance;
/// a template gets the empty object the engines hand theirs.
#[test]
fn malformed_tool_call_arguments_render_as_the_engines_render_them() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "北京天气"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\":\"北京\""}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "晴"}
        ]
    }))
    .unwrap();

    let template = render_with(Default::default(), &request);
    assert_eq!(
        template["messages"][1]["tool_calls"][0]["function"]["arguments"],
        json!({})
    );

    let raw = render_with(RAW_ARGUMENTS, &request);
    assert_eq!(
        raw["messages"][1]["tool_calls"][0]["function"]["arguments"],
        json!("{\"location\":\"北京\"")
    );
}

#[test]
fn deferred_renderer_gets_the_prefill_and_its_job_runs_in_the_tokenize_step() {
    let tokenizer = llm_tokenizer::MockTokenizer::new()
        .with_deferred_chat_ids(vec![7, 8, 9])
        .with_unbilled_prompt_tokens(3);
    let (processed, encoding) = process_chat_messages_with_placeholders(
        &prefill_request(),
        &tokenizer,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    assert_eq!(
        processed.text, "user: Hello\nassistant: Sure",
        "the prefill is passed into the render call"
    );
    assert!(matches!(encoding, PromptEncoding::Deferred(_)));
    assert_eq!(
        processed.unbilled_prompt_tokens, 3,
        "the renderer's unbilled count rides along"
    );
    let (flat, _) = process_chat_messages_with_placeholders(
        &prefill_request(),
        &llm_tokenizer::MockTokenizer::new(),
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    assert_eq!(flat.unbilled_prompt_tokens, 0);

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(tokenizer);
    let ids = block_on(encode_prompt_blocking(tokenizer, &processed.text, encoding)).unwrap();
    assert_eq!(ids.token_ids(), &[7, 8, 9]);
}

#[test]
fn deferred_encode_takes_the_offload_path_for_large_prompts() {
    use std::{
        sync::Mutex,
        thread::{self, ThreadId},
    };
    // The job records the thread it ran on. `block_on` polls on this
    // thread, so an offloaded job runs somewhere else.
    let ran_on: Arc<Mutex<Option<ThreadId>>> = Arc::new(Mutex::new(None));
    let probe = {
        let ran_on = ran_on.clone();
        move || *ran_on.lock().unwrap() = Some(thread::current().id())
    };
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(
        llm_tokenizer::MockTokenizer::new()
            .with_deferred_chat_ids(vec![1, 2])
            .with_deferred_chat_probe(probe),
    );
    let render = |content: &str| {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": content}]
        }))
        .unwrap();
        process_chat_messages_with_placeholders(
            &request,
            &*tokenizer,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap()
    };
    let encode = |text: &str, encoding: PromptEncoding| {
        block_on(encode_prompt_blocking(tokenizer.clone(), text, encoding)).unwrap()
    };

    let (processed, encoding) = render(&"x".repeat(ENCODE_OFFLOAD_MIN_BYTES * 4));
    assert!(processed.text.len() >= ENCODE_OFFLOAD_MIN_BYTES);
    assert_eq!(encode(&processed.text, encoding).token_ids(), &[1, 2]);
    let offloaded = ran_on.lock().unwrap().take().expect("the job ran");
    assert_ne!(
        offloaded,
        thread::current().id(),
        "a large deferred encode is offloaded"
    );

    let (processed, encoding) = render("hi");
    assert!(processed.text.len() < ENCODE_OFFLOAD_MIN_BYTES);
    encode(&processed.text, encoding);
    let inline = ran_on.lock().unwrap().take().expect("the job ran");
    assert_eq!(
        inline,
        thread::current().id(),
        "a small deferred encode runs inline"
    );
}

#[test]
fn public_entry_point_keeps_the_flat_text_for_a_deferred_renderer() {
    let tokenizer = llm_tokenizer::MockTokenizer::new().with_deferred_chat_ids(vec![7]);
    let processed = process_chat_messages(&prefill_request(), &tokenizer, None).unwrap();
    assert_eq!(processed.text, "user: Hello\nassistant: Sure");
}

/// A model nobody serves is a 404; a served model whose tokenizer is
/// still loading is a 503 with a `Retry-After`, marked as reported so it
/// logs once; a served model without a tokenizer when the gateway does
/// not load tokenizers itself is a 500.
#[test]
fn a_missing_tokenizer_is_a_404_a_503_while_it_loads_or_a_500_without_autoload() {
    assert_eq!(
        missing_tokenizer_response("stage", "nonexistent-model", false, true, None).status(),
        http::StatusCode::NOT_FOUND
    );
    let not_ready = missing_tokenizer_response("stage", "served-model", true, true, None);
    assert_eq!(not_ready.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        error::extract_error_code_from_response(&not_ready),
        TOKENIZER_NOT_READY_CODE
    );
    assert_eq!(
        not_ready
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    assert!(not_ready.extensions().get::<ReportedRefusal>().is_some());
    assert_eq!(
        missing_tokenizer_response("stage", "served-model", true, false, None).status(),
        http::StatusCode::INTERNAL_SERVER_ERROR
    );
}

/// A served model whose tokenizer load was skipped or failed for good is
/// a 503 of its own code without a `Retry-After`, marked as reported,
/// whether or not the gateway loads tokenizers itself; the message names
/// the reason class, never the path or the error.
#[tokio::test]
async fn a_served_model_whose_tokenizer_load_failed_is_a_503_without_retry_after() {
    let failed = TokenizerLoadFailure::Failed(
        "Workflow failed at step load_tokenizer: File not found: /models/secret".to_string(),
    );
    for (failure, autoload, reason) in [
        (&TokenizerLoadFailure::NoSource, true, "no tokenizer source"),
        (
            &TokenizerLoadFailure::Removed,
            true,
            "the tokenizer was removed",
        ),
        (&failed, true, "the load failed"),
        (&failed, false, "the load failed"),
    ] {
        let response =
            missing_tokenizer_response("stage", "served-model", true, autoload, Some(failure));
        assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            TOKENIZER_LOAD_FAILED_CODE
        );
        assert!(response.headers().get(RETRY_AFTER).is_none());
        assert!(response.extensions().get::<ReportedRefusal>().is_some());
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let message = String::from_utf8_lossy(&body);
        assert!(message.contains("served-model"), "{message}");
        assert!(message.contains(reason), "{message}");
        assert!(!message.contains("/models/secret"), "{message}");
    }
}

/// One table for both completion paths: the API's words and the engines'
/// own signals pass, as a word or as a JSON `type`; anything else is
/// "stop"; a local stop beats all.
#[test]
fn the_completion_finish_reason_is_one_table() {
    for (local_stop, engine, expected) in [
        (false, "", "stop"),
        (false, "stop", "stop"),
        (false, "length", "length"),
        (false, "content_filter", "content_filter"),
        (false, r#"{"type":"stop"}"#, "stop"),
        (false, r#"{"type":"length","length":7}"#, "length"),
        (false, r#"{"type":"content_filter"}"#, "content_filter"),
        (false, r#"{"type":"abort"}"#, "abort"),
        (false, "abort", "abort"),
        (false, "error", "error"),
        (false, "failed", "failed"),
        (false, r#"{"type":"timeout"}"#, "stop"),
        (false, "whatever", "stop"),
        (true, "length", "stop"),
    ] {
        assert_eq!(
            completion_finish_reason(local_stop, engine),
            expected,
            "local_stop={local_stop} engine={engine:?}"
        );
    }
}
