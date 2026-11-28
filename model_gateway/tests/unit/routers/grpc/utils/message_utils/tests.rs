use llm_tokenizer::traits::RendererCapabilities;
use messages::{InputMessage, Role, TextBlock};

use super::*;

/// `omit_thinking_text` empties the thinking blocks' text and touches
/// nothing else: the blocks stay, with their signature, in their place.
#[test]
fn omit_thinking_text_keeps_the_blocks_and_drops_their_text() {
    let mut content = vec![
        messages::ContentBlock::Thinking {
            thinking: "plan".to_string(),
            signature: "sig".to_string(),
        },
        messages::ContentBlock::Text {
            text: "Hello".to_string(),
            citations: None,
        },
        messages::ContentBlock::Thinking {
            thinking: "more".to_string(),
            signature: String::new(),
        },
    ];
    omit_thinking_text(&mut content);
    assert_eq!(
        serde_json::to_value(&content).unwrap(),
        json!([
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Hello"},
            {"type": "thinking", "thinking": "", "signature": ""},
        ])
    );
}

#[test]
fn anthropic_tool_use_id_swaps_standard_prefix() {
    assert_eq!(
        anthropic_tool_use_id("call_0123456789abcdef01234567"),
        "toolu_0123456789abcdef01234567"
    );
    // Deterministic: same input, same output.
    assert_eq!(
        anthropic_tool_use_id("call_0123456789abcdef01234567"),
        anthropic_tool_use_id("call_0123456789abcdef01234567"),
    );
}

#[test]
fn anthropic_tool_use_id_passes_other_shapes_through() {
    // Prompt-visible / reference-correlated id formats keep their shape.
    assert_eq!(anthropic_tool_use_id("Bash_3"), "Bash_3");
    assert_eq!(
        anthropic_tool_use_id("functions.get_weather:2"),
        "functions.get_weather:2"
    );
    // Already-native ids are untouched.
    assert_eq!(
        anthropic_tool_use_id("toolu_0123456789abcdef01234567"),
        "toolu_0123456789abcdef01234567"
    );
}

#[test]
fn test_simple_user_message() {
    let messages = vec![InputMessage {
        role: Role::User,
        content: InputContent::String("Hello".to_string()),
    }];

    let result = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::String,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["role"], "user");
    assert_eq!(result[0]["content"], "Hello");
}

#[test]
fn test_assistant_with_text() {
    let messages = vec![InputMessage {
        role: Role::Assistant,
        content: InputContent::Blocks(vec![InputContentBlock::Text(TextBlock {
            text: "Hi there".to_string(),
            cache_control: None,
            citations: None,
        })]),
    }];

    let result = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::String,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["role"], "assistant");
    assert_eq!(result[0]["content"], "Hi there");
}

#[test]
fn an_image_in_an_assistant_turn_is_refused_naming_the_block() {
    let messages: Vec<InputMessage> = serde_json::from_value(json!([{
        "role": "assistant",
        "content": [{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "AAAA"
            }
        }]
    }]))
    .unwrap();

    let error = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::OpenAI,
        None,
        MediaPartOrder::Authored,
    )
    .unwrap_err();

    assert!(error.is_refusal(), "{error:?}");
    assert_eq!(
        error.to_string(),
        "messages.0.content.0: image blocks are supported in user messages only"
    );
}

#[test]
fn an_image_in_a_system_turn_is_refused_naming_the_block() {
    let messages: Vec<InputMessage> = serde_json::from_value(json!([{
        "role": "system",
        "content": [{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "AAAA"
            }
        }]
    }]))
    .unwrap();

    let error = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::OpenAI,
        None,
        MediaPartOrder::Authored,
    )
    .unwrap_err();

    assert!(error.is_refusal(), "{error:?}");
    assert_eq!(
        error.to_string(),
        "messages.0.content.0: image blocks are supported in user messages only"
    );
}

#[test]
fn a_document_in_a_system_turn_is_refused_naming_the_block() {
    let content = serde_json::from_value(json!([{
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": "the notes"}
    }]))
    .unwrap();
    let error = convert_system_message(0, &content).unwrap_err();
    assert!(error.is_refusal(), "{error:?}");
    assert_eq!(
        error.to_string(),
        "messages.0.content.0: document blocks are supported in user messages only"
    );
}

#[test]
fn a_document_in_an_assistant_turn_is_refused_naming_the_block() {
    let messages: Vec<InputMessage> = serde_json::from_value(json!([{
        "role": "assistant",
        "content": [{
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": "the notes"
            }
        }]
    }]))
    .unwrap();

    let error = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::OpenAI,
        None,
        MediaPartOrder::Authored,
    )
    .unwrap_err();

    assert!(error.is_refusal(), "{error:?}");
    assert_eq!(
        error.to_string(),
        "messages.0.content.0: document blocks are supported in user messages only"
    );
}

#[test]
fn test_assistant_with_tool_use() {
    let messages = vec![InputMessage {
        role: Role::Assistant,
        content: InputContent::Blocks(vec![
            InputContentBlock::Text(TextBlock {
                text: "Let me check.".to_string(),
                cache_control: None,
                citations: None,
            }),
            InputContentBlock::ToolUse(messages::ToolUseBlock {
                id: "tu_1".to_string(),
                name: "calculator".to_string(),
                input: json!({"expr": "2+2"}),
                cache_control: None,
            }),
        ]),
    }];

    let result = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::String,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["role"], "assistant");
    assert_eq!(result[0]["content"], "Let me check.");
    let tool_calls = result[0]["tool_calls"].as_array().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0]["function"]["name"], "calculator");
}

/// An assistant turn without a text block (a tool-use-only turn) is
/// handed to the template as the empty content the engine's chat utils
/// hand it: `""` under the string format, `[]` under the OpenAI format,
/// never null, which a template that tests or concatenates the content
/// cannot take.
#[test]
fn an_assistant_turn_without_text_is_the_engines_empty_content_in_the_templates_format() {
    let messages = vec![InputMessage {
        role: Role::Assistant,
        content: InputContent::Blocks(vec![InputContentBlock::ToolUse(messages::ToolUseBlock {
            id: "tu_1".to_string(),
            name: "calc".to_string(),
            input: json!({"x": 1}),
            cache_control: None,
        })]),
    }];

    for (format, empty) in [
        (ChatTemplateContentFormat::String, json!("")),
        (ChatTemplateContentFormat::OpenAI, json!([])),
    ] {
        let result =
            process_message_content_format(&messages, format, None, MediaPartOrder::MediaFirst)
                .unwrap();
        assert_eq!(result[0]["content"], empty, "{format:?}");
        assert_eq!(
            result[0]["tool_calls"][0]["function"]["name"],
            json!("calc")
        );
    }
}

/// A tool-use-only assistant turn in the history renders through a
/// template that tests, splits and prints every assistant message's
/// content (a reasoning model's template strips the thought this way):
/// the check is false on the empty content and nothing is printed for
/// the turn, as the engine's own server renders it.
#[test]
fn a_tool_use_only_assistant_turn_renders_through_a_content_testing_template() {
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase).unwrap();
    tokenizer
        .set_chat_template(
            "{%- for message in messages -%}\
                 {%- if message.role == 'assistant' and '</think>' in message.content -%}\
                 {%- set content = message.content.split('</think>')[-1].lstrip() -%}\
                 {%- else -%}{%- set content = message.content -%}{%- endif -%}\
                 <|{{ message.role }}|>{{ content }}\
                 {%- if message.tool_calls -%}{%- for call in message.tool_calls -%}\
                 <call:{{ call.function.name }}>{%- endfor -%}{%- endif -%}\
                 <|end|>{%- endfor -%}\
                 {%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
                .to_string(),
        )
        .unwrap();
    assert_eq!(
        tokenizer.chat_template_content_format(),
        ChatTemplateContentFormat::String
    );
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "18 celsius"}
            ]}
        ]
    }))
    .unwrap();
    let rendered = process_messages(
        &request,
        &tokenizer,
        None,
        None,
        MediaPartOrder::MediaFirst,
        ProviderProfile::for_model(&request.model),
    )
    .unwrap()
    .0
    .text;
    assert_eq!(
        rendered,
        "<|user|>What is the weather in Paris?<|end|><|assistant|><call:get_weather><|end|>\
             <|tool|>18 celsius<|end|><|assistant|>"
    );
}

#[test]
fn test_user_media_order_follows_contract() {
    // A template that takes parts gets them in the request's order under
    // both media orders; a string template places the media as the order
    // says.
    let messages = vec![InputMessage {
        role: Role::User,
        content: InputContent::Blocks(vec![
            InputContentBlock::Text(TextBlock {
                text: "question".to_string(),
                cache_control: None,
                citations: None,
            }),
            InputContentBlock::Image(messages::ImageBlock {
                source: messages::ImageSource::Base64 {
                    media_type: "image/png".to_string(),
                    data: "AAAA".to_string().into(),
                },
                cache_control: None,
            }),
        ]),
    }];

    for order in [MediaPartOrder::MediaFirst, MediaPartOrder::Authored] {
        let parts = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::OpenAI,
            None,
            order,
        )
        .unwrap();
        let arr = parts[0]["content"].as_array().unwrap();
        assert_eq!(arr[0]["text"], "question", "{order:?}");
        assert_eq!(arr[1], json!({"type": "image"}), "{order:?}");
    }

    let mut tokens = PlaceholderTokens::default();
    tokens.insert(Modality::Image, "<image>".to_string());
    let string = |order| {
        process_message_content_format(
            &messages,
            ChatTemplateContentFormat::String,
            Some(&tokens),
            order,
        )
        .unwrap()[0]["content"]
            .clone()
    };
    assert_eq!(string(MediaPartOrder::MediaFirst), "<image>\nquestion");
    assert_eq!(string(MediaPartOrder::Authored), "question\n<image>");
}

/// The early return in `resolve_media` (an empty plan: no registry is read
/// for the order) serves this renderer too: a user turn of text and
/// document blocks carries no media part, so both orders render it the
/// same under both content formats, as the chat renderer's twin test pins
/// for a content array without media.
#[test]
fn blocks_without_media_render_the_same_under_either_order() {
    let messages = vec![InputMessage {
        role: Role::User,
        content: InputContent::Blocks(vec![
            InputContentBlock::Text(TextBlock {
                text: "question".to_string(),
                cache_control: None,
                citations: None,
            }),
            InputContentBlock::Document(messages::DocumentBlock {
                source: messages::DocumentSource::Text {
                    data: "the notes".to_string(),
                },
                cache_control: None,
                title: Some("notes".to_string()),
                context: None,
                citations: None,
            }),
        ]),
    }];
    for format in [
        ChatTemplateContentFormat::String,
        ChatTemplateContentFormat::OpenAI,
    ] {
        let media_first =
            process_message_content_format(&messages, format, None, MediaPartOrder::MediaFirst)
                .unwrap();
        let authored =
            process_message_content_format(&messages, format, None, MediaPartOrder::Authored)
                .unwrap();
        assert_eq!(media_first, authored, "{format:?}");
    }
}

/// The Messages path renders a user turn's blocks where the request put
/// them on a template that walks the parts, like the chat path.
#[test]
fn a_parts_template_renders_the_blocks_where_the_request_put_them() {
    let tokenizer = jinja(
        r"{%- for m in messages -%}<|turn|>{{ m.role }}<|body|>
{%- if m.content is string -%}{{ m.content }}{%- else -%}{%- for part in m.content -%}
{%- if part.type == 'image' -%}<|image|>{%- elif part.type == 'text' -%}{{ part.text }}{%- endif -%}
{%- endfor -%}{%- endif -%}<|end|>{%- endfor -%}
{%- if add_generation_prompt -%}<|turn|>assistant<|body|>{%- endif -%}",
    );
    assert_eq!(
        tokenizer.chat_template_content_format(),
        ChatTemplateContentFormat::OpenAI
    );
    let image = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}});
    let text = |text: &str| json!({"type": "text", "text": text});
    let render = |blocks: Vec<Value>| {
        let request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": blocks}]
        }))
        .unwrap();
        process_messages(
            &request,
            &tokenizer,
            None,
            None,
            MediaPartOrder::MediaFirst,
            ProviderProfile::for_model(&request.model),
        )
        .unwrap()
        .0
        .text
    };
    let turn = |body: &str| format!("<|turn|>user<|body|>{body}<|end|><|turn|>assistant<|body|>");
    assert_eq!(
        render(vec![text("What is in this picture?"), image.clone()]),
        turn("What is in this picture?<|image|>")
    );
    assert_eq!(
        render(vec![image.clone(), text("What is in this picture?")]),
        turn("<|image|>What is in this picture?")
    );
    assert_eq!(
        render(vec![
            text("Figure 1: "),
            image.clone(),
            text(" Figure 2: "),
            image.clone(),
            text(" Which?")
        ]),
        turn("Figure 1: <|image|> Figure 2: <|image|> Which?")
    );
    assert_eq!(
        render(vec![
            image.clone(),
            image,
            text("Which picture is brighter?")
        ]),
        turn("<|image|><|image|>Which picture is brighter?")
    );
    assert_eq!(
        render(vec![text("No picture here.")]),
        turn("No picture here.")
    );
}

#[test]
fn test_user_with_tool_result_splits() {
    let messages = vec![InputMessage {
        role: Role::User,
        content: InputContent::Blocks(vec![InputContentBlock::ToolResult(
            messages::ToolResultBlock {
                tool_use_id: "tu_1".to_string(),
                content: Some(ToolResultContent::String("4".to_string())),
                is_error: None,
                cache_control: None,
            },
        )]),
    }];

    let result = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::String,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    // Tool result becomes a "tool" role message, not a "user" message
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["role"], "tool");
    assert_eq!(result[0]["tool_call_id"], "tu_1");
    assert_eq!(result[0]["content"], "4");
}

#[test]
fn test_assistant_with_thinking() {
    // Single thinking block
    let messages = vec![InputMessage {
        role: Role::Assistant,
        content: InputContent::Blocks(vec![
            InputContentBlock::Thinking(messages::ThinkingBlock {
                thinking: "Let me reason...".to_string(),
                signature: "sig123".to_string(),
            }),
            InputContentBlock::Text(TextBlock {
                text: "The answer is 42.".to_string(),
                cache_control: None,
                citations: None,
            }),
        ]),
    }];

    let result = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::String,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["role"], "assistant");
    assert_eq!(result[0]["content"], "The answer is 42.");
    assert_eq!(result[0]["reasoning_content"], "Let me reason...");

    // Multiple thinking blocks are concatenated
    let messages = vec![InputMessage {
        role: Role::Assistant,
        content: InputContent::Blocks(vec![
            InputContentBlock::Thinking(messages::ThinkingBlock {
                thinking: "First thought.".to_string(),
                signature: "sig1".to_string(),
            }),
            InputContentBlock::Thinking(messages::ThinkingBlock {
                thinking: "Second thought.".to_string(),
                signature: "sig2".to_string(),
            }),
            InputContentBlock::Text(TextBlock {
                text: "Combined answer.".to_string(),
                cache_control: None,
                citations: None,
            }),
        ]),
    }];

    let result = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::String,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    assert_eq!(
        result[0]["reasoning_content"],
        "First thought.\nSecond thought."
    );
}

#[test]
fn tool_schemas_render_in_the_clients_order() {
    use llm_tokenizer::chat_template::ChatTemplateProcessor;

    // More keys than a hash map would keep in order by chance.
    let names = [
        "zeta", "alpha", "mu", "beta", "omega", "delta", "kappa", "gamma",
    ];
    let properties: serde_json::Map<String, Value> = names
        .iter()
        .map(|name| (name.to_string(), json!({"type": "string"})))
        .collect();
    let body = json!({
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "lookup", "input_schema": {
            "type": "object",
            "properties": properties,
            "required": ["mu", "alpha"],
            "additionalProperties": false
        }}]
    });
    let template = "{% for tool in tools %}{{ tool | tojson }}{% endfor %}";
    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
    // The tools of a fresh request, rendered as `process_messages` does.
    let render = || {
        let request: CreateMessageRequest = serde_json::from_value(body.clone()).unwrap();
        let tools: Vec<Value> = request
            .chat_tools()
            .iter()
            .map(|tool| serde_json::to_value(tool).unwrap())
            .collect();
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        processor.apply_chat_template(&[], params).unwrap()
    };

    // The same request renders the same prompt, so its prefix stays cached.
    let prompt = render();
    assert_eq!(prompt, render());
    let tool: Value = serde_json::from_str(&prompt).unwrap();
    let parameters = tool["function"]["parameters"].as_object().unwrap();
    let keys: Vec<&str> = parameters["properties"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, names);
    let keys: Vec<&str> = parameters.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        ["type", "properties", "required", "additionalProperties"]
    );
}

#[test]
fn test_tool_result_images_join_user_content_in_block_order() {
    let messages = vec![InputMessage {
        role: Role::User,
        content: InputContent::Blocks(vec![
            InputContentBlock::ToolResult(messages::ToolResultBlock {
                tool_use_id: "tu_1".to_string(),
                content: Some(ToolResultContent::Blocks(vec![
                    messages::ToolResultContentBlock::Text(TextBlock {
                        text: "screenshot taken".to_string(),
                        cache_control: None,
                        citations: None,
                    }),
                    messages::ToolResultContentBlock::Image(messages::ImageBlock {
                        source: messages::ImageSource::Base64 {
                            media_type: "image/png".to_string(),
                            data: "AAAA".to_string().into(),
                        },
                        cache_control: None,
                    }),
                ])),
                is_error: None,
                cache_control: None,
            }),
            InputContentBlock::Text(TextBlock {
                text: "what do you see".to_string(),
                cache_control: None,
                citations: None,
            }),
        ]),
    }];

    let result = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::OpenAI,
        None,
        MediaPartOrder::Authored,
    )
    .unwrap();

    // The user content carries the tool result's image ahead of the text
    // that followed it; the tool message keeps the result's text.
    assert_eq!(result.len(), 2);
    assert_eq!(result[0]["role"], "user");
    let parts = result[0]["content"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], json!({"type": "image"}));
    assert_eq!(parts[1]["text"], "what do you see");
    assert_eq!(result[1]["role"], "tool");
    assert_eq!(result[1]["tool_call_id"], "tu_1");
    assert_eq!(result[1]["content"], "screenshot taken");
}

#[test]
fn test_tool_result_image_blocks_skips_text_only_results() {
    let text_only = messages::ToolResultBlock {
        tool_use_id: "tu_1".to_string(),
        content: Some(ToolResultContent::String("4".to_string())),
        is_error: None,
        cache_control: None,
    };
    assert_eq!(tool_result_image_blocks(&text_only).count(), 0);

    let empty = messages::ToolResultBlock {
        tool_use_id: "tu_2".to_string(),
        content: None,
        is_error: None,
        cache_control: None,
    };
    assert_eq!(tool_result_image_blocks(&empty).count(), 0);
}

fn document(
    source: messages::DocumentSource,
    title: Option<&str>,
    context: Option<&str>,
) -> Vec<InputMessage> {
    vec![InputMessage {
        role: Role::User,
        content: InputContent::Blocks(vec![
            InputContentBlock::Document(messages::DocumentBlock {
                source,
                cache_control: None,
                title: title.map(str::to_owned),
                context: context.map(str::to_owned),
                citations: Some(messages::CitationsConfig {
                    enabled: Some(true),
                }),
            }),
            InputContentBlock::Text(TextBlock {
                text: "What is the powerhouse of the cell?".to_string(),
                cache_control: None,
                citations: None,
            }),
        ]),
    }]
}

/// A text-source document is prompt text with its title and context,
/// kept in the order the request wrote it, for both content formats.
#[test]
fn text_document_renders_as_prompt_text_in_authored_order() {
    let messages = document(
        messages::DocumentSource::Text {
            data: "The mitochondria is the powerhouse of the cell.".to_string(),
        },
        Some("Biology notes"),
        Some("From a textbook"),
    );
    let expected = "<document>\n<source>Biology notes</source>\n<document_context>From a textbook</document_context>\n<document_content>\nThe mitochondria is the powerhouse of the cell.\n</document_content>\n</document>";

    let parts = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::OpenAI,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    let arr = parts[0]["content"].as_array().unwrap();
    assert_eq!(arr[0], json!({"type": "text", "text": expected}));
    assert_eq!(arr[1]["text"], "What is the powerhouse of the cell?");

    let flat = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::String,
        None,
        MediaPartOrder::MediaFirst,
    )
    .unwrap();
    assert_eq!(
        flat[0]["content"],
        json!(format!("{expected}\nWhat is the powerhouse of the cell?"))
    );
}

/// A content source given as a string renders as the document's text, as
/// a text source does.
#[test]
fn a_content_document_given_as_a_string_renders_its_text() {
    let messages = document(
        messages::DocumentSource::Content {
            content: messages::DocumentContent::String("The whole text.".to_string()),
        },
        None,
        None,
    );
    let parts = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::OpenAI,
        None,
        MediaPartOrder::Authored,
    )
    .unwrap();
    assert_eq!(
        parts[0]["content"][0]["text"],
        "<document>\n<document_content>\nThe whole text.\n</document_content>\n</document>"
    );
}

/// A content-source document joins its text blocks; one holding anything
/// else is refused naming the block.
#[test]
fn content_document_joins_its_text_blocks() {
    let text = |t: &str| {
        InputContentBlock::Text(TextBlock {
            text: t.to_string(),
            cache_control: None,
            citations: None,
        })
    };
    let messages = document(
        messages::DocumentSource::Content {
            content: messages::DocumentContent::Blocks(vec![
                text("First paragraph."),
                text("Second paragraph."),
            ]),
        },
        None,
        None,
    );
    let parts = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::OpenAI,
        None,
        MediaPartOrder::Authored,
    )
    .unwrap();
    assert_eq!(
        parts[0]["content"][0]["text"],
        "<document>\n<document_content>\nFirst paragraph.\nSecond paragraph.\n</document_content>\n</document>"
    );

    let messages = document(
        messages::DocumentSource::Content {
            content: messages::DocumentContent::Blocks(vec![InputContentBlock::Image(
                messages::ImageBlock {
                    source: messages::ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: "AAAA".to_string().into(),
                    },
                    cache_control: None,
                },
            )]),
        },
        None,
        None,
    );
    let err = process_message_content_format(
        &messages,
        ChatTemplateContentFormat::OpenAI,
        None,
        MediaPartOrder::Authored,
    )
    .unwrap_err();
    assert!(err.is_refusal(), "{err:?}");
    let err = err.to_string();
    assert!(
        err.starts_with(
            "messages.0.content.0: a document content source may hold text blocks only"
        ),
        "{err}"
    );
}

/// A PDF or URL document has no reader here: refused with a message
/// naming the block, before any template sees it.
#[test]
fn pdf_and_url_documents_are_refused_naming_the_block() {
    for (source, needle) in [
        (
            messages::DocumentSource::Base64 {
                media_type: "application/pdf".to_string(),
                data: "JVBERi0=".to_string(),
            },
            "messages.0.content.0: document source type 'base64' (application/pdf) is not supported",
        ),
        (
            messages::DocumentSource::Url {
                url: "https://example.com/paper.pdf".to_string(),
            },
            "messages.0.content.0: document source type 'url' is not supported",
        ),
    ] {
        let err = process_message_content_format(
            &document(source, None, None),
            ChatTemplateContentFormat::OpenAI,
            None,
            MediaPartOrder::Authored,
        )
        .unwrap_err();
        assert!(err.is_refusal(), "{err:?}");
        assert!(err.to_string().starts_with(needle), "{err}");
    }
}

/// An image or document whose source is a file of the public API's Files
/// API, which no engine route can read, is refused naming the block that
/// holds it: an image, an image inside a tool result, a document.
#[test]
fn a_file_image_or_document_is_refused_naming_the_block() {
    let file_image = || messages::ImageBlock {
        source: messages::ImageSource::File {
            file_id: "file_1".to_string(),
        },
        cache_control: None,
    };
    let user = |blocks: Vec<InputContentBlock>| {
        vec![InputMessage {
            role: Role::User,
            content: InputContent::Blocks(blocks),
        }]
    };
    let text = InputContentBlock::Text(TextBlock {
        text: "Describe it.".to_string(),
        cache_control: None,
        citations: None,
    });
    let cases = [
        (
            user(vec![text, InputContentBlock::Image(file_image())]),
            "messages.0.content.1: image source type 'file' is not supported",
        ),
        (
            user(vec![InputContentBlock::ToolResult(
                messages::ToolResultBlock {
                    tool_use_id: "toolu_1".to_string(),
                    content: Some(ToolResultContent::Blocks(vec![
                        messages::ToolResultContentBlock::Image(file_image()),
                    ])),
                    is_error: None,
                    cache_control: None,
                },
            )]),
            "messages.0.content.0: image source type 'file' is not supported",
        ),
        (
            document(
                messages::DocumentSource::File {
                    file_id: "file_2".to_string(),
                },
                None,
                None,
            ),
            "messages.0.content.0: document source type 'file' is not supported",
        ),
    ];
    for (messages, needle) in cases {
        let err = process_message_content_format(
            &messages,
            ChatTemplateContentFormat::OpenAI,
            None,
            MediaPartOrder::Authored,
        )
        .unwrap_err();
        assert!(err.is_refusal(), "{err:?}");
        assert!(err.to_string().starts_with(needle), "{err}");
    }
}

/// count_tokens renders and encodes through the same path as a message
/// request: with a renderer that encodes itself, the count is the length
/// of the ids it hands back, not of a re-tokenization of its text.
#[tokio::test]
async fn count_input_tokens_reports_the_rendered_prompt_length() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(
        llm_tokenizer::mock::MockTokenizer::new().with_deferred_chat_ids(vec![7, 8, 9, 10, 11]),
    );
    let request: CountMessageTokensRequest = serde_json::from_value(json!({
        "model": "m",
        "system": "You are terse.",
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tools": [{
            "name": "get_weather",
            "description": "Weather for a city",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
        }],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [{"role": "user", "content": "Weather in Paris?"}]
    }))
    .unwrap();
    let chat_tools = request.chat_tools();
    let tools = template_tools(&chat_tools, request.tool_choice.as_ref());
    assert_eq!(tools.len(), 1, "a forced tool renders alone");

    let counted = count_input_tokens(
        MessagesPrompt::from(&request),
        tokenizer,
        &tools,
        None,
        MediaPartOrder::MediaFirst,
    )
    .await
    .unwrap();
    assert_eq!(counted, 5);
}

/// `tool_choice: {"type": "none"}` renders every tool, as on the chat
/// path and as the engine's own server renders it (the choice reaches
/// the template as a kwarg); `auto` renders them all too.
#[test]
fn tool_choice_none_renders_the_tools_as_auto_does() {
    let request = |tool_choice: Value| -> CountMessageTokensRequest {
        serde_json::from_value(json!({
            "model": "m",
            "tools": [{
                "name": "get_weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }],
            "tool_choice": tool_choice,
            "messages": [{"role": "user", "content": "Weather in Paris?"}]
        }))
        .unwrap()
    };
    let none = request(json!({"type": "none"}));
    let none_tools = none.chat_tools();
    let under_none = template_tools(&none_tools, none.tool_choice.as_ref());
    assert!(
        matches!(under_none, Cow::Borrowed(_)),
        "none narrows nothing: the tools are the request's own"
    );
    assert_eq!(under_none.len(), 1, "the tool stays declared under none");
    let auto = request(json!({"type": "auto"}));
    let auto_tools = auto.chat_tools();
    assert_eq!(
        template_tools(&auto_tools, auto.tool_choice.as_ref()).len(),
        1
    );
}

/// The Messages path tells the template the request's tool choice as the
/// chat path tells it, whatever the renderer: the chat value under the
/// chat path's kwarg (`any` as `required`, a named tool as its function
/// object, `none` as `none` with the tools narrowed away), and no kwargs
/// when the request names no choice.
#[test]
fn the_messages_path_tells_the_template_the_tool_choice() {
    let tok = llm_tokenizer::MockTokenizer::new().with_json_chat_template();
    let rendered = |tool_choice: Option<Value>| -> Value {
        let mut body = json!({
            "model": "m",
            "max_tokens": 16,
            "tools": [{
                "name": "get_weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }],
            "messages": [{"role": "user", "content": "Weather in Paris?"}]
        });
        if let Some(tool_choice) = tool_choice {
            body["tool_choice"] = tool_choice;
        }
        let request: CreateMessageRequest = serde_json::from_value(body).unwrap();
        let chat_tools = request.chat_tools();
        let tools = template_tools(&chat_tools, request.tool_choice.as_ref());
        let tools_for_template = if tools.is_empty() {
            None
        } else {
            Some(&*tools)
        };
        let (processed, _) =
            process_messages_with_encoding(&request, &tok, tools_for_template).unwrap();
        serde_json::from_str(&processed.text).unwrap()
    };
    let any = rendered(Some(json!({"type": "any"})));
    assert_eq!(any["template_kwargs"]["tool_choice"], json!("required"));
    assert_eq!(any["tools"].as_array().map(Vec::len), Some(1));
    let named = rendered(Some(json!({"type": "tool", "name": "get_weather"})));
    assert_eq!(
        named["template_kwargs"]["tool_choice"],
        json!({"type": "function", "function": {"name": "get_weather"}})
    );
    let none = rendered(Some(json!({"type": "none"})));
    assert_eq!(none["template_kwargs"]["tool_choice"], json!("none"));
    assert_eq!(
        none["tools"].as_array().map(Vec::len),
        Some(1),
        "the tool stays declared under none, the template told the choice: {none}"
    );
    let unset = rendered(None);
    assert!(
        unset.get("template_kwargs").is_none(),
        "no choice, no kwargs: {unset}"
    );
}

/// The K3 stand-in vocabulary of the kit test (`tests/grpc/kimi_k3_tool_choice_none.rs`):
/// byte-level ranks plus the K3 control tokens, under the config that
/// selects the renderer, so the rendering reads back as text.
fn k3_stand_in_dir() -> tempfile::TempDir {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    let model: String = (0u32..256)
        .map(|b| format!("{} {b}\n", STANDARD.encode([b as u8])))
        .collect();
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("tiktoken.model"), model).expect("tiktoken.model");
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"added_tokens_decoder": {
                "300": {"content": "<|open|>", "special": false},
                "301": {"content": "<|close|>", "special": false},
                "302": {"content": "<|sep|>", "special": false},
                "303": {"content": "<|end_of_msg|>", "special": true}
            }}"#,
    )
    .expect("tokenizer_config.json");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures": ["KimiK3ForConditionalGeneration"]}"#,
    )
    .expect("config.json");
    dir
}

/// On the Messages path the K3 profile renders `tool_choice: {"type":
/// "none"}` as its reference does and as the chat path renders it: the
/// tool declaration before the conversation, the instruction not to call
/// it after (the template learns the choice from the chat path's kwarg).
#[test]
fn k3_declares_the_tool_and_forbids_the_call_on_the_messages_path() {
    let dir = k3_stand_in_dir();
    let tok = llm_tokenizer::create_tokenizer(dir.path().to_str().expect("utf-8 path"))
        .expect("tokenizer");
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m",
        "max_tokens": 64,
        "tools": [{
            "name": "get_weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
        }],
        "tool_choice": {"type": "none"},
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}]
    }))
    .expect("the request");
    let chat_tools = request.chat_tools();
    let tools = template_tools(&chat_tools, request.tool_choice.as_ref());
    let (processed, _) = process_messages(
        &request,
        tok.as_ref(),
        Some(&*tools),
        None,
        MediaPartOrder::MediaFirst,
        ProviderProfile::for_model(&request.model),
    )
    .expect("render");
    let text = &processed.text;
    let declaration = text
        .find("Here are the available tools, described in JSONSchema.")
        .unwrap_or_else(|| panic!("no tool declaration in {text:?}"));
    assert!(text.contains(r#""name":"get_weather""#), "{text:?}");
    let user = text
        .find("What is the weather in Paris?")
        .unwrap_or_else(|| panic!("no user turn in {text:?}"));
    let forbids = text
        .find("You MUST NOT call any tools in the next message.")
        .unwrap_or_else(|| panic!("no tool_choice=none message in {text:?}"));
    assert!(
        declaration < user && user < forbids,
        "declaration at {declaration}, user turn at {user}, instruction at {forbids}"
    );
}

/// An image block the string format has no placeholder to write for is
/// refused, as the chat renderer refuses it, never dropped from a prompt
/// whose media still travels. A caller without model metadata passes no
/// placeholders, and its media parts are left out as before.
#[test]
fn an_image_block_with_no_placeholder_is_refused_not_dropped() {
    let messages: Vec<InputMessage> = serde_json::from_value(json!([{
        "role": "user",
        "content": [
            {"type": "text", "text": "question"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
        ]
    }]))
    .unwrap();
    let render = |placeholders: Option<&PlaceholderTokens>| {
        process_message_content_format(
            &messages,
            ChatTemplateContentFormat::String,
            placeholders,
            MediaPartOrder::MediaFirst,
        )
    };

    let refused = render(Some(&PlaceholderTokens::worker_only())).unwrap_err();
    assert!(refused.is_refusal(), "{refused:?}");
    assert_eq!(
        refused.to_string(),
        "missing image placeholder for string-format chat template"
    );
    assert_eq!(render(None).unwrap()[0]["content"], "question");
}

/// With a string-format template an image is the placeholder the model's
/// registry names; the count carries it, as the prompt `/v1/messages`
/// sends carries it. (The mock vocabulary knows "token", "Hello", "world".)
#[tokio::test]
async fn count_input_tokens_counts_the_image_placeholder_the_template_writes() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(
        llm_tokenizer::mock::MockTokenizer::new()
            .with_content_format(ChatTemplateContentFormat::String),
    );
    let request: CountMessageTokensRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Hello world"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
        ]}]
    }))
    .unwrap();
    let mut placeholders = PlaceholderTokens::default();
    placeholders.insert(Modality::Image, "token".to_string());

    let without = count_input_tokens(
        MessagesPrompt::from(&request),
        Arc::clone(&tokenizer),
        &[],
        None,
        MediaPartOrder::MediaFirst,
    )
    .await
    .unwrap();
    let with = count_input_tokens(
        MessagesPrompt::from(&request),
        tokenizer,
        &[],
        Some(&placeholders),
        MediaPartOrder::MediaFirst,
    )
    .await
    .unwrap();
    assert_eq!(without, 2, "the text alone: Hello, world");
    assert_eq!(with, 3, "the placeholder counts once");
}

/// A tokenizer whose encode fails is a gateway fault, told apart from a
/// request that does not render, so the router can answer 500 and 400 as
/// `/v1/messages` does.
#[tokio::test]
async fn count_input_tokens_tells_an_encode_failure_from_a_render_failure() {
    struct EncodeFails(llm_tokenizer::mock::MockTokenizer);
    impl llm_tokenizer::traits::Encoder for EncodeFails {
        fn encode(&self, _: &str, _: bool) -> anyhow::Result<llm_tokenizer::traits::Encoding> {
            Err(anyhow::anyhow!("the tokenizer files are unreadable"))
        }
        fn encode_batch(
            &self,
            _: &[&str],
            _: bool,
        ) -> anyhow::Result<Vec<llm_tokenizer::traits::Encoding>> {
            Err(anyhow::anyhow!("the tokenizer files are unreadable"))
        }
    }
    impl llm_tokenizer::traits::Decoder for EncodeFails {
        fn decode(&self, ids: &[u32], s: bool) -> anyhow::Result<String> {
            self.0.decode(ids, s)
        }
    }
    impl Tokenizer for EncodeFails {
        fn vocab_size(&self) -> usize {
            self.0.vocab_size()
        }
        fn get_special_tokens(&self) -> &llm_tokenizer::traits::SpecialTokens {
            self.0.get_special_tokens()
        }
        fn token_to_id(&self, t: &str) -> Option<u32> {
            self.0.token_to_id(t)
        }
        fn id_to_token(&self, id: u32) -> Option<String> {
            self.0.id_to_token(id)
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn apply_chat_template(
            &self,
            messages: &[Value],
            params: ChatTemplateParams,
        ) -> anyhow::Result<String> {
            self.0.apply_chat_template(messages, params)
        }
    }
    let request: CountMessageTokensRequest = serde_json::from_value(json!({
        "model": "m", "messages": [{"role": "user", "content": "Hello world"}]
    }))
    .unwrap();

    let failure = count_input_tokens(
        MessagesPrompt::from(&request),
        Arc::new(EncodeFails(llm_tokenizer::mock::MockTokenizer::new())),
        &[],
        None,
        MediaPartOrder::MediaFirst,
    )
    .await
    .unwrap_err();
    assert!(matches!(failure, CountError::Encode(_)), "{failure:?}");
}

/// A user turn, then `assistant` as the last message.
fn ending_with(assistant: Value) -> CreateMessageRequest {
    serde_json::from_value(json!({
        "model": "m",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "Hello"}, assistant]
    }))
    .unwrap()
}

fn render(request: &CreateMessageRequest, tokenizer: &dyn Tokenizer) -> String {
    process_messages(
        request,
        tokenizer,
        None,
        None,
        MediaPartOrder::MediaFirst,
        ProviderProfile::for_model(&request.model),
    )
    .unwrap()
    .0
    .text
}

/// A trailing assistant message with text is a prefill: a Jinja template
/// continues it inside the turn it renders for it, header included,
/// instead of closing it and opening a new turn.
#[test]
fn trailing_assistant_text_is_continued_inside_its_turn() {
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase).unwrap();
    tokenizer
        .set_chat_template(
            r"
{%- for m in messages -%}
{%- if m.role == 'assistant' -%}{{- '<|turn|>assistant<|to|>user<|body|>' + m.content + '<|end|>' -}}
{%- else -%}{{- '<|turn|>' + m.role + '<|body|>' + m.content + '<|end|>' -}}{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|turn|>assistant' -}}{%- endif -%}"
                .to_string(),
        )
        .unwrap();
    let request = ending_with(json!({"role": "assistant", "content": [
        {"type": "text", "text": "Sure"}
    ]}));
    assert_eq!(
        render(&request, &tokenizer),
        "<|turn|>user<|body|>Hello<|end|><|turn|>assistant<|to|>user<|body|>Sure"
    );
}

/// A renderer without native continuation gets the prefill appended
/// after the generation prompt, as chat `continue_final_message` does.
#[test]
fn trailing_assistant_text_follows_the_generation_prompt_elsewhere() {
    let tokenizer = llm_tokenizer::MockTokenizer::new();
    let request = ending_with(json!({"role": "assistant", "content": "Sure"}));
    assert_eq!(render(&request, &tokenizer), "user: Hello\nassistant: Sure");
}

/// A trailing assistant message without text (tool calls only) is not a
/// prefill: it is rendered as a closed turn before a new one.
#[test]
fn trailing_tool_use_still_opens_a_new_turn() {
    let tokenizer = llm_tokenizer::MockTokenizer::new();
    let request = ending_with(json!({"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
    ]}));
    assert_eq!(
        render(&request, &tokenizer),
        "user: Hello\nassistant: \nassistant: "
    );
}

/// The coding agent's shape: its environment block after the first user
/// turn and a note after every tool result, both `system`-role messages
/// inside `messages[]`, beside a top-level system prompt.
fn coding_agent_request() -> CreateMessageRequest {
    serde_json::from_value(json!({
        "model": "m",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "system": "You are a coding assistant.",
        "messages": [
            {"role": "user", "content": "Fix the failing test."},
            {"role": "system", "content": "# Environment\nworking directory: /work"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Run the suite first.", "signature": "sig"},
                {"type": "tool_use", "id": "toolu_01", "name": "Bash",
                 "input": {"command": "pytest -q"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": "1 failed, 14 passed"}
            ]},
            {"role": "system", "content": [
                {"type": "text", "text": "<total_tokens>14994851 tokens left</total_tokens>",
                 "cache_control": {"type": "ephemeral"}}
            ]}
        ]
    }))
    .expect("the request")
}

/// A `system`-role message inside `messages[]` is forwarded in place
/// unless the renderer writes no turn for a system message after the
/// first: then every one past the leading run is a user turn at its
/// position, and the leading run stays.
#[test]
fn system_messages_after_the_first_are_user_turns_for_a_leading_system_only_renderer() {
    let request = coding_agent_request();
    let in_place = llm_tokenizer::MockTokenizer::new();
    assert_eq!(
        render(&request, &in_place),
        "system: You are a coding assistant.\n\
             user: Fix the failing test.\n\
             system: # Environment\nworking directory: /work\n\
             assistant: \n\
             tool: 1 failed, 14 passed\n\
             system: <total_tokens>14994851 tokens left</total_tokens>\n\
             assistant: "
    );
    let leading_system_only =
        llm_tokenizer::MockTokenizer::new().with_renderer_capabilities(RendererCapabilities {
            leading_system_only: true,
            ..RendererCapabilities::default()
        });
    assert_eq!(
        render(&request, &leading_system_only),
        "system: You are a coding assistant.\n\
             user: Fix the failing test.\n\
             user: # Environment\nworking directory: /work\n\
             assistant: \n\
             tool: 1 failed, 14 passed\n\
             user: <total_tokens>14994851 tokens left</total_tokens>\n\
             assistant: "
    );
}

/// Without a top-level system prompt a leading `system` message is the
/// system prompt, which every renderer writes: it stays where it is.
#[test]
fn a_leading_system_message_stays_the_system_prompt_for_every_renderer() {
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m",
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Note"}
        ]
    }))
    .expect("the request");
    let leading_system_only =
        llm_tokenizer::MockTokenizer::new().with_renderer_capabilities(RendererCapabilities {
            leading_system_only: true,
            ..RendererCapabilities::default()
        });
    assert_eq!(
        render(&request, &leading_system_only),
        "system: Be terse.\nuser: Hi\nuser: Note\nassistant: "
    );
}

/// With a top-level system prompt and a `system` message opening the
/// array, both are the leading run: the system prompt, which stays as it
/// is for every renderer; only the system messages past it become user
/// turns for a leading-system-only renderer.
#[test]
fn a_top_level_system_and_a_leading_system_message_are_both_the_system_prompt() {
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m",
        "max_tokens": 8,
        "system": "Be terse.",
        "messages": [
            {"role": "system", "content": "# Environment"},
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Note"}
        ]
    }))
    .expect("the request");
    let leading_system_only =
        llm_tokenizer::MockTokenizer::new().with_renderer_capabilities(RendererCapabilities {
            leading_system_only: true,
            ..RendererCapabilities::default()
        });
    assert_eq!(
        render(&request, &leading_system_only),
        "system: Be terse.\nsystem: # Environment\nuser: Hi\nuser: Note\nassistant: "
    );
    assert_eq!(
        render(&request, &llm_tokenizer::MockTokenizer::new()),
        "system: Be terse.\nsystem: # Environment\nuser: Hi\nsystem: Note\nassistant: "
    );
}

/// A model directory the DeepSeek V4 renderer is chosen for, with a
/// tokenizer that loads (the render never encodes).
fn deepseek_v4_stand_in_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
                "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null, "decoder": null,
                "model": {"type": "BPE", "vocab": {"hello": 0}, "merges": []}}"#,
    )
    .expect("tokenizer.json");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"}"#,
    )
    .expect("config.json");
    dir
}

/// Through the DeepSeek V4 renderer the coding agent's notes are user
/// turns the encoder merges into the turn before them behind a blank
/// line: the prompt ends with the tool result, the note, the assistant
/// header and the open thought, and the historical assistant turn has
/// its header after the environment block (in place, the encoder glues
/// a system message's text to the text before it with no separator and
/// opens the same headers). The prompt is rendered before the parser is
/// chosen, so every parser reads this one.
#[test]
fn the_deepseek_v4_renderer_gets_the_coding_agents_notes_as_user_turns() {
    let dir = deepseek_v4_stand_in_dir();
    let tok = llm_tokenizer::create_tokenizer(dir.path().to_str().expect("utf-8 path"))
        .expect("tokenizer");
    assert!(tok.renderer_capabilities().leading_system_only);
    let text = render(&coding_agent_request(), tok.as_ref());
    assert!(
        text.ends_with(
            "<tool_result>1 failed, 14 passed</tool_result>\n\n\
                 <total_tokens>14994851 tokens left</total_tokens><｜Assistant｜><think>"
        ),
        "{text:?}"
    );
    assert!(
        text.contains(
            "<｜User｜>Fix the failing test.\n\n# Environment\nworking directory: /work<｜Assistant｜>"
        ),
        "{text:?}"
    );
}

/// An assistant turn as Qwen3 renders one: reasoning before the text,
/// tool calls after it.
const REASONING_TEXT_CALLS: &str = r"
{%- for m in messages -%}
{{- '<|im_start|>' + m.role + '\n' -}}
{%- if m.role == 'assistant' -%}
{%- if m.reasoning_content -%}{{- '<think>' + m.reasoning_content + '</think>' -}}{%- endif -%}
{{- m.content or '' -}}
{%- for tc in m.tool_calls or [] -%}{{- '<tool_call>' + tc.function.name + '</tool_call>' -}}{%- endfor -%}
{%- else -%}{{- m.content -}}{%- endif -%}
{{- '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}";

fn jinja(template: &str) -> llm_tokenizer::TiktokenTokenizer {
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase).unwrap();
    tokenizer.set_chat_template(template.to_string()).unwrap();
    tokenizer
}

/// Text followed by a tool call is a closed turn: continuing after the
/// text would drop the call, since transformers' `continue_final_message`
/// cuts the prompt there.
#[test]
fn trailing_text_with_tool_use_is_a_closed_turn() {
    let request = ending_with(json!({"role": "assistant", "content": [
        {"type": "text", "text": "Sure"},
        {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
    ]}));
    assert_eq!(
        render(&request, &jinja(REASONING_TEXT_CALLS)),
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n\
             Sure<tool_call>f</tool_call><|im_end|>\n<|im_start|>assistant\n"
    );
}

/// Thinking then text is continued after the text, its reasoning kept, as
/// transformers continues it.
#[test]
fn trailing_thinking_and_text_is_continued_with_its_reasoning() {
    let request = ending_with(json!({"role": "assistant", "content": [
        {"type": "thinking", "thinking": "plan", "signature": "s"},
        {"type": "text", "text": "Sure"}
    ]}));
    assert_eq!(
        render(&request, &jinja(REASONING_TEXT_CALLS)),
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>plan</think>Sure"
    );
}

/// Without native continuation only the text would follow the generation
/// prompt, so a message that also has thinking or a tool call is a
/// closed turn and keeps them.
#[test]
fn mixed_trailing_assistant_keeps_its_fields_elsewhere() {
    let tokenizer = llm_tokenizer::MockTokenizer::new().with_json_chat_template();
    let thinking = json!({"type": "thinking", "thinking": "plan", "signature": "s"});
    let text = json!({"type": "text", "text": "Sure"});
    let tool_use = json!({"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}});
    let cases = [
        (
            json!([thinking, text]),
            json!({"role": "assistant", "content": "Sure", "reasoning_content": "plan"}),
        ),
        (
            json!([text, tool_use]),
            json!({"role": "assistant", "content": "Sure", "tool_calls": [
                {"id": "toolu_1", "type": "function", "function": {"name": "f", "arguments": {}}}
            ]}),
        ),
    ];
    for (content, kept) in cases {
        let request = ending_with(json!({"role": "assistant", "content": content}));
        let rendered = render(&request, &tokenizer);
        let rendered: Value = serde_json::from_str(&rendered)
            .unwrap_or_else(|_| panic!("text appended after the prompt: {rendered}"));
        assert_eq!(rendered["add_generation_prompt"], json!(true));
        assert_eq!(rendered["messages"][1], kept);
    }
}

/// `output_config.effort` reads as a Chat `reasoning_effort` does, and
/// reaches the chat template as the `reasoning_effort` kwarg.
#[test]
fn output_config_effort_is_rendered_as_the_reasoning_effort() {
    let request = |output_config: Value| -> CreateMessageRequest {
        serde_json::from_value(json!({
            "model": "m", "max_tokens": 8, "output_config": output_config,
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap()
    };
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase).unwrap();
    tokenizer
        .set_chat_template(
            "{%- for m in messages -%}{{ m.role }}: {{ m.content }} {% endfor -%}\
                 effort={{ reasoning_effort | default('unset') }}"
                .to_string(),
        )
        .unwrap();
    let rendered = |output_config| {
        let request = request(output_config);
        process_messages(
            &request,
            &tokenizer,
            None,
            None,
            MediaPartOrder::MediaFirst,
            ProviderProfile::for_model(&request.model),
        )
        .unwrap()
        .0
        .text
    };
    assert_eq!(
        rendered(json!({"effort": "high"})),
        "user: Hello effort=high"
    );
    assert_eq!(rendered(json!({"effort": 64})), "user: Hello effort=64");
    assert_eq!(
        rendered(json!({"effort": null})),
        "user: Hello effort=unset"
    );
    assert_eq!(rendered(json!({})), "user: Hello effort=unset");
}

/// The reasoning parser is armed from the effort the template renders: on
/// a template that switches thinking through `reasoning_effort`, an
/// `output_config.effort` that closes the think block leaves the parser
/// unarmed, as the same effort does for a Chat request. Without a thinking
/// block, "none" turns a toggle template's thinking off, as for Chat.
#[test]
fn output_config_effort_arms_the_reasoning_parser_as_rendered() {
    use crate::routers::grpc::utils::{
        chat_reasoning_starts_in_prefill, messages_reasoning_starts_in_prefill,
    };
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase).unwrap();
    tokenizer
        .set_chat_template(
            "{# think_begin_token #}{%- for m in messages -%}{{ m.role }}: {{ m.content }}\n{% endfor -%}\
                 {%- if reasoning_effort is defined and reasoning_effort == 'no_think' -%}assistant: <think></think>\
                 {%- else -%}assistant: <think>{%- endif -%}"
                .to_string(),
        )
        .unwrap();
    for (effort, prompt_end, armed) in [
        (json!("no_think"), "<think></think>", false),
        (json!("high"), "<think>", true),
        (Value::Null, "<think>", true),
    ] {
        let request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 8, "output_config": {"effort": effort},
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap();
        let rendered = process_messages(
            &request,
            &tokenizer,
            None,
            None,
            MediaPartOrder::MediaFirst,
            ProviderProfile::for_model(&request.model),
        )
        .unwrap()
        .0
        .text;
        assert_eq!(rendered, format!("user: Hello\nassistant: {prompt_end}"));
        assert_eq!(
            messages_reasoning_starts_in_prefill(&request, false, &tokenizer),
            armed,
            "{effort}"
        );
        let chat: openai_protocol::chat::ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m", "reasoning_effort": effort,
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap();
        assert_eq!(
            chat_reasoning_starts_in_prefill(&chat, &tokenizer),
            armed,
            "chat {effort}"
        );
    }

    tokenizer
        .set_chat_template(
            "{%- for m in messages -%}{{ m.role }}: {{ m.content }} {% endfor -%}\
                 thinking={{ enable_thinking | default('unset') }}"
                .to_string(),
        )
        .unwrap();
    let rendered = |effort: &str| {
        let request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 8, "output_config": {"effort": effort},
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap();
        process_messages(
            &request,
            &tokenizer,
            None,
            None,
            MediaPartOrder::MediaFirst,
            ProviderProfile::for_model(&request.model),
        )
        .unwrap()
        .0
        .text
    };
    assert_eq!(rendered("none"), "user: Hello thinking=False");
    assert_eq!(rendered("high"), "user: Hello thinking=unset");
}

/// A block of the public API's server tools is refused on the engine
/// routes naming the block and its place, a server tool's result in the
/// assistant turn, a container upload in the user turn and either in a
/// `system` message of `messages[]` alike: the engine runs no server
/// tool, and a silent drop would answer a turn whose input it threw
/// away. A token count reads the same conversion.
#[test]
fn a_server_tool_block_is_refused_naming_the_block() {
    let convert = |messages: &[InputMessage]| {
        let err = process_message_content_format(
            messages,
            ChatTemplateContentFormat::String,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap_err();
        assert!(err.is_refusal(), "{err:?}");
        err.to_string()
    };
    let assistant: Vec<InputMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "bash_code_execution", "input": {"command": "ls"}},
            {"type": "bash_code_execution_tool_result", "tool_use_id": "srvtoolu_1",
             "content": {"type": "bash_code_execution_result", "stdout": "ok", "stderr": "", "return_code": 0, "content": []}},
            {"type": "text", "text": "done"}
        ]}
    ]))
    .unwrap();
    let err = convert(&assistant);
    assert!(
        err.starts_with(
            "messages.1.content.1: block type 'bash_code_execution_tool_result' is not supported on an engine route"
        ),
        "{err}"
    );
    let user: Vec<InputMessage> = serde_json::from_value(json!([
        {"role": "user", "content": [
            {"type": "text", "text": "run it"},
            {"type": "container_upload", "file_id": "file_1"}
        ]}
    ]))
    .unwrap();
    let err = convert(&user);
    assert!(
        err.starts_with(
            "messages.0.content.1: block type 'container_upload' is not supported on an engine route"
        ),
        "{err}"
    );
    let system: Vec<InputMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "run it"},
        {"role": "system", "content": [
            {"type": "text", "text": "the environment:"},
            {"type": "container_upload", "file_id": "file_1"}
        ]}
    ]))
    .unwrap();
    let err = convert(&system);
    assert!(
        err.starts_with(
            "messages.1.content.1: block type 'container_upload' is not supported on an engine route"
        ),
        "{err}"
    );
}

/// A replayed MCP exchange (`mcp_tool_use`, `mcp_tool_result`) is refused
/// on the engine routes naming the block and its place, in the assistant
/// turn the public API puts it in and in a user turn alike: the engine
/// runs no MCP loop, and a silent drop would answer without the call.
#[test]
fn a_replayed_mcp_exchange_is_refused_naming_the_block() {
    let convert = |messages: &[InputMessage]| {
        let err = process_message_content_format(
            messages,
            ChatTemplateContentFormat::String,
            None,
            MediaPartOrder::MediaFirst,
        )
        .unwrap_err();
        assert!(err.is_refusal(), "{err:?}");
        err.to_string()
    };
    let assistant: Vec<InputMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "look it up"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Looking."},
            {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup", "server_name": "s", "input": {"q": "x"}},
            {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}]},
            {"type": "text", "text": "it is 42"}
        ]}
    ]))
    .unwrap();
    let err = convert(&assistant);
    assert!(
        err.starts_with(
            "messages.1.content.1: block type 'mcp_tool_use' is not supported on an engine route"
        ),
        "{err}"
    );
    let user: Vec<InputMessage> = serde_json::from_value(json!([
        {"role": "user", "content": [
            {"type": "text", "text": "the result:"},
            {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}]}
        ]}
    ]))
    .unwrap();
    let err = convert(&user);
    assert!(
        err.starts_with(
            "messages.0.content.1: block type 'mcp_tool_result' is not supported on an engine route"
        ),
        "{err}"
    );
}
