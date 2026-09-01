use llm_tokenizer::chat_template::{
    detect_chat_template_content_format, ChatTemplateContentFormat, ChatTemplateParams,
    ChatTemplateProcessor,
};
use openai_protocol::{
    chat::{ChatMessage, MessageContent},
    common::{ContentPart, ImageUrl},
};

#[test]
fn test_simple_chat_template() {
    let template = r"
{%- for message in messages %}
<|{{ message.role }}|>{{ message.content }}<|end|>
{% endfor -%}
{%- if add_generation_prompt %}
<|assistant|>
{%- endif %}
";

    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();

    let messages = [ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Text("Test".to_string()),
        name: None,
    }];

    // Convert to JSON values like the router does
    let message_values: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    let params = ChatTemplateParams {
        add_generation_prompt: true,
        ..Default::default()
    };
    let result = processor
        .apply_chat_template(&message_values, params)
        .unwrap();
    assert!(result.contains("<|user|>Test<|end|>"));
    assert!(result.contains("<|assistant|>"));
}

#[test]
fn test_chat_template_with_tokens() {
    // Template that uses template kwargs for tokens
    let template = r"
{%- if bos_token -%}{{ bos_token }}{%- endif -%}
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}{%- if eos_token -%}{{ eos_token }}{%- endif -%}
{% endfor -%}
";

    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();

    let messages = [ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Text("Test".to_string()),
        name: None,
    }];

    // Convert to JSON values like the router does
    let message_values: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    // Use template_kwargs to pass tokens
    let mut template_kwargs = std::collections::HashMap::new();
    template_kwargs.insert(
        "bos_token".to_string(),
        serde_json::Value::String("<s>".to_string()),
    );
    template_kwargs.insert(
        "eos_token".to_string(),
        serde_json::Value::String("</s>".to_string()),
    );

    let params = ChatTemplateParams {
        template_kwargs: Some(&template_kwargs),
        ..Default::default()
    };

    let result = processor
        .apply_chat_template(&message_values, params)
        .unwrap();
    assert!(result.contains("<s>"));
    assert!(result.contains("</s>"));
}

#[test]
fn test_llama_style_template() {
    let template = r"
{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = '' -%}
{%- endif -%}

{{- bos_token if bos_token else '<|begin_of_text|>' }}
{%- if system_message %}
{{- '<|start_header_id|>system<|end_header_id|>\n\n' + system_message + '<|eot_id|>' }}
{%- endif %}

{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
{%- endfor %}

{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
";

    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();

    let messages = [
        ChatMessage::System {
            content: MessageContent::Text("You are a helpful assistant".to_string()),
            name: None,
            ext: Default::default(),
        },
        ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Text("What is 2+2?".to_string()),
            name: None,
        },
    ];

    // Convert to JSON values
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    // Use template_kwargs to pass the token
    let mut template_kwargs = std::collections::HashMap::new();
    template_kwargs.insert(
        "bos_token".to_string(),
        serde_json::Value::String("<|begin_of_text|>".to_string()),
    );

    let params = ChatTemplateParams {
        add_generation_prompt: true,
        template_kwargs: Some(&template_kwargs),
        ..Default::default()
    };
    let result = processor
        .apply_chat_template(&json_messages, params)
        .unwrap();

    // Check that the result contains expected markers
    assert!(result.contains("<|begin_of_text|>"));
    assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
    assert!(result.contains("You are a helpful assistant"));
    assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
    assert!(result.contains("What is 2+2?"));
    assert!(result.contains("<|start_header_id|>assistant<|end_header_id|>"));
}

#[test]
fn test_chatml_template() {
    let template = r"
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
";

    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();

    let messages = [
        ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Text("Hello".to_string()),
            name: None,
        },
        ChatMessage::Assistant {
            ext: Default::default(),
            content: Some(MessageContent::Text("Hi there!".to_string())),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        },
        ChatMessage::User {
            ext: Default::default(),
            content: MessageContent::Text("How are you?".to_string()),
            name: None,
        },
    ];

    // Convert to JSON values
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    let result = processor
        .apply_chat_template(
            &json_messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                ..Default::default()
            },
        )
        .unwrap();

    // Check ChatML format
    assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
    assert!(result.contains("<|im_start|>assistant\nHi there!<|im_end|>"));
    assert!(result.contains("<|im_start|>user\nHow are you?<|im_end|>"));
    assert!(result.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn test_template_without_generation_prompt() {
    let template = r"
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}
{% endfor -%}
{%- if add_generation_prompt -%}
assistant:
{%- endif -%}
";

    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();

    let messages = [ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Text("Test".to_string()),
        name: None,
    }];

    // Convert to JSON values
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    let result = processor
        .apply_chat_template(&json_messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(result.trim(), "user: Test");

    let result_with_prompt = processor
        .apply_chat_template(
            &json_messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(result_with_prompt.contains("assistant:"));
}

#[test]
fn test_empty_messages_template() {
    let template = r"{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}";

    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();

    let messages: Vec<serde_json::Value> = vec![];
    let result = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap();
    assert_eq!(result, "");
}

/// Test that tojson filter accepts ensure_ascii kwarg (HuggingFace compatibility)
/// This is the fix for: "unknown keyword argument 'ensure_ascii'"
#[test]
fn test_tojson_with_ensure_ascii() {
    // Template that uses tojson(ensure_ascii=False) like HuggingFace templates do
    let template = r"
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}
{%- if message.tool_calls is defined and message.tool_calls -%}
Tools: {{ message.tool_calls|tojson(ensure_ascii=False) }}
{%- endif -%}
{% endfor -%}
";

    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();

    let messages = [ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Text("Test with Unicode: 日本語".to_string()),
        name: None,
    }];

    // Convert to JSON values
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    // This should NOT fail with "unknown keyword argument 'ensure_ascii'"
    let result = processor
        .apply_chat_template(&json_messages, ChatTemplateParams::default())
        .unwrap();

    assert!(result.contains("user: Test with Unicode: 日本語"));
}

/// Test tojson with all HuggingFace kwargs
#[test]
fn test_tojson_with_all_huggingface_kwargs() {
    // Template using all the kwargs that HuggingFace's custom tojson accepts
    let template = r#"
Unsorted: {{ data|tojson }}
Sorted: {{ data|tojson(sort_keys=True) }}
Compact: {{ data|tojson(separators=(',', ':')) }}
Indented: {{ data|tojson(indent=2) }}
IndentedCompact: {{ data|tojson(indent=2, separators=(',', ':')) }}
Ascii: {{ "日本語"|tojson(ensure_ascii=True) }}
All: {{ data|tojson(ensure_ascii=False, sort_keys=True, indent=2, separators=(',', ': ')) }}
"#;

    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
    let messages: Vec<serde_json::Value> = vec![];
    let mut template_kwargs = std::collections::HashMap::new();
    template_kwargs.insert(
        "data".to_string(),
        serde_json::json!({"z_key": 1, "a_key": 2, "m_key": 3}),
    );

    // This should NOT fail - all kwargs should be accepted
    let result = processor
        .apply_chat_template(
            &messages,
            ChatTemplateParams {
                template_kwargs: Some(&template_kwargs),
                ..Default::default()
            },
        )
        .unwrap();

    // Verify sorted output contains keys in alphabetical order
    assert!(result.contains("Sorted:"));
    // By default, HuggingFace delegates to Python json.dumps, which preserves
    // insertion order and uses spaces after ',' and ':'.
    assert!(
        result.contains(r#"Unsorted: {"z_key": 1, "a_key": 2, "m_key": 3}"#),
        "default JSON should preserve insertion order and Python separators: {result}"
    );
    // The sorted output should have a_key before m_key before z_key
    let sorted_line = result.lines().find(|l| l.starts_with("Sorted:")).unwrap();
    let a_pos = sorted_line.find("a_key").unwrap();
    let m_pos = sorted_line.find("m_key").unwrap();
    let z_pos = sorted_line.find("z_key").unwrap();
    assert!(a_pos < m_pos && m_pos < z_pos, "Keys should be sorted");
    assert!(
        result.contains(r#"Compact: {"z_key":1,"a_key":2,"m_key":3}"#),
        "explicit separators should be honored: {result}"
    );

    // Verify indented output is pretty-printed with newlines
    assert!(
        result.contains("Indented: {\n"),
        "Indented JSON should be pretty-printed with newlines"
    );
    assert!(
        result.contains("IndentedCompact: {\n  \"z_key\":1,\n"),
        "custom separators should apply to indented JSON: {result}"
    );
    assert!(
        result.contains(r#"Ascii: "\u65e5\u672c\u8a9e""#),
        "ensure_ascii=True should escape non-ASCII text: {result}"
    );
}

#[test]
fn test_tojson_invalid_kwargs_rejected() {
    fn render_invalid_template(template: &str) -> String {
        let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();
        let messages: Vec<serde_json::Value> = vec![];
        let mut template_kwargs = std::collections::HashMap::new();
        template_kwargs.insert("data".to_string(), serde_json::json!({"k": 1}));

        processor
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    template_kwargs: Some(&template_kwargs),
                    ..Default::default()
                },
            )
            .unwrap_err()
            .to_string()
    }

    let err = render_invalid_template(r"{{ data|tojson(separators=',') }}");
    assert!(
        err.contains("separators must be a two-item sequence"),
        "unexpected separators error: {err}"
    );

    let err = render_invalid_template(r"{{ data|tojson(indent=-1) }}");
    assert!(
        err.contains("indent cannot be negative"),
        "unexpected indent error: {err}"
    );

    let err = render_invalid_template(r"{{ data|tojson(ensure_ascii='yes') }}");
    assert!(
        err.contains("ensure_ascii") || err.contains("bool"),
        "unexpected ensure_ascii error: {err}"
    );
}

#[test]
fn test_content_format_detection() {
    let string_template = r"
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}
{%- endfor -%}
";
    assert_eq!(
        detect_chat_template_content_format(string_template),
        ChatTemplateContentFormat::String
    );

    let openai_template = r"
{%- for message in messages -%}
  {%- for content in message.content -%}
    {{ content.type }}: {{ content.text }}
  {%- endfor -%}
{%- endfor -%}
";
    assert_eq!(
        detect_chat_template_content_format(openai_template),
        ChatTemplateContentFormat::OpenAI
    );
}

#[test]
fn test_template_with_multimodal_content() {
    let template = r#"
{%- for message in messages %}
{{ message.role }}:
{%- if message.content is string %}
{{ message.content }}
{%- else %}
{%- for part in message.content %}
  {%- if part.type == "text" %}
{{ part.text }}
  {%- elif part.type == "image_url" %}
[IMAGE]
  {%- endif %}
{%- endfor %}
{%- endif %}
{% endfor %}
"#;

    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();

    let messages = [ChatMessage::User {
        ext: Default::default(),
        content: MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Look at this:".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: None,
                    ext: Default::default(),
                },
            },
        ]),
        name: None,
    }];

    // Convert to JSON values
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|msg| serde_json::to_value(msg).unwrap())
        .collect();

    let result = processor
        .apply_chat_template(&json_messages, ChatTemplateParams::default())
        .unwrap();

    // Should contain both text and image parts
    assert!(result.contains("user:"));
    assert!(result.contains("Look at this:"));
    assert!(result.contains("[IMAGE]"));
}

#[test]
fn test_bos_token_injected_via_special_tokens() {
    // Simplified Llama-style template that uses {{ bos_token }}
    let template = r"{{- bos_token }}{%- for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>

{{ message.content }}<|eot_id|>
{%- endfor %}{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}";

    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();

    let messages = vec![serde_json::json!({
        "role": "user",
        "content": "Hello"
    })];

    let special_tokens = llm_tokenizer::SpecialTokens {
        bos_token: Some("<|begin_of_text|>".to_string()),
        eos_token: Some("<|eot_id|>".to_string()),
        ..Default::default()
    };

    let result = processor
        .apply_chat_template(
            &messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                special_tokens: Some(&special_tokens),
                ..Default::default()
            },
        )
        .unwrap();

    assert!(
        result.starts_with("<|begin_of_text|>"),
        "Must start with BOS token. Got: {:?}",
        &result[..50.min(result.len())]
    );
    assert!(result.contains("Hello"));
    assert!(result.ends_with("assistant<|end_header_id|>\n\n"));
}

#[test]
fn test_bos_token_missing_without_special_tokens() {
    // Same template but without special tokens — bos_token should be empty/undefined
    let template = r"{{- bos_token }}{%- for message in messages %}{{ message.role }}: {{ message.content }}
{%- endfor %}";

    let processor = ChatTemplateProcessor::new(template.to_string()).unwrap();

    let messages = vec![serde_json::json!({
        "role": "user",
        "content": "Hello"
    })];

    let result = processor
        .apply_chat_template(&messages, ChatTemplateParams::default())
        .unwrap();

    // Without special tokens, bos_token is undefined and renders as empty
    assert!(
        !result.contains("<|begin_of_text|>"),
        "Should NOT contain BOS without special tokens"
    );
    assert!(result.contains("Hello"));
}
