//! A checkpoint whose own tokenizer class sorts the keys of every map in the
//! tools before its template reads them (the Kimi family's
//! `tokenization_kimi.py`) renders them so here: the tool declaration the
//! template writes with `tojson` carries each object's keys in order at every
//! depth, whatever order the request declared them in. Only the tools are
//! sorted (a tool call's arguments are written as sent), a prepared tool set
//! renders the same bytes, and a checkpoint of another class keeps the
//! request's order.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fs;

use llm_tokenizer::{
    chat_template::{ChatTemplateParams, PreparedTools},
    traits::Tokenizer as TokenizerTrait,
    TiktokenTokenizer,
};
use serde_json::{json, Value};
use tempfile::TempDir;

const MIN_TIKTOKEN_MODEL: &str = "aGVsbG8= 0\n";

/// The family's tool declaration and tool-call lines, as the vendor's
/// templates write them: the tools through `tojson` with compact separators,
/// a call's arguments through the plain `tojson`.
const TEMPLATE: &str = "{%- if tools -%}<|im_system|>tool_declare<|im_middle|>\
    {{ tools | tojson(separators=(',', ':')) }}<|im_end|>{%- endif -%}\
    {%- for message in messages -%}\
    {%- if message['role'] == 'assistant' and message.get('tool_calls') -%}\
    <|im_assistant|>assistant<|im_middle|>{%- for tool_call in message['tool_calls'] -%}\
    <|tool_call_begin|>{{ tool_call['id'] }}<|tool_call_argument_begin|>\
    {{ tool_call['function']['arguments'] | tojson }}<|tool_call_end|>{%- endfor -%}<|im_end|>\
    {%- else -%}<|im_user|>{{ message['role'] }}<|im_middle|>{{ message['content'] }}<|im_end|>\
    {%- endif -%}{%- endfor -%}";

const KIMI_TOKENIZER_CONFIG: &str = r#"{
    "tokenizer_class": "TikTokenTokenizer",
    "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", null]}
}"#;

/// The tool declaration the reference writes for these tools: every map's
/// keys in order, at every depth.
const SORTED_DECLARATION: &str = concat!(
    r#"<|im_system|>tool_declare<|im_middle|>[{"function":{"description":"Get the current weather for a city","#,
    r#""name":"get_weather","parameters":{"properties":{"city":{"description":"City name","type":"string"},"#,
    r#""unit":{"enum":["celsius","fahrenheit"],"type":"string"}},"required":["city"],"type":"object"}},"#,
    r#""type":"function"},{"function":{"description":"Get the latest stock price for a ticker symbol","#,
    r#""name":"get_stock_price","parameters":{"properties":{"ticker":{"type":"string"}},"required":["ticker"],"#,
    r#""type":"object"}},"type":"function"}]<|im_end|>"#
);

fn tokenizer(tokenizer_config: &str) -> TiktokenTokenizer {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("tiktoken.model"), MIN_TIKTOKEN_MODEL).unwrap();
    fs::write(dir.path().join("tokenizer_config.json"), tokenizer_config).unwrap();
    fs::write(dir.path().join("chat_template.jinja"), TEMPLATE).unwrap();
    TiktokenTokenizer::from_dir(dir.path()).expect("tokenizer should load")
}

fn tools() -> Vec<Value> {
    vec![
        json!({"type": "function", "function": {"name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                "required": ["city"]}}}),
        json!({"type": "function", "function": {"name": "get_stock_price",
            "description": "Get the latest stock price for a ticker symbol",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]}}}),
    ]
}

fn messages() -> Vec<Value> {
    vec![
        json!({"role": "user", "content": "What is the weather in Paris?"}),
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": {"unit": "celsius", "city": "Paris"}}}]}),
    ]
}

fn render(tok: &TiktokenTokenizer, tools: &[Value], prepared: Option<&PreparedTools>) -> String {
    tok.apply_chat_template(
        &messages(),
        ChatTemplateParams {
            tools: Some(tools),
            prepared_tools: prepared,
            ..Default::default()
        },
    )
    .expect("render should succeed")
}

#[test]
fn the_family_renders_the_tools_key_sorted_and_the_call_arguments_as_sent() {
    let tok = tokenizer(KIMI_TOKENIZER_CONFIG);
    let rendered = render(&tok, &tools(), None);
    let expected = format!(
        "{SORTED_DECLARATION}<|im_user|>user<|im_middle|>What is the weather in Paris?<|im_end|>\
         <|im_assistant|>assistant<|im_middle|><|tool_call_begin|>call_1<|tool_call_argument_begin|>\
         {{\"unit\": \"celsius\", \"city\": \"Paris\"}}<|tool_call_end|><|im_end|>"
    );
    assert_eq!(rendered, expected);
}

#[test]
fn a_prepared_tool_set_renders_the_same_bytes() {
    let tok = tokenizer(KIMI_TOKENIZER_CONFIG);
    let tools = tools();
    let plain = render(&tok, &tools, None);
    let prepared = PreparedTools::new(tools.clone());
    for _ in 0..2 {
        assert_eq!(render(&tok, prepared.tools(), Some(&prepared)), plain);
    }
    assert!(plain.starts_with(SORTED_DECLARATION), "{plain}");
}

#[test]
fn a_checkpoint_of_another_class_keeps_the_requests_order() {
    let tok = tokenizer("{}");
    let rendered = render(&tok, &tools(), None);
    assert!(
        rendered.starts_with(
            r#"<|im_system|>tool_declare<|im_middle|>[{"type":"function","function":{"name":"get_weather","description":"#
        ),
        "{rendered}"
    );
}
