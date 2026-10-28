//! The DeepSeek V3.1 checkpoint template renders a request's tools as a
//! system message block the template itself never writes (see the tokenizer
//! crate's `huggingface::deepseek_v31`): a checkpoint of the V3 architecture
//! family whose template is the hybrid-thinking one gets the block, byte for
//! byte what the engines' tool chat template for the family renders between
//! the system prompt and the first user turn; a request without tools, the
//! family's earlier templates, another architecture with the same template
//! and a replacement template that reads `tools` render as before.
//!
//! The expected prompts were produced with transformers' `apply_chat_template`
//! on the checkpoint's template, with the block the engines' tool template
//! renders (jinja2 under transformers' settings) placed into the system
//! prompt as the model card documents it, `{system prompt}\n\n{tool
//! description}`.

#![allow(
    clippy::unwrap_used,
    reason = "test code: a failing setup step is a failing test"
)]

use std::{collections::HashMap, fs};

use llm_tokenizer::{
    chat_template::{ChatTemplateParams, ThinkingKeyName, ThinkingToggle},
    huggingface::HuggingFaceTokenizer,
    TokenizerTrait,
};
use serde_json::{json, Value};
use tempfile::TempDir;

use crate::common::MIN_TOKENIZER_JSON_WITH_SENTINELS;

/// The chat template of the V3.1 checkpoint (its `tokenizer_config.json`).
const V31_TEMPLATE: &str = r"{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if not thinking is defined %}{% set thinking = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, system_prompt='', is_first_sp=true, is_last_user=false) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '

' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}{{ bos_token }}{{ ns.system_prompt }}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{%- set ns.is_first = false -%}{%- set ns.is_last_user = true -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['tool_calls'] is defined and message['tool_calls'] is not none %}{%- if ns.is_last_user %}{{'<｜Assistant｜></think>'}}{%- endif %}{%- set ns.is_last_user = false -%}{%- set ns.is_first = false %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls'] %}{%- if not ns.is_first %}{%- if message['content'] is none %}{{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}{%- else %}{{message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}{%- endif %}{%- set ns.is_first = true -%}{%- else %}{{'<｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}{%- endif %}{%- endfor %}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- if message['role'] == 'assistant' and (message['tool_calls'] is not defined or message['tool_calls'] is none) %}{%- if ns.is_last_user %}{{'<｜Assistant｜>'}}{%- if message['prefix'] is defined and message['prefix'] and thinking %}{{'<think>'}}  {%- else %}{{'</think>'}}{%- endif %}{%- endif %}{%- set ns.is_last_user = false -%}{%- if ns.is_tool %}{{message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{%- set content = message['content'] -%}{%- if '</think>' in content %}{%- set content = content.split('</think>', 1)[1] -%}{%- endif %}{{content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_last_user = false -%}{%- set ns.is_tool = true -%}{{'<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endfor -%}{%- if add_generation_prompt and ns.is_last_user and not ns.is_tool %}{{'<｜Assistant｜>'}}{%- if not thinking %}{{'</think>'}}{%- else %}{{'<think>'}}{%- endif %}{% endif %}";

/// A template of the family's earlier releases in miniature: the system
/// texts, the turns, no thinking switch, nothing for `tools`.
const EARLIER_TEMPLATE: &str = r"{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% endif %}{% endfor %}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<｜User｜>' + message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<｜Assistant｜>' }}{% endif %}";

/// A replacement template with the V3.1 thinking switch that renders the
/// request's tools itself, as an engine's tool template does.
const TOOLS_AWARE_TEMPLATE: &str = r"{% if not thinking is defined %}{% set thinking = false %}{% endif %}{{ bos_token }}{% if tools is defined and tools is not none %}TOOLS:{{ tools | length }}{% endif %}{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}<｜Assistant｜>{% if not thinking %}</think>{% else %}<think>{% endif %}{% endif %}";

/// The issue's request: one small tool, the user's question, no system prompt.
const ONE_TOOL: &str = r#"<｜begin▁of▁sentence｜>

## Tools
You have access to the following tools:

### get_weather
Description: Get the current weather for a city

Parameters: {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- `tool_call_name` must be an exact match to one of the available tools
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>What is the weather in Paris today?<｜Assistant｜></think>"#;
/// The same request without tools: the checkpoint template's own prompt.
const NO_TOOLS: &str =
    r"<｜begin▁of▁sentence｜><｜User｜>What is the weather in Paris today?<｜Assistant｜></think>";
/// A system prompt and two tools, one of them without a description.
const SYSTEM_AND_TWO_TOOLS: &str = r#"<｜begin▁of▁sentence｜>You are a terse assistant.

## Tools
You have access to the following tools:

### get_weather
Description: Get the current weather for a city

Parameters: {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}

### get_stock_price
Description: Get the latest stock price for a ticker symbol

Parameters: {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- `tool_call_name` must be an exact match to one of the available tools
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>What is the weather in Paris?<｜Assistant｜></think>"#;
/// Thinking on: the block, then the open think block of the generation prompt.
const THINKING_ONE_TOOL: &str = r#"<｜begin▁of▁sentence｜>

## Tools
You have access to the following tools:

### is_prime
Description: Primality test

Parameters: {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- `tool_call_name` must be an exact match to one of the available tools
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>Is 97 prime?<｜Assistant｜><think>"#;
/// A tool call and its result in the history, rendered by the template as before.
const HISTORY_ONE_TOOL: &str = r#"<｜begin▁of▁sentence｜>

## Tools
You have access to the following tools:

### get_weather
Description: Get the current weather for a city

Parameters: {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- `tool_call_name` must be an exact match to one of the available tools
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>What is the weather in Paris?<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>{"temperature": 18, "unit": "celsius"}<｜tool▁output▁end｜>"#;

fn weather_tool() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        }, "required": ["city"]}
    }})
}

fn stock_tool() -> Value {
    json!({"type": "function", "function": {
        "name": "get_stock_price",
        "description": "Get the latest stock price for a ticker symbol",
        "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
    }})
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

/// A model directory with the architecture, the template inside
/// `tokenizer_config.json` beside the checkpoint's special tokens, and a
/// tokenizer that loads.
fn load(architecture: &str, template: &str) -> (TempDir, HuggingFaceTokenizer) {
    let dir = TempDir::new().unwrap();
    let tokenizer_path = dir.path().join("tokenizer.json");
    fs::write(&tokenizer_path, MIN_TOKENIZER_JSON_WITH_SENTINELS).unwrap();
    fs::write(
        dir.path().join("config.json"),
        json!({"architectures": [architecture], "model_type": "deepseek_v3"}).to_string(),
    )
    .unwrap();
    fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "chat_template": template,
        })
        .to_string(),
    )
    .unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
    (dir, tokenizer)
}

fn render(tokenizer: &HuggingFaceTokenizer, messages: &[Value], tools: &[Value]) -> String {
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        tools: (!tools.is_empty()).then_some(tools),
        ..ChatTemplateParams::default()
    };
    tokenizer.apply_chat_template(messages, params).unwrap()
}

#[test]
fn one_tool_renders_the_block_before_the_first_user_turn() {
    let (_dir, tokenizer) = load("DeepseekV3ForCausalLM", V31_TEMPLATE);
    let messages = [user("What is the weather in Paris today?")];
    assert_eq!(render(&tokenizer, &messages, &[weather_tool()]), ONE_TOOL);
}

#[test]
fn no_tools_renders_the_checkpoint_template_unchanged() {
    let (_dir, tokenizer) = load("DeepseekV3ForCausalLM", V31_TEMPLATE);
    let messages = [user("What is the weather in Paris today?")];
    assert_eq!(render(&tokenizer, &messages, &[]), NO_TOOLS);
    assert_eq!(
        tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: true,
                    tools: Some(&[]),
                    ..ChatTemplateParams::default()
                }
            )
            .unwrap(),
        NO_TOOLS
    );
}

#[test]
fn the_system_prompt_precedes_the_block() {
    let (_dir, tokenizer) = load("DeepseekV3ForCausalLM", V31_TEMPLATE);
    let messages = [
        json!({"role": "system", "content": "You are a terse assistant."}),
        user("What is the weather in Paris?"),
    ];
    assert_eq!(
        render(&tokenizer, &messages, &[weather_tool(), stock_tool()]),
        SYSTEM_AND_TWO_TOOLS
    );
}

#[test]
fn a_root_message_renders_as_the_system_prompt_before_the_block() {
    // The template has no `root` branch: the renderer hands the message on
    // as the leading system message, and the block follows its text.
    let (_dir, tokenizer) = load("DeepseekV3ForCausalLM", V31_TEMPLATE);
    let messages = [
        json!({"role": "root", "content": "You are a terse assistant."}),
        user("What is the weather in Paris?"),
    ];
    assert_eq!(
        render(&tokenizer, &messages, &[weather_tool(), stock_tool()]),
        SYSTEM_AND_TWO_TOOLS
    );
}

#[test]
fn thinking_mode_keeps_the_open_think_block_after_the_block() {
    let (_dir, tokenizer) = load("DeepseekV3ForCausalLM", V31_TEMPLATE);
    let messages = [user("Is 97 prime?")];
    let tools = [json!({"type": "function", "function": {
        "name": "is_prime", "description": "Primality test",
        "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    }})];
    let kwargs: HashMap<String, Value> = HashMap::from([("thinking".to_string(), json!(true))]);
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        tools: Some(&tools),
        template_kwargs: Some(&kwargs),
        ..ChatTemplateParams::default()
    };
    assert_eq!(
        tokenizer.apply_chat_template(&messages, params).unwrap(),
        THINKING_ONE_TOOL
    );
}

#[test]
fn history_turns_stay_the_template_s_own() {
    let (_dir, tokenizer) = load("DeepseekV3ForCausalLM", V31_TEMPLATE);
    let messages = [
        user("What is the weather in Paris?"),
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}),
        json!({"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature\": 18, \"unit\": \"celsius\"}"}),
    ];
    let tools = [json!({"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}
    }})];
    assert_eq!(render(&tokenizer, &messages, &tools), HISTORY_ONE_TOOL);
}

#[test]
fn the_encoding_path_renders_the_block_too() {
    let (_dir, tokenizer) = load("DeepseekV3ForCausalLM", V31_TEMPLATE);
    let messages = [user("What is the weather in Paris today?")];
    let tools = [weather_tool()];
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        tools: Some(&tools),
        ..ChatTemplateParams::default()
    };
    let output = tokenizer
        .apply_chat_template_with_encoding(&messages, params, None)
        .unwrap();
    assert_eq!(output.text, ONE_TOOL);
    assert!(!output.continued_final_message);
}

#[test]
fn the_template_answers_the_thinking_questions_as_before() {
    let (_dir, tokenizer) = load("DeepseekV3ForCausalLM", V31_TEMPLATE);
    assert_eq!(tokenizer.thinking_toggle(), ThinkingToggle::DefaultOff);
    assert_eq!(
        tokenizer.thinking_key_name(),
        Some(ThinkingKeyName::Thinking)
    );
    assert!(tokenizer.think_in_prefill());
    assert!(
        tokenizer
            .renderer_capabilities()
            .native_assistant_continuation
    );
}

#[test]
fn the_family_s_earlier_templates_get_no_block() {
    let (_dir, tokenizer) = load("DeepseekV3ForCausalLM", EARLIER_TEMPLATE);
    let messages = [
        json!({"role": "system", "content": "Be terse."}),
        user("Hi"),
    ];
    assert_eq!(
        render(&tokenizer, &messages, &[weather_tool()]),
        "<｜begin▁of▁sentence｜>Be terse.<｜User｜>Hi<｜Assistant｜>"
    );
}

#[test]
fn another_architecture_with_the_same_template_gets_no_block() {
    let (_dir, tokenizer) = load("LlamaForCausalLM", V31_TEMPLATE);
    let messages = [user("What is the weather in Paris today?")];
    assert_eq!(render(&tokenizer, &messages, &[weather_tool()]), NO_TOOLS);
}

#[test]
fn a_replacement_template_that_reads_tools_keeps_its_rendering() {
    let (_dir, mut tokenizer) = load("DeepseekV3ForCausalLM", V31_TEMPLATE);
    let messages = [user("Hi")];
    let tools = [weather_tool()];
    assert!(render(&tokenizer, &messages, &tools).contains("## Tools\n"));

    tokenizer
        .set_chat_template(TOOLS_AWARE_TEMPLATE.to_string())
        .unwrap();
    assert_eq!(
        render(&tokenizer, &messages, &tools),
        "<｜begin▁of▁sentence｜>TOOLS:1<|user|>Hi<｜Assistant｜></think>"
    );

    tokenizer
        .set_chat_template(V31_TEMPLATE.to_string())
        .unwrap();
    assert!(render(&tokenizer, &messages, &tools).contains("## Tools\n"));
}

/// An engine's tool template for the family in miniature: it serializes a
/// history call's arguments object instead of concatenating the string.
const SERIALIZING_TEMPLATE: &str = r"{{ bos_token }}{% for m in messages %}{% if m.tool_calls %}{% for t in m.tool_calls %}<call>{{ t.function.name }}:{{ t.function.arguments | tojson }}</call>{% endfor %}{% else %}<|{{ m.role }}|>{{ m.content }}{% endif %}{% endfor %}";

/// The first user turn after a call and its result, no tools in the request:
/// the checkpoint's template run by transformers' `apply_chat_template` on
/// the request as the API carries it, the call's arguments the string sent.
const FIRST_TURN_AFTER_A_CALL: &str = r#"<｜begin▁of▁sentence｜><｜User｜>What is the weather in Paris in celsius?<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Paris", "unit": "celsius"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>{"temp_c": 18, "sky": "cloudy"}<｜tool▁output▁end｜><｜User｜>Thanks. Now the same for Rome, also in celsius, with the tool.<｜Assistant｜></think>"#;

/// The same oracle on arguments spelled compactly, with non-ASCII text and
/// escapes: the template writes the string as sent, nothing re-serialized.
const ARGUMENTS_AS_WRITTEN: &str = r#"<｜begin▁of▁sentence｜><｜User｜>Weather in Zürich?<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Zürich", "note": "a\"b\\c 東京"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city":"Rome","unit":"celsius"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>{"temp_c": 12}<｜tool▁output▁end｜><｜tool▁output▁begin｜>{"temp_c": 24}<｜tool▁output▁end｜><｜User｜>Thanks.<｜Assistant｜></think>"#;

fn call(id: &str, arguments: &str) -> Value {
    json!({"id": id, "type": "function", "function": {"name": "get_weather", "arguments": arguments}})
}

fn tool_result(id: &str, text: &str) -> Value {
    json!({"role": "tool", "tool_call_id": id, "content": text})
}

/// The checkpoint's template writes a history call's arguments by string
/// concatenation, so it takes them as the text the request sent and says so
/// in its renderer capabilities (the gateway hands them on unparsed); a
/// replacement template that serializes the object answers for itself.
#[test]
fn the_checkpoint_template_takes_tool_call_arguments_as_the_text_the_request_sent() {
    let (_dir, mut tokenizer) = load("DeepseekV3ForCausalLM", V31_TEMPLATE);
    assert!(tokenizer.renderer_capabilities().raw_tool_call_arguments);
    tokenizer
        .set_chat_template(SERIALIZING_TEMPLATE.to_string())
        .unwrap();
    assert!(!tokenizer.renderer_capabilities().raw_tool_call_arguments);
    tokenizer
        .set_chat_template(V31_TEMPLATE.to_string())
        .unwrap();
    assert!(tokenizer.renderer_capabilities().raw_tool_call_arguments);
}

/// A history with a call and its result, then the first user turn after it,
/// renders byte for byte what the checkpoint's template renders for the
/// request as the API carries it; the arguments keep their spelling.
#[test]
fn a_history_call_renders_with_its_arguments_as_the_request_sent_them() {
    let (_dir, tokenizer) = load("DeepseekV3ForCausalLM", V31_TEMPLATE);
    let messages = [
        user("What is the weather in Paris in celsius?"),
        json!({"role": "assistant", "content": "", "tool_calls": [
            call("call_1", r#"{"city": "Paris", "unit": "celsius"}"#)
        ]}),
        tool_result("call_1", r#"{"temp_c": 18, "sky": "cloudy"}"#),
        user("Thanks. Now the same for Rome, also in celsius, with the tool."),
    ];
    assert_eq!(render(&tokenizer, &messages, &[]), FIRST_TURN_AFTER_A_CALL);

    let messages = [
        user("Weather in Zürich?"),
        json!({"role": "assistant", "content": null, "tool_calls": [
            call("call_1", r#"{"city": "Zürich", "note": "a\"b\\c 東京"}"#),
            call("call_2", r#"{"city":"Rome","unit":"celsius"}"#),
        ]}),
        tool_result("call_1", r#"{"temp_c": 12}"#),
        tool_result("call_2", r#"{"temp_c": 24}"#),
        user("Thanks."),
    ];
    assert_eq!(render(&tokenizer, &messages, &[]), ARGUMENTS_AS_WRITTEN);
}
