//! Checkpoint-specific DeepSeek V3.1 behavior: the tools block its template
//! lacks. The V3.1 release (the `DeepseekV3ForCausalLM` architecture with the
//! hybrid-thinking template, which reads a `thinking` switch that is off by
//! default) ships a chat template that renders every turn but never reads
//! the request's `tools`, so a request with tools renders the prompt of the
//! request without them and the model is never told what it may call. The
//! model card documents the format instead, `{system prompt}\n\n{tool
//! description}` before the first user turn, and the engines' tool chat
//! templates for the family write that block into the system prompt. This
//! module renders the block as those templates do, byte for byte, and hands
//! it to the checkpoint's own template as a system message, so the template
//! keeps rendering the turns, the history and the thinking switch itself.

use std::path::Path;

use serde_json::{json, Value};

use crate::{
    chat_template::{ChatTemplateParams, ChatTemplateState, ThinkingKeyName, ThinkingToggle},
    json_dumps,
};

/// Presence identifies a checkpoint of the V3 architecture family. Whether
/// the tools block applies is the active template's question
/// ([`Checkpoint::renders_tools_block`]): the family's other releases share
/// the architecture with formats of their own, and a replacement template
/// may render the tools itself.
pub(super) struct Checkpoint;

impl Checkpoint {
    /// Recognize the architecture family from `config.json`.
    pub(super) fn load(dir: &Path) -> Option<Self> {
        let config: Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
        config
            .get("architectures")?
            .as_array()?
            .iter()
            .any(|arch| arch.as_str() == Some("DeepseekV3ForCausalLM"))
            .then_some(Self)
    }

    /// Whether the gateway writes the tools block for the active template:
    /// the V3.1 hybrid-thinking template (a `thinking` switch, off by
    /// default; the family's earlier templates have no switch) when it
    /// renders a request with a tool as it renders the request without one.
    /// A template that reads `tools`, such as an engine's tool template set
    /// in place of the checkpoint's, keeps its own rendering.
    pub(super) fn renders_tools_block(template: &ChatTemplateState) -> bool {
        template.thinking_key_name() == Some(ThinkingKeyName::Thinking)
            && template.thinking_toggle() == ThinkingToggle::DefaultOff
            && ignores_tools(template)
    }
}

/// Compare two renders of one turn, with and without a tool, instead of
/// matching the template source: what counts is whether the active template
/// writes anything for the request's tools.
fn ignores_tools(template: &ChatTemplateState) -> bool {
    let messages = [json!({"role": "user", "content": "probe"})];
    let tools = [json!({"type": "function", "function": {
        "name": "probe", "description": "probe", "parameters": {"type": "object", "properties": {}}
    }})];
    let without = template.apply(&messages, ChatTemplateParams::default());
    let with = template.apply(
        &messages,
        ChatTemplateParams {
            tools: Some(&tools),
            ..ChatTemplateParams::default()
        },
    );
    matches!((without, with), (Ok(without), Ok(with)) if without == with)
}

/// The messages with the tools block as a system message right after the
/// last system message, so the template's system prompt (the system texts
/// joined by a blank line) ends with `\n\n{tool description}` as the model
/// card shows; a developer or root message counts, since the template has no
/// branch for either role and the renderer hands both on as system messages
/// (merged into the leading one, in their order). Without a system message
/// the block comes first, behind the blank line the engines' templates write
/// before it unconditionally. `None` when the request has no tools, so the
/// caller keeps its slice.
pub(super) fn messages_with_tools_block(
    messages: &[Value],
    tools: Option<&[Value]>,
) -> Option<Vec<Value>> {
    let tools = tools.filter(|tools| !tools.is_empty())?;
    let block = tools_block(tools);
    let last_system = messages.iter().rposition(|message| {
        matches!(
            message.get("role").and_then(Value::as_str),
            Some("system" | "developer" | "root")
        )
    });
    let mut owned = Vec::with_capacity(messages.len() + 1);
    match last_system {
        Some(index) => {
            owned.extend_from_slice(&messages[..=index]);
            owned.push(json!({"role": "system", "content": block}));
            owned.extend_from_slice(&messages[index + 1..]);
        }
        None => {
            owned.push(json!({"role": "system", "content": format!("\n\n{block}")}));
            owned.extend_from_slice(messages);
        }
    }
    Some(owned)
}

/// The call format the block ends with, as the engines' tool templates spell
/// it (the model card's prose differs in a blank line, the braces around
/// `additional_tool_calls` and the final newline; the engines' servers are
/// what the gateway's prompt is compared with).
const CALL_FORMAT: &str = "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n\
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>\
{{additional_tool_calls}}<｜tool▁calls▁end｜>\n\nWhere:\n\n\
- `tool_call_name` must be an exact match to one of the available tools\n\
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema\n\
- For multiple tool calls, chain them directly without separators or spaces\n";

/// The block the engines' tool templates for the family write into the
/// system prompt: the heading, each tool's name, its description when it has
/// one and its parameters as Python's `json.dumps` spells them, then the call
/// format. A tool without a function object renders nothing, as the
/// templates' `tool.function` reads would.
fn tools_block(tools: &[Value]) -> String {
    let mut block = String::from("## Tools\nYou have access to the following tools:\n");
    for function in tools.iter().filter_map(|tool| tool.get("function")) {
        block.push_str("\n### ");
        block.push_str(function.get("name").and_then(Value::as_str).unwrap_or(""));
        if let Some(description) = function.get("description").and_then(Value::as_str) {
            block.push_str("\nDescription: ");
            block.push_str(description);
        }
        block.push_str("\n\nParameters: ");
        block.push_str(&json_dumps::to_string(
            function.get("parameters").unwrap_or(&Value::Null),
        ));
        block.push('\n');
    }
    block.push_str(CALL_FORMAT);
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block the engines' tool chat template for the family renders for
    /// this tool (jinja2 under transformers' settings, the `tojson` filter as
    /// transformers defines it), read back between the system prompt and the
    /// first user turn.
    const WEATHER_BLOCK: &str = "## Tools\nYou have access to the following tools:\n\n\
### get_weather\nDescription: Get the current weather for a city\n\n\
Parameters: {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"City name\"}, \
\"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}}, \"required\": [\"city\"]}\n\
\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n\
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>\
{{additional_tool_calls}}<｜tool▁calls▁end｜>\n\nWhere:\n\n\
- `tool_call_name` must be an exact match to one of the available tools\n\
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema\n\
- For multiple tool calls, chain them directly without separators or spaces\n";

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

    fn content(message: &Value) -> &str {
        message["content"].as_str().unwrap()
    }

    #[test]
    fn block_matches_the_engine_template_byte_for_byte() {
        assert_eq!(tools_block(&[weather_tool()]), WEATHER_BLOCK);
    }

    #[test]
    fn every_tool_gets_its_entry_and_a_missing_description_its_own_shape() {
        let tools = [
            weather_tool(),
            json!({"type": "function", "function": {
                "name": "get_stock_price",
                "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
            }}),
        ];
        let block = tools_block(&tools);
        let (entries, format) = block.split_once("\nIMPORTANT").unwrap();
        assert!(entries.ends_with(
            "\n### get_weather\nDescription: Get the current weather for a city\n\n\
             Parameters: {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \
             \"description\": \"City name\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \
             \"fahrenheit\"]}}, \"required\": [\"city\"]}\n\
             \n### get_stock_price\n\nParameters: {\"type\": \"object\", \"properties\": {\"ticker\": \
             {\"type\": \"string\"}}, \"required\": [\"ticker\"]}\n"
        ));
        assert_eq!(format!("\nIMPORTANT{format}"), CALL_FORMAT);
    }

    #[test]
    fn parameters_keep_python_json_dumps_spelling_and_raw_unicode() {
        let tools = [json!({"type": "function", "function": {
            "name": "get_weather", "description": "查询城市天气 <city>",
            "parameters": {"type": "object", "properties": {"城市": {"type": "string", "description": "名称 & \"引号\""}}, "required": ["城市"]}
        }})];
        assert!(tools_block(&tools).contains(
            "\n### get_weather\nDescription: 查询城市天气 <city>\n\nParameters: {\"type\": \"object\", \
             \"properties\": {\"城市\": {\"type\": \"string\", \"description\": \"名称 & \\\"引号\\\"\"}}, \
             \"required\": [\"城市\"]}\n"
        ));
    }

    #[test]
    fn no_tools_keeps_the_messages() {
        let messages = [json!({"role": "user", "content": "Hi"})];
        assert!(messages_with_tools_block(&messages, None).is_none());
        assert!(messages_with_tools_block(&messages, Some(&[])).is_none());
    }

    #[test]
    fn without_a_system_message_the_block_leads_behind_a_blank_line() {
        let messages = [json!({"role": "user", "content": "Hi"})];
        let tools = [weather_tool()];
        let rewritten = messages_with_tools_block(&messages, Some(&tools)).unwrap();
        assert_eq!(rewritten.len(), 2);
        assert_eq!(rewritten[0]["role"], "system");
        assert_eq!(content(&rewritten[0]), format!("\n\n{WEATHER_BLOCK}"));
        assert_eq!(rewritten[1], messages[0]);
    }

    #[test]
    fn a_root_message_counts_as_the_system_prompt() {
        let tools = [weather_tool()];
        let messages = [
            json!({"role": "root", "content": "R"}),
            json!({"role": "user", "content": "Hi"}),
        ];
        let rewritten = messages_with_tools_block(&messages, Some(&tools)).unwrap();
        let roles: Vec<&str> = rewritten
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["root", "system", "user"]);
        assert_eq!(content(&rewritten[1]), WEATHER_BLOCK);
    }

    #[test]
    fn the_block_follows_the_last_system_or_developer_message() {
        let tools = [weather_tool()];
        let messages = [
            json!({"role": "system", "content": "First."}),
            json!({"role": "user", "content": "Hi"}),
            json!({"role": "developer", "content": "Second."}),
            json!({"role": "user", "content": "Again"}),
        ];
        let rewritten = messages_with_tools_block(&messages, Some(&tools)).unwrap();
        let roles: Vec<&str> = rewritten
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["system", "user", "developer", "system", "user"]);
        assert_eq!(content(&rewritten[3]), WEATHER_BLOCK);
        assert_eq!(rewritten[4], messages[3]);
    }
}
