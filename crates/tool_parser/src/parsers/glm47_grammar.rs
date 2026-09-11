use std::{collections::BTreeSet, sync::LazyLock};

use openai_protocol::common::Tool;
use serde_json::Value;

const SPECIAL_TOKENS: &[&str] = &[
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<|assistant|>",
];

static TEXT_RULES: LazyLock<String> = LazyLock::new(|| string_excluding("text", SPECIAL_TOKENS));
static THINKING_RULES: LazyLock<String> =
    LazyLock::new(|| string_excluding("thinking", &SPECIAL_TOKENS[1..8]));

const JSON_RULES: &str = r#"
json_string ::= "\"" ([^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [a-fA-F0-9]{4}))* "\""
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
ws ::= [ \n\t]*
array ::= "[" (ws value (ws "," ws value)*)? ws "]"
object ::= "{" (ws json_string ws ":" ws value (ws "," ws json_string ws ":" ws value)*)? ws "}"
value ::= number | boolean | null | json_string | array | object
"#;

fn literal(value: &str) -> String {
    Value::String(value.to_owned()).to_string()
}

fn string_excluding(name: &str, patterns: &[&str]) -> String {
    let mut prefixes = BTreeSet::from([String::new()]);
    let mut alphabet = BTreeSet::new();
    for pattern in patterns {
        for (index, character) in pattern.char_indices() {
            prefixes.insert(pattern[..index].to_owned());
            alphabet.insert(character);
        }
    }
    let prefixes: Vec<_> = prefixes.into_iter().collect();
    let excluded: String = alphabet
        .iter()
        .map(|character| match character {
            '\\' | ']' | '^' | '-' => format!("\\{character}"),
            _ => character.to_string(),
        })
        .collect();
    let mut rules = vec![format!("{name} ::= {name}_0")];
    for (index, prefix) in prefixes.iter().enumerate() {
        let mut choices = vec![format!("[^{excluded}] {name}_0"), "\"\"".to_owned()];
        for character in &alphabet {
            let candidate = format!("{prefix}{character}");
            if patterns.iter().any(|pattern| candidate.ends_with(pattern)) {
                continue;
            }
            let target = prefixes
                .iter()
                .enumerate()
                .filter(|(_, suffix)| candidate.ends_with(suffix.as_str()))
                .max_by_key(|(_, suffix)| suffix.len())
                .map_or(0, |(target, _)| target);
            choices.push(format!(
                "{} {name}_{target}",
                literal(&character.to_string())
            ));
        }
        rules.push(format!("{name}_{index} ::= {}", choices.join(" | ")));
    }
    rules.join("\n")
}

fn type_rule(schema_type: &str) -> &'static str {
    match schema_type {
        "integer" | "number" => "number",
        "boolean" => "boolean",
        "null" => "null",
        "array" => "array",
        "object" => "object",
        _ => "text",
    }
}

fn value_rule(schema: &Value) -> Result<String, String> {
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        if values.is_empty() {
            return Err("Tool parameter enum must not be empty".to_owned());
        }
        return Ok(values
            .iter()
            .map(|value| match value {
                Value::String(value) => literal(value),
                _ => literal(&value.to_string()),
            })
            .collect::<Vec<_>>()
            .join(" | "));
    }
    match schema.get("type") {
        Some(Value::String(schema_type)) => Ok(type_rule(schema_type).to_owned()),
        Some(Value::Array(schema_types)) if !schema_types.is_empty() => Ok(schema_types
            .iter()
            .map(|schema_type| type_rule(schema_type.as_str().unwrap_or("string")))
            .collect::<Vec<_>>()
            .join(" | ")),
        _ => Ok("text".to_owned()),
    }
}

pub(crate) fn generate(tools: &[Tool], enable_thinking: bool) -> Result<String, String> {
    let mut rules = vec![
        "root ::= assistant_turn".to_owned(),
        "assistant_turn ::= thinking_block text tool_calls".to_owned(),
        if enable_thinking {
            "thinking_block ::= thinking \"</think>\"".to_owned()
        } else {
            "thinking_block ::= \"\"".to_owned()
        },
        TEXT_RULES.clone(),
    ];
    if enable_thinking {
        rules.push(THINKING_RULES.clone());
    }
    if tools.is_empty() {
        rules.push("tool_calls ::= \"\"".to_owned());
    } else {
        rules.push("tool_calls ::= tool_call*".to_owned());
        rules.push(format!(
            "tool_call ::= \"<tool_call>\" ({}) \"</tool_call>\"",
            (0..tools.len())
                .map(|index| format!("call_{index}"))
                .collect::<Vec<_>>()
                .join(" | ")
        ));
        for (index, tool) in tools.iter().enumerate() {
            let mut arguments = Vec::new();
            if let Some(properties) = tool
                .function
                .parameters
                .get("properties")
                .and_then(Value::as_object)
            {
                for (key, schema) in properties {
                    arguments.push(format!(
                        "{} ({}) \"</arg_value>\"",
                        literal(&format!("<arg_key>{key}</arg_key><arg_value>")),
                        value_rule(schema)?,
                    ));
                }
            }
            let arguments = if arguments.is_empty() {
                "\"\"".to_owned()
            } else {
                format!("({})*", arguments.join(" | "))
            };
            rules.push(format!(
                "call_{index} ::= {} {arguments}",
                literal(&tool.function.name),
            ));
        }
        rules.push(JSON_RULES.to_owned());
    }
    Ok(rules.join("\n"))
}
