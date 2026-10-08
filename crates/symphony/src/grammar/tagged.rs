//! The tagged syntax written from the table: `<function=NAME>`, then `<parameter=KEY>` around
//! each value, `</function>`, between the call markers: Qwen 3.5 and later, Qwen3-Coder,
//! Seed-OSS. The engines' `qwen_xml_parameter` style writes Qwen's spelling of the parameters and
//! nothing else, so the tags are written here.
//!
//! A call is the call's opener, a newline, `<function=NAME>` and a newline; then one parameter per
//! property of the tool's schema, in the schema's order, each optional unless the schema requires
//! it: `<parameter=KEY>`, the value, `</parameter>` and a newline; then `</function>`, a newline
//! and the call's close. A newline may stand on either side of the value: Qwen 3.5's template puts
//! the value on a line of its own, Seed-OSS's and MiMo's write it between the tags directly, and
//! the assembler takes one newline away on each side when it is there.
//!
//! The value follows the property's type. A string is written as it is, any text up to the
//! parameter's close or a marker that ends the call. A number is JSON. A boolean or null is JSON
//! or Python's word (`True`, `False`, `None`), both of which the templates write and the assembler
//! reads. An object or a list is JSON the property's schema accepts where the template writes
//! JSON (Qwen's `tojson`), and any text where the template writes Python's repr (Seed-OSS's
//! `str`), which the assembler reads as the JSON it stands for ([`Spelling`]). A property without
//! a type, or a schema without properties, takes any text. When the calls sit in a block, the
//! block's ways in frame one or more calls, as Kimi K3's do.

use openai_protocol::common::Tool;
use serde_json::{json, Value};

use super::{
    block_of,
    schema::{self, shape, Definitions},
    CallMarkers, Grammar, Tag, WayIn,
};
use crate::tagged::{assembler::TAGS, Spelling};

const FUNCTION_OPEN: &str = TAGS[0];
const PARAMETER_OPEN: &str = TAGS[1];
const PARAMETER_CLOSE: &str = TAGS[2];
const FUNCTION_CLOSE: &str = TAGS[3];

/// The calls for `tools`: in a block, the block's ways in around one or more calls; without one,
/// one tag per tool, triggered by the call's opener.
pub(super) fn calls(
    markers: &CallMarkers<'_>,
    ways_in: &[WayIn<'_>],
    spelling: Spelling,
    tools: &[&Tool],
    at_least_one: bool,
) -> Grammar {
    let values = Values {
        spelling,
        call_open: markers.call_open,
        call_close: markers.call_close,
        block_close: markers.block.as_ref().map(|block| block.close),
    };
    let calls = tools.iter().map(|tool| {
        call(
            markers,
            &values,
            &tool.function.name,
            &tool.function.parameters,
        )
    });
    match markers.block.as_ref() {
        Some(block) => {
            let one_call = Grammar::Or(
                calls
                    .map(|(begin, content, end)| {
                        Grammar::Sequence(vec![
                            Grammar::ConstString(begin),
                            content,
                            Grammar::ConstString(end),
                        ])
                    })
                    .collect(),
            );
            block_of(
                ways_in,
                Grammar::Plus(Box::new(one_call)),
                block.close,
                at_least_one,
            )
        }
        None => Grammar::TriggeredTags {
            triggers: vec![markers.call_open.to_string()],
            tags: calls
                .map(|(begin, content, end)| Tag::new(begin, content, end))
                .collect(),
            at_least_one,
        },
    }
}

/// How a table's values are written: the family's spelling of what is not a string, and the
/// markers that end a value's text, which the engine takes wherever they stand (the call's opener
/// among them, since a call opened before the last closed ends it).
struct Values<'a> {
    spelling: Spelling,
    call_open: &'a str,
    call_close: &'a str,
    block_close: Option<&'a str>,
}

impl Values<'_> {
    /// A value written as it is: any text up to the parameter's close or a marker that ends the
    /// call.
    fn text(&self) -> Grammar {
        let mut excludes = vec![
            PARAMETER_CLOSE.to_string(),
            self.call_open.to_string(),
            self.call_close.to_string(),
        ];
        excludes.extend(self.block_close.map(str::to_string));
        Grammar::AnyText { excludes }
    }

    /// The value of a property, by the type its schema pins and the family's spelling.
    fn of(&self, property: &Value, definitions: Definitions<'_>) -> Grammar {
        match shape(property, definitions, || self.text()) {
            Some(("boolean", _)) => words(json!({"type": "boolean"}), ["True", "False"]),
            Some(("null", _)) => words_only(["null", "None"]),
            Some(("object" | "array", pinned)) => match self.spelling {
                Spelling::Json => pinned,
                Spelling::Python => self.text(),
            },
            Some((_, pinned)) => pinned,
            None => self.text(),
        }
    }
}

/// JSON the schema accepts, or one of Python's words for it.
fn words<const N: usize>(schema: Value, python: [&str; N]) -> Grammar {
    let mut options = vec![Grammar::JsonSchema {
        schema,
        style: None,
    }];
    options.extend(
        python
            .iter()
            .map(|word| Grammar::ConstString(word.to_string())),
    );
    Grammar::Or(options)
}

/// One of the words.
fn words_only<const N: usize>(spellings: [&str; N]) -> Grammar {
    Grammar::Or(
        spellings
            .iter()
            .map(|word| Grammar::ConstString(word.to_string()))
            .collect(),
    )
}

/// One call as its three parts: the opener, the function tag and its newlines; the parameters;
/// the function's close with the call's.
fn call(
    markers: &CallMarkers<'_>,
    values: &Values<'_>,
    name: &str,
    parameters: &Value,
) -> (String, Grammar, String) {
    (
        format!("{}\n{FUNCTION_OPEN}{name}>\n", markers.call_open),
        arguments(values, parameters),
        format!("{FUNCTION_CLOSE}\n{}", markers.call_close),
    )
}

/// The parameters the tool's schema asks for: one per property, in the schema's order, each
/// optional unless required; any parameters at all for a schema without properties.
fn arguments(values: &Values<'_>, parameters: &Value) -> Grammar {
    let parameter = |key: &str, property: &Value, definitions: Definitions<'_>| {
        argument(
            Grammar::ConstString(key.to_string()),
            values.of(property, definitions),
        )
    };
    schema::arguments(parameters, parameter, || {
        Grammar::Star(Box::new(any_argument(values)))
    })
}

/// One parameter: the opening tag with the key, the value with a newline allowed on either side,
/// the close and a newline.
fn argument(key: Grammar, value: Grammar) -> Grammar {
    let newline = || Grammar::Optional(Box::new(Grammar::ConstString("\n".to_string())));
    Grammar::Sequence(vec![
        Grammar::ConstString(PARAMETER_OPEN.to_string()),
        key,
        Grammar::ConstString(">".to_string()),
        newline(),
        value,
        newline(),
        Grammar::ConstString(format!("{PARAMETER_CLOSE}\n")),
    ])
}

/// A parameter with any key and any value.
fn any_argument(values: &Values<'_>) -> Grammar {
    let key = Grammar::AnyText {
        excludes: vec![">".to_string()],
    };
    argument(key, values.text())
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;
    use serde_json::json as value;

    use super::*;
    use crate::{format::CallSyntax, formats};

    fn tool(name: &str, parameters: Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
                strict: None,
            },
        }
    }

    fn weather_tools() -> Vec<Tool> {
        vec![
            tool(
                "get_weather",
                value!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer", "minimum": 1},
                        "metric": {"type": "boolean"},
                        "where": {"type": "object", "properties": {"lat": {"type": "number"}}},
                    },
                    "required": ["city", "days", "metric"],
                }),
            ),
            tool("ping", value!({"type": "object"})),
        ]
    }

    /// A parameter slot as the tag writes it: the tag with the key, the value, the close.
    fn slot(key: &str, value: Value) -> Value {
        let newline =
            value!({"type": "optional", "content": {"type": "const_string", "value": "\n"}});
        value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": "<parameter="},
            {"type": "const_string", "value": key},
            {"type": "const_string", "value": ">"},
            newline,
            value,
            newline,
            {"type": "const_string", "value": "</parameter>\n"},
        ]})
    }

    #[test]
    fn qwen_3_5_writes_each_parameter_with_a_newline_allowed_on_either_side_of_the_value() {
        // Qwen 3.5's template: `<tool_call>`, a newline, `<function=` and the name, `>` and a
        // newline; per argument `<parameter=` and the key, `>`, the value on a line of its own,
        // `</parameter>` and a newline; then `</function>`, a newline and `</tool_call>`. MiMo,
        // on the same table, writes the value between the tags directly, so the newlines are
        // allowed, not required. A string as it is; a boolean as JSON or Python's word; an object
        // as the JSON Qwen's `tojson` writes.
        let payload = formats::qwen3(CallSyntax::Tagged(Spelling::Json))
            .grammar(&weather_tools(), true, false)
            .expect("a grammar")
            .payload();
        let text = value!({
            "type": "any_text",
            "excludes": ["</parameter>", "<tool_call>", "</tool_call>"],
        });
        let days =
            value!({"type": "json_schema", "json_schema": {"type": "integer", "minimum": 1}});
        let metric = value!({"type": "or", "elements": [
            {"type": "json_schema", "json_schema": {"type": "boolean"}},
            {"type": "const_string", "value": "True"},
            {"type": "const_string", "value": "False"},
        ]});
        let place = value!({"type": "json_schema", "json_schema": {
            "type": "object", "properties": {"lat": {"type": "number"}},
        }});
        assert_eq!(
            payload,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": [
                    {
                        "type": "tag",
                        "begin": "<tool_call>\n<function=get_weather>\n",
                        "content": {"type": "sequence", "elements": [
                            slot("city", text.clone()),
                            slot("days", days),
                            slot("metric", metric),
                            {"type": "optional", "content": slot("where", place)},
                        ]},
                        "end": "</function>\n</tool_call>",
                    },
                    {
                        "type": "tag",
                        "begin": "<tool_call>\n<function=ping>\n",
                        "content": {"type": "star", "content": slot_any(text)},
                        "end": "</function>\n</tool_call>",
                    },
                ],
                "at_least_one": true,
            }})
        );
    }

    /// The slot of a parameter with any key.
    fn slot_any(text: Value) -> Value {
        let mut slot = slot("", text);
        slot["elements"][1] = value!({"type": "any_text", "excludes": [">"]});
        slot
    }

    #[test]
    fn seed_oss_writes_objects_as_pythons_text_under_its_own_markers() {
        // Seed-OSS's template writes every value with Python's `str`: a string as it is, an object
        // as its repr, which no JSON schema spells, so the object takes any text, and a boolean
        // as `True` or `False` beside JSON's words.
        let payload = formats::seed_oss()
            .grammar(&weather_tools(), false, false)
            .expect("a grammar")
            .payload();
        let format = &payload["format"];
        assert_eq!(format["triggers"], value!(["<seed:tool_call>"]));
        assert_eq!(format["at_least_one"], false);
        let weather = &format["tags"][0];
        assert_eq!(
            weather["begin"],
            "<seed:tool_call>\n<function=get_weather>\n"
        );
        assert_eq!(weather["end"], "</function>\n</seed:tool_call>");
        let text = value!({
            "type": "any_text",
            "excludes": ["</parameter>", "<seed:tool_call>", "</seed:tool_call>"],
        });
        let slots = weather["content"]["elements"]
            .as_array()
            .expect("four slots");
        assert_eq!(slots[0]["elements"][4], text);
        assert_eq!(slots[2]["elements"][4]["type"], "or");
        assert_eq!(
            slots[3]["content"]["elements"][4], text,
            "an object as Python's text"
        );
    }

    #[test]
    fn null_takes_jsons_word_or_pythons() {
        let payload = formats::qwen3(CallSyntax::Tagged(Spelling::Json))
            .grammar(
                &[tool(
                    "t",
                    value!({"type": "object", "properties": {"nothing": {"type": "null"}}}),
                )],
                true,
                false,
            )
            .expect("a grammar")
            .payload();
        let slot = &payload["format"]["tags"][0]["content"]["elements"][0]["content"];
        assert_eq!(
            slot["elements"][4],
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "null"},
                {"type": "const_string", "value": "None"},
            ]})
        );
    }

    #[test]
    fn with_the_reasoning_open_the_prefix_excludes_the_four_tags_too() {
        let payload = formats::qwen3(CallSyntax::Tagged(Spelling::Json))
            .grammar(&weather_tools(), true, true)
            .expect("a grammar")
            .payload();
        assert_eq!(payload["format"]["type"], "sequence");
        let prefix = &payload["format"]["elements"][0];
        assert_eq!(prefix["end"], "</think>");
        assert_eq!(
            prefix["content"]["excludes"],
            value!([
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<function=",
                "<parameter=",
                "</parameter>",
                "</function>",
            ])
        );
        assert_eq!(payload["format"]["elements"][1]["type"], "triggered_tags");
    }
}
