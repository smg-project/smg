//! The tagged syntax written from the table: `<function=NAME>`, then `<parameter=KEY>` around
//! each value, `</function>`, between the call markers, with what the family's template writes
//! around a value ([`Spelling`]): Qwen 3.5 and Qwen3-Coder put the value on a line of its own,
//! Seed-OSS writes it between the tags directly. The engines' `qwen_xml_parameter` style writes
//! Qwen's spelling of the parameters and nothing else, so both are written here from the tags.
//!
//! A call is the call's opener, a newline, `<function=NAME>` and a newline; then one parameter
//! per property of the tool's schema, in the schema's order, each optional unless the schema
//! requires it: `<parameter=KEY>`, the value with the spelling's newlines around it,
//! `</parameter>` and a newline; then `</function>`, a newline and the call's close. The value
//! follows the property's type as the templates write it (a string as it is, everything else as
//! JSON, which [`shape`] pins); a property without a type, or a schema without properties, takes
//! any text up to the parameter's close. When the calls sit in a block, the block's ways in frame
//! one or more calls, as Kimi K3's do.

use openai_protocol::common::Tool;
use serde_json::Value;

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
    let ends = Ends {
        call_close: markers.call_close,
        block_close: markers.block.as_ref().map(|block| block.close),
    };
    let calls = tools.iter().map(|tool| {
        call(
            markers,
            spelling,
            &ends,
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

/// What ends a value: the parameter's close, and the markers that end the call or the block,
/// which the engine takes wherever they stand.
struct Ends<'a> {
    call_close: &'a str,
    block_close: Option<&'a str>,
}

impl Ends<'_> {
    /// A value written as it is: any text up to one of the ends.
    fn text(&self) -> Grammar {
        let mut excludes = vec![PARAMETER_CLOSE.to_string(), self.call_close.to_string()];
        excludes.extend(self.block_close.map(str::to_string));
        Grammar::AnyText { excludes }
    }
}

/// One call as its three parts: the opener, the function tag and its newlines; the parameters;
/// the function's close with the call's.
fn call(
    markers: &CallMarkers<'_>,
    spelling: Spelling,
    ends: &Ends<'_>,
    name: &str,
    parameters: &Value,
) -> (String, Grammar, String) {
    (
        format!("{}\n{FUNCTION_OPEN}{name}>\n", markers.call_open),
        arguments(spelling, ends, parameters),
        format!("{FUNCTION_CLOSE}\n{}", markers.call_close),
    )
}

/// The parameters the tool's schema asks for: one per property, in the schema's order, each
/// optional unless required; any parameters at all for a schema without properties.
fn arguments(spelling: Spelling, ends: &Ends<'_>, parameters: &Value) -> Grammar {
    let parameter = |key: &str, property: &Value, definitions: Definitions<'_>| {
        let value = shape(property, definitions, || ends.text())
            .map_or_else(|| ends.text(), |(_, value)| value);
        argument(spelling, Grammar::ConstString(key.to_string()), value)
    };
    schema::arguments(parameters, parameter, || {
        Grammar::Star(Box::new(any_argument(spelling, ends)))
    })
}

/// One parameter: the opening tag with the key, the value with the spelling's text around it, the
/// close and a newline.
fn argument(spelling: Spelling, key: Grammar, value: Grammar) -> Grammar {
    let around = spelling.around_value;
    Grammar::Sequence(vec![
        Grammar::ConstString(PARAMETER_OPEN.to_string()),
        key,
        Grammar::ConstString(format!(">{around}")),
        value,
        Grammar::ConstString(format!("{around}{PARAMETER_CLOSE}\n")),
    ])
}

/// A parameter with any key and any value.
fn any_argument(spelling: Spelling, ends: &Ends<'_>) -> Grammar {
    let key = Grammar::AnyText {
        excludes: vec![">".to_string()],
    };
    argument(spelling, key, ends.text())
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
                        "units": {"type": "string", "enum": ["c", "f"]},
                    },
                    "required": ["city", "days"],
                }),
            ),
            tool("ping", value!({"type": "object"})),
        ]
    }

    #[test]
    fn qwen_3_5_puts_each_value_on_a_line_of_its_own() {
        // Qwen 3.5's template: `<tool_call>`, a newline, `<function=` and the name, `>` and a
        // newline; per argument `<parameter=` and the key, `>` and a newline, the value, a newline
        // and `</parameter>` and a newline; then `</function>`, a newline and `</tool_call>`. A
        // string as it is, a mapping or a list as JSON.
        let payload = formats::qwen3(CallSyntax::Tagged(Spelling::OWN_LINE))
            .grammar(&weather_tools(), true, false)
            .expect("a grammar")
            .payload();
        let text = value!({"type": "any_text", "excludes": ["</parameter>", "</tool_call>"]});
        let days =
            value!({"type": "json_schema", "json_schema": {"type": "integer", "minimum": 1}});
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
                            {"type": "sequence", "elements": [
                                {"type": "const_string", "value": "<parameter="},
                                {"type": "const_string", "value": "city"},
                                {"type": "const_string", "value": ">\n"},
                                text,
                                {"type": "const_string", "value": "\n</parameter>\n"},
                            ]},
                            {"type": "sequence", "elements": [
                                {"type": "const_string", "value": "<parameter="},
                                {"type": "const_string", "value": "days"},
                                {"type": "const_string", "value": ">\n"},
                                days,
                                {"type": "const_string", "value": "\n</parameter>\n"},
                            ]},
                            {"type": "optional", "content": {"type": "sequence", "elements": [
                                {"type": "const_string", "value": "<parameter="},
                                {"type": "const_string", "value": "units"},
                                {"type": "const_string", "value": ">\n"},
                                {"type": "or", "elements": [
                                    {"type": "const_string", "value": "c"},
                                    {"type": "const_string", "value": "f"},
                                ]},
                                {"type": "const_string", "value": "\n</parameter>\n"},
                            ]}},
                        ]},
                        "end": "</function>\n</tool_call>",
                    },
                    {
                        "type": "tag",
                        "begin": "<tool_call>\n<function=ping>\n",
                        "content": {"type": "star", "content": {"type": "sequence", "elements": [
                            {"type": "const_string", "value": "<parameter="},
                            {"type": "any_text", "excludes": [">"]},
                            {"type": "const_string", "value": ">\n"},
                            text,
                            {"type": "const_string", "value": "\n</parameter>\n"},
                        ]}},
                        "end": "</function>\n</tool_call>",
                    },
                ],
                "at_least_one": true,
            }})
        );
    }

    #[test]
    fn seed_oss_writes_the_value_between_its_tags_under_its_own_markers() {
        // Seed-OSS's template: `<seed:tool_call>`, a newline, the function tag and a newline; per
        // argument `<parameter=` and the key, `>`, the value, `</parameter>` and a newline; then
        // `</function>`, a newline and `</seed:tool_call>`.
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
        let city = &weather["content"]["elements"][0]["elements"];
        assert_eq!(city[0]["value"], "<parameter=");
        assert_eq!(city[1]["value"], "city");
        assert_eq!(city[2]["value"], ">");
        assert_eq!(
            city[3],
            value!({"type": "any_text", "excludes": ["</parameter>", "</seed:tool_call>"]})
        );
        assert_eq!(city[4]["value"], "</parameter>\n");
    }

    #[test]
    fn with_the_reasoning_open_the_prefix_excludes_the_four_tags_too() {
        let payload = formats::qwen3(CallSyntax::Tagged(Spelling::OWN_LINE))
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
