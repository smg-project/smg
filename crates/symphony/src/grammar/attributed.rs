//! Parameters named by an attribute, written from the table's own tags: MiniMax M2's. A call is
//! the invoke's opener with the tool's name, `">` and the template's newline, then one
//! `<parameter name="KEY">` per property of the tool's schema, in the schema's order, each optional
//! unless the schema requires it, the value as the templates write it (a string as it is,
//! everything else as JSON, which [`shape`] pins), `</parameter>` and a newline, then the invoke's
//! close and a newline; a schema without properties takes any pairs at all. The calls sit in the
//! block the table has, which the template opens after a newline and on a line of its own and
//! closes right after the last invoke's newline.

use openai_protocol::common::Tool;
use serde_json::Value;

use super::{
    schema::{self, shape, Definitions},
    CallMarkers, Grammar, Tag,
};
use crate::tagged::attributed::Tags;

/// The calls for `tools`, in the table's block; `None` for a table without one.
pub(super) fn calls(
    markers: &CallMarkers<'_>,
    tags: &Tags,
    tools: &[&Tool],
    at_least_one: bool,
) -> Option<Grammar> {
    let block = markers.block.as_ref()?;
    let ends = Ends {
        parameter_close: tags.parameter_close,
        call_exits: &markers.call_exits,
    };
    let invokes = tools
        .iter()
        .map(|tool| {
            Tag::new(
                format!("{}{}\">\n", markers.call_open, tool.function.name),
                arguments(tags, &ends, &tool.function.parameters),
                format!("{}\n", markers.call_close),
            )
        })
        .collect();
    Some(Grammar::Sequence(vec![
        Grammar::ConstString(format!("\n{}\n", block.open)),
        Grammar::TagsWithSeparator {
            tags: invokes,
            separator: String::new(),
            at_least_one,
        },
        Grammar::ConstString(block.close.to_string()),
    ]))
}

/// What ends a value: its own closing tag, and the terminals the table has a row for out of the
/// arguments state, which the engine takes wherever they stand inside a call.
struct Ends<'a> {
    parameter_close: &'a str,
    call_exits: &'a [&'a str],
}

impl Ends<'_> {
    /// A value written as it is: any text up to one of the ends.
    fn text(&self) -> Grammar {
        let mut excludes = vec![self.parameter_close.to_string()];
        excludes.extend(self.call_exits.iter().map(|exit| exit.to_string()));
        Grammar::AnyText { excludes }
    }
}

/// The arguments the tool's schema asks for: one pair per property, in the schema's order, each
/// optional unless required; any pairs at all for a schema without properties.
fn arguments(tags: &Tags, ends: &Ends<'_>, parameters: &Value) -> Grammar {
    let pair = |key: &str, property: &Value, definitions: Definitions<'_>| {
        let value = shape(property, definitions, || ends.text())
            .map_or_else(|| ends.text(), |(_, value)| value);
        argument(tags, Grammar::ConstString(key.to_string()), value)
    };
    schema::arguments(parameters, pair, || {
        Grammar::Star(Box::new(any_argument(tags, ends)))
    })
}

/// One pair: the opening tag up to the name's quote, the key, `">`, the value, the closing tag and
/// the template's newline.
fn argument(tags: &Tags, key: Grammar, value: Grammar) -> Grammar {
    Grammar::Sequence(vec![
        Grammar::ConstString(tags.parameter_open.to_string()),
        key,
        Grammar::ConstString("\">".to_string()),
        value,
        Grammar::ConstString(format!("{}\n", tags.parameter_close)),
    ])
}

/// A pair with any key and any value.
fn any_argument(tags: &Tags, ends: &Ends<'_>) -> Grammar {
    let key = Grammar::AnyText {
        excludes: vec!["\"".to_string()],
    };
    argument(tags, key, ends.text())
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;
    use serde_json::json as value;

    use super::*;
    use crate::formats;

    fn tool(name: &str, parameters: Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
                strict: None,
                extra: Default::default(),
            },
        }
    }

    #[test]
    fn minimax_m2_writes_its_block_each_invoke_and_each_pair_on_its_own_line() {
        // MiniMax M2's template: a newline and `<minimax:tool_call>` on its own line, then per call
        // `<invoke name="NAME">` and a newline, per argument the parameter tag, the value and the
        // closing tag with a newline, `</invoke>` and a newline, then `</minimax:tool_call>`; a
        // string as it is, the rest as JSON.
        let tools = [
            tool(
                "get_weather",
                value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1},
                }, "required": ["city"]}),
            ),
            tool("ping", value!({"type": "object"})),
        ];
        let payload = formats::minimax_m2()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        // What ends a value is what the table's rows leave the invoke on: its close, a new
        // invoke's opener, and the block's close.
        let text = value!({
            "type": "any_text",
            "excludes": ["</parameter>", "</invoke>", "<invoke name=\"", "</minimax:tool_call>"],
        });
        let days =
            value!({"type": "json_schema", "json_schema": {"type": "integer", "minimum": 1}});
        assert_eq!(
            payload,
            value!({"format": {"type": "sequence", "elements": [
                {"type": "const_string", "value": "\n<minimax:tool_call>\n"},
                {
                    "type": "tags_with_separator",
                    "tags": [
                        {
                            "type": "tag",
                            "begin": "<invoke name=\"get_weather\">\n",
                            "content": {"type": "sequence", "elements": [
                                {"type": "sequence", "elements": [
                                    {"type": "const_string", "value": "<parameter name=\""},
                                    {"type": "const_string", "value": "city"},
                                    {"type": "const_string", "value": "\">"},
                                    text,
                                    {"type": "const_string", "value": "</parameter>\n"},
                                ]},
                                {"type": "optional", "content": {"type": "sequence", "elements": [
                                    {"type": "const_string", "value": "<parameter name=\""},
                                    {"type": "const_string", "value": "days"},
                                    {"type": "const_string", "value": "\">"},
                                    days,
                                    {"type": "const_string", "value": "</parameter>\n"},
                                ]}},
                            ]},
                            "end": "</invoke>\n",
                        },
                        {
                            "type": "tag",
                            "begin": "<invoke name=\"ping\">\n",
                            "content": {"type": "star", "content": {
                                "type": "sequence",
                                "elements": [
                                    {"type": "const_string", "value": "<parameter name=\""},
                                    {"type": "any_text", "excludes": ["\""]},
                                    {"type": "const_string", "value": "\">"},
                                    text,
                                    {"type": "const_string", "value": "</parameter>\n"},
                                ],
                            }},
                            "end": "</invoke>\n",
                        },
                    ],
                    "separator": "",
                    "at_least_one": true,
                },
                {"type": "const_string", "value": "</minimax:tool_call>"},
            ]}})
        );
    }

    #[test]
    fn with_the_thought_open_the_block_follows_a_prefix_that_excludes_its_tags() {
        let payload = formats::minimax_m2()
            .grammar(&[tool("ping", value!({"type": "object"}))], true, true)
            .expect("a grammar")
            .payload();
        let parts = payload["format"]["elements"]
            .as_array()
            .expect("the prefix, then the block");
        let excludes = parts[0]["excludes"].as_array().expect("the excluded tags");
        for tag in [
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            "<invoke name=\"",
            "</invoke>",
            "<parameter name=\"",
            "</parameter>",
        ] {
            assert!(
                excludes.contains(&value!(tag)),
                "{tag} is excluded from the prefix"
            );
        }
        assert!(
            !excludes.contains(&value!("</think>")),
            "the thought's own close stays writable"
        );
        assert_eq!(parts[1]["elements"][0]["value"], "\n<minimax:tool_call>\n");
    }
}
