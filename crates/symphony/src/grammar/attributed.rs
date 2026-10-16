//! Parameters named by an attribute, written from the table's own tags in the family's spelling
//! ([`Tags`]): MiniMax M2's, MiniCPM5's, dots3's and DSML's. A call is the invoke's opener with the
//! tool's name and `">`, then one parameter tag per property of the tool's schema, in the schema's
//! order, each optional unless the schema requires it, the value as the templates write it (a
//! string as it is, everything else as JSON, which [`shape`] pins), its closing tag, then the
//! invoke's close; a schema without properties takes any pairs at all. Where the family writes each
//! tag on a line of its own (M2, dots3) a newline follows the invoke's opener, each parameter's
//! close and the invoke's close; where a value stands on its own line (dots3) a newline frames it
//! on both sides. The calls sit in the block the table has: M2's one block around the turn's calls,
//! opened after a newline and on a line of its own and closed right after the last invoke's
//! newline; dots3's block around each call, the block's opener triggering the tag so the newline
//! the template writes before it is free text. A table without a block (MiniCPM5) writes each call
//! as a tag the invoke's opener triggers, the newlines between calls free text.
//!
//! Where the tag's `string` attribute types the value (DSML, [`Typing::StringAttribute`]) the
//! schema is not consulted: any number of parameters with any name, each `string="true"` with
//! any text or `string="false"` with any JSON value, the shape the gateway hand-wrote for
//! DeepSeek V4.1; a schema-driven spelling is a change of its own.

use openai_protocol::common::Tool;
use serde_json::Value;

use super::{calls, CallMarkers, Count, Grammar, Tag};
use crate::tagged::attributed::{Tags, Typing};

/// The calls for `tools`, in the family's spelling: in the table's one block, in a block each, or
/// with no block at all, as many as `count` lets the model write. `None` for a syntax that
/// needs a block the table has not.
pub(super) fn calls(
    markers: &CallMarkers<'_>,
    tags: &Tags,
    tools: &[&Tool],
    count: Count<'_>,
) -> Option<Grammar> {
    if tags.typing == Typing::StringAttribute {
        return string_attribute_calls(markers, tags, tools, count);
    }
    let ends = Ends {
        parameter_close: tags.parameter_close,
        call_exits: &markers.call_exits,
    };
    let line = if tags.tags_on_their_own_lines {
        "\n"
    } else {
        ""
    };
    let invoke = |tool: &Tool, before: &str, after: &str| {
        Tag::new(
            format!(
                "{before}{}{}\">{line}",
                markers.call_open, tool.function.name
            ),
            arguments(tags, &ends, &tool.function.parameters),
            format!("{}{line}{after}", markers.call_close),
        )
    };
    let Some(block) = markers.block.as_ref() else {
        // No block: each call is a tag its opener triggers, the text between calls free.
        return Some(count.triggered(
            vec![markers.call_open.to_string()],
            tools.iter().map(|tool| invoke(tool, "", "")).collect(),
        ));
    };
    if tags.block_per_call {
        // A block around each call: the block's opener triggers the tag, so the newline the
        // template writes before it is free text.
        let before = format!("{}\n", block.open);
        return Some(
            count.triggered(
                vec![block.open.to_string()],
                tools
                    .iter()
                    .map(|tool| invoke(tool, &before, block.close))
                    .collect(),
            ),
        );
    }
    Some(Grammar::Sequence(vec![
        Grammar::ConstString(format!("{}{}\n", tags.before_block, block.open)),
        count.separated(tools.iter().map(|tool| invoke(tool, "", "")).collect()),
        Grammar::ConstString(count.end(block.close)),
    ]))
}

/// The calls where the tag's `string` attribute types the value: the block after what the
/// template writes before it, inside it one invoke per tool, inside an invoke any number of
/// parameter tags, each `string="true"` with any text up to a closing tag or `string="false"` with
/// any JSON value, every tag closed with the template's newline; as many calls as `count` lets the
/// model write.
fn string_attribute_calls(
    markers: &CallMarkers<'_>,
    tags: &Tags,
    tools: &[&Tool],
    count: Count<'_>,
) -> Option<Grammar> {
    let block = markers.block.as_ref()?;
    let parameter = Grammar::Tag(Tag::new(
        tags.parameter_open,
        Grammar::Sequence(vec![
            Grammar::Regex("[^\"]+".to_string()),
            Grammar::ConstString("\" string=\"".to_string()),
            Grammar::Or(vec![
                Grammar::Sequence(vec![
                    Grammar::ConstString("true\">".to_string()),
                    Grammar::AnyText {
                        excludes: vec![
                            tags.parameter_close.to_string(),
                            markers.call_close.to_string(),
                            block.close.to_string(),
                        ],
                    },
                ]),
                Grammar::Sequence(vec![
                    Grammar::ConstString("false\">".to_string()),
                    Grammar::JsonSchema {
                        schema: Value::Bool(true),
                        style: None,
                        excludes: Vec::new(),
                    },
                ]),
            ]),
        ]),
        format!("{}\n", tags.parameter_close),
    ));
    let invokes = tools
        .iter()
        .map(|tool| {
            Tag::new(
                format!("{}{}\">\n", markers.call_open, tool.function.name),
                Grammar::Star(Box::new(parameter.clone())),
                format!("{}\n", markers.call_close),
            )
        })
        .collect();
    Some(Grammar::Sequence(vec![
        Grammar::ConstString(format!("{}{}\n", tags.before_block, block.open)),
        count.separated(invokes),
        Grammar::ConstString(count.end(block.close)),
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
    calls::arguments(
        parameters,
        || ends.text(),
        |key, value| argument(tags, key, value),
        || any_argument(tags, ends),
    )
}

/// One pair: the opening tag up to the name's quote, the key, `">`, the value, the closing tag;
/// the value framed by newlines where it stands on its own line, and a newline after the closing
/// tag where each tag has its own line.
fn argument(tags: &Tags, key: Grammar, value: Grammar) -> Grammar {
    let frame = if tags.value_on_its_own_line { "\n" } else { "" };
    let line = if tags.tags_on_their_own_lines {
        "\n"
    } else {
        ""
    };
    Grammar::Sequence(vec![
        Grammar::ConstString(tags.parameter_open.to_string()),
        key,
        Grammar::ConstString(format!("\">{frame}")),
        value,
        Grammar::ConstString(format!("{frame}{}{line}", tags.parameter_close)),
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
    fn minicpm5_writes_each_call_as_a_tag_its_opener_triggers_with_nothing_between_the_tags() {
        // MiniCPM5's template: `<function name="NAME">`, then per argument the parameter tag, the
        // value and its closing tag with nothing between them, then `</function>`; one call after
        // another on lines of their own, so the newline between calls is free text.
        let tools = [tool(
            "play",
            value!({"type": "object", "properties": {
                "artist": {"type": "string"},
                "n": {"type": "integer"},
            }, "required": ["artist"]}),
        )];
        let payload = formats::minicpm5()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let text = value!({
            "type": "any_text",
            "excludes": ["</param>", "</function>"],
        });
        assert_eq!(
            payload,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<function name=\""],
                "tags": [{
                    "type": "tag",
                    "begin": "<function name=\"play\">",
                    "content": {"type": "sequence", "elements": [
                        {"type": "sequence", "elements": [
                            {"type": "const_string", "value": "<param name=\""},
                            {"type": "const_string", "value": "artist"},
                            {"type": "const_string", "value": "\">"},
                            text,
                            {"type": "const_string", "value": "</param>"},
                        ]},
                        {"type": "optional", "content": {"type": "sequence", "elements": [
                            {"type": "const_string", "value": "<param name=\""},
                            {"type": "const_string", "value": "n"},
                            {"type": "const_string", "value": "\">"},
                            {"type": "json_schema", "json_schema": {"type": "integer"}},
                            {"type": "const_string", "value": "</param>"},
                        ]}},
                    ]},
                    "end": "</function>",
                }],
                "at_least_one": true,
            }})
        );
    }

    #[test]
    fn dots3_writes_a_block_around_each_call_and_frames_each_value_with_newlines() {
        // dots3's template: a newline (free text here), `<dots_function_call>`, a newline,
        // `<invoke name="NAME">`, a newline, per argument the parameter tag, a newline, the
        // value, a newline, the closing tag and a newline, then `</invoke>`, a newline and the
        // block's close.
        let tools = [tool(
            "play",
            value!({"type": "object", "properties": {
                "artist": {"type": "string"},
            }, "required": ["artist"]}),
        )];
        let payload = formats::dots3()
            .grammar(&tools, false, false)
            .expect("a grammar")
            .payload();
        let text = value!({
            "type": "any_text",
            "excludes": ["</parameter>", "</invoke>", "</dots_function_call>"],
        });
        assert_eq!(
            payload,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<dots_function_call>"],
                "tags": [{
                    "type": "tag",
                    "begin": "<dots_function_call>\n<invoke name=\"play\">\n",
                    "content": {"type": "sequence", "elements": [
                        {"type": "sequence", "elements": [
                            {"type": "const_string", "value": "<parameter name=\""},
                            {"type": "const_string", "value": "artist"},
                            {"type": "const_string", "value": "\">\n"},
                            text,
                            {"type": "const_string", "value": "\n</parameter>\n"},
                        ]},
                    ]},
                    "end": "</invoke>\n</dots_function_call>",
                }],
                "at_least_one": false,
            }})
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
