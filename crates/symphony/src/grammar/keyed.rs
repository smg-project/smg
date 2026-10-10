//! Keyed arguments written from the table's own tags, for the spellings the engines' `glm_xml`
//! style does not write: Ling's, with the template's newline after the call's name, after each
//! `</arg_key>` and before `</tool_call>`, and Hy4's, every tag suffixed `:opensource` and the
//! calls in a block. GLM's and IQuest's compact spelling keeps the `glm_xml` style, the shape the
//! gateway writes for it today.
//!
//! A call is the call's opener and the tool's name, then one `<arg_key>` and `<arg_value>` pair per
//! property of the tool's schema, in the schema's order, each optional unless the schema requires
//! it, then the call's close, with the spelling's newline where the template writes one. The key
//! is the property's name; the value follows the property's type as the templates write it (a
//! string as it is, everything else as JSON, which [`shape`] pins); a property without a type, or
//! a schema without properties, takes any text up to the value's close. When the calls sit in a
//! block, the block's ways in frame one or more calls, as Kimi K3's do.

use openai_protocol::common::Tool;
use serde_json::Value;

use super::{
    block_of,
    schema::{self, shape, Definitions},
    CallMarkers, Grammar, Tag, WayIn,
};
use crate::tagged::keyed::Tags;

/// The calls for `tools`: in a block, the block's ways in around one or more calls; without one,
/// one tag per tool, triggered by the call's opener.
pub(super) fn calls(
    markers: &CallMarkers<'_>,
    ways_in: &[WayIn<'_>],
    tags: &Tags,
    tools: &[&Tool],
    at_least_one: bool,
) -> Grammar {
    let ends = Ends {
        value_close: tags.value_close,
        call_exits: &markers.call_exits,
    };
    let calls = tools.iter().map(|tool| {
        call(
            markers,
            tags,
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

/// What ends a value: its own closing tag, and the terminals the table has a row for out of the
/// arguments state, which the engine takes wherever they stand inside a call: the call's close,
/// its opener where a new call ends the one open (Ling), the block's close where it ends a call
/// (DSML's does, Hy4's does not).
struct Ends<'a> {
    value_close: &'a str,
    call_exits: &'a [&'a str],
}

impl Ends<'_> {
    /// A value written as it is: any text up to one of the ends.
    fn text(&self) -> Grammar {
        let mut excludes = vec![self.value_close.to_string()];
        excludes.extend(self.call_exits.iter().map(|exit| exit.to_string()));
        Grammar::AnyText { excludes }
    }
}

/// One call as its three parts: the opener with the name and the spelling's newline, the
/// arguments, and the newline with the close.
fn call(
    markers: &CallMarkers<'_>,
    tags: &Tags,
    ends: &Ends<'_>,
    name: &str,
    parameters: &Value,
) -> (String, Grammar, String) {
    (
        format!("{}{name}{}", markers.call_open, tags.between),
        arguments(tags, ends, parameters),
        format!("{}{}", tags.between, markers.call_close),
    )
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

/// One pair: the key between its tags, the spelling's newline, the value between its tags.
fn argument(tags: &Tags, key: Grammar, value: Grammar) -> Grammar {
    Grammar::Sequence(vec![
        Grammar::ConstString(tags.key_open.to_string()),
        key,
        Grammar::ConstString(format!(
            "{}{}{}",
            tags.key_close, tags.between, tags.value_open
        )),
        value,
        Grammar::ConstString(tags.value_close.to_string()),
    ])
}

/// A pair with any key and any value.
fn any_argument(tags: &Tags, ends: &Ends<'_>) -> Grammar {
    let key = Grammar::AnyText {
        excludes: vec![tags.key_close.to_string()],
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
    fn ling_writes_each_call_with_the_templates_newlines() {
        // Ling's template: `<tool_call>` and the name, a newline, then per argument `<arg_key>`,
        // the key, `</arg_key>`, a newline, `<arg_value>`, the value, `</arg_value>`, and after the
        // last a newline and `</tool_call>`; a string as it is, the rest as JSON.
        let payload = formats::ling()
            .grammar(&weather_tools(), true, false)
            .expect("a grammar")
            .payload();
        // What ends a value is what the table's rows leave the call on: its close, and a new call's
        // opener, which ends the one open.
        let text = value!({
            "type": "any_text",
            "excludes": ["</arg_value>", "</tool_call>", "<tool_call>"],
        });
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
                        "begin": "<tool_call>get_weather\n",
                        "content": {"type": "sequence", "elements": [
                            {"type": "sequence", "elements": [
                                {"type": "const_string", "value": "<arg_key>"},
                                {"type": "const_string", "value": "city"},
                                {"type": "const_string", "value": "</arg_key>\n<arg_value>"},
                                text,
                                {"type": "const_string", "value": "</arg_value>"},
                            ]},
                            {"type": "sequence", "elements": [
                                {"type": "const_string", "value": "<arg_key>"},
                                {"type": "const_string", "value": "days"},
                                {"type": "const_string", "value": "</arg_key>\n<arg_value>"},
                                days,
                                {"type": "const_string", "value": "</arg_value>"},
                            ]},
                            {"type": "optional", "content": {"type": "sequence", "elements": [
                                {"type": "const_string", "value": "<arg_key>"},
                                {"type": "const_string", "value": "units"},
                                {"type": "const_string", "value": "</arg_key>\n<arg_value>"},
                                {"type": "or", "elements": [
                                    {"type": "const_string", "value": "c"},
                                    {"type": "const_string", "value": "f"},
                                ]},
                                {"type": "const_string", "value": "</arg_value>"},
                            ]}},
                        ]},
                        "end": "\n</tool_call>",
                    },
                    {
                        "type": "tag",
                        "begin": "<tool_call>ping\n",
                        "content": {"type": "star", "content": {"type": "sequence", "elements": [
                            {"type": "const_string", "value": "<arg_key>"},
                            {"type": "any_text", "excludes": ["</arg_key>"]},
                            {"type": "const_string", "value": "</arg_key>\n<arg_value>"},
                            text,
                            {"type": "const_string", "value": "</arg_value>"},
                        ]}},
                        "end": "\n</tool_call>",
                    },
                ],
                "at_least_one": true,
            }})
        );
    }

    #[test]
    fn a_ling_call_without_arguments_writes_both_newlines() {
        // The template writes the newline after the name and the one before `</tool_call>`
        // whether or not an argument comes between: every one of the 259 zero-argument calls in
        // ling-3.0-flash's recorded parse sets reads `<tool_call>NAME\n\n</tool_call>`.
        let payload = formats::ling()
            .grammar(&[tool("ping", value!({"type": "object"}))], true, false)
            .expect("a grammar")
            .payload();
        let tag = &payload["format"]["tags"][0];
        let written = format!(
            "{}{}",
            tag["begin"].as_str().expect("begin"),
            tag["end"].as_str().expect("end")
        );
        assert_eq!(written, "<tool_call>ping\n\n</tool_call>");
    }

    #[test]
    fn hy4_with_the_reasoning_open_takes_the_prefix_and_its_one_way_in() {
        // Hy4's prompt opens the thought, so this is the path it takes most: the prefix is free
        // text that can write none of the four call terminals nor the four suffixed tags, the
        // thought's own two allowed so the model may close it where the engine runs the grammar
        // from the first token, and nothing owed at its end; the block behind it has its one way
        // in, from content, with no close of its own before the opener.
        let payload = formats::hy4()
            .grammar(&weather_tools(), true, true)
            .expect("a grammar")
            .payload();
        assert_eq!(payload["format"]["type"], "sequence");
        let prefix = &payload["format"]["elements"][0];
        assert_eq!(prefix["type"], "any_text");
        assert!(
            prefix.get("end").is_none(),
            "nothing owed at the prefix's end: {prefix}"
        );
        assert_eq!(
            prefix["excludes"],
            value!([
                "<tool_calls:opensource>",
                "</tool_calls:opensource>",
                "<tool_call:opensource>",
                "</tool_call:opensource>",
                "<arg_key:opensource>",
                "</arg_key:opensource>",
                "<arg_value:opensource>",
                "</arg_value:opensource>",
            ])
        );
        let block = &payload["format"]["elements"][1];
        assert_eq!(block["triggers"], value!(["<tool_calls:opensource>"]));
        assert_eq!(block["tags"][0]["begin"], "<tool_calls:opensource>");
        assert_eq!(block["at_least_one"], true);
    }

    #[test]
    fn ling_with_the_reasoning_open_takes_the_prefix_as_glm_does() {
        let payload = formats::ling()
            .grammar(&weather_tools(), false, true)
            .expect("a grammar")
            .payload();
        assert_eq!(payload["format"]["type"], "sequence");
        assert_eq!(payload["format"]["elements"][0]["type"], "any_text");
        assert_eq!(
            payload["format"]["elements"][0]["excludes"],
            value!([
                "<tool_call>",
                "</tool_call>",
                "<arg_key>",
                "</arg_key>",
                "<arg_value>",
                "</arg_value>",
            ])
        );
        assert_eq!(payload["format"]["elements"][1]["at_least_one"], false);
    }

    #[test]
    fn hy4_writes_its_block_and_its_suffixed_tags_with_nothing_between() {
        // Hy4's template writes the block, each call and each pair back to back, every tag
        // suffixed; the block is entered from content alone, so it has one way in.
        let payload = formats::hy4()
            .grammar(&weather_tools(), true, false)
            .expect("a grammar")
            .payload();
        let format = &payload["format"];
        assert_eq!(format["type"], "triggered_tags");
        assert_eq!(format["triggers"], value!(["<tool_calls:opensource>"]));
        let tags = format["tags"].as_array().expect("one way in");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0]["begin"], "<tool_calls:opensource>");
        assert_eq!(tags[0]["end"], "</tool_calls:opensource>");
        assert_eq!(tags[0]["content"]["type"], "plus");
        let calls = tags[0]["content"]["content"]["elements"]
            .as_array()
            .expect("one call per tool");
        assert_eq!(calls.len(), 2);
        let weather = calls[0]["elements"].as_array().expect("three parts");
        assert_eq!(weather[0]["value"], "<tool_call:opensource>get_weather");
        assert_eq!(weather[2]["value"], "</tool_call:opensource>");
        let city = &weather[1]["elements"][0]["elements"];
        assert_eq!(city[0]["value"], "<arg_key:opensource>");
        assert_eq!(city[1]["value"], "city");
        assert_eq!(
            city[2]["value"],
            "</arg_key:opensource><arg_value:opensource>"
        );
        // Hy4's table leaves a call on its close alone (no row from the call on the next call's
        // opener or the block's close), so nothing else ends a value.
        assert_eq!(
            city[3],
            value!({"type": "any_text", "excludes": [
                "</arg_value:opensource>",
                "</tool_call:opensource>",
            ]})
        );
        assert_eq!(city[4]["value"], "</arg_value:opensource>");
        assert_eq!(calls[1]["elements"][1]["type"], "star");
    }

    #[test]
    fn glm_and_iquest_keep_the_engines_compact_style() {
        let tools = weather_tools();
        for format in [formats::glm(), formats::iquest()] {
            let payload = format
                .grammar(&tools, true, false)
                .expect("a grammar")
                .payload();
            assert_eq!(
                payload["format"]["tags"][0]["content"]["style"],
                "glm_xml",
                "{}",
                format.name()
            );
        }
    }
}
