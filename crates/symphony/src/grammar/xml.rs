//! MiniMax M3's structural tag: the block of calls as the template writes it, from the table's
//! markers, its separator and the request's tools.
//!
//! The template writes its separator token before every tag. The block is `<tool_call>` and a
//! newline, one invoke or more, then the separator and `</tool_call>`; an invoke is the separator,
//! `<invoke name="NAME">`, its arguments, the separator, `</invoke>` and a newline. Each argument
//! is an element, the separator and `<key>`, its value, the separator and `</key>`, one per
//! property in the schema's order, present when the schema requires the property and otherwise
//! optional. `<tool_call>` triggers the tag, as the terminal that opens the block is the trigger
//! under every syntax: the text before it is free (the thought and its close, an answer, and the
//! separator the template writes before the block's opener), and free text cannot hold the
//! opener, since the opener is the trigger, so what the engine constrains and what the table
//! reads as the block's opening are the same bytes. The tag is the same whether the prompt left
//! the thought open or not.
//!
//! A value follows the property's type as the template writes it and the XML assembler reads it
//! ([`tagged::xml`](crate::tagged::xml)): a string as it is, any text that holds neither the
//! separator, nor the element's own closing tag, nor a marker that ends the call; a string `enum`
//! as one of its values; a number as the JSON its schema accepts; a boolean as one of the words the
//! assembler reads, `true`, `false`, `True` or `False`; an object with properties as nested
//! elements, one per property, as deep as the schema goes; an array with items as `<item>`
//! elements, each a value of the items' type; `null` as nothing at all, since the template leaves
//! out an argument whose value is None and writes an item that is None as an empty element. A
//! property that pins no type, an object without properties or an array without items takes any
//! text that holds neither its closing tag nor a marker that ends the call, separators and nested
//! tags included, which the assembler reads as it infers; a schema without properties takes the
//! same for the whole invoke.

use openai_protocol::common::Tool;
use serde_json::Value;

use super::{
    schema::{keyword, shape, Definitions},
    CallMarkers, Grammar, Tag,
};

/// The words the assembler reads as a boolean.
const BOOLEAN_WORDS: [&str; 4] = ["true", "false", "True", "False"];
/// A list's element tag.
const ITEM: &str = "item";
/// How deep the elements follow the schema; below that, a value is any text.
const DEPTH: usize = 16;

/// The block of calls for `tools`, written with `separator` before every tag, or `None` for a
/// table whose calls sit in no block.
pub(super) fn calls(
    markers: &CallMarkers<'_>,
    separator: &str,
    tools: &[&Tool],
    at_least_one: bool,
) -> Option<Grammar> {
    let block = markers.block.as_ref()?;
    let writer = Writer {
        separator,
        exits: &markers.call_exits,
    };
    let invokes = tools
        .iter()
        .map(|tool| writer.invoke(markers, &tool.function.name, &tool.function.parameters))
        .collect();
    Some(Grammar::TriggeredTags {
        triggers: vec![block.open.to_string()],
        tags: vec![Tag::new(
            format!("{}\n", block.open),
            Grammar::TagsWithSeparator {
                tags: invokes,
                separator: String::new(),
                at_least_one: true,
            },
            format!("{separator}{}", block.close),
        )],
        at_least_one,
    })
}

/// The spelling the table gives: the separator before every tag, and the markers that end a call
/// wherever they stand, which no value may hold.
struct Writer<'a> {
    separator: &'a str,
    exits: &'a [&'a str],
}

impl Writer<'_> {
    /// One invoke: the separator and the opener with the tool's name, the arguments, the
    /// separator, the call's close and a newline.
    fn invoke(&self, markers: &CallMarkers<'_>, name: &str, parameters: &Value) -> Tag {
        let definitions = Definitions::of(parameters);
        Tag::new(
            format!("{}{}{name}\">", self.separator, markers.call_open),
            self.members(parameters, definitions, markers.call_close, DEPTH),
            format!("{}{}\n", self.separator, markers.call_close),
        )
    }

    /// An object's elements: one per property in the schema's order, each optional unless the
    /// schema requires it, a property typed `null` left out; any text up to `close` for a schema
    /// without properties.
    fn members(
        &self,
        schema: &Value,
        definitions: Definitions<'_>,
        close: &str,
        depth: usize,
    ) -> Grammar {
        let target = definitions.target_of(schema);
        let properties = keyword(schema, target, "properties")
            .and_then(Value::as_object)
            .filter(|properties| !properties.is_empty());
        let Some(properties) = properties else {
            return self.any_text(close);
        };
        let required: Vec<&str> = keyword(schema, target, "required")
            .and_then(Value::as_array)
            .map(|names| names.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        Grammar::Sequence(
            properties
                .iter()
                .filter_map(|(key, schema)| {
                    let element = self.element(key, schema, definitions, depth)?;
                    Some(if required.contains(&key.as_str()) {
                        element
                    } else {
                        Grammar::Optional(Box::new(element))
                    })
                })
                .collect(),
        )
    }

    /// One element: the separator and `<key>`, the value, the separator and `</key>`; nothing for
    /// a property typed `null`, which the template leaves out.
    fn element(
        &self,
        key: &str,
        schema: &Value,
        definitions: Definitions<'_>,
        depth: usize,
    ) -> Option<Grammar> {
        let close = format!("</{key}>");
        let value = self.value(schema, definitions, &close, depth)?;
        Some(Grammar::Sequence(vec![
            Grammar::ConstString(format!("{}<{key}>", self.separator)),
            value,
            Grammar::ConstString(format!("{}{close}", self.separator)),
        ]))
    }

    /// The value between an element's tags, by the type the property's schema pins, or `None`
    /// for `null`, which is written as nothing.
    fn value(
        &self,
        schema: &Value,
        definitions: Definitions<'_>,
        close: &str,
        depth: usize,
    ) -> Option<Grammar> {
        let Some(depth) = depth.checked_sub(1) else {
            return Some(self.any_text(close));
        };
        let target = definitions.target_of(schema);
        match shape(schema, definitions, || self.text(close)) {
            Some(("null", _)) => None,
            Some(("boolean", _)) => Some(words(&BOOLEAN_WORDS)),
            Some(("object", _)) => Some(self.members(schema, definitions, close, depth)),
            Some(("array", _)) => Some(match keyword(schema, target, "items") {
                Some(items) => self.items(items, definitions, depth),
                None => self.any_text(close),
            }),
            Some((_, value)) => Some(value),
            None => Some(self.any_text(close)),
        }
    }

    /// A list's items: any number of `<item>` elements, each a value of the items' type, an item
    /// that is `null` an empty element.
    fn items(&self, items: &Value, definitions: Definitions<'_>, depth: usize) -> Grammar {
        let open = format!("{}<{ITEM}>", self.separator);
        let close = format!("{}</{ITEM}>", self.separator);
        let item = match self.value(items, definitions, &format!("</{ITEM}>"), depth) {
            Some(value) => Grammar::Sequence(vec![
                Grammar::ConstString(open),
                value,
                Grammar::ConstString(close),
            ]),
            None => Grammar::ConstString(format!("{open}{close}")),
        };
        Grammar::Star(Box::new(item))
    }

    /// A string written as it is: any text that holds neither the separator, nor `close`, nor a
    /// marker that ends the call.
    fn text(&self, close: &str) -> Grammar {
        self.excluding(&[self.separator, close])
    }

    /// A value the schema does not shape: any text that holds neither `close` nor a marker that
    /// ends the call, separators and nested tags included.
    fn any_text(&self, close: &str) -> Grammar {
        self.excluding(&[close])
    }

    /// Any text that holds none of `first` and none of the markers that end the call, each once.
    fn excluding(&self, first: &[&str]) -> Grammar {
        let mut excludes: Vec<String> = Vec::new();
        for text in first.iter().chain(self.exits.iter()) {
            if !excludes.iter().any(|known| known == text) {
                excludes.push(text.to_string());
            }
        }
        Grammar::AnyText { excludes }
    }
}

/// One of the words.
fn words(words: &[&str]) -> Grammar {
    Grammar::Or(
        words
            .iter()
            .map(|word| Grammar::ConstString(word.to_string()))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;
    use serde_json::json as value;

    use super::*;
    use crate::formats;

    const SEP: &str = "]<]minimax[>[";

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

    /// The markers that end a call wherever they stand, in the order of the table's rows.
    fn exits() -> Vec<&'static str> {
        vec!["</invoke>", "<invoke name=\"", "</tool_call>"]
    }

    /// An element: the separator and the key's tag, the value, the separator and the closing tag.
    fn element(key: &str, value: Value) -> Value {
        value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": format!("{SEP}<{key}>")},
            value,
            {"type": "const_string", "value": format!("{SEP}</{key}>")},
        ]})
    }

    /// A string's text inside the element `key`.
    fn text(key: &str) -> Value {
        let mut excludes = vec![SEP, ""];
        let close = format!("</{key}>");
        excludes[1] = &close;
        excludes.extend(exits());
        value!({"type": "any_text", "excludes": excludes})
    }

    /// The first invoke's elements of the block for `tools`.
    fn first_invoke(tools: &[Tool]) -> Value {
        let payload = formats::minimax_m3()
            .grammar(tools, true, false)
            .expect("a grammar")
            .payload();
        payload["format"]["tags"][0]["content"]["tags"][0].clone()
    }

    #[test]
    fn minimax_m3_writes_the_block_with_its_separator_before_every_tag() {
        // The recorded shape: `]<]minimax[>[<tool_call>\n`, each invoke
        // `]<]minimax[>[<invoke name="NAME">`, its elements each `]<]minimax[>[<key>VALUE` and
        // `]<]minimax[>[</key>`, then `]<]minimax[>[</invoke>\n`, and `]<]minimax[>[</tool_call>`.
        let payload = formats::minimax_m3()
            .grammar(&weather_tools(), true, false)
            .expect("a grammar")
            .payload();
        let weather = value!({
            "type": "tag",
            "begin": format!("{SEP}<invoke name=\"get_weather\">"),
            "content": {"type": "sequence", "elements": [
                element("city", text("city")),
                element(
                    "days",
                    value!({
                        "type": "json_schema",
                        "json_schema": {"type": "integer", "minimum": 1},
                    }),
                ),
                {"type": "optional", "content": element(
                    "units",
                    value!({"type": "or", "elements": [
                        {"type": "const_string", "value": "c"},
                        {"type": "const_string", "value": "f"},
                    ]}),
                )},
            ]},
            "end": format!("{SEP}</invoke>\n"),
        });
        let ping = value!({
            "type": "tag",
            "begin": format!("{SEP}<invoke name=\"ping\">"),
            "content": {"type": "any_text", "excludes": exits()},
            "end": format!("{SEP}</invoke>\n"),
        });
        assert_eq!(
            payload,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": [{
                    "type": "tag",
                    "begin": "<tool_call>\n",
                    "content": {
                        "type": "tags_with_separator",
                        "tags": [weather, ping],
                        "separator": "",
                        "at_least_one": true,
                    },
                    "end": format!("{SEP}</tool_call>"),
                }],
                "at_least_one": true,
            }})
        );
    }

    #[test]
    fn nested_objects_are_elements_lists_are_items_and_a_null_is_left_out() {
        let invoke = first_invoke(&[tool(
            "t",
            value!({
                "type": "object",
                "properties": {
                    "where": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}, "label": {"type": "string"}},
                        "required": ["x"],
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "flag": {"type": "boolean"},
                    "nothing": {"type": "null"},
                    "free": {"type": "object"},
                    "any": {},
                },
                "required": ["where", "tags", "flag", "nothing", "free", "any"],
            }),
        )]);
        let elements = invoke["content"]["elements"]
            .as_array()
            .expect("five elements");
        assert_eq!(elements.len(), 5, "the null is left out");
        assert_eq!(
            elements[0],
            element(
                "where",
                value!({"type": "sequence", "elements": [
                    element(
                        "x",
                        value!({"type": "json_schema", "json_schema": {"type": "number"}}),
                    ),
                    {"type": "optional", "content": element("label", text("label"))},
                ]}),
            )
        );
        assert_eq!(
            elements[1],
            element(
                "tags",
                value!({"type": "star", "content": element("item", text("item"))}),
            )
        );
        assert_eq!(
            elements[2],
            element(
                "flag",
                value!({"type": "or", "elements": [
                    {"type": "const_string", "value": "true"},
                    {"type": "const_string", "value": "false"},
                    {"type": "const_string", "value": "True"},
                    {"type": "const_string", "value": "False"},
                ]}),
            )
        );
        for (slot, key) in [(&elements[3], "free"), (&elements[4], "any")] {
            let mut excludes = vec![format!("</{key}>")];
            excludes.extend(exits().into_iter().map(str::to_string));
            assert_eq!(
                *slot,
                element(key, value!({"type": "any_text", "excludes": excludes})),
                "{key} takes any text"
            );
        }
    }

    #[test]
    fn a_reference_lends_its_targets_properties_and_a_list_of_nulls_holds_empty_items() {
        let invoke = first_invoke(&[tool(
            "t",
            value!({
                "type": "object",
                "$defs": {
                    "Point": {"type": "object", "properties": {"x": {"type": "integer"}}},
                    "Count": {"type": "integer"},
                },
                "properties": {
                    "at": {"$ref": "#/$defs/Point"},
                    "count": {"$ref": "#/$defs/Count"},
                    "holes": {"type": "array", "items": {"type": "null"}},
                },
                "required": ["at", "count", "holes"],
            }),
        )]);
        let elements = &invoke["content"]["elements"];
        assert_eq!(
            elements[0],
            element(
                "at",
                value!({"type": "sequence", "elements": [
                    {"type": "optional", "content": element(
                        "x",
                        value!({"type": "json_schema", "json_schema": {"type": "integer"}}),
                    )},
                ]}),
            )
        );
        assert_eq!(
            elements[1]["elements"][1],
            value!({"type": "json_schema", "json_schema": {
                "$ref": "#/$defs/Count",
                "$defs": {
                    "Point": {"type": "object", "properties": {"x": {"type": "integer"}}},
                    "Count": {"type": "integer"},
                },
            }})
        );
        assert_eq!(
            elements[2],
            element(
                "holes",
                value!({"type": "star", "content": {
                    "type": "const_string",
                    "value": format!("{SEP}<item>{SEP}</item>"),
                }}),
            )
        );
    }

    #[test]
    fn the_thought_open_gives_the_same_tag_and_without_a_required_call_the_block_is_optional() {
        let tools = weather_tools();
        let required = formats::minimax_m3()
            .grammar(&tools, true, false)
            .expect("a grammar");
        assert_eq!(
            formats::minimax_m3().grammar(&tools, true, true),
            Some(required.clone())
        );
        let either = formats::minimax_m3()
            .grammar(&tools, false, false)
            .expect("a grammar")
            .to_json();
        assert_eq!(either["at_least_one"], false);
        assert_eq!(
            either["tags"][0]["content"]["at_least_one"], true,
            "a block holds at least one invoke"
        );
        let mut same = required.to_json();
        same["at_least_one"] = Value::Bool(false);
        assert_eq!(either, same);
    }
}
