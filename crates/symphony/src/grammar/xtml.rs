//! Kimi K3's structural tag: the block of calls as the gateway's Kimi K3 parser writes it by hand,
//! here from the table's markers and the request's tools.
//!
//! The table gives the block its close and its ways in (`Format::ways_into`): Kimi K3's prompt
//! leaves the thought open in thinking mode and the answer otherwise, and a block the model could
//! not reach from there would force the calls inside a region the model can never close. So the
//! tag has one way in per state the block is entered from, each a path the table has: from the
//! thought or the answer, its close and then the row into the block from where the close returns;
//! from the turn itself, its own row. Each way is closed by the block's close, and its first text
//! triggers the tag. Inside, one or more calls, each `<|open|>call tool="NAME" index="N"<|sep|>`,
//! its arguments and the call's close, with the name as the request gives it (`&` and `"` escaped
//! as the template does).
//!
//! The arguments follow the tool's schema: one argument tag per property, in the schema's order,
//! each optional unless the schema requires it, its `type` attribute and value pinned by the
//! property's type. A string is written as it is, so its value is any text up to a marker; a
//! string `enum` is one of its values; a number, an object or an array is JSON the property's
//! schema accepts, with the tool's definitions attached so a `$ref` inside it still resolves; a
//! boolean is JSON `true` or `false`; `null` is the word. A property the schema leaves untyped, or
//! one whose `$ref` points at nothing, keeps its key but takes any type and any value; a schema
//! without properties takes any number of such arguments. The engine then accepts exactly what
//! the XTML assembler reads ([`tagged::xtml`](crate::tagged::xtml)).

use openai_protocol::common::Tool;
use serde_json::{json, Value};

use super::{CallMarkers, Grammar, Tag, WayIn};

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
/// The call's count of the turn's calls, from one.
const INDEX: &str = "[1-9][0-9]*";

/// The block of calls for `tools`, entered by `ways_in`, or `None` for a table whose calls sit
/// in no block.
pub(super) fn calls(
    markers: &CallMarkers<'_>,
    ways_in: &[WayIn<'_>],
    tools: &[&Tool],
    at_least_one: bool,
) -> Option<Grammar> {
    let block = markers.block.as_ref()?;
    let one_call = Grammar::Or(
        tools
            .iter()
            .map(|tool| call(markers, &tool.function.name, &tool.function.parameters))
            .collect(),
    );
    let content = Grammar::Plus(Box::new(one_call));
    let tags = ways_in
        .iter()
        .map(|way| Tag::new(way.begin(), content.clone(), block.close))
        .collect();
    let mut triggers: Vec<String> = Vec::new();
    for trigger in ways_in.iter().map(WayIn::trigger) {
        if !triggers.iter().any(|known| known == trigger) {
            triggers.push(trigger.to_string());
        }
    }
    Some(Grammar::TriggeredTags {
        triggers,
        tags,
        at_least_one,
    })
}

/// One call: the call's opener and the tool's name, its index, the arguments and the call's close.
fn call(markers: &CallMarkers<'_>, name: &str, parameters: &Value) -> Grammar {
    Grammar::Sequence(vec![
        Grammar::ConstString(format!("{}{}\" index=\"", markers.call_open, escape(name))),
        Grammar::Regex(INDEX.to_string()),
        Grammar::ConstString(format!("\"{SEP}")),
        arguments(parameters),
        Grammar::ConstString(markers.call_close.to_string()),
    ])
}

/// The arguments the tool's schema asks for: one tag per property, in the schema's order, each
/// optional unless required; any arguments at all for a schema without properties.
fn arguments(parameters: &Value) -> Grammar {
    let properties = parameters
        .get("properties")
        .and_then(Value::as_object)
        .filter(|properties| !properties.is_empty());
    let Some(properties) = properties else {
        return Grammar::Star(Box::new(any_argument()));
    };
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let definitions = Definitions::of(parameters);
    Grammar::Sequence(
        properties
            .iter()
            .map(|(key, schema)| {
                let tag = argument(key, schema, definitions);
                if required.contains(&key.as_str()) {
                    tag
                } else {
                    Grammar::Optional(Box::new(tag))
                }
            })
            .collect(),
    )
}

/// One argument tag: the key, the type the property's schema pins and a value of that type, or
/// the key with any type and any value when the schema pins none.
fn argument(key: &str, schema: &Value, definitions: Definitions<'_>) -> Grammar {
    let key = escape(key);
    match shape(schema, definitions) {
        Some((type_name, value)) => Grammar::Sequence(vec![
            Grammar::ConstString(format!(
                "{OPEN}argument key=\"{key}\" type=\"{type_name}\"{SEP}"
            )),
            value,
            Grammar::ConstString(format!("{CLOSE}argument{SEP}")),
        ]),
        None => Grammar::Sequence(vec![
            Grammar::ConstString(format!("{OPEN}argument key=\"{key}\" type=\"")),
            attribute_text(),
            Grammar::ConstString(format!("\"{SEP}")),
            any_value(),
            Grammar::ConstString(format!("{CLOSE}argument{SEP}")),
        ]),
    }
}

/// An argument tag with any key, any type and any value.
fn any_argument() -> Grammar {
    Grammar::Sequence(vec![
        Grammar::ConstString(format!("{OPEN}argument key=\"")),
        attribute_text(),
        Grammar::ConstString("\" type=\"".to_string()),
        attribute_text(),
        Grammar::ConstString(format!("\"{SEP}")),
        any_value(),
        Grammar::ConstString(format!("{CLOSE}argument{SEP}")),
    ])
}

/// The text of a quoted attribute: no quote, and not the start of a marker.
fn attribute_text() -> Grammar {
    Grammar::AnyText {
        excludes: vec!["\"".to_string(), "<|".to_string()],
    }
}

/// A value written as it is: any text up to a marker, a lone `<` or `|` included.
fn any_value() -> Grammar {
    Grammar::AnyText {
        excludes: vec![CLOSE.to_string(), OPEN.to_string(), SEP.to_string()],
    }
}

/// The `type` attribute and the value grammar a property's schema pins, or `None` when it pins
/// no type the syntax has, or a `$ref` in it points at nothing (a grammar with such a pointer
/// would not compile, and the whole call with it). A `$ref` lends its target's keywords, and the
/// property's own win, as in JSON Schema 2020-12. The template writes `integer` as `number`.
fn shape(schema: &Value, definitions: Definitions<'_>) -> Option<(&'static str, Grammar)> {
    if !definitions.pointers_resolve(schema) {
        return None;
    }
    let target = definitions.target_of(schema);
    let keyword = |name: &str| {
        schema
            .get(name)
            .or_else(|| target.and_then(|target| target.get(name)))
    };
    if let Some(values) = keyword("enum").and_then(Value::as_array) {
        if !values.is_empty() && values.iter().all(Value::is_string) {
            let options = values
                .iter()
                .filter_map(Value::as_str)
                .map(|value| Grammar::ConstString(value.to_string()))
                .collect();
            return Some(("string", Grammar::Or(options)));
        }
    }
    let json = |schema: Value| Grammar::JsonSchema {
        schema,
        style: None,
    };
    match keyword("type").and_then(Value::as_str)? {
        "string" => Some(("string", any_value())),
        "integer" | "number" => Some(("number", json(definitions.attached_to(schema)))),
        "boolean" => Some(("boolean", json(json!({"type": "boolean"})))),
        "object" => Some(("object", json(definitions.attached_to(schema)))),
        "array" => Some(("array", json(definitions.attached_to(schema)))),
        "null" => Some(("null", Grammar::ConstString("null".to_string()))),
        _ => None,
    }
}

/// What the template escapes in an attribute: `&` first, then `"`.
fn escape(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

/// The definitions at the root of a tool's schema, `$defs` and `definitions`, and the root itself.
/// Each property's schema goes to the engine as a document of its own, and a `$ref` such as
/// `#/$defs/Node` points from the root of the document it stands in, so the definitions travel
/// with the property's schema.
#[derive(Clone, Copy)]
struct Definitions<'a> {
    root: &'a Value,
    defs: Option<&'a Value>,
    definitions: Option<&'a Value>,
}

impl<'a> Definitions<'a> {
    fn of(root: &'a Value) -> Self {
        Self {
            root,
            defs: root.get("$defs"),
            definitions: root.get("definitions"),
        }
    }

    /// The schema a local pointer names: `#` the root, `#/$defs/Name` or `#/definitions/Name` an
    /// entry, with the pointer's escapes (`~1` for `/`, `~0` for `~`) undone.
    fn resolve(self, pointer: &str) -> Option<&'a Value> {
        if pointer == "#" || pointer == "#/" {
            return Some(self.root);
        }
        let mut segments = pointer.strip_prefix("#/")?.split('/');
        let mut node = match segments.next()? {
            "$defs" => self.defs?,
            "definitions" => self.definitions?,
            _ => return None,
        };
        for segment in segments {
            node = node.get(segment.replace("~1", "/").replace("~0", "~"))?;
        }
        Some(node)
    }

    /// The schema the property's own `$ref` names, one hop: its keywords stand beside the
    /// property's, and a cycle of definitions is the engine's to expand, not this function's.
    fn target_of(self, schema: &'a Value) -> Option<&'a Value> {
        self.resolve(schema.get("$ref")?.as_str()?)
    }

    /// Whether every local pointer in `schema` names something.
    fn pointers_resolve(self, schema: &Value) -> bool {
        match schema {
            Value::Object(map) => map.iter().all(|(key, value)| {
                let points_at_nothing = is_reference(key)
                    && value.as_str().is_some_and(|pointer| {
                        pointer.starts_with('#') && self.resolve(pointer).is_none()
                    });
                !points_at_nothing && self.pointers_resolve(value)
            }),
            Value::Array(items) => items.iter().all(|item| self.pointers_resolve(item)),
            _ => true,
        }
    }

    /// The property's schema with the root's definitions attached, so a local pointer in it still
    /// resolves when the engine compiles it on its own. A definition the schema declares itself
    /// stays; the root's fill in the names it lacks. A schema with no local pointer goes as it is.
    fn attached_to(self, schema: &Value) -> Value {
        let mut attached = schema.clone();
        if self.defs.is_none() && self.definitions.is_none() || !has_local_reference(schema) {
            return attached;
        }
        let Some(object) = attached.as_object_mut() else {
            return attached;
        };
        for (name, block) in [("$defs", self.defs), ("definitions", self.definitions)] {
            let Some(Value::Object(entries)) = block else {
                continue;
            };
            match object.get_mut(name) {
                Some(Value::Object(own)) => {
                    for (key, value) in entries {
                        own.entry(key.clone()).or_insert_with(|| value.clone());
                    }
                }
                Some(_) => {}
                None => {
                    object.insert(name.to_string(), Value::Object(entries.clone()));
                }
            }
        }
        attached
    }
}

/// Whether `key` is one of the keywords that point at another schema.
fn is_reference(key: &str) -> bool {
    matches!(key, "$ref" | "$dynamicRef" | "$recursiveRef")
}

/// Whether `schema` points anywhere inside its own document.
fn has_local_reference(schema: &Value) -> bool {
    match schema {
        Value::Object(map) => map.iter().any(|(key, value)| {
            is_reference(key)
                && value
                    .as_str()
                    .is_some_and(|pointer| pointer.starts_with('#'))
                || has_local_reference(value)
        }),
        Value::Array(items) => items.iter().any(has_local_reference),
        _ => false,
    }
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
            },
        }
    }

    /// An argument tag's opener for `key`, typed `type_name`.
    fn argument_open(key: &str, type_name: &str) -> String {
        format!("<|open|>argument key=\"{key}\" type=\"{type_name}\"<|sep|>")
    }

    const ARGUMENT_CLOSE: &str = "<|close|>argument<|sep|>";

    /// The first call of the tag's first way in, as JSON.
    fn first_call(tools: &[Tool]) -> Value {
        let payload = formats::kimi_k3()
            .grammar(tools, true, false)
            .expect("a grammar")
            .payload();
        payload["format"]["tags"][0]["content"]["content"]["elements"][0].clone()
    }

    #[test]
    fn a_call_pins_its_framing_and_the_tools_name() {
        // The old crate's `build_call_format`: the opener with the name, the index, `"<|sep|>`,
        // the arguments, the call's close.
        let call = first_call(&[tool(
            "get_weather",
            value!({"type": "object", "properties": {"city": {"type": "string"}}}),
        )]);
        assert_eq!(call["type"], "sequence");
        let elements = call["elements"].as_array().expect("five elements");
        assert_eq!(elements.len(), 5);
        assert_eq!(
            elements[0]["value"],
            "<|open|>call tool=\"get_weather\" index=\""
        );
        assert_eq!(
            elements[1],
            value!({"type": "regex", "pattern": "[1-9][0-9]*"})
        );
        assert_eq!(elements[2]["value"], "\"<|sep|>");
        assert_eq!(elements[3]["type"], "sequence");
        assert_eq!(elements[4]["value"], "<|close|>call<|sep|>");
    }

    #[test]
    fn the_arguments_follow_the_schema_one_tag_per_property_in_its_order() {
        // The old crate's `build_args_format` and `classify_arg`: a required string as any text up
        // to a marker, a required integer as `number` with the property's schema, an optional
        // string enum as one of its values inside `optional`.
        let call = first_call(&[tool(
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
        )]);
        let markers = value!(["<|close|>", "<|open|>", "<|sep|>"]);
        assert_eq!(
            call["elements"][3],
            value!({"type": "sequence", "elements": [
                {"type": "sequence", "elements": [
                    {"type": "const_string", "value": argument_open("city", "string")},
                    {"type": "any_text", "excludes": markers},
                    {"type": "const_string", "value": ARGUMENT_CLOSE},
                ]},
                {"type": "sequence", "elements": [
                    {"type": "const_string", "value": argument_open("days", "number")},
                    {"type": "json_schema", "json_schema": {"type": "integer", "minimum": 1}},
                    {"type": "const_string", "value": ARGUMENT_CLOSE},
                ]},
                {"type": "optional", "content": {"type": "sequence", "elements": [
                    {"type": "const_string", "value": argument_open("units", "string")},
                    {"type": "or", "elements": [
                        {"type": "const_string", "value": "c"},
                        {"type": "const_string", "value": "f"},
                    ]},
                    {"type": "const_string", "value": ARGUMENT_CLOSE},
                ]}},
            ]})
        );
    }

    #[test]
    fn every_type_the_syntax_has_pins_its_attribute_and_value() {
        let call = first_call(&[tool(
            "t",
            value!({
                "type": "object",
                "properties": {
                    "flag": {"type": "boolean", "description": "ignored"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "where": {"type": "object", "properties": {"x": {"type": "number"}}},
                    "nothing": {"type": "null"},
                    "ratio": {"type": "number"},
                },
                "required": ["flag", "tags", "where", "nothing", "ratio"],
            }),
        )]);
        let slots = call["elements"][3]["elements"]
            .as_array()
            .expect("five slots");
        let opener = |slot: &Value| slot["elements"][0]["value"].clone();
        let pinned = |slot: &Value| slot["elements"][1].clone();
        assert_eq!(
            opener(&slots[0]),
            "<|open|>argument key=\"flag\" type=\"boolean\"<|sep|>"
        );
        assert_eq!(
            pinned(&slots[0]),
            value!({"type": "json_schema", "json_schema": {"type": "boolean"}})
        );
        assert_eq!(
            opener(&slots[1]),
            "<|open|>argument key=\"tags\" type=\"array\"<|sep|>"
        );
        assert_eq!(
            pinned(&slots[1])["json_schema"],
            value!({"type": "array", "items": {"type": "string"}})
        );
        assert_eq!(
            opener(&slots[2]),
            "<|open|>argument key=\"where\" type=\"object\"<|sep|>"
        );
        assert_eq!(
            opener(&slots[3]),
            "<|open|>argument key=\"nothing\" type=\"null\"<|sep|>"
        );
        assert_eq!(
            pinned(&slots[3]),
            value!({"type": "const_string", "value": "null"})
        );
        assert_eq!(
            opener(&slots[4]),
            "<|open|>argument key=\"ratio\" type=\"number\"<|sep|>"
        );
    }

    #[test]
    fn a_reference_brings_the_tools_definitions_along_and_a_dangling_one_pins_nothing() {
        let call = first_call(&[tool(
            "t",
            value!({
                "type": "object",
                "$defs": {"Count": {"type": "integer"}, "Unused": {"type": "string"}},
                "properties": {
                    "count": {"$ref": "#/$defs/Count"},
                    "lost": {"$ref": "#/$defs/Missing"},
                    "untyped": {"description": "anything"},
                },
                "required": ["count", "lost", "untyped"],
            }),
        )]);
        let slots = call["elements"][3]["elements"]
            .as_array()
            .expect("three slots");
        // The `$ref` lends its target's type, and the definitions travel with the schema.
        assert_eq!(
            slots[0]["elements"][0]["value"],
            "<|open|>argument key=\"count\" type=\"number\"<|sep|>"
        );
        assert_eq!(
            slots[0]["elements"][1]["json_schema"],
            value!({
                "$ref": "#/$defs/Count",
                "$defs": {"Count": {"type": "integer"}, "Unused": {"type": "string"}},
            })
        );
        // A pointer at nothing, or no type: the key stays, the type and the value are the model's.
        for (slot, key) in [(&slots[1], "lost"), (&slots[2], "untyped")] {
            let begin = format!("<|open|>argument key=\"{key}\" type=\"");
            assert_eq!(
                *slot,
                value!({"type": "sequence", "elements": [
                    {"type": "const_string", "value": begin},
                    {"type": "any_text", "excludes": ["\"", "<|"]},
                    {"type": "const_string", "value": "\"<|sep|>"},
                    {"type": "any_text", "excludes": ["<|close|>", "<|open|>", "<|sep|>"]},
                    {"type": "const_string", "value": ARGUMENT_CLOSE},
                ]})
            );
        }
    }

    #[test]
    fn attribute_values_are_escaped_as_the_template_writes_them() {
        let parameters = value!({
            "type": "object",
            "properties": {"k\"q": {"type": "string"}},
            "required": ["k\"q"],
        });
        let call = first_call(&[tool("a&b\"c", parameters)]);
        assert_eq!(
            call["elements"][0]["value"],
            "<|open|>call tool=\"a&amp;b&quot;c\" index=\""
        );
        assert_eq!(
            call["elements"][3]["elements"][0]["elements"][0]["value"],
            "<|open|>argument key=\"k&quot;q\" type=\"string\"<|sep|>"
        );
    }

    #[test]
    fn a_schema_without_properties_takes_any_arguments() {
        for parameters in [
            value!({"type": "object"}),
            value!({}),
            value!({"properties": {}}),
        ] {
            let call = first_call(&[tool("ping", parameters)]);
            assert_eq!(call["elements"][3]["type"], "star");
            assert_eq!(
                call["elements"][3]["content"]["elements"][0]["value"],
                "<|open|>argument key=\""
            );
        }
    }
}
