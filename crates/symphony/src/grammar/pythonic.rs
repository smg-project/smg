//! Python calls written from the table's markers: Olmo 3's, one call per line between
//! `<function_calls>` and `</function_calls>`, and LFM2.5's, a Python list between
//! `<|tool_call_start|>` and `<|tool_call_end|>`.
//!
//! A call is the tool's name, `(`, its arguments and `)`. The arguments are `key=value` pairs in
//! the schema's order, joined by `, `, each present when the schema requires it and otherwise
//! optional; since the comma stands between two present pairs only, the pairs are written as one
//! choice per pair that may come first, the required ones and those after it following each with
//! its comma. A value follows the property's type in the dialect's spelling of a value, which is
//! the template's: LFM2.5 writes Python, a string in either quote (`"Paris"`, `'Paris'`), a
//! string `enum` as one of its values quoted the same way, a boolean as `True` or `False`, `null`
//! as `None`; Olmo 3 writes every value through `tojson`, so JSON, a string in double quotes, an
//! `enum` value the same, `true`, `false` and `null`. A number, an object or an array is the JSON
//! the property's schema accepts under both, which the templates write with `tojson` and the
//! assembler reads as it reads a Python literal. An `enum` behind a `$ref` is the target's, as
//! every keyword is. A property that pins no type, or a schema without properties, takes any text
//! up to the call's `)`.

use openai_protocol::common::Tool;
use serde_json::{json, Value};

use super::{
    schema::{keyword, shape, Definitions},
    CallMarkers, Grammar, Tag,
};
use crate::pythonic::{Dialect, Framing, Values};

/// The calls for `tools` in the region the table's markers frame: one tag, opened by the region's
/// opener (and the list's `[`), holding one call or more in the dialect's framing, each value in
/// its spelling, closed by the list's `]` and the region's close.
pub(super) fn calls(
    markers: &CallMarkers<'_>,
    dialect: Dialect,
    tools: &[&Tool],
    at_least_one: bool,
) -> Grammar {
    let one_call = Grammar::Or(
        tools
            .iter()
            .map(|tool| {
                call(
                    &tool.function.name,
                    &tool.function.parameters,
                    dialect.values,
                )
            })
            .collect(),
    );
    let (open, between, close) = match dialect.framing {
        Framing::OnePerLine => ("", "\n", ""),
        Framing::List => ("[", ", ", "]"),
    };
    let content = Grammar::Sequence(vec![
        one_call.clone(),
        Grammar::Star(Box::new(Grammar::Sequence(vec![
            Grammar::ConstString(between.to_string()),
            one_call,
        ]))),
    ]);
    Grammar::TriggeredTags {
        triggers: vec![markers.call_open.to_string()],
        tags: vec![Tag::new(
            format!("{}{open}", markers.call_open),
            content,
            format!("{close}{}", markers.call_close),
        )],
        at_least_one,
    }
}

/// One call: the name and `(`, the arguments, `)`.
fn call(name: &str, parameters: &Value, values: Values) -> Grammar {
    Grammar::Sequence(vec![
        Grammar::ConstString(format!("{name}(")),
        arguments(parameters, values),
        Grammar::ConstString(")".to_string()),
    ])
}

/// The arguments the tool's schema asks for, joined by `, `: every pair the schema requires, and
/// each other pair optional. The comma stands between two present pairs only, so the sequence is
/// one choice per pair that may stand first (every pair up to the first required one), each
/// followed by the pairs after it, a required one with its comma, an optional one with its comma
/// inside the option. A schema without properties takes any text up to the `)`.
fn arguments(parameters: &Value, values: Values) -> Grammar {
    let Some(properties) = parameters
        .get("properties")
        .and_then(Value::as_object)
        .filter(|properties| !properties.is_empty())
    else {
        return any_arguments();
    };
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let definitions = Definitions::of(parameters);
    let pairs: Vec<(Grammar, bool)> = properties
        .iter()
        .map(|(key, schema)| {
            (
                argument(key, schema, definitions, values),
                required.contains(&key.as_str()),
            )
        })
        .collect();
    let first_required = pairs.iter().position(|(_, required)| *required);
    let may_start = first_required.map_or(pairs.len(), |at| at + 1);
    let starts: Vec<Grammar> = (0..may_start)
        .map(|first| {
            let mut elements = vec![pairs[first].0.clone()];
            for (pair, required) in &pairs[first + 1..] {
                let with_comma =
                    Grammar::Sequence(vec![Grammar::ConstString(", ".to_string()), pair.clone()]);
                elements.push(if *required {
                    with_comma
                } else {
                    Grammar::Optional(Box::new(with_comma))
                });
            }
            Grammar::Sequence(elements)
        })
        .collect();
    let some = match starts.len() {
        1 => starts.into_iter().next().unwrap_or_else(any_arguments),
        _ => Grammar::Or(starts),
    };
    if first_required.is_some() {
        some
    } else {
        Grammar::Optional(Box::new(some))
    }
}

/// One pair: the key, `=`, and the value the property's schema pins.
fn argument(key: &str, schema: &Value, definitions: Definitions<'_>, values: Values) -> Grammar {
    Grammar::Sequence(vec![
        Grammar::ConstString(format!("{key}=")),
        value(schema, definitions, values),
    ])
}

/// A value as the dialect writes one: a string `enum` as one of its values quoted the dialect's
/// way, a string, a boolean and `null` in the dialect's spelling, the rest as the JSON the schema
/// accepts; any text up to the call's end where the schema pins nothing. The `enum` is read
/// through the property's `$ref`, as [`shape`] reads every keyword, so that a reference to a
/// definition holding one (an `Enum` field in Pydantic's schemas) is quoted as the template
/// quotes it and not written bare.
fn value(schema: &Value, definitions: Definitions<'_>, values: Values) -> Grammar {
    let target = definitions.target_of(schema);
    if let Some(options) = keyword(schema, target, "enum").and_then(Value::as_array) {
        if !options.is_empty() && options.iter().all(Value::is_string) {
            return Grammar::Or(
                options
                    .iter()
                    .filter_map(Value::as_str)
                    .flat_map(|option| quoted(option, values))
                    .collect(),
            );
        }
    }
    let string = move || match values {
        Values::Python => python_string(),
        Values::Json => json_string(),
    };
    match shape(schema, definitions, string) {
        Some(("boolean", _)) => match values {
            Values::Python => words(&["True", "False"]),
            Values::Json => words(&["true", "false"]),
        },
        Some(("null", _)) => Grammar::ConstString(
            match values {
                Values::Python => "None",
                Values::Json => "null",
            }
            .to_string(),
        ),
        Some((_, value)) => value,
        None => any_value(),
    }
}

/// An `enum` value as the dialect quotes it: in either quote under Python, each with its own
/// escaping; as the one JSON string under JSON.
fn quoted(option: &str, values: Values) -> Vec<Grammar> {
    match values {
        Values::Python => vec![
            Grammar::ConstString(format!("\"{}\"", escape(option, '"'))),
            Grammar::ConstString(format!("'{}'", escape(option, '\''))),
        ],
        Values::Json => vec![Grammar::ConstString(
            Value::String(option.to_string()).to_string(),
        )],
    }
}

/// A Python string in either quote: the quote, any text with its quotes and backslashes escaped,
/// the same quote.
fn python_string() -> Grammar {
    Grammar::Regex(r#""([^"\\]|\\.)*"|'([^'\\]|\\.)*'"#.to_string())
}

/// A JSON string, as `tojson` writes one.
fn json_string() -> Grammar {
    Grammar::JsonSchema {
        schema: json!({"type": "string"}),
        style: None,
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

/// A value the schema does not type: any text that holds no `)`.
fn any_value() -> Grammar {
    Grammar::AnyText {
        excludes: vec![")".to_string()],
    }
}

/// The arguments of a tool whose schema has no properties: any text that holds no `)`.
fn any_arguments() -> Grammar {
    any_value()
}

/// `value` with the backslashes and the `quote` it will stand in escaped.
fn escape(value: &str, quote: char) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        if c == '\\' || c == quote {
            out.push('\\');
        }
        out.push(c);
    }
    out
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

    fn python_string_json() -> Value {
        value!({"type": "regex", "pattern": r#""([^"\\]|\\.)*"|'([^'\\]|\\.)*'"#})
    }

    fn json_string_json() -> Value {
        value!({"type": "json_schema", "json_schema": {"type": "string"}})
    }

    #[test]
    fn olmo3_frames_one_call_per_line_and_types_each_argument() {
        // Olmo 3's template: `<function_calls>`, one call per line, `</function_calls>`; a call is
        // the name, `(`, `key=value` pairs joined by `, `, `)`, every value through `tojson`: a
        // string and an enum value as JSON strings, a number as JSON.
        let payload = formats::olmo3()
            .grammar(&weather_tools(), true, false)
            .expect("a grammar")
            .payload();
        let days =
            value!({"type": "json_schema", "json_schema": {"type": "integer", "minimum": 1}});
        let weather = value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": "get_weather("},
            {"type": "sequence", "elements": [
                {"type": "sequence", "elements": [
                    {"type": "const_string", "value": "city="},
                    json_string_json(),
                ]},
                {"type": "sequence", "elements": [
                    {"type": "const_string", "value": ", "},
                    {"type": "sequence", "elements": [
                        {"type": "const_string", "value": "days="},
                        days,
                    ]},
                ]},
                {"type": "optional", "content": {"type": "sequence", "elements": [
                    {"type": "const_string", "value": ", "},
                    {"type": "sequence", "elements": [
                        {"type": "const_string", "value": "units="},
                        {"type": "or", "elements": [
                            {"type": "const_string", "value": "\"c\""},
                            {"type": "const_string", "value": "\"f\""},
                        ]},
                    ]},
                ]}},
            ]},
            {"type": "const_string", "value": ")"},
        ]});
        let ping = value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": "ping("},
            {"type": "any_text", "excludes": [")"]},
            {"type": "const_string", "value": ")"},
        ]});
        let one_call = value!({"type": "or", "elements": [weather, ping]});
        assert_eq!(
            payload,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<function_calls>"],
                "tags": [{
                    "type": "tag",
                    "begin": "<function_calls>",
                    "content": {"type": "sequence", "elements": [
                        one_call,
                        {"type": "star", "content": {"type": "sequence", "elements": [
                            {"type": "const_string", "value": "\n"},
                            one_call,
                        ]}},
                    ]},
                    "end": "</function_calls>",
                }],
                "at_least_one": true,
            }})
        );
    }

    #[test]
    fn lfm2_5_frames_the_calls_as_a_list_between_its_markers() {
        let payload = formats::lfm2_5()
            .grammar(&weather_tools(), false, false)
            .expect("a grammar")
            .payload();
        let tag = &payload["format"]["tags"][0];
        assert_eq!(tag["begin"], "<|tool_call_start|>[");
        assert_eq!(tag["end"], "]<|tool_call_end|>");
        assert_eq!(
            payload["format"]["triggers"],
            value!(["<|tool_call_start|>"])
        );
        assert_eq!(payload["format"]["at_least_one"], false);
        let between = &tag["content"]["elements"][1]["content"]["elements"][0];
        assert_eq!(between, &value!({"type": "const_string", "value": ", "}));
        // LFM2.5's values are Python's: a string in either quote, an enum value quoted both ways.
        let weather = &tag["content"]["elements"][0]["elements"][0]["elements"][1];
        assert_eq!(weather["elements"][0]["elements"][1], python_string_json());
        assert_eq!(
            weather["elements"][2]["content"]["elements"][1]["elements"][1],
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "\"c\""},
                {"type": "const_string", "value": "'c'"},
                {"type": "const_string", "value": "\"f\""},
                {"type": "const_string", "value": "'f'"},
            ]})
        );
    }

    #[test]
    fn olmo3_writes_a_boolean_and_a_null_as_json_and_lfm2_5_as_python() {
        // Olmo 3's template writes `key=value | tojson`: every recorded boolean is `true` or
        // `false` and every null `null`; LFM2.5's writes Python's `True`, `False` and `None`.
        let tools = vec![tool(
            "connect",
            value!({"type": "object", "properties": {
                "use_ssl": {"type": "boolean"},
                "proxy": {"type": "null"},
            }, "required": ["use_ssl", "proxy"]}),
        )];
        let olmo3 = formats::olmo3()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let olmo3 =
            &olmo3["format"]["tags"][0]["content"]["elements"][0]["elements"][0]["elements"][1];
        assert_eq!(
            olmo3["elements"][0]["elements"][1],
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "true"},
                {"type": "const_string", "value": "false"},
            ]})
        );
        assert_eq!(
            olmo3["elements"][1]["elements"][1]["elements"][1],
            value!({"type": "const_string", "value": "null"})
        );
        let lfm2_5 = formats::lfm2_5()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let lfm2_5 =
            &lfm2_5["format"]["tags"][0]["content"]["elements"][0]["elements"][0]["elements"][1];
        assert_eq!(
            lfm2_5["elements"][0]["elements"][1],
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "True"},
                {"type": "const_string", "value": "False"},
            ]})
        );
        assert_eq!(
            lfm2_5["elements"][1]["elements"][1]["elements"][1],
            value!({"type": "const_string", "value": "None"})
        );
    }

    #[test]
    fn a_string_enum_behind_a_reference_is_quoted_as_the_dialect_quotes_it() {
        // Pydantic writes an `Enum` field as a `$ref` to a definition holding the `enum`.
        let root = value!({
            "type": "object",
            "properties": {"units": {"$ref": "#/$defs/Unit"}},
            "$defs": {"Unit": {"type": "string", "enum": ["c", "f"]}},
        });
        let definitions = Definitions::of(&root);
        let units = &root["properties"]["units"];
        assert_eq!(
            value(units, definitions, Values::Json).to_json(),
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "\"c\""},
                {"type": "const_string", "value": "\"f\""},
            ]})
        );
        assert_eq!(
            value(units, definitions, Values::Python).to_json(),
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "\"c\""},
                {"type": "const_string", "value": "'c'"},
                {"type": "const_string", "value": "\"f\""},
                {"type": "const_string", "value": "'f'"},
            ]})
        );
    }

    #[test]
    fn optional_arguments_put_the_comma_between_present_pairs_only() {
        // Two optional pairs and no required one: a alone, a then b, b alone, or no pair at all.
        let schema = value!({"type": "object", "properties": {
            "a": {"type": "integer"},
            "b": {"type": "boolean"},
        }});
        let grammar = arguments(&schema, Values::Python);
        let a = value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": "a="},
            {"type": "json_schema", "json_schema": {"type": "integer"}},
        ]});
        let b = value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": "b="},
            {"type": "or", "elements": [
                {"type": "const_string", "value": "True"},
                {"type": "const_string", "value": "False"},
            ]},
        ]});
        assert_eq!(
            grammar.to_json(),
            value!({"type": "optional", "content": {"type": "or", "elements": [
                {"type": "sequence", "elements": [
                    a,
                    {"type": "optional", "content": {"type": "sequence", "elements": [
                        {"type": "const_string", "value": ", "},
                        b,
                    ]}},
                ]},
                {"type": "sequence", "elements": [b]},
            ]}})
        );
        // A required pair after an optional one: the optional may start, or the required does.
        let schema = value!({"type": "object", "properties": {
            "a": {"type": "integer"},
            "b": {"type": "boolean"},
        }, "required": ["b"]});
        let json = arguments(&schema, Values::Python).to_json();
        assert_eq!(json["type"], "or");
        assert_eq!(json["elements"].as_array().expect("two starts").len(), 2);
    }

    #[test]
    fn a_null_and_a_quoted_enum_value_are_spelled_as_the_dialect_writes_them() {
        let root = value!({});
        let definitions = Definitions::of(&root);
        assert_eq!(
            value(&value!({"type": "null"}), definitions, Values::Python).to_json(),
            value!({"type": "const_string", "value": "None"})
        );
        assert_eq!(
            value(&value!({"enum": ["it's"]}), definitions, Values::Python).to_json(),
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "\"it's\""},
                {"type": "const_string", "value": "'it\\'s'"},
            ]})
        );
        // JSON: one string, escaped as `tojson` escapes it.
        assert_eq!(
            value(&value!({"enum": ["say \"hi\""]}), definitions, Values::Json).to_json(),
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "\"say \\\"hi\\\"\""},
            ]})
        );
        for values in [Values::Python, Values::Json] {
            assert_eq!(
                value(&value!({}), definitions, values).to_json(),
                value!({"type": "any_text", "excludes": [")"]})
            );
        }
    }
}
