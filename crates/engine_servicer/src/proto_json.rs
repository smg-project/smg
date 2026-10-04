//! JSON text as the proto `Struct`/`Value` the info RPCs carry: the launcher
//! hands the servicers their `server_args` and `scheduler_info` as JSON
//! objects, computed in Python before the engine starts.

use prost_types::{value::Kind, ListValue, Struct, Value};

/// A JSON object as a proto `Struct`; anything else (including unparsable
/// text) is an empty one.
pub(crate) fn struct_from_json(json: &str) -> Struct {
    match serde_json::from_str::<serde_json::Value>(json) {
        Ok(serde_json::Value::Object(map)) => Struct {
            fields: map
                .into_iter()
                .map(|(key, value)| (key, value_from_json(value)))
                .collect(),
        },
        _ => Struct::default(),
    }
}

pub(crate) fn value_from_json(value: serde_json::Value) -> Value {
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(flag) => Kind::BoolValue(flag),
        serde_json::Value::Number(number) => Kind::NumberValue(number.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(text) => Kind::StringValue(text),
        serde_json::Value::Array(items) => Kind::ListValue(ListValue {
            values: items.into_iter().map(value_from_json).collect(),
        }),
        serde_json::Value::Object(map) => Kind::StructValue(Struct {
            fields: map
                .into_iter()
                .map(|(key, value)| (key, value_from_json(value)))
                .collect(),
        }),
    };
    Value { kind: Some(kind) }
}

/// Whether `json` parses as a JSON object (what the info RPCs can carry).
pub(crate) fn is_json_object(json: &str) -> bool {
    matches!(
        serde_json::from_str::<serde_json::Value>(json),
        Ok(serde_json::Value::Object(_))
    )
}

/// An integer as a proto number `Value`.
pub(crate) fn number(value: i32) -> Value {
    Value {
        kind: Some(Kind::NumberValue(f64::from(value))),
    }
}
