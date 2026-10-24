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

/// The handshake's capacity figures a scheduler servicer adds to the
/// launcher's `scheduler_info` once the engine is linked, under the keys the
/// Python servicer reports them by.
pub(crate) fn insert_capacity(
    scheduler_info: &mut Struct,
    max_total_num_tokens: i32,
    page_size: u64,
    max_running_requests: i32,
) {
    scheduler_info.fields.insert(
        "max_total_num_tokens".to_string(),
        number(max_total_num_tokens),
    );
    scheduler_info.fields.insert(
        "page_size".to_string(),
        number(i32::try_from(page_size).unwrap_or(i32::MAX)),
    );
    scheduler_info.fields.insert(
        "max_running_requests".to_string(),
        number(max_running_requests),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_capacity_figures_join_the_launchers_entries_under_the_python_servicers_keys() {
        fn number_of(scheduler_info: &Struct, key: &str) -> f64 {
            match scheduler_info.fields[key].kind {
                Some(Kind::NumberValue(value)) => value,
                ref other => panic!("{key}: {other:?}"),
            }
        }
        let mut scheduler_info = struct_from_json(r#"{"is_generation":true}"#);
        insert_capacity(&mut scheduler_info, 4096, 16, 64);
        assert_eq!(number_of(&scheduler_info, "max_total_num_tokens"), 4096.0);
        assert_eq!(number_of(&scheduler_info, "page_size"), 16.0);
        assert_eq!(number_of(&scheduler_info, "max_running_requests"), 64.0);
        assert!(matches!(
            scheduler_info.fields["is_generation"].kind,
            Some(Kind::BoolValue(true))
        ));
        // A page size past i32 saturates, as every figure on this wire does.
        insert_capacity(&mut scheduler_info, 0, u64::MAX, 0);
        assert_eq!(number_of(&scheduler_info, "page_size"), f64::from(i32::MAX));
    }
}
