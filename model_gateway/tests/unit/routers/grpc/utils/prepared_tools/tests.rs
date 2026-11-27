use std::sync::Mutex;

use serde_json::json;

use super::*;

/// The table is one per process: the tests that read its state run one
/// at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn tool(name: &str, description: &str) -> Tool {
    serde_json::from_value(json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
        }
    }))
    .unwrap()
}

#[test]
fn a_tool_set_is_prepared_once_and_shared_by_the_next_request_with_the_same_tools() {
    let _serial = SERIAL.lock().unwrap();
    let tools = [
        tool("prepared_once_a", "the first"),
        tool("prepared_once_b", "the second"),
    ];
    let first = prepare_tools(&tools).unwrap();
    assert_eq!(
        first.tools(),
        tools
            .iter()
            .map(|tool| serde_json::to_value(tool).unwrap())
            .collect::<Vec<_>>(),
        "the prepared values are the request's tools as JSON values"
    );
    let equal = [tools[0].clone(), tools[1].clone()];
    assert!(
        Arc::ptr_eq(&first, &prepare_tools(&equal).unwrap()),
        "equal tools share the one kept value"
    );
    let other = [tool("prepared_once_a", "another description")];
    let third = prepare_tools(&other).unwrap();
    assert!(
        !Arc::ptr_eq(&first, &third),
        "other tools are another value"
    );
    assert_eq!(third.tools(), [serde_json::to_value(&other[0]).unwrap()]);
    let reordered = [tools[1].clone(), tools[0].clone()];
    assert!(
        !Arc::ptr_eq(&first, &prepare_tools(&reordered).unwrap()),
        "the same tools in another order render differently and are another value"
    );
}

#[test]
fn the_table_starts_over_at_its_bound_and_keeps_no_oversize_set() {
    let _serial = SERIAL.lock().unwrap();
    let first_tools = [tool("bound_first", "kept until the bound")];
    let first = prepare_tools(&first_tools).unwrap();
    for i in 0..PREPARED_TOOLS_KEPT {
        prepare_tools(&[tool(&format!("bound_{i}"), "one of many")]).unwrap();
    }
    assert!(
        PREPARED_TOOLS.read().sets.len() <= PREPARED_TOOLS_KEPT,
        "the table never holds more than its bound"
    );
    assert!(
        !Arc::ptr_eq(&first, &prepare_tools(&first_tools).unwrap()),
        "a set the restart dropped is prepared again"
    );
    let oversize = [tool("oversize", &"x".repeat(PREPARED_TOOLS_MAX_KEY_BYTES))];
    let once = prepare_tools(&oversize).unwrap();
    let twice = prepare_tools(&oversize).unwrap();
    assert!(
        !Arc::ptr_eq(&once, &twice),
        "a set above the size bound is prepared for its request alone"
    );
    assert_eq!(once.tools(), twice.tools());
    KEY.with(|key| {
        assert!(
            key.borrow().capacity() <= PREPARED_TOOLS_MAX_KEY_BYTES,
            "the buffer is dropped back after an oversize set"
        );
    });
}

#[test]
fn the_table_starts_over_when_the_kept_sets_would_exceed_their_weight() {
    let _serial = SERIAL.lock().unwrap();
    let first_tools = [tool("weight_first", "kept until the budget")];
    let first = prepare_tools(&first_tools).unwrap();
    // A set of many small values: light on the wire, heavy kept. Under the
    // per-set cap, so a run of them fills the table's weight long before
    // its count.
    let enum_tool = |name: &str| -> Tool {
        serde_json::from_value(json!({"type": "function", "function": {"name": name,
            "parameters": {"type": "object", "properties": {"code": {"type": "integer", "enum": (0..4000).collect::<Vec<_>>()}}}}}))
        .unwrap()
    };
    let one = [enum_tool("weight_probe")];
    let weight = set_weight(
        serde_json::to_vec(&one).unwrap().len(),
        &one.iter()
            .map(|t| serde_json::to_value(t).unwrap())
            .collect::<Vec<_>>(),
    );
    assert!(weight < PREPARED_TOOLS_MAX_SET_WEIGHT, "{weight}");
    let sets = PREPARED_TOOLS_KEPT_WEIGHT / weight + 2;
    assert!(
        sets < PREPARED_TOOLS_KEPT,
        "{sets} sets of {weight} fill the weight before the count"
    );
    for i in 0..sets {
        prepare_tools(&[enum_tool(&format!("weight_{i}"))]).unwrap();
    }
    let kept = PREPARED_TOOLS.read();
    assert!(
        kept.weight <= PREPARED_TOOLS_KEPT_WEIGHT,
        "the kept sets stay within their weight ({})",
        kept.weight
    );
    drop(kept);
    assert!(
        !Arc::ptr_eq(&first, &prepare_tools(&first_tools).unwrap()),
        "a set the weight's restart dropped is prepared again"
    );
    // Heavy kept, light on the wire: 12,000 one-digit numbers serialize to
    // about 24 KiB, under the key cap, and weigh past the per-set cap.
    let heavy = [serde_json::from_value::<Tool>(json!({"type": "function", "function": {"name": "heavy",
        "parameters": {"type": "object", "properties": {"code": {"type": "integer", "enum": (0..12000).map(|i| i % 10).collect::<Vec<_>>()}}}}})).unwrap()];
    let heavy_key = serde_json::to_vec(&heavy).unwrap().len();
    let heavy_weight = set_weight(
        heavy_key,
        &heavy
            .iter()
            .map(|t| serde_json::to_value(t).unwrap())
            .collect::<Vec<_>>(),
    );
    assert!(
        heavy_key <= PREPARED_TOOLS_MAX_KEY_BYTES && heavy_weight > PREPARED_TOOLS_MAX_SET_WEIGHT,
        "{heavy_key} bytes, {heavy_weight} weight"
    );
    assert!(
        !Arc::ptr_eq(
            &prepare_tools(&heavy).unwrap(),
            &prepare_tools(&heavy).unwrap()
        ),
        "a set above the weight cap is prepared for its request alone"
    );
}

fn declared(function: Value) -> Tool {
    serde_json::from_value(json!({"type": "function", "function": function})).unwrap()
}

/// A tool declared without a schema (the field absent or `null`) reaches
/// the template with `"parameters": null` in the schema's place, as the
/// engine's own template reads the declaration; a client's own `{}`
/// stays `{}`. The kept sets tell the two apart: the serialized key of a
/// schema-less tool leaves the field out, the key of a `{}` tool carries it.
#[test]
fn a_tool_without_a_schema_reaches_the_template_with_a_null_schema_in_its_place() {
    let _serial = SERIAL.lock().unwrap();
    let without = [
        declared(json!({"name": "schema_less_absent", "description": "no schema given"})),
        declared(json!({"name": "schema_less_null", "parameters": null, "strict": false})),
    ];
    let prepared = prepare_tools(&without).unwrap();
    assert_eq!(
        prepared.tools(),
        [
            json!({"type": "function", "function": {
                "name": "schema_less_absent", "description": "no schema given", "parameters": null}}),
            json!({"type": "function", "function": {
                "name": "schema_less_null", "parameters": null, "strict": false}}),
        ],
        "the schema's place is after the description, or after the name, before the rest"
    );
    let keys: Vec<Vec<&str>> = prepared
        .tools()
        .iter()
        .map(|tool| {
            tool["function"]
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect()
        })
        .collect();
    assert_eq!(
        keys,
        [
            vec!["name", "description", "parameters"],
            vec!["name", "parameters", "strict"]
        ]
    );

    assert!(
        Arc::ptr_eq(
            &prepare_tools(&without[..1]).unwrap(),
            &prepare_tools(&without[..1]).unwrap()
        ),
        "a schema-less set is kept and shared like any other"
    );
    let with_empty = [declared(json!({"name": "schema_less_absent",
        "description": "no schema given", "parameters": {}}))];
    let empty = prepare_tools(&with_empty).unwrap();
    assert_eq!(
        empty.tools()[0]["function"]["parameters"],
        json!({}),
        "a client's own empty schema renders as it was given"
    );
    assert!(
        !Arc::ptr_eq(&prepare_tools(&without[..1]).unwrap(), &empty),
        "a schema-less tool and the same tool with an empty schema are kept apart"
    );
}
