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
/// stays `{}`. The kept sets tell the two apart: the key of a schema-less
/// tool marks the schema absent, the key of a `{}` tool carries it.
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

fn key_of(tools: &[Tool]) -> Vec<u8> {
    let mut key = Vec::new();
    write_key(&mut key, tools).unwrap();
    key
}

/// The key stands for the serialized tools: two sets share a key exactly
/// when they serialize to the same bytes. Each function below differs from
/// every other in one field, one value, one order or the place of one
/// boundary; a set is one of them, or two in a row.
#[test]
fn two_tool_sets_share_a_key_exactly_when_they_serialize_alike() {
    let functions = [
        json!({"name": "a"}),
        json!({"name": "b"}),
        json!({"name": "a", "description": ""}),
        json!({"name": "a", "description": "d"}),
        json!({"name": "ad", "description": ""}),
        json!({"name": "a", "strict": false}),
        json!({"name": "a", "strict": true}),
        json!({"name": "a", "description": "d", "strict": true}),
        json!({"name": "a", "parameters": {}}),
        json!({"name": "a", "parameters": {}, "strict": false}),
        json!({"name": "a", "parameters": []}),
        json!({"name": "a", "parameters": ""}),
        json!({"name": "a", "parameters": false}),
        json!({"name": "a", "parameters": true}),
        json!({"name": "a", "parameters": 0}),
        json!({"name": "a", "parameters": "0"}),
        json!({"name": "a", "parameters": 0.0}),
        json!({"name": "a", "parameters": -0.0}),
        json!({"name": "a", "parameters": -1}),
        json!({"name": "a", "parameters": 1}),
        json!({"name": "a", "parameters": 1.5}),
        json!({"name": "a", "parameters": [1]}),
        json!({"name": "a", "parameters": [1, 2]}),
        json!({"name": "a", "parameters": [12]}),
        json!({"name": "a", "parameters": [[1], [2]]}),
        json!({"name": "a", "parameters": [[1, 2]]}),
        json!({"name": "a", "parameters": [[1], 2]}),
        json!({"name": "a", "parameters": [[], []]}),
        json!({"name": "a", "parameters": [[[]]]}),
        json!({"name": "a", "parameters": [null]}),
        json!({"name": "a", "parameters": [false]}),
        json!({"name": "a", "parameters": ["ab", "c"]}),
        json!({"name": "a", "parameters": ["a", "bc"]}),
        json!({"name": "a", "parameters": ["abc"]}),
        json!({"name": "a", "parameters": {"x": 1, "y": 2}}),
        json!({"name": "a", "parameters": {"y": 2, "x": 1}}),
        json!({"name": "a", "parameters": {"x": {"y": 2}}}),
        json!({"name": "a", "parameters": {"x": 1}}),
        json!({"name": "a", "parameters": {"x": null}}),
        json!({"name": "a", "parameters": {"x": "null"}}),
        json!({"name": "a", "parameters": {"x": []}}),
        json!({"name": "a", "parameters": {"x": {}}}),
        json!({"name": "a", "response": {"x": 1}}),
        json!({"name": "a", "response": null}),
        json!({"name": "a", "response": 1, "result": 2}),
        json!({"name": "a", "result": 2, "response": 1}),
        json!({"name": "a", "description": "x".repeat(127)}),
        json!({"name": "a", "description": "x".repeat(128)}),
        json!({"name": "a", "description": "x".repeat(16_384)}),
    ];
    let mut sets: Vec<Vec<Tool>> = vec![Vec::new()];
    sets.extend(
        functions
            .iter()
            .map(|function| vec![declared(function.clone())]),
    );
    let (a, b) = (
        declared(functions[0].clone()),
        declared(functions[1].clone()),
    );
    sets.extend([
        vec![a.clone(), b.clone()],
        vec![b.clone(), a.clone()],
        vec![a.clone(), a],
        vec![b.clone(), b],
    ]);
    sets.push(vec![serde_json::from_value(
        json!({"type": "custom", "function": {"name": "a"}}),
    )
    .unwrap()]);

    let texts: Vec<Vec<u8>> = sets
        .iter()
        .map(|set| serde_json::to_vec(set).unwrap())
        .collect();
    let keys: Vec<Vec<u8>> = sets.iter().map(|set| key_of(set.as_slice())).collect();
    for (i, text) in texts.iter().enumerate() {
        for (j, other) in texts.iter().enumerate() {
            assert_eq!(
                text == other,
                i == j,
                "the list holds no set twice: {i} and {j}"
            );
            assert_eq!(
                keys[i] == keys[j],
                i == j,
                "the keys of {} and {}",
                String::from_utf8_lossy(text),
                String::from_utf8_lossy(other)
            );
        }
    }

    // Read from another text of the same tools, the key is the same: a
    // schema given as `null` is a schema not given.
    let absent = [declared(json!({"name": "a"}))];
    let null = [declared(json!({"name": "a", "parameters": null}))];
    assert_eq!(
        serde_json::to_vec(&absent).unwrap(),
        serde_json::to_vec(&null).unwrap()
    );
    assert_eq!(key_of(&absent), key_of(&null));
    assert_eq!(key_of(&absent), keys[1], "and it is the key of the list's");
}

/// The table's bounds were set by the serialized bytes and read the key: a
/// key is no longer than its set serialized, whatever the set is made of
/// (many numbers, arrays in arrays, many fields, texts at the lengths where
/// a length takes one more byte), so a set the table kept it still keeps.
#[test]
fn a_key_is_no_longer_than_its_tools_serialized() {
    let many_fields: Map<String, Value> = (0..300).map(|i| (format!("k{i}"), json!(i))).collect();
    let shapes = [
        json!({"name": "n"}),
        json!({"name": "n", "parameters": {"enum": (0..12_000).map(|i| i % 10).collect::<Vec<_>>()}}),
        json!({"name": "n", "parameters": (0..200).map(|_| vec![0; 128]).collect::<Vec<_>>()}),
        json!({"name": "n", "parameters": (0..200).map(|_| json!([])).collect::<Vec<_>>()}),
        json!({"name": "n", "parameters": many_fields}),
        json!({"name": "n", "parameters": ["x".repeat(127), "x".repeat(128), "x".repeat(16_383)]}),
        json!({"name": "n".repeat(200), "description": "d".repeat(20_000),
            "parameters": {("k".repeat(130)): "v".repeat(16_384)}}),
    ];
    for shape in shapes {
        let set = [declared(shape)];
        let (key, text) = (key_of(&set), serde_json::to_vec(&set).unwrap());
        assert!(
            text.len() <= PREPARED_TOOLS_MAX_KEY_BYTES && key.len() <= text.len(),
            "{} key bytes for {} serialized",
            key.len(),
            text.len()
        );
    }
}
