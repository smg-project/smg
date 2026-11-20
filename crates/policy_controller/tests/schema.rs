//! The committed `policyspec.schema.json` is generated from the spec types.
//! The tests hold the artifact to the generated schema and check its
//! document shapes. `SMG_POLICY_SCHEMA_WRITE=1` writes the generated
//! schema to the file.

use std::{collections::BTreeSet, fs, path::Path};

use serde_json::Value;
use smg_policy_controller::{
    spec::{MAX_BLOCK_HASHES, MAX_ENGINES, MAX_HEADERS, MAX_LABELS, MAX_SIGNALS, MAX_ZONES},
    Catalog, PolicySpec,
};

fn schema_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("policyspec.schema.json")
}

#[test]
fn the_committed_schema_matches_the_spec_types() {
    let generated = PolicySpec::schema_json().expect("the schema serializes");
    let path = schema_path();
    if std::env::var_os("SMG_POLICY_SCHEMA_WRITE").is_some() {
        fs::write(&path, &generated).expect("the schema file is writable");
        return;
    }
    let committed = fs::read_to_string(&path).unwrap_or_default();
    assert!(
        committed == generated,
        "{} does not match the spec types: run \
         `SMG_POLICY_SCHEMA_WRITE=1 cargo test -p smg-policy-controller --test schema` \
         and commit the file",
        path.display()
    );
}

#[test]
fn the_committed_schema_admits_least_cost_with_terms_and_a_margin() {
    let committed = fs::read_to_string(schema_path()).expect("the committed schema");
    let schema: Value = serde_json::from_str(&committed).expect("a JSON schema");
    let defs = &schema["$defs"];
    let entries = defs["ScoreStage"]["oneOf"]
        .as_array()
        .expect("the scorer entries");
    let entry = entries
        .iter()
        .find(|entry| entry["properties"].get("least-cost").is_some())
        .expect("the least-cost entry");
    assert_eq!(entry["required"], serde_json::json!(["least-cost"]));
    assert_eq!(
        entry["properties"]["least-cost"]["$ref"],
        "#/$defs/LeastCostParams"
    );
    let params = &defs["LeastCostParams"];
    assert_eq!(params["required"], serde_json::json!(["terms"]));
    let terms = &params["properties"]["terms"];
    assert_eq!(
        terms["propertyNames"]["enum"],
        serde_json::json!([
            "candidate.in_flight",
            "load.active_kv_permille",
            "load.kv_usage_permille",
            "load.queued",
            "load.resident_tokens",
            "load.running",
            "load.waiting_uncached_tokens",
            "sent.requests",
            "sent.tokens",
        ])
    );
    let coefficient = &terms["additionalProperties"];
    assert_eq!(coefficient["type"], "number");
    assert_eq!(coefficient["minimum"], -1e15);
    assert_eq!(coefficient["maximum"], 1e15);
    assert_eq!(coefficient["not"]["const"], 0);
    assert_eq!(params["properties"]["margin"]["$ref"], "#/$defs/CostMargin");
    assert_eq!(defs["CostMargin"]["properties"]["absolute"]["default"], 0.0);
    assert_eq!(defs["CostMargin"]["properties"]["relative"]["default"], 0.0);
    assert_eq!(
        params["properties"]["unreported"]["anyOf"][0]["$ref"],
        "#/$defs/Unreported"
    );
    let unreported = &defs["Unreported"]["oneOf"];
    assert_eq!(unreported[1]["const"], "missing-as-zero");
    let over_least = &unreported[0]["properties"]["over-least"];
    assert_eq!(over_least["required"], serde_json::json!(["request_cost"]));
    assert_eq!(over_least["properties"]["request_cost"]["type"], "number");
    let bare = defs["ScoreSpec"]["anyOf"][0]["enum"]
        .as_array()
        .expect("the bare names");
    assert!(!bare.iter().any(|name| name == "least-cost"));
}

#[test]
fn the_committed_schema_admits_a_filter_by_its_name_and_by_its_name_with_parameters() {
    let committed = fs::read_to_string(schema_path()).expect("the committed schema");
    let schema: Value = serde_json::from_str(&committed).expect("a JSON schema");
    let defs = &schema["$defs"];
    assert_eq!(
        defs["ProfileSpec"]["properties"]["filter"]["items"]["$ref"],
        "#/$defs/FilterSpec"
    );
    // A bare name is any string: a host adds names. A name with parameters
    // is one key mapped to a map.
    assert_eq!(
        defs["FilterSpec"]["anyOf"],
        serde_json::json!([
            { "type": "string" },
            {
                "type": "object",
                "minProperties": 1,
                "maxProperties": 1,
                "additionalProperties": { "type": "object" }
            }
        ])
    );
}

/// Every object schema under `root` either lists its properties and refuses
/// the rest, or is a map (`additionalProperties` a schema), never open.
fn objects_are_closed(value: &Value, at: &str, open: &mut Vec<String>) {
    if let Some(map) = value.as_object() {
        let closed = map.get("type").and_then(Value::as_str) != Some("object")
            || !map.contains_key("properties")
            || map.get("additionalProperties") == Some(&Value::Bool(false));
        if !closed {
            open.push(at.to_string());
        }
        for (key, child) in map {
            objects_are_closed(child, &format!("{at}/{key}"), open);
        }
    } else if let Some(items) = value.as_array() {
        for (i, child) in items.iter().enumerate() {
            objects_are_closed(child, &format!("{at}/{i}"), open);
        }
    }
}

#[test]
fn the_schema_names_every_section_and_refuses_unknown_properties() {
    let schema = serde_json::to_value(PolicySpec::schema()).expect("a JSON value");
    let root = schema.as_object().expect("an object schema");
    let properties: BTreeSet<&str> = root["properties"]
        .as_object()
        .expect("root properties")
        .keys()
        .map(String::as_str)
        .collect();
    let expected: BTreeSet<&str> = [
        "apiVersion",
        "generation",
        "view",
        "index",
        "signals",
        "profiles",
        "plans",
        "bindings",
        "explain",
    ]
    .into_iter()
    .collect();
    assert_eq!(properties, expected);
    assert_eq!(root["additionalProperties"], Value::Bool(false));
    assert!(!properties.contains("kind"));
    let required: Vec<&str> = root["required"]
        .as_array()
        .expect("required")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert_eq!(required, ["apiVersion", "profiles"]);

    let mut open = Vec::new();
    objects_are_closed(&schema, "", &mut open);
    assert!(open.is_empty(), "open object schemas: {open:?}");

    let defs = schema["$defs"].as_object().expect("$defs");
    for name in [
        "ViewSpec",
        "IndexSpec",
        "PlansSpec",
        "ProfileSpec",
        "ScoreSpec",
        "ScoreStage",
        "PickSpec",
        "BindingSpec",
        "Match",
        "OverloadSpec",
        "SessionsSpec",
        "CanarySpec",
    ] {
        assert!(defs.contains_key(name), "{name} missing from $defs");
    }
    assert_eq!(
        defs["ViewSpec"]["properties"]["max_block_hashes"]["default"],
        256
    );
    assert_eq!(
        defs["SessionsSpec"]["properties"]["ttl_s"]["default"],
        14_400
    );
    // The bare names are the catalog minus the stages whose parameters have
    // no default, so a document written against the schema validates.
    let bare = |def: &str| -> Vec<String> {
        defs[def]["anyOf"][0]["enum"]
            .as_array()
            .expect("the bare names")
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect()
    };
    // Read from the stock catalog, the one the validator resolves a name in.
    // The catalog's pickers are a table of its own, so a picker in one and
    // not in the other fails here. Its scorers are the list the schema
    // reads, so the literals are what pins a published spelling.
    let stock = Catalog::builtin();
    let scorers: Vec<&str> = stock
        .scorers()
        .filter(|name| !matches!(*name, "resident-work" | "least-cost"))
        .collect();
    assert_eq!(bare("ScoreSpec"), scorers);
    assert_eq!(
        scorers,
        [
            "least-load",
            "least-requests",
            "fewest-sessions",
            "session-affinity",
            "prefix-affinity",
            "kv-pressure",
            "warmup-slice",
            "bucket",
            "consistent-hash",
        ]
    );
    let pickers: Vec<&str> = stock
        .pickers()
        .filter(|name| *name != "best-with-temperature")
        .collect();
    assert_eq!(bare("PickSpec"), pickers);
    assert_eq!(
        pickers,
        [
            "best",
            "round-robin",
            "random",
            "power-of-two",
            "count-first",
            "power-of-two-count-first",
            "passthrough",
            "manual",
        ]
    );
    // The ranges the validator enforces are the schema's too, so an editor
    // refuses on shape what apply would refuse.
    let field = |def: &str, name: &str| defs[def]["properties"][name].clone();
    assert_eq!(field("CanarySpec", "percent")["maximum"], 100);
    assert_eq!(field("CanarySpec", "percent")["minimum"], 1);
    assert_eq!(field("SessionsSpec", "inflight_cap")["minimum"], 1);
    assert_eq!(field("OverloadSpec", "waiting_requests")["minimum"], 1);
    assert_eq!(field("OverloadSpec", "token_usage")["exclusiveMinimum"], 0);
    assert_eq!(field("OverloadSpec", "token_usage")["maximum"], 1.0);
    assert_eq!(field("Weight", "weight")["exclusiveMinimum"], 0);
    assert_eq!(field("PrefixAffinityParams", "threshold")["maximum"], 1.0);
    assert_eq!(field("BucketParams", "rel_threshold")["minimum"], 1.0);
    assert_eq!(field("IndexSpec", "ttl_s")["minimum"], 1);
    assert_eq!(field("ViewSpec", "block_size")["minimum"], 1);
    assert_eq!(field("ViewSpec", "headers")["maxItems"], MAX_HEADERS);
    assert_eq!(field("ViewSpec", "labels")["maxItems"], MAX_LABELS);
    assert_eq!(field("ViewSpec", "zones")["maxItems"], MAX_ZONES);
    assert_eq!(field("ViewSpec", "engines")["maxItems"], MAX_ENGINES);
    assert_eq!(
        field("ViewSpec", "max_block_hashes")["maximum"],
        MAX_BLOCK_HASHES
    );
    assert_eq!(field("SignalSpec", "max_age_ms")["minimum"], 1);
    assert_eq!(field("SignalSpec", "max_age_ms")["maximum"], 3_599_999);
    // A unit's length and its alphabet, as the validator takes them.
    assert_eq!(field("SignalSpec", "unit")["minLength"], 1);
    assert_eq!(field("SignalSpec", "unit")["maxLength"], 31);
    assert_eq!(field("SignalSpec", "unit")["pattern"], "^[!-~]*$");
    // A signal's name and the count of signals, as the validator takes them.
    let signals = &root["properties"]["signals"];
    assert_eq!(signals["maxProperties"], MAX_SIGNALS);
    assert_eq!(
        signals["propertyNames"]["pattern"],
        "^[a-z][a-z0-9_]{0,62}$"
    );
}
