//! The committed `policyspec.schema.json` is generated from the spec types.
//! The first test fails when the two drift; `SMG_POLICY_SCHEMA_WRITE=1`
//! rewrites the file from the types instead. The second pins the schema's
//! shape: every section named, no unknown property anywhere, no `kind`.

use std::{collections::BTreeSet, fs, path::Path};

use serde_json::Value;
use smg_policy_controller::{spec::MAX_HEADERS, PolicySpec};

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
        "profiles",
        "plans",
        "bindings",
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
    assert_eq!(
        bare("ScoreSpec"),
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
    assert_eq!(
        bare("PickSpec"),
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
}
