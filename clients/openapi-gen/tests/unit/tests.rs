use std::collections::BTreeSet;

use serde_json::json;

use super::*;

fn document_value() -> Value {
    serde_json::to_value(document().expect("the document builds")).expect("it serializes")
}

fn listed_routes() -> BTreeSet<(String, String)> {
    CONTROL_PLANE_ROUTES
        .iter()
        .chain(RL_CONTROL_PLANE_ROUTES)
        .map(|(method, path)| (method.to_lowercase(), path.to_string()))
        .collect()
}

#[test]
fn every_control_plane_route_has_an_operation_that_takes_the_bearer() {
    let doc = document_value();

    for (method, path) in listed_routes() {
        let operation = &doc["paths"][&path][&method];
        assert!(operation.is_object(), "{method} {path} has no operation");
        assert_eq!(
            operation["security"],
            json!([{ BEARER_SCHEME: [] }]),
            "{method} {path} does not take the bearer"
        );
    }
    assert_eq!(
        doc["components"]["securitySchemes"][BEARER_SCHEME]["scheme"],
        json!("bearer")
    );
}

#[test]
fn only_the_listed_routes_take_the_bearer() {
    let doc = document_value();
    let listed = listed_routes();

    for (path, item) in doc["paths"].as_object().expect("the paths") {
        for (method, operation) in item.as_object().expect("a path item") {
            let takes_bearer = operation.get("security").is_some();
            assert_eq!(
                takes_bearer,
                listed.contains(&(method.clone(), path.clone())),
                "{method} {path}: takes the bearer {takes_bearer}, listed otherwise"
            );
        }
    }
}

#[test]
fn the_worker_routes_take_the_types_the_gateway_reads() {
    let doc = document_value();
    let item = &doc["paths"]["/workers/{worker_id}"];
    let body_schema = |method: &str| {
        item[method]["requestBody"]["content"]["application/json"]["schema"]["$ref"].clone()
    };

    assert_eq!(body_schema("put"), json!("#/components/schemas/WorkerSpec"));
    assert_eq!(
        body_schema("patch"),
        json!("#/components/schemas/WorkerUpdateRequest")
    );
    assert_eq!(
        item["delete"]["responses"]["4XX"]["content"]["application/json"]["schema"]["$ref"],
        json!("#/components/schemas/WorkerErrorResponse")
    );
    assert_eq!(
        doc["paths"]["/workers"]["get"]["parameters"][0]["name"],
        json!("model")
    );
}

#[test]
fn the_input_items_list_names_its_paging_parameters() {
    let doc = document_value();
    let parameters: Vec<_> = doc["paths"]["/v1/responses/{response_id}/input_items"]["get"]
        ["parameters"]
        .as_array()
        .expect("the parameters")
        .iter()
        .map(|parameter| {
            (
                parameter["name"].as_str().unwrap_or(""),
                parameter["in"].as_str().unwrap_or(""),
                parameter["required"].as_bool(),
                parameter["schema"]["type"].as_str().unwrap_or(""),
                parameter["schema"]["items"]["type"].as_str(),
            )
        })
        .collect();
    assert_eq!(
        parameters,
        vec![
            ("response_id", "path", Some(true), "string", None),
            ("limit", "query", Some(false), "integer", None),
            ("order", "query", Some(false), "string", None),
            ("after", "query", Some(false), "string", None),
            ("include", "query", Some(false), "array", Some("string")),
        ]
    );
}

#[test]
fn the_rate_limit_document_write_declares_its_precondition_header() {
    let doc = document_value();
    let parameters = doc["paths"]["/rate-limits/tenants"]["put"]["parameters"]
        .as_array()
        .expect("the write's parameters")
        .clone();
    let if_match = parameters
        .iter()
        .find(|parameter| parameter["name"] == json!("If-Match"))
        .expect("the If-Match parameter");
    assert_eq!(if_match["in"], json!("header"));
    assert_eq!(if_match["required"], json!(true));
    assert_eq!(if_match["schema"]["type"], json!("string"));
    assert!(
        parameters
            .iter()
            .any(|parameter| parameter["name"] == json!("dry_run")),
        "the dry run stays declared"
    );
}

#[test]
fn the_module_and_tokenizer_routes_refuse_in_the_worker_envelope() {
    let doc = document_value();
    let envelope = json!("#/components/schemas/WorkerErrorResponse");
    for (path, method, status) in [
        ("/v1/tokenizers", "post", "409"),
        ("/v1/tokenizers", "post", "4XX"),
        ("/v1/tokenizers/{tokenizer_id}", "get", "4XX"),
        ("/v1/tokenizers/{tokenizer_id}", "delete", "4XX"),
        ("/v1/tokenizers/{tokenizer_id}/status", "get", "4XX"),
        ("/wasm", "post", "4XX"),
        ("/wasm/{module_uuid}", "delete", "4XX"),
    ] {
        assert_eq!(
            doc["paths"][path][method]["responses"][status]["content"]["application/json"]
                ["schema"]["$ref"],
            envelope,
            "{method} {path} {status}"
        );
    }
}

/// The route that answers a kept routing decision takes the request's id
/// as its one path parameter, answers an object that holds the decision,
/// and refuses in the control plane's envelope.
#[test]
fn the_kept_decision_route_names_its_parameter_and_refuses_in_the_worker_envelope() {
    let doc = document_value();
    let get = &doc["paths"]["/policy/explain/{request_id}"]["get"];
    assert_eq!(
        get["parameters"],
        json!([{"name": "request_id", "in": "path", "required": true, "schema": {"type": "string"}}]),
        "{get}"
    );
    let answer = &get["responses"]["200"]["content"]["application/json"]["schema"];
    assert_eq!(answer["required"], json!(["decision"]), "{get}");
    assert_eq!(
        get["responses"]["4XX"]["content"]["application/json"]["schema"]["$ref"],
        json!("#/components/schemas/WorkerErrorResponse"),
        "{get}"
    );
}

/// The WASM load's explicit 400 carries two bodies, the per-module results
/// and, for a refused body, the envelope; the explicit status hides the
/// 4XX range, so the document declares both under it.
#[test]
fn the_wasm_load_400_declares_the_per_module_shape_and_the_envelope() {
    let doc = document_value();
    let schema =
        &doc["paths"]["/wasm"]["post"]["responses"]["400"]["content"]["application/json"]["schema"];
    let shapes = schema["oneOf"]
        .as_array()
        .unwrap_or_else(|| panic!("the 400 carries no oneOf: {schema}"));
    assert_eq!(shapes.len(), 2, "{schema}");
    assert_eq!(shapes[0]["required"], json!(["modules"]));
    assert_eq!(
        shapes[1]["$ref"],
        json!("#/components/schemas/WorkerErrorResponse")
    );
}
