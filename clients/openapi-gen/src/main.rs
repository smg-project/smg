use std::{collections::BTreeMap, io::Write};

use schemars::{schema_for, Schema};
use serde::Serialize;
use serde_json::Value;

// ============================================================================
// OpenAPI 3.1 document structure (minimal, just what we need)
// ============================================================================

#[derive(Serialize)]
struct OpenApiDoc {
    openapi: String,
    info: Info,
    paths: BTreeMap<String, PathItem>,
    components: Components,
}

#[derive(Serialize)]
struct Info {
    title: String,
    version: String,
    description: String,
}

#[derive(Serialize)]
struct Components {
    schemas: BTreeMap<String, Value>,
}

#[derive(Default, Serialize)]
struct PathItem {
    #[serde(skip_serializing_if = "Option::is_none")]
    get: Option<Operation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    post: Option<Operation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    put: Option<Operation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delete: Option<Operation>,
}

#[derive(Serialize)]
struct Operation {
    #[serde(rename = "operationId")]
    operation_id: String,
    summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Vec<Parameter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "requestBody")]
    request_body: Option<RequestBody>,
    responses: BTreeMap<String, Response>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    deprecated: bool,
}

impl Operation {
    fn deprecated(mut self) -> Self {
        self.deprecated = true;
        self
    }
}

#[derive(Clone, Serialize)]
struct Parameter {
    name: String,
    #[serde(rename = "in")]
    location: String,
    required: bool,
    schema: ParameterSchema,
}

#[derive(Clone, Serialize)]
struct ParameterSchema {
    #[serde(rename = "type")]
    schema_type: String,
}

#[derive(Serialize)]
struct RequestBody {
    required: bool,
    content: BTreeMap<String, MediaType>,
}

#[derive(Serialize)]
struct MediaType {
    schema: Value,
}

#[derive(Serialize)]
struct Response {
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<BTreeMap<String, MediaType>>,
}

// ============================================================================
// Schema collection
// ============================================================================

/// Post-process a JSON value tree to fix schemars output for OpenAPI compatibility:
/// 1. Rewrite `#/$defs/X` → `#/components/schemas/X`
/// 2. Replace boolean `true` in `anyOf`/`oneOf`/`allOf` arrays with `{}` (empty = any)
fn fixup_schema(value: &mut Value) {
    match value {
        Value::String(s) => {
            if let Some(rest) = s.strip_prefix("#/$defs/") {
                *s = format!("#/components/schemas/{rest}");
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                fixup_schema(item);
            }
        }
        Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                if matches!(k.as_str(), "anyOf" | "oneOf" | "allOf") {
                    if let Value::Array(arr) = v {
                        for item in arr.iter_mut() {
                            if *item == Value::Bool(true) {
                                *item = Value::Object(serde_json::Map::new());
                            } else {
                                fixup_schema(item);
                            }
                        }
                        continue;
                    }
                }
                fixup_schema(v);
            }
        }
        _ => {}
    }
}

/// Collect a root schema and all its definitions into the schemas map.
/// Returns the top-level schema name.
///
/// schemars 1.0 returns a single `Schema` (a JSON object) holding the root
/// schema inline, plus `$defs` for referenced subschemas and a `$schema`
/// meta-schema key.
fn collect_schema(root: Schema, schemas: &mut BTreeMap<String, Value>) -> anyhow::Result<String> {
    let mut root_value = root.to_value();
    let Value::Object(map) = &mut root_value else {
        anyhow::bail!("expected schema object, got {root_value}");
    };

    let title = match map.get("title") {
        Some(Value::String(t)) => t.clone(),
        _ => "Unknown".to_string(),
    };

    // Pull out the referenced subschemas and the meta-schema key so only the
    // root schema body remains.
    let defs = map.remove("$defs");
    map.remove("$schema");

    // Add all definitions (sub-schemas referenced by the root)
    if let Some(Value::Object(defs)) = defs {
        for (name, mut value) in defs {
            fixup_schema(&mut value);
            schemas.insert(name, value);
        }
    }

    // Add the root schema itself
    fixup_schema(&mut root_value);
    schemas.insert(title.clone(), root_value);

    Ok(title)
}

// ============================================================================
// Helper functions
// ============================================================================

fn schema_ref(name: &str) -> Value {
    serde_json::json!({ "$ref": format!("#/components/schemas/{name}") })
}

fn json_body(name: &str) -> RequestBody {
    let mut content = BTreeMap::new();
    content.insert(
        "application/json".to_string(),
        MediaType {
            schema: schema_ref(name),
        },
    );
    RequestBody {
        required: true,
        content,
    }
}

fn json_response(name: &str, desc: &str) -> BTreeMap<String, Response> {
    let mut responses = BTreeMap::new();
    let mut content = BTreeMap::new();
    content.insert(
        "application/json".to_string(),
        MediaType {
            schema: schema_ref(name),
        },
    );
    responses.insert(
        "200".to_string(),
        Response {
            description: desc.to_string(),
            content: Some(content),
        },
    );
    responses
}

fn no_content_response(desc: &str) -> BTreeMap<String, Response> {
    let mut responses = BTreeMap::new();
    responses.insert(
        "204".to_string(),
        Response {
            description: desc.to_string(),
            content: None,
        },
    );
    responses
}

fn path_param(name: &str) -> Parameter {
    Parameter {
        name: name.to_string(),
        location: "path".to_string(),
        required: true,
        schema: ParameterSchema {
            schema_type: "string".to_string(),
        },
    }
}

fn optional_query_param(name: &str) -> Parameter {
    Parameter {
        name: name.to_string(),
        location: "query".to_string(),
        required: false,
        schema: ParameterSchema {
            schema_type: "string".to_string(),
        },
    }
}

fn operation(
    op_id: &str,
    summary: &str,
    params: Option<Vec<Parameter>>,
    request_body: Option<RequestBody>,
    responses: BTreeMap<String, Response>,
) -> Operation {
    Operation {
        operation_id: op_id.to_string(),
        summary: summary.to_string(),
        parameters: params,
        request_body,
        responses,
        deprecated: false,
    }
}

fn post_endpoint(op_id: &str, summary: &str, req_name: &str, resp_name: &str) -> PathItem {
    PathItem {
        post: Some(operation(
            op_id,
            summary,
            None,
            Some(json_body(req_name)),
            json_response(resp_name, summary),
        )),
        ..PathItem::default()
    }
}

fn get_endpoint(op_id: &str, summary: &str, resp_name: &str) -> PathItem {
    PathItem {
        get: Some(operation(
            op_id,
            summary,
            None,
            None,
            json_response(resp_name, summary),
        )),
        ..PathItem::default()
    }
}

// ============================================================================
// Main: generate the OpenAPI spec
// ============================================================================

fn main() -> anyhow::Result<()> {
    let mut schemas = BTreeMap::new();
    let mut paths = BTreeMap::new();

    // ---- Chat Completions ----
    use openai_protocol::chat::*;
    let req_name = collect_schema(schema_for!(ChatCompletionRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(ChatCompletionResponse), &mut schemas)?;
    collect_schema(schema_for!(ChatCompletionStreamResponse), &mut schemas)?;
    paths.insert(
        "/v1/chat/completions".to_string(),
        post_endpoint(
            "createChatCompletion",
            "Create chat completion",
            &req_name,
            &resp_name,
        ),
    );

    // ---- Completions ----
    use openai_protocol::completion::*;
    let req_name = collect_schema(schema_for!(CompletionRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(CompletionResponse), &mut schemas)?;
    collect_schema(schema_for!(CompletionStreamResponse), &mut schemas)?;
    paths.insert(
        "/v1/completions".to_string(),
        post_endpoint(
            "createCompletion",
            "Create completion",
            &req_name,
            &resp_name,
        ),
    );

    // ---- Embeddings ----
    use openai_protocol::embedding::*;
    let req_name = collect_schema(schema_for!(EmbeddingRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(EmbeddingResponse), &mut schemas)?;
    paths.insert(
        "/v1/embeddings".to_string(),
        post_endpoint("createEmbedding", "Create embedding", &req_name, &resp_name),
    );

    // ---- Rerank ----
    use openai_protocol::rerank::*;
    let req_name = collect_schema(schema_for!(RerankRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(RerankResponse), &mut schemas)?;
    paths.insert(
        "/v1/rerank".to_string(),
        post_endpoint("createRerank", "Rerank documents", &req_name, &resp_name),
    );

    // ---- Messages (Anthropic) ----
    use openai_protocol::messages::*;
    let req_name = collect_schema(schema_for!(CreateMessageRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(Message), &mut schemas)?;
    collect_schema(schema_for!(MessageStreamEvent), &mut schemas)?;
    paths.insert(
        "/v1/messages".to_string(),
        post_endpoint(
            "createMessage",
            "Create message (Anthropic)",
            &req_name,
            &resp_name,
        ),
    );

    // ---- Responses API ----
    use openai_protocol::responses::*;
    let req_name = collect_schema(schema_for!(ResponsesRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(ResponsesResponse), &mut schemas)?;
    paths.insert(
        "/v1/responses".to_string(),
        post_endpoint("createResponse", "Create response", &req_name, &resp_name),
    );
    // GET + DELETE /v1/responses/{response_id}
    let response_id_param = vec![path_param("response_id")];
    paths.insert(
        "/v1/responses/{response_id}".to_string(),
        PathItem {
            get: Some(operation(
                "getResponse",
                "Get a response by ID",
                Some(response_id_param.clone()),
                None,
                json_response(&resp_name, "Get a response by ID"),
            )),
            delete: Some(operation(
                "deleteResponse",
                "Delete a response",
                Some(response_id_param.clone()),
                None,
                no_content_response("Response deleted"),
            )),
            ..PathItem::default()
        },
    );
    // POST /v1/responses/{response_id}/cancel
    paths.insert(
        "/v1/responses/{response_id}/cancel".to_string(),
        PathItem {
            post: Some(operation(
                "cancelResponse",
                "Cancel an in-progress response",
                Some(response_id_param.clone()),
                None,
                json_response(&resp_name, "Cancel an in-progress response"),
            )),
            ..PathItem::default()
        },
    );
    // GET /v1/responses/{response_id}/input_items — returns generic JSON list
    let mut input_items_content = BTreeMap::new();
    input_items_content.insert(
        "application/json".to_string(),
        MediaType {
            schema: serde_json::json!({
                "type": "object",
                "description": "Paginated list envelope with {object, data, first_id, last_id, has_more}"
            }),
        },
    );
    let mut input_items_responses = BTreeMap::new();
    input_items_responses.insert(
        "200".to_string(),
        Response {
            description: "List input items for a response".to_string(),
            content: Some(input_items_content),
        },
    );
    paths.insert(
        "/v1/responses/{response_id}/input_items".to_string(),
        PathItem {
            get: Some(operation(
                "listResponseInputItems",
                "List input items for a response",
                Some(response_id_param),
                None,
                input_items_responses,
            )),
            ..PathItem::default()
        },
    );

    // ---- Classify ----
    use openai_protocol::classify::*;
    let req_name = collect_schema(schema_for!(ClassifyRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(ClassifyResponse), &mut schemas)?;
    collect_schema(schema_for!(ClassifyData), &mut schemas)?;
    paths.insert(
        "/v1/classify".to_string(),
        post_endpoint("classify", "Classify text", &req_name, &resp_name),
    );

    // ---- Parser ----
    use openai_protocol::parser::*;
    let req_name = collect_schema(schema_for!(ParseFunctionCallRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(ParseFunctionCallResponse), &mut schemas)?;
    paths.insert(
        "/parse/function_call".to_string(),
        post_endpoint(
            "parseFunctionCall",
            "Parse function calls from model output",
            &req_name,
            &resp_name,
        ),
    );
    let req_name = collect_schema(schema_for!(SeparateReasoningRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(SeparateReasoningResponse), &mut schemas)?;
    paths.insert(
        "/parse/reasoning".to_string(),
        post_endpoint(
            "separateReasoning",
            "Separate reasoning from model output",
            &req_name,
            &resp_name,
        ),
    );

    // ---- Workers ----
    // Note: worker mutation endpoints return 202 Accepted with ad-hoc JSON
    // (not WorkerApiResponse), and list returns a different stats shape than
    // WorkerListResponse. We use inline schemas matching the actual server responses.
    use openai_protocol::worker::*;
    let worker_spec_name = collect_schema(schema_for!(WorkerSpec), &mut schemas)?;
    let worker_info_name = collect_schema(schema_for!(WorkerInfo), &mut schemas)?;
    let worker_update_name = collect_schema(schema_for!(WorkerUpdateRequest), &mut schemas)?;

    // Inline schema for 202 Accepted mutation responses
    let worker_accepted_schema = serde_json::json!({
        "type": "object",
        "description": "202 Accepted response with {status, worker_id, message/url/location}",
        "properties": {
            "status": {"type": "string"},
            "worker_id": {"type": "string"},
            "message": {"type": "string"},
            "url": {"type": "string"},
            "location": {"type": "string"}
        }
    });
    let worker_accepted_response = |desc: &str| -> BTreeMap<String, Response> {
        let mut responses = BTreeMap::new();
        let mut content = BTreeMap::new();
        content.insert(
            "application/json".to_string(),
            MediaType {
                schema: worker_accepted_schema.clone(),
            },
        );
        responses.insert(
            "202".to_string(),
            Response {
                description: desc.to_string(),
                content: Some(content),
            },
        );
        responses
    };

    // Inline schema for worker list response
    let worker_list_schema = serde_json::json!({
        "type": "object",
        "description": "Worker list with stats",
        "properties": {
            "workers": {
                "type": "array",
                "items": { "$ref": format!("#/components/schemas/{worker_info_name}") }
            },
            "total": {"type": "integer"},
            "stats": {
                "type": "object",
                "properties": {
                    "prefill_count": {"type": "integer"},
                    "decode_count": {"type": "integer"},
                    "regular_count": {"type": "integer"}
                }
            }
        }
    });
    let mut worker_list_content = BTreeMap::new();
    worker_list_content.insert(
        "application/json".to_string(),
        MediaType {
            schema: worker_list_schema,
        },
    );
    let mut worker_list_responses = BTreeMap::new();
    worker_list_responses.insert(
        "200".to_string(),
        Response {
            description: "List all workers".to_string(),
            content: Some(worker_list_content),
        },
    );

    // POST + GET /workers
    paths.insert(
        "/workers".to_string(),
        PathItem {
            get: Some(operation(
                "listWorkers",
                "List all workers",
                None,
                None,
                worker_list_responses,
            )),
            post: Some(operation(
                "createWorker",
                "Register a new worker",
                None,
                Some(json_body(&worker_spec_name)),
                worker_accepted_response("Worker creation accepted"),
            )),
            ..PathItem::default()
        },
    );
    // GET + PUT + DELETE /workers/{worker_id}
    let worker_id_param = vec![path_param("worker_id")];
    paths.insert(
        "/workers/{worker_id}".to_string(),
        PathItem {
            get: Some(operation(
                "getWorker",
                "Get worker by ID",
                Some(worker_id_param.clone()),
                None,
                json_response(&worker_info_name, "Get worker by ID"),
            )),
            put: Some(operation(
                "updateWorker",
                "Update a worker",
                Some(worker_id_param.clone()),
                Some(json_body(&worker_update_name)),
                worker_accepted_response("Worker update accepted"),
            )),
            delete: Some(operation(
                "deleteWorker",
                "Remove a worker",
                Some(worker_id_param),
                None,
                worker_accepted_response("Worker deletion accepted"),
            )),
            ..PathItem::default()
        },
    );

    // ---- Fleet load ----
    let worker_load_name = collect_schema(schema_for!(WorkerLoadResponse), &mut schemas)?;
    let loads_operation = |op_id: &str, summary: &str| {
        operation(
            op_id,
            summary,
            Some(vec![optional_query_param("model")]),
            None,
            json_response(&worker_load_name, summary),
        )
    };
    paths.insert(
        "/loads".to_string(),
        PathItem {
            get: Some(loads_operation(
                "getLoads",
                "Cached engine load for every worker, plus a fleet aggregate",
            )),
            ..PathItem::default()
        },
    );
    paths.insert(
        "/get_loads".to_string(),
        PathItem {
            get: Some(loads_operation("getLoadsLegacy", "Deprecated alias of /loads").deprecated()),
            ..PathItem::default()
        },
    );

    // ---- Generate (SGLang native) ----
    use openai_protocol::generate::*;
    let req_name = collect_schema(schema_for!(GenerateRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(GenerateResponse), &mut schemas)?;
    paths.insert(
        "/generate".to_string(),
        post_endpoint(
            "generate",
            "Generate (SGLang native)",
            &req_name,
            &resp_name,
        ),
    );

    // ---- Tokenize / Detokenize ----
    use openai_protocol::tokenize::*;
    let req_name = collect_schema(schema_for!(TokenizeRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(TokenizeResponse), &mut schemas)?;
    paths.insert(
        "/v1/tokenize".to_string(),
        post_endpoint("tokenize", "Tokenize text", &req_name, &resp_name),
    );

    let req_name = collect_schema(schema_for!(DetokenizeRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(DetokenizeResponse), &mut schemas)?;
    paths.insert(
        "/v1/detokenize".to_string(),
        post_endpoint("detokenize", "Detokenize tokens", &req_name, &resp_name),
    );

    // ---- Models ----
    use openai_protocol::models::ListModelsResponse as OpenAIListModelsResponse;
    let resp_name = collect_schema(schema_for!(OpenAIListModelsResponse), &mut schemas)?;
    paths.insert(
        "/v1/models".to_string(),
        get_endpoint("listModels", "List available models", &resp_name),
    );

    // ---- Shared types (not tied to a specific endpoint) ----
    use openai_protocol::common::{Detail, ErrorDetail, ErrorResponse};
    collect_schema(schema_for!(ErrorResponse), &mut schemas)?;
    collect_schema(schema_for!(ErrorDetail), &mut schemas)?;
    collect_schema(schema_for!(Detail), &mut schemas)?;

    // ---- Assemble OpenAPI document ----
    let doc = OpenApiDoc {
        openapi: "3.1.0".to_string(),
        info: Info {
            title: "SMG (Shepherd Model Gateway) API".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "OpenAI-compatible API with Anthropic Messages and SGLang native support"
                .to_string(),
        },
        paths,
        components: Components { schemas },
    };

    // Output path: first CLI arg or default
    let output_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "clients/openapi/smg-openapi.yaml".to_string());

    let yaml = serde_yaml_ng::to_string(&doc)?;

    if output_path == "-" {
        write!(std::io::stdout(), "{yaml}")?;
    } else {
        std::fs::write(&output_path, &yaml)?;
        writeln!(std::io::stderr(), "Wrote OpenAPI spec to {output_path}")?;
    }

    Ok(())
}
