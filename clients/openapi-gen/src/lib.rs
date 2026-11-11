//! The gateway's OpenAPI 3.1 document, built from the protocol crate's types.
//!
//! `document` is the whole document; the binary writes it as YAML. The
//! control-plane routes are listed as data (`CONTROL_PLANE_ROUTES`), so the
//! gateway's route-table test and this crate's document test read one list.

use std::collections::BTreeMap;

use schemars::{schema_for, Schema};
use serde::Serialize;
use serde_json::Value;

// ============================================================================
// OpenAPI 3.1 document structure (minimal, just what we need)
// ============================================================================

/// The document: serialize it as YAML or JSON.
#[derive(Serialize)]
pub struct OpenApiDoc {
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
    #[serde(rename = "securitySchemes")]
    security_schemes: BTreeMap<String, SecurityScheme>,
}

/// The credential a route takes: the control plane's bearer.
#[derive(Serialize)]
struct SecurityScheme {
    #[serde(rename = "type")]
    scheme_type: String,
    scheme: String,
    description: String,
}

/// The name of the bearer scheme in `components` and on the operations.
const BEARER_SCHEME: &str = "bearerAuth";

fn bearer_scheme() -> BTreeMap<String, SecurityScheme> {
    BTreeMap::from([(
        BEARER_SCHEME.to_string(),
        SecurityScheme {
            scheme_type: "http".to_string(),
            scheme: "bearer".to_string(),
            description: "A control-plane API key or JWT. The admin role reaches every \
                          method, the user role GET and HEAD."
                .to_string(),
        },
    )])
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
    patch: Option<Operation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delete: Option<Operation>,
}

impl PathItem {
    /// The operation of `method` (an HTTP method name in capitals), if any.
    fn operation_mut(&mut self, method: &str) -> Option<&mut Operation> {
        match method {
            "GET" => self.get.as_mut(),
            "POST" => self.post.as_mut(),
            "PUT" => self.put.as_mut(),
            "PATCH" => self.patch.as_mut(),
            "DELETE" => self.delete.as_mut(),
            _ => None,
        }
    }
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
    /// The credential the route takes: set on every control-plane route.
    #[serde(skip_serializing_if = "Option::is_none")]
    security: Option<Vec<BTreeMap<String, Vec<String>>>>,
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

fn json_content(name: &str) -> BTreeMap<String, MediaType> {
    inline_content(schema_ref(name))
}

/// A JSON media map over a schema written in place (a shape the protocol
/// crate does not define).
fn inline_content(schema: Value) -> BTreeMap<String, MediaType> {
    BTreeMap::from([("application/json".to_string(), MediaType { schema })])
}

fn json_body(name: &str) -> RequestBody {
    RequestBody {
        required: true,
        content: json_content(name),
    }
}

/// A body the route reads when given and defaults when absent.
fn optional_json_body(name: &str) -> RequestBody {
    RequestBody {
        required: false,
        content: json_content(name),
    }
}

fn json_response(name: &str, desc: &str) -> BTreeMap<String, Response> {
    status_response("200", name, desc)
}

/// One response, `status`, with the schema `name`.
fn status_response(status: &str, name: &str, desc: &str) -> BTreeMap<String, Response> {
    BTreeMap::from([(
        status.to_string(),
        Response {
            description: desc.to_string(),
            content: Some(json_content(name)),
        },
    )])
}

/// A 200 response with a schema written in place.
fn inline_response(desc: &str, schema: Value) -> BTreeMap<String, Response> {
    BTreeMap::from([(
        "200".to_string(),
        Response {
            description: desc.to_string(),
            content: Some(inline_content(schema)),
        },
    )])
}

/// `responses` plus the refusal envelope `name` on every 4xx status.
fn refused_with(
    mut responses: BTreeMap<String, Response>,
    name: &str,
) -> BTreeMap<String, Response> {
    responses.insert(
        "4XX".to_string(),
        Response {
            description: "The request was refused; the body says why".to_string(),
            content: Some(json_content(name)),
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
        security: None,
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

/// The control-plane routes (method, path) the gateway mounts behind its
/// control-plane authentication, as its route table registers them. The
/// gateway's own test reads this list against the route table; the test
/// below reads it against the document, so the two cannot drift apart unseen.
pub const CONTROL_PLANE_ROUTES: &[(&str, &str)] = &[
    ("GET", "/inflight"),
    ("GET", "/config"),
    ("POST", "/secrets/reload"),
    ("POST", "/flush_cache"),
    ("POST", "/start_profile"),
    ("POST", "/stop_profile"),
    ("POST", "/heap_profile"),
    ("GET", "/get_loads"),
    ("GET", "/policy/explain/{id}"),
    ("POST", "/parse/function_call"),
    ("POST", "/parse/reasoning"),
    ("POST", "/wasm"),
    ("GET", "/wasm"),
    ("DELETE", "/wasm/{module_uuid}"),
    ("POST", "/v1/tokenizers"),
    ("GET", "/v1/tokenizers"),
    ("GET", "/v1/tokenizers/{tokenizer_id}"),
    ("DELETE", "/v1/tokenizers/{tokenizer_id}"),
    ("GET", "/v1/tokenizers/{tokenizer_id}/status"),
    ("POST", "/workers"),
    ("GET", "/workers"),
    ("GET", "/workers/{worker_id}"),
    ("PUT", "/workers/{worker_id}"),
    ("PATCH", "/workers/{worker_id}"),
    ("DELETE", "/workers/{worker_id}"),
];

/// The RL control-plane routes, behind the same authentication, mounted only
/// with `--enable-rl`. The gateway's route-list test reads this list against
/// the RL crate's router; the test below reads it against the document.
pub const RL_CONTROL_PLANE_ROUTES: &[(&str, &str)] = &[
    ("GET", "/v1/rl/workers"),
    ("GET", "/v1/rl/workers/{worker_id}"),
    ("GET", "/v1/rl/workers/{worker_id}/engine/{path}"),
    ("POST", "/v1/rl/workers/{worker_id}/engine/{path}"),
    ("GET", "/v1/rl/engine/{path}"),
    ("POST", "/v1/rl/engine/{path}"),
];

/// The whole document. Fails when a listed control-plane route has no
/// operation to carry the bearer scheme.
pub fn document() -> anyhow::Result<OpenApiDoc> {
    let mut schemas = BTreeMap::new();
    let mut paths = BTreeMap::new();

    // ---- Shared types (not tied to a specific endpoint) ----
    use openai_protocol::common::{Detail, ErrorDetail, ErrorResponse};
    let error_name = collect_schema(schema_for!(ErrorResponse), &mut schemas)?;
    collect_schema(schema_for!(ErrorDetail), &mut schemas)?;
    collect_schema(schema_for!(Detail), &mut schemas)?;

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
    let req_name = collect_schema(schema_for!(CountMessageTokensRequest), &mut schemas)?;
    let resp_name = collect_schema(schema_for!(CountMessageTokensResponse), &mut schemas)?;
    paths.insert(
        "/v1/messages/count_tokens".to_string(),
        post_endpoint(
            "countMessageTokens",
            "Count message tokens (Anthropic)",
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
    let worker_error_name = collect_schema(schema_for!(WorkerErrorResponse), &mut schemas)?;

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
                "List the workers, or with format=manifest the file provider's manifest",
                Some(vec![
                    optional_query_param("model"),
                    optional_query_param("format"),
                    optional_query_param("origin"),
                ]),
                None,
                refused_with(worker_list_responses, &worker_error_name),
            )),
            post: Some(operation(
                "createWorker",
                "Register a new worker",
                None,
                Some(json_body(&worker_spec_name)),
                refused_with(
                    worker_accepted_response("Worker creation accepted"),
                    &worker_error_name,
                ),
            )),
            ..PathItem::default()
        },
    );
    // GET + PUT + PATCH + DELETE /workers/{worker_id}
    let worker_id_param = vec![path_param("worker_id")];
    paths.insert(
        "/workers/{worker_id}".to_string(),
        PathItem {
            post: None,
            get: Some(operation(
                "getWorker",
                "Get worker by ID",
                Some(worker_id_param.clone()),
                None,
                refused_with(
                    json_response(&worker_info_name, "Get worker by ID"),
                    &worker_error_name,
                ),
            )),
            put: Some(operation(
                "replaceWorker",
                "Replace a worker: its whole spec, re-registered",
                Some(worker_id_param.clone()),
                Some(json_body(&worker_spec_name)),
                refused_with(
                    worker_accepted_response("Worker replacement accepted"),
                    &worker_error_name,
                ),
            )),
            patch: Some(operation(
                "updateWorker",
                "Update a worker: priority, cost, labels, API key or health checks",
                Some(worker_id_param.clone()),
                Some(json_body(&worker_update_name)),
                refused_with(
                    worker_accepted_response("Worker update accepted"),
                    &worker_error_name,
                ),
            )),
            delete: Some(operation(
                "deleteWorker",
                "Remove a worker",
                Some(worker_id_param),
                None,
                refused_with(
                    worker_accepted_response("Worker deletion accepted"),
                    &worker_error_name,
                ),
            )),
        },
    );

    // ---- Fleet control (control plane) ----
    let flush_name = collect_schema(schema_for!(FlushCacheResult), &mut schemas)?;
    paths.insert(
        "/flush_cache".to_string(),
        PathItem {
            post: Some(operation(
                "flushCache",
                "Flush the KV cache of every worker",
                None,
                None,
                json_response(&flush_name, "The workers flushed and the ones that failed"),
            )),
            ..PathItem::default()
        },
    );
    let profile_name = collect_schema(schema_for!(ProfileResult), &mut schemas)?;
    let start_profile_name = collect_schema(schema_for!(StartProfileRequest), &mut schemas)?;
    let stop_profile_name = collect_schema(schema_for!(StopProfileRequest), &mut schemas)?;
    paths.insert(
        "/start_profile".to_string(),
        PathItem {
            post: Some(operation(
                "startProfile",
                "Start a profiling run on every worker, or on the one `url` names",
                None,
                Some(optional_json_body(&start_profile_name)),
                json_response(
                    &profile_name,
                    "The workers that started and the ones that failed",
                ),
            )),
            ..PathItem::default()
        },
    );
    paths.insert(
        "/stop_profile".to_string(),
        PathItem {
            post: Some(operation(
                "stopProfile",
                "Stop the profiling run on every worker, or on the one `url` names",
                None,
                Some(optional_json_body(&stop_profile_name)),
                json_response(
                    &profile_name,
                    "The workers that stopped and the ones that failed",
                ),
            )),
            ..PathItem::default()
        },
    );
    let mut heap_profile_responses = inline_response(
        "The profile was written",
        serde_json::json!({
            "type": "object",
            "description": "Where the heap profile of the gateway process was written",
            "properties": {
                "path": {"type": "string"},
                "prof_active": {"type": "boolean", "description": "Whether the allocator was sampling at the dump"}
            },
            "required": ["path", "prof_active"]
        }),
    );
    heap_profile_responses.insert(
        "default".to_string(),
        Response {
            description:
                "No profile: the build lacks the profiling feature (501), no directory is \
                          configured (404), the allocator is not sampling or the dump failed"
                    .to_string(),
            content: Some(json_content(&error_name)),
        },
    );
    paths.insert(
        "/heap_profile".to_string(),
        PathItem {
            post: Some(operation(
                "dumpHeapProfile",
                "Write a heap profile of the gateway process",
                None,
                None,
                heap_profile_responses,
            )),
            ..PathItem::default()
        },
    );
    paths.insert(
        "/inflight".to_string(),
        PathItem {
            get: Some(operation(
                "listInflightRequests",
                "The requests in flight on this gateway, by process-local id",
                None,
                None,
                inline_response(
                    "The in-flight snapshot",
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "instance_id": {"type": "string", "format": "uuid"},
                            "request_ids": {"type": "array", "items": {"type": "integer", "minimum": 0}}
                        },
                        "required": ["instance_id", "request_ids"]
                    }),
                ),
            )),
            ..PathItem::default()
        },
    );

    paths.insert(
        "/config".to_string(),
        PathItem {
            get: Some(operation(
                "getEffectiveConfig",
                "The effective configuration with every secret redacted: the admin role alone, mounted only with control-plane authentication",
                None,
                None,
                inline_response(
                    "The redacted configuration view: the build identity, the secret sources and the configuration",
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "build": {"type": "object"},
                            "secret_sources": {"type": "object"},
                            "config": {"type": "object"}
                        },
                        "required": ["build", "secret_sources", "config"]
                    }),
                ),
            )),
            ..PathItem::default()
        },
    );

    paths.insert(
        "/secrets/reload".to_string(),
        PathItem {
            post: Some(operation(
                "reloadSecrets",
                "Read the key files the start read again and put their keys in force, for the requests the gateway accepts; the shared key it sends upstream to its workers keeps the start's value until a restart. The admin role alone, mounted only with control-plane authentication",
                None,
                None,
                {
                    let mut responses = inline_response(
                        "The files read with their entry counts, and whether the serving and the control-plane key sets changed",
                        serde_json::json!({
                            "type": "object",
                            "properties": {
                                "files": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "setting": {"type": "string"},
                                            "path": {"type": "string"},
                                            "entries": {"type": "integer"}
                                        },
                                        "required": ["setting", "path", "entries"]
                                    }
                                },
                                "serving_changed": {"type": "boolean"},
                                "control_plane_changed": {"type": "boolean"}
                            },
                            "required": ["files", "serving_changed", "control_plane_changed"]
                        }),
                    );
                    responses.insert(
                        "409".to_string(),
                        Response {
                            description: "No key was read from a file at start; nothing to reload"
                                .to_string(),
                            content: None,
                        },
                    );
                    responses.insert(
                        "422".to_string(),
                        Response {
                            description: "A file could not be used (unreadable, unparsable, or a key set the start's rules refuse); every key in force stands. The body names the field and the path, never the content"
                                .to_string(),
                            content: None,
                        },
                    );
                    responses
                },
            )),
            ..PathItem::default()
        },
    );

    // ---- WASM modules (control plane) ----
    // The module types live in the WASM runtime crate, which this generator
    // does not build; their shape is written here as the gateway answers it.
    let wasm_descriptor = serde_json::json!({
        "type": "object",
        "description": "A module to load, and after the load its result",
        "properties": {
            "name": {"type": "string"},
            "file_path": {"type": "string"},
            "module_type": {"type": "string", "enum": ["Middleware"]},
            "attach_points": {"type": "array", "items": {"type": "object", "additionalProperties": true}},
            "add_result": {
                "description": "`{\"Success\": \"<uuid>\"}` or `{\"Error\": \"<reason>\"}`, after the load",
                "type": "object",
                "additionalProperties": true
            }
        },
        "required": ["name", "file_path", "module_type", "attach_points"]
    });
    let wasm_modules = |desc: &str| {
        serde_json::json!({
            "type": "object",
            "description": desc,
            "properties": {"modules": {"type": "array", "items": wasm_descriptor}},
            "required": ["modules"]
        })
    };
    let mut wasm_added = inline_response(
        "Every module loaded; each carries its uuid in `add_result`",
        wasm_modules("The modules with their results"),
    );
    wasm_added.insert(
        "400".to_string(),
        Response {
            description: "A module failed to load; each carries its result in `add_result`. A refused body answers the error envelope instead.".to_string(),
            content: Some(inline_content(serde_json::json!({
                "oneOf": [
                    wasm_modules("The modules with their results"),
                    schema_ref(&worker_error_name)
                ]
            }))),
        },
    );
    paths.insert(
        "/wasm".to_string(),
        PathItem {
            post: Some(operation(
                "addWasmModules",
                "Load WASM modules; 200 when every module loaded, 400 with the per-module result otherwise",
                None,
                Some(RequestBody {
                    required: true,
                    content: inline_content(wasm_modules("The modules to load")),
                }),
                refused_with(wasm_added, &worker_error_name),
            )),
            get: Some(operation(
                "listWasmModules",
                "The loaded WASM modules and their execution metrics",
                None,
                None,
                inline_response(
                    "The modules and the metrics",
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "modules": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "module_uuid": {"type": "string", "format": "uuid"},
                                        "module_meta": {"type": "object", "additionalProperties": true}
                                    },
                                    "required": ["module_uuid", "module_meta"]
                                }
                            },
                            "metrics": {
                                "type": "object",
                                "properties": {
                                    "total_executions": {"type": "integer", "minimum": 0},
                                    "successful_executions": {"type": "integer", "minimum": 0},
                                    "failed_executions": {"type": "integer", "minimum": 0},
                                    "total_execution_time_ms": {"type": "integer", "minimum": 0},
                                    "max_execution_time_ms": {"type": "integer", "minimum": 0},
                                    "average_execution_time_ms": {"type": "number"}
                                }
                            }
                        },
                        "required": ["modules", "metrics"]
                    }),
                ),
            )),
            ..PathItem::default()
        },
    );
    paths.insert(
        "/wasm/{module_uuid}".to_string(),
        PathItem {
            delete: Some(operation(
                "removeWasmModule",
                "Unload a WASM module by uuid",
                Some(vec![path_param("module_uuid")]),
                None,
                refused_with(
                    BTreeMap::from([(
                        "200".to_string(),
                        Response {
                            description: "The module was removed".to_string(),
                            content: None,
                        },
                    )]),
                    &worker_error_name,
                ),
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

    // GET /policy/explain/{id}: the policy engine's record of one traced decision
    let explain_worker = serde_json::json!({
        "worker_id": {"type": "string", "description": "The worker's id, sixteen hex digits"},
        "worker": {
            "type": ["string", "null"],
            "description": "The worker's address, credentials redacted; null once it left the pool"
        }
    });
    let explain_rows = |more: Value, required: Value| {
        let mut properties = explain_worker.clone();
        for (name, schema) in more.as_object().into_iter().flatten() {
            properties[name.as_str()] = schema.clone();
        }
        serde_json::json!({"type": "object", "properties": properties, "required": required})
    };
    let explain_target = explain_rows(
        serde_json::json!({
            "position": {"type": "integer", "minimum": 0},
            "dp_rank": {"type": "integer", "minimum": 0}
        }),
        serde_json::json!(["worker_id", "worker", "position", "dp_rank"]),
    );
    let mut explain_pair_target = explain_target.clone();
    explain_pair_target["properties"]["decode"] = explain_target;
    explain_pair_target["properties"]["decode"]["description"] =
        Value::from("The decode side of a pair; absent on a single-side target");
    let explain_record = serde_json::json!({
        "type": "object",
        "description": "The decision as the engine recorded it",
        "properties": {
            "decision_id": {"type": "integer", "minimum": 0},
            "at_unix_micros": {"type": "integer", "minimum": 0},
            "model": {"type": "string"},
            "phase": {"type": "string", "enum": ["single", "prefill", "decode", "encode"]},
            "profile": {
                "type": ["string", "null"],
                "description": "The profile that decided; null when no binding matched"
            },
            "outcome": {
                "type": "string",
                "enum": ["pick", "queue", "saturated", "flow_limit", "no_endpoints", "policy_declined", "deadline"]
            },
            "stage": {
                "type": "string",
                "description": "The picker or filter that settled the decision; binding for a miss"
            },
            "survivors": {"type": "integer", "minimum": 0},
            "pool_size": {"type": "integer", "minimum": 0},
            "attempt": {"type": "integer", "minimum": 0},
            "excluded": {"type": "integer", "minimum": 0},
            "seed": {"type": "integer", "minimum": 0},
            "generation": {"type": "integer", "minimum": 0},
            "top": {
                "type": "array",
                "description": "The best-scored candidates, best first",
                "items": explain_rows(
                    serde_json::json!({"score": {"type": "number"}}),
                    serde_json::json!(["worker_id", "worker", "score"])
                )
            },
            "targets": {"type": "array", "description": "The pick's targets in order", "items": explain_pair_target}
        },
        "required": [
            "decision_id", "at_unix_micros", "model", "phase", "profile", "outcome", "stage", "survivors",
            "pool_size", "attempt", "excluded", "seed", "generation", "top", "targets"
        ]
    });
    paths.insert(
        "/policy/explain/{id}".to_string(),
        PathItem {
            get: Some(operation(
                "explainPolicyDecision",
                "The policy engine's record of one traced decision, by its id",
                Some(vec![path_param("id")]),
                None,
                refused_with(
                    inline_response("The decision as the engine recorded it", explain_record),
                    &error_name,
                ),
            )),
            ..PathItem::default()
        },
    );

    // ---- RL control plane ----
    // Engine-native routes are proxied verbatim: `{path}` is the engine route
    // (it may contain slashes) and the body is whatever that route takes, so
    // both are free-form here. The per-worker proxy mirrors the engine's
    // status, hence the `default` response.
    use openai_protocol::rl::*;
    let rl_workers_name = collect_schema(schema_for!(RlWorkersResponse), &mut schemas)?;
    let rl_entry_name = collect_schema(schema_for!(RlWorkerEntry), &mut schemas)?;
    let rl_outcome_name = collect_schema(schema_for!(RlCallOutcome), &mut schemas)?;
    let rl_fanout_name = collect_schema(schema_for!(RlFanoutResponse), &mut schemas)?;
    let engine_body = || {
        let mut content = BTreeMap::new();
        content.insert(
            "application/json".to_string(),
            MediaType {
                schema: serde_json::json!({
                    "type": "object",
                    "description": "Engine-defined request body, forwarded verbatim",
                    "additionalProperties": true
                }),
            },
        );
        RequestBody {
            required: false,
            content,
        }
    };
    paths.insert(
        "/v1/rl/workers".to_string(),
        get_endpoint(
            "listRlWorkers",
            "List workers for RL control: engine, topology, health, weight version, capabilities",
            &rl_workers_name,
        ),
    );
    paths.insert(
        "/v1/rl/workers/{worker_id}".to_string(),
        PathItem {
            get: Some(operation(
                "getRlWorker",
                "Get one worker for RL control",
                Some(vec![path_param("worker_id")]),
                None,
                json_response(&rl_entry_name, "Get one worker for RL control"),
            )),
            ..PathItem::default()
        },
    );
    let rl_proxy = |op_id: &str, body: Option<RequestBody>| {
        let mut responses =
            json_response(&rl_outcome_name, "The engine's status and body, wrapped");
        responses.insert(
            "default".to_string(),
            Response {
                description: "The engine answered a non-2xx status, wrapped the same way"
                    .to_string(),
                content: Some(json_content(&rl_outcome_name)),
            },
        );
        operation(
            op_id,
            "Proxy one engine-native route to one worker",
            Some(vec![path_param("worker_id"), path_param("path")]),
            body,
            responses,
        )
    };
    paths.insert(
        "/v1/rl/workers/{worker_id}/engine/{path}".to_string(),
        PathItem {
            get: Some(rl_proxy("proxyRlEngineRouteGet", None)),
            post: Some(rl_proxy("proxyRlEngineRoute", Some(engine_body()))),
            ..PathItem::default()
        },
    );
    let rl_fanout = |op_id: &str, body: Option<RequestBody>| {
        let mut responses = json_response(&rl_fanout_name, "Every selected worker succeeded");
        responses.insert(
            "207".to_string(),
            Response {
                description: "At least one selected worker failed; see `failed`".to_string(),
                content: Some(json_content(&rl_fanout_name)),
            },
        );
        let selector = Parameter {
            name: "selector".to_string(),
            location: "query".to_string(),
            required: true,
            schema: ParameterSchema {
                schema_type: "string".to_string(),
            },
        };
        operation(
            op_id,
            "Fan one engine-native route out to every worker matching `selector`",
            Some(vec![path_param("path"), selector]),
            body,
            responses,
        )
    };
    paths.insert(
        "/v1/rl/engine/{path}".to_string(),
        PathItem {
            get: Some(rl_fanout("fanoutRlEngineRouteGet", None)),
            post: Some(rl_fanout("fanoutRlEngineRoute", Some(engine_body()))),
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

    // ---- Tokenizer management (control plane) ----
    let add_tokenizer_name = collect_schema(schema_for!(AddTokenizerRequest), &mut schemas)?;
    let tokenizer_added_name = collect_schema(schema_for!(AddTokenizerResponse), &mut schemas)?;
    let tokenizer_list_name = collect_schema(schema_for!(ListTokenizersResponse), &mut schemas)?;
    let tokenizer_info_name = collect_schema(schema_for!(TokenizerInfo), &mut schemas)?;
    let tokenizer_removed_name =
        collect_schema(schema_for!(RemoveTokenizerResponse), &mut schemas)?;
    let mut tokenizer_added = status_response(
        "202",
        &tokenizer_added_name,
        "The load was queued; `id` names the tokenizer, `/status` follows it",
    );
    tokenizer_added.extend(status_response(
        "409",
        &worker_error_name,
        "A tokenizer of that name exists; the error names its id",
    ));
    paths.insert(
        "/v1/tokenizers".to_string(),
        PathItem {
            post: Some(operation(
                "addTokenizer",
                "Load a tokenizer from a path or a hub id",
                None,
                Some(json_body(&add_tokenizer_name)),
                refused_with(tokenizer_added, &worker_error_name),
            )),
            get: Some(operation(
                "listTokenizers",
                "The loaded tokenizers",
                None,
                None,
                json_response(&tokenizer_list_name, "The loaded tokenizers"),
            )),
            ..PathItem::default()
        },
    );
    let tokenizer_id_param = vec![path_param("tokenizer_id")];
    paths.insert(
        "/v1/tokenizers/{tokenizer_id}".to_string(),
        PathItem {
            get: Some(operation(
                "getTokenizer",
                "One loaded tokenizer, by id or name",
                Some(tokenizer_id_param.clone()),
                None,
                refused_with(
                    json_response(&tokenizer_info_name, "The tokenizer"),
                    &worker_error_name,
                ),
            )),
            delete: Some(operation(
                "removeTokenizer",
                "Unload a tokenizer, by id or name",
                Some(tokenizer_id_param.clone()),
                None,
                refused_with(
                    json_response(&tokenizer_removed_name, "The tokenizer was removed"),
                    &worker_error_name,
                ),
            )),
            ..PathItem::default()
        },
    );
    paths.insert(
        "/v1/tokenizers/{tokenizer_id}/status".to_string(),
        PathItem {
            get: Some(operation(
                "getTokenizerStatus",
                "Where a queued tokenizer load stands: pending, processing, completed or failed",
                Some(tokenizer_id_param),
                None,
                refused_with(
                    json_response(&tokenizer_added_name, "The load's status"),
                    &worker_error_name,
                ),
            )),
            ..PathItem::default()
        },
    );

    // ---- Models ----
    use openai_protocol::models::ListModelsResponse as OpenAIListModelsResponse;
    let resp_name = collect_schema(schema_for!(OpenAIListModelsResponse), &mut schemas)?;
    paths.insert(
        "/v1/models".to_string(),
        get_endpoint("listModels", "List available models", &resp_name),
    );

    // ---- The control-plane routes take the bearer ----
    let bearer = vec![BTreeMap::from([(BEARER_SCHEME.to_string(), Vec::new())])];
    for (method, path) in CONTROL_PLANE_ROUTES.iter().chain(RL_CONTROL_PLANE_ROUTES) {
        let operation = paths
            .get_mut(*path)
            .and_then(|item: &mut PathItem| item.operation_mut(method))
            .ok_or_else(|| {
                anyhow::anyhow!("{method} {path} is a control-plane route without an operation")
            })?;
        operation.security = Some(bearer.clone());
    }

    // ---- Assemble OpenAPI document ----
    Ok(OpenApiDoc {
        openapi: "3.1.0".to_string(),
        info: Info {
            title: "SMG (Shepherd Model Gateway) API".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "OpenAI-compatible API with Anthropic Messages and SGLang native support"
                .to_string(),
        },
        paths,
        components: Components {
            schemas,
            security_schemes: bearer_scheme(),
        },
    })
}

#[cfg(test)]
mod tests {
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

    /// The WASM load's explicit 400 carries two bodies, the per-module results
    /// and, for a refused body, the envelope; the explicit status hides the
    /// 4XX range, so the document declares both under it.
    #[test]
    fn the_wasm_load_400_declares_the_per_module_shape_and_the_envelope() {
        let doc = document_value();
        let schema = &doc["paths"]["/wasm"]["post"]["responses"]["400"]["content"]
            ["application/json"]["schema"];
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
}
