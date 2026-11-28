use std::sync::atomic::{AtomicUsize, Ordering};

use axum::http::StatusCode;
use openai_protocol::{
    model_card::ModelCard, responses::ResponseStatus, worker::HealthCheckConfig, UNKNOWN_MODEL_ID,
};
use serde_json::json;
use smg_data_connector::{
    ConditionalStore, ListParams, MemoryConversationItemStorage, MemoryConversationStorage,
    MemoryResponseStorage, NewConversation, ResponseId, ResponseStorageError, SortOrder,
    StoredResponse,
};

use super::*;
use crate::routers::grpc::{
    backend_client::BasicWorkerBuilder,
    common::responses::test_support::{completed_response, response_storage_over},
};

/// A chain ending in a record its client asked not to store is no
/// previous response (the loaders of both model families call this; the
/// item-reference resolver shares its predicate); a stored chain is
/// returned, an unknown id refused the same way: 404, the resource the
/// request names is absent.
#[tokio::test]
async fn an_unstored_record_is_no_previous_response() {
    let storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
    for (id, store) in [("resp_unstored", false), ("resp_stored", true)] {
        let mut response = StoredResponse::new(None);
        response.id = ResponseId::from(id);
        response.raw_response =
            json!({"id": id, "store": store, "status": "completed", "output": []});
        storage.store_response(response).await.unwrap();
    }

    let refused = previous_response_chain(&storage, "resp_unstored", None)
        .await
        .expect_err("an unstored record is no previous response");
    assert_eq!(refused.status(), StatusCode::NOT_FOUND);
    let chain = previous_response_chain(&storage, "resp_stored", None)
        .await
        .expect("a stored record is a previous response");
    assert_eq!(chain.len(), 1);
    let unknown = previous_response_chain(&storage, "resp_unknown", None)
        .await
        .expect_err("an unknown id is refused");
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
}

/// A store read one record at a time (the trait's own chain walk), counted.
struct CountedReads {
    inner: MemoryResponseStorage,
    reads: AtomicUsize,
}

response_storage_over! {
    CountedReads.inner;
    passes store_response, delete_response, list_identifier_responses, delete_identifier_responses;
    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.get_response(response_id).await
    }
}

/// A previous response the caller may not read is refused after one read
/// of the store; a chain is walked no further than the first such record.
#[tokio::test]
async fn a_previous_response_the_caller_may_not_read_is_refused_after_one_read() {
    let counted = Arc::new(CountedReads {
        inner: MemoryResponseStorage::new(),
        reads: AtomicUsize::new(0),
    });
    let storage: Arc<dyn ResponseStorage> = counted.clone();
    let mut previous = None;
    for id in ["a1", "a2", "a3", "b4"] {
        let mut response = StoredResponse::new(previous.replace(ResponseId::from(id)));
        response.id = ResponseId::from(id);
        response.tenant = Some(id[..1].to_string()); // the letter of its id
        storage.store_response(response).await.unwrap();
    }

    let mut loads = Vec::new();
    for (named, tenant) in [("a3", "a"), ("a3", "b"), ("b4", "b")] {
        let chain = previous_response_chain(&storage, named, Some(tenant)).await;
        let ids: Vec<&str> = chain.iter().flatten().map(|r| r.id.0.as_str()).collect();
        loads.push((
            chain.as_ref().err().map(|refused| refused.status()),
            ids.join(" "),
            counted.reads.swap(0, Ordering::SeqCst),
        ));
    }
    assert_eq!(
        loads,
        [
            (None, "a1 a2 a3".to_string(), 3),
            (Some(StatusCode::NOT_FOUND), String::new(), 1),
            (None, "b4".to_string(), 2),
        ]
    );
}

/// `background: true` with `store: false`, the public API's probe: the
/// result is written over the record the client polls and nothing is
/// linked to the conversation; a foreground `store: false` response
/// stores nothing at all.
#[tokio::test]
async fn an_unstored_background_response_finishes_in_its_record_and_the_history_keeps_nothing() {
    let conversation_storage: Arc<dyn ConversationStorage> =
        Arc::new(MemoryConversationStorage::new());
    let item_storage: Arc<dyn ConversationItemStorage> =
        Arc::new(MemoryConversationItemStorage::new());
    let response_storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
    let conversation = conversation_storage
        .create_conversation(NewConversation::default())
        .await
        .unwrap();
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "m",
        "input": "hi",
        "background": true,
        "store": false,
        "conversation": conversation.id.0,
        "max_output_tokens": 64
    }))
    .unwrap();
    background::store_record(
        &response_storage,
        &background::queued_response("resp_u", &request),
        &request,
        None,
        None,
    )
    .await
    .unwrap();

    persist_response_if_needed(
        conversation_storage.clone(),
        item_storage.clone(),
        response_storage.clone(),
        &completed_response("resp_u", &request),
        &request,
        None,
        None,
    )
    .await
    .expect("nothing is reported to the caller here");

    let stored = response_storage
        .get_response(&ResponseId::from("resp_u"))
        .await
        .unwrap()
        .expect("the record is polled");
    assert_eq!(
        stored.raw_response["status"], "completed",
        "{}",
        stored.raw_response
    );
    assert_eq!(stored.raw_response["store"], false);
    assert_eq!(
        stored.raw_response["output"][0]["content"][0]["text"],
        "hello"
    );
    let items = item_storage
        .list_items(
            &conversation.id,
            ListParams {
                limit: 10,
                order: SortOrder::Asc,
                after: None,
            },
        )
        .await
        .unwrap();
    assert!(
        items.is_empty(),
        "nothing is linked to the conversation: {items:?}"
    );

    let mut foreground = request;
    foreground.background = None;
    persist_response_if_needed(
        conversation_storage,
        item_storage,
        response_storage.clone(),
        &completed_response("resp_f", &foreground),
        &foreground,
        None,
        None,
    )
    .await
    .expect("nothing is reported to the caller here");
    assert!(response_storage
        .get_response(&ResponseId::from("resp_f"))
        .await
        .unwrap()
        .is_none());
}

/// A response store that is down: every call answers a storage error.
struct DownResponseStorage;

#[async_trait::async_trait]
impl ResponseStorage for DownResponseStorage {
    async fn store_response(
        &self,
        _response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        Err(ResponseStorageError::Unavailable(
            "connection refused".to_string(),
        ))
    }

    async fn get_response(
        &self,
        _response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        Err(ResponseStorageError::Unavailable(
            "connection refused".to_string(),
        ))
    }

    async fn delete_response(&self, _response_id: &ResponseId) -> Result<(), ResponseStorageError> {
        Err(ResponseStorageError::Unavailable(
            "connection refused".to_string(),
        ))
    }

    async fn list_identifier_responses(
        &self,
        _identifier: &str,
        _tenant: Option<&str>,
        _limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
        Err(ResponseStorageError::Unavailable(
            "connection refused".to_string(),
        ))
    }

    async fn delete_identifier_responses(
        &self,
        _identifier: &str,
        _tenant: Option<&str>,
    ) -> Result<usize, ResponseStorageError> {
        Err(ResponseStorageError::Unavailable(
            "connection refused".to_string(),
        ))
    }
}

/// A foreground response the client expects stored whose record could
/// not be written is reported to the caller with the store-class code; an
/// unstored request has nothing to report.
#[tokio::test]
async fn a_foreground_stored_response_whose_record_was_not_written_is_reported() {
    let conversation_storage: Arc<dyn ConversationStorage> =
        Arc::new(MemoryConversationStorage::new());
    let item_storage: Arc<dyn ConversationItemStorage> =
        Arc::new(MemoryConversationItemStorage::new());
    let down: Arc<dyn ResponseStorage> = Arc::new(DownResponseStorage);
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "m",
        "input": "hi",
        "max_output_tokens": 64
    }))
    .unwrap();
    assert!(stored_foreground(&request));

    let not_stored = persist_response_if_needed(
        conversation_storage.clone(),
        item_storage.clone(),
        down.clone(),
        &completed_response("resp_s", &request),
        &request,
        None,
        None,
    )
    .await
    .expect_err("the record was not written");
    assert!(not_stored.retryable, "{not_stored:?}");
    assert_eq!(not_stored.code(), "storage_unavailable");
    assert!(
        not_stored.message.contains("connection refused"),
        "{}",
        not_stored.message
    );
    let error = not_stored.error_object();
    assert_eq!(error["code"], "storage_unavailable");
    let answer = not_stored.into_response();
    assert_eq!(answer.status(), StatusCode::SERVICE_UNAVAILABLE);

    let unstorable = NotStored::new(false, "no output");
    assert_eq!(unstorable.code(), "store_response_failed");
    assert_eq!(
        unstorable.into_response().status(),
        StatusCode::INTERNAL_SERVER_ERROR
    );

    let mut unstored = request.clone();
    unstored.store = Some(false);
    assert!(!stored_foreground(&unstored));
    persist_response_if_needed(
        conversation_storage,
        item_storage,
        down,
        &completed_response("resp_n", &unstored),
        &unstored,
        None,
        None,
    )
    .await
    .expect("nothing to store, nothing to report");
}

/// A stored response gives its `temperature` and `top_p` to a `GET` of
/// its id as the answer itself gave them, not widened to 64 bits: the
/// record of a stored response, the queued record of a background one
/// and the record an unstored background run ends in.
#[tokio::test]
async fn a_get_of_a_stored_response_reads_the_sampling_numbers_unwidened() {
    use crate::endpoints::responses::get_response;

    let storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
    let read = |id: &'static str| {
        let storage = storage.clone();
        async move {
            let answer = get_response(&storage, None, id, None).await;
            let body = axum::body::to_bytes(answer.into_body(), 1 << 20).await;
            let read: Value = serde_json::from_slice(&body.unwrap()).unwrap();
            (read["temperature"].clone(), read["top_p"].clone())
        }
    };
    let exact = (json!(0.7), json!(0.9));
    let mut request = json!({
        "model": "m",
        "input": "hi",
        "temperature": 0.7,
        "top_p": 0.9
    });
    let stored: ResponsesRequest = serde_json::from_value(request.clone()).unwrap();
    persist_response_if_needed(
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
        storage.clone(),
        &completed_response("resp_g", &stored),
        &stored,
        None,
        None,
    )
    .await
    .expect("the record is written");
    assert_eq!(read("resp_g").await, exact);

    request["background"] = json!(true);
    request["store"] = json!(false);
    let unstored: ResponsesRequest = serde_json::from_value(request).unwrap();
    background::store_record(
        &storage,
        &background::queued_response("resp_q", &unstored),
        &unstored,
        None,
        None,
    )
    .await
    .expect("the queued record is written");
    assert_eq!(read("resp_q").await, exact);
    persist_response_if_needed(
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
        storage.clone(),
        &completed_response("resp_q", &unstored),
        &unstored,
        None,
        None,
    )
    .await
    .expect("the unstored end is written");
    assert_eq!(read("resp_q").await, exact);
}

/// A background response deleted while its work ran (its record is gone
/// when the result arrives): the result is not stored under the id again
/// and nothing of it reaches the conversation; the delete holds.
#[tokio::test]
async fn a_background_result_whose_record_was_deleted_is_not_stored() {
    let conversation_storage: Arc<dyn ConversationStorage> =
        Arc::new(MemoryConversationStorage::new());
    let item_storage: Arc<dyn ConversationItemStorage> =
        Arc::new(MemoryConversationItemStorage::new());
    let response_storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
    let conversation = conversation_storage
        .create_conversation(NewConversation::default())
        .await
        .unwrap();
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "m",
        "input": "hi",
        "background": true,
        "conversation": conversation.id.0,
        "max_output_tokens": 64
    }))
    .unwrap();
    background::store_record(
        &response_storage,
        &background::queued_response("resp_d", &request),
        &request,
        None,
        None,
    )
    .await
    .unwrap();
    response_storage
        .delete_response(&ResponseId::from("resp_d"))
        .await
        .unwrap();

    persist_response_if_needed(
        conversation_storage,
        item_storage.clone(),
        response_storage.clone(),
        &completed_response("resp_d", &request),
        &request,
        None,
        None,
    )
    .await
    .expect("a deleted record is not a store failure");

    assert!(
        response_storage
            .get_response(&ResponseId::from("resp_d"))
            .await
            .unwrap()
            .is_none(),
        "the deleted record is not re-created by the result"
    );
    let items = item_storage
        .list_items(
            &conversation.id,
            ListParams {
                limit: 10,
                order: SortOrder::Asc,
                after: None,
            },
        )
        .await
        .unwrap();
    assert!(
        items.is_empty(),
        "nothing is linked to the conversation: {items:?}"
    );
}

/// A store that takes the write and then answers a timeout: the call's
/// bound ran out after the store had accepted the write.
struct TimingOutAfterWriteStorage {
    inner: MemoryResponseStorage,
    timeouts_left: AtomicUsize,
    conditional_stores: AtomicUsize,
}

response_storage_over! {
    TimingOutAfterWriteStorage.inner;
    passes store_response, update_response_object_if_open, get_response, delete_response, list_identifier_responses, delete_identifier_responses;
    async fn store_response_if_open(
        &self,
        response: StoredResponse,
    ) -> Result<ConditionalStore, ResponseStorageError> {
        let outcome = self.inner.store_response_if_open(response).await?;
        self.conditional_stores.fetch_add(1, Ordering::SeqCst);
        let left = self.timeouts_left.load(Ordering::SeqCst);
        if left > 0 {
            self.timeouts_left.store(left - 1, Ordering::SeqCst);
            return Err(ResponseStorageError::Timeout(
                "store_response_if_open exceeded its bound".to_string(),
            ));
        }
        Ok(outcome)
    }
}

/// A background response's terminal persist retried after a timeout
/// whose write had landed: the retry finds the record closed with this
/// run's own result and links the items into the conversation, as the
/// first attempt would have had its answer arrived.
#[tokio::test(start_paused = true)]
async fn a_retried_terminal_persist_whose_first_write_landed_links_its_items() {
    let conversation_storage: Arc<dyn ConversationStorage> =
        Arc::new(MemoryConversationStorage::new());
    let item_storage: Arc<dyn ConversationItemStorage> =
        Arc::new(MemoryConversationItemStorage::new());
    let timing_out = Arc::new(TimingOutAfterWriteStorage {
        inner: MemoryResponseStorage::new(),
        timeouts_left: AtomicUsize::new(1),
        conditional_stores: AtomicUsize::new(0),
    });
    let response_storage: Arc<dyn ResponseStorage> = timing_out.clone();
    let conversation = conversation_storage
        .create_conversation(NewConversation::default())
        .await
        .unwrap();
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "m",
        "input": "hi",
        "background": true,
        "conversation": conversation.id.0,
        "max_output_tokens": 64
    }))
    .unwrap();
    background::store_record(
        &response_storage,
        &background::queued_response("resp_r", &request),
        &request,
        None,
        None,
    )
    .await
    .unwrap();

    persist_response_if_needed(
        conversation_storage,
        item_storage.clone(),
        response_storage.clone(),
        &completed_response("resp_r", &request),
        &request,
        None,
        None,
    )
    .await
    .expect("nothing is reported to the caller here");

    assert_eq!(
        timing_out.conditional_stores.load(Ordering::SeqCst),
        2,
        "the write whose answer was lost was retried once"
    );
    let stored = response_storage
        .get_response(&ResponseId::from("resp_r"))
        .await
        .unwrap()
        .expect("the record");
    assert_eq!(stored.raw_response["status"], "completed");
    let items = item_storage
        .list_items(
            &conversation.id,
            ListParams {
                limit: 10,
                order: SortOrder::Asc,
                after: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(
        items.len(),
        2,
        "the input and the output are linked into the conversation: {items:?}"
    );
}

fn registry_with_aliased_worker() -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = BasicWorkerBuilder::new("http://worker:8080")
        .model(ModelCard::new("canonical-model").with_alias("model-alias"))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    registry.register_or_replace(Arc::new(worker));
    registry
}

#[test]
fn worker_availability_accepts_alias_and_preserves_unknown_rejection() {
    let registry = registry_with_aliased_worker();

    assert!(validate_worker_availability(&registry, "canonical-model").is_none());
    assert!(validate_worker_availability(&registry, "model-alias").is_none());

    let response = validate_worker_availability(&registry, UNKNOWN_MODEL_ID)
        .expect("unknown model should remain rejected for Responses");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[test]
fn worker_availability_rejects_alias_once_its_worker_is_gone() {
    let registry = registry_with_aliased_worker();
    let worker_id = registry.get_id_by_url("http://worker:8080").unwrap();
    assert!(registry.remove(&worker_id).is_some());

    let response = validate_worker_availability(&registry, "model-alias")
        .expect("alias must stop resolving with no workers behind it");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

/// The public API's function-tool keys say how the API calls a tool, not
/// what the model calls, so the chat template's copy of a function tool,
/// alone or in a namespace, carries none of them.
#[test]
fn the_chat_templates_function_tools_leave_out_the_apis_tool_keys() {
    let function = serde_json::json!({
        "type": "function", "name": "lookup", "parameters": {},
        "async": true, "defer_loading": true,
        "output_schema": {"type": "object"}, "allowed_callers": ["direct"]
    });
    let request: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "input": "hi",
        "tools": [
            function.clone(),
            {"type": "namespace", "name": "weather", "description": "Weather tools",
             "tools": [function]}
        ]
    }))
    .unwrap();
    let tools = extract_tools_from_response_tools(request.tools.as_deref());
    assert_eq!(
        tools
            .iter()
            .map(|t| t.function.name.as_str())
            .collect::<Vec<_>>(),
        vec!["lookup", "weather.lookup"]
    );
    for tool in &tools {
        assert!(
            tool.function.extra.is_empty(),
            "{}: {:?}",
            tool.function.name,
            tool.function.extra
        );
    }
}

/// A tool only code may call (`allowed_callers` without `direct`) is
/// refused, whether a function or custom tool, alone or in a namespace,
/// and a tool the model may call directly is served.
#[test]
fn a_tool_without_a_direct_caller_is_refused() {
    let request = |tools: Value| -> ResponsesRequest {
        serde_json::from_value(json!({"model": "m", "input": "hi", "tools": tools})).unwrap()
    };
    let function = |callers: Value| json!({"type": "function", "name": "lookup", "parameters": {}, "allowed_callers": callers});
    let custom =
        |callers: Value| json!({"type": "custom", "name": "run", "allowed_callers": callers});
    for (tools, at) in [
        (json!([function(json!(["programmatic"]))]), "tools[0]"),
        (json!([function(json!([]))]), "tools[0]"),
        (
            json!([{"type": "mcp", "server_label": "docs", "server_url": "https://example.com/mcp",
                     "allowed_callers": ["programmatic"]}]),
            "tools[0]",
        ),
        (
            json!([{"type": "code_interpreter", "allowed_callers": ["programmatic"]}]),
            "tools[0]",
        ),
        (
            json!([{"type": "shell", "allowed_callers": ["programmatic"]}]),
            "tools[0]",
        ),
        (
            json!([{"type": "apply_patch", "allowed_callers": ["programmatic"]}]),
            "tools[0]",
        ),
        (
            json!([{"type": "function", "name": "a", "parameters": {}}, custom(json!(["programmatic"]))]),
            "tools[1]",
        ),
        (
            json!([{"type": "namespace", "name": "weather", "description": "Weather tools",
                    "tools": [function(json!(["direct"])), custom(json!(["programmatic"]))]}]),
            "tools[0].tools[1]",
        ),
    ] {
        let request = request(tools);
        let response = refuse_tools_without_a_direct_caller(request.tools.as_deref())
            .unwrap_or_else(|| panic!("{at} has no direct caller"));
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{at}");
    }
    for tools in [
        json!([function(json!(["direct"]))]),
        json!([function(json!(["direct", "programmatic"]))]),
        json!([function(Value::Null), custom(json!(["direct"]))]),
        json!([{"type": "function", "name": "a", "parameters": {}}, {"type": "custom", "name": "b"}]),
    ] {
        let request = request(tools.clone());
        assert!(
            refuse_tools_without_a_direct_caller(request.tools.as_deref()).is_none(),
            "{tools}"
        );
    }
    assert!(refuse_tools_without_a_direct_caller(None).is_none());
}

#[test]
fn namespace_function_identity_roundtrips_and_preserves_literal_names() {
    let mut request: ResponsesRequest = namespace_test_request();
    let tools = extract_tools_from_response_tools(request.tools.as_deref());
    assert_eq!(
        tools
            .iter()
            .map(|t| t.function.name.as_str())
            .collect::<Vec<_>>(),
        vec!["weather.lookup", "travel.lookup"]
    );
    for namespace in ["weather", "travel"] {
        assert_eq!(
            resolve_function_identity(request.tools.as_deref(), &format!("{namespace}.lookup")),
            ("lookup".into(), Some(namespace.into()))
        );
    }
    for name in ["lookup", "unknown.lookup"] {
        assert_eq!(
            resolve_function_identity(request.tools.as_deref(), name),
            (name.into(), None)
        );
    }
    request.tools.as_mut().unwrap().push(
        serde_json::from_value(
            serde_json::json!({"type":"function","name":"weather.lookup","parameters":{}}),
        )
        .unwrap(),
    );
    assert_eq!(
        resolve_function_identity(request.tools.as_deref(), "weather.lookup"),
        ("weather.lookup".into(), None)
    );
}

#[test]
fn a_namespaced_name_is_read_at_the_dot() {
    assert!(is_namespaced("weather.lookup", "weather", "lookup"));
    for (name, namespace, member) in [
        ("weatherx.lookup", "weather", "lookup"),
        ("weather.lookups", "weather", "lookup"),
        ("weather.lookup.now", "weather", "lookup"),
        ("weather", "weather", ""),
        ("lookup", "", "lookup"),
    ] {
        assert!(!is_namespaced(name, namespace, member), "{name}");
    }
}

#[test]
fn a_call_whose_arguments_are_no_json_value_is_incomplete_whatever_the_finish() {
    for finish in [
        Some("length"),
        Some("abort"),
        Some("stop"),
        Some("tool_calls"),
        None,
    ] {
        for arguments in ["{\"a\": \"b", "", "weather in Paris"] {
            assert_eq!(
                function_call_status(finish, arguments),
                "incomplete",
                "{finish:?} {arguments:?}"
            );
        }
        assert_eq!(
            function_call_status(finish, "{\"a\": 1}"),
            "completed",
            "{finish:?}"
        );
    }
    assert_eq!(function_call_status(Some("failed"), "{"), "in_progress");
    assert_eq!(
        response_status(Some("abort"), true),
        (ResponseStatus::Incomplete, None)
    );
    assert_eq!(
        response_status(Some("abort"), false),
        (ResponseStatus::Completed, None)
    );
    assert_eq!(
        response_status(Some("length"), false),
        (
            ResponseStatus::Incomplete,
            Some(IncompleteDetails {
                reason: IncompleteReason::MaxOutputTokens
            })
        )
    );
    assert_eq!(
        response_status(Some("error"), true),
        (ResponseStatus::Failed, None)
    );
}

#[test]
fn the_safety_net_allows_the_limits_generations_and_the_answer_on_the_last_result() {
    assert!(!iteration_exceeds_safety_limit(
        DEFAULT_MAX_ITERATIONS,
        false
    ));
    assert!(!iteration_exceeds_safety_limit(
        DEFAULT_MAX_ITERATIONS + 1,
        false
    ));
    assert!(iteration_exceeds_safety_limit(
        DEFAULT_MAX_ITERATIONS + 2,
        false
    ));
    assert!(!iteration_exceeds_safety_limit(
        DEFAULT_MAX_ITERATIONS + 2,
        true
    ));
}

#[test]
fn a_responses_usage_is_the_sum_of_its_model_calls() {
    let counts = |usage: &ResponseUsage| {
        (
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.input_tokens_details.cached_tokens,
            usage.output_tokens_details.reasoning_tokens,
        )
    };
    let mut total = None;
    add_model_call_usage(&mut total, None);
    assert!(total.is_none());
    // One call: what a response served by a single call reports.
    let first = Usage::from_counts(11, 5).with_cached_tokens(4);
    add_model_call_usage(&mut total, Some(&first));
    assert_eq!(
        total.as_ref().map(counts),
        Some(counts(&ResponseUsage::from(&first)))
    );
    let second = Usage::from_counts(23, 7).with_reasoning_tokens(3);
    add_model_call_usage(&mut total, Some(&second));
    add_model_call_usage(&mut total, None);
    assert_eq!(total.as_ref().map(counts), Some((34, 12, 46, 4, 3)));
}

#[test]
fn the_tool_call_limit_is_the_users_figure_under_the_safety_limit() {
    assert_eq!(tool_call_limit(None), DEFAULT_MAX_ITERATIONS);
    assert_eq!(tool_call_limit(Some(3)), 3);
    assert_eq!(tool_call_limit(Some(0)), 0);
    assert_eq!(
        tool_call_limit(Some(DEFAULT_MAX_ITERATIONS + 5)),
        DEFAULT_MAX_ITERATIONS
    );
    assert_eq!(tool_call_allowance(Some(3), 1), 2);
    assert_eq!(tool_call_allowance(Some(3), 5), 0);
    assert_eq!(tool_call_allowance(None, 4), DEFAULT_MAX_ITERATIONS - 4);
}

#[test]
fn a_batch_past_the_cap_keeps_its_permitted_prefix_and_names_the_cap() {
    assert_eq!(tool_call_overflow(Some(3), 1, 2), None);
    assert_eq!(tool_call_overflow(None, 0, DEFAULT_MAX_ITERATIONS), None);
    assert_eq!(
        tool_call_overflow(Some(3), 1, 4),
        Some((McpToolCallLimit::User, 2))
    );
    assert_eq!(
        tool_call_overflow(Some(1), 1, 1),
        Some((McpToolCallLimit::User, 0))
    );
    assert_eq!(
        tool_call_overflow(Some(DEFAULT_MAX_ITERATIONS), DEFAULT_MAX_ITERATIONS, 1),
        Some((McpToolCallLimit::User, 0))
    );
    assert_eq!(
        tool_call_overflow(None, DEFAULT_MAX_ITERATIONS - 1, 2),
        Some((McpToolCallLimit::Safety, 1))
    );
    assert_eq!(
        tool_call_overflow(Some(DEFAULT_MAX_ITERATIONS + 5), DEFAULT_MAX_ITERATIONS, 1),
        Some((McpToolCallLimit::Safety, 0))
    );
}

#[test]
fn custom_tool_input_extracts_input_or_preserves_raw_arguments() {
    assert_eq!(
        custom_tool_input(r#"{"input":"*** Begin Patch"}"#),
        "*** Begin Patch"
    );
    assert_eq!(
        custom_tool_input(r#"{"patch":"raw"}"#),
        r#"{"patch":"raw"}"#
    );
    assert_eq!(custom_tool_input("*** Begin Patch"), "*** Begin Patch");
    assert_eq!(custom_tool_input(""), "");
}
