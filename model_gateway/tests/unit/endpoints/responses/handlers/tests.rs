use async_trait::async_trait;
use axum::http::header::CONTENT_TYPE;
use smg_data_connector::{
    MemoryConfig, MemoryResponseStorage, ResponseStorageError, StoredResponse,
};

use super::*;
use crate::{middleware::test_support::json_of, routers::error::HEADER_X_SMG_ERROR_CODE};

/// A store that fails every read with the given error.
struct FailingReads(fn() -> ResponseStorageError);

#[async_trait]
impl ResponseStorage for FailingReads {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        Ok(response.id)
    }

    async fn get_response(
        &self,
        _response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        Err((self.0)())
    }

    async fn delete_response(&self, _response_id: &ResponseId) -> Result<(), ResponseStorageError> {
        Ok(())
    }

    async fn list_identifier_responses(
        &self,
        _identifier: &str,
        _tenant: Option<&str>,
        _limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
        Ok(Vec::new())
    }

    async fn delete_identifier_responses(
        &self,
        _identifier: &str,
        _tenant: Option<&str>,
    ) -> Result<usize, ResponseStorageError> {
        Ok(0)
    }
}

/// A store that did not answer within its bound, or is not available at
/// all, makes the GET, the DELETE and the input-items listing a 503
/// `storage_unavailable`; a failure the store answered stays the 500
/// `storage_error` it was.
#[tokio::test]
async fn a_store_that_timed_out_is_answered_as_unavailable() {
    fn timed_out() -> ResponseStorageError {
        ResponseStorageError::Timeout(
            "the postgres get_response call exceeded its bound of 5 s".to_string(),
        )
    }
    fn refused() -> ResponseStorageError {
        ResponseStorageError::Unavailable("connection refused".to_string())
    }
    fn answered() -> ResponseStorageError {
        ResponseStorageError::StorageError("duplicate key".to_string())
    }
    for (failure, status, code) in [
        (
            timed_out as fn() -> ResponseStorageError,
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_unavailable",
        ),
        (
            refused,
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_unavailable",
        ),
        (answered, StatusCode::INTERNAL_SERVER_ERROR, "storage_error"),
    ] {
        let storage: Arc<dyn ResponseStorage> = Arc::new(FailingReads(failure));
        for response in [
            get_response(&storage, None, "resp_1", None).await,
            delete_response(&storage, None, "resp_1").await,
            list_response_input_items(&storage, None, "resp_1", None).await,
        ] {
            assert_eq!(response.status(), status);
            assert_eq!(
                response.headers().get(HEADER_X_SMG_ERROR_CODE).unwrap(),
                code
            );
        }
    }
}

const A: Option<&str> = Some("header:tenant-a");
const B: Option<&str> = Some("header:tenant-b");

async fn storage_with(id: &str, tenant: Option<&str>) -> Arc<dyn ResponseStorage> {
    let storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
    let mut stored = StoredResponse::new(None);
    stored.id = ResponseId::from(id);
    stored.input = json!([{"type": "message", "role": "user", "content": "secret prompt"}]);
    stored.raw_response = json!({"id": id, "status": "completed", "output": []});
    stored.tenant = tenant.map(str::to_string);
    storage.store_response(stored).await.unwrap();
    storage
}

/// A response stored under a tenant is read, listed and deleted by that
/// tenant only; another tenant, and a caller without one, get the answer
/// of an unknown id and change nothing.
#[tokio::test]
async fn another_tenants_response_answers_not_found_on_every_by_id_route() {
    let storage = storage_with("resp_a", A).await;
    for caller in [B, None] {
        assert_eq!(
            get_response(&storage, caller, "resp_a", None)
                .await
                .status(),
            StatusCode::NOT_FOUND,
            "{caller:?}"
        );
        assert_eq!(
            list_response_input_items(&storage, caller, "resp_a", None)
                .await
                .status(),
            StatusCode::NOT_FOUND,
            "{caller:?}"
        );
        assert_eq!(
            delete_response(&storage, caller, "resp_a").await.status(),
            StatusCode::NOT_FOUND,
            "{caller:?}"
        );
    }
    assert!(
        storage
            .get_response(&ResponseId::from("resp_a"))
            .await
            .unwrap()
            .is_some(),
        "a refused delete leaves the record"
    );
    assert_eq!(
        get_response(&storage, A, "resp_a", None).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        list_response_input_items(&storage, A, "resp_a", None)
            .await
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        delete_response(&storage, A, "resp_a").await.status(),
        StatusCode::OK
    );
    assert_eq!(
        get_response(&storage, A, "resp_a", None).await.status(),
        StatusCode::NOT_FOUND
    );
}

/// A response stored under no tenant (no tenant identity configured, or
/// stored before records carried one) is served as before.
#[tokio::test]
async fn a_response_without_a_tenant_is_served_to_every_caller() {
    let storage = storage_with("resp_shared", None).await;
    for caller in [None, A] {
        assert_eq!(
            get_response(&storage, caller, "resp_shared", None)
                .await
                .status(),
            StatusCode::OK,
            "{caller:?}"
        );
        assert_eq!(
            list_response_input_items(&storage, caller, "resp_shared", None)
                .await
                .status(),
            StatusCode::OK,
            "{caller:?}"
        );
    }
    assert_eq!(
        delete_response(&storage, B, "resp_shared").await.status(),
        StatusCode::OK
    );
}

/// The answer for another tenant's id is the answer for an unknown id.
#[tokio::test]
async fn the_refusal_is_the_unknown_id_answer() {
    let storage = storage_with("resp_a", A).await;
    let other = get_response(&storage, B, "resp_a", None).await;
    let missing = get_response(&storage, B, "resp_missing", None).await;
    assert_eq!(other.status(), missing.status());
    let body = |response: Response| async {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        json["error"]["code"].clone()
    };
    assert_eq!(body(other).await, body(missing).await);
}

/// `resp_paged`: three stored input messages in the order the request gave
/// them, each with the id the store minted.
async fn paged_storage() -> Arc<dyn ResponseStorage> {
    let storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
    let mut stored = StoredResponse::new(None);
    stored.id = ResponseId::from("resp_paged");
    stored.input = Value::Array(
        (0..3)
            .map(|i| {
                json!({"id": format!("msg_{i}"), "type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": format!("q{i}")}]})
            })
            .collect(),
    );
    stored.raw_response = json!({"id": "resp_paged", "status": "completed", "output": []});
    storage.store_response(stored).await.unwrap();
    storage
}

fn texts(list: &Value) -> Vec<String> {
    list["data"]
        .as_array()
        .expect("a data array")
        .iter()
        .map(|item| {
            item["content"][0]["text"]
                .as_str()
                .unwrap_or("?")
                .to_owned()
        })
        .collect()
}

/// The input items list pages in the order asked for from the cursor
/// given: the newest first by default, `has_more` only while items follow,
/// a cursor the list does not hold starting the page at its first item, and
/// twenty items to a page when the query names no limit.
#[tokio::test]
async fn the_input_items_list_pages_in_the_order_asked_from_the_cursor_given() {
    let storage = paged_storage().await;
    for (query, expected, has_more, first_id, last_id) in [
        ("", vec!["q2", "q1", "q0"], false, "msg_2", "msg_0"),
        ("limit=1", vec!["q2"], true, "msg_2", "msg_2"),
        ("limit=1&after=msg_2", vec!["q1"], true, "msg_1", "msg_1"),
        ("limit=1&after=msg_1", vec!["q0"], false, "msg_0", "msg_0"),
        (
            "limit=2&after=msg_2",
            vec!["q1", "q0"],
            false,
            "msg_1",
            "msg_0",
        ),
        (
            "order=desc&limit=100",
            vec!["q2", "q1", "q0"],
            false,
            "msg_2",
            "msg_0",
        ),
        ("order=asc", vec!["q0", "q1", "q2"], false, "msg_0", "msg_2"),
        (
            "order=asc&limit=2",
            vec!["q0", "q1"],
            true,
            "msg_0",
            "msg_1",
        ),
        (
            "order=asc&after=msg_0&limit=5",
            vec!["q1", "q2"],
            false,
            "msg_1",
            "msg_2",
        ),
        (
            "after=msg_unknown&limit=2",
            vec!["q2", "q1"],
            true,
            "msg_2",
            "msg_1",
        ),
        (
            "include[]=message.input_image.image_url",
            vec!["q2", "q1", "q0"],
            false,
            "msg_2",
            "msg_0",
        ),
    ] {
        let response = list_response_input_items(&storage, None, "resp_paged", Some(query)).await;
        assert_eq!(response.status(), StatusCode::OK, "{query}");
        let list = json_of(response).await;
        assert_eq!(list["object"], "list", "{query}");
        assert_eq!(texts(&list), expected, "{query}: {list}");
        assert_eq!(list["has_more"], has_more, "{query}: {list}");
        assert_eq!(list["first_id"], first_id, "{query}: {list}");
        assert_eq!(list["last_id"], last_id, "{query}: {list}");
    }
    // A cursor past the last item is an empty page that nothing follows.
    let response =
        list_response_input_items(&storage, None, "resp_paged", Some("after=msg_0")).await;
    let list = json_of(response).await;
    assert!(texts(&list).is_empty(), "{list}");
    assert_eq!(list["has_more"], false, "{list}");
    assert_eq!(list["first_id"], Value::Null, "{list}");

    let mut long = StoredResponse::new(None);
    long.id = ResponseId::from("resp_long");
    long.input = Value::Array(
        (0..25)
            .map(|i| {
                json!({"id": format!("msg_{i}"), "type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": format!("q{i}")}]})
            })
            .collect(),
    );
    storage.store_response(long).await.unwrap();
    let response = list_response_input_items(&storage, None, "resp_long", None).await;
    let list = json_of(response).await;
    assert_eq!(texts(&list).len(), InputItemsQuery::DEFAULT_LIMIT, "{list}");
    assert_eq!(list["has_more"], true, "{list}");
    assert_eq!(list["first_id"], "msg_24", "{list}");
    assert_eq!(list["last_id"], "msg_5", "{list}");
    let response =
        list_response_input_items(&storage, None, "resp_long", Some("after=msg_5")).await;
    let list = json_of(response).await;
    assert_eq!(texts(&list), ["q4", "q3", "q2", "q1", "q0"], "{list}");
    assert_eq!(list["has_more"], false, "{list}");
}

/// A record whose items were stored without ids (from before ids were
/// minted at store time) pages like any other: the ids it lists are the
/// same on every read, so a walk by `after=<last_id>` reaches `has_more:
/// false` with every item seen once, and a second walk reads the same ids.
#[tokio::test]
async fn a_record_of_id_less_items_pages_to_its_end_under_the_same_ids() {
    let storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::new());
    let mut legacy = StoredResponse::new(None);
    legacy.id = ResponseId::from("resp_legacy");
    legacy.input = Value::Array(
        (0..25)
            .map(|i| {
                let mut item = json!({"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": format!("q{i}")}]});
                // every fifth item carries the empty id the store reads as missing
                if i % 5 == 0 {
                    item["id"] = json!("");
                }
                item
            })
            .collect(),
    );
    legacy.raw_response = json!({"id": "resp_legacy", "status": "completed", "output": []});
    storage.store_response(legacy).await.unwrap();

    /// One walk of the record, ten items to a page: the ids and the texts
    /// in the order read, and the number of pages.
    async fn walk(storage: &Arc<dyn ResponseStorage>) -> (Vec<String>, Vec<String>, usize) {
        let (mut ids, mut seen, mut pages): (Vec<String>, Vec<String>, usize) =
            (Vec::new(), Vec::new(), 0);
        let mut query = Some("limit=10".to_owned());
        while let Some(query_now) = query.take() {
            let list = json_of(
                list_response_input_items(storage, None, "resp_legacy", Some(&query_now)).await,
            )
            .await;
            pages += 1;
            assert!(pages <= 4, "the walk did not end in four pages: {ids:?}");
            let page = list["data"].as_array().unwrap();
            assert!(page.len() <= 10, "a page of at most ten: {list}");
            ids.extend(
                page.iter()
                    .map(|item| item["id"].as_str().unwrap().to_owned()),
            );
            seen.extend(texts(&list));
            if list["has_more"] == true {
                query = Some(format!(
                    "limit=10&after={}",
                    list["last_id"].as_str().unwrap()
                ));
            }
        }
        (ids, seen, pages)
    }
    let (ids, seen, pages) = walk(&storage).await;
    assert_eq!(pages, 3, "{ids:?}");
    assert_eq!(
        seen,
        (0..25).rev().map(|i| format!("q{i}")).collect::<Vec<_>>()
    );
    assert_eq!(
        ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
        25,
        "{ids:?}"
    );
    assert_eq!(
        walk(&storage).await.0,
        ids,
        "a second walk reads the same ids"
    );
}

/// A query value the contract does not take is refused with the parameter
/// named, before the store is read (an unknown id with such a query is the
/// 400, not the 404); an unknown `include` value answers the refusal the
/// conversation item routes give it, on both read routes.
#[tokio::test]
async fn a_query_value_outside_the_contract_is_refused_with_the_parameter_named() {
    let storage = paged_storage().await;
    for (query, parameter) in [
        ("limit=0", "limit"),
        ("limit=101", "limit"),
        ("limit=-1", "limit"),
        ("limit=abc", "limit"),
        ("order=sideways", "order"),
        ("limit=1&order=", "order"),
    ] {
        for id in ["resp_paged", "resp_missing"] {
            let response = list_response_input_items(&storage, None, id, Some(query)).await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "{query} on {id}"
            );
            assert_eq!(
                response.headers().get(HEADER_X_SMG_ERROR_CODE).unwrap(),
                INVALID_QUERY_PARAMETER,
                "{query} on {id}"
            );
            let json = json_of(response).await;
            assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
            let message = json["error"]["message"].as_str().unwrap_or_default();
            assert!(
                message.starts_with(&format!("{parameter}:")),
                "{query}: the message names the parameter: {json}"
            );
        }
    }
    for response in [
        list_response_input_items(&storage, None, "resp_paged", Some("include[]=bogus")).await,
        get_response(&storage, None, "resp_paged", Some("include[]=bogus")).await,
        get_response(
            &storage,
            None,
            "resp_paged",
            Some("include=message.output_text.logprobs&include[1]=bogus"),
        )
        .await,
    ] {
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let json = json_of(response).await;
        assert_eq!(json["error"]["code"], "invalid_value", "{json}");
        assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
        assert!(
            json["error"]["param"]
                .as_str()
                .unwrap_or_default()
                .starts_with("include["),
            "{json}"
        );
    }
}

/// A stored response is read as one JSON object: `stream=true` is refused
/// by name (the gateway has no event stream of it to join) instead of
/// answered as JSON to a client that asked for events; `stream=false`, a
/// valid `include` and a parameter the contract does not name read the
/// object unchanged.
#[tokio::test]
async fn a_stream_read_of_a_stored_response_is_refused_not_answered_as_json() {
    let storage = storage_with("resp_a", A).await;
    for query in [
        "stream=true",
        "stream=maybe",
        "stream=true&starting_after=3",
    ] {
        let response = get_response(&storage, A, "resp_a", Some(query)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{query}");
        assert_eq!(
            response.headers().get(HEADER_X_SMG_ERROR_CODE).unwrap(),
            INVALID_QUERY_PARAMETER,
            "{query}"
        );
        let json = json_of(response).await;
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .starts_with("stream:"),
            "{query}: {json}"
        );
    }
    for query in [
        "",
        "stream=false",
        "include[]=reasoning.encrypted_content&include[]=message.output_text.logprobs",
        "starting_after=3",
    ] {
        let response = get_response(&storage, A, "resp_a", Some(query)).await;
        assert_eq!(response.status(), StatusCode::OK, "{query}");
        assert_eq!(
            response.headers().get(CONTENT_TYPE).unwrap(),
            "application/json",
            "{query}"
        );
        assert_eq!(
            json_of(response).await,
            json!({"id": "resp_a", "status": "completed", "output": []}),
            "{query}"
        );
    }
}

/// A response the bounded memory store evicted answers as an unknown id.
#[tokio::test]
async fn an_evicted_response_answers_not_found() {
    let config = MemoryConfig {
        max_responses: 2,
        max_items: 1,
        retention_secs: None,
    };
    let storage: Arc<dyn ResponseStorage> = Arc::new(MemoryResponseStorage::with_config(&config));
    let mut ids = Vec::new();
    for i in 0..3 {
        let mut stored = StoredResponse::new(None);
        stored.raw_response = json!({"id": stored.id.0, "status": "completed", "n": i});
        ids.push(storage.store_response(stored).await.unwrap().0);
    }
    assert_eq!(
        get_response(&storage, None, &ids[0], None).await.status(),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        list_response_input_items(&storage, None, &ids[0], None)
            .await
            .status(),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        get_response(&storage, None, &ids[2], None).await.status(),
        StatusCode::OK
    );
}
