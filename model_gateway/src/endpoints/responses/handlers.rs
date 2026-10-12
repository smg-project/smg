use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::responses::generate_id;
use serde_json::{json, Value};
use smg_data_connector::{ResponseId, ResponseStorage, ResponseStorageError, StoredResponse};
use tracing::{info, warn};

use crate::routers::error;

/// The response stored under `id` as a caller of `tenant` may see it: `None`
/// for an id that is not stored and for one stored under another tenant, the
/// same answer, so an id tells a caller nothing about another tenant's records.
async fn visible_response(
    response_storage: &Arc<dyn ResponseStorage>,
    id: &ResponseId,
    tenant: Option<&str>,
) -> Result<Option<StoredResponse>, ResponseStorageError> {
    Ok(response_storage
        .get_response(id)
        .await?
        .filter(|stored| stored.is_visible_to(tenant)))
}

/// `GET /v1/responses/{id}` for a caller of `tenant`.
pub async fn get_response(
    response_storage: &Arc<dyn ResponseStorage>,
    tenant: Option<&str>,
    response_id: &str,
) -> Response {
    let id = ResponseId::from(response_id);
    match visible_response(response_storage, &id, tenant).await {
        Ok(Some(stored)) => (StatusCode::OK, Json(stored.raw_response)).into_response(),
        Ok(None) => error::not_found(
            "not_found",
            format!("No response found with id '{response_id}'"),
        ),
        Err(e) => error::storage_failure("storage_error", "Failed to get response", &e),
    }
}

/// `DELETE /v1/responses/{id}` for a caller of `tenant`.
pub async fn delete_response(
    response_storage: &Arc<dyn ResponseStorage>,
    tenant: Option<&str>,
    response_id: &str,
) -> Response {
    let id = ResponseId::from(response_id);

    match visible_response(response_storage, &id, tenant).await {
        Ok(Some(_)) => match response_storage.delete_response(&id).await {
            Ok(()) => {
                info!(response_id = %id.0, "Deleted response");
                (
                    StatusCode::OK,
                    Json(json!({
                        "id": id.0,
                        "object": "response.deleted",
                        "deleted": true,
                    })),
                )
                    .into_response()
            }
            Err(e) => error::storage_failure("storage_error", "Failed to delete response", &e),
        },
        Ok(None) => error::not_found(
            "not_found",
            format!("No response found with id '{response_id}'"),
        ),
        Err(e) => error::storage_failure("storage_error", "Failed to get response", &e),
    }
}

/// `GET /v1/responses/{id}/input_items` for a caller of `tenant`.
pub async fn list_response_input_items(
    response_storage: &Arc<dyn ResponseStorage>,
    tenant: Option<&str>,
    response_id: &str,
) -> Response {
    let resp_id = ResponseId::from(response_id);

    match visible_response(response_storage, &resp_id, tenant).await {
        Ok(Some(stored)) => {
            let items = stored.input.as_array().cloned().unwrap_or_default();

            let items_with_ids: Vec<Value> = items
                .into_iter()
                .map(|mut item| {
                    if item.get("id").is_none() {
                        if let Some(obj) = item.as_object_mut() {
                            obj.insert("id".to_string(), json!(generate_id("msg")));
                        }
                    }
                    item
                })
                .collect();

            let response_body = json!({
                "object": "list",
                "data": items_with_ids,
                "first_id": items_with_ids.first().and_then(|v| v.get("id").and_then(|i| i.as_str())),
                "last_id": items_with_ids.last().and_then(|v| v.get("id").and_then(|i| i.as_str())),
                "has_more": false
            });

            (StatusCode::OK, Json(response_body)).into_response()
        }
        Ok(None) => error::not_found(
            "not_found",
            format!("No response found with id '{response_id}'"),
        ),
        Err(e) => {
            warn!("Failed to retrieve input items for {}: {}", response_id, e);
            error::storage_failure("storage_error", "Failed to retrieve input items", &e)
        }
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use smg_data_connector::{MemoryResponseStorage, ResponseStorageError, StoredResponse};

    use super::*;
    use crate::routers::error::HEADER_X_SMG_ERROR_CODE;

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

        async fn delete_response(
            &self,
            _response_id: &ResponseId,
        ) -> Result<(), ResponseStorageError> {
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
                get_response(&storage, None, "resp_1").await,
                delete_response(&storage, None, "resp_1").await,
                list_response_input_items(&storage, None, "resp_1").await,
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
                get_response(&storage, caller, "resp_a").await.status(),
                StatusCode::NOT_FOUND,
                "{caller:?}"
            );
            assert_eq!(
                list_response_input_items(&storage, caller, "resp_a")
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
            get_response(&storage, A, "resp_a").await.status(),
            StatusCode::OK
        );
        assert_eq!(
            list_response_input_items(&storage, A, "resp_a")
                .await
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            delete_response(&storage, A, "resp_a").await.status(),
            StatusCode::OK
        );
        assert_eq!(
            get_response(&storage, A, "resp_a").await.status(),
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
                get_response(&storage, caller, "resp_shared").await.status(),
                StatusCode::OK,
                "{caller:?}"
            );
            assert_eq!(
                list_response_input_items(&storage, caller, "resp_shared")
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
        let other = get_response(&storage, B, "resp_a").await;
        let missing = get_response(&storage, B, "resp_missing").await;
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
}
