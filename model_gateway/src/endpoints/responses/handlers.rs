use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::responses::generate_id;
use serde_json::{json, Value};
use smg_data_connector::{ResponseId, ResponseStorage};
use tracing::{info, warn};

use crate::routers::error;

pub async fn get_response(
    response_storage: &Arc<dyn ResponseStorage>,
    response_id: &str,
) -> Response {
    let id = ResponseId::from(response_id);
    match response_storage.get_response(&id).await {
        Ok(Some(stored)) => (StatusCode::OK, Json(stored.raw_response)).into_response(),
        Ok(None) => error::not_found(
            "not_found",
            format!("No response found with id '{response_id}'"),
        ),
        Err(e) => error::storage_failure("storage_error", "Failed to get response", &e),
    }
}

pub async fn delete_response(
    response_storage: &Arc<dyn ResponseStorage>,
    response_id: &str,
) -> Response {
    let id = ResponseId::from(response_id);

    match response_storage.get_response(&id).await {
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

pub async fn list_response_input_items(
    response_storage: &Arc<dyn ResponseStorage>,
    response_id: &str,
) -> Response {
    let resp_id = ResponseId::from(response_id);

    match response_storage.get_response(&resp_id).await {
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
    use smg_data_connector::{ResponseStorageError, StoredResponse};

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
            _limit: Option<usize>,
        ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
            Ok(Vec::new())
        }

        async fn delete_identifier_responses(
            &self,
            _identifier: &str,
        ) -> Result<usize, ResponseStorageError> {
            Ok(0)
        }
    }

    /// A store that did not answer within its bound makes the GET, the DELETE
    /// and the input-items listing a 503 `storage_unavailable`; any other store
    /// failure stays the 500 `storage_error` it was.
    #[tokio::test]
    async fn a_store_that_timed_out_is_answered_as_unavailable() {
        fn timed_out() -> ResponseStorageError {
            ResponseStorageError::Timeout(
                "the postgres get_response call exceeded its bound of 5 s".to_string(),
            )
        }
        fn refused() -> ResponseStorageError {
            ResponseStorageError::StorageError("connection refused".to_string())
        }
        for (failure, status, code) in [
            (
                timed_out as fn() -> ResponseStorageError,
                StatusCode::SERVICE_UNAVAILABLE,
                "storage_unavailable",
            ),
            (refused, StatusCode::INTERNAL_SERVER_ERROR, "storage_error"),
        ] {
            let storage: Arc<dyn ResponseStorage> = Arc::new(FailingReads(failure));
            for response in [
                get_response(&storage, "resp_1").await,
                delete_response(&storage, "resp_1").await,
                list_response_input_items(&storage, "resp_1").await,
            ] {
                assert_eq!(response.status(), status);
                assert_eq!(
                    response.headers().get(HEADER_X_SMG_ERROR_CODE).unwrap(),
                    code
                );
            }
        }
    }
}
