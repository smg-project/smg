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
        Err(e) => error::internal_error("storage_error", format!("Failed to get response: {e}")),
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
            Err(e) => {
                error::internal_error("storage_error", format!("Failed to delete response: {e}"))
            }
        },
        Ok(None) => error::not_found(
            "not_found",
            format!("No response found with id '{response_id}'"),
        ),
        Err(e) => error::internal_error("storage_error", format!("Failed to get response: {e}")),
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
            error::internal_error(
                "storage_error",
                format!("Failed to retrieve input items: {e}"),
            )
        }
    }
}
