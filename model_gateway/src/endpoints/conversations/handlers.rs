//! Conversation CRUD handlers for the /v1/conversations API - shared across routers

use std::{collections::HashSet, fmt::Display, sync::Arc};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use serde_json::{json, Value};
use smg_data_connector::{
    AddedAtSequence, Conversation, ConversationId, ConversationItem, ConversationItemId,
    ConversationItemStorage, ConversationStorage, ListParams, NewConversation, NewConversationItem,
    SortOrder, StorageFailure,
};
use tracing::info;

use crate::routers::{
    common::{openai_bridge, persistence_utils::item_to_json},
    error as route_error,
};

// ============================================================================
// Constants
// ============================================================================

pub const MAX_METADATA_PROPERTIES: usize = 16;
const MAX_ITEMS_PER_REQUEST: usize = 20;
/// The longest item id a client may supply, in bytes: the id columns of the
/// SQL stores are 64 wide, and every backend gives the same refusal before any
/// store is asked, where one of them used to fail the insert instead.
pub const MAX_ITEM_ID_LENGTH: usize = 64;

const SUPPORTED_ITEM_TYPES: &[&str] = &[
    "message",
    "reasoning",
    "mcp_list_tools",
    "mcp_call",
    "item_reference",
    "function_call",
    "function_call_output",
    "file_search_call",
    "computer_call",
    "computer_call_output",
    "web_search_call",
    "image_generation_call",
    "code_interpreter_call",
    "local_shell_call",
    "local_shell_call_output",
    "mcp_approval_request",
    "mcp_approval_response",
    "custom_tool_call",
    "custom_tool_call_output",
];

const IMPLEMENTED_ITEM_TYPES: &[&str] = &[
    "message",
    "reasoning",
    "mcp_list_tools",
    "mcp_call",
    "item_reference",
];

// ============================================================================
// Error Response Helpers
// ============================================================================

// The gateway's envelope (`{"error": {"message", "type", "param", "code"}}`),
// not a bare string: an SDK decodes `error` as an object.

fn bad_request(message: impl Into<String>) -> Response {
    route_error::bad_request("invalid_request", message)
}

fn not_found(message: impl Into<String>) -> Response {
    route_error::not_found("not_found", message)
}

/// A failed store call: `503 storage_unavailable` when the store did not
/// answer within its bound, else the `500 internal_error` these handlers
/// answer; `context` opens the message.
fn store_error(context: &str, error: &(impl StorageFailure + Display)) -> Response {
    route_error::storage_failure("internal_error", context, error)
}

fn bad_request_structured(error_obj: Value) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": error_obj}))).into_response()
}

/// The refusal of an item the conversation already has, whether it arrives as an
/// `item_reference` or under its own id.
fn item_already_in_conversation() -> Response {
    bad_request_structured(json!({
        "message": "Item already in conversation",
        "type": "invalid_request_error",
        "param": "items",
        "code": "item_already_in_conversation"
    }))
}

/// The refusal of a client-supplied item id longer than the stores keep, under
/// the public API's code for a string over its bound.
fn item_id_above_max_length(id: &str) -> Response {
    bad_request_structured(json!({
        "message": format!(
            "Item id must be at most {MAX_ITEM_ID_LENGTH} bytes long, got {}",
            id.len()
        ),
        "type": "invalid_request_error",
        "param": "items[].id",
        "code": "string_above_max_length"
    }))
}

// ============================================================================
// Storage Helpers
// ============================================================================

async fn ensure_conversation_exists(
    storage: &Arc<dyn ConversationStorage>,
    conv_id: &ConversationId,
) -> Result<Conversation, Response> {
    match storage.get_conversation(conv_id).await {
        Ok(Some(conv)) => Ok(conv),
        Ok(None) => Err(not_found("Conversation not found")),
        Err(e) => Err(store_error("Failed to get conversation", &e)),
    }
}

// ============================================================================
// Metadata Operations
// ============================================================================

fn validate_metadata(value: &Value) -> Result<Option<serde_json::Map<String, Value>>, String> {
    match value.get("metadata") {
        Some(Value::Object(map)) => {
            if map.len() > MAX_METADATA_PROPERTIES {
                Err(format!(
                    "metadata cannot have more than {MAX_METADATA_PROPERTIES} properties"
                ))
            } else {
                Ok(Some(map.clone()))
            }
        }
        Some(_) => Err("metadata must be an object".to_string()),
        None => Ok(None),
    }
}

fn apply_metadata_patches(
    current: Option<serde_json::Map<String, Value>>,
    body: &Value,
) -> Result<Option<serde_json::Map<String, Value>>, String> {
    let patch_map = match body.get("metadata") {
        Some(Value::Object(map)) => map,
        Some(_) => return Err("metadata must be an object".to_string()),
        None => return Ok(current),
    };

    let mut result = current.unwrap_or_default();
    for (k, v) in patch_map {
        if v.is_null() {
            result.remove(k);
        } else {
            result.insert(k.clone(), v.clone());
        }
    }

    if result.len() > MAX_METADATA_PROPERTIES {
        return Err(format!(
            "metadata cannot have more than {MAX_METADATA_PROPERTIES} properties"
        ));
    }

    Ok(if result.is_empty() {
        None
    } else {
        Some(result)
    })
}

// ============================================================================
// Conversation CRUD Handlers
// ============================================================================

pub async fn create_conversation(storage: &Arc<dyn ConversationStorage>, body: Value) -> Response {
    let metadata = match validate_metadata(&body) {
        Ok(m) => m,
        Err(msg) => return bad_request(msg),
    };

    let new_conv = NewConversation { id: None, metadata };

    match storage.create_conversation(new_conv).await {
        Ok(conversation) => {
            info!(conversation_id = %conversation.id.0, "Created conversation");
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Err(e) => store_error("Failed to create conversation", &e),
    }
}

pub async fn get_conversation(storage: &Arc<dyn ConversationStorage>, conv_id: &str) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    match storage.get_conversation(&conversation_id).await {
        Ok(Some(conversation)) => {
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Ok(None) => not_found("Conversation not found"),
        Err(e) => store_error("Failed to get conversation", &e),
    }
}

pub async fn update_conversation(
    storage: &Arc<dyn ConversationStorage>,
    conv_id: &str,
    body: Value,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    let current = match ensure_conversation_exists(storage, &conversation_id).await {
        Ok(c) => c,
        Err(response) => return response,
    };

    let final_metadata = match apply_metadata_patches(current.metadata.clone(), &body) {
        Ok(m) => m,
        Err(msg) => return bad_request(msg),
    };

    match storage
        .update_conversation(&conversation_id, final_metadata)
        .await
    {
        Ok(Some(conversation)) => {
            info!(conversation_id = %conversation_id.0, "Updated conversation");
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Ok(None) => not_found("Conversation not found"),
        Err(e) => store_error("Failed to update conversation", &e),
    }
}

pub async fn delete_conversation(
    storage: &Arc<dyn ConversationStorage>,
    conv_id: &str,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    match storage.delete_conversation(&conversation_id).await {
        Ok(true) => {
            info!(conversation_id = %conversation_id.0, "Deleted conversation");
            (
                StatusCode::OK,
                Json(json!({
                    "id": conversation_id.0,
                    "object": "conversation.deleted",
                    "deleted": true
                })),
            )
                .into_response()
        }
        Ok(false) => not_found("Conversation not found"),
        Err(e) => store_error("Failed to delete conversation", &e),
    }
}

// ============================================================================
// Conversation Item Handlers
// ============================================================================

pub async fn list_conversation_items(
    conversation_storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &str,
    limit: Option<usize>,
    order: Option<&str>,
    after: Option<&str>,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    if let Err(response) = ensure_conversation_exists(conversation_storage, &conversation_id).await
    {
        return response;
    }

    let limit = limit.unwrap_or(100);
    let order = match order {
        Some("asc") => SortOrder::Asc,
        _ => SortOrder::Desc,
    };

    let params = ListParams {
        limit,
        order,
        after: after.map(String::from),
    };

    match item_storage.list_items(&conversation_id, params).await {
        Ok(items) => {
            // The item objects as the public API lists them: each item's own
            // fields, the same object the single-item GET answers.
            let item_values: Vec<Value> = items.iter().map(item_to_json).collect();

            (
                StatusCode::OK,
                Json(json!({
                    "object": "list",
                    "data": item_values,
                    "has_more": items.len() == limit,
                    "first_id": items.first().map(|item| &item.id.0),
                    "last_id": items.last().map(|item| &item.id.0),
                })),
            )
                .into_response()
        }
        Err(e) => store_error("Failed to list items", &e),
    }
}

pub async fn create_conversation_items(
    conversation_storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &str,
    body: Value,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    if let Err(response) = ensure_conversation_exists(conversation_storage, &conversation_id).await
    {
        return response;
    }

    let items_array = match body.get("items").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return bad_request("Missing or invalid 'items' field"),
    };

    if items_array.len() > MAX_ITEMS_PER_REQUEST {
        return bad_request(format!(
            "Cannot add more than {MAX_ITEMS_PER_REQUEST} items at a time"
        ));
    }

    // Bounded before any store is asked, for the whole batch: the stores' id columns are
    // this wide, and a later item's over-long id must not leave earlier items behind.
    if let Some(id_str) = items_array
        .iter()
        .filter_map(|item| item.get("id").and_then(Value::as_str))
        .find(|id| id.len() > MAX_ITEM_ID_LENGTH)
    {
        return item_id_above_max_length(id_str);
    }

    let mut created_items = Vec::new();
    let mut warnings = Vec::new();
    let mut link_pairs = Vec::new();
    let mut seen_ids = HashSet::new();
    // One stamp per item, each later than the one before: the stores order a conversation's
    // items by `added_at`, so the items of this call list in the order the request gave them.
    let mut stamps = AddedAtSequence::new();

    for item_val in items_array {
        let added_at = stamps.next_added_at();
        match process_item(item_storage, &conversation_id, item_val, added_at).await {
            Ok((item_json, item_id, warning)) => {
                if seen_ids.insert(item_id.0.clone()) {
                    link_pairs.push((item_id, added_at));
                }
                created_items.push(item_json);
                if let Some(w) = warning {
                    warnings.push(w);
                }
            }
            Err(response) => return response,
        }
    }

    // Batch-link all items in a single operation
    if let Err(e) = item_storage.link_items(&conversation_id, &link_pairs).await {
        return store_error("Failed to link items to conversation", &e);
    }

    let mut response = json!({
        "object": "list",
        "data": created_items,
        "first_id": created_items.first().and_then(|v| v.get("id")),
        "last_id": created_items.last().and_then(|v| v.get("id")),
        "has_more": false
    });

    if !warnings.is_empty() {
        if let Some(obj) = response.as_object_mut() {
            obj.insert("warnings".to_string(), json!(warnings));
        }
    }

    (StatusCode::OK, Json(response)).into_response()
}

/// Process a single item for creation. Returns (json, item_id, warning).
/// Linking is deferred to the caller for batch operation.
async fn process_item(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conversation_id: &ConversationId,
    item_val: &Value,
    added_at: chrono::DateTime<Utc>,
) -> Result<(Value, ConversationItemId, Option<String>), Response> {
    let item_type = item_val
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message");

    // Handle item_reference specially - just resolve existing item
    if item_type == "item_reference" {
        return process_item_reference(item_storage, conversation_id, item_val, added_at).await;
    }

    let user_provided_id = item_val.get("id").and_then(|v| v.as_str());

    let (item, warning) = if let Some(id_str) = user_provided_id {
        process_item_with_id(item_storage, conversation_id, item_val, id_str).await?
    } else {
        process_new_item(item_storage, item_val).await?
    };

    let item_id = item.id.clone();
    Ok((item_to_json(&item), item_id, warning))
}

/// Process an item_reference - resolve an existing item for linking.
/// Linking is deferred to the caller for batch operation. A reference to an item the
/// conversation already has is refused, as an item arriving under such an id is: linked
/// again, the item would be listed twice by one backend, moved by another and rejected
/// by a third.
async fn process_item_reference(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conversation_id: &ConversationId,
    item_val: &Value,
    _added_at: chrono::DateTime<Utc>,
) -> Result<(Value, ConversationItemId, Option<String>), Response> {
    let ref_id = item_val
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| bad_request("item_reference requires 'id' field"))?;

    let item_id = ConversationItemId::from(ref_id);

    let is_linked = item_storage
        .is_item_linked(conversation_id, &item_id)
        .await
        .map_err(|e| store_error("Failed to check item link", &e))?;
    if is_linked {
        return Err(item_already_in_conversation());
    }

    let existing_item = match item_storage.get_item(&item_id).await {
        Ok(Some(item)) => item,
        Ok(None) => return Err(not_found(format!("Referenced item '{ref_id}' not found"))),
        Err(e) => return Err(store_error("Failed to get referenced item", &e)),
    };

    let item_id = existing_item.id.clone();
    Ok((item_to_json(&existing_item), item_id, None))
}

/// Process an item with a user-provided ID
async fn process_item_with_id(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conversation_id: &ConversationId,
    item_val: &Value,
    id_str: &str,
) -> Result<(ConversationItem, Option<String>), Response> {
    // Bounded before any store is asked: the stores' id columns are this wide,
    // and an id over the bound must be the same 400 on every backend.
    if id_str.len() > MAX_ITEM_ID_LENGTH {
        return Err(item_id_above_max_length(id_str));
    }
    let item_id = ConversationItemId::from(id_str);

    // Check if already linked
    let is_linked = item_storage
        .is_item_linked(conversation_id, &item_id)
        .await
        .map_err(|e| store_error("Failed to check item link", &e))?;

    if is_linked {
        return Err(item_already_in_conversation());
    }

    // Check if item exists globally
    match item_storage.get_item(&item_id).await {
        Ok(Some(existing)) => Ok((existing, None)),
        Ok(None) => {
            // Create new item with the provided ID
            let (mut new_item, warning) = parse_item_from_value(item_val).map_err(bad_request)?;
            new_item.id = Some(item_id);

            let created = item_storage
                .create_item(new_item)
                .await
                .map_err(|e| store_error("Failed to create item", &e))?;

            Ok((created, warning))
        }
        Err(e) => Err(store_error("Failed to check item existence", &e)),
    }
}

/// Process a new item without a user-provided ID
async fn process_new_item(
    item_storage: &Arc<dyn ConversationItemStorage>,
    item_val: &Value,
) -> Result<(ConversationItem, Option<String>), Response> {
    let (new_item, warning) = parse_item_from_value(item_val).map_err(bad_request)?;

    let created = item_storage
        .create_item(new_item)
        .await
        .map_err(|e| store_error("Failed to create item", &e))?;

    Ok((created, warning))
}

/// The `include` values a conversation item GET accepts (the public API's
/// includable fields).
const ITEM_INCLUDE_VALUES: [&str; 8] = [
    "code_interpreter_call.outputs",
    "computer_call_output.output.image_url",
    "file_search_call.results",
    "message.input_image.image_url",
    "message.output_text.logprobs",
    "reasoning.encrypted_content",
    "web_search_call.action.sources",
    "web_search_call.results",
];

/// Parse the `include` parameter of `GET /v1/conversations/{id}/items/{item}`
/// from the raw query string (`include[]=x`, `include=x`, `include[0]=x`).
/// An empty or unknown value is rejected with `invalid_value` on
/// `include[<index>]`, as the public API answers.
pub fn parse_item_include_query(query: Option<&str>) -> Result<Option<Vec<String>>, Response> {
    let Some(query) = query else {
        return Ok(None);
    };
    let mut include = Vec::new();
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        let is_include = key == "include"
            || key
                .strip_prefix("include[")
                .is_some_and(|rest| rest.ends_with(']'));
        if !is_include {
            continue;
        }
        if !ITEM_INCLUDE_VALUES.contains(&value.as_ref()) {
            let index = include.len();
            return Err(bad_request_structured(json!({
                "message": format!(
                    "Invalid value: '{value}'. Supported values are: {}.",
                    ITEM_INCLUDE_VALUES
                        .iter()
                        .map(|v| format!("'{v}'"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                "type": "invalid_request_error",
                "param": format!("include[{index}]"),
                "code": "invalid_value"
            })));
        }
        include.push(value.into_owned());
    }
    Ok((!include.is_empty()).then_some(include))
}

pub async fn get_conversation_item(
    conversation_storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &str,
    item_id: &str,
    _include: Option<Vec<String>>,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);
    let item_id = ConversationItemId::from(item_id);

    if let Err(response) = ensure_conversation_exists(conversation_storage, &conversation_id).await
    {
        return response;
    }

    let is_linked = match item_storage
        .is_item_linked(&conversation_id, &item_id)
        .await
    {
        Ok(linked) => linked,
        Err(e) => return store_error("Failed to check item link", &e),
    };

    if !is_linked {
        return not_found("Item not found in this conversation");
    }

    match item_storage.get_item(&item_id).await {
        Ok(Some(item)) => (StatusCode::OK, Json(item_to_json(&item))).into_response(),
        Ok(None) => not_found("Item not found"),
        Err(e) => store_error("Failed to get item", &e),
    }
}

pub async fn delete_conversation_item(
    conversation_storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
    conv_id: &str,
    item_id: &str,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);
    let item_id = ConversationItemId::from(item_id);

    let conversation =
        match ensure_conversation_exists(conversation_storage, &conversation_id).await {
            Ok(conv) => conv,
            Err(response) => return response,
        };

    match item_storage.delete_item(&conversation_id, &item_id).await {
        Ok(()) => {
            info!(
                conversation_id = %conversation_id.0,
                item_id = %item_id.0,
                "Deleted conversation item"
            );
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Err(e) => store_error("Failed to delete item", &e),
    }
}

// ============================================================================
// Parsing and Serialization
// ============================================================================

fn parse_item_from_value(
    item_val: &Value,
) -> Result<(NewConversationItem, Option<String>), String> {
    let item_type = item_val
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message");

    if !SUPPORTED_ITEM_TYPES.contains(&item_type) {
        return Err(format!(
            "Unsupported item type '{}'. Supported types: {}",
            item_type,
            SUPPORTED_ITEM_TYPES.join(", ")
        ));
    }

    let warning = if IMPLEMENTED_ITEM_TYPES.contains(&item_type) {
        None
    } else {
        Some(format!(
            "Item type '{item_type}' is accepted but not yet implemented. \
             The item will be stored but may not function as expected."
        ))
    };

    let role = item_val
        .get("role")
        .and_then(|v| v.as_str())
        .map(String::from);

    if item_type == "message" && role.is_none() {
        return Err("Message items require 'role' field".to_string());
    }

    let status = item_val
        .get("status")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| Some("completed".to_string()));

    let content = if item_type == "message" {
        message_content_parts(role.as_deref(), item_val.get("content"))
    } else if item_type == "reasoning" {
        item_val.get("content").cloned().unwrap_or(json!([]))
    } else {
        // Strip image_generation_call.result base64 before storage. The
        // compactor is a no-op for non-image item types, so it's safe to
        // apply unconditionally on the non-message branch.
        let mut content = item_val.clone();
        openai_bridge::compact_image_generation_outputs_json(std::slice::from_mut(&mut content));
        content
    };

    Ok((
        NewConversationItem {
            id: None,
            response_id: None,
            item_type: item_type.to_string(),
            role,
            content,
            status,
        },
        warning,
    ))
}

/// The content of a message item as it is stored and listed: a plain string
/// becomes the one typed part the public API makes of it for the message's
/// role, so an item added as `"content": "text"` has the shape of one added
/// with typed parts or created by a response. An assistant message gets an
/// `output_text` part (with empty `annotations`), the text part the public
/// API accepts on that role when the history loaders replay the conversation
/// upstream; every other role gets an `input_text` part. Typed parts stay as
/// sent, an absent content is an empty list.
fn message_content_parts(role: Option<&str>, content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(text)) if role == Some("assistant") => {
            json!([{ "type": "output_text", "text": text, "annotations": [] }])
        }
        Some(Value::String(text)) => json!([{ "type": "input_text", "text": text }]),
        Some(parts) => parts.clone(),
        None => json!([]),
    }
}

/// The Conversation object. `metadata` is always present, an empty object
/// for a conversation created without any, as the public API answers it.
pub fn conversation_to_json(conversation: &Conversation) -> Value {
    json!({
        "id": conversation.id.0,
        "object": "conversation",
        "created_at": conversation.created_at.timestamp(),
        "metadata": Value::Object(conversation.metadata.clone().unwrap_or_default()),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use smg_data_connector::{MemoryConversationItemStorage, MemoryConversationStorage};

    use super::*;

    /// A client-supplied item id longer than the stores' id columns is refused
    /// at the handler, before any store is asked, so every backend answers the
    /// same 400 under the public API's code; an id at the bound is kept and
    /// linked under that id.
    #[tokio::test]
    async fn an_item_id_over_the_bound_is_refused_before_the_store_and_one_at_the_bound_is_kept() {
        let conversation_storage: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let item_storage: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::new());
        let conversation = conversation_storage
            .create_conversation(NewConversation {
                id: None,
                metadata: None,
            })
            .await
            .unwrap();
        let conv_id = conversation.id.0.clone();
        let items = |id: &str| json!({"items": [{"type": "message", "role": "user", "content": "x", "id": id}]});

        // 65 and 64 characters: one over the stores' 64-character id columns, one at the bound.
        let over = format!("msg_{}", "a".repeat(61));
        assert_eq!(over.len(), 65);
        let response =
            create_conversation_items(&conversation_storage, &item_storage, &conv_id, items(&over))
                .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "string_above_max_length", "{json}");
        assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
        assert_eq!(json["error"]["param"], "items[].id", "{json}");
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("64"),
            "the message names the bound: {json}"
        );

        // The bound holds for the whole batch before any store is asked: an over-long
        // id on the second item leaves the first item unstored and unlinked.
        let batch = json!({"items": [
            {"type": "message", "role": "user", "content": "first", "id": "msg_first"},
            {"type": "message", "role": "user", "content": "second", "id": over}
        ]});
        let response =
            create_conversation_items(&conversation_storage, &item_storage, &conv_id, batch).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let listed = item_storage
            .list_items(
                &ConversationId::from(conv_id.as_str()),
                ListParams {
                    limit: 10,
                    order: SortOrder::Asc,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert!(
            listed.is_empty(),
            "nothing is stored for a refused batch: {listed:?}"
        );
        // The store itself, not the conversation's links: the old per-item check
        // created the first item before refusing the second, linking none.
        assert!(
            item_storage
                .get_item(&ConversationItemId::from("msg_first"))
                .await
                .unwrap()
                .is_none(),
            "an earlier item of a refused batch is not stored"
        );
        let over_id = ConversationItemId::from(over.as_str());
        assert!(
            item_storage.get_item(&over_id).await.unwrap().is_none(),
            "nothing is stored under the refused id"
        );
        assert!(!item_storage
            .is_item_linked(&conversation.id, &over_id)
            .await
            .unwrap());

        let at_bound = format!("msg_{}", "a".repeat(60));
        assert_eq!(at_bound.len(), 64);
        let response = create_conversation_items(
            &conversation_storage,
            &item_storage,
            &conv_id,
            items(&at_bound),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(item_storage
            .is_item_linked(
                &conversation.id,
                &ConversationItemId::from(at_bound.as_str())
            )
            .await
            .unwrap());
    }

    /// The CRUD helpers answer with the gateway's envelope, an object an SDK
    /// decodes (`type` from the public vocabulary, a string `code`, `param`),
    /// not a bare string under `error`.
    #[tokio::test]
    async fn error_helpers_build_the_object_envelope() {
        for (response, status, code) in [
            (
                bad_request("bad"),
                StatusCode::BAD_REQUEST,
                "invalid_request",
            ),
            (
                not_found("Conversation not found"),
                StatusCode::NOT_FOUND,
                "not_found",
            ),
            (
                store_error(
                    "Failed to get conversation",
                    &smg_data_connector::ResponseStorageError::Timeout(
                        "the redis get_conversation call exceeded its bound of 5 s".to_string(),
                    ),
                ),
                StatusCode::SERVICE_UNAVAILABLE,
                "storage_unavailable",
            ),
            (
                store_error(
                    "Failed to get conversation",
                    &smg_data_connector::ResponseStorageError::StorageError("db error".to_string()),
                ),
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
            ),
        ] {
            assert_eq!(response.status(), status);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let json: Value = serde_json::from_slice(&body).unwrap();
            let error = json["error"]
                .as_object()
                .unwrap_or_else(|| panic!("error must be an object: {json}"));
            assert_eq!(error["code"], code);
            assert_eq!(error["type"], route_error::error_type_for_status(status));
            assert!(error["message"].is_string());
            assert!(error["param"].is_null());
        }
    }
}
