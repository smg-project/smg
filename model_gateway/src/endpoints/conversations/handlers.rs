//! Conversation CRUD handlers for the /v1/conversations API - shared across routers

use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    sync::Arc,
};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::{
    common::json_type_name,
    responses::{
        validate_input_message_role, validate_metadata_pair_count, validate_metadata_pairs,
        ResponseContentPart,
    },
};
use serde_json::{json, Value};
use smg_data_connector::{
    AddedAtSequence, Conversation, ConversationId, ConversationItem, ConversationItemId,
    ConversationItemStorage, ConversationStorage, ListParams, NewConversation, NewConversationItem,
    SortOrder, StorageFailure,
};
use tracing::{info, warn};

use crate::routers::{
    common::{openai_bridge, persistence_utils::item_to_json},
    error as route_error,
};

// ============================================================================
// Constants
// ============================================================================

const MAX_ITEMS_PER_REQUEST: usize = 20;
/// The longest item id a client may supply, in bytes: the id columns of the
/// SQL stores are 64 wide, and every backend gives the same refusal before any
/// store is asked, where one of them used to fail the insert instead.
pub const MAX_ITEM_ID_LENGTH: usize = 64;
/// The `status` values a message item may carry (the public API's set for a
/// message; the other item types have sets of their own and keep the status
/// they are sent).
const MESSAGE_STATUSES: [&str; 3] = ["in_progress", "completed", "incomplete"];

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

/// A refusal in the public API's words: `code` and `message` as a validator
/// of the protocols crate gives them, under `param` (none for the body as a
/// whole).
fn refused(param: Option<&str>, code: &str, message: &str) -> Response {
    bad_request_structured(json!({
        "message": message,
        "type": "invalid_request_error",
        "param": param,
        "code": code
    }))
}

/// The refusal of a `metadata` object outside the public API's bounds, as
/// `validate_metadata_pairs` words it.
fn metadata_refused(code: &str, message: Option<&str>) -> Response {
    refused(Some("metadata"), code, message.unwrap_or_default())
}

/// A request body that is not a JSON object is refused as the typed
/// extractors refuse one (`invalid_type`), before any field is read.
fn object_body(body: &Value) -> Result<(), Response> {
    if body.is_object() {
        return Ok(());
    }
    Err(refused(
        None,
        "invalid_type",
        &format!(
            "Invalid type for the request body: expected an object, but got {} instead.",
            json_type_name(body)
        ),
    ))
}

// ============================================================================
// Storage Helpers
// ============================================================================

/// The conversation under `conv_id` as a caller of `tenant` may see it. A
/// conversation stored under another tenant answers as an unknown id does, so
/// an id tells a caller nothing about another tenant's conversations.
async fn ensure_conversation_exists(
    storage: &Arc<dyn ConversationStorage>,
    conv_id: &ConversationId,
    tenant: Option<&str>,
) -> Result<Conversation, Response> {
    match storage.get_conversation(conv_id).await {
        Ok(Some(conv)) if conv.is_visible_to(tenant) => Ok(conv),
        Ok(_) => Err(not_found("Conversation not found")),
        Err(e) => Err(store_error("Failed to get conversation", &e)),
    }
}

// ============================================================================
// Metadata Operations
// ============================================================================

/// The `metadata` of a create body: none when absent, the object once the
/// public API's bounds hold (`validate_metadata_pairs`: 16 pairs, 64-character
/// keys, 512-character string values, strings only); a bound crossed and a
/// value of another type are refused under the public API's codes.
fn validate_metadata(value: &Value) -> Result<Option<serde_json::Map<String, Value>>, Response> {
    match value.get("metadata") {
        Some(Value::Object(map)) => {
            validate_metadata_pairs(map.iter())
                .map_err(|error| metadata_refused(&error.code, error.message.as_deref()))?;
            Ok(Some(map.clone()))
        }
        Some(other) => Err(metadata_not_an_object(other)),
        None => Ok(None),
    }
}

/// `current` with the `metadata` patch of an update body applied (a null
/// value removes its key). The key and value rules are read on the patch's
/// own pairs, so a pair stored before these rules held does not block an
/// unrelated patch; the 16-pair bound is the stored object's and is read on
/// the merged result. A patch that crosses either changes nothing.
fn apply_metadata_patches(
    current: Option<serde_json::Map<String, Value>>,
    body: &Value,
) -> Result<Option<serde_json::Map<String, Value>>, Response> {
    let patch_map = match body.get("metadata") {
        Some(Value::Object(map)) => map,
        Some(other) => return Err(metadata_not_an_object(other)),
        None => return Ok(current),
    };

    // A null value removes its key and is not a value under the rules.
    let patched: Vec<(&String, &Value)> = patch_map
        .iter()
        .filter(|(_, value)| !value.is_null())
        .collect();
    validate_metadata_pairs(patched.iter().copied())
        .map_err(|error| metadata_refused(&error.code, error.message.as_deref()))?;
    let mut result = current.unwrap_or_default();
    for (k, v) in patch_map {
        if v.is_null() {
            result.remove(k);
        } else {
            result.insert(k.clone(), v.clone());
        }
    }
    validate_metadata_pair_count(result.len())
        .map_err(|error| metadata_refused(&error.code, error.message.as_deref()))?;

    Ok(if result.is_empty() {
        None
    } else {
        Some(result)
    })
}

/// The refusal of a `metadata` field that is not an object.
fn metadata_not_an_object(value: &Value) -> Response {
    metadata_refused(
        "invalid_type",
        Some(&format!(
            "Invalid type for 'metadata': expected an object, but got {} instead.",
            json_type_name(value)
        )),
    )
}

// ============================================================================
// Conversation CRUD Handlers
// ============================================================================

/// `POST /v1/conversations`: the conversation is scoped to the caller's
/// `tenant`. Its `items`, when the request carries any, are prepared, stored
/// and linked through the path of `POST /v1/conversations/{id}/items` (the
/// same bounds, refusals and `warnings`), so a conversation created with items
/// and one created empty and given the same items list the same. The whole
/// batch is checked before the conversation exists (`prepare_items`: the
/// bounds, an unsupported type, a message without a role, an unknown
/// reference, an id of another tenant), so a refused create creates no
/// conversation; a store fault during the writes, a reference the writes
/// evicted from a bounded store, or an id another caller's write took since
/// the batch was prepared, deletes the conversation just created (the records
/// the batch wrote stay in the item store, unlinked).
pub async fn create_conversation(
    storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
    tenant: Option<&str>,
    body: Value,
) -> Response {
    if let Err(response) = object_body(&body) {
        return response;
    }
    let metadata = match validate_metadata(&body) {
        Ok(m) => m,
        Err(response) => return response,
    };
    let items = match items_of(&body, false) {
        Ok(items) => items,
        Err(response) => return response,
    };
    // The whole batch is checked against the stores before the conversation
    // exists, so a refused batch leaves nothing behind.
    let prepared = match prepare_items(item_storage, None, items, tenant).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };

    let new_conv = NewConversation {
        id: None,
        metadata,
        tenant: tenant.map(str::to_string),
    };

    let conversation = match storage.create_conversation(new_conv).await {
        Ok(conversation) => conversation,
        Err(e) => return store_error("Failed to create conversation", &e),
    };
    info!(conversation_id = %conversation.id.0, "Created conversation");
    let mut answer = conversation_to_json(&conversation);
    if !prepared.is_empty() {
        match store_items(item_storage, &conversation.id, prepared, tenant).await {
            Ok((_, warnings)) => {
                if !warnings.is_empty() {
                    if let Some(obj) = answer.as_object_mut() {
                        obj.insert("warnings".to_string(), json!(warnings));
                    }
                }
            }
            Err(response) => {
                // A store fault, a referenced item the writes evicted, or an
                // id another tenant's write took since the batch was prepared
                // reaches here: every other refusal was found before the
                // conversation was created.
                if let Err(e) = storage.delete_conversation(&conversation.id).await {
                    warn!(
                        conversation_id = %conversation.id.0,
                        error = %e,
                        "A conversation whose items were refused could not be deleted"
                    );
                }
                return response;
            }
        }
    }
    (StatusCode::OK, Json(answer)).into_response()
}

pub async fn get_conversation(
    storage: &Arc<dyn ConversationStorage>,
    tenant: Option<&str>,
    conv_id: &str,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    match ensure_conversation_exists(storage, &conversation_id, tenant).await {
        Ok(conversation) => {
            (StatusCode::OK, Json(conversation_to_json(&conversation))).into_response()
        }
        Err(response) => response,
    }
}

pub async fn update_conversation(
    storage: &Arc<dyn ConversationStorage>,
    tenant: Option<&str>,
    conv_id: &str,
    body: Value,
) -> Response {
    if let Err(response) = object_body(&body) {
        return response;
    }
    let conversation_id = ConversationId::from(conv_id);

    let current = match ensure_conversation_exists(storage, &conversation_id, tenant).await {
        Ok(c) => c,
        Err(response) => return response,
    };

    let final_metadata = match apply_metadata_patches(current.metadata.clone(), &body) {
        Ok(m) => m,
        Err(response) => return response,
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
    tenant: Option<&str>,
    conv_id: &str,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    if let Err(response) = ensure_conversation_exists(storage, &conversation_id, tenant).await {
        return response;
    }

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
    tenant: Option<&str>,
    conv_id: &str,
    limit: Option<usize>,
    order: Option<&str>,
    after: Option<&str>,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);

    if let Err(response) =
        ensure_conversation_exists(conversation_storage, &conversation_id, tenant).await
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
            let has_more = items.len() == limit;
            let first_id = items.first().map(|item| item.id.0.clone());
            let last_id = items.last().map(|item| item.id.0.clone());
            // The item objects as the public API lists them: each item's own
            // fields, the same object the single-item GET answers; the page is
            // this answer's own, so each item moves into its object.
            let item_values: Vec<Value> = items.into_iter().map(item_to_json).collect();

            (
                StatusCode::OK,
                Json(json!({
                    "object": "list",
                    "data": item_values,
                    "has_more": has_more,
                    "first_id": first_id,
                    "last_id": last_id,
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
    tenant: Option<&str>,
    conv_id: &str,
    body: Value,
) -> Response {
    if let Err(response) = object_body(&body) {
        return response;
    }
    let conversation_id = ConversationId::from(conv_id);

    if let Err(response) =
        ensure_conversation_exists(conversation_storage, &conversation_id, tenant).await
    {
        return response;
    }

    let items = match items_of(&body, true) {
        Ok(items) => items,
        Err(response) => return response,
    };
    let prepared = match prepare_items(item_storage, Some(&conversation_id), items, tenant).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    let (created_items, warnings) =
        match store_items(item_storage, &conversation_id, prepared, tenant).await {
            Ok(stored) => stored,
            Err(response) => return response,
        };

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

/// The `items` of a request body: the array, its per-request bound and the
/// id-length bound, read before any store is asked. An absent or null field is
/// the refusal when `required` (the add-items route) and an empty batch
/// otherwise (the create route, whose `items` are optional).
fn items_of(body: &Value, required: bool) -> Result<&[Value], Response> {
    let items_array: &[Value] = match body.get("items") {
        Some(Value::Array(items)) => items,
        None | Some(Value::Null) if !required => &[],
        _ => return Err(bad_request("Missing or invalid 'items' field")),
    };

    if items_array.len() > MAX_ITEMS_PER_REQUEST {
        return Err(bad_request(format!(
            "Cannot add more than {MAX_ITEMS_PER_REQUEST} items at a time"
        )));
    }

    // Bounded before any store is asked, for the whole batch: the stores' id columns are
    // this wide, and a later item's over-long id must not leave earlier items behind.
    if let Some(id_str) = items_array
        .iter()
        .filter_map(|item| item.get("id").and_then(Value::as_str))
        .find(|id| id.len() > MAX_ITEM_ID_LENGTH)
    {
        return Err(item_id_above_max_length(id_str));
    }
    Ok(items_array)
}

/// A parsed item and the warning of an accepted-but-unimplemented type.
type OwnContent = (NewConversationItem, Option<String>);

/// One item of a request batch once the stores have been asked and before
/// anything is written: an item that exists already and is taken as it is (an
/// `item_reference`, an item arriving under an id the caller may see), a new
/// item to write with the warning of a type accepted but not implemented, or a
/// repeat of an earlier item of the same batch (the same client-supplied id
/// twice, a reference to an item the batch itself adds).
enum PreparedItem {
    /// An `item_reference`: the stored item it names.
    Existing(ConversationItem),
    /// An item adopted under its own client-supplied id: the stored item, and
    /// the request's own parsed content to write under that id should the
    /// batch's writes evict the stored one (`None` for a bare id the parser
    /// does not take).
    Adopted(ConversationItem, Option<Box<OwnContent>>),
    New(NewConversationItem, Option<String>),
    SameAs(usize),
}

/// An adopted item awaiting the re-check after the batch's writes: its id and
/// the request's own content.
struct Adoption {
    id: ConversationItemId,
    own: Option<Box<OwnContent>>,
}

/// The whole batch checked against the stores before any write, in the order
/// given: every refusal the routes give (the parser's, an unknown reference, an
/// id of another tenant, an item the conversation already has) is found here,
/// so a refused batch stores nothing on either route. `conversation_id` is
/// `None` while the conversation does not exist yet (the create route):
/// nothing can be linked to it.
async fn prepare_items(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conversation_id: Option<&ConversationId>,
    items: &[Value],
    tenant: Option<&str>,
) -> Result<Vec<PreparedItem>, Response> {
    let mut prepared = Vec::with_capacity(items.len());
    // The ids this batch brings itself (a new item under its own id): a later
    // item of the batch that names one is a repeat, not a lookup the store
    // cannot answer before the write.
    let mut batch_ids: HashMap<&str, usize> = HashMap::new();
    for item_val in items {
        let item_type = item_val
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        let id = item_val.get("id").and_then(Value::as_str);
        if let Some(index) = id.and_then(|id| batch_ids.get(id)) {
            prepared.push(PreparedItem::SameAs(*index));
            continue;
        }
        let item = if item_type == "item_reference" {
            prepare_item_reference(item_storage, conversation_id, id, tenant).await?
        } else if let Some(id_str) = id {
            prepare_item_with_id(item_storage, conversation_id, item_val, id_str, tenant).await?
        } else {
            let (new_item, warning) = parse_item_from_value(item_val, tenant)?;
            PreparedItem::New(new_item, warning)
        };
        if let (Some(id_str), PreparedItem::New(..)) = (id, &item) {
            batch_ids.insert(id_str, prepared.len());
        }
        prepared.push(item);
    }
    Ok(prepared)
}

/// An `item_reference`: the existing item it names, as the caller's `tenant`
/// may see it. A reference to an item the conversation already has is refused,
/// as an item arriving under such an id is: the public API does not re-add an
/// item a conversation has (the stores would move it to the new position).
async fn prepare_item_reference(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conversation_id: Option<&ConversationId>,
    id: Option<&str>,
    tenant: Option<&str>,
) -> Result<PreparedItem, Response> {
    let ref_id = id.ok_or_else(|| bad_request("item_reference requires 'id' field"))?;
    let item_id = ConversationItemId::from(ref_id);
    refuse_an_item_the_conversation_has(item_storage, conversation_id, &item_id).await?;

    // An item of another tenant is as unknown as a missing one.
    match item_storage.get_item(&item_id).await {
        Ok(Some(item)) if item.is_visible_to(tenant) => Ok(PreparedItem::Existing(item)),
        Ok(_) => Err(not_found(format!("Referenced item '{ref_id}' not found"))),
        Err(e) => Err(store_error("Failed to get referenced item", &e)),
    }
}

/// An item arriving under its own id: the caller's own (or an untenanted one)
/// is taken as it is, another tenant's is neither served nor overwritten, an
/// id the store does not hold is parsed as a new item under that id.
async fn prepare_item_with_id(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conversation_id: Option<&ConversationId>,
    item_val: &Value,
    id_str: &str,
    tenant: Option<&str>,
) -> Result<PreparedItem, Response> {
    // Bounded before any store is asked: the stores' id columns are this wide,
    // and an id over the bound must be the same 400 on every backend.
    if id_str.len() > MAX_ITEM_ID_LENGTH {
        return Err(item_id_above_max_length(id_str));
    }
    let item_id = ConversationItemId::from(id_str);
    refuse_an_item_the_conversation_has(item_storage, conversation_id, &item_id).await?;

    match item_storage.get_item(&item_id).await {
        Ok(Some(existing)) if existing.is_visible_to(tenant) => {
            // The request's own content, kept beside the adoption: written
            // under this id should the batch's writes evict the stored item.
            let own = parse_item_from_value(item_val, tenant)
                .ok()
                .map(|(mut item, warning)| {
                    item.id = Some(item_id.clone());
                    Box::new((item, warning))
                });
            Ok(PreparedItem::Adopted(existing, own))
        }
        Ok(Some(_)) => Err(bad_request(format!("Item id '{id_str}' cannot be used"))),
        Ok(None) => {
            let (mut new_item, warning) = parse_item_from_value(item_val, tenant)?;
            new_item.id = Some(item_id);
            Ok(PreparedItem::New(new_item, warning))
        }
        Err(e) => Err(store_error("Failed to check item existence", &e)),
    }
}

/// The refusal of an item a write under a caller-supplied id answered that the
/// caller may not see: the store keeps the item it already holds under an id,
/// so a caller that prepared the id as absent and lost the race to another
/// caller's write is answered that caller's item. It is refused as the prepare
/// step refuses such an id, before anything of the batch is linked.
fn refuse_an_item_the_caller_may_not_see(
    item: &ConversationItem,
    tenant: Option<&str>,
) -> Result<(), Response> {
    if item.is_visible_to(tenant) {
        Ok(())
    } else {
        Err(bad_request(format!(
            "Item id '{}' cannot be used",
            item.id.0
        )))
    }
}

/// The refusal of an item the conversation already has; nothing is linked to a
/// conversation that does not exist yet.
async fn refuse_an_item_the_conversation_has(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conversation_id: Option<&ConversationId>,
    item_id: &ConversationItemId,
) -> Result<(), Response> {
    let Some(conversation_id) = conversation_id else {
        return Ok(());
    };
    let is_linked = item_storage
        .is_item_linked(conversation_id, item_id)
        .await
        .map_err(|e| store_error("Failed to check item link", &e))?;
    if is_linked {
        return Err(item_already_in_conversation());
    }
    Ok(())
}

/// The prepared items of one request written and linked into `conversation_id`
/// in the order given: each item as the public API lists it, and the warnings
/// of the item types accepted but not implemented. Every refusal was found by
/// `prepare_items`; what ends the batch here is a store fault, or a referenced
/// item the batch's own writes evicted (a bounded store drops its oldest
/// records on a write): after a batch that wrote a new item, every reference
/// and adoption is read again before the link. A reference that is gone fails
/// the batch, so nothing of it is linked (its new records stay in the store
/// unlinked: the store's delete unlinks only, and a retry under the same id
/// adopts them); an adopted id that is gone takes the request's own content
/// under that id, as the write-as-you-go order answered, and every entry of
/// the answer that names that id (the adoption, a reference to it, a repeat
/// of it) lists the item as it is stored then. A batch without a new item
/// asks the stores nothing more. A write under a caller-supplied id answers
/// the item the store already holds under it, and the re-check reads whatever
/// item sits under an adopted or referenced id: one another caller wrote
/// since the batch was prepared, which `tenant` may not see, is refused
/// before any link, as the prepare step refuses such an id (an adoption) or
/// does not find it (a reference).
async fn store_items(
    item_storage: &Arc<dyn ConversationItemStorage>,
    conversation_id: &ConversationId,
    items: Vec<PreparedItem>,
    tenant: Option<&str>,
) -> Result<(Vec<Value>, Vec<String>), Response> {
    let mut created_items: Vec<Value> = Vec::with_capacity(items.len());
    let mut item_ids: Vec<ConversationItemId> = Vec::with_capacity(items.len());
    let mut warnings = Vec::new();
    let mut link_pairs = Vec::new();
    let mut seen_ids = HashSet::new();
    let mut referenced: Vec<ConversationItemId> = Vec::new();
    let mut adopted: Vec<Adoption> = Vec::new();
    let mut wrote = false;
    // One stamp per item, each later than the one before: the stores order a conversation's
    // items by `added_at`, so the items of this call list in the order the request gave them.
    let mut stamps = AddedAtSequence::new();

    for prepared in items {
        let added_at = stamps.next_added_at();
        let (item_json, item_id, warning) = match prepared {
            PreparedItem::Existing(item) => {
                referenced.push(item.id.clone());
                let id = item.id.clone();
                (item_to_json(item), id, None)
            }
            PreparedItem::Adopted(item, own) => {
                adopted.push(Adoption {
                    id: item.id.clone(),
                    own,
                });
                let id = item.id.clone();
                (item_to_json(item), id, None)
            }
            PreparedItem::New(new_item, warning) => {
                wrote = true;
                let created = item_storage
                    .create_item(new_item)
                    .await
                    .map_err(|e| store_error("Failed to create item", &e))?;
                refuse_an_item_the_caller_may_not_see(&created, tenant)?;
                let id = created.id.clone();
                (item_to_json(created), id, warning)
            }
            PreparedItem::SameAs(index) => {
                (created_items[index].clone(), item_ids[index].clone(), None)
            }
        };
        if seen_ids.insert(item_id.0.clone()) {
            link_pairs.push((item_id.clone(), added_at));
        }
        created_items.push(item_json);
        item_ids.push(item_id);
        if let Some(w) = warning {
            warnings.push(w);
        }
    }

    // The references and adoptions were resolved before the writes; a bounded
    // store may have dropped one of them to make room for this batch's new
    // items. An adopted id that is gone takes the request's own content (a
    // write of its own, read before the references), and every entry of the
    // answer that names that id lists the item as written; a reference that is
    // gone ends the batch before anything is linked, as a missing reference
    // does.
    if wrote {
        for Adoption { id: item_id, own } in adopted {
            match item_storage.get_item(&item_id).await {
                Ok(Some(item)) => refuse_an_item_the_caller_may_not_see(&item, tenant)?,
                Ok(None) => match own.map(|boxed| *boxed) {
                    Some((new_item, warning)) => {
                        let created = item_storage
                            .create_item(new_item)
                            .await
                            .map_err(|e| store_error("Failed to create item", &e))?;
                        refuse_an_item_the_caller_may_not_see(&created, tenant)?;
                        let written = item_to_json(created);
                        for (entry, id) in created_items.iter_mut().zip(&item_ids) {
                            if *id == item_id {
                                *entry = written.clone();
                            }
                        }
                        if let Some(w) = warning {
                            warnings.push(w);
                        }
                    }
                    None => return Err(not_found(format!("Item '{}' not found", item_id.0))),
                },
                Err(e) => return Err(store_error("Failed to re-check item", &e)),
            }
        }
        for item_id in &referenced {
            match item_storage.get_item(item_id).await {
                Ok(Some(item)) if item.is_visible_to(tenant) => {}
                Ok(_) => {
                    return Err(not_found(format!(
                        "Referenced item '{}' not found",
                        item_id.0
                    )))
                }
                Err(e) => return Err(store_error("Failed to re-check referenced item", &e)),
            }
        }
    }

    // Batch-link all items in a single operation
    item_storage
        .link_items(conversation_id, &link_pairs)
        .await
        .map_err(|e| store_error("Failed to link items to conversation", &e))?;
    Ok((created_items, warnings))
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
    tenant: Option<&str>,
    conv_id: &str,
    item_id: &str,
    _include: Option<Vec<String>>,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);
    let item_id = ConversationItemId::from(item_id);

    if let Err(response) =
        ensure_conversation_exists(conversation_storage, &conversation_id, tenant).await
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
        Ok(Some(item)) => (StatusCode::OK, Json(item_to_json(item))).into_response(),
        Ok(None) => not_found("Item not found"),
        Err(e) => store_error("Failed to get item", &e),
    }
}

pub async fn delete_conversation_item(
    conversation_storage: &Arc<dyn ConversationStorage>,
    item_storage: &Arc<dyn ConversationItemStorage>,
    tenant: Option<&str>,
    conv_id: &str,
    item_id: &str,
) -> Response {
    let conversation_id = ConversationId::from(conv_id);
    let item_id = ConversationItemId::from(item_id);

    let conversation =
        match ensure_conversation_exists(conversation_storage, &conversation_id, tenant).await {
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

/// A request item as the stores take it, with the warning of a type accepted
/// but not implemented; a refusal is the response the caller answers. A
/// message item's `role`, `status` and content parts are checked against the
/// public API's sets (`validate_message_fields`) before any store is asked.
fn parse_item_from_value(
    item_val: &Value,
    tenant: Option<&str>,
) -> Result<(NewConversationItem, Option<String>), Response> {
    let item_type = item_val
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message");

    if !SUPPORTED_ITEM_TYPES.contains(&item_type) {
        return Err(bad_request(format!(
            "Unsupported item type '{}'. Supported types: {}",
            item_type,
            SUPPORTED_ITEM_TYPES.join(", ")
        )));
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

    if item_type == "message" {
        if let Some(other) = item_val
            .get("role")
            .filter(|role| !role.is_null() && !role.is_string())
        {
            return Err(refused(
                Some("items[].role"),
                "invalid_type",
                &format!(
                    "Invalid type for 'items[].role': expected a string, but got {} instead.",
                    json_type_name(other)
                ),
            ));
        }
        if role.is_none() {
            return Err(bad_request("Message items require 'role' field"));
        }
        validate_message_fields(item_val)?;
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
            tenant: tenant.map(str::to_string),
        },
        warning,
    ))
}

/// A message item's `role`, `status` and `content` under the public API's
/// sets, refused in its words before any store is asked: the role through the
/// rule the Responses request validates with, the status in the message set,
/// the content a string or an array of the content parts the Responses
/// request takes (each part read with the typed part, so an unknown type, a
/// missing field and a wrong shape are refused as the typed extractors refuse
/// them). An item of another type keeps its fields as sent.
fn validate_message_fields(item_val: &Value) -> Result<(), Response> {
    if let Some(role) = item_val.get("role").and_then(Value::as_str) {
        validate_input_message_role(role).map_err(|error| {
            refused(
                Some("items[].role"),
                &error.code,
                error.message.as_deref().unwrap_or_default(),
            )
        })?;
    }
    if let Some(status) = item_val.get("status").filter(|status| !status.is_null()) {
        if !status
            .as_str()
            .is_some_and(|status| MESSAGE_STATUSES.contains(&status))
        {
            let shown = status
                .as_str()
                .map_or_else(|| status.to_string(), str::to_owned);
            return Err(refused(
                Some("items[].status"),
                "invalid_value",
                &format!(
                    "Invalid value: '{shown}'. Supported values are: 'in_progress', 'completed', and 'incomplete'."
                ),
            ));
        }
    }
    match item_val.get("content") {
        None | Some(Value::String(_)) => Ok(()),
        Some(Value::Array(parts)) => {
            for (index, part) in parts.iter().enumerate() {
                if let Err(error) = serde_json::from_value::<ResponseContentPart>(part.clone()) {
                    return Err(part_refused(index, part, &error.to_string()));
                }
            }
            Ok(())
        }
        Some(other) => Err(refused(
            Some("items[].content"),
            "invalid_type",
            &format!(
                "Invalid type for 'items[].content': expected a string or an array of content parts, but got {} instead.",
                json_type_name(other)
            ),
        )),
    }
}

/// The refusal of a message content part the typed part does not read, under
/// the public API's code for serde's reason, on the field the reason names
/// without a guess: a part that is not an object is `invalid_type`; one
/// without a `type` is `missing_required_parameter` on its type, one whose
/// `type` is not a string `invalid_type` on it, one whose `type` is outside
/// the part set `invalid_value` on it (a type the set knows is never blamed
/// for a nested variant of the same name); a missing field of a part that holds
/// nothing nested is `missing_required_parameter` on that field. serde names
/// a missing field or an unknown variant without its path, so an error that
/// may come from inside a nested value (an annotation's field, a `detail`
/// outside its set) names the part itself, under the code of its kind.
fn part_refused(index: usize, part: &Value, reason: &str) -> Response {
    let part_param = format!("items[].content[{index}]");
    let missing_field = reason
        .strip_prefix("missing field `")
        .and_then(|rest| rest.split_once('`'))
        .map(|(field, _)| field);
    let (param, code) = match part.get("type") {
        _ if !part.is_object() => (part_param.clone(), "invalid_type"),
        None | Some(Value::Null) => (format!("{part_param}.type"), "missing_required_parameter"),
        Some(Value::String(own)) if unknown_variant_is(reason, own) && !part_type_is_known(own) => {
            (format!("{part_param}.type"), "invalid_value")
        }
        Some(Value::String(_)) => {
            let nested = part
                .as_object()
                .is_some_and(|fields| fields.values().any(|v| v.is_object() || v.is_array()));
            match missing_field {
                Some(field) if !nested => (
                    format!("{part_param}.{field}"),
                    "missing_required_parameter",
                ),
                Some(_) => (part_param.clone(), "missing_required_parameter"),
                None if reason.starts_with("unknown variant") => {
                    (part_param.clone(), "invalid_value")
                }
                None => (part_param.clone(), "invalid_type"),
            }
        }
        Some(_) => (format!("{part_param}.type"), "invalid_type"),
    };
    refused(
        Some(&param),
        code,
        &format!("Invalid '{part_param}': {reason}"),
    )
}

/// Whether the part set knows `type_name`: a part of that type alone fails on a
/// missing field (or passes), never on an unknown variant, so an unknown-variant
/// reason on such a part comes from a value nested inside it.
fn part_type_is_known(type_name: &str) -> bool {
    match serde_json::from_value::<ResponseContentPart>(json!({ "type": type_name })) {
        Ok(_) => true,
        Err(error) => !error.to_string().starts_with("unknown variant"),
    }
}

/// Whether serde's reason is the unknown-variant refusal of `name` itself
/// (``unknown variant `name`, expected ...``), not of a value nested in it.
fn unknown_variant_is(reason: &str, name: &str) -> bool {
    reason
        .strip_prefix("unknown variant `")
        .and_then(|rest| rest.strip_prefix(name))
        .is_some_and(|rest| rest.starts_with('`'))
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
mod tests;
