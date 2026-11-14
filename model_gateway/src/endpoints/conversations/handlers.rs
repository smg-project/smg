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
    let metadata = match validate_metadata(&body) {
        Ok(m) => m,
        Err(msg) => return bad_request(msg),
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
    let conversation_id = ConversationId::from(conv_id);

    let current = match ensure_conversation_exists(storage, &conversation_id, tenant).await {
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
            let (new_item, warning) =
                parse_item_from_value(item_val, tenant).map_err(bad_request)?;
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
            let (mut new_item, warning) =
                parse_item_from_value(item_val, tenant).map_err(bad_request)?;
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

fn parse_item_from_value(
    item_val: &Value,
    tenant: Option<&str>,
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
            tenant: tenant.map(str::to_string),
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
    use smg_data_connector::{
        MemoryConfig, MemoryConversationItemStorage, MemoryConversationStorage,
    };

    use super::*;
    use crate::middleware::test_support::json_of;

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
                tenant: None,
            })
            .await
            .unwrap();
        let conv_id = conversation.id.0.clone();
        let items = |id: &str| json!({"items": [{"type": "message", "role": "user", "content": "x", "id": id}]});

        // 65 and 64 characters: one over the stores' 64-character id columns, one at the bound.
        let over = format!("msg_{}", "a".repeat(61));
        assert_eq!(over.len(), 65);
        let response = create_conversation_items(
            &conversation_storage,
            &item_storage,
            None,
            &conv_id,
            items(&over),
        )
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
            create_conversation_items(&conversation_storage, &item_storage, None, &conv_id, batch)
                .await;
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
            None,
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

    /// A conversation created with `items` lists them, in the order given, as
    /// a conversation created empty and given the same items does; the
    /// accepted-but-unimplemented types answer the add-items route's
    /// `warnings`; a batch the add-items route refuses before any store (over
    /// the bound, an unsupported type, a malformed field) creates no
    /// conversation, and one the store refuses leaves none behind.
    #[tokio::test]
    async fn a_conversation_created_with_items_lists_them_as_the_add_items_route_stores_them() {
        let conversations: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let items: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::new());
        let two = json!([
            {"type": "message", "role": "user", "content": "first"},
            {"type": "message", "role": "assistant", "content": "second"}
        ]);
        let shape = |listed: &Value| -> Vec<Value> {
            listed["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| json!([item["type"], item["role"], item["content"], item["status"]]))
                .collect()
        };

        let response = create_conversation(
            &conversations,
            &items,
            None,
            json!({"items": two, "metadata": {"a": "b"}}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let created = json_of(response).await;
        assert_eq!(created["object"], "conversation", "{created}");
        assert_eq!(created["metadata"]["a"], "b", "{created}");
        assert!(created.get("warnings").is_none(), "{created}");
        let with_items = created["id"].as_str().unwrap().to_string();
        let listed = json_of(
            list_conversation_items(
                &conversations,
                &items,
                None,
                &with_items,
                None,
                Some("asc"),
                None,
            )
            .await,
        )
        .await;
        assert_eq!(listed["data"].as_array().map(Vec::len), Some(2), "{listed}");
        assert_eq!(listed["data"][0]["content"][0]["text"], "first", "{listed}");
        assert_eq!(
            listed["data"][1]["content"][0]["text"], "second",
            "{listed}"
        );

        let with_null =
            create_conversation(&conversations, &items, None, json!({"items": null})).await;
        assert_eq!(
            with_null.status(),
            StatusCode::OK,
            "a null items field is an absent one"
        );
        let empty =
            json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
        let then_added = empty["id"].as_str().unwrap().to_string();
        let added = create_conversation_items(
            &conversations,
            &items,
            None,
            &then_added,
            json!({"items": two}),
        )
        .await;
        assert_eq!(added.status(), StatusCode::OK);
        let listed_after = json_of(
            list_conversation_items(
                &conversations,
                &items,
                None,
                &then_added,
                None,
                Some("asc"),
                None,
            )
            .await,
        )
        .await;
        assert_eq!(
            shape(&listed),
            shape(&listed_after),
            "{listed}\n{listed_after}"
        );

        let response = create_conversation(
            &conversations,
            &items,
            None,
            json!({"items": [{"type": "web_search_call", "id": "ws_1", "status": "completed"}]}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let warned = json_of(response).await;
        assert_eq!(
            warned["warnings"].as_array().map(Vec::len),
            Some(1),
            "{warned}"
        );

        let before = conversations.stored_record_count();
        let items_before = items.stored_record_count();
        let too_many: Vec<Value> = (0..=MAX_ITEMS_PER_REQUEST)
            .map(|i| json!({"type": "message", "role": "user", "content": format!("m{i}")}))
            .collect();
        for (body, status) in [
            (json!({"items": too_many}), StatusCode::BAD_REQUEST),
            (
                json!({"items": [{"type": "bogus"}]}),
                StatusCode::BAD_REQUEST,
            ),
            (
                json!({"items": [{"type": "message", "content": "no role"}]}),
                StatusCode::BAD_REQUEST,
            ),
            (json!({"items": "first"}), StatusCode::BAD_REQUEST),
            (
                json!({"items": [{"type": "item_reference", "id": "msg_nowhere"}]}),
                StatusCode::NOT_FOUND,
            ),
        ] {
            let response = create_conversation(&conversations, &items, None, body.clone()).await;
            assert_eq!(response.status(), status, "{body}");
            assert_eq!(
                conversations.stored_record_count(),
                before,
                "a refused create creates nothing: {body}"
            );
            assert_eq!(
                items.stored_record_count(),
                items_before,
                "a refused create stores no item: {body}"
            );
        }

        // An id the caller already holds is adopted without a parse, as the
        // add-items route adopts it: no role is asked of it.
        let held = listed["data"][0]["id"].as_str().unwrap().to_string();
        let adopted = create_conversation(
            &conversations,
            &items,
            None,
            json!({"items": [{"id": held}]}),
        )
        .await;
        assert_eq!(adopted.status(), StatusCode::OK);
    }

    /// A batch refused after its first item (an unknown reference, an
    /// unsupported type) stores nothing on either route: the whole batch is
    /// prepared before the first write, so a corrected retry under the same id
    /// is a fresh write with the retry's content.
    #[tokio::test]
    async fn a_refused_batch_stores_nothing_on_either_route() {
        let conversations: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let items: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::new());
        let empty =
            json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
        let conversation = empty["id"].as_str().unwrap().to_string();
        let conversations_before = conversations.stored_record_count();
        let items_before = items.stored_record_count();
        let first = json!({"id": "msg_first", "type": "message", "role": "user", "content": "A"});

        for (second, status) in [
            (
                json!({"type": "item_reference", "id": "msg_nowhere"}),
                StatusCode::NOT_FOUND,
            ),
            (json!({"type": "bogus"}), StatusCode::BAD_REQUEST),
        ] {
            let body = json!({"items": [first, second]});
            let on_create = create_conversation(&conversations, &items, None, body.clone()).await;
            assert_eq!(on_create.status(), status, "{body}");
            let on_add = create_conversation_items(
                &conversations,
                &items,
                None,
                &conversation,
                body.clone(),
            )
            .await;
            assert_eq!(on_add.status(), status, "{body}");
            assert_eq!(
                conversations.stored_record_count(),
                conversations_before,
                "no conversation: {body}"
            );
            assert_eq!(items.stored_record_count(), items_before, "no item: {body}");
        }

        let retried = create_conversation_items(
            &conversations,
            &items,
            None,
            &conversation,
            json!({"items": [{"id": "msg_first", "type": "message", "role": "user", "content": "B"}]}),
        )
        .await;
        assert_eq!(retried.status(), StatusCode::OK);
        let listed = json_of(
            list_conversation_items(
                &conversations,
                &items,
                None,
                &conversation,
                None,
                Some("asc"),
                None,
            )
            .await,
        )
        .await;
        assert_eq!(listed["data"][0]["content"][0]["text"], "B", "{listed}");
    }

    const A: Option<&str> = Some("header:tenant-a");
    const B: Option<&str> = Some("header:tenant-b");

    /// A conversation created by tenant a with one item; the stores and the ids.
    async fn conversation_of_a() -> (
        Arc<dyn ConversationStorage>,
        Arc<dyn ConversationItemStorage>,
        String,
        String,
    ) {
        let conversations: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let items: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::new());
        let created =
            json_of(create_conversation(&conversations, &items, A, json!({})).await).await;
        let conv_id = created["id"].as_str().unwrap().to_string();
        let added = json_of(
            create_conversation_items(
                &conversations,
                &items,
                A,
                &conv_id,
                json!({"items": [{"type": "message", "role": "user", "content": "secret"}]}),
            )
            .await,
        )
        .await;
        let item_id = added["data"][0]["id"].as_str().unwrap().to_string();
        (conversations, items, conv_id, item_id)
    }

    /// A conversation created under a tenant is read, listed, written into and
    /// deleted by that tenant only; another tenant, and a caller without one,
    /// get the answer of an unknown id on every route and change nothing.
    #[tokio::test]
    async fn another_tenants_conversation_answers_not_found_on_every_route() {
        let (conversations, items, conv_id, item_id) = conversation_of_a().await;
        let stored = conversations
            .get_conversation(&ConversationId::from(conv_id.as_str()))
            .await
            .unwrap()
            .expect("created");
        assert_eq!(
            stored.tenant.as_deref(),
            A,
            "the conversation carries its tenant"
        );

        for caller in [B, None] {
            let answers = [
                get_conversation(&conversations, caller, &conv_id).await,
                update_conversation(
                    &conversations,
                    caller,
                    &conv_id,
                    json!({"metadata": {"k": "v"}}),
                )
                .await,
                list_conversation_items(&conversations, &items, caller, &conv_id, None, None, None)
                    .await,
                create_conversation_items(
                    &conversations,
                    &items,
                    caller,
                    &conv_id,
                    json!({"items": [{"type": "message", "role": "user", "content": "intruder"}]}),
                )
                .await,
                get_conversation_item(&conversations, &items, caller, &conv_id, &item_id, None)
                    .await,
                delete_conversation_item(&conversations, &items, caller, &conv_id, &item_id).await,
                delete_conversation(&conversations, caller, &conv_id).await,
            ];
            for answer in answers {
                assert_eq!(answer.status(), StatusCode::NOT_FOUND, "{caller:?}");
            }
        }

        // Nothing changed for its owner: the one item is still there, the
        // conversation too, and the owner's own calls work.
        let listed = json_of(
            list_conversation_items(&conversations, &items, A, &conv_id, None, None, None).await,
        )
        .await;
        assert_eq!(listed["data"].as_array().unwrap().len(), 1);
        assert_eq!(
            get_conversation_item(&conversations, &items, A, &conv_id, &item_id, None)
                .await
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            delete_conversation_item(&conversations, &items, A, &conv_id, &item_id)
                .await
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            delete_conversation(&conversations, A, &conv_id)
                .await
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            get_conversation(&conversations, A, &conv_id).await.status(),
            StatusCode::NOT_FOUND
        );
    }

    /// An item of another tenant is as unknown as a missing one to an
    /// `item_reference`, and an item arriving under that id is refused rather
    /// than served or overwritten; the owner references its own item freely.
    #[tokio::test]
    async fn another_tenants_item_cannot_be_referenced_or_adopted() {
        let (conversations, items, _conv_a, item_a) = conversation_of_a().await;
        let created =
            json_of(create_conversation(&conversations, &items, B, json!({})).await).await;
        let conv_b = created["id"].as_str().unwrap().to_string();

        let reference = json!({"items": [{"type": "item_reference", "id": item_a}]});
        let answer = create_conversation_items(&conversations, &items, B, &conv_b, reference).await;
        assert_eq!(answer.status(), StatusCode::NOT_FOUND);
        let adopt = json!({"items": [{"type": "message", "id": item_a, "role": "user", "content": "mine now"}]});
        let answer = create_conversation_items(&conversations, &items, B, &conv_b, adopt).await;
        assert_eq!(answer.status(), StatusCode::BAD_REQUEST);
        // a's item is untouched and still a's.
        let stored = items
            .get_item(&ConversationItemId::from(item_a.as_str()))
            .await
            .unwrap()
            .expect("still there");
        assert_eq!(stored.tenant.as_deref(), A);
        assert_eq!(stored.content[0]["text"], "secret");
        let listed = json_of(
            list_conversation_items(&conversations, &items, B, &conv_b, None, None, None).await,
        )
        .await;
        assert_eq!(listed["data"].as_array().map(Vec::len), Some(0));

        // The owner references its own item into a second conversation of its own.
        let created =
            json_of(create_conversation(&conversations, &items, A, json!({})).await).await;
        let conv_a2 = created["id"].as_str().unwrap().to_string();
        let reference = json!({"items": [{"type": "item_reference", "id": item_a}]});
        let answer =
            create_conversation_items(&conversations, &items, A, &conv_a2, reference).await;
        assert_eq!(answer.status(), StatusCode::OK);
    }

    /// Without tenant identity nothing changes: a conversation created by a
    /// caller without a tenant is served to every caller, as before.
    #[tokio::test]
    async fn a_conversation_without_a_tenant_is_served_to_every_caller() {
        let conversations: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let items: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::new());
        let created =
            json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
        let conv_id = created["id"].as_str().unwrap().to_string();
        for caller in [None, A] {
            assert_eq!(
                get_conversation(&conversations, caller, &conv_id)
                    .await
                    .status(),
                StatusCode::OK,
                "{caller:?}"
            );
            assert_eq!(
                list_conversation_items(&conversations, &items, caller, &conv_id, None, None, None)
                    .await
                    .status(),
                StatusCode::OK,
                "{caller:?}"
            );
        }
        assert_eq!(
            delete_conversation(&conversations, B, &conv_id)
                .await
                .status(),
            StatusCode::OK
        );
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

    /// A reference to an item the same batch adds, and a new id given twice in
    /// one batch, store the item once and list it once on both routes: the
    /// batch's own ids are known before any write, so no store is asked for an
    /// item that is not written yet; an accepted-but-unimplemented type warns
    /// once.
    #[tokio::test]
    async fn a_reference_to_an_item_of_the_same_batch_and_a_repeated_id_store_the_item_once() {
        let conversations: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let items: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::new());
        let empty =
            json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
        let conversation = empty["id"].as_str().unwrap().to_string();
        let message =
            |id: &str| json!({"id": id, "type": "message", "role": "user", "content": "x"});
        let ids_of = |listed: &Value| -> Vec<String> {
            listed["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item["id"].as_str().unwrap().to_string())
                .collect()
        };

        // The create route: a reference to the item the batch itself adds.
        let before = items.stored_record_count().unwrap();
        let created = create_conversation(
            &conversations,
            &items,
            None,
            json!({"items": [message("msg_a"), {"type": "item_reference", "id": "msg_a"}]}),
        )
        .await;
        assert_eq!(created.status(), StatusCode::OK);
        let created = json_of(created).await;
        assert!(created.get("warnings").is_none(), "{created}");
        let with_items = created["id"].as_str().unwrap().to_string();
        assert_eq!(items.stored_record_count(), Some(before + 1));
        let listed = json_of(
            list_conversation_items(
                &conversations,
                &items,
                None,
                &with_items,
                None,
                Some("asc"),
                None,
            )
            .await,
        )
        .await;
        assert_eq!(ids_of(&listed), vec!["msg_a"], "{listed}");

        // The add-items route: the same new id twice, then an accepted-but-unimplemented
        // type referenced by its own batch (one warning, not two).
        for (batch, id, warnings) in [
            (json!([message("msg_b"), message("msg_b")]), "msg_b", 0),
            (
                json!([
                    {"id": "ws_c", "type": "web_search_call", "status": "completed"},
                    {"type": "item_reference", "id": "ws_c"}
                ]),
                "ws_c",
                1,
            ),
        ] {
            let before = items.stored_record_count().unwrap();
            let added = create_conversation_items(
                &conversations,
                &items,
                None,
                &conversation,
                json!({"items": batch}),
            )
            .await;
            assert_eq!(added.status(), StatusCode::OK, "{id}");
            let added = json_of(added).await;
            assert_eq!(added["data"].as_array().map(Vec::len), Some(2), "{added}");
            assert_eq!(added["data"][1]["id"], id, "{added}");
            assert_eq!(
                added["warnings"].as_array().map(Vec::len).unwrap_or(0),
                warnings,
                "{added}"
            );
            assert_eq!(items.stored_record_count(), Some(before + 1), "{id}");
        }
        let listed = json_of(
            list_conversation_items(
                &conversations,
                &items,
                None,
                &conversation,
                None,
                Some("asc"),
                None,
            )
            .await,
        )
        .await;
        assert_eq!(ids_of(&listed), vec!["msg_b", "ws_c"], "{listed}");
    }

    /// A reference resolved before the writes can name an item a bounded
    /// store drops to make room for the batch's own new items. Every reference
    /// is read again after the writes: one that is gone fails the batch before
    /// anything is linked, on both routes, with the answer of a missing
    /// reference.
    #[tokio::test]
    async fn a_reference_the_batch_s_own_writes_evict_fails_the_batch_before_the_link() {
        for route in ["create", "add-items"] {
            let conversations: Arc<dyn ConversationStorage> =
                Arc::new(MemoryConversationStorage::new());
            let two_items = MemoryConfig {
                max_items: 2,
                ..MemoryConfig::default()
            };
            let items: Arc<dyn ConversationItemStorage> =
                Arc::new(MemoryConversationItemStorage::with_config(&two_items));
            let stored = |id: &str| NewConversationItem {
                id: Some(ConversationItemId::from(id)),
                response_id: None,
                item_type: "message".to_string(),
                role: Some("user".to_string()),
                content: json!([{"type": "input_text", "text": id}]),
                status: None,
                tenant: None,
            };
            items.create_item(stored("msg_one")).await.unwrap();
            items.create_item(stored("msg_two")).await.unwrap();
            let empty =
                json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
            let conversation = empty["id"].as_str().unwrap().to_string();
            let conversations_before = conversations.stored_record_count();

            // The new item takes the store over its bound: one of the two
            // stored items goes, and the batch references both.
            let batch = json!({"items": [
                {"id": "msg_new", "type": "message", "role": "user", "content": "new"},
                {"type": "item_reference", "id": "msg_one"},
                {"type": "item_reference", "id": "msg_two"}
            ]});
            let response = match route {
                "create" => create_conversation(&conversations, &items, None, batch).await,
                _ => {
                    create_conversation_items(&conversations, &items, None, &conversation, batch)
                        .await
                }
            };
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{route}");
            let body = json_of(response).await;
            let message = body["error"]["message"].as_str().unwrap_or_default();
            assert!(
                message.contains("msg_one") || message.contains("msg_two"),
                "{route}: the answer names the evicted reference: {body}"
            );
            let kept = [
                items
                    .get_item(&ConversationItemId::from("msg_one"))
                    .await
                    .unwrap(),
                items
                    .get_item(&ConversationItemId::from("msg_two"))
                    .await
                    .unwrap(),
            ];
            assert_eq!(
                kept.iter().filter(|item| item.is_some()).count(),
                1,
                "{route}: one stored item went to make room"
            );
            assert_eq!(
                conversations.stored_record_count(),
                conversations_before,
                "{route}: no conversation is left behind"
            );
            let listed = json_of(
                list_conversation_items(
                    &conversations,
                    &items,
                    None,
                    &conversation,
                    None,
                    Some("asc"),
                    None,
                )
                .await,
            )
            .await;
            assert_eq!(
                listed["data"].as_array().map(Vec::len),
                Some(0),
                "{route}: {listed}"
            );
            assert!(
                !items
                    .is_item_linked(
                        &ConversationId::from(conversation.as_str()),
                        &ConversationItemId::from("msg_new")
                    )
                    .await
                    .unwrap(),
                "{route}: nothing of the batch is linked"
            );
        }
    }

    /// An item adopted under its own id is not a reference: when the batch's
    /// writes evict the stored item, the request's own content is written
    /// under that id (the answer the write-as-you-go order gave), on both
    /// routes; a bare id that is gone answers as a missing item.
    #[tokio::test]
    async fn an_adopted_id_the_batch_s_own_writes_evict_takes_the_request_s_content() {
        let two_items = MemoryConfig {
            max_items: 2,
            ..MemoryConfig::default()
        };
        let stored = |id: &str| NewConversationItem {
            id: Some(ConversationItemId::from(id)),
            response_id: None,
            item_type: "message".to_string(),
            role: Some("user".to_string()),
            content: json!([{"type": "input_text", "text": "stored"}]),
            status: None,
            tenant: None,
        };
        for route in ["create", "add-items"] {
            let conversations: Arc<dyn ConversationStorage> =
                Arc::new(MemoryConversationStorage::new());
            let items: Arc<dyn ConversationItemStorage> =
                Arc::new(MemoryConversationItemStorage::with_config(&two_items));
            // The oldest record goes first: the two stored items get distinct stamps.
            items.create_item(stored("msg_oldest")).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            items.create_item(stored("msg_other")).await.unwrap();
            let empty =
                json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
            let conversation = empty["id"].as_str().unwrap().to_string();
            let batch = json!({"items": [
                {"type": "message", "role": "user", "content": "new"},
                {"id": "msg_oldest", "type": "message", "role": "user", "content": "own"}
            ]});
            let response = match route {
                "create" => create_conversation(&conversations, &items, None, batch).await,
                _ => {
                    create_conversation_items(&conversations, &items, None, &conversation, batch)
                        .await
                }
            };
            assert_eq!(response.status(), StatusCode::OK, "{route}");
            let body = json_of(response).await;
            let listed_in = if route == "create" {
                body["id"].as_str().unwrap().to_string()
            } else {
                assert_eq!(
                    body["data"][1]["content"][0]["text"], "own",
                    "{route}: {body}"
                );
                conversation.clone()
            };
            let listed = json_of(
                list_conversation_items(
                    &conversations,
                    &items,
                    None,
                    &listed_in,
                    None,
                    Some("asc"),
                    None,
                )
                .await,
            )
            .await;
            let texts: Vec<&str> = listed["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item["content"][0]["text"].as_str().unwrap_or(""))
                .collect();
            assert_eq!(texts, vec!["new", "own"], "{route}: {listed}");
            assert_eq!(listed["data"][1]["id"], "msg_oldest", "{route}: {listed}");
        }

        // A bare id (nothing the parser takes) adopted and then evicted by the
        // batch's write answers as a missing item, not a reference.
        let conversations: Arc<dyn ConversationStorage> =
            Arc::new(MemoryConversationStorage::new());
        let items: Arc<dyn ConversationItemStorage> =
            Arc::new(MemoryConversationItemStorage::with_config(&two_items));
        items.create_item(stored("msg_oldest")).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        items.create_item(stored("msg_other")).await.unwrap();
        let empty =
            json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
        let conversation = empty["id"].as_str().unwrap().to_string();
        let response = create_conversation_items(
            &conversations,
            &items,
            None,
            &conversation,
            json!({"items": [
                {"type": "message", "role": "user", "content": "new"},
                {"id": "msg_oldest"}
            ]}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = json_of(response).await;
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("msg_oldest") && !message.contains("Referenced"),
            "{body}"
        );
    }
    /// An adopted id the batch's writes evict is written once with the
    /// request's own content, and every entry of the answer that names it (the
    /// adoption, a reference to it, a repeat of it) lists the item as the
    /// listing does, on both routes.
    #[tokio::test]
    async fn a_recreated_adopted_id_lists_the_same_in_every_entry_of_the_answer() {
        let two_items = MemoryConfig {
            max_items: 2,
            ..MemoryConfig::default()
        };
        let stored = |id: &str| NewConversationItem {
            id: Some(ConversationItemId::from(id)),
            response_id: None,
            item_type: "message".to_string(),
            role: Some("user".to_string()),
            content: json!([{"type": "input_text", "text": "stored"}]),
            status: None,
            tenant: None,
        };
        for route in ["create", "add-items"] {
            let conversations: Arc<dyn ConversationStorage> =
                Arc::new(MemoryConversationStorage::new());
            let items: Arc<dyn ConversationItemStorage> =
                Arc::new(MemoryConversationItemStorage::with_config(&two_items));
            // The oldest record goes first: the two stored items get distinct stamps.
            items.create_item(stored("msg_oldest")).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            items.create_item(stored("msg_other")).await.unwrap();
            let empty =
                json_of(create_conversation(&conversations, &items, None, json!({})).await).await;
            let conversation = empty["id"].as_str().unwrap().to_string();
            // The new item evicts the oldest; the batch then adopts that id, references it and
            // adopts it again.
            let batch = json!({"items": [
                {"type": "message", "role": "user", "content": "new"},
                {"id": "msg_oldest", "type": "message", "role": "user", "content": "own"},
                {"type": "item_reference", "id": "msg_oldest"},
                {"id": "msg_oldest", "type": "message", "role": "user", "content": "own again"}
            ]});
            let response = match route {
                "create" => create_conversation(&conversations, &items, None, batch).await,
                _ => {
                    create_conversation_items(&conversations, &items, None, &conversation, batch)
                        .await
                }
            };
            assert_eq!(response.status(), StatusCode::OK, "{route}");
            let body = json_of(response).await;
            let listed_in = if route == "create" {
                body["id"].as_str().unwrap().to_string()
            } else {
                let texts: Vec<&str> = body["data"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item["content"][0]["text"].as_str().unwrap_or(""))
                    .collect();
                assert_eq!(texts, vec!["new", "own", "own", "own"], "{route}: {body}");
                conversation.clone()
            };
            let listed = json_of(
                list_conversation_items(
                    &conversations,
                    &items,
                    None,
                    &listed_in,
                    None,
                    Some("asc"),
                    None,
                )
                .await,
            )
            .await;
            let texts: Vec<&str> = listed["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item["content"][0]["text"].as_str().unwrap_or(""))
                .collect();
            assert_eq!(texts, vec!["new", "own"], "{route}: {listed}");
            assert_eq!(listed["data"][1]["id"], "msg_oldest", "{route}: {listed}");
            assert!(
                items
                    .get_item(&ConversationItemId::from("msg_other"))
                    .await
                    .unwrap()
                    .is_none(),
                "{route}: the other stored item went to make room for the recreation"
            );
        }
    }

    /// The post-write re-check applies the prepare step's visibility rule: when
    /// the batch's own writes evicted an adopted or referenced item and another
    /// caller's write took the id before the re-check, the item now under the id
    /// is refused as the prepare step refuses it (an adoption's 400, a
    /// reference's 404) and nothing of the batch is linked; the other caller's
    /// item is untouched.
    #[tokio::test]
    async fn an_id_another_caller_took_before_the_re_check_is_refused_before_the_link() {
        for (case, entry, status) in [
            (
                "adoption",
                json!({"id": "msg_o", "type": "message", "role": "user", "content": "own"}),
                StatusCode::BAD_REQUEST,
            ),
            (
                "reference",
                json!({"type": "item_reference", "id": "msg_o"}),
                StatusCode::NOT_FOUND,
            ),
        ] {
            let conversations: Arc<dyn ConversationStorage> =
                Arc::new(MemoryConversationStorage::new());
            let two_items = MemoryConfig {
                max_items: 2,
                ..MemoryConfig::default()
            };
            let items: Arc<dyn ConversationItemStorage> =
                Arc::new(MemoryConversationItemStorage::with_config(&two_items));
            let stored = |id: &str, tenant: Option<&str>| NewConversationItem {
                id: Some(ConversationItemId::from(id)),
                response_id: None,
                item_type: "message".to_string(),
                role: Some("user".to_string()),
                content: json!([{"type": "input_text", "text": id}]),
                status: None,
                tenant: tenant.map(str::to_string),
            };
            items.create_item(stored("msg_o", A)).await.unwrap();
            items.create_item(stored("msg_u", A)).await.unwrap();
            let created =
                json_of(create_conversation(&conversations, &items, A, json!({})).await).await;
            let conv_id = created["id"].as_str().unwrap().to_string();
            let conversation = ConversationId::from(conv_id.as_str());

            // The batch is prepared while msg_o is a's own item.
            let batch = [
                json!({"id": "msg_new", "type": "message", "role": "user", "content": "new"}),
                entry,
            ];
            let prepared = match prepare_items(&items, Some(&conversation), &batch, A).await {
                Ok(prepared) => prepared,
                Err(response) => panic!("{case}: prepared: {}", json_of(response).await),
            };
            // Between the prepare and the writes: a write evicts msg_o (the
            // oldest record of the bounded store) and another caller's write
            // takes the id.
            items.create_item(stored("msg_filler", A)).await.unwrap();
            items.create_item(stored("msg_o", B)).await.unwrap();

            let response = store_items(&items, &conversation, prepared, A)
                .await
                .expect_err("the id now holds another caller's item");
            assert_eq!(response.status(), status, "{case}");
            let body = json_of(response).await;
            assert!(
                body["error"]["message"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("msg_o"),
                "{case}: the answer names the id: {body}"
            );
            for id in ["msg_o", "msg_new"] {
                assert!(
                    !items
                        .is_item_linked(&conversation, &ConversationItemId::from(id))
                        .await
                        .unwrap(),
                    "{case}: {id} is not linked"
                );
            }
            let listed = json_of(
                list_conversation_items(
                    &conversations,
                    &items,
                    A,
                    &conv_id,
                    None,
                    Some("asc"),
                    None,
                )
                .await,
            )
            .await;
            assert_eq!(
                listed["data"].as_array().map(Vec::len),
                Some(0),
                "{case}: {listed}"
            );
            let other = items
                .get_item(&ConversationItemId::from("msg_o"))
                .await
                .unwrap()
                .expect("the other caller's item stands");
            assert_eq!(other.tenant.as_deref(), B, "{case}");
            assert_eq!(other.content[0]["text"], "msg_o", "{case}");
        }
    }
}
