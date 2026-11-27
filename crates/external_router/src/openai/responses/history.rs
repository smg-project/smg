//! Input history loading for the Responses API.
//!
//! Loads conversation history and/or previous response chains into the request
//! input before forwarding to the upstream provider.

use std::{collections::HashSet, fmt::Display, sync::Arc};

use axum::response::Response;
use openai_protocol::{
    event_types::ItemType,
    responses::{
        from_turn_boundary, without_calls_answered_by, without_outputs_of_missing_calls,
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesRequest,
    },
};
use serde::Deserialize;
use serde_json::Value;
use smg_data_connector::{
    ConversationId, ConversationItem, ListParams, ResponseId, ResponseStorageError, SortOrder,
    StorageFailure, StoredResponse,
};
use tracing::warn;

use super::super::context::ResponsesComponents;
use crate::{error, metrics::metrics_labels, persistence_utils::split_stored_message_content};

/// How many of a conversation's stored items a request replays: its newest
/// ones, oldest first; what is older is not sent. A window this limit cuts
/// can begin inside a turn, so it begins at its first turn boundary (a
/// message the client sent, or a call with its output) and the outputs of
/// the calls before the window are left out of it.
const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;

pub(crate) struct LoadedInputHistory {
    pub previous_response_id: Option<String>,
    pub existing_mcp_list_tools_labels: Vec<String>,
}

/// Load conversation history and/or previous response chain into request input.
///
/// Mutates `request_body.input` with the loaded items. `tenant` is the
/// caller's tenant: a previous response or a conversation of another tenant
/// answers as a missing one does.
/// Returns `Ok(LoadedInputHistory)` on success, or `Err(response)` on validation failure.
pub(crate) async fn load_input_history(
    components: &ResponsesComponents,
    conversation: Option<&str>,
    request_body: &mut ResponsesRequest,
    model: &str,
    tenant: Option<&str>,
) -> Result<LoadedInputHistory, Response> {
    let previous_response_id = request_body
        .previous_response_id
        .take()
        .filter(|id| !id.is_empty());
    let mut existing_mcp_list_tools_labels = HashSet::new();

    // Load items from previous response chain if specified
    let mut chain_items: Option<Vec<ResponseInputOutputItem>> = None;
    if let Some(prev_id_str) = &previous_response_id {
        let responses = visible_previous_chain(components, prev_id_str, tenant, model).await?;
        existing_mcp_list_tools_labels.extend(responses.iter().flat_map(|stored| {
            extract_mcp_list_tools_labels(stored.raw_response.get("output").unwrap_or(&Value::Null))
        }));
        chain_items = Some(chain_input_items(&responses));
    }

    // Load conversation history if specified
    if let Some(conv_id_str) = conversation {
        let conv_id = ConversationId::from(conv_id_str);
        check_conversation_visible(components, &conv_id, tenant, model).await?;
        let mut items = conversation_window_items(components, &conv_id, tenant, model).await?;
        let own_input =
            std::mem::replace(&mut request_body.input, ResponseInput::Items(Vec::new()));
        append_current_input(&mut items, own_input, conv_id_str);
        request_body.input = ResponseInput::Items(items);
    }

    // Apply previous response chain items if loaded.
    // Note: conversation and previous_response_id are mutually exclusive
    // (enforced by the caller in route_responses), so this branch and the
    // conversation branch above never both modify request_body.input.
    if let Some(mut items) = chain_items {
        let id_suffix = previous_response_id.as_deref().unwrap_or("new");
        let own_input =
            std::mem::replace(&mut request_body.input, ResponseInput::Items(Vec::new()));
        append_current_input(&mut items, own_input, id_suffix);
        request_body.input = ResponseInput::Items(items);
    }

    Ok(LoadedInputHistory {
        previous_response_id,
        existing_mcp_list_tools_labels: existing_mcp_list_tools_labels.into_iter().collect(),
    })
}

/// A storage failure while the history is loaded: logged in the context's
/// words, recorded as an internal error on `model`, answered as a storage
/// failure under `code`.
fn storage_failed(
    model: &str,
    code: &str,
    context: impl Display,
    err: &(impl StorageFailure + Display),
) -> Response {
    warn!("{context}: {err}");
    super::record_error(
        model,
        metrics_labels::ERROR_INTERNAL,
        metrics_labels::REASON_INTERNAL,
    );
    error::storage_failure(code, context, err)
}

/// The previous-response chain as the caller may see it: the records up to the
/// first of another tenant (the chain ends there, as it would at a missing
/// one); a chain ending in a record its client asked not to store (a
/// background run's record on the shared store, kept for its polling window
/// only) would break when it goes: no previous response, as for an unstored
/// id.
async fn visible_previous_chain(
    components: &ResponsesComponents,
    prev_id_str: &str,
    tenant: Option<&str>,
    model: &str,
) -> Result<Vec<Arc<StoredResponse>>, Response> {
    let prev_id = ResponseId::from(prev_id_str);
    match components
        .response_storage
        .get_response_chain_for(&prev_id, tenant)
        .await
    {
        Ok(chain) if !chain.responses.is_empty() => Ok(chain.responses),
        Ok(_) | Err(ResponseStorageError::ResponseNotFound(_)) => {
            super::record_error(
                model,
                metrics_labels::ERROR_VALIDATION,
                metrics_labels::REASON_INVALID_REQUEST,
            );
            Err(error::not_found(
                "previous_response_not_found",
                format!("Previous response with id '{prev_id_str}' not found."),
            ))
        }
        Err(e) => Err(storage_failed(
            model,
            "load_previous_response_chain_failed",
            format!("Failed to load previous response chain for {prev_id_str}"),
            &e,
        )),
    }
}

/// Every record's input items followed by its output items, in chain order.
fn chain_input_items(responses: &[Arc<StoredResponse>]) -> Vec<ResponseInputOutputItem> {
    responses
        .iter()
        .flat_map(|stored| {
            deserialize_items_from_array(&stored.input)
                .into_iter()
                .chain(deserialize_items_from_array(
                    stored
                        .raw_response
                        .get("output")
                        .unwrap_or(&Value::Array(vec![])),
                ))
        })
        .collect()
}

/// A conversation of another tenant answers as a missing one.
async fn check_conversation_visible(
    components: &ResponsesComponents,
    conv_id: &ConversationId,
    tenant: Option<&str>,
    model: &str,
) -> Result<(), Response> {
    match components
        .conversation_storage
        .get_conversation(conv_id)
        .await
    {
        Ok(Some(conversation)) if conversation.is_visible_to(tenant) => Ok(()),
        Ok(_) => {
            super::record_error(
                model,
                metrics_labels::ERROR_VALIDATION,
                metrics_labels::REASON_INVALID_REQUEST,
            );
            Err(error::not_found(
                "not_found",
                format!("No conversation found with id '{}'", conv_id.0),
            ))
        }
        Err(e) => Err(storage_failed(
            model,
            "check_conversation_failed",
            format!("Failed to check conversation {}", conv_id.0),
            &e,
        )),
    }
}

/// The conversation's newest items as request input: listed newest first so
/// that the limit cuts the oldest off, then put back in the order they were
/// stored. One more than the limit is asked for: a conversation of exactly
/// the limit replays whole, and only one with an item past the window is cut.
/// A window cut at the limit begins at its first turn boundary (the tail of
/// the turn the cut split is not sent), and an output whose call is before it
/// is not either.
async fn conversation_window_items(
    components: &ResponsesComponents,
    conv_id: &ConversationId,
    tenant: Option<&str>,
    model: &str,
) -> Result<Vec<ResponseInputOutputItem>, Response> {
    let params = ListParams {
        limit: MAX_CONVERSATION_HISTORY_ITEMS + 1,
        order: SortOrder::Desc,
        after: None,
    };
    let mut stored_items = match components
        .conversation_item_storage
        .list_items(conv_id, params)
        .await
    {
        Ok(stored_items) => stored_items,
        Err(e) => {
            return Err(storage_failed(
                model,
                "load_conversation_history_failed",
                "Failed to load conversation history",
                &e,
            ));
        }
    };
    stored_items.reverse();
    let cut = stored_items.len() > MAX_CONVERSATION_HISTORY_ITEMS;
    if cut {
        stored_items.drain(..stored_items.len() - MAX_CONVERSATION_HISTORY_ITEMS);
    }
    let mut hidden = false;
    // The rows the rule hid, converted: the call of an output among them is
    // not replayed either.
    let mut skipped = Vec::new();
    let items: Vec<ResponseInputOutputItem> = stored_items
        .into_iter()
        .filter_map(|row| {
            let visible = row.is_visible_to(tenant);
            hidden |= !visible;
            let item = stored_item_as_input(row)?;
            if !visible {
                skipped.push(item);
                return None;
            }
            Some(item)
        })
        .collect();
    let mut items = without_calls_answered_by(items, &skipped);
    if cut {
        items = without_outputs_of_missing_calls(from_turn_boundary(items));
    } else if hidden {
        // The output of a call the rule hid is no history, as after a cut.
        items = without_outputs_of_missing_calls(items);
    }
    Ok(items)
}

/// A stored conversation item as the request input item it was; nothing for
/// one that is not replayed (a reasoning item, an unknown type, a record that
/// does not deserialize: the last two are logged).
fn stored_item_as_input(item: ConversationItem) -> Option<ResponseInputOutputItem> {
    match item.item_type.as_str() {
        "message" => {
            // Stored content may be either the raw content array
            // (legacy shape) or an object `{content: [...], phase: ...}`
            // when the message carried a phase label (P3).
            let (content_value, stored_phase) = split_stored_message_content(item.content);
            match serde_json::from_value::<Vec<ResponseContentPart>>(content_value) {
                Ok(content_parts) => Some(ResponseInputOutputItem::Message {
                    id: item.id.0,
                    role: item.role.unwrap_or_else(|| "user".to_string()),
                    content: content_parts,
                    status: item.status,
                    phase: stored_phase,
                }),
                Err(e) => {
                    tracing::error!("Failed to deserialize message content: {}", e);
                    None
                }
            }
        }
        ItemType::FUNCTION_CALL => {
            match serde_json::from_value::<ResponseInputOutputItem>(item.content) {
                Ok(func_call) => Some(func_call),
                Err(e) => {
                    tracing::error!("Failed to deserialize function_call: {}", e);
                    None
                }
            }
        }
        ItemType::FUNCTION_CALL_OUTPUT => {
            tracing::debug!(
                item_id = %item.id.0,
                "Loading function_call_output from DB"
            );
            match serde_json::from_value::<ResponseInputOutputItem>(item.content) {
                Ok(func_output) => {
                    tracing::debug!("Successfully deserialized function_call_output");
                    Some(func_output)
                }
                Err(e) => {
                    tracing::error!("Failed to deserialize function_call_output: {}", e);
                    None
                }
            }
        }
        "additional_tools" | "agent_message" | "tool_search_call" | "tool_search_output"
        | "web_search_call" | "program" | "program_output" => {
            match serde_json::from_value::<ResponseInputOutputItem>(item.content) {
                Ok(replayed) => Some(replayed),
                Err(e) => {
                    tracing::error!("Failed to deserialize {}: {}", item.item_type, e);
                    None
                }
            }
        }
        "reasoning" => None,
        _ => {
            warn!("Unknown item type in conversation: {}", item.item_type);
            None
        }
    }
}

/// Deserialize ResponseInputOutputItems from a JSON array value
fn deserialize_items_from_array(array: &Value) -> Vec<ResponseInputOutputItem> {
    array
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    ResponseInputOutputItem::deserialize(item)
                        .map_err(|e| warn!("Failed to deserialize item: {}. Item: {}", e, item))
                        .ok()
                })
                .collect()
        })
        .unwrap_or_default()
}

fn extract_mcp_list_tools_labels(array: &Value) -> Vec<String> {
    array
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    (item.get("type").and_then(|t| t.as_str()) == Some(ItemType::MCP_LIST_TOOLS))
                        .then(|| item.get("server_label").and_then(|v| v.as_str()))
                        .flatten()
                        .map(ToOwned::to_owned)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Append current request input to items list, creating a user message if needed.
/// The input is taken by value: its text or its items move into the list.
fn append_current_input(
    items: &mut Vec<ResponseInputOutputItem>,
    input: ResponseInput,
    id_suffix: &str,
) {
    match input {
        ResponseInput::Text(text) => {
            items.push(ResponseInputOutputItem::Message {
                id: format!("msg_u_{id_suffix}"),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText { text }],
                status: Some("completed".to_string()),
                phase: None,
            });
        }
        ResponseInput::Items(current_items) => {
            items.extend(
                current_items
                    .into_iter()
                    .map(openai_protocol::responses::normalize_input_item),
            );
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/openai/responses/history/tests.rs"]
mod tests;
