//! Input history loading for the Responses API.
//!
//! Loads conversation history and/or previous response chains into the request
//! input before forwarding to the upstream provider.

use std::{collections::HashSet, sync::Arc};

use axum::response::Response;
use openai_protocol::{
    event_types::ItemType,
    responses::{
        from_turn_boundary, without_outputs_of_missing_calls, ResponseContentPart, ResponseInput,
        ResponseInputOutputItem, ResponsesRequest,
    },
};
use serde_json::Value;
use smg_data_connector::{
    is_unstored, visible_chain, ConversationId, ConversationItem, ListParams, ResponseId,
    ResponseStorageError, SortOrder, StoredResponse,
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
        let mut items = conversation_window_items(components, &conv_id, model).await?;
        append_current_input(&mut items, &request_body.input, conv_id_str);
        request_body.input = ResponseInput::Items(items);
    }

    // Apply previous response chain items if loaded.
    // Note: conversation and previous_response_id are mutually exclusive
    // (enforced by the caller in route_responses), so this branch and the
    // conversation branch above never both modify request_body.input.
    if let Some(mut items) = chain_items {
        let id_suffix = previous_response_id.as_deref().unwrap_or("new");
        append_current_input(&mut items, &request_body.input, id_suffix);
        request_body.input = ResponseInput::Items(items);
    }

    Ok(LoadedInputHistory {
        previous_response_id,
        existing_mcp_list_tools_labels: existing_mcp_list_tools_labels.into_iter().collect(),
    })
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
        .get_response_chain(&prev_id, None)
        .await
        .map(|chain| visible_chain(chain.responses, tenant))
    {
        Ok(responses)
            if !responses.is_empty()
                && !responses
                    .last()
                    .is_some_and(|response| is_unstored(response)) =>
        {
            Ok(responses)
        }
        Ok(_) | Err(ResponseStorageError::ResponseNotFound(_)) => {
            super::record_error(
                model,
                metrics_labels::ERROR_VALIDATION,
                metrics_labels::REASON_INVALID_REQUEST,
            );
            Err(error::bad_request(
                "previous_response_not_found",
                format!("Previous response with id '{prev_id_str}' not found."),
            ))
        }
        Err(e) => {
            warn!(
                "Failed to load previous response chain for {}: {}",
                prev_id_str, e
            );
            super::record_error(
                model,
                metrics_labels::ERROR_INTERNAL,
                metrics_labels::REASON_INTERNAL,
            );
            Err(error::storage_failure(
                "load_previous_response_chain_failed",
                format!("Failed to load previous response chain for {prev_id_str}"),
                &e,
            ))
        }
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
        Err(e) => {
            warn!("Failed to check conversation {}: {}", conv_id.0, e);
            super::record_error(
                model,
                metrics_labels::ERROR_INTERNAL,
                metrics_labels::REASON_INTERNAL,
            );
            Err(error::storage_failure(
                "check_conversation_failed",
                format!("Failed to check conversation {}", conv_id.0),
                &e,
            ))
        }
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
            warn!("Failed to load conversation history: {}", e);
            super::record_error(
                model,
                metrics_labels::ERROR_INTERNAL,
                metrics_labels::REASON_INTERNAL,
            );
            return Err(error::storage_failure(
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
    let mut items: Vec<ResponseInputOutputItem> = stored_items
        .into_iter()
        .filter_map(stored_item_as_input)
        .collect();
    if cut {
        items = without_outputs_of_missing_calls(from_turn_boundary(items));
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
                    serde_json::from_value::<ResponseInputOutputItem>(item.clone())
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

/// Append current request input to items list, creating a user message if needed
fn append_current_input(
    items: &mut Vec<ResponseInputOutputItem>,
    input: &ResponseInput,
    id_suffix: &str,
) {
    match input {
        ResponseInput::Text(text) => {
            items.push(ResponseInputOutputItem::Message {
                id: format!("msg_u_{id_suffix}"),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText { text: text.clone() }],
                status: Some("completed".to_string()),
                phase: None,
            });
        }
        ResponseInput::Items(current_items) => {
            for item in current_items {
                items.push(openai_protocol::responses::normalize_input_item(item));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rustls::crypto::ring;
    use serde_json::json;
    use smg_data_connector::{
        AddedAtSequence, ConversationItemId, ConversationStorage, MemoryConversationItemStorage,
        MemoryConversationStorage, MemoryResponseStorage, NewConversation, NewConversationItem,
    };
    use smg_mcp::{McpConfig, McpOrchestrator};

    use super::*;
    use crate::openai::context::{ResponsesComponents, SharedComponents};

    /// The tests' HTTP client, the process-level TLS crypto provider
    /// installed first; the loader under test sends nothing with it.
    #[expect(
        clippy::disallowed_methods,
        reason = "a test binary's client factory; the process TLS crypto provider is installed first"
    )]
    fn client() -> reqwest::Client {
        let _ = ring::default_provider().install_default();
        reqwest::Client::new()
    }

    /// The components of one router over in-memory stores, with the
    /// conversation `conv_1`.
    async fn components() -> ResponsesComponents {
        let conversations = Arc::new(MemoryConversationStorage::new());
        conversations
            .create_conversation(NewConversation {
                id: Some(ConversationId::from("conv_1")),
                metadata: None,
                tenant: None,
            })
            .await
            .expect("the conversation is created");
        ResponsesComponents {
            shared: Arc::new(SharedComponents { client: client() }),
            mcp_orchestrator: Arc::new(
                McpOrchestrator::new(McpConfig::default())
                    .await
                    .expect("an orchestrator without servers"),
            ),
            mcp_format_registry: Default::default(),
            response_storage: Arc::new(MemoryResponseStorage::new()),
            conversation_storage: conversations,
            conversation_item_storage: Arc::new(MemoryConversationItemStorage::new()),
        }
    }

    /// Links `rows` to `conv_1` in order, each stamped later than the one
    /// before, as the conversations API does.
    async fn link_rows(components: &ResponsesComponents, rows: Vec<NewConversationItem>) {
        let conv_id = ConversationId::from("conv_1");
        let mut stamps = AddedAtSequence::new();
        for row in rows {
            let created = components
                .conversation_item_storage
                .create_item(row)
                .await
                .expect("the row is created");
            components
                .conversation_item_storage
                .link_item(&conv_id, &created.id, stamps.next_added_at())
                .await
                .expect("the row is linked");
        }
    }

    /// Message `n` of a conversation: a user's for an odd `n`, the
    /// assistant's for an even one, its text `item n`.
    fn message_row(n: usize) -> NewConversationItem {
        let (role, content) = if n % 2 == 1 {
            (
                "user",
                json!([{"type": "input_text", "text": format!("item {n}")}]),
            )
        } else {
            (
                "assistant",
                json!([{"type": "output_text", "text": format!("item {n}"), "annotations": []}]),
            )
        };
        NewConversationItem {
            id: None,
            response_id: None,
            item_type: "message".to_string(),
            role: Some(role.to_string()),
            content,
            status: Some("completed".to_string()),
            tenant: None,
        }
    }

    fn call_row(call_id: &str) -> NewConversationItem {
        NewConversationItem {
            id: None,
            response_id: None,
            item_type: ItemType::FUNCTION_CALL.to_string(),
            role: None,
            content: json!({
                "type": "function_call", "call_id": call_id, "name": "calc",
                "arguments": "{}", "status": "completed"
            }),
            status: Some("completed".to_string()),
            tenant: None,
        }
    }

    fn output_row(call_id: &str) -> NewConversationItem {
        NewConversationItem {
            id: None,
            response_id: None,
            item_type: ItemType::FUNCTION_CALL_OUTPUT.to_string(),
            role: None,
            content: json!({"type": "function_call_output", "call_id": call_id, "output": "42"}),
            status: Some("completed".to_string()),
            tenant: None,
        }
    }

    /// The kinds of the items, with the call id of a tool item, and the text
    /// of a message, for the window assertions.
    fn shape(items: &[ResponseInputOutputItem]) -> Vec<String> {
        items
            .iter()
            .map(|item| match item {
                ResponseInputOutputItem::Message { role, content, .. } => {
                    let text = content
                        .iter()
                        .find_map(|part| match part {
                            ResponseContentPart::InputText { text }
                            | ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
                            _ => None,
                        })
                        .unwrap_or("");
                    format!("{role}:{text}")
                }
                ResponseInputOutputItem::FunctionToolCall { call_id, .. } => {
                    format!("function_call:{call_id}")
                }
                ResponseInputOutputItem::FunctionCallOutput { call_id, .. } => {
                    format!("function_call_output:{call_id}")
                }
                other => format!("{other:?}"),
            })
            .collect()
    }

    async fn loaded_input(
        components: &ResponsesComponents,
        text: &str,
    ) -> Vec<ResponseInputOutputItem> {
        let mut request = ResponsesRequest {
            input: ResponseInput::Text(text.to_string()),
            ..Default::default()
        };
        load_input_history(components, Some("conv_1"), &mut request, "m", None)
            .await
            .expect("the conversation loads");
        match request.input {
            ResponseInput::Items(items) => items,
            ResponseInput::Text(_) => panic!("the loaded input is an item list"),
        }
    }

    /// A conversation of 120 items replays its newest 100, oldest first,
    /// then the request's input: the first 20 are not sent.
    #[tokio::test]
    async fn a_conversation_longer_than_the_limit_replays_its_newest_items() {
        let components = components().await;
        link_rows(&components, (1..=120).map(message_row).collect()).await;

        let items = loaded_input(&components, "item 121").await;

        let expected: Vec<String> = (21..=121)
            .map(|n| format!("{}:item {n}", if n % 2 == 1 { "user" } else { "assistant" }))
            .collect();
        assert_eq!(shape(&items), expected);
    }

    /// A conversation one item longer than the limit, whose first items are
    /// two calls with their outputs: its window begins at the first output,
    /// inside a turn whose request is before the window. The window replays
    /// from its first client message; the output of the call before it and
    /// the call with its output after it go with the turn the cut split.
    #[tokio::test]
    async fn a_cut_window_begins_at_its_first_client_message() {
        let components = components().await;
        let mut rows = vec![
            call_row("call_A"),
            output_row("call_A"),
            call_row("call_B"),
            output_row("call_B"),
        ];
        rows.extend((5..=101).map(message_row));
        link_rows(&components, rows).await;

        let items = loaded_input(&components, "item 102").await;

        let kinds = shape(&items);
        assert_eq!(kinds[0], "user:item 5", "{kinds:?}");
        assert!(
            !kinds.iter().any(|kind| kind.starts_with("function_call")),
            "{kinds:?}"
        );
        // the calls and outputs are the tail of the turn the cut split: the
        // messages replay, then the request
        assert_eq!(items.len(), 101 - 4 + 1);
        assert_eq!(kinds.last().map(String::as_str), Some("user:item 102"));
    }

    /// A conversation of exactly the limit is not cut: every item replays,
    /// an output whose call is not among them included (the window did not
    /// begin after its call).
    #[tokio::test]
    async fn a_conversation_of_exactly_the_limit_replays_whole() {
        let components = components().await;
        let mut rows = vec![output_row("call_A")];
        rows.extend((2..=100).map(message_row));
        link_rows(&components, rows).await;

        let items = loaded_input(&components, "item 101").await;

        let kinds = shape(&items);
        assert_eq!(kinds[0], "function_call_output:call_A", "{kinds:?}");
        assert_eq!(items.len(), 100 + 1);
        assert_eq!(kinds.last().map(String::as_str), Some("user:item 101"));
    }
    /// A cut that lands inside a turn (row 51 of 150 is the output of turn
    /// 13's call): the window begins at the next turn boundary, the user's
    /// message of turn 14 (row 53); the output and the answer of turn 13
    /// before it are not sent.
    #[tokio::test]
    async fn a_cut_window_that_begins_inside_a_turn_starts_at_the_next_turn_boundary() {
        let components = components().await;
        let mut rows = Vec::new();
        for turn in 1..=37 {
            rows.push(message_row(4 * turn - 3));
            rows.push(call_row(&format!("call_{turn}")));
            rows.push(output_row(&format!("call_{turn}")));
            rows.push(message_row(4 * turn));
        }
        rows.push(message_row(149));
        rows.push(call_row("call_38"));
        link_rows(&components, rows).await;

        let items = loaded_input(&components, "item 151").await;

        let kinds = shape(&items);
        assert_eq!(
            &kinds[..4],
            [
                "user:item 53",
                "function_call:call_14",
                "function_call_output:call_14",
                "assistant:item 56"
            ],
            "{kinds:?}"
        );
        assert!(
            !kinds.contains(&"assistant:item 52".to_string()),
            "{kinds:?}"
        );
        assert!(
            !kinds.contains(&"function_call_output:call_13".to_string()),
            "{kinds:?}"
        );
        // rows 53 to 150, then the request
        assert_eq!(items.len(), 150 - 52 + 1);
        assert_eq!(kinds.last().map(String::as_str), Some("user:item 151"));
    }

    /// A stored row replays as the item it was (a message by its role and
    /// content, a call as itself); a reasoning row, a row of an unknown type
    /// and a message whose content does not read are not replayed.
    #[test]
    fn a_stored_row_replays_as_its_item_or_not_at_all() {
        let row = |item_type: &str, content: Value| ConversationItem {
            id: ConversationItemId::from("item_1"),
            response_id: None,
            item_type: item_type.to_string(),
            role: Some("assistant".to_string()),
            content,
            status: Some("completed".to_string()),
            created_at: chrono::Utc::now(),
            tenant: None,
        };
        let message = stored_item_as_input(row(
            "message",
            json!([{"type": "output_text", "text": "item 1", "annotations": []}]),
        ))
        .expect("a message replays");
        assert_eq!(shape(&[message]), vec!["assistant:item 1"]);
        let call = stored_item_as_input(row(
            ItemType::FUNCTION_CALL,
            json!({"type": "function_call", "call_id": "call_A", "name": "calc", "arguments": "{}", "status": "completed"}),
        ))
        .expect("a call replays");
        assert_eq!(shape(&[call]), vec!["function_call:call_A"]);
        assert!(stored_item_as_input(row("reasoning", json!({}))).is_none());
        assert!(stored_item_as_input(row("mystery", json!({}))).is_none());
        assert!(
            stored_item_as_input(row("message", json!({"text": "no content array"}))).is_none()
        );
    }
}
