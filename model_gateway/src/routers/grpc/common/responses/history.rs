//! The history a Responses request continues, as the input items the
//! engine-side builders consume.
//!
//! Every path that rebuilds a request's history (the buffered, streaming and
//! background responses of both pipelines) goes through [`load_history`]: the
//! stored chain behind `previous_response_id`, then the stored items of
//! `conversation`, then the request's own input, each stored item converted
//! back by the one conversion of this module. A stored turn's tool calls and
//! their outputs come back as they were stored, in the order they were stored.

use std::sync::Arc;

use axum::response::Response;
use openai_protocol::responses::{
    generate_id, normalize_input_item, McpToolCallError, ResponseContentPart, ResponseInput,
    ResponseInputOutputItem, ResponsesRequest,
};
use serde_json::Value;
use smg_data_connector::{
    self as data_connector, ConversationId, ConversationItem, ConversationItemStorage,
    ConversationStorage, ResponseStorage, StoredResponse,
};
use tracing::{debug, warn};

use super::{previous_response_chain, ResponsesContext};
use crate::routers::{common::persistence_utils::split_stored_message_content, error};

/// How many of a conversation's stored items a request replays.
const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;

/// The stores a request's history is read from.
#[derive(Clone, Copy)]
pub(crate) struct HistoryStores<'a> {
    pub responses: &'a Arc<dyn ResponseStorage>,
    pub conversations: &'a dyn ConversationStorage,
    pub conversation_items: &'a dyn ConversationItemStorage,
}

impl ResponsesContext {
    /// The stores this context reads a request's history from.
    pub(crate) fn history_stores(&self) -> HistoryStores<'_> {
        HistoryStores {
            responses: &self.response_storage,
            conversations: self.conversation_storage.as_ref(),
            conversation_items: self.conversation_item_storage.as_ref(),
        }
    }
}

/// `request` with the history it continues loaded into its input: the items
/// of the `previous_response_id` chain, then the items of `conversation`,
/// then the request's own input, every `item_reference` among them resolved.
/// `previous_response_id` is cleared once its chain is loaded; a request that
/// continues nothing keeps its input as it came. `tenant` is the caller's
/// tenant: a previous response or a conversation of another tenant answers as
/// a missing one does.
pub(crate) async fn load_history(
    stores: HistoryStores<'_>,
    request: &ResponsesRequest,
    tenant: Option<&str>,
) -> Result<ResponsesRequest, Response> {
    let mut history: Vec<ResponseInputOutputItem> = Vec::new();
    let mut continues = false;

    if let Some(prev_id) = request.previous_response_id.as_deref() {
        let chain = previous_response_chain(stores.responses, prev_id, tenant).await?;
        history.extend(chain.iter().flat_map(stored_response_items));
        continues = true;
    }

    if let Some(conversation) = &request.conversation {
        history.extend(conversation_items(stores, conversation.as_id(), tenant).await?);
        continues = true;
    }

    let mut loaded = request.clone();
    if continues {
        history.extend(current_input_items(&request.input));
        loaded.input = ResponseInput::Items(history);
        loaded.previous_response_id = None;
    }

    // Resolve `item_reference` input items against what this request already
    // loaded (the previous_response_id chain, the conversation history) and,
    // failing that, the response store.
    if let ResponseInput::Items(items) = &loaded.input {
        if items
            .iter()
            .any(|item| matches!(item, ResponseInputOutputItem::ItemReference { .. }))
        {
            let resolved =
                resolve_item_references(stores.responses.as_ref(), items.clone(), tenant).await?;
            loaded.input = ResponseInput::Items(resolved);
        }
    }

    debug!(
        has_previous_response = request.previous_response_id.is_some(),
        has_conversation = request.conversation.is_some(),
        "Loaded conversation history"
    );

    Ok(loaded)
}

/// The request's own input as items: a text input is the user message it
/// stands for; an item list is normalized (a bare `{role, content}` message
/// becomes a message item), every other item kept as it came.
pub(crate) fn current_input_items(input: &ResponseInput) -> Vec<ResponseInputOutputItem> {
    match input {
        ResponseInput::Text(text) => vec![ResponseInputOutputItem::Message {
            id: generate_id("msg"),
            role: "user".to_string(),
            content: vec![ResponseContentPart::InputText { text: text.clone() }],
            status: Some("completed".to_string()),
            phase: None,
        }],
        ResponseInput::Items(items) => items.iter().map(normalize_input_item).collect(),
    }
}

/// The items a stored response replays as: its input items, then its output
/// items, each as it was stored.
pub(crate) fn stored_response_items(stored: &StoredResponse) -> Vec<ResponseInputOutputItem> {
    let output = stored.raw_response.get("output").and_then(Value::as_array);
    stored
        .input
        .as_array()
        .into_iter()
        .flatten()
        .chain(output.into_iter().flatten())
        .filter_map(stored_item_as_input)
        .filter(replays)
        .collect()
}

/// Whether a stored item is one the history replays: see [`unreplayed_kind`].
fn replays(item: &ResponseInputOutputItem) -> bool {
    match unreplayed_kind(item) {
        Some(kind) => {
            debug!(
                item_type = kind,
                "Stored item of a kind the history does not replay; left out"
            );
            false
        }
        None => true,
    }
}

/// The name of a stored item's kind when neither conversion takes it as
/// input (a hosted image generation, an approval the client sent, the
/// computer, shell, patch and local-shell tools, a compaction), `None` for a
/// kind the history replays. Such rows are left out when the history loads
/// rather than refused later: the gateway wrote them itself (a hosted tool's
/// output, the client's approval turn), and a continuation that fails on
/// them until the client deletes the row is a conversation the client cannot
/// carry on. A request's own input items are not filtered and keep their
/// refusal. Every variant is named, so a new one has to say whether the
/// history replays it.
fn unreplayed_kind(item: &ResponseInputOutputItem) -> Option<&'static str> {
    match item {
        ResponseInputOutputItem::ImageGenerationCall { .. } => Some("image_generation_call"),
        ResponseInputOutputItem::McpApprovalResponse { .. } => Some("mcp_approval_response"),
        ResponseInputOutputItem::ComputerCall { .. } => Some("computer_call"),
        ResponseInputOutputItem::ComputerCallOutput { .. } => Some("computer_call_output"),
        ResponseInputOutputItem::Compaction { .. } => Some("compaction"),
        ResponseInputOutputItem::ShellCall { .. } => Some("shell_call"),
        ResponseInputOutputItem::ShellCallOutput { .. } => Some("shell_call_output"),
        ResponseInputOutputItem::ApplyPatchCall { .. } => Some("apply_patch_call"),
        ResponseInputOutputItem::ApplyPatchCallOutput { .. } => Some("apply_patch_call_output"),
        ResponseInputOutputItem::LocalShellCall { .. } => Some("local_shell_call"),
        ResponseInputOutputItem::LocalShellCallOutput { .. } => Some("local_shell_call_output"),
        ResponseInputOutputItem::Message { .. }
        | ResponseInputOutputItem::SimpleInputMessage { .. }
        | ResponseInputOutputItem::Reasoning { .. }
        | ResponseInputOutputItem::FunctionToolCall { .. }
        | ResponseInputOutputItem::FunctionCallOutput { .. }
        | ResponseInputOutputItem::McpApprovalRequest { .. }
        | ResponseInputOutputItem::McpCall { .. }
        | ResponseInputOutputItem::McpListTools { .. }
        | ResponseInputOutputItem::CustomToolCall { .. }
        | ResponseInputOutputItem::CustomToolCallOutput { .. }
        | ResponseInputOutputItem::ItemReference { .. } => None,
    }
}

/// What a stored hosted tool call replays as its result: its output when it
/// has one, else its error's text when it failed (a failed call stores an
/// empty output and the failure under `error`; replaying it as an empty
/// result would tell the model the call succeeded and said nothing), else
/// nothing (a call that has not run).
pub(crate) fn hosted_call_result(
    output: Option<&str>,
    error: Option<&McpToolCallError>,
    status: Option<&str>,
) -> Option<String> {
    if let Some(output) = output.filter(|output| !output.is_empty()) {
        return Some(output.to_owned());
    }
    match error {
        Some(McpToolCallError::ProtocolError { code, message }) => Some(format!(
            "The tool call failed (protocol error {code}): {message}"
        )),
        Some(McpToolCallError::HttpError { code, message }) => {
            Some(format!("The tool call failed (HTTP {code}): {message}"))
        }
        Some(McpToolCallError::ToolExecutionError { content }) => Some(format!(
            "The tool call failed: {}",
            content
                .as_str()
                .map_or_else(|| content.to_string(), str::to_owned)
        )),
        None if status == Some("failed") => Some("The tool call failed.".to_owned()),
        None => output.map(str::to_owned),
    }
}

/// The input item a conversation row replays as: a message row from its
/// content parts and the phase label it may carry, a reasoning row added
/// through the conversations API from the bare content it stores, any other
/// row from the whole item it stores. `None` for a row that has no input form
/// (a built-in tool call, an older schema), which is left out.
pub(crate) fn conversation_item_as_input(
    item: &ConversationItem,
) -> Option<ResponseInputOutputItem> {
    match item.item_type.as_str() {
        "message" => {
            let (content, phase) = split_stored_message_content(item.content.clone());
            match serde_json::from_value::<Vec<ResponseContentPart>>(content) {
                Ok(content) => Some(ResponseInputOutputItem::Message {
                    id: item.id.0.clone(),
                    role: item.role.clone().unwrap_or_else(|| "user".to_string()),
                    content,
                    status: item.status.clone(),
                    phase,
                }),
                Err(e) => {
                    warn!(item_id = %item.id.0, error = %e, "Stored message item has no readable content; left out of the history");
                    None
                }
            }
        }
        "reasoning" if item.content.is_array() => {
            match serde_json::from_value(item.content.clone()) {
                Ok(content) => Some(ResponseInputOutputItem::new_reasoning(
                    item.id.0.clone(),
                    vec![],
                    content,
                    item.status.clone(),
                )),
                Err(e) => {
                    warn!(item_id = %item.id.0, error = %e, "Stored reasoning item has no readable content; left out of the history");
                    None
                }
            }
        }
        _ => {
            // A whole-item row keeps the item under its own fields; a row
            // added through the conversations API carries its id on the row
            // only, and an item is found by id when referenced.
            let mut whole = item.content.clone();
            if let Some(fields) = whole.as_object_mut() {
                fields
                    .entry("id")
                    .or_insert_with(|| Value::String(item.id.0.clone()));
            }
            stored_item_as_input(&whole)
        }
    }
}

/// The input item a stored item replays as; `None` for one with no input form
/// (a built-in tool call, an older schema), which is left out.
fn stored_item_as_input(item: &Value) -> Option<ResponseInputOutputItem> {
    serde_json::from_value::<ResponseInputOutputItem>(item.clone())
        .map_err(|e| warn!("Failed to deserialize stored item: {}. Item: {}", e, item))
        .ok()
}

/// The replayable items of a conversation, oldest first, as the caller's
/// `tenant` may see it: a conversation of another tenant is as missing.
async fn conversation_items(
    stores: HistoryStores<'_>,
    conversation_id: &str,
    tenant: Option<&str>,
) -> Result<Vec<ResponseInputOutputItem>, Response> {
    let conv_id = ConversationId::from(conversation_id);
    let conversation = stores
        .conversations
        .get_conversation(&conv_id)
        .await
        .map_err(|e| {
            error::storage_failure(
                "check_conversation_failed",
                "Failed to check conversation",
                &e,
            )
        })?;
    if !conversation.is_some_and(|conversation| conversation.is_visible_to(tenant)) {
        return Err(error::not_found(
            "conversation_not_found",
            format!(
                "Conversation '{conversation_id}' not found. Please create the conversation first using the conversations API."
            ),
        ));
    }

    let params = data_connector::ListParams {
        limit: MAX_CONVERSATION_HISTORY_ITEMS,
        order: data_connector::SortOrder::Asc,
        after: None,
    };
    // A listing the store did not answer (or refused) is not an answer
    // without the conversation's context: the client reads a 503 for a
    // timed-out store and a 500 for any other failure, as every other
    // store-dependent answer of this path does.
    let stored = stores
        .conversation_items
        .list_items(&conv_id, params)
        .await
        .map_err(|e| {
            error::storage_failure(
                "load_conversation_history_failed",
                "Failed to load conversation history",
                &e,
            )
        })?;
    Ok(stored
        .iter()
        .filter_map(conversation_item_as_input)
        .filter(replays)
        .collect())
}

/// Replace every `item_reference` input item by the item it points to: an item
/// already in the list (loaded from the previous_response_id chain or the
/// conversation history) or an output item of a stored response the caller's
/// `tenant` may see. An unknown id answers 404, as the public API does, and
/// so does an item of another tenant's response.
pub(crate) async fn resolve_item_references(
    storage: &dyn ResponseStorage,
    items: Vec<ResponseInputOutputItem>,
    tenant: Option<&str>,
) -> Result<Vec<ResponseInputOutputItem>, Response> {
    let mut resolved: Vec<ResponseInputOutputItem> = Vec::with_capacity(items.len());
    for item in items {
        let ResponseInputOutputItem::ItemReference { id, .. } = &item else {
            resolved.push(item);
            continue;
        };
        if let Some(found) = resolved
            .iter()
            .find(|candidate| input_item_id(candidate) == Some(id.as_str()))
            .cloned()
        {
            resolved.push(found);
            continue;
        }
        let stored = storage
            .find_response_by_output_item(id)
            .await
            .map_err(|e| {
                error::storage_failure(
                    "resolve_item_reference_failed",
                    format!("Failed to resolve item '{id}'"),
                    &e,
                )
            })?
            // The items of a response its client asked not to store are not
            // stored items (its record is kept for its polling window only),
            // and a response of another tenant has none the caller may see.
            .filter(|response| {
                !data_connector::is_unstored(response) && response.is_visible_to(tenant)
            });
        let candidate = stored
            .as_ref()
            .and_then(|response| response.raw_response.get("output"))
            .and_then(Value::as_array)
            .and_then(|output| {
                output
                    .iter()
                    .find(|candidate| candidate.get("id").and_then(Value::as_str) == Some(id))
            });
        let Some(candidate) = candidate else {
            return Err(error::not_found(
                "item_not_found",
                format!("Item with id '{id}' not found."),
            ));
        };
        // a stored item that has no input form (a built-in tool call, an older
        // schema) exists but cannot be re-injected: say so, not "not found"
        match serde_json::from_value::<ResponseInputOutputItem>(candidate.clone()) {
            Ok(found) => resolved.push(found),
            Err(e) => {
                let item_type = candidate
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                warn!(
                    "item_reference '{id}' (type '{item_type}') names a stored output item that cannot be used as input: {e}"
                );
                return Err(error::bad_request(
                    "item_not_usable_as_input",
                    format!("Item '{id}' of type '{item_type}' cannot be used as an input item."),
                ));
            }
        }
    }
    Ok(resolved)
}

/// The id an input item carries, for `item_reference` resolution. Every
/// variant is named, so a new one has to say whether it carries an id.
fn input_item_id(item: &ResponseInputOutputItem) -> Option<&str> {
    match item {
        ResponseInputOutputItem::Message { id, .. }
        | ResponseInputOutputItem::Reasoning { id, .. }
        | ResponseInputOutputItem::McpApprovalRequest { id, .. }
        | ResponseInputOutputItem::ImageGenerationCall { id, .. }
        | ResponseInputOutputItem::ComputerCall { id, .. }
        | ResponseInputOutputItem::LocalShellCall { id, .. }
        | ResponseInputOutputItem::LocalShellCallOutput { id, .. }
        | ResponseInputOutputItem::McpCall { id, .. }
        | ResponseInputOutputItem::McpListTools { id, .. } => Some(id.as_str()),
        ResponseInputOutputItem::FunctionToolCall { id, .. }
        | ResponseInputOutputItem::FunctionCallOutput { id, .. }
        | ResponseInputOutputItem::McpApprovalResponse { id, .. }
        | ResponseInputOutputItem::Compaction { id, .. }
        | ResponseInputOutputItem::ComputerCallOutput { id, .. }
        | ResponseInputOutputItem::CustomToolCall { id, .. }
        | ResponseInputOutputItem::CustomToolCallOutput { id, .. }
        | ResponseInputOutputItem::ShellCall { id, .. }
        | ResponseInputOutputItem::ShellCallOutput { id, .. }
        | ResponseInputOutputItem::ApplyPatchCall { id, .. }
        | ResponseInputOutputItem::ApplyPatchCallOutput { id, .. } => id.as_deref(),
        // a reference is what gets resolved, never a resolution; a bare input
        // message carries no id
        ResponseInputOutputItem::ItemReference { .. }
        | ResponseInputOutputItem::SimpleInputMessage { .. } => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use axum::http;
    use openai_protocol::{common::ConversationRef, responses::StringOrContentParts};
    use serde_json::json;
    use smg_data_connector::{
        ConversationItemId, MemoryConversationItemStorage, MemoryConversationStorage,
        MemoryResponseStorage, NewConversation, NewConversationItem,
    };

    use super::*;
    use crate::routers::common::persistence_utils::persist_conversation_items;

    /// The stores of one gateway, in memory, with the conversation `conv_1`.
    pub(crate) struct Stores {
        pub responses: Arc<dyn ResponseStorage>,
        pub conversations: Arc<dyn ConversationStorage>,
        pub conversation_items: Arc<dyn ConversationItemStorage>,
    }

    impl Stores {
        pub(crate) async fn with_conversation() -> Self {
            let stores = Self {
                responses: Arc::new(MemoryResponseStorage::new()),
                conversations: Arc::new(MemoryConversationStorage::new()),
                conversation_items: Arc::new(MemoryConversationItemStorage::new()),
            };
            stores
                .conversations
                .create_conversation(NewConversation {
                    id: Some(ConversationId::from("conv_1")),
                    metadata: None,
                    tenant: None,
                })
                .await
                .expect("the conversation is created");
            stores
        }

        pub(crate) fn history(&self) -> HistoryStores<'_> {
            HistoryStores {
                responses: &self.responses,
                conversations: self.conversations.as_ref(),
                conversation_items: self.conversation_items.as_ref(),
            }
        }

        /// Stores a turn as the gateway persists one: `request` answered by
        /// a response of `output` items under `response_id`.
        pub(crate) async fn store_turn(
            &self,
            response_id: &str,
            request: &ResponsesRequest,
            output: Value,
        ) {
            let response = json!({
                "id": response_id,
                "object": "response",
                "model": "m",
                "status": "completed",
                "output": output,
            });
            persist_conversation_items(
                self.conversations.clone(),
                self.conversation_items.clone(),
                self.responses.clone(),
                &response,
                request,
                None,
                None,
            )
            .await
            .expect("the turn is stored");
        }
    }

    /// Turn 1 of the tests: the user's question, answered by one function
    /// call the client is to run.
    pub(crate) fn function_call_output() -> Value {
        json!([{
            "type": "function_call", "id": "fc_1", "call_id": "call_A",
            "name": "calc", "arguments": "{\"expr\":\"6*7\"}", "status": "completed"
        }])
    }

    /// Turn 2 of the tests: the client answers call_A.
    pub(crate) fn function_call_output_input() -> ResponseInput {
        ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
            id: None,
            call_id: "call_A".to_string(),
            output: StringOrContentParts::String("42".to_string()),
            status: None,
        }])
    }

    /// The kinds of the items, with the call id of a tool item, for the
    /// order assertions.
    pub(crate) fn shape(items: &[ResponseInputOutputItem]) -> Vec<String> {
        items
            .iter()
            .map(|item| match item {
                ResponseInputOutputItem::Message { role, .. } => format!("message:{role}"),
                ResponseInputOutputItem::FunctionToolCall {
                    call_id, output, ..
                } => format!(
                    "function_call:{call_id}{}",
                    if output.is_some() { "+output" } else { "" }
                ),
                ResponseInputOutputItem::FunctionCallOutput { call_id, .. } => {
                    format!("function_call_output:{call_id}")
                }
                ResponseInputOutputItem::Reasoning { .. } => "reasoning".to_string(),
                other => serde_json::to_value(other).expect("an item serializes")["type"]
                    .as_str()
                    .unwrap_or("?")
                    .to_string(),
            })
            .collect()
    }

    fn turn_1_in_conversation() -> ResponsesRequest {
        ResponsesRequest {
            input: ResponseInput::Text("What is 6*7?".to_string()),
            conversation: Some(ConversationRef::Id("conv_1".to_string())),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn a_conversation_reload_keeps_the_stored_turns_function_call_and_the_output_in_order() {
        let stores = Stores::with_conversation().await;
        stores
            .store_turn("resp_1", &turn_1_in_conversation(), function_call_output())
            .await;
        let turn_2 = ResponsesRequest {
            input: function_call_output_input(),
            conversation: Some(ConversationRef::Id("conv_1".to_string())),
            ..Default::default()
        };

        let loaded = load_history(stores.history(), &turn_2, None)
            .await
            .expect("the conversation loads");

        let ResponseInput::Items(items) = &loaded.input else {
            panic!("the loaded input is an item list");
        };
        assert_eq!(
            shape(items),
            [
                "message:user",
                "function_call:call_A",
                "function_call_output:call_A"
            ]
        );
        assert!(matches!(
            &items[1],
            ResponseInputOutputItem::FunctionToolCall { name, arguments, .. }
                if name == "calc" && arguments == "{\"expr\":\"6*7\"}"
        ));
    }

    /// A stored row of a kind no conversion replays (a hosted image
    /// generation the gateway wrote, the client's approval turn) is left out
    /// when the history loads, through `conversation` and through
    /// `previous_response_id` alike, so the continuation goes on instead of
    /// failing on the row until the client deletes it.
    #[tokio::test]
    async fn stored_kinds_no_conversion_replays_are_left_out_of_the_history() {
        let stores = Stores::with_conversation().await;
        stores
            .store_turn(
                "resp_1",
                &turn_1_in_conversation(),
                json!([
                    {"type": "image_generation_call", "id": "ig_1", "status": "completed",
                     "result": "aGVsbG8="},
                    {"type": "message", "id": "msg_1", "role": "assistant",
                     "content": [{"type": "output_text", "text": "Drawn.", "annotations": []}]}
                ]),
            )
            .await;
        let approval: ResponseInputOutputItem = serde_json::from_value(json!({
            "type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": true
        }))
        .expect("an approval response item");
        // Turn 2 continues the conversation and the chain alike, so both
        // loaders walk past the rows they do not replay.
        let turn_2 = ResponsesRequest {
            input: ResponseInput::Items(vec![approval]),
            conversation: Some(ConversationRef::Id("conv_1".to_string())),
            previous_response_id: Some("resp_1".to_string()),
            ..Default::default()
        };
        stores
            .store_turn(
                "resp_2",
                &turn_2,
                json!([
                    {"type": "message", "id": "msg_2", "role": "assistant",
                     "content": [{"type": "output_text", "text": "Approved.", "annotations": []}]}
                ]),
            )
            .await;
        let expected = [
            "message:user",
            "message:assistant",
            "message:assistant",
            "message:user",
        ];

        let turn_3 = ResponsesRequest {
            input: ResponseInput::Text("And now?".to_string()),
            conversation: Some(ConversationRef::Id("conv_1".to_string())),
            ..Default::default()
        };
        let loaded = load_history(stores.history(), &turn_3, None)
            .await
            .expect("the conversation loads past the rows it does not replay");
        let ResponseInput::Items(items) = &loaded.input else {
            panic!("the loaded input is an item list");
        };
        assert_eq!(shape(items), expected);

        let turn_3 = ResponsesRequest {
            input: ResponseInput::Text("And now?".to_string()),
            previous_response_id: Some("resp_2".to_string()),
            ..Default::default()
        };
        let loaded = load_history(stores.history(), &turn_3, None)
            .await
            .expect("the chain loads past the items it does not replay");
        let ResponseInput::Items(items) = &loaded.input else {
            panic!("the loaded input is an item list");
        };
        assert_eq!(shape(items), expected);
    }

    /// A stored hosted call replays its output; failing that, its failure
    /// as text (never a successful empty result for a call that failed); a
    /// call that has not run replays no result.
    #[test]
    fn a_stored_hosted_call_replays_its_output_else_its_failure() {
        let execution = McpToolCallError::execution("wiki is down");
        let http = McpToolCallError::HttpError {
            code: 503,
            message: "unavailable".to_string(),
        };
        assert_eq!(
            hosted_call_result(Some("found"), None, Some("completed")).as_deref(),
            Some("found")
        );
        assert_eq!(
            hosted_call_result(Some("found"), Some(&execution), Some("failed")).as_deref(),
            Some("found"),
            "an output that is there is the result"
        );
        assert_eq!(
            hosted_call_result(Some(""), Some(&execution), Some("failed")).as_deref(),
            Some("The tool call failed: wiki is down")
        );
        assert_eq!(
            hosted_call_result(None, Some(&http), None).as_deref(),
            Some("The tool call failed (HTTP 503): unavailable")
        );
        assert_eq!(
            hosted_call_result(None, None, Some("failed")).as_deref(),
            Some("The tool call failed.")
        );
        assert_eq!(hosted_call_result(None, None, Some("in_progress")), None);
        assert_eq!(
            hosted_call_result(Some(""), None, Some("completed")).as_deref(),
            Some(""),
            "a completed call with an empty output keeps it"
        );
    }

    #[tokio::test]
    async fn a_previous_response_reload_keeps_the_stored_turns_function_call_and_the_output_in_order(
    ) {
        let stores = Stores::with_conversation().await;
        let turn_1 = ResponsesRequest {
            input: ResponseInput::Text("What is 6*7?".to_string()),
            ..Default::default()
        };
        stores
            .store_turn("resp_1", &turn_1, function_call_output())
            .await;
        let turn_2 = ResponsesRequest {
            input: function_call_output_input(),
            previous_response_id: Some("resp_1".to_string()),
            ..Default::default()
        };

        let loaded = load_history(stores.history(), &turn_2, None)
            .await
            .expect("the chain loads");

        assert_eq!(loaded.previous_response_id, None);
        let ResponseInput::Items(items) = &loaded.input else {
            panic!("the loaded input is an item list");
        };
        assert_eq!(
            shape(items),
            [
                "message:user",
                "function_call:call_A",
                "function_call_output:call_A"
            ]
        );
    }

    /// Turn 3 after the client's answer was stored: the whole exchange comes
    /// back, the call before its output, through both continuations.
    #[tokio::test]
    async fn a_stored_function_call_output_reloads_after_its_call() {
        let stores = Stores::with_conversation().await;
        stores
            .store_turn("resp_1", &turn_1_in_conversation(), function_call_output())
            .await;
        let turn_2 = ResponsesRequest {
            input: function_call_output_input(),
            conversation: Some(ConversationRef::Id("conv_1".to_string())),
            previous_response_id: Some("resp_1".to_string()),
            ..Default::default()
        };
        stores
            .store_turn(
                "resp_2",
                &turn_2,
                json!([{
                    "type": "message", "id": "msg_a2", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "42", "annotations": []}]
                }]),
            )
            .await;
        let expected = [
            "message:user",
            "function_call:call_A",
            "function_call_output:call_A",
            "message:assistant",
            "message:user",
        ];

        for turn_3 in [
            ResponsesRequest {
                input: ResponseInput::Text("Thanks".to_string()),
                conversation: Some(ConversationRef::Id("conv_1".to_string())),
                ..Default::default()
            },
            ResponsesRequest {
                input: ResponseInput::Text("Thanks".to_string()),
                previous_response_id: Some("resp_2".to_string()),
                ..Default::default()
            },
        ] {
            let loaded = load_history(stores.history(), &turn_3, None)
                .await
                .expect("the history loads");
            let ResponseInput::Items(items) = &loaded.input else {
                panic!("the loaded input is an item list");
            };
            assert_eq!(shape(items), expected, "{:?}", turn_3.conversation);
        }
    }

    /// A function call added through the conversations API is stored whole,
    /// its id on the row: it reloads under that id.
    #[tokio::test]
    async fn a_function_call_added_to_the_conversation_reloads_under_the_rows_id() {
        let stores = Stores::with_conversation().await;
        let conv_id = ConversationId::from("conv_1");
        let row = stores
            .conversation_items
            .create_item(NewConversationItem {
                id: None,
                response_id: None,
                item_type: "function_call".to_string(),
                role: None,
                content: json!({
                    "type": "function_call", "call_id": "call_B", "name": "calc",
                    "arguments": "{}", "status": "completed"
                }),
                status: Some("completed".to_string()),
                tenant: None,
            })
            .await
            .expect("the row is created");
        stores
            .conversation_items
            .link_item(&conv_id, &row.id, chrono::Utc::now())
            .await
            .expect("the row is linked");
        let request = ResponsesRequest {
            input: ResponseInput::Text("go on".to_string()),
            conversation: Some(ConversationRef::Id("conv_1".to_string())),
            ..Default::default()
        };

        let loaded = load_history(stores.history(), &request, None)
            .await
            .expect("the conversation loads");

        let ResponseInput::Items(items) = &loaded.input else {
            panic!("the loaded input is an item list");
        };
        assert_eq!(shape(items), ["function_call:call_B", "message:user"]);
        assert_eq!(input_item_id(&items[0]), Some(row.id.0.as_str()));
    }

    /// A message row keeps the phase label it was stored with, and a reasoning
    /// row added through the conversations API (bare content) comes back as
    /// the reasoning item.
    #[test]
    fn message_and_reasoning_rows_rebuild_their_items() {
        let row = |item_type: &str, role: Option<&str>, content: Value| ConversationItem {
            id: ConversationItemId::from(format!("{item_type}_1").as_str()),
            response_id: None,
            item_type: item_type.to_string(),
            role: role.map(str::to_string),
            content,
            status: Some("completed".to_string()),
            created_at: chrono::Utc::now(),
            tenant: None,
        };

        let message = conversation_item_as_input(&row(
            "message",
            Some("assistant"),
            json!({"content": [{"type": "output_text", "text": "hi", "annotations": []}], "phase": "final_answer"}),
        ))
        .expect("a message row rebuilds");
        assert!(matches!(
            message,
            ResponseInputOutputItem::Message { ref id, ref role, phase: Some(openai_protocol::responses::MessagePhase::FinalAnswer), .. }
                if id == "message_1" && role == "assistant"
        ));

        let reasoning = conversation_item_as_input(&row(
            "reasoning",
            None,
            json!([{"type": "reasoning_text", "text": "thinking"}]),
        ))
        .expect("a reasoning row rebuilds");
        assert!(matches!(
            reasoning,
            ResponseInputOutputItem::Reasoning { ref id, ref content, .. }
                if id == "reasoning_1" && content.len() == 1
        ));

        // a row with no input form is left out, not an error
        assert!(conversation_item_as_input(&row(
            "web_search_call",
            None,
            json!({"type": "web_search_call", "status": "completed"})
        ))
        .is_none());
    }

    fn stored_response_with_message(item_id: &str, text: &str) -> StoredResponse {
        let mut stored = StoredResponse::new(None);
        stored.raw_response = json!({
            "id": "resp_stored",
            "output": [{
                "type": "message", "id": item_id, "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]
            }]
        });
        stored
    }

    fn reference(id: &str) -> ResponseInputOutputItem {
        serde_json::from_value(json!({"type": "item_reference", "id": id})).unwrap()
    }

    #[tokio::test]
    async fn item_reference_resolves_a_stored_output_item() {
        let storage = MemoryResponseStorage::new();
        storage
            .store_response(stored_response_with_message("msg_stored", "earlier answer"))
            .await
            .unwrap();
        let user = normalize_input_item(
            &serde_json::from_value(json!({"role": "user", "content": "continue"})).unwrap(),
        );

        let resolved = resolve_item_references(&storage, vec![reference("msg_stored"), user], None)
            .await
            .expect("a stored output item resolves");

        assert_eq!(resolved.len(), 2);
        match &resolved[0] {
            ResponseInputOutputItem::Message {
                id, role, content, ..
            } => {
                assert_eq!(id, "msg_stored");
                assert_eq!(role, "assistant");
                assert!(matches!(
                    &content[0],
                    ResponseContentPart::OutputText { text, .. } if text == "earlier answer"
                ));
            }
            other => panic!("expected the stored message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_item_reference_to_an_unstored_response_is_not_found() {
        let storage = MemoryResponseStorage::new();
        let mut unstored = stored_response_with_message("msg_unstored", "a background answer");
        unstored.raw_response["store"] = json!(false);
        storage.store_response(unstored).await.unwrap();

        let refused = resolve_item_references(&storage, vec![reference("msg_unstored")], None)
            .await
            .expect_err("an unstored response's item is no stored item");

        assert_eq!(refused.status(), http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn item_reference_prefers_an_item_already_loaded() {
        let storage = MemoryResponseStorage::new();
        let loaded = ResponseInputOutputItem::Message {
            id: "msg_loaded".to_string(),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: "from the chain".to_string(),
                annotations: vec![],
                logprobs: None,
            }],
            status: Some("completed".to_string()),
            phase: None,
        };

        let resolved =
            resolve_item_references(&storage, vec![loaded, reference("msg_loaded")], None)
                .await
                .expect("an item already in the list resolves without storage");

        assert_eq!(resolved.len(), 2);
        assert_eq!(input_item_id(&resolved[1]), Some("msg_loaded"));
        assert!(!matches!(
            resolved[1],
            ResponseInputOutputItem::ItemReference { .. }
        ));
    }

    #[tokio::test]
    async fn item_reference_to_a_stored_item_with_no_input_form_is_400() {
        let storage = MemoryResponseStorage::new();
        let mut stored = StoredResponse::new(None);
        stored.raw_response = json!({
            "id": "resp_stored",
            "output": [{"type": "web_search_call", "id": "ws_1", "status": "completed"}]
        });
        storage.store_response(stored).await.unwrap();

        let response = resolve_item_references(&storage, vec![reference("ws_1")], None)
            .await
            .expect_err("a stored item without an input form is refused");

        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn item_reference_finds_a_loaded_custom_tool_call_by_its_id() {
        let storage = MemoryResponseStorage::new();
        let loaded: ResponseInputOutputItem = serde_json::from_value(json!({
            "type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1",
            "name": "lookup", "input": "weather"
        }))
        .unwrap();

        let resolved = resolve_item_references(&storage, vec![loaded, reference("ctc_1")], None)
            .await
            .expect("an item already in the list resolves without storage");

        assert_eq!(resolved.len(), 2);
        assert_eq!(input_item_id(&resolved[1]), Some("ctc_1"));
    }

    /// An output item of another tenant's response is as unknown as a missing
    /// one; the owner's reference resolves.
    #[tokio::test]
    async fn item_reference_to_another_tenants_output_item_is_404() {
        let storage = MemoryResponseStorage::new();
        let mut stored = stored_response_with_message("msg_a", "a's answer");
        stored.tenant = Some("header:tenant-a".to_string());
        storage.store_response(stored).await.unwrap();

        for caller in [Some("header:tenant-b"), None] {
            let response = resolve_item_references(&storage, vec![reference("msg_a")], caller)
                .await
                .expect_err("another tenant's item is refused");
            assert_eq!(response.status(), http::StatusCode::NOT_FOUND, "{caller:?}");
        }
        let resolved =
            resolve_item_references(&storage, vec![reference("msg_a")], Some("header:tenant-a"))
                .await
                .expect("the owner's reference resolves");
        assert_eq!(input_item_id(&resolved[0]), Some("msg_a"));
    }

    #[tokio::test]
    async fn item_reference_to_an_unknown_id_is_404() {
        let storage = MemoryResponseStorage::new();

        let response = resolve_item_references(&storage, vec![reference("msg_missing")], None)
            .await
            .expect_err("an unknown item is refused");

        assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
    }
}
