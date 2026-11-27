//! The history a Responses request continues, as the input items the
//! engine-side builders consume.
//!
//! Every path that rebuilds a request's history (the buffered, streaming and
//! background responses of both pipelines) goes through [`load_history`]: the
//! stored chain behind `previous_response_id`, then the stored items of
//! `conversation`, then the request's own input, each stored item converted
//! back by the one conversion of this module. A stored turn's tool calls and
//! their outputs come back as they were stored, in the order they were stored;
//! a conversation longer than [`MAX_CONVERSATION_HISTORY_ITEMS`] replays its
//! newest items, from a turn boundary on.

use std::sync::Arc;

use axum::response::Response;
use openai_protocol::{
    responses::{
        from_turn_boundary, generate_id, normalize_input_item, without_calls_answered_by,
        without_outputs_of_missing_calls, McpToolCallError, ResponseContentPart, ResponseInput,
        ResponseInputOutputItem, ResponsesRequest, StringOrContentParts,
    },
    validated::Normalizable,
};
use serde::Deserialize;
use serde_json::Value;
use smg_data_connector::{
    self as data_connector, ConversationId, ConversationItem, ConversationItemStorage,
    ConversationStorage, ResponseStorage, StoredResponse,
};
use tracing::{debug, warn};

use super::{
    previous_response_chain, utils::refuse_tools_without_a_direct_caller, ResponsesContext,
};
use crate::routers::{
    common::{
        persistence_utils::split_stored_message_content,
        responses_compat::lower_responses_request_for_local_backend,
    },
    error,
};

/// How many of a conversation's stored items a request replays: its newest
/// ones, oldest first; what is older is not in the prompt. A window this
/// limit cuts can begin inside a turn, so it begins at its first turn
/// boundary (a message the client sent, or a call with its output) and the
/// outputs of the calls before the window are left out of it.
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
    into: ReplayInto,
) -> Result<ResponsesRequest, Response> {
    let mut history: Vec<ResponseInputOutputItem> = Vec::new();
    let mut continues = false;

    if let Some(prev_id) = request.previous_response_id.as_deref() {
        let chain = previous_response_chain(stores.responses, prev_id, tenant).await?;
        history.extend(
            chain
                .iter()
                .flat_map(|stored| stored_response_items(stored, into)),
        );
        continues = true;
    }
    // Where the chain's items end, for the pass below (resolution maps every
    // item to one item, so it holds after it).
    let chain_items = history.len();

    if let Some(conversation) = &request.conversation {
        history.extend(conversation_items(stores, conversation.as_id(), tenant, into).await?);
        continues = true;
    }

    let mut loaded = request.clone();
    if continues {
        // The copy of the request's own input made just above is the one that
        // joins the history, moved out of the loaded request; the normaliser
        // reshapes it in place instead of copying every item a second time.
        let own_input = std::mem::replace(&mut loaded.input, ResponseInput::Items(Vec::new()));
        history.extend(current_input_items(own_input));
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

    // The chain's items, their references resolved (a stored call behind an
    // `item_reference` counts as loaded): the results of calls that did not
    // survive the reload go. A chain is cut at the store's depth cap, at an
    // expired or deleted record or at another tenant's, with no word of it,
    // and the oldest turn kept then opens with the result of a call the turn
    // before it issued, which is gone: it answers nothing the request
    // replays, and the contract profiles refuse a tool result with no call
    // before it (`ProviderProfile::unanswerable_tool_result`), so keeping it
    // would refuse every later request on the chain for a call the client
    // never sent. A conversation's window keeps its own rule
    // (`conversation_items`), and the request's own items stay out of it: a
    // result among them with no call before it is the client's and keeps its
    // refusal.
    if chain_items > 0 {
        if let ResponseInput::Items(items) = &mut loaded.input {
            let rest = items.split_off(chain_items);
            let before = items.len();
            let mut kept = without_outputs_of_missing_calls(std::mem::take(items));
            if kept.len() < before {
                debug!(
                    dropped = before - kept.len(),
                    "dropped the chain's stored tool results whose calls did not survive the reload"
                );
            }
            kept.extend(rest);
            *items = kept;
        }
    }

    // Lower any inline `additional_tools` / `agent_message` items (from the
    // loaded history or the current turn) into standard top-level `tools` and
    // `user` messages for local backends, while keeping the caller's original
    // `request` untouched for persistence so subsequent turns can replay them.
    lower_responses_request_for_local_backend(&mut loaded)?;
    if continues && loaded.tool_choice.is_none() {
        loaded.normalize();
    }
    if let Some(error_response) = refuse_tools_without_a_direct_caller(loaded.tools.as_deref()) {
        return Err(error_response);
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
/// becomes a message item), every other item kept as it came. The input is
/// taken by value: what is not reshaped moves, nothing is copied.
pub(crate) fn current_input_items(input: ResponseInput) -> Vec<ResponseInputOutputItem> {
    match input {
        ResponseInput::Text(text) => vec![ResponseInputOutputItem::Message {
            id: generate_id("msg"),
            role: "user".to_string(),
            content: vec![ResponseContentPart::InputText { text }],
            status: Some("completed".to_string()),
            phase: None,
        }],
        ResponseInput::Items(items) => items.into_iter().map(normalize_input_item).collect(),
    }
}

/// The items a stored response replays as: its input items, then its output
/// items, each as it was stored.
fn stored_response_items(
    stored: &StoredResponse,
    into: ReplayInto,
) -> Vec<ResponseInputOutputItem> {
    let output = stored.raw_response.get("output").and_then(Value::as_array);
    stored
        .input
        .as_array()
        .into_iter()
        .flatten()
        .chain(output.into_iter().flatten())
        .filter_map(stored_item_as_input)
        .filter(replays)
        .map(|item| replayed_parts(item, into))
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

/// The pipeline a loaded history is replayed into. A stored message replays
/// the content parts that pipeline carried when the turn was answered: the
/// chat conversion never carried an `input_file`, an `input_image` without
/// an `image_url` (a `file_id` is not resolvable here) or a media part of a
/// `developer` message (the developer turn folds into the system text), the
/// Harmony builder no `input_image` at all. Such parts are left out when the
/// history loads, as an unreplayed item is ([`unreplayed_kind`]): the turn
/// was answered without them, and a continuation refused on them until the
/// client edits a chain it cannot edit is a conversation the client cannot
/// carry on. A request's own input is not filtered and keeps its refusal, and
/// so does a conversation item the client added itself (no response wrote
/// it, so no pipeline ever answered without its parts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplayInto {
    /// The chat conversion (`responses_to_chat`).
    Chat,
    /// The Harmony builder.
    Harmony,
}

impl ReplayInto {
    /// Whether the pipeline carried `part` of a message with `role` when a
    /// stored turn was answered.
    fn carried(self, role: &str, part: &ResponseContentPart) -> bool {
        match part {
            ResponseContentPart::InputFile { .. } => false,
            ResponseContentPart::InputImage { image_url, .. } => match self {
                Self::Chat => image_url.is_some() && role != "developer",
                Self::Harmony => false,
            },
            ResponseContentPart::OutputText { .. }
            | ResponseContentPart::InputText { .. }
            | ResponseContentPart::Refusal { .. } => true,
        }
    }
}

/// `item` with the content parts `into` never carried left out (see
/// [`ReplayInto`]); an item that is not a message is returned as it is.
fn replayed_parts(mut item: ResponseInputOutputItem, into: ReplayInto) -> ResponseInputOutputItem {
    fn keep_carried(role: &str, parts: &mut Vec<ResponseContentPart>, into: ReplayInto) {
        parts.retain(|part| {
            let carried = into.carried(role, part);
            if !carried {
                let part_type = match part {
                    ResponseContentPart::InputFile { .. } => "input_file",
                    _ => "input_image",
                };
                debug!(
                    part_type,
                    "Stored message part the pipeline never carried; left out of the replay"
                );
            }
            carried
        });
    }
    match &mut item {
        ResponseInputOutputItem::Message { role, content, .. } => keep_carried(role, content, into),
        ResponseInputOutputItem::SimpleInputMessage {
            role,
            content: StringOrContentParts::Array(parts),
            ..
        } => keep_carried(role, parts, into),
        _ => {}
    }
    item
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
        ResponseInputOutputItem::Program { .. } => Some("program"),
        ResponseInputOutputItem::ProgramOutput { .. } => Some("program_output"),
        ResponseInputOutputItem::AdditionalTools { .. }
        | ResponseInputOutputItem::AgentMessage { .. }
        | ResponseInputOutputItem::ToolSearchCall { .. }
        | ResponseInputOutputItem::ToolSearchOutput { .. }
        | ResponseInputOutputItem::WebSearchCall { .. }
        | ResponseInputOutputItem::Message { .. }
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
/// (a built-in tool call, an older schema), which is left out. The row is
/// consumed: its strings become the item's, with no copy; a row that carries
/// its id on the row only has that id written into the item.
fn conversation_item_as_input(item: ConversationItem) -> Option<ResponseInputOutputItem> {
    match item.item_type.as_str() {
        "message" => {
            let (content, phase) = split_stored_message_content(item.content);
            match serde_json::from_value::<Vec<ResponseContentPart>>(content) {
                Ok(content) => Some(ResponseInputOutputItem::Message {
                    id: item.id.0,
                    role: item.role.unwrap_or_else(|| "user".to_string()),
                    content,
                    status: item.status,
                    phase,
                }),
                Err(e) => {
                    warn!(item_id = %item.id.0, error = %e, "Stored message item has no readable content; left out of the history");
                    None
                }
            }
        }
        "reasoning" if item.content.is_array() => match serde_json::from_value(item.content) {
            Ok(content) => Some(ResponseInputOutputItem::new_reasoning(
                item.id.0,
                vec![],
                content,
                item.status,
            )),
            Err(e) => {
                warn!(item_id = %item.id.0, error = %e, "Stored reasoning item has no readable content; left out of the history");
                None
            }
        },
        _ => {
            // A whole-item row keeps the item under its own fields; a row
            // added through the conversations API carries its id on the row
            // only, and an item is found by id when referenced.
            let mut whole = item.content;
            let id = item.id.0;
            if let Some(fields) = whole.as_object_mut() {
                fields
                    .entry("id")
                    .or_insert_with(|| Value::String(id.clone()));
            }
            match serde_json::from_value(whole) {
                Ok(input) => Some(input),
                Err(e) => {
                    warn!(item_id = %id, item_type = %item.item_type, error = %e, "Stored item has no readable form; left out of the history");
                    None
                }
            }
        }
    }
}

/// The input item a stored item replays as, read from the stored value in
/// place; `None` for one with no input form (a built-in tool call, an older
/// schema), which is left out.
fn stored_item_as_input(item: &Value) -> Option<ResponseInputOutputItem> {
    ResponseInputOutputItem::deserialize(item)
        .map_err(|e| warn!("Failed to deserialize stored item: {}. Item: {}", e, item))
        .ok()
}

/// The replayable items of a conversation, its newest
/// [`MAX_CONVERSATION_HISTORY_ITEMS`] oldest first, as the caller's `tenant`
/// may see it: a conversation of another tenant is as missing. An
/// `item_reference` row of the window is resolved as one of the request's own
/// is ([`resolve_item_references`], its 404 and its 400 included).
async fn conversation_items(
    stores: HistoryStores<'_>,
    conversation_id: &str,
    tenant: Option<&str>,
    into: ReplayInto,
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

    // The newest items, listed newest first so that the limit cuts the
    // oldest off, then put back in the order they were stored. One more than
    // the limit is asked for: a conversation of exactly the limit replays
    // whole, and only one with an item past the window is cut.
    let params = data_connector::ListParams {
        limit: MAX_CONVERSATION_HISTORY_ITEMS + 1,
        order: data_connector::SortOrder::Desc,
        after: None,
    };
    // A listing the store did not answer (or refused) is not an answer
    // without the conversation's context: the client reads a 503 for a
    // timed-out store and a 500 for any other failure, as every other
    // store-dependent answer of this path does.
    let mut stored = stores
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
    stored.reverse();
    let cut = stored.len() > MAX_CONVERSATION_HISTORY_ITEMS;
    if cut {
        stored.drain(..stored.len() - MAX_CONVERSATION_HISTORY_ITEMS);
    }
    let mut hidden = false;
    // The rows the rule hid, converted: the call of an output among them is
    // not replayed either.
    let mut skipped = Vec::new();
    // A row a stored response wrote was answered without the parts the
    // pipeline never carried; a row the client added to the conversation
    // itself (no response id) was answered by nobody and keeps every part,
    // so the conversion refuses it as it would the request's own.
    let items: Vec<ResponseInputOutputItem> = stored
        .into_iter()
        .filter_map(|row| {
            let visible = row.is_visible_to(tenant);
            hidden |= !visible;
            let answered = row.response_id.is_some();
            let item = conversation_item_as_input(row).filter(replays)?;
            if !visible {
                skipped.push(item);
                return None;
            }
            Some(if answered {
                replayed_parts(item, into)
            } else {
                item
            })
        })
        .collect();
    let mut items = without_calls_answered_by(items, &skipped);
    if cut {
        // The window is cut at the limit: it begins at its first turn boundary
        // (the tail of the turn the cut split is not replayed), and an output
        // whose call is before it is not either.
        debug!(
            conversation_id,
            limit = MAX_CONVERSATION_HISTORY_ITEMS,
            "Conversation history cut to its newest items"
        );
        items = from_turn_boundary(items);
    }
    // The output of a call the rule hid is no history, as after a cut.
    let lost_rows = cut || hidden;
    // A call the window holds as an `item_reference` row pairs as a stored
    // call does: what the boundary left is resolved before the outputs are
    // matched with their calls.
    let references: Vec<(usize, ResponseInputOutputItem)> = items
        .iter()
        .enumerate()
        .filter(|(_, item)| matches!(item, ResponseInputOutputItem::ItemReference { .. }))
        .map(|(at, item)| (at, item.clone()))
        .collect();
    if !references.is_empty() {
        let mut marked = resolve_item_references(stores.responses.as_ref(), items, tenant).await?;
        // Each reference row goes back in front of the item it named, as a
        // mark: the pair rule passes it by, whatever it takes out.
        for (at, row) in references.into_iter().rev() {
            marked.insert(at, row);
        }
        if !cut {
            // Not after a cut: the boundary may have dropped the output the
            // first run kept a stored call for, and a run here would take
            // that call from under the output the request brings.
            marked = without_calls_answered_by(marked, &skipped);
        }
        // An output before a reference pairs with the calls before that
        // reference alone: a call a reference names must not keep an
        // older output of its id.
        items = Vec::with_capacity(marked.len());
        for item in marked {
            if !matches!(item, ResponseInputOutputItem::ItemReference { .. }) {
                items.push(item);
            } else if lost_rows {
                items = without_outputs_of_missing_calls(items);
            }
        }
    }
    Ok(if lost_rows {
        without_outputs_of_missing_calls(items)
    } else {
        items
    })
}

/// Replace every `item_reference` input item by the item it points to: an item
/// already in the list (loaded from the previous_response_id chain or the
/// conversation history) or an output item of a stored response the caller's
/// `tenant` may see. An unknown id answers 404, as the public API does, and
/// so does an item of another tenant's response.
async fn resolve_item_references(
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
        match ResponseInputOutputItem::deserialize(candidate) {
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
        | ResponseInputOutputItem::McpApprovalRequest { id, .. }
        | ResponseInputOutputItem::ComputerCall { id, .. }
        | ResponseInputOutputItem::LocalShellCallOutput { id, .. }
        | ResponseInputOutputItem::McpCall { id, .. }
        | ResponseInputOutputItem::McpListTools { id, .. } => Some(id.as_str()),
        ResponseInputOutputItem::Reasoning { id, .. }
        | ResponseInputOutputItem::AdditionalTools { id, .. }
        | ResponseInputOutputItem::AgentMessage { id, .. }
        | ResponseInputOutputItem::ToolSearchCall { id, .. }
        | ResponseInputOutputItem::ToolSearchOutput { id, .. }
        | ResponseInputOutputItem::WebSearchCall { id, .. }
        | ResponseInputOutputItem::Program { id, .. }
        | ResponseInputOutputItem::ProgramOutput { id, .. }
        | ResponseInputOutputItem::ImageGenerationCall { id, .. }
        | ResponseInputOutputItem::LocalShellCall { id, .. }
        | ResponseInputOutputItem::FunctionToolCall { id, .. }
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
#[path = "../../../../../tests/unit/routers/grpc/common/responses/history/tests.rs"]
pub(crate) mod tests;
