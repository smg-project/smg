//! Message threads on the Messages route (beta): the turns of a conversation
//! kept in this process, so a client sends each turn's new messages only.
//!
//! A `thread: {"type": "create"}` request is a whole conversation; its
//! answer is kept under the answer's `id`, with the request's `system`,
//! `tools`, messages and fixed fields. A `thread: {"type": "continue",
//! "previous_message_id"}` request names an answer and carries the messages
//! since it; the gateway rebuilds the conversation from the stored turns
//! (the create turn's `system` and `tools`, every turn's messages and
//! answer, then the new messages), checks the fixed fields against the
//! create turn's, and routes the plain request this would have been. The
//! store is the Responses history's in-memory store with its record schema
//! unchanged: a turn is a [`StoredResponse`] whose `input` holds the turn's
//! request fields ([`TurnInput`]) and whose `raw_response` holds the answer,
//! chained by `previous_response_id`, so a thread holds its conversation
//! once, not once per turn. The public API deletes a thread no more than
//! an hour after its last use, the reference a deployment's bound is read
//! against. The store keeps the turns the operator's `memory` bounds
//! allow, the `max_responses` and `retention_secs` of the memory history
//! backend (unset: 10,000 turns, each for a day after it was written; a
//! `retention_secs` of 3,600 stays within the API's hour), the oldest
//! evicted first, on every write; a thread is read back whole up to that
//! count. A turn is written under its answer's id.
//! An id that is already a stored turn's (a repeated `rid`, a backend's
//! repeated id, a guess) would hand the client an id that continues another
//! conversation, so the answer is reissued under a fresh id, in the answer
//! the client reads and in the store.
//!
//! A thread turn's message id is the gateway's: the request's `rid` is
//! replaced by a fresh `msg_` id before the route (on the regular gRPC path
//! the request id is the message id, so a client's repeated `rid` would name
//! a stored turn), and the client's `rid` is echoed as the answer's
//! `request-id` header, the trace id it stays, under the request-id layer's
//! inbound rule and where the route set none; a provider never receives
//! `rid`, the adapter that mounts a provider router clears it. Every turn of
//! a thread carries the token minted at its create turn (the record's
//! `conversation_id`); a root under a reused id, written after the thread's
//! own root was evicted, carries another, so an old child's continue onto it
//! is the 404, never a splice of two conversations.
//!
//! A streamed answer is stored as well: the thread layer reads the events
//! on their way to the client, takes the turn's place in the store at
//! `message_start` (a taken id is reissued there, in the event the client
//! reads), folds the content blocks and the final `message_delta` into the
//! answer, and writes it whole at `message_stop`; the bytes the client reads
//! are the router's own, except a reissued `message_start`.
//!
//! Best effort, as the public API's threads are: a turn that was not stored
//! (an error, an answer the gateway could not read, a write the store
//! refused, a stream cut before its `message_stop`, an evicted record) makes
//! the next continue a 404 `thread_not_found`, and the client replays the
//! conversation in a new create request, as the public SDKs' thread helper
//! does on its own. An id may be continued more than once (a fork); the
//! public API's `thread_already_continued` is not answered. A continue whose
//! rebuilt conversation breaks a request rule is the same 404: the public
//! API's codes name no such case, and the client's replay (the whole
//! conversation as a create request) meets the 400 that names the rule.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use axum::{
    body::{to_bytes, Body, Bytes},
    http::{header, HeaderValue, StatusCode},
    response::Response,
};
use http_body::{Body as HttpBody, Frame, SizeHint};
use openai_protocol::messages::{
    ContentBlock, ContentBlockDelta, CreateMessageRequest, ErrorDetails, ErrorEnvelope,
    ErrorResponse, InputContent, InputContentBlock, InputMessage, Message, MessageDelta,
    MessageDeltaUsage, MessageStreamEvent, MessageThread, Role, SystemContent, Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use smg_data_connector::{
    visible_chain, MemoryConfig, MemoryResponseStorage, ResponseId, ResponseStorage,
    ResponseStorageError, StoredResponse,
};
use tracing::{debug, warn};

use crate::{
    middleware::{
        messages_error_type,
        request_id::{is_acceptable_inbound_id, REQUEST_ID},
    },
    observability::metrics::Metrics,
    routers::{
        common::sse_rechunk::{find_frame_end, DELIMITER_OVERLAP},
        error as route_error,
    },
};

/// An answer over this cannot be read back for its record: the Messages
/// route's own answers are far smaller.
const ANSWER_LIMIT: usize = 64 << 20;

/// What a stored turn holds in its record's `input`: the fields the
/// conversation is rebuilt from. The create turn carries the thread's
/// `system`, `tools` and fixed fields; every turn carries the messages it
/// added (the create request's conversation, a continue request's new
/// messages); the answer is the record's `raw_response`.
#[derive(Deserialize)]
struct TurnInput {
    #[serde(default)]
    system: Option<SystemContent>,
    #[serde(default)]
    tools: Option<Vec<Tool>>,
    messages: Vec<InputMessage>,
    #[serde(default)]
    fixed: Map<String, Value>,
}

impl TurnInput {
    /// The record's `input` of the turn `request` sends: the create turn
    /// (`root`) with the thread's `system`, `tools` and fixed fields, any
    /// other with its messages alone.
    fn record(request: &CreateMessageRequest, root: bool) -> Value {
        if root {
            json!({
                "system": request.system,
                "tools": request.tools,
                "messages": request.messages,
                "fixed": fixed_fields(request),
            })
        } else {
            json!({ "messages": request.messages })
        }
    }
}

/// The request fields a continue request repeats as the create request sent
/// them, by name, as JSON: `model`, `tool_choice`, `thinking`,
/// `mcp_servers`, `container`, `output_config`, `output_format`, and every
/// field the typed request does not model (the beta fields in `other`). The
/// sampling fields, `max_tokens`, `stop_sequences`, `metadata`, `stream`,
/// `service_tier` and `rid` may change from turn to turn.
fn fixed_fields(request: &CreateMessageRequest) -> Map<String, Value> {
    let mut fixed = Map::new();
    let mut keep = |name: &str, value: Value| {
        if !value.is_null() {
            fixed.insert(name.to_string(), value);
        }
    };
    keep("model", Value::String(request.model.clone()));
    keep("tool_choice", json_of(&request.tool_choice));
    keep("thinking", json_of(&request.thinking));
    keep("mcp_servers", json_of(&request.mcp_servers));
    keep("container", json_of(&request.container));
    keep("output_config", json_of(&request.output_config));
    keep("output_format", json_of(&request.output_format));
    for (name, value) in &request.other {
        keep(name, value.clone());
    }
    fixed
}

fn json_of<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// The fixed fields that differ between the create turn's and a continue
/// request's, by name, sorted.
fn differing(stored: &Map<String, Value>, sent: &Map<String, Value>) -> Vec<String> {
    let mut names: Vec<&String> = stored
        .keys()
        .chain(sent.keys())
        .filter(|name| stored.get(*name) != sent.get(*name))
        .collect();
    names.sort();
    names.dedup();
    names.into_iter().cloned().collect()
}

/// The stored answer as the assistant message of the conversation: its
/// content blocks read in their wire shape as input blocks (text,
/// `tool_use`, the thinking blocks and the server-tool blocks share it).
fn answer_message(answer: &Value) -> Result<InputMessage, Refusal> {
    let content = answer
        .get("content")
        .ok_or_else(|| Refusal::NotFound("a stored answer has no content".to_string()))?;
    let blocks = Vec::<InputContentBlock>::deserialize(content).map_err(|error| {
        Refusal::NotFound(format!("a stored answer's content is no input: {error}"))
    })?;
    Ok(InputMessage {
        role: Role::Assistant,
        content: InputContent::Blocks(blocks),
    })
}

/// Why a continue request is refused before the model runs.
enum Refusal {
    /// Nothing this request can continue: the public API's 404, which the
    /// client answers with a replay. The reason goes to the log alone, as
    /// the public API's refusal does not say.
    NotFound(String),
    /// The fixed fields the request changed.
    Mismatch(Vec<String>),
}

impl Refusal {
    fn outcome(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "not_found",
            Self::Mismatch(_) => "mismatch",
        }
    }

    fn into_response(self, previous_message_id: &str) -> Response {
        match self {
            Self::NotFound(why) => {
                debug!(previous_message_id, why = %why, "message thread: continue refused");
                refused(
                    StatusCode::NOT_FOUND,
                    "thread_not_found",
                    format!(
                        "thread: no thread can be continued from `{previous_message_id}`; \
                         send the whole conversation in a new create request"
                    ),
                )
            }
            Self::Mismatch(names) => refused(
                StatusCode::BAD_REQUEST,
                "thread_fingerprint_mismatch",
                format!(
                    "thread: the request changes {}: a continue request repeats the create \
                     request's fixed fields and sends the new messages only",
                    names.join(", ")
                ),
            ),
        }
    }
}

/// A refusal in the Messages envelope with its `details.error_code`, under
/// the gateway's error-code header and metric label (`create_error`); the
/// envelope layer fills in the request id.
fn refused(status: StatusCode, code: &str, message: String) -> Response {
    let mut response = route_error::create_error(status, code, message.as_str());
    let error = ErrorResponse {
        error_type: messages_error_type(status).to_string(),
        message,
        details: Some(ErrorDetails {
            error_code: code.to_string(),
            unsupported: None,
        }),
    };
    let body = serde_json::to_vec(&ErrorEnvelope::new(error, "")).unwrap_or_default();
    *response.body_mut() = Body::from(body);
    response
}

/// How a turn's record was written.
enum Written {
    /// Under the answer's id.
    Kept,
    /// The answer's id was a stored turn's: the answer, serialized again
    /// with the fresh id it was reissued under.
    Reissued(Bytes),
    /// Not written: the answer has no id, or the reissued write was refused.
    Refused,
}

/// A fresh message id, as the request-id layer mints them.
fn fresh_message_id() -> String {
    format!(
        "msg_{}",
        uuid::Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)).simple()
    )
}

/// A fresh thread token, minted at the create turn and carried by every turn
/// of the thread: a root written later under a reused id carries another.
fn fresh_thread_token() -> String {
    format!(
        "thr_{}",
        uuid::Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)).simple()
    )
}

/// A thread turn in flight: its op, the client's `rid` when it sent one
/// (echoed as the answer's `request-id`, the trace id it stays), its record
/// and whether its answer streams (a streamed answer is stored from its
/// events as the client reads them, not read back).
pub(crate) struct ThreadTurn {
    op: &'static str,
    trace: Option<String>,
    record: TurnRecord,
    streamed: bool,
}

impl ThreadTurn {
    /// The client's `rid`, when it sent one: the trace id, and the lineage a
    /// routing key is derived from, since the request's own `rid` is the
    /// minted message id by the time a router reads it.
    pub(crate) fn trace(&self) -> Option<&str> {
        self.trace.as_deref()
    }
}

/// What a stored turn's record holds besides the answer, taken from the
/// request before the thread's history is spliced into it; `thread` is the
/// token minted at the create turn, which every turn of the thread carries.
struct TurnRecord {
    previous: Option<ResponseId>,
    input: Value,
    model: String,
    tenant: Option<String>,
    thread: String,
}

impl TurnRecord {
    /// The store's record of this turn: `answer` under `id`, the turn's
    /// request as the record's input, the thread's token.
    fn record(self, id: ResponseId, answer: Value) -> StoredResponse {
        let mut record = StoredResponse::new(self.previous);
        record.id = id;
        record.input = self.input;
        record.model = Some(self.model);
        record.raw_response = answer;
        record.tenant = self.tenant;
        record.conversation_id = Some(self.thread);
        record
    }

    /// The turn's place in the store while its answer streams: `id` taken,
    /// the `message_start` message as the answer and no input yet (a
    /// continue meanwhile is the 404); the record replaces it at
    /// `message_stop`, a cut stream removes it.
    fn reservation(&self, id: ResponseId, message: &Message) -> StoredResponse {
        let mut record = StoredResponse::new(self.previous.clone());
        record.id = id;
        record.model = Some(self.model.clone());
        record.raw_response = serde_json::to_value(message).unwrap_or(Value::Null);
        record.tenant.clone_from(&self.tenant);
        record.conversation_id = Some(self.thread.clone());
        record
    }
}

/// The threads of this gateway, in the store of their turns.
pub(crate) struct MessageThreads {
    store: MemoryResponseStorage,
    /// The store's turn count, the depth a thread is read back to.
    max_turns: usize,
    /// The bytes an answer may hold, read whole or streamed: `ANSWER_LIMIT`.
    answer_limit: usize,
}

impl MessageThreads {
    /// The threads under the default bounds ([`MemoryConfig::default`]).
    pub(crate) fn new() -> Self {
        Self::from_config(&MemoryConfig::default())
    }

    /// The threads in a store bounded as the operator's `memory` section
    /// says: at most `max_responses` turns, each kept for `retention_secs`
    /// after it was written, the oldest evicted first on every write. The
    /// count is also the depth a thread is read back to, so no kept thread
    /// is cut short of its create turn.
    pub(crate) fn from_config(config: &MemoryConfig) -> Self {
        Self {
            store: MemoryResponseStorage::with_config(config),
            max_turns: config.max_responses,
            answer_limit: ANSWER_LIMIT,
        }
    }

    /// The threads in a store of `max_turns` turns under the default
    /// retention.
    #[cfg(test)]
    fn bounded(max_turns: usize) -> Self {
        Self::from_config(&MemoryConfig {
            max_responses: max_turns,
            ..MemoryConfig::default()
        })
    }

    /// The threads in a store of `max_turns` turns, each kept for
    /// `retention_secs` after it was written.
    #[cfg(test)]
    fn kept(max_turns: usize, retention_secs: u64) -> Self {
        Self::from_config(&MemoryConfig {
            max_responses: max_turns,
            retention_secs: Some(retention_secs),
            ..MemoryConfig::default()
        })
    }

    /// The threads with answers bounded at `answer_limit` bytes.
    #[cfg(test)]
    fn limited(mut self, answer_limit: usize) -> Self {
        self.answer_limit = answer_limit;
        self
    }

    /// The request to route and, for a thread turn, the turn to store with
    /// its answer: a create request as it came, its `thread` taken off; a
    /// continue request as the conversation it continues, rebuilt from the
    /// stored turns. Or the public API's refusal of a continue: a 404
    /// `thread_not_found` when nothing can be continued from the id
    /// (unknown, evicted, another tenant's, a turn the gateway cannot read,
    /// a conversation that does not take the new messages), a 400
    /// `thread_fingerprint_mismatch` naming the fixed fields the request
    /// changed.
    pub(crate) async fn resolve(
        &self,
        mut request: CreateMessageRequest,
        tenant: Option<String>,
    ) -> Result<(CreateMessageRequest, Option<ThreadTurn>), Response> {
        let Some(thread) = request.thread.take() else {
            return Ok((request, None));
        };
        let streamed = request.is_stream();
        let model = request.model.clone();
        // The gateway mints the turn's message id: on the regular gRPC path
        // the request id is the message id, so a client's repeated rid would
        // name a stored turn. The rid stays a trace id, echoed as the
        // answer's request-id header.
        let trace = request.rid.replace(fresh_message_id());
        match thread {
            MessageThread::Create => {
                let record = TurnRecord {
                    previous: None,
                    input: TurnInput::record(&request, true),
                    model,
                    tenant,
                    thread: fresh_thread_token(),
                };
                Ok((
                    request,
                    Some(ThreadTurn {
                        op: "create",
                        trace,
                        record,
                        streamed,
                    }),
                ))
            }
            MessageThread::Continue {
                previous_message_id,
            } => {
                let input = TurnInput::record(&request, false);
                match self
                    .continue_from(&previous_message_id, tenant.as_deref(), &mut request)
                    .await
                {
                    Ok(thread) => {
                        let record = TurnRecord {
                            previous: Some(ResponseId::from(previous_message_id)),
                            input,
                            model,
                            tenant,
                            thread,
                        };
                        Ok((
                            request,
                            Some(ThreadTurn {
                                op: "continue",
                                trace,
                                record,
                                streamed,
                            }),
                        ))
                    }
                    Err(refusal) => {
                        Metrics::record_message_thread_turn("continue", refusal.outcome());
                        Err(refusal.into_response(&previous_message_id))
                    }
                }
            }
        }
    }

    /// `request` as the conversation of the thread ending at `id`, continued
    /// by its own messages; answers the thread's token, for the new turn's
    /// record.
    async fn continue_from(
        &self,
        id: &str,
        tenant: Option<&str>,
        request: &mut CreateMessageRequest,
    ) -> Result<String, Refusal> {
        let (thread, turns) = self.turns(id, tenant).await?;
        let root = TurnInput::deserialize(&turns[0].input)
            .map_err(|error| Refusal::NotFound(format!("the create turn is no turn: {error}")))?;
        let changed = differing(&root.fixed, &fixed_fields(request));
        if !changed.is_empty() {
            return Err(Refusal::Mismatch(changed));
        }
        let mut messages = root.messages;
        messages.push(answer_message(&turns[0].raw_response)?);
        for turn in &turns[1..] {
            let input = TurnInput::deserialize(&turn.input)
                .map_err(|error| Refusal::NotFound(format!("a stored turn is no turn: {error}")))?;
            messages.extend(input.messages);
            messages.push(answer_message(&turn.raw_response)?);
        }
        messages.append(&mut request.messages);
        request.system = root.system;
        request.tools = root.tools;
        request.messages = messages;
        if let Some(violation) = request.contract_violation() {
            return Err(Refusal::NotFound(format!(
                "the conversation does not take the new messages: {violation}"
            )));
        }
        Ok(thread)
    }

    /// The stored turns of the thread ending at `id` that `tenant` may see,
    /// the create turn first, with the thread's token. A root whose token
    /// not every turn carries is another thread's create turn, written under
    /// the id of a root the store let go: refused, never spliced.
    async fn turns(
        &self,
        id: &str,
        tenant: Option<&str>,
    ) -> Result<(String, Vec<Arc<StoredResponse>>), Refusal> {
        let chain = match self
            .store
            .get_response_chain(&ResponseId::from(id), Some(self.max_turns))
            .await
        {
            Ok(chain) => chain.responses,
            Err(ResponseStorageError::ResponseNotFound(_)) => {
                return Err(Refusal::NotFound("no turn under the id".to_string()));
            }
            Err(error) => {
                return Err(Refusal::NotFound(format!("the store failed: {error}")));
            }
        };
        let turns = visible_chain(chain, tenant);
        let thread = match turns.first() {
            None => return Err(Refusal::NotFound("no turn under the id".to_string())),
            Some(root) if root.previous_response_id.is_some() => {
                return Err(Refusal::NotFound("the create turn is gone".to_string()));
            }
            Some(root) => root.conversation_id.clone(),
        };
        let same_thread =
            |turn: &Arc<StoredResponse>| turn.conversation_id == thread && thread.is_some();
        match thread {
            Some(thread) if turns.iter().all(same_thread) => Ok((thread, turns)),
            _ => Err(Refusal::NotFound("the create turn is gone".to_string())),
        }
    }

    /// The answer, with the turn stored under the answer's id when it is a
    /// 200 message; an id that is already a stored turn's is reissued, in
    /// the answer and in the store. A streamed answer is stored from its
    /// events by [`StreamedTurn`] as the client reads them. An error, an
    /// answer the gateway cannot read or a write the store refused leaves
    /// the turn unstored and the answer as it is: the next continue is a
    /// 404 and the client replays.
    pub(crate) async fn stored(&self, turn: ThreadTurn, response: Response) -> Response {
        let mut response = response;
        // The client's rid, the trace id, on the answer as the header the
        // SDKs read, under the request-id layer's inbound rule (one to 128
        // visible-ASCII characters: a client cannot size the header) and
        // only where the route set none (a relayed upstream id is the one
        // its support asks for); the layer keeps a request-id the route set.
        if let Some(value) = turn
            .trace
            .as_deref()
            .filter(|rid| is_acceptable_inbound_id(rid))
            .and_then(|rid| HeaderValue::from_str(rid).ok())
        {
            response.headers_mut().entry(&REQUEST_ID).or_insert(value);
        }
        let op = turn.op;
        if response.status() != StatusCode::OK {
            Metrics::record_message_thread_turn(op, "unstored");
            return response;
        }
        if turn.streamed {
            let (parts, body) = response.into_parts();
            let streamed =
                StreamedTurn::new(body, turn.record, op, self.store.clone(), self.answer_limit);
            return Response::from_parts(parts, Body::new(streamed));
        }
        let turn = turn.record;
        let (mut parts, body) = response.into_parts();
        let bytes = match to_bytes(body, self.answer_limit).await {
            Ok(bytes) => bytes,
            Err(error) => {
                warn!(%error, "message thread: the answer could not be read back");
                Metrics::record_message_thread_turn(op, "unstored");
                return route_error::internal_error(
                    "thread_answer_unreadable",
                    "The answer of a thread turn could not be read back",
                );
            }
        };
        let (outcome, bytes) = match serde_json::from_slice::<Value>(&bytes) {
            Ok(answer) => match self.store(turn, answer) {
                Written::Kept => ("stored", bytes),
                Written::Reissued(reissued) => {
                    parts.headers.remove(header::CONTENT_LENGTH);
                    ("reissued", reissued)
                }
                Written::Refused => ("unstored", bytes),
            },
            Err(error) => {
                warn!(%error, "message thread: the answer is not JSON, the turn is not stored");
                ("unstored", bytes)
            }
        };
        Metrics::record_message_thread_turn(op, outcome);
        Response::from_parts(parts, Body::from(bytes))
    }

    /// The record of `turn` with `answer`, written under the answer's id
    /// while nothing is stored under it. An id that is already a stored
    /// turn's would hand the client an id that continues another
    /// conversation, so the answer is reissued under a fresh id.
    fn store(&self, turn: TurnRecord, answer: Value) -> Written {
        let Some(id) = answer
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(ResponseId::from)
        else {
            warn!("message thread: the answer has no id, the turn is not stored");
            return Written::Refused;
        };
        let record = turn.record(id, answer);
        let mut record = match self.store.store_response_if_absent(record) {
            Ok(_) => return Written::Kept,
            Err(record) => *record,
        };
        let fresh = fresh_message_id();
        debug!(
            taken = %record.id.0,
            fresh,
            "message thread: the answer's id is a stored turn's; reissued"
        );
        record.raw_response["id"] = Value::String(fresh.clone());
        record.id = ResponseId::from(fresh);
        let Ok(bytes) = serde_json::to_vec(&record.raw_response) else {
            return Written::Refused;
        };
        match self.store.store_response_if_absent(record) {
            Ok(_) => Written::Reissued(Bytes::from(bytes)),
            Err(record) => {
                warn!(
                    id = %record.id.0,
                    "message thread: the reissued write was refused, the turn is not stored"
                );
                Written::Refused
            }
        }
    }
}

/// A streamed answer on its way to the client, read for its record. The turn
/// takes its place in the store at `message_start` (under a fresh id when
/// the event's id is a stored turn's, rewritten in the event the client
/// reads), the content blocks and the final `message_delta` are folded as
/// the events pass, and the whole answer replaces the place at
/// `message_stop`. A stream that ends, fails or is dropped before that
/// leaves no turn: the next continue is the 404. The bytes the client reads
/// are the router's own, except a reissued `message_start`.
struct StreamedTurn {
    inner: Body,
    store: MemoryResponseStorage,
    turn: Option<TurnRecord>,
    op: &'static str,
    /// The bytes before the first complete event, held until it is whole: a
    /// `message_start` may be rewritten. Empty once the stream passes.
    held: Vec<u8>,
    /// The bytes of the event not yet complete, parsed when it is.
    carry: Vec<u8>,
    /// Where the last scan of `held` and of `carry` stopped: the next scan
    /// resumes there, so an event that arrives in many frames is read once.
    held_scanned: usize,
    carry_scanned: usize,
    /// Every byte passes as it comes: `message_start` is through, or the
    /// stream opened with an error.
    passing: bool,
    /// The place taken in the store.
    place: Option<Place>,
    answer: Option<Assembled>,
    /// The bytes of content an answer may retain, the bound the whole
    /// answer is read under on the unstreamed path.
    limit: usize,
    /// The outcome counted, once.
    outcome: Option<&'static str>,
    ended: bool,
}

/// A streamed turn's place in the store: the reservation as stored, the
/// identity the final write and the removal compare against (a place the
/// store's bounds dropped meanwhile, and the turn that took its id since,
/// are left alone), and whether it was taken under a fresh id.
struct Place {
    held: Arc<StoredResponse>,
    reissued: bool,
}

/// The answer under construction: the `message_start` message, its content
/// blocks by index, each tool call's input JSON as it streams, and the bytes
/// retained so far: each block's start event, and the text, thinking,
/// signature and input its deltas add.
struct Assembled {
    message: Message,
    blocks: Vec<ContentBlock>,
    partial_json: Vec<String>,
    retained: usize,
}

/// An answer may hold this many content blocks. One with a block past them
/// is not stored: the record is the answer the client read, or nothing.
const MAX_BLOCKS: usize = 1024;

/// More bytes than this without an event's end, before the first event or
/// within one, are not a stream this gateway reads: they pass to the client
/// as they are and the turn is not stored.
const EVENT_LIMIT: usize = 1 << 20;

impl StreamedTurn {
    fn new(
        inner: Body,
        turn: TurnRecord,
        op: &'static str,
        store: MemoryResponseStorage,
        limit: usize,
    ) -> Self {
        Self {
            inner,
            store,
            op,
            turn: Some(turn),
            held: Vec::new(),
            carry: Vec::new(),
            held_scanned: 0,
            carry_scanned: 0,
            passing: false,
            place: None,
            answer: None,
            limit,
            outcome: None,
            ended: false,
        }
    }

    /// The bytes the client reads for `data`: all of them, once the first
    /// complete event is through; until then they are held.
    fn pass(&mut self, data: Bytes) -> Option<Bytes> {
        if self.passing {
            self.fold(&data);
            return Some(data);
        }
        self.held.extend_from_slice(&data);
        let mut out = Vec::new();
        while let Some(end) = event_end(&self.held, self.held_scanned) {
            self.held_scanned = 0;
            match parse_event(&self.held[..end]) {
                Ok(Some(MessageStreamEvent::MessageStart { message })) => {
                    let event: Vec<u8> = self.held.drain(..end).collect();
                    out.extend_from_slice(&self.start(message, &event));
                    self.passing = true;
                    let rest = std::mem::take(&mut self.held);
                    self.fold(&rest);
                    out.extend_from_slice(&rest);
                    break;
                }
                Ok(Some(MessageStreamEvent::Error { .. })) => {
                    self.passing = true;
                    out.append(&mut self.held);
                    break;
                }
                // Nothing is recorded before `message_start`: every other
                // event, readable or not, passes.
                _ => out.extend(self.held.drain(..end)),
            }
        }
        self.held_scanned = self.held.len();
        if !self.passing && self.held.len() > EVENT_LIMIT {
            self.passing = true;
            out.append(&mut self.held);
            self.held_scanned = 0;
            self.decline("no event end within the size limit");
        }
        (!out.is_empty()).then(|| Bytes::from(out))
    }

    /// The `message_start` bytes the client reads: the event as it came, or
    /// rewritten with a fresh id when the message's id is a stored turn's.
    /// The turn's place in the store is taken here.
    fn start(&mut self, mut message: Message, event: &[u8]) -> Vec<u8> {
        let Some(turn) = &self.turn else {
            return event.to_vec();
        };
        let place = turn.reservation(ResponseId::from(message.id.as_str()), &message);
        let mut reissued = false;
        let held = match self.store.store_response_if_absent(place) {
            Ok(held) => held,
            Err(_) => {
                let fresh = fresh_message_id();
                debug!(
                    taken = %message.id,
                    fresh,
                    "message thread: a streamed answer's id is a stored turn's; reissued"
                );
                message.id = fresh;
                let place = turn.reservation(ResponseId::from(message.id.as_str()), &message);
                let Ok(held) = self.store.store_response_if_absent(place) else {
                    return event.to_vec();
                };
                reissued = true;
                held
            }
        };
        let bytes = if reissued {
            match start_event(&message) {
                Some(bytes) => bytes,
                None => {
                    self.store.remove_response_if_held(&held);
                    return event.to_vec();
                }
            }
        } else {
            event.to_vec()
        };
        self.place = Some(Place { held, reissued });
        self.answer = Some(Assembled {
            message,
            blocks: Vec::new(),
            partial_json: Vec::new(),
            retained: 0,
        });
        bytes
    }

    /// The events in `data` read for the record, while a place is held. Data
    /// this gateway cannot read gives the place back: the record is the
    /// answer the client read, or nothing.
    fn fold(&mut self, data: &[u8]) {
        if self.place.is_none() {
            return;
        }
        self.carry.extend_from_slice(data);
        while let Some(end) = event_end(&self.carry, self.carry_scanned) {
            self.carry_scanned = 0;
            let parsed = parse_event(&self.carry[..end]);
            self.carry.drain(..end);
            match parsed {
                Ok(Some(event)) => self.apply(event, end),
                Ok(None) => {}
                Err(Unreadable) => self.decline("data this gateway cannot read"),
            }
            if self.place.is_none() {
                self.carry.clear();
                break;
            }
        }
        self.carry_scanned = self.carry.len();
        if self.place.is_some() && self.carry.len() > EVENT_LIMIT {
            self.decline("no event end within the size limit");
            self.carry.clear();
            self.carry_scanned = 0;
        }
    }

    /// `event` folded into the answer; `size` is the bytes it arrived in.
    fn apply(&mut self, event: MessageStreamEvent, size: usize) {
        let Some(answer) = &mut self.answer else {
            return;
        };
        match event {
            MessageStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                if answer.start_block(index, content_block, size).is_err() {
                    self.decline("a content block past the bound on blocks");
                }
            }
            MessageStreamEvent::ContentBlockDelta { index, delta } => answer.delta(index, delta),
            MessageStreamEvent::ContentBlockStop { index } => {
                if answer.stop_block(index).is_err() {
                    self.decline("a tool call's input deltas are no JSON document");
                }
            }
            MessageStreamEvent::MessageDelta { delta, usage } => answer.message_delta(delta, usage),
            MessageStreamEvent::MessageStop => self.finish(),
            MessageStreamEvent::Error { .. } => self.give_up(),
            MessageStreamEvent::MessageStart { .. } | MessageStreamEvent::Ping => {}
        }
        if self
            .answer
            .as_ref()
            .is_some_and(|answer| answer.retained > self.limit)
        {
            self.decline("an answer over the size limit");
        }
    }

    /// The whole answer in the turn's place, at `message_stop`. A place the
    /// store's bounds dropped meanwhile is left to whatever took its id: the
    /// turn is not stored.
    fn finish(&mut self) {
        let (Some(place), Some(turn), Some(mut answer)) =
            (self.place.take(), self.turn.take(), self.answer.take())
        else {
            return;
        };
        answer.message.content = answer.blocks;
        let value = match serde_json::to_value(&answer.message) {
            Ok(value) => value,
            Err(error) => {
                warn!(%error, "message thread: the streamed answer does not serialize, the turn is not stored");
                self.store.remove_response_if_held(&place.held);
                self.count("unstored");
                return;
            }
        };
        let id = place.held.id.clone();
        if self
            .store
            .replace_response_if_held(&place.held, turn.record(id, value))
        {
            self.count(if place.reissued { "reissued" } else { "stored" });
        } else {
            warn!(
                id = %place.held.id.0,
                "message thread: the streamed answer's place was dropped by the store's bounds, the turn is not stored"
            );
            self.count("unstored");
        }
    }

    /// The place given back: an error event, a cut stream, a dropped body.
    /// A place the store's bounds dropped meanwhile, and the turn that took
    /// its id since, are left alone.
    fn give_up(&mut self) {
        if let Some(place) = self.place.take() {
            self.store.remove_response_if_held(&place.held);
        }
        self.answer = None;
        self.count("unstored");
    }

    /// The place given back for an answer this gateway declines to store:
    /// data it cannot read, or more of it than the bounds allow. The log
    /// says why.
    fn decline(&mut self, why: &'static str) {
        warn!(why, "message thread: the streamed turn is not stored");
        self.give_up();
    }

    fn count(&mut self, outcome: &'static str) {
        if self.outcome.is_none() {
            self.outcome = Some(outcome);
            Metrics::record_message_thread_turn(self.op, outcome);
        }
    }
}

impl Drop for StreamedTurn {
    fn drop(&mut self) {
        self.give_up();
    }
}

impl HttpBody for StreamedTurn {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // `Body` is `Unpin`, and so is everything else held here.
        let this = self.get_mut();
        loop {
            if this.ended {
                return Poll::Ready(None);
            }
            match Pin::new(&mut this.inner).poll_frame(cx) {
                Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                    Ok(data) => {
                        if let Some(out) = this.pass(data) {
                            return Poll::Ready(Some(Ok(Frame::data(out))));
                        }
                    }
                    Err(frame) => return Poll::Ready(Some(Ok(frame))),
                },
                Poll::Ready(None) => {
                    this.ended = true;
                    this.give_up();
                    let held = std::mem::take(&mut this.held);
                    this.held_scanned = 0;
                    return if held.is_empty() {
                        Poll::Ready(None)
                    } else {
                        Poll::Ready(Some(Ok(Frame::data(Bytes::from(held)))))
                    };
                }
                Poll::Ready(Some(Err(error))) => {
                    this.ended = true;
                    this.give_up();
                    return Poll::Ready(Some(Err(error)));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.ended || (self.inner.is_end_stream() && self.held.is_empty())
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

impl Assembled {
    /// The block at `index`, as its start event of `size` bytes gave it. A
    /// block may arrive whole in that event (a server tool's result, a block
    /// this gateway does not read), so the event counts toward the bytes
    /// retained, as a delta does; one that replaces a block counts too. A
    /// block past `MAX_BLOCKS` has no place in the record, which would then
    /// not be the answer the client read.
    fn start_block(
        &mut self,
        index: u32,
        block: ContentBlock,
        size: usize,
    ) -> Result<(), Unreadable> {
        let Some(index) = slot(index) else {
            return Err(Unreadable);
        };
        self.retained += size;
        while self.blocks.len() < index {
            self.blocks.push(ContentBlock::Text {
                text: String::new(),
                citations: None,
            });
            self.partial_json.push(String::new());
        }
        if index == self.blocks.len() {
            self.blocks.push(block);
            self.partial_json.push(String::new());
        } else {
            self.blocks[index] = block;
            self.partial_json[index].clear();
        }
        Ok(())
    }

    fn delta(&mut self, index: u32, delta: ContentBlockDelta) {
        let Some(index) = slot(index) else {
            return;
        };
        let (Some(block), Some(partial)) =
            (self.blocks.get_mut(index), self.partial_json.get_mut(index))
        else {
            return;
        };
        match (delta, block) {
            (ContentBlockDelta::TextDelta { text: more }, ContentBlock::Text { text, .. }) => {
                self.retained += more.len();
                text.push_str(&more);
            }
            (ContentBlockDelta::InputJsonDelta { partial_json }, _) => {
                self.retained += partial_json.len();
                partial.push_str(&partial_json);
            }
            (
                ContentBlockDelta::ThinkingDelta { thinking: more },
                ContentBlock::Thinking { thinking, .. },
            ) => {
                self.retained += more.len();
                thinking.push_str(&more);
            }
            (
                ContentBlockDelta::SignatureDelta { signature: more },
                ContentBlock::Thinking { signature, .. },
            ) => {
                self.retained += more.len();
                signature.push_str(&more);
            }
            (
                ContentBlockDelta::CitationsDelta { citation },
                ContentBlock::Text { citations, .. },
            ) => citations.get_or_insert_with(Vec::new).push(citation),
            _ => {}
        }
    }

    /// A tool call's input, parsed from its deltas when the block closes.
    /// Deltas that are no JSON document make the answer unreadable.
    fn stop_block(&mut self, index: u32) -> Result<(), Unreadable> {
        let Some(index) = slot(index) else {
            return Ok(());
        };
        let (Some(block), Some(partial)) =
            (self.blocks.get_mut(index), self.partial_json.get_mut(index))
        else {
            return Ok(());
        };
        if partial.is_empty() {
            return Ok(());
        }
        let Ok(value) = serde_json::from_str::<Value>(partial) else {
            return Err(Unreadable);
        };
        match block {
            ContentBlock::ToolUse { input, .. } | ContentBlock::ServerToolUse { input, .. } => {
                *input = value;
            }
            _ => {}
        }
        partial.clear();
        Ok(())
    }

    /// The final `message_delta`: the stop and the usage of the whole answer.
    fn message_delta(&mut self, delta: MessageDelta, usage: MessageDeltaUsage) {
        let message = &mut self.message;
        if delta.stop_reason.is_some() {
            message.stop_reason = delta.stop_reason;
        }
        if delta.stop_sequence.is_some() {
            message.stop_sequence = delta.stop_sequence;
        }
        if delta.stop_details.is_some() {
            message.stop_details = delta.stop_details;
        }
        if delta.container.is_some() {
            message.container = delta.container;
        }
        message.usage.output_tokens = usage.output_tokens;
        if let Some(input_tokens) = usage.input_tokens {
            message.usage.input_tokens = input_tokens;
        }
        if usage.cache_creation_input_tokens.is_some() {
            message.usage.cache_creation_input_tokens = usage.cache_creation_input_tokens;
        }
        if usage.cache_read_input_tokens.is_some() {
            message.usage.cache_read_input_tokens = usage.cache_read_input_tokens;
        }
        if usage.server_tool_use.is_some() {
            message.usage.server_tool_use = usage.server_tool_use;
        }
        if usage.output_tokens_details.is_some() {
            message.usage.output_tokens_details = usage.output_tokens_details;
        }
    }
}

/// A content block's slot, for an index this gateway follows.
fn slot(index: u32) -> Option<usize> {
    usize::try_from(index)
        .ok()
        .filter(|&index| index < MAX_BLOCKS)
}

/// The end of the first complete event in `buffer` (the index after its
/// blank line, in any of the three framings the relay reads), when there is
/// one. `scanned` is where the last scan of this buffer stopped; the scan
/// resumes a delimiter's overlap before it, since a four-byte delimiter can
/// straddle the frame boundary the last scan ended at. An event that
/// arrives in many frames is read once this way, not once per frame.
fn event_end(buffer: &[u8], scanned: usize) -> Option<usize> {
    let from = scanned.saturating_sub(DELIMITER_OVERLAP);
    let end = find_frame_end(buffer, from);
    #[cfg(test)]
    tests::SCAN_POSITIONS.with(|cell| {
        let visited = end.map_or(buffer.len(), |(at, _)| at + 1);
        cell.set(cell.get() + visited.saturating_sub(from));
    });
    end.map(|(at, len)| at + len)
}

/// Data this gateway cannot read as a stream event. The answer it is part
/// of is not the one the client reads, so that answer is not stored.
struct Unreadable;

/// The event `block`'s data as the stream event it names: `Ok(None)` for a
/// comment or a keep-alive without data, `Err` for data this gateway does
/// not know. The one-line case, every token's, borrows its line.
fn parse_event(block: &[u8]) -> Result<Option<MessageStreamEvent>, Unreadable> {
    let text = std::str::from_utf8(block).map_err(|_| Unreadable)?;
    let mut lines = text
        .split(['\r', '\n'])
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|rest| rest.strip_prefix(' ').unwrap_or(rest));
    let Some(first) = lines.next() else {
        return Ok(None);
    };
    let parsed = match lines.next() {
        None if first.trim().is_empty() => return Ok(None),
        None => serde_json::from_str(first),
        Some(second) => {
            let mut data = String::with_capacity(block.len());
            data.push_str(first);
            for line in std::iter::once(second).chain(lines) {
                data.push('\n');
                data.push_str(line);
            }
            serde_json::from_str(&data)
        }
    };
    parsed.map(Some).map_err(|_| Unreadable)
}

/// The `message_start` event with `message`, as the bytes of one event.
fn start_event(message: &Message) -> Option<Vec<u8>> {
    let data = serde_json::to_string(&MessageStreamEvent::MessageStart {
        message: message.clone(),
    })
    .ok()?;
    Some(format!("event: message_start\ndata: {data}\n\n").into_bytes())
}

#[cfg(test)]
#[path = "../../tests/unit/routers/message_threads/tests.rs"]
mod tests;
