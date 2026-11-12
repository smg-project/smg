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
/// of text, thinking, signature and input retained so far.
struct Assembled {
    message: Message,
    blocks: Vec<ContentBlock>,
    partial_json: Vec<String>,
    retained: usize,
}

/// More content blocks than this in one answer are not followed.
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
                Ok(Some(event)) => self.apply(event),
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

    fn apply(&mut self, event: MessageStreamEvent) {
        let Some(answer) = &mut self.answer else {
            return;
        };
        match event {
            MessageStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => answer.start_block(index, content_block),
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
    fn start_block(&mut self, index: u32, block: ContentBlock) {
        let Some(index) = slot(index) else {
            return;
        };
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
mod tests {
    use std::{cell::Cell, time::Duration};

    use axum::{body::Bytes, http::header, response::IntoResponse, Json};
    use chrono::Utc;
    use futures::{stream, StreamExt};
    use tracing_test::traced_test;

    use super::*;

    fn request(body: Value) -> CreateMessageRequest {
        let mut body = body;
        let object = body.as_object_mut().expect("an object");
        object.entry("model").or_insert(json!("m"));
        object.entry("max_tokens").or_insert(json!(64));
        serde_json::from_value(body).expect("a well-formed request")
    }

    fn answer(id: &str, content: Value) -> Response {
        Json(json!({
            "id": id, "type": "message", "role": "assistant", "model": "m",
            "content": content, "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 2}
        }))
        .into_response()
    }

    async fn body_of(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("a body");
        serde_json::from_slice(&bytes).expect("a JSON body")
    }

    type Resolved = (CreateMessageRequest, Option<ThreadTurn>);

    /// The resolution of a request the thread store accepts.
    fn accepted(outcome: Result<Resolved, Response>) -> Resolved {
        match outcome {
            Ok(resolved) => resolved,
            Err(refused) => panic!("refused with {}", refused.status()),
        }
    }

    /// The refusal of a request the thread store refuses.
    fn refusal(outcome: Result<Resolved, Response>) -> Response {
        match outcome {
            Ok(_) => panic!("accepted"),
            Err(refused) => refused,
        }
    }

    /// A thread of `turns` user turns under `tenant`, each answered by a
    /// text reply under the id `msg_<tag>_<turn>`; answers the last id.
    async fn thread(
        threads: &MessageThreads,
        tenant: Option<&str>,
        turns: usize,
        tag: &str,
    ) -> String {
        let mut previous: Option<String> = None;
        for index in 1..=turns {
            let body = match &previous {
                None => json!({
                    "system": "be brief",
                    "tools": [{"name": "f", "input_schema": {"type": "object"}}],
                    "messages": [{"role": "user", "content": "turn 1"}],
                    "thread": {"type": "create"}
                }),
                Some(id) => json!({
                    "messages": [{"role": "user", "content": format!("turn {index}")}],
                    "thread": {"type": "continue", "previous_message_id": id}
                }),
            };
            let (_, turn) = accepted(
                threads
                    .resolve(request(body), tenant.map(str::to_string))
                    .await,
            );
            let id = format!("msg_{tag}_{index}");
            let reply = answer(
                &id,
                json!([{"type": "text", "text": format!("reply {index}")}]),
            );
            let response = threads.stored(turn.expect("a thread turn"), reply).await;
            assert_eq!(response.status(), StatusCode::OK);
            previous = Some(id);
        }
        previous.expect("at least one turn")
    }

    /// A request without `thread` is routed as it came.
    #[tokio::test]
    async fn a_plain_request_is_no_turn() {
        let threads = MessageThreads::new();
        let (request, turn) = accepted(
            threads
                .resolve(
                    request(json!({"messages": [{"role": "user", "content": "hi"}]})),
                    None,
                )
                .await,
        );
        assert!(turn.is_none());
        assert_eq!(request.messages.len(), 1);
    }

    /// A stored answer that carries an MCP exchange (the `mcp_tool_use` and
    /// `mcp_tool_result` blocks the MCP loop writes beside its text) is read
    /// back as the assistant's turn, blocks and all, so the thread continues.
    #[tokio::test]
    async fn a_stored_mcp_exchange_is_continued() {
        let threads = MessageThreads::new();
        let (_, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "look it up"}],
                        "thread": {"type": "create"}
                    })),
                    None,
                )
                .await,
        );
        let answered = threads
            .stored(
                turn.expect("a turn"),
                answer(
                    "msg_mcp",
                    json!([
                        {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup", "server_name": "s", "input": {"q": "x"}},
                        {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}], "is_error": false},
                        {"type": "text", "text": "it is 42"}
                    ]),
                ),
            )
            .await;
        assert_eq!(answered.status(), StatusCode::OK);
        let (rebuilt, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "thanks"}],
                        "thread": {"type": "continue", "previous_message_id": "msg_mcp"}
                    })),
                    None,
                )
                .await,
        );
        assert!(turn.is_some());
        assert_eq!(rebuilt.messages.len(), 3);
        assert_eq!(rebuilt.messages[1].role, Role::Assistant);
        let InputContent::Blocks(blocks) = &rebuilt.messages[1].content else {
            panic!("blocks");
        };
        assert!(
            matches!(
                &blocks[..],
                [
                    InputContentBlock::McpToolUse(mcp_use),
                    InputContentBlock::McpToolResult(result),
                    InputContentBlock::Text(text)
                ] if mcp_use.id == "mcptoolu_1" && result.tool_use_id == "mcptoolu_1" && text.text == "it is 42"
            ),
            "{blocks:?}"
        );
    }

    /// A continue request becomes the whole conversation: the create turn's
    /// `system` and `tools`, every turn's messages and answer, then the new
    /// messages; the answer the client gets is the router's, byte for byte.
    #[tokio::test]
    async fn a_continue_rebuilds_the_conversation() {
        let threads = MessageThreads::new();
        let last = thread(&threads, None, 3, "c").await;
        let (rebuilt, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "turn 4"}],
                        "thread": {"type": "continue", "previous_message_id": last}
                    })),
                    None,
                )
                .await,
        );
        assert!(turn.is_some());
        assert!(rebuilt.thread.is_none());
        assert!(rebuilt.system.is_some());
        assert_eq!(rebuilt.tools.as_ref().map(Vec::len), Some(1));
        let texts: Vec<String> = rebuilt
            .messages
            .iter()
            .map(|message| match &message.content {
                InputContent::String(text) => text.clone(),
                InputContent::Blocks(blocks) => match &blocks[..] {
                    [InputContentBlock::Text(text)] => text.text.clone(),
                    other => panic!("one text block, not {other:?}"),
                },
            })
            .collect();
        assert_eq!(
            texts,
            ["turn 1", "reply 1", "turn 2", "reply 2", "turn 3", "reply 3", "turn 4"]
        );
        let roles: Vec<&Role> = rebuilt.messages.iter().map(|m| &m.role).collect();
        assert_eq!(roles.len(), 7);
        assert!(roles.iter().step_by(2).all(|role| **role == Role::User));
    }

    /// A continue from an id nobody issued, or whose create turn is gone, is
    /// the public API's 404 with its code in `error.details`.
    #[tokio::test]
    async fn an_unknown_id_is_not_found() {
        let threads = MessageThreads::new();
        let refused = refusal(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "again"}],
                        "thread": {"type": "continue", "previous_message_id": "msg_none"}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            refused
                .headers()
                .get(route_error::HEADER_X_SMG_ERROR_CODE)
                .and_then(|value| value.to_str().ok()),
            Some("thread_not_found")
        );
        let body = body_of(refused).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "not_found_error");
        assert_eq!(body["error"]["details"]["error_code"], "thread_not_found");
    }

    /// A continue that changes a fixed field names it, as a 400.
    #[tokio::test]
    async fn a_changed_fixed_field_is_a_mismatch() {
        let threads = MessageThreads::new();
        let last = thread(&threads, None, 1, "f").await;
        let refused = refusal(
            threads
                .resolve(
                    request(json!({
                        "model": "other", "thinking": {"type": "enabled", "budget_tokens": 1024},
                        "messages": [{"role": "user", "content": "again"}],
                        "thread": {"type": "continue", "previous_message_id": last}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
        let body = body_of(refused).await;
        assert_eq!(
            body["error"]["details"]["error_code"],
            "thread_fingerprint_mismatch"
        );
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(
            message.starts_with("thread: the request changes model, thinking:"),
            "{message}"
        );
    }

    /// A turn stored under a tenant is continued by that tenant alone; one
    /// stored under none by every caller.
    #[tokio::test]
    async fn a_thread_stays_with_its_tenant() {
        let threads = MessageThreads::new();
        let theirs = thread(&threads, Some("auth:a"), 1, "theirs").await;
        let anyones = thread(&threads, None, 1, "anyones").await;
        let cont = |id: &str| {
            request(json!({
                "messages": [{"role": "user", "content": "again"}],
                "thread": {"type": "continue", "previous_message_id": id}
            }))
        };
        for (id, tenant, continued) in [
            (&theirs, Some("auth:a"), true),
            (&theirs, Some("auth:b"), false),
            (&theirs, None, false),
            (&anyones, Some("auth:b"), true),
            (&anyones, None, true),
        ] {
            let outcome = threads.resolve(cont(id), tenant.map(str::to_string)).await;
            assert_eq!(outcome.is_ok(), continued, "{id} as {tenant:?}");
        }
    }

    /// A streamed answer with no events, an error and an answer with no id
    /// leave the turn unstored, and the answer as it was; the next continue
    /// is the 404.
    #[tokio::test]
    async fn an_unstored_turn_cannot_be_continued() {
        let threads = MessageThreads::new();
        let create = |stream: bool| {
            request(json!({
                "stream": stream,
                "messages": [{"role": "user", "content": "hi"}],
                "thread": {"type": "create"}
            }))
        };
        let cases: [(bool, Response); 3] = [
            (true, answer("msg_streamed", json!([]))),
            (false, route_error::bad_request("some_code", "refused")),
            (false, answer("", json!([{"type": "text", "text": "x"}]))),
        ];
        for (stream, response) in cases {
            let status = response.status();
            let (_, turn) = accepted(threads.resolve(create(stream), None).await);
            let answered = threads.stored(turn.expect("a turn"), response).await;
            assert_eq!(answered.status(), status);
        }
        let refused = refusal(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "again"}],
                        "thread": {"type": "continue", "previous_message_id": "msg_streamed"}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
    }

    /// The tool loop: an answer that calls a tool is continued by the
    /// result alone, paired against the stored call.
    #[tokio::test]
    async fn a_tool_result_continues_the_stored_call() {
        let threads = MessageThreads::new();
        let (_, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "tools": [{"name": "f", "input_schema": {"type": "object"}}],
                        "messages": [{"role": "user", "content": "call f"}],
                        "thread": {"type": "create"}
                    })),
                    None,
                )
                .await,
        );
        let call = answer(
            "msg_call",
            json!([{"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}]),
        );
        threads.stored(turn.expect("a turn"), call).await;
        let (rebuilt, _) = accepted(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}
                        ]}],
                        "thread": {"type": "continue", "previous_message_id": "msg_call"}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(rebuilt.messages.len(), 3);
        let unpaired = refusal(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_other", "content": "42"}
                        ]}],
                        "thread": {"type": "continue", "previous_message_id": "msg_call"}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(unpaired.status(), StatusCode::NOT_FOUND);
    }

    /// The gateway mints a thread turn's message id: a client's `rid` would
    /// become the answer's id on the regular gRPC path, so a repeated one
    /// would name a stored turn. The rid stays a trace id, echoed as the
    /// answer's `request-id`, on a stored and on a streamed turn; a turn
    /// without one gets no echo.
    #[tokio::test]
    async fn a_thread_turn_takes_a_minted_message_id() {
        let threads = MessageThreads::new();
        let (request_sent, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "rid": "trace-1",
                        "messages": [{"role": "user", "content": "hi"}],
                        "thread": {"type": "create"}
                    })),
                    None,
                )
                .await,
        );
        let minted = request_sent.rid.clone().expect("a minted id");
        assert!(
            minted.starts_with("msg_") && minted != "trace-1",
            "{minted}"
        );
        let answered = threads
            .stored(
                turn.expect("a turn"),
                answer(&minted, json!([{"type": "text", "text": "x"}])),
            )
            .await;
        assert_eq!(
            answered.headers()["request-id"].to_str().unwrap(),
            "trace-1"
        );
        let (streamed, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "rid": "trace-1",
                        "stream": true,
                        "messages": [{"role": "user", "content": "more"}],
                        "thread": {"type": "continue", "previous_message_id": minted}
                    })),
                    None,
                )
                .await,
        );
        let again = streamed.rid.clone().expect("a minted id");
        assert!(again.starts_with("msg_") && again != minted && again != "trace-1");
        let answered = threads
            .stored(turn.expect("a turn"), answer("msg_streamed", json!([])))
            .await;
        assert_eq!(
            answered.headers()["request-id"].to_str().unwrap(),
            "trace-1"
        );
        let (plain, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "hi"}],
                        "thread": {"type": "create"}
                    })),
                    None,
                )
                .await,
        );
        assert!(plain
            .rid
            .as_deref()
            .is_some_and(|id| id.starts_with("msg_")));
        let answered = threads
            .stored(turn.expect("a turn"), answer("msg_plain", json!([])))
            .await;
        assert!(answered.headers().get("request-id").is_none());
    }

    /// An evicted root's id written again by another create (a backend's
    /// repeated id) is another thread's root: an old child's continue onto
    /// it is refused, never spliced onto the new root's system and tools.
    /// Retention runs from the write, so the root expires while the child,
    /// written later, stays.
    #[tokio::test]
    async fn a_reused_root_id_is_refused_not_spliced() {
        let threads = MessageThreads::kept(MemoryConfig::default().max_responses, 4);
        let create = |content: &str| {
            request(json!({
                "system": "be brief",
                "messages": [{"role": "user", "content": content}],
                "thread": {"type": "create"}
            }))
        };
        let (_, turn) = accepted(threads.resolve(create("first"), None).await);
        threads
            .stored(
                turn.expect("a turn"),
                answer("msg_root", json!([{"type": "text", "text": "a"}])),
            )
            .await;
        tokio::time::sleep(Duration::from_millis(2000)).await;
        let (_, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "second"}],
                        "thread": {"type": "continue", "previous_message_id": "msg_root"}
                    })),
                    None,
                )
                .await,
        );
        threads
            .stored(
                turn.expect("a turn"),
                answer("msg_child", json!([{"type": "text", "text": "b"}])),
            )
            .await;
        tokio::time::sleep(Duration::from_millis(2300)).await;
        let (_, turn) = accepted(threads.resolve(create("another conversation"), None).await);
        let answered = threads
            .stored(
                turn.expect("a turn"),
                answer("msg_root", json!([{"type": "text", "text": "c"}])),
            )
            .await;
        assert_eq!(
            answered.status(),
            StatusCode::OK,
            "the new root under the reused id"
        );
        let refused = refusal(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "third"}],
                        "thread": {"type": "continue", "previous_message_id": "msg_child"}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
        let error = body_of(refused).await;
        assert_eq!(error["error"]["details"]["error_code"], "thread_not_found");
    }

    /// A reissued answer is serialized again under its fresh id, so the
    /// `content-length` of the answer it replaces does not go out with it.
    #[tokio::test]
    async fn a_reissued_answer_drops_the_stale_content_length() {
        let threads = MessageThreads::new();
        let create = || {
            request(json!({
                "messages": [{"role": "user", "content": "hi"}],
                "thread": {"type": "create"}
            }))
        };
        let content = json!([{"type": "text", "text": "x"}]);
        let (_, first) = accepted(threads.resolve(create(), None).await);
        threads
            .stored(first.expect("a turn"), answer("msg_taken", content.clone()))
            .await;
        let (_, second) = accepted(threads.resolve(create(), None).await);
        let body = serde_json::to_vec(&json!({
            "id": "msg_taken", "type": "message", "role": "assistant", "model": "m",
            "content": content, "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 2}
        }))
        .expect("an answer");
        let sized = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_LENGTH, body.len())
            .body(Body::from(body))
            .expect("a response");
        let reissued = threads.stored(second.expect("a turn"), sized).await;
        assert_eq!(reissued.status(), StatusCode::OK);
        assert!(
            reissued.headers().get(header::CONTENT_LENGTH).is_none(),
            "the replaced answer's length is gone"
        );
        let answer = body_of(reissued).await;
        assert_ne!(answer["id"], json!("msg_taken"));
    }

    /// A thread whose create turn the store let go (the count bound reached
    /// under its continues) cannot be continued: the client replays. The
    /// store itself let the turn go, not only the walk's depth.
    #[tokio::test]
    async fn an_evicted_create_turn_ends_the_thread() {
        let threads = MessageThreads::from_config(&MemoryConfig {
            max_responses: 2,
            ..MemoryConfig::default()
        });
        assert_eq!(
            threads.max_turns, 2,
            "the chain is read back to the configured count, not the default"
        );
        let last = thread(&threads, None, 3, "bounded").await;
        assert!(
            threads
                .store
                .get_response(&ResponseId::from("msg_bounded_1"))
                .await
                .expect("a store read")
                .is_none(),
            "the create turn's record is gone from the store"
        );
        let refused = refusal(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "turn 4"}],
                        "thread": {"type": "continue", "previous_message_id": last}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
        let error = body_of(refused).await;
        assert_eq!(error["error"]["details"]["error_code"], "thread_not_found");
    }

    /// A second answer under an id that is already a stored turn (another
    /// caller's guess) leaves the first turn as it was and is reissued under
    /// a fresh id: the owner continues the first, the second caller
    /// continues their own under the fresh id and gets the 404 under the
    /// guessed one.
    #[tokio::test]
    async fn an_answer_under_a_stored_id_leaves_the_first_turn() {
        let threads = MessageThreads::new();
        let create = |system: &str| {
            request(json!({
                "system": system,
                "messages": [{"role": "user", "content": "hi"}],
                "thread": {"type": "create"}
            }))
        };
        let reply = |text: &str| answer("msg_taken", json!([{"type": "text", "text": text}]));
        let mut ids = Vec::new();
        for (system, tenant) in [("first", "auth:a"), ("second", "auth:b")] {
            let (_, turn) = accepted(
                threads
                    .resolve(create(system), Some(tenant.to_string()))
                    .await,
            );
            let answered = threads.stored(turn.expect("a turn"), reply(system)).await;
            assert_eq!(answered.status(), StatusCode::OK);
            ids.push(
                body_of(answered).await["id"]
                    .as_str()
                    .expect("an id")
                    .to_string(),
            );
        }
        assert_eq!(ids[0], "msg_taken");
        assert_ne!(ids[1], "msg_taken", "the second answer is reissued");
        assert!(ids[1].starts_with("msg_"), "{}", ids[1]);
        let cont = |id: &str, tenant: &str| {
            threads.resolve(
                request(json!({
                    "messages": [{"role": "user", "content": "again"}],
                    "thread": {"type": "continue", "previous_message_id": id}
                })),
                Some(tenant.to_string()),
            )
        };
        let (rebuilt, _) = accepted(cont("msg_taken", "auth:a").await);
        assert!(
            matches!(rebuilt.system, Some(SystemContent::String(ref text)) if text == "first"),
            "the first turn's system, not {:?}",
            json_of(&rebuilt.system)
        );
        assert_eq!(rebuilt.messages.len(), 3);
        let (theirs, _) = accepted(cont(&ids[1], "auth:b").await);
        assert!(
            matches!(theirs.system, Some(SystemContent::String(ref text)) if text == "second"),
            "the second caller's own turn under the fresh id"
        );
        assert_eq!(
            refusal(cont("msg_taken", "auth:b").await).status(),
            StatusCode::NOT_FOUND
        );
    }

    /// A client that sends one `rid` on every turn: the second answer comes
    /// back under a fresh id, the turn is stored under it, and the third
    /// turn's conversation is whole. Before the reissue the second turn was
    /// unstored and the third turn silently lost it.
    #[tokio::test]
    async fn a_repeated_rid_keeps_the_thread_whole() {
        let threads = MessageThreads::new();
        let turn = |text: &str, previous: Option<&str>| match previous {
            None => request(json!({
                "messages": [{"role": "user", "content": text}],
                "thread": {"type": "create"}
            })),
            Some(id) => request(json!({
                "messages": [{"role": "user", "content": text}],
                "thread": {"type": "continue", "previous_message_id": id}
            })),
        };
        let (_, created) = accepted(threads.resolve(turn("m1", None), None).await);
        let first = threads
            .stored(
                created.expect("a turn"),
                answer("R", json!([{"type": "text", "text": "a1"}])),
            )
            .await;
        assert_eq!(body_of(first).await["id"], "R");
        let (_, continued) = accepted(threads.resolve(turn("m2", Some("R")), None).await);
        let second = threads
            .stored(
                continued.expect("a turn"),
                answer("R", json!([{"type": "text", "text": "a2"}])),
            )
            .await;
        assert_eq!(second.status(), StatusCode::OK);
        let second = body_of(second).await;
        let reissued = second["id"].as_str().expect("an id");
        assert_ne!(reissued, "R", "the colliding id is reissued");
        assert_eq!(
            second["content"][0]["text"], "a2",
            "the answer itself is unchanged"
        );
        let (rebuilt, _) = accepted(threads.resolve(turn("m3", Some(reissued)), None).await);
        let texts: Vec<String> = rebuilt
            .messages
            .iter()
            .map(|message| match &message.content {
                InputContent::String(text) => text.clone(),
                InputContent::Blocks(blocks) => match &blocks[..] {
                    [InputContentBlock::Text(text)] => text.text.clone(),
                    other => panic!("one text block, not {other:?}"),
                },
            })
            .collect();
        assert_eq!(texts, ["m1", "a1", "m2", "a2", "m3"]);
    }

    /// A streamed answer as the router writes it: `events` as `event:` and
    /// `data:` lines, then cut into frames at the byte offsets `cuts`.
    fn sse(events: &[(&str, Value)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, data) in events {
            out.extend_from_slice(format!("event: {name}\ndata: {data}\n\n").as_bytes());
        }
        out
    }

    fn streamed(bytes: &[u8], cuts: &[usize]) -> Response {
        let mut frames = Vec::new();
        let mut at = 0;
        for &cut in cuts {
            frames.push(Bytes::copy_from_slice(&bytes[at..cut]));
            at = cut;
        }
        frames.push(Bytes::copy_from_slice(&bytes[at..]));
        let body = Body::from_stream(stream::iter(
            frames.into_iter().map(Ok::<_, std::io::Error>),
        ));
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(body)
            .expect("a response")
    }

    fn message_start(id: &str) -> (&'static str, Value) {
        (
            "message_start",
            json!({"type": "message_start", "message": {
                "id": id, "type": "message", "role": "assistant", "model": "m", "content": [],
                "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 7, "output_tokens": 0}
            }}),
        )
    }

    fn text_events(id: &str, text: &str) -> Vec<(&'static str, Value)> {
        vec![
            message_start(id),
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            ),
            (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
            ),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            ),
            (
                "message_delta",
                json!({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 3}}),
            ),
            ("message_stop", json!({"type": "message_stop"})),
        ]
    }

    fn create(text: &str, streamed: bool) -> CreateMessageRequest {
        request(json!({
            "stream": streamed,
            "messages": [{"role": "user", "content": text}],
            "thread": {"type": "create"}
        }))
    }

    fn continue_from(id: &str, text: &str) -> CreateMessageRequest {
        request(json!({
            "messages": [{"role": "user", "content": text}],
            "thread": {"type": "continue", "previous_message_id": id}
        }))
    }

    /// The texts of a rebuilt conversation, one per message.
    fn texts(request: &CreateMessageRequest) -> Vec<String> {
        request
            .messages
            .iter()
            .map(|message| match &message.content {
                InputContent::String(text) => text.clone(),
                InputContent::Blocks(blocks) => match &blocks[..] {
                    [InputContentBlock::Text(text)] => text.text.clone(),
                    other => panic!("one text block, not {other:?}"),
                },
            })
            .collect()
    }

    /// A streamed create is stored from its events, cut across frames as
    /// they come: the client reads the router's bytes unchanged, and the
    /// continue from the `message_start` id carries the streamed text, the
    /// stop reason and the usage of the final `message_delta`.
    #[tokio::test]
    async fn a_streamed_answer_is_stored_from_its_events() {
        let threads = MessageThreads::new();
        let bytes = sse(&text_events("msg_s1", "Hello, world"));
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        let cuts = [17, 230, 231, bytes.len() - 5];
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &cuts))
            .await;
        assert_eq!(answered.status(), StatusCode::OK);
        let read = to_bytes(answered.into_body(), usize::MAX)
            .await
            .expect("the stream");
        assert_eq!(
            read.as_ref(),
            bytes.as_slice(),
            "the client's bytes are the router's"
        );
        let (rebuilt, _) = accepted(
            threads
                .resolve(continue_from("msg_s1", "again"), None)
                .await,
        );
        assert_eq!(texts(&rebuilt), ["hi", "Hello, world", "again"]);
        let (_, turns) = threads
            .turns("msg_s1", None)
            .await
            .ok()
            .expect("the thread");
        let stored = &turns[0].raw_response;
        assert_eq!(stored["stop_reason"], "end_turn");
        assert_eq!(stored["usage"]["input_tokens"], 7);
        assert_eq!(stored["usage"]["output_tokens"], 3);
        assert_eq!(turns[0].input["messages"][0]["content"], "hi");
    }

    /// A streamed tool call's input is the JSON its deltas spell, parsed when
    /// the block closes.
    #[tokio::test]
    async fn a_streamed_tool_call_is_stored_whole() {
        let threads = MessageThreads::new();
        let bytes = sse(&[
            message_start("msg_t1"),
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}}),
            ),
            (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
            ),
            (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": " [1, 2]}"}}),
            ),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            ),
            (
                "message_delta",
                json!({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"output_tokens": 9}}),
            ),
            ("message_stop", json!({"type": "message_stop"})),
        ]);
        let (_, turn) = accepted(threads.resolve(create("call f", true), None).await);
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &[]))
            .await;
        to_bytes(answered.into_body(), usize::MAX)
            .await
            .expect("the stream");
        let (_, turns) = threads
            .turns("msg_t1", None)
            .await
            .ok()
            .expect("the thread");
        let stored = &turns[0].raw_response;
        assert_eq!(stored["content"][0]["type"], "tool_use");
        assert_eq!(stored["content"][0]["input"], json!({"a": [1, 2]}));
        assert_eq!(stored["stop_reason"], "tool_use");
    }

    /// A streamed answer whose `message_start` id is a stored turn's is
    /// reissued in that event: the client continues under the fresh id, the
    /// first turn keeps its own.
    #[tokio::test]
    async fn a_streamed_answer_under_a_stored_id_is_reissued() {
        let threads = MessageThreads::new();
        let (_, first) = accepted(threads.resolve(create("one", false), None).await);
        let kept = threads
            .stored(
                first.expect("a turn"),
                answer("msg_dup", json!([{"type": "text", "text": "first answer"}])),
            )
            .await;
        assert_eq!(body_of(kept).await["id"], "msg_dup");
        let bytes = sse(&text_events("msg_dup", "second answer"));
        let (_, second) = accepted(threads.resolve(create("two", true), None).await);
        let answered = threads
            .stored(second.expect("a turn"), streamed(&bytes, &[40]))
            .await;
        let read = to_bytes(answered.into_body(), usize::MAX)
            .await
            .expect("the stream");
        let text = std::str::from_utf8(&read).expect("utf-8");
        let first_event = text.split("\n\n").next().expect("an event");
        let data = first_event
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("the message_start data");
        let start: Value = serde_json::from_str(data).expect("json");
        let fresh = start["message"]["id"].as_str().expect("an id").to_string();
        assert_ne!(
            fresh, "msg_dup",
            "the colliding id is reissued in message_start"
        );
        assert!(text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
        let (rebuilt, _) = accepted(threads.resolve(continue_from(&fresh, "three"), None).await);
        assert_eq!(texts(&rebuilt), ["two", "second answer", "three"]);
        let (rebuilt, _) = accepted(
            threads
                .resolve(continue_from("msg_dup", "more"), None)
                .await,
        );
        assert_eq!(texts(&rebuilt), ["one", "first answer", "more"]);
    }

    /// A stream that ends before its `message_stop`, or that carries an error
    /// event (here one a `message_stop` still follows), leaves no turn: the
    /// place is given back by the stream's end (at the error event on the
    /// failed path, at the end-of-stream arm on the cut one), while the body
    /// is still held (the drop guard is the other path), and the continue
    /// from its id is the 404.
    #[tokio::test]
    async fn a_cut_or_failed_stream_leaves_no_turn() {
        let threads = MessageThreads::new();
        let mut cut = text_events("msg_cut", "half");
        cut.truncate(3);
        let mut failed = text_events("msg_failed", "half");
        failed.truncate(3);
        failed.push((
            "error",
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "later"}}),
        ));
        failed.extend(text_events("msg_failed", "half").into_iter().skip(4));
        for (id, events) in [("msg_cut", cut), ("msg_failed", failed)] {
            let bytes = sse(&events);
            let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
            let answered = threads
                .stored(turn.expect("a turn"), streamed(&bytes, &[]))
                .await;
            let mut data = answered.into_body().into_data_stream();
            let mut read = Vec::new();
            while let Some(frame) = data.next().await {
                read.extend_from_slice(&frame.expect("the stream"));
            }
            assert_eq!(read.as_slice(), bytes.as_slice());
            let kept = threads
                .store
                .get_response(&ResponseId::from(id))
                .await
                .expect("the store");
            assert!(
                kept.is_none(),
                "{id}: the place is given back by the stream's end, while the body is still held"
            );
            drop(data);
            let refused = refusal(threads.resolve(continue_from(id, "again"), None).await);
            assert_eq!(refused.status(), StatusCode::NOT_FOUND, "{id}");
        }
    }

    /// A stream with data this gateway cannot read (a content block of a
    /// kind it does not know, a tool call whose input deltas are no JSON
    /// document) reaches the client as it came and leaves no turn: the
    /// record is the answer the client read or nothing, so the continue
    /// from its id is the 404.
    #[tokio::test]
    async fn an_unreadable_stream_leaves_no_turn() {
        let threads = MessageThreads::new();
        let mut unknown_block = text_events("msg_unknown", "known");
        unknown_block.splice(
            4..4,
            [
                (
                    "content_block_start",
                    json!({"type": "content_block_start", "index": 1, "content_block": {"type": "bash_code_execution_tool_result", "tool_use_id": "srvtoolu_1", "content": []}}),
                ),
                (
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": 1}),
                ),
                (
                    "content_block_start",
                    json!({"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}}),
                ),
                (
                    "content_block_delta",
                    json!({"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "after"}}),
                ),
                (
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": 2}),
                ),
            ],
        );
        let no_json = vec![
            message_start("msg_nojson"),
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}}),
            ),
            (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
            ),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            ),
            (
                "message_delta",
                json!({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"output_tokens": 9}}),
            ),
            ("message_stop", json!({"type": "message_stop"})),
        ];
        for (id, events) in [("msg_unknown", unknown_block), ("msg_nojson", no_json)] {
            let bytes = sse(&events);
            let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
            let answered = threads
                .stored(turn.expect("a turn"), streamed(&bytes, &[]))
                .await;
            let read = to_bytes(answered.into_body(), usize::MAX)
                .await
                .expect("the stream");
            assert_eq!(read.as_ref(), bytes.as_slice(), "{id}");
            let kept = threads
                .store
                .get_response(&ResponseId::from(id))
                .await
                .expect("the store");
            assert!(kept.is_none(), "{id}: the place is given back");
            let refused = refusal(threads.resolve(continue_from(id, "again"), None).await);
            assert_eq!(refused.status(), StatusCode::NOT_FOUND, "{id}");
        }
    }

    /// A stream framed with CRLF blank lines (one of the relay's three
    /// framings) reaches the client event by event as it comes, and is
    /// stored whole.
    #[tokio::test]
    async fn a_crlf_framed_stream_passes_as_it_comes_and_is_stored() {
        let threads = MessageThreads::new();
        let mut bytes = Vec::new();
        let mut cuts = Vec::new();
        for (name, data) in &text_events("msg_crlf", "carriage") {
            bytes.extend_from_slice(format!("event: {name}\r\ndata: {data}\r\n\r\n").as_bytes());
            cuts.push(bytes.len());
        }
        cuts.pop();
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &cuts))
            .await;
        let mut data = answered.into_body().into_data_stream();
        let first = data
            .next()
            .await
            .expect("a frame")
            .expect("the start event");
        assert_eq!(
            first.as_ref(),
            &bytes[..cuts[0]],
            "the first event is through on its own"
        );
        let mut read = first.to_vec();
        while let Some(frame) = data.next().await {
            read.extend_from_slice(&frame.expect("a frame"));
        }
        assert_eq!(read, bytes, "the client's bytes are the router's");
        let (rebuilt, _) = accepted(
            threads
                .resolve(continue_from("msg_crlf", "again"), None)
                .await,
        );
        assert_eq!(texts(&rebuilt), ["hi", "carriage", "again"]);
    }

    /// A body dropped before its `message_stop` (the client went away) gives
    /// its place back: no record stays under the dropped id and the thread
    /// cannot be continued from it.
    #[tokio::test]
    async fn a_dropped_stream_leaves_no_turn() {
        let threads = MessageThreads::new();
        let events = text_events("msg_dropped", "half");
        let start = sse(&events[..1]).len();
        let bytes = sse(&events);
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &[start]))
            .await;
        let mut data = answered.into_body().into_data_stream();
        let first = data
            .next()
            .await
            .expect("a frame")
            .expect("the start event");
        assert!(first.starts_with(b"event: message_start"), "{first:?}");
        let reserved = threads
            .store
            .get_response(&ResponseId::from("msg_dropped"))
            .await
            .expect("the store");
        assert!(reserved.is_some(), "the place is taken at message_start");
        drop(data);
        let kept = threads
            .store
            .get_response(&ResponseId::from("msg_dropped"))
            .await
            .expect("the store");
        assert!(kept.is_none(), "the dropped place is given back");
        let refused = refusal(
            threads
                .resolve(continue_from("msg_dropped", "again"), None)
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
    }

    /// A streamed turn whose place the store's count bound dropped before
    /// its `message_stop`, and whose id another turn took meanwhile, leaves
    /// that turn alone: the streamed answer is not written over it, and the
    /// continue from the id is the other turn's.
    #[tokio::test]
    async fn an_evicted_place_is_left_to_the_turn_that_took_its_id() {
        let threads = MessageThreads::bounded(1);
        let events = text_events("msg_evicted", "streamed");
        let bytes = sse(&events);
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        let answered = threads
            .stored(
                turn.expect("a turn"),
                streamed(&bytes, &[sse(&events[..1]).len()]),
            )
            .await;
        let mut data = answered.into_body().into_data_stream();
        let first = data
            .next()
            .await
            .expect("a frame")
            .expect("the start event");
        assert!(first.starts_with(b"event: message_start"), "{first:?}");
        let (_, other) = accepted(threads.resolve(create("other", false), None).await);
        threads
            .stored(
                other.expect("a turn"),
                answer("msg_other", json!([{"type": "text", "text": "pushes"}])),
            )
            .await;
        let (_, taker) = accepted(threads.resolve(create("taker", false), None).await);
        threads
            .stored(
                taker.expect("a turn"),
                answer(
                    "msg_evicted",
                    json!([{"type": "text", "text": "took the id"}]),
                ),
            )
            .await;
        let mut read = first.to_vec();
        while let Some(frame) = data.next().await {
            read.extend_from_slice(&frame.expect("a frame"));
        }
        assert_eq!(read, bytes, "the client's bytes are the router's");
        let (rebuilt, _) = accepted(
            threads
                .resolve(continue_from("msg_evicted", "again"), None)
                .await,
        );
        assert_eq!(texts(&rebuilt), ["taker", "took the id", "again"]);
    }

    /// A stream cut after the store's count bound dropped its place, and
    /// after another turn took the id, removes nothing: the other turn stays
    /// and the continue from the id is its own.
    #[tokio::test]
    async fn a_cut_stream_leaves_an_evicted_place_to_the_turn_that_took_its_id() {
        let threads = MessageThreads::bounded(1);
        let events = text_events("msg_cut_evicted", "streamed");
        let bytes = sse(&events);
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        let answered = threads
            .stored(
                turn.expect("a turn"),
                streamed(&bytes, &[sse(&events[..1]).len()]),
            )
            .await;
        let mut data = answered.into_body().into_data_stream();
        let first = data
            .next()
            .await
            .expect("a frame")
            .expect("the start event");
        assert!(first.starts_with(b"event: message_start"), "{first:?}");
        let (_, other) = accepted(threads.resolve(create("other", false), None).await);
        threads
            .stored(
                other.expect("a turn"),
                answer("msg_other", json!([{"type": "text", "text": "pushes"}])),
            )
            .await;
        let (_, taker) = accepted(threads.resolve(create("taker", false), None).await);
        threads
            .stored(
                taker.expect("a turn"),
                answer(
                    "msg_cut_evicted",
                    json!([{"type": "text", "text": "took the id"}]),
                ),
            )
            .await;
        drop(data);
        let (rebuilt, _) = accepted(
            threads
                .resolve(continue_from("msg_cut_evicted", "again"), None)
                .await,
        );
        assert_eq!(texts(&rebuilt), ["taker", "took the id", "again"]);
    }

    /// An answer with more content than the limit is not stored: the place
    /// is given back as the deltas pass the bound, the client reads the
    /// stream as it came, and the continue from its id is the 404.
    #[tokio::test]
    async fn an_answer_over_the_limit_leaves_no_turn() {
        let threads = MessageThreads::new().limited(16);
        let bytes = sse(&text_events("msg_big", "more than sixteen bytes of text"));
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &[]))
            .await;
        let read = to_bytes(answered.into_body(), usize::MAX)
            .await
            .expect("the stream");
        assert_eq!(read.as_ref(), bytes.as_slice());
        let refused = refusal(
            threads
                .resolve(continue_from("msg_big", "again"), None)
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
    }

    thread_local! {
        /// The scan positions `event_end` visited on this thread, read by the
        /// test that feeds one event in many frames.
        pub(super) static SCAN_POSITIONS: Cell<usize> = const { Cell::new(0) };
    }

    /// An event that arrives in many frames is scanned once: the scan of
    /// each buffer resumes where it stopped, three bytes back, instead of
    /// starting from the first byte at every frame.
    #[tokio::test]
    async fn an_event_in_many_frames_is_scanned_once() {
        let threads = MessageThreads::new();
        let text = "x".repeat(64 * 1024);
        let bytes = sse(&text_events("msg_frames", &text));
        let cuts: Vec<usize> = (1..bytes.len() / 1024).map(|k| k * 1024).collect();
        let frames = cuts.len() + 1;
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        SCAN_POSITIONS.with(|cell| cell.set(0));
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &cuts))
            .await;
        let read = to_bytes(answered.into_body(), usize::MAX)
            .await
            .expect("the stream");
        let positions = SCAN_POSITIONS.with(Cell::get);
        assert_eq!(read.as_ref(), bytes.as_slice());
        assert!(
            positions <= bytes.len() + 4 * frames,
            "{positions} scan positions over {} bytes in {frames} frames",
            bytes.len()
        );
        let stored = threads
            .store
            .get_response(&ResponseId::from("msg_frames"))
            .await
            .expect("the store");
        assert!(stored.is_some(), "the turn is stored");
    }

    /// A frame boundary inside an event's delimiter still ends the event, at
    /// every split of the two-byte and of the four-byte delimiter: the
    /// resumed scan starts a delimiter's overlap back, before the first
    /// `message_start` and within the answer alike.
    #[tokio::test]
    async fn a_delimiter_split_by_a_frame_boundary_still_ends_the_event() {
        let events = text_events("msg_split", "hello");
        let crlf: Vec<u8> = events
            .iter()
            .flat_map(|(name, data)| format!("event: {name}\r\ndata: {data}\r\n\r\n").into_bytes())
            .collect();
        for (bytes, delimiter) in [(sse(&events), 2), (crlf, 4)] {
            let first_end = event_end(&bytes, 0).expect("the first event");
            let second_end =
                event_end(&bytes[first_end..], 0).expect("the second event") + first_end;
            for cut in (1..delimiter).flat_map(|back| [first_end - back, second_end - back]) {
                let threads = MessageThreads::new();
                let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
                let answered = threads
                    .stored(turn.expect("a turn"), streamed(&bytes, &[cut]))
                    .await;
                let read = to_bytes(answered.into_body(), usize::MAX)
                    .await
                    .expect("the stream");
                assert_eq!(
                    read.as_ref(),
                    bytes.as_slice(),
                    "cut at {cut} of a {delimiter}-byte delimiter"
                );
                let stored = threads
                    .store
                    .get_response(&ResponseId::from("msg_split"))
                    .await
                    .expect("the store");
                assert!(
                    stored.is_some(),
                    "the turn is stored with the cut at {cut} of a {delimiter}-byte delimiter"
                );
            }
        }
    }

    /// A body with no event end within the limit is not a stream this
    /// gateway reads. Before `message_start`, the bytes pass to the client
    /// once they are over it, not at the body's end. After it, the place is
    /// given back and the bytes keep passing; the continue is the 404.
    #[tokio::test]
    async fn a_body_without_an_event_end_passes_once_over_the_limit() {
        let threads = MessageThreads::new();
        let big = vec![b'x'; EVENT_LIMIT + 1];
        let mut bytes = big.clone();
        bytes.extend_from_slice(b"tail");
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &[big.len()]))
            .await;
        let mut data = answered.into_body().into_data_stream();
        let first = data.next().await.expect("a frame").expect("the held bytes");
        assert_eq!(
            first.len(),
            big.len(),
            "the held bytes pass before the body ends"
        );
        let mut read = first.to_vec();
        while let Some(frame) = data.next().await {
            read.extend_from_slice(&frame.expect("a frame"));
        }
        assert_eq!(read, bytes);

        let start = sse(&text_events("msg_noend", "")[..1]);
        let mut bytes = start.clone();
        bytes.extend_from_slice(b"data: ");
        bytes.extend_from_slice(&vec![b'y'; EVENT_LIMIT + 1]);
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &[start.len()]))
            .await;
        let mut data = answered.into_body().into_data_stream();
        let first = data
            .next()
            .await
            .expect("a frame")
            .expect("the start event");
        assert_eq!(first.as_ref(), start.as_slice());
        let reserved = threads
            .store
            .get_response(&ResponseId::from("msg_noend"))
            .await
            .expect("the store");
        assert!(reserved.is_some(), "the place is taken at message_start");
        let second = data
            .next()
            .await
            .expect("a frame")
            .expect("the unending event");
        assert_eq!(second.len(), bytes.len() - start.len());
        let kept = threads
            .store
            .get_response(&ResponseId::from("msg_noend"))
            .await
            .expect("the store");
        assert!(kept.is_none(), "the place is given back over the limit");
        assert!(data.next().await.is_none());
        let refused = refusal(
            threads
                .resolve(continue_from("msg_noend", "again"), None)
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
    }

    /// A turn older than the operator's retention is gone when the store
    /// next evicts: the thread ends with the public 404 and the client
    /// replays.
    #[tokio::test]
    async fn a_turn_past_the_retention_ends_the_thread() {
        let threads = MessageThreads::from_config(&MemoryConfig {
            retention_secs: Some(60),
            ..MemoryConfig::default()
        });
        let last = thread(&threads, None, 2, "aged").await;
        let cont = || {
            request(json!({
                "messages": [{"role": "user", "content": "turn 3"}],
                "thread": {"type": "continue", "previous_message_id": last}
            }))
        };
        accepted(threads.resolve(cont(), None).await);
        threads
            .store
            .evict_at(Utc::now() + chrono::Duration::seconds(61));
        let refused = refusal(threads.resolve(cont(), None).await);
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
    }

    /// A thread longer than the history store's default chain depth of 100
    /// is read back whole: the walk goes as deep as the store keeps turns.
    #[tokio::test]
    async fn a_long_thread_is_read_back_whole() {
        let threads = MessageThreads::new();
        let last = thread(&threads, None, 120, "long").await;
        let (rebuilt, _) = accepted(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "turn 121"}],
                        "thread": {"type": "continue", "previous_message_id": last}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(rebuilt.messages.len(), 241);
    }

    /// The echoed `request-id` keeps the request-id layer's rules: a `rid`
    /// over 128 characters or with a space in it is not echoed (a client
    /// cannot size the header), and a `request-id` the route set stays (a
    /// relayed upstream id is the one its support asks for).
    #[tokio::test]
    async fn the_echoed_request_id_keeps_the_layers_rules() {
        let threads = MessageThreads::new();
        for (rid, id) in [
            ("x".repeat(129), "msg_long"),
            ("trace 1".to_string(), "msg_space"),
        ] {
            let (_, turn) = accepted(
                threads
                    .resolve(
                        request(json!({
                            "rid": rid,
                            "messages": [{"role": "user", "content": "hi"}],
                            "thread": {"type": "create"}
                        })),
                        None,
                    )
                    .await,
            );
            let answered = threads
                .stored(turn.expect("a turn"), answer(id, json!([])))
                .await;
            assert_eq!(answered.status(), StatusCode::OK);
            assert!(answered.headers().get("request-id").is_none(), "{rid:?}");
        }
        let (_, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "rid": "trace-1",
                        "messages": [{"role": "user", "content": "hi"}],
                        "thread": {"type": "create"}
                    })),
                    None,
                )
                .await,
        );
        let mut relayed = answer("msg_relayed", json!([]));
        relayed
            .headers_mut()
            .insert(&REQUEST_ID, HeaderValue::from_static("upstream-1"));
        let answered = threads.stored(turn.expect("a turn"), relayed).await;
        assert_eq!(
            answered.headers()["request-id"].to_str().unwrap(),
            "upstream-1"
        );
    }

    fn create_turn() -> CreateMessageRequest {
        request(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "thread": {"type": "create"}
        }))
    }

    /// A 200 answer that is not JSON goes to the client as it came; the turn
    /// is left unstored and the log says why.
    #[tokio::test]
    #[traced_test]
    async fn a_non_json_answer_is_left_unstored_with_a_warning() {
        let threads = MessageThreads::new();
        let (_, turn) = accepted(threads.resolve(create_turn(), None).await);
        let plain = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/plain")
            .body(Body::from("not json"))
            .expect("a response");
        let answered = threads.stored(turn.expect("a turn"), plain).await;
        assert_eq!(answered.status(), StatusCode::OK);
        let bytes = to_bytes(answered.into_body(), usize::MAX)
            .await
            .expect("a body");
        assert_eq!(bytes.as_ref(), b"not json");
        assert!(logs_contain("the answer is not JSON"));
    }

    /// A JSON answer without an id goes to the client as it came; the turn
    /// is left unstored and the log says why.
    #[tokio::test]
    #[traced_test]
    async fn an_answer_without_an_id_is_left_unstored_with_a_warning() {
        let threads = MessageThreads::new();
        let (_, turn) = accepted(threads.resolve(create_turn(), None).await);
        let answered = threads
            .stored(
                turn.expect("a turn"),
                answer("", json!([{"type": "text", "text": "x"}])),
            )
            .await;
        assert_eq!(answered.status(), StatusCode::OK);
        assert_eq!(body_of(answered).await["id"], "");
        assert!(logs_contain("the answer has no id"));
    }
}
