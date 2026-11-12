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
//! once, not once per turn. The store keeps the newest [`MAX_TURNS`] turns
//! for [`TURN_RETENTION_SECS`] after they were written, the oldest evicted
//! first, on every write; a thread is read back whole up to that count. A
//! turn is written under its answer's id. An id that is already a stored
//! turn's (a repeated `rid`, a backend's repeated id, a guess) would hand
//! the client an id that continues another conversation, so the answer is
//! reissued under a fresh id, in the answer the client reads and in the
//! store.
//!
//! A thread turn's message id is the gateway's: the request's `rid` is
//! replaced by a fresh `msg_` id before the route (on the regular gRPC path
//! the request id is the message id, so a client's repeated `rid` would name
//! a stored turn, and a streamed answer is never reissued), and the client's
//! `rid` is echoed as the answer's `request-id` header, the trace id it
//! stays, under the request-id layer's inbound rule and where the route set
//! none; a provider never receives `rid`, the adapter that mounts a provider
//! router clears it. Every turn of a thread carries the token minted at its
//! create turn (the record's `conversation_id`); a root under a reused id,
//! written after the thread's own root was evicted, carries another, so an
//! old child's continue onto it is the 404, never a splice of two
//! conversations.
//!
//! Best effort, as the public API's threads are: a turn that was not stored
//! (a streamed answer, an error, an answer the gateway could not read, a
//! write the store refused, an evicted record) makes the next continue a
//! 404 `thread_not_found`, and the
//! client replays the conversation in a new create request, as the public
//! SDKs' thread helper does on its own. An id
//! may be continued more than once (a fork); the public API's
//! `thread_already_continued` is not answered. A continue whose rebuilt
//! conversation breaks a request rule is the same 404: the public API's
//! codes name no such case, and the client's replay (the whole conversation
//! as a create request) meets the 400 that names the rule.

use std::sync::Arc;

use axum::{
    body::{to_bytes, Body, Bytes},
    http::{header, HeaderValue, StatusCode},
    response::Response,
};
use openai_protocol::messages::{
    CreateMessageRequest, ErrorDetails, ErrorEnvelope, ErrorResponse, InputContent,
    InputContentBlock, InputMessage, MessageThread, Role, SystemContent, Tool,
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
    routers::error as route_error,
};

/// The turns the store keeps: the newest 10,000, each for an hour after it
/// was written (the public API deletes a thread no more than an hour after
/// its last use). Over the count the oldest go first; both bounds are
/// applied on every write. The count is also the depth a thread is read
/// back to, so no kept thread is cut short of its create turn.
const MAX_TURNS: usize = 10_000;
const TURN_RETENTION_SECS: u64 = 60 * 60;

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
/// (echoed as the answer's `request-id`, the trace id it stays) and, when
/// its answer is read back, its record. A streamed turn carries no record:
/// its answer is not read back, so the request is not copied for a turn
/// that is never stored.
pub(crate) struct ThreadTurn {
    op: &'static str,
    trace: Option<String>,
    record: Option<TurnRecord>,
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

/// The threads of this gateway, in the store of their turns.
pub(crate) struct MessageThreads {
    store: MemoryResponseStorage,
    /// The store's turn count, the depth a thread is read back to.
    max_turns: usize,
}

impl MessageThreads {
    pub(crate) fn new() -> Self {
        Self::bounded(MAX_TURNS)
    }

    /// The threads in a store of `max_turns` turns, each kept for the
    /// retention period.
    fn bounded(max_turns: usize) -> Self {
        Self::kept(max_turns, TURN_RETENTION_SECS)
    }

    /// The threads in a store of `max_turns` turns, each kept for
    /// `retention_secs` after it was written.
    fn kept(max_turns: usize, retention_secs: u64) -> Self {
        Self {
            store: MemoryResponseStorage::with_config(&MemoryConfig {
                max_responses: max_turns,
                retention_secs: Some(retention_secs),
                ..MemoryConfig::default()
            }),
            max_turns,
        }
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
        // name a stored turn (a streamed answer is never reissued). The rid
        // stays a trace id, echoed as the answer's request-id header.
        let trace = request.rid.replace(fresh_message_id());
        match thread {
            MessageThread::Create => {
                let record = (!streamed).then(|| TurnRecord {
                    previous: None,
                    input: TurnInput::record(&request, true),
                    model,
                    tenant,
                    thread: fresh_thread_token(),
                });
                Ok((
                    request,
                    Some(ThreadTurn {
                        op: "create",
                        trace,
                        record,
                    }),
                ))
            }
            MessageThread::Continue {
                previous_message_id,
            } => {
                let input = (!streamed).then(|| TurnInput::record(&request, false));
                match self
                    .continue_from(&previous_message_id, tenant.as_deref(), &mut request)
                    .await
                {
                    Ok(thread) => {
                        let record = input.map(|input| TurnRecord {
                            previous: Some(ResponseId::from(previous_message_id)),
                            input,
                            model,
                            tenant,
                            thread,
                        });
                        Ok((
                            request,
                            Some(ThreadTurn {
                                op: "continue",
                                trace,
                                record,
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
    /// complete 200 message; an id that is already a stored turn's is
    /// reissued, in the answer and in the store. A streamed answer, an
    /// error, an answer the gateway cannot read or a write the store refused
    /// leaves the turn unstored and the answer as it is: the next continue
    /// is a 404 and the client replays.
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
        let Some(turn) = turn.record.filter(|_| response.status() == StatusCode::OK) else {
            Metrics::record_message_thread_turn(op, "unstored");
            return response;
        };
        let (mut parts, body) = response.into_parts();
        let bytes = match to_bytes(body, ANSWER_LIMIT).await {
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
        let mut record = StoredResponse::new(turn.previous);
        record.id = id;
        record.input = turn.input;
        record.model = Some(turn.model);
        record.raw_response = answer;
        record.tenant = turn.tenant;
        record.conversation_id = Some(turn.thread);
        let mut record = match self.store.store_response_if_absent(record) {
            Ok(()) => return Written::Kept,
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
            Ok(()) => Written::Reissued(Bytes::from(bytes)),
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::{body::Bytes, http::header, response::IntoResponse, Json};
    use futures::{stream, StreamExt};
    use tokio::time::timeout;
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

    /// A streamed answer, an error and an answer with no id leave the turn
    /// unstored, and the answer as it was; the next continue is the 404.
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
        let threads = MessageThreads::kept(MAX_TURNS, 4);
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

    /// A streamed turn's answer is not read back, so `resolve` takes no
    /// record of it: the request is not copied for a turn that is never
    /// stored. An unstreamed turn's record is taken.
    #[tokio::test]
    async fn a_streamed_turn_takes_no_record() {
        let threads = MessageThreads::new();
        let create = |stream: bool| {
            request(json!({
                "stream": stream,
                "messages": [{"role": "user", "content": "hi"}],
                "thread": {"type": "create"}
            }))
        };
        let (_, turn) = accepted(threads.resolve(create(true), None).await);
        assert!(turn.expect("a turn").record.is_none(), "a streamed create");
        let (_, turn) = accepted(threads.resolve(create(false), None).await);
        let turn = turn.expect("a turn");
        assert!(turn.record.is_some(), "a create");
        threads
            .stored(
                turn,
                answer("msg_root", json!([{"type": "text", "text": "x"}])),
            )
            .await;
        let cont = |stream: bool| {
            request(json!({
                "stream": stream,
                "messages": [{"role": "user", "content": "more"}],
                "thread": {"type": "continue", "previous_message_id": "msg_root"}
            }))
        };
        let (_, turn) = accepted(threads.resolve(cont(true), None).await);
        assert!(
            turn.expect("a turn").record.is_none(),
            "a streamed continue"
        );
        let (_, turn) = accepted(threads.resolve(cont(false), None).await);
        assert!(turn.expect("a turn").record.is_some(), "a continue");
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

    /// A streamed turn goes to the client as it comes: the store reads no
    /// stream back, and the turn stays unstored.
    #[tokio::test]
    async fn a_streamed_turn_is_not_read_back() {
        let threads = MessageThreads::new();
        let (_, turn) = accepted(
            threads
                .resolve(
                    request(json!({
                        "stream": true,
                        "messages": [{"role": "user", "content": "hi"}],
                        "thread": {"type": "create"}
                    })),
                    None,
                )
                .await,
        );
        let first = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_sse\"}}\n\n";
        let never_ending =
            stream::iter([Ok::<_, std::io::Error>(Bytes::from(first))]).chain(stream::pending());
        let sse = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(never_ending))
            .expect("a response");
        let answered = timeout(
            Duration::from_secs(2),
            threads.stored(turn.expect("a turn"), sse),
        )
        .await
        .expect("a streamed turn is answered before its stream ends");
        assert_eq!(answered.status(), StatusCode::OK);
        let mut events = answered.into_body().into_data_stream();
        let event = events
            .next()
            .await
            .expect("the first event")
            .expect("a readable event");
        assert_eq!(event, first.as_bytes());
        let refused = refusal(
            threads
                .resolve(
                    request(json!({
                        "messages": [{"role": "user", "content": "again"}],
                        "thread": {"type": "continue", "previous_message_id": "msg_sse"}
                    })),
                    None,
                )
                .await,
        );
        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
    }

    /// A thread whose create turn the store let go (the count bound reached
    /// under its continues) cannot be continued: the client replays. The
    /// store itself let the turn go, not only the walk's depth.
    #[tokio::test]
    async fn an_evicted_create_turn_ends_the_thread() {
        let threads = MessageThreads::bounded(2);
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
