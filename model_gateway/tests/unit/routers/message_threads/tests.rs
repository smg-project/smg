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
async fn thread(threads: &MessageThreads, tenant: Option<&str>, turns: usize, tag: &str) -> String {
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

/// A streamed answer with a block of a kind this gateway does not read
/// reaches the client as it came and is stored with the block as the
/// upstream sent it, beside the text it reads. A continue from it reads
/// the stored blocks as the request's input, which has no form for a kind
/// no API has, so it is the 404.
#[tokio::test]
async fn a_streamed_block_this_gateway_does_not_read_is_stored_as_sent() {
    let threads = MessageThreads::new();
    let block =
        json!({"type": "a_block_kind_of_no_api", "tool_use_id": "srvtoolu_1", "detail": "x"});
    let mut events = text_events("msg_unknown", "known");
    events.splice(
        4..4,
        [
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": 1, "content_block": block.clone()}),
            ),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 1}),
            ),
        ],
    );
    let bytes = sse(&events);
    let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
    let answered = threads
        .stored(turn.expect("a turn"), streamed(&bytes, &[]))
        .await;
    let read = to_bytes(answered.into_body(), usize::MAX)
        .await
        .expect("the stream");
    assert_eq!(read.as_ref(), bytes.as_slice());
    let (_, turns) = threads
        .turns("msg_unknown", None)
        .await
        .ok()
        .expect("the thread");
    let stored = &turns[0].raw_response;
    assert_eq!(stored["content"][0]["text"], "known");
    assert_eq!(stored["content"][1], block);
    let refused = refusal(
        threads
            .resolve(continue_from("msg_unknown", "again"), None)
            .await,
    );
    assert_eq!(
        refused.status(),
        StatusCode::NOT_FOUND,
        "the request does not read the block"
    );
}

/// A block that arrives whole in its start event counts toward the limit
/// as deltas do: an answer whose blocks pass the bound with no delta at
/// all is not stored, and the same answer under a wider bound is.
#[tokio::test]
async fn a_block_started_whole_counts_toward_the_limit() {
    let block = json!({"type": "a_block_kind_of_no_api", "detail": "x".repeat(2048)});
    let mut events = text_events("msg_whole", "known");
    events.splice(
        4..4,
        [
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": 1, "content_block": block}),
            ),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 1}),
            ),
        ],
    );
    let bytes = sse(&events);
    for (limit, kept) in [(1 << 20, true), (1024, false)] {
        let threads = MessageThreads::new().limited(limit);
        let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
        let answered = threads
            .stored(turn.expect("a turn"), streamed(&bytes, &[]))
            .await;
        let read = to_bytes(answered.into_body(), usize::MAX)
            .await
            .expect("the stream");
        assert_eq!(read.as_ref(), bytes.as_slice());
        let stored = threads.turns("msg_whole", None).await.is_ok();
        assert_eq!(stored, kept, "a limit of {limit} bytes");
    }
}

/// An answer with a block past the bound on blocks is not stored, as
/// the record would lack that block: the client reads the stream as it
/// came, and the continue from its id is the 404.
#[tokio::test]
async fn a_block_past_the_block_bound_leaves_no_turn() {
    let threads = MessageThreads::new();
    let mut events = text_events("msg_many", "known");
    events.splice(
        4..4,
        [
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": MAX_BLOCKS, "content_block": {"type": "text", "text": "past the bound"}}),
            ),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": MAX_BLOCKS}),
            ),
        ],
    );
    let bytes = sse(&events);
    let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
    let answered = threads
        .stored(turn.expect("a turn"), streamed(&bytes, &[]))
        .await;
    let read = to_bytes(answered.into_body(), usize::MAX)
        .await
        .expect("the stream");
    assert_eq!(read.as_ref(), bytes.as_slice());
    let stored = threads.turns("msg_many", None).await.is_ok();
    assert!(!stored, "a turn that lacks a block the client read");
}

/// A stream with data this gateway cannot read (a tool call whose input
/// deltas are no JSON document) reaches the client as it came and leaves
/// no turn: the record is the answer the client read or nothing, so the
/// continue from its id is the 404.
#[tokio::test]
async fn an_unreadable_stream_leaves_no_turn() {
    let threads = MessageThreads::new();
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
    let bytes = sse(&no_json);
    let (_, turn) = accepted(threads.resolve(create("hi", true), None).await);
    let answered = threads
        .stored(turn.expect("a turn"), streamed(&bytes, &[]))
        .await;
    let read = to_bytes(answered.into_body(), usize::MAX)
        .await
        .expect("the stream");
    assert_eq!(read.as_ref(), bytes.as_slice());
    let kept = threads
        .store
        .get_response(&ResponseId::from("msg_nojson"))
        .await
        .expect("the store");
    assert!(kept.is_none(), "the place is given back");
    let refused = refusal(
        threads
            .resolve(continue_from("msg_nojson", "again"), None)
            .await,
    );
    assert_eq!(refused.status(), StatusCode::NOT_FOUND);
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
        let second_end = event_end(&bytes[first_end..], 0).expect("the second event") + first_end;
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
