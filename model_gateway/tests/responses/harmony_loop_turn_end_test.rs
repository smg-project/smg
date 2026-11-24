//! The ways a Harmony Responses turn ends on a call to a server tool in the
//! plain and streamed forms. The outcomes are alike except when the token
//! limit cuts the first call inside its arguments: only the streamed form
//! recovers that call.
//! A turn that ended on its call runs it: the loop asks the model again and
//! answers. A turn the token limit cut inside a tool's name read no call:
//! nothing runs, and the response keeps the truncation.

use serde_json::{json, Value};

use super::harmony_loop_harness::{serve, serve_with_input, token_ids, End, Served, Turn, ANSWER};

/// A call to the mock tool server's search tool up to its arguments, the
/// arguments whole and cut, and the call's closing token.
const CALL: &str =
    "<|channel|>commentary to=functions.brave_web_search <|constrain|>json<|message|>";
/// A call to the mock tool server's other search tool, up to its arguments.
const OTHER_CALL: &str =
    "<|channel|>commentary to=functions.brave_local_search <|constrain|>json<|message|>";
const WHOLE: &str = r#"{"query": "usage"}"#;
const CUT: &str = r#"{"query": "usa"#;
const CLOSE: &str = "<|call|>";
/// A call the token limit cut inside the tool's name.
const CUT_NAME: &str = "<|channel|>commentary to=functions.brave_web";
/// The start of the turn's next message.
const NEXT: &str = "<|start|>assistant";

/// One way a turn ends, and what the client is due.
struct Exit {
    name: &'static str,
    turn: String,
    end: End,
    /// Whether the call runs; the loop then asks the model again and answers.
    runs: bool,
    /// Whether only the streamed form is checked because it recovers a cut call.
    stream_only: bool,
    status: &'static str,
    reason: Option<&'static str>,
}

impl Exit {
    fn new(
        name: &'static str,
        turn: &[&str],
        end: End,
        runs: bool,
        status: &'static str,
        reason: Option<&'static str>,
    ) -> Self {
        Self {
            name,
            turn: turn.concat(),
            end,
            runs,
            stream_only: false,
            status,
            reason,
        }
    }

    fn stream_only(mut self) -> Self {
        self.stream_only = true;
        self
    }
}

fn exits() -> Vec<Exit> {
    vec![
        // The turn ended on the call: it runs, and the loop answers on its
        // result.
        Exit::new(
            "a stop behind the closed call",
            &[CALL, WHOLE, CLOSE],
            End::Finish("stop"),
            true,
            "completed",
            None,
        ),
        Exit::new(
            "the stream's end behind whole arguments",
            &[CALL, WHOLE],
            End::Eof,
            true,
            "completed",
            None,
        ),
        Exit::new(
            "the token limit behind the closed call",
            &[CALL, WHOLE, CLOSE],
            End::Finish("length"),
            true,
            "completed",
            None,
        ),
        // The streamed form recovers this cut call and runs it: its cut
        // arguments do not parse, so the tool server gets `{}`. The plain form
        // reads only completed messages and reports `incomplete` without running it.
        Exit::new(
            "the token limit inside the first call",
            &[CALL, CUT],
            End::Finish("length"),
            true,
            "completed",
            None,
        )
        .stream_only(),
        // The token limit cut the first call right behind its header: a
        // server tool's call with no argument bytes is not read, nothing runs.
        Exit::new(
            "the token limit behind the first call's header",
            &[CALL],
            End::Finish("length"),
            false,
            "incomplete",
            Some("max_output_tokens"),
        ),
        // The call closed and the token limit cut a second one: the closed
        // call runs and the cut one is not read.
        Exit::new(
            "the token limit inside a second call",
            &[CALL, WHOLE, CLOSE, NEXT, OTHER_CALL, CUT],
            End::Finish("length"),
            true,
            "completed",
            None,
        ),
        Exit::new(
            "the stream's end inside a second call",
            &[CALL, WHOLE, CLOSE, NEXT, OTHER_CALL, CUT],
            End::Eof,
            true,
            "completed",
            None,
        ),
        Exit::new(
            "the token limit before a second call's arguments",
            &[CALL, WHOLE, CLOSE, NEXT, OTHER_CALL],
            End::Finish("length"),
            true,
            "completed",
            None,
        ),
        // The token limit cut the turn before it named a tool: no call was
        // read, nothing runs, the response is cut.
        Exit::new(
            "the token limit inside the tool's name",
            &[CUT_NAME],
            End::Finish("length"),
            false,
            "incomplete",
            Some("max_output_tokens"),
        ),
    ]
}

/// The status of each call item of an output.
fn call_statuses(output: &Value) -> Vec<&str> {
    output
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| {
            item["type"]
                .as_str()
                .is_some_and(|kind| kind.ends_with("_call"))
        })
        .filter_map(|item| item["status"].as_str())
        .collect()
}

/// The lines in which `served` differs from what `exit` is due.
fn misreads(exit: &Exit, stream: bool, served: &Served) -> Vec<String> {
    let mut wrong = Vec::new();
    let mut due = |what: String, got: Value, want: Value| {
        if got != want {
            wrong.push(format!("{what}: {got}, due {want}"));
        }
    };
    let ran = usize::from(exit.runs);
    due(
        "generations".to_string(),
        json!(served.generations),
        json!(1 + ran),
    );
    due(
        "tool calls the server received".to_string(),
        json!(served.tool_calls),
        json!(ran),
    );
    // A call that ran is an item of the response, completed; a turn that
    // read no call leaves none.
    let calls = vec!["completed"; ran];
    for (name, object) in [
        ("the response", &served.response),
        ("the stored object", &served.stored),
    ] {
        due(
            format!("{name}'s status"),
            object["status"].clone(),
            json!(exit.status),
        );
        due(
            format!("{name}'s incomplete reason"),
            object["incomplete_details"]["reason"].clone(),
            json!(exit.reason),
        );
        due(
            format!("{name}'s call items"),
            json!(call_statuses(&object["output"])),
            json!(calls),
        );
    }
    if stream {
        let events_of = |kind: &str| {
            served
                .events
                .iter()
                .filter(|event| event["type"].as_str().is_some_and(|name| name == kind))
                .count()
        };
        let terminals: Vec<&str> = served
            .events
            .iter()
            .filter_map(|event| event["type"].as_str())
            .filter(|kind| {
                matches!(
                    *kind,
                    "response.completed" | "response.incomplete" | "response.failed"
                )
            })
            .collect();
        due(
            "terminal events".to_string(),
            json!(terminals),
            json!([format!("response.{}", exit.status)]),
        );
        due(
            "mcp_call.completed events".to_string(),
            json!(events_of("response.mcp_call.completed")),
            json!(ran),
        );
        let closed: Vec<&Value> = served
            .events
            .iter()
            .filter(|event| {
                event["type"] == "response.output_item.done"
                    && event["item"]["type"] == "mcp_call"
                    && event["item"]["status"] == "completed"
            })
            .map(|event| &event["item"]["status"])
            .collect();
        due(
            "the closed mcp_call items".to_string(),
            json!(closed),
            json!(calls),
        );
        let nonterminal: Vec<&Value> = served
            .events
            .iter()
            .filter(|event| event["type"] == "response.output_item.done")
            .map(|event| &event["item"])
            .filter(|item| {
                item["type"]
                    .as_str()
                    .is_some_and(|kind| kind.ends_with("_call"))
                    && !matches!(
                        item["status"].as_str(),
                        Some("completed" | "incomplete" | "failed")
                    )
            })
            .collect();
        due(
            "output_item.done items with a nonterminal status".to_string(),
            json!(nonterminal),
            json!([]),
        );
        let ids_of = |kind: &str| -> Vec<&Value> {
            served
                .events
                .iter()
                .filter(|event| event["type"] == kind)
                .map(|event| &event["item"]["id"])
                .collect()
        };
        let done = ids_of("response.output_item.done");
        let call_ids_of = |kind: &str| -> Vec<&Value> {
            served
                .events
                .iter()
                .filter(|event| {
                    event["type"] == kind
                        && event["item"]["type"]
                            .as_str()
                            .is_some_and(|item_type| item_type.ends_with("_call"))
                })
                .map(|event| &event["item"]["id"])
                .collect()
        };
        due(
            "call items closed exactly once".to_string(),
            json!(call_ids_of("response.output_item.done")),
            json!(call_ids_of("response.output_item.added")),
        );
        let argument_done: Vec<&Value> = served
            .events
            .iter()
            .filter(|event| event["type"] == "response.mcp_call_arguments.done")
            .map(|event| &event["item_id"])
            .collect();
        due(
            "call arguments closed exactly once".to_string(),
            json!(argument_done),
            json!(call_ids_of("response.output_item.added")),
        );
        let open: Vec<&Value> = ids_of("response.output_item.added")
            .into_iter()
            .filter(|id| !done.contains(id))
            .collect();
        due(
            "items added and never closed".to_string(),
            json!(open),
            json!([]),
        );
    }
    wrong
}

#[tokio::test]
async fn the_loop_runs_the_server_call_a_turn_ended_on() {
    let exits = exits();
    let texts: Vec<&str> = exits
        .iter()
        .map(|exit| exit.turn.as_str())
        .chain([ANSWER])
        .collect();
    let mut ids = token_ids(&texts);
    let answer = Turn {
        output_ids: ids.pop().unwrap(),
        prompt_tokens: 9,
        end: End::Finish("stop"),
    };

    // Every successfully served exit is read before any is judged, so one run
    // shows all response mismatches.
    let mut misread = Vec::new();
    for (exit, output_ids) in exits.iter().zip(ids) {
        let turns = [
            Turn {
                output_ids,
                prompt_tokens: 7,
                end: exit.end,
            },
            answer.clone(),
        ];
        for stream in [false, true] {
            // The plain form reads the terminal frame: an exit without one
            // is the stream's alone.
            if !stream && (matches!(exit.end, End::Eof) || exit.stream_only) {
                continue;
            }
            let served = serve(&turns, stream).await;
            for line in misreads(exit, stream, &served) {
                misread.push(format!("{} (stream={stream}): {line}", exit.name));
            }
        }
    }
    assert!(
        misread.is_empty(),
        "{} misreads:\n{}",
        misread.len(),
        misread.join("\n")
    );
}

/// Codex replays a patch result and asks for another edit. The model sees its
/// function-shaped tool, while the client receives raw patch text and a call id
/// it can echo on the next turn.
#[tokio::test]
async fn harmony_replays_and_emits_codex_custom_patch_calls() {
    let patch = "*** Begin Patch\n*** Add File: hello.txt\n+Hello\n*** End Patch\n";
    let reply = format!(
        "<|channel|>commentary to=functions.apply_patch <|constrain|>json<|message|>{}<|call|>",
        json!({"input": patch})
    );
    let turns = [Turn {
        output_ids: token_ids(&[&reply]).remove(0),
        prompt_tokens: 24,
        end: End::Finish("stop"),
    }];
    let input = json!([
        {"type":"additional_tools", "role":"developer", "tools":[
            {"type":"namespace", "name":"functions", "description":"Default tools", "tools":[
                {"type":"custom", "name":"apply_patch"}
            ]}
        ]},
        {"role":"user", "content":"Create a README and a greeting."},
        {"type":"custom_tool_call", "name":"apply_patch", "call_id":"call_readme",
         "input":"*** Begin Patch\n*** Add File: README.md\n+# Demo\n*** End Patch"},
        {"type":"custom_tool_call_output", "call_id":"call_readme", "output":"Exit code: 0\nWall time: 0.1 seconds\nOutput:\nSuccess. Updated the following files:\nA README.md\n"}
    ]);
    for stream in [false, true] {
        let served = serve_with_input(&turns, stream, input.clone()).await;
        let item = served.response["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["name"] == "apply_patch")
            .unwrap();
        assert_eq!(item["type"], "custom_tool_call");
        assert_eq!(item["input"], patch);
        assert_eq!(served.response["output"], served.stored["output"]);
        assert_eq!(served.tool_calls, 0);
        if stream {
            let events: Vec<_> = served
                .events
                .iter()
                .filter(|e| e["item"]["id"] == item["id"] || e["item_id"] == item["id"])
                .collect();
            let types: Vec<_> = events.iter().filter_map(|e| e["type"].as_str()).collect();
            assert_eq!(
                types,
                [
                    "response.output_item.added",
                    "response.custom_tool_call_input.delta",
                    "response.custom_tool_call_input.done",
                    "response.output_item.done"
                ]
            );
            assert_eq!(events[0]["item"]["call_id"], item["call_id"]);
            assert_eq!(events[1]["delta"], patch);
            assert_eq!(events[2]["input"], patch);
        }
    }
}

/// A generation returned an exec command and started a patch before it ended.
/// The parser returned only the command. Close the announced patch item without
/// telling the client that it received an empty patch to run.
#[tokio::test]
async fn harmony_omits_a_cut_patch_after_a_complete_client_call() {
    let text = concat!(
        "<|channel|>commentary to=functions.exec_command <|constrain|>json<|message|>",
        r#"{"cmd":"ls"}"#,
        "<|call|><|start|>assistant",
        "<|channel|>commentary to=functions.apply_patch <|constrain|>json<|message|>",
        r#"{"input":"*** Begin Patch"#,
    );
    let turns = [Turn {
        output_ids: token_ids(&[text]).remove(0),
        prompt_tokens: 24,
        end: End::Finish("stop"),
    }];
    let input = json!([
        {"type":"additional_tools","role":"developer","tools":[
            {"type":"function","name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}},
            {"type":"custom","name":"apply_patch"}
        ]},
        {"role":"user","content":"List the files and patch the README."}
    ]);
    for stream in [false, true] {
        let served = serve_with_input(&turns, stream, input.clone()).await;
        assert_eq!(
            served.response["status"], "completed",
            "stream={stream}: {}",
            served.response
        );
        assert_eq!(served.stored["status"], "completed");
        assert_eq!(served.response["output"], served.stored["output"]);
        let calls: Vec<_> = served.response["output"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| item["name"].as_str())
            .collect();
        assert_eq!(calls, ["exec_command"]);
        if stream {
            let patch = served
                .events
                .iter()
                .find(|event| {
                    event["type"] == "response.output_item.added"
                        && event["item"]["name"] == "apply_patch"
                })
                .unwrap();
            let id = &patch["item"]["id"];
            assert!(served
                .events
                .iter()
                .any(|event| event["type"] == "response.output_item.done"
                    && event["item"]["id"] == *id
                    && event["item"]["status"] == "incomplete"));
            assert!(served.events.iter().all(|event| event["item_id"] != *id));
        }
    }
}
