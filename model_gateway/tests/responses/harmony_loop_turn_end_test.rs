//! The ways a Harmony Responses turn ends inside or behind a call to a
//! server tool, through the tool loop, in the plain and the streamed form.
//! The loop runs the call only when the turn ended on it. A turn the token
//! limit, a stop, an abort or the stream's end left inside the call, or
//! behind arguments the turn did not finish on, runs nothing: the response
//! keeps the truncation, and the call reaches the client neither as an item
//! of the response nor through an event of success.

use serde_json::{json, Value};

use super::harmony_loop_usage_test::{serve, token_ids, End, Served, Turn, ANSWER};

/// A call to the mock tool server's search tool up to its arguments, the
/// arguments whole and cut, and the call's closing token.
const CALL: &str =
    "<|channel|>commentary to=functions.brave_web_search <|constrain|>json<|message|>";
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
    /// Whether the turn comes second, behind a turn whose server call ran and
    /// used up the request's cap of one tool call: its own allowance is none,
    /// so a server call of it opens no item on the stream.
    past_the_cap: bool,
    /// Whether the call runs; the loop then asks the model again and answers.
    runs: bool,
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
            past_the_cap: false,
            runs,
            status,
            reason,
        }
    }

    fn past_the_cap(mut self) -> Self {
        self.past_the_cap = true;
        self
    }
}

fn exits() -> Vec<Exit> {
    let limit = Some("max_output_tokens");
    vec![
        // The turn did not end on the call: nothing runs, the response is cut.
        Exit::new(
            "a stop inside the arguments",
            &[CALL, CUT],
            End::Finish("stop"),
            false,
            "incomplete",
            None,
        ),
        // The same stop on a turn past the request's cap: the call has no
        // item on the stream, and the response is cut all the same.
        Exit::new(
            "a stop inside the arguments, past the cap",
            &[CALL, CUT],
            End::Finish("stop"),
            false,
            "incomplete",
            None,
        )
        .past_the_cap(),
        Exit::new(
            "the token limit inside the arguments",
            &[CALL, CUT],
            End::Finish("length"),
            false,
            "incomplete",
            limit,
        ),
        Exit::new(
            "the token limit inside the tool's name",
            &[CUT_NAME],
            End::Finish("length"),
            false,
            "incomplete",
            limit,
        ),
        Exit::new(
            "the token limit behind whole arguments",
            &[CALL, WHOLE],
            End::Finish("length"),
            false,
            "incomplete",
            limit,
        ),
        Exit::new(
            "an abort inside the arguments",
            &[CALL, CUT],
            End::Finish("abort"),
            false,
            "incomplete",
            None,
        ),
        Exit::new(
            "an abort behind whole arguments",
            &[CALL, WHOLE],
            End::Finish("abort"),
            false,
            "incomplete",
            None,
        ),
        Exit::new(
            "the stream's end inside the arguments",
            &[CALL, CUT],
            End::Eof,
            false,
            "incomplete",
            None,
        ),
        // The turn ended on the call: it runs, and the loop answers on its
        // result.
        Exit::new(
            "a stop at the closing token, not delivered",
            &[CALL, WHOLE],
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
        // The call closed and the token limit cut a second one: the closed
        // call runs and the cut one is not read.
        Exit::new(
            "the token limit inside a second call",
            &[CALL, WHOLE, CLOSE, NEXT, CALL, CUT],
            End::Finish("length"),
            true,
            "completed",
            None,
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
    // The calls that ran: the one of the turn before, under the cap, and this
    // turn's own.
    let ran = usize::from(exit.past_the_cap) + usize::from(exit.runs);
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
    // A call that ran is an item of the response, completed; one that did
    // not run is no item of it.
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
        due(
            "terminal events of the status".to_string(),
            json!(events_of(&format!("response.{}", exit.status))),
            json!(1),
        );
        // No event of success for a call that did not run, and its item, once
        // announced, closes as cut.
        due(
            "mcp_call.completed events".to_string(),
            json!(events_of("response.mcp_call.completed")),
            json!(ran),
        );
        let closed: Vec<&Value> = served
            .events
            .iter()
            .filter(|event| {
                event["type"] == "response.output_item.done" && event["item"]["type"] == "mcp_call"
            })
            .map(|event| &event["item"]["status"])
            .collect();
        // The un-run call has an item when the turn read a call and stood
        // within its allowance.
        let mut closed_due = calls;
        if !exit.runs && !exit.past_the_cap && exit.turn.starts_with(CALL) {
            closed_due.push("incomplete");
        }
        due(
            "the closed mcp_call items".to_string(),
            json!(closed),
            json!(closed_due),
        );
    }
    wrong
}

#[tokio::test]
async fn the_loop_runs_a_server_call_only_when_its_turn_ended_on_it() {
    let exits = exits();
    // The turn before a turn past the cap: a whole call, which runs.
    let whole_call = [CALL, WHOLE, CLOSE].concat();
    let texts: Vec<&str> = exits
        .iter()
        .map(|exit| exit.turn.as_str())
        .chain([whole_call.as_str(), ANSWER])
        .collect();
    let mut ids = token_ids(&texts);
    let answer = Turn {
        output_ids: ids.pop().unwrap(),
        prompt_tokens: 9,
        end: End::Finish("stop"),
    };
    let first = Turn {
        output_ids: ids.pop().unwrap(),
        prompt_tokens: 5,
        end: End::Finish("stop"),
    };

    // Every exit is read before any is judged, so one run shows them all.
    let mut misread = Vec::new();
    for (exit, output_ids) in exits.iter().zip(ids) {
        let turn = Turn {
            output_ids,
            prompt_tokens: 7,
            end: exit.end,
        };
        let turns = if exit.past_the_cap {
            vec![first.clone(), turn, answer.clone()]
        } else {
            vec![turn, answer.clone()]
        };
        let max_tool_calls = exit.past_the_cap.then_some(1);
        for stream in [false, true] {
            // The plain form reads the terminal frame: an exit without one
            // is the stream's alone.
            if matches!(exit.end, End::Eof) && !stream {
                continue;
            }
            let served = serve(&turns, stream, max_tool_calls).await;
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
