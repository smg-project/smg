//! The turns a checkpoint's chat template renders, read back by its table.
//!
//! `tests/fixtures/turns/render.py` renders, with each checkpoint's own chat template, an
//! assistant message of every shape a tool-calling conversation takes (a plain answer, one call,
//! several, a call with no arguments, a call after content, string arguments that hold a bracket,
//! a newline or the list's close, non-ASCII text, nested values) and records, one JSON line per
//! turn in `tests/fixtures/turns/<table>.jsonl`, the bytes the template wrote for the turn, the
//! generation prompt's tail before it, the request's tools and the message the template rendered.
//! This test replays every turn through the table named on it, after the tail, whole, at every
//! two-way split, character by character and on thirty seeded plans, folds the events into the
//! assistant message with the chat adapter, and requires the rendered message back at every cut:
//! the content, and each call's name and arguments as JSON values. Where bellwether has recorded
//! the checkpoint, its hand-written set renders the same bytes for the same shapes, so these turns
//! are the recorded ones for a table whose corpus is not yet in the fixture pin, and the call
//! shapes for a table whose template the recorder could not render.

#[expect(
    dead_code,
    reason = "the integration tests share these helpers, and this one replays after a prompt only"
)]
mod common;

use std::{fs, path::PathBuf};

use common::{bytes_of, chunkings, replay_after};
use openai_protocol::common::Tool;
use serde::Deserialize;
use serde_json::Value;
use symphony::{adapt, formats::Table, Declared, Engine, EngineFinish};

/// One rendered turn.
#[derive(Deserialize)]
struct Turn {
    table: String,
    case: String,
    tail: String,
    output: String,
    tools: Vec<Tool>,
    content: String,
    /// Each call's name and its arguments as the request gave them, written as JSON.
    calls: Vec<(String, String)>,
}

/// The message a replay gave, in the terms the fixture uses: the content, each call's name and
/// its arguments as a JSON value, and the finish.
#[derive(Debug, PartialEq, Eq)]
struct Message {
    content: String,
    calls: Vec<(String, Value)>,
    finish: String,
}

impl Message {
    #[expect(
        clippy::expect_used,
        reason = "a fixture whose arguments are not JSON ends the test with the case"
    )]
    fn of_turn(turn: &Turn) -> Self {
        Self {
            content: turn.content.clone(),
            calls: turn
                .calls
                .iter()
                .map(|(name, arguments)| {
                    (
                        name.clone(),
                        serde_json::from_str(arguments).expect("the fixture's arguments are JSON"),
                    )
                })
                .collect(),
            finish: if turn.calls.is_empty() {
                "stop".to_string()
            } else {
                "tool_calls".to_string()
            },
        }
    }

    fn of_events(events: &[symphony::Event]) -> Self {
        let choice = adapt::chat::message(0, events);
        Self {
            content: choice.message.content.unwrap_or_default(),
            calls: choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| {
                    let arguments = call.function.arguments.unwrap_or_default();
                    (
                        call.function.name,
                        serde_json::from_str(&arguments).unwrap_or(Value::String(arguments)),
                    )
                })
                .collect(),
            finish: choice.finish_reason.unwrap_or_default(),
        }
    }
}

#[expect(
    clippy::panic,
    clippy::expect_used,
    reason = "a fixture that cannot be read ends the test with its path"
)]
fn turns() -> Vec<Turn> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/turns");
    let mut files: Vec<PathBuf> = fs::read_dir(&directory)
        .unwrap_or_else(|e| panic!("{}: {e}", directory.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no turn fixtures under {}",
        directory.display()
    );
    files
        .iter()
        .flat_map(|path| {
            let text =
                fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            text.lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    serde_json::from_str::<Turn>(line)
                        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
#[expect(
    clippy::print_stdout,
    reason = "the count of turns is the run's summary"
)]
fn every_rendered_turn_reads_back_as_the_message_the_template_rendered_at_every_cut() {
    let turns = turns();
    let mut tables_seen = Vec::new();
    for turn in &turns {
        let table = Table::named(&turn.table).unwrap_or_else(|| {
            panic!(
                "{}/{}: no table named {:?}",
                turn.table, turn.case, turn.table
            )
        });
        if !tables_seen.contains(&table) {
            tables_seen.push(table);
        }
        let expected = Message::of_turn(turn);
        for (plan, cuts) in chunkings(&turn.output, usize::MAX) {
            let mut parser = Engine::new(table.format(), Declared::of(&turn.tools));
            let events = replay_after(
                &mut parser,
                &turn.tail,
                &turn.output,
                &cuts,
                &EngineFinish::Stop,
                false,
            )
            .unwrap_or_else(|e| panic!("{}/{}: {plan}: {e}", turn.table, turn.case));
            let bytes: String = events.iter().map(bytes_of).collect();
            assert_eq!(
                bytes, turn.output,
                "{}/{}: {plan}: the bytes",
                turn.table, turn.case
            );
            assert_eq!(
                Message::of_events(&events),
                expected,
                "{}/{}: {plan}",
                turn.table,
                turn.case
            );
        }
    }
    println!(
        "{} turns of {} table(s) read back at every cut",
        turns.len(),
        tables_seen.len()
    );
}
