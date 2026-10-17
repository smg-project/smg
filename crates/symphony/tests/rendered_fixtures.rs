//! The turns a family's chat template renders, read back through its table. A table whose family
//! has no recorded set stands on these until bellwether records one: `tests/fixtures/rendered/
//! <table>.json`, written by the script the file names under `scripts/` from the checkpoint's
//! chat template, or from its reference encoder where the checkpoint ships one instead, for the
//! shapes a tool-calling conversation takes (an answer, a thought and an answer, one call, several,
//! a call after a thought, a call with no arguments, nested and non-ASCII values, the turn after
//! a tool result). Each turn names the generation prompt's tail, the output the model writes (the
//! stop token the decoder hides left out) and the message the turn stands for: its thought, its
//! content, and its calls as names and arguments. The table reads every turn whole, at every
//! two-way split, character by character and under the seeded plans, and gives every byte in
//! order, the message's thought and content, its calls with the same arguments, no malformed
//! text, and a stop finish that counts the calls.

#[expect(
    dead_code,
    reason = "the integration tests share these helpers, and this one replays after a prompt alone"
)]
mod common;

use std::{fs, path::PathBuf};

use common::{bytes_of, chunkings, replay_after};
use openai_protocol::common::Tool;
use serde::Deserialize;
use serde_json::Value;
use symphony::{formats::Table, Declared, Engine, EngineFinish, Event, FinishReason};

/// One file under `tests/fixtures/rendered/`.
#[derive(Deserialize)]
struct Fixture {
    /// The table's name, as [`Table::named`] takes it.
    table: String,
    /// The request's tools, which type a call's values where the table reads them so.
    #[serde(default)]
    tools: Vec<Tool>,
    turns: Vec<Turn>,
}

/// A turn as rendered, and the message it stands for.
#[derive(Deserialize)]
struct Turn {
    name: String,
    /// The generation prompt's tail, from its last user or tool message to its end.
    tail: String,
    /// What the model writes after the prompt, without the stop token that ends the turn.
    output: String,
    reasoning: String,
    content: String,
    calls: Vec<Call>,
}

#[derive(Deserialize)]
struct Call {
    name: String,
    arguments: Value,
}

/// What the events say: the thought, the content, the calls with their arguments as the JSON
/// value their fragments make, the malformed text, and the finish with its count of calls.
struct Reading {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
    malformed: Vec<String>,
    finish: Option<(FinishReason, u32)>,
}

#[expect(
    clippy::panic,
    reason = "a call's arguments that are not JSON fail the turn, with the turn named"
)]
fn reading(events: &[Event], place: &dyn Fn() -> String) -> Reading {
    let mut reading = Reading {
        reasoning: String::new(),
        content: String::new(),
        calls: Vec::new(),
        malformed: Vec::new(),
        finish: None,
    };
    let mut arguments: Vec<String> = Vec::new();
    for event in events {
        match event {
            Event::Reasoning(text) => reading.reasoning.push_str(&text.text),
            Event::Content(text) => reading.content.push_str(&text.text),
            Event::ToolCallStart { index, name, .. } => {
                assert_eq!(*index as usize, arguments.len(), "{}", place());
                reading.calls.push((name.clone(), Value::Null));
                arguments.push(String::new());
            }
            Event::ToolCallArguments { index, json, .. } => {
                arguments[*index as usize].push_str(json);
            }
            Event::Malformed { text, .. } => reading.malformed.push(text.text.to_string()),
            Event::Finish {
                reason, tool_calls, ..
            } => reading.finish = Some((reason.clone(), *tool_calls)),
            _ => {}
        }
    }
    for (call, text) in reading.calls.iter_mut().zip(arguments) {
        call.1 = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{}: {}: arguments {text:?}: {e}", place(), call.0));
    }
    reading
}

#[expect(
    clippy::panic,
    clippy::expect_used,
    reason = "a fixture that cannot be read fails the test, with the file named"
)]
fn fixtures() -> Vec<(PathBuf, Fixture)> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rendered");
    let mut paths: Vec<PathBuf> = fs::read_dir(&directory)
        .unwrap_or_else(|e| panic!("{}: {e}", directory.display()))
        .map(|entry| entry.expect("an entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let text =
                fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let fixture: Fixture =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            (path, fixture)
        })
        .collect()
}

#[test]
fn every_fixture_names_a_registry_table_and_its_turns_once() {
    let found = fixtures();
    assert!(!found.is_empty(), "no rendered fixtures");
    let mut tables: Vec<String> = Vec::new();
    for (path, fixture) in &found {
        let table = Table::named(&fixture.table)
            .unwrap_or_else(|| panic!("{}: no table named {:?}", path.display(), fixture.table));
        assert_eq!(
            path.file_stem().and_then(|stem| stem.to_str()),
            Some(table.name().replace('.', "_").as_str()),
            "{}: the file is named after its table",
            path.display()
        );
        assert!(
            !tables.contains(&fixture.table),
            "{}: a second fixture for {}",
            path.display(),
            fixture.table
        );
        tables.push(fixture.table.clone());
        let mut names: Vec<&str> = fixture
            .turns
            .iter()
            .map(|turn| turn.name.as_str())
            .collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "{}: turns named twice", path.display());
        assert!(count >= 6, "{}: too few turns to stand on", path.display());
    }
}

#[test]
fn every_rendered_turn_reads_back_as_the_message_it_stands_for() {
    for (path, fixture) in fixtures() {
        let table = Table::named(&fixture.table).expect("a registry table");
        let declared = Declared::of(&fixture.tools);
        for turn in &fixture.turns {
            let expected_calls: Vec<(String, Value)> = turn
                .calls
                .iter()
                .map(|call| (call.name.clone(), call.arguments.clone()))
                .collect();
            let count = u32::try_from(turn.calls.len()).expect("a few calls");
            for (plan, cuts) in chunkings(&turn.output, usize::MAX) {
                let place = || format!("{}: {}: {plan}", path.display(), turn.name);
                let mut engine = Engine::new(table.format(), declared.clone());
                let events = replay_after(
                    &mut engine,
                    &turn.tail,
                    &turn.output,
                    &cuts,
                    &EngineFinish::Stop,
                    false,
                )
                .unwrap_or_else(|e| panic!("{}: {e}", place()));
                let conserved: String = events.iter().map(bytes_of).collect();
                assert_eq!(conserved, turn.output, "{}", place());
                assert!(
                    matches!(events.last(), Some(Event::Finish { .. })),
                    "{}: Finish is last",
                    place()
                );
                let read = reading(&events, &place);
                assert_eq!(read.reasoning, turn.reasoning, "{}", place());
                assert_eq!(read.content, turn.content, "{}", place());
                assert_eq!(read.calls, expected_calls, "{}", place());
                assert_eq!(read.malformed, Vec::<String>::new(), "{}", place());
                assert_eq!(
                    read.finish,
                    Some((FinishReason::Stop, count)),
                    "{}",
                    place()
                );
            }
        }
    }
}
