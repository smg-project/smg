//! The harness every fuzz target shares.
//!
//! A target hands the fuzzer's bytes to [`run`] with the table it fuzzes, or none for the target
//! that lets the bytes pick one. [`decode`] reads the bytes as a case: the text, and from the
//! last twelve bytes the plan that cuts the text into deltas, whether empty deltas come between
//! the pieces, how long the prompt's tail at the front of the text is, how the engine finished and
//! which table reads it. [`check`] replays the case whole and in pieces and asserts the crate's
//! contract, so that a breach is a panic the fuzzer keeps with the bytes that caused it:
//!
//! - no `feed` fails and nothing panics, on any bytes, cut anywhere;
//! - conservation: the bytes of the events, in order, are the output, with `Finish` last;
//! - what the parser says does not depend on the cuts: the whole replay and the pieces agree on
//!   the content, each reasoning region, the dropped and malformed text by reason, the calls with
//!   their arguments and the finish;
//! - well-formedness: one start and one end per call, arguments only between them and a valid
//!   JSON prefix after every fragment, reasoning text only inside a region, one `Finish` that
//!   counts the calls; and a call its own terminal closed has arguments that are whole JSON,
//!   unless the engine said `Malformed` while the call was open;
//! - the adapters render any event sequence without a panic, and the Chat Completions stream,
//!   concatenated, says what the whole message says.
//!
//! Every breach message starts with `breach <name>:`, the invariant's name, so that the artifacts
//! of a campaign group by class.
//!
//! The seeds `prepare` writes from the crate's recorded output slices are cases in this layout,
//! so a seed replays as its table's own output and every mutation of it stays a case.

#![forbid(unsafe_code)]

use std::{fs, path::Path, sync::OnceLock};

/// The macro a target is written with, so that the targets and the harness share one dependency.
pub use libfuzzer_sys::fuzz_target;
use openai_protocol::common::{Function, Tool};
use serde::Deserialize;
use serde_json::{json, Value};
use symphony::{
    adapt::{chat, messages, responses},
    formats::Table,
    json::PartialJson,
    Declared, DropReason, Engine, EngineFinish, Event, Events, FinishReason, Input,
    MalformedReason, Parser,
};

/// The targets and the table each fuzzes; `None` lets the input's last byte pick one of
/// [`Table::ALL`]. `prepare` writes one dictionary per entry; a test keeps the entries and the
/// files under `fuzz_targets/` in step.
pub const TARGETS: &[(&str, Option<Table>)] = &[
    ("any_table", None),
    ("json_calls", Some(Table::Qwen3)),
    ("tagged_calls", Some(Table::Qwen3Tagged)),
    ("keyed_calls", Some(Table::Glm)),
    ("attributed_calls", Some(Table::DeepSeekV4_1)),
    ("message_calls", Some(Table::KimiK3)),
    ("xml_calls", Some(Table::MinimaxM3)),
    ("gemma_calls", Some(Table::Gemma4)),
    ("pythonic_calls", Some(Table::Lfm2_5)),
];

/// Whether the breach named `name` is allowed for this run: `SYMPHONY_FUZZ_ALLOW` lists, comma
/// separated, the names of the invariants a run looks past, so that a class already reported does
/// not stop the fuzzer from finding the next one. Only `whole-json` honors it; the others are
/// not known to be broken.
pub fn allowed(name: &str) -> bool {
    static ALLOWED: OnceLock<Vec<String>> = OnceLock::new();
    ALLOWED
        .get_or_init(|| {
            std::env::var("SYMPHONY_FUZZ_ALLOW")
                .map(|list| list.split(',').map(|n| n.trim().to_string()).collect())
                .unwrap_or_default()
        })
        .iter()
        .any(|allowed| allowed == name)
}

/// The table the target named `target` fuzzes: `Some(None)` for the target whose input picks the
/// table, `None` for no target of that name.
pub fn table_of(target: &str) -> Option<Option<Table>> {
    TARGETS
        .iter()
        .find(|(name, _)| *name == target)
        .map(|(_, table)| *table)
}

/// One case: the table, the prompt's tail the engine is seeded with, the output, where the output
/// is cut into deltas, whether an empty delta comes between the pieces, and the engine's finish.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Case {
    pub table: Table,
    pub prompt: String,
    pub output: String,
    pub cuts: Vec<usize>,
    pub empty_between: bool,
    pub finish: EngineFinish,
}

/// The parameters at the end of an input: eight piece sizes, the flags, the prompt's length, the
/// pieces' scale and the table.
const TAIL: usize = 12;
const FLAG_EMPTY_BETWEEN: u8 = 1;
const FLAG_WHOLE: u8 = 8;

/// Reads `data` as a case for `table`, or for the table its last byte picks when `table` is
/// `None`. Shorter than the tail, the bytes are the output alone, read whole with a stop.
pub fn decode(data: &[u8], table: Option<Table>) -> Case {
    let (text, tail) = if data.len() >= TAIL {
        data.split_at(data.len() - TAIL)
    } else {
        (data, &[][..])
    };
    let text = String::from_utf8_lossy(text).into_owned();
    let byte = |at: usize| tail.get(at).copied().unwrap_or(0);
    let table = table.unwrap_or(Table::ALL[byte(11) as usize % Table::ALL.len()]);
    let flags = byte(8);
    let finish = match (flags >> 1) & 3 {
        0 => EngineFinish::Stop,
        1 => EngineFinish::Length,
        2 => EngineFinish::Abort,
        _ => EngineFinish::Other("content_filter".to_string()),
    };
    let (prompt, output) = split_prompt(text, byte(9) as usize);
    let scale = 1 + byte(10) as usize % 8;
    let cuts = if flags & FLAG_WHOLE != 0 || tail.is_empty() {
        Vec::new()
    } else {
        let sizes: Vec<usize> = tail[..8]
            .iter()
            .map(|b| scale * (1 + *b as usize % 16))
            .collect();
        plan(&output, &sizes)
    };
    Case {
        table,
        prompt,
        output,
        cuts,
        empty_between: flags & FLAG_EMPTY_BETWEEN != 0,
        finish,
    }
}

/// The input that reads as `table`'s `output` after `prompt`, cut into pieces of about a token,
/// with a stop: the seed `prepare` writes for a recorded slice.
pub fn encode(table: Table, prompt: &str, output: &str) -> Result<Vec<u8>, String> {
    let prompt_len = u8::try_from(prompt.len()).map_err(|_| {
        format!(
            "a prompt tail of {} bytes does not fit the seed",
            prompt.len()
        )
    })?;
    let Some(index) = Table::ALL.iter().position(|t| *t == table) else {
        return Err(format!("{} is not in Table::ALL", table.name()));
    };
    let mut data = Vec::with_capacity(prompt.len() + output.len() + TAIL);
    data.extend_from_slice(prompt.as_bytes());
    data.extend_from_slice(output.as_bytes());
    data.extend_from_slice(&[3; 8]);
    data.push(0);
    data.push(prompt_len);
    data.push(0);
    data.push(index as u8);
    Ok(data)
}

/// `text` split after its first `len` bytes, moved back to a character boundary and clamped to
/// the text.
fn split_prompt(text: String, len: usize) -> (String, String) {
    let mut at = len.min(text.len());
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    let output = text[at..].to_string();
    let mut prompt = text;
    prompt.truncate(at);
    (prompt, output)
}

/// The byte offsets that cut `text` into pieces of `sizes` characters, cycling through the sizes.
fn plan(text: &str, sizes: &[usize]) -> Vec<usize> {
    let mut cuts = Vec::new();
    let mut in_piece = 0;
    let mut which = 0;
    for (at, _) in text.char_indices() {
        if at > 0 && in_piece == sizes[which % sizes.len()] {
            cuts.push(at);
            in_piece = 0;
            which += 1;
        }
        in_piece += 1;
    }
    cuts
}

/// A decoded case, checked.
pub fn run(data: &[u8], table: Option<Table>) {
    check(&decode(data, table));
}

/// Replays `case` whole and in its pieces and asserts the contract on both.
pub fn check(case: &Case) {
    let whole = replay(case, &[]);
    let pieces = replay(case, &case.cuts);
    for (which, replayed) in [("whole", &whole), ("pieces", &pieces)] {
        let conserved: String = replayed.events.iter().map(bytes_of).collect();
        assert_eq!(
            conserved,
            case.output,
            "breach conservation: {}: the {which} replay does not conserve the output",
            describe(case)
        );
        assert!(
            matches!(replayed.events.last(), Some(Event::Finish { .. })),
            "breach finish-last: {}: the {which} replay does not end with Finish",
            describe(case)
        );
        check_well_formed(case, which, replayed);
    }
    assert_eq!(
        Said::of(&whole.events),
        Said::of(&pieces.events),
        "breach chunking: {}: the pieces say something else than the whole",
        describe(case)
    );
    check_adapters(case, &whole.events, &pieces.events);
}

/// The request's tools: a string and an integer, a string and an object.
pub fn declared() -> Declared {
    let tool = |name: &str, parameters: Value| Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters,
            strict: None,
            extra: Default::default(),
        },
    };
    Declared::of(&[
        tool(
            "get_weather",
            json!({"type": "object", "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            }}),
        ),
        tool(
            "change_drink",
            json!({"type": "object", "properties": {
                "drink_id": {"type": "string"},
                "new_preferences": {"type": "object"},
            }}),
        ),
    ])
}

/// The names [`declared`] declares, for a dictionary.
pub fn declared_names() -> Vec<&'static str> {
    vec![
        "get_weather",
        "change_drink",
        "city",
        "days",
        "drink_id",
        "new_preferences",
    ]
}

/// One recorded output slice of a table family, as the crate's fixtures record it.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Slice {
    pub family: String,
    pub prompt: String,
    pub output: String,
}

/// The slices under `dir`: one `<table>.jsonl` per table, one slice per line.
pub fn slices(dir: &Path) -> Result<Vec<(Table, String, Slice)>, String> {
    let mut out = Vec::new();
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut paths: Vec<_> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    paths.sort();
    for path in paths {
        let name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| format!("{}: no table name", path.display()))?;
        let table = Table::named(name)
            .ok_or_else(|| format!("{}: no table is named {name}", path.display()))?;
        let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        for (line, json) in text.lines().enumerate() {
            let slice: Slice = serde_json::from_str(json)
                .map_err(|e| format!("{}:{}: {e}", path.display(), line + 1))?;
            out.push((table, slice.family.clone(), slice));
        }
    }
    Ok(out)
}

/// The events of one replay and how many of them came before the end of the stream.
struct Replayed {
    events: Vec<Event>,
    before_end: usize,
}

/// `case` through a new engine, its output cut at `cuts`, the events drained after every feed as
/// a stream's reader drains them.
fn replay(case: &Case, cuts: &[usize]) -> Replayed {
    let mut engine = Engine::new(case.table.format(), declared());
    let mut out = Events::new();
    let mut events = Vec::new();
    let delta = |text: &'static str| Input::Delta {
        token_ids: &[],
        text,
        spans: &[],
    };
    feed(
        case,
        &mut engine,
        Input::Prompt {
            token_ids: &[],
            text: &case.prompt,
        },
        &mut out,
        &mut events,
    );
    let mut from = 0;
    for &cut in cuts.iter().chain(std::iter::once(&case.output.len())) {
        if cut > from {
            if case.empty_between {
                feed(case, &mut engine, delta(""), &mut out, &mut events);
            }
            feed(
                case,
                &mut engine,
                Input::Delta {
                    token_ids: &[],
                    text: &case.output[from..cut],
                    spans: &[],
                },
                &mut out,
                &mut events,
            );
            from = cut;
        }
    }
    if case.empty_between {
        feed(case, &mut engine, delta(""), &mut out, &mut events);
    }
    let before_end = events.len();
    feed(
        case,
        &mut engine,
        Input::End {
            finish: case.finish.clone(),
        },
        &mut out,
        &mut events,
    );
    Replayed { events, before_end }
}

fn feed(
    case: &Case,
    engine: &mut Engine,
    input: Input<'_>,
    out: &mut Events,
    events: &mut Vec<Event>,
) {
    let step = format!("{input:?}");
    if let Err(why) = engine.feed(input, out) {
        panic!(
            "breach feed-failed: {}: feed of {step} failed: {why}",
            describe(case)
        );
    }
    events.extend(out.drain());
}

/// The case, for a message.
fn describe(case: &Case) -> String {
    format!(
        "table {} prompt {:?} output {:?} cuts {:?} empty_between {} finish {:?}",
        case.table.name(),
        case.prompt,
        case.output,
        case.cuts,
        case.empty_between,
        case.finish
    )
}

/// The output bytes `event` accounts for.
fn bytes_of(event: &Event) -> &str {
    match event {
        Event::Content(t) | Event::Reasoning(t) => t.text.as_str(),
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => source.text.as_str(),
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
    }
}

/// Whether `text` is a JSON prefix the prefix parser takes whole.
fn valid_json_prefix(text: &str) -> bool {
    matches!(PartialJson::default().parse(text, true), Ok((_, consumed)) if consumed == text.len())
}

/// What a client relies on in the shape of the stream, checked event by event.
fn check_well_formed(case: &Case, which: &str, replayed: &Replayed) {
    let mut started: Vec<u32> = Vec::new();
    let mut ended: Vec<u32> = Vec::new();
    let mut arguments: Vec<(u32, String)> = Vec::new();
    let mut malformed_inside: Vec<u32> = Vec::new();
    let mut reasoning_open = false;
    let mut finished = false;
    for (at, event) in replayed.events.iter().enumerate() {
        let place = || format!("{} ({which}), event {at}: {event:?}", describe(case));
        assert!(
            !finished,
            "breach after-finish: {}: an event after Finish",
            place()
        );
        match event {
            Event::ToolCallStart { index, .. } => {
                assert!(
                    !started.contains(index),
                    "breach second-start: {}: a second start",
                    place()
                );
                started.push(*index);
                arguments.push((*index, String::new()));
            }
            Event::ToolCallArguments { index, json, .. } => {
                assert!(
                    started.contains(index),
                    "breach arguments-before-start: {}: arguments before the call's start",
                    place()
                );
                assert!(
                    !ended.contains(index),
                    "breach arguments-after-end: {}: arguments after the call's end",
                    place()
                );
                if let Some((_, so_far)) = arguments.iter_mut().find(|(i, _)| i == index) {
                    so_far.push_str(json);
                    assert!(
                        valid_json_prefix(so_far),
                        "breach json-prefix: {}: the arguments so far, {so_far:?}, are no valid JSON prefix",
                        place()
                    );
                }
            }
            Event::ToolCallEnd { index, .. } => {
                assert!(
                    started.contains(index),
                    "breach end-before-start: {}: an end before the call's start",
                    place()
                );
                assert!(
                    !ended.contains(index),
                    "breach second-end: {}: a second end",
                    place()
                );
                ended.push(*index);
                if at < replayed.before_end
                    && !malformed_inside.contains(index)
                    && !allowed("whole-json")
                {
                    let so_far = arguments
                        .iter()
                        .find(|(i, _)| i == index)
                        .map(|(_, so_far)| so_far.as_str())
                        .unwrap_or_default();
                    assert!(
                        serde_json::from_str::<Value>(so_far).is_ok(),
                        "breach whole-json: {}: the call closed with arguments that are no whole JSON: {so_far:?}",
                        place()
                    );
                }
            }
            Event::Malformed { .. } => {
                for index in &started {
                    if !ended.contains(index) && !malformed_inside.contains(index) {
                        malformed_inside.push(*index);
                    }
                }
            }
            Event::ReasoningStart => {
                assert!(
                    !reasoning_open,
                    "breach reasoning-reopened: {}: reasoning already open",
                    place()
                );
                reasoning_open = true;
            }
            Event::Reasoning(_) => {
                assert!(
                    reasoning_open,
                    "breach reasoning-outside: {}: reasoning outside a region",
                    place()
                );
            }
            Event::ReasoningEnd => {
                assert!(
                    reasoning_open,
                    "breach reasoning-closed-twice: {}: no reasoning open",
                    place()
                );
                reasoning_open = false;
            }
            Event::Finish { tool_calls, .. } => {
                assert_eq!(
                    *tool_calls as usize,
                    started.len(),
                    "breach finish-count: {}: the finish does not count the calls that started",
                    place()
                );
                finished = true;
            }
            Event::Content(_) | Event::Dropped { .. } => {}
        }
    }
    let place = format!("{} ({which})", describe(case));
    assert!(finished, "breach no-finish: {place}: no Finish");
    assert!(
        !reasoning_open,
        "breach reasoning-left-open: {place}: reasoning left open"
    );
    assert_eq!(
        started.len(),
        ended.len(),
        "breach call-without-end: {place}: a call without an end"
    );
}

/// What the parser has said, as the streams a client would assemble.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Said {
    content: String,
    reasoning: Vec<String>,
    dropped: Vec<(DropReason, String)>,
    malformed: Vec<(MalformedReason, String)>,
    calls: Vec<(u32, String, String)>,
    arguments: Vec<(u32, String)>,
    ended: Vec<u32>,
    finish: Option<(FinishReason, u32, u32)>,
}

impl Said {
    fn of(events: &[Event]) -> Self {
        let mut said = Self::default();
        let mut region_open = false;
        for event in events {
            match event {
                Event::Content(t) => said.content.push_str(&t.text),
                Event::ReasoningStart => {
                    said.reasoning.push(String::new());
                    region_open = true;
                }
                Event::Reasoning(t) => {
                    if !region_open {
                        said.reasoning.push(String::new());
                        region_open = true;
                    }
                    if let Some(region) = said.reasoning.last_mut() {
                        region.push_str(&t.text);
                    }
                }
                Event::ReasoningEnd => region_open = false,
                Event::Dropped { text, why } => joined_by(&mut said.dropped, why, &text.text),
                Event::Malformed { text, why } => joined_by(&mut said.malformed, why, &text.text),
                Event::ToolCallStart {
                    index, id, name, ..
                } => {
                    said.calls.push((*index, id.clone(), name.clone()));
                    said.arguments.push((*index, String::new()));
                }
                Event::ToolCallArguments { index, json, .. } => {
                    if let Some((_, so_far)) = said.arguments.iter_mut().find(|(i, _)| i == index) {
                        so_far.push_str(json);
                    }
                }
                Event::ToolCallEnd { index, .. } => said.ended.push(*index),
                Event::Finish {
                    reason,
                    tool_calls,
                    reasoning_tokens,
                } => said.finish = Some((reason.clone(), *tool_calls, *reasoning_tokens)),
            }
        }
        said
    }
}

/// Append `text` to the entry for `why`, adding the entry when it is the reason's first text.
fn joined_by<R: Clone + PartialEq>(list: &mut Vec<(R, String)>, why: &R, text: &str) {
    match list.iter_mut().find(|(reason, _)| reason == why) {
        Some((_, joined)) => joined.push_str(text),
        None => list.push((why.clone(), text.to_string())),
    }
}

/// The adapters over both replays: no panic, and the Chat Completions stream of the pieces,
/// concatenated, is the message of the whole.
fn check_adapters(case: &Case, whole: &[Event], pieces: &[Event]) {
    let message = chat::message(0, whole);
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls: Vec<(u32, Option<String>, String)> = Vec::new();
    for choice in chat::deltas(0, pieces) {
        if let Some(text) = &choice.delta.content {
            content.push_str(text);
        }
        if let Some(text) = &choice.delta.reasoning_content {
            reasoning.push_str(text);
        }
        for call in choice.delta.tool_calls.iter().flatten() {
            let at = match calls.iter().position(|(index, _, _)| *index == call.index) {
                Some(at) => at,
                None => {
                    calls.push((call.index, None, String::new()));
                    calls.len() - 1
                }
            };
            if let Some(function) = &call.function {
                if let Some(name) = &function.name {
                    calls[at].1 = Some(name.clone());
                }
                if let Some(arguments) = &function.arguments {
                    calls[at].2.push_str(arguments);
                }
            }
        }
    }
    if content.trim().is_empty() {
        content.clear();
    }
    assert_eq!(
        message.message.content.clone().unwrap_or_default(),
        content,
        "breach chat-content: {}: the chat stream's content is not the message's",
        describe(case)
    );
    assert_eq!(
        message
            .message
            .reasoning_content
            .clone()
            .unwrap_or_default(),
        reasoning,
        "breach chat-reasoning: {}: the chat stream's reasoning is not the message's",
        describe(case)
    );
    let message_calls: Vec<(u32, Option<String>, String)> = message
        .message
        .tool_calls
        .iter()
        .flatten()
        .enumerate()
        .map(|(index, call)| {
            (
                index as u32,
                Some(call.function.name.clone()),
                call.function.arguments.clone().unwrap_or_default(),
            )
        })
        .collect();
    assert_eq!(
        message_calls,
        calls,
        "breach chat-calls: {}: the chat stream's calls are not the message's",
        describe(case)
    );

    let mut stream = messages::Stream::new();
    let mut out = Vec::new();
    let mut stop = None;
    for event in pieces {
        if let Some(reason) = stream.feed(event, &mut out) {
            stop = Some(reason);
        }
    }
    stream.close(&mut out);
    let output = messages::output(whole);
    assert_eq!(
        serde_json::to_string(&output.stop_reason).ok(),
        serde_json::to_string(&stop).ok(),
        "breach messages-stop: {}: the Messages stream's stop reason is not the output's",
        describe(case)
    );

    let mut stream = responses::Stream::new("resp");
    let mut out = Vec::new();
    for event in pieces {
        stream.feed(event, &mut out);
    }
    stream.close(&mut out);
    let _ = responses::output("resp", whole);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/slices")
    }

    #[test]
    fn every_recorded_slice_holds_the_contract() -> Result<(), String> {
        let slices = slices(&fixtures())?;
        assert_eq!(slices.len(), 40, "eight tables, five families each");
        for (table, family, slice) in &slices {
            let data = encode(*table, &slice.prompt, &slice.output)?;
            let case = decode(&data, None);
            assert_eq!(case.table, *table, "{family}");
            assert_eq!(case.prompt, slice.prompt, "{family}");
            assert_eq!(case.output, slice.output, "{family}");
            assert!(!case.cuts.is_empty(), "{family}: a seed replays in pieces");
            check(&case);
            check(&decode(&data, Some(*table)));
        }
        Ok(())
    }

    #[test]
    fn short_bytes_are_an_output_read_whole() {
        let case = decode(b"Hello", Some(Table::Qwen3));
        assert_eq!(case.output, "Hello");
        assert!(case.cuts.is_empty() && case.prompt.is_empty());
        assert_eq!(case.finish, EngineFinish::Stop);
        check(&case);
        check(&decode(b"", None));
    }

    #[test]
    fn the_plan_cuts_on_character_boundaries_in_the_sizes_given() {
        assert_eq!(plan("héllo wörld", &[2, 3]), vec![3, 6, 8, 12]);
        assert_eq!(plan("ab", &[5]), Vec::<usize>::new());
    }

    #[test]
    fn the_targets_name_the_tables_their_files_fuzz() -> Result<(), String> {
        for (name, table) in TARGETS {
            let path =
                Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("fuzz_targets/{name}.rs"));
            let source =
                fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let spelled = match table {
                Some(table) => format!("Some(Table::{table:?})"),
                None => "None".to_string(),
            };
            assert!(source.contains(&spelled), "{name}: expected {spelled}");
        }
        Ok(())
    }
}
