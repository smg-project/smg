//! Replaying a compiled table: feeding an output to it cut at given places, the ways to cut an
//! output, what a client assembles from the events, and the contract the events keep. Shared by
//! the compile's tests.

#![allow(
    clippy::allow_attributes,
    clippy::unwrap_used,
    clippy::panic,
    dead_code,
    reason = "test helpers shared by the compile's tests; a bad fixture or table should panic"
)]

use std::path::{Path, PathBuf};

use serde_json::Value;
use smg_response_template::{
    read_response_template,
    to_symphony::{compile, Compiled, Refusals},
};
use symphony::{
    json::PartialJson, Declared, DropReason, Engine, EngineFinish, Event, Events, FinishReason,
    Format, Input, MalformedReason, Parser,
};

use crate::common::{chunk_lengths, read_jsonl, untag};

/// The public templates and transformers' sessions for them.
pub fn compiled_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compiled")
}

/// A template row: its id, the template, and the texts of the tokens the model stops on.
pub struct Row {
    pub id: String,
    pub spec: Value,
    pub stops: Vec<String>,
}

pub fn rows(dir: &Path) -> Vec<Row> {
    read_jsonl(dir, "templates")
        .into_iter()
        .map(|row| Row {
            id: row["id"].as_str().unwrap().to_owned(),
            spec: untag(&row["template"]),
            stops: row
                .get("stops")
                .and_then(Value::as_array)
                .map(|stops| {
                    stops
                        .iter()
                        .map(|s| s.as_str().unwrap().to_owned())
                        .collect()
                })
                .unwrap_or_default(),
        })
        .collect()
}

/// `row`'s template compiled with its stops: `None` when it is not read.
pub fn compiled(row: &Row) -> Option<Result<Compiled, Refusals>> {
    let template = read_response_template(&row.spec).ok()?;
    let stops: Vec<&str> = row.stops.iter().map(String::as_str).collect();
    Some(compile(&template, &stops))
}

/// Compare `seen` with the pinned list `name` (rewritten with `TO_SYMPHONY_WRITE=1`).
pub fn pinned(name: &str, seen: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name);
    if std::env::var_os("TO_SYMPHONY_WRITE").is_some() {
        std::fs::write(&path, seen).unwrap();
    }
    let pinned = std::fs::read_to_string(&path).unwrap_or_default();
    if seen != pinned {
        let diff: Vec<String> = seen
            .lines()
            .filter(|line| !pinned.lines().any(|p| p == *line))
            .map(|line| format!("now    {line}"))
            .chain(
                pinned
                    .lines()
                    .filter(|line| !seen.lines().any(|s| s == *line))
                    .map(|line| format!("pinned {line}")),
            )
            .collect();
        panic!(
            "{name}: compile outcomes changed (TO_SYMPHONY_WRITE=1 rewrites the list):\n{}",
            diff.join("\n")
        );
    }
}

/// Feed `pieces` of the output after `prompt` (none when empty), then the end.
pub fn run(
    format: &Format,
    declared: &Declared,
    prompt: &str,
    text: &str,
    cuts: &[usize],
    end: &EngineFinish,
) -> Vec<Event> {
    let mut parser = Engine::new(format.clone(), declared.clone());
    let mut out = Events::new();
    if !prompt.is_empty() {
        let prompt = Input::Prompt {
            token_ids: &[],
            text: prompt,
        };
        parser.feed(prompt, &mut out).unwrap();
    }
    let mut from = 0;
    for &cut in cuts.iter().chain(std::iter::once(&text.len())) {
        if cut > from {
            let delta = Input::Delta {
                token_ids: &[],
                text: &text[from..cut],
                spans: &[],
            };
            parser.feed(delta, &mut out).unwrap();
            from = cut;
        }
    }
    parser
        .feed(
            Input::End {
                finish: end.clone(),
            },
            &mut out,
        )
        .unwrap();
    out.drain()
}

/// The places to cut `text` at, each plan named: whole, a character at a time, steps of three
/// and seven, once at every place (at most `every_split_up_to` bytes), once around every
/// occurrence of a marker, and thirty seeded plans of one to eight characters.
pub fn plans(text: &str, markers: &[&str], every_split_up_to: usize) -> Vec<(String, Vec<usize>)> {
    let starts: Vec<usize> = text.char_indices().map(|(at, _)| at).collect();
    let bytes_of = |lengths: Vec<usize>| -> Vec<usize> {
        let mut at = 0;
        let mut cuts = Vec::new();
        for length in lengths {
            at += length;
            if at < starts.len() {
                cuts.push(starts[at]);
            }
        }
        cuts
    };
    let mut plans = vec![("whole".to_owned(), Vec::new())];
    for feed in ["chars", "step:3", "step:7"] {
        plans.push((feed.to_owned(), bytes_of(chunk_lengths(text, feed))));
    }
    let mut once: Vec<usize> = Vec::new();
    if text.len() <= every_split_up_to {
        once.extend(starts.iter().skip(1));
    } else {
        for marker in markers {
            for (at, _) in text.match_indices(marker) {
                once.extend(
                    starts
                        .iter()
                        .copied()
                        .filter(|&cut| cut > 0 && cut + 1 >= at && cut <= at + marker.len() + 1),
                );
            }
        }
        once.sort_unstable();
        once.dedup();
    }
    plans.extend(
        once.into_iter()
            .map(|cut| (format!("cut at {cut}"), vec![cut])),
    );
    for seed in 1..=30u64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut cuts = Vec::new();
        let mut at = 0;
        while at < starts.len() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            at += 1 + (state >> 33) as usize % 8;
            if at < starts.len() {
                cuts.push(starts[at]);
            }
        }
        plans.push((format!("seeded plan {seed}"), cuts));
    }
    plans
}

/// What a client assembles from the events: the content, each reasoning region, dropped and
/// malformed text by reason, the calls with their arguments, and the finish.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Said {
    pub content: String,
    pub reasoning: Vec<String>,
    pub dropped: Vec<(DropReason, String)>,
    pub malformed: Vec<(MalformedReason, String)>,
    pub calls: Vec<(u32, String, String)>,
    pub arguments: Vec<(u32, String)>,
    pub ended: Vec<u32>,
    pub finish: Option<(FinishReason, u32, u32)>,
}

pub fn joined_by<R: Clone + PartialEq>(list: &mut Vec<(R, String)>, why: &R, text: &str) {
    match list.iter_mut().find(|(reason, _)| reason == why) {
        Some((_, joined)) => joined.push_str(text),
        None => list.push((why.clone(), text.to_owned())),
    }
}

impl Said {
    pub fn of(events: &[Event]) -> Self {
        let mut said = Self::default();
        let mut open = false;
        for event in events {
            match event {
                Event::Content(t) => said.content.push_str(&t.text),
                Event::ReasoningStart => {
                    said.reasoning.push(String::new());
                    open = true;
                }
                Event::Reasoning(t) => {
                    if !open {
                        said.reasoning.push(String::new());
                        open = true;
                    }
                    if let Some(region) = said.reasoning.last_mut() {
                        region.push_str(&t.text);
                    }
                }
                Event::ReasoningEnd => open = false,
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

/// The output bytes `event` accounts for.
pub fn bytes_of(event: &Event) -> &str {
    match event {
        Event::Content(t) | Event::Reasoning(t) => t.text.as_str(),
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => source.text.as_str(),
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
    }
}

/// Every byte of `text` in exactly one event, in order, `Finish` last; reasoning only inside a
/// region; a call's arguments a JSON prefix after every fragment, and only between its start and
/// its end.
pub fn check_contract(place: &str, text: &str, events: &[Event]) {
    let conserved: String = events.iter().map(bytes_of).collect();
    assert_eq!(conserved, text, "{place}: the bytes of the events");
    assert!(
        matches!(events.last(), Some(Event::Finish { .. })),
        "{place}: Finish is last"
    );
    let mut open = false;
    let mut started: Vec<u32> = Vec::new();
    let mut ended: Vec<u32> = Vec::new();
    let mut arguments: Vec<(u32, String)> = Vec::new();
    for event in events {
        match event {
            Event::ReasoningStart => {
                assert!(!open, "{place}: reasoning opened twice");
                open = true;
            }
            Event::ReasoningEnd => {
                assert!(open, "{place}: reasoning closed while closed");
                open = false;
            }
            Event::Reasoning(_) => assert!(open, "{place}: reasoning outside a region"),
            Event::ToolCallStart { index, .. } => {
                assert!(!started.contains(index), "{place}: a second start");
                started.push(*index);
                arguments.push((*index, String::new()));
            }
            Event::ToolCallArguments { index, json, .. } => {
                assert!(
                    started.contains(index) && !ended.contains(index),
                    "{place}: arguments outside their call"
                );
                let so_far = &mut arguments.iter_mut().find(|(i, _)| i == index).unwrap().1;
                so_far.push_str(json);
                let prefix = matches!(
                    PartialJson::default().parse(so_far, true),
                    Ok((_, consumed)) if consumed == so_far.len()
                );
                assert!(
                    prefix,
                    "{place}: the arguments {so_far:?} are no JSON prefix"
                );
            }
            Event::ToolCallEnd { index, .. } => {
                assert!(started.contains(index), "{place}: an end before the start");
                ended.push(*index);
            }
            Event::Finish { tool_calls, .. } => {
                assert_eq!(
                    *tool_calls as usize,
                    started.len(),
                    "{place}: the finish counts the calls"
                );
            }
            Event::Content(_) | Event::Dropped { .. } | Event::Malformed { .. } => {}
        }
    }
    assert!(!open, "{place}: reasoning left open");
}
