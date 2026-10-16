//! Every table under its own markers, scrambled. Each table of the registry, its variants without
//! calls and without the thought (the ones the gateway makes of it), with tools declared and
//! without, reads seeded outputs built from its own terminals and tags, whole and cut short, tool
//! names, argument fragments in every call syntax, whole surrogate escapes, text in several
//! scripts and stray delimiters,
//! with seeded corruptions on top (a span deleted or doubled, a piece inserted, the tail cut, a
//! character swapped), after an empty prompt or one made of the table's own markers, ended by
//! every engine finish. Nothing a model could write may panic the parser or the adapters, and
//! every output keeps the contract whatever it says: every byte in exactly one event with `Finish`
//! last, a well-formed stream, the same content, reasoning, calls and finish however the output
//! is cut into deltas, with or without token spans, and with spans every token counted once, in
//! the event that carries its first byte.
//!
//! A surrogate escape broken inside a top-level string is left out of the outputs on purpose:
//! a string-valued arguments streams as written, and the JSON prefix judge refuses such a string
//! whole, so there the pieces say the string's bytes before the break where the whole says
//! nothing; the one stated exception to the chunking rule, held by the assemblers' own tests.
//! No piece holds a broken escape, and an output in which a concatenation (a lone `\` before an
//! escape) or a corruption (a cut inside an escape) built one anyway is skipped under its seed (a
//! few per cent of the outputs; the count is on the diagnostic line), so a long audit never
//! reports the known exception as a failure.
//!
//! `SYMPHONY_SCRAMBLE_CASES=n` sets the outputs per table, variant and declaration (one hundred
//! by default; a longer run for an audit), `SYMPHONY_SCRAMBLE_SEED=n` the first seed, so a run can
//! be repeated or continued, and `SYMPHONY_SCRAMBLE_REPORT=file` writes every failure to a file
//! where the message shows the first twenty. A failure names the table, the variant, the seed and
//! the output, so that it replays.

#[expect(
    dead_code,
    reason = "the integration tests share these helpers, and this one takes the bytes of an event alone"
)]
mod common;

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::atomic::{AtomicUsize, Ordering},
};

use common::bytes_of;
use openai_protocol::common::{Function, Tool};
use serde_json::json as value;
use symphony::{
    adapt, formats::Table, json::PartialJson, Declared, DropReason, Engine, EngineFinish, Event,
    Events, FinishReason, Format, Input, MalformedReason, ParseError, Parser, TokenSpan,
};

const CASES_ENV: &str = "SYMPHONY_SCRAMBLE_CASES";
const SEED_ENV: &str = "SYMPHONY_SCRAMBLE_SEED";
/// A file every failure is written to, for an audit that wants more than the first [`SHOWN`].
const REPORT_ENV: &str = "SYMPHONY_SCRAMBLE_REPORT";
const DEFAULT_CASES: u64 = 100;
/// How many failures the message lists.
const SHOWN: usize = 20;

/// The fragments every syntax has a use for: tool names, the arguments in each spelling, the
/// pieces of a value, the delimiters of every markup, whitespace, text in several scripts, and
/// the markers of other families, which a table must read as text.
const COMMON: &[&str] = &[
    "get_weather",
    "play",
    "spotify_play",
    "add_integers",
    "f",
    "nosuch",
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}",
    "{\"name\": \"play\", \"arguments\": {\"artist\": \"Taylor Swift\", \"n\": 20}}",
    "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}",
    "{\"name\": \"f\", \"arguments\": {}}",
    "[{\"name\": \"f\", \"arguments\": {}}]",
    "{\"city\": \"Paris\"",
    "\"days\": 3",
    "{",
    "}",
    "[",
    "]",
    ",",
    ":",
    "\"",
    "\\",
    "\\\"",
    "\"\\ud83c\\ude00\"",
    "\\ud83c\\ude00",
    "\\u00e9",
    "city",
    "Paris",
    "days",
    "3",
    "true",
    "null",
    "-1.5e3",
    "\"quoted\"",
    "name=\"get_weather\"",
    " name=\"city\"",
    " string=\"true\"",
    " string=\"false\"",
    "get_weather(city=\"Paris\", days=3)",
    "[get_weather(city=\"Paris\")]",
    "play(artist='Swift')",
    "call:play{artist:<|\"|>Taylor Swift<|\"|>,n:20}",
    "<",
    ">",
    "/",
    "<>",
    "</",
    "/>",
    "=",
    "\n",
    "\n\n",
    " ",
    "\t",
    "  \n ",
    "The user asks about Paris.",
    "Paris is sunny.",
    "héllo wörld",
    "東京の天気",
    "🌧️ rain",
    "a",
    "x y z",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<|im_end|>",
    "<|eot_id|>",
];

/// The characters a corruption swaps in.
const SWAPPED_IN: &[char] = &['<', '>', '"', '\n', ' ', 'x', '}', ']', 'é', '🌧'];

/// The tools the outputs name, with every kind of parameter the syntaxes type by.
fn tools() -> Vec<Tool> {
    let tool = |name: &str, parameters: serde_json::Value| Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters,
            strict: None,
            extra: Default::default(),
        },
    };
    vec![
        tool(
            "get_weather",
            value!({"type": "object", "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "metric": {"type": "boolean"},
                "options": {"type": "object", "properties": {"unit": {"type": "string"}}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "note": {"type": ["string", "null"]},
            }}),
        ),
        tool(
            "play",
            value!({"type": "object", "properties": {
                "artist": {"type": "string"},
                "n": {"type": "integer"},
            }}),
        ),
        tool(
            "spotify_play",
            value!({"type": "object", "properties": {
                "artist": {"type": "string"},
                "duration": {"type": "integer"},
                "shuffle": {"type": "boolean"},
                "tags": {"type": "array"},
            }}),
        ),
        tool(
            "add_integers",
            value!({"type": "object", "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            }}),
        ),
        tool("f", value!({"type": "object", "properties": {}})),
    ]
}

/// A seeded generator, the same sequence for a seed every time.
struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// A number below `n`; zero when `n` is.
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() as usize) % n
        }
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

/// The variants of a table the gateway reads with: the table, and the table as a request without
/// tools or without separate reasoning asks for it.
#[derive(Clone, Copy, Debug)]
enum Variant {
    Whole,
    WithoutCalls,
    WithoutReasoning,
}

impl Variant {
    const ALL: [Variant; 3] = [Self::Whole, Self::WithoutCalls, Self::WithoutReasoning];

    fn of(self, table: Table) -> Format {
        let format = table.format();
        match self {
            Self::Whole => format,
            Self::WithoutCalls => format.without_calls(),
            Self::WithoutReasoning => format.without_reasoning(),
        }
    }
}

/// The pieces a table's outputs are built from: its own terminals and tags, each also cut short
/// at both ends and in the middle, so that half a marker and a marker missing its first
/// character are as common as a whole one, and the fragments every syntax has a use for.
fn vocabulary(format: &Format) -> Vec<String> {
    let mut markers: Vec<String> = format.terminal_texts().map(str::to_string).collect();
    if let Some(syntax) = format.call_syntax() {
        markers.extend(syntax.tag_texts().into_iter().map(str::to_string));
    }
    let mut pieces = markers.clone();
    for marker in &markers {
        let chars: Vec<char> = marker.chars().collect();
        if chars.len() >= 2 {
            pieces.push(chars[..chars.len() / 2].iter().collect());
            pieces.push(chars[..chars.len() - 1].iter().collect());
            pieces.push(chars[1..].iter().collect());
        }
    }
    pieces.extend(COMMON.iter().map(|piece| piece.to_string()));
    pieces
}

/// A seeded output of `vocabulary`: one to twenty-four pieces, then up to three corruptions.
fn output(random: &mut Random, vocabulary: &[String]) -> String {
    let pieces = 1 + random.below(24);
    let mut text = String::new();
    for _ in 0..pieces {
        text.push_str(random.pick(vocabulary));
    }
    for _ in 0..random.below(4) {
        text = corrupted(random, &text, vocabulary);
    }
    text
}

/// `text` with one seeded corruption, on character boundaries: a span deleted, a span doubled,
/// a piece inserted, the tail cut, or a span swapped for one character.
fn corrupted(random: &mut Random, text: &str, vocabulary: &[String]) -> String {
    let bounds: Vec<usize> = text
        .char_indices()
        .map(|(at, _)| at)
        .chain(std::iter::once(text.len()))
        .collect();
    let (from, to) = {
        let a = *random.pick(&bounds);
        let b = *random.pick(&bounds);
        (a.min(b), a.max(b))
    };
    match random.below(5) {
        0 => format!("{}{}", &text[..from], &text[to..]),
        1 => format!("{}{}{}", &text[..to], &text[from..to], &text[to..]),
        2 => format!(
            "{}{}{}",
            &text[..from],
            random.pick(vocabulary),
            &text[from..]
        ),
        3 => text[..to].to_string(),
        _ => format!(
            "{}{}{}",
            &text[..from],
            random.pick(SWAPPED_IN),
            &text[to..]
        ),
    }
}

/// A seeded prompt: nothing, most of the time, else one or two of the table's own markers, so
/// that the replay of the prompt starts the table in each of its states.
fn prompt(random: &mut Random, format: &Format) -> String {
    let markers: Vec<&str> = format.terminal_texts().collect();
    if markers.is_empty() || random.below(2) == 0 {
        return String::new();
    }
    let mut text = (*random.pick(&markers)).to_string();
    if random.below(2) == 0 {
        text.push_str(random.pick(&markers));
    }
    text
}

fn finish(random: &mut Random) -> EngineFinish {
    match random.below(4) {
        0 => EngineFinish::Length,
        1 => EngineFinish::Abort,
        2 => EngineFinish::Other("other".to_string()),
        _ => EngineFinish::Stop,
    }
}

/// A seeded plan of cuts into `text`, each piece one to eight characters.
fn plan(random: &mut Random, text: &str) -> Vec<usize> {
    let starts: Vec<usize> = text.char_indices().map(|(at, _)| at).collect();
    let mut cuts = Vec::new();
    let mut at = 0;
    while at < starts.len() {
        at += 1 + random.below(8);
        if at < starts.len() {
            cuts.push(starts[at]);
        }
    }
    cuts
}

/// The spans of a delta's `text` as a decoder would release it: a token per one to three
/// characters, ids counting on from `next_id`.
fn spans(text: &str, next_id: &mut u32) -> Vec<TokenSpan> {
    let mut spans = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while chars.peek().is_some() {
        let mut end = start;
        for _ in 0..1 + (*next_id as usize % 3) {
            match chars.next() {
                Some((at, c)) => end = at + c.len_utf8(),
                None => break,
            }
        }
        spans.push(TokenSpan {
            token_id: *next_id,
            start,
            end,
            continued: false,
        });
        *next_id += 1;
        start = end;
    }
    spans
}

/// Feed `prompt_text`, then `text` cut at `cuts`, then the end, and return the events with the
/// number of token spans sent. With `with_spans` every delta carries synthetic token spans; with
/// `empty_between` an empty delta comes before every piece and after the last.
fn replay(
    parser: &mut dyn Parser,
    prompt_text: &str,
    text: &str,
    cuts: &[usize],
    finish: &EngineFinish,
    with_spans: bool,
    empty_between: bool,
) -> Result<(Vec<Event>, u32), ParseError> {
    let mut out = Events::new();
    parser.feed(
        Input::Prompt {
            token_ids: &[],
            text: prompt_text,
        },
        &mut out,
    )?;
    let mut from = 0;
    let mut next_id = 1;
    let mut sent = 0;
    for &cut in cuts.iter().chain(std::iter::once(&text.len())) {
        if cut > from {
            if empty_between {
                parser.feed(
                    Input::Delta {
                        token_ids: &[],
                        text: "",
                        spans: &[],
                    },
                    &mut out,
                )?;
            }
            let piece = &text[from..cut];
            if with_spans {
                let spans = spans(piece, &mut next_id);
                sent += spans.len() as u32;
                let ids: Vec<u32> = spans.iter().map(|span| span.token_id).collect();
                parser.feed(
                    Input::Delta {
                        token_ids: &ids,
                        text: piece,
                        spans: &spans,
                    },
                    &mut out,
                )?;
            } else {
                parser.feed(
                    Input::Delta {
                        token_ids: &[],
                        text: piece,
                        spans: &[],
                    },
                    &mut out,
                )?;
            }
            from = cut;
        }
    }
    if empty_between {
        parser.feed(
            Input::Delta {
                token_ids: &[],
                text: "",
                spans: &[],
            },
            &mut out,
        )?;
    }
    parser.feed(
        Input::End {
            finish: finish.clone(),
        },
        &mut out,
    )?;
    Ok((out.drain(), sent))
}

/// Token identity on a replay with spans: every token counted once, in the event that carries
/// its first byte, so the counts over all events add up to the tokens sent and no event that
/// carries text is uncounted; the first breach, if any.
fn check_tokens(events: &[Event], sent: u32) -> Result<(), String> {
    let mut counted = 0;
    for event in events {
        let text = match event {
            Event::Content(text) | Event::Reasoning(text) => text,
            Event::Dropped { text, .. } | Event::Malformed { text, .. } => text,
            Event::ToolCallStart { source, .. }
            | Event::ToolCallArguments { source, .. }
            | Event::ToolCallEnd { source, .. } => source,
            Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => continue,
        };
        match text.tokens {
            Some(tokens) => counted += tokens,
            None if text.text.is_empty() => {}
            None => return Err(format!("an uncounted event: {event:?}")),
        }
    }
    if counted != sent {
        return Err(format!("{counted} tokens counted of {sent} sent"));
    }
    Ok(())
}

/// What the parser has said, as the streams a client would assemble: the token counts aside,
/// which the spans decide and the contract suite checks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Said {
    content: String,
    reasoning: Vec<String>,
    dropped: Vec<(DropReason, String)>,
    malformed: Vec<(MalformedReason, String)>,
    calls: Vec<(u32, String, String)>,
    arguments: Vec<(u32, String)>,
    ended: Vec<u32>,
    finish: Option<(FinishReason, u32)>,
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
                    reason, tool_calls, ..
                } => said.finish = Some((reason.clone(), *tool_calls)),
            }
        }
        said
    }
}

fn joined_by<R: Clone + PartialEq>(list: &mut Vec<(R, String)>, why: &R, text: &str) {
    match list.iter_mut().find(|(reason, _)| reason == why) {
        Some((_, joined)) => joined.push_str(text),
        None => list.push((why.clone(), text.to_string())),
    }
}

fn valid_json_prefix(text: &str) -> bool {
    matches!(PartialJson::default().parse(text, true), Ok((_, consumed)) if consumed == text.len())
}

/// The contract one replay keeps: conservation, `Finish` last and alone, and a well-formed
/// stream; the first breach, if any.
fn check_contract(text: &str, events: &[Event]) -> Result<(), String> {
    let conserved: String = events.iter().map(bytes_of).collect();
    if conserved != text {
        return Err(format!("the events carry {conserved:?}"));
    }
    if !matches!(events.last(), Some(Event::Finish { .. })) {
        return Err("Finish is not last".to_string());
    }
    let mut started: Vec<u32> = Vec::new();
    let mut ended: Vec<u32> = Vec::new();
    let mut arguments: Vec<(u32, String)> = Vec::new();
    let mut reasoning_open = false;
    let mut finished = false;
    for (at, event) in events.iter().enumerate() {
        let place = |what: &str| format!("event {at} {event:?}: {what}");
        if finished {
            return Err(place("after Finish"));
        }
        match event {
            Event::ToolCallStart { index, .. } => {
                if started.contains(index) {
                    return Err(place("a second start"));
                }
                started.push(*index);
                arguments.push((*index, String::new()));
            }
            Event::ToolCallArguments { index, json, .. } => {
                if !started.contains(index) {
                    return Err(place("arguments before the call's start"));
                }
                if ended.contains(index) {
                    return Err(place("arguments after the call's end"));
                }
                if let Some((_, so_far)) = arguments.iter_mut().find(|(i, _)| i == index) {
                    so_far.push_str(json);
                    if !valid_json_prefix(so_far) {
                        return Err(place(&format!(
                            "the arguments so far, {so_far:?}, are no valid JSON prefix"
                        )));
                    }
                }
            }
            Event::ToolCallEnd { index, .. } => {
                if !started.contains(index) {
                    return Err(place("an end before the call's start"));
                }
                if ended.contains(index) {
                    return Err(place("a second end"));
                }
                ended.push(*index);
            }
            Event::ReasoningStart => {
                if reasoning_open {
                    return Err(place("reasoning already open"));
                }
                reasoning_open = true;
            }
            Event::Reasoning(_) => {
                if !reasoning_open {
                    return Err(place("reasoning outside a region"));
                }
            }
            Event::ReasoningEnd => {
                if !reasoning_open {
                    return Err(place("no reasoning open"));
                }
                reasoning_open = false;
            }
            Event::Finish { tool_calls, .. } => {
                if *tool_calls as usize != started.len() {
                    return Err(place("the finish does not count the calls that started"));
                }
                finished = true;
            }
            Event::Content(_) | Event::Dropped { .. } | Event::Malformed { .. } => {}
        }
    }
    if reasoning_open {
        return Err("reasoning left open".to_string());
    }
    if started.len() != ended.len() {
        return Err("a call without an end".to_string());
    }
    Ok(())
}

/// The adapters over the events, whole and streamed, each of which the gateway runs on every
/// response: they must not panic, whatever the events say.
fn adapt_all(events: &[Event]) {
    let _ = adapt::chat::message(0, events);
    for event in events {
        let _ = adapt::chat::delta(0, event);
    }
    let _ = adapt::messages::output(events);
    let mut stream = adapt::messages::Stream::new();
    let mut out = Vec::new();
    for event in events {
        let _ = stream.feed(event, &mut out);
    }
    stream.close(&mut out);
    let _ = adapt::responses::output("resp_scramble", events);
    let mut stream = adapt::responses::Stream::new("resp_scramble");
    let mut out = Vec::new();
    for event in events {
        let _ = stream.feed(event, &mut out);
    }
    let _ = stream.close(&mut out);
}

/// One case: the table's `format` with `declared`, the output after the prompt, ended by
/// `finish`, replayed whole and on every plan; the first breach, if any.
fn check(
    format: &Format,
    declared: &Declared,
    prompt_text: &str,
    text: &str,
    finish: &EngineFinish,
    random: &mut Random,
) -> Result<(), String> {
    let new = || Engine::new(format.clone(), declared.clone());
    let (whole, _) = replay(&mut new(), prompt_text, text, &[], finish, false, false)
        .map_err(|e| format!("whole: {e}"))?;
    check_contract(text, &whole).map_err(|e| format!("whole: {e}"))?;
    adapt_all(&whole);
    let expected = Said::of(&whole);
    let per_char: Vec<usize> = text.char_indices().map(|(at, _)| at).skip(1).collect();
    let mut plans = vec![
        ("character by character", per_char.clone(), false, false),
        ("character by character with spans", per_char, true, false),
    ];
    for at in 1..=3 {
        plans.push(("seeded plan", plan(random, text), false, false));
        if at == 1 {
            plans.push(("seeded plan with spans", plans[2].1.clone(), true, false));
            plans.push((
                "seeded plan with empty deltas",
                plans[2].1.clone(),
                false,
                true,
            ));
        }
    }
    for (name, cuts, with_spans, empty_between) in plans {
        let (events, sent) = replay(
            &mut new(),
            prompt_text,
            text,
            &cuts,
            finish,
            with_spans,
            empty_between,
        )
        .map_err(|e| format!("{name} {cuts:?}: {e}"))?;
        check_contract(text, &events).map_err(|e| format!("{name} {cuts:?}: {e}"))?;
        if with_spans {
            check_tokens(&events, sent).map_err(|e| format!("{name} {cuts:?}: {e}"))?;
        }
        adapt_all(&events);
        let said = Said::of(&events);
        if said != expected {
            return Err(format!(
                "{name} {cuts:?} says something else:\n  whole: {expected:?}\n  cut:   {said:?}"
            ));
        }
    }
    Ok(())
}

fn cases() -> u64 {
    std::env::var(CASES_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_CASES)
}

fn first_seed() -> u64 {
    std::env::var(SEED_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// The message of a panic, as `panic!` formatted it.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "a panic without a message".to_string())
}

/// Every table, variant and declaration: one unit of work for a thread.
fn subjects() -> Vec<(Table, Variant, bool)> {
    let mut subjects = Vec::new();
    for table in Table::ALL {
        for variant in Variant::ALL {
            for with_tools in [true, false] {
                subjects.push((table, variant, with_tools));
            }
        }
    }
    subjects
}

/// The scrambles of one subject, each in its own seed: the failures, each naming what replays it.
/// Outputs skipped for holding a broken surrogate escape (the stated exception), for the
/// diagnostic line.
static SKIPPED: AtomicUsize = AtomicUsize::new(0);

/// Whether `text` holds a JSON surrogate escape that is broken: a high surrogate (`\ud800` to
/// `\udbff`) not followed by a low one (`\udc00` to `\udfff`), or a low one not preceded by a
/// high one. An escape begins at a backslash that is not itself escaped; an escape cut short
/// (the tail cut mid-escape) is not a broken one, as whole and pieces hold it alike.
fn holds_a_broken_surrogate_escape(text: &str) -> bool {
    let bytes = text.as_bytes();
    let escape_at = |at: usize| -> Option<u32> {
        // `\uXXXX` at `at`, the backslash unescaped: its code unit.
        if bytes.get(at) != Some(&b'\\') || bytes.get(at + 1) != Some(&b'u') {
            return None;
        }
        let preceding = bytes[..at]
            .iter()
            .rev()
            .take_while(|b| **b == b'\\')
            .count();
        if preceding % 2 == 1 {
            return None;
        }
        let digits = bytes.get(at + 2..at + 6)?;
        let digits = std::str::from_utf8(digits).ok()?;
        u32::from_str_radix(digits, 16).ok()
    };
    let mut at = 0;
    let mut after_high = false;
    while at < bytes.len() {
        match escape_at(at) {
            Some(unit) if (0xD800..0xDC00).contains(&unit) => {
                if after_high {
                    return true;
                }
                after_high = true;
                at += 6;
            }
            Some(unit) if (0xDC00..0xE000).contains(&unit) => {
                if !after_high {
                    return true;
                }
                after_high = false;
                at += 6;
            }
            Some(_) => {
                if after_high {
                    return true;
                }
                at += 6;
            }
            None => {
                if after_high {
                    return true;
                }
                at += 1;
            }
        }
    }
    false
}

#[test]
fn a_broken_surrogate_escape_is_told_from_a_whole_one() {
    for whole in [
        "",
        "plain text",
        r"\ud83c\ude00",
        r#"{"e": "\ud83c\ude00"}"#,
        r"\u00e9 \n \\",
        r"\ud83",
        r"ab\udc0",
        // A high escape the text ends on: cut short, held alike by whole and pieces.
        r"\ud83c",
    ] {
        assert!(!holds_a_broken_surrogate_escape(whole), "{whole:?}");
    }
    for broken in [
        r"\udc00",
        r#""ab\udc00""#,
        r"\\ud83c\ude00",
        r#"\ud83c""#,
        r"\ud83c\x",
        r"\ud83c\ud83c\ude00",
        r"\ud83c ",
    ] {
        assert!(holds_a_broken_surrogate_escape(broken), "{broken:?}");
    }
}

fn scramble(
    table: Table,
    variant: Variant,
    with_tools: bool,
    cases: u64,
    first: u64,
) -> Vec<String> {
    let format = variant.of(table);
    let declared = if with_tools {
        Declared::of(&tools())
    } else {
        Declared::default()
    };
    let vocabulary = vocabulary(&format);
    let mut failures = Vec::new();
    for seed in first..first + cases {
        let mut random = Random::new(
            seed ^ ((table as u64) << 40)
                ^ ((variant as u64) << 48)
                ^ (u64::from(with_tools) << 56),
        );
        let prompt_text = prompt(&mut random, &format);
        let text = output(&mut random, &vocabulary);
        // The stated exception (see the module doc) stays out of the outputs.
        if holds_a_broken_surrogate_escape(&text) {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let finish = finish(&mut random);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            check(
                &format,
                &declared,
                &prompt_text,
                &text,
                &finish,
                &mut random,
            )
        }));
        let failure = match outcome {
            Ok(Ok(())) => continue,
            Ok(Err(breach)) => breach,
            Err(payload) => format!("PANIC: {}", panic_message(payload)),
        };
        failures.push(format!(
            "{} ({variant:?}, {}), seed {seed}, prompt {prompt_text:?}, finish {finish:?}, \
             output {text:?}:\n  {failure}",
            table.name(),
            if with_tools {
                "tools declared"
            } else {
                "no tools"
            },
        ));
    }
    failures
}

#[test]
#[expect(
    clippy::print_stdout,
    reason = "the count of scrambles read is the test's diagnostic output"
)]
fn every_table_reads_every_scramble_without_a_panic_and_keeps_the_contract() {
    let cases = cases();
    let first = first_seed();
    let subjects = subjects();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(8);
    let failures: Vec<String> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let (next, subjects) = (&next, &subjects);
                scope.spawn(move || {
                    let mut failures = Vec::new();
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(table, variant, with_tools)) = subjects.get(at) else {
                            break failures;
                        };
                        failures.extend(scramble(table, variant, with_tools, cases, first));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| {
                worker
                    .join()
                    .unwrap_or_else(|payload| vec![panic_message(payload)])
            })
            .collect()
    });
    let skipped = SKIPPED.load(Ordering::Relaxed) as u64;
    println!(
        "{} scrambles read: {} tables x {} variants x 2 declarations x {cases} seeds from {first}, \
         {skipped} skipped for a broken surrogate escape (the stated exception)",
        subjects.len() as u64 * cases - skipped,
        Table::ALL.len(),
        Variant::ALL.len()
    );
    if let Some(path) = std::env::var_os(REPORT_ENV) {
        let report = failures.join("\n");
        std::fs::write(&path, report).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        println!("every failure: {}", path.display());
    }
    assert!(
        failures.is_empty(),
        "{} of {} scrambles broke the contract or panicked; the first {}:\n{}",
        failures.len(),
        subjects.len() as u64 * cases,
        failures.len().min(SHOWN),
        failures
            .iter()
            .take(SHOWN)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn a_corruption_keeps_the_text_on_character_boundaries_and_a_plan_cuts_there() {
    let vocabulary = vec!["<a>".to_string(), "é".to_string()];
    let mut random = Random::new(3);
    for _ in 0..500 {
        let text = output(&mut random, &vocabulary);
        for cut in plan(&mut random, &text) {
            assert!(text.is_char_boundary(cut), "{text:?} cut at {cut}");
        }
        let mut next_id = 1;
        let spans = spans(&text, &mut next_id);
        assert!(TokenSpan::partitions(&text, &spans), "{text:?}: {spans:?}");
    }
}
