//! Compiled tables read against transformers' own reading of recorded sessions.
//!
//! - A compiled table reads every recorded session of its template as transformers 5.18.0 does
//!   (`parse_response(output + stop, template, prefix=prompt, tools=tools)` for the public
//!   templates of `fixtures/compiled/`, and the crate's own recordings in `fixtures/hf/`), whole
//!   and at every chunking: whole, a character at a time, in steps of three and seven, cut once at
//!   every place (for an output of at most 2 KiB) and around every marker, and thirty seeded
//!   plans. A session whose message differs is listed in `to_symphony_differs.tsv` with the codes
//!   of the documented differences that explain it, each checked by undoing it; a difference none
//!   explains fails, and so does a listed session that reads as transformers does. A session that
//!   gives a marker back (an open whose header does not match, a close without its trailer) reads
//!   exactly as transformers does, with no difference listed.
//! - At every chunking the table keeps symphony's contract: every byte of the output in exactly
//!   one event, in order; what it says the same whatever the chunking; a call's arguments a JSON
//!   prefix after every fragment; every token counted once, in the event that carries its first
//!   byte.
//! - Where transformers' own stream differs from its whole-text reading (`streamed` records), the
//!   table reads the whole text's message at that chunking too.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "a test: a bad fixture or table should panic"
)]

mod common;
mod replay;

use std::{
    collections::BTreeMap,
    path::Path,
    sync::OnceLock,
    time::{Duration, Instant},
};

use common::{fixtures_dir, read_json, read_jsonl, untag};
use openai_protocol::common::Tool;
use replay::{check_contract, compiled, compiled_dir, pinned, plans, rows, run, Said};
use serde_json::Value;
use smg_response_template::{
    load_response_template, Event as RegionEvent, ResponseParser, ResponseTemplate,
};
use symphony::{
    adapt, Declared, Engine, EngineFinish, Event, Events, Format, Input, MalformedReason, Parser,
    TokenSpan,
};

/// The compiled tables of a fixture directory by template id, each with the literal text of its
/// template's delimiters, compiled once.
type Tables = BTreeMap<String, (Format, Vec<String>, Option<ResponseTemplate>)>;

fn tables(dir: &Path, with_stops: bool) -> Tables {
    rows(dir)
        .into_iter()
        .filter_map(|mut row| {
            if !with_stops {
                row.stops.clear();
            }
            match compiled(&row)? {
                Ok(compiled) => {
                    let runs = load_response_template(&row.spec).ok();
                    Some((
                        row.id.clone(),
                        (compiled.format, literal_texts(&row.spec), runs),
                    ))
                }
                Err(_) => None,
            }
        })
        .collect()
}

fn public_tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| tables(&compiled_dir(), true))
}

fn fixture_tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| tables(&fixtures_dir(), false))
}

/// The tools a session names, as the API declares them, and as JSON for the template's reader.
fn tools(dir: &Path, session: &Value) -> (Declared, Vec<Value>) {
    let Some(set) = session.get("tools").and_then(Value::as_str) else {
        return (Declared::default(), Vec::new());
    };
    let listed = untag(&read_json(dir, "tools.json")[set]);
    let listed = listed.as_array().cloned().unwrap_or_default();
    // A set may hold a bare function, which transformers reads and the API does not.
    let api: Vec<Tool> = listed
        .iter()
        .filter(|tool| tool.get("type").is_some())
        .map(|tool| serde_json::from_value(tool.clone()).unwrap())
        .collect();
    (Declared::of(&api), listed)
}

/// A synthetic tokenization of `text`, one to three characters a token, the same every time.
fn token_boundaries(text: &str) -> Vec<usize> {
    let chars: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let mut boundaries = vec![0];
    let mut at = 0;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    while at < chars.len() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        at += 1 + (state % 3) as usize;
        if at < chars.len() {
            boundaries.push(chars[at]);
        }
    }
    boundaries.push(text.len());
    boundaries.dedup();
    boundaries
}

/// Every token counted once, in the event that carries its first byte, at the cuts given.
fn check_tokens(
    place: &str,
    format: &Format,
    declared: &Declared,
    prompt: &str,
    text: &str,
    cuts: &[usize],
) {
    let boundaries = token_boundaries(text);
    let tokens: Vec<(usize, usize)> = boundaries.windows(2).map(|w| (w[0], w[1])).collect();
    let mut parser = Engine::new(format.clone(), declared.clone());
    let mut out = Events::new();
    if !prompt.is_empty() {
        let input = Input::Prompt {
            token_ids: &[],
            text: prompt,
        };
        parser.feed(input, &mut out).unwrap();
    }
    let mut from = 0;
    for &cut in cuts.iter().chain(std::iter::once(&text.len())) {
        if cut > from {
            let spans: Vec<TokenSpan> = tokens
                .iter()
                .filter(|&&(start, end)| end > from && start < cut)
                .map(|&(start, end)| TokenSpan {
                    token_id: 0,
                    start: start.max(from) - from,
                    end: end.min(cut) - from,
                    continued: start < from,
                })
                .collect();
            let delta = Input::Delta {
                token_ids: &[],
                text: &text[from..cut],
                spans: &spans,
            };
            parser.feed(delta, &mut out).unwrap();
            from = cut;
        }
    }
    let end = Input::End {
        finish: EngineFinish::Stop,
    };
    parser.feed(end, &mut out).unwrap();
    let starts: Vec<usize> = boundaries[..boundaries.len() - 1].to_vec();
    let mut at = 0;
    let mut total = 0;
    for event in out.drain() {
        let (counted, length) = match &event {
            Event::Content(t) | Event::Reasoning(t) => (t.tokens, t.text.len()),
            Event::Dropped { text, .. } | Event::Malformed { text, .. } => {
                (text.tokens, text.text.len())
            }
            Event::ToolCallStart { source, .. }
            | Event::ToolCallArguments { source, .. }
            | Event::ToolCallEnd { source, .. } => (source.tokens, source.text.len()),
            Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => continue,
        };
        let expected = starts
            .iter()
            .filter(|&&start| start >= at && start < at + length)
            .count();
        let counted = counted.unwrap_or_else(|| panic!("{place}: uncounted {event:?}"));
        assert_eq!(
            counted as usize, expected,
            "{place}: {event:?} at byte {at}"
        );
        at += length;
        total += expected;
    }
    assert_eq!(total, starts.len(), "{place}: every token counted once");
}

/// A message as both readers give it: reasoning, content, and each call's name and arguments.
#[derive(Clone, Debug, PartialEq)]
struct Message {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
}

/// The message a client folds from the events, with every call the stream cut closed: its
/// arguments' object closed and the text the cut gave back, which no parameter took, left out.
/// `None` when the stream cut no call.
fn closed(events: &[Event]) -> Option<Message> {
    let kept: Vec<Event> = events
        .iter()
        .filter(|event| {
            !matches!(
                event,
                Event::Malformed {
                    why: MalformedReason::UnterminatedRegion,
                    ..
                }
            )
        })
        .cloned()
        .collect();
    let mut message = ours(&kept);
    let mut cut = false;
    for (_, arguments) in &mut message.calls {
        let Value::String(text) = arguments else {
            continue;
        };
        let whole = if text.is_empty() {
            "{}".to_owned()
        } else {
            format!("{text}}}")
        };
        if let Ok(value) = serde_json::from_str::<Value>(&whole) {
            *arguments = value;
            cut = true;
        }
    }
    cut.then_some(message)
}

fn ours(events: &[Event]) -> Message {
    let message = adapt::chat::message(0, events).message;
    Message {
        reasoning: message.reasoning_content.unwrap_or_default(),
        content: message.content.unwrap_or_default(),
        calls: message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|call| {
                let arguments = call.function.arguments.unwrap_or_default();
                let arguments =
                    serde_json::from_str(&arguments).unwrap_or(Value::String(arguments));
                (call.function.name, arguments)
            })
            .collect(),
    }
}

fn theirs(message: &Value) -> Message {
    let text = |key: &str| {
        message
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    Message {
        reasoning: text("thinking") + &text("reasoning_content"),
        content: text("content"),
        calls: message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|call| {
                let function = &call["function"];
                (
                    function["name"].as_str().unwrap_or_default().to_owned(),
                    function["arguments"].clone(),
                )
            })
            .collect(),
    }
}

/// A session to replay: its id, prompt, output (without the stop text), how it ended, its tools,
/// transformers' message (`None` where transformers raised), and what it covers.
#[derive(Clone)]
struct Session {
    id: String,
    template: String,
    /// What the session was recorded to cover (`give-back`, `cut`, ...).
    covers: Vec<String>,
    prompt: String,
    output: String,
    /// The stop text transformers was fed after the output, which the stream does not show.
    stop: String,
    end: EngineFinish,
    declared: Declared,
    /// The tools as transformers read them.
    tools: Vec<Value>,
    message: Option<Value>,
    streamed: Vec<Value>,
}

/// The sessions of `fixtures/compiled/`: the output as the stop decoder hands it over.
fn public_sessions() -> Vec<Session> {
    let dir = compiled_dir();
    read_jsonl(&dir, "sessions")
        .into_iter()
        .map(|row| Session {
            id: row["id"].as_str().unwrap().to_owned(),
            template: row["template"].as_str().unwrap().to_owned(),
            covers: row["covers"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
            prompt: row["prompt"].as_str().unwrap().to_owned(),
            output: row["output"].as_str().unwrap().to_owned(),
            stop: row["stop"].as_str().unwrap_or_default().to_owned(),
            end: if row["end"] == "stop" {
                EngineFinish::Stop
            } else {
                EngineFinish::Length
            },
            declared: tools(&dir, &row).0,
            tools: tools(&dir, &row).1,
            message: row["transformers"].get("message").map(untag),
            streamed: row
                .get("streamed")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
        })
        .collect()
}

/// The sessions of `fixtures/hf/`: the whole text, which transformers read with no stop.
fn fixture_sessions() -> Vec<Session> {
    let dir = fixtures_dir();
    ["sessions", "random"]
        .iter()
        .flat_map(|name| read_jsonl(&dir, name))
        .map(|row| {
            let unary = &row["unary"]["final"];
            Session {
                id: row["id"].as_str().unwrap().to_owned(),
                template: row["template"].as_str().unwrap().to_owned(),
                covers: Vec::new(),
                prompt: row["prefix"].as_str().unwrap_or_default().to_owned(),
                output: row["text"].as_str().unwrap().to_owned(),
                stop: String::new(),
                end: EngineFinish::Stop,
                declared: tools(&dir, &row).0,
                tools: tools(&dir, &row).1,
                message: (unary[0] == "final").then(|| untag(&unary[1])),
                streamed: Vec::new(),
            }
        })
        .collect()
}

/// Replay every session of a compiled template at every chunking: the contract holds at each,
/// and what the table says is the same at each. Returns the sessions whose message differs from
/// transformers', each with the code of the documented difference whose check it passes, or
/// `unlisted`, and both readings.
fn replay_sessions(
    corpus: &str,
    sessions: &[Session],
    tables: &Tables,
) -> BTreeMap<String, (String, String)> {
    let mut differs = BTreeMap::new();
    let mut replayed = 0;
    for session in sessions {
        let Some((format, literals, template)) = tables.get(&session.template) else {
            continue;
        };
        replayed += 1;
        let markers: Vec<&str> = literals.iter().map(String::as_str).collect();
        let whole = run(
            format,
            &session.declared,
            &session.prompt,
            &session.output,
            &[],
            &session.end,
        );
        let said = Said::of(&whole);
        for (plan, cuts) in plans(&session.output, &markers, 2048) {
            let place = format!("{} ({plan})", session.id);
            let events = run(
                format,
                &session.declared,
                &session.prompt,
                &session.output,
                &cuts,
                &session.end,
            );
            check_contract(&place, &session.output, &events);
            assert_eq!(Said::of(&events), said, "{place}: what the table says");
        }
        for plan in ["whole", "chars", "step:3"] {
            let cuts: Vec<usize> = plans(&session.output, &[], 0)
                .into_iter()
                .find(|(name, _)| name == plan)
                .map(|(_, cuts)| cuts)
                .unwrap_or_default();
            check_tokens(
                &format!("{} ({plan}, tokens)", session.id),
                format,
                &session.declared,
                &session.prompt,
                &session.output,
                &cuts,
            );
        }
        let ours = ours(&whole);
        let theirs = session.message.as_ref().map(theirs);
        if theirs.as_ref() == Some(&ours) {
            continue;
        }
        let readings = match &theirs {
            Some(theirs) => format!("ours   {ours:?}\ntheirs {theirs:?}"),
            None => format!("ours   {ours:?}\ntheirs raised"),
        };
        // A marker given back reads exactly as transformers reads it: no documented difference
        // explains a session that covers one.
        let code = if session.covers.iter().any(|covered| covered == "give-back") {
            "unlisted".to_owned()
        } else if theirs.is_none() {
            "X5".to_owned()
        } else {
            explain(session, format, template.as_ref(), theirs.as_ref())
                .unwrap_or_else(|| "unlisted".to_owned())
        };
        differs.insert(format!("{corpus}\t{}", session.id), (code, readings));
    }
    assert!(
        replayed > 0,
        "no session of a compiled template in {corpus}"
    );
    differs
}

/// The documented differences that explain why the table's reading of `session` differs from
/// transformers' `theirs` (`None` where transformers raised), as codes in the order they apply:
/// with each undone, the table reads transformers' message. `None` when none do.
///
/// - X5: transformers raised; the table's events are defined.
/// - X2: the prompt wrote text after its anchor, which the table leaves to the prompt. Generated
///   instead, it reads as transformers reads it.
/// - X9: the prompt holds no anchor, so the table reads from the output on, where transformers
///   reads the whole prompt as the message's start. With the prompt read as output, it reads as
///   transformers reads it.
/// - X3: the stop text, which the stream does not show, is text transformers keeps. Shown, it
///   reads as transformers reads it.
/// - X1: a field that does not repeat was written more than once; the table keeps every stay
///   where transformers keeps the last. The table reads what the template's reader closes in
///   each field, joined.
/// - X5: the stream was cut for length inside a call, which the table reports cut (its arguments'
///   object left open, the text of a parameter the cut left unfinished malformed), where
///   transformers closes the region at the end of the text. With each cut call closed and that
///   text left out, it reads as transformers reads it. The stream's stop closes every call, so
///   this never explains a session that stopped.
fn explain(
    session: &Session,
    format: &Format,
    template: Option<&ResponseTemplate>,
    theirs: Option<&Message>,
) -> Option<String> {
    let theirs = theirs?;
    let anchor_end = template.and_then(|template| template.last_anchor_end(&session.prompt));
    // The prompt as the table reads it, and what of it is read as output.
    let mut shifts: Vec<(Vec<&str>, &str, String)> =
        vec![(Vec::new(), session.prompt.as_str(), String::new())];
    match anchor_end {
        Some(end) if end < session.prompt.len() => shifts.push((
            vec!["X2"],
            &session.prompt[..end],
            session.prompt[end..].to_owned(),
        )),
        None if !session.prompt.is_empty() && template.is_some() => {
            shifts.push((vec!["X9"], "", session.prompt.clone()));
        }
        _ => {}
    }
    for (codes, prompt, shifted) in shifts {
        let stops: &[bool] = if session.stop.is_empty() {
            &[false]
        } else {
            &[false, true]
        };
        for &shown in stops {
            let mut codes = codes.clone();
            let stop = if shown {
                codes.push("X3");
                session.stop.as_str()
            } else {
                ""
            };
            let output = format!("{shifted}{}{stop}", session.output);
            let events = run(
                format,
                &session.declared,
                prompt,
                &output,
                &[],
                &session.end,
            );
            let read = ours(&events);
            if read == *theirs {
                return Some(codes.join(" "));
            }
            if !matches!(session.end, EngineFinish::Stop)
                && closed(&events).as_ref() == Some(theirs)
            {
                codes.push("X5");
                return Some(codes.join(" "));
            }
            let moved = Session {
                prompt: prompt.to_owned(),
                output: format!("{shifted}{}", session.output),
                ..session.clone()
            };
            if template.and_then(|template| joined(template, &moved)) == Some(read) {
                codes.push("X1");
                return Some(codes.join(" "));
            }
        }
    }
    None
}

/// What the template's reader closes in each field of the output, every region's value joined in
/// order: the message the table gives where a field is written more than once.
fn joined(template: &ResponseTemplate, session: &Session) -> Option<Message> {
    let mut parser = ResponseParser::new(template, &session.prompt, &session.tools).ok()?;
    let mut events = parser
        .feed(&format!("{}{}", session.output, session.stop))
        .ok()?;
    events.extend(parser.finalize().ok()?.1);
    let mut message = Message {
        reasoning: String::new(),
        content: String::new(),
        calls: Vec::new(),
    };
    for event in events {
        if let RegionEvent::RegionClose { field, value } = event {
            let text = value.as_str().map(str::to_owned);
            match (field.as_str(), text) {
                ("thinking" | "reasoning_content", Some(text)) => message.reasoning.push_str(&text),
                ("content", Some(text)) => message.content.push_str(&text),
                _ => {}
            }
        }
    }
    Some(message)
}

/// The literal text a template's delimiters write, for the cuts around markers: the literals of
/// `open`, `close` and `start_anchor`, and the runs of literal characters of their patterns and
/// of a tag pattern.
fn literal_texts(spec: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let mut add = |value: &Value, pattern: bool| {
        let strings: Vec<&str> = match value {
            Value::String(s) => vec![s.as_str()],
            Value::Array(items) => items.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        for s in strings {
            if pattern {
                let mut run = String::new();
                let mut chars = s.chars();
                while let Some(c) = chars.next() {
                    match c {
                        '\\' => match chars.next() {
                            Some(next) if !next.is_ascii_alphanumeric() => run.push(next),
                            _ => out.push(std::mem::take(&mut run)),
                        },
                        '(' | ')' | '[' | ']' | '{' | '}' | '|' | '?' | '*' | '+' | '.' | '^'
                        | '$' => {
                            out.push(std::mem::take(&mut run));
                        }
                        c => run.push(c),
                    }
                }
                out.push(run);
            } else {
                out.push(s.to_owned());
            }
        }
    };
    add(&spec["start_anchor"], false);
    add(&spec["start_anchor_pattern"], true);
    for field in spec["fields"]
        .as_object()
        .into_iter()
        .flat_map(|f| f.values())
    {
        add(&field["open"], false);
        add(&field["close"], false);
        add(&field["open_pattern"], true);
        add(&field["close_pattern"], true);
        add(&field["content_args"]["tag_pattern"], true);
    }
    out.retain(|text| text.chars().count() >= 2);
    out.sort();
    out.dedup();
    out
}

/// The sessions whose message differs, each with its documented difference, as
/// `to_symphony_differs.tsv` pins them (`TO_SYMPHONY_WRITE=1` rewrites it): a session no
/// difference explains fails, and so does a listed session that reads as transformers does.
#[test]
fn compiled_tables_read_the_messages_transformers_parses() {
    let mut differs = replay_sessions("compiled", &public_sessions(), public_tables());
    differs.extend(replay_sessions("hf", &fixture_sessions(), fixture_tables()));
    let unexplained: Vec<String> = differs
        .iter()
        .filter(|(_, (code, _))| code == "unlisted")
        .map(|(id, (_, readings))| format!("{id}\n{readings}"))
        .collect();
    assert!(
        unexplained.is_empty(),
        "differences no documented difference explains:\n{}",
        unexplained.join("\n\n")
    );
    let seen: String = differs
        .iter()
        .map(|(id, (code, _))| format!("{id}\t{code}\n"))
        .collect();
    pinned("to_symphony_differs.tsv", &seen);
}

#[test]
fn where_transformers_own_stream_differs_the_table_reads_the_whole_text() {
    let tables = public_tables();
    for session in public_sessions() {
        let Some((format, _, _)) = tables.get(&session.template) else {
            continue;
        };
        let whole = ours(&run(
            format,
            &session.declared,
            &session.prompt,
            &session.output,
            &[],
            &session.end,
        ));
        let starts: Vec<usize> = session.output.char_indices().map(|(at, _)| at).collect();
        for record in &session.streamed {
            // Offsets in characters, or one character a chunk.
            let cuts: Vec<usize> = match record["cuts"].as_array() {
                Some(offsets) => offsets
                    .iter()
                    .map(|c| starts[c.as_u64().unwrap() as usize])
                    .collect(),
                None => starts.iter().skip(1).copied().collect(),
            };
            let events = run(
                format,
                &session.declared,
                &session.prompt,
                &session.output,
                &cuts,
                &session.end,
            );
            assert_eq!(ours(&events), whole, "{} at {cuts:?}", session.id);
        }
    }
}

/// An output of at least `bytes` bytes for `template`: its sessions' outputs, one after another.
fn long_output(sessions: &[Session], template: &str, bytes: usize) -> String {
    let outputs: Vec<&str> = sessions
        .iter()
        .filter(|session| session.template == template && session.prompt.is_empty())
        .chain(
            sessions
                .iter()
                .filter(|session| session.template == template),
        )
        .map(|session| session.output.as_str())
        .filter(|output| !output.is_empty())
        .collect();
    let mut out = String::new();
    while out.len() < bytes {
        for output in &outputs {
            out.push_str(output);
            out.push('\n');
        }
    }
    out
}

/// The time a table takes to read `output` in pieces of sixteen bytes, the least of five runs.
fn read_time(format: &Format, output: &str) -> Duration {
    let starts: Vec<usize> = output.char_indices().map(|(at, _)| at).collect();
    let cuts: Vec<usize> = starts
        .iter()
        .copied()
        .filter(|&at| at > 0 && at % 16 < 4)
        .collect();
    (0..5)
        .map(|_| {
            let started = Instant::now();
            let events = run(
                format,
                &Declared::default(),
                "",
                output,
                &cuts,
                &EngineFinish::Stop,
            );
            assert!(!events.is_empty());
            started.elapsed()
        })
        .min()
        .unwrap()
}

/// A compiled table reads in linear time: per byte, 64 KiB take at most 1.5 times what 8 KiB
/// take. Timing; run alone in a release build (`cargo test --release -- --ignored`).
#[test]
#[ignore = "timing: run alone in a release build"]
#[expect(clippy::print_stdout, reason = "the timings are the test's record")]
fn compiled_tables_read_in_linear_time() {
    let sessions = public_sessions();
    for (template, (format, _, _)) in public_tables() {
        let short = long_output(&sessions, template, 8 << 10);
        let long = long_output(&sessions, template, 64 << 10);
        let per_byte = |output: &str| read_time(format, output).as_secs_f64() / output.len() as f64;
        let (at_8k, at_64k) = (per_byte(&short), per_byte(&long));
        println!(
            "{template}: {:.1} ns/byte at {} bytes, {:.1} ns/byte at {} bytes, ratio {:.2}",
            at_8k * 1e9,
            short.len(),
            at_64k * 1e9,
            long.len(),
            at_64k / at_8k
        );
        assert!(
            at_64k <= 1.5 * at_8k,
            "{template}: {at_64k} vs {at_8k} per byte"
        );
    }
}
