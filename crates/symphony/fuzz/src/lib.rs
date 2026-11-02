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
//!   their ids and arguments and the finish (both replays count their call ids from one nonce,
//!   as one request's engine does);
//! - well-formedness: one start and one end per call, arguments only between them and a valid
//!   JSON prefix after every fragment until a `Malformed { InvalidArguments }` event of no bytes
//!   says where the arguments stopped being one (from there every fragment is the model's bytes
//!   as written), reasoning text only inside a region, one `Finish` that
//!   counts the calls; a fragment of the arguments that closes a string, writes a member under an
//!   empty key or writes an empty string carries the output bytes that wrote it (a reader makes
//!   no byte up); and a call that ended, at its own terminal or at the stream's end, has
//!   arguments that are whole JSON, unless the engine said `Malformed` while the call was open
//!   (three names for the breach: `empty-arguments`, `tolerated-json`, `whole-json`, so that a
//!   run allows a class already reported and still catches the others); and a pythonic call
//!   that ended without its own `)` has its object left open, closed for it by nothing;
//! - the adapters render any event sequence without a panic, and the Chat Completions stream,
//!   concatenated, says what the whole message says.
//!
//! Every breach message starts with `breach <name>:`, the invariant's name, so that the artifacts
//! of a campaign group by class.
//!
//! The seeds `prepare` writes from the recorded output slices of bellwether's hand-written sets
//! ([`slices`], read from the tree `BELLWETHER_FIXTURES` names) are cases in this layout, so a
//! seed replays as its table's own output and every mutation of it stays a case.

#![forbid(unsafe_code)]

use std::{fs, io::Write, path::Path, sync::OnceLock};

/// The macro a target is written with, so that the targets and the harness share one dependency.
pub use libfuzzer_sys::fuzz_target;
use openai_protocol::common::{Function, Tool};
use serde_json::{json, Value};
use symphony::{
    adapt::{chat, messages, responses},
    formats::Table,
    json::PartialJson,
    CallIds, CallSyntax, Declared, DropReason, Engine, EngineFinish, Event, Events, FinishReason,
    Input, MalformedReason, Parser,
};

/// The recorded output slices the seeds are made from, one definition with the crate's bench.
#[path = "../../tests/common/slices.rs"]
pub mod slices;

pub use slices::{Recorded, Slice};

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

/// The breach names this run looks past: `SYMPHONY_FUZZ_ALLOW`, comma separated, read once.
fn allowances() -> &'static [String] {
    static ALLOWED: OnceLock<Vec<String>> = OnceLock::new();
    ALLOWED.get_or_init(|| {
        std::env::var("SYMPHONY_FUZZ_ALLOW")
            .map(|list| list.split(',').map(|n| n.trim().to_string()).collect())
            .unwrap_or_default()
    })
}

/// Whether the breach named `name` is allowed for this run: `SYMPHONY_FUZZ_ALLOW` lists, comma
/// separated, the names the breach messages carry, so that a class already reported does not stop
/// the fuzzer from finding the next one. Every check honors it, through [`breach`].
pub fn allowed(name: &str) -> bool {
    allowances().iter().any(|allowed| allowed == name)
}

/// The check named `name`: a panic whose message starts with `breach <name>:` unless `holds`, or
/// the run allows the breach. `message` is built only when the panic is.
fn breach(name: &str, holds: bool, message: impl FnOnce() -> String) {
    if !holds && !allowed(name) {
        panic!("breach {name}: {}", message());
    }
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

/// The events of `case` replayed with its output cut at `cuts` (none for the whole replay,
/// `case.cuts` for the pieces): what a reader of the stream sees, for reading an artifact
/// outside the shape checks (`explain --events`).
///
/// # Panics
///
/// As a replay does: a `feed` that fails is the `feed-failed` breach, and an engine that panics
/// panics here.
pub fn events(case: &Case, cuts: &[usize]) -> Vec<Event<'static>> {
    replay(case, cuts).events
}

/// Replays `case` whole and in its pieces and asserts the contract on both.
pub fn check(case: &Case) {
    let whole = replay(case, &[]);
    let pieces = replay(case, &case.cuts);
    for (which, replayed) in [("whole", &whole), ("pieces", &pieces)] {
        let conserved: String = replayed.events.iter().map(bytes_of).collect();
        breach("conservation", conserved == case.output, || {
            format!(
                "{}: the {which} replay does not conserve the output; its events carry {conserved:?}",
                describe(case)
            )
        });
        breach(
            "finish-last",
            matches!(replayed.events.last(), Some(Event::Finish { .. })),
            || {
                format!(
                    "{}: the {which} replay does not end with Finish",
                    describe(case)
                )
            },
        );
        check_well_formed(case, which, replayed);
    }
    let whole_said = Said::of(&whole.events);
    let pieces_said = Said::of(&pieces.events);
    breach("chunking", whole_said == pieces_said, || {
        format!(
            "{}: the pieces say something else than the whole: whole {whole_said:?}, pieces {pieces_said:?}",
            describe(case)
        )
    });
    check_adapters(case, &whole.events, &pieces.events);
}

/// [`declared`], built once: the tools and their schemas are the same for every replay, and a
/// replay runs twice per execution.
fn declared_shared() -> Declared {
    static DECLARED: OnceLock<Declared> = OnceLock::new();
    DECLARED.get_or_init(declared).clone()
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

/// What a run starts from, as [`prepare`] counts it, and the groups the tree had not recorded,
/// each a notice for the caller to print.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prepared {
    pub dictionaries: usize,
    pub seeds: usize,
    pub missing: Vec<String>,
}

/// libFuzzer's longest dictionary word, in bytes: a longer entry it skips without a word.
pub const DICTIONARY_WORD_MAX: usize = 64;

/// Writes what a run starts from under `out`: `dict/<target>.dict`, a dictionary per target in
/// libFuzzer's syntax (every terminal of its table, of every table for `any_table`, the declared
/// names and the atoms of the call syntaxes), and `seeds/<table>/<family>`, a seed per recorded
/// slice of the groups recorded under `root`, the tree `BELLWETHER_FIXTURES` names
/// ([`slices::recorded`]), each the case that replays the slice ([`encode`]).
pub fn prepare(root: &Path, out: &Path) -> Result<Prepared, String> {
    let dict = out.join("dict");
    fs::create_dir_all(&dict).map_err(|e| format!("{}: {e}", dict.display()))?;
    for (target, table) in TARGETS {
        let tables: Vec<Table> = match table {
            Some(table) => vec![*table],
            None => Table::ALL.to_vec(),
        };
        let path = dict.join(format!("{target}.dict"));
        let mut file = fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut entries: Vec<String> = Vec::new();
        for table in tables {
            let format = table.format();
            entries.extend(format.terminal_texts().map(str::to_owned));
        }
        entries.extend(declared_names().into_iter().map(str::to_owned));
        entries.extend(ATOMS.iter().map(|atom| (*atom).to_owned()));
        entries.sort();
        entries.dedup();
        for entry in entries {
            if entry.len() > DICTIONARY_WORD_MAX {
                return Err(format!(
                    "{}: {entry:?} is {} bytes; a dictionary word is at most {DICTIONARY_WORD_MAX}",
                    path.display(),
                    entry.len()
                ));
            }
            writeln!(file, "{}", quoted(&entry)).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    let recorded = slices::recorded(root)?;
    let mut seeds = 0;
    for (table, slice) in &recorded.slices {
        let dir = out.join("seeds").join(table.name());
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = dir.join(&slice.family);
        let seed = encode(*table, &slice.prompt, &slice.output)?;
        fs::write(&path, seed).map_err(|e| format!("{}: {e}", path.display()))?;
        seeds += 1;
    }
    Ok(Prepared {
        dictionaries: TARGETS.len(),
        seeds,
        missing: recorded.missing,
    })
}

/// Pieces of the call syntaxes a mutation composes markers with, each within a dictionary word.
const ATOMS: &[&str] = &[
    "{\"name\": \"get_weather\", \"arguments\": ",
    "{\"city\": \"Paris\", \"days\": 3}}",
    "\"arguments\"",
    "\"name\"",
    "city",
    "days",
    "Paris",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<function=",
    "<parameter=",
    "</parameter>",
    "</function>",
    " name=\"",
    " string=\"true\"",
    " key=\"",
    " type=\"string\"",
    " index=\"1\"",
    "call:",
    "<|\"|>",
    "[get_weather(city='Paris')]",
    "\\u00e9",
    "\\\"",
    "\u{fffd}",
];

/// `entry` in libFuzzer's dictionary syntax: quoted, with every byte outside printable ASCII, the
/// quote and the backslash escaped as `\xNN`.
fn quoted(entry: &str) -> String {
    let mut out = String::from("\"");
    for byte in entry.bytes() {
        match byte {
            b'"' | b'\\' => out.push_str(&format!("\\x{byte:02x}")),
            0x20..=0x7e => out.push(byte as char),
            _ => out.push_str(&format!("\\x{byte:02x}")),
        }
    }
    out.push('"');
    out
}

/// The events of one replay.
struct Replayed {
    events: Vec<Event<'static>>,
}

/// The nonce both replays of a case count their call ids from: an engine draws its own at random
/// (one request, one nonce), so two engines over one output would name the same call differently
/// and the chunking check could never compare them.
const NONCE: &str = "0123456789abcdef00000000";

/// `case` through a new engine, its output cut at `cuts`, the events drained after every feed as
/// a stream's reader drains them.
fn replay(case: &Case, cuts: &[usize]) -> Replayed {
    // The nonce is this module's constant; `the_nonce_is_one_the_engine_takes` holds it valid, so
    // an invalid one fails that test rather than falling back to a random nonce in silence.
    let Some(ids) = CallIds::from_nonce(NONCE) else {
        unreachable!("NONCE is not 24 hex characters")
    };
    let mut engine = Engine::with_call_ids(case.table.shared(), declared_shared(), ids);
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
    feed(
        case,
        &mut engine,
        Input::End {
            finish: case.finish.clone(),
        },
        &mut out,
        &mut events,
    );
    Replayed { events }
}

fn feed<'d>(
    case: &Case,
    engine: &mut Engine,
    input: Input<'d>,
    out: &mut Events<'d>,
    events: &mut Vec<Event<'static>>,
) {
    let step = input.clone();
    let fed = engine.feed(input, out);
    breach("feed-failed", fed.is_ok(), || {
        let why = fed
            .as_ref()
            .err()
            .map_or_else(String::new, ToString::to_string);
        format!("{}: feed of {step:?} failed: {why}", describe(case))
    });
    events.extend(out.drain().into_iter().map(Event::into_owned));
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
fn bytes_of<'a>(event: &'a Event<'_>) -> &'a str {
    match event {
        Event::Content(t) | Event::Reasoning(t) => &t.text,
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => &text.text,
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => &source.text,
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
    }
}

/// Whether `fragment`, appended to the arguments `so_far`, closes a JSON string `so_far` left
/// open: the quote that ends a value the reader was streaming.
fn closes_an_open_string(so_far: &str, fragment: &str) -> bool {
    let mut inside = false;
    let mut escaped = false;
    for c in so_far.chars() {
        if !inside {
            inside = c == '"';
        } else if escaped {
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            inside = false;
        }
    }
    if !inside {
        return false;
    }
    for c in fragment.chars() {
        if escaped {
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            return true;
        }
    }
    false
}

/// Whether `json`, a fragment of the arguments, writes an empty string for a value the model left
/// out: the fragment alone (`""`, `, ""`, `[""`) with no byte of the output behind it (none, or
/// whitespace), or a whole member (`{"city": ""`, `, "city": ""`) whose source holds nothing of a
/// value once the key's own text is taken out: whitespace and the separators `=`, `:` and `,`
/// alone (`city=`). The model's own empty value leaves its quotes or its tags in the source
/// (`city=""`, `<parameter=city></parameter>`), and the JSON syntaxes' source is the member itself.
/// A trailing `}`, `,` or `]` (`{"city": ""}`, `"city": "",`, `"tags": [""]`) is set aside before
/// the string is judged.
fn made_up_empty_string(json: &str, source: &str) -> bool {
    // A whole member may close its object or list or carry the next separator (`{"key": ""}`,
    // `"key": "",`, `"tags": [""]`): the empty string is judged without that trailing byte.
    let json = json.strip_suffix(['}', ',', ']']).unwrap_or(json);
    if !json.ends_with("\"\"") {
        return false;
    }
    let member = json
        .strip_prefix('{')
        .or_else(|| json.strip_prefix(", "))
        .unwrap_or(json);
    let key = member
        .strip_prefix('"')
        .and_then(|rest| rest.split_once("\": "))
        .map(|(key, _)| key)
        .filter(|key| !key.is_empty());
    match key {
        Some(key) => source
            .replacen(key, "", 1)
            .chars()
            .all(|c| c.is_whitespace() || matches!(c, '=' | ':' | ',')),
        None => source.trim().is_empty(),
    }
}

/// Whether `text` is one whole JSON value: the syntax, as a client's reader takes it, not a
/// reader's number range. `serde_json`'s `Value` refuses `9.99e999999` for its range alone, a
/// number JSON writes and a client reads as it can; `IgnoredAny` walks the same grammar (the
/// brackets and commas, a word spelled whole, a number's digits, decimals and exponent, the
/// escapes, a raw control character refused inside a string) without converting the number.
/// It takes a `\u` escape of any code point, a lone surrogate included, as the grammar does;
/// a reader that drops a byte of the output fails conservation before it reaches this judge.
fn whole_json(text: &str) -> bool {
    serde_json::from_str::<serde::de::IgnoredAny>(text).is_ok()
}

/// Whether `text` has the shape of one whole JSON value, whatever JSON says of its tokens: an
/// object or a list opened at the front and closed at the end, every bracket outside a string's
/// contents matched; or one scalar token, a string that is closed, a number or a word. The shape
/// the prefix judge took as whole when the call ended; whether JSON takes the tokens inside is
/// [`whole_json`]'s question, and the two apart name the breach.
fn closed_structure(text: &str) -> bool {
    let text = text.trim();
    if text.is_empty() || (text.starts_with(['{', '[']) && !text.ends_with(['}', ']'])) {
        return false;
    }
    let mut open: Vec<char> = Vec::new();
    let mut inside = false;
    let mut escaped = false;
    for c in text.chars() {
        if inside {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                inside = false;
            }
        } else {
            match c {
                '"' => inside = true,
                '{' | '[' => open.push(c),
                '}' | ']' => {
                    let opener = if c == '}' { '{' } else { '[' };
                    if open.pop() != Some(opener) {
                        return false;
                    }
                }
                _ => {}
            }
        }
    }
    open.is_empty() && !inside
}

/// Whether `text` is a JSON prefix the prefix parser takes whole.
fn valid_json_prefix(text: &str) -> bool {
    matches!(PartialJson::default().parse(text, true), Ok((_, consumed)) if consumed == text.len())
}

/// What a client relies on in the shape of the stream, checked event by event.
fn check_well_formed(case: &Case, which: &str, replayed: &Replayed) {
    // A pythonic call ends at its own `)`, which every template writes before the region's
    // marker; the other syntaxes' ends may carry no bytes by design (the block's close ends
    // the call).
    let pythonic = matches!(
        case.table.format().call_syntax(),
        Some(CallSyntax::Pythonic(_))
    );
    let mut started: Vec<u32> = Vec::new();
    let mut ended: Vec<u32> = Vec::new();
    let mut arguments: Vec<(u32, String)> = Vec::new();
    let mut malformed_inside: Vec<u32> = Vec::new();
    // The calls whose arguments stopped being JSON, said by a `Malformed { InvalidArguments }`
    // event of no bytes while the call was open: from there the fragments are the model's bytes.
    let mut broke: Vec<u32> = Vec::new();
    let mut reasoning_open = false;
    let mut finished = false;
    // Whether the last event was the word of a call, or a string in it, that the stream's stop
    // closed (`Malformed { OpenAtTheStop }`, of no bytes), or of a string the call's closing
    // marker closed (`Malformed { OpenAtTheMarker }`): the fragment right after it may close the
    // string with no byte of the output behind it, since the model's own end of output, or of
    // the call, is what closed it.
    let mut after_the_stops_word = false;
    for (at, event) in replayed.events.iter().enumerate() {
        let place = || format!("{} ({which}), event {at}: {event:?}", describe(case));
        breach("after-finish", !finished, || {
            format!("{}: an event after Finish", place())
        });
        let stop_closed = std::mem::take(&mut after_the_stops_word);
        match event {
            Event::ToolCallStart { index, .. } => {
                breach("second-start", !started.contains(index), || {
                    format!("{}: a second start", place())
                });
                started.push(*index);
                arguments.push((*index, String::new()));
            }
            Event::ToolCallArguments {
                index,
                json,
                source,
            } => {
                breach("arguments-before-start", started.contains(index), || {
                    format!("{}: arguments before the call's start", place())
                });
                breach("arguments-after-end", !ended.contains(index), || {
                    format!("{}: arguments after the call's end", place())
                });
                if let Some((_, so_far)) = arguments.iter_mut().find(|(i, _)| i == index) {
                    // A fragment that closes a string the arguments so far left open carries the
                    // bytes that closed it, the model's quote, closing tag or token: a closing
                    // quote with no byte of the output behind it (none, or whitespace alone) is
                    // one the reader made up for a value the model never closed. The one
                    // exception is the model's own end of output: the stop has no bytes, and the
                    // word right before the fragment says the stop closed the string.
                    breach(
                        "made-up-quote",
                        !closes_an_open_string(so_far, json)
                            || !source.text.trim().is_empty()
                            || stop_closed,
                        || {
                            format!(
                                "{}: the fragment {json:?} closes a string with no byte of the output behind it",
                                place()
                            )
                        },
                    );
                    // A member under an empty key is the model's to write (`{"": 1}` in a JSON
                    // syntax: the source carries the key's quotes); a reader that spells one for
                    // a key the model left out made the key up, whether it writes the key as a
                    // fragment of its own or opens the member with it (`{"": "`, `, "": 1`).
                    let member = json
                        .strip_prefix('{')
                        .or_else(|| json.strip_prefix(", "))
                        .unwrap_or(json);
                    breach(
                        "empty-key-member",
                        !(member.starts_with("\"\": ") && !source.text.contains("\"\"")),
                        || {
                            format!(
                                "{}: the fragment {json:?} writes a member under an empty key the output did not write",
                                place()
                            )
                        },
                    );
                    // An empty string no byte of the output wrote: the reader quoted an empty bare
                    // value where the model left one out before a comma or a close, as a
                    // fragment of its own or at the end of a whole member (`, "city": ""`).
                    breach(
                        "made-up-empty-string",
                        !made_up_empty_string(json, &source.text),
                        || {
                            format!(
                                "{}: the fragment {json:?} writes an empty string with no byte of a value behind it",
                                place()
                            )
                        },
                    );
                    so_far.push_str(json);
                    if broke.contains(index) {
                        breach("verbatim-after-break", json == &source.text, || {
                            format!(
                                "{}: the fragment {json:?} after the break is not the model's bytes as written",
                                place()
                            )
                        });
                    } else {
                        breach("json-prefix", valid_json_prefix(so_far), || {
                            format!(
                                "{}: the arguments so far, {so_far:?}, are no valid JSON prefix",
                                place()
                            )
                        });
                    }
                }
            }
            Event::ToolCallEnd { index, source } => {
                breach("end-before-start", started.contains(index), || {
                    format!("{}: an end before the call's start", place())
                });
                breach("second-end", !ended.contains(index), || {
                    format!("{}: a second end", place())
                });
                ended.push(*index);
                // The stream's end closes an open call too, and the promise holds there as at a
                // marker's close: a call the end cut says `Malformed` before its end, or its
                // arguments are whole. A reader that ends a cut call there without a word is a
                // breach, whichever its syntax.
                if !malformed_inside.contains(index) {
                    let so_far = arguments
                        .iter()
                        .find(|(i, _)| i == index)
                        .map(|(_, so_far)| so_far.as_str())
                        .unwrap_or_default();
                    // Three names for one promise, so that a run allows a class already
                    // reported and still catches the others: a call that ended with no fragment
                    // at all; a value whose shape closed but whose tokens JSON refuses (what the
                    // prefix judge tolerates: a value left out, a trailing comma, an escape JSON
                    // lacks, a short `\u`, whitespace outside JSON's, a number or a word cut
                    // short); and the rest, the committed prefix of a value that never closed.
                    if so_far.is_empty() {
                        breach("empty-arguments", false, || {
                            format!(
                                "{}: the call ended with no arguments fragment at all",
                                place()
                            )
                        });
                    } else if closed_structure(so_far) {
                        breach("tolerated-json", whole_json(so_far), || {
                            format!(
                                "{}: the call ended with arguments whose shape closed but which JSON refuses: {so_far:?}",
                                place()
                            )
                        });
                    } else {
                        breach("whole-json", whole_json(so_far), || {
                            format!(
                                "{}: the call ended with arguments that are the prefix of a value that never closed: {so_far:?}",
                                place()
                            )
                        });
                    }
                }
                // A pythonic end with no bytes of its own is a call the region's marker or the
                // stream's end cut: nothing closes its object for it, whatever it said. A cut
                // call that said nothing and left its object open is one of the whole-JSON
                // names' breaches above, so no event is named twice.
                if pythonic && source.text.is_empty() {
                    let so_far = arguments
                        .iter()
                        .find(|(i, _)| i == index)
                        .map(|(_, so_far)| so_far.as_str())
                        .unwrap_or_default();
                    breach("pythonic-cut-call-closed", !whole_json(so_far), || {
                        format!(
                            "{}: a pythonic call ended without its `)` with arguments {so_far:?}: a cut call's object stays open",
                            place()
                        )
                    });
                }
            }
            Event::Malformed { text, why } => {
                // The word of a call, or a string in it, that the stream's stop closed, or of a
                // string the call's closing marker closed: the fragment right after it may close
                // the string with no bytes behind it, and the call's arguments at its end are
                // whole JSON, which the end checks as for a call no word was said about.
                let stops_word = matches!(
                    why,
                    MalformedReason::OpenAtTheStop | MalformedReason::OpenAtTheMarker
                ) && text.text.is_empty();
                after_the_stops_word = stops_word;
                for index in &started {
                    if !ended.contains(index) && !malformed_inside.contains(index) && !stops_word {
                        malformed_inside.push(*index);
                    }
                    if !ended.contains(index)
                        && *why == MalformedReason::InvalidArguments
                        && text.text.is_empty()
                        && !broke.contains(index)
                    {
                        broke.push(*index);
                    }
                }
            }
            Event::ReasoningStart => {
                breach("reasoning-reopened", !reasoning_open, || {
                    format!("{}: reasoning already open", place())
                });
                reasoning_open = true;
            }
            Event::Reasoning(_) => {
                breach("reasoning-outside", reasoning_open, || {
                    format!("{}: reasoning outside a region", place())
                });
            }
            Event::ReasoningEnd => {
                breach("reasoning-closed-twice", reasoning_open, || {
                    format!("{}: no reasoning open", place())
                });
                reasoning_open = false;
            }
            Event::Finish { tool_calls, .. } => {
                breach(
                    "finish-count",
                    *tool_calls as usize == started.len(),
                    || {
                        format!(
                            "{}: the finish counts {tool_calls} calls, {} started",
                            place(),
                            started.len()
                        )
                    },
                );
                finished = true;
            }
            Event::Content(_) | Event::Dropped { .. } => {}
        }
    }
    let place = format!("{} ({which})", describe(case));
    breach("no-finish", finished, || format!("{place}: no Finish"));
    breach("reasoning-left-open", !reasoning_open, || {
        format!("{place}: reasoning left open")
    });
    breach("call-without-end", started.len() == ended.len(), || {
        format!(
            "{place}: {} calls started, {} ended",
            started.len(),
            ended.len()
        )
    });
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
    // The stream as the gateway renders it: each event given up to `into_delta`, the path the
    // streaming processor takes; the borrowing `delta` is the robustness suite's.
    for choice in pieces
        .iter()
        .cloned()
        .filter_map(|event| chat::into_delta(0, event))
    {
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
    let message_content = message.message.content.clone().unwrap_or_default();
    breach("chat-content", message_content == content, || {
        format!(
            "{}: the chat stream's content {content:?} is not the message's {message_content:?}",
            describe(case)
        )
    });
    let message_reasoning = message
        .message
        .reasoning_content
        .clone()
        .unwrap_or_default();
    breach("chat-reasoning", message_reasoning == reasoning, || {
        format!(
            "{}: the chat stream's reasoning {reasoning:?} is not the message's {message_reasoning:?}",
            describe(case)
        )
    });
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
    breach("chat-calls", message_calls == calls, || {
        format!(
            "{}: the chat stream's calls {calls:?} are not the message's {message_calls:?}",
            describe(case)
        )
    });

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
    let output_stop = serde_json::to_string(&output.stop_reason).ok();
    let stream_stop = serde_json::to_string(&stop).ok();
    breach("messages-stop", output_stop == stream_stop, || {
        format!(
            "{}: the Messages stream's stop reason {stream_stop:?} is not the output's {output_stop:?}",
            describe(case)
        )
    });

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
    use symphony::Text;

    use super::*;

    /// The tree `BELLWETHER_FIXTURES` names, or None with the notice a skipped test prints: the
    /// seeds are read from bellwether's recorded sets, which the repository does not hold.
    fn fixtures() -> Option<std::path::PathBuf> {
        let root = slices::root();
        if root.is_none() {
            eprintln!(
                "skipping: {} is not set; point it at the tree `bellwether unpack` writes",
                slices::ENV
            );
        }
        root
    }

    /// The recorded slices under `root`, every group the tree lacks named, and every recorded
    /// group checked to yield the families a seed set needs: plain text, one call, parallel calls
    /// and the malformed call cut from the one call's (a thought and a call where the group's
    /// template has a thought).
    fn recorded_slices(root: &Path) -> Result<Recorded, String> {
        let recorded = slices::recorded(root)?;
        for notice in &recorded.missing {
            eprintln!("skipping {notice}");
        }
        if recorded.slices.is_empty() {
            return Err(format!(
                "none of the eight groups is recorded under {}",
                root.display()
            ));
        }
        for group in &slices::GROUPS {
            let families: Vec<&str> = recorded
                .slices
                .iter()
                .filter(|(table, _)| table.name() == group.table)
                .map(|(_, slice)| slice.family.as_str())
                .collect();
            if families.is_empty() {
                continue;
            }
            println!("{}: {}", group.table, families.join(", "));
            for family in ["plain", "one_call", "parallel_calls", "malformed"] {
                if !families.contains(&family) {
                    return Err(format!(
                        "{}: no {family} slice among {families:?}",
                        group.table
                    ));
                }
            }
        }
        Ok(recorded)
    }

    #[test]
    fn every_recorded_slice_holds_the_contract() -> Result<(), String> {
        let Some(root) = fixtures() else {
            return Ok(());
        };
        for (table, slice) in &recorded_slices(&root)?.slices {
            let data = encode(*table, &slice.prompt, &slice.output)?;
            let case = decode(&data, None);
            assert_eq!(case.table, *table, "{}", slice.family);
            assert_eq!(case.prompt, slice.prompt, "{}", slice.family);
            assert_eq!(case.output, slice.output, "{}", slice.family);
            assert!(
                !case.cuts.is_empty(),
                "{}: a seed replays in pieces",
                slice.family
            );
            check(&case);
            check(&decode(&data, Some(*table)));
            // The malformed slice is the one call's with its body cut short before the call's
            // close: the table says so, where a cut before another terminal would read whole.
            if slice.family == "malformed" {
                let whole = replay(&case, &[]);
                assert!(
                    whole.events.iter().any(|event| matches!(
                        event,
                        Event::Malformed { why, .. } if *why != MalformedReason::UnknownTool
                    )),
                    "{}: the malformed slice reads whole: {:?}",
                    table.name(),
                    slice.output
                );
            }
        }
        Ok(())
    }

    /// The byte length of a dictionary line once libFuzzer reads it: the quotes off, `\xNN` one byte.
    fn word_len(line: &str) -> usize {
        let inner = line.trim_matches('"');
        inner
            .split("\\x")
            .enumerate()
            .map(|(i, part)| {
                if i == 0 {
                    part.len()
                } else {
                    1 + part.len().saturating_sub(2)
                }
            })
            .sum()
    }

    #[test]
    fn the_seeds_prepare_writes_decode_as_the_slices() -> Result<(), String> {
        let Some(root) = fixtures() else {
            return Ok(());
        };
        let recorded = recorded_slices(&root)?;
        let out =
            std::env::temp_dir().join(format!("symphony-fuzz-prepare-{}", std::process::id()));
        let prepared = prepare(&root, &out)?;
        assert_eq!(prepared.dictionaries, TARGETS.len());
        assert_eq!(prepared.seeds, recorded.slices.len());
        assert_eq!(prepared.missing, recorded.missing);
        for (table, slice) in &recorded.slices {
            let family = slice.family.as_str();
            let path = out.join("seeds").join(table.name()).join(family);
            let data = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            for case in [decode(&data, None), decode(&data, Some(*table))] {
                assert_eq!(case.table, *table, "{family}");
                assert_eq!(case.prompt, slice.prompt, "{family}");
                assert_eq!(case.output, slice.output, "{family}");
                assert_eq!(case.finish, EngineFinish::Stop, "{family}");
                assert!(!case.cuts.is_empty(), "{family}: a seed replays in pieces");
            }
        }
        fs::remove_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
        Ok(())
    }

    #[test]
    fn the_dictionaries_prepare_writes_fit_libfuzzers_words() -> Result<(), String> {
        // A tree with no group recorded: the dictionaries come from the tables alone.
        let out = std::env::temp_dir().join(format!("symphony-fuzz-dict-{}", std::process::id()));
        let empty = out.join("empty-tree");
        fs::create_dir_all(&empty).map_err(|e| format!("{}: {e}", empty.display()))?;
        let prepared = prepare(&empty, &out.join("prepared"))?;
        assert_eq!((prepared.dictionaries, prepared.seeds), (TARGETS.len(), 0));
        assert_eq!(prepared.missing.len(), slices::GROUPS.len());
        let out_dir = out.join("prepared");
        for (target, _) in TARGETS {
            let path = out_dir.join("dict").join(format!("{target}.dict"));
            let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            assert!(text.lines().count() > 1, "{target}: an empty dictionary");
            for line in text.lines() {
                assert!(
                    word_len(line) <= DICTIONARY_WORD_MAX,
                    "{target}: {line} is longer than a dictionary word"
                );
            }
        }
        fs::remove_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
        Ok(())
    }

    /// The name of the breach `check_well_formed` reports on `events`, as the stream of one
    /// replay, or None when it holds: the checks on the shape of a stream, without a parser
    /// behind it, so that a check is tried on the event list it is about.
    fn breach_in(table: Table, events: Vec<Event<'static>>) -> Option<String> {
        // An allowance would turn a breach into a hold and the assertion into a riddle.
        assert!(
            allowances().is_empty(),
            "the harness's own tests run with SYMPHONY_FUZZ_ALLOW unset; it allows {:?}",
            allowances()
        );
        let case = Case {
            table,
            prompt: String::new(),
            output: String::new(),
            cuts: Vec::new(),
            empty_between: false,
            finish: EngineFinish::Stop,
        };
        let replayed = Replayed { events };
        let outcome = std::panic::catch_unwind(|| check_well_formed(&case, "synthetic", &replayed));
        let payload = outcome.err()?;
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()));
        match message.as_deref().and_then(|m| m.strip_prefix("breach ")) {
            Some(rest) => rest.split(':').next().map(str::to_string),
            // Any other panic is no answer of the checks: it is raised again, not read as a hold.
            None => std::panic::resume_unwind(payload),
        }
    }

    fn start(index: u32) -> Event<'static> {
        Event::ToolCallStart {
            index,
            id: format!("call_{index}"),
            name: "get_weather".to_string(),
            source: Text::uncounted("<tool_call>"),
        }
    }

    fn arguments(index: u32, json: &str, source: &str) -> Event<'static> {
        Event::ToolCallArguments {
            index,
            json: json.to_string().into(),
            source: Text::uncounted(source.to_string()),
        }
    }

    fn end(index: u32, source: &str) -> Event<'static> {
        Event::ToolCallEnd {
            index,
            source: Text::uncounted(source.to_string()),
        }
    }

    fn malformed() -> Event<'static> {
        Event::Malformed {
            text: Text::uncounted(""),
            why: MalformedReason::UnterminatedRegion,
        }
    }

    fn finish(tool_calls: u32) -> Event<'static> {
        Event::Finish {
            reason: FinishReason::ToolCalls,
            tool_calls,
            reasoning_tokens: 0,
        }
    }

    #[test]
    fn a_call_the_streams_end_closed_keeps_the_whole_json_promise() {
        // The end of the stream closed a call whose arguments are a prefix: without a word it is
        // a breach, as it is for a call a marker closed; with `Malformed` before the end it holds.
        let cut = || {
            vec![
                start(0),
                arguments(0, "{\"city\": \"Par", "{\"city\": \"Par"),
            ]
        };
        let mut without_a_word = cut();
        without_a_word.extend([end(0, ""), finish(1)]);
        assert_eq!(
            breach_in(Table::Qwen3Tagged, without_a_word).as_deref(),
            Some("whole-json")
        );
        let mut said = cut();
        said.extend([malformed(), end(0, ""), finish(1)]);
        assert_eq!(breach_in(Table::Qwen3Tagged, said), None);
        let mut whole = cut();
        whole.extend([
            arguments(0, "is\"}", "is</parameter>"),
            end(0, ""),
            finish(1),
        ]);
        assert_eq!(breach_in(Table::Qwen3Tagged, whole), None);
    }

    #[test]
    fn a_quote_that_closes_a_value_carries_a_byte_of_the_output() {
        // The arguments so far leave a string open; a closing quote with no source behind it is
        // the reader's, one with the closing tag behind it is the model's.
        let open = || {
            vec![
                start(0),
                arguments(0, "{\"city\": \"Par", "{\"city\": \"Par"),
            ]
        };
        for source in ["", "\n"] {
            let mut made_up = open();
            made_up.extend([arguments(0, "\"}", source), end(0, ""), finish(1)]);
            assert_eq!(
                breach_in(Table::Qwen3Tagged, made_up).as_deref(),
                Some("made-up-quote"),
                "source {source:?}"
            );
        }
        let mut closed = open();
        closed.extend([
            arguments(0, "\"}", "</parameter>\n</function>"),
            end(0, "</tool_call>"),
            finish(1),
        ]);
        assert_eq!(breach_in(Table::Qwen3Tagged, closed), None);
        assert!(closes_an_open_string("{\"a\": \"x", "\""));
        assert!(closes_an_open_string("{\"a\": \"x\\\"", "y\""));
        assert!(!closes_an_open_string("{\"a\": \"x\"", "}"));
        assert!(!closes_an_open_string("{\"a\": \"x", "\\\""));
    }

    #[test]
    fn a_member_under_an_empty_key_is_the_models_or_a_breach() {
        // The JSON syntaxes' own `{"": "Paris"}`: the fragment's source carries the key's quotes.
        let models = vec![
            start(0),
            arguments(0, "{", "{"),
            arguments(0, "\"\": ", "\"\": "),
            arguments(0, "\"Paris\"}", "\"Paris\"}"),
            end(0, "</tool_call>"),
            finish(1),
        ];
        assert_eq!(breach_in(Table::Qwen3, models), None);
        let whole = vec![
            start(0),
            arguments(0, "{\"\": \"Paris\"}", "{\"\": \"Paris\"}"),
            end(0, "</tool_call>"),
            finish(1),
        ];
        assert_eq!(breach_in(Table::Qwen3, whole), None);
        // A reader that spells the member for a key-less value: the source has no quotes,
        // whether the key comes as a fragment of its own or opens the member.
        for (json, source) in [
            ("\"\": ", ":"),
            ("\"\": ", "<parameter=>"),
            ("\"\": ", ""),
            ("{\"\": \"", "<parameter=>"),
            (", \"\": 1", "<arg_key></arg_key><arg_value>1</arg_value>"),
        ] {
            let readers = vec![
                start(0),
                arguments(0, "{\"a\": 1", "{\"a\": 1"),
                arguments(0, json, source),
                arguments(0, "}", "}"),
                end(0, ""),
                finish(1),
            ];
            assert_eq!(
                breach_in(Table::Gemma4, readers).as_deref(),
                Some("empty-key-member"),
                "fragment {json:?} with source {source:?}"
            );
        }
    }

    #[test]
    fn an_empty_string_carries_a_byte_of_the_output() {
        // A value the model left out: the lone fragment with nothing behind it, a list's first
        // item alike, or a whole member whose source holds the key and its separator alone (the
        // pythonic reader's `city=`); the model's own empty value leaves its quotes or its tags.
        let first = || {
            vec![
                start(0),
                arguments(0, "{\"city\": \"Paris\"", "city='Paris'"),
            ]
        };
        for (json, source) in [
            (", \"days\": \"\"", ", days="),
            (", \"days\": \"\"", "days:"),
            (", \"days\": \"\"", "\n"),
        ] {
            let mut made_up = first();
            made_up.extend([
                arguments(0, json, source),
                arguments(0, "}", ""),
                end(0, ")"),
                finish(1),
            ]);
            assert_eq!(
                breach_in(Table::Lfm2_5, made_up).as_deref(),
                Some("made-up-empty-string"),
                "fragment {json:?} with source {source:?}"
            );
        }
        for (json, source) in [
            (", \"days\": \"\"", ", days=\"\""),
            (", \"days\": \"\"", ", days=''"),
            (", \"days\": \"\"", "\n<parameter=days>\n</parameter>"),
            (
                ", \"days\": \"\"",
                "<arg_key>days</arg_key><arg_value></arg_value>",
            ),
            (", \"days\": \"\"", ", \"days\": \"\""),
        ] {
            let mut written = first();
            written.extend([
                arguments(0, json, source),
                arguments(0, "}", ")"),
                end(0, "]"),
                finish(1),
            ]);
            assert_eq!(
                breach_in(Table::Lfm2_5, written),
                None,
                "fragment {json:?} with source {source:?}"
            );
        }
        for (before, json, source, close) in [
            ("{\"city\": ", "\"\"", "", "}"),
            ("{\"city\": ", "\"\"", " ", "}"),
            ("{\"tags\": ", "[\"\"", "", "]}"),
        ] {
            let lone = vec![
                start(0),
                arguments(0, before, before),
                arguments(0, json, source),
                arguments(0, close, close),
                end(0, ""),
                finish(1),
            ];
            assert_eq!(
                breach_in(Table::Gemma4, lone).as_deref(),
                Some("made-up-empty-string"),
                "fragment {json:?} with source {source:?}"
            );
        }
        let written = vec![
            start(0),
            arguments(0, "{\"city\": ", "{\"city\": "),
            arguments(0, "\"\"", "</parameter>"),
            arguments(0, "}", "</function>"),
            end(0, "</tool_call>"),
            finish(1),
        ];
        assert_eq!(breach_in(Table::Qwen3Tagged, written), None);
        assert!(made_up_empty_string("{\"city\": \"\"", "city="));
        assert!(!made_up_empty_string("{\"city\": \"\"", "city=\"\""));
        assert!(!made_up_empty_string("{\"city\": \"x\"", ""));
        // A whole member through its closing byte: the object's or the list's close, or the
        // next member's separator.
        assert!(made_up_empty_string("{\"city\": \"\"}", "city="));
        assert!(made_up_empty_string(", \"city\": \"\",", "city="));
        assert!(made_up_empty_string("\"tags\": [\"\"]", "tags="));
        assert!(made_up_empty_string("[\"\"]", " "));
        assert!(!made_up_empty_string("{\"city\": \"\"}", "city=\"\""));
        assert!(!made_up_empty_string("\"tags\": [\"\"]", "tags=[\"\"]"));
    }

    #[test]
    fn the_json_syntaxes_own_empty_key_holds_in_pieces() -> Result<(), String> {
        // The seed a first draft of the empty-key check flagged: the model's own member.
        let output = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"\": \"Paris\"}}\n</tool_call>";
        let data = encode(Table::Qwen3, "", output)?;
        let case = decode(&data, Some(Table::Qwen3));
        assert!(!case.cuts.is_empty());
        check(&case);
        Ok(())
    }

    #[test]
    fn the_whole_json_judge_is_the_syntax_not_a_readers_number_range() -> Result<(), String> {
        for whole in [
            "9.99e999999",
            "8888E8888",
            "{\"days\": -1e999}",
            "{}",
            "[1, 2]",
            "\"x\"",
            "null",
            // The grammar's escape, whatever the code point: a lone surrogate is syntax.
            "{\"a\": \"\\ud800\"}",
        ] {
            assert!(whole_json(whole), "{whole}");
        }
        for refused in [
            "84.",
            "0e",
            "1e+",
            "-",
            "nu",
            "{\"a\": ,}",
            "{\"a\": 1,}",
            "[,,]",
            "\"\\h\"",
            "\"\\u0l\"",
            "{\"a\": \"b",
            "",
            "{\"a\":\u{b}1}",
            // A raw control character inside a string, which the escapes exist for.
            "\"a\nb\"",
            "{\"a\": \"b\tc\"}",
        ] {
            assert!(!whole_json(refused), "{refused:?}");
        }
        // The case the range judge flagged: the arguments pass through as the model wrote them,
        // whole JSON text a client reads.
        let output = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"days\": 9.99e999999}}\n</tool_call>";
        let data = encode(Table::Qwen3, "", output)?;
        check(&decode(&data, Some(Table::Qwen3)));
        Ok(())
    }

    #[test]
    fn the_whole_json_promise_names_its_three_breaches() {
        // The same stream, three arguments: no fragment at all; a shape that closed with a
        // value left out, or one scalar token cut short; and a prefix that never closed.
        let ended_with = |json: &str| {
            let mut events = vec![start(0)];
            if !json.is_empty() {
                events.push(arguments(0, json, json));
            }
            events.extend([end(0, "</tool_call>"), finish(1)]);
            breach_in(Table::Qwen3, events)
        };
        assert_eq!(ended_with("").as_deref(), Some("empty-arguments"));
        for tolerated in [
            "{\"city\": \"Paris\", \"days\": ,}",
            "{\"a\": 1,}",
            "0e",
            "n",
            "\"\\h\"",
        ] {
            assert_eq!(
                ended_with(tolerated).as_deref(),
                Some("tolerated-json"),
                "{tolerated:?}"
            );
        }
        for prefix in [
            "{\"city\": \"Par",
            "{\"city\": \"Paris\"",
            "[1, [2",
            "\"Par",
        ] {
            assert_eq!(
                ended_with(prefix).as_deref(),
                Some("whole-json"),
                "{prefix:?}"
            );
        }
        for whole in ["{\"city\": \"Paris\"}", "{}", "[]", "1", "\"x\""] {
            assert_eq!(ended_with(whole), None, "{whole:?}");
        }
        assert!(closed_structure("{\"a\": \"}\"}") && !closed_structure("{\"a\": \"}\""));
        assert!(!closed_structure("}") && !closed_structure("{]") && !closed_structure("  "));
    }

    #[test]
    fn the_events_of_both_replays_are_the_streams_a_reader_sees() -> Result<(), String> {
        let output = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let case = decode(&encode(Table::Qwen3, "", output)?, Some(Table::Qwen3));
        let (whole, pieces) = (events(&case, &[]), events(&case, &case.cuts));
        assert!(
            pieces.len() > whole.len(),
            "the pieces come as the deltas arrive"
        );
        for replay in [&whole, &pieces] {
            let conserved: String = replay.iter().map(bytes_of).collect();
            assert_eq!(conserved, output);
            assert!(matches!(
                replay.last(),
                Some(Event::Finish { tool_calls: 1, .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn a_pythonic_call_that_ended_without_its_close_said_so_and_stays_open() {
        // The region's marker cut the call: an end of no bytes with the object closed for it is
        // a breach, with a word or without; with the word and the object left open it holds;
        // and a tagged call ends without bytes of its own by design.
        let closed_for_it = vec![
            start(0),
            arguments(0, "{\"city\": \"Paris\"}", "city='Paris'"),
            end(0, ""),
            finish(1),
        ];
        assert_eq!(
            breach_in(Table::Lfm2_5, closed_for_it).as_deref(),
            Some("pythonic-cut-call-closed")
        );
        let said_and_closed = vec![
            start(0),
            arguments(0, "{\"city\": \"Paris\"", "city='Paris'"),
            malformed(),
            arguments(0, "}", ""),
            end(0, ""),
            finish(1),
        ];
        assert_eq!(
            breach_in(Table::Lfm2_5, said_and_closed).as_deref(),
            Some("pythonic-cut-call-closed")
        );
        let said = vec![
            start(0),
            arguments(0, "{\"city\": \"Paris\"", "city='Paris'"),
            malformed(),
            end(0, ""),
            finish(1),
        ];
        assert_eq!(breach_in(Table::Lfm2_5, said), None);
        let closed_itself = vec![
            start(0),
            arguments(0, "{\"city\": \"Paris\"}", "city='Paris')"),
            end(0, ")"),
            finish(1),
        ];
        assert_eq!(breach_in(Table::Lfm2_5, closed_itself), None);
        let tagged = vec![start(0), arguments(0, "{}", ""), end(0, ""), finish(1)];
        assert_eq!(breach_in(Table::Qwen3Tagged, tagged), None);
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
    fn the_nonce_is_one_the_engine_takes() {
        assert!(CallIds::from_nonce(NONCE).is_some(), "{NONCE}");
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
