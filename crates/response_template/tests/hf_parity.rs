//! Replays transformers 5.17.0 over the recordings in `tests/fixtures/hf/`
//! (written by `scripts/generate_hf_fixtures.py`) and compares value for
//! value: same keys in the same order, int vs float kept, floats bit-exact.
//!
//! - Loading: a template transformers accepts loads here unless
//!   `tests/unsupported.tsv` lists it with the reason this crate reports; one
//!   transformers rejects is rejected here too.
//! - Every session of a loaded template: the whole output in one `feed` (the
//!   final message and the trace of events), every two-way split (every final
//!   message, and the traces as one digest), and the recorded chunkings.
//!
//! `RESPONSE_TEMPLATE_PARITY_REPORT=<path>` writes per-template tallies.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "a test: a bad fixture should panic"
)]

mod common;

use std::{
    collections::{BTreeMap, HashMap},
    thread,
};

use common::{
    canonical, chunk_lengths, digest_traces, fixtures_dir, read_json, read_jsonl, split_chars, tag,
    untag,
};
use serde_json::{json, Value};
use smg_response_template::{
    load_response_template, Event, LoadError, ParseError, PyErrorKind, ResponseParser,
    ResponseTemplate,
};

/// The transformers release the crate reproduces; the fixtures record the
/// release they were taken from.
const TRANSFORMERS_VERSION: &str = "5.17.0";

/// `tests/unsupported.tsv`: template id to the reason this crate refuses it.
fn unsupported() -> BTreeMap<String, String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/unsupported.tsv");
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let (id, reason) = l.split_once('\t').unwrap();
            (id.to_owned(), reason.to_owned())
        })
        .collect()
}

fn events_json(events: &[Event]) -> Value {
    tag(&serde_json::to_value(events).unwrap())
}

/// The class transformers raises, or the name of the explicit error.
fn class(e: &ParseError) -> String {
    match e {
        ParseError::Content { kind, .. } => kind.to_string(),
        ParseError::MissingRequired(_) => "ValueError".to_owned(),
        ParseError::Unrepresentable(_) => "Unrepresentable".to_owned(),
        other => panic!("unexpected error {other}"),
    }
}

/// An error that ends a trace: after it the port's state is not transformers'.
fn ends_trace(e: &ParseError) -> bool {
    matches!(
        e,
        ParseError::Unrepresentable(_)
            | ParseError::Content {
                kind: PyErrorKind::Recursion,
                ..
            }
    )
}

/// The trace of one session, in the generator's format.
fn run_trace(template: &ResponseTemplate, prefix: &str, tools: &[Value], feeds: &[&str]) -> Value {
    let mut parser = match ResponseParser::new(template, prefix, tools) {
        Ok(p) => p,
        Err(e) => return json!([["init_error", class(&e)]]),
    };
    let mut steps = vec![json!(["init", events_json(parser.initial_events())])];
    for chunk in feeds {
        match parser.feed(chunk) {
            Ok(events) => steps.push(json!(["feed", events_json(&events)])),
            Err(e) => {
                steps.push(json!(["feed_error", class(&e)]));
                if ends_trace(&e) {
                    return Value::Array(steps);
                }
            }
        }
    }
    steps.push(match parser.finalize() {
        Ok((message, events)) => {
            json!(["final", tag(&Value::Object(message)), events_json(&events)])
        }
        Err(e) => json!(["final_error", class(&e)]),
    });
    Value::Array(steps)
}

fn final_of(trace: &Value) -> &Value {
    trace.as_array().unwrap().last().unwrap()
}

struct Loaded {
    templates: HashMap<String, Result<ResponseTemplate, LoadError>>,
    sources: HashMap<String, Value>,
    tools: HashMap<String, Vec<Value>>,
}

fn load_all() -> Loaded {
    let dir = fixtures_dir();
    let mut templates = HashMap::new();
    let mut sources = HashMap::new();
    for row in read_jsonl(&dir, "templates") {
        let id = row["id"].as_str().unwrap().to_owned();
        templates.insert(id.clone(), load_response_template(&untag(&row["template"])));
        sources.insert(id, row);
    }
    let tools = read_json(&dir, "tools.json")
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), untag(v).as_array().unwrap().clone()))
        .collect();
    Loaded {
        templates,
        sources,
        tools,
    }
}

#[test]
fn fixtures_record_this_transformers_version() {
    let meta = read_json(&fixtures_dir(), "meta.json");
    if let Some(version) = meta.get("transformers") {
        assert_eq!(
            version, TRANSFORMERS_VERSION,
            "fixtures from another transformers"
        );
    }
}

#[test]
fn templates_load_like_transformers() {
    let loaded = load_all();
    let listed = unsupported();
    let mut failures = Vec::new();
    let mut unsupported_seen = BTreeMap::new();
    let mut ids: Vec<&String> = loaded.templates.keys().collect();
    ids.sort();
    for id in ids {
        let hf_ok = loaded.sources[id]["hf"].get("ok").is_some();
        let expected = listed.get(id).map(String::as_str);
        match (&loaded.templates[id], hf_ok) {
            (Ok(_), true) => {
                if let Some(feature) = expected {
                    failures.push(format!("{id}: loads now; unlist it ({feature})"));
                }
            }
            (Ok(_), false) => {
                failures.push(format!("{id}: transformers rejects it, it loads here"));
            }
            (Err(LoadError::Invalid { message, .. }), true) => {
                failures.push(format!(
                    "{id}: transformers accepts it, rejected here: {message}"
                ));
            }
            (Err(LoadError::Unsupported { feature, .. }), true) => {
                let feature = feature.to_string();
                if expected != Some(feature.as_str()) {
                    failures.push(format!(
                        "{id}: unsupported {feature}, expected {expected:?}"
                    ));
                }
                unsupported_seen.insert(id.clone(), feature);
            }
            (Err(_), false) => {
                if let Some(feature) = expected {
                    failures.push(format!(
                        "{id}: transformers rejects it; unlist it ({feature})"
                    ));
                }
            }
            (Err(e), true) => failures.push(format!("{id}: unexpected error {e}")),
        }
    }
    // Local fixture directories (RESPONSE_TEMPLATE_HF_FIXTURES) hold other templates.
    if std::env::var_os("RESPONSE_TEMPLATE_HF_FIXTURES").is_none() {
        for id in listed.keys() {
            if !loaded.templates.contains_key(id) {
                failures.push(format!("{id}: listed as unsupported but not recorded"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} load mismatches:\n{}\nunsupported seen: {unsupported_seen:#?}",
        failures.len(),
        failures.join("\n")
    );
}

/// The outcome of one session.
#[derive(Default, Clone)]
struct Outcome {
    unary: bool,
    split2: Option<bool>,
    chunkings: bool,
    notes: Vec<String>,
}

fn check_case(case: &Value, template: &ResponseTemplate, tools: &[Value]) -> Outcome {
    let text = case["text"].as_str().unwrap();
    let prefix = case["prefix"].as_str().unwrap_or("");
    let mut out = Outcome::default();

    let unary = run_trace(template, prefix, tools, &[text]);
    let expected_final = &case["unary"]["final"];
    let final_ok = canonical(final_of(&unary)) == canonical(expected_final);
    let digest_ok = digest_traces(std::slice::from_ref(&unary)) == case["unary"]["digest"];
    out.unary = final_ok && digest_ok;
    if !out.unary {
        out.notes.push(format!(
            "unary: got {}\n  expected final {}\n  expected trace {}",
            canonical(&unary),
            canonical(expected_final),
            case["unary"]
                .get("trace")
                .map(canonical)
                .unwrap_or_default()
        ));
    }

    if let Some(split2) = case.get("split2") {
        let bounds: Vec<usize> = text.char_indices().map(|(b, _)| b).skip(1).collect();
        let mut traces = Vec::with_capacity(bounds.len());
        let mut finals_ok = true;
        for (k, b) in bounds.iter().enumerate() {
            let trace = run_trace(template, prefix, tools, &[&text[..*b], &text[*b..]]);
            let expected = split2["final_differs"]
                .get((k + 1).to_string())
                .unwrap_or(expected_final);
            if canonical(final_of(&trace)) != canonical(expected) {
                if finals_ok {
                    out.notes.push(format!(
                        "split at char {}: got {}, expected {}",
                        k + 1,
                        canonical(final_of(&trace)),
                        canonical(expected)
                    ));
                }
                finals_ok = false;
            }
            traces.push(trace);
        }
        let digest_ok = digest_traces(&traces) == split2["digest"];
        if finals_ok && !digest_ok {
            out.notes.push("split2: events differ (digest)".into());
        }
        out.split2 = Some(finals_ok && digest_ok);
    }

    let chunkings = &case["chunkings"];
    let mut traces = Vec::new();
    let mut finals_ok = true;
    for (k, spec) in chunkings["feeds"].as_array().unwrap().iter().enumerate() {
        let lengths = chunk_lengths(text, spec.as_str().unwrap());
        let chunks = split_chars(text, &lengths);
        let trace = run_trace(template, prefix, tools, &chunks);
        let expected = chunkings["final_differs"]
            .get(k.to_string())
            .unwrap_or(expected_final);
        if canonical(final_of(&trace)) != canonical(expected) {
            if finals_ok {
                out.notes.push(format!(
                    "chunking {spec}: got {}, expected {}",
                    canonical(final_of(&trace)),
                    canonical(expected)
                ));
            }
            finals_ok = false;
        }
        traces.push(trace);
    }
    let digest_ok = digest_traces(&traces) == chunkings["digest"];
    if finals_ok && !digest_ok {
        out.notes.push("chunkings: events differ (digest)".into());
    }
    out.chunkings = finals_ok && digest_ok;
    out
}

/// Tallies per template: sessions replayed, and those that matched in one
/// feed, in every two-way split, and in every recorded chunking.
#[derive(Default)]
struct Tally {
    sessions: usize,
    unary: usize,
    split2_cases: usize,
    split2: usize,
    chunkings: usize,
    skipped_unsupported: usize,
}

fn replay(name: &str) {
    let loaded = load_all();
    let cases = read_jsonl(&fixtures_dir(), name);
    let workers = thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(32);
    let chunk = cases.len().div_ceil(workers).max(1);
    let results: Vec<(String, String, Option<Outcome>)> = thread::scope(|scope| {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|part| {
                let loaded = &loaded;
                scope.spawn(move || {
                    part.iter()
                        .map(|case| {
                            let id = case["id"].as_str().unwrap().to_owned();
                            let tid = case["template"].as_str().unwrap().to_owned();
                            let outcome = loaded.templates[&tid].as_ref().ok().map(|template| {
                                let tools = match case.get("tools") {
                                    Some(Value::String(set)) => loaded.tools[set].clone(),
                                    _ => Vec::new(),
                                };
                                check_case(case, template, &tools)
                            });
                            (id, tid, outcome)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });

    let mut tallies: BTreeMap<String, Tally> = BTreeMap::new();
    let mut failures = Vec::new();
    for (id, tid, outcome) in &results {
        let tally = tallies.entry(tid.clone()).or_default();
        let Some(outcome) = outcome else {
            tally.skipped_unsupported += 1;
            continue;
        };
        tally.sessions += 1;
        tally.unary += usize::from(outcome.unary);
        if let Some(ok) = outcome.split2 {
            tally.split2_cases += 1;
            tally.split2 += usize::from(ok);
        }
        tally.chunkings += usize::from(outcome.chunkings);
        if !outcome.unary || outcome.split2 == Some(false) || !outcome.chunkings {
            failures.push(format!("{id}:\n  {}", outcome.notes.join("\n  ")));
        }
    }
    if let Some(path) = std::env::var_os("RESPONSE_TEMPLATE_PARITY_REPORT") {
        let report: BTreeMap<&String, Value> = tallies
            .iter()
            .map(|(tid, t)| {
                let tier = match &loaded.templates[tid] {
                    Ok(_) => "exact".to_owned(),
                    Err(LoadError::Unsupported { feature, .. }) => {
                        format!("unsupported: {feature}")
                    }
                    Err(e) => format!("error: {e}"),
                };
                (
                    tid,
                    json!({
                        "tier": tier, "sessions": t.sessions, "unary": t.unary,
                        "split2_cases": t.split2_cases, "split2": t.split2,
                        "chunkings": t.chunkings, "skipped_unsupported": t.skipped_unsupported,
                    }),
                )
            })
            .collect();
        let path = std::path::PathBuf::from(path).with_extension(format!("{name}.json"));
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
    assert!(
        failures.is_empty(),
        "{} of {} sessions differ from transformers:\n{}",
        failures.len(),
        results.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn sessions_replay_like_transformers() {
    replay("sessions");
}

#[test]
fn random_sessions_replay_like_transformers() {
    replay("random");
}
