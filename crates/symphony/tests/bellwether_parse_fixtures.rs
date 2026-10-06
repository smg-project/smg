//! Parity with bellwether's parse fixtures.
//!
//! bellwether (smg-project/bellwether) records, per model, what a model's output must parse to: the
//! output text, its token ids, chunk plans, and the reference assistant message from the round trip
//! through the checkpoint's template. This test replays every Qwen3-8B parse case through
//! [`Qwen3`], folds the events into the assistant message with [`adapt::chat::message`], and
//! compares it with the reference: whole, at every byte split, byte by byte, and on thirty seeded
//! byte plans. Every replay also has to conserve the output's bytes across its events and agree
//! with every other replay of the same case.
//!
//! The run is opt-in: `BELLWETHER_FIXTURES` points at the `fixtures/` directory of a bellwether
//! checkout; without it the test prints a skip notice and passes.
//!
//! Two policy questions stand between the parser and bitwise parity, and the test declares them
//! rather than hides them. Bellwether #17: the template's separator bytes (the newline after
//! `<think>`, the two after `</think>`) are in the parser's content and reasoning and not in the
//! reference's; a case that matches once the newlines at both ends are trimmed is counted as
//! "separators only", and any other whitespace still counts as a difference. Two of
//! the bitwise cases, the parallel calls, owe their count to the adapter rather than the parser:
//! there the parser's content is nothing but the separator between the calls, and the adapter's
//! rule that whitespace-only content is absent folds the difference away before the comparison.
//! Bellwether #16: two probe cases put marker strings inside text, and the parser reads them as
//! markers like every marker parser; they are listed in [`KNOWN_DIFFERENCES`] with their reason and
//! with the call count and finish reason the parser gives, so the list allows that difference and
//! no other. The run fails on any other difference, when a listed case starts matching, and when a
//! listed case is no longer among the fixtures, so the list cannot rot. Each replay ends with the
//! engine finish the reference implies. The fixtures' token-level chunk plans are not replayed
//! here: that needs token attribution, which will read the pieces bellwether records as
//! `output_pieces`.

mod common;

use std::{fs, path::PathBuf};

use common::{bytes_of, chunkings, replay};
use serde::Deserialize;
use symphony::{adapt, EngineFinish, Event, Qwen3};

const FIXTURES_ENV: &str = "BELLWETHER_FIXTURES";
const SLUG: &str = "qwen3-8b";

/// A case known to differ from the reference beyond the separator bytes: why, and what the parser
/// says instead, its call count and finish reason, so that the list allows that difference and no
/// other.
struct KnownDifference {
    id: &'static str,
    reason: &'static str,
    calls: usize,
    finish: &'static str,
}

const KNOWN_DIFFERENCES: &[KnownDifference] = &[
    KnownDifference {
        id: "qwen3-8b/parse/reasoning-with-marker-text",
        reason: "the reasoning holds a `</think>`; the parser ends the reasoning there, as every marker \
                 parser does, and the reference keeps the marker as reasoning text (bellwether #16)",
        calls: 0,
        finish: "stop",
    },
    KnownDifference {
        id: "qwen3-8b/parse/content-with-marker-in-code-fence",
        reason: "the content holds a complete `<tool_call>` block inside a code fence; the parser makes \
                 a call of it, as every marker parser does, and the reference keeps it as content \
                 (bellwether #16)",
        calls: 1,
        finish: "tool_calls",
    },
];

#[derive(Deserialize)]
struct Fixture {
    id: String,
    reference: Reference,
}

#[derive(Deserialize)]
struct Reference {
    text: String,
    message: ReferenceMessage,
    finish_reason: String,
}

#[derive(Deserialize)]
struct ReferenceMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ReferenceCall>,
}

#[derive(Deserialize)]
struct ReferenceCall {
    function: ReferenceFunction,
}

#[derive(Deserialize)]
struct ReferenceFunction {
    name: String,
    arguments: String,
}

/// What the parser said an output means, in the terms the reference uses.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Said {
    content: Option<String>,
    reasoning: Option<String>,
    calls: Vec<(String, String)>,
    finish: String,
}

impl Said {
    fn of(events: &[Event]) -> Self {
        let choice = adapt::chat::message(0, events);
        Self {
            content: choice.message.content,
            reasoning: choice.message.reasoning_content,
            calls: choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| {
                    (
                        call.function.name,
                        call.function.arguments.unwrap_or_default(),
                    )
                })
                .collect(),
            finish: choice.finish_reason.unwrap_or_default(),
        }
    }

    fn of_reference(reference: &Reference) -> Self {
        Self {
            content: reference.message.content.clone().filter(|c| !c.is_empty()),
            reasoning: reference.message.reasoning_content.clone(),
            calls: reference
                .message
                .tool_calls
                .iter()
                .map(|call| (call.function.name.clone(), call.function.arguments.clone()))
                .collect(),
            finish: reference.finish_reason.clone(),
        }
    }

    /// The same, with the separator bytes trimmed from content and reasoning (bellwether #17). The
    /// template's separators are newlines, so only newlines are trimmed: any other whitespace the
    /// parser kept or lost still counts as a difference.
    fn trimmed(&self) -> Self {
        let trim = |part: &Option<String>| {
            part.as_deref()
                .map(|text| text.trim_matches('\n'))
                .filter(|t| !t.is_empty())
                .map(str::to_string)
        };
        Self {
            content: trim(&self.content),
            reasoning: trim(&self.reasoning),
            calls: self.calls.clone(),
            finish: self.finish.clone(),
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "the skip notice and the per-case report are test diagnostic output"
)]
fn qwen3_parse_fixtures_match_the_reference() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; point it at the fixtures/ directory of a bellwether checkout"
        );
        return;
    };
    let fixtures = read_fixtures(&root.join(SLUG).join("parse")).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        !fixtures.is_empty(),
        "no parse fixtures under {}",
        root.display()
    );

    let known = |id: &str| KNOWN_DIFFERENCES.iter().find(|known| known.id == id);
    let mut failures = Vec::new();
    let (mut bitwise, mut separators_only, mut listed) = (0, 0, 0);
    for fixture in &fixtures {
        let text = fixture.reference.text.as_str();
        let expected = Said::of_reference(&fixture.reference);
        let mut saids = Vec::new();
        let finish = engine_finish(&fixture.reference.finish_reason);
        for plan in chunkings(text) {
            let events = replay(&mut Qwen3::new(), text, &plan, &finish, false)
                .unwrap_or_else(|e| panic!("{}: plan {plan:?}: {e}", fixture.id));
            let conserved: String = events.iter().map(bytes_of).collect();
            if conserved != text {
                failures.push(format!(
                    "{}: bytes not conserved on plan {plan:?}",
                    fixture.id
                ));
            }
            let mut index = 0;
            for event in &events {
                if let Event::ToolCallStart { index: i, id, .. } = event {
                    if *i != index || *id != format!("call_{index}") {
                        failures.push(format!(
                            "{}: call index {i} / id {id} on plan {plan:?}",
                            fixture.id
                        ));
                    }
                    index += 1;
                }
            }
            saids.push(Said::of(&events));
        }
        let said = saids[0].clone();
        if saids.iter().any(|s| *s != said) {
            failures.push(format!(
                "{}: the replays disagree with each other",
                fixture.id
            ));
        }
        let verdict = if said == expected {
            bitwise += 1;
            "bitwise"
        } else if said.trimmed() == expected.trimmed() {
            separators_only += 1;
            "separators only (bellwether #17)"
        } else if let Some(listed_case) = known(&fixture.id) {
            if said.calls.len() != listed_case.calls || said.finish != listed_case.finish {
                failures.push(format!(
                    "{}: listed in KNOWN_DIFFERENCES for {} call(s) and `{}`, but the parser says {} \
                     call(s) and `{}`",
                    fixture.id,
                    listed_case.calls,
                    listed_case.finish,
                    said.calls.len(),
                    said.finish
                ));
            }
            listed += 1;
            "listed (bellwether #16)"
        } else {
            failures.push(format!(
                "{}: differs from the reference\n    said:      {said:?}\n    reference: {expected:?}",
                fixture.id
            ));
            "DIFFERS"
        };
        if known(&fixture.id).is_some() && said.trimmed() == expected.trimmed() {
            failures.push(format!(
                "{}: listed in KNOWN_DIFFERENCES but matching the reference now; remove it",
                fixture.id
            ));
        }
        println!("  {verdict:34} {}", fixture.id);
        if let Some(listed_case) = known(&fixture.id) {
            println!("  {:34} {}", "", listed_case.reason);
        }
    }
    for listed_case in KNOWN_DIFFERENCES {
        if !fixtures.iter().any(|fixture| fixture.id == listed_case.id) {
            failures.push(format!(
                "{}: listed in KNOWN_DIFFERENCES but not among the fixtures; remove it",
                listed_case.id
            ));
        }
    }
    println!(
        "{} cases: {bitwise} bitwise, {separators_only} separators only, {listed} listed",
        fixtures.len()
    );
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The engine's finish for a reference's finish reason. The adapter turns `stop` after a call into
/// `tool_calls`, so only a truncation needs its own engine reason.
fn engine_finish(reference: &str) -> EngineFinish {
    match reference {
        "length" => EngineFinish::Length,
        _ => EngineFinish::Stop,
    }
}

fn read_fixtures(dir: &std::path::Path) -> Result<Vec<Fixture>, String> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();
    let mut fixtures = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file)
            .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
        for (number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let fixture: Fixture = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", file.display(), number + 1))?;
            fixtures.push(fixture);
        }
    }
    Ok(fixtures)
}
