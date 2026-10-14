//! Layouts learned from public chat templates, over the renderings `fixtures/chat_templates/`
//! records with transformers 5.18.0.
//!
//! - The probe in code is the one the renderings were recorded with (`probe.json`): its tools and
//!   each case's conversation, byte for byte as JSON.
//! - Tables spelled from two public response templates whose calls are tagged parameters, as
//!   symphony's own tests spell them and with this crate's value reader, learn from their chat
//!   templates' renderings the layouts symphony's tests write by hand for them: Muse Glimmer
//!   (`meta-models/Muse-Glimmer-30B`) and `transformers serve`'s Qwen 3.5 template, with
//!   `Qwen/Qwen3.5-9B`'s chat template.
//! - Every pair of a response template and a chat template the recordings' manifest names is
//!   compiled and, where the template compiles, its layout learned in every setting recorded.
//!   `layouts.tsv` pins the outcomes (`TO_SYMPHONY_WRITE=1` rewrites it). A pair whose template
//!   compiles has a layout in the default setting where the manifest says it should, and one of a
//!   single call where it says so.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "a test: a bad fixture or table should panic"
)]

mod common;
mod replay;

use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use common::{read_json, read_jsonl};
use openai_protocol::common::Tool;
use replay::{compiled, compiled_dir, pinned, rows};
use serde_json::{json, Value};
use smg_response_template::to_symphony::{
    layout::{self, learn, Case, Learned, NoLayout},
    Transformers,
};
use symphony::{
    grammar::Options,
    shape::{
        self, Arguments, Atom, CallLayout, CallShape, Class, Dialect, End, Layout, Message,
        NameSource, Nested, Pattern, Segment, Slot, Spellings, ValueParser, Words, Written,
    },
    CallSyntax, Emits, Format,
};

fn chat_templates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chat_templates")
}

/// The rendering of `case` by `template` in `setting`, as recorded.
fn rendering(renders: &[Value], template: &str, setting: &str, case: Case) -> Option<String> {
    renders
        .iter()
        .find(|row| {
            row["template"] == template
                && row["setting"] == setting
                && row["case"] == case.name()
                && row.get("text").is_some()
        })
        .map(|row| row["text"].as_str().unwrap().to_owned())
}

/// The stop texts recorded for a public response template of `fixtures/compiled/`.
fn stops_of(id: &str) -> Vec<String> {
    rows(&compiled_dir())
        .into_iter()
        .find(|row| row.id == id)
        .unwrap_or_else(|| panic!("{id} is not recorded"))
        .stops
}

#[test]
fn the_probe_is_the_one_the_renderings_were_recorded_with() {
    let probe = read_json(&chat_templates_dir(), "probe.json");
    assert_eq!(probe["tools"].to_string(), layout::tools().to_string());
    let cases = probe["cases"].as_object().unwrap();
    for case in Case::ALL {
        let recorded = &cases[&case.name()];
        assert_eq!(
            recorded["messages"].to_string(),
            case.messages().to_string(),
            "{case}"
        );
        assert_eq!(
            recorded["add_generation_prompt"],
            json!(case.generation_prompt()),
            "{case}"
        );
    }
    // The one recorded case the layout does not read: the prompt after a tool's result.
    let unread: Vec<&String> = cases
        .keys()
        .filter(|name| Case::ALL.iter().all(|case| case.name() != **name))
        .collect();
    assert_eq!(unread, ["prompt_after_tool"]);
}

/// A class of Python `regex`'s, as symphony's shape recordings hold it (`word` for `\w`, `space`
/// for `\s`): the class a compile resolves the escape into.
fn recorded_class(table: &str) -> Class {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../symphony/tests/fixtures/shape/classes.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let tables: Value = serde_json::from_str(&text).unwrap();
    let ranges = tables[table].as_array().unwrap().iter().map(|range| {
        let mut chars = range.as_str().unwrap().chars();
        match (chars.next(), chars.next(), chars.next()) {
            (Some(first), Some(last), None) => (first, last),
            _ => panic!("a range is two characters: {range}"),
        }
    });
    Class::new(ranges, false)
}

fn run(class: Class, min: u16, greedy: bool, capture: Option<Slot>) -> Atom {
    Atom::Run {
        class,
        min,
        max: None,
        greedy,
        capture,
    }
}

fn none_of(chars: &str) -> Class {
    Class::new(chars.chars().map(|c| (c, c)), true)
}

fn lit(text: &str) -> Atom {
    Atom::Lit(text.into())
}

/// The response template of `meta-models/Muse-Glimmer-30B` (`hub/muse-glimmer` in
/// `fixtures/compiled/`), spelled as symphony's tests spell it: the turn after
/// `<|start|>assistant`, a thought to the model itself and an answer to the user, both stripped,
/// nothing kept outside them; a call's open `<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">` is the
/// terminal `<atem:invoke` and a header that names the tool, its close `</atem:invoke>`, and its
/// parameters `<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>`,
/// each value read as JSON where it is JSON.
fn muse() -> Format {
    let shape = shape::intern(CallShape {
        header: Pattern(
            [
                Atom::Boundary,
                run(none_of(">"), 0, false, None),
                Atom::Boundary,
                lit("name=\""),
                run(none_of("\""), 1, true, Some(Slot::Name)),
                lit("\">"),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Params {
            tag: Pattern(
                [
                    lit("<atem:parameter"),
                    Atom::Boundary,
                    run(none_of(">"), 0, false, None),
                    Atom::Boundary,
                    lit("name=\""),
                    run(none_of("\""), 1, true, Some(Slot::Key)),
                    lit("\""),
                    run(none_of(">"), 0, false, None),
                    lit(">"),
                    run(Class::any(), 0, false, Some(Slot::Value)),
                ]
                .into(),
            ),
            ends: [End {
                consume: Pattern([lit("</atem:parameter>")].into()),
                ahead: Pattern::default(),
            }]
            .into(),
            values: ValueParser::Json {
                dialect: Dialect::default(),
                allow_non_json: true,
                strip: true,
            },
        },
        trailer: Pattern::default(),
        word: recorded_class("word"),
        hold_cap: 8192,
    });
    Format::new("muse glimmer template")
        .terminal("thought", "to=self<|message|>")
        .terminal("answer", "to=user<|message|>")
        .terminal("eot", "<|eot|>")
        .terminal("eom", "<|eom|>")
        .terminal("call_open", "<atem:invoke")
        .terminal("call_close", "</atem:invoke>")
        .state("outside", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("call", Emits::Arguments)
        .transition("outside", "thought", "reasoning")
        .transition("outside", "answer", "content")
        .transition("reasoning", "eom", "outside")
        .transition("content", "eot", "outside")
        .transition("content", "eom", "outside")
        .transition("outside", "call_open", "call")
        .transition("call", "call_close", "outside")
        .calls(CallSyntax::Shaped(shape))
        .framing("outside")
        .turn_anchor("<|start|>assistant")
        .trims("reasoning")
        .trims("content")
        .stop_tokens(["<|end_of_text|>", "<|eot|>"])
        .reads_values_with(Arc::new(Transformers))
}

/// transformers v5.18.0's `transformers serve` template for Qwen 3.5 (`serve/qwen3_5` in
/// `fixtures/compiled/`), spelled as symphony's tests spell it: the turn after
/// `<|im_start|>assistant\n`, a thought between `<think>` and `</think>`, and the content outside
/// them, stripped; a call's open `\s*<tool_call>\s*<function=(?P<name>[^>\n]+)>` is the terminal
/// `<tool_call>` and a header that names the tool, its close `</function>\s*</tool_call>` the
/// trailer `</function>\s*` and the terminal `</tool_call>`, and its parameters
/// `<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>`, each value its text, typed by the
/// request's tools.
fn serve_qwen3_5() -> Format {
    let space = recorded_class("space");
    let name = none_of(">\n");
    let shape = shape::intern(CallShape {
        header: Pattern(
            [
                run(space.clone(), 0, true, None),
                lit("<function="),
                run(name.clone(), 1, true, Some(Slot::Name)),
                lit(">"),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Params {
            tag: Pattern(
                [
                    lit("<parameter="),
                    run(name, 1, true, Some(Slot::Key)),
                    lit(">"),
                    run(space.clone(), 0, true, None),
                    run(Class::any(), 0, false, Some(Slot::Value)),
                ]
                .into(),
            ),
            ends: [End {
                consume: Pattern([run(space.clone(), 0, true, None), lit("</parameter>")].into()),
                ahead: Pattern::default(),
            }]
            .into(),
            values: ValueParser::Raw,
        },
        trailer: Pattern([lit("</function>"), run(space, 0, true, None)].into()),
        word: recorded_class("word"),
        hold_cap: 8192,
    });
    Format::new("qwen 3.5 serve template")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("call_open", "<tool_call>")
        .terminal("call_close", "</tool_call>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "call")
        .transition("call", "call_close", "content")
        .calls(CallSyntax::Shaped(shape))
        .turn_anchor("<|im_start|>assistant\n")
        .trims("content")
        .trims("reasoning")
        .stop_tokens(["<|im_end|>", "<|endoftext|>"])
        .reads_values_with(Arc::new(Transformers))
}

fn written(segments: impl IntoIterator<Item = Segment>) -> Written {
    Written(segments.into_iter().collect())
}

fn piece(text: &str) -> Segment {
    Segment::Lit(text.into())
}

/// The layout symphony's tests write by hand for Muse Glimmer, from its chat template's
/// rendering: a call is a message to the tool, ` to=NAME` and `<|message|>`, with
/// `<atem:function_calls>` on a line of its own; the invoke's one attribute is the name; a
/// parameter a line; between two calls the block's close, `<|eom|>` and the next message's start;
/// after the last, the block's close. The thought and the answer are messages to the model itself
/// and to the user, a space before the recipient; the answer ends at the stop.
fn muse_layout() -> Layout {
    Layout {
        call: CallLayout {
            lead: written([
                piece(" to="),
                Segment::Slot(Slot::Name),
                piece("<|message|><atem:function_calls>\n"),
            ]),
            header: written([piece(" name=\""), Segment::Slot(Slot::Name), piece("\">")]),
            before: "\n".into(),
            parameter: written([
                piece("<atem:parameter name=\""),
                Segment::Slot(Slot::Key),
                piece("\">"),
                Segment::Slot(Slot::Value),
                piece("</atem:parameter>"),
            ]),
            between: "\n".into(),
            after: "\n".into(),
            trailer: "".into(),
            close: Some("</atem:invoke>".into()),
        },
        between_calls: Some("\n</atem:function_calls><|eom|><|start|>assistant".into()),
        after_calls: "\n</atem:function_calls>".into(),
        values: Spellings::default(),
        thought: Some(Message {
            before: " ".into(),
            after: "<|start|>assistant".into(),
            stops: false,
            close: Some("<|eom|>".into()),
        }),
        answer: Some(Message {
            before: " ".into(),
            after: "".into(),
            stops: true,
            close: None,
        }),
    }
}

/// The layout symphony's tests write by hand for Qwen 3.5: `<tool_call>`, the function's tag and
/// each parameter's tag, value and close on lines of their own, `True` and `None` as Python writes
/// them, a newline between two calls, nothing after the last but the stop; the thought's close
/// followed by a blank line.
fn serve_qwen3_5_layout() -> Layout {
    Layout {
        call: CallLayout {
            lead: Written::default(),
            header: written([piece("\n<function="), Segment::Slot(Slot::Name), piece(">")]),
            before: "\n".into(),
            parameter: written([
                piece("<parameter="),
                Segment::Slot(Slot::Key),
                piece(">\n"),
                Segment::Slot(Slot::Value),
                piece("\n</parameter>"),
            ]),
            between: "\n".into(),
            after: "\n".into(),
            trailer: "</function>\n".into(),
            close: Some("</tool_call>".into()),
        },
        between_calls: Some("\n".into()),
        after_calls: "".into(),
        values: Spellings {
            words: Words::Python,
            nested: Nested::Json,
        },
        thought: Some(Message {
            before: "".into(),
            after: "\n\n".into(),
            stops: false,
            close: Some("</think>".into()),
        }),
        answer: Some(Message {
            before: "".into(),
            after: "".into(),
            stops: true,
            close: None,
        }),
    }
}

/// The layout `format` learns from `template`'s renderings in `setting`, with the stops of the
/// response template `id`.
fn learned(format: &Format, id: &str, template: &str, setting: &str) -> Result<Learned, NoLayout> {
    let renders = read_jsonl(&chat_templates_dir(), "renders");
    let stops = stops_of(id);
    let stops: Vec<&str> = stops.iter().map(String::as_str).collect();
    learn(format, &stops, |case| {
        rendering(&renders, template, setting, case)
    })
}

/// That a table with `layout` has a tag for a forced call, a named function, one call and a JSON
/// answer, with the prompt's thought open and not.
fn every_mode_has_a_tag(format: &Format, layout: Layout) {
    let format = format.clone().with_layout(layout);
    let tools: Vec<Tool> = serde_json::from_value(layout::tools()).unwrap();
    for reasoning_open in [false, true] {
        for (at_least_one, one_call) in [(true, false), (true, true), (false, true)] {
            let options = Options {
                at_least_one,
                reasoning_open,
                one_call,
            };
            assert!(format.grammar_for(&tools, options).is_some(), "{options:?}");
        }
        let answer = format.answer_grammar(&json!({"type": "object"}), reasoning_open);
        assert!(answer.is_some(), "{reasoning_open}");
    }
}

#[test]
fn muse_glimmers_layout_is_the_one_its_chat_template_renders() {
    let format = muse();
    let learned = learned(
        &format,
        "hub/muse-glimmer",
        "meta-models/Muse-Glimmer-30B",
        "default",
    )
    .unwrap();
    assert!(!learned.thinks_first);
    assert_eq!(learned.layout, muse_layout());
    every_mode_has_a_tag(&format, learned.layout);
}

#[test]
fn qwen3_5s_layout_is_the_one_its_chat_template_renders() {
    let format = serve_qwen3_5();
    let on = learned(&format, "serve/qwen3_5", "Qwen/Qwen3.5-9B", "default").unwrap();
    // The prompt opens the thought, so the turn's thought is the prompt's: no thinking first.
    assert!(!on.thinks_first);
    assert_eq!(on.layout, serve_qwen3_5_layout());
    every_mode_has_a_tag(&format, on.layout);
    // With thinking turned off, the prompt closes an empty thought and no turn writes the probe's:
    // the calls and the answer are written alike, and the thought has no layout.
    let off = learned(
        &format,
        "serve/qwen3_5",
        "Qwen/Qwen3.5-9B",
        "disable_thinking",
    )
    .unwrap();
    assert!(!off.thinks_first);
    assert_eq!(
        off.layout,
        Layout {
            thought: None,
            ..serve_qwen3_5_layout()
        }
    );
}

/// One line for the pinned list: the pair, the setting, and the outcome.
fn outcome(id: &str, template: &str, setting: &str) -> String {
    let row = rows(&compiled_dir())
        .into_iter()
        .find(|row| row.id == id)
        .unwrap_or_else(|| panic!("{id} is not recorded"));
    let what = match compiled(&row) {
        None => "not read".to_owned(),
        Some(Err(refusals)) => {
            let codes: Vec<&str> = refusals
                .constructs()
                .into_iter()
                .map(|construct| construct.code())
                .collect();
            format!("refused\t{}", codes.join(" "))
        }
        Some(Ok(compiled)) => match learned(&compiled.format, id, template, setting) {
            Ok(learned) => format!(
                "layout\t{}{}",
                layout::summary(&learned.layout),
                if learned.thinks_first {
                    "; thinks first"
                } else {
                    ""
                }
            ),
            Err(why) => format!("no layout\t{} ({why})", why.code()),
        },
    };
    format!("{id}\t{template}\t{setting}\t{what}\n")
}

#[test]
fn the_recorded_pairs_compile_and_learn_their_layouts_as_pinned() {
    let manifest = read_json(&chat_templates_dir(), "manifest.json");
    let sources = manifest["sources"].as_array().unwrap();
    let mut seen = String::new();
    for pair in manifest["pairs"].as_array().unwrap() {
        let id = pair["response_template"].as_str().unwrap();
        let template = pair["chat_template"].as_str().unwrap();
        let settings = sources
            .iter()
            .find(|source| source["id"] == template)
            .and_then(|source| source["settings"].as_array())
            .unwrap();
        for setting in settings {
            let setting = setting.as_str().unwrap();
            let line = outcome(id, template, setting);
            if setting == "default" {
                let wanted = pair["layout"].as_bool().unwrap();
                let fields: Vec<&str> = line.trim_end().split('\t').collect();
                if fields[3] != "refused" && fields[3] != "not read" {
                    assert_eq!(fields[3] == "layout", wanted, "{line}");
                }
                if pair["max_calls"] == json!(1) && fields[3] == "layout" {
                    assert!(line.contains("between calls None"), "{line}");
                }
            }
            let _ = write!(seen, "{line}");
        }
    }
    pinned("layouts.tsv", &seen);
}
