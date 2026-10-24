//! The crate's public surface, pinned: the modules the root declares public and the names it
//! re-exports are what the gateway, the bindings and these tests build on. A change to them is a
//! change of the contract, made here on purpose.

use symphony::{
    declared::Declared,
    engine::Engine,
    event::{DropReason, Event, Events, FinishReason, MalformedReason, Text},
    format::{CallId, CallSyntax, Emits, Format, Whitespace},
    grammar::{
        Bound, Grammar, GrammarStart, Output, OutputAnswer, OutputChoice, OutputPosition,
        OutputSpelling, Tag, Turn,
    },
    input::{EngineFinish, Input, RecordedInput, TokenSpan},
    parser::{ParseError, Parser},
    shape::{Inferred, ValueRead},
};

const ROOT: &str = include_str!("../../src/lib.rs");

/// The modules the root declares public, in the root's order.
const MODULES: [&str; 13] = [
    "adapt", "declared", "engine", "event", "format", "formats", "gemma", "grammar", "input",
    "json", "parser", "shape", "tagged",
];

/// The names the root re-exports, by the module they come from, in the root's order.
const RE_EXPORTS: [(&str, &[&str]); 9] = [
    ("declared", &["Declared"]),
    ("engine", &["Engine"]),
    (
        "event",
        &[
            "DropReason",
            "Event",
            "Events",
            "FinishReason",
            "MalformedReason",
            "Text",
        ],
    ),
    (
        "format",
        &["CallId", "CallSyntax", "Emits", "Format", "Whitespace"],
    ),
    (
        "grammar",
        &[
            "Bound",
            "Grammar",
            "GrammarStart",
            "Output",
            "OutputAnswer",
            "OutputChoice",
            "OutputPosition",
            "OutputSpelling",
            "Tag",
            "Turn",
        ],
    ),
    ("ids", &["CallIds"]),
    (
        "input",
        &["EngineFinish", "Input", "RecordedInput", "TokenSpan"],
    ),
    ("parser", &["ParseError", "Parser"]),
    ("shape", &["Inferred", "ValueRead"]),
];

/// The `pub mod` lines of the root, in order.
fn public_modules() -> Vec<&'static str> {
    ROOT.lines()
        .filter_map(|line| line.strip_prefix("pub mod "))
        .map(|rest| rest.trim_end_matches(';'))
        .collect()
}

/// The `pub use` items of the root, in order: the module and the names, braces and line breaks
/// read through.
fn re_exports() -> Vec<(&'static str, Vec<&'static str>)> {
    let mut out = Vec::new();
    let mut rest = ROOT;
    while let Some(at) = rest.find("\npub use ") {
        let item = &rest[at + "\npub use ".len()..];
        let Some(end) = item.find(';') else { break };
        let Some((module, names)) = item[..end].split_once("::") else {
            break;
        };
        let names = names
            .trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace())
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .collect();
        out.push((module, names));
        rest = &item[end..];
    }
    out
}

#[test]
fn the_root_declares_public_exactly_the_modules_this_test_lists() {
    assert_eq!(public_modules(), MODULES);
}

#[test]
fn the_root_re_exports_exactly_the_names_this_test_lists() {
    let pinned: Vec<(&str, Vec<&str>)> = RE_EXPORTS
        .iter()
        .map(|(module, names)| (*module, names.to_vec()))
        .collect();
    assert_eq!(re_exports(), pinned);
}

/// Every pinned name resolves under its module (the imports above), and the engine is the
/// parser the contract names.
#[test]
fn every_pinned_name_resolves_under_its_module() {
    fn named<T: ?Sized>() {}
    fn parses<P: Parser>() {}
    fn reads<R: ValueRead>() {}
    named::<Declared>();
    named::<(
        DropReason,
        Event,
        Events,
        FinishReason,
        MalformedReason,
        Text,
    )>();
    named::<(CallId, CallSyntax, Emits, Format, Whitespace)>();
    named::<(
        Bound,
        Grammar,
        GrammarStart,
        Output,
        OutputAnswer,
        OutputChoice,
    )>();
    named::<(OutputPosition, OutputSpelling, Tag, Turn)>();
    named::<(EngineFinish, Input, RecordedInput, TokenSpan, ParseError)>();
    named::<Inferred>();
    reads::<Inferred>();
    parses::<Engine>();
}

/// The re-exported names resolve at the root as well, where nothing is imported.
mod at_the_root {
    fn named<T: ?Sized>() {}

    #[test]
    fn every_pinned_name_resolves_at_the_root() {
        named::<symphony::Declared>();
        named::<symphony::Engine>();
        named::<(
            symphony::DropReason,
            symphony::Event,
            symphony::Events,
            symphony::FinishReason,
            symphony::MalformedReason,
            symphony::Text,
        )>();
        named::<(
            symphony::CallId,
            symphony::CallSyntax,
            symphony::Emits,
            symphony::Format,
            symphony::Whitespace,
        )>();
        named::<(
            symphony::Bound,
            symphony::Grammar,
            symphony::GrammarStart,
            symphony::Output,
            symphony::OutputAnswer,
            symphony::OutputChoice,
            symphony::OutputPosition,
            symphony::OutputSpelling,
            symphony::Tag,
            symphony::Turn,
        )>();
        named::<(
            symphony::EngineFinish,
            symphony::Input,
            symphony::RecordedInput,
            symphony::TokenSpan,
        )>();
        named::<(symphony::ParseError, symphony::Inferred)>();
        fn parses<P: symphony::Parser>() {}
        parses::<symphony::Engine>();
        fn reads<R: symphony::ValueRead>() {}
        reads::<symphony::Inferred>();
        named::<symphony::CallIds>();
    }
}
