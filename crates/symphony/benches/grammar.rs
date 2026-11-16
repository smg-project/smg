//! What a required call's constraint costs: the grammar over the request's tools and its text for
//! the engine's compiler, which a gateway builds once per request whose `tool_choice` requires a
//! call (`required`; a named choice sets `one_call`), from where the subject's prompt leaves the
//! model. One subject per call syntax, the prompts of `throughput.rs`, the tools a request
//! declares.
//!
//! A bench target of its own, beside `throughput`: that binary's prose and parse rows are gated
//! (`.github/bench-gates/symphony.toml`), and with thin LTO in one codegen unit the inlining of
//! the crate's hot path inside it follows the binary's set of callers, so linking `grammar_for`
//! and its `Format` callers into the throughput binary read its prose rows 47 % slower with the
//! crate's parser untouched (2026-11-08). The grammar rows live here, where they change nothing
//! they do not measure.
//!
//! `cargo bench -p smg-symphony --bench grammar`; one core, release: the row per subject is the
//! grammar's construction and its text, in microseconds.
#![allow(clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use openai_protocol::common::{Function, Tool};
use serde_json::json;
use symphony::{
    formats::Table,
    grammar::{GrammarStart, Options},
};

/// One table under measurement and the prompt tail that places the output, as `throughput.rs`
/// spells them.
const SUBJECTS: &[(Table, &str)] = &[
    (Table::Qwen3, "<|im_start|>assistant\n"),
    (Table::Qwen3Tagged, "<|im_start|>assistant\n"),
    (Table::Glm, "<|assistant|><think>"),
    (Table::DeepSeekV4_1, "<｜Assistant｜><think>"),
    (
        Table::KimiK3,
        "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>",
    ),
    (Table::MinimaxM3, "]~b]ai\n"),
    (Table::Gemma4, "<|turn>model\n"),
    (Table::Lfm2_5, "<|im_start|>assistant\n"),
    (Table::Json, ""),
];

/// The request's tools, the two `throughput.rs` declares.
fn tools() -> Vec<Tool> {
    let tool = |name: &str, parameters: serde_json::Value| Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters: Some(parameters),
            strict: None,
            extra: Default::default(),
        },
    };
    vec![
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
    ]
}

/// The required call's grammar and its text per subject: `grammar_for` over the tools with the
/// options a gateway passes for `tool_choice: required`, from where the prompt leaves the model,
/// then `payload_text`.
fn grammar(c: &mut Criterion) {
    let tools = tools();
    let mut group = c.benchmark_group("grammar");
    for (table, prompt) in SUBJECTS {
        let format = table.format();
        let options = Options {
            at_least_one: true,
            reasoning_open: format.reasoning_open(prompt),
            thought_closed: format.thought_closed(prompt),
            one_call: false,
            start: GrammarStart::Unknown,
        };
        group.bench_function(BenchmarkId::new("required", table.name()), |b| {
            b.iter(|| {
                black_box(
                    format
                        .grammar_for(&tools, options)
                        .expect("every subject builds a required call's grammar")
                        .payload_text(),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, grammar);
criterion_main!(benches);
