//! What a table costs: the bytes per second of an output fed whole and in small pieces, the bytes
//! per second of prose with no marker in it (the common streaming case), and the construction of
//! an engine, which is the parser's per-request setup. One subject per call syntax, each a turn as
//! the table's own tests spell it: a thought, an answer and two calls, the tools declared as a
//! request declares them. The parse and prose rows time the reading alone: the engine is made
//! before the timed closure and dropped after it, and the setup rows time the making and the drop.
//!
//! `cargo bench -p smg-symphony --bench throughput`; a filter narrows it (`-- parse/whole`).
#![allow(clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use openai_protocol::common::{Function, Tool};
use serde_json::json;
use symphony::{formats::Table, Declared, Engine, EngineFinish, Events, Input, Parser};

/// One table under measurement: the prompt tail that places the output, the output, and the prompt
/// tail that leaves the model in its answer, for the prose rows.
struct Subject {
    table: Table,
    prompt: &'static str,
    output: &'static str,
    content_prompt: &'static str,
}

const SUBJECTS: &[Subject] = &[
    Subject {
        table: Table::Qwen3,
        prompt: "<|im_start|>assistant\n",
        output: concat!(
            "<think>\nThe user wants the weather and a drink.\n</think>\n\n",
            "Let me check both.\n",
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>\n",
            "<tool_call>\n{\"name\": \"change_drink\", \"arguments\": {\"drink_id\": \"latte\", ",
            "\"new_preferences\": {\"size\": \"large\", \"temperature\": \"hot\"}}}\n</tool_call>"
        ),
        content_prompt: "<|im_start|>assistant\n",
    },
    Subject {
        table: Table::Qwen3Tagged,
        prompt: "<|im_start|>assistant\n",
        output: concat!(
            "<think>\nThe user wants the weather and a drink.\n</think>\n\n",
            "Let me check both.\n",
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
            "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n",
            "<tool_call>\n<function=change_drink>\n<parameter=drink_id>\nlatte\n</parameter>\n",
            "<parameter=new_preferences>\n{\"size\": \"large\", \"temperature\": \"hot\"}\n</parameter>\n",
            "</function>\n</tool_call>"
        ),
        content_prompt: "<|im_start|>assistant\n",
    },
    Subject {
        table: Table::Glm,
        prompt: "<|assistant|><think>",
        output: concat!(
            "The user wants the weather and a drink.</think>Let me check both.",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>",
            "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
            "<tool_call>change_drink<arg_key>drink_id</arg_key><arg_value>latte</arg_value>",
            "<arg_key>new_preferences</arg_key><arg_value>",
            "{\"size\": \"large\", \"temperature\": \"hot\"}</arg_value></tool_call>"
        ),
        content_prompt: "<|assistant|>",
    },
    Subject {
        table: Table::DeepSeekV4_1,
        prompt: "<｜Assistant｜><think>",
        output: concat!(
            "The user wants the weather and a drink.</think>\n\nLet me check both.",
            "<｜DSML｜ calls>\n",
            "<｜DSML｜ invoke name=\"get_weather\">\n",
            "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n",
            "<｜DSML｜ invoke name=\"change_drink\">\n",
            "<｜DSML｜ parameter name=\"drink_id\" string=\"true\">latte</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"new_preferences\" string=\"false\">",
            "{\"size\": \"large\", \"temperature\": \"hot\"}</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n",
            "</｜DSML｜ calls>"
        ),
        content_prompt: "<｜Assistant｜></think>",
    },
    Subject {
        table: Table::KimiK3,
        prompt: "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>",
        output: concat!(
            "The user wants the weather and a drink.<|close|>think<|sep|>",
            "<|open|>response<|sep|>Let me check both.<|close|>response<|sep|>",
            "<|open|>tools<|sep|><|open|>call tool=\"get_weather\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"city\" type=\"string\"<|sep|>Paris<|close|>argument<|sep|>",
            "<|open|>argument key=\"days\" type=\"number\"<|sep|>3<|close|>argument<|sep|>",
            "<|close|>call<|sep|><|open|>call tool=\"change_drink\" index=\"2\"<|sep|>",
            "<|open|>argument key=\"drink_id\" type=\"string\"<|sep|>latte<|close|>argument<|sep|>",
            "<|open|>argument key=\"new_preferences\" type=\"object\"<|sep|>",
            "{\"size\": \"large\", \"temperature\": \"hot\"}<|close|>argument<|sep|>",
            "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
        ),
        content_prompt: "<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>",
    },
    Subject {
        table: Table::MinimaxM3,
        prompt: "]~b]ai\n",
        output: concat!(
            "<mm:think>The user wants the weather and a drink.</mm:think>Let me check both.",
            "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">",
            "]<]minimax[>[<city>Paris]<]minimax[>[</city>]<]minimax[>[<days>3]<]minimax[>[</days>",
            "]<]minimax[>[</invoke>\n]<]minimax[>[<invoke name=\"change_drink\">",
            "]<]minimax[>[<drink_id>latte]<]minimax[>[</drink_id>]<]minimax[>[<new_preferences>",
            "]<]minimax[>[<size>large]<]minimax[>[</size>]<]minimax[>[<temperature>hot",
            "]<]minimax[>[</temperature>]<]minimax[>[</new_preferences>]<]minimax[>[</invoke>\n",
            "]<]minimax[>[</tool_call>"
        ),
        content_prompt: "]~b]ai\n",
    },
    Subject {
        table: Table::Gemma4,
        prompt: "<|turn>model\n",
        output: concat!(
            "<|channel>thought\nThe user wants the weather and a drink.\n<channel|>",
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3}<tool_call|>",
            "<|tool_call>call:change_drink{drink_id:<|\"|>latte<|\"|>,",
            "new_preferences:{size:<|\"|>large<|\"|>,temperature:<|\"|>hot<|\"|>}}<tool_call|>",
            "Let me check both."
        ),
        content_prompt: "<|turn>model\n",
    },
    Subject {
        table: Table::Lfm2_5,
        prompt: "<|im_start|>assistant\n",
        output: concat!(
            "<think>The user wants the weather and a drink.</think>Let me check both.",
            "<|tool_call_start|>[get_weather(city='Paris', days=3), ",
            "change_drink(drink_id='latte', new_preferences={'size': 'large', 'temperature': 'hot'})]",
            "<|tool_call_end|>"
        ),
        content_prompt: "<|im_start|>assistant\n",
    },
];

/// Prose of about four kilobytes with no marker of any table in it: what most deltas carry.
fn prose() -> String {
    let sentence = "The forecast for the week stays mild, with light wind from the west and a chance of rain on Thursday afternoon; ";
    let mut text = String::new();
    while text.len() < 4096 {
        text.push_str(sentence);
    }
    text
}

/// The request's tools: a string and an integer, a string and an object.
fn declared() -> Declared {
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

/// `text` cut into pieces of at least `size` bytes, each ending on a character boundary.
fn pieces(text: &str, size: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (at, _) in text.char_indices() {
        if at - start >= size {
            out.push(&text[start..at]);
            start = at;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// An engine over `table`, seeded with the subject's prompt.
fn engine(table: Table, prompt: &str, declared: &Declared) -> Engine {
    let mut engine = Engine::new(table.format(), declared.clone());
    let mut out = Events::new();
    engine
        .feed(
            Input::Prompt {
                token_ids: &[],
                text: prompt,
            },
            &mut out,
        )
        .expect("prompt");
    engine
}

/// Feeds `pieces` and the end of the stream; the events drained after every piece, as a stream's
/// reader drains them.
fn read<'d>(engine: &mut Engine, pieces: &[&'d str], out: &mut Events<'d>) {
    for piece in pieces {
        engine
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: piece,
                    spans: &[],
                },
                out,
            )
            .expect("delta");
        black_box(out.drain());
    }
    engine
        .feed(
            Input::End {
                finish: EngineFinish::Stop,
            },
            out,
        )
        .expect("end");
    black_box(out.drain());
}

fn parse(c: &mut Criterion) {
    let declared = declared();
    let mut group = c.benchmark_group("parse");
    for subject in SUBJECTS {
        let name = subject.table.name();
        group.throughput(Throughput::Bytes(subject.output.len() as u64));
        group.bench_with_input(BenchmarkId::new("whole", name), subject, |b, subject| {
            b.iter_batched_ref(
                || engine(subject.table, subject.prompt, &declared),
                |engine| {
                    let mut out = Events::new();
                    read(engine, &[subject.output], &mut out);
                },
                BatchSize::SmallInput,
            );
        });
        let cut = pieces(subject.output, 8);
        group.bench_with_input(BenchmarkId::new("pieces8", name), subject, |b, subject| {
            b.iter_batched_ref(
                || engine(subject.table, subject.prompt, &declared),
                |engine| {
                    let mut out = Events::new();
                    read(engine, &cut, &mut out);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// The prose rows: every subject's table with the output placed in its answer, and the table with
/// no markers at all, which is the floor.
fn prose_only(c: &mut Criterion) {
    let declared = declared();
    let text = prose();
    let cut = pieces(&text, 8);
    let mut group = c.benchmark_group("prose");
    group.throughput(Throughput::Bytes(text.len() as u64));
    let subjects = SUBJECTS
        .iter()
        .map(|subject| (subject.table, subject.content_prompt))
        .chain(std::iter::once((Table::Plain, "")));
    for (table, prompt) in subjects {
        let name = table.name();
        group.bench_with_input(BenchmarkId::new("whole", name), &table, |b, table| {
            b.iter_batched_ref(
                || engine(*table, prompt, &declared),
                |engine| {
                    let mut out = Events::new();
                    read(engine, &[text.as_str()], &mut out);
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("pieces8", name), &table, |b, table| {
            b.iter_batched_ref(
                || engine(*table, prompt, &declared),
                |engine| {
                    let mut out = Events::new();
                    read(engine, &cut, &mut out);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn setup(c: &mut Criterion) {
    let declared = declared();
    let mut group = c.benchmark_group("setup");
    for subject in SUBJECTS {
        let name = subject.table.name();
        group.bench_with_input(BenchmarkId::new("table", name), subject, |b, subject| {
            b.iter(|| black_box(subject.table.format()));
        });
        group.bench_with_input(BenchmarkId::new("engine", name), subject, |b, subject| {
            b.iter(|| black_box(engine(subject.table, subject.prompt, &declared)));
        });
    }
    group.finish();
}

criterion_group!(benches, parse, prose_only, setup);
criterion_main!(benches);
