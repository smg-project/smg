//! What a table costs: the bytes per second of an output fed whole and in small pieces, the bytes
//! per second of prose with no marker in it (the common streaming case), and the construction of
//! an engine, which is the parser's per-request setup. One subject per call syntax, each a turn as
//! the table's own tests spell it: a thought, an answer and two calls, the tools declared as a
//! request declares them. The parse and prose rows time the reading alone: the engine is made
//! before the timed closure and dropped after it, and the setup rows time the making and the drop.
//!
//! The recorded slices of bellwether's hand-written sets (`tests/common/slices.rs`: per table
//! family, plain text, one call, parallel calls, a thought and a call, and a malformed call cut
//! from the one call's) add the cost of the output fed whole, one character at a time and in
//! pieces of about a token (`slice/<family>/{whole,chars,tokens}`, in bytes per second or tokens
//! per second), the parts of the per-request setup (`setup/{table,validate,scanner,declared,
//! engine}`) and the adapters over the events a slice produces
//! (`adapt/{chat_deltas,chat_message,messages_stream}`, per event). A slice row is the reading
//! alone, the engine built outside the timing; `setup/engine` states the per-request setup. The
//! slices are read from the tree `BELLWETHER_FIXTURES` names (the one `bellwether unpack` writes
//! from the commit CI pins); without it the `slice` and `adapt` groups are skipped, and the run
//! says so once.
//!
//! `cargo bench -p smg-symphony --bench throughput`; a filter narrows it (`-- parse/whole`). The
//! README's evidence section says how a change produces its throughput row from these.
#![allow(clippy::expect_used)]

use std::{hint::black_box, sync::OnceLock, time::Duration};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use openai_protocol::common::{Function, Tool};
use serde_json::json;
use slices::{Slice, FAMILIES};
use symphony::{
    adapt::{chat, messages},
    formats::Table,
    Declared, Engine, EngineFinish, Event, Events, Input, Parser,
};

#[path = "../tests/common/slices.rs"]
mod slices;

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
    // Llama 3.1's calls alone in the turn: no thought and no answer, the call token, two calls
    // with `; ` between them, as the model writes them.
    Subject {
        table: Table::Json,
        prompt: "",
        output: concat!(
            "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\", \"days\": 3}}; ",
            "{\"name\": \"change_drink\", \"parameters\": {\"drink_id\": \"latte\", ",
            "\"new_preferences\": {\"size\": \"large\", \"temperature\": \"hot\"}}}"
        ),
        content_prompt: "",
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

/// The slices of the eight tables that stand for the call syntaxes, read once from the tree
/// `BELLWETHER_FIXTURES` names; empty, with the reason printed once, without the tree, and a
/// group the tree has not recorded is named and left out.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice is the bench's diagnostic output"
)]
fn recorded() -> &'static [(Table, Slice)] {
    static RECORDED: OnceLock<Vec<(Table, Slice)>> = OnceLock::new();
    RECORDED.get_or_init(|| {
        let Some(root) = slices::root() else {
            eprintln!(
                "skipping the slice and adapt rows: {} is not set; point it at the tree \
                 `bellwether unpack` writes",
                slices::ENV
            );
            return Vec::new();
        };
        let recorded = slices::recorded(&root).expect("the recorded slices");
        for notice in &recorded.missing {
            eprintln!("skipping {notice}");
        }
        recorded.slices
    })
}

/// The table's terminals, longest first, so that the longest matches first.
fn terminals(table: Table) -> Vec<String> {
    let format = table.format();
    let mut terminals: Vec<String> = format.terminal_texts().map(str::to_owned).collect();
    terminals.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    terminals.dedup();
    terminals
}

/// `text` cut as a tokenizer would, near enough for a cost per token: a terminal of the table is
/// one piece, and the text between is cut into pieces of up to four characters that end before a
/// space, a newline or a terminal.
fn tokens<'a>(text: &'a str, terminals: &[String]) -> Vec<&'a str> {
    let terminal_at = |at: usize| {
        terminals
            .iter()
            .find(|terminal| text[at..].starts_with(terminal.as_str()))
    };
    let mut out = Vec::new();
    let mut start = 0;
    while start < text.len() {
        if let Some(terminal) = terminal_at(start) {
            out.push(&text[start..start + terminal.len()]);
            start += terminal.len();
            continue;
        }
        let mut end = start;
        for (count, (offset, ch)) in text[start..].char_indices().enumerate() {
            let at = start + offset;
            if count > 0 && (count == 4 || ch == ' ' || ch == '\n' || terminal_at(at).is_some()) {
                break;
            }
            end = at + ch.len_utf8();
        }
        out.push(&text[start..end]);
        start = end;
    }
    out
}

/// The events `slice` produces through `table` read in token pieces: the stream an adapter
/// renders, the prompt's own events first (a prompt that opens the thought starts the stream
/// with its reasoning event, which the adapters render before any delta).
fn events_of(table: Table, slice: &Slice, declared: &Declared) -> Vec<Event<'static>> {
    let terminals = terminals(table);
    let mut engine = Engine::new(table.format(), declared.clone());
    let mut out = Events::new();
    let mut events = Vec::new();
    engine
        .feed(
            Input::Prompt {
                token_ids: &[],
                text: &slice.prompt,
            },
            &mut out,
        )
        .expect("prompt");
    events.extend(out.drain().into_iter().map(Event::into_owned));
    for piece in tokens(&slice.output, &terminals) {
        engine
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: piece,
                    spans: &[],
                },
                &mut out,
            )
            .expect("delta");
        events.extend(out.drain().into_iter().map(Event::into_owned));
    }
    engine
        .feed(
            Input::End {
                finish: EngineFinish::Stop,
            },
            &mut out,
        )
        .expect("end");
    events.extend(out.drain().into_iter().map(Event::into_owned));
    events
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
    let tools = declared();
    let mut group = c.benchmark_group("setup");
    for subject in SUBJECTS {
        let name = subject.table.name();
        group.bench_with_input(BenchmarkId::new("table", name), subject, |b, subject| {
            b.iter(|| black_box(subject.table.format()));
        });
        group.bench_with_input(BenchmarkId::new("validate", name), subject, |b, subject| {
            let format = subject.table.format();
            b.iter(|| black_box(format.validate()));
        });
        group.bench_with_input(BenchmarkId::new("engine", name), subject, |b, subject| {
            b.iter(|| black_box(engine(subject.table, subject.prompt, &tools)));
        });
    }
    group.bench_function("declared", |b| b.iter(|| black_box(declared())));
    group.finish();
}

fn slices(c: &mut Criterion) {
    let recorded = recorded();
    if recorded.is_empty() {
        return;
    }
    let declared = declared();
    for family in FAMILIES {
        let mut group = c.benchmark_group(format!("slice/{family}"));
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_millis(1500));
        for (table, slice) in recorded.iter().filter(|(_, slice)| slice.family == family) {
            let name = table.name();
            let terminals = terminals(*table);
            let by_token = tokens(&slice.output, &terminals);
            let by_char = pieces(&slice.output, 1);
            group.throughput(Throughput::Bytes(slice.output.len() as u64));
            // The engine is built outside the timing: a slice is short, and its row is the
            // reading alone; `setup/engine` states the per-request setup on its own.
            group.bench_with_input(BenchmarkId::new("whole", name), slice, |b, slice| {
                b.iter_batched_ref(
                    || engine(*table, &slice.prompt, &declared),
                    |engine| {
                        let mut out = Events::new();
                        read(engine, &[slice.output.as_str()], &mut out);
                    },
                    BatchSize::SmallInput,
                );
            });
            group.bench_with_input(BenchmarkId::new("chars", name), slice, |b, slice| {
                b.iter_batched_ref(
                    || engine(*table, &slice.prompt, &declared),
                    |engine| {
                        let mut out = Events::new();
                        read(engine, &by_char, &mut out);
                    },
                    BatchSize::SmallInput,
                );
            });
            group.throughput(Throughput::Elements(by_token.len() as u64));
            group.bench_with_input(BenchmarkId::new("tokens", name), slice, |b, slice| {
                b.iter_batched_ref(
                    || engine(*table, &slice.prompt, &declared),
                    |engine| {
                        let mut out = Events::new();
                        read(engine, &by_token, &mut out);
                    },
                    BatchSize::SmallInput,
                );
            });
        }
        group.finish();
    }
}

fn adapters(c: &mut Criterion) {
    let recorded = recorded();
    if recorded.is_empty() {
        return;
    }
    let declared = declared();
    let mut group = c.benchmark_group("adapt");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_millis(1500));
    for (table, slice) in recorded
        .iter()
        .filter(|(_, slice)| slice.family == "thought_call")
    {
        let name = table.name();
        let events = events_of(*table, slice, &declared);
        group.throughput(Throughput::Elements(events.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("chat_deltas", name),
            &events,
            |b, events| {
                b.iter(|| black_box(chat::deltas(0, events)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("chat_message", name),
            &events,
            |b, events| {
                b.iter(|| black_box(chat::message(0, events)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("messages_stream", name),
            &events,
            |b, events| {
                b.iter(|| {
                    let mut stream = messages::Stream::new();
                    let mut out = Vec::new();
                    for event in events {
                        black_box(stream.feed(event, &mut out));
                    }
                    stream.close(&mut out);
                    black_box(out)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, parse, prose_only, setup, slices, adapters);
criterion_main!(benches);
