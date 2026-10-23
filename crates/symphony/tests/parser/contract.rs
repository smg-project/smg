//! The contract every Symphony parser keeps, checked for every subject over a corpus of its outputs
//! and every way of chunking them.
//!
//! Checked for each subject in [`FORMATS`]:
//!
//! - conservation: every byte of the output lands in exactly one event, in order, with `Finish`
//!   last;
//! - chunking invariance: what the parser says (content, each reasoning region, dropped and
//!   malformed text by reason, the calls with their arguments, and the finish with its reason and
//!   counts) does not depend on how the output was cut into deltas, nor on empty deltas between
//!   them;
//! - well-formedness: arguments and an end only for a call that has started and not ended, one
//!   start per index, reasoning text only inside an open region, exactly one `Finish`, counting the
//!   calls that started, after every fragment each call's arguments so far a prefix of JSON a
//!   strict reader accepts, which is what prefix stability promises a client, until a
//!   `Malformed { InvalidArguments }` event of no bytes says where they stopped being one, from
//!   which every fragment is the model's bytes as written (`json` equal to `source`), so that
//!   the call carries exactly what the model wrote, and a call that ended, by its own
//!   terminal, the region's or the end of the stream, has arguments that are whole JSON unless a
//!   `Malformed` event came while it was open, so that a client is never handed a cut call as a
//!   whole one without a word, and a fragment that closes a string the arguments left open
//!   carries the bytes of the output that closed it, so that no closing quote is made up for a
//!   value the model never closed, and a fragment that is an empty string carries bytes of the
//!   output too, so that no value is made up for a member that has none;
//! - order: calls are numbered from zero as they start, with ids that follow the index;
//! - the lifecycle: one `Prompt`, deltas, one `End`, and an input out of order is a `Lifecycle`
//!   error that emits nothing.
//!
//! The design's "committed stays committed" is not a separate check: `Events` is append-only and
//! conservation forbids saying a byte twice, so nothing a parser pushed can be taken back through
//! the event list. The fifth property, token identity (every token counted once, in the event that
//! carries its first byte, whatever the cuts), is checked event by event with a synthetic
//! tokenization of each output. A subject joins the contract with one entry in [`FORMATS`], its
//! constructor and its corpus.

use openai_protocol::common::{Function, Tool};
use serde_json::json as value;
use symphony::{
    format::CallId,
    formats::{self, Table},
    shape::{
        self, Arguments, Atom, CallShape, Class, Dialect, End, NameSource, Path, Pattern, Slot,
        ValueParser,
    },
    CallSyntax, Declared, DropReason, Emits, Engine, EngineFinish, Event, Events, FinishReason,
    Format, Input, MalformedReason, ParseError, Parser, TokenSpan,
};

use crate::{
    common,
    common::{bytes_of, chunkings, delta, prompt},
};

/// A subject under test: how to make its parser, the outputs it is checked over, and the registry
/// table it stands for, when it is one: such a subject builds its parser from `Table::format`, so
/// the label is what is tested ([`every_registry_table_has_a_subject`] keeps every table of
/// [`Table::ALL`] in the contract).
struct Subject {
    name: &'static str,
    new: fn() -> Box<dyn Parser>,
    outputs: &'static [&'static str],
    table: Option<Table>,
}

fn qwen3() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Qwen3.format(), Declared::default()))
}

fn qwen3_tagged() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Table::Qwen3Tagged.format(),
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "note": {"type": ["string", "null"]},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }]),
    ))
}

fn qwen2_5() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Qwen2_5.format(), Declared::default()))
}

fn seed_oss() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Table::SeedOss.format(),
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "note": {"type": ["string", "null"]},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }]),
    ))
}

fn deepseek_v4_1() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Table::DeepSeekV4_1.format(),
        Declared::default(),
    ))
}

fn keyed_tools() -> Declared {
    Declared::of(&[Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "get_weather".to_string(),
            description: None,
            parameters: value!({"type": "object", "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "note": {"type": ["string", "null"]},
            }}),
            strict: None,
            extra: Default::default(),
        },
    }])
}

fn hy4() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Hy4.format(), keyed_tools()))
}

fn ling() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Ling.format(), keyed_tools()))
}

fn glm() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Glm.format(), keyed_tools()))
}

/// The GLM table where the stream's stop closes the last call, as when a family's last call ends
/// with a stop token the stop decoder hides.
fn glm_closed_at_stop() -> Box<dyn Parser> {
    Box::new(Engine::new(
        formats::glm().closes_at_stop("call"),
        keyed_tools(),
    ))
}

/// Muse Glimmer's messages, spelled from the response template of `meta-models/Muse-Glimmer-30B`
/// (its `tokenizer_config.json`): a thought addressed to the model itself, an answer addressed to
/// the user, and nothing kept outside them. The calls are left out.
fn muse_messages() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Format::new("muse messages")
            .terminal("thought", "to=self<|message|>")
            .terminal("answer", "to=user<|message|>")
            .terminal("eot", "<|eot|>")
            .terminal("eom", "<|eom|>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("content", Emits::Content)
            .transition("outside", "thought", "reasoning")
            .transition("outside", "answer", "content")
            .transition("reasoning", "eom", "outside")
            .transition("content", "eot", "outside")
            .transition("content", "eom", "outside")
            .framing("outside")
            .trims("reasoning")
            .trims("content"),
        Declared::default(),
    ))
}

/// ERNIE's turn, spelled from transformers v5.18.0's `tests/utils/test_chat_parsing.py`
/// (`ernie_template`): the thought opens with `(?:^|<think>\s*)`, so a turn that starts with text
/// starts in it; the answer opens with `<response>\n`, a call with `<tool_call>`, and nothing is
/// kept outside them.
fn ernie() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Format::new("ernie")
            .terminal("think_open", "<think>")
            .terminal("think_close", "</think>")
            .terminal("answer_open", "<response>\n")
            .terminal("answer_close", "</response>")
            .terminal("call_open", "<tool_call>")
            .terminal("call_close", "</tool_call>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("content", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("outside", "think_open", "reasoning")
            .transition("reasoning", "think_close", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "answer_close", "outside")
            .transition("outside", "call_open", "call")
            .transition("call", "call_close", "outside")
            .calls(CallSyntax::Json)
            .framing("outside")
            .turn_anchor("Assistant:")
            .starts_in("reasoning")
            .trims("reasoning")
            .trims("content"),
        Declared::default(),
    ))
}

/// Gemma 4's turn, spelled from the response template of `google/gemma-4-31B-it` (its
/// `tokenizer_config.json`): two anchors, a thought, the calls in Gemma's spelling, the content
/// outside them trimmed stay by stay, and its non-stop close, `<|tool_response>`, a row back into
/// the content that drops the marker.
fn gemma4_template() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Format::new("gemma 4 template")
            .terminal("think_open", "<|channel>thought\n")
            .terminal("think_close", "<channel|>")
            .terminal("call_open", "<|tool_call>")
            .terminal("call_close", "<tool_call|>")
            .terminal("response_open", "<|tool_response>")
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("call", Emits::Arguments)
            .transition("content", "think_open", "reasoning")
            .transition("reasoning", "think_close", "content")
            .transition("content", "call_open", "call")
            .transition("call", "call_close", "content")
            .transition("content", "response_open", "content")
            .calls(CallSyntax::Gemma(symphony::gemma::Spelling::GEMMA4))
            .turn_anchor("<|turn>model\n")
            .turn_anchor("<tool_response|>")
            .trims("content")
            .trims("reasoning"),
        Declared::default(),
    ))
}

/// Inkling's messages, spelled from transformers v5.18.0's `tests/utils/test_chat_parsing.py`
/// (`inkling_template`): each message is a thought, an answer or a call after its own marker, all
/// of them closed by `<|end_message|>`, and nothing is kept outside them; the thoughts and the
/// answers repeat, joined with `""`, as the engine's stays follow one another, and not stripped
/// (no trim row).
fn inkling_template() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Format::new("inkling template")
            .terminal("thought", "<|content_thinking|>")
            .terminal("answer", "<|content_text|>")
            .terminal("call", "<|content_invoke_tool_json|>")
            .terminal("end", "<|end_message|>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("content", Emits::Content)
            .state("calls", Emits::Arguments)
            .transition("outside", "thought", "reasoning")
            .transition("outside", "answer", "content")
            .transition("outside", "call", "calls")
            .transition("reasoning", "end", "outside")
            .transition("content", "end", "outside")
            .transition("calls", "end", "outside")
            .calls(CallSyntax::Json)
            .framing("outside")
            .turn_anchor("<|message_model|>"),
        Declared::default(),
    ))
}

fn iquest() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::IQuest.format(), keyed_tools()))
}

fn olmo3() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Olmo3.format(), Declared::default()))
}

fn lfm2_5() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Lfm2_5.format(), Declared::default()))
}

fn llama4() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Llama4.format(), recorded_tools()))
}

fn xlam() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Xlam.format(), Declared::default()))
}

fn minimax_m3() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Table::MinimaxM3.format(),
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "opts": {"type": "object", "properties": {"unit": {"type": "string"}}},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }]),
    ))
}

fn kimi_k3() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::KimiK3.format(), Declared::default()))
}

fn muse_spark() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::MuseSpark.format(), Declared::default()))
}

fn kimi_k2() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::KimiK2.format(), Declared::default()))
}

fn mistral() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Mistral.format(), Declared::default()))
}

fn mistral_v13() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::MistralV13.format(), Declared::default()))
}

fn llama_nemotron() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Table::LlamaNemotron.format(),
        Declared::default(),
    ))
}

fn function_gemma() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Table::FunctionGemma.format(),
        Declared::default(),
    ))
}

fn forced() -> Box<dyn Parser> {
    Box::new(Engine::new(
        formats::forced(),
        Declared::default().forcing("get_weather"),
    ))
}

fn plain() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Plain.format(), Declared::default()))
}

/// A tool of bellwether's common set, by name and parameters.
fn tool(name: &str, parameters: serde_json::Value) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters,
            strict: None,
            extra: Default::default(),
        },
    }
}

/// The tools of bellwether's common set, which type the values of the outputs recorded with them
/// under the tagged, keyed and attributed syntaxes.
fn recorded_tools() -> Declared {
    Declared::of(&[
        tool(
            "get_weather",
            value!({"type": "object", "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string"},
            }}),
        ),
        tool(
            "get_stock_price",
            value!({"type": "object", "properties": {"ticker": {"type": "string"}}}),
        ),
        tool(
            "save_note",
            value!({"type": "object", "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "meta": {"type": "object"},
            }}),
        ),
    ])
}

fn qwen3_tagged_direct() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Table::Qwen3TaggedDirect.format(),
        recorded_tools(),
    ))
}

fn deepseek_v4() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::DeepSeekV4.format(), Declared::default()))
}

fn deepseek_v3_2() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Table::DeepSeekV3_2.format(),
        Declared::default(),
    ))
}

fn deepseek_v3_1() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Table::DeepSeekV3_1.format(),
        Declared::default(),
    ))
}

fn deepseek_v3() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::DeepSeekV3.format(), Declared::default()))
}

fn json() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Json.format(), Declared::default()))
}

fn minimax_m2() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::MinimaxM2.format(), recorded_tools()))
}

fn minicpm5() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::MiniCpm5.format(), recorded_tools()))
}

fn dots3() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Dots3.format(), recorded_tools()))
}

fn glm4() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Glm4.format(), recorded_tools()))
}

fn k2_horizon() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::K2Horizon.format(), recorded_tools()))
}

fn gemma4() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Gemma4.format(), Declared::default()))
}

fn inkling() -> Box<dyn Parser> {
    Box::new(Engine::new(Table::Inkling.format(), Declared::default()))
}

/// Inkling as a request without tools reads it: the message header content, its markers dropped.
fn inkling_without_calls() -> Box<dyn Parser> {
    Box::new(Engine::new(
        Table::Inkling.format().without_calls(),
        Declared::default(),
    ))
}

/// `\w` as these outputs need it: the ASCII letters, digits and `_` their names are written in.
fn word() -> Class {
    Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false)
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

/// The Qwen3 table with its calls given as data: one JSON object that names the call, right
/// after the region's marker, as the hand-written JSON syntax reads it.
fn qwen3_shaped() -> Box<dyn Parser> {
    let shape = shape::intern(CallShape {
        header: Pattern::default(),
        name: NameSource::Body,
        arguments: Arguments::Envelope {
            dialect: Dialect::default(),
            name: Path::new(["name"]),
            arguments: Path::new(["arguments"]),
        },
        trailer: Pattern::default(),
        word: word(),
        hold_cap: 8192,
    });
    Box::new(Engine::new(
        formats::qwen3(CallSyntax::Shaped(shape)),
        Declared::default(),
    ))
}

/// transformers v5.18.0's `tests/utils/test_chat_parsing.py` (`gpt_oss_template`): Harmony's
/// analysis and final channels, nothing kept outside them, and the calls the commentary channel
/// opens: `<|channel|>commentary to=functions.` is the terminal, `(?P<name>\w+).*?<\|message\|>`
/// the header, the arguments one JSON value, and the stop token `<|call|>` the close.
fn gpt_oss_template() -> Box<dyn Parser> {
    let shape = shape::intern(CallShape {
        header: Pattern(
            [
                run(word(), 1, true, Some(Slot::Name)),
                run(Class::any(), 0, false, None),
                Atom::Lit("<|message|>".into()),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Json(Dialect::default()),
        trailer: Pattern::default(),
        word: word(),
        hold_cap: 8192,
    });
    Box::new(Engine::new(
        Format::new("gpt-oss test template")
            .terminal("analysis", "<|channel|>analysis<|message|>")
            .terminal("final", "<|channel|>final<|message|>")
            .terminal("end", "<|end|>")
            .terminal("call_open", "<|channel|>commentary to=functions.")
            .terminal("call_close", "<|call|>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("content", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("outside", "analysis", "reasoning")
            .transition("outside", "final", "content")
            .transition("reasoning", "end", "outside")
            .transition("content", "end", "outside")
            .transition("outside", "call_open", "call")
            .transition("call", "call_close", "outside")
            .calls(CallSyntax::Shaped(shape))
            .framing("outside")
            .closes_at_stop("call")
            .turn_anchor("<|start|>assistant")
            .trims("reasoning")
            .trims("content")
            .stop_tokens(["<|return|>", "<|endoftext|>", "<|call|>"]),
        Declared::default(),
    ))
}

fn space() -> Class {
    Class::new([('\t', '\r'), (' ', ' ')], false)
}

fn none_of(chars: &str) -> Class {
    Class::new(chars.chars().map(|c| (c, c)), true)
}

fn ends_at(consume: impl IntoIterator<Item = Atom>, ahead: impl IntoIterator<Item = Atom>) -> End {
    End {
        consume: Pattern(consume.into_iter().collect()),
        ahead: Pattern(ahead.into_iter().collect()),
    }
}

/// transformers v5.18.0's `transformers serve` template for Qwen 3.5 (`qwen3_5` in
/// `src/transformers/cli/serving/utils.py`): a thought, the content outside it stripped, and calls
/// opened by `<tool_call>` and the header `\s*<function=(?P<name>[^>\n]+)>`, closed by the
/// trailer `</function>\s*` and `</tool_call>`, their parameters tags with a key and a value up
/// to `\s*</parameter>`, typed by the request's tools.
fn qwen3_5_serve_shaped() -> Box<dyn Parser> {
    Box::new(Engine::new(qwen3_5_serve_table(), keyed_tools()))
}

/// The Qwen 3.5 serve template where the stream's stop closes a call, as the template's reader
/// closes a region at the end of its text, whatever the trailer.
fn qwen3_5_serve_closed_at_stop() -> Box<dyn Parser> {
    Box::new(Engine::new(
        qwen3_5_serve_table().closes_at_stop("call"),
        keyed_tools(),
    ))
}

fn qwen3_5_serve_table() -> Format {
    let name = none_of(">\n");
    let shape = shape::intern(CallShape {
        header: Pattern(
            [
                run(space(), 0, true, None),
                Atom::Lit("<function=".into()),
                run(name.clone(), 1, true, Some(Slot::Name)),
                Atom::Lit(">".into()),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Params {
            tag: Pattern(
                [
                    Atom::Lit("<parameter=".into()),
                    run(name, 1, true, Some(Slot::Key)),
                    Atom::Lit(">".into()),
                    run(space(), 0, true, None),
                    run(Class::any(), 0, false, Some(Slot::Value)),
                ]
                .into(),
            ),
            ends: [ends_at(
                [
                    run(space(), 0, true, None),
                    Atom::Lit("</parameter>".into()),
                ],
                [],
            )]
            .into(),
            values: ValueParser::Raw,
        },
        trailer: Pattern([Atom::Lit("</function>".into()), run(space(), 0, true, None)].into()),
        word: word(),
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
}

/// The response template of `meta-models/Muse-Glimmer-30B` (its `tokenizer_config.json`) with its
/// calls: the terminal `<atem:invoke`, a header that skips other attributes and names the tool, and
/// parameter tags that do the same for their keys, each value up to `</atem:parameter>`.
fn muse_shaped() -> Box<dyn Parser> {
    let skip = || run(none_of(">"), 0, false, None);
    let shape = shape::intern(CallShape {
        header: Pattern(
            [
                Atom::Boundary,
                skip(),
                Atom::Boundary,
                Atom::Lit("name=\"".into()),
                run(none_of("\""), 1, true, Some(Slot::Name)),
                Atom::Lit("\">".into()),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Params {
            tag: Pattern(
                [
                    Atom::Lit("<atem:parameter".into()),
                    Atom::Boundary,
                    skip(),
                    Atom::Boundary,
                    Atom::Lit("name=\"".into()),
                    run(none_of("\""), 1, true, Some(Slot::Key)),
                    Atom::Lit("\"".into()),
                    skip(),
                    Atom::Lit(">".into()),
                    run(Class::any(), 0, false, Some(Slot::Value)),
                ]
                .into(),
            ),
            ends: [ends_at([Atom::Lit("</atem:parameter>".into())], [])].into(),
            values: ValueParser::Json {
                dialect: Dialect::default(),
                allow_non_json: true,
                strip: true,
            },
        },
        trailer: Pattern::default(),
        word: word(),
        hold_cap: 8192,
    });
    Box::new(Engine::new(
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
            .trims("content"),
        Declared::default(),
    ))
}

/// Parameters on abstract tags: a call named by the template, its parameters `<k=KEY>` and a
/// value that ends at `</k>` or where the next parameter's tag begins.
fn params_tags() -> Box<dyn Parser> {
    let shape = shape::intern(CallShape {
        header: Pattern::default(),
        name: NameSource::Literal("f".into()),
        arguments: Arguments::Params {
            tag: Pattern(
                [
                    Atom::Lit("<k=".into()),
                    run(word(), 1, true, Some(Slot::Key)),
                    Atom::Lit(">".into()),
                    run(Class::any(), 0, false, Some(Slot::Value)),
                ]
                .into(),
            ),
            ends: [
                ends_at([Atom::Lit("</k>".into())], []),
                ends_at([], [Atom::Lit("<k=".into())]),
            ]
            .into(),
            values: ValueParser::Raw,
        },
        trailer: Pattern::default(),
        word: word(),
        hold_cap: 8192,
    });
    Box::new(Engine::new(
        Format::new("parameters on abstract tags")
            .terminal("call_open", "<c>")
            .terminal("call_close", "</c>")
            .state("content", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("content", "call_open", "call")
            .transition("call", "call_close", "content")
            .calls(CallSyntax::Shaped(shape))
            .closes_at_stop("call"),
        Declared::default(),
    ))
}

/// Calls given as data on abstract tags: a region between `<c>` and `</c>` beside a thought
/// between `<t>` and `</t>`, each call named by a `<n=NAME>` header, its arguments one JSON value,
/// and `</f>` with the whitespace after it as the trailer, which the stream's stop may end too.
fn shaped_tags() -> Box<dyn Parser> {
    let shape = shape::intern(CallShape {
        header: Pattern(
            [
                Atom::Lit("<n=".into()),
                run(word(), 1, true, Some(Slot::Name)),
                Atom::Lit(">".into()),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Json(Dialect::default()),
        trailer: Pattern(
            [
                Atom::Lit("</f>".into()),
                run(Class::new([('\t', '\r'), (' ', ' ')], false), 0, true, None),
            ]
            .into(),
        ),
        word: word(),
        hold_cap: 8192,
    });
    Box::new(Engine::new(
        Format::new("shaped tags")
            .terminal("think_open", "<t>")
            .terminal("think_close", "</t>")
            .terminal("call_open", "<c>")
            .terminal("call_close", "</c>")
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("call", Emits::Arguments)
            .transition("content", "think_open", "reasoning")
            .transition("reasoning", "think_close", "content")
            .transition("content", "call_open", "call")
            .transition("reasoning", "call_open", "call")
            .transition("call", "call_close", "content")
            .calls(CallSyntax::Shaped(shape))
            .closes_at_stop("call"),
        Declared::default(),
    ))
}

const FORMATS: &[Subject] = &[
    Subject {
        name: "qwen3",
        new: qwen3,
        outputs: QWEN3_OUTPUTS,
        table: Some(Table::Qwen3),
    },
    Subject {
        name: "qwen3 tagged",
        new: qwen3_tagged,
        outputs: QWEN3_TAGGED_OUTPUTS,
        table: Some(Table::Qwen3Tagged),
    },
    Subject {
        name: "qwen2.5",
        new: qwen2_5,
        outputs: QWEN3_OUTPUTS,
        table: Some(Table::Qwen2_5),
    },
    Subject {
        name: "deepseek v4.1",
        new: deepseek_v4_1,
        outputs: DSML_OUTPUTS,
        table: Some(Table::DeepSeekV4_1),
    },
    Subject {
        name: "seed-oss",
        new: seed_oss,
        outputs: SEED_OUTPUTS,
        table: Some(Table::SeedOss),
    },
    Subject {
        name: "hy4",
        new: hy4,
        outputs: HY4_OUTPUTS,
        table: Some(Table::Hy4),
    },
    Subject {
        name: "ling",
        new: ling,
        outputs: LING_OUTPUTS,
        table: Some(Table::Ling),
    },
    Subject {
        name: "glm",
        new: glm,
        outputs: GLM_OUTPUTS,
        table: Some(Table::Glm),
    },
    Subject {
        name: "glm, closed at stop",
        new: glm_closed_at_stop,
        outputs: GLM_STOPPED_OUTPUTS,
        table: None,
    },
    Subject {
        name: "muse messages, framed",
        new: muse_messages,
        outputs: MUSE_OUTPUTS,
        table: None,
    },
    Subject {
        name: "ernie, started in the thought",
        new: ernie,
        outputs: ERNIE_OUTPUTS,
        table: None,
    },
    Subject {
        name: "gemma 4 template, trimmed",
        new: gemma4_template,
        outputs: GEMMA4_TEMPLATE_OUTPUTS,
        table: None,
    },
    Subject {
        name: "inkling template, framed",
        new: inkling_template,
        outputs: INKLING_TEMPLATE_OUTPUTS,
        table: None,
    },
    Subject {
        name: "iquest",
        new: iquest,
        outputs: IQUEST_OUTPUTS,
        table: Some(Table::IQuest),
    },
    Subject {
        name: "olmo3",
        new: olmo3,
        outputs: OLMO3_OUTPUTS,
        table: Some(Table::Olmo3),
    },
    Subject {
        name: "lfm2.5",
        new: lfm2_5,
        outputs: LFM_OUTPUTS,
        table: Some(Table::Lfm2_5),
    },
    Subject {
        name: "llama4",
        new: llama4,
        outputs: LLAMA4_OUTPUTS,
        table: Some(Table::Llama4),
    },
    Subject {
        name: "xlam",
        new: xlam,
        outputs: XLAM_OUTPUTS,
        table: Some(Table::Xlam),
    },
    Subject {
        name: "minimax m3",
        new: minimax_m3,
        outputs: M3_OUTPUTS,
        table: Some(Table::MinimaxM3),
    },
    Subject {
        name: "kimi k3",
        new: kimi_k3,
        outputs: KIMI_K3_OUTPUTS,
        table: Some(Table::KimiK3),
    },
    Subject {
        name: "muse spark",
        new: muse_spark,
        outputs: MUSE_SPARK_OUTPUTS,
        table: Some(Table::MuseSpark),
    },
    Subject {
        name: "kimi k2",
        new: kimi_k2,
        outputs: KIMI_K2_OUTPUTS,
        table: Some(Table::KimiK2),
    },
    Subject {
        name: "plain",
        new: plain,
        outputs: PLAIN_OUTPUTS,
        table: Some(Table::Plain),
    },
    Subject {
        name: "qwen3 tagged, the value between the tags",
        new: qwen3_tagged_direct,
        outputs: QWEN3_TAGGED_DIRECT_OUTPUTS,
        table: Some(Table::Qwen3TaggedDirect),
    },
    Subject {
        name: "deepseek v4",
        new: deepseek_v4,
        outputs: DEEPSEEK_V4_OUTPUTS,
        table: Some(Table::DeepSeekV4),
    },
    Subject {
        name: "deepseek v3.2",
        new: deepseek_v3_2,
        outputs: DEEPSEEK_V3_2_OUTPUTS,
        table: Some(Table::DeepSeekV3_2),
    },
    Subject {
        name: "deepseek v3.1",
        new: deepseek_v3_1,
        outputs: DEEPSEEK_V3_1_OUTPUTS,
        table: Some(Table::DeepSeekV3_1),
    },
    Subject {
        name: "deepseek v3",
        new: deepseek_v3,
        outputs: DEEPSEEK_V3_OUTPUTS,
        table: Some(Table::DeepSeekV3),
    },
    Subject {
        name: "json",
        new: json,
        outputs: JSON_OUTPUTS,
        table: Some(Table::Json),
    },
    Subject {
        name: "minimax m2",
        new: minimax_m2,
        outputs: MINIMAX_M2_OUTPUTS,
        table: Some(Table::MinimaxM2),
    },
    Subject {
        name: "minicpm5",
        new: minicpm5,
        outputs: MINICPM5_OUTPUTS,
        table: Some(Table::MiniCpm5),
    },
    Subject {
        name: "dots3",
        new: dots3,
        outputs: DOTS3_OUTPUTS,
        table: Some(Table::Dots3),
    },
    Subject {
        name: "glm4",
        new: glm4,
        outputs: GLM4_OUTPUTS,
        table: Some(Table::Glm4),
    },
    Subject {
        name: "k2 horizon",
        new: k2_horizon,
        outputs: K2_HORIZON_OUTPUTS,
        table: Some(Table::K2Horizon),
    },
    Subject {
        name: "gemma 4",
        new: gemma4,
        outputs: GEMMA4_OUTPUTS,
        table: Some(Table::Gemma4),
    },
    Subject {
        name: "inkling",
        new: inkling,
        outputs: INKLING_OUTPUTS,
        table: Some(Table::Inkling),
    },
    Subject {
        name: "inkling, no tools",
        new: inkling_without_calls,
        outputs: INKLING_NO_TOOLS_OUTPUTS,
        table: None,
    },
    Subject {
        name: "mistral",
        new: mistral,
        outputs: MISTRAL_LIST_OUTPUTS,
        table: Some(Table::Mistral),
    },
    Subject {
        name: "mistral v13",
        new: mistral_v13,
        outputs: MISTRAL_OUTPUTS,
        table: Some(Table::MistralV13),
    },
    Subject {
        name: "llama nemotron",
        new: llama_nemotron,
        outputs: LLAMA_NEMOTRON_OUTPUTS,
        table: Some(Table::LlamaNemotron),
    },
    Subject {
        name: "function gemma",
        new: function_gemma,
        outputs: FUNCTION_GEMMA_OUTPUTS,
        table: Some(Table::FunctionGemma),
    },
    Subject {
        name: "qwen3, its calls as a shape",
        new: qwen3_shaped,
        outputs: QWEN3_OUTPUTS,
        table: None,
    },
    Subject {
        name: "gpt-oss test template, its calls as a shape",
        new: gpt_oss_template,
        outputs: GPT_OSS_OUTPUTS,
        table: None,
    },
    Subject {
        name: "shaped calls on abstract tags",
        new: shaped_tags,
        outputs: SHAPED_TAGS_OUTPUTS,
        table: None,
    },
    Subject {
        name: "qwen 3.5 serve template, its calls as a shape",
        new: qwen3_5_serve_shaped,
        outputs: QWEN3_5_SERVE_OUTPUTS,
        table: None,
    },
    Subject {
        name: "qwen 3.5 serve template, its calls as a shape, closed at stop",
        new: qwen3_5_serve_closed_at_stop,
        outputs: QWEN3_5_SERVE_OUTPUTS,
        table: None,
    },
    Subject {
        name: "muse glimmer template, its calls as a shape",
        new: muse_shaped,
        outputs: MUSE_CALLS_OUTPUTS,
        table: None,
    },
    Subject {
        name: "parameters on abstract tags",
        new: params_tags,
        outputs: PARAMS_TAGS_OUTPUTS,
        table: None,
    },
    Subject {
        name: "forced",
        new: forced,
        outputs: FORCED_OUTPUTS,
        table: None,
    },
];

/// The Qwen 3.5 serve template: a thought, an answer and a call; two calls; a close without the
/// trailer; a marker with no function; text between parameters; a value that never ends; values
/// across lines and empty; calls cut in a value, in a tag and in the header; and nothing.
const QWEN3_5_SERVE_OUTPUTS: &[&str] = &[
    "<think>\nThe user wants the weather.\n</think>\n\nLet me check.\n<tool_call>\n\
     <function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n\
     </parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n\
     </tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n\
     </function>\n</tool_call>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</tool_call>\n\
     More text</function></tool_call>",
    "<tool_call>\nnot a function</tool_call>Hi",
    "<tool_call>\n<function=get_weather>\nnoise <parameter=city>  Paris  </parameter> junk \
     <parameter=days>4</parameter></function>\n</tool_call>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</function>\n</tool_call>",
    "<tool_call>\n<function=get_weather>\n<parameter=days>\n</parameter>\n<parameter=city>\n\
     New\nYork\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
    "<tool_call>\n<function=get_weather>\n<parame",
    "<tool_call>\n<fun",
    "",
];

/// Muse Glimmer's calls: after a thought, with attributes and text between parameters, an invoke
/// without a name, two calls, a tag whose name runs on, a call cut short, and nothing.
const MUSE_CALLS_OUTPUTS: &[&str] = &[
    "to=self<|message|>Plan.<|eom|><|start|>assistant<atem:invoke name=\"get_weather\">\
     <atem:parameter name=\"city\">Paris</atem:parameter><atem:parameter name=\"days\">3\
     </atem:parameter></atem:invoke>",
    "<atem:invoke id=\"1\" name=\"lookup\"><atem:parameter type=\"string\" name=\"q\">x y\
     </atem:parameter> <atem:parameter name=\"ids\">[1, 2]</atem:parameter></atem:invoke>",
    "<atem:invoke id=\"1\">text</atem:invoke> to=user<|message|>Hi<|eot|>",
    "<atem:invoke name=\"f\"></atem:invoke><atem:invoke name=\"g\"><atem:parameter \
     name=\"o\">{\"a\": true}</atem:parameter></atem:invoke>",
    "<atem:invoke name=\"f\"><atem:parameters name=\"x\">1</atem:parameter></atem:invoke>",
    "<atem:invoke name=\"f\"><atem:parameter name=\"q\">cut",
    "",
];

/// Parameters on abstract tags: values ended by their close and by the next tag, text between
/// them, a tag that does not match, a key written twice, calls cut in a value and in a tag, a
/// call the stop closes, and nothing.
const PARAMS_TAGS_OUTPUTS: &[&str] = &[
    "<c><k=a>1<k=b>2</k></c>",
    "Hi <c>junk<k=a>1</k> mid <k=b>x y</k>tail</c> bye",
    "<c><k=a b>1</k><k=c>[1, 2]</k></c>",
    "<c><k=a>1</k><k=a>2</k></c>",
    "<c><k=a>1<k=b>2",
    "<c><k=",
    "<c><k=a>{\"x\": [1, {\"y\": null}]}</k>",
    "",
];

/// The gpt-oss test template: a thought and a call the stop closes, a call cut short, a header
/// that cannot match (no name after the marker) and one the end leaves unfinished, an answer with
/// text outside it, arguments that are not JSON, text after the arguments, and nothing.
const GPT_OSS_OUTPUTS: &[&str] = &[
    "<|channel|>analysis<|message|>User asks for the weather.<|end|><|start|>assistant\
     <|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>\
     {\"city\": \"Paris\", \"days\": 3}",
    "<|channel|>commentary to=functions.get_weather<|message|>{\"city\": \"Par",
    "<|channel|>commentary to=functions.<|channel|>final<|message|>Hi",
    "<|channel|>commentary to=functions.get_weather json",
    "Hello <|channel|>final<|message|> It is sunny. <|end|> after",
    "<|channel|>commentary to=functions.f<|message|>not json",
    "<|channel|>commentary to=functions.f<|message|>{} trailing",
    "",
    "<|channel|>commentary to=functions.get_weather<|message|>{\"city\": \"Paris\", \"days\": ,}",
];

/// Calls on abstract tags: a call its trailer closes, terminals whose header does not match (in
/// the content, in a thought, twice in a row), a close the trailer does not end at, a call cut
/// short, a value the reader refuses before the trailer closes the call, a call the stop closes
/// after its trailer, a header the end leaves unfinished, and nothing.
const SHAPED_TAGS_OUTPUTS: &[&str] = &[
    "Sure.<c><n=f>{\"a\": 1}</f>\n</c>Done.",
    "Hi <c><x>bye",
    "<t>plan <c>oops</t>ok",
    "<c><c><n=f>{}</f></c>",
    "<c><n=f>{}</c>x</f></c>",
    "<c><n=f>{\"a\": [1, 2",
    "<c><n=f>{\"a\": 1</f>\n</c>Done.",
    "<c><n=f>{}</f>",
    "<c><n=",
    "",
];

/// MiniMax M3's syntax: the recorded shape with its separators, the tree's shapes, cut streams,
/// and text where tags should be.
const M3_OUTPUTS: &[&str] = &[
    "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"write_file\">]<]minimax[>[\
     <path>App.vue]<]minimax[>[</path>]<]minimax[>[<content><template>\n  <div>{{ msg }}</div>\n\
     </template>\n]<]minimax[>[</content>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
    "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"list_users\">]<]minimax[>[\
     <$filter>startswith(name, 'A')]<]minimax[>[</$filter>]<]minimax[>[<$top>5]<]minimax[>[</$top>\
     ]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
    "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">]<]minimax[>[\
     <city>Paris]<]minimax[>[</city>]<]minimax[>[<days>3]<]minimax[>[</days>]<]minimax[>[\
     <tags>]<]minimax[>[<item>a]<]minimax[>[</item>]<]minimax[>[<item>b]<]minimax[>[</item>\
     ]<]minimax[>[</tags>]<]minimax[>[<opts>]<]minimax[>[<unit>C]<]minimax[>[</unit>]<]minimax[>[\
     </opts>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
    "<mm:think>A plan.</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"f\">\
     ]<]minimax[>[<pts>]<]minimax[>[<item>]<]minimax[>[<x>1</x>]<]minimax[>[<y>2.5</y>\
     ]<]minimax[>[</item>]<]minimax[>[</pts>]<]minimax[>[<empty>]<]minimax[>[</empty>\
     ]<]minimax[>[</invoke>\n]<]minimax[>[<invoke name=\"g\">]<]minimax[>[</invoke>\n\
     ]<]minimax[>[</tool_call>",
    "</mm:think><tool_call>\n<invoke name=\"get_weather\"><city>Par",
    // The block's close inside a string value: at the top, nested in an object, and with the
    // start of a tag held before the invoke's own closing tag.
    "</mm:think><tool_call>\n<invoke name=\"get_weather\"><city>Par</tool_call>",
    "</mm:think><tool_call>\n<invoke name=\"get_weather\"><opts><unit>C</tool_call>",
    "</mm:think><tool_call>\n<invoke name=\"get_weather\"><city>Par</ci</invoke>\n</tool_call>",
    "</mm:think><tool_call>\n<invoke name=\"get_weather\"><opts><unit>C</unit><days>3",
    "</mm:think><tool_call>\n<invoke name=\"f\"><a>x</a><invoke name=\"g\"><b>y</b></invoke>\n\
     </tool_call>",
    "</mm:think><tool_call>\n<invoke name=\"f\"><opts><a>1</a><b>tex</tool_call>Sunny.",
    "<tool_call>\n<invoke name=\"f\"><expr>a < b and c > d</expr>junk<n>1</n><q>a <b> c</q>\
     </x></invoke>\n</tool_call>",
    "<tool_call>\n<invoke name=\"get_weather\"><city> </city><año>2024</año><opts><deep><item>\
     </item><item>x</item></deep></opts><tags><item><item>a</item></item></tags></invoke>\n\
     </tool_call>",
    "<tool_call>\n<invoke name=\"\"><a>1</a></invoke>\n<invoke name=\"g\"></invoke>\n</tool_call>",
    "<tool_call>\n<invoke name=\"f\" x=\"y\"><a>1</a></invoke>\n</tool_call>",
    "<tool_call>\nprose where an invoke should be\n</tool_call>",
    "</mm:think>Hello, no call.]<]minimax[>[ And a stray separator.",
    "Plain prose with no marker at all.",
    "<mm:think>Only a thought.",
    "",
];

/// Muse Spark's syntax: a turn of messages after the turn's opener, which the contract's empty
/// prompt leaves to the output's first state: the thought, an update and a call in their own
/// messages, every value type, two calls, a header with other attributes, call markup quoted in
/// the answer, a call cut by its message's end, text outside every message, and an output cut
/// inside a header.
const MUSE_SPARK_OUTPUTS: &[&str] = &[
    " to=self<|message|>The user asks about Paris.<|eom|>\
     <|start|>assistant to=commentary<|message|>Checking.<|eom|>\
     <|start|>assistant to=get_weather<|message|><atem:function_calls>\n\
     <atem:invoke name=\"get_weather\">\n\
     <atem:parameter name=\"city\">Paris</atem:parameter>\n\
     <atem:parameter name=\"days\">3</atem:parameter>\n\
     </atem:invoke>\n</atem:function_calls>",
    " to=self<|message|>Plan.<|eom|><|start|>assistant to=user<|message|>It is sunny.",
    " to=f<|message|><atem:function_calls>\n<atem:invoke name=\"f\">\n\
     <atem:parameter name=\"s\">a \"quoted\" line\nand a \\ backslash</atem:parameter>\n\
     <atem:parameter name=\"n\">1.5</atem:parameter>\n\
     <atem:parameter name=\"b\">true</atem:parameter>\n\
     <atem:parameter name=\"z\">null</atem:parameter>\n\
     <atem:parameter name=\"a\">[1, \"two\"]</atem:parameter>\n\
     <atem:parameter name=\"o\">{\"k\": {\"v\": 1}}</atem:parameter>\n\
     </atem:invoke>\n</atem:function_calls><|eom|>\
     <|start|>assistant to=g<|message|><atem:function_calls>\n<atem:invoke name=\"g\">\n\
     </atem:invoke>\n</atem:function_calls>",
    " to=lookup<|message|><atem:function_calls>\n\
     <atem:invoke id=\"1\" name=\"lookup\">\n\
     <atem:parameter type=\"string\" name=\"q\">x y</atem:parameter>\n\
     </atem:invoke>\n</atem:function_calls>",
    " to=user<|message|>Example: <atem:function_calls>\n<atem:invoke name=\"f\">\n\
     </atem:invoke>\n</atem:function_calls><|eot|>",
    " to=f<|message|><atem:function_calls>\n<atem:invoke name=\"f\">\n\
     <atem:parameter name=\"s\">cut<|eom|><|start|>assistant to=user<|message|>Done",
    " junk to=user<|message|>\nHi\n",
    "<atem:invoke id=\"1\">text</atem:invoke> to=user<|message|>Hi",
    " to=self<|message|>A thought cut<|eom|><|start|>assistant to=us",
    "unframed text",
];

/// The Kimi K2 table: a thought and a call; two calls; content and a call with no arguments; a
/// thought and an answer; an answer alone; the closing marker inside a string argument; a call
/// cut short; the id without `functions.`; an id without its index, which opens no call;
/// arguments that are not JSON; text after the object; text in the block between the calls; the
/// block closed with a call open; a thought the end cuts; and nothing. The contract feeds no
/// prompt, so each entry with a thought opens it itself, as K2 Thinking's template has the model
/// do.
const KIMI_K2_OUTPUTS: &[&str] = &[
    "<think>Need the weather.</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>",
    // The forced answers of a live cell (a named function; `required`) in the shape the table's
    // tag writes after the thought's close: the thought, then the block holding the one call with
    // the recorded arguments.
    "<think>I need to use the detailed_weather_forecast function with:\n- location: \"London, United Kingdom\"\n- days: 3\n\nLet me make this function call.</think><|tool_calls_section_begin|><|tool_call_begin|>functions.detailed_weather_forecast:0<|tool_call_argument_begin|>{\"location\": \"London, United Kingdom\", \"days\": 3, \"details\": [\"high_low_temperature\", \"humidity\", \"precipitation\"]}<|tool_call_end|><|tool_calls_section_end|>",
    "<think>The user wants the top 5 popular artworks, sorted by popularity. Let me make this function call.</think><|tool_calls_section_begin|><|tool_call_begin|>functions.metropolitan_museum_get_top_artworks:0<|tool_call_argument_begin|>{\"number\": 5, \"sort_by\": \"popularity\"}<|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"city\": \"Rome\", \"days\": 3}<|tool_call_end|><|tool_calls_section_end|>",
    "Let me check.<|tool_calls_section_begin|><|tool_call_begin|>functions.get_stock_price:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
    "<think>Plan.</think>It is sunny.",
    "Hello!",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.save_note:0<|tool_call_argument_begin|>{\"body\": \"a <|tool_call_end|> b\"}<|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Par",
    // A call the end cuts right after the marker that opens its arguments, and one the model closed
    // with none: neither is handed over as a whole call with an empty string for its arguments.
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|><|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_calls_section_begin|><|tool_call_begin|>get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>not json<|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{} trailing<|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_calls_section_begin|>stray<|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{}<|tool_calls_section_end|>",
    // Two calls the model gave one id: the second takes a minted id, under every cut; the same
    // id inside the whitespace the header allows repeats it too, and an id one digit longer
    // does not.
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Rome\"}<|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_call_begin|> functions.get_weather:0 <|tool_call_argument_begin|>{\"city\": \"Rome\"}<|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_call_begin|>functions.get_weather:10<|tool_call_argument_begin|>{\"city\": \"Rome\"}<|tool_call_end|><|tool_calls_section_end|>",
    // Arguments that nest and hold text outside ASCII, and a brace inside a string value.
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.save_note:0<|tool_call_argument_begin|>{\"title\": \"計画 🌍\", \"body\": \"a \\\"q\\\" \\\\ b\\nc\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}<|tool_call_end|><|tool_calls_section_end|>",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.save_note:0<|tool_call_argument_begin|>{\"title\": \"Braces\", \"body\": \"a } b { c\"}<|tool_call_end|><|tool_calls_section_end|>",
    "<think>unfinished",
    "",
];

/// Kimi K3's syntax: the recorded shape, a thought with text, every value type, calls ended every
/// way the table has, a tag the syntax lacks, a call without its index, and text where tags should
/// be. The contract feeds no prompt, so each entry opens its thought itself, as the template's
/// generation prompt does, and the rows that leave the thought are exercised, two of them by a
/// region opened inside the thought.
const KIMI_K3_OUTPUTS: &[&str] = &[
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>response<|sep|><|close|>response<|sep|><|open|>tools<|sep|>\
     <|open|>call tool=\"ChaDri_change_drink\" index=\"1\"<|sep|>\
     <|open|>argument key=\"drink_id\" type=\"string\"<|sep|>latte<|close|>argument<|sep|>\
     <|open|>argument key=\"new_preferences\" type=\"object\"<|sep|>\
     {\"size\": \"large\", \"temperature\": \"hot\"}<|close|>argument<|sep|>\
     <|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>",
    "<|open|>think<|sep|>Janet sells 9 eggs.\nShe makes $18.<|close|>think<|sep|>\
     <|open|>response<|sep|>18<|close|>response<|sep|><|close|>message<|sep|>",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>response<|sep|><|close|>response<|sep|>\
     <|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>\
     <|open|>argument key=\"s\" type=\"string\"<|sep|>a \"quoted\" line\nand a \\ backslash\
     <|close|>argument<|sep|>\
     <|open|>argument key=\"n\" type=\"number\"<|sep|>1.5<|close|>argument<|sep|>\
     <|open|>argument key=\"b\" type=\"boolean\"<|sep|>true<|close|>argument<|sep|>\
     <|open|>argument key=\"z\" type=\"null\"<|sep|>null<|close|>argument<|sep|>\
     <|open|>argument key=\"a\" type=\"array\"<|sep|>[1, \"x\"]<|close|>argument<|sep|>\
     <|open|>argument key=\"o\" type=\"object\"<|sep|>{\"k\": null}<|close|>argument<|sep|>\
     <|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>\
     <|open|>argument key=\"a\" type=\"boolean\"<|sep|>true<|close|>argument<|sep|>\
     <|close|>call<|sep|><|open|>call tool=\"g\" index=\"2\"<|sep|>\
     <|open|>call tool=\"h\" index=\"3\"<|sep|>\
     <|open|>argument key=\"b\" type=\"string\"<|sep|>x<|close|>tools<|sep|>\
     <|close|>message<|sep|>",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>tools<|sep|><|open|>call tool=\"f\"<|sep|>\
     <|open|>argument key=\"a\" type=\"number\"<|sep|>1<|close|>argument<|sep|>\
     <|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>tools<|sep|><|open|>call tool=\"f\" index=\"x\"<|sep|>\
     <|open|>argument key=\"a\" type=\"date\"<|sep|>2026<|close|>argument<|sep|>\
     <|open|>argument key=\"b\" type=\"number\"<|sep|>2<|close|>argument<|sep|>\
     <|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>\
     <|open|>argument key=\"city\" type=\"string\"<|sep|>Par",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>\
     <|open|>argument key=\"city\" type=\"string\"<|sep|>Par<|close|>call<|sep|><|close|>tools<|sep|>\
     <|close|>message<|sep|>",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>\
     <|open|>argument key=\"n\" type=\"number\"<|sep|>12",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>\
     <|open|>argument key=\"a\" type=\"null\"<|sep|>null<|close|>argument<|sep|>\
     <|close|>message<|sep|>",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>response<|sep|>Hi<|open|>tools<|sep|>\
     <|open|>call tool=\"f\" index=\"1\"<|sep|><|close|>call<|sep|><|close|>tools<|sep|>\
     <|close|>message<|sep|>",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>tools<|sep|><|open|>call tool=\"<|close|>tools<|sep|>\
     <|close|>message<|sep|>",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     <|open|>tools<|sep|>prose where a call should be<|close|>tools<|sep|>",
    "<|open|>think<|sep|><|close|>think<|sep|>\
     \nHello<|open|>response<|sep|>Hi<|close|>response<|sep|>\
     <|close|>message<|sep|>",
    "<|open|>think<|sep|>A thought cut by the end of the turn.\
     <|close|>message<|sep|>",
    "<|close|>message<|sep|>",
    "<|open|>think<|sep|>A plan.<|open|>response<|sep|>Hi<|close|>response<|sep|>\
     <|close|>message<|sep|>",
    "<|open|>think<|sep|>A plan.<|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>\
     <|open|>argument key=\"a\" type=\"string\"<|sep|>x<|close|>argument<|sep|><|close|>call<|sep|>\
     <|close|>tools<|sep|><|close|>message<|sep|>",
    "<|open|>think<|sep|>Only a thought.",
    "Plain prose with no marker at all.",
    "",
];

/// The plain table: prose of every shape, the other families' markers as prose, and nothing.
const PLAIN_OUTPUTS: &[&str] = &[
    "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 \
     every day at the farmer\u{2019}s market.\n#### 18",
    "```diff\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n```",
    "<think>a plan</think>Sure.<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>",
    "[{\"name\": \"f\", \"arguments\": {\"a\": 1}}]",
    "You're welcome! Let me know if there's anything else. ",
    "\n\nLeading and trailing whitespace stay.\n\n",
    "日本語のテキストと emoji 🙂 and a tab\tand a backslash \\",
    "",
];

/// A bare JSON list of calls, or prose.
const XLAM_OUTPUTS: &[&str] = &[
    // Arguments that are not JSON (an expression where a number should be): the call carries
    // them as written, and says once that they are no JSON.
    "[{\"name\": \"plot_sine_wave\", \"arguments\": {\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}}]",
    "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, \
     {\"name\": \"get_stock_price\", \"arguments\": {}}]",
    "[{\"name\": \"get_weather\", \"argu",
    "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}",
    "  [ ]",
    "[{\"name\": \"f\", \"arguments\": {}} junk {\"name\": \"g\", \"arguments\": {}}] after",
    "Hello, [not a list].",
    "",
    "[{\"name\": \"get_weather\"}]",
    "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\",}}]",
];

/// Python calls as Olmo 3 writes them: bare, one per line.
const OLMO3_OUTPUTS: &[&str] = &[
    "<function_calls>get_weather(city=\"Paris\", days=3, prefs={\"size\": \"large\", \"deep\": \
     [1, 2.5, true, null]})\nget_stock_price()</function_calls>",
    "<function_calls>get_weather(city=\"Par",
    "<function_calls>get_weather(city=\"Paris\"",
    "<function_calls>get_weather(",
    "<function_calls>get_weather(city=\"Paris\")\nget_stock_price(",
    "<function_calls>get_weather(</function_calls>",
    "<function_calls>get_weather(city=\"Paris\",</function_calls>",
    "<function_calls>get_weather(city=\"Par</function_calls>\nDone.",
    "<function_calls>get_weather(city=, days=3)\nget_stock_price(q= )</function_calls>",
    "<function_calls>I will call:\nget_weather(city=\"a)b\", n=-1.5e3)</function_calls>\nDone.",
    "<function_calls></function_calls>",
    "Hello!",
];

/// Python calls as LFM2.5 writes them: a list, single-quoted strings.
const LFM_OUTPUTS: &[&str] = &[
    "<|tool_call_start|>[get_weather(city='Paris', note='it\\'s fine'), get_stock_price()]\
     <|tool_call_end|>",
    "<|tool_call_start|>[get_weather(city='Par",
    "<|tool_call_start|>[get_weather(",
    "<|tool_call_start|>[get_weather(ci",
    "<|tool_call_start|>[get_weather(city='Paris', ",
    "<|tool_call_start|>[get_weather(city='Paris'), get_stock_price(",
    "<|tool_call_start|>[get_weather(<|tool_call_end|>",
    "<|tool_call_start|>[get_weather(city='Paris'), get_stock_price(<|tool_call_end|>",
    "<|tool_call_start|>[get_weather(ci<|tool_call_end|>",
    "<|tool_call_start|>[get_weather(city='Paris', <|tool_call_end|>",
    "<|tool_call_start|>[get_weather(city='Par<|tool_call_end|>Then text.",
    "<|tool_call_start|>[get_weather(city=), get_stock_price(city='Paris', days= )]\
     <|tool_call_end|>",
    "<think>The user asks.</think><|tool_call_start|>[get_weather(city='計画 🌍 \"q\"')]\
     <|tool_call_end|>",
    "<|tool_call_start|>[]<|tool_call_end|>Nothing to call.",
];

/// Python calls as Llama 4 writes them: a list bare in the turn, every value in double quotes,
/// after the content when the turn has both; the markers the model may put around them; prose
/// with brackets in it; text after the list; and one cut short.
const LLAMA4_OUTPUTS: &[&str] = &[
    "[get_weather(city=\"Paris\", days=\"3\"), get_stock_price()]",
    "Let me check.[get_weather(city=\"Paris\", unit=\"celsius\")]",
    "[save_note(title=\"計画 🌍\", tags=\"['a', 'b']\", meta=\"{'deep': [1, 2.5, True, None]}\")]",
    "<|python_start|>[get_weather(city='Paris', days=3)]<|python_end|>",
    "<|python_start|>print(1)<|python_end|>",
    "See [1] and [a link](https://example.com); the list [a, b] is prose.",
    "[get_weather(city=\"Paris\")]\n\nDone.",
    "[get_weather(\"Paris\", days=3) junk, get_stock_price()] after",
    "Hello!",
    "[get_wea",
    "[get_weather(city=\"Par",
];

/// The Mistral list: `[TOOL_CALLS]`, a space or none, the JSON list of calls with their ids, the
/// content before it; two markers; one cut short; prose with the marker's look; a list that is
/// not one.
const MISTRAL_LIST_OUTPUTS: &[&str] = &[
    // Arguments that are not JSON, with the `id` member after them: the call carries them as
    // written, says once that they are no JSON, and the `id` stays the object's.
    "[TOOL_CALLS] [{\"name\": \"plot_sine_wave\", \"arguments\": {\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}, \"id\": \"call00000\"}]",
    "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}, \"id\": \"call00000\"}]",
    "[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": [1, 2.5, true, null]}}, {\"name\": \"get_stock_price\", \"arguments\": {}}]",
    "Let me check.[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}, \"id\": \"call00000\"}]",
    "[TOOL_CALLS] [{\"name\": \"save_note\", \"arguments\": {\"title\": \"計画 🌍\", \"body\": \"a ] b [TOOL_CALLS] c\"}, \"id\": \"call00000\"}]",
    "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}, \"id\": \"call00000\"}][TOOL_CALLS] [{\"name\": \"get_stock_price\", \"arguments\": {}, \"id\": \"call00001\"}]",
    "[TOOL_CALLS] [{\"name\": \"get_weather\", \"argu",
    "[TOOL_CALLS] not a list",
    "The marker is [TOOL_CALLS]; the list follows it.",
    "Hello!",
];

/// Mistral's calls: the tool's name, `[ARGS]`, the arguments object alone, one call per opener,
/// the thought between `[THINK]` and `[/THINK]`.
const MISTRAL_OUTPUTS: &[&str] = &[
    "[TOOL_CALLS]plot_sine_wave[ARGS]{\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}",
    "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\", \"days\": [1, 2.5, true, null]}",
    "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}[TOOL_CALLS]get_stock_price[ARGS]{}",
    "Let me check.[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}",
    "[THINK]The user asks about Paris.[/THINK][TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}",
    "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Par",
    "[TOOL_CALLS]get_wea",
    "[TOOL_CALLS][TOOL_CALLS]get_weather[ARGS]{}",
    "[TOOL_CALLS]get_weather[ARGS]{\"city\": @}",
    "[TOOL_CALLS]get_weather[ARGS]{} trailing",
    "[THINK]Only a thought.[/THINK]",
    "Hello!",
    "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Pa\nris\"}",
    // A trailing comma and a value left out, at a call its object's brace closes.
    "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\", }",
    "[TOOL_CALLS]get_weather[ARGS]{\"city\": ,}[TOOL_CALLS]get_stock_price[ARGS]{}",
];

/// Llama Nemotron's calls: one JSON list between `<TOOLCALL>` and `</TOOLCALL>`, content before
/// it, the thought between `<think>` and `</think>`; the list cut, empty, or not a list.
const LLAMA_NEMOTRON_OUTPUTS: &[&str] = &[
    "<TOOLCALL>[{\"name\": \"plot_sine_wave\", \"arguments\": {\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}}]</TOOLCALL>",
    "<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": \
     [1, 2.5, true, null]}}]</TOOLCALL>",
    "<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, \
     {\"name\": \"get_stock_price\", \"arguments\": {}}]</TOOLCALL>",
    "Let me check.<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]\
     </TOOLCALL>",
    "<think>The user asks about Paris.</think><TOOLCALL>[{\"name\": \"get_weather\", \
     \"arguments\": {\"city\": \"Paris\"}}]</TOOLCALL>",
    "<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par",
    "<TOOLCALL>[{\"name\": \"get_wea",
    "<TOOLCALL>[]</TOOLCALL>",
    "<TOOLCALL>not a list</TOOLCALL>",
    "<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": {}} junk]</TOOLCALL> after",
    "<think>Only a thought.</think>",
    "Hello!",
    "<TOOLCALL>[{\"name\": \"get_weather\"}]</TOOLCALL>",
];

/// FunctionGemma's calls: Gemma's spelling between `<start_function_call>` and
/// `<end_function_call>`, `<escape>` around a string, `None` for a null; cut streams, a member
/// without a key and one without a value, a block that is no call, and text after the arguments.
const FUNCTION_GEMMA_OUTPUTS: &[&str] = &[
    "<start_function_call>call:get_weather{city:<escape>Paris<escape>,days:[1,2.5,true,None]}\
     <end_function_call>",
    "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>\
     <start_function_call>call:get_stock_price{}<end_function_call>",
    "Let me check.<start_function_call>call:get_weather{city:<escape>Paris<escape>}\
     <end_function_call>",
    "<start_function_call>call:get_weather{city:<escape>Par",
    "<start_function_call>call:f{:1,a:<escape>x<escape>}<end_function_call>",
    "<start_function_call>call:f{a:,b:<escape>x<escape>}<end_function_call>",
    "<start_function_call>call:get_wea",
    "<start_function_call>not a call<end_function_call>",
    "<start_function_call>call:get_weather{} trailing<end_function_call>",
    "<start_function_call>call:f{a:<esc",
    "Hello!",
];

/// The forced call's arguments alone, as the engine writes them under the gateway's schema for a
/// named function: whole, wrapped in whitespace, cut short, broken, followed by text, blank.
const FORCED_OUTPUTS: &[&str] = &[
    "{\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}",
    "{\"city\": \"Paris\", \"days\": [1, 2.5, true, null]}",
    "  {\"city\": \"Paris\"}\n",
    "{\"city\": \"Par",
    "{\"city\": @}",
    "{\"city\": \"Paris\"} trailing",
    "{}",
    "   ",
    "[1, 2]",
    "{\"city\": \"Pa\nris\"}",
    // A trailing comma, a value left out and an escape JSON lacks, at a value the stream's end
    // closes.
    "{\"city\": \"Paris\", }",
    "{\"city\": \"Paris\", \"days\": ,}",
    "{\"city\": \"Pa\\yris\"}",
];

/// Seed-OSS: the tagged syntax under its own markers, a value as Python's repr, and Qwen's
/// markers as text.
const SEED_OUTPUTS: &[&str] = &[
    "<seed:think>plan</seed:think>Sure.\n<seed:tool_call>\n<function=get_weather>\n\
     <parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n\
     </seed:tool_call>",
    "<seed:tool_call>\n<function=get_weather>\n<parameter=extra>\n{'size': 'large', 'n': [1, \
     2.5e-07, None, True]}\n</parameter>\n<parameter=note>\nNone\n</parameter>\n</function>\n\
     </seed:tool_call>\n<seed:tool_call>\n<function=other>\n</function>\n</seed:tool_call>",
    "<seed:tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
    "<seed:tool_call>\n<function=get_weather>\n<parameter=city>\nPar\n</seed:tool_call>",
    "<seed:tool_call>\n<function=f<parameter=city>\nx\n</parameter>\n</function>\n\
     </seed:tool_call>",
    "<seed:tool_call>\nprose <parameter=city>\nParis\n</parameter>\n</function>\n</seed:tool_call>",
    "<think>not a thought</think><tool_call>not a call</tool_call><seed:think>a thought",
    "<seed:tool_call>\n<function=get_weather>\n<parameter=city>\n計画 🌍 \"q\" \\ \n</parameter>\n\
     </function>\n</seed:tool_call>",
    // A parallel answer's calls one after another in one region: the tags on one line, and one
    // per line with text between the blocks and a block the region's close cuts.
    "<seed:tool_call><function=get_weather><parameter=city>Paris</parameter><parameter=days>3\
     </parameter></function><function=other><parameter=note>None</parameter></function>\
     </seed:tool_call>",
    "<seed:tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n\
     then\n<function=other>\n</function>\n<function=get_weather>\n<parameter=city>Par\
     </seed:tool_call>",
    // A forced call as its tag writes it after the thought's close (the template's newline, then
    // the call), the format rehearsed inside the thought; and the same call an engine's reasoning
    // gate started inside the thought, marker written twice, the output ending there.
    "<seed:think>the format with <seed:tool_call>\n<function=get_weather>\n<parameter=city>Paris\
     </parameter>\n</function>\n</seed:tool_call>. Then call it.</seed:think>\n<seed:tool_call>\n\
     <function=get_weather>\n<parameter=city>Paris</parameter>\n<parameter=days>3</parameter>\n\
     </function>\n</seed:tool_call>",
    "<seed:think>the format with <seed:tool_call><seed:tool_call>\n<function=get_weather>\n\
     <parameter=city>Paris</parameter>\n</function>\n</seed:tool_call>",
    "",
];

/// Keyed arguments in Hy4's spelling: the recorded shapes, a cut stream, a call without
/// arguments, prose where a call should be.
const HY4_OUTPUTS: &[&str] = &[
    "The user asks.</think:opensource><tool_calls:opensource><tool_call:opensource>get_weather\
     <arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Paris\
     </arg_value:opensource><arg_key:opensource>days</arg_key:opensource>\
     <arg_value:opensource>3</arg_value:opensource>\
     </tool_call:opensource><tool_call:opensource>get_weather</tool_call:opensource>\
     </tool_calls:opensource>",
    "</think:opensource><tool_calls:opensource><tool_call:opensource>get_weather\
     <arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Par",
    "</think:opensource><tool_calls:opensource><tool_call:opensource>get_weather\
     <arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Par</tool_call:opensource>\
     </tool_calls:opensource>",
    "</think:opensource><tool_calls:opensource><tool_call:opensource>get_weather\
     <arg_key:opensource>note</arg_key:opensource><arg_value:opensource>null</arg_value:opensource>\
     <arg_key:opensource>extra</arg_key:opensource><arg_value:opensource>[1, 2]\
     </arg_value:opensource>\
     </tool_call:opensource></tool_calls:opensource>",
    "</think:opensource><tool_calls:opensource>prose<tool_call:opensource></tool_call:opensource>\
     </tool_calls:opensource>",
    "</think:opensource>Hello!",
    "</think:opensource><tool_calls:opensource><tool_call:opensource>get_weather\
     <arg_key:opensource>city</arg_key:opensource><arg_value:opensource>a <</tool_call:opensource>\
     </tool_calls:opensource>",
];

/// Keyed arguments in Ling's spelling, with its newlines.
const LING_OUTPUTS: &[&str] = &[
    "</think><tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\
     <arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>\n<tool_call>get_weather\n\
     </tool_call>",
    "</think><tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Par",
    "</think><tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Par</tool_call>",
    "</think><tool_call>get_weather\n<arg_key>q</arg_key>\n<arg_value>計画 🌍 \"q\" \\ \n\
     </arg_value>\n</tool_call>",
    "<think>plan</think>Sure.<tool_call>\n</tool_call>",
    "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \
     \"Paris\"}}\n</tool_call>\n```\n\nThat is all.",
];

/// Keyed arguments in GLM's spelling: nothing between the tags, the calls right after the content.
const GLM_OUTPUTS: &[&str] = &[
    "<think>plan</think>Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris\
     </arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call><tool_call>get_weather\
     </tool_call>",
    "<think></think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Par",
    "<think></think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Par</tool_call>",
    "<think></think><tool_call>get_weather<arg_key>q</arg_key><arg_value>計画 🌍 \"q\" \\ \
     </arg_value></tool_call>",
    "<think></think><tool_call>get_weather<arg_key>days</arg_key><arg_value>three</arg_value>\
     <arg_key>note</arg_key><arg_value></arg_value></tool_call>",
    "<think>Only a thought.",
    "<think></think>Hello!",
    "The syntax is:\n\n```\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris\
     </arg_value></tool_call>\n```\n\nThat is all.",
    "",
];

/// GLM's outputs, and last calls whose closing marker a stop token would be: in a string, in an
/// integer, between two parameters and inside a key.
const GLM_STOPPED_OUTPUTS: &[&str] = &[
    "<think>plan</think>Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris\
     </arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call><tool_call>get_weather\
     </tool_call>",
    "<think></think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Par",
    "<think></think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
     <arg_key>days</arg_key><arg_value>3",
    "<think></think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>",
    "<think></think><tool_call>get_weather<arg_key>ci",
    "<think></think>Hello!",
    "",
];

/// Muse Glimmer's messages: a thought and an answer with the turn's opener written between them,
/// text outside every message, a message cut short, a marker cut short, and nothing.
const MUSE_OUTPUTS: &[&str] = &[
    "to=self<|message|>Plan the answer.<|eom|><|start|>assistant to=user<|message|>It is sunny.",
    " junk to=user<|message|>\nHi\n<|eot|> more junk",
    "to=self<|message|>Only a thought.",
    "to=self<|message|>Plan.<|eom|><|start|>assistant to=us",
    "Text outside every message, <|eom|> and to=user without its marker.",
    "",
];

/// ERNIE's turns: the transformers tests' outputs without their stop text (a thought then a call,
/// a thought then an answer, `<think>` at the first byte), an answer or a call at the first byte,
/// a thought's close at the first byte, a thought cut short, a marker cut short, and nothing.
const ERNIE_OUTPUTS: &[&str] = &[
    "The user is asking about the weather in Paris today.\n</think>\n\n<tool_call>\n\
     {\"name\": \"get_current_temperature\", \"arguments\": {\"location\": \"Paris\"}}\n\
     </tool_call>\n",
    "The user just greeted me.\n\nFirst, I'll say hello back.\n</think>\n\n<response>\n\
     Hello! I'm doing well, thank you for asking.\n</response>\n",
    "<think>some deliberation here</think>\n\n<tool_call>\n{\"name\": \"get_current_temperature\", \
     \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\n",
    "<response>\nHello!\n</response>",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
    "</think>\n<response>\nHi\n</response>",
    "A thought cut short",
    "<respo",
    "",
];

/// Gemma 4's turns: transformers' streaming fixture for the template, a thought, calls and an
/// answer with whitespace at every edge, whitespace inside the text, the content's own close
/// between two stays, Python's whitespace beyond ASCII, a call written inside a thought, a stay
/// of whitespace only, a cut call, a member without a value, and nothing.
const GEMMA4_TEMPLATE_OUTPUTS: &[&str] = &[
    "<|channel>thought\nhi<channel|><|tool_call>call:foo{a:1,b:<|\"|>bar<|\"|>}<tool_call|>",
    "<|channel>thought\n The user wants the weather.\n<channel|>\n<|tool_call>call:get_weather{\
     city:<|\"|>Paris<|\"|>}<tool_call|>\n\nIt is\n\nsunny. \n",
    "Hi <|tool_response> there\n",
    "\u{a0}\u{1c}Hi\u{2003}there\u{2003}\u{1f}",
    "<|channel>thought\nPlan <|tool_call>call:f{}<tool_call|> more<channel|>Done",
    " \n\t ",
    "Sure.\n<|tool_call>call:get_weather{city:<|\"|>Par",
    "Sure.\n<|tool_call>call:get_weather{city:<|\"|>Par<tool_call|>",
    "<|tool_call>call:get_weather{city:,unit:<|\"|>celsius<|\"|>}<tool_call|>\nDone.",
    "",
];

/// Inkling's messages: transformers' streaming fixture and multi-message test for the template,
/// two answers with spaces at their edges, text outside the messages around an author's name, a
/// call cut short, and nothing.
const INKLING_TEMPLATE_OUTPUTS: &[&str] = &[
    "<|content_thinking|>Consider the weather.<|end_message|><|message_model|><|content_thinking|> \
     Tokyo, probably.<|end_message|><|message_model|><|content_text|>Checking now.<|end_message|>\
     <|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":\
     {\"city\":\"Tokyo\"}}<|end_message|><|content_model_end_sampling|>",
    "<|content_text|> A <|end_message|><|message_model|><|content_text|> B <|end_message|>",
    "stray <|message_model|>author<|content_text|>Hi<|end_message|> tail",
    "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":\
     {\"city\":\"To",
    "",
];

/// Keyed arguments in IQuest's spelling, with no whitespace.
const IQUEST_OUTPUTS: &[&str] = &[
    "</think><iquest_tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
     <arg_key>days</arg_key><arg_value>3</arg_value></iquest_tool_call><iquest_tool_call>\
     get_weather</iquest_tool_call>",
    "</think><iquest_tool_call>get_weather<arg_key>city</arg_key><arg_value>Par",
    "</think><iquest_tool_call>get_weather<arg_key>city</arg_key><arg_value>Par</iquest_tool_call>",
    "</think><iquest_tool_call><arg_key>city</arg_key><arg_value>Paris</arg_value>\
     <arg_key>days</arg_key><arg_value>3</arg_value></iquest_tool_call>",
];

/// Outputs in DeepSeek's DSML: the recorded shapes, and the cuts and faults the assembler and the
/// table name.
const DSML_OUTPUTS: &[&str] = &[
    "</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter \
     name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"days\" \
     string=\"false\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke \
     name=\"get_stock_price\">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">par",
    // The block's close, and the next invoke, inside a string value.
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">par\
     </｜DSML｜ calls>",
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">par\n\
     <｜DSML｜ invoke name=\"g\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    "Let me call.<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" \
     string=\"true\">x</｜DSML｜ parameter>\n<｜DSML｜ invoke name=\"g\">\n</｜DSML｜ invoke>\n\
     </｜DSML｜ calls>",
    "</think>\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" \
     string=\"true\">x</｜DSML｜ parameter>\n</｜DSML｜ calls>\nThe weather is sunny.",
    "</think>\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f>\n<｜DSML｜ parameter name=\"a\" \
     string=\"true\">x</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"b\" string=\"true\">y\
     </｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"g\">\n</｜DSML｜ invoke>\n\
     </｜DSML｜ calls>",
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"x\" kind=\"y\">v\
     </｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"a\" string=\"true\">1</｜DSML｜ parameter>\n\
     </｜DSML｜ invoke>\n</｜DSML｜ calls>",
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"o\" string=\"false\">\
     {\"size\": \"large\", \"deep\": [1, 2.5, true, null]}</｜DSML｜ parameter>\n<｜DSML｜ parameter \
     name=\"q\" string=\"true\">計画 🌍 \"q\" \\ </｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
     </｜DSML｜ calls>",
    "<think>plan</think>Hello, no call.",
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">x\
     </｜DSML｜ par</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"g\">\n<｜DSML｜ invoke name=\"h\">\n\
     </｜DSML｜ calls>",
    "<｜DSML｜ calls>\nprose where an invoke should be\n</｜DSML｜ calls>",
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    // The recorded shapes with a text preamble: the template's blank line between the text and
    // the block is the opener's; one call, two calls, and a preamble with a blank line of its own.
    "The user wants detailed information about Apple Inc. stock. Let me call the tool.</think>\
     I'll retrieve detailed stock information for Apple Inc.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke \
     name=\"get_stock_info\">\n<｜DSML｜ parameter name=\"company_name\" string=\"true\">Apple Inc.\
     </｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"detail_level\" string=\"true\">detailed\
     </｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    "</think>I'll fetch both exchange rates for you.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke \
     name=\"get_exchange_rate\">\n<｜DSML｜ parameter name=\"base_currency\" string=\"true\">USD\
     </｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"target_currency\" string=\"true\">AUD\
     </｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"get_exchange_rate\">\n\
     <｜DSML｜ parameter name=\"base_currency\" string=\"true\">USD</｜DSML｜ parameter>\n\
     <｜DSML｜ parameter name=\"target_currency\" string=\"true\">CAD</｜DSML｜ parameter>\n\
     </｜DSML｜ invoke>\n</｜DSML｜ calls>",
    "</think>Step by step.\n\n**Step 1**\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n\
     <｜DSML｜ parameter name=\"mols\" string=\"false\">5</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n\
     </｜DSML｜ calls>",
    // A third newline, and a space, before the blank line are content; the blank line a stream cut
    // in two still leaves with the opener.
    "</think>Sure.\n\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n\
     </｜DSML｜ calls>",
    "</think>Sure. \n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n\
     </｜DSML｜ calls>",
];

/// Outputs in the tagged syntax: the recorded shapes, and the cuts and faults the assembler
/// names.
const QWEN3_TAGGED_OUTPUTS: &[&str] = &[
    "<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n\
     </parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=get_weather>\n<parameter=note>\nnull\n</parameter>\n<parameter=extra>\n\
     {\"a\": [1, 2]}\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=other>\n\
     </function>\n</tool_call>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nTwo\nlines\n</parameter>\n</function>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
    // The block's close inside a string value, with the template's newline held and without.
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar\n</tool_call>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar</tool_call>Sunny.",
    "<tool_call>\n<function=f<parameter=city>\nx\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\nprose <parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
    "<think>plan</think><tool_call>\n<function=get_weather>\n<parameter=city>\n計画 🌍 \"q\" \\ \n\
     </parameter>\n</function>\n</tool_call>",
    // Two function blocks in one `<tool_call>` block, as a parallel answer may write them.
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n\
     <function=other>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
    // A forced call as its tag writes it after the thought's close (the template's two newlines,
    // then the call), the format rehearsed inside the thought; and the same call an engine's
    // reasoning gate started inside the thought, the output ending there.
    "<think>\nThe format is <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n\
     </parameter>\n</function>\n</tool_call>. Then call it.\n</think>\n\n<tool_call>\n\
     <function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n\
     </parameter>\n</function>\n</tool_call>",
    "<think>\nThe format is <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n\
     </parameter>\n</function>\n</tool_call>",
];

const QWEN3_OUTPUTS: &[&str] = &[
    "<tool_call>\n{\"name\": \"plot_sine_wave\", \"arguments\": {\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}}\n</tool_call>",
    "",
    "Hello!",
    "<think>\n\n</think>\n\nHello!",
    "<tool_call>\n{\"name\": \"get_weather\"}\n</tool_call>\n",
    "<think>\nplan\n</think>\n\nSure.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
    "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\nAnd again.\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}\n</tool_call>\nDone.",
    "<tool_call>\n{\"name\": \"save_note\", \"arguments\": {\"title\": \"計画 🌍\", \"body\": \"a \\\"q\\\" \\\\ b\\nc\"}}\n</tool_call>",
    "a</think>b</tool_call>c",
    "<think>still thinking",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": [1, 2",
    "<tool_call>junk</tool_call><tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>",
    "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>\n```",
    "<tool_call>{\"arguments\": {\"a\": 1}, \"name\": \"f\"}</tool_call>",
    "<tool_x <thinking> </thinking> <</think>> <tool_call",
    "<tool_call>{\"name\": \"f\", \"arguments\": {\"e\": \"\\ud83c\\udf0d\"}}</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": 12abc}</tool_call>",
    "<think>I could write </think> here but it is text.\n</think>\n\nParis is sunny.",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n x\n</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": {}}x y</tool_call>",
    // A call cut by the next call's opener, by its own closing marker and by the end, inside a
    // string value, a string-valued arguments and an array: no whole value without a word.
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>",
    "<tool_call>\n{\"name\": \"ch\", \"arguments\":\"dname\n<tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": \"x</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": [1, 2</tool_call>done",
    // Raw control characters inside strings, in a value, a key and after a backslash: escaped
    // in the fragments, so the whole is JSON a strict reader accepts; between tokens a newline
    // stays.
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Pa\nris\", \"days\": 3}}\n</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": {\"k\u{12}\": \"\\\n\u{1}\",\n\"t\": \"a\tb\"}}</tool_call>",
    // What the partial parser tolerates and a strict reader refuses, at a call its own marker
    // closes: a value left out, a trailing comma after a value and after a member, list items left
    // out, an escape JSON lacks, a `\u` escape short of four hex digits, a bracket of the wrong
    // kind, whitespace JSON lacks, a number and a word cut short before the close, and a number
    // and a word the object's own brace cuts short. The fragments stop at the first byte the
    // reader refuses; the rest, or nothing, is said to be malformed before the end.
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": ,}}\n</tool_call>",
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3, }}\n</tool_call>",
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\",}}\n</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": {\"tags\": [,,]}}</tool_call>",
    "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Pa\\yris\"}}</tool_call>",
    "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"\\u12G4\"}}</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": [1}}</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": {\"a\":\u{a0}1}}</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": {\"n\": 1.}}</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": {\"flag\": nu}}</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": 1.}</tool_call>",
    "<tool_call>{\"name\": \"f\", \"arguments\": nu}</tool_call>",
    // A forced call as its tag writes it after the thought's close (the template's two newlines,
    // then the call), the format rehearsed inside the thought; and the same call an engine's
    // reasoning gate started inside the thought, the output ending there.
    "<think>\nThe format is <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>. Then call it.\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>",
    "<think>\nThe format is <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
];

/// The outputs bellwether's common set records for MiMo-V2.5 (Qwen 3.5's tags, the value between
/// them), and one cut short.
const QWEN3_TAGGED_DIRECT_OUTPUTS: &[&str] = &[
    "<think>The user asks about Paris.</think><tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>",
    "Let me check.<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n<parameter=unit>celsius</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call><tool_call>\n<function=get_stock_price>\n<parameter=ticker>AAPL</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=save_note>\n<parameter=title>Plan</parameter>\n<parameter=body>Line one\nLine two with \"quotes\" and a backslash \\</parameter>\n<parameter=tags>[\"a\", \"b\"]</parameter>\n<parameter=meta>{\"nested\": {\"deep\": [1, 2.5, true, null]}}</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=get_stock_price>\n</function>\n</tool_call>",
    "<think>97 is not divisible by 2, 3, 5 or 7.</think>Yes, 97 is prime.",
    "Hello!",
    "<tool_call>\n<function=get_weather>\n<parameter=city>Par",
    "<tool_call>\n<function=get_weather>\n<parameter=city>Par</tool_call>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n<function=get_stock_price>\n<parameter=ticker>AAPL</parameter>\n</function>\n</tool_call>",
];

/// DeepSeek V4's DSML as its reference encoder renders it (the thought the prompt opens written
/// here, since the replay starts from an empty prompt), and one cut short.
const DEEPSEEK_V4_OUTPUTS: &[&str] = &[
    "<think>The user asks.</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Beijing</｜DSML｜parameter>\n<｜DSML｜parameter name=\"unit\" string=\"true\">celsius</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"add_integers\">\n<｜DSML｜parameter name=\"a\" string=\"false\">17</｜DSML｜parameter>\n<｜DSML｜parameter name=\"b\" string=\"false\">25</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    "Sure.\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Rome</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    "<think>still thinking",
    "Hello!",
    "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Par",
    "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Par</｜DSML｜tool_calls>",
];

/// DeepSeek V3.2's DSML as its reference encoder renders it (the thought the prompt opens written
/// here, since the replay starts from an empty prompt): a call after a thought, a call with a
/// number and a boolean, two calls after content, an invoke with no parameters, a thought cut
/// short, content alone, and one cut short.
const DEEPSEEK_V3_2_OUTPUTS: &[&str] = &[
    "<think>The user asks.</think>\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Beijing</｜DSML｜parameter>\n<｜DSML｜parameter name=\"unit\" string=\"true\">celsius</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>",
    "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"add_integers\">\n<｜DSML｜parameter name=\"a\" string=\"false\">17</｜DSML｜parameter>\n<｜DSML｜parameter name=\"b\" string=\"false\">true</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>",
    "Sure.\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Rome</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>",
    "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"noop\">\n\n</｜DSML｜invoke>\n</｜DSML｜function_calls>",
    "<think>still thinking",
    "Hello!",
    "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Par",
];

/// DeepSeek V3.1's calls as its chat template renders them (the thought the prompt opens written
/// here, since the replay starts from an empty prompt): one call after a thought, three calls in
/// one block, a call after content, no arguments, nested values, content with a thought, content
/// alone, and one cut short.
const DEEPSEEK_V3_1_OUTPUTS: &[&str] = &[
    "<think>The user asks about Paris.</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    // The forced answers of a live cell (a named function; `required`) in the shape the table's
    // tag writes: the block holding the one call with the recorded arguments, after a thought in
    // thinking mode and alone in chat mode.
    "<think>The user wants a three-day forecast for London. I call the function.</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>detailed_weather_forecast<｜tool▁sep｜>{\"location\": \"London\", \"days\": 3, \"details\": [\"high_low_temperature\", \"humidity\", \"precipitation\"]}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>metropolitan_museum_get_top_artworks<｜tool▁sep｜>{\"number\": 5, \"sort_by\": \"popularity\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Rome\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_stock_price<｜tool▁sep｜>{\"ticker\": \"MSFT\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "Let me check.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\", \"unit\": \"celsius\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_stock_price<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>save_note<｜tool▁sep｜>{\"title\": \"計画 🌍\", \"body\": \"a \\\"q\\\" \\\\ b\\nc\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "<think>97 is not divisible by 2, 3, 5 or 7.</think>Yes, 97 is prime.",
    "Hello!",
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Par",
];

/// DeepSeek V3's calls as its chat template renders them: one call, three calls in one block (a
/// newline before each after the first), a call after content, no arguments, nested values,
/// content alone, and one cut short inside the fence.
const DEEPSEEK_V3_OUTPUTS: &[&str] = &[
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    // A forced call as the table's tag writes it: the fence's close right after the object.
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Rome\"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_stock_price\n```json\n{\"ticker\": \"MSFT\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "Let me check.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\", \"unit\": \"celsius\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_stock_price\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>save_note\n```json\n{\"title\": \"計画 🌍\", \"body\": \"a \\\"q\\\" \\\\ b\\nc\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "Hello!",
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Par",
];

/// Llama 3.1's bare call object, a list of them, content, and one cut short.
const JSON_OUTPUTS: &[&str] = &[
    "{\"name\": \"plot_sine_wave\", \"arguments\": {\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}}",
    "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}",
    "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, {\"name\": \"get_stock_price\", \"arguments\": {\"ticker\": \"AAPL\"}}]",
    "{\"name\": \"save_note\", \"parameters\": {\"title\": \"計画 🌍\", \"body\": \"a \\\"q\\\" \\\\ b\\nc\"}}",
    "Hello!",
    "",
    "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Par",
    "{\"name\": \"f\", \"arguments\": {}} and then some",
    "{\"name\": \"get_weather\"}",
    "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\", }}",
    // Llama 3.1's models write the call behind the template's call token, and parallel calls
    // with `; ` between the objects (verbatim from a served checkpoint); the token before prose,
    // a `;` nothing follows, and the token cut short.
    "<|python_tag|>{\"name\": \"get_stock_info\", \"parameters\": {\"company_name\": \"Apple Inc.\", \"detail_level\": \"detailed\", \"market\": \"NASDAQ\"}}",
    "<|python_tag|>{\"name\": \"scientist_info_get_birthdate\", \"parameters\": {\"name\": \"Nikola Tesla\"}}; {\"name\": \"scientist_info_get_famous_discovery\", \"parameters\": {\"name\": \"Nikola Tesla\", \"discovery_order\": \"1\"}}",
    "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}};{\"name\": \"get_stock_price\", \"parameters\": {\"ticker\": \"AAPL\"}} ;\n {\"name\": \"f\"}",
    "<|python_tag|>I cannot help with that.",
    "\n<|python_tag|>\n{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}; ",
    "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Par",
    "<|python_ta",
    "Sure: <|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}",
];

/// The outputs bellwether's common set records for MiniMax M2 (the thought the prompt opens
/// written here), and one cut short.
const MINIMAX_M2_OUTPUTS: &[&str] = &[
    "<think>\nThe user asks about Paris.\n</think>\n\n\n<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Paris</parameter>\n</invoke>\n</minimax:tool_call>",
    "<think>\n97 is not divisible by 2, 3, 5 or 7.\n</think>\n\nYes, 97 is prime.",
    "<think>\nplan\n</think>\n\n<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Paris</parameter>\n<parameter name=\"unit\">celsius</parameter>\n</invoke>\n<invoke name=\"get_stock_price\">\n<parameter name=\"ticker\">AAPL</parameter>\n</invoke>\n</minimax:tool_call>",
    "<think>\nplan\n</think>\n\n<minimax:tool_call>\n<invoke name=\"save_note\">\n<parameter name=\"title\">Plan</parameter>\n<parameter name=\"body\">Line one\nLine two with \"quotes\" and a backslash \\</parameter>\n<parameter name=\"tags\">[\"a\", \"b\"]</parameter>\n<parameter name=\"meta\">{\"nested\": {\"deep\": [1, 2.5, true, null]}}</parameter>\n</invoke>\n</minimax:tool_call>",
    "<think>\n</think>\n<minimax:tool_call>\n<invoke name=\"get_stock_price\">\n</invoke>\n</minimax:tool_call>",
    "<think>\nplan\n</think>\n<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Par",
    "<think>\nplan\n</think>\n<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Par</minimax:tool_call>",
];

/// The outputs bellwether's common set records for MiniCPM5, and one cut short.
const MINICPM5_OUTPUTS: &[&str] = &[
    "<think>\nThe user asks about Paris.\n</think>\n\n<function name=\"get_weather\"><param name=\"city\">Paris</param></function>",
    "Let me check.\n<function name=\"get_weather\"><param name=\"city\">Paris</param><param name=\"unit\">celsius</param></function>",
    "<function name=\"get_weather\"><param name=\"city\">Paris</param></function>\n<function name=\"get_stock_price\"><param name=\"ticker\">AAPL</param></function>",
    "<function name=\"save_note\"><param name=\"title\">Plan</param><param name=\"body\"><![CDATA[Line one\nLine two with \"quotes\" and a backslash \\]]></param><param name=\"tags\">['a', 'b']</param><param name=\"meta\">{'nested': {'deep': [1, 2.5, True, None]}}</param></function>",
    "<function name=\"get_stock_price\"></function>",
    "<think>\n97 is not divisible by 2, 3, 5 or 7.\n</think>\n\nYes, 97 is prime.",
    "<function name=\"save_note\"><param name=\"body\"><![CDATA[Line one\nLine two",
    "<function name=\"get_weather\"><param name=\"city\">Par</function>",
    // A forced call an engine's reasoning gate started inside the thought, the output ending
    // there (the table's row out of the thought on the opener reads it); the call as its tag
    // writes it after the close, with the template's two newlines, is the first row.
    "<think>\nThe user asks about Paris: <function name=\"get_weather\"><param name=\"city\">Paris</param></function>",
];

/// The outputs bellwether's common set records for dots3, and one cut short.
const DOTS3_OUTPUTS: &[&str] = &[
    "<think>\nThe user asks about Paris.\n</think>\n\n\n<dots_function_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">\nParis\n</parameter>\n</invoke>\n</dots_function_call>",
    "Let me check.\n<dots_function_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">\nParis\n</parameter>\n<parameter name=\"unit\">\ncelsius\n</parameter>\n</invoke>\n</dots_function_call>",
    "\n<dots_function_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">\nParis\n</parameter>\n</invoke>\n</dots_function_call>\n<dots_function_call>\n<invoke name=\"get_stock_price\">\n<parameter name=\"ticker\">\nAAPL\n</parameter>\n</invoke>\n</dots_function_call>",
    "\n<dots_function_call>\n<invoke name=\"save_note\">\n<parameter name=\"title\">\nPlan\n</parameter>\n<parameter name=\"body\">\nLine one\nLine two with \"quotes\" and a backslash \\\n</parameter>\n<parameter name=\"tags\">\n[\"a\", \"b\"]\n</parameter>\n<parameter name=\"meta\">\n{\"nested\": {\"deep\": [1, 2.5, true, null]}}\n</parameter>\n</invoke>\n</dots_function_call>",
    "\n<dots_function_call>\n<invoke name=\"get_stock_price\">\n</invoke>\n</dots_function_call>",
    "<think>\n97 is not divisible by 2, 3, 5 or 7.\n</think>\n\nYes, 97 is prime.",
    "<dots_function_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">\nPar",
    "<dots_function_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">\nPar\n</dots_function_call>",
    // A forced call as its tag writes it after the thought's close (the template's two newlines,
    // then the block, without the newline the template writes before every block, which the first
    // row has); and the same call an engine's reasoning gate started inside the thought, the
    // output ending there (the table's row out of the thought on the opener reads it).
    "<think>\nThe user asks about Paris.\n</think>\n\n<dots_function_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">\nParis\n</parameter>\n</invoke>\n</dots_function_call>",
    "<think>\nThe user asks about Paris: <dots_function_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">\nParis\n</parameter>\n</invoke>\n</dots_function_call>",
];

/// The outputs bellwether's common set records for GLM 4.6, and one cut short.
const GLM4_OUTPUTS: &[&str] = &[
    "\n<think>The user asks about Paris.</think>\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>",
    "\nLet me check.\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>unit</arg_key>\n<arg_value>celsius</arg_value>\n</tool_call>",
    "\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>\n<tool_call>get_stock_price\n<arg_key>ticker</arg_key>\n<arg_value>AAPL</arg_value>\n</tool_call>",
    "\n<tool_call>save_note\n<arg_key>title</arg_key>\n<arg_value>Plan</arg_value>\n<arg_key>body</arg_key>\n<arg_value>Line one\nLine two with \"quotes\" and a backslash \\</arg_value>\n<arg_key>tags</arg_key>\n<arg_value>[\"a\", \"b\"]</arg_value>\n<arg_key>meta</arg_key>\n<arg_value>{\"nested\": {\"deep\": [1, 2.5, true, null]}}</arg_value>\n</tool_call>",
    "\n<tool_call>get_stock_price\n</tool_call>",
    "\nHello!",
    "\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Par",
    "\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Par</tool_call>",
    // A forced call as its tag writes it after the thought's close (the template's newline, then
    // the call), the format rehearsed inside the thought; and the same call an engine's reasoning
    // gate started inside the thought, the output ending there.
    "\n<think>The format is <tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>. Then call it.</think>\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>",
    "\n<think>The format is <tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>",
];

/// The outputs bellwether's common set records for K2-Horizon (the thought the prompt opens
/// written here), and one cut short.
const K2_HORIZON_OUTPUTS: &[&str] = &[
    "<ifm|think>The user asks about Paris.</ifm|think><ifm|tool_calls>\n<ifm|tool_call>get_weather\n<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>Paris</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>",
    "<ifm|think>97 is not divisible by 2, 3, 5 or 7.</ifm|think>Yes, 97 is prime.",
    "<ifm|think>plan</ifm|think><ifm|tool_calls>\n<ifm|tool_call>get_weather\n<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>Paris</ifm|arg_value>\n<ifm|arg_key>unit</ifm|arg_key>\n<ifm|arg_value>celsius</ifm|arg_value>\n</ifm|tool_call>\n<ifm|tool_call>get_stock_price\n<ifm|arg_key>ticker</ifm|arg_key>\n<ifm|arg_value>AAPL</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>",
    "<ifm|think>plan</ifm|think><ifm|tool_calls>\n<ifm|tool_call>save_note\n<ifm|arg_key>title</ifm|arg_key>\n<ifm|arg_value>Plan</ifm|arg_value>\n<ifm|arg_key>tags</ifm|arg_key>\n<ifm|arg_value>[\"a\", \"b\"]</ifm|arg_value>\n<ifm|arg_key>meta</ifm|arg_key>\n<ifm|arg_value>{\"nested\": {\"deep\": [1, 2.5, true, null]}}</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>",
    "<ifm|think>still thinking",
    "<ifm|think>plan</ifm|think><ifm|tool_calls>\n<ifm|tool_call>get_weather\n<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>Par",
    "<ifm|think>plan</ifm|think><ifm|tool_calls>\n<ifm|tool_call>get_weather\n<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>Par</ifm|tool_call>\n</ifm|tool_calls>",
];

/// The outputs bellwether's common set records for Gemma 4, one cut short, and the members the
/// template never writes: cut, without a key, without a value.
const GEMMA4_OUTPUTS: &[&str] = &[
    "<|channel>thought\nThe user asks about Paris.\n<channel|><|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
    "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,unit:<|\"|>celsius<|\"|>}<tool_call|>Let me check.",
    "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_call>call:get_stock_price{ticker:<|\"|>AAPL<|\"|>}<tool_call|>",
    "<|tool_call>call:save_note{body:<|\"|>Line one\nLine two with \"quotes\" and a backslash \\<|\"|>,meta:{nested:{deep:[1,2.5,true,null]}},tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],title:<|\"|>Plan<|\"|>}<tool_call|>",
    "<|tool_call>call:get_stock_price{}<tool_call|>",
    "<|channel>thought\n97 is not divisible by 2, 3, 5 or 7.\n<channel|>Yes, 97 is prime.",
    "Hello!",
    "<|tool_call>call:get_weather{city:<|\"|>Par",
    // A string the closing marker cuts, alone and nested in a list.
    "<|tool_call>call:get_weather{city:<|\"|>Par<tool_call|>",
    "<|tool_call>call:f{tags:[<|\"|>a<|\"|>,<|\"|>b<tool_call|>",
    // A member without a key: alone, after a member with a container, and cut by the marker.
    "<|tool_call>call:f{:1,a:2}<tool_call|>",
    "<|tool_call>call:f{a:2,:{x:[<|\"|>y<|\"|>]}}<tool_call|>",
    "<|tool_call>call:f{:<|\"|>te<tool_call|>",
    // A member cut between its colon and its value, by the closing marker and by the end.
    "<|tool_call>call:get_weather{city:<tool_call|>",
    "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:<tool_call|>",
    "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:",
    "<|tool_call>call:get_weather{tags:[<tool_call|>",
    // A member without a value, before a comma and before the close, a list item missing before
    // a comma, and a member without a key around one.
    "<|tool_call>call:get_weather{city:,days:3}<tool_call|>",
    "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days: }<tool_call|>",
    "<|tool_call>call:f{a:{x:},b:[1,,2]}<tool_call|>",
    "<|tool_call>call:f{:{:1},a:<|\"|><|\"|>}<tool_call|>",
    // Nesting past the prefix parser's bound, closed and cut.
    "<|tool_call>call:get_weather{days:[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]}<tool_call|>",
    "<|tool_call>call:get_weather{days:[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[",
];

/// The outputs bellwether's common set records for Inkling, and one cut short.
const INKLING_OUTPUTS: &[&str] = &[
    "<|content_thinking|>The user asks about Paris.<|end_message|><|message_model|><|content_text|><|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|>",
    "<|content_text|>Let me check.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"unit\":\"celsius\"}}<|end_message|>",
    "<|content_text|><|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|><|message_model|>get_stock_price<|content_invoke_tool_json|>{\"name\":\"get_stock_price\",\"args\":{\"ticker\":\"AAPL\"}}<|end_message|>",
    "<|content_text|><|end_message|><|message_model|>save_note<|content_invoke_tool_json|>{\"name\":\"save_note\",\"args\":{\"body\":\"Line one\\nLine two with \\\"quotes\\\" and a backslash \\\\\",\"meta\":{\"nested\":{\"deep\":[1,2.5,true,null]}},\"tags\":[\"a\",\"b\"],\"title\":\"Plan\"}}<|end_message|>",
    "<|content_text|>Hello!<|end_message|>",
    "<|content_thinking|>97 is not divisible by 2, 3, 5 or 7.<|end_message|><|message_model|><|content_text|>Yes, 97 is prime.<|end_message|>",
    "<|content_text|><|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"ci",
    "<|content_text|><|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\"}<|end_message|>",
];

/// Outputs recorded from requests that declared no tools: a thought and a short answer, a thought
/// and a longer one, a JSON answer whose grammar wrote the text marker again after the model's
/// own, the common set's call written without tools, and one cut inside the answer.
const INKLING_NO_TOOLS_OUTPUTS: &[&str] = &[
    "<|content_thinking|>We need solve math problem. Shop sells pens at 3 for 2 dollars. So price per pen = 2/3 dollars. 18 pens cost 18 * 2/3 = 12 dollars. Alternatively 18/3=6 groups, each 2 dollars => 12. Answer: 12.<|end_message|><|message_model|><|content_text|>12<|end_message|>",
    "<|content_thinking|>The user wants to rewrite \"The meeting was postponed by the manager\" in the active voice.\n\nOriginal: The meeting was postponed by the manager.\n- Subject: The meeting\n- Verb: was postponed (passive)\n- Agent: by the manager\n\nActive voice: The manager postponed the meeting.\n\nThis is straightforward. I should provide the rewritten sentence clearly.<|end_message|><|message_model|><|content_text|>The manager postponed the meeting.<|end_message|>",
    "<|content_thinking|>The user asks: \"Give the facts about Tokyo.\" The system instruction says: \"Answer with a JSON object with the keys city, country and population.\"\n\nSo I need to provide facts about Tokyo, but specifically format the answer as a JSON object with keys: city, country, population.\n\nI should include the relevant values:\n- city: Tokyo\n- country: Japan\n- population: Approximately 13.9 million (for Tokyo proper) or 37 million (for Greater Tokyo Area). Usually \"Tokyo\" as a city refers to the prefecture/capital, but the city proper is about 14 million. I should provide a clear figure. Let's use ~13.96 million (as of recent estimates) or just \"13.9 million\". To be safe and accurate, I can specify the metropolitan area, but the key is just population. Let's provide \"13.9 million\" for the city proper, or mention both? The instruction is strict: JSON object with those keys. So just those three.\n\nLet's construct:\n{\n  \"city\": \"Tokyo\",\n  \"country\": \"Japan\",\n  \"population\": \"13.9 million\"\n}\n\nThis satisfies. I will not add extra text outside JSON? The instruction says \"Answer with a JSON object...\". Usually that means the response should be just the JSON. I will provide just the JSON.<|end_message|><|message_model|><|content_text|><|content_text|>{\"city\": \"Tokyo\", \"country\": \"Japan\", \"population\": \"13.9 million\"}<|end_message|>",
    "<|content_thinking|>The user asks about Paris.<|end_message|><|message_model|><|content_text|><|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|>",
    "<|content_thinking|>Short.<|end_message|><|message_model|><|content_text|>The answer is cut he",
];

/// Every subject with each of its outputs.
fn corpus() -> impl Iterator<Item = (&'static Subject, &'static str)> {
    FORMATS
        .iter()
        .flat_map(|subject| subject.outputs.iter().map(move |text| (subject, *text)))
}

/// What the parser has said, as the streams a client would assemble.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Said {
    content: String,
    /// The text of each reasoning region, in order.
    reasoning: Vec<String>,
    /// Dropped text by reason, each reason's text joined in order.
    dropped: Vec<(DropReason, String)>,
    /// Malformed text by reason, each reason's text joined in order.
    malformed: Vec<(MalformedReason, String)>,
    calls: Vec<(u32, String, String)>,
    arguments: Vec<(u32, String)>,
    ended: Vec<u32>,
    /// The finish's reason, call count and reasoning token count.
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
                    said.calls
                        .push((*index, common::id_shape(id), name.clone()));
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

fn end() -> Input<'static> {
    Input::End {
        finish: EngineFinish::Stop,
    }
}

/// Replay `text` through a new parser of `subject`, cut at `cuts`, ending with `stop`.
fn replay(
    subject: &Subject,
    text: &str,
    cuts: &[usize],
    empty_between: bool,
) -> Result<Vec<Event<'static>>, ParseError> {
    common::replay(
        &mut *(subject.new)(),
        text,
        cuts,
        &EngineFinish::Stop,
        empty_between,
    )
}

/// Whether `text` is a prefix of JSON a strict reader accepts: whole, or cut where the reader asks
/// for more bytes rather than refusing one it saw. The crate's partial parser is no oracle here: its
/// prefix mode tolerates what JSON lacks (a trailing comma, a value left out, an escape JSON lacks),
/// which is what a client's reader refuses once the call ends.
fn strict_json_prefix(text: &str) -> bool {
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(_) => true,
        Err(error) => error.is_eof(),
    }
}

/// Whether `fragment`, appended to the arguments `so_far`, closes a JSON string `so_far` left
/// open: the quote that ends a value the parser was streaming.
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

/// What a client relies on in the shape of the stream, checked event by event.
fn check_well_formed(subject: &Subject, text: &str, cuts: &[usize], events: &[Event]) {
    let name = subject.name;
    // A pythonic call's end is its own `)`, which every template writes before the region's
    // marker; the other syntaxes' ends may carry no bytes (the region's marker closes them).
    let pythonic = subject
        .table
        .is_some_and(|table| matches!(table.format().call_syntax(), Some(CallSyntax::Pythonic(_))));
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
    for (at, event) in events.iter().enumerate() {
        let place = || format!("{name}: {text:?} cut at {cuts:?}, event {at}: {event:?}");
        assert!(!finished, "{}: after Finish", place());
        let stop_closed = std::mem::take(&mut after_the_stops_word);
        match event {
            Event::ToolCallStart { index, .. } => {
                assert!(!started.contains(index), "{}: a second start", place());
                started.push(*index);
                arguments.push((*index, String::new()));
            }
            Event::ToolCallArguments {
                index,
                json,
                source,
            } => {
                assert!(
                    started.contains(index),
                    "{}: before the call's start",
                    place()
                );
                assert!(!ended.contains(index), "{}: after the call's end", place());
                if let Some((_, so_far)) = arguments.iter_mut().find(|(i, _)| i == index) {
                    // A fragment that closes a string the arguments so far left open carries the
                    // bytes that closed it, the model's quote, closing tag or token: a closing
                    // quote with no byte of the output behind it (none, or whitespace alone) is
                    // one the reader made up for a value the model never closed. The one
                    // exception is the model's own end of output: the stop has no bytes, and the
                    // word right before the fragment says the stop closed the string.
                    assert!(
                        !closes_an_open_string(so_far, json)
                            || !source.text.trim().is_empty()
                            || stop_closed,
                        "{}: the fragment {json:?} closes an open string with no byte of the output behind it, a made-up closing quote",
                        place()
                    );
                    // An empty string a fragment writes has bytes of the output behind it, the
                    // string the model opened and closed at once or the JSON it wrote: one with
                    // none, or whitespace alone, is made up for a member that has no value.
                    assert!(
                        !(matches!(json.as_str(), "\"\"" | ", \"\"")
                            && source.text.trim().is_empty()),
                        "{}: the fragment {json:?} writes an empty string with no byte of the output behind it, a made-up value",
                        place()
                    );
                    so_far.push_str(json);
                    if broke.contains(index) {
                        assert_eq!(
                            json,
                            &source.text,
                            "{}: a fragment after the break is not the model's bytes as written",
                            place()
                        );
                    } else {
                        assert!(
                            strict_json_prefix(so_far),
                            "{}: the arguments so far, {so_far:?}, are no prefix of JSON a strict reader accepts",
                            place()
                        );
                    }
                }
            }
            Event::ToolCallEnd { index, source } => {
                assert!(
                    started.contains(index),
                    "{}: before the call's start",
                    place()
                );
                assert!(!ended.contains(index), "{}: a second end", place());
                ended.push(*index);
                // A pythonic call that ended without its `)` was cut by the region's marker or
                // the stream's end: it says so, and nothing closes its object for it.
                if pythonic && source.text.is_empty() {
                    let so_far = arguments
                        .iter()
                        .find(|(i, _)| i == index)
                        .map_or("", |(_, so_far)| so_far.as_str());
                    assert!(
                        malformed_inside.contains(index)
                            && serde_json::from_str::<serde_json::Value>(so_far).is_err(),
                        "{}: a pythonic call ended without its `)` with arguments {so_far:?}: a cut call says so and its object stays open",
                        place()
                    );
                }
                // A call that ended is whole, or the parser said it is not, whether its own
                // terminal, the region's or the end of the stream ended it: a client that reads
                // the arguments as JSON relies on one or the other.
                if !malformed_inside.contains(index) {
                    let so_far = arguments
                        .iter()
                        .find(|(i, _)| i == index)
                        .map_or("", |(_, so_far)| so_far.as_str());
                    assert!(
                        serde_json::from_str::<serde_json::Value>(so_far).is_ok(),
                        "{}: the call closed with arguments that are no whole JSON, {so_far:?}, and no Malformed event",
                        place()
                    );
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
                assert!(!reasoning_open, "{}: reasoning already open", place());
                reasoning_open = true;
            }
            Event::Reasoning(_) => {
                assert!(reasoning_open, "{}: reasoning outside a region", place());
            }
            Event::ReasoningEnd => {
                assert!(reasoning_open, "{}: no reasoning open", place());
                reasoning_open = false;
            }
            Event::Finish { tool_calls, .. } => {
                assert_eq!(
                    *tool_calls as usize,
                    started.len(),
                    "{}: the finish counts the calls that started",
                    place()
                );
                finished = true;
            }
            Event::Content(_) | Event::Dropped { .. } => {}
        }
    }
    let place = format!("{name}: {text:?} cut at {cuts:?}");
    assert!(finished, "{place}: no Finish");
    assert!(!reasoning_open, "{place}: reasoning left open");
    assert_eq!(started.len(), ended.len(), "{place}: a call without an end");
}

/// Every table of the registry is a subject here, so that a new table joins the contract with its
/// outputs before it reads a model's.
#[test]
fn every_registry_table_has_a_subject() {
    for table in Table::ALL {
        assert!(
            FORMATS.iter().any(|subject| subject.table == Some(table)),
            "{} has no subject in FORMATS",
            table.name()
        );
    }
}

#[test]
fn every_byte_of_every_output_lands_in_exactly_one_event_in_order() {
    for (subject, text) in corpus() {
        for (_, cuts) in chunkings(text, usize::MAX) {
            let events = replay(subject, text, &cuts, false)
                .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", subject.name));
            let conserved: String = events.iter().map(bytes_of).collect();
            assert_eq!(conserved, *text, "{}: cuts {cuts:?}", subject.name);
            assert!(
                matches!(events.last(), Some(Event::Finish { .. })),
                "{}: {text:?}: Finish is last",
                subject.name
            );
        }
    }
}

#[test]
fn what_the_parser_says_does_not_depend_on_the_chunking() {
    for (subject, text) in corpus() {
        let whole = replay(subject, text, &[], false)
            .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", subject.name));
        let expected = Said::of(&whole);
        for (_, cuts) in chunkings(text, usize::MAX) {
            let events = replay(subject, text, &cuts, false)
                .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", subject.name));
            assert_eq!(
                Said::of(&events),
                expected,
                "{}: {text:?} cut at {cuts:?}",
                subject.name
            );
        }
    }
}

#[test]
fn empty_deltas_between_the_pieces_change_nothing() {
    for (subject, text) in corpus() {
        let whole = replay(subject, text, &[], false)
            .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", subject.name));
        let per_char: Vec<usize> = text.char_indices().map(|(i, _)| i).skip(1).collect();
        for cuts in [Vec::new(), per_char] {
            let events = replay(subject, text, &cuts, true)
                .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", subject.name));
            let conserved: String = events.iter().map(bytes_of).collect();
            assert_eq!(conserved, *text, "{}: cuts {cuts:?}", subject.name);
            assert_eq!(
                Said::of(&events),
                Said::of(&whole),
                "{}: {text:?} cut at {cuts:?}",
                subject.name
            );
        }
    }
}

#[test]
fn the_event_stream_is_well_formed_under_every_chunking() {
    for (subject, text) in corpus() {
        for (_, cuts) in chunkings(text, usize::MAX) {
            let events = replay(subject, text, &cuts, false)
                .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", subject.name));
            check_well_formed(subject, text, &cuts, &events);
        }
    }
}

#[test]
fn calls_are_numbered_from_zero_in_order_with_ids_that_count_from_a_nonce_of_the_engines_own() {
    // The whole output is enough: chunking invariance above makes every other cut say the same.
    fn starts(events: &[Event]) -> Vec<(u32, String, String)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart {
                    index, id, source, ..
                } => Some((*index, id.clone(), source.text.to_string())),
                _ => None,
            })
            .collect()
    }
    for (subject, text) in corpus() {
        let events = replay(subject, text, &[], false)
            .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", subject.name));
        let calls = starts(&events);
        // A table whose call header writes the id carries the model's own: not minted, found in
        // the header's bytes, and the same from another engine over the same output.
        let from_header = subject
            .table
            .is_some_and(|table| table.format().call_id_source() == CallId::Header);
        if from_header {
            // A call's id is its header's, found in the header's bytes; a header whose id
            // repeats an earlier call's gets a minted id instead, so the ids stay distinct. The
            // engine reads the header's id as its text without the ending literal, trimmed; here
            // the literal is what followed the earlier call's id in its header, so a header
            // repeats when, past the whitespace the pattern allows, it is that id and that
            // literal again.
            for (position, (index, id, source)) in calls.iter().enumerate() {
                assert_eq!(*index as usize, position, "{}: {text:?}", subject.name);
                let repeated = calls[..position]
                    .iter()
                    .any(|(_, earlier, earlier_source)| {
                        if common::minted_id(earlier).is_some() {
                            return false;
                        }
                        let Some(at) = earlier_source.find(earlier.as_str()) else {
                            return false;
                        };
                        let literal = earlier_source[at + earlier.len()..].trim_start();
                        source
                            .trim_start()
                            .strip_prefix(earlier.as_str())
                            .is_some_and(|rest| rest.trim_start() == literal)
                    });
                if repeated {
                    assert!(
                        common::minted_id(id).is_some(),
                        "{}: {text:?}: a repeated header ({source:?}) gave {id:?}, not a minted id",
                        subject.name
                    );
                } else {
                    assert!(
                        !id.is_empty() && common::minted_id(id).is_none() && source.contains(id),
                        "{}: {text:?}: id {id:?} is not the header's ({source:?})",
                        subject.name
                    );
                }
            }
            let mut distinct: Vec<&String> = calls.iter().map(|(_, id, _)| id).collect();
            distinct.sort();
            distinct.dedup();
            assert_eq!(distinct.len(), calls.len(), "{}: {text:?}", subject.name);
            // Another engine gives the same ids, the minted fallbacks compared by their shape:
            // each engine draws its own nonce.
            let shaped = |calls: &[(u32, String, String)]| -> Vec<(u32, String, String)> {
                calls
                    .iter()
                    .map(|(index, id, source)| (*index, common::id_shape(id), source.clone()))
                    .collect()
            };
            let again = replay(subject, text, &[], false)
                .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", subject.name));
            assert_eq!(
                shaped(&calls),
                shaped(&starts(&again)),
                "{}: {text:?}",
                subject.name
            );
            continue;
        }
        let first = calls.first().and_then(|(_, id, _)| common::minted_id(id));
        for (position, (index, id, _)) in calls.iter().enumerate() {
            assert_eq!(*index as usize, position, "{}: {text:?}", subject.name);
            let number = common::minted_id(id).unwrap_or_else(|| {
                panic!(
                    "{}: {text:?}: id {id:?} is not call_ and 24 hex",
                    subject.name
                )
            });
            assert_eq!(
                Some(number),
                first.map(|first| first + position as u128),
                "{}: {text:?}: {id} does not count from the first id",
                subject.name
            );
        }
        // Another engine over the same output is another request: it repeats none of the ids.
        let again = replay(subject, text, &[], false)
            .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", subject.name));
        for ((_, id, _), (_, other, _)) in calls.iter().zip(starts(&again)) {
            assert_ne!(id, &other, "{}: {text:?}", subject.name);
        }
    }
}

/// One call with nothing in it, as each table's template writes it, by the subject's name.
const CALLS_WITHOUT_ARGUMENTS: &[(&str, &str)] = &[
    ("qwen3", "<tool_call>\n{\"name\": \"get_weather\"}\n</tool_call>\n"),
    ("qwen2.5", "<tool_call>\n{\"name\": \"get_weather\"}\n</tool_call>"),
    (
        "qwen3 tagged",
        "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
    ),
    (
        "qwen3 tagged, the value between the tags",
        "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
    ),
    (
        "seed-oss",
        "<seed:tool_call>\n<function=get_weather>\n</function>\n</seed:tool_call>",
    ),
    (
        "deepseek v4.1",
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    ),
    (
        "deepseek v4",
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    ),
    (
        "hy4",
        "<tool_calls:opensource><tool_call:opensource>get_weather</tool_call:opensource>\
         </tool_calls:opensource>",
    ),
    ("ling", "<tool_call>get_weather\n</tool_call>"),
    ("glm", "<tool_call>get_weather</tool_call>"),
    ("glm4", "<tool_call>get_weather\n</tool_call>"),
    ("iquest", "<iquest_tool_call>get_weather</iquest_tool_call>"),
    (
        "k2 horizon",
        "<ifm|tool_calls>\n<ifm|tool_call>get_weather\n</ifm|tool_call>\n</ifm|tool_calls>",
    ),
    (
        "minimax m2",
        "<minimax:tool_call>\n<invoke name=\"get_weather\">\n</invoke>\n</minimax:tool_call>",
    ),
    (
        "minimax m3",
        "<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">]<]minimax[>[</invoke>\n\
         ]<]minimax[>[</tool_call>",
    ),
    ("minicpm5", "<function name=\"get_weather\"></function>"),
    (
        "dots3",
        "<dots_function_call>\n<invoke name=\"get_weather\">\n</invoke>\n</dots_function_call>",
    ),
    (
        "kimi k3",
        "<|open|>think<|sep|><|close|>think<|sep|><|open|>tools<|sep|>\
         <|open|>call tool=\"get_weather\" index=\"1\"<|sep|><|close|>call<|sep|>\
         <|close|>tools<|sep|><|close|>message<|sep|>",
    ),
    ("gemma 4", "<|tool_call>call:get_weather{}<tool_call|>"),
    (
        "function gemma",
        "<start_function_call>call:get_weather{}<end_function_call>",
    ),
    ("olmo3", "<function_calls>get_weather()</function_calls>"),
    ("lfm2.5", "<|tool_call_start|>[get_weather()]<|tool_call_end|>"),
    ("mistral v13", "[TOOL_CALLS]get_weather[ARGS]{}"),
    ("forced", "{}"),
    ("xlam", "[{\"name\": \"get_weather\"}]"),
    ("llama nemotron", "<TOOLCALL>[{\"name\": \"get_weather\"}]</TOOLCALL>"),
    ("json", "{\"name\": \"get_weather\"}"),
    (
        "inkling",
        "<|content_text|><|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>\
         {\"name\":\"get_weather\"}<|end_message|>",
    ),
];

/// A call that takes no arguments reads the same from every family, and the same as one written
/// with an empty object: its arguments are `{}` under every chunking, whichever syntax the table
/// reads and whether the syntax writes the empty object or leaves the member out.
#[test]
fn a_call_with_nothing_in_it_has_an_empty_object_for_its_arguments_whatever_the_syntax() {
    for (name, text) in CALLS_WITHOUT_ARGUMENTS {
        let subject = FORMATS
            .iter()
            .find(|subject| subject.name == *name)
            .unwrap_or_else(|| panic!("{name}: no such subject"));
        for (plan, cuts) in chunkings(text, usize::MAX) {
            let events = replay(subject, text, &cuts, false)
                .unwrap_or_else(|e| panic!("{name}: {text:?} {plan}: {e}"));
            let said = Said::of(&events);
            assert_eq!(said.calls.len(), 1, "{name}: {text:?} {plan}: {said:?}");
            assert_eq!(
                said.arguments,
                [(0, "{}".to_string())],
                "{name}: {text:?} {plan}: {said:?}"
            );
            assert_eq!(said.ended, [0], "{name}: {text:?} {plan}: {said:?}");
        }
    }
}

/// A synthetic tokenization of `text`: token boundaries at character offsets, one to three
/// characters per token, deterministic in the text.
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

/// Replay `text` through a new parser of `subject` cut at `cuts`, after the prompt, each delta
/// carrying the spans of the synthetic tokens it holds, a token cut by a delta boundary continuing
/// into the next delta, and a byte-less token at the very end, as an end-of-turn token would be.
fn replay_counted(
    subject: &Subject,
    text: &str,
    cuts: &[usize],
) -> Result<Vec<Event<'static>>, ParseError> {
    let boundaries = token_boundaries(text);
    let mut tokens: Vec<(usize, usize)> = boundaries.windows(2).map(|w| (w[0], w[1])).collect();
    tokens.push((text.len(), text.len()));
    // Each delta's spans are built first: a delta's events borrow it for as long as the buffer
    // they are read into lives, here the whole output.
    let mut deltas: Vec<(usize, usize, Vec<TokenSpan>)> = Vec::new();
    let mut from = 0;
    for &cut in cuts.iter().chain(std::iter::once(&text.len())) {
        if cut > from {
            let spans: Vec<TokenSpan> = tokens
                .iter()
                .filter(|&&(start, end)| {
                    (end > from && start < cut)
                        || (start == end && start == cut && cut == text.len())
                })
                .map(|&(start, end)| TokenSpan {
                    token_id: 0,
                    start: start.max(from) - from,
                    end: end.min(cut) - from,
                    continued: start < from,
                })
                .collect();
            deltas.push((from, cut, spans));
            from = cut;
        }
    }
    let mut parser = (subject.new)();
    let mut out = Events::new();
    parser.feed(prompt(), &mut out)?;
    for (from, cut, spans) in &deltas {
        parser.feed(
            Input::Delta {
                token_ids: &[],
                text: &text[*from..*cut],
                spans,
            },
            &mut out,
        )?;
    }
    if text.is_empty() {
        // No delta carried text, so the byte-less token arrives in an empty one.
        parser.feed(
            Input::Delta {
                token_ids: &[],
                text: "",
                spans: &[TokenSpan {
                    token_id: 0,
                    start: 0,
                    end: 0,
                    continued: false,
                }],
            },
            &mut out,
        )?;
    }
    parser.feed(
        Input::End {
            finish: EngineFinish::Stop,
        },
        &mut out,
    )?;
    Ok(out.into_owned().drain())
}

fn tokens_of(event: &Event) -> Option<u32> {
    match event {
        Event::Content(t) | Event::Reasoning(t) => t.tokens,
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.tokens,
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => source.tokens,
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => None,
    }
}

#[test]
fn every_token_is_counted_in_the_event_that_carries_its_first_byte_whatever_the_cuts() {
    for (subject, text) in corpus() {
        let boundaries = token_boundaries(text);
        // Where each synthetic token starts, and the byte-less one at the end.
        let mut starts: Vec<usize> = boundaries[..boundaries.len() - 1].to_vec();
        starts.push(text.len());
        for (_, cuts) in chunkings(text, usize::MAX) {
            let events = replay_counted(subject, text, &cuts)
                .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", subject.name));
            let place = || format!("{}: {text:?} cut at {cuts:?}", subject.name);
            let mut at = 0;
            let mut total = 0;
            for event in &events {
                if matches!(
                    event,
                    Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. }
                ) {
                    continue;
                }
                let tokens =
                    tokens_of(event).unwrap_or_else(|| panic!("{}: uncounted {event:?}", place()));
                let length = bytes_of(event).len();
                let expected = match event {
                    // The byte-less token at the end has no byte to follow it.
                    Event::Dropped {
                        why: DropReason::ControlToken,
                        ..
                    } if length == 0 => starts.iter().filter(|&&start| start >= text.len()).count(),
                    _ => starts
                        .iter()
                        .filter(|&&start| start >= at && start < at + length)
                        .count(),
                };
                assert_eq!(
                    tokens as usize,
                    expected,
                    "{}: {event:?} at byte {at}",
                    place()
                );
                at += length;
                total += tokens as usize;
            }
            assert_eq!(total, starts.len(), "{}: every token counted once", place());
            let reasoning: u32 = events
                .iter()
                .filter_map(|e| match e {
                    Event::Reasoning(t) => t.tokens,
                    _ => None,
                })
                .sum();
            let Some(Event::Finish {
                reasoning_tokens, ..
            }) = events.last()
            else {
                panic!("{}: Finish is last", place());
            };
            assert_eq!(
                *reasoning_tokens,
                reasoning,
                "{}: Finish counts the reasoning tokens",
                place()
            );
        }
    }
}

/// Feed `input`, which is out of order, and check that it is a `Lifecycle` error and emits nothing.
fn rejected<'d>(parser: &mut dyn Parser, input: Input<'d>, out: &mut Events<'d>, what: &str) {
    let before = out.len();
    assert!(
        matches!(parser.feed(input, out), Err(ParseError::Lifecycle(_))),
        "{what}: a lifecycle error"
    );
    assert_eq!(out.len(), before, "{what}: nothing emitted");
}

#[test]
fn the_lifecycle_is_one_prompt_then_deltas_then_one_end() {
    for subject in FORMATS {
        let mut parser = (subject.new)();
        let mut out = Events::new();
        parser.feed(prompt(), &mut out).expect("a prompt first");
        rejected(&mut *parser, prompt(), &mut out, "a second prompt");
        parser.feed(delta("a"), &mut out).expect("a delta");
        rejected(
            &mut *parser,
            prompt(),
            &mut out,
            "a prompt after output began",
        );
        parser.feed(end(), &mut out).expect("the end");
        rejected(&mut *parser, delta("b"), &mut out, "a delta after the end");
        rejected(&mut *parser, end(), &mut out, "a second end");

        let mut without_prompt = (subject.new)();
        let mut out = Events::new();
        without_prompt
            .feed(delta("a"), &mut out)
            .expect("a delta first is fine: the prompt is optional");
    }
}
