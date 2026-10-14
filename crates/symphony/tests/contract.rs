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
//!   calls that started, and after every fragment each call's arguments so far a valid JSON prefix,
//!   which is what prefix stability promises a client;
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

mod common;

use common::{bytes_of, chunkings, delta, prompt};
use openai_protocol::common::{Function, Tool};
use serde_json::json as value;
use symphony::{
    formats,
    json::PartialJson,
    shape::{
        self, Arguments, Atom, CallShape, Class, Dialect, End, NameSource, Path, Pattern, Slot,
        ValueParser,
    },
    tagged::{Placement, Spelling},
    CallSyntax, Declared, DropReason, Emits, Engine, EngineFinish, Event, Events, FinishReason,
    Format, Input, MalformedReason, ParseError, Parser, TokenSpan,
};

/// A subject under test: how to make its parser, and the outputs it is checked over.
struct Subject {
    name: &'static str,
    new: fn() -> Box<dyn Parser>,
    outputs: &'static [&'static str],
}

fn qwen3() -> Box<dyn Parser> {
    Box::new(Engine::new(
        formats::qwen3(CallSyntax::Json),
        Declared::default(),
    ))
}

fn qwen3_tagged() -> Box<dyn Parser> {
    Box::new(Engine::new(
        formats::qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine)),
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
    Box::new(Engine::new(formats::qwen2_5(), Declared::default()))
}

fn seed_oss() -> Box<dyn Parser> {
    Box::new(Engine::new(
        formats::seed_oss(),
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
    Box::new(Engine::new(formats::deepseek_v4_1(), Declared::default()))
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
    Box::new(Engine::new(formats::hy4(), keyed_tools()))
}

fn ling() -> Box<dyn Parser> {
    Box::new(Engine::new(formats::ling(), keyed_tools()))
}

fn glm() -> Box<dyn Parser> {
    Box::new(Engine::new(formats::glm(), keyed_tools()))
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
/// answers repeat, joined with `""` (no row) and not stripped (no trim row).
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
    Box::new(Engine::new(formats::iquest(), keyed_tools()))
}

fn olmo3() -> Box<dyn Parser> {
    Box::new(Engine::new(formats::olmo3(), Declared::default()))
}

fn lfm2_5() -> Box<dyn Parser> {
    Box::new(Engine::new(formats::lfm2_5(), Declared::default()))
}

fn xlam() -> Box<dyn Parser> {
    Box::new(Engine::new(formats::xlam(), Declared::default()))
}

fn minimax_m3() -> Box<dyn Parser> {
    Box::new(Engine::new(
        formats::minimax_m3(),
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
    Box::new(Engine::new(formats::kimi_k3(), Declared::default()))
}

fn mistral_v13() -> Box<dyn Parser> {
    Box::new(Engine::new(formats::mistral_v13(), Declared::default()))
}

fn llama_nemotron() -> Box<dyn Parser> {
    Box::new(Engine::new(formats::llama_nemotron(), Declared::default()))
}

fn function_gemma() -> Box<dyn Parser> {
    Box::new(Engine::new(formats::function_gemma(), Declared::default()))
}

fn forced() -> Box<dyn Parser> {
    Box::new(Engine::new(
        formats::forced(),
        Declared::default().forcing("get_weather"),
    ))
}

fn plain() -> Box<dyn Parser> {
    Box::new(Engine::new(formats::plain(), Declared::default()))
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
    },
    Subject {
        name: "qwen3 tagged",
        new: qwen3_tagged,
        outputs: QWEN3_TAGGED_OUTPUTS,
    },
    Subject {
        name: "qwen2.5",
        new: qwen2_5,
        outputs: QWEN3_OUTPUTS,
    },
    Subject {
        name: "deepseek v4.1",
        new: deepseek_v4_1,
        outputs: DSML_OUTPUTS,
    },
    Subject {
        name: "seed-oss",
        new: seed_oss,
        outputs: SEED_OUTPUTS,
    },
    Subject {
        name: "hy4",
        new: hy4,
        outputs: HY4_OUTPUTS,
    },
    Subject {
        name: "ling",
        new: ling,
        outputs: LING_OUTPUTS,
    },
    Subject {
        name: "glm",
        new: glm,
        outputs: GLM_OUTPUTS,
    },
    Subject {
        name: "glm, closed at stop",
        new: glm_closed_at_stop,
        outputs: GLM_STOPPED_OUTPUTS,
    },
    Subject {
        name: "muse messages, framed",
        new: muse_messages,
        outputs: MUSE_OUTPUTS,
    },
    Subject {
        name: "ernie, started in the thought",
        new: ernie,
        outputs: ERNIE_OUTPUTS,
    },
    Subject {
        name: "gemma 4 template, trimmed",
        new: gemma4_template,
        outputs: GEMMA4_TEMPLATE_OUTPUTS,
    },
    Subject {
        name: "inkling template, framed",
        new: inkling_template,
        outputs: INKLING_TEMPLATE_OUTPUTS,
    },
    Subject {
        name: "iquest",
        new: iquest,
        outputs: IQUEST_OUTPUTS,
    },
    Subject {
        name: "olmo3",
        new: olmo3,
        outputs: OLMO3_OUTPUTS,
    },
    Subject {
        name: "lfm2.5",
        new: lfm2_5,
        outputs: LFM_OUTPUTS,
    },
    Subject {
        name: "xlam",
        new: xlam,
        outputs: XLAM_OUTPUTS,
    },
    Subject {
        name: "minimax m3",
        new: minimax_m3,
        outputs: M3_OUTPUTS,
    },
    Subject {
        name: "kimi k3",
        new: kimi_k3,
        outputs: KIMI_K3_OUTPUTS,
    },
    Subject {
        name: "plain",
        new: plain,
        outputs: PLAIN_OUTPUTS,
    },
    Subject {
        name: "mistral v13",
        new: mistral_v13,
        outputs: MISTRAL_OUTPUTS,
    },
    Subject {
        name: "llama nemotron",
        new: llama_nemotron,
        outputs: LLAMA_NEMOTRON_OUTPUTS,
    },
    Subject {
        name: "function gemma",
        new: function_gemma,
        outputs: FUNCTION_GEMMA_OUTPUTS,
    },
    Subject {
        name: "qwen3, its calls as a shape",
        new: qwen3_shaped,
        outputs: QWEN3_OUTPUTS,
    },
    Subject {
        name: "gpt-oss test template, its calls as a shape",
        new: gpt_oss_template,
        outputs: GPT_OSS_OUTPUTS,
    },
    Subject {
        name: "shaped calls on abstract tags",
        new: shaped_tags,
        outputs: SHAPED_TAGS_OUTPUTS,
    },
    Subject {
        name: "qwen 3.5 serve template, its calls as a shape",
        new: qwen3_5_serve_shaped,
        outputs: QWEN3_5_SERVE_OUTPUTS,
    },
    Subject {
        name: "qwen 3.5 serve template, its calls as a shape, closed at stop",
        new: qwen3_5_serve_closed_at_stop,
        outputs: QWEN3_5_SERVE_OUTPUTS,
    },
    Subject {
        name: "muse glimmer template, its calls as a shape",
        new: muse_shaped,
        outputs: MUSE_CALLS_OUTPUTS,
    },
    Subject {
        name: "parameters on abstract tags",
        new: params_tags,
        outputs: PARAMS_TAGS_OUTPUTS,
    },
    Subject {
        name: "forced",
        new: forced,
        outputs: FORCED_OUTPUTS,
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
];

/// Calls on abstract tags: a call its trailer closes, terminals whose header does not match (in
/// the content, in a thought, twice in a row), a close the trailer does not end at, a call cut
/// short, a call the stop closes after its trailer, a header the end leaves unfinished, and
/// nothing.
const SHAPED_TAGS_OUTPUTS: &[&str] = &[
    "Sure.<c><n=f>{\"a\": 1}</f>\n</c>Done.",
    "Hi <c><x>bye",
    "<t>plan <c>oops</t>ok",
    "<c><c><n=f>{}</f></c>",
    "<c><n=f>{}</c>x</f></c>",
    "<c><n=f>{\"a\": [1, 2",
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
    "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, \
     {\"name\": \"get_stock_price\", \"arguments\": {}}]",
    "[{\"name\": \"get_weather\", \"argu",
    "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}",
    "  [ ]",
    "[{\"name\": \"f\", \"arguments\": {}} junk {\"name\": \"g\", \"arguments\": {}}] after",
    "Hello, [not a list].",
    "",
];

/// Python calls as Olmo 3 writes them: bare, one per line.
const OLMO3_OUTPUTS: &[&str] = &[
    "<function_calls>get_weather(city=\"Paris\", days=3, prefs={\"size\": \"large\", \"deep\": \
     [1, 2.5, true, null]})\nget_stock_price()</function_calls>",
    "<function_calls>get_weather(city=\"Par",
    "<function_calls>get_weather(city=\"Paris\"",
    "<function_calls>I will call:\nget_weather(city=\"a)b\", n=-1.5e3)</function_calls>\nDone.",
    "<function_calls></function_calls>",
    "Hello!",
];

/// Python calls as LFM2.5 writes them: a list, single-quoted strings.
const LFM_OUTPUTS: &[&str] = &[
    "<|tool_call_start|>[get_weather(city='Paris', note='it\\'s fine'), get_stock_price()]\
     <|tool_call_end|>",
    "<|tool_call_start|>[get_weather(city='Par",
    "<think>The user asks.</think><|tool_call_start|>[get_weather(city='計画 🌍 \"q\"')]\
     <|tool_call_end|>",
    "<|tool_call_start|>[]<|tool_call_end|>Nothing to call.",
];

/// Mistral's calls: the tool's name, `[ARGS]`, the arguments object alone, one call per opener,
/// the thought between `[THINK]` and `[/THINK]`.
const MISTRAL_OUTPUTS: &[&str] = &[
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
];

/// Llama Nemotron's calls: one JSON list between `<TOOLCALL>` and `</TOOLCALL>`, content before
/// it, the thought between `<think>` and `</think>`; the list cut, empty, or not a list.
const LLAMA_NEMOTRON_OUTPUTS: &[&str] = &[
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
];

/// FunctionGemma's calls: Gemma's spelling between `<start_function_call>` and
/// `<end_function_call>`, `<escape>` around a string, `None` for a null; cut streams, a block that
/// is no call, and text after the arguments.
const FUNCTION_GEMMA_OUTPUTS: &[&str] = &[
    "<start_function_call>call:get_weather{city:<escape>Paris<escape>,days:[1,2.5,true,None]}\
     <end_function_call>",
    "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>\
     <start_function_call>call:get_stock_price{}<end_function_call>",
    "Let me check.<start_function_call>call:get_weather{city:<escape>Paris<escape>}\
     <end_function_call>",
    "<start_function_call>call:get_weather{city:<escape>Par",
    "<start_function_call>call:get_wea",
    "<start_function_call>not a call<end_function_call>",
    "<start_function_call>call:get_weather{} trailing<end_function_call>",
    "<start_function_call>call:f{a:<esc",
    "Hello!",
];

/// The forced call's arguments alone, as the engine writes them under the gateway's schema for a
/// named function: whole, wrapped in whitespace, cut short, broken, followed by text, blank.
const FORCED_OUTPUTS: &[&str] = &[
    "{\"city\": \"Paris\", \"days\": [1, 2.5, true, null]}",
    "  {\"city\": \"Paris\"}\n",
    "{\"city\": \"Par",
    "{\"city\": @}",
    "{\"city\": \"Paris\"} trailing",
    "{}",
    "   ",
    "[1, 2]",
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
    "<seed:tool_call>\n<function=f<parameter=city>\nx\n</parameter>\n</function>\n\
     </seed:tool_call>",
    "<seed:tool_call>\nprose <parameter=city>\nParis\n</parameter>\n</function>\n</seed:tool_call>",
    "<think>not a thought</think><tool_call>not a call</tool_call><seed:think>a thought",
    "<seed:tool_call>\n<function=get_weather>\n<parameter=city>\n計画 🌍 \"q\" \\ \n</parameter>\n\
     </function>\n</seed:tool_call>",
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
/// of whitespace only, a cut call, and nothing.
const GEMMA4_TEMPLATE_OUTPUTS: &[&str] = &[
    "<|channel>thought\nhi<channel|><|tool_call>call:foo{a:1,b:<|\"|>bar<|\"|>}<tool_call|>",
    "<|channel>thought\n The user wants the weather.\n<channel|>\n<|tool_call>call:get_weather{\
     city:<|\"|>Paris<|\"|>}<tool_call|>\n\nIt is\n\nsunny. \n",
    "Hi <|tool_response> there\n",
    "\u{a0}\u{1c}Hi\u{2003}there\u{2003}\u{1f}",
    "<|channel>thought\nPlan <|tool_call>call:f{}<tool_call|> more<channel|>Done",
    " \n\t ",
    "Sure.\n<|tool_call>call:get_weather{city:<|\"|>Par",
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
    "<tool_call>\n<function=f<parameter=city>\nx\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\nprose <parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
    "<think>plan</think><tool_call>\n<function=get_weather>\n<parameter=city>\n計画 🌍 \"q\" \\ \n\
     </parameter>\n</function>\n</tool_call>",
];

const QWEN3_OUTPUTS: &[&str] = &[
    "",
    "Hello!",
    "<think>\n\n</think>\n\nHello!",
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
) -> Result<Vec<Event>, ParseError> {
    common::replay(
        &mut *(subject.new)(),
        text,
        cuts,
        &EngineFinish::Stop,
        empty_between,
    )
}

/// Whether `text` is a JSON prefix the prefix parser takes whole.
fn valid_json_prefix(text: &str) -> bool {
    matches!(PartialJson::default().parse(text, true), Ok((_, consumed)) if consumed == text.len())
}

/// What a client relies on in the shape of the stream, checked event by event.
fn check_well_formed(name: &str, text: &str, cuts: &[usize], events: &[Event]) {
    let mut started: Vec<u32> = Vec::new();
    let mut ended: Vec<u32> = Vec::new();
    let mut arguments: Vec<(u32, String)> = Vec::new();
    let mut reasoning_open = false;
    let mut finished = false;
    for (at, event) in events.iter().enumerate() {
        let place = || format!("{name}: {text:?} cut at {cuts:?}, event {at}: {event:?}");
        assert!(!finished, "{}: after Finish", place());
        match event {
            Event::ToolCallStart { index, .. } => {
                assert!(!started.contains(index), "{}: a second start", place());
                started.push(*index);
                arguments.push((*index, String::new()));
            }
            Event::ToolCallArguments { index, json, .. } => {
                assert!(
                    started.contains(index),
                    "{}: before the call's start",
                    place()
                );
                assert!(!ended.contains(index), "{}: after the call's end", place());
                if let Some((_, so_far)) = arguments.iter_mut().find(|(i, _)| i == index) {
                    so_far.push_str(json);
                    assert!(
                        valid_json_prefix(so_far),
                        "{}: the arguments so far, {so_far:?}, are no valid JSON prefix",
                        place()
                    );
                }
            }
            Event::ToolCallEnd { index, .. } => {
                assert!(
                    started.contains(index),
                    "{}: before the call's start",
                    place()
                );
                assert!(!ended.contains(index), "{}: a second end", place());
                ended.push(*index);
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
            Event::Content(_) | Event::Dropped { .. } | Event::Malformed { .. } => {}
        }
    }
    let place = format!("{name}: {text:?} cut at {cuts:?}");
    assert!(finished, "{place}: no Finish");
    assert!(!reasoning_open, "{place}: reasoning left open");
    assert_eq!(started.len(), ended.len(), "{place}: a call without an end");
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
            check_well_formed(subject.name, text, &cuts, &events);
        }
    }
}

#[test]
fn calls_are_numbered_from_zero_in_order_with_ids_that_follow_the_index() {
    // The whole output is enough: chunking invariance above makes every other cut say the same.
    for (subject, text) in corpus() {
        let events = replay(subject, text, &[], false)
            .unwrap_or_else(|e| panic!("{}: {text:?}: {e}", subject.name));
        for (position, (index, id, _)) in Said::of(&events).calls.iter().enumerate() {
            assert_eq!(*index as usize, position, "{}: {text:?}", subject.name);
            assert_eq!(id, &format!("call_{index}"), "{}: {text:?}", subject.name);
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
fn replay_counted(subject: &Subject, text: &str, cuts: &[usize]) -> Result<Vec<Event>, ParseError> {
    let boundaries = token_boundaries(text);
    let mut tokens: Vec<(usize, usize)> = boundaries.windows(2).map(|w| (w[0], w[1])).collect();
    tokens.push((text.len(), text.len()));
    let mut parser = (subject.new)();
    let mut out = Events::new();
    parser.feed(prompt(), &mut out)?;
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
            parser.feed(
                Input::Delta {
                    token_ids: &[],
                    text: &text[from..cut],
                    spans: &spans,
                },
                &mut out,
            )?;
            from = cut;
        }
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
    Ok(out.drain())
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
fn rejected(parser: &mut dyn Parser, input: Input<'_>, out: &mut Events, what: &str) {
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
