//! Tables spelled from public response templates, read as Hugging Face's parser reads the same
//! outputs. A response template is the `response_template` a checkpoint's `tokenizer_config.json`
//! declares, or one of transformers' own; each table here spells one with the rows a template
//! needs: a framing state for the text outside every field, turn anchors, a start row, trimmed
//! stays, and the stop tokens that are terminals. The calls are spelled where a call
//! syntax the crate has reads the template's (Gemma's, the JSON object, a call given as data whose
//! header names the tool), and left out where none does yet.
//!
//! Each case is a prompt, an output and the finish, with the message Hugging Face's parser gives
//! for them: `parse_response(output, template, prefix=prompt)` from transformers 5.18.0, the
//! output without the stop token's text, which the stop decoder hides (where the stop text would
//! change the message, the case says so). Every case replays whole, at every two-way split,
//! character by character and on thirty seeded plans, and the message the chat adapter folds from
//! the events is the recorded one at every cut: the reasoning, the content, and each call's name
//! and arguments as JSON values. The cases where the table reads an output on purpose otherwise
//! are listed with what it gives and why.

#[expect(
    dead_code,
    reason = "the integration tests share these helpers, and this one replays after a prompt only"
)]
mod common;

use std::{fs, path::PathBuf};

use common::{bytes_of, chunkings, replay_after};
use openai_protocol::common::{Function, Tool};
use serde_json::{json, Value};
use symphony::{
    adapt,
    grammar::Options,
    shape::{
        self, Arguments, Atom, CallLayout, CallShape, Class, Dialect, End, Layout, Message,
        NameSource, Nested, Pattern, Segment, Slot, Spellings, ValueParser, Words, Written,
    },
    CallSyntax, Declared, Emits, Engine, EngineFinish, Format, Grammar, Parser, Tag,
};

/// One output and the message it reads as.
struct Case {
    prompt: &'static str,
    output: &'static str,
    finish: EngineFinish,
    reasoning: Option<&'static str>,
    content: Option<&'static str>,
    /// Each call's name and its arguments as JSON.
    calls: &'static [(&'static str, &'static str)],
}

/// Python `regex`'s `\w`, as `tests/fixtures/shape/classes.json` records it: the class a
/// template's `\w` holds, which whoever lowers a template resolves into ranges.
#[expect(
    clippy::panic,
    clippy::expect_used,
    reason = "a fixture that cannot be read ends the test with its path"
)]
fn python_word() -> Class {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/shape/classes.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let tables: Value = serde_json::from_str(&text).expect("classes.json");
    let ranges = tables["word"]
        .as_array()
        .expect("the word table")
        .iter()
        .map(|range| {
            let mut chars = range.as_str().expect("a range").chars();
            match (chars.next(), chars.next(), chars.next()) {
                (Some(first), Some(last), None) => (first, last),
                _ => panic!("a range is two characters: {range}"),
            }
        });
    Class::new(ranges, false)
}

/// transformers v5.18.0's `tests/utils/test_chat_parsing.py` (`gpt_oss_template`): Harmony's
/// messages, the analysis channel a thought and the final channel the answer, each closed by
/// `<|end|>`, nothing kept outside them, the turn after `<|start|>assistant`. A call's open,
/// `<\|channel\|>commentary to=functions\.(?P<name>\w+).*?<\|message\|>`, is the terminal
/// `<|channel|>commentary to=functions.` and a header that names the tool; its arguments are one
/// JSON value; its close, `<|call|>`, is a stop token, so the stream's stop closes it. The stop
/// tokens are `<|return|>`, `<|endoftext|>` and `<|call|>` (the `generation_config.json` of
/// `openai/gpt-oss-20b`).
fn gpt_oss() -> Format {
    let word = python_word();
    let shape = shape::intern(CallShape {
        header: Pattern(
            [
                Atom::Run {
                    class: word.clone(),
                    min: 1,
                    max: None,
                    greedy: true,
                    capture: Some(Slot::Name),
                },
                Atom::Run {
                    class: Class::any(),
                    min: 0,
                    max: None,
                    greedy: false,
                    capture: None,
                },
                Atom::Lit("<|message|>".into()),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Json(Dialect::default()),
        trailer: Pattern::default(),
        word,
        hold_cap: 8192,
    });
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
        .stop_tokens(["<|return|>", "<|endoftext|>", "<|call|>"])
}

/// The response template of `google/gemma-4-31B-it` (its `tokenizer_config.json`), as
/// transformers v5.18.0's tests write it (`gemma4_template`): a turn after `<|turn>model\n` or
/// after `<tool_response|>`, a thought, the calls in Gemma's spelling, and the content outside
/// them, stripped. The content closes at `<turn|>`, `<|tool_response>` or `<eos>`, and all three
/// are the stop tokens (its `generation_config.json`): the first and the last have no row, and
/// `<|tool_response>`, which a stream that does not stop on it carries, is a row back into the
/// content that drops the marker.
fn gemma4() -> Format {
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
        .trims("reasoning")
        .stop_tokens(["<eos>", "<turn|>", "<|tool_response>"])
}

/// transformers v5.18.0's `tests/utils/test_chat_parsing.py` (`ernie_template`): the turn after
/// `Assistant:`, the thought opened by `(?:^|<think>\s*)`, so a turn that starts with text starts
/// in it, the answer between `<response>\n` and `\n?</response>`, a call between `<tool_call>` and
/// `</tool_call>` as one JSON object, and nothing kept outside them. `</s>` is the stop token.
fn ernie() -> Format {
    Format::new("ernie test template")
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
        .trims("content")
        .stop_tokens(["</s>"])
}

/// transformers v5.18.0's `tests/utils/test_chat_parsing.py` (`inkling_template`): the turn after
/// `<|message_model|>`, each message a thought, an answer or a call after its own marker (an
/// author's name and other text before the marker are outside), all of them closed by
/// `<|end_message|>`, and nothing kept outside them. The thoughts and the answers repeat, joined
/// with `""` and not stripped: no trim row. `<|end_message|>` ends a message
/// inside the turn; the stop token, `<|content_model_end_sampling|>` (the `config.json` of
/// `thinkingmachines/Inkling`), ends the turn and is not a terminal here.
fn inkling() -> Format {
    Format::new("inkling test template")
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
        .turn_anchor("<|message_model|>")
        .stop_tokens(["<|content_model_end_sampling|>"])
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

/// Python `regex`'s `\s`, as `tests/fixtures/shape/classes.json` records it.
#[expect(
    clippy::panic,
    clippy::expect_used,
    reason = "a fixture that cannot be read ends the test with its path"
)]
fn python_space() -> Class {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/shape/classes.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let tables: Value = serde_json::from_str(&text).expect("classes.json");
    let ranges = tables["space"]
        .as_array()
        .expect("the space table")
        .iter()
        .map(|range| {
            let mut chars = range.as_str().expect("a range").chars();
            match (chars.next(), chars.next(), chars.next()) {
                (Some(first), Some(last), None) => (first, last),
                _ => panic!("a range is two characters: {range}"),
            }
        });
    Class::new(ranges, false)
}

/// The response template of `meta-models/Muse-Glimmer-30B` (its `tokenizer_config.json`): the turn
/// after `<|start|>assistant`, a thought addressed to the model itself, an answer addressed to the
/// user, both stripped, and nothing kept outside them. A call's open,
/// `<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">`, is the terminal `<atem:invoke` and a header
/// that skips other attributes and names the tool; its close is `</atem:invoke>`; its parameters
/// are tags read as `finditer` reads `<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>`,
/// then the value up to `</atem:parameter>`. The template reads a value as JSON where it is JSON
/// and as text otherwise, which the values here read the same as the crate's own reading does.
/// `<|end_of_text|>` and `<|eot|>` are the stop tokens (its `generation_config.json`); `<|eom|>`
/// ends a message inside the turn, and the model goes on after it.
fn muse() -> Format {
    let word = python_word();
    let any = Class::any();
    let shape = shape::intern(CallShape {
        header: Pattern(
            [
                Atom::Boundary,
                run(none_of(">"), 0, false, None),
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
                    run(none_of(">"), 0, false, None),
                    Atom::Boundary,
                    Atom::Lit("name=\"".into()),
                    run(none_of("\""), 1, true, Some(Slot::Key)),
                    Atom::Lit("\"".into()),
                    run(none_of(">"), 0, false, None),
                    Atom::Lit(">".into()),
                    run(any, 0, false, Some(Slot::Value)),
                ]
                .into(),
            ),
            ends: [End {
                consume: Pattern([Atom::Lit("</atem:parameter>".into())].into()),
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
        word,
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
}

/// transformers v5.18.0's `transformers serve` template for Qwen 3.5 (`qwen3_5` in
/// `src/transformers/cli/serving/utils.py`): the turn after `<|im_start|>assistant\n`, a thought
/// between `<think>` and `</think>`, and the content outside them, stripped, which the stop ends.
/// A call's open, `\s*<tool_call>\s*<function=(?P<name>[^>\n]+)>`, is the terminal `<tool_call>`
/// after the content's whitespace and a header that names the tool; its close,
/// `</function>\s*</tool_call>`, is the trailer `</function>\s*` and the terminal `</tool_call>`;
/// its parameters are tags read as `finditer` reads `<parameter=(?P<key>[^>\n]+)>\s*`, then the
/// value up to `\s*</parameter>`, each value the text, typed by the request's tools.
/// `<|im_end|>` and `<|endoftext|>` are the stop tokens.
fn serve_qwen3_5() -> Format {
    let space = python_space();
    let name = none_of(">\n");
    let shape = shape::intern(CallShape {
        header: Pattern(
            [
                run(space.clone(), 0, true, None),
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
                    run(space.clone(), 0, true, None),
                    run(Class::any(), 0, false, Some(Slot::Value)),
                ]
                .into(),
            ),
            ends: [End {
                consume: Pattern(
                    [
                        run(space.clone(), 0, true, None),
                        Atom::Lit("</parameter>".into()),
                    ]
                    .into(),
                ),
                ahead: Pattern::default(),
            }]
            .into(),
            values: ValueParser::Raw,
        },
        trailer: Pattern([Atom::Lit("</function>".into()), run(space, 0, true, None)].into()),
        word: python_word(),
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
}

/// The tools the Qwen 3.5 cases' requests declare: one weather tool, its city a string and its
/// days an integer.
fn weather() -> Declared {
    Declared::of(&[Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            }}),
            strict: None,
            extra: Default::default(),
        },
    }])
}

/// The gpt-oss test template's cases. With the stop token's text fed, as transformers feeds it,
/// the answers of the first two would keep `<|return|>` (and the second its edge before it): the
/// stop decoder hides it here. The calls end at `<|call|>`, which the stream's stop stands for:
/// after a thought, right after the marker, with words before `<|message|>` that the header
/// skips, after an answer with spaces around the arguments, and a marker that names no tool
/// (its header needs a word right after it), which opens no call.
const GPT_OSS: &[Case] = &[
    Case {
        prompt: "<|start|>user<|message|>Weather in Paris?<|end|><|start|>assistant",
        output: "<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant\
                 <|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>\
                 {\"city\": \"Paris\"}",
        finish: EngineFinish::Stop,
        reasoning: Some("Need the weather."),
        content: None,
        calls: &[("get_weather", "{\"city\": \"Paris\"}")],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>commentary to=functions.get_weather<|message|>{\"city\": \"Paris\", \
                 \"days\": 3}",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[("get_weather", "{\"city\": \"Paris\", \"days\": 3}")],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>commentary to=functions.get_weather json<|message|>{}",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[("get_weather", "{}")],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>final<|message|>Checking.<|end|><|start|>assistant<|channel|>\
                 commentary to=functions.lookup <|constrain|>json<|message|> {\"q\": [1, 2.5, \
                 null]} ",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Checking."),
        calls: &[("lookup", "{\"q\": [1, 2.5, null]}")],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>commentary to=functions.<|channel|>final<|message|>Hi",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
    Case {
        prompt: "<|start|>user<|message|>Weather?<|end|><|start|>assistant",
        output: "<|channel|>analysis<|message|>User asks for the weather.<|end|><|start|>\
                 assistant<|channel|>final<|message|>It is sunny.",
        finish: EngineFinish::Stop,
        reasoning: Some("User asks for the weather."),
        content: Some("It is sunny."),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>analysis<|message|>\n  Plan.  \n<|end|><|start|>assistant<|channel|>\
                 final<|message|> Hi! \n",
        finish: EngineFinish::Stop,
        reasoning: Some("Plan."),
        content: Some("Hi!"),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "Stray text<|channel|>final<|message|>Hi<|end|>more stray",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>analysis<|message|>Plan.<|end|><|start|>assistant<|chan",
        finish: EngineFinish::Length,
        reasoning: Some("Plan."),
        content: None,
        calls: &[],
    },
    Case {
        prompt: "hello ",
        output: "<|channel|>final<|message|>Hi",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant<|channel|>final<|message|>  ",
        output: " more.",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("more."),
        calls: &[],
    },
];

/// The Gemma 4 template's cases: a thought and a call, edges, a turn after a tool's response and
/// after a later turn, an answer before a call, a call written inside a thought (the thought's
/// text, as the template reads it), a thought the prompt opened, and Python's whitespace.
const GEMMA4: &[Case] = &[
    Case {
        prompt: "<|turn>user\nWeather?<turn|>\n<|turn>model\n",
        output: "<|channel>thought\nThe user wants the weather.\n<channel|><|tool_call>\
                 call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
        finish: EngineFinish::Stop,
        reasoning: Some("The user wants the weather."),
        content: None,
        calls: &[("get_weather", "{\"city\":\"Paris\"}")],
    },
    Case {
        prompt: "<|turn>model\n",
        output: "\n\nIt is sunny.\n\n",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("It is sunny."),
        calls: &[],
    },
    Case {
        prompt: "<|turn>model\n<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>\
                 <|tool_response>response:get_weather{temp:20}<tool_response|>",
        output: "It is 20 degrees.",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("It is 20 degrees."),
        calls: &[],
    },
    Case {
        prompt: "<|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>\
                 response:f{}<tool_response|>Done.<turn|>\n<|turn>user\nThanks<turn|>\n<|turn>\
                 model\n",
        output: "Hello.",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hello."),
        calls: &[],
    },
    Case {
        prompt: "<|turn>model\n",
        output: "Let me check.\n<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,\
                 days:3}<tool_call|>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Let me check."),
        calls: &[("get_weather", "{\"city\":\"Paris\",\"days\":3}")],
    },
    Case {
        prompt: "<|turn>model\n",
        output: "<|channel>thought\nPlan <|tool_call>call:f{}<tool_call|> more<channel|>Done",
        finish: EngineFinish::Stop,
        reasoning: Some("Plan <|tool_call>call:f{}<tool_call|> more"),
        content: Some("Done"),
        calls: &[],
    },
    Case {
        prompt: "<|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>\
                 response:f{}<tool_response|><|channel>thought\n",
        output: "Plan.\n<channel|>Answer.",
        finish: EngineFinish::Stop,
        reasoning: Some("Plan."),
        content: Some("Answer."),
        calls: &[],
    },
    Case {
        prompt: "<|turn>model\n",
        output: "\u{a0}\u{1c}Hi\u{2003}there\u{2003}\u{1f}",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi\u{2003}there"),
        calls: &[],
    },
];

/// The ERNIE test template's cases: transformers' tests' outputs without their stop text (a
/// thought then a call, a thought then an answer, `<think>` at the turn's first byte), an answer
/// or a close at the first byte, a prompt that writes the first byte, and a thought after an
/// answer.
const ERNIE: &[Case] = &[
    Case {
        prompt: "User: Weather?\nAssistant:",
        output: "The user is asking about the weather in Paris today. Let me check the available \
                 tools.\n</think>\n\n<tool_call>\n{\"name\": \"get_current_temperature\", \
                 \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\n",
        finish: EngineFinish::Stop,
        reasoning: Some(
            "The user is asking about the weather in Paris today. Let me check the \
             available tools.",
        ),
        content: None,
        calls: &[("get_current_temperature", "{\"location\":\"Paris\"}")],
    },
    Case {
        prompt: "User: Hi! How are you?\nAssistant:",
        output:
            "The user just greeted me.\n\nFirst, I'll say hello back.\n</think>\n\n<response>\n\
             Hello! I'm doing well, thank you for asking.\n</response>\n",
        finish: EngineFinish::Stop,
        reasoning: Some("The user just greeted me.\n\nFirst, I'll say hello back."),
        content: Some("Hello! I'm doing well, thank you for asking."),
        calls: &[],
    },
    Case {
        prompt: "Assistant:",
        output: "<think>some deliberation here</think>\n\n<tool_call>\n{\"name\": \
                 \"get_current_temperature\", \"arguments\": {\"location\": \"Paris\"}}\n\
                 </tool_call>\n",
        finish: EngineFinish::Stop,
        reasoning: Some("<think>some deliberation here"),
        content: None,
        calls: &[("get_current_temperature", "{\"location\":\"Paris\"}")],
    },
    Case {
        prompt: "Assistant:",
        output: "<response>\nHello!\n</response>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hello!"),
        calls: &[],
    },
    Case {
        prompt: "Assistant:",
        output: "</think>\n<response>\nHi\n</response>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
    Case {
        prompt: "Assistant: ",
        output: "<response>\nHi\n</response>",
        finish: EngineFinish::Stop,
        reasoning: Some("<response>\nHi\n</response>"),
        content: None,
        calls: &[],
    },
    Case {
        prompt: "Assistant:",
        output: "<response>\nHi\n</response><think>More.</think>",
        finish: EngineFinish::Stop,
        reasoning: Some("More."),
        content: Some("Hi"),
        calls: &[],
    },
];

/// The Inkling test template's cases: transformers' multi-message test without the stop token's
/// text, answers with spaces at their edges, thoughts joined with an answer after them, and text
/// outside the messages.
const INKLING: &[Case] = &[
    Case {
        prompt: "<|message_system|><|content_text|>Be brief.<|end_message|><|message_model|>",
        output: "<|content_thinking|>Consider the weather.<|end_message|><|message_model|>\
                 <|content_thinking|> Tokyo, probably.<|end_message|><|message_model|>\
                 <|content_text|>Checking the weather now.<|end_message|><|message_model|>\
                 get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\
                 \"args\":{\"city\":\"Tokyo\",\"units\":\"C\"}}<|end_message|>",
        finish: EngineFinish::Stop,
        reasoning: Some("Consider the weather. Tokyo, probably."),
        content: Some("Checking the weather now."),
        calls: &[("get_weather", "{\"city\":\"Tokyo\",\"units\":\"C\"}")],
    },
    Case {
        prompt: "<|message_model|>",
        output: "<|content_text|> A <|end_message|><|message_model|><|content_text|> B \
                 <|end_message|>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some(" A  B "),
        calls: &[],
    },
    Case {
        prompt: "<|message_model|>",
        output: "<|content_thinking|>one <|end_message|><|message_model|><|content_thinking|>\
                 two<|end_message|><|message_model|><|content_text|>\nHi\n<|end_message|>",
        finish: EngineFinish::Stop,
        reasoning: Some("one two"),
        content: Some("\nHi\n"),
        calls: &[],
    },
    Case {
        prompt: "<|message_model|>",
        output: "stray <|message_model|>author<|content_text|>Hi<|end_message|> tail",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
];

/// The Muse Glimmer template's cases: a thought then an answer, text outside the messages, and a
/// thought alone; then calls: after a thought, with attributes beside the names and text between
/// the parameters, an invoke without a name (no call: text outside the messages), two calls the
/// second without parameters, and a parameter tag whose name runs on, which no parameter begins.
const MUSE: &[Case] = &[
    Case {
        prompt: "<|start|>user<|message|>Weather?<|eot|><|start|>assistant",
        output: "to=self<|message|>Plan.<|eom|><|start|>assistant<atem:invoke \
                 name=\"get_weather\"><atem:parameter name=\"city\">Paris</atem:parameter>\
                 <atem:parameter name=\"days\">3</atem:parameter></atem:invoke>",
        finish: EngineFinish::Stop,
        reasoning: Some("Plan."),
        content: None,
        calls: &[("get_weather", "{\"city\": \"Paris\", \"days\": 3}")],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<atem:invoke id=\"1\" name=\"lookup\"><atem:parameter type=\"string\" \
                 name=\"q\">x y</atem:parameter> <atem:parameter name=\"ids\">[1, 2]\
                 </atem:parameter></atem:invoke>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[("lookup", "{\"q\": \"x y\", \"ids\": [1, 2]}")],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<atem:invoke id=\"1\">text</atem:invoke> to=user<|message|>Hi<|eot|>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<atem:invoke name=\"f\"><atem:parameter name=\"o\">{\"a\": true}\
                 </atem:parameter><atem:parameter name=\"p\">plain words</atem:parameter>\
                 </atem:invoke><atem:invoke name=\"g\"></atem:invoke>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[
            ("f", "{\"o\": {\"a\": true}, \"p\": \"plain words\"}"),
            ("g", "{}"),
        ],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<atem:invoke name=\"f\"><atem:parameters name=\"x\">1</atem:parameter>\
                 <atem:parameter name=\"y\">2</atem:parameter></atem:invoke>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[("f", "{\"y\": 2}")],
    },
    Case {
        prompt: "<|start|>user<|message|>Weather?<|eot|><|start|>assistant",
        output: "to=self<|message|>Plan the answer.<|eom|><|start|>assistant to=user<|message|>It \
                 is sunny.",
        finish: EngineFinish::Stop,
        reasoning: Some("Plan the answer."),
        content: Some("It is sunny."),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant",
        output: " junk to=user<|message|>\nHi\n",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant",
        output: " to=self<|message|> think <|eom|>",
        finish: EngineFinish::Stop,
        reasoning: Some("think"),
        content: None,
        calls: &[],
    },
];

/// The Qwen 3.5 serve template's cases, the requests declaring the weather tool: a thought, an
/// answer and a call; two calls; a `</tool_call>` without `</function>` before it, which is the
/// call's text, so the call closes at the later one; a call marker with no function after it,
/// which is the answer's text; text between parameters and spaces around a value; a name with a
/// space; a key written twice (the last value stands); a value that never ends before the call
/// closes; and a thought the prompt opened, with an empty value and a value on two lines.
const SERVE_QWEN3_5: &[Case] = &[
    Case {
        prompt: "<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n",
        output: "<think>\nThe user wants the weather.\n</think>\n\nLet me check.\n<tool_call>\n\
                 <function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
                 <parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
        finish: EngineFinish::Stop,
        reasoning: Some("The user wants the weather."),
        content: Some("Let me check."),
        calls: &[("get_weather", "{\"city\": \"Paris\", \"days\": 3}")],
    },
    Case {
        prompt: "<|im_start|>assistant\n",
        output: "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
                 </function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n\
                 <parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[
            ("get_weather", "{\"city\": \"Paris\"}"),
            ("get_weather", "{\"city\": \"Rome\"}"),
        ],
    },
    Case {
        prompt: "<|im_start|>assistant\n",
        output: "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
                 </tool_call>\nMore text</function></tool_call>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[("get_weather", "{\"city\": \"Paris\"}")],
    },
    Case {
        prompt: "<|im_start|>assistant\n",
        output: "<tool_call>\nnot a function</tool_call>Hi",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("<tool_call>\nnot a function</tool_call>Hi"),
        calls: &[],
    },
    Case {
        prompt: "<|im_start|>assistant\n",
        output: "<tool_call>\n<function=get_weather>\nnoise <parameter=city>  Paris  </parameter> \
                 junk <parameter=days>4</parameter></function>\n</tool_call>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[("get_weather", "{\"city\": \"Paris\", \"days\": 4}")],
    },
    Case {
        prompt: "<|im_start|>assistant\n",
        output: "<tool_call>\n<function=get weather>\n<parameter=city>\nParis\n</parameter>\n\
                 </function>\n</tool_call>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[("get weather", "{\"city\": \"Paris\"}")],
    },
    Case {
        prompt: "<|im_start|>assistant\n",
        output: "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
                 <parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[("get_weather", "{\"city\": \"Rome\"}")],
    },
    Case {
        prompt: "<|im_start|>assistant\n",
        output: "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</function>\n\
                 </tool_call>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[("get_weather", "{}")],
    },
    Case {
        prompt: "<|im_start|>assistant\n<think>\n",
        output: "plan\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=days>\n\
                 </parameter>\n<parameter=city>\nNew\nYork\n</parameter>\n</function>\n\
                 </tool_call>",
        finish: EngineFinish::Stop,
        reasoning: Some("plan"),
        content: None,
        calls: &[("get_weather", "{\"days\": \"\", \"city\": \"New\\nYork\"}")],
    },
];

/// Replays `case` through `format` at every chunking, and checks that the message is the
/// recorded one at each and that the events account for the output's bytes.
fn check(name: &str, format: &Format, case: &Case) {
    check_declared(name, format, case, &Declared::default());
}

/// [`check`] for a request whose tools are `declared`.
#[expect(
    clippy::panic,
    clippy::expect_used,
    reason = "a check that fails fails the test that runs it"
)]
fn check_declared(name: &str, format: &Format, case: &Case, declared: &Declared) {
    for (plan, cuts) in chunkings(case.output, usize::MAX) {
        let mut parser = Engine::new(format.clone(), declared.clone());
        let events = replay_after(
            &mut parser as &mut dyn Parser,
            case.prompt,
            case.output,
            &cuts,
            &case.finish,
            false,
        )
        .unwrap_or_else(|e| panic!("{name}: {:?}: {e}", case.output));
        let place = || format!("{name}: {:?} after {:?}, {plan}", case.output, case.prompt);
        let conserved: String = events.iter().map(bytes_of).collect();
        assert_eq!(conserved, case.output, "{}", place());
        let message = adapt::chat::message(0, &events).message;
        assert_eq!(
            message.reasoning_content.as_deref(),
            case.reasoning,
            "{}",
            place()
        );
        assert_eq!(message.content.as_deref(), case.content, "{}", place());
        let calls: Vec<(String, Value)> = message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|call| {
                let arguments = call.function.arguments.unwrap_or_default();
                let value = serde_json::from_str(&arguments)
                    .unwrap_or_else(|e| panic!("{}: arguments {arguments:?}: {e}", place()));
                (call.function.name, value)
            })
            .collect();
        let expected: Vec<(String, Value)> = case
            .calls
            .iter()
            .map(|(name, arguments)| {
                let value = serde_json::from_str(arguments).expect("recorded arguments");
                (name.to_string(), value)
            })
            .collect();
        assert_eq!(calls, expected, "{}", place());
    }
}

#[test]
fn gpt_oss_test_template_reads_as_hugging_face_reads_it() {
    let format = gpt_oss();
    for case in GPT_OSS {
        check("gpt-oss", &format, case);
    }
}

#[test]
fn gemma4_template_reads_as_hugging_face_reads_it() {
    let format = gemma4();
    for case in GEMMA4 {
        check("gemma 4", &format, case);
    }
}

#[test]
fn ernie_test_template_reads_as_hugging_face_reads_it() {
    let format = ernie();
    for case in ERNIE {
        check("ernie", &format, case);
    }
}

#[test]
fn inkling_test_template_reads_as_hugging_face_reads_it() {
    let format = inkling();
    for case in INKLING {
        check("inkling", &format, case);
    }
}

#[test]
fn muse_glimmer_template_reads_as_hugging_face_reads_it() {
    let format = muse();
    for case in MUSE {
        check("muse", &format, case);
    }
}

#[test]
fn qwen3_5_serve_template_reads_as_hugging_face_reads_it() {
    let (format, declared) = (serve_qwen3_5(), weather());
    for case in SERVE_QWEN3_5 {
        check_declared("qwen 3.5 serve", &format, case, &declared);
    }
}

#[test]
fn a_field_written_twice_reads_as_both_stays() {
    // The content's own close between two stays, in a stream that does not stop on it: Hugging
    // Face's parser keeps the last stay ("there"); a stream has sent the first by then, so the
    // table reads both, each trimmed.
    check(
        "gemma 4",
        &gemma4(),
        &Case {
            prompt: "<|turn>model\n",
            output: "Hi <|tool_response> there",
            finish: EngineFinish::Stop,
            reasoning: None,
            content: Some("Hithere"),
            calls: &[],
        },
    );
}

#[test]
fn every_table_here_is_valid_and_names_its_stop_tokens() {
    for (format, stops) in [
        (gpt_oss(), &[][..]),
        (gemma4(), &["<|tool_response>"][..]),
        (ernie(), &[][..]),
        (inkling(), &[][..]),
        (muse(), &["<|eot|>"][..]),
        (serve_qwen3_5(), &[][..]),
    ] {
        assert_eq!(format.validate(), Ok(()), "{}", format.name());
        for stop in stops {
            assert!(format.is_stop_token(stop), "{}: {stop}", format.name());
        }
    }
    // A stop token that is not a terminal is not kept, and a terminal that ends a message inside
    // the turn is not a stop token.
    assert!(gpt_oss().is_stop_token("<|call|>"));
    assert!(!gpt_oss().is_stop_token("<|return|>"));
    assert!(!gemma4().is_stop_token("<turn|>"));
    assert!(!inkling().is_stop_token("<|end_message|>"));
    assert!(!muse().is_stop_token("<|eom|>"));
}

// The structural tags of the shaped calls above, written from their chat templates' layouts.

/// A text of the layout, from its literals and places.
fn written(segments: impl IntoIterator<Item = Segment>) -> Written {
    Written(segments.into_iter().collect())
}

fn piece(text: &str) -> Segment {
    Segment::Lit(text.into())
}

/// How the chat template of `meta-models/Muse-Glimmer-30B` (revision `a4e59da52a7b`, its
/// `chat_template.jinja`) writes a turn, as transformers 5.18.0's `apply_chat_template` rendered
/// the probe conversation ([`probe_tools`]): a call is a message to the tool, ` to=NAME`
/// and `<|message|>`, with `<atem:function_calls>` on a line of its own, which the response
/// template reads as text outside every field; the invoke's one attribute is the name; a
/// parameter a line, its value between the tags; between two calls the block's close, `<|eom|>`
/// and the next message's start; after the last, the block's close, then `<|eot|>`, a stop token.
/// The thought and the answer are messages to the model itself and to the user, with a space
/// before the recipient; the answer ends at `<|eot|>`.
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

/// How the chat template of `Qwen/Qwen3.5-9B` (revision `c20223623576`, its `chat_template.jinja`)
/// writes a turn, as transformers 5.18.0's `apply_chat_template` rendered the probe conversation:
/// `<tool_call>`, the function's tag and each parameter's tag, value and close on lines of their
/// own, `True` and `None` as Python writes them, lists and objects as JSON, a newline between two
/// calls, nothing after the last but `<|im_end|>`, a stop token; the thought's close followed by
/// a blank line.
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

/// The probe conversation's tools: `fn_alpha`, a required parameter of each JSON type (`z` a
/// nullable string), and `fn_beta`, one string.
fn probe_tools() -> Vec<Tool> {
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
    vec![
        tool(
            "fn_alpha",
            json!({
                "type": "object",
                "properties": {
                    "s": {"type": "string"},
                    "i": {"type": "integer"},
                    "n": {"type": "number"},
                    "b": {"type": "boolean"},
                    "a": {"type": "array", "items": {"type": "integer"}},
                    "o": {"type": "object"},
                    "z": {"type": "string", "nullable": true},
                },
                "required": ["s", "i", "n", "b", "a", "o", "z"],
            }),
        ),
        tool(
            "fn_beta",
            json!({
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            }),
        ),
    ]
}

/// The probe's calls: `fn_alpha` with a value of each type, then `fn_beta`, as the tables here
/// read them. The nullable `z` reads as null from the word the template writes for it (`null` in
/// Muse Glimmer's calls, `None` in Qwen 3.5's), since its schema admits null.
fn probe_calls(both: bool) -> Value {
    let alpha = json!({"name": "fn_alpha", "arguments": {
        "s": "sigma", "i": 42, "n": 2.5, "b": true, "a": [1, 2], "o": {"m": "mu"}, "z": null,
    }});
    let beta = json!({"name": "fn_beta", "arguments": {"q": "kappa"}});
    if both {
        json!([alpha, beta])
    } else {
        json!([alpha])
    }
}

/// What Muse Glimmer's chat template rendered after the generation prompt (`<|start|>assistant`)
/// for the probe's two calls, its first call alone and a JSON answer, each up to `<|eot|>`.
const MUSE_CALLS: &str = " to=fn_alpha<|message|><atem:function_calls>\n<atem:invoke \
     name=\"fn_alpha\">\n<atem:parameter name=\"s\">sigma</atem:parameter>\n<atem:parameter \
     name=\"i\">42</atem:parameter>\n<atem:parameter name=\"n\">2.5</atem:parameter>\n\
     <atem:parameter name=\"b\">true</atem:parameter>\n<atem:parameter name=\"a\">[1, \
     2]</atem:parameter>\n<atem:parameter name=\"o\">{\"m\": \"mu\"}</atem:parameter>\n\
     <atem:parameter name=\"z\">null</atem:parameter>\n</atem:invoke>\n</atem:function_calls>\
     <|eom|><|start|>assistant to=fn_beta<|message|><atem:function_calls>\n<atem:invoke \
     name=\"fn_beta\">\n<atem:parameter name=\"q\">kappa</atem:parameter>\n</atem:invoke>\n\
     </atem:function_calls>";
const MUSE_ONE_CALL: &str = " to=fn_alpha<|message|><atem:function_calls>\n<atem:invoke \
     name=\"fn_alpha\">\n<atem:parameter name=\"s\">sigma</atem:parameter>\n<atem:parameter \
     name=\"i\">42</atem:parameter>\n<atem:parameter name=\"n\">2.5</atem:parameter>\n\
     <atem:parameter name=\"b\">true</atem:parameter>\n<atem:parameter name=\"a\">[1, \
     2]</atem:parameter>\n<atem:parameter name=\"o\">{\"m\": \"mu\"}</atem:parameter>\n\
     <atem:parameter name=\"z\">null</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
const MUSE_ANSWER: &str = " to=user<|message|>{\"probe\": 1}";
/// The generation prompt's end for Muse Glimmer.
const MUSE_PROMPT: &str = "<|start|>user<|message|>Call the probe tools.<|eot|><|start|>assistant";

/// What Qwen 3.5's chat template rendered for the probe's two calls and a JSON answer, each up to
/// `<|im_end|>`: after its default generation prompt, which opens the thought, and after the one
/// with thinking turned off, which opens and closes it.
const QWEN3_5_CALLS_THINKING: &str = "\n</think>\n\n<tool_call>\n<function=fn_alpha>\n\
     <parameter=s>\nsigma\n</parameter>\n<parameter=i>\n42\n</parameter>\n<parameter=n>\n2.5\n\
     </parameter>\n<parameter=b>\nTrue\n</parameter>\n<parameter=a>\n[1, 2]\n</parameter>\n\
     <parameter=o>\n{\"m\": \"mu\"}\n</parameter>\n<parameter=z>\nNone\n</parameter>\n\
     </function>\n</tool_call>\n<tool_call>\n<function=fn_beta>\n<parameter=q>\nkappa\n\
     </parameter>\n</function>\n</tool_call>";
const QWEN3_5_ANSWER_THINKING: &str = "\n</think>\n\n{\"probe\": 1}";
const QWEN3_5_PROMPT_THINKING: &str =
    "<|im_start|>user\nCall the probe tools.<|im_end|>\n<|im_start|>assistant\n<think>\n";
const QWEN3_5_PROMPT: &str = "<|im_start|>user\nCall the probe tools.<|im_end|>\n\
     <|im_start|>assistant\n<think>\n\n</think>\n\n";

/// The message `output` reads as under `format` after `prompt`, typed by the probe's tools.
#[expect(
    clippy::expect_used,
    reason = "an output that does not parse fails the test that reads it"
)]
fn read_probe(format: &Format, prompt: &str, output: &str) -> Value {
    let mut parser = Engine::new(format.clone(), Declared::of(&probe_tools()));
    let events = replay_after(
        &mut parser as &mut dyn Parser,
        prompt,
        output,
        &[],
        &EngineFinish::Stop,
        false,
    )
    .expect("a parse");
    let message = adapt::chat::message(0, &events).message;
    let calls: Vec<Value> = message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|call| {
            let arguments = call.function.arguments.unwrap_or_default();
            json!({
                "name": call.function.name,
                "arguments": serde_json::from_str::<Value>(&arguments).expect("JSON arguments"),
            })
        })
        .collect();
    json!({
        "reasoning": message.reasoning_content,
        "content": message.content,
        "calls": calls,
    })
}

/// Where `grammar`, matched in `text` from `at`, can end, as an engine reads a structural tag: a
/// JSON schema's leaf takes one JSON value of the type it names, free text holds no trigger, and
/// a tag list's free text and tags follow the engines' `at_least_one` and `stop_after_first`.
fn ends(grammar: &Grammar, text: &str, at: usize) -> Vec<usize> {
    let mut found: Vec<usize> = match grammar {
        Grammar::ConstString(literal) | Grammar::Token(literal) => text[at..]
            .starts_with(literal.as_str())
            .then_some(at + literal.len())
            .into_iter()
            .collect(),
        Grammar::AnyText { excludes } => free(text, at, excludes),
        Grammar::JsonSchema { schema, .. } => json_ends(schema, text, at),
        Grammar::Regex(_) => Vec::new(),
        Grammar::Sequence(parts) => parts.iter().fold(vec![at], |starts, part| {
            starts
                .iter()
                .flat_map(|&start| ends(part, text, start))
                .collect()
        }),
        Grammar::Or(options) => options
            .iter()
            .flat_map(|option| ends(option, text, at))
            .collect(),
        Grammar::Optional(inner) => std::iter::once(at).chain(ends(inner, text, at)).collect(),
        Grammar::Star(inner) => repeated(inner, text, at, 0),
        Grammar::Plus(inner) => repeated(inner, text, at, 1),
        Grammar::Tag(tag) => tag_ends(tag, text, at),
        Grammar::TriggeredTags {
            triggers,
            tags,
            at_least_one,
            stop_after_first,
        } => triggered(triggers, tags, *at_least_one, *stop_after_first, text, at),
        Grammar::TagsWithSeparator {
            tags,
            separator,
            at_least_one,
            stop_after_first,
        } => {
            let mut found = if *at_least_one { Vec::new() } else { vec![at] };
            let mut frontier: Vec<usize> = tags
                .iter()
                .flat_map(|tag| tag_ends(tag, text, at))
                .collect();
            while let Some(end) = frontier.pop() {
                if found.contains(&end) {
                    continue;
                }
                found.push(end);
                if *stop_after_first || !text[end..].starts_with(separator.as_str()) {
                    continue;
                }
                let next = end + separator.len();
                frontier.extend(tags.iter().flat_map(|tag| tag_ends(tag, text, next)));
            }
            found
        }
    };
    found.sort_unstable();
    found.dedup();
    found
}

/// Where free text from `at` can end: before any of `excludes` is whole in it.
fn free(text: &str, at: usize, excludes: &[String]) -> Vec<usize> {
    let mut found = vec![at];
    for (offset, c) in text[at..].char_indices() {
        let end = at + offset + c.len_utf8();
        if excludes
            .iter()
            .any(|exclude| !exclude.is_empty() && text[at..end].contains(exclude.as_str()))
        {
            break;
        }
        found.push(end);
    }
    found
}

/// Where a JSON value from `at` can end: each place up to which the text is one JSON value of the
/// type `schema` names (any value for a schema without one), with no space at its edges.
fn json_ends(schema: &Value, text: &str, at: usize) -> Vec<usize> {
    let fits = |value: &Value| match schema.get("type").and_then(Value::as_str) {
        Some("integer") => value.is_i64() || value.is_u64(),
        Some("number") => value.is_number(),
        Some("boolean") => value.is_boolean(),
        Some("string") => value.is_string(),
        Some("array") => value.is_array(),
        Some("object") => value.is_object(),
        Some("null") => value.is_null(),
        _ => true,
    };
    text[at..]
        .char_indices()
        .map(|(offset, c)| at + offset + c.len_utf8())
        .filter(|&end| {
            let candidate = &text[at..end];
            candidate.trim() == candidate
                && serde_json::from_str::<Value>(candidate).is_ok_and(|value| fits(&value))
        })
        .collect()
}

fn repeated(inner: &Grammar, text: &str, at: usize, least: usize) -> Vec<usize> {
    let mut reached: Vec<(usize, usize)> = vec![(at, 0)];
    let mut found = Vec::new();
    while let Some((start, count)) = reached.pop() {
        if count >= least && !found.contains(&start) {
            found.push(start);
        }
        for end in ends(inner, text, start) {
            if end > start && !reached.contains(&(end, count + 1)) && count < 64 {
                reached.push((end, count + 1));
            }
        }
    }
    found
}

fn tag_ends(tag: &Tag, text: &str, at: usize) -> Vec<usize> {
    let begin = Grammar::ConstString(tag.begin.clone());
    let end = Grammar::ConstString(tag.end.clone());
    ends(
        &Grammar::Sequence(vec![begin, (*tag.content).clone(), end]),
        text,
        at,
    )
}

fn triggered(
    triggers: &[String],
    tags: &[Tag],
    at_least_one: bool,
    stop_after_first: bool,
    text: &str,
    at: usize,
) -> Vec<usize> {
    let tags_from = |start: usize| -> Vec<usize> {
        tags.iter()
            .flat_map(|tag| tag_ends(tag, text, start))
            .collect()
    };
    let mut found: Vec<usize> = Vec::new();
    // (where free text may start, whether a tag was written)
    let mut frontier: Vec<(usize, bool)> = if at_least_one {
        tags_from(at).into_iter().map(|end| (end, true)).collect()
    } else {
        vec![(at, false)]
    };
    let mut seen: Vec<(usize, bool)> = Vec::new();
    while let Some((start, written)) = frontier.pop() {
        if seen.contains(&(start, written)) {
            continue;
        }
        seen.push((start, written));
        if written && stop_after_first {
            found.push(start);
            continue;
        }
        for end in free(text, start, triggers) {
            found.push(end);
            if triggers
                .iter()
                .any(|trigger| text[end..].starts_with(trigger.as_str()))
            {
                frontier.extend(tags_from(end).into_iter().map(|after| (after, true)));
            }
        }
    }
    found
}

/// Whether `grammar` takes all of `text`.
fn accepts(grammar: &Grammar, text: &str) -> bool {
    ends(grammar, text, 0).contains(&text.len())
}

#[test]
fn every_layout_here_fits_its_table() {
    assert_eq!(muse_layout().check(&muse()), Ok(()));
    assert_eq!(serve_qwen3_5_layout().check(&serve_qwen3_5()), Ok(()));
    // Recorded with the chat template, read back through the table: the probe's calls.
    let muse = muse();
    assert_eq!(
        read_probe(&muse, MUSE_PROMPT, MUSE_CALLS),
        json!({"reasoning": null, "content": null, "calls": probe_calls(true)})
    );
    assert_eq!(
        read_probe(&muse, MUSE_PROMPT, MUSE_ONE_CALL),
        json!({"reasoning": null, "content": null, "calls": probe_calls(false)})
    );
    let qwen = serve_qwen3_5();
    assert_eq!(
        read_probe(&qwen, QWEN3_5_PROMPT_THINKING, QWEN3_5_CALLS_THINKING),
        json!({"reasoning": null, "content": null, "calls": probe_calls(true)})
    );
    // The JSON answers read back as that answer and no call.
    let answer = json!({"reasoning": null, "content": "{\"probe\": 1}", "calls": []});
    assert_eq!(read_probe(&muse, MUSE_PROMPT, MUSE_ANSWER), answer);
    assert_eq!(
        read_probe(&qwen, QWEN3_5_PROMPT_THINKING, QWEN3_5_ANSWER_THINKING),
        answer
    );
}

/// The tags of `format` for the probe's tools: a forced call, a named function (one call, at
/// least one), one call at most, and a JSON answer; with `reasoning_open` as the prompt leaves it.
#[expect(
    clippy::expect_used,
    reason = "a table with no tag for a mode fails the test that asks for it"
)]
fn modes(format: &Format, reasoning_open: bool) -> [Grammar; 4] {
    let tools = probe_tools();
    let options = |at_least_one, one_call| Options {
        at_least_one,
        reasoning_open,
        one_call,
    };
    let tag = |options| format.grammar_for(&tools, options).expect("a tag");
    [
        tag(options(true, false)),
        tag(options(true, true)),
        tag(options(false, true)),
        format
            .answer_grammar(&json!({"type": "object"}), reasoning_open)
            .expect("an answer tag"),
    ]
}

#[test]
fn muse_glimmers_tags_take_what_its_chat_template_writes() {
    let format = muse().with_layout(muse_layout());
    assert!(!format.reasoning_open(MUSE_PROMPT));
    let [required, named, one, answer] = modes(&format, false);
    assert!(accepts(&required, MUSE_CALLS));
    assert!(accepts(&required, MUSE_ONE_CALL));
    assert!(accepts(&named, MUSE_ONE_CALL));
    assert!(!accepts(&named, MUSE_CALLS), "one call");
    assert!(accepts(&one, MUSE_ONE_CALL));
    assert!(
        accepts(&one, MUSE_ANSWER),
        "no call owed: the answer is free text"
    );
    assert!(accepts(&answer, MUSE_ANSWER));
    // A forced call starts with the lead the template writes for the tool, and the invoke's
    // marker triggers each call.
    let payload = required.payload();
    assert_eq!(
        payload["format"]["elements"][0],
        json!({"type": "or", "elements": [
            {"type": "const_string", "value": " to=fn_alpha<|message|><atem:function_calls>\n"},
            {"type": "const_string", "value": " to=fn_beta<|message|><atem:function_calls>\n"},
        ]})
    );
    let calls = &payload["format"]["elements"][1];
    assert_eq!(calls["triggers"], json!(["<atem:invoke"]));
    assert_eq!(
        calls["tags"][1]["begin"],
        "<atem:invoke name=\"fn_beta\">\n"
    );
    assert_eq!(
        calls["tags"][1]["end"], "</atem:invoke>",
        "the newline after the last parameter is its own"
    );
    // Near misses: an unknown tool, a missing required key, a call without its lead or without
    // its close.
    for miss in [
        MUSE_ONE_CALL.replace("fn_alpha", "fn_gamma"),
        MUSE_ONE_CALL.replace("<atem:parameter name=\"s\">sigma</atem:parameter>\n", ""),
        MUSE_ONE_CALL.replace(" to=fn_alpha<|message|><atem:function_calls>\n", ""),
        MUSE_ONE_CALL.replace("</atem:invoke>", ""),
        MUSE_ONE_CALL.replace("name=\"fn_alpha\"", "nom=\"fn_alpha\""),
    ] {
        assert!(!accepts(&required, &miss), "{miss:?}");
        assert!(!accepts(&named, &miss), "{miss:?}");
    }
    assert!(!accepts(&answer, "{\"probe\": 1}"), "without the recipient");
}

#[test]
fn qwen3_5s_tags_take_what_its_chat_template_writes() {
    let format = serve_qwen3_5().with_layout(serve_qwen3_5_layout());
    // The default prompt leaves the thought open: the tag follows a prefix, never a refusal.
    assert!(format.reasoning_open(QWEN3_5_PROMPT_THINKING));
    assert!(!format.reasoning_open(QWEN3_5_PROMPT));
    let [required, named, one, answer] = modes(&format, true);
    assert_eq!(required.payload()["format"]["type"], "sequence");
    assert!(accepts(&required, QWEN3_5_CALLS_THINKING));
    let one_call = QWEN3_5_CALLS_THINKING
        .split_once("\n<tool_call>\n<function=fn_beta>")
        .map(|(first, _)| first)
        .expect("two calls");
    assert!(accepts(&named, one_call));
    assert!(!accepts(&named, QWEN3_5_CALLS_THINKING), "one call");
    assert!(accepts(&one, one_call));
    // With no call owed, the thought is the tag's own free text: no prefix whose end could split
    // the call's marker, so a second call cannot pass as text.
    assert_eq!(one.payload()["format"]["type"], "triggered_tags");
    assert!(!accepts(&one, QWEN3_5_CALLS_THINKING), "one call");
    assert!(accepts(&answer, QWEN3_5_ANSWER_THINKING));
    let thought = format!("Probe thought.{QWEN3_5_CALLS_THINKING}");
    assert!(accepts(&required, &thought), "the thought before its close");
    // With thinking off, the prompt closed the thought: the calls start at once.
    let [required, named, _, answer] = modes(&format, false);
    let calls = QWEN3_5_CALLS_THINKING.trim_start_matches("\n</think>\n\n");
    assert!(accepts(&required, calls));
    assert!(accepts(
        &named,
        one_call.trim_start_matches("\n</think>\n\n")
    ));
    assert!(accepts(&answer, "{\"probe\": 1}"));
    assert!(!accepts(&required, QWEN3_5_CALLS_THINKING));
    for miss in [
        calls.replace("fn_beta", "fn_gamma"),
        calls.replace("<parameter=q>\nkappa\n</parameter>\n", ""),
        calls.replace("\n<function=fn_alpha>", "\nfn_alpha"),
        calls.replace("</function>\n</tool_call>", "</tool_call>"),
        calls.replace("True", "true"),
    ] {
        assert!(!accepts(&required, &miss), "{miss:?}");
    }
    for (prompt, output) in [
        (QWEN3_5_PROMPT, calls),
        (QWEN3_5_PROMPT_THINKING, QWEN3_5_CALLS_THINKING),
    ] {
        assert_eq!(
            read_probe(&format, prompt, output),
            json!({"reasoning": null, "content": null, "calls": probe_calls(true)})
        );
    }
    assert_eq!(
        read_probe(&format, QWEN3_5_PROMPT_THINKING, one_call),
        json!({"reasoning": null, "content": null, "calls": probe_calls(false)})
    );
}

#[test]
fn qwen3_5s_tag_is_the_tagged_tables_where_their_layouts_agree() {
    // The registry's Qwen 3.5 table writes the same calls by hand: the same trigger, the same
    // begin and end for every tool, and each parameter's tag, close and newline. The values
    // differ only where the hand-written tag allows a spelling the chat template does not write:
    // a newline on either side of a value that is not text, which the template always writes, and
    // JSON's `true` beside Python's words. A string's text holds the same markers.
    let tools = probe_tools();
    let shaped = serve_qwen3_5()
        .with_layout(serve_qwen3_5_layout())
        .grammar(&tools, true, false)
        .expect("a tag")
        .payload();
    let tagged = symphony::formats::qwen3(CallSyntax::Tagged(
        symphony::tagged::Spelling::Json,
        symphony::tagged::Placement::OwnLine,
    ))
    .grammar(&tools, true, false)
    .expect("a tag")
    .payload();
    let (shaped, tagged) = (&shaped["format"], &tagged["format"]);
    assert_eq!(shaped["triggers"], tagged["triggers"]);
    assert_eq!(shaped["at_least_one"], tagged["at_least_one"]);
    for at in 0..2 {
        for field in ["begin", "end"] {
            assert_eq!(
                shaped["tags"][at][field], tagged["tags"][at][field],
                "{field} of tag {at}"
            );
        }
    }
    let parameter = |tag: &Value, at: usize| tag["content"]["elements"][at].clone();
    let newline = json!({"type": "optional", "content": {"type": "const_string", "value": "\n"}});
    let alpha = (&shaped["tags"][0], &tagged["tags"][0]);
    // `s`, a string: the tag, the text, the close and the newline, alike but for the text's
    // markers, which leave out the opener of a call that would be text inside the value.
    assert_eq!(
        parameter(alpha.0, 0),
        json!({"type": "sequence", "elements": [
            {"type": "const_string", "value": "<parameter=s>\n"},
            {"type": "any_text", "excludes": ["</parameter>", "</tool_call>"]},
            {"type": "const_string", "value": "\n</parameter>\n"},
        ]})
    );
    // `i`, an integer: the template's newlines, where the hand-written tag allows them.
    assert_eq!(
        parameter(alpha.0, 1)["elements"][1],
        parameter(alpha.1, 1)["elements"][4]
    );
    assert_eq!(parameter(alpha.1, 1)["elements"][3], newline);
    // `b`, a boolean: Python's words, as the template writes them; the hand-written tag takes
    // JSON's too.
    assert_eq!(
        parameter(alpha.0, 3)["elements"][1],
        json!({"type": "or", "elements": [
            {"type": "const_string", "value": "True"},
            {"type": "const_string", "value": "False"},
        ]})
    );
    assert_eq!(
        parameter(alpha.1, 3)["elements"][4]["elements"][1],
        json!({"type": "const_string", "value": "True"})
    );
}
