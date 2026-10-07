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
    formats, json::PartialJson, CallSyntax, Declared, DropReason, Engine, EngineFinish, Event,
    Events, FinishReason, Input, MalformedReason, ParseError, Parser, TokenSpan,
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
        formats::qwen3(CallSyntax::Tagged),
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
        },
    }])
}

fn hy4() -> Box<dyn Parser> {
    Box::new(Engine::new(formats::hy4(), keyed_tools()))
}

fn ling() -> Box<dyn Parser> {
    Box::new(Engine::new(formats::ling(), keyed_tools()))
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
            },
        }]),
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
        for cuts in chunkings(text) {
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
        for cuts in chunkings(text) {
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
        for cuts in chunkings(text) {
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
        for cuts in chunkings(text) {
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
