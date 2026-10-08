//! The structural tag a table yields: the constraint an engine takes for a forced or required
//! tool call, built from the same terminals and call syntax the parser reads, so the constrained
//! output is exactly what the format parses.
//!
//! [`Grammar`] is Symphony's own tree, in the shape the engines' structural-tag format takes
//! (xgrammar's, which vLLM and SGLang accept): a `triggered_tags` of per-tool `tag`s, each a
//! `begin`, a content grammar and an `end`; `sequence`, `or`, `star`, `plus` and `optional` to
//! compose; `const_string`, `any_text`, `regex` and `json_schema` as leaves. [`Grammar::to_json`]
//! writes it as the engine reads it, and [`Grammar::payload`] wraps it as the request carries it.
//!
//! [`Format::grammar`] derives the tree from the table. The call markers are the terminals that
//! enter and leave the arguments state, the block markers the ones around a wrapper state, the
//! thought's close the terminal that leaves the reasoning state, and the request's tools give each
//! call its name and schema. What the syntax inside the markers constrains comes from the call
//! syntax: a JSON object around each tool's schema for the JSON family; the engines' `glm_xml`
//! style of a schema for keyed arguments in GLM's spelling; a tag grammar per parameter for DSML.
//! The gateway's old parsers wrote these tags by hand, one function per parser; where one exists
//! for a format Symphony has, the derived tag is pinned to it in the tests, so the switch changes
//! nothing an engine sees.
//!
//! With the prompt's reasoning still open, the tag is wrapped in a prefix: free text that can
//! write none of the table's markers nor the call syntax's inner tags, closed by the thought's
//! close, so the forced call follows the reasoning instead of replacing it, as the gateway's
//! `wrap_in_reasoning_prefix` does. A table without a thought has nothing to wrap, and the tag
//! comes back as it is; a thought that closes somewhere other than content has no prefix yet, and
//! the table gives `None` rather than a tag the model could write inside its thought.
//!
//! Not derived yet, so [`Format::grammar`] gives `None` for them: the tagged syntax (Qwen 3.5 and
//! later, Seed-OSS), keyed arguments in another spelling than the compact one (Ling's newlines,
//! Hy4's suffixed tags), MiniMax M3's XML tree, Kimi K3's XTML, the pythonic tables, xLAM's list,
//! and a table without calls. The gateway then takes the JSON-schema path, as it does for those
//! models today.

use openai_protocol::common::Tool;
use serde_json::{json, Value};

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::{dsml, keyed},
};

/// A structural-tag grammar, as the engines take it.
#[derive(Clone, Debug, PartialEq)]
pub enum Grammar {
    /// Exactly this text.
    ConstString(String),
    /// Any text that contains none of `excludes`.
    AnyText { excludes: Vec<String> },
    /// Text the regular expression accepts.
    Regex(String),
    /// A JSON value the schema accepts, written in the engine's `style` when one is named
    /// (`glm_xml` writes it as `<arg_key>`/`<arg_value>` pairs).
    JsonSchema {
        schema: Value,
        style: Option<String>,
    },
    /// The elements one after another.
    Sequence(Vec<Grammar>),
    /// One of the elements.
    Or(Vec<Grammar>),
    /// The content or nothing.
    Optional(Box<Grammar>),
    /// The content any number of times, none included.
    Star(Box<Grammar>),
    /// The content one or more times.
    Plus(Box<Grammar>),
    /// `begin`, the content, `end`.
    Tag(Tag),
    /// Free text until one of `triggers` opens a tag; with `at_least_one`, at least one tag.
    TriggeredTags {
        triggers: Vec<String>,
        tags: Vec<Tag>,
        at_least_one: bool,
    },
    /// Tags one after another with `separator` between them; with `at_least_one`, at least one.
    TagsWithSeparator {
        tags: Vec<Tag>,
        separator: String,
        at_least_one: bool,
    },
}

/// A tagged region: `begin`, what the content may be, `end`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tag {
    pub begin: String,
    pub content: Box<Grammar>,
    pub end: String,
}

impl Tag {
    fn new(begin: impl Into<String>, content: Grammar, end: impl Into<String>) -> Self {
        Self {
            begin: begin.into(),
            content: Box::new(content),
            end: end.into(),
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "type": "tag",
            "begin": self.begin,
            "content": self.content.to_json(),
            "end": self.end,
        })
    }
}

impl Grammar {
    /// The grammar as the engine reads it.
    pub fn to_json(&self) -> Value {
        match self {
            Self::ConstString(value) => json!({"type": "const_string", "value": value}),
            Self::AnyText { excludes } => json!({"type": "any_text", "excludes": excludes}),
            Self::Regex(pattern) => json!({"type": "regex", "pattern": pattern}),
            Self::JsonSchema { schema, style } => {
                let mut written = json!({"type": "json_schema", "json_schema": schema});
                if let Some(style) = style {
                    written["style"] = Value::String(style.clone());
                }
                written
            }
            Self::Sequence(elements) => json!({
                "type": "sequence",
                "elements": elements.iter().map(Self::to_json).collect::<Vec<_>>(),
            }),
            Self::Or(elements) => json!({
                "type": "or",
                "elements": elements.iter().map(Self::to_json).collect::<Vec<_>>(),
            }),
            Self::Optional(content) => json!({"type": "optional", "content": content.to_json()}),
            Self::Star(content) => json!({"type": "star", "content": content.to_json()}),
            Self::Plus(content) => json!({"type": "plus", "content": content.to_json()}),
            Self::Tag(tag) => tag.to_json(),
            Self::TriggeredTags {
                triggers,
                tags,
                at_least_one,
            } => json!({
                "type": "triggered_tags",
                "triggers": triggers,
                "tags": tags.iter().map(Tag::to_json).collect::<Vec<_>>(),
                "at_least_one": at_least_one,
            }),
            Self::TagsWithSeparator {
                tags,
                separator,
                at_least_one,
            } => json!({
                "type": "tags_with_separator",
                "tags": tags.iter().map(Tag::to_json).collect::<Vec<_>>(),
                "separator": separator,
                "at_least_one": at_least_one,
            }),
        }
    }

    /// The grammar as a request carries it to the engine: under `format`.
    pub fn payload(&self) -> Value {
        json!({"format": self.to_json()})
    }
}

/// The markers a table's rows give a call: the terminals that enter and leave the arguments state,
/// and the ones around a wrapper state when the calls sit in a block.
struct CallMarkers<'a> {
    block: Option<(&'a str, &'a str)>,
    call_open: &'a str,
    call_close: &'a str,
}

impl Format {
    /// The structural tag for a forced or required call to one of `tools`, or `None` when the
    /// table's call syntax has no derivation yet, the table has no calls, or no tool has a name.
    /// `at_least_one` asks for at least one call (`tool_choice` `required`, a named function, or
    /// an allowed-tools list in `required` mode). With `reasoning_open`, the prompt left the model
    /// inside its thought, and the tag follows a reasoning prefix the thought's close ends; a
    /// table without a thought has no prefix, and the tag comes back as it is; a table whose
    /// thought closes somewhere other than content has no prefix yet, and gives `None`.
    pub fn grammar(
        &self,
        tools: &[Tool],
        at_least_one: bool,
        reasoning_open: bool,
    ) -> Option<Grammar> {
        let named: Vec<&Tool> = tools
            .iter()
            .filter(|tool| !tool.function.name.is_empty())
            .collect();
        if named.is_empty() {
            return None;
        }
        let markers = self.call_markers()?;
        let calls = match self.call_syntax()? {
            CallSyntax::Json => Self::json_calls(&markers, &named, at_least_one),
            CallSyntax::Keyed(tags) if *tags == keyed::Tags::PLAIN => {
                Self::keyed_calls(&markers, &named, at_least_one)
            }
            CallSyntax::Dsml => Self::dsml_calls(&markers, &named, at_least_one)?,
            CallSyntax::Keyed(_)
            | CallSyntax::Tagged
            | CallSyntax::Pythonic
            | CallSyntax::JsonList
            | CallSyntax::Xml
            | CallSyntax::Xtml => return None,
        };
        if !reasoning_open {
            return Some(calls);
        }
        if !self.has_state(Emits::Reasoning) {
            return Some(calls);
        }
        let think_close = self.marker(Emits::Reasoning, Emits::Content)?;
        let mut excludes: Vec<String> = self.terminal_texts().map(str::to_string).collect();
        let inner: &[&str] = match self.call_syntax() {
            Some(CallSyntax::Keyed(tags)) => &[
                tags.key_open,
                tags.key_close,
                tags.value_open,
                tags.value_close,
            ],
            Some(CallSyntax::Dsml) => &[dsml::PARAMETER_OPEN, dsml::PARAMETER_CLOSE],
            _ => &[],
        };
        excludes.extend(inner.iter().map(|tag| tag.to_string()));
        let prefix = Grammar::Tag(Tag::new("", Grammar::AnyText { excludes }, think_close));
        Some(Grammar::Sequence(vec![prefix, calls]))
    }

    /// The JSON family: each call is the call markers around one JSON object whose `name` is the
    /// tool's and whose `arguments` the tool's schema accepts, the template's newlines included.
    fn json_calls(markers: &CallMarkers<'_>, tools: &[&Tool], at_least_one: bool) -> Grammar {
        let tags = tools
            .iter()
            .map(|tool| {
                let name = Value::String(tool.function.name.clone());
                Tag::new(
                    format!("{}\n{{\"name\": {name}, \"arguments\": ", markers.call_open),
                    Grammar::JsonSchema {
                        schema: tool.function.parameters.clone(),
                        style: None,
                    },
                    format!("}}\n{}", markers.call_close),
                )
            })
            .collect();
        Grammar::TriggeredTags {
            triggers: vec![markers.call_open.to_string()],
            tags,
            at_least_one,
        }
    }

    /// Keyed arguments in the compact spelling (GLM 4.7 and later, IQuest): each call is the call
    /// opener and the tool's name, the schema written in the engines' `glm_xml` style (`<arg_key>`
    /// and `<arg_value>` pairs with nothing between them), and the call's close; the shape
    /// xgrammar's built-in GLM tag has and the gateway writes today. Ling's spelling, with the
    /// template's newlines, is not this one, so it has no derivation yet.
    fn keyed_calls(markers: &CallMarkers<'_>, tools: &[&Tool], at_least_one: bool) -> Grammar {
        let tags = tools
            .iter()
            .map(|tool| {
                Tag::new(
                    format!("{}{}", markers.call_open, tool.function.name),
                    Grammar::JsonSchema {
                        schema: tool.function.parameters.clone(),
                        style: Some("glm_xml".to_string()),
                    },
                    markers.call_close,
                )
            })
            .collect();
        Grammar::TriggeredTags {
            triggers: vec![markers.call_open.to_string()],
            tags,
            at_least_one,
        }
    }

    /// DSML: the calls block, inside it one invoke per call, inside an invoke any number of
    /// parameter tags whose `string` attribute types the value (`true`: any text up to a closing
    /// tag; `false`: any JSON value), with the template's newlines; the shape the gateway writes
    /// for DeepSeek V4.1 today. The block follows the template's two newlines.
    fn dsml_calls(
        markers: &CallMarkers<'_>,
        tools: &[&Tool],
        at_least_one: bool,
    ) -> Option<Grammar> {
        let (block_open, block_close) = markers.block?;
        let parameter_close = dsml::PARAMETER_CLOSE;
        let parameter = Grammar::Tag(Tag::new(
            dsml::PARAMETER_OPEN,
            Grammar::Sequence(vec![
                Grammar::Regex("[^\"]+".to_string()),
                Grammar::ConstString("\" string=\"".to_string()),
                Grammar::Or(vec![
                    Grammar::Sequence(vec![
                        Grammar::ConstString("true\">".to_string()),
                        Grammar::AnyText {
                            excludes: vec![
                                parameter_close.to_string(),
                                markers.call_close.to_string(),
                                block_close.to_string(),
                            ],
                        },
                    ]),
                    Grammar::Sequence(vec![
                        Grammar::ConstString("false\">".to_string()),
                        Grammar::JsonSchema {
                            schema: Value::Bool(true),
                            style: None,
                        },
                    ]),
                ]),
            ]),
            format!("{parameter_close}\n"),
        ));
        let invokes = tools
            .iter()
            .map(|tool| {
                Tag::new(
                    format!("{}{}\">\n", markers.call_open, tool.function.name),
                    Grammar::Star(Box::new(parameter.clone())),
                    format!("{}\n", markers.call_close),
                )
            })
            .collect();
        Some(Grammar::Sequence(vec![
            Grammar::ConstString(format!("\n\n{block_open}\n")),
            Grammar::TagsWithSeparator {
                tags: invokes,
                separator: String::new(),
                at_least_one,
            },
            Grammar::ConstString(block_close.to_string()),
        ]))
    }

    /// The call markers the rows give: the terminal from the wrapper state, or without one from
    /// the content state, into the arguments state, and the one back; and, when the calls sit in
    /// a block, the terminals from the content state into the wrapper state and back.
    fn call_markers(&self) -> Option<CallMarkers<'_>> {
        let call_open = self
            .marker(Emits::Wrapper, Emits::Arguments)
            .or_else(|| self.marker(Emits::Content, Emits::Arguments))?;
        let call_close = self
            .marker(Emits::Arguments, Emits::Wrapper)
            .or_else(|| self.marker(Emits::Arguments, Emits::Content))?;
        let block = self
            .marker(Emits::Content, Emits::Wrapper)
            .zip(self.marker(Emits::Wrapper, Emits::Content));
        Some(CallMarkers {
            block,
            call_open,
            call_close,
        })
    }

    /// The text of the terminal that moves the engine from a state emitting `from` to one
    /// emitting `to`: the two states name the marker, not the order of the rows.
    fn marker(&self, from: Emits, to: Emits) -> Option<&str> {
        self.transitions()
            .find(|&(source, _, target)| self.emits(source) == from && self.emits(target) == to)
            .map(|(_, on, _)| self.terminal_text(on))
    }

    /// Whether a row of the table enters or leaves a state emitting `emits`.
    fn has_state(&self, emits: Emits) -> bool {
        self.transitions()
            .any(|(source, _, target)| self.emits(source) == emits || self.emits(target) == emits)
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;
    use serde_json::json as value;

    use super::*;
    use crate::formats;

    fn tool(name: &str, strict: Option<bool>, parameters: Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
                strict,
            },
        }
    }

    fn weather_tools() -> Vec<Tool> {
        let schema = value!({"type": "object", "properties": {"city": {"type": "string"}}});
        vec![
            tool("get_weather", None, schema.clone()),
            tool("lookup", Some(false), schema.clone()),
            tool("ping", None, value!({})),
            tool("", None, schema),
        ]
    }

    #[test]
    fn glm_derives_the_tag_the_gateway_hand_writes_today() {
        // The old crate's `Glm4MoeParser::build_structural_tag`, which mirrors xgrammar's built-in
        // `glm_4_7` tag: one trigger, a `glm_xml`-styled schema per named tool as written (a
        // no-argument tool's `{}` included), `at_least_one` as asked.
        let schema = value!({"type": "object", "properties": {"city": {"type": "string"}}});
        let grammar = formats::glm()
            .grammar(&weather_tools(), true, false)
            .expect("a grammar");
        let per_tool = |name: &str, schema: &Value| {
            value!({
                "type": "tag",
                "begin": format!("<tool_call>{name}"),
                "content": {"type": "json_schema", "json_schema": schema, "style": "glm_xml"},
                "end": "</tool_call>",
            })
        };
        assert_eq!(
            grammar.payload(),
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": [
                    per_tool("get_weather", &schema),
                    per_tool("lookup", &schema),
                    per_tool("ping", &value!({})),
                ],
                "at_least_one": true,
            }})
        );
        let optional = formats::glm()
            .grammar(&weather_tools(), false, false)
            .expect("a grammar");
        assert_eq!(optional.payload()["format"]["at_least_one"], false);
    }

    #[test]
    fn the_reasoning_prefix_wraps_the_tag_as_the_gateway_does() {
        // `wrap_in_reasoning_prefix(tag, Glm4MoeParser::reasoning_prefix())`: a sequence of free
        // text that writes none of the markers, closed by `</think>`, then the tag.
        let grammar = formats::glm()
            .grammar(&weather_tools(), true, true)
            .expect("a grammar");
        let payload = grammar.payload();
        assert_eq!(payload["format"]["type"], "sequence");
        let elements = payload["format"]["elements"]
            .as_array()
            .expect("two elements");
        assert_eq!(
            elements[0],
            value!({
                "type": "tag",
                "begin": "",
                "content": {"type": "any_text", "excludes": [
                    "<think>", "</think>", "<tool_call>", "</tool_call>",
                    "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
                ]},
                "end": "</think>",
            })
        );
        assert_eq!(elements[1]["type"], "triggered_tags");
    }

    #[test]
    fn iquest_takes_the_compact_tag_under_its_own_markers_and_ling_none() {
        let tools = weather_tools();
        let payload = formats::iquest()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        assert_eq!(
            payload["format"]["triggers"],
            value!(["<iquest_tool_call>"])
        );
        assert_eq!(
            payload["format"]["tags"][0]["begin"],
            "<iquest_tool_call>get_weather"
        );
        assert_eq!(payload["format"]["tags"][0]["content"]["style"], "glm_xml");
        assert_eq!(payload["format"]["tags"][0]["end"], "</iquest_tool_call>");
        // Ling writes the template's newlines between the tags, which the compact style does not,
        // so its table says so and gets no tag until that spelling has a derivation.
        assert_eq!(keyed::Tags::LING.between, "\n");
        assert!(formats::ling().grammar(&tools, true, false).is_none());
        assert!(formats::ling().grammar(&tools, true, true).is_none());
    }

    #[test]
    fn the_prefix_excludes_each_tables_own_terminals_and_inner_tags() {
        // No old builder wraps these two, so the derived shape is pinned here: the prefix's
        // excludes are the table's terminals in order and the syntax's inner tags, its end the
        // thought's close back to content, and the second element the tag as derived without it.
        let tools = weather_tools();
        for (format, excludes) in [
            (
                formats::deepseek_v4_1(),
                value!([
                    "<think>",
                    "</think>",
                    "<｜DSML｜ calls>",
                    "</｜DSML｜ calls>",
                    "<｜DSML｜ invoke name=\"",
                    "</｜DSML｜ invoke>",
                    "<｜DSML｜ parameter name=\"",
                    "</｜DSML｜ parameter>",
                ]),
            ),
            (
                formats::qwen3(CallSyntax::Json),
                value!(["<think>", "</think>", "<tool_call>", "</tool_call>"]),
            ),
        ] {
            let wrapped = format
                .grammar(&tools, true, true)
                .expect("a grammar")
                .to_json();
            let bare = format
                .grammar(&tools, true, false)
                .expect("a grammar")
                .to_json();
            assert_eq!(wrapped["type"], "sequence", "{}", format.name());
            assert_eq!(
                wrapped["elements"][0],
                value!({
                    "type": "tag",
                    "begin": "",
                    "content": {"type": "any_text", "excludes": excludes},
                    "end": "</think>",
                }),
                "{}",
                format.name()
            );
            assert_eq!(wrapped["elements"][1], bare, "{}", format.name());
        }
    }

    #[test]
    fn a_table_without_a_thought_has_no_prefix_to_wrap() {
        // Qwen 2.5 has no reasoning state: a prompt cannot leave the model inside a thought, and
        // the tag comes back as it is, as the old registry's does when a parser has no prefix.
        let tools = weather_tools();
        let bare = formats::qwen2_5().grammar(&tools, true, false);
        assert!(bare.is_some());
        assert_eq!(formats::qwen2_5().grammar(&tools, true, true), bare);
    }

    #[test]
    fn a_thought_that_closes_somewhere_other_than_content_has_no_prefix_yet() {
        // A thought that closes into the calls block, as Kimi K3's closes into its message: the
        // tag derives without the prefix, but a prefix ending at the block's opener would force
        // the call inside the thought, so with the reasoning open the table gives `None`.
        let into_the_block = Format::new("into_the_block")
            .terminal("think_open", "<think>")
            .terminal("think_close", "</think>")
            .terminal("calls_open", "<calls>")
            .terminal("calls_close", "</calls>")
            .terminal("call_open", "<tool_call>")
            .terminal("call_close", "</tool_call>")
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("calls", Emits::Wrapper)
            .state("call", Emits::Arguments)
            .transition("content", "think_open", "reasoning")
            .transition("reasoning", "think_close", "calls")
            .transition("content", "calls_open", "calls")
            .transition("calls", "call_open", "call")
            .transition("call", "call_close", "calls")
            .transition("calls", "calls_close", "content")
            .calls(CallSyntax::Json);
        let tools = weather_tools();
        assert!(into_the_block.grammar(&tools, true, false).is_some());
        assert!(into_the_block.grammar(&tools, true, true).is_none());
    }

    #[test]
    fn the_markers_come_from_the_states_not_from_the_order_of_the_rows() {
        // DeepSeek V4.1's rows with the reasoning state's two exits swapped and the invoke's two
        // exits swapped: the thought still closes back to content, and the call still closes back
        // to the block.
        let swapped = Format::new("swapped")
            .terminal("think_open", "<think>")
            .terminal("think_close", "</think>")
            .terminal("calls_open", "<｜DSML｜ calls>")
            .terminal("calls_close", "</｜DSML｜ calls>")
            .terminal("invoke_open", "<｜DSML｜ invoke name=\"")
            .terminal("invoke_close", dsml::INVOKE_CLOSE)
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("calls", Emits::Wrapper)
            .state("invoke", Emits::Arguments)
            .transition("content", "think_open", "reasoning")
            .transition("reasoning", "calls_open", "calls")
            .transition("reasoning", "think_close", "content")
            .transition("content", "calls_open", "calls")
            .transition("calls", "invoke_open", "invoke")
            .transition("invoke", "calls_close", "content")
            .transition("invoke", "invoke_close", "calls")
            .transition("invoke", "invoke_open", "invoke")
            .transition("calls", "calls_close", "content")
            .calls(CallSyntax::Dsml);
        let tools = weather_tools();
        assert_eq!(
            swapped.grammar(&tools, true, true),
            formats::deepseek_v4_1().grammar(&tools, true, true)
        );
    }

    #[test]
    fn dsml_derives_the_v41_tag_the_gateway_hand_writes_today() {
        // The old crate's `DeepSeekDsmlParser::build_v41_structural_tag`: the template's two
        // newlines, the calls block, one invoke tag per named tool holding any number of parameter
        // tags, each typed by its `string` attribute, every tag closed with the template's newline.
        let tools = vec![tool(
            "get_weather",
            None,
            value!({"type": "object", "properties": {"city": {"type": "string"}}}),
        )];
        let grammar = formats::deepseek_v4_1()
            .grammar(&tools, true, false)
            .expect("a grammar");
        let parameter = value!({
            "type": "tag",
            "begin": "<｜DSML｜ parameter name=\"",
            "content": {
                "type": "sequence",
                "elements": [
                    {"type": "regex", "pattern": "[^\"]+"},
                    {"type": "const_string", "value": "\" string=\""},
                    {"type": "or", "elements": [
                        {"type": "sequence", "elements": [
                            {"type": "const_string", "value": "true\">"},
                            {"type": "any_text", "excludes": [
                                "</｜DSML｜ parameter>",
                                "</｜DSML｜ invoke>",
                                "</｜DSML｜ calls>",
                            ]},
                        ]},
                        {"type": "sequence", "elements": [
                            {"type": "const_string", "value": "false\">"},
                            {"type": "json_schema", "json_schema": true},
                        ]},
                    ]},
                ],
            },
            "end": "</｜DSML｜ parameter>\n",
        });
        assert_eq!(
            grammar.payload(),
            value!({"format": {
                "type": "sequence",
                "elements": [
                    {"type": "const_string", "value": "\n\n<｜DSML｜ calls>\n"},
                    {
                        "type": "tags_with_separator",
                        "tags": [{
                            "type": "tag",
                            "begin": "<｜DSML｜ invoke name=\"get_weather\">\n",
                            "content": {"type": "star", "content": parameter},
                            "end": "</｜DSML｜ invoke>\n",
                        }],
                        "separator": "",
                        "at_least_one": true,
                    },
                    {"type": "const_string", "value": "</｜DSML｜ calls>"},
                ],
            }})
        );
    }

    #[test]
    fn the_json_family_wraps_each_tools_schema_in_the_call_markers() {
        let tools = weather_tools();
        for format in [formats::qwen3(CallSyntax::Json), formats::qwen2_5()] {
            let grammar = format.grammar(&tools, true, false).expect("a grammar");
            let payload = grammar.payload();
            assert_eq!(
                payload["format"]["type"],
                "triggered_tags",
                "{}",
                format.name()
            );
            assert_eq!(payload["format"]["triggers"], value!(["<tool_call>"]));
            let tags = payload["format"]["tags"].as_array().expect("tags");
            assert_eq!(tags.len(), 3, "a nameless tool has no tag");
            assert_eq!(
                tags[0]["begin"],
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
            );
            assert_eq!(tags[0]["end"], "}\n</tool_call>");
            assert_eq!(tags[0]["content"]["type"], "json_schema");
            assert!(tags[0]["content"].get("style").is_none());
            assert_eq!(tags[2]["content"]["json_schema"], value!({}));
        }
    }

    #[test]
    fn no_named_tool_and_no_derivation_yet_give_none() {
        let tools = weather_tools();
        assert!(formats::glm().grammar(&[], true, false).is_none());
        assert!(formats::glm()
            .grammar(&[tool("", None, value!({}))], true, false)
            .is_none());
        for format in [
            formats::qwen3(CallSyntax::Tagged),
            formats::seed_oss(),
            formats::ling(),
            formats::hy4(),
            formats::minimax_m3(),
            formats::kimi_k3(),
            formats::olmo3(),
            formats::lfm2_5(),
            formats::xlam(),
            formats::plain(),
        ] {
            assert!(
                format.grammar(&tools, true, false).is_none(),
                "{}",
                format.name()
            );
        }
    }

    #[test]
    fn every_grammar_node_writes_the_engines_keys() {
        let leaf = Grammar::Sequence(vec![
            Grammar::ConstString("a".to_string()),
            Grammar::AnyText {
                excludes: vec!["<".to_string()],
            },
            Grammar::Regex("[0-9]+".to_string()),
            Grammar::Or(vec![
                Grammar::Optional(Box::new(Grammar::ConstString("b".to_string()))),
                Grammar::Star(Box::new(Grammar::ConstString("c".to_string()))),
                Grammar::Plus(Box::new(Grammar::ConstString("d".to_string()))),
            ]),
        ]);
        assert_eq!(
            leaf.to_json(),
            value!({"type": "sequence", "elements": [
                {"type": "const_string", "value": "a"},
                {"type": "any_text", "excludes": ["<"]},
                {"type": "regex", "pattern": "[0-9]+"},
                {"type": "or", "elements": [
                    {"type": "optional", "content": {"type": "const_string", "value": "b"}},
                    {"type": "star", "content": {"type": "const_string", "value": "c"}},
                    {"type": "plus", "content": {"type": "const_string", "value": "d"}},
                ]},
            ]})
        );
    }
}
