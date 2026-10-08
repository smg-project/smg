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
//! write none of the table's markers, closed by the thought's close, so the forced call follows
//! the reasoning instead of replacing it, as the gateway's `wrap_in_reasoning_prefix` does.
//!
//! Not derived yet, so [`Format::grammar`] gives `None` for them: the tagged syntax (Qwen 3.5 and
//! later, Seed-OSS), keyed arguments in another spelling than GLM's (Hy4), MiniMax M3's XML tree,
//! Kimi K3's XTML, the pythonic tables, xLAM's list, and a table without calls. The gateway then
//! takes the JSON-schema path, as it does for those models today.

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
    /// inside its thought, and the tag follows a reasoning prefix the thought's close ends.
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
        let think_close =
            self.marker(|from| from == Emits::Reasoning, |to| to != Emits::Reasoning)?;
        let mut excludes: Vec<String> = self.terminal_texts().map(str::to_string).collect();
        if let Some(CallSyntax::Keyed(tags)) = self.call_syntax() {
            excludes.extend(
                [
                    tags.key_open,
                    tags.key_close,
                    tags.value_open,
                    tags.value_close,
                ]
                .into_iter()
                .map(str::to_string),
            );
        }
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

    /// Keyed arguments in GLM's spelling: each call is the call opener and the tool's name, the
    /// schema written in the engines' `glm_xml` style (`<arg_key>` and `<arg_value>` pairs), and
    /// the call's close; the shape xgrammar's built-in GLM tag has and the gateway writes today.
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

    /// The call markers the rows give: the terminal that enters the arguments state from outside
    /// it and the one that leaves it, and the terminals around a wrapper state when there is one.
    fn call_markers(&self) -> Option<CallMarkers<'_>> {
        let in_call = |emits: Emits| matches!(emits, Emits::Arguments | Emits::Wrapper);
        let call_open =
            self.marker(|from| from != Emits::Arguments, |to| to == Emits::Arguments)?;
        let call_close =
            self.marker(|from| from == Emits::Arguments, |to| to != Emits::Arguments)?;
        let block = self
            .marker(|from| !in_call(from), |to| to == Emits::Wrapper)
            .zip(self.marker(|from| from == Emits::Wrapper, |to| !in_call(to)));
        Some(CallMarkers {
            block,
            call_open,
            call_close,
        })
    }

    /// The text of the first terminal that moves the engine from a state whose emission `from`
    /// accepts to one whose emission `to` accepts.
    fn marker(&self, from: impl Fn(Emits) -> bool, to: impl Fn(Emits) -> bool) -> Option<&str> {
        self.transitions()
            .find(|&(source, _, target)| from(self.emits(source)) && to(self.emits(target)))
            .map(|(_, on, _)| self.terminal_text(on))
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
        // Ling has the same rows under its own opener, so the same tag comes of it.
        assert_eq!(
            formats::ling().grammar(&weather_tools(), true, true),
            formats::glm().grammar(&weather_tools(), true, true)
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
