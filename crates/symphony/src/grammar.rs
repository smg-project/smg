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
//! [`Format::grammar`] derives the tree from the table. A region of the table (a call, a block of
//! calls, the thought, the answer) is opened by the terminal of a row into its state and closed
//! by the terminal of a row back to a state it is entered from; the terminal every state has a
//! row for is the end of the turn, which ends whatever is open and closes no region of its own.
//! So the call markers are the rows into and out of the arguments state, the block markers the
//! rows into and out of the wrapper state the calls sit in, and the thought's close the row from
//! the reasoning state back to where the thought opened; the request's tools give each call its
//! name and schema. What the syntax inside the markers constrains comes from the call syntax: a
//! JSON object around each tool's schema for the JSON family; the engines' `glm_xml` style of a
//! schema for keyed arguments in the compact spelling, and the table's own tags written one pair
//! per property for Ling's newlines and Hy4's suffixed tags (the `keyed` module); `<function=`
//! and `<parameter=` tags, one per property, for Qwen 3.5's and Seed-OSS's tagged syntax (the
//! `tagged` module); a tag grammar per parameter for DSML; for Kimi K3's XTML, one argument tag
//! per property of the schema (the `xtml` module). The gateway's
//! old parsers wrote these tags by hand, one function per parser; where one exists for a format
//! Symphony has, the derived tag is pinned to it in the tests, so the switch changes nothing an
//! engine sees.
//!
//! With the prompt's reasoning still open, the tag is wrapped in a prefix: free text that can
//! write none of the table's markers nor the call syntax's inner tags, closed by the thought's
//! close, so the forced call follows the reasoning instead of replacing it, as the gateway's
//! `wrap_in_reasoning_prefix` does. A table without a thought has nothing to wrap, and the tag
//! comes back as it is; a thought that does not close back to where it opened has no prefix yet,
//! and the table gives `None` rather than a tag the model could write inside its thought. Kimi
//! K3's tag takes no prefix: the block of calls can be entered from the turn itself, from the
//! thought or from the answer, and the tag has one way in per state: from a region with a close
//! of its own, that close and then the row into the block from where the close returns; from any
//! other state, its own row into the block. So the model closes whatever the prompt left open on
//! its way to the calls, whichever that was.
//!
//! Not derived yet, so [`Format::grammar`] gives `None` for them: MiniMax M3's XML tree, the
//! pythonic tables, xLAM's list, and a table without calls. The gateway then takes the JSON-schema
//! path, as it does for those models today.

use openai_protocol::common::Tool;
use serde_json::{json, Value};

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::{assembler::TAGS as TAGGED_TAGS, dsml, keyed::Tags as KeyedTags},
};

mod keyed;
mod schema;
mod tagged;
mod xtml;

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

/// The markers a table's rows give a call: the terminals into and out of the arguments state, and
/// the block's own when the calls sit in a block.
struct CallMarkers<'a> {
    block: Option<Block<'a>>,
    call_open: &'a str,
    call_close: &'a str,
    /// The terminals the table has a row for out of the arguments state, in the order of the
    /// rows, each once: what the engine takes wherever it stands inside a call, a value included.
    call_exits: Vec<&'a str>,
}

/// A block of calls: its wrapper state, and the terminals into and out of it.
struct Block<'a> {
    state: usize,
    open: &'a str,
    close: &'a str,
}

/// One way into a block of calls, from a state with a row into it: from a region with a close of
/// its own (a thought, an answer) whose home has a row into the block, that close first, then the
/// home's row, so the model closes what the prompt left open and goes on from where that leaves
/// it; from any other state (the turn's own, one with no close, one whose home has no row into
/// the block), the opener of its own row. Every way in is a path the table has.
struct WayIn<'a> {
    close: Option<&'a str>,
    open: &'a str,
}

impl WayIn<'_> {
    /// What the model writes to enter the block this way.
    fn begin(&self) -> String {
        format!("{}{}", self.close.unwrap_or(""), self.open)
    }

    /// The text that triggers this way in: the close when there is one, else the opener.
    fn trigger(&self) -> &str {
        self.close.unwrap_or(self.open)
    }
}

/// A block of calls as the engine takes it: one tag per way in, each `content` between the way's
/// text and the block's close, triggered by each way's first text, the same trigger once.
fn block_of(ways_in: &[WayIn<'_>], content: Grammar, close: &str, at_least_one: bool) -> Grammar {
    let tags = ways_in
        .iter()
        .map(|way| Tag::new(way.begin(), content.clone(), close))
        .collect();
    let mut triggers: Vec<String> = Vec::new();
    for trigger in ways_in.iter().map(WayIn::trigger) {
        if !triggers.iter().any(|known| known == trigger) {
            triggers.push(trigger.to_string());
        }
    }
    Grammar::TriggeredTags {
        triggers,
        tags,
        at_least_one,
    }
}

/// A region's two markers, the terminal of the row into its state and of the row back out, and
/// the state the row back out returns to.
struct Region<'a> {
    open: &'a str,
    close: &'a str,
    home: usize,
}

impl Format {
    /// The structural tag for a forced or required call to one of `tools`, or `None` when the
    /// table's call syntax has no derivation yet, the table has no calls, or no tool has a name.
    /// `at_least_one` asks for at least one call (`tool_choice` `required`, a named function, or
    /// an allowed-tools list in `required` mode). With `reasoning_open`, the prompt left the model
    /// inside its thought, and the tag follows a reasoning prefix the thought's close ends; a
    /// table without a thought has no prefix, and the tag comes back as it is; a table whose
    /// thought does not close back to where it opened has no prefix yet, and gives `None`. Kimi
    /// K3's tag has a way in per state its block is entered from, the state's close first when it
    /// has one, so it is the same either way.
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
            CallSyntax::Keyed(tags) if *tags == KeyedTags::PLAIN => {
                Self::keyed_calls(&markers, &named, at_least_one)
            }
            CallSyntax::Dsml => Self::dsml_calls(&markers, &named, at_least_one)?,
            CallSyntax::Keyed(tags) => {
                let ways_in = self.ways_into_block(&markers);
                keyed::calls(&markers, &ways_in, tags, &named, at_least_one)
            }
            CallSyntax::Xtml => {
                let ways_in = self.ways_into(markers.block.as_ref()?.state);
                return xtml::calls(&markers, &ways_in, &named, at_least_one);
            }
            CallSyntax::Tagged(spelling, _) => {
                let ways_in = self.ways_into_block(&markers);
                tagged::calls(&markers, &ways_in, *spelling, &named, at_least_one)
            }
            CallSyntax::Pythonic | CallSyntax::JsonList | CallSyntax::Xml => return None,
        };
        if !reasoning_open {
            return Some(calls);
        }
        let Some(reasoning) = self.state_emitting(Emits::Reasoning) else {
            return Some(calls);
        };
        let think_close = self
            .region(reasoning, |state| self.is_call_state(state))?
            .close;
        let mut excludes: Vec<String> = self.terminal_texts().map(str::to_string).collect();
        let inner: &[&str] = match self.call_syntax() {
            Some(CallSyntax::Keyed(tags)) => &[
                tags.key_open,
                tags.key_close,
                tags.value_open,
                tags.value_close,
            ],
            Some(CallSyntax::Dsml) => &[dsml::PARAMETER_OPEN, dsml::PARAMETER_CLOSE],
            Some(CallSyntax::Tagged(..)) => &TAGGED_TAGS,
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
    /// xgrammar's built-in GLM tag has and the gateway writes today. Ling's and Hy4's spellings
    /// are not this one, and are written from their own tags instead (the `keyed` module).
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
        let block = markers.block.as_ref()?;
        let (block_open, block_close) = (block.open, block.close);
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

    /// The call markers the rows give: the region of the arguments state, and of the wrapper state
    /// with a row into it when the calls sit in a block.
    fn call_markers(&self) -> Option<CallMarkers<'_>> {
        let arguments = self.state_emitting(Emits::Arguments)?;
        let is_arguments = |state: usize| self.emits(state) == Emits::Arguments;
        let call = self.region(arguments, |_| false)?;
        let block = self
            .transitions()
            .find(|&(source, _, target)| {
                target == arguments && self.emits(source) == Emits::Wrapper
            })
            .map(|(source, _, _)| source);
        let block = match block {
            Some(state) => {
                let Region { open, close, .. } = self.region(state, is_arguments)?;
                Some(Block { state, open, close })
            }
            None => None,
        };
        let mut call_exits: Vec<&str> = Vec::new();
        for (_, on, _) in self
            .transitions()
            .filter(|&(source, _, _)| source == arguments)
        {
            let text = self.terminal_text(on);
            if !call_exits.contains(&text) {
                call_exits.push(text);
            }
        }
        Some(CallMarkers {
            block,
            call_open: call.open,
            call_close: call.close,
            call_exits,
        })
    }

    /// The ways into the block the calls sit in, or none when they sit in no block.
    fn ways_into_block(&self, markers: &CallMarkers<'_>) -> Vec<WayIn<'_>> {
        markers
            .block
            .as_ref()
            .map(|block| self.ways_into(block.state))
            .unwrap_or_default()
    }

    /// The first state emitting `emits`.
    fn state_emitting(&self, emits: Emits) -> Option<usize> {
        (0..self.states()).find(|&state| self.emits(state) == emits)
    }

    /// The states with a row into `state`, other than itself and the states `within` it, in the
    /// order of the states: the states the region is entered from.
    fn entered_from(&self, state: usize, within: impl Fn(usize) -> bool) -> Vec<usize> {
        let mut from: Vec<usize> = self
            .transitions()
            .filter(|&(source, _, target)| target == state && source != state && !within(source))
            .map(|(source, _, _)| source)
            .collect();
        from.sort_unstable();
        from.dedup();
        from
    }

    /// The markers of the region `state` is: its close is the terminal of the row from `state`
    /// back to a state it is entered from, other than the end of the turn, and its opener the
    /// terminal of that state's row into `state`; that state is its home. The states name the
    /// markers, not the order of the rows. The turn's own state, where the opener leaves the
    /// engine, is no region: the rows back into it close the regions it holds.
    fn region(&self, state: usize, within: impl Fn(usize) -> bool + Copy) -> Option<Region<'_>> {
        if state == 0 {
            return None;
        }
        let from = self.entered_from(state, within);
        let turn_close = self.turn_close();
        let (_, close, home) = self.transitions().find(|&(source, on, target)| {
            source == state && from.contains(&target) && Some(on) != turn_close
        })?;
        let (_, open, _) = self
            .transitions()
            .find(|&(source, _, target)| source == home && target == state)?;
        Some(Region {
            open: self.terminal_text(open),
            close: self.terminal_text(close),
            home,
        })
    }

    /// The ways into the block `state` is, one per state it is entered from, in the order of the
    /// states, the same way in once: from a region with a close whose home has a row into the
    /// block, the close and then that row; from any other state, its own row.
    fn ways_into(&self, state: usize) -> Vec<WayIn<'_>> {
        let door = |from: usize| {
            self.transitions()
                .find(|&(source, _, target)| source == from && target == state)
                .map(|(_, on, _)| self.terminal_text(on))
        };
        let mut ways: Vec<WayIn<'_>> = Vec::new();
        for from in self.entered_from(state, |state| self.is_arguments(state)) {
            let through_home = self
                .region(from, |state| self.is_call_state(state))
                .and_then(|Region { close, home, .. }| Some((Some(close), door(home)?)));
            let (close, open) = match through_home {
                Some((close, open)) => (close, Some(open)),
                None => (None, door(from)),
            };
            let Some(open) = open else { continue };
            let way = WayIn { close, open };
            if !ways.iter().any(|known| known.begin() == way.begin()) {
                ways.push(way);
            }
        }
        ways
    }

    /// Whether `state` emits a call's arguments.
    fn is_arguments(&self, state: usize) -> bool {
        self.emits(state) == Emits::Arguments
    }

    /// Whether `state` is one of the calls': a call's arguments, or the block the calls sit in,
    /// the wrapper state with a row into an arguments state. A thought or an answer is a region
    /// outside them, so a row into the calls is no close of its own.
    fn is_call_state(&self, state: usize) -> bool {
        self.is_arguments(state)
            || self.emits(state) == Emits::Wrapper
                && self
                    .transitions()
                    .any(|(source, _, target)| source == state && self.is_arguments(target))
    }

    /// The terminal that every state has a row for, each back to the turn's own state: the end of
    /// the turn, which ends whatever is open and closes no region of its own.
    fn turn_close(&self) -> Option<usize> {
        let terminals = self.terminal_texts().count();
        (0..terminals).find(|&on| {
            (0..self.states()).all(|state| {
                self.transitions()
                    .any(|(source, t, target)| source == state && t == on && target == 0)
            })
        })
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
    fn iquest_takes_the_compact_tag_under_its_own_markers() {
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
        // so its table says so and its tag is written from its own tags (the `keyed` module).
        assert_eq!(KeyedTags::LING.between, "\n");
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
    fn a_thought_that_closes_into_the_block_has_no_prefix_but_its_own_door() {
        // A thought that closes into the calls block and nowhere else has no close of its own, so
        // with the reasoning open the prefix path gives `None` rather than a tag the model could
        // write inside its thought; without it the tag derives, and under XTML the thought's way
        // in is its own row into the block, `</think>`, since that is a path the table has.
        let into_the_block = |syntax: CallSyntax| {
            Format::new("into_the_block")
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
                .calls(syntax)
        };
        let tools = weather_tools();
        assert!(into_the_block(CallSyntax::Json)
            .grammar(&tools, true, false)
            .is_some());
        assert!(into_the_block(CallSyntax::Json)
            .grammar(&tools, true, true)
            .is_none());
        let as_xtml = into_the_block(CallSyntax::Xtml)
            .grammar(&tools, true, true)
            .expect("a grammar")
            .payload();
        assert_eq!(
            as_xtml["format"]["triggers"],
            value!(["<calls>", "</think>"])
        );
        assert_eq!(begins(&as_xtml), ["<calls>", "</think>"]);
    }

    #[test]
    fn a_thought_whose_home_has_no_row_into_the_block_enters_by_its_own_row() {
        // The block is reachable from inside the thought alone and closes back into it, the shape
        // of a model that calls tools mid-thought. The thought has a close, but content, where the
        // close returns, has no row into the block, so `</think><calls>` would be no path the
        // table has: the way in is the thought's own row, `<calls>`, with the thought left open.
        let mid_thought = Format::new("mid_thought")
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
            .transition("reasoning", "think_close", "content")
            .transition("reasoning", "calls_open", "calls")
            .transition("calls", "call_open", "call")
            .transition("call", "call_close", "calls")
            .transition("calls", "calls_close", "reasoning")
            .calls(CallSyntax::Xtml);
        let payload = mid_thought
            .grammar(&weather_tools(), true, true)
            .expect("a grammar")
            .payload();
        assert_eq!(payload["format"]["triggers"], value!(["<calls>"]));
        assert_eq!(begins(&payload), ["<calls>"]);
        assert_eq!(payload["format"]["tags"][0]["end"], "</calls>");
    }

    #[test]
    fn a_state_the_calls_return_to_enters_the_block_by_its_own_row() {
        // MiniMax M3's table under XTML: the block is entered from the turn's start, from the
        // thought and from the text after it, which the block's close returns to. None of the
        // three is a region with a close of its own (the text's row into the block is no close,
        // the block being one of the calls' states), so each enters by its own row, and the one
        // row they share makes one way in.
        let tools = weather_tools();
        let as_xtml = formats::minimax_m3()
            .calls(CallSyntax::Xtml)
            .grammar(&tools, true, true)
            .expect("a grammar")
            .payload();
        assert_eq!(as_xtml["format"]["triggers"], value!(["<tool_call>"]));
        assert_eq!(begins(&as_xtml), ["<tool_call>"]);
        assert_eq!(as_xtml["format"]["tags"][0]["end"], "</tool_call>");
    }

    /// The `begin` of each way into a block, from a `triggered_tags` payload.
    fn begins(payload: &Value) -> Vec<String> {
        payload["format"]["tags"]
            .as_array()
            .expect("tags")
            .iter()
            .map(|tag| tag["begin"].as_str().expect("a begin").to_string())
            .collect()
    }

    #[test]
    fn a_blocks_opener_is_the_row_from_the_state_its_close_returns_to() {
        // The block is entered by two terminals, `<calls-from-thought>` from the thought (the
        // first row in) and `<calls>` from content; its close returns to content, so `<calls>` is
        // its opener, whichever row comes first, and DSML writes it as the block's opening. The
        // way in from the thought closes it, which leaves the engine in content, so it goes on
        // through content's door, `<calls>`, not the thought's own row.
        let two_doors = |syntax: CallSyntax| {
            Format::new("two_doors")
                .terminal("think_open", "<think>")
                .terminal("think_close", "</think>")
                .terminal("from_thought", "<calls-from-thought>")
                .terminal("calls_open", "<calls>")
                .terminal("calls_close", "</calls>")
                .terminal("call_open", "<tool_call>")
                .terminal("call_close", "</tool_call>")
                .state("content", Emits::Content)
                .state("reasoning", Emits::Reasoning)
                .state("calls", Emits::Wrapper)
                .state("call", Emits::Arguments)
                .transition("content", "think_open", "reasoning")
                .transition("reasoning", "think_close", "content")
                .transition("reasoning", "from_thought", "calls")
                .transition("content", "calls_open", "calls")
                .transition("calls", "call_open", "call")
                .transition("call", "call_close", "calls")
                .transition("calls", "calls_close", "content")
                .calls(syntax)
        };
        let tools = weather_tools();
        let as_dsml = two_doors(CallSyntax::Dsml)
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        assert_eq!(as_dsml["format"]["elements"][0]["value"], "\n\n<calls>\n");
        let as_xtml = two_doors(CallSyntax::Xtml)
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        assert_eq!(
            as_xtml["format"]["triggers"],
            value!(["<calls>", "</think>"])
        );
        assert_eq!(begins(&as_xtml), ["<calls>", "</think><calls>"]);
    }

    #[test]
    fn kimi_k3_derives_the_same_tag_whatever_the_order_of_its_rows() {
        // Kimi K3's rows in reverse: the end of the turn is still the terminal every state's row
        // sends back to the turn's state, the block's close is still the tools tag and not that
        // end, and the three ways in keep the order of the states.
        let rows = [
            ("message", "think_open", "reasoning"),
            ("reasoning", "think_close", "message"),
            ("reasoning", "response_open", "content"),
            ("reasoning", "tools_open", "tools"),
            ("reasoning", "message_close", "message"),
            ("message", "response_open", "content"),
            ("content", "response_close", "message"),
            ("content", "message_close", "message"),
            ("content", "tools_open", "tools"),
            ("message", "tools_open", "tools"),
            ("tools", "call_open", "call"),
            ("call", "call_close", "tools"),
            ("call", "call_open", "call"),
            ("call", "tools_close", "message"),
            ("call", "message_close", "message"),
            ("tools", "tools_close", "message"),
            ("tools", "message_close", "message"),
            ("message", "message_close", "message"),
        ];
        let mut reversed = Format::new("kimi_k3_reversed")
            .terminal("think_open", "<|open|>think<|sep|>")
            .terminal("think_close", "<|close|>think<|sep|>")
            .terminal("response_open", "<|open|>response<|sep|>")
            .terminal("response_close", "<|close|>response<|sep|>")
            .terminal("tools_open", "<|open|>tools<|sep|>")
            .terminal("tools_close", "<|close|>tools<|sep|>")
            .terminal("call_open", "<|open|>call tool=\"")
            .terminal("call_close", "<|close|>call<|sep|>")
            .terminal("message_close", "<|close|>message<|sep|>")
            .state("message", Emits::Wrapper)
            .state("reasoning", Emits::Reasoning)
            .state("content", Emits::Content)
            .state("tools", Emits::Wrapper)
            .state("call", Emits::Arguments);
        for (from, on, to) in rows.iter().rev() {
            reversed = reversed.transition(from, on, to);
        }
        let reversed = reversed.calls(CallSyntax::Xtml);
        let tools = weather_tools();
        assert_eq!(
            reversed.grammar(&tools, true, true),
            formats::kimi_k3().grammar(&tools, true, true)
        );
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
            formats::minimax_m3(),
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
    fn kimi_k3_derives_the_tag_the_gateway_hand_writes_today() {
        // The old crate's `KimiK3Parser::build_structural_tag`: one tag per way into the block
        // of calls (from the turn itself, after the thought's close, after the answer's close),
        // each the same `plus` of an `or` of the tools' calls, closed by the block's close, with
        // the opener and the two closes as triggers.
        let tools = weather_tools();
        let payload = formats::kimi_k3()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let format = &payload["format"];
        assert_eq!(format["type"], "triggered_tags");
        assert_eq!(
            format["triggers"],
            value!([
                "<|open|>tools<|sep|>",
                "<|close|>think<|sep|>",
                "<|close|>response<|sep|>"
            ])
        );
        assert_eq!(format["at_least_one"], true);
        let tags = format["tags"].as_array().expect("three ways in");
        assert_eq!(
            tags.iter().map(|tag| &tag["begin"]).collect::<Vec<_>>(),
            [
                "<|open|>tools<|sep|>",
                "<|close|>think<|sep|><|open|>tools<|sep|>",
                "<|close|>response<|sep|><|open|>tools<|sep|>",
            ]
        );
        for tag in tags {
            assert_eq!(tag["end"], "<|close|>tools<|sep|>");
            assert_eq!(tag["content"], tags[0]["content"]);
        }
        assert_eq!(tags[0]["content"]["type"], "plus");
        let calls = tags[0]["content"]["content"]["elements"]
            .as_array()
            .expect("one call per named tool");
        assert_eq!(calls.len(), 3, "a nameless tool has no call");
        assert_eq!(
            calls[0]["elements"][0]["value"],
            "<|open|>call tool=\"get_weather\" index=\""
        );
        // The prompt's open thought changes nothing: the tag closes it on its own way in.
        assert_eq!(
            formats::kimi_k3().grammar(&tools, false, true),
            formats::kimi_k3().grammar(&tools, false, false)
        );
        assert_eq!(
            formats::kimi_k3()
                .grammar(&tools, false, false)
                .expect("a grammar")
                .payload()["format"]["at_least_one"],
            false
        );
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
