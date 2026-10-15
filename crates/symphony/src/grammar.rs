//! The structural tag a table yields: the constraint an engine takes for a forced or required
//! tool call, built from the same terminals and call syntax the parser reads, so the constrained
//! output is exactly what the format parses.
//!
//! [`Grammar`] is Symphony's own tree, in the shape the engines' structural-tag format takes
//! (xgrammar's, which the serving engines accept): a `triggered_tags` of per-tool `tag`s, each a
//! `begin`, a content grammar and an `end`; `sequence`, `or`, `star`, `plus` and `optional` to
//! compose; `const_string`, `any_text`, `regex`, `json_schema` and `token` as leaves.
//! [`Grammar::to_json`] writes it as the engine reads it, and [`Grammar::payload`] wraps it as the
//! request carries it.
//!
//! [`Format::grammar`] derives the tree from the table's rows and the request's tools, one spelling
//! per call syntax. Notes on how, and on what it leaves open: `docs/grammar.md`.

use std::collections::VecDeque;

use openai_protocol::common::Tool;
use serde_json::{json, Value};

use crate::{
    format::{CallSyntax, Emits, Format},
    shape::{Layout, Message},
    tagged::{attributed::Typing, keyed::Tags as KeyedTags},
};

mod attributed;
mod gemma;
mod keyed;
mod pythonic;
mod schema;
mod shaped;
mod tagged;
mod xml;
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
    /// One token of the model's vocabulary, by its text: a marker written as that token, never
    /// spelled in others ([`Format::marker_tokens`]).
    Token(String),
    /// Free text until one of `triggers` opens a tag; with `at_least_one`, at least one tag, the
    /// first at once; with `stop_after_first`, nothing after the first tag.
    TriggeredTags {
        triggers: Vec<String>,
        tags: Vec<Tag>,
        at_least_one: bool,
        stop_after_first: bool,
    },
    /// Tags one after another with `separator` between them; with `at_least_one`, at least one;
    /// with `stop_after_first`, at most one.
    TagsWithSeparator {
        tags: Vec<Tag>,
        separator: String,
        at_least_one: bool,
        stop_after_first: bool,
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
            Self::Token(token) => json!({"type": "token", "token": token}),
            Self::TriggeredTags {
                triggers,
                tags,
                at_least_one,
                stop_after_first,
            } => stopping(
                json!({
                    "type": "triggered_tags",
                    "triggers": triggers,
                    "tags": tags.iter().map(Tag::to_json).collect::<Vec<_>>(),
                    "at_least_one": at_least_one,
                }),
                *stop_after_first,
            ),
            Self::TagsWithSeparator {
                tags,
                separator,
                at_least_one,
                stop_after_first,
            } => stopping(
                json!({
                    "type": "tags_with_separator",
                    "tags": tags.iter().map(Tag::to_json).collect::<Vec<_>>(),
                    "separator": separator,
                    "at_least_one": at_least_one,
                }),
                *stop_after_first,
            ),
        }
    }

    /// The grammar as a request carries it to the engine: under `format`.
    pub fn payload(&self) -> Value {
        json!({"format": self.to_json()})
    }

    /// The nodes in the tree, this one included: its size, for a derivation that bounds it. A
    /// JSON schema leaf counts by the size of its schema (a node per four of the schema's own, a
    /// long string weighing its length), since the definitions attached to it are copied with it.
    pub(crate) fn nodes(&self) -> usize {
        1 + match self {
            Self::ConstString(_) | Self::AnyText { .. } | Self::Regex(_) | Self::Token(_) => 0,
            Self::JsonSchema { schema, .. } => json_nodes(schema) / 4,
            Self::Sequence(elements) | Self::Or(elements) => elements.iter().map(Self::nodes).sum(),
            Self::Optional(content) | Self::Star(content) | Self::Plus(content) => content.nodes(),
            Self::Tag(tag) => tag.content.nodes(),
            Self::TriggeredTags { tags, .. } | Self::TagsWithSeparator { tags, .. } => {
                tags.iter().map(|tag| tag.content.nodes()).sum()
            }
        }
    }
}

/// `grammar` with each of `tokens`, marker texts the longest first, written as a token element
/// where it stands in a constant string; a tag keeps its begin and end, and a tag list its
/// separator.
fn bound(grammar: Grammar, tokens: &[String]) -> Grammar {
    let in_tag = |tag: Tag| Tag {
        content: Box::new(bound(*tag.content, tokens)),
        ..tag
    };
    let all = |elements: Vec<Grammar>| -> Vec<Grammar> {
        elements
            .into_iter()
            .map(|element| bound(element, tokens))
            .collect()
    };
    match grammar {
        Grammar::ConstString(text) => split_tokens(text, tokens),
        Grammar::Sequence(elements) => Grammar::Sequence(all(elements)),
        Grammar::Or(elements) => Grammar::Or(all(elements)),
        Grammar::Optional(content) => Grammar::Optional(Box::new(bound(*content, tokens))),
        Grammar::Star(content) => Grammar::Star(Box::new(bound(*content, tokens))),
        Grammar::Plus(content) => Grammar::Plus(Box::new(bound(*content, tokens))),
        Grammar::Tag(tag) => Grammar::Tag(in_tag(tag)),
        Grammar::TriggeredTags {
            triggers,
            tags,
            at_least_one,
            stop_after_first,
        } => Grammar::TriggeredTags {
            triggers,
            tags: tags.into_iter().map(in_tag).collect(),
            at_least_one,
            stop_after_first,
        },
        Grammar::TagsWithSeparator {
            tags,
            separator,
            at_least_one,
            stop_after_first,
        } => Grammar::TagsWithSeparator {
            tags: tags.into_iter().map(in_tag).collect(),
            separator,
            at_least_one,
            stop_after_first,
        },
        leaf @ (Grammar::AnyText { .. }
        | Grammar::Regex(_)
        | Grammar::JsonSchema { .. }
        | Grammar::Token(_)) => leaf,
    }
}

/// `text` as constant strings and tokens: each of `tokens` where it stands, leftmost first and of
/// two at one place the longer; `text` as it is when it holds none.
fn split_tokens(text: String, tokens: &[String]) -> Grammar {
    if !tokens.iter().any(|token| text.contains(token.as_str())) {
        return Grammar::ConstString(text);
    }
    let mut pieces: Vec<Grammar> = Vec::new();
    let mut plain = String::new();
    let mut rest = text.as_str();
    while let Some(first) = rest.chars().next() {
        if let Some(token) = tokens.iter().find(|token| rest.starts_with(token.as_str())) {
            if !plain.is_empty() {
                pieces.push(Grammar::ConstString(std::mem::take(&mut plain)));
            }
            pieces.push(Grammar::Token(token.clone()));
            rest = &rest[token.len()..];
        } else {
            plain.push(first);
            rest = &rest[first.len_utf8()..];
        }
    }
    if !plain.is_empty() {
        pieces.push(Grammar::ConstString(plain));
    }
    if pieces.len() == 1 {
        if let Some(piece) = pieces.pop() {
            return piece;
        }
    }
    Grammar::Sequence(pieces)
}

/// `written` with `stop_after_first` set when the tags stop after the first: the key is left out
/// otherwise, the engines' default, so a tag that asks for any number of calls is written as it
/// was before the key existed.
fn stopping(mut written: Value, stop_after_first: bool) -> Value {
    if stop_after_first {
        written["stop_after_first"] = Value::Bool(true);
    }
    written
}

/// The nodes of a JSON value: one per scalar, one per object or array and one per member, and
/// one more per sixty-four bytes of a string, so a long `description` weighs what it costs.
fn json_nodes(value: &Value) -> usize {
    match value {
        Value::Object(map) => {
            1 + map
                .values()
                .map(|member| 1 + json_nodes(member))
                .sum::<usize>()
        }
        Value::Array(items) => 1 + items.iter().map(json_nodes).sum::<usize>(),
        Value::String(text) => 1 + text.len() / 64,
        _ => 1,
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

/// One way into a block of calls, from a state with a path into it: from a region with a close of
/// its own (a thought, an answer) whose close leaves the engine where a path into the block
/// starts, that close first, then the path, so the model closes what the prompt left open and
/// goes on from where that leaves it; from any other state (the turn's own, one with no close,
/// one whose close leads nowhere into the block), its own path. A path is the rows from one state
/// to the block through framing states alone, the shortest there is: one row where a state has a
/// row into the block, and more where the template frames its messages (the close of a thought,
/// the row that starts the next message, the row into the block). Every way in is a path the
/// table has.
struct WayIn<'a> {
    close: Option<&'a str>,
    rows: Vec<&'a str>,
}

impl WayIn<'_> {
    /// What the model writes to enter the block this way.
    fn begin(&self) -> String {
        let mut begin = self.close.unwrap_or("").to_string();
        begin.extend(self.rows.iter().copied());
        begin
    }

    /// The text that triggers this way in: the close when there is one, else the first row's.
    fn trigger(&self) -> &str {
        self.close
            .or_else(|| self.rows.first().copied())
            .unwrap_or("")
    }
}

/// A block of calls as the engine takes it: one tag per way in, each `content` between the way's
/// text and the block's close, triggered by each way's first text, the same trigger once.
fn block_of(ways_in: &[WayIn<'_>], content: Grammar, close: &str, count: Count<'_>) -> Grammar {
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
    count.triggered(triggers, tags)
}

/// What a structural tag is asked for: how many calls the request lets the model make, and where
/// its prompt left the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// At least one call: `tool_choice` `required`, a named function, or an allowed-tools list in
    /// `required` mode.
    pub at_least_one: bool,
    /// The prompt left the model inside its thought.
    pub reasoning_open: bool,
    /// At most one call: a named function, or `parallel_tool_calls: false`. The tag stops after
    /// the first call, and the call's end carries what the template writes after the last call to
    /// end the turn, so nothing else can follow.
    pub one_call: bool,
}

/// How many calls a syntax's tag holds, and what follows the last: the request's two bounds and
/// the table's end of the turn, which the tag writes after its one call.
#[derive(Clone, Copy)]
struct Count<'a> {
    at_least_one: bool,
    one: bool,
    /// The terminal that ends the turn wherever the model stands (Kimi K3's message close), or
    /// nothing for a table whose turn a stop token ends.
    last: &'a str,
}

impl Count<'_> {
    /// `end`, followed by the end of the turn when the tag holds one call: nothing may follow it.
    fn end(&self, end: impl Into<String>) -> String {
        let mut end = end.into();
        if self.one {
            end.push_str(self.last);
        }
        end
    }

    /// The calls a region holds: `call` once when the tag holds one, else one or more of it.
    fn calls(&self, call: Grammar) -> Grammar {
        if self.one {
            call
        } else {
            Grammar::Plus(Box::new(call))
        }
    }

    /// Tags `triggers` open in free text, as many as the count lets the model write; with one
    /// call, each tag's end carries the end of the turn.
    fn triggered(&self, triggers: Vec<String>, tags: Vec<Tag>) -> Grammar {
        let tags = tags
            .into_iter()
            .map(|tag| Tag {
                end: self.end(tag.end),
                ..tag
            })
            .collect();
        Grammar::TriggeredTags {
            triggers,
            tags,
            at_least_one: self.at_least_one,
            stop_after_first: self.one,
        }
    }

    /// Tags one after another with nothing between them, as many as the count lets the model
    /// write; the end of the turn is the caller's to write, after what closes them.
    fn separated(&self, tags: Vec<Tag>) -> Grammar {
        Grammar::TagsWithSeparator {
            tags,
            separator: String::new(),
            at_least_one: self.at_least_one,
            stop_after_first: self.one,
        }
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
    /// table has no calls or no tool has a name. `at_least_one` asks for at least one call
    /// (`tool_choice` `required`, a named function, or an allowed-tools list in `required` mode).
    /// With `reasoning_open`, the prompt left the model inside its thought, and the tag follows a
    /// prefix of free text without the call markers, nothing owed at its end (the thought and its
    /// close, or nothing, by where the engine starts the grammar); a table without a thought has
    /// no prefix, and the tag comes back as it is; a table whose thought closes only into the
    /// block has no prefix, its close being its own way in, and gives `None`; one whose thought
    /// closes into the text after it (MiniMax M3's) takes the prefix. Kimi K3's tag has a way in per state
    /// its block is entered from, the state's close first when it has one, so it is the same
    /// either way; so is xLAM's list, whose table has no thought. With the thought left to the
    /// model (the turn's own state opens it and closes an empty one: MiniMax M3), the tag admits
    /// the thought, or the bare close, or nothing, before the calls, since a forced call is
    /// written at once ([`Self::led`]). The tag for any number of calls:
    /// [`Self::grammar_for`] with these two and [`Options::one_call`] unset.
    pub fn grammar(
        &self,
        tools: &[Tool],
        at_least_one: bool,
        reasoning_open: bool,
    ) -> Option<Grammar> {
        self.grammar_for(
            tools,
            Options {
                at_least_one,
                reasoning_open,
                one_call: false,
            },
        )
    }

    /// The structural tag for a call to one of `tools` as `options` ask: [`Self::grammar`]'s tag,
    /// and with [`Options::one_call`] the same tag holding one call. A tag of one call stops after
    /// it (`stop_after_first`): a region that holds calls (a block, a list, Python's calls)
    /// holds that one, and the call's end carries the terminal that ends the turn wherever the
    /// model stands, when the table has one (Kimi K3's message close), since nothing may follow;
    /// a table whose turn ends at a stop token writes nothing after the call. `None` as for
    /// [`Self::grammar`].
    pub fn grammar_for(&self, tools: &[Tool], options: Options) -> Option<Grammar> {
        let named: Vec<&Tool> = tools
            .iter()
            .filter(|tool| !tool.function.name.is_empty())
            .collect();
        if named.is_empty() {
            return None;
        }
        let count = Count {
            at_least_one: options.at_least_one,
            one: options.one_call,
            last: self.turn_end(),
        };
        // The markers are the syntaxes' with a region of calls; xLAM's list has none.
        let markers = self.call_markers();
        let markers = markers.as_ref();
        // Whether the tag takes the prefix when the prompt left the thought open: the syntaxes
        // whose tag enters the calls from the thought on its own way in (Kimi K3's, Inkling's)
        // and xLAM's list, whose table has no thought, take none. MiniMax M3's takes it: its
        // block opens from the text its thought closes into, and a forced block, written at
        // once, leaves no free text for the thought's close before the trigger.
        let (calls, prefixed) = match self.call_syntax()? {
            // xLAM's list stands alone in the turn, no marker before or after it, and its table
            // has no thought: the list is the same either way; a call alone (`json`) is forced as
            // the list, as the gateway's older `json` parser forces it. Between markers (Llama
            // Nemotron) the list is one tag's content, and the thought takes the prefix.
            CallSyntax::JsonList | CallSyntax::JsonAlone => match markers {
                None => (Self::json_list_calls(&named, count), false),
                Some(markers) => (Self::json_list_tag(markers, &named, count), true),
            },
            CallSyntax::Json => (Self::json_calls(markers?, &named, count), true),
            CallSyntax::Keyed(tags) if *tags == KeyedTags::PLAIN => {
                (Self::keyed_calls(markers?, &named, count), true)
            }
            // XTML's grammar is its own: the block's ways in come from the table.
            CallSyntax::Attributed(tags) if tags.typing == Typing::TypeAttribute => {
                let markers = markers?;
                let ways_in = self.ways_into(markers.block.as_ref()?.state);
                (xtml::calls(markers, &ways_in, &named, count)?, false)
            }
            CallSyntax::Attributed(tags) => {
                (attributed::calls(markers?, tags, &named, count)?, true)
            }
            CallSyntax::Keyed(tags) => {
                let markers = markers?;
                let ways_in = self.ways_into_block(markers);
                (keyed::calls(markers, &ways_in, tags, &named, count), true)
            }
            CallSyntax::Tagged(spelling, _) => {
                let markers = markers?;
                let ways_in = self.ways_into_block(markers);
                (
                    tagged::calls(markers, &ways_in, *spelling, &named, count),
                    true,
                )
            }
            CallSyntax::Pythonic(dialect) => {
                (pythonic::calls(markers?, *dialect, &named, count), true)
            }
            CallSyntax::NamedArguments(marker) => {
                (self.named_arguments_calls(marker, &named, count)?, true)
            }
            CallSyntax::Xml => (xml::calls(markers?, self.separator(), &named, count)?, true),
            CallSyntax::Gemma(spelling) => (gemma::calls(markers?, *spelling, &named, count), true),
            CallSyntax::NamedJson(marker) => (self.named_json_calls(marker, &named, count)?, false),
            // A template's patterns do not say which text a run of whitespace or a skipped
            // attribute stands for, and a tag writes only what the template renders: a shaped
            // call's tag is written from the layout its chat template renders, and a table
            // without one has none. With no call owed, the tag takes no prefix: its own free text
            // writes the thought, where a prefix before it would let an engine end the prefix
            // inside the call's terminal and write the rest as free text, a call no tag constrains.
            CallSyntax::Shaped(shape) => (
                shaped::calls(self, shape, self.layout()?, &named, count)?,
                options.at_least_one,
            ),
            // The forced call's constraint is the request's own schema, which the gateway writes
            // for the function `tool_choice` names; the format derives nothing of its own.
            CallSyntax::ForcedArguments => return None,
        };
        self.led(calls, options.reasoning_open, prefixed)
            .map(|grammar| self.bounded(grammar))
    }

    /// The structural tag for an answer that is JSON `schema` accepts (`response_format`
    /// `json_schema`, or `json_object` with an object's schema): the rows into the state the
    /// answer is written in, the JSON, and the answer's close with what ends the turn after it.
    /// Where the output starts in that state (the content of ChatML's tables, of Gemma 4's,
    /// MiniMax's, DeepSeek's), the tag is the schema alone, after the prefix [`Self::grammar`]
    /// writes when the prompt left the model inside its thought; a table whose turn ends at a
    /// stop token writes nothing after it. Otherwise the model starts in the turn's own state, or
    /// in its thought when the prompt opened it, and the tag is one `tag`: the way into the answer
    /// from there, the thought's close first as a call's way in has it, the schema, then the
    /// answer's close and the end of the turn (Kimi K3's response region, Inkling's text message);
    /// an answer a stop token may close ends at the model's stop, with nothing after the schema.
    /// The answer's state is the first content state whose text the template keeps: a framing
    /// state, whose text a response template drops, holds no answer. `None` for a table with no
    /// such state (xLAM, whose answer is anything but a list), with no way into it from where the
    /// model starts, or, with the thought open, a thought that closes only into the block, as for
    /// [`Self::grammar`]; a thought the model may write first leads the answer as it leads a call.
    pub fn answer_grammar(&self, schema: &Value, reasoning_open: bool) -> Option<Grammar> {
        let answer = Grammar::JsonSchema {
            schema: schema.clone(),
            style: None,
        };
        let content = self.answer_state()?;
        let (before, after) = self.around(|layout| layout.answer.as_ref());
        if content == 0 {
            let mut parts = vec![answer];
            if !before.is_empty() {
                parts.insert(0, Grammar::ConstString(before.to_string()));
            }
            let end = format!("{after}{}", self.turn_end());
            if !end.is_empty() {
                parts.push(Grammar::ConstString(end));
            }
            let answer = match parts.len() {
                1 => parts.remove(0),
                _ => Grammar::Sequence(parts),
            };
            return self
                .led(answer, reasoning_open, true)
                .map(|grammar| self.bounded(grammar));
        }
        // The model starts in its thought when the prompt opened it, and otherwise in the turn's
        // own state, which a thought written first also closes back to.
        let start = match self.state_emitting(Emits::Reasoning) {
            Some(reasoning) if reasoning_open => reasoning,
            _ => 0,
        };
        let way = self.way_in(start, content)?;
        // The thought's close and what the layout writes after a thought, then the rows into the
        // answer, what the layout writes before an answer standing right before the last of them,
        // the terminal that opens it.
        let mut begin = way.close.unwrap_or("").to_string();
        if way.close.is_some() {
            begin.push_str(self.around(|layout| layout.thought.as_ref()).1);
        }
        match way.rows.split_last() {
            Some((opener, rows)) => {
                begin.extend(rows.iter().copied());
                begin.push_str(before);
                begin.push_str(opener);
            }
            None => begin.push_str(before),
        }
        // An answer the stop ends, as the layout writes it or where a stop token may close it,
        // ends at the model's stop, which writes nothing.
        let stops = self
            .layout()
            .and_then(|layout| layout.answer.as_ref())
            .map_or_else(|| self.stopped(content), |message| message.stops);
        let mut end = match self.close_of(content) {
            Some((close, _)) if !stops => close.to_string(),
            _ => String::new(),
        };
        end.push_str(after);
        end.push_str(self.turn_end());
        self.led(
            Grammar::Tag(Tag::new(begin, answer, end)),
            reasoning_open,
            false,
        )
        .map(|grammar| self.bounded(grammar))
    }

    /// `body` after what the model writes before it. With the prompt's thought open, the prefix
    /// for a tag that takes one (`prefixed`, [`Self::after_the_thought`]), and nothing for a tag
    /// whose ways in close the thought themselves. With the thought not open, the thought itself
    /// where the table writes one before every call and answer ([`Format::thinks_first`]): the
    /// thought's opener, its text, which writes none of the table's terminals nor the call
    /// syntax's inner tags, and its close; `None` when the thought does not open from the turn's
    /// own state, where the output starts, and close back there. A table that leaves the thought
    /// to the model, the turn's own state opening it and closing an empty one, admits it first
    /// ([`Self::thought_the_model_may_write_first`]); one that does neither has nothing before
    /// `body`.
    fn led(&self, body: Grammar, reasoning_open: bool, prefixed: bool) -> Option<Grammar> {
        if reasoning_open {
            return if prefixed {
                self.after_the_thought(body)
            } else {
                Some(body)
            };
        }
        if !self.writes_thought_first() {
            return Some(self.thought_the_model_may_write_first(body));
        }
        let reasoning = self.state_emitting(Emits::Reasoning)?;
        let thought = self.region(reasoning, |state| self.is_call_state(state))?;
        if thought.home != 0 {
            return None;
        }
        let text = Grammar::AnyText {
            excludes: self.excluded_but(&[]),
        };
        let (before, after) = self.around(|layout| layout.thought.as_ref());
        Some(Grammar::Sequence(vec![
            Grammar::Tag(Tag::new(
                format!("{before}{}", thought.open),
                text,
                format!("{}{after}", thought.close),
            )),
            body,
        ]))
    }

    /// What the table's layout writes before a region's opener and after its close, for the
    /// region `which` picks (a thought or an answer); nothing for a table without one.
    fn around(&self, which: impl Fn(&Layout) -> Option<&Message>) -> (&str, &str) {
        self.layout()
            .and_then(which)
            .map_or(("", ""), |message| (&*message.before, &*message.after))
    }

    /// `body` after the prefix a prompt's open thought takes: free text that can write none of the
    /// table's terminals but the thought's own, nor the call syntax's inner tags, with nothing
    /// owed at its end. A table without a thought has no prefix, and gives `body` as it is; a
    /// thought that closes only into the block has no prefix, its close being its own way in, and
    /// gives `None`; one that closes into the text after it takes the prefix with that close as
    /// its own ([`Self::thought_markers`]).
    fn after_the_thought(&self, body: Grammar) -> Option<Grammar> {
        let Some(reasoning) = self.state_emitting(Emits::Reasoning) else {
            return Some(body);
        };
        // The thought's own markers stay writable: the model closes its thought inside the
        // prefix where the engine applies the grammar from the first token, and an engine that
        // runs a reasoning parser has consumed them before the grammar starts. They are its
        // opener and every close the model writes for it, since the model writes the one its
        // chat template renders, whichever the tags would write.
        let (open, closes) = self.thought_markers(reasoning)?;
        let own: Vec<&str> = std::iter::once(open).chain(closes).collect();
        let excludes = self.excluded_but(&own);
        Some(Grammar::Sequence(vec![Grammar::AnyText { excludes }, body]))
    }

    /// The thought's opener and the closes the model writes for it. Where the thought closes
    /// back to where it opened, the region's opener and every row back there (other than the
    /// turn's end and a stop); where it closes into a state outside the calls instead, as
    /// MiniMax M3's closes into the text after it, the opener of the row into it from the turn's
    /// own state and those closes, the one the table's layout names first. `None` for a thought
    /// that closes only into the block: its close is its own way in, not a marker the text before
    /// the calls could write.
    fn thought_markers(&self, reasoning: usize) -> Option<(&str, Vec<&str>)> {
        let turn_close = self.turn_close();
        let closes_to = |within: &dyn Fn(usize) -> bool| -> Vec<&str> {
            let mut closes: Vec<&str> = Vec::new();
            for (source, on, target) in self.transitions() {
                let text = self.terminal_text(on);
                if source == reasoning
                    && within(target)
                    && Some(on) != turn_close
                    && !self.stops(on)
                    && !closes.contains(&text)
                {
                    closes.push(text);
                }
            }
            closes
        };
        if let Some(thought) = self.region(reasoning, |state| self.is_call_state(state)) {
            let home = thought.home;
            let closes = closes_to(&|target| target == home);
            return Some((thought.open, closes));
        }
        let (_, open, _) = self
            .transitions()
            .find(|&(source, _, target)| source == 0 && target == reasoning)?;
        let mut closes = closes_to(&|target| !self.is_call_state(target) && target != reasoning);
        if closes.is_empty() {
            return None;
        }
        if let Some(named) = self.written_close(reasoning) {
            if let Some(at) = closes.iter().position(|close| *close == named) {
                closes.swap(0, at);
            }
        }
        Some((self.terminal_text(open), closes))
    }

    /// `body` after what the model writes first when the template leaves the thought to it, which
    /// the table says with two rows from the turn's own state, where the output starts: one on the
    /// thought's opener and one on its close, the model writing the thought or closing an empty
    /// one (MiniMax M3). The tag then admits the thought (its opener, its text free of the table's
    /// terminals and the call syntax's inner tags, and its close), or the bare close, or nothing
    /// at all, before `body`: a forced call is written at once, and without this the tag would
    /// take the thought away. A table whose prompt opens the thought has only the opener's row
    /// there, if that, and adds nothing. A prompt with the thought closed is both the template's
    /// adaptive default and a request with thinking switched off (its generation prompt ends with
    /// the thought's close): the tag admits the thought, or a second close, in both, and an answer
    /// that writes one with thinking off reads back as content, as the legacy parser's tag has it.
    fn thought_the_model_may_write_first(&self, body: Grammar) -> Grammar {
        let Some(reasoning) = self.state_emitting(Emits::Reasoning) else {
            return body;
        };
        let opens_from_start = self
            .transitions()
            .any(|(source, _, target)| source == 0 && target == reasoning);
        let Some((open, closes)) = opens_from_start
            .then(|| self.thought_markers(reasoning))
            .flatten()
        else {
            return body;
        };
        let Some(&close) = closes.first() else {
            return body;
        };
        let closes_from_start = self
            .transitions()
            .any(|(source, on, _)| source == 0 && self.terminal_text(on) == close);
        if !closes_from_start {
            return body;
        }
        let text = Grammar::AnyText {
            excludes: self.excluded_but(&[]),
        };
        let first = Grammar::Or(vec![
            Grammar::Tag(Tag::new(open, text, close)),
            Grammar::ConstString(close.to_string()),
        ]);
        Grammar::Sequence(vec![Grammar::Optional(Box::new(first)), body])
    }

    /// What free text before a tag may not write: the table's terminals, but `kept` and those
    /// written inside one of them (a kept close `<e></t>` holds the terminal `<e>`, and free text
    /// that excluded `<e>` could never write the close), and the call syntax's inner tags.
    fn excluded_but(&self, kept: &[&str]) -> Vec<String> {
        let mut excludes: Vec<String> = self
            .terminal_texts()
            .filter(|text| !kept.iter().any(|kept| kept.contains(text)))
            .map(str::to_string)
            .collect();
        let inner = self
            .call_syntax()
            .map(CallSyntax::tag_texts)
            .unwrap_or_default();
        excludes.extend(inner.iter().map(|tag| tag.to_string()));
        excludes
    }

    /// `grammar` with every marker that is one token ([`Format::marker_tokens`]) written as that
    /// token where it stands in a constant string: inside a tag's content and between tags. A
    /// tag's begin and end, its triggers, a separator and the texts free text excludes stay
    /// strings, as the engines take them. Without such markers, `grammar` as it is.
    fn bounded(&self, grammar: Grammar) -> Grammar {
        let tokens = self.marker_token_texts();
        if tokens.is_empty() {
            return grammar;
        }
        bound(grammar, tokens)
    }

    /// The JSON family: each call is the call markers around one JSON object whose `name` is the
    /// tool's and whose `arguments` the tool's schema accepts, the template's newlines included.
    fn json_calls(markers: &CallMarkers<'_>, tools: &[&Tool], count: Count<'_>) -> Grammar {
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
        count.triggered(vec![markers.call_open.to_string()], tags)
    }

    /// Inkling's named JSON: the tag the gateway's old Inkling parser hand-writes. The marker that
    /// says a message is a call triggers it, inside whatever message the model is writing, the
    /// first one included: the marker, `{"name":"NAME","args":` as the template writes it, the
    /// arguments the tool's schema accepts, `}` and the message's close (the terminal of the row
    /// from the turn's own state, where a message starts, to the wrapper state a closed message
    /// leaves the engine in). The tool's name before the marker stays free text: the assembler
    /// takes the call's name from the object and drops the header as the template's framing. The
    /// messages before the call (a thought, the template's empty answer) are free text too, and a
    /// thought is a message of its own, so the tag is the same whether the prompt left the model
    /// inside one or not. The tag is wider than the parser at one point, as the gateway's is: a
    /// marker outside a message's header (after an answer's text, or after a message's close with
    /// no opener between) is writable here and reads as that message's text, so an output written
    /// there satisfies `at_least_one` and yields no call; the model does not write it, the
    /// sequence being off its training. `None` for a table whose messages have no close.
    fn named_json_calls(&self, marker: &str, tools: &[&Tool], count: Count<'_>) -> Option<Grammar> {
        let (_, close, _) = self
            .transitions()
            .find(|&(source, _, target)| source == 0 && self.emits(target) == Emits::Wrapper)?;
        let close = self.terminal_text(close);
        let tags = tools
            .iter()
            .map(|tool| {
                let quoted = Value::String(tool.function.name.clone());
                Tag::new(
                    format!("{marker}{{\"name\":{quoted},\"args\":"),
                    Grammar::JsonSchema {
                        schema: tool.function.parameters.clone(),
                        style: None,
                    },
                    format!("}}{close}"),
                )
            })
            .collect();
        Some(count.triggered(vec![marker.to_string()], tags))
    }

    /// Mistral's named arguments: the call's opener triggers a tag per tool, the opener, the
    /// tool's name and the marker as its `begin`, the arguments the tool's schema accepts as its
    /// content, and nothing as its `end`, since the template writes no closing marker: the object
    /// the schema accepts ends the tag, and the next opener or the end of the turn follows. The
    /// opener is the terminal of the row from the turn's own state into the arguments state;
    /// `None` for a table without one.
    fn named_arguments_calls(
        &self,
        marker: &str,
        tools: &[&Tool],
        count: Count<'_>,
    ) -> Option<Grammar> {
        let (_, open, _) = self
            .transitions()
            .find(|&(source, _, target)| source == 0 && self.emits(target) == Emits::Arguments)?;
        let open = self.terminal_text(open);
        let tags = tools
            .iter()
            .map(|tool| {
                Tag::new(
                    format!("{open}{}{marker}", tool.function.name),
                    Grammar::JsonSchema {
                        schema: tool.function.parameters.clone(),
                        style: None,
                    },
                    "",
                )
            })
            .collect();
        Some(count.triggered(vec![open.to_string()], tags))
    }

    /// xLAM's list: the whole output is `[`, the calls joined by `, `, `]`, each call a JSON object
    /// whose `name` is the tool's and whose `arguments` the tool's schema accepts, spelled as the
    /// template spells it (`{"name": "NAME", "arguments": {...}}`, a space after each colon and
    /// comma); with `at_least_one`, the list alone, and otherwise the list or prose, since the
    /// model answers in prose instead. The prose is the assembler's rule exactly: its first byte
    /// that is not whitespace is not `[`, and the rest is free, a `[` later in it included (a
    /// Markdown link, in 592 of the 13,085 recorded prose answers). The one place the grammar is
    /// narrower than the assembler: an output that starts with `[` and opens no object is content
    /// to the assembler and not writable here.
    /// The list of the tools' calls as xLAM and Llama Nemotron write it: `[`, each call as
    /// `{"name": NAME, "arguments": ` and the tool's schema, `, ` between calls, `]`; with one
    /// call, that call alone, and after the `]` what the count writes after the last call.
    fn json_list(tools: &[&Tool], count: Count<'_>) -> Grammar {
        let one_call = Grammar::Or(
            tools
                .iter()
                .map(|tool| {
                    let name = Value::String(tool.function.name.clone());
                    Grammar::Sequence(vec![
                        Grammar::ConstString(format!("{{\"name\": {name}, \"arguments\": ")),
                        Grammar::JsonSchema {
                            schema: tool.function.parameters.clone(),
                            style: None,
                        },
                        Grammar::ConstString("}".to_string()),
                    ])
                })
                .collect(),
        );
        let mut list = vec![Grammar::ConstString("[".to_string()), one_call.clone()];
        if !count.one {
            list.push(Grammar::Star(Box::new(Grammar::Sequence(vec![
                Grammar::ConstString(", ".to_string()),
                one_call,
            ]))));
        }
        list.push(Grammar::ConstString(count.end("]")));
        Grammar::Sequence(list)
    }

    /// xLAM's list alone in the turn: the list, or, when no call is required, prose that does
    /// not start with a bracket.
    fn json_list_calls(tools: &[&Tool], count: Count<'_>) -> Grammar {
        let list = Self::json_list(tools, count);
        if count.at_least_one {
            list
        } else {
            Grammar::Or(vec![
                Grammar::Sequence(vec![
                    Grammar::Regex(r"\s*[^\s\[]".to_string()),
                    Grammar::AnyText {
                        excludes: Vec::new(),
                    },
                ]),
                list,
            ])
        }
    }

    /// Llama Nemotron's list between its markers: one tag, the opener, the list, the closer.
    fn json_list_tag(markers: &CallMarkers<'_>, tools: &[&Tool], count: Count<'_>) -> Grammar {
        // The tag's end carries the end of the turn, so the list inside it writes none.
        let inside = Count { last: "", ..count };
        count.triggered(
            vec![markers.call_open.to_string()],
            vec![Tag::new(
                markers.call_open,
                Self::json_list(tools, inside),
                markers.call_close,
            )],
        )
    }

    /// Keyed arguments in the compact spelling (GLM 4.7 and later, IQuest): each call is the call
    /// opener and the tool's name, the schema written in the engines' `glm_xml` style (`<arg_key>`
    /// and `<arg_value>` pairs with nothing between them), and the call's close; the shape
    /// xgrammar's built-in GLM tag has and the gateway writes today. Ling's and Hy4's spellings
    /// are not this one, and are written from their own tags instead (the `keyed` module).
    fn keyed_calls(markers: &CallMarkers<'_>, tools: &[&Tool], count: Count<'_>) -> Grammar {
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
        count.triggered(vec![markers.call_open.to_string()], tags)
    }

    /// The text the table drops wherever it stands, which the template writes before every tag
    /// (MiniMax M3's separator), or nothing for a table without one.
    fn separator(&self) -> &str {
        (0..self.terminal_texts().count())
            .find(|&index| self.ignored(index))
            .map_or("", |index| self.terminal_text(index))
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

    /// The state an answer is written in: the first content state whose text the template keeps,
    /// a framing row's state (the text outside a response template's fields) being none.
    fn answer_state(&self) -> Option<usize> {
        (0..self.states())
            .find(|&state| self.emits(state) == Emits::Content && !self.is_framing(state))
    }

    /// Whether a stop token closes the region `state` is: a row out of it on one, which the
    /// model's stop takes.
    fn stopped(&self, state: usize) -> bool {
        self.transitions()
            .any(|(source, on, target)| source == state && target != state && self.stops(on))
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
    /// back to a state it is entered from, other than the end of the turn and a stop token (the
    /// model's stop ends the region there, with no text a tag could write), and its opener the
    /// terminal of that state's row into `state`; that state is its home. The states name the
    /// markers, not the order of the rows. Where the template closes the region more than one way,
    /// the close is the one the table's layout names for a thought or an answer, the one its chat
    /// template renders, and otherwise the first such row. The turn's own state, where the opener
    /// leaves the engine, is no region: the rows back into it close the regions it holds.
    fn region(&self, state: usize, within: impl Fn(usize) -> bool + Copy) -> Option<Region<'_>> {
        if state == 0 {
            return None;
        }
        let from = self.entered_from(state, within);
        let turn_close = self.turn_close();
        let closes = || {
            self.transitions().filter(|&(source, on, target)| {
                source == state
                    && from.contains(&target)
                    && Some(on) != turn_close
                    && !self.stops(on)
            })
        };
        let written = self.written_close(state);
        let (_, close, home) = written
            .and_then(|text| closes().find(|&(_, on, _)| self.terminal_text(on) == text))
            .or_else(|| closes().next())?;
        let (_, open, _) = self
            .transitions()
            .find(|&(source, _, target)| source == home && target == state)?;
        Some(Region {
            open: self.terminal_text(open),
            close: self.terminal_text(close),
            home,
        })
    }

    /// The close the table's layout names for the region `state` is, the thought or the answer:
    /// the terminal its chat template renders there.
    fn written_close(&self, state: usize) -> Option<&str> {
        let layout = self.layout()?;
        let message = if Some(state) == self.state_emitting(Emits::Reasoning) {
            layout.thought.as_ref()
        } else if Some(state) == self.answer_state() {
            layout.answer.as_ref()
        } else {
            None
        };
        message?.close.as_deref()
    }

    /// The ways into the block `state` is, one per state with a path into it (a row into it, or
    /// rows through framing states), in the order of the states, the same way in once: from a
    /// region with a close that leaves the engine where a path into the block starts, the close
    /// and then that path; from any other state, its own path.
    fn ways_into(&self, state: usize) -> Vec<WayIn<'_>> {
        let mut ways: Vec<WayIn<'_>> = Vec::new();
        let reaching = (0..self.states()).filter(|&from| {
            from != state && !self.is_arguments(from) && self.path(from, state).is_some()
        });
        for from in reaching {
            let Some(way) = self.way_in(from, state) else {
                continue;
            };
            if !ways.iter().any(|known| known.begin() == way.begin()) {
                ways.push(way);
            }
        }
        ways
    }

    /// The way into `state` from `from`: the close of the region `from` is, when it has one and a
    /// path into `state` starts where the close leaves the engine, then that path; else the path
    /// from `from` itself.
    fn way_in(&self, from: usize, state: usize) -> Option<WayIn<'_>> {
        let through_close = self.close_of(from).and_then(|(close, landing)| {
            Some(WayIn {
                close: Some(close),
                rows: self.path(landing, state)?,
            })
        });
        through_close.or_else(|| {
            Some(WayIn {
                close: None,
                rows: self.path(from, state)?,
            })
        })
    }

    /// The close of the region `state` is, and the state it leaves the engine in: the row back to
    /// the state the region opened from, its home ([`Self::region`]); for a region with no such
    /// row, its row into a framing state, as a thought closes into the wrapping between the
    /// template's messages (Inkling's message close). Neither is the end of the turn, a stop token
    /// or a row into the calls' block; the turn's own state is no region and has no close.
    fn close_of(&self, state: usize) -> Option<(&str, usize)> {
        if let Some(Region { close, home, .. }) =
            self.region(state, |state| self.is_call_state(state))
        {
            return Some((close, home));
        }
        if state == 0 {
            return None;
        }
        let turn_close = self.turn_close();
        let block = self
            .call_markers()
            .and_then(|markers| markers.block)
            .map(|block| block.state);
        self.transitions()
            .find(|&(source, on, target)| {
                source == state
                    && target != state
                    && Some(on) != turn_close
                    && !self.stops(on)
                    && self.between_messages(target)
                    && Some(target) != block
            })
            .map(|(_, on, target)| (self.terminal_text(on), target))
    }

    /// The shortest path of rows from `from` to `to` through framing states alone, the terminals
    /// in the order the model writes them; `None` when there is none. One row where `from` has a
    /// row into `to`. The end of the turn is no step of a path, nor is a stop token: a way in that
    /// wrote either would end the turn before the calls. Rows are tried in the table's order, so
    /// of two paths of one length the one whose rows come first is taken: the first row into `to`
    /// when there are two.
    fn path(&self, from: usize, to: usize) -> Option<Vec<&str>> {
        if from == to {
            return Some(Vec::new());
        }
        let turn_close = self.turn_close();
        let mut reached_by: Vec<Option<(usize, usize)>> = vec![None; self.states()];
        let mut seen = vec![false; self.states()];
        seen[from] = true;
        let mut frontier = VecDeque::from([from]);
        while let Some(state) = frontier.pop_front() {
            for (source, on, target) in self.transitions() {
                if source != state || seen[target] || Some(on) == turn_close || self.stops(on) {
                    continue;
                }
                seen[target] = true;
                reached_by[target] = Some((state, on));
                if target == to {
                    let mut rows = Vec::new();
                    let mut at = to;
                    while let Some((before, on)) = reached_by[at] {
                        rows.push(self.terminal_text(on));
                        at = before;
                    }
                    rows.reverse();
                    return Some(rows);
                }
                if self.between_messages(target) {
                    frontier.push_back(target);
                }
            }
        }
        None
    }

    /// Whether `state` lies between the template's messages: its wrapping (a wrapper state), or
    /// text the template discards (a framing row: the text outside a response template's
    /// fields). The model writes only the markers around its messages and calls there, so a way
    /// in may pass through it.
    fn between_messages(&self, state: usize) -> bool {
        self.emits(state) == Emits::Wrapper || self.is_framing(state)
    }

    /// Whether terminal `on` is a token the model stops on ([`Format::is_stop_token`]): the stop
    /// decoder hides it, so a tag cannot write it, and the region it closes ends at the model's
    /// stop.
    fn stops(&self, on: usize) -> bool {
        self.is_stop_token(self.terminal_text(on))
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

    /// What the model writes to end its turn wherever it stands: the end of the turn's terminal,
    /// or nothing for a table without one, whose turn a stop token ends.
    fn turn_end(&self) -> &str {
        self.turn_close()
            .map_or("", |close| self.terminal_text(close))
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;
    use serde_json::json as value;

    use super::*;
    use crate::{formats, tagged::attributed::Tags as AttributedTags};

    fn tool(name: &str, strict: Option<bool>, parameters: Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
                strict,
                extra: Default::default(),
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
        // `wrap_in_reasoning_prefix(tag, Glm4MoeParser::reasoning_prefix(..))`: a sequence of free
        // text that writes none of the call markers, with nothing owed at its end (an engine that
        // runs a reasoning parser applies the grammar only after the model's own `</think>`, so a
        // prefix closed by `</think>` would be owed a second time), then the tag.
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
                "type": "any_text",
                "excludes": [
                    "<tool_call>", "</tool_call>",
                    "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
                ],
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
        assert_eq!(KeyedTags::LING.newlines.after_name, "\n");
    }

    #[test]
    fn the_prefix_excludes_each_tables_own_terminals_and_inner_tags() {
        // No old builder wraps these two, so the derived shape is pinned here: the prefix is free
        // text whose excludes are the table's terminals in order, the thought's own two left out
        // so the model may close it where the engine runs the grammar from the first token, and
        // the syntax's inner tags; nothing is owed at its end; the second element is the tag as
        // derived without it.
        let tools = weather_tools();
        for (format, excludes) in [
            (
                formats::deepseek_v4_1(),
                value!([
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
                value!(["<tool_call>", "</tool_call>"]),
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
                value!({"type": "any_text", "excludes": excludes}),
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
        let as_xtml = into_the_block(CallSyntax::Attributed(AttributedTags::XTML))
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
            .calls(CallSyntax::Attributed(AttributedTags::XTML));
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
            .calls(CallSyntax::Attributed(AttributedTags::XTML))
            .grammar(&tools, true, true)
            .expect("a grammar")
            .payload();
        assert_eq!(as_xtml["format"]["triggers"], value!(["<tool_call>"]));
        assert_eq!(begins(&as_xtml), ["<tool_call>"]);
        assert_eq!(as_xtml["format"]["tags"][0]["end"], "</tool_call>");
    }

    #[test]
    fn a_way_in_follows_the_framing_between_messages() {
        // A turn of messages: `<|start|>` opens one from the wrapping between them, its kind
        // follows, and `<|end|>` closes a thought back into the wrapping. The block is entered
        // from a message's header alone, so the turn's start goes in by starting a message, and
        // the thought by closing, starting the next message and entering the block: a close, a
        // row between messages and the block's opener. The header, with its own row, keeps it.
        let messages = Format::new("messages")
            .terminal("start", "<|start|>")
            .terminal("think", "<|think|>")
            .terminal("end", "<|end|>")
            .terminal("calls_open", "<|calls|>")
            .terminal("calls_close", "<|/calls|>")
            .terminal("call_open", "<|call|>")
            .terminal("call_close", "<|/call|>")
            .state("between", Emits::Wrapper)
            .state("header", Emits::Wrapper)
            .state("reasoning", Emits::Reasoning)
            .state("calls", Emits::Wrapper)
            .state("call", Emits::Arguments)
            .transition("between", "start", "header")
            .transition("header", "think", "reasoning")
            .transition("reasoning", "end", "between")
            .transition("header", "calls_open", "calls")
            .transition("calls", "call_open", "call")
            .transition("call", "call_close", "calls")
            .transition("calls", "calls_close", "header")
            .calls(CallSyntax::Attributed(AttributedTags::XTML));
        let tools = weather_tools();
        let payload = messages
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        assert_eq!(
            begins(&payload),
            [
                "<|start|><|calls|>",
                "<|calls|>",
                "<|end|><|start|><|calls|>"
            ]
        );
        assert_eq!(
            payload["format"]["triggers"],
            value!(["<|start|>", "<|calls|>", "<|end|>"])
        );
        assert_eq!(payload["format"]["tags"][0]["end"], "<|/calls|>");
        // The open thought changes nothing: the tag closes it on its own way in.
        assert_eq!(
            messages.grammar(&tools, true, true),
            messages.grammar(&tools, true, false)
        );
        // The end of the turn is no step of a way in. Here the answer closes into an aside, and
        // the aside's rows back to the turn's state are the end of the turn, the first of them,
        // and `<|next|>`: the answer's way in closes it and goes on by `<|next|>`, as the aside's
        // own does, never by ending the turn.
        let aside = Format::new("aside")
            .terminal("response_open", "<|response|>")
            .terminal("response_close", "<|/response|>")
            .terminal("next", "<|next|>")
            .terminal("calls_open", "<|calls|>")
            .terminal("calls_close", "<|/calls|>")
            .terminal("call_open", "<|call|>")
            .terminal("call_close", "<|/call|>")
            .terminal("turn_end", "<|turn_end|>")
            .state("turn", Emits::Wrapper)
            .state("content", Emits::Content)
            .state("aside", Emits::Wrapper)
            .state("calls", Emits::Wrapper)
            .state("call", Emits::Arguments)
            .transition("turn", "response_open", "content")
            .transition("content", "response_close", "aside")
            .transition("aside", "turn_end", "turn")
            .transition("aside", "next", "turn")
            .transition("turn", "calls_open", "calls")
            .transition("calls", "call_open", "call")
            .transition("call", "call_close", "calls")
            .transition("calls", "calls_close", "turn")
            .transition("turn", "turn_end", "turn")
            .transition("content", "turn_end", "turn")
            .transition("calls", "turn_end", "turn")
            .transition("call", "turn_end", "turn")
            .calls(CallSyntax::Attributed(AttributedTags::XTML));
        let payload = aside
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        assert_eq!(
            begins(&payload),
            [
                "<|calls|>",
                "<|/response|><|next|><|calls|>",
                "<|next|><|calls|>"
            ]
        );
    }

    #[test]
    fn a_thought_that_closes_into_its_answer_has_no_way_of_its_own() {
        // Hy4's shape: the block is entered from content alone, and the thought closes into
        // content, which is text, not framing; the thought has no path of its own, and the open
        // thought takes the prefix instead, as before.
        let payload = formats::hy4()
            .grammar(&weather_tools(), true, false)
            .expect("a grammar")
            .payload();
        assert_eq!(begins(&payload), ["<tool_calls:opensource>"]);
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
        let as_dsml = two_doors(CallSyntax::Attributed(AttributedTags::DSML))
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        assert_eq!(as_dsml["format"]["elements"][0]["value"], "\n\n<calls>\n");
        let as_xtml = two_doors(CallSyntax::Attributed(AttributedTags::XTML))
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
        let reversed = reversed.calls(CallSyntax::Attributed(AttributedTags::XTML));
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
            .terminal("invoke_close", AttributedTags::DSML.invoke_close)
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
            .calls(CallSyntax::Attributed(AttributedTags::DSML));
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
    fn xlam_writes_the_list_alone_each_call_named_and_its_arguments_by_the_schema() {
        // The recorded outputs: `[` first, `{"name": "NAME", "arguments": {...}}` with a space
        // after each colon and comma, `, ` between calls, `]` last and nothing after it. The table
        // has no thought, so the thought's state changes nothing.
        let tools = weather_tools();
        let payload = formats::xlam()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let call = |name: &str, parameters: Value| {
            value!({"type": "sequence", "elements": [
                {
                    "type": "const_string",
                    "value": format!("{{\"name\": \"{name}\", \"arguments\": "),
                },
                {"type": "json_schema", "json_schema": parameters},
                {"type": "const_string", "value": "}"},
            ]})
        };
        let one_call = value!({"type": "or", "elements": tools
            .iter()
            .filter(|tool| !tool.function.name.is_empty())
            .map(|tool| call(&tool.function.name, tool.function.parameters.clone()))
            .collect::<Vec<_>>()});
        assert_eq!(
            payload,
            value!({"format": {"type": "sequence", "elements": [
                {"type": "const_string", "value": "["},
                one_call,
                {"type": "star", "content": {"type": "sequence", "elements": [
                    {"type": "const_string", "value": ", "},
                    one_call,
                ]}},
                {"type": "const_string", "value": "]"},
            ]}})
        );
        assert_eq!(
            formats::xlam()
                .grammar(&tools, true, true)
                .expect("a grammar")
                .payload(),
            payload
        );
    }

    #[test]
    fn xlam_without_a_required_call_writes_the_list_or_prose_without_a_bracket() {
        let tools = weather_tools();
        let list = formats::xlam()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .to_json();
        let either = formats::xlam()
            .grammar(&tools, false, false)
            .expect("a grammar")
            .to_json();
        assert_eq!(
            either,
            value!({"type": "or", "elements": [
                {"type": "sequence", "elements": [
                    {"type": "regex", "pattern": "\\s*[^\\s\\[]"},
                    {"type": "any_text", "excludes": []},
                ]},
                list,
            ]})
        );
    }

    #[test]
    fn inkling_derives_the_tag_the_gateway_hand_writes_today() {
        // The old crate's `InklingParser::build_structural_tag`: one trigger, the call marker; per
        // named tool a tag from the marker and `{"name":"NAME","args":` to `}<|end_message|>`
        // around the tool's schema; `at_least_one` as asked. The messages before the call (a
        // thought, the template's empty answer, the tool's name before the marker) are free text,
        // so the recorded thinking turn stays writable, and a call in the first message is
        // constrained too. The same tag with the thought open.
        let tools = weather_tools();
        let payload = formats::inkling()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let schema = value!({"type": "object", "properties": {"city": {"type": "string"}}});
        let per_tool = |name: &str, schema: &Value| {
            value!({
                "type": "tag",
                "begin": format!("<|content_invoke_tool_json|>{{\"name\":\"{name}\",\"args\":"),
                "content": {"type": "json_schema", "json_schema": schema},
                "end": "}<|end_message|>",
            })
        };
        assert_eq!(
            payload,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<|content_invoke_tool_json|>"],
                "tags": [
                    per_tool("get_weather", &schema),
                    per_tool("lookup", &schema),
                    per_tool("ping", &value!({})),
                ],
                "at_least_one": true,
            }})
        );
        assert_eq!(
            formats::inkling()
                .grammar(&tools, false, true)
                .expect("a grammar")
                .payload()["format"]["at_least_one"],
            false
        );
        assert_eq!(
            formats::inkling().grammar(&tools, true, true),
            formats::inkling().grammar(&tools, true, false)
        );
    }

    #[test]
    fn llama_nemotron_derives_one_tag_around_the_list_of_calls() {
        // No hand-written tag to pin to: the gateway's older parsers have none for this syntax.
        // One tag: `<TOOLCALL>` begins it, xLAM's list of the tools' calls is its content and
        // `</TOOLCALL>` ends it. With the thought open, the tag follows a prefix of free text
        // without the call's markers, the thought's own still writable.
        let tools = weather_tools();
        let payload = formats::llama_nemotron()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let list = formats::xlam()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .to_json();
        assert_eq!(
            payload,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<TOOLCALL>"],
                "tags": [{
                    "type": "tag",
                    "begin": "<TOOLCALL>",
                    "content": list,
                    "end": "</TOOLCALL>",
                }],
                "at_least_one": true,
            }})
        );
        let open = formats::llama_nemotron()
            .grammar(&tools, true, true)
            .expect("a grammar");
        assert!(matches!(
            &open,
            Grammar::Sequence(parts)
                if parts.len() == 2
                    && matches!(&parts[0], Grammar::AnyText { excludes }
                        if excludes.contains(&"<TOOLCALL>".to_string())
                            && excludes.contains(&"</TOOLCALL>".to_string())
                            && !excludes.contains(&"<think>".to_string())
                            && !excludes.contains(&"</think>".to_string()))
        ));
    }

    #[test]
    fn mistral_derives_a_tag_per_tool_from_the_opener_to_the_arguments_object() {
        // No hand-written tag to pin to: the gateway's older `mistral` parser reads Mistral 7B
        // v0.3's `[TOOL_CALLS][{…}]` list, another syntax. Per named tool, `[TOOL_CALLS]`, the
        // name and `[ARGS]` begin the tag, the tool's schema is its content, and nothing ends it:
        // the template writes no closing marker, so the object ends the tag and the next opener
        // or the end of the turn follows. Mistral's prompt never leaves the thought open, and the
        // tag with it open is the same tag after a prefix of free text.
        let tools = weather_tools();
        let payload = formats::mistral_v13()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let schema = value!({"type": "object", "properties": {"city": {"type": "string"}}});
        let per_tool = |name: &str, schema: &Value| {
            value!({
                "type": "tag",
                "begin": format!("[TOOL_CALLS]{name}[ARGS]"),
                "content": {"type": "json_schema", "json_schema": schema},
                "end": "",
            })
        };
        assert_eq!(
            payload,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["[TOOL_CALLS]"],
                "tags": [
                    per_tool("get_weather", &schema),
                    per_tool("lookup", &schema),
                    per_tool("ping", &value!({})),
                ],
                "at_least_one": true,
            }})
        );
        let open = formats::mistral_v13()
            .grammar(&tools, true, true)
            .expect("a grammar");
        assert!(matches!(
            &open,
            Grammar::Sequence(parts)
                if parts.len() == 2
                    && matches!(&parts[0], Grammar::AnyText { excludes }
                        if excludes.contains(&"[TOOL_CALLS]".to_string())
                            && excludes.contains(&"[ARGS]".to_string())
                            && !excludes.contains(&"[THINK]".to_string()))
        ));
    }

    #[test]
    fn no_named_tool_and_no_calls_give_none() {
        let tools = weather_tools();
        assert!(formats::glm().grammar(&[], true, false).is_none());
        assert!(formats::glm()
            .grammar(&[tool("", None, value!({}))], true, false)
            .is_none());
        assert!(formats::plain().grammar(&tools, true, false).is_none());
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

    /// The tag for one call to one of `tools`.
    fn one_call(
        format: &Format,
        tools: &[Tool],
        at_least_one: bool,
        reasoning_open: bool,
    ) -> Value {
        format
            .grammar_for(
                tools,
                Options {
                    at_least_one,
                    reasoning_open,
                    one_call: true,
                },
            )
            .expect("a grammar")
            .payload()
    }

    /// `payload` with `stop_after_first` set on the node at `pointer`.
    fn stopping_at(mut payload: Value, pointer: &str) -> Value {
        payload.pointer_mut(pointer).expect("the node")["stop_after_first"] = Value::Bool(true);
        payload
    }

    #[test]
    fn one_call_is_the_qwen3_tag_stopping_after_the_first() {
        // A named function, or `parallel_tool_calls: false`: one tag per named tool, as for any
        // number of calls, and the tags stop after the first. A stop token ends the turn, so
        // nothing follows the call's close.
        let schema = value!({"type": "object", "properties": {"city": {"type": "string"}}});
        let per_tool = |name: &str, schema: &Value| {
            value!({
                "type": "tag",
                "begin": format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": "),
                "content": {"type": "json_schema", "json_schema": schema},
                "end": "}\n</tool_call>",
            })
        };
        assert_eq!(
            one_call(
                &formats::qwen3(CallSyntax::Json),
                &weather_tools(),
                true,
                false
            ),
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": [
                    per_tool("get_weather", &schema),
                    per_tool("lookup", &schema),
                    per_tool("ping", &value!({})),
                ],
                "at_least_one": true,
                "stop_after_first": true,
            }})
        );
    }

    #[test]
    fn one_call_stops_glm_and_gemma4_after_the_first_and_changes_nothing_else() {
        // Each call is a tag of its own and a stop token ends the turn: the tags for any number
        // of calls, pinned above and in `gemma`, stopping after the first, with the call optional
        // or required and with the thought open or not (the prefix is the same).
        let tools = weather_tools();
        for format in [formats::glm(), formats::gemma4(), formats::iquest()] {
            for at_least_one in [true, false] {
                let many = format
                    .grammar(&tools, at_least_one, false)
                    .expect("a grammar")
                    .payload();
                assert_eq!(
                    one_call(&format, &tools, at_least_one, false),
                    stopping_at(many, "/format"),
                    "{}",
                    format.name()
                );
                let open = format
                    .grammar(&tools, at_least_one, true)
                    .expect("a grammar")
                    .payload();
                assert_eq!(
                    one_call(&format, &tools, at_least_one, true),
                    stopping_at(open, "/format/elements/1"),
                    "{}",
                    format.name()
                );
            }
        }
    }

    #[test]
    fn one_call_in_a_block_of_invokes_is_one_invoke_then_the_blocks_close() {
        // MiniMax M2's one block around the turn's calls and DeepSeek V4.1's: the block opens as
        // before, holds one invoke, and closes; the invokes are tags with nothing between them,
        // which stop after the first.
        let tools = weather_tools();
        for (format, block) in [(formats::minimax_m2(), 1), (formats::deepseek_v4_1(), 1)] {
            let many = format
                .grammar(&tools, true, false)
                .expect("a grammar")
                .payload();
            assert_eq!(
                one_call(&format, &tools, true, false),
                stopping_at(many, &format!("/format/elements/{block}")),
                "{}",
                format.name()
            );
        }
        let one = one_call(&formats::minimax_m2(), &tools, true, false);
        assert_eq!(
            one["format"]["elements"][0]["value"],
            "\n<minimax:tool_call>\n"
        );
        assert_eq!(
            one["format"]["elements"][1]["tags"][0]["begin"],
            "<invoke name=\"get_weather\">\n"
        );
        assert_eq!(
            one["format"]["elements"][2]["value"],
            "</minimax:tool_call>"
        );
        // dots3's block around each call: a tag per tool, stopping after the first.
        let dots3 = formats::dots3()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        assert_eq!(
            one_call(&formats::dots3(), &tools, true, false),
            stopping_at(dots3, "/format")
        );
    }

    #[test]
    fn one_call_in_kimi_k3s_block_holds_one_call_and_ends_the_turn() {
        // Kimi K3 writes its turn's close after the block, and a tag of one call has nothing after
        // it: each way in holds one call, not one or more, and the block's close is followed by
        // the end of the turn.
        let tools = weather_tools();
        let many = formats::kimi_k3()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let one = one_call(&formats::kimi_k3(), &tools, true, false);
        assert_eq!(one["format"]["stop_after_first"], true);
        assert_eq!(one["format"]["triggers"], many["format"]["triggers"]);
        assert_eq!(begins(&one), begins(&many));
        for (tag, before) in one["format"]["tags"]
            .as_array()
            .expect("tags")
            .iter()
            .zip(many["format"]["tags"].as_array().expect("tags"))
        {
            assert_eq!(tag["end"], "<|close|>tools<|sep|><|close|>message<|sep|>");
            assert_eq!(
                tag["content"], before["content"]["content"],
                "one call, no plus"
            );
        }
    }

    #[test]
    fn one_call_is_one_item_of_a_list_and_one_python_call() {
        // xLAM's list holds one call between its brackets, and so do LFM2.5's list and Olmo 3's
        // lines; Hy4's block and MiniMax M3's hold one call each.
        let tools = weather_tools();
        let xlam = one_call(&formats::xlam(), &tools, true, false);
        let elements = xlam["format"]["elements"].as_array().expect("the list");
        assert_eq!(elements.len(), 3, "`[`, one call, `]`");
        assert_eq!(elements[0]["value"], "[");
        assert_eq!(elements[1]["type"], "or");
        assert_eq!(elements[2]["value"], "]");
        assert!(xlam["format"].get("stop_after_first").is_none());
        for format in [formats::lfm2_5(), formats::olmo3()] {
            let one = one_call(&format, &tools, true, false);
            assert_eq!(one["format"]["stop_after_first"], true, "{}", format.name());
            assert_eq!(
                one["format"]["tags"][0]["content"]["type"],
                "or",
                "{}: one call of one of the tools",
                format.name()
            );
        }
        let hy4 = one_call(&formats::hy4(), &tools, true, false);
        assert_eq!(hy4["format"]["stop_after_first"], true);
        assert_ne!(hy4["format"]["tags"][0]["content"]["type"], "plus");
        let m3 = one_call(&formats::minimax_m3(), &tools, true, false);
        let block = &m3["format"]["elements"][1]["elements"][1];
        assert_eq!(
            block["stop_after_first"], true,
            "the block after the thought the tag admits"
        );
        let invokes = &block["tags"][0]["content"];
        assert_eq!(invokes["type"], "tags_with_separator");
        assert_eq!(invokes["at_least_one"], true);
        assert_eq!(invokes["stop_after_first"], true);
    }

    #[test]
    fn without_one_call_every_tag_is_the_one_grammar_gives() {
        let tools = weather_tools();
        for table in [
            formats::Table::Qwen3,
            formats::Table::Qwen3Tagged,
            formats::Table::DeepSeekV4_1,
            formats::Table::Hy4,
            formats::Table::Xlam,
            formats::Table::MinimaxM3,
            formats::Table::KimiK3,
            formats::Table::Inkling,
            formats::Table::Gemma4,
        ] {
            let format = table.format();
            for (at_least_one, reasoning_open) in
                [(true, false), (true, true), (false, false), (false, true)]
            {
                assert_eq!(
                    format.grammar_for(
                        &tools,
                        Options {
                            at_least_one,
                            reasoning_open,
                            one_call: false,
                        },
                    ),
                    format.grammar(&tools, at_least_one, reasoning_open),
                    "{}",
                    format.name()
                );
            }
        }
    }

    /// The message `output` reads as under `format` after `prompt`, typed by `tools`.
    fn read(format: &Format, tools: &[Tool], prompt: &str, output: &str) -> Value {
        use crate::{
            adapt,
            engine::Engine,
            event::Events,
            input::{EngineFinish, Input},
            parser::Parser as _,
            tagged::Declared,
        };
        let mut engine = Engine::new(format.clone(), Declared::of(tools));
        let mut out = Events::new();
        engine
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut out,
            )
            .expect("the prompt");
        engine
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: output,
                    spans: &[],
                },
                &mut out,
            )
            .expect("the output");
        engine
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("the end");
        let events = out.drain();
        let message = adapt::chat::message(0, &events).message;
        let calls: Vec<Value> = message
            .tool_calls
            .unwrap_or_default()
            .iter()
            .map(|call| {
                let arguments = call.function.arguments.as_deref().unwrap_or("{}");
                value!({
                    "name": call.function.name,
                    "arguments": serde_json::from_str::<Value>(arguments).expect("JSON"),
                })
            })
            .collect();
        value!({
            "content": message.content,
            "reasoning": message.reasoning_content,
            "calls": calls,
        })
    }

    #[test]
    fn a_one_call_tags_sample_reads_back_as_that_call() {
        // Each table's one-call tag written out for a call to `get_weather` (the samples an engine
        // checks against the tags): the table reads it back as that call and nothing else.
        let tools = weather_tools();
        let call = value!({
            "content": null,
            "reasoning": null,
            "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
        });
        for (format, sample) in [
            (
                formats::qwen3(CallSyntax::Json),
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n\
                 </tool_call>",
            ),
            (
                formats::glm(),
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
                 </tool_call>",
            ),
            (
                formats::gemma4(),
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
            ),
            (
                formats::minimax_m2(),
                "\n<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter \
                 name=\"city\">Paris</parameter>\n</invoke>\n</minimax:tool_call>",
            ),
        ] {
            assert_eq!(read(&format, &tools, "", sample), call, "{}", format.name());
        }
    }

    /// A JSON answer's schema.
    fn answer_schema() -> Value {
        value!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
    }

    /// The JSON answer's tag for `format`, as a payload.
    fn answer(format: &Format, reasoning_open: bool) -> Option<Value> {
        format
            .answer_grammar(&answer_schema(), reasoning_open)
            .map(|grammar| grammar.payload())
    }

    #[test]
    fn a_json_answer_where_the_output_starts_in_content_is_the_schema_alone() {
        // Qwen3, GLM, Gemma 4 and MiniMax M2 start their output in content and end their turn at
        // a stop token: the answer is the schema and nothing else, the constraint the gateway
        // gives a `json_schema` response today.
        let alone = value!({"format": {"type": "json_schema", "json_schema": answer_schema()}});
        for format in [
            formats::qwen3(CallSyntax::Json),
            formats::glm(),
            formats::gemma4(),
            formats::minimax_m2(),
            formats::deepseek_v4_1(),
            formats::plain(),
        ] {
            assert_eq!(
                answer(&format, false),
                Some(alone.clone()),
                "{}",
                format.name()
            );
        }
    }

    #[test]
    fn with_the_thought_open_the_answer_follows_the_prefix_a_forced_call_takes() {
        // The prefix is the forced call's: the thought's own markers writable, so the model
        // closes its thought where the engine runs the grammar from the first token; every other
        // terminal and the syntax's inner tags not; nothing owed at its end.
        let schema = value!({"type": "json_schema", "json_schema": answer_schema()});
        assert_eq!(
            answer(&formats::qwen3(CallSyntax::Json), true),
            Some(value!({"format": {"type": "sequence", "elements": [
                {"type": "any_text", "excludes": ["<tool_call>", "</tool_call>"]},
                schema,
            ]}}))
        );
        let tools = weather_tools();
        for format in [
            formats::glm(),
            formats::gemma4(),
            formats::minimax_m2(),
            formats::deepseek_v4_1(),
        ] {
            let call = format
                .grammar(&tools, true, true)
                .expect("a grammar")
                .payload();
            assert_eq!(
                answer(&format, true),
                Some(value!({"format": {"type": "sequence", "elements": [
                    call["format"]["elements"][0].clone(),
                    schema,
                ]}})),
                "{}",
                format.name()
            );
        }
        // A table without a thought has no prefix.
        assert_eq!(
            answer(&formats::plain(), true),
            answer(&formats::plain(), false)
        );
    }

    #[test]
    fn kimi_k3s_answer_is_its_response_region_then_the_end_of_the_turn() {
        // The turn is regions: the answer opens the response region from the turn's own state,
        // and closes it and the turn. The prompt opens the thought, so the answer's way in closes
        // it first, as the block of calls' does.
        let closed = value!({"format": {
            "type": "tag",
            "begin": "<|open|>response<|sep|>",
            "content": {"type": "json_schema", "json_schema": answer_schema()},
            "end": "<|close|>response<|sep|><|close|>message<|sep|>",
        }});
        assert_eq!(answer(&formats::kimi_k3(), false), Some(closed.clone()));
        let mut open = closed;
        open["format"]["begin"] = Value::from("<|close|>think<|sep|><|open|>response<|sep|>");
        assert_eq!(answer(&formats::kimi_k3(), true), Some(open));
    }

    #[test]
    fn inklings_answer_is_a_text_message() {
        // The output starts after the message's opener, where the text marker makes the message
        // the answer, and the message's close ends it. The generation prompt opens no thought; a
        // thought a prompt left open reaches a text message only through the next message's
        // header, which reads calls and is no framing, so it has no answer tag.
        assert_eq!(
            answer(&formats::inkling(), false),
            Some(value!({"format": {
                "type": "tag",
                "begin": "<|content_text|>",
                "content": {"type": "json_schema", "json_schema": answer_schema()},
                "end": "<|end_message|>",
            }}))
        );
        assert_eq!(answer(&formats::inkling(), true), None);
    }

    #[test]
    fn a_table_without_content_has_no_answer_tag_and_a_thought_closing_into_its_own_content_leads_it(
    ) {
        // xLAM's one state reads calls, and its answer is whatever is not a list. MiniMax M3's
        // thought closes into a content state of its own, not where it opened: as for a forced
        // call, the answer admits the thought the model may write first, and the open thought
        // takes the prefix with its close kept writable.
        assert_eq!(answer(&formats::xlam(), false), None);
        let closed = answer(&formats::minimax_m3(), false).expect("an answer tag");
        assert_eq!(closed["format"]["type"], "sequence");
        assert_eq!(closed["format"]["elements"][0]["type"], "optional");
        assert_eq!(
            closed["format"]["elements"][0]["content"]["elements"][1],
            value!({"type": "const_string", "value": "</mm:think>"})
        );
        assert_eq!(closed["format"]["elements"][1]["type"], "json_schema");
        let open = answer(&formats::minimax_m3(), true).expect("an answer tag");
        assert_eq!(
            open["format"]["elements"][0],
            value!({"type": "any_text", "excludes": [
                "<tool_call>", "</tool_call>", "<invoke name=\"", "</invoke>", "]<]minimax[>[",
            ]})
        );
        assert_eq!(open["format"]["elements"][1]["type"], "json_schema");
    }

    #[test]
    fn a_json_answers_sample_reads_back_as_that_content_and_no_call() {
        let json = "{\"city\": \"Paris\"}";
        for (format, prompt, sample) in [
            (formats::qwen3(CallSyntax::Json), "", json.to_string()),
            (
                formats::qwen3(CallSyntax::Json),
                "<|im_start|>assistant\n<think>\n",
                format!("Paris, then.\n</think>\n\n{json}"),
            ),
            (formats::gemma4(), "", json.to_string()),
            (
                formats::kimi_k3(),
                "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>",
                format!(
                    "<|close|>think<|sep|><|open|>response<|sep|>{json}<|close|>response<|sep|>\
                     <|close|>message<|sep|>"
                ),
            ),
            (
                formats::inkling(),
                "<|message_model|>",
                format!("<|content_text|>{json}<|end_message|>"),
            ),
        ] {
            let message = read(&format, &[], prompt, &sample);
            assert_eq!(
                message["content"].as_str().map(str::trim),
                Some(json),
                "{}",
                format.name()
            );
            assert_eq!(message["calls"], value!([]), "{}", format.name());
        }
    }

    /// A ChatML turn up to the assistant's opener, the user asking for the weather.
    const CHATML_TURN: &str =
        "<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n";

    #[test]
    fn the_prompts_replay_says_whether_the_thought_is_open_wherever_the_prompt_ends() {
        // Read where the prompt ends, not by whether it ends at the opener: Qwen 3.5's generation
        // prompt opens the thought after the opener, and with thinking off opens and closes it
        // there, as Nemotron 3's does without the newlines; Qwen3 writes its own `<think>`.
        let qwen3_5 = formats::Table::Qwen3Tagged.format();
        assert!(qwen3_5.reasoning_open(&format!("{CHATML_TURN}<think>\n")));
        assert!(!qwen3_5.reasoning_open(&format!("{CHATML_TURN}<think>\n\n</think>\n\n")));
        assert!(!qwen3_5.reasoning_open(&format!("{CHATML_TURN}<think></think>")));
        assert!(!formats::qwen3(CallSyntax::Json).reasoning_open(CHATML_TURN));
        // A thought quoted in an earlier turn moves nothing: the replay starts at the last opener.
        assert!(!qwen3_5.reasoning_open(
            "<|im_start|>user\nWhat is <think>?<|im_end|>\n<|im_start|>assistant\n"
        ));
        // DeepSeek's templates open the thought after `<｜Assistant｜>`, R1's distills with a
        // newline after it; Kimi K3's after its message tag.
        let deepseek = formats::deepseek_v4_1();
        assert!(deepseek.reasoning_open("<｜User｜>Weather in Paris?<｜Assistant｜><think>"));
        assert!(deepseek.reasoning_open("<｜User｜>Weather in Paris?<｜Assistant｜><think>\n"));
        assert!(!deepseek.reasoning_open("<｜User｜>Hi<｜Assistant｜><think>\n\n</think>\n\n"));
        assert!(formats::kimi_k3()
            .reasoning_open("<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>"));
        // A table without a thought never has one open.
        assert!(!formats::qwen2_5().reasoning_open(&format!("{CHATML_TURN}<think>\n")));
    }

    #[test]
    fn a_prompt_left_inside_the_thought_gets_the_prefixed_tag_not_a_refusal() {
        // Where the replay leaves the model in its thought, the forced call and the JSON answer
        // follow the prefix, so the model finishes its thought first; neither is refused.
        let tools = weather_tools();
        let prompt = format!("{CHATML_TURN}<think>\n");
        for table in [formats::Table::Qwen3Tagged, formats::Table::Qwen3] {
            let format = table.format();
            let reasoning_open = format.reasoning_open(&prompt);
            assert!(reasoning_open, "{table:?}");
            for one_call in [false, true] {
                let call = format
                    .grammar_for(
                        &tools,
                        Options {
                            at_least_one: true,
                            reasoning_open,
                            one_call,
                        },
                    )
                    .expect("a tag")
                    .payload();
                assert_eq!(call["format"]["type"], "sequence", "{table:?}");
                assert_eq!(call["format"]["elements"][0]["type"], "any_text");
            }
            let answer = answer(&format, reasoning_open).expect("an answer");
            assert_eq!(answer["format"]["elements"][0]["type"], "any_text");
        }
        let deepseek = formats::deepseek_v4_1();
        let reasoning_open = deepseek.reasoning_open("<｜User｜>q<｜Assistant｜><think>\n");
        let call = deepseek
            .grammar(&tools, true, reasoning_open)
            .expect("a tag")
            .payload();
        assert_eq!(call["format"]["elements"][0]["type"], "any_text");
    }

    #[test]
    fn the_replay_agrees_with_where_the_engine_starts_the_output() {
        // The engine replays the same walk: the output is reasoning exactly where the replay says
        // the thought is open.
        for (format, prompt) in [
            (
                formats::Table::Qwen3Tagged.format(),
                format!("{CHATML_TURN}<think>\n"),
            ),
            (
                formats::Table::Qwen3Tagged.format(),
                format!("{CHATML_TURN}<think>\n\n</think>\n\n"),
            ),
            (formats::qwen3(CallSyntax::Json), CHATML_TURN.to_string()),
            (
                formats::deepseek_v4_1(),
                "<｜User｜>q<｜Assistant｜><think>".to_string(),
            ),
            (
                formats::kimi_k3(),
                "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>".to_string(),
            ),
            (formats::inkling(), "<|message_model|>".to_string()),
        ] {
            let read = read(&format, &[], &prompt, "Paris.");
            assert_eq!(
                read["reasoning"].is_string(),
                format.reasoning_open(&prompt),
                "{}: {prompt:?}",
                format.name()
            );
        }
    }

    /// A table whose template writes a thought before every call and answer: `<think>` blocks
    /// and `<tool_call>` blocks of JSON calls, everything else content, the turn's thought first.
    fn thinking_first() -> Format {
        Format::new("thinking_first")
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
            .calls(CallSyntax::Json)
            .thinks_first()
    }

    /// The thought a table that thinks first writes before a tag, as a payload node.
    fn thought_node(excludes: Value) -> Value {
        value!({
            "type": "tag",
            "begin": "<think>",
            "content": {"type": "any_text", "excludes": excludes},
            "end": "</think>",
        })
    }

    #[test]
    fn a_table_that_thinks_first_writes_the_thought_before_the_call() {
        // The template writes a thought before every call, so the forced call's tag does too
        // when the prompt has not opened it: the thought's opener, text that writes no marker, its
        // close, then the call the table's syntax writes. With the thought open, the prefix as for
        // any table; nothing is owed there, as an engine that runs a reasoning parser has read the
        // thought's close before the grammar starts.
        let tools = weather_tools();
        let format = thinking_first();
        let excludes = value!(["<think>", "</think>", "<tool_call>", "</tool_call>"]);
        let plain = formats::qwen3(CallSyntax::Json);
        for one_call in [false, true] {
            let options = Options {
                at_least_one: true,
                reasoning_open: false,
                one_call,
            };
            let calls = plain
                .grammar_for(&tools, options)
                .expect("a grammar")
                .to_json();
            assert_eq!(
                format
                    .grammar_for(&tools, options)
                    .expect("a grammar")
                    .payload(),
                value!({"format": {"type": "sequence", "elements": [
                    thought_node(excludes.clone()),
                    calls,
                ]}})
            );
        }
        assert_eq!(
            format.grammar(&tools, true, true),
            plain.grammar(&tools, true, true)
        );
        // The same rows without the property, Qwen3's, give the tag alone.
        assert_eq!(
            plain
                .grammar(&tools, true, false)
                .expect("a grammar")
                .payload()["format"]["type"],
            "triggered_tags"
        );
    }

    #[test]
    fn a_table_that_thinks_first_writes_the_thought_before_the_answer() {
        // The answer where the output starts in content: the thought, then the schema. With the
        // thought open, the prefix and the schema, as for any table.
        let format = thinking_first();
        let schema = value!({"type": "json_schema", "json_schema": answer_schema()});
        assert_eq!(
            answer(&format, false),
            Some(value!({"format": {"type": "sequence", "elements": [
                thought_node(value!(["<think>", "</think>", "<tool_call>", "</tool_call>"])),
                schema,
            ]}}))
        );
        assert_eq!(
            answer(&format, true),
            answer(&formats::qwen3(CallSyntax::Json), true)
        );
    }

    #[test]
    fn a_turn_of_regions_that_thinks_first_closes_the_thought_then_goes_in() {
        // Kimi K3's regions, thinking first: the thought is written whole, and the block of
        // calls and the answer are entered from where its close leaves the engine, the turn's
        // own state, as Kimi K3's prompt does not open the thought here.
        let tools = weather_tools();
        let format = formats::kimi_k3().thinks_first();
        let calls = format
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        assert_eq!(calls["format"]["type"], "sequence");
        assert_eq!(
            calls["format"]["elements"][0]["begin"],
            "<|open|>think<|sep|>"
        );
        assert_eq!(
            calls["format"]["elements"][0]["end"],
            "<|close|>think<|sep|>"
        );
        assert_eq!(
            calls["format"]["elements"][1],
            formats::kimi_k3()
                .grammar(&tools, true, false)
                .expect("a grammar")
                .to_json()
        );
        let answer = answer(&format, false).expect("an answer");
        assert_eq!(
            answer["format"]["elements"][1]["begin"],
            "<|open|>response<|sep|>"
        );
        // With the prompt's thought open, Kimi K3's tags as they are.
        assert_eq!(
            format.grammar(&tools, true, true),
            formats::kimi_k3().grammar(&tools, true, true)
        );
    }

    #[test]
    fn a_thought_written_first_opens_where_the_output_starts() {
        // The thought opens from a message's header, not from the turn's own state where the
        // output starts: no lead the table has writes it first there, so the tags give `None`
        // rather than a thought the model could not open.
        let tools = weather_tools();
        let headed = Format::new("headed")
            .terminal("start", "<|start|>")
            .terminal("think", "<|think|>")
            .terminal("end", "<|end|>")
            .terminal("calls_open", "<|calls|>")
            .terminal("calls_close", "<|/calls|>")
            .terminal("call_open", "<|call|>")
            .terminal("call_close", "<|/call|>")
            .state("between", Emits::Wrapper)
            .state("header", Emits::Wrapper)
            .state("reasoning", Emits::Reasoning)
            .state("calls", Emits::Wrapper)
            .state("call", Emits::Arguments)
            .transition("between", "start", "header")
            .transition("header", "think", "reasoning")
            .transition("reasoning", "end", "header")
            .transition("header", "calls_open", "calls")
            .transition("calls", "call_open", "call")
            .transition("call", "call_close", "calls")
            .transition("calls", "calls_close", "header")
            .calls(CallSyntax::Attributed(AttributedTags::XTML));
        assert!(headed.grammar(&tools, true, false).is_some());
        assert!(headed
            .clone()
            .thinks_first()
            .grammar(&tools, true, false)
            .is_none());
        assert_eq!(
            headed.clone().thinks_first().grammar(&tools, true, true),
            headed.grammar(&tools, true, true)
        );
    }

    #[test]
    fn a_thought_the_model_may_write_first_is_admitted_before_a_forced_call() {
        // MiniMax M3 leaves the thought to the model: a row from the turn's start opens it and
        // another closes an empty one. A forced call, written at once, would take the thought
        // away, so the tag admits the thought, or the bare close, or nothing, before the block.
        // A table whose prompt opens the thought (GLM) adds nothing; neither does the open thought,
        // whose prefix the tag takes instead.
        let tools = weather_tools();
        let closed = formats::minimax_m3()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let first = &closed["format"]["elements"][0];
        assert_eq!(first["type"], "optional");
        assert_eq!(first["content"]["type"], "or");
        let choices = first["content"]["elements"]
            .as_array()
            .expect("two choices");
        assert_eq!(choices[0]["type"], "tag");
        assert_eq!(choices[0]["begin"], "<mm:think>");
        assert_eq!(choices[0]["end"], "</mm:think>");
        assert_eq!(
            choices[1],
            value!({"type": "const_string", "value": "</mm:think>"})
        );
        assert_eq!(
            closed["format"]["elements"][1]["elements"][1]["at_least_one"],
            true
        );
        let open = formats::minimax_m3()
            .grammar(&tools, true, true)
            .expect("a grammar")
            .payload();
        assert_eq!(open["format"]["elements"][0]["type"], "any_text");
        assert_eq!(
            open["format"]["elements"][1],
            closed["format"]["elements"][1]
        );
        let glm = formats::glm()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        assert_eq!(glm["format"]["type"], "triggered_tags");
    }

    #[test]
    fn minimax_m3_forced_call_samples_read_back_with_the_thought() {
        // What the tag admits is what the table reads: the thought, the bare close of an empty
        // one, or nothing, then the block with the separator before every tag.
        let tools = weather_tools();
        let block = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">\
                     ]<]minimax[>[<city>Paris]<]minimax[>[</city>]<]minimax[>[</invoke>\n\
                     ]<]minimax[>[</tool_call>";
        // The block without the separator before its opener, as the grammar forced it until now.
        let bare = &block["]<]minimax[>[".len()..];
        for (before, block, reasoning) in [
            (
                "<mm:think>Paris it is.</mm:think>",
                block,
                value!("Paris it is."),
            ),
            ("</mm:think>", block, Value::Null),
            ("", block, Value::Null),
            (
                "<mm:think>Paris it is.</mm:think>",
                bare,
                value!("Paris it is."),
            ),
            ("", bare, Value::Null),
        ] {
            let read_back = read(
                &formats::minimax_m3(),
                &tools,
                "",
                &format!("{before}{block}"),
            );
            assert_eq!(read_back["reasoning"], reasoning, "{before:?}");
            assert_eq!(
                read_back["calls"],
                value!([{"name": "get_weather", "arguments": {"city": "Paris"}}]),
                "{before:?}"
            );
        }
    }

    #[test]
    fn a_thought_first_samples_read_back_as_the_thought_and_the_call() {
        let tools = weather_tools();
        let sample = "<think>Paris it is.</think><tool_call>\n{\"name\": \"get_weather\", \
                      \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        assert_eq!(
            read(&thinking_first(), &tools, "", sample),
            value!({
                "content": null,
                "reasoning": "Paris it is.",
                "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
            })
        );
    }

    /// A table spelled as a response template's fields read, on abstract tags: the text outside
    /// every field is framing; a thought between `<t>` and `</t>`; an answer after `<a>` that
    /// `</a>` closes, or the stop token `<s>`, whose row comes first; and calls of JSON between
    /// `<c>` and `</c>`.
    fn framed() -> Format {
        Format::new("framed")
            .terminal("think_open", "<t>")
            .terminal("think_close", "</t>")
            .terminal("answer_open", "<a>")
            .terminal("answer_close", "</a>")
            .terminal("stop", "<s>")
            .terminal("call_open", "<c>")
            .terminal("call_close", "</c>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("content", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("outside", "think_open", "reasoning")
            .transition("reasoning", "think_close", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "stop", "outside")
            .transition("content", "answer_close", "outside")
            .transition("outside", "call_open", "call")
            .transition("call", "call_close", "outside")
            .calls(CallSyntax::Json)
            .framing("outside")
            .stop_tokens(["<s>"])
    }

    #[test]
    fn a_framing_state_holds_no_answer_and_a_stop_token_ends_one() {
        // The answer is the first content state the template keeps, not the framing the turn
        // starts in; a stop token closes it, so the tag ends at the model's stop.
        let schema = answer_schema();
        let json_schema = value!({"type": "json_schema", "json_schema": schema});
        let format = framed();
        assert_eq!(
            answer(&format, false),
            Some(value!({"format": {
                "type": "tag",
                "begin": "<a>",
                "content": json_schema,
                "end": "",
            }}))
        );
        // With the thought open, the way in is its close, then the row into the answer.
        assert_eq!(
            answer(&format, true).map(|payload| payload["format"]["begin"].clone()),
            Some(value!("</t><a>"))
        );
        // Without the stop token, the answer's own close ends it.
        let format = Format::new("framed")
            .terminal("answer_open", "<a>")
            .terminal("answer_close", "</a>")
            .state("outside", Emits::Content)
            .state("content", Emits::Content)
            .transition("outside", "answer_open", "content")
            .transition("content", "answer_close", "outside")
            .framing("outside");
        assert_eq!(
            answer(&format, false).map(|payload| payload["format"]["end"].clone()),
            Some(value!("</a>"))
        );
        let json = "{\"city\": \"Paris\"}";
        let message = read(&framed(), &[], "", &format!("<a>{json}"));
        assert_eq!(message["content"], value!(json));
        assert_eq!(message["calls"], value!([]));
    }

    #[test]
    fn a_regions_close_and_a_ways_step_are_never_a_stop_token() {
        // The stop row out of the answer comes first in the table, but a tag cannot write the
        // stop token: the region's close is the answer's own, and no path steps on the token.
        let format = framed();
        let content = format.answer_state().expect("an answer state");
        let region = format
            .region(content, |state| format.is_call_state(state))
            .expect("the answer's region");
        assert_eq!((region.open, region.close), ("<a>", "</a>"));
        assert_eq!(format.close_of(content), Some(("</a>", 0)));
        assert_eq!(
            format.path(content, 3),
            Some(vec!["</a>", "<c>"]),
            "out through the answer's close and the framing, not the stop token"
        );
        assert!(format.stopped(content));
        assert!(!format.stopped(1));
    }

    #[test]
    fn a_way_in_passes_through_the_framing_between_two_markers() {
        // The thought closes into framing of its own, and a second marker leads on into the
        // framing the turn starts in, from where the answer opens: the way in is all three.
        let format = Format::new("two markers")
            .terminal("think_open", "<t>")
            .terminal("think_close", "</t>")
            .terminal("next", "<n>")
            .terminal("answer_open", "<a>")
            .terminal("answer_close", "</a>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("gap", Emits::Content)
            .state("content", Emits::Content)
            .transition("outside", "think_open", "reasoning")
            .transition("reasoning", "think_close", "gap")
            .transition("gap", "next", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "answer_close", "outside")
            .framing("outside")
            .framing("gap");
        assert_eq!(format.validate(), Ok(()));
        assert_eq!(format.path(2, 3), Some(vec!["<n>", "<a>"]));
        assert_eq!(
            answer(&format, true).map(|payload| payload["format"]["begin"].clone()),
            Some(value!("</t><n><a>"))
        );
        let json = "{\"city\": \"Paris\"}";
        let message = read(&format, &[], "<t>", &format!("plan</t><n><a>{json}</a>"));
        assert_eq!(message["content"], value!(json));
        assert_eq!(message["reasoning"], value!("plan"));
    }

    #[test]
    fn a_layouts_text_before_an_answer_stands_before_the_terminal_that_opens_it() {
        // The way in from the open thought is its close and two rows, `<n>` and the answer's
        // opener `<a>`: the layout's text after a thought follows the close, and its text before
        // an answer comes right before `<a>`, not before the first row.
        let format = Format::new("two markers")
            .terminal("think_open", "<t>")
            .terminal("think_close", "</t>")
            .terminal("next", "<n>")
            .terminal("answer_open", "<a>")
            .terminal("answer_close", "</a>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("gap", Emits::Content)
            .state("content", Emits::Content)
            .transition("outside", "think_open", "reasoning")
            .transition("reasoning", "think_close", "gap")
            .transition("gap", "next", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "answer_close", "outside")
            .framing("outside")
            .framing("gap")
            .with_layout(Layout {
                thought: Some(Message {
                    after: "\n".into(),
                    ..Message::default()
                }),
                answer: Some(Message {
                    before: " ".into(),
                    ..Message::default()
                }),
                ..Layout::default()
            });
        assert_eq!(
            answer(&format, true).map(|payload| payload["format"]["begin"].clone()),
            Some(value!("</t>\n<n> <a>"))
        );
        assert_eq!(
            answer(&format, false).map(|payload| payload["format"]["begin"].clone()),
            Some(value!(" <a>"))
        );
    }

    /// A thought the template closes two ways, `<u>` or `<e></t>`, the row on `<u>` first in the
    /// table; `<e>` alone closes the answer; calls of JSON between `<c>` and `</c>`. The template
    /// writes the thought before every call and answer.
    fn closed_two_ways() -> Format {
        Format::new("closed two ways")
            .terminal("think_open", "<t>")
            .terminal("think_close", "<u>")
            .terminal("think_end", "<e></t>")
            .terminal("answer_open", "<a>")
            .terminal("answer_close", "<e>")
            .terminal("call_open", "<c>")
            .terminal("call_close", "</c>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("content", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("outside", "think_open", "reasoning")
            .transition("reasoning", "think_close", "outside")
            .transition("reasoning", "think_end", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "answer_close", "outside")
            .transition("outside", "call_open", "call")
            .transition("call", "call_close", "outside")
            .calls(CallSyntax::Json)
            .framing("outside")
            .thinks_first()
    }

    #[test]
    fn a_thought_closed_more_than_one_way_is_closed_as_its_layout_says_the_template_writes_it() {
        let format = closed_two_ways();
        assert_eq!(format.validate(), Ok(()));
        let thought_end = |format: &Format| {
            answer(format, false).map(|payload| payload["format"]["elements"][0]["end"].clone())
        };
        // Without a layout that names one, the tags write the first row out of the thought.
        assert_eq!(thought_end(&format), Some(value!("<u>")));
        // The layout names the close the chat template renders: the thought written first ends
        // with it, and with the thought open the way into the answer starts with it.
        let format = format.with_layout(Layout {
            thought: Some(Message {
                close: Some("<e></t>".into()),
                ..Message::default()
            }),
            ..Layout::default()
        });
        assert_eq!(thought_end(&format), Some(value!("<e></t>")));
        assert_eq!(
            answer(&format, true).map(|payload| payload["format"]["begin"].clone()),
            Some(value!("<e></t><a>"))
        );
        // With the thought open, the prefix a forced call takes keeps both of the thought's closes
        // writable, and `<e>` inside the one the template renders, but no other marker.
        let wrapped = format
            .grammar(&weather_tools(), true, true)
            .expect("a grammar")
            .to_json();
        assert_eq!(
            wrapped["elements"][0],
            value!({"type": "any_text", "excludes": ["<a>", "<c>", "</c>"]})
        );
        // Either close ends the thought, and `<e>` alone is the thought's text.
        let json = "{\"city\": \"Paris\"}";
        let message = read(&format, &[], "<t>", &format!("plan<e></t><a>{json}<e>"));
        assert_eq!(message["reasoning"], value!("plan"));
        assert_eq!(message["content"], value!(json));
        let message = read(&format, &[], "<t>", "plan<e>more<u><a>x<e>");
        assert_eq!(message["reasoning"], value!("plan<e>more"));
        assert_eq!(message["content"], value!("x"));
    }

    #[test]
    fn an_answer_closed_more_than_one_way_ends_as_its_layout_says_the_template_writes_it() {
        // `closed_two_ways` with its answer closed by `</a>` as well, the row on `<e>` first.
        let format = closed_two_ways().terminal("answer_end", "</a>");
        let format = format.transition("content", "answer_end", "outside");
        assert_eq!(format.validate(), Ok(()));
        let answer_end = |format: &Format| {
            answer(format, false).map(|payload| payload["format"]["elements"][1]["end"].clone())
        };
        // Without a layout that names one, the answer ends with the first row out of it.
        assert_eq!(answer_end(&format), Some(value!("<e>")));
        // The layout names the close the chat template renders, and the answer ends with it.
        let format = format.with_layout(Layout {
            answer: Some(Message {
                close: Some("</a>".into()),
                ..Message::default()
            }),
            ..Layout::default()
        });
        assert_eq!(answer_end(&format), Some(value!("</a>")));
    }

    /// Every node of `payload` of `kind`, depth first.
    fn nodes_of<'v>(payload: &'v Value, kind: &str, out: &mut Vec<&'v Value>) {
        match payload {
            Value::Object(map) => {
                if map.get("type").and_then(Value::as_str) == Some(kind) {
                    out.push(payload);
                }
                for member in map.values() {
                    nodes_of(member, kind, out);
                }
            }
            Value::Array(items) => {
                for item in items {
                    nodes_of(item, kind, out);
                }
            }
            _ => {}
        }
    }

    /// `grammar` with every token written back as its text: each constant string a marker
    /// token cut apart joined again, so a bounded tag compares with the tag without tokens.
    fn unbound(grammar: Grammar) -> Grammar {
        let in_tag = |tag: Tag| Tag {
            content: Box::new(unbound(*tag.content)),
            ..tag
        };
        match grammar {
            Grammar::Token(text) => Grammar::ConstString(text),
            Grammar::Sequence(elements) => {
                let cut = elements.iter().any(|e| matches!(e, Grammar::Token(_)))
                    && elements
                        .iter()
                        .all(|e| matches!(e, Grammar::Token(_) | Grammar::ConstString(_)));
                if cut {
                    let mut text = String::new();
                    for element in elements {
                        if let Grammar::Token(piece) | Grammar::ConstString(piece) = element {
                            text.push_str(&piece);
                        }
                    }
                    return Grammar::ConstString(text);
                }
                Grammar::Sequence(elements.into_iter().map(unbound).collect())
            }
            Grammar::Or(elements) => Grammar::Or(elements.into_iter().map(unbound).collect()),
            Grammar::Optional(content) => Grammar::Optional(Box::new(unbound(*content))),
            Grammar::Star(content) => Grammar::Star(Box::new(unbound(*content))),
            Grammar::Plus(content) => Grammar::Plus(Box::new(unbound(*content))),
            Grammar::Tag(tag) => Grammar::Tag(in_tag(tag)),
            Grammar::TriggeredTags {
                triggers,
                tags,
                at_least_one,
                stop_after_first,
            } => Grammar::TriggeredTags {
                triggers,
                tags: tags.into_iter().map(in_tag).collect(),
                at_least_one,
                stop_after_first,
            },
            Grammar::TagsWithSeparator {
                tags,
                separator,
                at_least_one,
                stop_after_first,
            } => Grammar::TagsWithSeparator {
                tags: tags.into_iter().map(in_tag).collect(),
                separator,
                at_least_one,
                stop_after_first,
            },
            other => other,
        }
    }

    #[test]
    fn a_marker_that_is_one_token_is_written_as_that_token_inside_a_tag() {
        // Gemma 4's string delimiter is one token of its vocabulary: inside each call's tag,
        // every string stands between that token, not its spelling in other tokens; the tag's
        // begin and end, the trigger and the prefix's excludes stay text, as the engines take
        // them. Written back as text, the tag is the one without tokens.
        let tools = weather_tools();
        let plain = formats::gemma4();
        let bounded = formats::gemma4().marker_tokens(["<|\"|>", "<|tool_call>", "<tool_call|>"]);
        for (at_least_one, reasoning_open, one_call) in [
            (true, false, false),
            (true, true, false),
            (false, false, true),
        ] {
            let options = Options {
                at_least_one,
                reasoning_open,
                one_call,
            };
            let before = plain.grammar_for(&tools, options).expect("a grammar");
            let after = bounded.grammar_for(&tools, options).expect("a grammar");
            assert_eq!(unbound(after.clone()), before);
            let payload = after.payload();
            let (mut tokens, mut strings) = (Vec::new(), Vec::new());
            nodes_of(&payload, "token", &mut tokens);
            nodes_of(&payload, "const_string", &mut strings);
            assert!(!tokens.is_empty());
            assert!(tokens.iter().all(|token| token["token"] == "<|\"|>"));
            assert!(strings
                .iter()
                .all(|string| !string["value"].as_str().unwrap_or("").contains("<|\"|>")));
            let tags = if reasoning_open {
                &payload["format"]["elements"][1]
            } else {
                &payload["format"]
            };
            assert_eq!(tags["triggers"], value!(["<|tool_call>"]));
            assert_eq!(tags["tags"][0]["begin"], "<|tool_call>call:get_weather{");
        }
        // The answer holds no marker, and is the same.
        assert_eq!(answer(&bounded, true), answer(&plain, true));
    }

    #[test]
    fn a_token_inside_a_longer_marker_is_cut_out_of_it() {
        // DeepSeek V4.1's block opener begins with one token, `<｜DSML｜`: the opener, written
        // before the invokes, takes that token and the rest as text; the close, which does not
        // hold it, stays text, and the invokes' tags keep their begins and ends.
        let tools = weather_tools();
        let plain = formats::deepseek_v4_1();
        let bounded = formats::deepseek_v4_1().marker_tokens(["<｜DSML｜"]);
        let grammar = bounded.grammar(&tools, true, false).expect("a grammar");
        assert_eq!(
            Some(unbound(grammar.clone())),
            plain.grammar(&tools, true, false)
        );
        let payload = grammar.payload();
        assert_eq!(
            payload["format"]["elements"][0],
            value!({"type": "sequence", "elements": [
                {"type": "const_string", "value": "\n\n"},
                {"type": "token", "token": "<｜DSML｜"},
                {"type": "const_string", "value": " calls>\n"},
            ]})
        );
        assert_eq!(
            payload["format"]["elements"][2],
            value!({"type": "const_string", "value": "</｜DSML｜ calls>"})
        );
        assert_eq!(
            payload["format"]["elements"][1]["tags"][0]["begin"],
            "<｜DSML｜ invoke name=\"get_weather\">\n"
        );
        // Kimi K3's tags are made of its structural tokens: every argument tag inside a call
        // writes them as tokens, and the block's ways in stay text.
        let kimi = formats::kimi_k3().marker_tokens(["<|open|>", "<|close|>", "<|sep|>"]);
        let grammar = kimi.grammar(&tools, true, true).expect("a grammar");
        assert_eq!(
            Some(unbound(grammar.clone())),
            formats::kimi_k3().grammar(&tools, true, true)
        );
        let payload = grammar.payload();
        let mut tokens = Vec::new();
        nodes_of(&payload, "token", &mut tokens);
        assert!(tokens.iter().any(|token| token["token"] == "<|sep|>"));
        assert_eq!(
            payload["format"]["triggers"],
            value!([
                "<|open|>tools<|sep|>",
                "<|close|>think<|sep|>",
                "<|close|>response<|sep|>"
            ])
        );
    }

    #[test]
    fn the_marker_tokens_are_the_tables_own_the_longest_first() {
        // A token no marker holds is no marker's; the empty text is none; of two, the longer is
        // tried first, so a token inside a longer one never cuts it.
        let format = formats::kimi_k3().marker_tokens([
            "<|sep|>",
            "hello",
            "",
            "<|open|>",
            "<|close|>message<|sep|>",
            "<|sep|>",
        ]);
        assert_eq!(
            format.marker_token_texts(),
            ["<|close|>message<|sep|>", "<|open|>", "<|sep|>"]
        );
        assert!(formats::kimi_k3().marker_token_texts().is_empty());
        // The syntax's own tags count: Gemma 4's string delimiter is no terminal.
        assert_eq!(
            formats::gemma4()
                .marker_tokens(["<|\"|>"])
                .marker_token_texts(),
            ["<|\"|>"]
        );
    }

    #[test]
    fn every_syntax_says_its_inner_tags() {
        use crate::{pythonic::Dialect, tagged::attributed::Tags as AttributedTags};
        assert_eq!(
            CallSyntax::Tagged(
                crate::tagged::Spelling::Json,
                crate::tagged::Placement::OwnLine
            )
            .tag_texts(),
            ["<function=", "<parameter=", "</parameter>", "</function>"]
        );
        assert_eq!(
            CallSyntax::Keyed(KeyedTags::PLAIN).tag_texts(),
            ["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"]
        );
        assert_eq!(
            CallSyntax::Attributed(AttributedTags::DSML).tag_texts(),
            [
                AttributedTags::DSML.parameter_open,
                AttributedTags::DSML.parameter_close
            ]
        );
        assert_eq!(
            CallSyntax::Attributed(AttributedTags::MINIMAX_M2).tag_texts(),
            ["<parameter name=\"", "</parameter>"]
        );
        assert_eq!(
            CallSyntax::Attributed(AttributedTags::XTML).tag_texts(),
            [
                "<|open|>argument key=\"",
                "<|close|>argument<|sep|>",
                "<|sep|>"
            ]
        );
        assert_eq!(
            CallSyntax::Gemma(crate::gemma::Spelling::GEMMA4).tag_texts(),
            ["<|\"|>"]
        );
        assert_eq!(
            CallSyntax::NamedJson("<|content_invoke_tool_json|>").tag_texts(),
            ["<|content_invoke_tool_json|>"]
        );
        for syntax in [
            CallSyntax::Json,
            CallSyntax::JsonList,
            CallSyntax::Xml,
            CallSyntax::Pythonic(Dialect::OLMO3),
        ] {
            assert!(syntax.tag_texts().is_empty(), "{syntax:?}");
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
