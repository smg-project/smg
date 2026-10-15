//! The grammar of a whole turn of a table whose calls are addressed to a tool, from the first
//! token the model writes: its thought, then either an answer or its calls.
//!
//! A call is addressed to a tool where the table gives it as data ([`CallSyntax::Shaped`]) in one
//! shape: after the region's terminal, the tool's name, read up to whitespace or `<`, and a
//! literal that closes the header; then the parameters, each a literal that opens the key, the
//! key, the key's close and a newline, the value as raw text (`.*?`), then a newline and the
//! value's close, or, where the next key closes a value too, a newline before the next key's
//! opening. In a neutral spelling, a turn is
//!
//! ```text
//! <|via|>notes<|body|>THOUGHT<|stop|><|head|>assistant
//! ( <|via|>reply<|body|>ANSWER<|done|>
//! | CALL (<|stop|><|head|>assistant CALL)* <eos> )
//! ```
//!
//! where `CALL` is ` to=NAME<|via|>tool<|body|>` and then the parameters, each the key's tag
//! `<arg name="KEY">`, a newline, the value, a newline and `</arg>`, a newline allowed before
//! each. Every text of that comes from the table's rows and the call's shape
//! ([`Format::turn_spelling`]), none from a model:
//! - **the thought** is the markers into the reasoning state from the turn's own state, then text
//!   that is not empty and does not start with a character any of its closes starts with, ended
//!   by any close of the reasoning state (every row out of it but a stop's), longest first; a stop
//!   token never ends a turn inside its thought, so the text holds none of the table's stops;
//! - **the answer** is the markers into the content state from where the thought's closes leave
//!   the turn, then the request's answer format or free text, which holds none of the content
//!   state's closes, then, optional, the close that is a stop no call ends on;
//! - **a call** is the markers into the arguments state from there, the tool's name, the header's
//!   close, then the parameters; between two calls the row out of a call that is no stop, after
//!   the last the one that is (the engine stops there, and the call closes on it);
//! - **values**: a string is raw text that holds no call terminator (each row out of a call
//!   without the turn opener it may end with), no value close and, where the next key closes a
//!   value, no next key's opening; a string `pattern` xgrammar compiles is that regex instead; a
//!   string constant is itself, left out when it holds one of those; every other value is JSON of
//!   its part of the schema with the call terminators left out of its strings. Each JSON value
//!   whose objects do not list their properties by name also takes them listed by name, the
//!   order the model writes them in.
//!
//! The parameters are the schema's own, each at most once, required ones always, in the schema's
//! order or in the order of their names; free keys where `additionalProperties` allows them or
//! the schema declares none. Where the next key closes a value, a value but the last may end
//! there, and the lists are built by halving, so a call of n parameters holds each value O(log n)
//! times.
//!
//! Under a choice that leaves the model the answer (`auto`), a call may also name any tool other
//! than the request's declared ones, with any arguments: the table reads it back and the client
//! can tell the model the tool does not exist.
//!
//! Where the prompt leaves the model ([`Format::turn_position`]) decides where the grammar starts:
//! at the turn's start, inside a thought the prompt opened, after a thought the prompt closed (a
//! continued message: the turn holds one thought), or inside an answer the prompt opened (the rest
//! of a free answer). Anywhere else (inside a call, past text no field holds) no grammar starts.

mod values;

use std::collections::BTreeMap;

use openai_protocol::common::Tool;
use serde_json::{json, Map, Value};

use self::values::{Kind, Root, Source, ValueOption};
use super::{Bound, Grammar, Tag};
use crate::{
    format::{CallSyntax, Emits, Format},
    markers::{Piece, Scanner},
    shape::{Arguments, Atom, CallShape, Class, End, NameSource, Slot, ValueParser},
};

/// What a newline written between two parameters is: the template's.
const BETWEEN_PARAMETERS: &str = "\n";

/// The keys a schema that declares none takes, as xgrammar's XML styles write them.
const FREE_KEY: &str = "[a-zA-Z_][a-zA-Z0-9_]*";

/// The characters a call's name never holds, past which the header's close follows it: the name
/// is read up to whitespace or `<`.
const NAME_STOPS: &str = r"\s<";

/// The texts of a turn the table's rows declare; see the module documentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnSpelling {
    thought: Option<Thought>,
    answer: Option<Answer>,
    call_lead: String,
    header_close: String,
    key_open: String,
    key_close: String,
    value_close: String,
    next_key_closes: bool,
    separator: String,
    call_end: String,
    call_terminators: Vec<String>,
    stops: Vec<String>,
    /// What a tool's name, a parameter's key or a string constant may not hold: the value's close,
    /// the marker tokens, and the rows out of a call.
    delimiters: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Thought {
    open: String,
    ends: Vec<String>,
    first: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Answer {
    open: String,
    closes: Vec<String>,
    end: Option<String>,
}

/// Where the prompt leaves the model's turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnPosition {
    /// Nothing written past the turn's opener.
    Start,
    /// Inside a thought the prompt opened.
    InThought,
    /// Past a thought the prompt closed, in a message it continues.
    AfterThought,
    /// Inside an answer the prompt opened.
    InAnswer,
    /// Anywhere else: inside a call, a marker cut short, text no field holds.
    Elsewhere,
}

/// How the request's tool choice shapes the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnChoice {
    /// An answer or calls (`auto`, or `none` with nothing callable).
    Auto,
    /// One or more calls.
    Required,
    /// Exactly one call.
    One,
}

/// The request's own constraint on the answer.
#[derive(Clone, Copy, Debug)]
pub enum TurnAnswer<'a> {
    /// JSON of the schema.
    Json(&'a Value),
    Regex(&'a str),
    /// An EBNF grammar; only xgrammar's own syntax (`::=`) composes into a turn.
    Ebnf(&'a str),
}

impl TurnAnswer<'_> {
    /// Whether the turn's grammar can hold it: not an EBNF grammar in Lark's syntax (`rule:`).
    pub fn composes(&self) -> bool {
        match self {
            Self::Ebnf(grammar) => grammar.contains("::="),
            Self::Json(_) | Self::Regex(_) => true,
        }
    }
}

/// What one request asks of a turn's grammar.
#[derive(Clone, Copy, Debug)]
pub struct Turn<'a> {
    pub position: TurnPosition,
    /// Every tool the request declares: an undeclared call names none of them.
    pub declared: &'a [Tool],
    /// The tools the choice lets the model call.
    pub callable: &'a [Tool],
    pub choice: TurnChoice,
    /// More than one call may be made (ignored for [`TurnChoice::One`]).
    pub parallel: bool,
    pub answer: Option<TurnAnswer<'a>>,
}

/// How an addressed call is spelled, read off its shape (see the module documentation).
struct Addressed<'s> {
    header_close: &'s str,
    key_open: &'s str,
    key_close: &'s str,
    value_close: &'s str,
    /// A newline and the next key's opening close a value too.
    next_key_closes: bool,
}

impl<'s> Addressed<'s> {
    /// The spelling of `shape`'s calls, or what keeps them from being addressed to a tool.
    fn of(shape: &'s CallShape) -> Result<Self, String> {
        let not = |what: &str| format!("the table's calls are not addressed to a tool: {what}");
        let [Atom::Run {
            class: name,
            min: name_min,
            capture: Some(Slot::Name),
            ..
        }, Atom::Lit(header_close)] = &*shape.header.0
        else {
            return Err(not("the header is not a name and a literal"));
        };
        if shape.name != NameSource::Header || *name_min == 0 || !reads_names(name) {
            return Err(not(
                "the header does not read the name up to whitespace or <",
            ));
        }
        if !shape.trailer.0.is_empty() {
            return Err(not("a trailer before the call's close"));
        }
        let Arguments::Params { tag, ends, values } = &shape.arguments else {
            return Err(not("the arguments are not tagged parameters"));
        };
        if !matches!(
            values,
            ValueParser::Raw | ValueParser::Text { strip: false }
        ) {
            return Err(not("a value that is not read as raw text"));
        }
        let [Atom::Lit(key_open), Atom::Run {
            class: key,
            min: key_min,
            capture: Some(Slot::Key),
            ..
        }, Atom::Lit(after_key), Atom::Run {
            class: value,
            min: 0,
            max: None,
            greedy: false,
            capture: Some(Slot::Value),
        }] = &*tag.0
        else {
            return Err(not("a parameter that is not a key's tag and a value"));
        };
        let Some(key_close) = after_key
            .strip_suffix('\n')
            .filter(|close| !close.is_empty())
        else {
            return Err(not("a key's tag not followed by a newline"));
        };
        let key_stop = key_close.chars().next().unwrap_or('\n');
        if *key_min == 0 || key.contains(key_stop) || *value != Class::any() {
            return Err(not("a key or a value read otherwise"));
        }
        let closed = |end: &'s End| match (&*end.consume.0, &*end.ahead.0) {
            ([Atom::Lit(close)], []) => close.strip_prefix('\n').filter(|close| !close.is_empty()),
            _ => None,
        };
        let next_key = |end: &End| {
            matches!((&*end.consume.0, &*end.ahead.0),
                ([Atom::Lit(newline)], [Atom::Lit(next)]) if &**newline == "\n" && next == key_open)
        };
        let (value_close, next_key_closes) = match &**ends {
            [close] => (closed(close), false),
            [close, next] if next_key(next) => (closed(close), true),
            _ => (None, false),
        };
        let Some(value_close) = value_close else {
            return Err(not("a value's close that is not a newline and a literal"));
        };
        Ok(Self {
            header_close,
            key_open,
            key_close,
            value_close,
            next_key_closes,
        })
    }
}

/// Whether a name's run reads a tool's name up to whitespace or `<`: it holds neither, and it
/// holds the characters a tool's name is made of.
fn reads_names(class: &Class) -> bool {
    let stops = [' ', '\t', '\n', '\r', '<'];
    let names = [
        'a', 'Z', '0', '_', '-', '.', ':', '/', '"', '>', '=', '\u{e9}',
    ];
    !stops.iter().any(|&c| class.contains(c)) && names.iter().all(|&c| class.contains(c))
}

impl Format {
    /// The texts of this table's turns, or why its rows give none, naming what is missing: calls
    /// that are not addressed, no turn opener or anchor, a thought whose closes lead to different
    /// states, no row out of a call on a stop token or toward another call.
    pub fn turn_spelling(&self) -> Result<TurnSpelling, String> {
        let Some(CallSyntax::Shaped(shape)) = self.call_syntax() else {
            return Err("the table's calls are not given as data".to_string());
        };
        let spelling = Addressed::of(shape)?;
        let openers = self.turn_texts();
        if openers.is_empty() {
            return Err("the table names no turn opener or anchor".to_string());
        }
        let arguments = (0..self.states())
            .find(|&state| self.emits(state) == Emits::Arguments)
            .ok_or("the table has no arguments state")?;
        let exits = |state: usize| -> Vec<(&str, usize)> {
            let mut exits: Vec<(&str, usize)> = Vec::new();
            for (_, on, target) in self.transitions().filter(|&(from, _, _)| from == state) {
                let text = self.terminal_text(on);
                if !exits.iter().any(|(known, _)| *known == text) {
                    exits.push((text, target));
                }
            }
            exits
        };

        let start = 0;
        let reasoning = (0..self.states()).find(|&state| {
            self.emits(state) == Emits::Reasoning && self.path(start, state).is_some()
        });
        let (thought, home) = match reasoning {
            Some(reasoning) => {
                let open = self.path(start, reasoning).unwrap_or_default().concat();
                let closes: Vec<(&str, usize)> = exits(reasoning)
                    .into_iter()
                    .filter(|(text, _)| !self.is_stop_token(text))
                    .collect();
                let Some(&(_, home)) = closes.first() else {
                    return Err("the thought has no close".to_string());
                };
                if closes.iter().any(|&(_, target)| target != home) {
                    return Err("the thought's closes lead to different states".to_string());
                }
                let mut ends: Vec<String> =
                    closes.iter().map(|(text, _)| text.to_string()).collect();
                ends.sort_by_key(|end| std::cmp::Reverse(end.len()));
                let mut leads: Vec<char> =
                    ends.iter().filter_map(|end| end.chars().next()).collect();
                leads.sort_unstable();
                leads.dedup();
                let first = format!(
                    "[^{}]",
                    leads.iter().map(|&c| class_char(c)).collect::<String>()
                );
                (Some(Thought { open, ends, first }), home)
            }
            None => (None, start),
        };

        let call_stops: Vec<&str> = exits(arguments)
            .into_iter()
            .filter(|(text, _)| self.is_stop_token(text))
            .map(|(text, _)| text)
            .collect();
        let call_end = call_stops
            .first()
            .ok_or("no row out of a call is on a stop token")?
            .to_string();
        let call_lead = self
            .path(home, arguments)
            .ok_or("no markers lead from the turn to a call")?
            .concat();
        let joiner = exits(arguments)
            .into_iter()
            .filter(|(text, _)| !self.is_stop_token(text))
            .find_map(|(text, target)| {
                Some(format!("{text}{}", self.path(target, arguments)?.concat()))
            })
            .ok_or("no row out of a call leads to another call")?;
        let separator = joiner
            .strip_suffix(call_lead.as_str())
            .ok_or_else(|| {
                format!(
                    "the markers between two calls ({joiner:?}) do not end with a call's own \
                     ({call_lead:?})"
                )
            })?
            .to_string();
        let mut call_terminators: Vec<String> = Vec::new();
        for (text, _) in exits(arguments) {
            let terminator = openers
                .iter()
                .find_map(|opener| text.strip_suffix(opener).filter(|rest| !rest.is_empty()))
                .unwrap_or(text)
                .to_string();
            if !call_terminators.contains(&terminator) {
                call_terminators.push(terminator);
            }
        }

        let answer = (0..self.states()).find_map(|state| {
            let answers = self.emits(state) == Emits::Content && !self.is_framing(state);
            let open = answers.then(|| self.path(home, state)).flatten()?.concat();
            let texts: Vec<&str> = exits(state).into_iter().map(|(text, _)| text).collect();
            let closes: Vec<String> = texts
                .iter()
                .filter(|text| {
                    !texts
                        .iter()
                        .any(|other| other.len() < text.len() && text.contains(other))
                })
                .map(|text| text.to_string())
                .collect();
            let end = texts
                .iter()
                .find(|text| self.is_stop_token(text) && !call_stops.contains(text))
                .map(|text| text.to_string());
            Some(Answer { open, closes, end })
        });

        let tokens = self.marker_token_texts();
        let holds_token = |text: &str| tokens.iter().any(|token| text.contains(token.as_str()));
        let mut delimiters = vec![spelling.value_close.to_string()];
        delimiters.extend(tokens.iter().cloned());
        for (text, _) in exits(arguments) {
            if !holds_token(text) && !delimiters.iter().any(|known| known == text) {
                delimiters.push(text.to_string());
            }
        }

        Ok(TurnSpelling {
            thought,
            answer,
            call_lead,
            header_close: spelling.header_close.to_string(),
            key_open: spelling.key_open.to_string(),
            key_close: spelling.key_close.to_string(),
            value_close: spelling.value_close.to_string(),
            next_key_closes: spelling.next_key_closes,
            separator,
            call_end,
            call_terminators,
            stops: self.stop_token_texts().to_vec(),
            delimiters,
        })
    }

    /// Where `prompt`, the rendered prompt, leaves the model's turn: the text past the last turn
    /// opener or anchor, its markers replayed over the table from the state the output starts in
    /// there. A prompt that ends at the opener starts the turn, unless it `continued` the final
    /// assistant message and the message the opener ends was a thought: the turn is past its
    /// thought then.
    pub fn turn_position(&self, prompt: &str, continued: bool) -> TurnPosition {
        let Some((at, tail, from)) = self.turn_start(prompt) else {
            return TurnPosition::Elsewhere;
        };
        if tail.is_empty() && from == 0 {
            let thought_before = continued
                && self
                    .turn_start(&prompt[..at])
                    .is_some_and(|(_, before, from)| {
                        self.walk(before, from)
                            .iter()
                            .any(|&state| self.emits(state) == Emits::Reasoning)
                    });
            return if thought_before {
                TurnPosition::AfterThought
            } else {
                TurnPosition::Start
            };
        }
        let state = self.walk(tail, from).last().copied().unwrap_or(from);
        match self.emits(state) {
            Emits::Reasoning => TurnPosition::InThought,
            Emits::Content if !self.is_framing(state) && state != 0 => TurnPosition::InAnswer,
            _ => TurnPosition::Elsewhere,
        }
    }

    /// The states `text`'s markers move the engine through from `from`, that state first.
    fn walk(&self, text: &str, from: usize) -> Vec<usize> {
        let mut scanner = Scanner::new(self.terminal_texts());
        let mut states = vec![from];
        let pieces = scanner.feed(text).into_iter().chain(scanner.finish());
        for piece in pieces {
            if let Piece::Marker(index) = piece {
                let state = *states.last().unwrap_or(&from);
                if let Some(to) = self.next(state, index) {
                    states.push(to);
                }
            }
        }
        states
    }
}

impl TurnSpelling {
    /// The most bytes of JSON text a turn's grammar may take: a gateway leaves a turn whose grammar
    /// is longer unconstrained, or refuses it where the request asked for a constraint. A call
    /// holds each parameter's value once, or at most 1 + ⌈log₂ n⌉ times among n parameters none
    /// of which is required, but each JSON value carries the definitions it refers to, as a
    /// structural tag cannot refer to a schema written elsewhere.
    pub const MAX_TAG_BYTES: usize = 4 << 20;

    /// The most parameters (keys of its schema's `properties`) a tool may declare for a turn's
    /// grammar ([`Self::check_parameter_counts`]). Tools declare tens; a tool's body compiles in a
    /// time that grows with the square of its parameters.
    pub const MAX_TOOL_PARAMETERS: usize = 1024;

    /// The most parameters the tools of one turn may declare together
    /// ([`Self::check_parameter_counts`]). Agent tool sets declare up to about 1,100.
    pub const MAX_TOTAL_PARAMETERS: usize = 4096;

    /// Refuses a tool that declares more than [`Self::MAX_TOOL_PARAMETERS`] parameters, naming it
    /// and the limit, then tools that declare more than [`Self::MAX_TOTAL_PARAMETERS`] together.
    /// Only a tool's own parameters count, not the fields of a parameter's value.
    pub fn check_parameter_counts(tools: &[Tool]) -> Result<(), String> {
        let mut total: usize = 0;
        for tool in tools {
            let count = tool
                .function
                .parameters
                .get("properties")
                .and_then(Value::as_object)
                .map_or(0, Map::len);
            if count > Self::MAX_TOOL_PARAMETERS {
                return Err(format!(
                    "tool '{}' declares {count} parameters, over this model's limit of {} \
                     parameters per tool",
                    tool.function.name,
                    Self::MAX_TOOL_PARAMETERS
                ));
            }
            total = total.saturating_add(count);
        }
        if total > Self::MAX_TOTAL_PARAMETERS {
            return Err(format!(
                "the tools declare {total} parameters together, over this model's limit of {} \
                 parameters per request",
                Self::MAX_TOTAL_PARAMETERS
            ));
        }
        Ok(())
    }

    /// Refuses tools whose calls the grammar cannot write so that the table reads them back: a
    /// name the call's header does not read whole (empty, with whitespace or `<`, or holding a
    /// delimiter or the header's close), a parameters schema that is not an object, a required
    /// parameter it does not declare, a parameter name the key's tag cannot carry, an empty enum,
    /// and a string enum value holding a delimiter.
    pub fn check_tools(&self, tools: &[Tool]) -> Result<(), String> {
        let key_stop = self.key_close.chars().next();
        for tool in tools {
            let name = &tool.function.name;
            if !self.readable_name(name) {
                return Err(format!(
                    "tool name '{name}' cannot be read back from this model's call format \
                     (it must be non-empty, without whitespace or '<')"
                ));
            }
            let parameters = &tool.function.parameters;
            let object = parameters.as_object().filter(|schema| {
                schema
                    .get("type")
                    .is_none_or(|kind| kind.as_str() == Some("object"))
            });
            let Some(object) = object else {
                return Err(format!(
                    "tool '{name}': this model's call format takes an object parameters schema"
                ));
            };
            let declared = object.get("properties").and_then(Value::as_object);
            let required = object.get("required").and_then(Value::as_array);
            for key in required.into_iter().flatten() {
                if !key
                    .as_str()
                    .is_some_and(|key| declared.is_some_and(|d| d.contains_key(key)))
                {
                    return Err(format!(
                        "tool '{name}': required parameter {key} is not among its properties"
                    ));
                }
            }
            for (key, schema) in declared.into_iter().flatten() {
                if schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .is_some_and(Vec::is_empty)
                {
                    return Err(format!(
                        "tool '{name}': parameter '{key}' has an empty enum"
                    ));
                }
                if key.is_empty()
                    || key_stop.is_some_and(|stop| key.contains(stop))
                    || self.holds_delimiter(key)
                {
                    return Err(format!(
                        "tool '{name}': parameter name '{key}' cannot be written in this \
                         model's call format"
                    ));
                }
                if string_enum(schema)
                    .is_some_and(|values| values.iter().any(|v| self.holds_delimiter(v)))
                {
                    return Err(format!(
                        "tool '{name}': a value of parameter '{key}' holds a delimiter of \
                         this model's call format"
                    ));
                }
            }
        }
        Ok(())
    }

    /// The name's run takes the whole name (no whitespace, no `<`), and the header's close
    /// follows it.
    fn readable_name(&self, name: &str) -> bool {
        !name.is_empty()
            && !name.chars().any(|c| c == '<' || c.is_whitespace())
            && !name.contains(self.header_close.as_str())
            && !self.holds_delimiter(name)
    }

    fn holds_delimiter(&self, text: &str) -> bool {
        self.delimiters
            .iter()
            .any(|delimiter| text.contains(delimiter.as_str()))
    }
}

/// The values of a string `enum` (or a string `const`).
fn string_enum(schema: &Value) -> Option<Vec<&String>> {
    if let Some(Value::String(value)) = schema.get("const") {
        return Some(vec![value]);
    }
    let values: Vec<&String> = schema
        .get("enum")?
        .as_array()?
        .iter()
        .map(|v| match v {
            Value::String(s) => Some(s),
            _ => None,
        })
        .collect::<Option<_>>()?;
    (!values.is_empty()).then_some(values)
}

/// `c` inside a regex class.
fn class_char(c: char) -> String {
    if r"\]^-[".contains(c) {
        format!("\\{c}")
    } else {
        c.to_string()
    }
}

impl TurnSpelling {
    /// The grammar of the turn `turn` asks for, or `None` where none starts: anywhere the position
    /// is [`TurnPosition::Elsewhere`]; inside an answer, unless it is free and no call is asked
    /// for; and where a call is asked for and none is callable, or the turn could only answer and
    /// the table has no answer.
    pub fn grammar(&self, turn: &Turn<'_>) -> Option<Grammar> {
        let thought = match turn.position {
            TurnPosition::Start => self.thought.as_ref().map(|thought| {
                Grammar::Tag(Tag::bounded(
                    thought.open.as_str(),
                    Grammar::Sequence(vec![
                        Grammar::Regex(thought.first.clone()),
                        self.thought_text(),
                    ]),
                    Bound::Texts(thought.ends.clone()),
                ))
            }),
            TurnPosition::InThought => {
                let thought = self.thought.as_ref()?;
                Some(Grammar::Tag(Tag::bounded(
                    "",
                    self.thought_text(),
                    Bound::Texts(thought.ends.clone()),
                )))
            }
            TurnPosition::AfterThought => None,
            TurnPosition::InAnswer => {
                if turn.answer.is_some() || turn.choice != TurnChoice::Auto {
                    return None;
                }
                return Some(sequence(self.answer_rest(None)?));
            }
            TurnPosition::Elsewhere => return None,
        };
        let output = self.output(turn)?;
        Some(sequence(thought.into_iter().chain([output]).collect()))
    }

    /// The thought's text: anything but the table's stop tokens.
    fn thought_text(&self) -> Grammar {
        Grammar::AnyText {
            excludes: self.stops.clone(),
        }
    }

    fn output(&self, turn: &Turn<'_>) -> Option<Grammar> {
        let answer = || self.answer(turn.answer);
        if turn.callable.is_empty() {
            return match turn.choice {
                TurnChoice::Auto => answer(),
                TurnChoice::Required | TurnChoice::One => None,
            };
        }
        let mut calls: Vec<Grammar> = turn
            .callable
            .iter()
            .map(|tool| {
                let name = &tool.function.name;
                let header = format!("{}{name}{}", self.call_lead, self.header_close);
                Grammar::Sequence(vec![
                    Grammar::ConstString(header),
                    self.arguments(&tool.function.parameters),
                ])
            })
            .collect();
        if turn.choice == TurnChoice::Auto {
            let names: Vec<&str> = turn
                .declared
                .iter()
                .map(|tool| tool.function.name.as_str())
                .collect();
            let header = Grammar::Sequence(vec![
                Grammar::ConstString(self.call_lead.clone()),
                Grammar::Regex(other_names(&names)),
                Grammar::ConstString(self.header_close.clone()),
            ]);
            calls.push(Grammar::Sequence(vec![
                header,
                self.arguments(&Value::Bool(true)),
            ]));
        }
        let call = one_of(calls);
        let mut run = vec![call.clone()];
        if turn.parallel && turn.choice != TurnChoice::One {
            run.push(Grammar::Star(Box::new(Grammar::Sequence(vec![
                Grammar::ConstString(self.separator.clone()),
                call,
            ]))));
        }
        // The last call ends with the stop the table closes a call on, where the engine stops: a
        // turn the engine ended with another stop would leave the call cut.
        run.push(Grammar::ConstString(self.call_end.clone()));
        let calls = Grammar::Sequence(run);
        match (turn.choice, answer()) {
            (TurnChoice::Auto, Some(answer)) => Some(Grammar::Or(vec![calls, answer])),
            _ => Some(calls),
        }
    }

    /// An answer: its markers, then `format` or free text, then its end.
    fn answer(&self, format: Option<TurnAnswer<'_>>) -> Option<Grammar> {
        let open = Grammar::ConstString(self.answer.as_ref()?.open.clone());
        let rest = self.answer_rest(format)?;
        Some(Grammar::Sequence([vec![open], rest].concat()))
    }

    /// The rest of an open answer: `format`, or text that holds none of the answer's closes, then
    /// the close that ends it, optional. A JSON answer's strings hold none of the closes either.
    fn answer_rest(&self, format: Option<TurnAnswer<'_>>) -> Option<Vec<Grammar>> {
        let answer = self.answer.as_ref()?;
        let body = match format {
            None => Grammar::AnyText {
                excludes: answer.closes.clone(),
            },
            Some(TurnAnswer::Json(schema)) => Grammar::JsonSchema {
                schema: schema.clone(),
                style: None,
                excludes: answer.closes.clone(),
            },
            Some(TurnAnswer::Regex(pattern)) => Grammar::Regex(pattern.to_string()),
            Some(TurnAnswer::Ebnf(grammar)) => Grammar::Ebnf(grammar.to_string()),
        };
        let mut rest = vec![body];
        rest.extend(
            answer
                .end
                .as_ref()
                .map(|end| optional(Grammar::ConstString(end.clone()))),
        );
        Some(rest)
    }

    /// The arguments under `schema`, the tool's parameters schema.
    fn arguments(&self, schema: &Value) -> Grammar {
        if self.next_key_closes {
            let joined = self.parameters(schema, Syntax::Joined);
            let last = self.parameters(schema, Syntax::Last);
            lenient(joined, last, &self.key_open)
        } else {
            closed(self.parameters(schema, Syntax::Closed))
        }
    }

    /// What a raw string value holds none of, for one of the three ways a parameter is written.
    fn raw_excludes(&self, syntax: Syntax) -> Vec<String> {
        let mut excludes = self.call_terminators.clone();
        excludes.push(self.value_close.clone());
        if syntax != Syntax::Closed {
            excludes.push(format!("{BETWEEN_PARAMETERS}{}", self.key_open));
        }
        excludes
    }

    /// The parameters of one call under `schema`, each written as `syntax` writes it.
    fn parameters(&self, schema: &Value, syntax: Syntax) -> Parameters {
        let root = Root(schema);
        let raw = self.raw_excludes(syntax);
        let parameter = |key: Key<'_>, schema: &Value| {
            let values: Vec<Grammar> = root
                .options(schema)
                .iter()
                .filter_map(|option| self.value(root, option, &raw))
                .collect();
            (!values.is_empty()).then(|| self.parameter(key, one_of(values), syntax))
        };
        match root.resolved() {
            Value::Object(object) => {
                let additional = match object.get("additionalProperties") {
                    Some(Value::Bool(false)) => None,
                    Some(additional) => parameter(Key::Free, additional),
                    None if object.contains_key("properties") => None,
                    None => parameter(Key::Free, &values::ANY),
                };
                let required: Vec<&str> = object
                    .get("required")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .collect();
                let declared = object
                    .get("properties")
                    .and_then(Value::as_object)
                    .into_iter()
                    .flatten()
                    .filter_map(|(key, schema)| {
                        Some(Parameter {
                            key: key.clone(),
                            grammar: parameter(Key::Declared(key), schema)?,
                            required: required.contains(&key.as_str()),
                        })
                    })
                    .collect();
                Parameters {
                    declared,
                    additional,
                }
            }
            Value::Bool(false) => Parameters {
                declared: Vec::new(),
                additional: None,
            },
            _ => Parameters {
                declared: Vec::new(),
                additional: parameter(Key::Free, &values::ANY),
            },
        }
    }

    /// One parameter whose value is `value`, as `syntax` writes it.
    fn parameter(&self, key: Key<'_>, value: Grammar, syntax: Syntax) -> Grammar {
        let key_close = format!("{}{BETWEEN_PARAMETERS}", self.key_close);
        let close = format!("{BETWEEN_PARAMETERS}{}", self.value_close);
        match syntax {
            Syntax::Closed => Grammar::Sequence(vec![
                separator(),
                key.tag(&self.key_open, &key_close, value, Bound::Text(close)),
            ]),
            Syntax::Joined => {
                let next = &self.key_open;
                let ends = vec![
                    format!("{close}{next}"),
                    format!("{close}{BETWEEN_PARAMETERS}{next}"),
                    format!("{BETWEEN_PARAMETERS}{next}"),
                ];
                key.tag("", &key_close, value, Bound::Texts(ends))
            }
            Syntax::Last => key.tag("", &key_close, value, Bound::Text(close)),
        }
    }

    /// One option of a value: a string as raw text that holds none of `raw` (its `pattern` when
    /// xgrammar compiles it; a constant as itself, left out when it holds one), anything else as
    /// JSON.
    fn value(&self, root: Root<'_>, option: &ValueOption<'_>, raw: &[String]) -> Option<Grammar> {
        if option.kind == Kind::String {
            let text = || Grammar::AnyText {
                excludes: raw.to_vec(),
            };
            return Some(match &option.source {
                Source::Literal(value) => {
                    let value = value.as_str()?;
                    if raw.iter().any(|word| value.contains(word.as_str())) {
                        return None;
                    }
                    Grammar::ConstString(value.to_string())
                }
                Source::Schema(schema) => match schema.get("pattern").and_then(Value::as_str) {
                    Some(pattern) if values::compiles(pattern) => {
                        Grammar::Regex(pattern.to_string())
                    }
                    _ => text(),
                },
                Source::Any => text(),
            });
        }
        let schema = match &option.source {
            Source::Schema(schema) => {
                Value::Object(values::narrowed(schema, option.kind).into_owned())
            }
            Source::Literal(value) => json!({"const": value}),
            Source::Any => option.kind.any_schema(),
        };
        Some(self.json(root, schema))
    }

    /// JSON of `schema`, which carries the tool's definitions so its `$ref`s resolve, without
    /// the patterns xgrammar does not compile, its strings holding no call terminator; and the
    /// same JSON with every object's properties in the order of their names, when that differs.
    fn json(&self, root: Root<'_>, mut schema: Value) -> Grammar {
        if let (Some(schema), Some(root)) = (schema.as_object_mut(), root.0.as_object()) {
            for key in ["$defs", "definitions"] {
                if let Some(definitions) = root.get(key) {
                    schema.entry(key).or_insert_with(|| definitions.clone());
                }
            }
        }
        values::drop_uncompiled_patterns(&mut schema);
        let json = |schema: Value| Grammar::JsonSchema {
            schema,
            style: None,
            excludes: self.call_terminators.clone(),
        };
        let mut by_name = schema.clone();
        if values::sort_properties(&mut by_name) {
            Grammar::Or(vec![json(schema), json(by_name)])
        } else {
            json(schema)
        }
    }
}

/// The three ways a parameter is written: with its own opening and close (`Closed`); where the
/// next key closes a value, past its opening and ended by the next one's (`Joined`), or as the
/// last, past its opening and ended by its close (`Last`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Syntax {
    Closed,
    Joined,
    Last,
}

/// A parameter's key: one the schema declares, or any free key.
#[derive(Clone, Copy, Debug)]
enum Key<'a> {
    Declared(&'a str),
    Free,
}

impl Key<'_> {
    /// `prefix KEY suffix`, `content`, `end`: a tag for a declared key, a sequence around a free
    /// one.
    fn tag(&self, prefix: &str, suffix: &str, content: Grammar, end: Bound) -> Grammar {
        match self {
            Self::Declared(key) => {
                Grammar::Tag(Tag::bounded(format!("{prefix}{key}{suffix}"), content, end))
            }
            Self::Free => Grammar::Sequence(vec![
                Grammar::ConstString(prefix.to_string()),
                Grammar::Regex(FREE_KEY.to_string()),
                Grammar::ConstString(suffix.to_string()),
                content,
                match end {
                    Bound::Text(text) => Grammar::ConstString(text),
                    Bound::Texts(texts) => {
                        one_of(texts.into_iter().map(Grammar::ConstString).collect())
                    }
                },
            ]),
        }
    }
}

/// A call's parameters: the declared ones in the schema's order, and one free parameter when
/// the schema allows any number of them.
struct Parameters {
    declared: Vec<Parameter>,
    additional: Option<Grammar>,
}

struct Parameter {
    key: String,
    grammar: Grammar,
    required: bool,
}

/// Arguments whose every value ends with its close.
fn closed(
    Parameters {
        declared,
        additional,
    }: Parameters,
) -> Grammar {
    let additional = additional.map(|parameter| Grammar::Star(Box::new(parameter)));
    let lists: Vec<Grammar> = orders(&declared, |parameter| &parameter.key)
        .into_iter()
        .map(|order| {
            order
                .into_iter()
                .map(|parameter| listed(parameter.grammar.clone(), parameter.required))
                .chain(additional.clone())
                .collect::<Vec<_>>()
        })
        .filter(|list| !list.is_empty())
        .map(Grammar::Sequence)
        .collect();
    if lists.is_empty() {
        Grammar::ConstString(String::new())
    } else {
        one_of(lists)
    }
}

/// A declared parameter written with another after it, `joined`, and as the last, `last`.
struct Lenient {
    key: String,
    joined: Grammar,
    last: Grammar,
    required: bool,
}

impl Lenient {
    fn joined(&self) -> Grammar {
        listed(self.joined.clone(), self.required)
    }
}

/// Arguments whose values but the last end where the next one opens, `key_open` written once
/// before the first: the last present one is a declared one past every required one, or a free
/// one.
fn lenient(joined: Parameters, last: Parameters, key_open: &str) -> Grammar {
    // Both lists have the same parameters: the same options survive the same exclusions.
    let declared: Vec<Lenient> = joined
        .declared
        .into_iter()
        .zip(last.declared)
        .map(|(joined, last)| Lenient {
            key: joined.key,
            joined: joined.grammar,
            last: last.grammar,
            required: joined.required,
        })
        .collect();
    let additional = joined.additional.zip(last.additional);
    let mut lists = Vec::new();
    for order in orders(&declared, |parameter| &parameter.key) {
        if !order.is_empty() {
            let first = order
                .iter()
                .rposition(|parameter| parameter.required)
                .unwrap_or(0);
            let mut list: Vec<Grammar> = order[..first].iter().map(|p| p.joined()).collect();
            list.push(last_present(&order[first..]));
            lists.push(Grammar::Sequence(list));
        }
        if let Some((joined, last)) = &additional {
            let mut list: Vec<Grammar> = order.iter().map(|p| p.joined()).collect();
            list.extend([Grammar::Star(Box::new(joined.clone())), last.clone()]);
            lists.push(Grammar::Sequence(list));
        }
    }
    if lists.is_empty() {
        return Grammar::ConstString(String::new());
    }
    let arguments = Grammar::Sequence(vec![
        separator(),
        Grammar::ConstString(key_open.to_string()),
        one_of(lists),
    ]);
    if declared.iter().any(|parameter| parameter.required) {
        arguments
    } else {
        optional(arguments)
    }
}

/// `parameters`, at least one, in order, the last present one written last: it is in the first
/// half, or past all of it. Halving keeps the grammar O(n log n) long and O(log n) deep.
fn last_present(parameters: &[&Lenient]) -> Grammar {
    if let [only] = parameters {
        return only.last.clone();
    }
    let (head, tail) = parameters.split_at(parameters.len() / 2);
    let mut after_head: Vec<Grammar> = head.iter().map(|parameter| parameter.joined()).collect();
    after_head.push(last_present(tail));
    Grammar::Or(vec![last_present(head), Grammar::Sequence(after_head)])
}

/// `parameters` in the schema's order, and in the order of their keys when that differs: the
/// model writes them sorted.
fn orders<P>(parameters: &[P], key: impl Fn(&P) -> &str) -> Vec<Vec<&P>> {
    let declared: Vec<&P> = parameters.iter().collect();
    let mut sorted = declared.clone();
    sorted.sort_by_key(|parameter| key(parameter));
    if sorted
        .iter()
        .map(|parameter| key(parameter))
        .eq(declared.iter().map(|parameter| key(parameter)))
    {
        vec![declared]
    } else {
        vec![declared, sorted]
    }
}

/// `parameter`, optional unless `required`.
fn listed(parameter: Grammar, required: bool) -> Grammar {
    if required {
        parameter
    } else {
        optional(parameter)
    }
}

/// The newline the template may write before a parameter.
fn separator() -> Grammar {
    optional(Grammar::ConstString(BETWEEN_PARAMETERS.to_string()))
}

fn optional(content: Grammar) -> Grammar {
    Grammar::Optional(Box::new(content))
}

/// The elements one after another; one element stands as it is.
fn sequence(mut elements: Vec<Grammar>) -> Grammar {
    if elements.len() == 1 {
        return elements.remove(0);
    }
    Grammar::Sequence(elements)
}

/// One of the elements; one element stands as it is.
fn one_of(mut elements: Vec<Grammar>) -> Grammar {
    if elements.len() == 1 {
        return elements.remove(0);
    }
    Grammar::Or(elements)
}

/// A regex of the names a call's header can spell other than `names`: none whitespace or `<`, and
/// none exactly one of `names`. It walks a trie of the names: at each node, a character that
/// leaves every name, or one that follows a name further.
fn other_names(names: &[&str]) -> String {
    #[derive(Default)]
    struct Node {
        children: BTreeMap<char, Node>,
        end: bool,
    }

    fn escaped(c: char, meta: &str) -> String {
        if meta.contains(c) {
            format!("\\{c}")
        } else {
            c.to_string()
        }
    }

    /// A name's character other than `chars`.
    fn other<'a>(chars: impl Iterator<Item = &'a char>) -> String {
        let chars: String = chars.map(|&c| escaped(c, r"\]^-[")).collect();
        format!("[^{NAME_STOPS}{chars}]")
    }

    /// The rest of a name past `node` that ends on none of the names.
    fn rest(node: &Node, first: bool) -> String {
        let any = other([].iter());
        let mut branches = vec![format!("{}{any}*", other(node.children.keys()))];
        branches.extend(
            node.children.iter().map(|(&c, child)| {
                format!("{}{}", escaped(c, r".^$*+?{}[]\|()"), rest(child, false))
            }),
        );
        let optional = if first || node.end { "" } else { "?" };
        format!("(?:{}){optional}", branches.join("|"))
    }

    let mut root = Node::default();
    for name in names {
        let mut node = &mut root;
        for c in name.chars() {
            node = node.children.entry(c).or_default();
        }
        node.end = true;
    }
    rest(&root, true)
}

#[cfg(test)]
mod tests;
