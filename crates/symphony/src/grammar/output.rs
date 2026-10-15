//! The output grammar for a whole assistant turn of a table whose calls are addressed to a tool,
//! from the first
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
//! ([`Format::output_spelling`]), none from a model:
//! - **the thought** is the markers into the reasoning state from the turn's own state, then text
//!   that is not empty and does not start with a character any of its closes starts with, ended
//!   by any close of the reasoning state (every row out of it but a stop's), longest first; a stop
//!   token never ends a turn inside its thought, so the text holds none of the table's stops;
//! - **the answer** is the markers into the content state from where the thought's closes leave
//!   the turn, then the request's answer format (JSON of its schema, its strings free) or free
//!   text, which holds none of the content state's closes but the stop a call ends on, then,
//!   optional, the close that is a stop no call ends on;
//! - **a call** is the markers into the arguments state from there, the tool's name, the header's
//!   close, then the parameters; between two calls the row out of a call that is no stop, after
//!   the last, optional, the one that is: the engine stops there, and where its matcher stops on
//!   that token too, the token ends the call without the text;
//! - **values**: a string is raw text that holds no call terminator (each row out of a call
//!   without the turn opener it may end with), no value close and, where the next key closes a
//!   value, no next key's opening; a string `pattern` xgrammar compiles is that regex instead; a
//!   string constant is itself, left out when it holds one of those; every other value is JSON of
//!   its part of the schema, its keys in the schema's order, with the call terminators left out
//!   of its strings, carrying the tool's definitions it reaches through its `$ref`s, and without
//!   the annotations (`title`, `description`, `examples`) that constrain nothing.
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
//! Where the prompt leaves the model ([`Format::output_position`]) decides where the grammar starts:
//! at the turn's start, inside a thought the prompt opened, after a thought the prompt closed (a
//! continued message: the turn holds one thought), inside an answer the prompt opened (the rest of
//! a free answer), or inside a call the prompt opened (the rest of that call is free text, and
//! each call after it is held). Anywhere else (past text no field holds) no grammar starts.
//!
//! A tool that asks for its calls to follow its schema (`strict`) may hold parts of it the grammar
//! takes more loosely ([`OutputSpelling::unenforced`]), and a tool or parameter may have a name the
//! calls cannot carry ([`OutputSpelling::check_names`]).

mod build;
mod strict;
mod values;

use std::collections::BTreeMap;

use openai_protocol::common::Tool;
use serde_json::{Map, Value};

use self::{
    build::{Build, Json, Length, Tree},
    values::{Kind, Root, Source, ValueOption},
};
use super::{json_len, Bound, Grammar};
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
/// is read up to whitespace or `<`. The whitespace is the set xgrammar's regex reads for `\s`,
/// spelled in the escapes of its EBNF, whose character classes take no `\s`.
const NAME_STOPS: &str = r"\f\n\r\t\v\u0020\u00a0<";

/// The texts of a turn the table's rows declare; see the module documentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputSpelling {
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
    /// What free text in the answer holds none of: each close of the content state that holds no
    /// shorter one, but the stop a call ends on, where the engine stops wherever it is written.
    closes: Vec<String>,
    end: Option<String>,
}

/// Where the prompt leaves the model's turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputPosition {
    /// Nothing written past the turn's opener.
    Start,
    /// Inside a thought the prompt opened.
    InThought,
    /// Past a thought the prompt closed, in a message it continues.
    AfterThought,
    /// Inside an answer the prompt opened.
    InAnswer,
    /// Inside a call the prompt opened.
    InCall,
    /// Anywhere else: a marker cut short, text no field holds.
    Elsewhere,
}

/// How the request's tool choice shapes the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputChoice {
    /// An answer or calls (`auto`, or `none` with nothing callable).
    Auto,
    /// One or more calls.
    Required,
    /// Exactly one call.
    One,
}

/// The request's own constraint on the answer.
#[derive(Clone, Copy, Debug)]
pub enum OutputAnswer<'a> {
    /// JSON of the schema.
    Json(&'a Value),
    Regex(&'a str),
    /// An EBNF grammar; only xgrammar's own syntax (`::=`) composes into a turn.
    Ebnf(&'a str),
}

impl OutputAnswer<'_> {
    /// Whether the output grammar can hold it: not an EBNF grammar in Lark's syntax (`rule:`).
    pub fn composes(&self) -> bool {
        match self {
            Self::Ebnf(grammar) => grammar.contains("::="),
            Self::Json(_) | Self::Regex(_) => true,
        }
    }
}

/// What one request asks of an output grammar.
#[derive(Clone, Copy, Debug)]
pub struct Output<'a> {
    pub position: OutputPosition,
    /// Every tool the request declares: an undeclared call names none of them.
    pub declared: &'a [Tool],
    /// The tools the choice lets the model call.
    pub callable: &'a [Tool],
    pub choice: OutputChoice,
    /// More than one call may be made (ignored for [`OutputChoice::One`]).
    pub parallel: bool,
    pub answer: Option<OutputAnswer<'a>>,
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
    pub fn output_spelling(&self) -> Result<OutputSpelling, String> {
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
                    !call_stops.contains(text)
                        && !texts
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

        Ok(OutputSpelling {
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
    pub fn output_position(&self, prompt: &str, continued: bool) -> OutputPosition {
        let Some((at, tail, from)) = self.turn_start(prompt) else {
            return OutputPosition::Elsewhere;
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
                OutputPosition::AfterThought
            } else {
                OutputPosition::Start
            };
        }
        let state = self.walk(tail, from).last().copied().unwrap_or(from);
        match self.emits(state) {
            Emits::Reasoning => OutputPosition::InThought,
            Emits::Content if !self.is_framing(state) && state != 0 => OutputPosition::InAnswer,
            Emits::Arguments => OutputPosition::InCall,
            _ => OutputPosition::Elsewhere,
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

impl OutputSpelling {
    /// The most bytes of JSON text an output grammar may take: a gateway leaves a turn whose grammar
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
        for tool in tools {
            let name = &tool.function.name;
            self.check_name(name)?;
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
                self.check_key(name, key)?;
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

    /// Refuses tools a call cannot name so that the table reads the name back, or whose parameter
    /// a call cannot name: the name and parameter checks of [`Self::check_tools`], for any tool a
    /// request declares, whatever its choice.
    pub fn check_names(&self, tools: &[Tool]) -> Result<(), String> {
        for tool in tools {
            let name = &tool.function.name;
            self.check_name(name)?;
            if let Some(object) = tool.function.parameters.as_object() {
                for key in object
                    .get("properties")
                    .and_then(Value::as_object)
                    .into_iter()
                    .flatten()
                    .map(|(key, _)| key)
                {
                    self.check_key(name, key)?;
                }
            }
        }
        Ok(())
    }

    fn check_name(&self, name: &str) -> Result<(), String> {
        if self.readable_name(name) {
            return Ok(());
        }
        Err(format!(
            "tool name '{name}' cannot be read back from this model's call format (it must be \
             non-empty, without whitespace or '<')"
        ))
    }

    fn check_key(&self, name: &str, key: &str) -> Result<(), String> {
        let key_stop = self.key_close.chars().next();
        if key.is_empty()
            || key_stop.is_some_and(|stop| key.contains(stop))
            || self.holds_delimiter(key)
        {
            return Err(format!(
                "tool '{name}': parameter name '{key}' cannot be written in this model's call \
                 format"
            ));
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

impl OutputSpelling {
    /// The output grammar `request` asks for, or `None` where none starts: anywhere the position
    /// is [`OutputPosition::Elsewhere`]; inside an answer the request gives a format, which applies
    /// to the answer's start; and where a call is asked for and none is callable, or the turn
    /// could only answer and the table has no answer. Inside an answer the rest of a free one is
    /// the grammar whatever the choice, since no call follows an answer; inside a call the rest of
    /// it is free, then calls follow where more than one may be made.
    pub fn grammar(&self, request: &Output<'_>) -> Option<Grammar> {
        self.build(&Tree::default(), request)
    }

    /// The length of the text [`Grammar::payload_text`] writes for [`OutputSpelling::grammar`], or
    /// `None` where that is `None`, worked out without building the grammar: a turn over a size
    /// bound is refused in about the time it takes to read its tools.
    pub fn payload_len(&self, request: &Output<'_>) -> Option<usize> {
        let format = self.build(&Length::new(), request)?;
        let payload = json_len(&serde_json::json!({"format": null})) - json_len(&Value::Null);
        Some(payload.saturating_add(format))
    }

    fn build<B: Build>(&self, b: &B, request: &Output<'_>) -> Option<B::Node> {
        let thought = match request.position {
            OutputPosition::Start => self.thought.as_ref().map(|thought| {
                b.tag(
                    thought.open.clone(),
                    b.sequence(vec![
                        b.leaf(Grammar::Regex(thought.first.clone())),
                        b.leaf(self.thought_text()),
                    ]),
                    Bound::Texts(thought.ends.clone()),
                )
            }),
            OutputPosition::InThought => {
                let thought = self.thought.as_ref()?;
                Some(b.tag(
                    String::new(),
                    b.leaf(self.thought_text()),
                    Bound::Texts(thought.ends.clone()),
                ))
            }
            OutputPosition::AfterThought => None,
            OutputPosition::InAnswer => {
                if request.answer.is_some() {
                    return None;
                }
                return Some(sequence(b, self.answer_rest(b, None)?));
            }
            OutputPosition::InCall => return Some(self.rest_of_call(b, request)),
            OutputPosition::Elsewhere => return None,
        };
        let output = self.output(b, request)?;
        Some(sequence(b, thought.into_iter().chain([output]).collect()))
    }

    /// The thought's text: anything but the table's stop tokens.
    fn thought_text(&self) -> Grammar {
        Grammar::AnyText {
            excludes: self.stops.clone(),
        }
    }

    fn output<B: Build>(&self, b: &B, request: &Output<'_>) -> Option<B::Node> {
        let answer = || self.answer(b, request.answer);
        let Some(call) = self.call(b, request) else {
            return match request.choice {
                OutputChoice::Auto => answer(),
                OutputChoice::Required | OutputChoice::One => None,
            };
        };
        let mut run = vec![call.clone()];
        if request.parallel && request.choice != OutputChoice::One {
            run.push(self.more_calls(b, call));
        }
        run.push(self.call_end(b));
        let calls = b.sequence(run);
        match (request.choice, answer()) {
            (OutputChoice::Auto, Some(answer)) => Some(b.or(vec![calls, answer])),
            _ => Some(calls),
        }
    }

    /// The rest of a call the prompt opened, free text up to the row toward another call, then,
    /// where more than one call may be made, calls held as at the turn's start; then the last
    /// call's end.
    fn rest_of_call<B: Build>(&self, b: &B, request: &Output<'_>) -> B::Node {
        let mut rest = vec![b.leaf(Grammar::AnyText {
            excludes: vec![self.separator.clone()],
        })];
        if request.parallel && request.choice != OutputChoice::One {
            rest.extend(self.call(b, request).map(|call| self.more_calls(b, call)));
        }
        rest.push(self.call_end(b));
        b.sequence(rest)
    }

    /// Any number of calls more, each after the row toward it.
    fn more_calls<B: Build>(&self, b: &B, call: B::Node) -> B::Node {
        b.star(b.sequence(vec![
            b.leaf(Grammar::ConstString(self.separator.clone())),
            call,
        ]))
    }

    /// The stop the table closes a call on, where the engine stops, optional: an engine whose
    /// matcher stops on its token takes it as the grammar's end, not as its text.
    fn call_end<B: Build>(&self, b: &B) -> B::Node {
        b.optional(b.leaf(Grammar::ConstString(self.call_end.clone())))
    }

    /// One call to a tool the choice lets the model call, or under `auto` to a tool the request
    /// does not declare; `None` when nothing is callable.
    fn call<B: Build>(&self, b: &B, request: &Output<'_>) -> Option<B::Node> {
        if request.callable.is_empty() {
            return None;
        }
        let mut calls: Vec<B::Node> = request
            .callable
            .iter()
            .map(|tool| {
                let name = &tool.function.name;
                let header = format!("{}{name}{}", self.call_lead, self.header_close);
                b.sequence(vec![
                    b.leaf(Grammar::ConstString(header)),
                    self.arguments_in(b, &tool.function.parameters),
                ])
            })
            .collect();
        if request.choice == OutputChoice::Auto {
            let names: Vec<&str> = request
                .declared
                .iter()
                .map(|tool| tool.function.name.as_str())
                .collect();
            // The undeclared name and the header's close: one grammar (see `other_names`).
            let header = b.sequence(vec![
                b.leaf(Grammar::ConstString(self.call_lead.clone())),
                b.leaf(Grammar::Ebnf(other_names(&names, &self.header_close))),
            ]);
            calls.push(b.sequence(vec![header, self.arguments_in(b, &Value::Bool(true))]));
        }
        Some(one_of(b, calls))
    }

    /// An answer: its markers, then `format` or free text, then its end.
    fn answer<B: Build>(&self, b: &B, format: Option<OutputAnswer<'_>>) -> Option<B::Node> {
        let open = b.leaf(Grammar::ConstString(self.answer.as_ref()?.open.clone()));
        let rest = self.answer_rest(b, format)?;
        Some(b.sequence([vec![open], rest].concat()))
    }

    /// The rest of an open answer: `format`, or text that holds none of the answer's closes, then
    /// the close that ends it, optional.
    fn answer_rest<B: Build>(
        &self,
        b: &B,
        format: Option<OutputAnswer<'_>>,
    ) -> Option<Vec<B::Node>> {
        let answer = self.answer.as_ref()?;
        let body = match format {
            None => b.leaf(Grammar::AnyText {
                excludes: answer.closes.clone(),
            }),
            Some(OutputAnswer::Json(schema)) => b.schema(schema),
            Some(OutputAnswer::Regex(pattern)) => b.leaf(Grammar::Regex(pattern.to_string())),
            Some(OutputAnswer::Ebnf(grammar)) => b.leaf(Grammar::Ebnf(grammar.to_string())),
        };
        let mut rest = vec![body];
        rest.extend(
            answer
                .end
                .as_ref()
                .map(|end| b.optional(b.leaf(Grammar::ConstString(end.clone())))),
        );
        Some(rest)
    }

    /// The arguments under `schema`, the tool's parameters schema.
    #[cfg(test)]
    fn arguments(&self, schema: &Value) -> Grammar {
        self.arguments_in(&Tree::default(), schema)
    }

    fn arguments_in<B: Build>(&self, b: &B, schema: &Value) -> B::Node {
        if self.next_key_closes {
            let joined = self.parameters_in(b, schema, Syntax::Joined);
            let last = self.parameters_in(b, schema, Syntax::Last);
            lenient(b, joined, last, &self.key_open)
        } else {
            closed(b, self.parameters_in(b, schema, Syntax::Closed))
        }
    }

    /// What a raw string value holds none of, for one of the three ways a parameter is written:
    /// the call terminators, the value's close, and the next key's opening where it closes a
    /// value.
    fn raw_excludes(&self, syntax: Syntax) -> Vec<String> {
        let mut excludes = self.call_terminators.clone();
        excludes.push(self.value_close.clone());
        if syntax != Syntax::Closed {
            excludes.push(format!("{BETWEEN_PARAMETERS}{}", self.key_open));
        }
        excludes
    }

    /// The parameters of one call under `schema`, each written as `syntax` writes it.
    #[cfg(test)]
    fn parameters(&self, schema: &Value, syntax: Syntax) -> Parameters<Grammar> {
        self.parameters_in(&Tree::default(), schema, syntax)
    }

    fn parameters_in<B: Build>(
        &self,
        b: &B,
        schema: &Value,
        syntax: Syntax,
    ) -> Parameters<B::Node> {
        let root = Root(schema);
        let raw = self.raw_excludes(syntax);
        let parameter = |key: Key<'_>, schema: &Value| {
            let values: Vec<B::Node> = root
                .options(schema)
                .iter()
                .filter_map(|option| self.value(b, root, option, &raw))
                .collect();
            (!values.is_empty()).then(|| self.parameter(b, key, one_of(b, values), syntax))
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
    fn parameter<B: Build>(&self, b: &B, key: Key<'_>, value: B::Node, syntax: Syntax) -> B::Node {
        let key_close = format!("{}{BETWEEN_PARAMETERS}", self.key_close);
        let close = format!("{BETWEEN_PARAMETERS}{}", self.value_close);
        match syntax {
            Syntax::Closed => b.sequence(vec![
                separator(b),
                key.tag(b, &self.key_open, &key_close, value, Bound::Text(close)),
            ]),
            Syntax::Joined => {
                let next = &self.key_open;
                let ends = vec![
                    format!("{close}{next}"),
                    format!("{close}{BETWEEN_PARAMETERS}{next}"),
                    format!("{BETWEEN_PARAMETERS}{next}"),
                ];
                key.tag(b, "", &key_close, value, Bound::Texts(ends))
            }
            Syntax::Last => key.tag(b, "", &key_close, value, Bound::Text(close)),
        }
    }

    /// One option of a value: a string as raw text that holds none of `raw` (its `pattern` when
    /// xgrammar compiles it; a constant as itself, left out when it holds one), anything else as
    /// JSON ([`Build::value`]).
    fn value<B: Build>(
        &self,
        b: &B,
        root: Root<'_>,
        option: &ValueOption<'_>,
        raw: &[String],
    ) -> Option<B::Node> {
        if option.kind == Kind::String {
            let text = || {
                b.leaf(Grammar::AnyText {
                    excludes: raw.to_vec(),
                })
            };
            return Some(match &option.source {
                Source::Literal(value) => {
                    let value = value.as_str()?;
                    if raw.iter().any(|word| value.contains(word.as_str())) {
                        return None;
                    }
                    b.leaf(Grammar::ConstString(value.to_string()))
                }
                Source::Schema(schema) => match schema.get("pattern").and_then(Value::as_str) {
                    Some(pattern) if values::compiles(pattern) => {
                        b.leaf(Grammar::Regex(pattern.to_string()))
                    }
                    _ => text(),
                },
                Source::Any => text(),
            });
        }
        let json = match &option.source {
            Source::Schema(schema) => Json::Schema(schema, option.kind),
            Source::Literal(value) => Json::Literal(value),
            Source::Any => Json::Any(option.kind),
        };
        Some(b.value(json, root, &self.call_terminators))
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
    fn tag<B: Build>(
        &self,
        b: &B,
        prefix: &str,
        suffix: &str,
        content: B::Node,
        end: Bound,
    ) -> B::Node {
        match self {
            Self::Declared(key) => b.tag(format!("{prefix}{key}{suffix}"), content, end),
            Self::Free => b.sequence(vec![
                b.leaf(Grammar::ConstString(prefix.to_string())),
                b.leaf(Grammar::Regex(FREE_KEY.to_string())),
                b.leaf(Grammar::ConstString(suffix.to_string())),
                content,
                match end {
                    Bound::Text(text) => b.leaf(Grammar::ConstString(text)),
                    Bound::Texts(texts) => one_of(
                        b,
                        texts
                            .into_iter()
                            .map(|text| b.leaf(Grammar::ConstString(text)))
                            .collect(),
                    ),
                },
            ]),
        }
    }
}

/// A call's parameters: the declared ones in the schema's order, and one free parameter when
/// the schema allows any number of them.
struct Parameters<N> {
    declared: Vec<Parameter<N>>,
    additional: Option<N>,
}

struct Parameter<N> {
    key: String,
    grammar: N,
    required: bool,
}

/// Arguments whose every value ends with its close.
fn closed<B: Build>(
    b: &B,
    Parameters {
        declared,
        additional,
    }: Parameters<B::Node>,
) -> B::Node {
    let additional = additional.map(|parameter| b.star(parameter));
    let lists: Vec<B::Node> = orders(&declared, |parameter| &parameter.key)
        .into_iter()
        .map(|order| {
            order
                .into_iter()
                .map(|parameter| listed(b, parameter.grammar.clone(), parameter.required))
                .chain(additional.clone())
                .collect::<Vec<_>>()
        })
        .filter(|list| !list.is_empty())
        .map(|list| b.sequence(list))
        .collect();
    if lists.is_empty() {
        b.leaf(Grammar::ConstString(String::new()))
    } else {
        one_of(b, lists)
    }
}

/// A declared parameter written with another after it, `joined`, and as the last, `last`.
struct Lenient<N> {
    key: String,
    joined: N,
    last: N,
    required: bool,
}

impl<N: Clone> Lenient<N> {
    fn joined<B: Build<Node = N>>(&self, b: &B) -> N {
        listed(b, self.joined.clone(), self.required)
    }
}

/// Arguments whose values but the last end where the next one opens, `key_open` written once
/// before the first: the last present one is a declared one past every required one, or a free
/// one.
fn lenient<B: Build>(
    b: &B,
    joined: Parameters<B::Node>,
    last: Parameters<B::Node>,
    key_open: &str,
) -> B::Node {
    // Both lists have the same parameters: the same options survive the same exclusions.
    let declared: Vec<Lenient<B::Node>> = joined
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
            let mut list: Vec<B::Node> = order[..first].iter().map(|p| p.joined(b)).collect();
            list.push(last_present(b, &order[first..]));
            lists.push(b.sequence(list));
        }
        if let Some((joined, last)) = &additional {
            let mut list: Vec<B::Node> = order.iter().map(|p| p.joined(b)).collect();
            list.extend([b.star(joined.clone()), last.clone()]);
            lists.push(b.sequence(list));
        }
    }
    if lists.is_empty() {
        return b.leaf(Grammar::ConstString(String::new()));
    }
    let arguments = b.sequence(vec![
        separator(b),
        b.leaf(Grammar::ConstString(key_open.to_string())),
        one_of(b, lists),
    ]);
    if declared.iter().any(|parameter| parameter.required) {
        arguments
    } else {
        b.optional(arguments)
    }
}

/// `parameters`, at least one, in order, the last present one written last: it is in the first
/// half, or past all of it. Halving keeps the grammar O(n log n) long and O(log n) deep.
fn last_present<B: Build>(b: &B, parameters: &[&Lenient<B::Node>]) -> B::Node {
    if let [only] = parameters {
        return only.last.clone();
    }
    let (head, tail) = parameters.split_at(parameters.len() / 2);
    let mut after_head: Vec<B::Node> = head.iter().map(|parameter| parameter.joined(b)).collect();
    after_head.push(last_present(b, tail));
    b.or(vec![last_present(b, head), b.sequence(after_head)])
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
fn listed<B: Build>(b: &B, parameter: B::Node, required: bool) -> B::Node {
    if required {
        parameter
    } else {
        b.optional(parameter)
    }
}

/// The newline the template may write before a parameter.
fn separator<B: Build>(b: &B) -> B::Node {
    b.optional(b.leaf(Grammar::ConstString(BETWEEN_PARAMETERS.to_string())))
}

/// The elements one after another; one element stands as it is.
fn sequence<B: Build>(b: &B, mut elements: Vec<B::Node>) -> B::Node {
    if elements.len() == 1 {
        return elements.remove(0);
    }
    b.sequence(elements)
}

/// One of the elements; one element stands as it is.
fn one_of<B: Build>(b: &B, mut elements: Vec<B::Node>) -> B::Node {
    if elements.len() == 1 {
        return elements.remove(0);
    }
    b.or(elements)
}

/// An EBNF grammar of the names a call's header can spell other than `names`, each followed by
/// `close`, the header's close: none whitespace or `<`, and none exactly one of `names`. Its rules
/// walk a trie of the names, one rule a node: a character that leaves every name and then any
/// name's rest (`t`, one rule for every node, the close at its end), a character that follows a
/// name further, or, where no name ends, the close. The close inside the name's own rules lets
/// the engine tell a token that ends the name and opens the close without looking past the
/// grammar, which keeps the mask of each token in the name a few microseconds.
fn other_names(names: &[&str], close: &str) -> String {
    #[derive(Default)]
    struct Node {
        children: BTreeMap<char, Node>,
        end: bool,
    }

    /// `c` as EBNF writes it where the characters of `meta` and control characters are escaped.
    fn escaped(c: char, meta: &str) -> String {
        if c.is_control() || meta.contains(c) {
            format!("\\U{:08x}", u32::from(c))
        } else {
            c.to_string()
        }
    }

    fn string(text: &str) -> String {
        let text: String = text.chars().map(|c| escaped(c, "\"\\")).collect();
        format!("\"{text}\"")
    }

    /// The rule of `node` and, after it, those of the nodes under it; `first` for the root, where
    /// a name has not started yet.
    fn rules(node: &Node, first: bool, close: &str, out: &mut Vec<String>) -> usize {
        let at = out.len();
        out.push(String::new());
        let others: String = node
            .children
            .keys()
            .map(|&c| escaped(c, r"\]^-["))
            .collect();
        let mut branches = vec![format!("[^{NAME_STOPS}{others}] t")];
        for (&c, child) in &node.children {
            let child = rules(child, false, close, out);
            branches.push(format!("{} n{child}", string(&c.to_string())));
        }
        if !first && !node.end {
            branches.push(close.to_string());
        }
        out[at] = format!("n{at} ::= {}", branches.join(" | "));
        at
    }

    let mut root = Node::default();
    for name in names {
        let mut node = &mut root;
        for c in name.chars() {
            node = node.children.entry(c).or_default();
        }
        node.end = true;
    }
    let close = string(close);
    let mut out = vec![
        "root ::= n0".to_string(),
        format!("t ::= [^{NAME_STOPS}]* {close}"),
    ];
    let mut nodes = Vec::new();
    rules(&root, true, &close, &mut nodes);
    out.extend(nodes);
    out.join("\n")
}

#[cfg(test)]
mod tests;
