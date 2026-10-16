//! A format is data: the terminals a model writes between its text, the states that text falls
//! into, the transitions the terminals make between them, and the syntax of a call. The [`Engine`]
//! runs one; a family's definition is a function under [`formats`] today and a TOML file later.
//!
//! The table is the design's section 4. What it says:
//!
//! - A **terminal** is a spelling the model writes, `<think>` or `</tool_call>`, named so that a
//!   transition can speak of it. Token ids come in a later step; today a terminal is its text.
//! - A **state** says what the text inside it is: content, reasoning, a call's arguments, or the
//!   template's wrapping between calls. Text in a content state is `Content`; in a reasoning
//!   state `Reasoning`; in an arguments state it is fed to the call syntax's assembler and becomes
//!   the call's events; in a wrapper state whitespace is dropped and anything else is malformed.
//! - A **transition** `from + terminal = to` moves the engine between states when the terminal
//!   arrives in `from`. A terminal with no transition from the current state is text, where the
//!   model put it: `</think>` in content is content. Entering a reasoning state pushes
//!   `ReasoningStart` and leaving one `ReasoningEnd`; entering an arguments state opens a call and
//!   leaving one ends it, so `calls + call_open = calls` ends the call that was open and starts
//!   the next. The terminal itself is `Dropped { Wrapper }` on a transition.
//! - The **call syntax** says how the model writes a call inside an arguments state, and the
//!   assembler for it: a family's, or a call given as data ([`CallSyntax::Shaped`]), whose
//!   terminal opens a call only when the header after it matches. A format with no arguments
//!   state has none.
//! - An **ignored terminal** is a spelling the engine drops wherever it stands, a call's arguments
//!   included, without moving: the separator token MiniMax M3's template writes before every tag.
//!   No transition can name it.
//! - A **stop row** names an arguments state whose call the stream's stop closes. A family whose
//!   last call ends with a stop token writes no other marker after it, and the stop decoder hides
//!   that token, so without the row the call would read as cut short. With it, a stream that
//!   stops in that state ends the call as its closing marker would, with no bytes; a stream cut
//!   for length, or aborted, is still cut, and so is one that ended with any finish but `Stop`.
//!   `EngineFinish::Stop` also covers a stop sequence of the request's, which the row would take
//!   for the stop token and close the call on, with whatever arguments had arrived; a driver that
//!   knows a stop sequence ended the stream hands the end over as another finish
//!   (`EngineFinish::Other`), and the call stays cut, since a stop sequence never completes a
//!   call. A table read from a response template may name it for any call, since the template's
//!   reader closes a region at the end of its text whatever the region's closing pattern: a call
//!   given as data closes there as [`CallSyntax::Shaped`] says.
//! - A **framing row** names a content or wrapper state whose text the template discards: the
//!   space around a response template's fields, which Hugging Face's parser drops (its null sink),
//!   or a field the API has no place for. Every byte of text there, a terminal with no row from
//!   the state included, is `Dropped { Wrapper }`, and so is what the scanner still held there
//!   when the stream ended, at every finish: neither the client nor `Malformed` sees it. The
//!   state's rows move the engine as before.
//! - A **trim row** names a content or reasoning state whose text is trimmed stay by stay, as
//!   Hugging Face's `text` parser strips a field's value (`strip`, on unless a template turns it
//!   off). A stay is the text between the row that enters the state and the row that leaves it,
//!   a row back into the same state ending one stay and starting the next. Whitespace before the
//!   stay's first other character is `Dropped { Whitespace }`; a run of it after the last one is
//!   held until more text in the stay shows it was inside the text, or the stay ends and drops it.
//!   Whitespace is what Python's `str.strip` removes: Unicode's white space and the separators
//!   U+001C to U+001F.
//! - The **stop tokens** are the terminals that are tokens the engine stops on (Harmony's
//!   `<|call|>`): the stop decoder hides them, so the engine never reads one, and a constraint
//!   derived from the table can end a region one of them closes with no text of its own, since
//!   the model's stop ends the region there ([`Format::is_stop_token`]).
//! - A table **thinks first** when its template writes the thought before every call and answer
//!   it renders, which the template and its rendering tell, not the call syntax: the structural
//!   tags then write the thought before the call or the answer when the prompt has not opened it.
//!   A template whose thought the model may leave out (gpt-oss's analysis channel) does not.
//!
//! The first state in the table is where an output starts, unless the prompt moved the engine
//! before the output began: the engine replays the prompt's terminals over the same table, from
//! the last **turn opener** the table names (`<|im_start|>assistant` for Qwen's ChatML), so that
//! a marker quoted in an earlier turn moves nothing. A table that names no opener is replayed
//! from the prompt's start.
//!
//! A table read from a response template names its **turn anchors** instead, as Hugging Face's
//! `start_anchor` does: one literal or several (Gemma 4's template starts a turn after
//! `<|turn>model\n` or after `<tool_response|>`). The anchors are found as a regular expression's
//! `finditer` finds an alternation of them: from the prompt's start, the earliest match, of the
//! anchors that match there the first in the table's order, then on from that match's end, so
//! that matches never overlap; the last match is the turn's. The replay starts after its end, in
//! the state the anchor names: the first state, or one the anchor already entered (a template
//! whose anchor ends with the thought's opener). A prompt that holds no anchor replays nothing,
//! and the output starts in the first state. A table names a turn opener or turn anchors, not
//! both.
//!
//! A **start row** names the state a turn starts in when it does not start with a terminal: a
//! response template's open with a `^` alternative (`(?:^|<think>\s*)`), which matches with no
//! text at the turn's first byte. Hugging Face's parser takes the longest delimiter at a place,
//! and a regular expression the first alternative that matches: so at the turn's first byte a
//! terminal whose row leads into another state wins, being longer than the empty match, while a
//! terminal whose row leads into the start state is text there, the `^` having matched first
//! (`<think>` at the first byte is the thought's text). Any other first piece starts the turn in
//! the start state and is read there. The turn's first byte is where the replay starts, after the
//! anchor; when the prompt leaves nothing to replay, it is the output's. A turn an anchor started
//! in a state of its own has no start to choose.
//!
//! The table holds what is static about a format. What belongs to one request, the tools that
//! type a tagged call's values, reaches the [`Engine`] with the request, not the table.
//!
//! [`Engine`]: crate::Engine
//! [`formats`]: crate::formats

use std::sync::Arc;

use crate::{
    engine::is_strip_space,
    gemma,
    markers::{Piece, Scanner},
    pythonic,
    shape::{CallShape, Layout, ValueRead, Values},
    tagged::{self, attributed::TagEnd, keyed},
};

/// How the model writes a call between the call markers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallSyntax {
    /// One JSON object, `{"name": …, "arguments": {…}}`: Qwen3.
    Json,
    /// `<function=NAME>` and then `<parameter=KEY>` around each value's text, typed by the
    /// request's tools, which reach the engine with the request, with the family's spelling of a
    /// value that is not a string and where its template puts a value between the tags: Qwen 3.5
    /// and later and Qwen3-Coder (on a line of its own), Seed-OSS and MiMo (between the tags
    /// directly).
    Tagged(tagged::Spelling, tagged::Placement),
    /// The call's name as text, then `<arg_key>` and `<arg_value>` pairs, in the family's spelling
    /// of the four tags, typed by the request's tools: GLM, Hy4, Ling, IQuest. The terminals that
    /// enter and leave the state are the call's own markers, and the call's end carries the
    /// closing one.
    Keyed(keyed::Tags),
    /// Python calls, `name(key=value, ...)`, one or several, in the family's dialect: framed as
    /// its template frames them (one per line: Olmo 3; a Python list: LFM2.5, Llama 3.2's
    /// pythonic template), the values as it writes them (Python's words and quotes: LFM2.5; JSON,
    /// through `tojson`: Olmo 3). The region's markers are the terminals; each call ends at its
    /// own `)`, and the region's closing marker is wrapping. The assembler reads either framing
    /// and both spellings; the grammar derived from the table writes the dialect named.
    Pythonic(pythonic::Dialect),
    /// A JSON list of calls with no markers around it, or content: xLAM. The table's one state is
    /// an arguments state entered at the output's start, and the assembler decides at the first
    /// byte that is not whitespace.
    JsonList,
    /// One JSON call object, or a list of them, alone in the turn with no markers around it, or
    /// content: the format the gateway's older `json` parser reads. The table's one state is an
    /// arguments state entered at the output's start, and the assembler decides at the first byte
    /// that is not whitespace: `{` is one call, `[` a list, anything else content.
    JsonAlone,
    /// An XML tree of arguments: `<invoke name="…">`, then one element per argument, `<key>` to
    /// `</key>`, nested for an object and with `<item>` elements for a list, typed by the request's
    /// tools at every depth: MiniMax M3. The terminal that enters the state is the invoke tag's
    /// opening, and the one that leaves it is `</invoke>`, which the call's end carries.
    Xml,
    /// Parameters named by an attribute: the arguments state is one `<invoke name="…">` block
    /// whose `<parameter name="…">` tags carry the key in the attribute, typed by the request's
    /// tools or by an attribute of the tag, in the family's spelling of the tags: MiniMax M2,
    /// dots3, MiniCPM5, DeepSeek's DSML, Kimi K3's XTML. The terminal that enters the state is the
    /// invoke tag's opening, and the one that leaves it is the invoke's closing tag, which the
    /// call's end carries.
    Attributed(tagged::attributed::Tags),
    /// The tool's name as text, then this marker, then one JSON object `{"name": …, "args": …}`:
    /// Inkling, whose every message starts the arguments state and says what it is only after
    /// its opener (`<|content_thinking|>`, `<|content_text|>`, or the name and the marker). The
    /// terminals that leave the state are the message's own; the object's close is the call's end,
    /// and the message's closing terminal is the region's.
    NamedJson(&'static str),
    /// The tool's name as text, then this marker, then the arguments object alone: Mistral's
    /// `[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}`. The terminal that enters the state is the
    /// call's opener, which also opens the next call; the object's close is the call's end, and
    /// the region ends at the next call's opener or the end of the turn, since the template writes
    /// no closing marker.
    NamedArguments(&'static str),
    /// The arguments object alone, named by the function the request forces
    /// (`Declared::forced`): what the engine writes under the JSON-schema constraint the gateway
    /// builds when `tool_choice` names a function, the one shape known from the request rather
    /// than from the model. The table's one state is an arguments state entered at the output's
    /// start; the object's close ends the call, and what follows is wrapping or malformed.
    ForcedArguments,
    /// Gemma's `call:NAME{key:value,…}`: keys bare, a string between two tokens, numbers and
    /// words bare, lists and objects nested the same way, the keys sorted by the template, in the
    /// family's spelling of the string token and the null word (Gemma 4: `<|"|>` and `null`;
    /// FunctionGemma: `<escape>` and `None`). The terminals that enter and leave the state are the
    /// call's own markers, and the call's end carries the closing one.
    Gemma(gemma::Spelling),
    /// A call given as data, as a response template writes one: what follows the region's
    /// terminal (the header, which may capture the tool's name), how the arguments are written,
    /// and what comes after them before the region's closing terminal (see
    /// [`CallShape`]). The terminal that enters the state opens a call only when the header after
    /// it matches; otherwise it is the text of the state it came in. A closing terminal closes the
    /// region only where the trailer ends right before it, and is the call's end; otherwise it is
    /// the call's text. The stream's stop, where [`Format::closes_at_stop`] names the state, closes
    /// one JSON value or object where the trailer ends with the stream, and tagged parameters
    /// wherever the trailer stands. The shape is [`intern`](crate::shape::intern)ed, so the syntax
    /// stays `Copy`.
    Shaped(&'static CallShape),
}

impl CallSyntax {
    /// The tags the syntax writes inside a call besides the table's terminals: the markers a
    /// grammar writes inside a call's region, which free text before a call may not write, and
    /// which a client that shows a call as text has to strip. JSON, Python and xLAM's list have
    /// none, their calls being their own text, and neither has MiniMax M3's XML tree, whose
    /// element tags are the arguments' keys. A call given as data has its parameters' (see
    /// [`CallShape::tag_texts`]).
    pub fn tag_texts(&self) -> Vec<&'static str> {
        match self {
            Self::Tagged(..) => tagged::assembler::TAGS.to_vec(),
            Self::Keyed(tags) => vec![
                tags.key_open,
                tags.key_close,
                tags.value_open,
                tags.value_close,
            ],
            // XTML's `<|sep|>` ends a tag's tail and is a marker of its own.
            Self::Attributed(tags) => match tags.tag_end {
                TagEnd::Marker(marker) => vec![tags.parameter_open, tags.parameter_close, marker],
                TagEnd::Angle => vec![tags.parameter_open, tags.parameter_close],
            },
            Self::NamedJson(marker) | Self::NamedArguments(marker) => vec![*marker],
            Self::Gemma(spelling) => vec![spelling.string],
            Self::Shaped(shape) => shape.tag_texts(),
            Self::Json
            | Self::Pythonic(_)
            | Self::JsonList
            | Self::JsonAlone
            | Self::ForcedArguments
            | Self::Xml => Vec::new(),
        }
    }
}

/// What the text inside a state is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emits {
    Content,
    Reasoning,
    /// A call's arguments, assembled by the format's call syntax.
    Arguments,
    /// The template's wrapping between calls: whitespace is `Dropped { Wrapper }`, anything else
    /// `Malformed`.
    Wrapper,
}

/// One format's table. Built with [`Format::new`] and the methods that add a row each; the
/// first state added is the initial one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Format {
    name: String,
    terminals: Vec<Terminal>,
    states: Vec<State>,
    transitions: Vec<Transition>,
    calls: Option<CallSyntax>,
    turn_opener: Option<String>,
    /// The turn anchors, in the order a match at one place prefers them.
    anchors: Vec<Anchor>,
    /// The state a turn starts in when it does not start with a terminal.
    start: Option<usize>,
    structural_tokens: Vec<String>,
    /// The texts of the tokens the engine stops on that are terminals.
    stop_tokens: Vec<String>,
    /// How a shaped call's values read.
    values: Values,
    thinks_first: bool,
    marker_tokens: Vec<String>,
    /// How the model's chat template writes a turn, where the table holds one.
    layout: Option<Box<Layout>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Terminal {
    name: String,
    text: String,
    /// Dropped wherever it stands, never a move.
    ignored: bool,
    /// Read as content text where it moves the engine, in place of being dropped: a table's
    /// variant shows the markers of the channel a request does not separate.
    shown: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    name: String,
    emits: Emits,
    /// The template discards the state's text (the framing row).
    framing: bool,
    /// The state's text is trimmed stay by stay (the trim row).
    trimmed: bool,
    /// The stream's stop closes the call open in the state (the stop row).
    closes_at_stop: bool,
}

/// What the replay of a prompt leaves ([`Format::replayed`]): the state the output starts in,
/// whether the turn's first piece is still to come, and the stay the output continues.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Replay {
    /// The state the walk ends in; where the prompt left a call open, the state the turn started
    /// in.
    pub(crate) state: usize,
    /// Whether the turn's first piece is still to come, for a table with a start row.
    pub(crate) at_turn_start: bool,
    /// Whether the prompt wrote text other than whitespace in the stay the output continues: a
    /// prefill the output goes on with. The turn opener and the table's markers are the
    /// template's framing, not the stay's text.
    pub(crate) begun: bool,
}

/// A literal that starts the model's turn in the prompt, and the state the output starts in after
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Anchor {
    text: String,
    state: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Transition {
    from: usize,
    on: usize,
    to: usize,
}

/// A row that names a state or terminal the table does not have. Definitions are written in the
/// crate, so this is a programming error, and the builder panics with the name.
#[expect(
    clippy::panic,
    reason = "a definition that names a row it never added is a programming error in the crate"
)]
fn unknown(kind: &str, name: &str, format: &str) -> ! {
    panic!("format {format}: no {kind} named {name:?}")
}

impl Format {
    /// An empty table with a name; add terminals, states and transitions to it.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            terminals: Vec::new(),
            states: Vec::new(),
            transitions: Vec::new(),
            calls: None,
            turn_opener: None,
            anchors: Vec::new(),
            start: None,
            structural_tokens: Vec::new(),
            stop_tokens: Vec::new(),
            values: Values::Inferred,
            thinks_first: false,
            marker_tokens: Vec::new(),
            layout: None,
        }
    }

    /// A terminal and its text spelling.
    #[must_use]
    pub fn terminal(mut self, name: &str, text: &str) -> Self {
        self.terminals.push(Terminal {
            name: name.to_string(),
            text: text.to_string(),
            ignored: false,
            shown: false,
        });
        self
    }

    /// A terminal the engine drops wherever it stands, inside a call's arguments too, without
    /// moving: a template's separator token. No transition can name it.
    #[must_use]
    pub fn ignores(mut self, name: &str, text: &str) -> Self {
        self.terminals.push(Terminal {
            name: name.to_string(),
            text: text.to_string(),
            ignored: true,
            shown: false,
        });
        self
    }

    /// A token of the model's vocabulary that the format's tags begin with (`<|open|>`,
    /// `<|close|>`). It reaches the engine whole or not at all, so a tail at the end of the output
    /// that begins with one and is the beginning of a terminal the state has a row for is a tag
    /// the end cut short, never the model's text: the engine drops it as it drops the whole tag,
    /// instead of releasing it as content. A tail shorter than the token stays text, as every
    /// parser keeps a partial marker at the end of the output; so does the beginning of a
    /// terminal the state keeps as text when whole.
    #[must_use]
    pub fn structural_token(mut self, text: &str) -> Self {
        self.structural_tokens.push(text.to_string());
        self
    }

    /// A state and what its text is; the first one added is where an output starts.
    #[must_use]
    pub fn state(mut self, name: &str, emits: Emits) -> Self {
        self.states.push(State {
            name: name.to_string(),
            emits,
            framing: false,
            trimmed: false,
            closes_at_stop: false,
        });
        self
    }

    /// `from + on = to`: the terminal `on`, arriving in state `from`, moves the engine to `to`.
    /// Both states and the terminal must already be in the table.
    #[must_use]
    pub fn transition(mut self, from: &str, on: &str, to: &str) -> Self {
        let transition = Transition {
            from: self.state_named(from),
            on: self.terminal_named(on),
            to: self.state_named(to),
        };
        self.transitions.push(transition);
        self
    }

    /// The syntax of a call inside an arguments state.
    #[must_use]
    pub fn calls(mut self, syntax: CallSyntax) -> Self {
        self.calls = Some(syntax);
        self
    }

    /// The arguments state `state` closes its call when the stream stops in it: the family ends
    /// its last call with a stop token, which the stop decoder hides, or a response template's
    /// reader closes the call's region at the end of its text. A stream cut for length,
    /// aborted, or ended with any finish but `Stop` is still cut; a driver hands a stop sequence of
    /// the request's over as such a finish (see the stop row in [`crate::format`]). The state must
    /// already be in the table.
    #[must_use]
    pub fn closes_at_stop(mut self, state: &str) -> Self {
        let state = self.state_named(state);
        self.states[state].closes_at_stop = true;
        self
    }

    /// The template discards the text in `state`, a content or wrapper state: every byte of it is
    /// `Dropped { Wrapper }`, held bytes included, at every finish (see the framing row in
    /// [`crate::format`]). The state must already be in the table.
    #[must_use]
    pub fn framing(mut self, state: &str) -> Self {
        let state = self.state_named(state);
        self.states[state].framing = true;
        self
    }

    /// The text of `state`, a content or reasoning state, is trimmed stay by stay: whitespace at
    /// a stay's edges is `Dropped { Whitespace }` (see the trim row in [`crate::format`]). The
    /// state must already be in the table.
    #[must_use]
    pub fn trims(mut self, state: &str) -> Self {
        let state = self.state_named(state);
        self.states[state].trimmed = true;
        self
    }

    /// The texts among `stops` (the model's end-of-sequence and stop tokens) that are terminals of
    /// the table, each kept once in the order given; other texts are not kept. The terminals must
    /// already be in the table. Parsing does not change, since the stop decoder hides these
    /// tokens; [`Self::is_stop_token`] tells a constraint derived from the table which closing
    /// terminals the model's stop writes.
    #[must_use]
    pub fn stop_tokens<'t>(mut self, stops: impl IntoIterator<Item = &'t str>) -> Self {
        for stop in stops {
            let terminal = self.terminals.iter().any(|terminal| terminal.text == stop);
            if terminal && !self.stop_tokens.iter().any(|known| known == stop) {
                self.stop_tokens.push(stop.to_string());
            }
        }
        self
    }

    /// Whether `text` is a terminal of the table that is a token the engine stops on
    /// ([`Self::stop_tokens`]): a region it closes ends at the model's stop, so a constraint can
    /// end that region with no text of its own.
    pub fn is_stop_token(&self, text: &str) -> bool {
        self.stop_tokens.iter().any(|stop| stop == text)
    }

    /// The stop tokens the table keeps ([`Self::stop_tokens`]), in the order given.
    pub(crate) fn stop_token_texts(&self) -> &[String] {
        &self.stop_tokens
    }

    /// The table's shaped calls read their values with `reader`: a tagged value's text by the
    /// shape's parser, typed by the parameter's schema; a member of JSON arguments held and typed
    /// where the reader says; a key written twice as it says. Without the row they read as the
    /// tagged tables read theirs ([`Inferred`](crate::shape::Inferred)). No other call syntax
    /// reads it.
    #[must_use]
    pub fn reads_values_with(mut self, reader: Arc<dyn ValueRead>) -> Self {
        self.values = Values::Reader(reader);
        self
    }

    /// The reader a shaped call's values read with.
    pub(crate) fn value_reader(&self) -> Values {
        self.values.clone()
    }

    /// The text that opens the turn the model writes, `<|im_start|>assistant` for ChatML: the
    /// prompt's replay starts at its last occurrence.
    #[must_use]
    pub fn opens_turn(mut self, text: &str) -> Self {
        self.turn_opener = Some(text.to_string());
        self
    }

    /// A literal that starts the model's turn in the prompt, as a response template's
    /// `start_anchor`: the prompt's replay starts after the end of the last anchor found (see the
    /// turn anchors in [`crate::format`]), in the first state. Anchors added earlier are preferred
    /// where several match at one place.
    #[must_use]
    pub fn turn_anchor(mut self, text: &str) -> Self {
        self.anchors.push(Anchor {
            text: text.to_string(),
            state: 0,
        });
        self
    }

    /// A turn anchor after which the output starts in `state`: an anchor that already opened the
    /// thought, or closed it. The state must already be in the table.
    #[must_use]
    pub fn turn_anchor_starting_in(mut self, text: &str, state: &str) -> Self {
        let state = self.state_named(state);
        self.anchors.push(Anchor {
            text: text.to_string(),
            state,
        });
        self
    }

    /// The state a turn starts in when its first piece is not a terminal whose row leads into
    /// another state, as a response template's `^` alternative of the state's open (see the start
    /// row in [`crate::format`]). The state must already be in the table.
    #[must_use]
    pub fn starts_in(mut self, state: &str) -> Self {
        self.start = Some(self.state_named(state));
        self
    }

    /// The state the start row names, if the table has one.
    pub(crate) fn start(&self) -> Option<usize> {
        self.start
    }

    /// The template writes the thought before every call and answer it renders: a property of the
    /// template and its rendering, not of the call syntax. The structural tags write the thought
    /// first where the prompt has not opened it ([`Format::grammar`], [`Format::answer_grammar`]);
    /// parsing does not change. The table must have a thought; the tags write it where it opens
    /// from the turn's own state and closes back there, and give `None` otherwise.
    #[must_use]
    pub fn thinks_first(mut self) -> Self {
        self.thinks_first = true;
        self
    }

    /// Whether the template writes the thought before every call and answer.
    pub(crate) fn writes_thought_first(&self) -> bool {
        self.thinks_first
    }

    /// The texts among `tokens` (the model's vocabulary's added tokens, by their text) that the
    /// table's markers are made of: a terminal, a structural token or one of the call syntax's
    /// tags that is one token, or a token inside one (`<|open|>` in Kimi K3's tags). The
    /// structural tags write each as that token wherever they write it inside a tag or between
    /// tags, so the model cannot spell it in ordinary tokens there; a tag's own begin and end and
    /// its triggers stay text. Parsing does not change. It reads the markers the table has when
    /// it is called, so it comes after the rows, and called again it replaces the set.
    #[must_use]
    pub fn marker_tokens<T: AsRef<str>>(mut self, tokens: impl IntoIterator<Item = T>) -> Self {
        let mut markers: Vec<&str> = self
            .terminals
            .iter()
            .map(|terminal| terminal.text.as_str())
            .chain(self.structural_tokens.iter().map(String::as_str))
            .collect();
        let tags = self
            .calls
            .map(|syntax| syntax.tag_texts())
            .unwrap_or_default();
        markers.extend(tags);
        let mut kept: Vec<String> = tokens
            .into_iter()
            .map(|token| token.as_ref().to_string())
            .filter(|token| {
                !token.is_empty() && markers.iter().any(|marker| marker.contains(token.as_str()))
            })
            .collect();
        // The longest first, so a token inside a longer one is never taken for part of it.
        kept.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        kept.dedup();
        self.marker_tokens = kept;
        self
    }

    /// The marker texts that are one token each, the longest first.
    pub(crate) fn marker_token_texts(&self) -> &[String] {
        &self.marker_tokens
    }

    /// How the model's chat template writes a turn of this table where the response template's
    /// patterns leave the text open ([`Layout`]), as whoever compiled the table learned it from a
    /// rendering. Parsing does not read it. The structural tags of a shaped call are written from
    /// it, and every table's tags write a thought and an answer as it does
    /// ([`Format::grammar_for`], [`Format::answer_grammar`]). [`Layout::check`] says whether it
    /// fits the table; a shaped call whose layout does not fit has no tag. Given again, it
    /// replaces the one before.
    #[must_use]
    pub fn with_layout(mut self, layout: Layout) -> Self {
        self.layout = Some(Box::new(layout));
        self
    }

    /// The table's layout, if it has one.
    pub fn layout(&self) -> Option<&Layout> {
        self.layout.as_deref()
    }

    /// The table for a request that does not separate reasoning (`separate_reasoning: false`), as
    /// the legacy path runs no reasoning parser then: the thought's text is content, and every
    /// marker that is not a call's is shown, read as content text where it moves the engine, so
    /// the bytes stay in the content as the model wrote them. The rows stay, so a call inside or
    /// after the thought reads as before, from wherever a start row, an anchor or the prompt's
    /// replay puts the turn. A call's markers are the terminals on the rows into or out of an
    /// arguments state or a calls wrapper; a table whose thought is framed by them alone
    /// (Inkling's) shows none. A state's own text rows stay with it: a trimmed stay is still
    /// trimmed, a framing state's text still dropped; and a tag the end cut short is still
    /// dropped, as the legacy parsers drop it on both their paths. A table without a thought
    /// changes only in what it shows; the thought-first mark and the layout, which serve the tags
    /// alone, go.
    #[must_use]
    pub fn without_reasoning(mut self) -> Self {
        let calls = self.terminals_of_calls();
        for (index, terminal) in self.terminals.iter_mut().enumerate() {
            if !terminal.ignored && !calls.contains(&index) {
                terminal.shown = true;
            }
        }
        for state in &mut self.states {
            if state.emits == Emits::Reasoning {
                state.emits = Emits::Content;
            }
        }
        self.thinks_first = false;
        self.layout = None;
        self
    }

    /// The table for a request that declares no tools or whose `tool_choice` is `none`, as the
    /// legacy path runs no tool parser then: the arguments states and the calls wrappers are
    /// content, with no call syntax to read them, and a call's markers are shown, read as content
    /// text where they move the engine, so a call's bytes stay in the content as the model wrote
    /// them; a marker that also opens the thought is not shown, so the thought reads as before,
    /// its markers dropped. The turn's own wrapper, where a table starts in one (Kimi K3's
    /// message), is the turn's frame, not a calls region: it and its markers stay. A table whose
    /// output starts in its calls (`json`, `xlam`, `forced`) becomes content from its first byte.
    /// A state's own text rows stay with it, as in [`Self::without_reasoning`]. A stop that closed
    /// a call closes nothing; the layout goes.
    #[must_use]
    pub fn without_calls(mut self) -> Self {
        let demoted = self.calls_regions();
        let calls = self.terminals_of_calls();
        let opens_thought: Vec<usize> = self
            .transitions
            .iter()
            .filter(|row| self.states[row.to].emits == Emits::Reasoning)
            .map(|row| row.on)
            .collect();
        for (index, terminal) in self.terminals.iter_mut().enumerate() {
            if calls.contains(&index) && !opens_thought.contains(&index) {
                terminal.shown = true;
            }
        }
        for (index, state) in self.states.iter_mut().enumerate() {
            if demoted[index] {
                state.emits = Emits::Content;
                state.closes_at_stop = false;
            }
        }
        self.calls = None;
        self.layout = None;
        self
    }

    /// Which states are a call's region: the arguments states, and the wrapper states other than
    /// the first, since a table that starts in a wrapper (Kimi K3's message) starts in the turn's
    /// frame.
    fn calls_regions(&self) -> Vec<bool> {
        self.states
            .iter()
            .enumerate()
            .map(|(index, state)| match state.emits {
                Emits::Arguments => true,
                Emits::Wrapper => index != 0,
                Emits::Content | Emits::Reasoning => false,
            })
            .collect()
    }

    /// The terminals that are a call's markers: those whose every row enters or leaves a call's
    /// region ([`Self::calls_regions`]). A terminal with a row between two other states as well
    /// (Kimi K3's message close) frames the turn, not a call.
    fn terminals_of_calls(&self) -> Vec<usize> {
        let regions = self.calls_regions();
        (0..self.terminals.len())
            .filter(|&on| {
                let mut rows = self.transitions.iter().filter(|row| row.on == on);
                let mut any = false;
                let all = rows.all(|row| {
                    any = true;
                    regions[row.from] || regions[row.to]
                });
                any && all
            })
            .collect()
    }

    /// The turn opener, if the table names one.
    pub(crate) fn turn_opener(&self) -> Option<&str> {
        self.turn_opener.as_deref()
    }

    /// The texts that start the model's turn: the turn opener, or the turn anchors in the table's
    /// order; none for a table that names neither.
    pub(crate) fn turn_texts(&self) -> Vec<&str> {
        match self.turn_opener() {
            Some(opener) => vec![opener],
            None => self
                .anchors
                .iter()
                .map(|anchor| anchor.text.as_str())
                .collect(),
        }
    }

    /// Where the model's turn starts in `prompt`: where the last turn anchor found
    /// ([`Self::last_anchor`]), or else the last turn opener, starts, the text after it, and the
    /// state the output starts in after it. `None` when the prompt holds neither.
    pub(crate) fn turn_start<'p>(&self, prompt: &'p str) -> Option<(usize, &'p str, usize)> {
        if !self.anchors.is_empty() {
            let (at, anchor) = self.last_anchor(prompt)?;
            return Some((at, &prompt[at + anchor.text.len()..], anchor.state));
        }
        let opener = self.turn_opener()?;
        let at = prompt.rfind(opener)?;
        Some((at, &prompt[at + opener.len()..], 0))
    }

    /// The part of `prompt` the engine replays to find the state the generation starts in: from
    /// the last occurrence of the turn opener, or the whole prompt when the table names none or
    /// the prompt does not hold it; for a table with turn anchors, from the start of the last
    /// anchor found, or nothing when the prompt holds none. A caller that keeps the prompt for a
    /// later `Input::Prompt` need keep only this: replayed again, it gives itself.
    pub fn replay<'p>(&self, prompt: &'p str) -> &'p str {
        if !self.anchors.is_empty() {
            let from = self.last_anchor(prompt).map_or(prompt.len(), |(at, _)| at);
            return &prompt[from..];
        }
        match self.turn_opener().and_then(|opener| prompt.rfind(opener)) {
            Some(at) => &prompt[at..],
            None => prompt,
        }
    }

    /// The bytes a piece of the replay stands for.
    fn piece_len(&self, piece: &Piece) -> usize {
        match piece {
            Piece::Text(text) => text.len(),
            Piece::Marker(index) => self.terminal_text(*index).len(),
        }
    }

    /// The text of `prompt` the engine replays, and the state the replay starts in: after the
    /// last turn anchor, in the state it names; otherwise [`Self::replay`]'s part, from the first
    /// state.
    pub(crate) fn turn<'p>(&self, prompt: &'p str) -> (&'p str, usize) {
        if self.anchors.is_empty() {
            return (self.replay(prompt), 0);
        }
        match self.last_anchor(prompt) {
            Some((at, anchor)) => (&prompt[at + anchor.text.len()..], anchor.state),
            None => ("", 0),
        }
    }

    /// The last turn anchor in `prompt` and where it starts, found as `finditer` finds the
    /// alternation of the anchors: the earliest match, the first anchor in the table's order
    /// where several match there, then on from that match's end. Each anchor's next occurrence is
    /// searched for again only once the matches have passed it, so a prompt costs one pass per
    /// anchor. An anchor with no text never matches.
    fn last_anchor(&self, prompt: &str) -> Option<(usize, &Anchor)> {
        let find = |anchor: &Anchor, from: usize| {
            if anchor.text.is_empty() {
                return None;
            }
            prompt[from..]
                .find(anchor.text.as_str())
                .map(|at| from + at)
        };
        let mut next: Vec<Option<usize>> =
            self.anchors.iter().map(|anchor| find(anchor, 0)).collect();
        let mut from = 0;
        let mut last = None;
        loop {
            for (anchor, at) in self.anchors.iter().zip(next.iter_mut()) {
                if at.is_some_and(|at| at < from) {
                    *at = find(anchor, from);
                }
            }
            let Some((at, index)) = next
                .iter()
                .enumerate()
                .filter_map(|(index, at)| at.map(|at| (at, index)))
                .min()
            else {
                return last;
            };
            let anchor = &self.anchors[index];
            last = Some((at, anchor));
            from = at + anchor.text.len();
        }
    }

    /// The state the turn starts in once its first piece decides it, for a table with a start row:
    /// the start row's state, unless the piece is a terminal whose row from `state` leads into
    /// another one, which wins as the longer match; then `state`. A terminal the table ignores
    /// decides nothing, and gives `None`. A table without a start row starts where it stands.
    pub(crate) fn opening(&self, state: usize, piece: &Piece) -> Option<usize> {
        let Some(start) = self.start else {
            return Some(state);
        };
        match *piece {
            Piece::Marker(index) if self.ignored(index) => None,
            Piece::Marker(index) if self.next(state, index).is_some_and(|to| to != start) => {
                Some(state)
            }
            _ => Some(start),
        }
    }

    /// Where `prompt` leaves the table: [`Self::turn`]'s text replayed over the rows from the state
    /// the turn starts in, the turn's first piece deciding its start where the table has a start
    /// row ([`Self::opening`]), a terminal the table ignores, or one with no row from where the
    /// walk stands, moving nothing; and the stay the output continues. A call the prompt left
    /// open is not entered: the walk rests where the turn started. The engine starts the output
    /// where this leaves it, and [`Self::reasoning_open`] reads the same walk.
    pub(crate) fn replayed(&self, prompt: &str) -> Replay {
        let (turn, start) = self.turn(prompt);
        // The turn opener's own bytes, where the replay begins with them: the template's framing,
        // not text of the stay.
        let framing = self
            .turn_opener()
            .filter(|opener| self.anchors.is_empty() && turn.starts_with(opener))
            .map_or(0, str::len);
        let mut scanner = Scanner::new(self.terminal_texts());
        let (mut state, mut rest) = (start, start);
        // Whether the prompt wrote text other than whitespace in the stay the output continues;
        // a marker is the template's, whether or not a row reads it here.
        let mut begun = false;
        let mut at = 0;
        // An anchor that names its own state leaves no start to choose.
        let mut at_turn_start = self.start.is_some() && start == 0;
        let pieces = scanner.feed(turn).into_iter().chain(scanner.finish());
        for piece in pieces {
            if at_turn_start {
                if let Some(opened) = self.opening(state, &piece) {
                    at_turn_start = false;
                    if opened != state {
                        begun = false;
                    }
                    (state, rest) = (opened, opened);
                }
            }
            let from = at;
            at += self.piece_len(&piece);
            match piece {
                Piece::Marker(index) if self.ignored(index) => {}
                Piece::Marker(index) => {
                    if let Some(to) = self.next(state, index) {
                        (state, begun) = (to, false);
                    }
                }
                Piece::Text(text) => {
                    let written = &text[framing.saturating_sub(from).min(text.len())..];
                    begun |= !written.chars().all(is_strip_space);
                }
            }
        }
        if self.emits(state) == Emits::Arguments {
            (state, begun) = (rest, false);
        }
        Replay {
            state,
            at_turn_start,
            begun,
        }
    }

    /// The state `prompt` leaves the table in ([`Self::replayed`]): where the engine starts the
    /// output.
    pub(crate) fn prompt_state(&self, prompt: &str) -> usize {
        self.replayed(prompt).state
    }

    /// Whether `prompt` leaves the model inside its thought: its terminals replayed over the table
    /// from the turn's opener or its last anchor, as the engine replays them before the output.
    /// The value the structural tags take as `reasoning_open` ([`Format::grammar`],
    /// [`Format::answer_grammar`]), read where the prompt ends, whatever it writes after the
    /// opener: Qwen 3.5's and DeepSeek's templates open the thought there, and a template with
    /// thinking turned off opens and closes it there.
    pub fn reasoning_open(&self, prompt: &str) -> bool {
        self.emits(self.prompt_state(prompt)) == Emits::Reasoning
    }

    /// The format's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the table reads calls from the output's first byte, with no marker before them:
    /// the output starts in an arguments state that no row enters, as xLAM's list and the `json`
    /// table do. Such a table reads a bare JSON answer as a call. Inkling starts in an arguments
    /// state too, but a row enters it, and its call has a marker after the tool's name; Llama
    /// Nemotron writes xLAM's list, but between markers: neither reads a bare answer as a call.
    pub fn starts_in_calls(&self) -> bool {
        let starts_in_arguments = self
            .states
            .first()
            .is_some_and(|state| state.emits == Emits::Arguments);
        starts_in_arguments
            && !self
                .transitions
                .iter()
                .any(|row| self.states[row.to].emits == Emits::Arguments)
    }

    fn state_named(&self, name: &str) -> usize {
        self.states
            .iter()
            .position(|state| state.name == name)
            .unwrap_or_else(|| unknown("state", name, &self.name))
    }

    fn terminal_named(&self, name: &str) -> usize {
        self.terminals
            .iter()
            .position(|terminal| terminal.name == name && !terminal.ignored)
            .unwrap_or_else(|| unknown("terminal", name, &self.name))
    }

    /// The terminals' texts, in the order the engine's scanner numbers them: the markers the
    /// table's rows move on, and those it ignores. Whoever learns the table's [`Layout`] from a
    /// rendering reads them to tell a marker from the text the layout writes around it.
    pub fn terminal_texts(&self) -> impl Iterator<Item = &str> {
        self.terminals.iter().map(|terminal| terminal.text.as_str())
    }

    /// Whether `tail`, at the end of the output in `state`, is a tag the end cut short: it begins
    /// with a whole structural token and is the beginning of a terminal that, whole, would move
    /// the engine out of `state` or be dropped where it stands. The beginning of a terminal the
    /// state has no row for is text, as the whole terminal would be.
    pub(crate) fn is_cut_tag(&self, state: usize, tail: &str) -> bool {
        self.structural_tokens
            .iter()
            .any(|token| tail.starts_with(token.as_str()))
            && self.terminals.iter().enumerate().any(|(index, terminal)| {
                terminal.text.len() > tail.len()
                    && terminal.text.starts_with(tail)
                    && (terminal.ignored || self.next(state, index).is_some())
            })
    }

    /// The text of terminal `index`.
    pub(crate) fn terminal_text(&self, index: usize) -> &str {
        &self.terminals[index].text
    }

    /// Whether terminal `index` is one the engine drops wherever it stands.
    pub(crate) fn ignored(&self, index: usize) -> bool {
        self.terminals[index].ignored
    }

    /// Whether terminal `index` is read as content text where it moves the engine.
    pub(crate) fn shown(&self, index: usize) -> bool {
        self.terminals[index].shown
    }

    /// What the text in state `index` is.
    pub(crate) fn emits(&self, index: usize) -> Emits {
        self.states[index].emits
    }

    /// How many states the table has; the first is the one the turn opener leaves the engine in.
    pub(crate) fn states(&self) -> usize {
        self.states.len()
    }

    /// Where terminal `on` takes the engine from state `from`, if the table says.
    pub(crate) fn next(&self, from: usize, on: usize) -> Option<usize> {
        self.transitions
            .iter()
            .find(|transition| transition.from == from && transition.on == on)
            .map(|transition| transition.to)
    }

    /// Every row of the table as `(from, on, to)`, in the order the rows were added.
    pub(crate) fn transitions(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.transitions
            .iter()
            .map(|transition| (transition.from, transition.on, transition.to))
    }

    /// The call syntax, for a format with an arguments state: how the model writes a call there.
    /// For a call given as data it is the shape's patterns, which whoever compiled the table reads
    /// again to learn the table's [`Layout`] from a rendering of the model's chat template.
    pub fn call_syntax(&self) -> Option<&CallSyntax> {
        self.calls.as_ref()
    }

    /// Whether the stream's stop closes the call in state `index`.
    pub(crate) fn stop_closes(&self, index: usize) -> bool {
        self.states[index].closes_at_stop
    }

    /// Whether the template discards the text in state `index`.
    pub(crate) fn is_framing(&self, index: usize) -> bool {
        self.states[index].framing
    }

    /// Whether the text of state `index` is trimmed stay by stay.
    pub(crate) fn is_trimmed(&self, index: usize) -> bool {
        self.states[index].trimmed
    }

    /// What every table must hold before an engine runs it: a state to start in, which does not
    /// emit reasoning (a thought the output starts inside is entered by the prompt's replay or by
    /// the start row, either of which pushes `ReasoningStart`; an arguments state at the start has
    /// its call opened by the engine), a call syntax when a state emits arguments, a text for
    /// every terminal, a turn opener and turn anchors that are not empty and not both named, a
    /// start row only where a turn's first byte is read from the first state and no opener replays
    /// it, a stop row only for an arguments state, a framing row only for a state of content or
    /// wrapping, a trim row only for a state of content or reasoning that is not framing,
    /// a thought when the table thinks first, and, for a shaped call, a shape a stream can run,
    /// opened only through its terminal (the turn does not start in its state), no terminal the
    /// table ignores, and, when its header is not empty, no row into its state from a state that
    /// holds a call. The builder keeps a row from naming what the table lacks; this checks the
    /// rest.
    pub fn validate(&self) -> Result<(), String> {
        let Some(first) = self.states.first() else {
            return Err(format!("format {}: a table with no state", self.name));
        };
        if first.emits == Emits::Reasoning {
            return Err(format!(
                "format {}: the first state {:?} emits reasoning; a thought the output starts \
                 inside is entered by the prompt",
                self.name, first.name
            ));
        }
        if self
            .states
            .iter()
            .any(|state| state.emits == Emits::Arguments)
            && self.calls.is_none()
        {
            return Err(format!(
                "format {}: an arguments state with no call syntax",
                self.name
            ));
        }
        if let Some(terminal) = self
            .terminals
            .iter()
            .find(|terminal| terminal.text.is_empty())
        {
            return Err(format!(
                "format {}: terminal {:?} has no text",
                self.name, terminal.name
            ));
        }
        if self.turn_opener.as_deref() == Some("") {
            return Err(format!("format {}: an empty turn opener", self.name));
        }
        if self.anchors.iter().any(|anchor| anchor.text.is_empty()) {
            return Err(format!("format {}: an empty turn anchor", self.name));
        }
        if self.turn_opener.is_some() && !self.anchors.is_empty() {
            return Err(format!(
                "format {}: a turn opener and turn anchors; a table names one",
                self.name
            ));
        }
        if self.start.is_some() {
            if first.emits == Emits::Arguments {
                return Err(format!(
                    "format {}: a start row, but the first state {:?} holds a call",
                    self.name, first.name
                ));
            }
            if self.turn_opener.is_some() {
                return Err(format!(
                    "format {}: a start row needs the turn's start, which a turn opener replays",
                    self.name
                ));
            }
        }
        if let Some(state) = self
            .states
            .iter()
            .find(|state| state.closes_at_stop && state.emits != Emits::Arguments)
        {
            return Err(format!(
                "format {}: a stop closes state {:?}, which holds no call",
                self.name, state.name
            ));
        }
        if let Some(state) = self.states.iter().find(|state| {
            state.framing && matches!(state.emits, Emits::Reasoning | Emits::Arguments)
        }) {
            return Err(format!(
                "format {}: state {:?} is framing, but it emits {:?}",
                self.name, state.name, state.emits
            ));
        }
        let misfit = self
            .states
            .iter()
            .filter(|state| state.trimmed)
            .find_map(|state| {
                if matches!(state.emits, Emits::Arguments | Emits::Wrapper) {
                    Some(format!(
                        "format {}: state {:?} is trimmed, but it emits {:?}",
                        self.name, state.name, state.emits
                    ))
                } else if state.framing {
                    Some(format!(
                        "format {}: state {:?} is trimmed, but its text is framing",
                        self.name, state.name
                    ))
                } else {
                    None
                }
            });
        if let Some(why) = misfit {
            return Err(why);
        }
        if self.thinks_first
            && !self
                .states
                .iter()
                .any(|state| state.emits == Emits::Reasoning)
        {
            return Err(format!(
                "format {}: a table that thinks first has no thought",
                self.name
            ));
        }
        if let Some(CallSyntax::Shaped(shape)) = self.calls {
            self.validate_shaped(shape)?;
        }
        Ok(())
    }

    /// What a table with a shaped call must hold besides: a shape a stream can run, every call
    /// opened by a terminal (a shaped call opens through its terminal and its header, so the
    /// output does not start in one, nor does a start row lead into one), no terminal the table
    /// ignores (a shaped call has no place to keep one among the bytes it holds), and, when the
    /// header is not empty, no row from a state that holds a call into one: a terminal whose
    /// header does not match is the text of the state it came in, which a closed call cannot take.
    fn validate_shaped(&self, shape: &CallShape) -> Result<(), String> {
        if let Err(why) = shape.check() {
            return Err(format!("format {}: the call's shape: {why}", self.name));
        }
        let calls = |index: usize| self.states[index].emits == Emits::Arguments;
        let anchored = self.anchors.iter().any(|anchor| calls(anchor.state));
        if calls(0) || self.start.is_some_and(calls) || anchored {
            return Err(format!(
                "format {}: a shaped call opens through its terminal, but the turn may start in \
                 one",
                self.name
            ));
        }
        if let Some(terminal) = self.terminals.iter().find(|terminal| terminal.ignored) {
            return Err(format!(
                "format {}: terminal {:?} is ignored, which a shaped call cannot keep in place",
                self.name, terminal.name
            ));
        }
        if shape.header.0.is_empty() {
            return Ok(());
        }
        match self
            .transitions
            .iter()
            .find(|row| calls(row.from) && calls(row.to))
        {
            Some(row) => Err(format!(
                "format {}: a row from {:?} into {:?}, both holding calls, but the call's header \
                 may not match, and then its terminal is the text of a state that holds none",
                self.name, self.states[row.from].name, self.states[row.to].name
            )),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_states() -> Format {
        Format::new("t")
            .terminal("open", "<a>")
            .terminal("close", "</a>")
            .state("content", Emits::Content)
            .state("inside", Emits::Reasoning)
            .transition("content", "open", "inside")
            .transition("inside", "close", "content")
    }

    #[test]
    fn rows_are_looked_up_by_the_names_they_were_added_under() {
        let format = two_states();
        assert_eq!(format.name(), "t");
        assert_eq!(format.terminal_texts().collect::<Vec<_>>(), ["<a>", "</a>"]);
        assert_eq!(format.terminal_text(1), "</a>");
        assert_eq!(format.emits(0), Emits::Content);
        assert_eq!(format.emits(1), Emits::Reasoning);
        assert_eq!(format.next(0, 0), Some(1));
        assert_eq!(format.next(1, 1), Some(0));
        assert_eq!(format.next(0, 1), None, "a closer in content has no row");
        assert_eq!(format.next(1, 0), None, "an opener inside has no row");
        assert!(format.call_syntax().is_none());
        assert!(format.validate().is_ok());
        assert_eq!(
            Format::new("empty").validate(),
            Err("format empty: a table with no state".to_string())
        );
        assert_eq!(
            Format::new("calls")
                .state("calls", Emits::Arguments)
                .validate(),
            Err("format calls: an arguments state with no call syntax".to_string())
        );
        assert_eq!(
            Format::new("blank")
                .terminal("t", "")
                .state("c", Emits::Content)
                .validate(),
            Err("format blank: terminal \"t\" has no text".to_string())
        );
        assert_eq!(
            Format::new("opener")
                .state("c", Emits::Content)
                .opens_turn("")
                .validate(),
            Err("format opener: an empty turn opener".to_string())
        );
        assert_eq!(
            Format::new("first")
                .state("c", Emits::Content)
                .thinks_first()
                .validate(),
            Err("format first: a table that thinks first has no thought".to_string())
        );
        assert!(two_states().thinks_first().validate().is_ok());
        assert_eq!(
            Format::new("thought")
                .state("r", Emits::Reasoning)
                .validate(),
            Err(
                "format thought: the first state \"r\" emits reasoning; a thought the output \
                 starts inside is entered by the prompt"
                    .to_string()
            )
        );
    }

    /// A thought and calls, the calls' opener read inside the thought too, as Gemma 4 and dots3
    /// have it.
    fn with_calls() -> Format {
        Format::new("q")
            .terminal("think_open", "<think>")
            .terminal("think_close", "</think>")
            .terminal("call_open", "<tool_call>")
            .terminal("call_close", "</tool_call>")
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("calls", Emits::Arguments)
            .transition("content", "think_open", "reasoning")
            .transition("reasoning", "think_close", "content")
            .transition("content", "call_open", "calls")
            .transition("calls", "call_close", "content")
            .transition("reasoning", "call_open", "calls")
            .calls(CallSyntax::Json)
            .opens_turn("<|im_start|>assistant")
    }

    fn events(format: Format, prompt: &str, output: &str) -> Vec<crate::event::Event> {
        use crate::{
            engine::Engine,
            event::Events,
            input::{EngineFinish, Input},
            parser::Parser,
            tagged::Declared,
        };
        let mut parser = Engine::new(format, Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut out,
            )
            .expect("prompt");
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: output,
                    spans: &[],
                },
                &mut out,
            )
            .expect("delta");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        out.drain()
    }

    fn said(events: &[crate::event::Event]) -> (String, String, usize) {
        use crate::event::Event;
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls = 0;
        for event in events {
            match event {
                Event::Content(text) => content.push_str(&text.text),
                Event::Reasoning(text) => reasoning.push_str(&text.text),
                Event::ToolCallStart { .. } => calls += 1,
                _ => {}
            }
        }
        (content, reasoning, calls)
    }

    #[test]
    fn without_reasoning_shows_the_thoughts_markers_and_reads_the_calls() {
        let format = with_calls().without_reasoning();
        assert!(format.validate().is_ok());
        assert_eq!(
            format.emits(1),
            Emits::Content,
            "the thought is content now"
        );
        assert_eq!(
            format.next(0, 0),
            Some(1),
            "its opener still moves the engine"
        );
        assert!(
            format.shown(0) && format.shown(1),
            "and is shown, as its closer is"
        );
        assert!(
            !format.shown(2) && !format.shown(3),
            "the call's markers are not"
        );
        let output = "<think>a thought</think>an answer<tool_call>\n{\"name\": \"f\", \
                      \"arguments\": {}}\n</tool_call>";
        let (content, reasoning, calls) = said(&events(format.clone(), "", output));
        assert_eq!(content, "<think>a thought</think>an answer");
        assert_eq!(reasoning, "");
        assert_eq!(calls, 1);
        // A prompt that opened the thought leaves the output in it: its text and its closer are
        // content, and the call after it reads.
        let (content, reasoning, calls) = said(&events(
            format,
            "<|im_start|>assistant\n<think>\n",
            "still a thought</think><tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n\
             </tool_call>",
        ));
        assert_eq!(
            (content.as_str(), reasoning.as_str(), calls),
            ("still a thought</think>", "", 1)
        );
        // A table without a thought or a marker is itself.
        let plain = Format::new("p").state("c", Emits::Content);
        assert_eq!(plain.clone().without_reasoning(), plain);
    }

    #[test]
    fn without_calls_shows_the_calls_markers_and_reads_the_thought() {
        let format = with_calls().without_calls();
        assert!(format.validate().is_ok());
        assert_eq!(
            format.emits(2),
            Emits::Content,
            "the calls state is content now"
        );
        assert_eq!(format.call_syntax(), None);
        assert!(
            format.shown(2) && format.shown(3),
            "the call's markers are shown"
        );
        assert!(
            !format.shown(0) && !format.shown(1),
            "the thought's are not"
        );
        let output = "<think>a thought</think><tool_call>\n{\"name\": \"f\", \
                      \"arguments\": {}}\n</tool_call>";
        let (content, reasoning, calls) = said(&events(format, "", output));
        assert_eq!(
            content,
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>"
        );
        assert_eq!(reasoning, "a thought");
        assert_eq!(calls, 0);
        // A trimmed stay is still trimmed up to the shown marker; the call's own text is not.
        let trimmed = with_calls().trims("content").without_calls();
        let (content, _, _) = said(&events(
            trimmed,
            "",
            "an answer \n<tool_call>\n{}\n</tool_call>",
        ));
        assert_eq!(content, "an answer<tool_call>\n{}\n</tool_call>");
        // A stop that closed the call closes nothing once the call is content.
        let stopped = with_calls().closes_at_stop("calls").without_calls();
        assert!(stopped.validate().is_ok());
        assert!(!stopped.stop_closes(2));
        // A table whose output starts in its calls is content from its first byte.
        let alone = Format::new("j")
            .state("calls", Emits::Arguments)
            .calls(CallSyntax::JsonList)
            .opens_turn("<|a|>")
            .without_calls();
        assert!(alone.validate().is_ok());
        assert_eq!(alone.turn_opener(), Some("<|a|>"));
        assert_eq!(alone.emits(0), Emits::Content);
        let (content, _, calls) =
            said(&events(alone, "", "[{\"name\": \"f\", \"arguments\": {}}]"));
        assert_eq!(content, "[{\"name\": \"f\", \"arguments\": {}}]");
        assert_eq!(calls, 0);
    }

    /// Kimi K3's turn after its opener: a thought, an answer and one call.
    const KIMI_TURN: &str = concat!(
        "<|open|>think<|sep|>a thought<|close|>think<|sep|>",
        "<|open|>response<|sep|>an answer<|close|>response<|sep|>",
        "<|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>",
        "<|open|>argument key=\"q\" type=\"string\"<|sep|>x<|close|>argument<|sep|>",
        "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
    );
    const KIMI_OPENER: &str = "<|open|>message role=\"assistant\"<|sep|>";

    #[test]
    fn without_reasoning_shows_kimi_k3s_frame_and_still_reads_its_call() {
        // The legacy reasoning parser removes the think, response and message tags and leaves the
        // tools channel to the tool parser; without it those tags are text, and the call reads.
        let format = crate::formats::kimi_k3().without_reasoning();
        assert!(format.validate().is_ok());
        let (content, reasoning, calls) = said(&events(format, KIMI_OPENER, KIMI_TURN));
        assert_eq!(
            content,
            "<|open|>think<|sep|>a thought<|close|>think<|sep|><|open|>response<|sep|>an answer\
             <|close|>response<|sep|><|close|>message<|sep|>"
        );
        assert_eq!(reasoning, "");
        assert_eq!(calls, 1);
        // A tag the end cut short is dropped, shown or not: the legacy parsers drop it on both
        // paths, so a response the length cut inside its closing tag ends the same way.
        let cut = "<|open|>response<|sep|>an answer<|close|>response";
        let out = events(
            crate::formats::kimi_k3().without_reasoning(),
            KIMI_OPENER,
            cut,
        );
        assert_eq!(said(&out).0, "<|open|>response<|sep|>an answer");
        let dropped = out.iter().rev().nth(1);
        assert!(
            matches!(dropped, Some(crate::event::Event::Dropped { text, .. })
                if text.text == "<|close|>response"),
            "{out:?}"
        );
    }

    #[test]
    fn without_calls_keeps_kimi_k3s_frame_and_reads_its_thought_and_answer() {
        let format = crate::formats::kimi_k3().without_calls();
        assert!(format.validate().is_ok());
        assert_eq!(
            format.emits(0),
            Emits::Wrapper,
            "the message is the turn's frame"
        );
        let out = events(format, KIMI_OPENER, KIMI_TURN);
        let (content, reasoning, calls) = said(&out);
        assert_eq!(reasoning, "a thought");
        assert_eq!(
            content,
            "an answer<|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>\
             <|open|>argument key=\"q\" type=\"string\"<|sep|>x<|close|>argument<|sep|>\
             <|close|>call<|sep|><|close|>tools<|sep|>"
        );
        assert_eq!(calls, 0);
        assert!(
            !out.iter()
                .any(|e| matches!(e, crate::event::Event::Malformed { .. })),
            "{out:?}"
        );
    }

    /// Inkling's turn after its opener: a thought, an answer and one call, then the sampling end.
    const INKLING_TURN: &str = concat!(
        "<|content_thinking|>check sources<|end_message|>",
        "<|message_model|><|content_text|>Here is the answer.<|end_message|>",
        "<|message_model|>search<|content_invoke_tool_json|>",
        "{\"name\": \"search\", \"args\": {\"q\": \"rust\"}}<|end_message|>",
        "<|content_model_end_sampling|>"
    );

    #[test]
    fn without_calls_keeps_inklings_thought_and_shows_its_call_message_as_text() {
        // The message header is Inkling's arguments state, so its framing is shown here where the
        // legacy reasoning parser removes it; the thought still reads, the ignored terminal is
        // still dropped, and the call's message is text.
        let format = crate::formats::inkling().without_calls();
        assert!(format.validate().is_ok());
        let out = events(format, "<|message_model|>", INKLING_TURN);
        let (content, reasoning, calls) = said(&out);
        assert_eq!(reasoning, "check sources");
        assert_eq!(calls, 0);
        assert!(
            content.contains("Here is the answer.")
                && content.contains(
                    "search<|content_invoke_tool_json|>{\"name\": \"search\", \"args\": \
                     {\"q\": \"rust\"}}"
                ),
            "{content}"
        );
        assert!(
            !content.contains("<|content_model_end_sampling|>"),
            "{content}"
        );
    }

    #[test]
    fn without_reasoning_reads_inklings_call_after_a_thinking_message() {
        let format = crate::formats::inkling().without_reasoning();
        assert!(format.validate().is_ok());
        let (content, reasoning, calls) = said(&events(format, "<|message_model|>", INKLING_TURN));
        assert_eq!(reasoning, "");
        assert_eq!(calls, 1);
        assert_eq!(content, "check sourcesHere is the answer.");
    }

    #[test]
    fn without_reasoning_keeps_both_of_minimax_m3s_thought_markers() {
        let format = crate::formats::minimax_m3().without_reasoning();
        let output = "<mm:think>a thought</mm:think>an answer";
        let (content, reasoning, calls) = said(&events(format, "]~b]ai\n", output));
        assert_eq!(content, output);
        assert_eq!((reasoning.as_str(), calls), ("", 0));
    }

    #[test]
    fn without_reasoning_reads_a_call_after_a_thought_the_turn_starts_in() {
        // ERNIE-style: the turn starts inside the thought, with no opener to read.
        let format = Format::new("e")
            .terminal("think_close", "</think>")
            .terminal("call_open", "<tool_call>")
            .terminal("call_close", "</tool_call>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("call", Emits::Arguments)
            .transition("reasoning", "think_close", "outside")
            .transition("outside", "call_open", "call")
            .transition("call", "call_close", "outside")
            .calls(CallSyntax::Json)
            .turn_anchor("Assistant:")
            .starts_in("reasoning")
            .without_reasoning();
        assert!(format.validate().is_ok());
        let output =
            "a thought</think>\n<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>";
        let (content, reasoning, calls) = said(&events(format, "User: hi\nAssistant:", output));
        assert_eq!(
            (content.as_str(), reasoning.as_str(), calls),
            ("a thought</think>\n", "", 1)
        );
    }

    #[test]
    fn without_reasoning_drops_the_thought_first_mark_with_the_thought() {
        let format = with_calls().thinks_first();
        assert!(format.validate().is_ok());
        assert!(format.without_reasoning().validate().is_ok());
    }

    #[test]
    fn a_stop_row_names_an_arguments_state() {
        let format = Format::new("stop")
            .terminal("open", "<call>")
            .state("content", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("content", "open", "call")
            .calls(CallSyntax::Json)
            .closes_at_stop("call");
        assert!(format.stop_closes(1));
        assert!(!format.stop_closes(0));
        assert!(format.validate().is_ok());
        assert_eq!(
            format.closes_at_stop("content").validate(),
            Err("format stop: a stop closes state \"content\", which holds no call".to_string())
        );
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"call\"")]
    fn a_stop_row_for_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().closes_at_stop("call");
    }

    /// Gemma 4's two anchors, spelled from the response template of `google/gemma-4-31B-it` (its
    /// `tokenizer_config.json`), around a thought and the content.
    fn gemma_anchors() -> Format {
        Format::new("anchors")
            .terminal("think_open", "<|channel>thought\n")
            .terminal("think_close", "<channel|>")
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .transition("content", "think_open", "reasoning")
            .transition("reasoning", "think_close", "content")
            .turn_anchor("<|turn>model\n")
            .turn_anchor("<tool_response|>")
    }

    #[test]
    fn the_last_anchor_of_either_kind_starts_the_turn_after_its_end() {
        let format = gemma_anchors();
        let turn = "<|turn>user\nWeather?<turn|>\n<|turn>model\n";
        assert_eq!(format.replay(turn), "<|turn>model\n");
        assert_eq!(format.turn(turn), ("", 0));
        let after_a_tool = "<|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>\
                            response:f{}<tool_response|><|channel>thought\n";
        assert_eq!(
            format.replay(after_a_tool),
            "<tool_response|><|channel>thought\n"
        );
        assert_eq!(format.turn(after_a_tool), ("<|channel>thought\n", 0));
        let later_turn = "<|turn>model\n<tool_response|>Done.<turn|>\n<|turn>user\nThanks<turn|>\n\
                          <|turn>model\n";
        assert_eq!(format.turn(later_turn), ("", 0));
    }

    #[test]
    fn anchors_are_found_as_finditer_finds_them() {
        let doubled = Format::new("doubled")
            .state("content", Emits::Content)
            .turn_anchor("aa");
        // `finditer` takes `aa` at 0 and goes on from 2: the turn starts after the first two.
        assert_eq!(doubled.turn("aaa"), ("a", 0));
        assert_eq!(doubled.turn("aaaa"), ("", 0));
        let longest_first = Format::new("longest first")
            .state("content", Emits::Content)
            .turn_anchor("ab")
            .turn_anchor("a");
        assert_eq!(longest_first.turn("xab"), ("", 0));
        let shortest_first = Format::new("shortest first")
            .state("content", Emits::Content)
            .turn_anchor("a")
            .turn_anchor("ab");
        assert_eq!(shortest_first.turn("xab"), ("b", 0));
        // A later anchor that starts inside the last match is not one.
        let overlapping = Format::new("overlapping")
            .state("content", Emits::Content)
            .turn_anchor("<a><b>")
            .turn_anchor("<b>c");
        assert_eq!(overlapping.turn("<a><b>c"), ("c", 0));
        assert_eq!(overlapping.turn("<b>c<a><b>c"), ("c", 0));
    }

    #[test]
    fn a_prompt_without_an_anchor_replays_nothing_and_a_replay_gives_itself() {
        let format = gemma_anchors();
        assert_eq!(format.replay("hello "), "");
        assert_eq!(format.turn("hello "), ("", 0));
        assert_eq!(format.turn(""), ("", 0));
        for prompt in [
            "<|turn>model\n",
            "<|turn>model\n<|channel>thought\nPlan",
            "a<tool_response|>b<|turn>model\nc<tool_response|>d",
            "no anchor at all",
        ] {
            let kept = format.replay(prompt);
            assert_eq!(format.replay(kept), kept, "{prompt:?}");
            assert_eq!(format.turn(kept), format.turn(prompt), "{prompt:?}");
        }
    }

    #[test]
    fn an_anchor_names_the_state_the_output_starts_in() {
        // TRL's Nemotron 3 template: the anchor pattern `<\|im_start\|>assistant\n<think>(?:\n|
        // </think>)?` is three literals, the longer ones first as the greedy `?` tries them.
        let format = Format::new("nemotron 3")
            .terminal("think_close", "</think>")
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .transition("reasoning", "think_close", "content")
            .turn_anchor_starting_in("<|im_start|>assistant\n<think>\n", "reasoning")
            .turn_anchor_starting_in("<|im_start|>assistant\n<think></think>", "content")
            .turn_anchor_starting_in("<|im_start|>assistant\n<think>", "reasoning");
        assert!(format.validate().is_ok());
        assert_eq!(
            format.turn("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"),
            ("", 1)
        );
        assert_eq!(
            format.turn("<|im_start|>assistant\n<think></think>"),
            ("", 0)
        );
        assert_eq!(format.turn("<|im_start|>assistant\n<think>"), ("", 1));
        assert_eq!(format.turn("<|im_start|>assistant\n<think>x"), ("x", 1));
    }

    #[test]
    fn an_anchor_is_not_empty_and_a_table_names_an_opener_or_anchors() {
        assert_eq!(
            Format::new("empty")
                .state("c", Emits::Content)
                .turn_anchor("")
                .validate(),
            Err("format empty: an empty turn anchor".to_string())
        );
        assert_eq!(
            Format::new("both")
                .state("c", Emits::Content)
                .opens_turn("<|assistant|>")
                .turn_anchor("<|assistant|>")
                .validate(),
            Err("format both: a turn opener and turn anchors; a table names one".to_string())
        );
        // An empty anchor never matches, even in a table nobody validated.
        let unchecked = Format::new("unchecked")
            .state("c", Emits::Content)
            .turn_anchor("")
            .turn_anchor("x");
        assert_eq!(unchecked.turn("axb"), ("b", 0));
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"thought\"")]
    fn an_anchor_starting_in_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().turn_anchor_starting_in("<turn>", "thought");
    }

    #[test]
    fn a_start_row_needs_a_first_state_without_a_call_and_no_opener() {
        let format = two_states().starts_in("inside");
        assert_eq!(format.start(), Some(1));
        assert!(format.validate().is_ok());
        assert_eq!(two_states().start(), None);
        assert_eq!(
            Format::new("list")
                .state("calls", Emits::Arguments)
                .state("content", Emits::Content)
                .calls(CallSyntax::JsonList)
                .starts_in("content")
                .validate(),
            Err("format list: a start row, but the first state \"calls\" holds a call".to_string())
        );
        assert_eq!(
            format.opens_turn("<turn>").validate(),
            Err(
                "format t: a start row needs the turn's start, which a turn opener replays"
                    .to_string()
            )
        );
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"thought\"")]
    fn a_start_row_for_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().starts_in("thought");
    }

    #[test]
    fn a_framing_row_names_a_state_of_content_or_wrapping() {
        let format = Format::new("framing")
            .terminal("open", "<call>")
            .state("outside", Emits::Content)
            .state("between", Emits::Wrapper)
            .state("thought", Emits::Reasoning)
            .state("call", Emits::Arguments)
            .transition("outside", "open", "call")
            .calls(CallSyntax::Json)
            .framing("outside")
            .framing("between")
            .framing("outside");
        let framed: Vec<usize> = (0..format.states())
            .filter(|&s| format.is_framing(s))
            .collect();
        assert_eq!(framed, [0, 1], "a state named twice is kept once");
        assert!(format.is_framing(0));
        assert!(format.is_framing(1));
        assert!(!format.is_framing(2));
        assert!(!format.is_framing(3));
        assert!(format.validate().is_ok());
        assert_eq!(
            format.clone().framing("thought").validate(),
            Err("format framing: state \"thought\" is framing, but it emits Reasoning".to_string())
        );
        assert_eq!(
            format.framing("call").validate(),
            Err("format framing: state \"call\" is framing, but it emits Arguments".to_string())
        );
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"outside\"")]
    fn a_framing_row_for_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().framing("outside");
    }

    #[test]
    fn a_trim_row_names_a_state_of_content_or_reasoning_that_is_not_framing() {
        let format = Format::new("trimmed")
            .terminal("open", "<call>")
            .state("content", Emits::Content)
            .state("thought", Emits::Reasoning)
            .state("between", Emits::Wrapper)
            .state("call", Emits::Arguments)
            .state("outside", Emits::Content)
            .transition("content", "open", "call")
            .calls(CallSyntax::Json)
            .framing("outside")
            .trims("content")
            .trims("thought")
            .trims("content");
        let trimmed: Vec<usize> = (0..format.states())
            .filter(|&s| format.is_trimmed(s))
            .collect();
        assert_eq!(trimmed, [0, 1], "a state named twice is kept once");
        assert!(format.is_trimmed(0) && format.is_trimmed(1) && !format.is_trimmed(2));
        assert!(format.validate().is_ok());
        assert_eq!(
            format.clone().trims("between").validate(),
            Err("format trimmed: state \"between\" is trimmed, but it emits Wrapper".to_string())
        );
        assert_eq!(
            format.clone().trims("call").validate(),
            Err("format trimmed: state \"call\" is trimmed, but it emits Arguments".to_string())
        );
        assert_eq!(
            format.trims("outside").validate(),
            Err(
                "format trimmed: state \"outside\" is trimmed, but its text is framing".to_string()
            )
        );
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"answer\"")]
    fn a_trim_row_for_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().trims("answer");
    }

    #[test]
    fn the_stop_tokens_kept_are_the_terminals_among_the_stops_each_once() {
        let format = two_states().terminal("call", "<|call|>").stop_tokens([
            "<|return|>",
            "<|call|>",
            "</a>",
            "<|call|>",
        ]);
        assert!(format.is_stop_token("<|call|>"));
        assert!(format.is_stop_token("</a>"));
        assert!(
            !format.is_stop_token("<|return|>"),
            "a stop the table has no terminal for is not kept"
        );
        assert!(!format.is_stop_token("<a>"));
        assert_eq!(format.stop_tokens, ["<|call|>", "</a>"]);
        assert!(
            !two_states()
                .stop_tokens(["<|call|>"])
                .is_stop_token("<|call|>"),
            "a stop named before its terminal is not one yet"
        );
        assert!(format.validate().is_ok());
    }

    /// A shape on abstract tags, its header `<n=NAME>` when `header`, empty otherwise.
    fn shape(header: bool) -> CallShape {
        use crate::shape::{Arguments, Atom, Class, Dialect, NameSource, Pattern, Slot};
        let word = Class::new([('a', 'z')], false);
        let (header, name) = if header {
            let atoms = [
                Atom::Lit("<n=".into()),
                Atom::Run {
                    class: word.clone(),
                    min: 1,
                    max: None,
                    greedy: true,
                    capture: Some(Slot::Name),
                },
                Atom::Lit(">".into()),
            ];
            (Pattern(atoms.into()), NameSource::Header)
        } else {
            (Pattern::default(), NameSource::Literal("f".into()))
        };
        CallShape {
            header,
            name,
            arguments: Arguments::Json(Dialect::default()),
            trailer: Pattern::default(),
            word,
            hold_cap: 64,
        }
    }

    fn shaped(header: bool) -> CallSyntax {
        CallSyntax::Shaped(crate::shape::intern(shape(header)))
    }

    #[test]
    fn a_shaped_call_opens_from_a_state_that_holds_none() {
        let table = |syntax: CallSyntax| {
            Format::new("shaped")
                .terminal("open", "<c>")
                .terminal("close", "</c>")
                .state("content", Emits::Content)
                .state("call", Emits::Arguments)
                .transition("content", "open", "call")
                .transition("call", "close", "content")
                .calls(syntax)
        };
        assert!(table(shaped(true)).validate().is_ok());
        assert_eq!(
            table(shaped(true))
                .transition("call", "open", "call")
                .validate(),
            Err(
                "format shaped: a row from \"call\" into \"call\", both holding calls, but the \
                 call's header may not match, and then its terminal is the text of a state that \
                 holds none"
                    .to_string()
            )
        );
        assert!(
            table(shaped(false))
                .transition("call", "open", "call")
                .validate()
                .is_ok(),
            "a call with no header always opens at its terminal"
        );
        assert_eq!(
            table(shaped(true)).ignores("sep", "|").validate(),
            Err(
                "format shaped: terminal \"sep\" is ignored, which a shaped call cannot keep in \
                 place"
                    .to_string()
            )
        );
        for starts_in_a_call in [
            table(shaped(true)).starts_in("call"),
            table(shaped(true)).turn_anchor_starting_in("<turn>", "call"),
        ] {
            assert_eq!(
                starts_in_a_call.validate(),
                Err(
                    "format shaped: a shaped call opens through its terminal, but the turn may \
                     start in one"
                        .to_string()
                )
            );
        }
        let unnamed = CallSyntax::Shaped(crate::shape::intern(CallShape {
            name: crate::shape::NameSource::Body,
            ..shape(true)
        }));
        assert_eq!(
            table(unnamed).validate(),
            Err(
                "format shaped: the call's shape: the name comes from the arguments, which are \
                 no envelope"
                    .to_string()
            )
        );
    }

    #[test]
    fn the_prompts_replay_walks_the_rows_from_the_last_opener() {
        // Terminals before the last opener move nothing; after it, each moves the walk where its
        // row says, a terminal with no row from where the walk stands moving nothing; a terminal
        // the prompt cut short is text.
        let format = two_states().opens_turn("<turn>");
        assert_eq!(format.prompt_state("<turn>"), 0);
        assert_eq!(format.prompt_state("<turn><a>"), 1);
        assert!(format.reasoning_open("<turn><a>"));
        assert!(!format.reasoning_open("<a><turn>"));
        assert!(!format.reasoning_open("<turn><a>x</a>"));
        assert!(
            format.reasoning_open("<turn></a><a>"),
            "a close in content is text"
        );
        assert!(
            format.reasoning_open("<turn><a></a"),
            "a cut close moves nothing"
        );
        assert!(format.reasoning_open("<a>"), "no opener: the whole prompt");
    }

    #[test]
    fn an_ignored_terminal_is_scanned_for_and_cannot_be_named_in_a_row() {
        let format = two_states().ignores("sep", "|");
        assert_eq!(
            format.terminal_texts().collect::<Vec<_>>(),
            ["<a>", "</a>", "|"]
        );
        assert!(format.ignored(2));
        assert!(!format.ignored(0));
    }

    #[test]
    #[should_panic(expected = "format t: no terminal named \"sep\"")]
    fn a_transition_on_an_ignored_terminal_is_a_programming_error() {
        let _ = two_states()
            .ignores("sep", "|")
            .transition("content", "sep", "inside");
    }

    #[test]
    #[should_panic(expected = "format t: no terminal named \"missing\"")]
    fn a_transition_on_a_terminal_the_table_lacks_is_a_programming_error() {
        let _ = two_states().transition("content", "missing", "inside");
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"nowhere\"")]
    fn a_transition_to_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().transition("content", "open", "nowhere");
    }
}
