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
//!   assembler for it. A format with no arguments state has none.
//!
//! The first state in the table is where an output starts, unless the prompt moved the engine
//! before the output began: the engine replays the prompt's terminals over the same table, from
//! the last **turn opener** the table names (`<|im_start|>assistant` for Qwen's ChatML), so that
//! a marker quoted in an earlier turn moves nothing. A table that names no opener is replayed
//! from the prompt's start.
//!
//! The table holds what is static about a format. What belongs to one request, the tools that
//! type a tagged call's values, reaches the [`Engine`] with the request, not the table.
//!
//! [`Engine`]: crate::Engine
//! [`formats`]: crate::formats

use crate::tagged::keyed;

/// How the model writes a call between the call markers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallSyntax {
    /// One JSON object, `{"name": …, "arguments": {…}}`: Qwen3.
    Json,
    /// `<function=NAME>` and then `<parameter=KEY>` around each value's text, typed by the
    /// request's tools, which reach the engine with the request: Qwen 3.5 and later, Qwen3-Coder.
    Tagged,
    /// DeepSeek's DSML: the arguments state is one `<｜DSML｜ invoke name="…">` block, whose
    /// parameter tags carry a `string` attribute that types each value. The terminal that enters
    /// the state is the invoke tag's opening, and the one that leaves it is the invoke's closing
    /// tag, which the call's end carries.
    Dsml,
    /// The call's name as text, then `<arg_key>` and `<arg_value>` pairs, in the family's spelling
    /// of the four tags, typed by the request's tools: GLM, Hy4, Ling, IQuest. The terminals that
    /// enter and leave the state are the call's own markers, and the call's end carries the
    /// closing one.
    Keyed(keyed::Tags),
    /// Python calls, `name(key=value, ...)`, one or several, bare or in a list: Olmo 3, LFM2.5,
    /// Llama 3.2's pythonic template. The region's markers are the terminals; each call ends at
    /// its own `)`, and the region's closing marker is wrapping.
    Pythonic,
    /// A JSON list of calls with no markers around it, or content: xLAM. The table's one state is
    /// an arguments state entered at the output's start, and the assembler decides at the first
    /// byte that is not whitespace.
    JsonList,
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
#[derive(Clone, Debug)]
pub struct Format {
    name: String,
    terminals: Vec<Terminal>,
    states: Vec<State>,
    transitions: Vec<Transition>,
    calls: Option<CallSyntax>,
    turn_opener: Option<String>,
}

#[derive(Clone, Debug)]
struct Terminal {
    name: String,
    text: String,
}

#[derive(Clone, Debug)]
struct State {
    name: String,
    emits: Emits,
}

#[derive(Clone, Copy, Debug)]
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
        }
    }

    /// A terminal and its text spelling.
    #[must_use]
    pub fn terminal(mut self, name: &str, text: &str) -> Self {
        self.terminals.push(Terminal {
            name: name.to_string(),
            text: text.to_string(),
        });
        self
    }

    /// A state and what its text is; the first one added is where an output starts.
    #[must_use]
    pub fn state(mut self, name: &str, emits: Emits) -> Self {
        self.states.push(State {
            name: name.to_string(),
            emits,
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

    /// The text that opens the turn the model writes, `<|im_start|>assistant` for ChatML: the
    /// prompt's replay starts at its last occurrence.
    #[must_use]
    pub fn opens_turn(mut self, text: &str) -> Self {
        self.turn_opener = Some(text.to_string());
        self
    }

    /// The turn opener, if the table names one.
    pub(crate) fn turn_opener(&self) -> Option<&str> {
        self.turn_opener.as_deref()
    }

    /// The format's name.
    pub fn name(&self) -> &str {
        &self.name
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
            .position(|terminal| terminal.name == name)
            .unwrap_or_else(|| unknown("terminal", name, &self.name))
    }

    /// The terminals' texts, in the order the engine's scanner numbers them.
    pub(crate) fn terminal_texts(&self) -> impl Iterator<Item = &str> {
        self.terminals.iter().map(|terminal| terminal.text.as_str())
    }

    /// The text of terminal `index`.
    pub(crate) fn terminal_text(&self, index: usize) -> &str {
        &self.terminals[index].text
    }

    /// What the text in state `index` is.
    pub(crate) fn emits(&self, index: usize) -> Emits {
        self.states[index].emits
    }

    /// Where terminal `on` takes the engine from state `from`, if the table says.
    pub(crate) fn next(&self, from: usize, on: usize) -> Option<usize> {
        self.transitions
            .iter()
            .find(|transition| transition.from == from && transition.on == on)
            .map(|transition| transition.to)
    }

    /// The call syntax, for a format with an arguments state.
    pub(crate) fn call_syntax(&self) -> Option<&CallSyntax> {
        self.calls.as_ref()
    }

    /// What every table must hold before an engine runs it: a state to start in, which does not
    /// emit reasoning (a thought the output starts inside is entered by the prompt's replay, which
    /// pushes `ReasoningStart`; an arguments state at the start has its call opened by the
    /// engine), a call syntax when a state emits arguments, a text for every terminal and a turn
    /// opener that is not empty. The builder keeps a row from naming what the table lacks; this
    /// checks the rest.
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
        Ok(())
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
