//! A compiled table's layout, learned from the model's chat template.
//!
//! A response template reads more than one spelling of a call: `\s*` takes any whitespace, a skip
//! such as `[^>]*?` any attributes, and the text outside every field is dropped whatever it is. A
//! structural tag writes one spelling, and it should be the one the model's chat template renders,
//! or the constraint pushes the model off the text it was trained on. symphony's [`Layout`] is that
//! spelling. [`learn`] reads it off the chat template's renderings of a probe conversation: the
//! probe's [`tools`], and the conversations of each [`Case`].
//!
//! - **Read back.** Each turn's rendering, after the prompt's rendering and up to the first text
//!   of a token the model stops on, is read through the table as the model's output would be read:
//!   the prompt, the turn, the stop. The turn with the calls must read back as exactly the probe's
//!   calls (their names, and their arguments equal once the table has typed them), with no content
//!   and no thought; the turn with the answer as the probe's answer alone; a turn with the probe's
//!   thought as that thought and the rest. A chat template that writes one call in a turn (the turn
//!   with the calls reads back as its first call alone, or has no rendering) is read with the turn
//!   with one call, and its layout holds one call in a tag. One that renders no turn without a
//!   thought (it raises) is read from the turns with the probe's thought, in the first field whose
//!   turns read back as that thought.
//! - **Parts.** Each part is the text the rendering wrote there, cut where the table's events and
//!   the shape's patterns cut it: the text before the call's terminal (the lead, with the tool's
//!   name in its place), the header (with the name where the header captures it), the text before
//!   the first parameter, a parameter (with the key and the value where its tag captures them), the
//!   text between two parameters and after the last, the trailer and the close; then the text
//!   between two calls and after the last, how `true`, `null`, a list and an object are spelled,
//!   and the text around the thought and the answer, with the terminal that closed each (a
//!   template may close a region more than one way, and the tags write the close it renders).
//!   Where a thought comes before the first call (the prompt opened it, or the turn writes one),
//!   the text between its close and the call is the thought's, and the lead is empty.
//! - **Checks.** Every call and every parameter is written alike but for the name, the key and the
//!   value, the parameters in the order of the tool's schema; numbers are JSON, and `true` and
//!   `null` are spelled as JSON or as Python spells them; the parts write the turn with the calls
//!   again byte for byte, and the turns with one call and with the thought write the same parts,
//!   the answer too; the answer's text stands right after the terminal that opens its region and
//!   right before the one that closes it; the text after a thought, and before the terminal that
//!   opens an answer's region, holds no terminal of the table, since the structural tags write the
//!   rows into a region themselves; no text the layout writes for every tool names one; and the
//!   layout fits the table ([`Layout::check`]).
//!
//! The renderings also say whether the template thinks first ([`Learned::thinks_first`]): it
//! writes a thought before every call and answer, its own opener included, in turns whose message
//! has none; or it renders no turn without one; or it writes one before the calls and renders no
//! answer without one. The structural tags then write the thought first where the prompt has not
//! opened it. A template that writes its own thought before the calls but not before the answer,
//! or the other way round, gives no layout.
//!
//! Anything else gives no layout, with a stable code ([`NoLayout::code`]); a table without a
//! layout has no structural tag for its calls.

#[cfg(test)]
mod tests;

use std::{fmt, ops::Range};

use openai_protocol::common::Tool;
use serde_json::{json, Map, Value};
use symphony::{
    adapt,
    shape::{
        Arguments, CallLayout, CallShape, Class, End, Layout, LayoutError, Match, Matcher, Message,
        NameSource, Nested, Pattern, Segment, Slot, Spellings, Suffix, Verdict, Words, Written,
    },
    CallSyntax, Declared, DropReason, Engine, EngineFinish, Event, Events, Format, Input, Parser,
};

/// The user's message every conversation of the probe starts with.
const ASK: &str = "Call the probe tools.";
/// The thought a probe turn writes where it writes one.
const THOUGHT: &str = "Probe thought.";
/// The probe's answer, a JSON object.
const ANSWER: &str = "{\"probe\": 1}";
/// The keys of `fn_alpha`'s parameters, in the order of its schema.
const ALPHA_KEYS: [&str; 7] = ["s", "i", "n", "b", "a", "o", "z"];

/// The probe's tools, as a request's `tools`: `fn_alpha`, with a required parameter of each JSON
/// type (`s` a string, `i` an integer, `n` a number, `b` a boolean, `a` a list of integers, `o` an
/// object, `z` a nullable string), and `fn_beta`, with one string `q`.
pub fn tools() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "fn_alpha",
                "description": "A probe tool with one parameter of each JSON type.",
                "parameters": {
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
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fn_beta",
                "description": "A probe tool with one string parameter.",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        },
    ])
}

/// The probe's calls: `fn_alpha` with a value of each type, then `fn_beta`, each with its id and
/// its arguments as an object.
fn probe_calls() -> [Value; 2] {
    [
        json!({
            "id": "call_alpha",
            "type": "function",
            "function": {
                "name": "fn_alpha",
                "arguments": {
                    "s": "sigma", "i": 42, "n": 2.5, "b": true, "a": [1, 2], "o": {"m": "mu"},
                    "z": null,
                },
            },
        }),
        json!({
            "id": "call_beta",
            "type": "function",
            "function": {"name": "fn_beta", "arguments": {"q": "kappa"}},
        }),
    ]
}

/// The message field a probe turn writes its thought in. Chat templates read an assistant's
/// reasoning from one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ThoughtField {
    /// `reasoning_content`.
    ReasoningContent,
    /// `reasoning`.
    Reasoning,
    /// `thinking`.
    Thinking,
}

impl ThoughtField {
    /// Every field, in the order [`learn`] tries them.
    pub const ALL: [Self; 3] = [Self::ReasoningContent, Self::Reasoning, Self::Thinking];

    /// The field's key in a message.
    pub fn key(self) -> &'static str {
        match self {
            Self::ReasoningContent => "reasoning_content",
            Self::Reasoning => "reasoning",
            Self::Thinking => "thinking",
        }
    }
}

/// One conversation of the probe, which the chat template renders with the probe's [`tools`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Case {
    /// The user's message, with the generation prompt: the prompt every turn below follows.
    Prompt,
    /// The user's message and an assistant turn with the probe's two calls.
    Calls,
    /// The same turn with its first call alone.
    OneCall,
    /// The user's message and an assistant turn with the probe's JSON answer.
    Answer,
    /// [`Case::Calls`] with the probe's thought in a field.
    ThoughtCalls(ThoughtField),
    /// [`Case::OneCall`] with the probe's thought in a field.
    ThoughtOneCall(ThoughtField),
    /// [`Case::Answer`] with the probe's thought in a field.
    ThoughtAnswer(ThoughtField),
}

impl Case {
    /// Every case, the prompt first.
    pub const ALL: [Self; 13] = [
        Self::Prompt,
        Self::Calls,
        Self::OneCall,
        Self::Answer,
        Self::ThoughtCalls(ThoughtField::ReasoningContent),
        Self::ThoughtCalls(ThoughtField::Reasoning),
        Self::ThoughtCalls(ThoughtField::Thinking),
        Self::ThoughtOneCall(ThoughtField::ReasoningContent),
        Self::ThoughtOneCall(ThoughtField::Reasoning),
        Self::ThoughtOneCall(ThoughtField::Thinking),
        Self::ThoughtAnswer(ThoughtField::ReasoningContent),
        Self::ThoughtAnswer(ThoughtField::Reasoning),
        Self::ThoughtAnswer(ThoughtField::Thinking),
    ];

    /// The case's name: `prompt`, `calls`, `one_call`, `answer`, and each of the last three with
    /// `+` and the thought's field (`calls+reasoning_content`).
    pub fn name(self) -> String {
        match self {
            Self::Prompt => "prompt".to_owned(),
            Self::Calls => "calls".to_owned(),
            Self::OneCall => "one_call".to_owned(),
            Self::Answer => "answer".to_owned(),
            Self::ThoughtCalls(field) => format!("calls+{}", field.key()),
            Self::ThoughtOneCall(field) => format!("one_call+{}", field.key()),
            Self::ThoughtAnswer(field) => format!("answer+{}", field.key()),
        }
    }

    /// The conversation, as a request's `messages`. Each call's arguments are an object; where the
    /// chat template raises on that, a renderer passes them as compact JSON text instead, as
    /// `scripts/record_chat_template_renders.py` does.
    pub fn messages(self) -> Value {
        let user = json!({"role": "user", "content": ASK});
        let [alpha, beta] = probe_calls();
        let (thought, key, body) = match self {
            Self::Prompt => return json!([user]),
            Self::Calls => (None, "tool_calls", json!([alpha, beta])),
            Self::OneCall => (None, "tool_calls", json!([alpha])),
            Self::Answer => (None, "content", json!(ANSWER)),
            Self::ThoughtCalls(field) => (Some(field), "tool_calls", json!([alpha, beta])),
            Self::ThoughtOneCall(field) => (Some(field), "tool_calls", json!([alpha])),
            Self::ThoughtAnswer(field) => (Some(field), "content", json!(ANSWER)),
        };
        let mut turn = Map::new();
        turn.insert("role".to_owned(), json!("assistant"));
        if let Some(field) = thought {
            turn.insert(field.key().to_owned(), json!(THOUGHT));
        }
        turn.insert(key.to_owned(), body);
        json!([user, turn])
    }

    /// Whether the rendering adds the generation prompt: the prompt's does, a turn's does not.
    pub fn generation_prompt(self) -> bool {
        self == Self::Prompt
    }

    /// The same turn with the probe's thought in `field`.
    fn thinking(self, field: ThoughtField) -> Self {
        match self {
            Self::Calls => Self::ThoughtCalls(field),
            Self::OneCall => Self::ThoughtOneCall(field),
            Self::Answer => Self::ThoughtAnswer(field),
            other => other,
        }
    }
}

impl fmt::Display for Case {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.name())
    }
}

/// Why a table gets no layout from its chat template's renderings, each with a stable code
/// ([`NoLayout::code`]).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum NoLayout {
    /// The table's calls are not tagged parameters given as data, the one way of writing calls a
    /// layout has places for.
    #[error("the table's calls are not tagged parameters given as data")]
    Calls,
    /// The table does not pass its own check, so no engine reads it.
    #[error("the table is not valid: {0}")]
    Table(String),
    /// A case the layout needs has no rendering: the chat template raised, or it was not rendered.
    #[error("the chat template has no rendering of {0}")]
    Unrendered(Case),
    /// A turn's rendering does not begin with the prompt's, so what the model writes after the
    /// prompt is not known.
    #[error("the rendering of {0} does not begin with the prompt's")]
    Prompt(Case),
    /// A turn's rendering does not read back through the table as the probe's message: the chat
    /// template writes no calls, or writes them otherwise than the response template reads them.
    #[error("the rendering of {case} reads back as {read}, not as the probe's message")]
    ReadBack { case: Case, read: String },
    /// A part written differently from one call or parameter to the next, or by the turn with one
    /// call or with the thought than by the turn with the calls.
    #[error("the {0} is not written alike in every call and turn")]
    Uneven(&'static str),
    /// A text the layout writes for every tool that names one: the probe's tools are named where
    /// the layout has no place for a name (the text after a thought, where the prompt opened it,
    /// holds the call's lead too).
    #[error("the {0} names a tool where the layout has no place for its name")]
    Named(&'static str),
    /// Parameters written in another order than the tool's schema's.
    #[error("the parameters are not written in the order of the tool's schema")]
    Order,
    /// A value spelled in a way a layout has no words for: a number that is not JSON, or `true`
    /// and `null` spelled otherwise than as JSON or as Python spells them.
    #[error("the {0} is spelled in a way a layout does not write")]
    Spelling(&'static str),
    /// The parts taken from a turn do not write it again as the structural tags write them, or its
    /// events cannot be cut into them: text stands between the answer's text and the terminals of
    /// its region, or the text after a thought, or before the terminal that opens an answer's
    /// region, holds a terminal of the table, which the tags write themselves on the way in.
    #[error("the parts taken from the rendering of {0} do not write it again")]
    Unexplained(Case),
    /// The layout taken from the renderings does not fit the table.
    #[error("the layout does not fit the table: {0}")]
    Unfit(LayoutError),
}

impl NoLayout {
    /// The stable code: `Calls`, `Table`, `Unrendered`, `Prompt`, `ReadBack`, `Uneven`, `Named`,
    /// `Order`, `Spelling`, `Unexplained` or `Unfit`.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Calls => "Calls",
            Self::Table(_) => "Table",
            Self::Unrendered(_) => "Unrendered",
            Self::Prompt(_) => "Prompt",
            Self::ReadBack { .. } => "ReadBack",
            Self::Uneven(_) => "Uneven",
            Self::Named(_) => "Named",
            Self::Order => "Order",
            Self::Spelling(_) => "Spelling",
            Self::Unexplained(_) => "Unexplained",
            Self::Unfit(_) => "Unfit",
        }
    }
}

/// What a table learns from its chat template's renderings: the layout of its turns, and whether
/// the template thinks first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Learned {
    /// How the chat template writes a turn where the response template's patterns leave the text
    /// open.
    pub layout: Layout,
    /// Whether the template writes a thought before every call and answer: the turns with the
    /// calls and with the answer write one, its opener included, where the message has none; the
    /// template renders no turn without one; or the turn with the calls writes one and the
    /// template renders no answer without one.
    pub thinks_first: bool,
}

impl Learned {
    /// `format` with what it learned: the layout ([`Format::with_layout`]), and the mark of a
    /// template that thinks first ([`Format::thinks_first`]) where it does.
    pub fn apply(self, format: Format) -> Format {
        let format = format.with_layout(self.layout);
        if self.thinks_first {
            format.thinks_first()
        } else {
            format
        }
    }
}

/// The layout of `format`'s turns, learned from the chat template's renderings of the probe.
/// `render` gives the rendering of each [`Case`], with the probe's [`tools`] and with the
/// generation prompt where the case asks for it, or `None` for one that raised or was not
/// rendered. `stops` are the texts of the tokens the model stops on, as
/// [`compile`](super::compile) takes them: a turn's rendering ends at the first one. The module
/// documentation says how the layout is read and checked; anything else gives no layout.
pub fn learn(
    format: &Format,
    stops: &[&str],
    render: impl Fn(Case) -> Option<String>,
) -> Result<Learned, NoLayout> {
    let shape = match format.call_syntax() {
        Some(CallSyntax::Shaped(shape)) => *shape,
        _ => return Err(NoLayout::Calls),
    };
    let Arguments::Params { tag, ends, .. } = &shape.arguments else {
        return Err(NoLayout::Calls);
    };
    format.validate().map_err(NoLayout::Table)?;
    let tools: Vec<Tool> = serde_json::from_value(tools()).unwrap_or_default();
    let prompt = render(Case::Prompt).ok_or(NoLayout::Unrendered(Case::Prompt))?;
    Learner {
        format,
        shape,
        tag,
        ends,
        stops,
        declared: Declared::of(&tools),
        prompt,
        render: &render,
    }
    .learn()
}

/// A layout in one line, for logs and pinned lists: each part as written, `{name}`, `{key}` and
/// `{value}` standing for its places.
pub fn summary(layout: &Layout) -> String {
    let call = &layout.call;
    let message = |message: Option<&Message>| match message {
        None => "none".to_owned(),
        Some(message) => format!(
            "before {:?}{} after {:?}{}",
            message.before,
            message
                .close
                .as_ref()
                .map_or_else(String::new, |close| format!(" close {close:?}")),
            message.after,
            if message.stops { " stops" } else { "" }
        ),
    };
    format!(
        "lead {:?} header {:?} before {:?} parameter {:?} between {:?} after {:?} trailer {:?} \
         close {:?}; between calls {:?}; after calls {:?}; values {:?} {:?}; thought {}; answer {}",
        spelled(&call.lead),
        spelled(&call.header),
        call.before,
        spelled(&call.parameter),
        call.between,
        call.after,
        call.trailer,
        call.close,
        layout.between_calls,
        layout.after_calls,
        layout.values.words,
        layout.values.nested,
        message(layout.thought.as_ref()),
        message(layout.answer.as_ref()),
    )
}

/// A written text with its places spelled `{name}`, `{key}` and `{value}`.
fn spelled(written: &Written) -> String {
    written
        .0
        .iter()
        .map(|segment| match segment {
            Segment::Lit(text) => text.to_string(),
            Segment::Slot(Slot::Name) => "{name}".to_owned(),
            Segment::Slot(Slot::Key) => "{key}".to_owned(),
            Segment::Slot(Slot::Value) => "{value}".to_owned(),
        })
        .collect()
}

/// What a layout is learned with: the table, its call's shape, the stops, the probe's tools, the
/// prompt's rendering and the renderings.
struct Learner<'a, R> {
    format: &'a Format,
    shape: &'static CallShape,
    tag: &'static Pattern,
    ends: &'static [End],
    stops: &'a [&'a str],
    declared: Declared,
    prompt: String,
    render: &'a R,
}

/// A turn's rendering read back through the table: the output (what follows the prompt's
/// rendering, up to the first stop text) and the table's events for it, each with the bytes of
/// the output it accounts for.
struct Reading {
    case: Case,
    output: String,
    events: Vec<(Range<usize>, Event)>,
}

/// What a reading says, as the Chat API's message holds it: the thought, the content, and each
/// call's name and arguments.
#[derive(Debug, PartialEq)]
struct Said {
    reasoning: Option<String>,
    content: Option<String>,
    calls: Vec<(String, Value)>,
}

impl Said {
    /// The probe's message: its first `calls` calls, or its answer for none, with or without its
    /// thought.
    fn probe(calls: usize, thought: bool) -> Self {
        let called = probe_calls()
            .into_iter()
            .take(calls)
            .map(|call| {
                let function = &call["function"];
                let name = function["name"].as_str().unwrap_or_default().to_owned();
                (name, function["arguments"].clone())
            })
            .collect();
        Self {
            reasoning: thought.then(|| THOUGHT.to_owned()),
            content: (calls == 0).then(|| ANSWER.to_owned()),
            calls: called,
        }
    }

    fn describe(&self) -> String {
        let calls: Vec<Value> = self
            .calls
            .iter()
            .map(|(name, arguments)| json!({"name": name, "arguments": arguments}))
            .collect();
        json!({"reasoning": self.reasoning, "content": self.content, "calls": calls}).to_string()
    }
}

/// One call as a turn wrote it: where each of its parts stands in the output.
struct Call {
    name: String,
    /// The terminal that opened the call's region.
    terminal: Range<usize>,
    /// The header after the terminal, and where it captured the name.
    header: Range<usize>,
    name_at: Option<Range<usize>>,
    params: Vec<Param>,
    /// Where the trailer starts, in the text from the last parameter to the close.
    trailer_at: usize,
    /// What the call's end accounts for: the terminal that closed the region, or nothing where
    /// the stream's stop closed it.
    close: Range<usize>,
}

/// One parameter as a turn wrote it: its tag through the end its value took, and where its key and
/// its value stand.
struct Param {
    span: Range<usize>,
    key: Range<usize>,
    value: Range<usize>,
}

/// A turn's thought: the terminal that opened it, where the turn wrote one (none where the prompt
/// opened it), and the terminal that closed it.
struct Thought {
    open: Option<Range<usize>>,
    close: Range<usize>,
}

/// A turn's answer: the terminal that opened its region, where it has one (none where the answer
/// is the text the turn starts in), its text, and the terminal that closed it (none where the
/// stop ended it).
struct Answer {
    open: Option<Range<usize>>,
    text: Range<usize>,
    close: Option<Range<usize>>,
}

impl Answer {
    /// Where the answer's region starts: its opener, or its text.
    fn start(&self) -> usize {
        self.open
            .as_ref()
            .map_or(self.text.start, |open| open.start)
    }
}

/// The answer as a turn wrote it: the text around it, the turn's text from where the answer's
/// region starts, whether the turn was the one without the probe's thought, and whether it wrote a
/// thought of its own, its opener included, before the answer.
struct Answered {
    message: Message,
    region: String,
    plain: bool,
    opened: bool,
}

/// The calls a turn wrote, and the thought before the first.
struct Calls {
    reading: Reading,
    calls: Vec<Call>,
    thought: Option<Thought>,
}

/// The turn the layout of the calls is learned from: its calls, whether it holds both, and the
/// field of the probe's thought it writes, if any.
struct Turn {
    calls: Calls,
    many: bool,
    field: Option<ThoughtField>,
}

impl Turn {
    fn new(calls: Calls, many: bool, field: Option<ThoughtField>) -> Self {
        Self { calls, many, field }
    }
}

impl<R: Fn(Case) -> Option<String>> Learner<'_, R> {
    fn learn(&self) -> Result<Learned, NoLayout> {
        let turn = self.calls()?;
        let mut layout = call_layout(&turn.calls)?;
        unnamed(&layout)?;
        if turn.many {
            self.check_one_call(&turn, &layout)?;
        }
        let (thought, field) = match turn.field {
            // The turn with the calls writes the probe's thought: the chat template writes no
            // turn without one.
            Some(field) => (held(&turn.calls), Some(field)),
            None => self.thought(&turn.calls, &layout)?,
        };
        // The tags write the rows from the thought's close into a call or an answer themselves, so
        // the text the layout writes after the thought holds none of the table's markers.
        if thought
            .as_ref()
            .is_some_and(|thought| self.marked(&thought.after))
        {
            let case = turn.calls.reading.case;
            return Err(NoLayout::Unexplained(match (turn.field, field) {
                (None, Some(field)) => case.thinking(field),
                _ => case,
            }));
        }
        layout.thought = thought;
        unnamed(&layout)?;
        let answer = self.answer(&layout, field)?;
        layout.answer = Some(answer.message);
        unnamed(&layout)?;
        if let (Some(field), true) = (field, answer.plain) {
            self.check_thought_answer(field, &layout, &answer.region)?;
        }
        layout.check(self.format).map_err(NoLayout::Unfit)?;
        // The template thinks first where it renders no turn without a thought, or where its turns
        // without one write a thought of their own before the calls and before the answer, or
        // before the calls where it renders no answer without the probe's thought. A thought of
        // its own before the calls alone, or before the answer alone, no tag writes as it does.
        let opened = |thought: Option<&Thought>| thought.is_some_and(|t| t.open.is_some());
        let thinks_first = match turn.field {
            Some(_) => true,
            None => {
                let calls = opened(turn.calls.thought.as_ref());
                if answer.plain && answer.opened != calls {
                    return Err(NoLayout::Uneven("thought"));
                }
                calls
            }
        };
        Ok(Learned {
            layout,
            thinks_first,
        })
    }

    /// The turn with the calls: both, or the first alone where the chat template writes one call
    /// in a turn (the turn with the calls reads back as the first call alone, or has no
    /// rendering). Where the chat template writes no turn with calls and without a thought (it
    /// requires one, and raises), the turn with the probe's thought in the first field whose
    /// turns read back as the thought and the calls.
    fn calls(&self) -> Result<Turn, NoLayout> {
        if let Some(turn) = self.calls_in(None)? {
            return Ok(turn);
        }
        for field in ThoughtField::ALL {
            if let Some(turn) = self.calls_in(Some(field))? {
                return Ok(turn);
            }
        }
        Err(NoLayout::Unrendered(Case::Calls))
    }

    /// The turn with both calls, or with the first alone, with the probe's thought in `field` or
    /// without one. `None` where neither turn is rendered, or, with a thought, where neither reads
    /// back as the thought and the calls (the chat template does not read that field, or its
    /// rendering of it does not follow the prompt).
    fn calls_in(&self, field: Option<ThoughtField>) -> Result<Option<Turn>, NoLayout> {
        let thought = field.is_some();
        let case = |case: Case| field.map_or(case, |field| case.thinking(field));
        // With a thought, a field the chat template does not read is no error: the next is tried.
        let skip = |error: &NoLayout| match error {
            NoLayout::Unrendered(_) => true,
            NoLayout::Prompt(_) | NoLayout::ReadBack { .. } => thought,
            _ => false,
        };
        let mut first_only = false;
        match self.read(case(Case::Calls)) {
            Ok(reading) => {
                let read = said(&reading);
                if read == Said::probe(2, thought) {
                    let calls = self.cut(reading)?;
                    return Ok(Some(Turn::new(calls, true, field)));
                }
                if read != Said::probe(1, thought) {
                    if thought {
                        return Ok(None);
                    }
                    return Err(NoLayout::ReadBack {
                        case: Case::Calls,
                        read: read.describe(),
                    });
                }
                first_only = true;
            }
            Err(error) if skip(&error) => {}
            Err(error) => return Err(error),
        }
        match self.read(case(Case::OneCall)) {
            Ok(reading) => {
                let read = said(&reading);
                if read == Said::probe(1, thought) {
                    let calls = self.cut(reading)?;
                    return Ok(Some(Turn::new(calls, false, field)));
                }
                if thought {
                    return Ok(None);
                }
                Err(NoLayout::ReadBack {
                    case: Case::OneCall,
                    read: read.describe(),
                })
            }
            // A turn with the calls that read back as the first call alone needs the turn with
            // one call.
            Err(error) if skip(&error) && (thought || !first_only) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// The rendering of `case`, read back through the table.
    fn read(&self, case: Case) -> Result<Reading, NoLayout> {
        let rendering = (self.render)(case).ok_or(NoLayout::Unrendered(case))?;
        let turn = rendering
            .strip_prefix(self.prompt.as_str())
            .ok_or(NoLayout::Prompt(case))?;
        let end = self
            .stops
            .iter()
            .filter(|stop| !stop.is_empty())
            .filter_map(|stop| turn.find(stop))
            .min()
            .unwrap_or(turn.len());
        let output = turn[..end].to_owned();
        let mut engine = Engine::new(self.format.clone(), self.declared.clone());
        let mut out = Events::new();
        let prompt = Input::Prompt {
            token_ids: &[],
            text: &self.prompt,
        };
        let delta = Input::Delta {
            token_ids: &[],
            text: &output,
            spans: &[],
        };
        let end = Input::End {
            finish: EngineFinish::Stop,
        };
        let fed = engine
            .feed(prompt, &mut out)
            .and_then(|()| engine.feed(delta, &mut out))
            .and_then(|()| engine.feed(end, &mut out));
        if let Err(error) = fed {
            return Err(NoLayout::ReadBack {
                case,
                read: error.to_string(),
            });
        }
        // Every byte of the output is in one event, in order (conservation), so each event's
        // bytes stand where the bytes before them end.
        if out.text() != output {
            return Err(NoLayout::Unexplained(case));
        }
        let mut events = Vec::new();
        let mut at = 0;
        for event in out.drain() {
            let length = bytes_of(&event).len();
            events.push((at..at + length, event));
            at += length;
        }
        Ok(Reading {
            case,
            output,
            events,
        })
    }

    /// Where each call's parts stand in `reading`, and the thought before the first call.
    fn cut(&self, reading: Reading) -> Result<Calls, NoLayout> {
        let calls = self
            .cut_calls(&reading)
            .ok_or(NoLayout::Unexplained(reading.case))?;
        let thought = thought_of(&reading, calls[0].terminal.start);
        Ok(Calls {
            reading,
            calls,
            thought,
        })
    }

    /// Where each call's parts stand: its terminal is the bytes dropped just before its start, its
    /// header its start's bytes, its parameters its arguments' bytes, with their key and value where
    /// the shape's tag captures them, and its close its end's bytes; the trailer is the earliest
    /// text before the close that the shape's trailer reads. `None` for events that do not cut so.
    fn cut_calls(&self, reading: &Reading) -> Option<Vec<Call>> {
        let output = reading.output.as_str();
        let mut calls: Vec<Call> = Vec::new();
        for (at, (range, event)) in reading.events.iter().enumerate() {
            match event {
                Event::ToolCallStart { name, .. } => {
                    let (terminal, before) = &reading.events[at.checked_sub(1)?];
                    if !matches!(before, Event::Dropped { .. })
                        || terminal.is_empty()
                        || terminal.end != range.start
                    {
                        return None;
                    }
                    calls.push(Call {
                        name: name.clone(),
                        terminal: terminal.clone(),
                        header: range.clone(),
                        name_at: self.header_name(output, range.clone())?,
                        params: Vec::new(),
                        trailer_at: range.end,
                        close: range.end..range.end,
                    });
                }
                Event::ToolCallArguments { .. } if !range.is_empty() => {
                    let param = self.param(output, range.clone())?;
                    calls.last_mut()?.params.push(param);
                }
                Event::ToolCallEnd { .. } => {
                    let call = calls.last_mut()?;
                    let last = call.params.last().map_or(call.header.end, |p| p.span.end);
                    call.trailer_at = self.trailer_start(output, last, range.start)?;
                    call.close = range.clone();
                }
                _ => {}
            }
        }
        if calls.is_empty() || calls.iter().any(|call| call.params.is_empty()) {
            return None;
        }
        Some(calls)
    }

    /// Where the header at `header` captured the tool's name, as the shape's header reads it on
    /// the output from there; `Some(None)` for a name the template writes itself.
    fn header_name(&self, output: &str, header: Range<usize>) -> Option<Option<Range<usize>>> {
        if !matches!(self.shape.name, NameSource::Header) {
            return Some(None);
        }
        let before = output[..header.start].chars().next_back();
        let found = matched(
            &self.shape.header,
            &[],
            &self.shape.word,
            before,
            &output[header.start..],
        )?;
        if found.end() != header.len() {
            return None;
        }
        let span = found.span(Slot::Name)?;
        Some(Some(header.start + span.start..header.start + span.end))
    }

    /// The parameter whose tag starts at `span`: where its key and its value stand, as the shape's
    /// tag and ends read them on the output from there.
    fn param(&self, output: &str, span: Range<usize>) -> Option<Param> {
        let before = output[..span.start].chars().next_back();
        let found = matched(
            self.tag,
            self.ends,
            &self.shape.word,
            before,
            &output[span.start..],
        )?;
        if found.end() != span.len() {
            return None;
        }
        let at = |slot: Slot| {
            let range = found.span(slot)?;
            Some(span.start + range.start..span.start + range.end)
        };
        Some(Param {
            key: at(Slot::Key)?,
            value: at(Slot::Value)?,
            span,
        })
    }

    /// Where the trailer starts in the text from `from` to `close` (the closing terminal, or the
    /// end of the output where the stop closed the call): the earliest place the shape's trailer
    /// reads from up to there, or `close` for a shape without one.
    fn trailer_start(&self, output: &str, from: usize, close: usize) -> Option<usize> {
        let trailer = &self.shape.trailer;
        if trailer.0.is_empty() {
            return Some(close);
        }
        let before = output[..from].chars().next_back();
        let mut search = Suffix::new(trailer, &self.shape.word, before).hold_cap(None);
        search.feed(&output[from..close]);
        let start = search.start(output[close..].chars().next())?;
        Some(from + start)
    }

    /// Where the turn with the calls wrote both: the turn with one call, where the chat template
    /// renders it, writes the first call as that turn does, then what the layout writes after the
    /// last call.
    fn check_one_call(&self, turn: &Turn, layout: &Layout) -> Result<(), NoLayout> {
        let case = turn
            .field
            .map_or(Case::OneCall, |field| Case::OneCall.thinking(field));
        let one = match self.read(case) {
            Ok(one) => one,
            Err(NoLayout::Unrendered(_)) => return Ok(()),
            Err(error) => return Err(error),
        };
        expect(&one, &Said::probe(1, turn.field.is_some()))?;
        let calls = &turn.calls;
        let first = &calls.calls[0];
        let end = match layout.call.close {
            Some(_) => first.close.end,
            None => first.close.start,
        };
        let written = format!("{}{}", &calls.reading.output[..end], layout.after_calls);
        if one.output != written {
            return Err(NoLayout::Uneven("turn with one call"));
        }
        Ok(())
    }

    /// The thought as the turns write it, and the field the chat template reads it from. It comes
    /// from the turn with the calls and the probe's thought in the first field whose rendering
    /// reads back as that thought and the calls: the text before the thought's opener, where the
    /// turn writes one, and the text between its close and the first call's lead. That turn writes
    /// its calls as `layout` does. Where the turn with the calls holds a thought too (the prompt
    /// opened it), what follows its close is the same. Where no turn writes the probe's thought,
    /// the thought the turn with the calls holds, if any.
    fn thought(
        &self,
        calls: &Calls,
        layout: &Layout,
    ) -> Result<(Option<Message>, Option<ThoughtField>), NoLayout> {
        let lead = &layout.call.lead;
        let one = layout.between_calls.is_none();
        let (case, count) = if one {
            (Case::OneCall, 1)
        } else {
            (Case::Calls, 2)
        };
        let held = held(calls);
        for field in ThoughtField::ALL {
            let Ok(reading) = self.read(case.thinking(field)) else {
                continue;
            };
            if said(&reading) != Said::probe(count, true) {
                continue;
            }
            let turn = self.cut(reading)?;
            let first = &turn.calls[0];
            let Some(thought) = &turn.thought else {
                continue;
            };
            let output = turn.reading.output.as_str();
            let mut message = around(output, thought, first.terminal.start);
            let first_lead = fill(lead, &first.name);
            let after = message
                .after
                .strip_suffix(first_lead.as_str())
                .ok_or(NoLayout::Uneven("lead"))?;
            message.after = after.into();
            if held
                .as_ref()
                .is_some_and(|held| held.after != message.after || held.close != message.close)
            {
                return Err(NoLayout::Uneven("thought"));
            }
            // After the thought, the turn writes its calls as the turn without it does.
            let from = first.terminal.start - first_lead.len();
            if write_calls(output, layout, &turn.calls) != output[from..] {
                return Err(NoLayout::Uneven("turn with the thought"));
            }
            return Ok((Some(message), Some(field)));
        }
        Ok((held, None))
    }

    /// The answer as the turn with the probe's answer writes it, or, where the chat template
    /// renders no such turn without a thought, the turn with the probe's thought in `field`: the
    /// text before the terminal that opens its region (before its text, where the answer is the
    /// text the turn starts in), and the text after its close, or after its text where the stop
    /// ends it. A thought before it writes the thought's text after its close first. The answer's
    /// text stands right after the terminal that opens its region and right before the one that
    /// closes it, and the text before that opener holds no terminal of the table. Says too
    /// whether the turn was the one without the probe's thought, and whether it wrote a thought of
    /// its own before the answer.
    fn answer(&self, layout: &Layout, field: Option<ThoughtField>) -> Result<Answered, NoLayout> {
        let (reading, plain) = match (self.read(Case::Answer), field) {
            (Ok(reading), _) => (reading, true),
            (Err(NoLayout::Unrendered(_)), Some(field)) => {
                (self.read(Case::ThoughtAnswer(field))?, false)
            }
            (Err(error), _) => return Err(error),
        };
        expect(&reading, &Said::probe(0, !plain))?;
        let answer = answer_of(&reading).ok_or(NoLayout::Unexplained(reading.case))?;
        whole(&reading, &answer)?;
        let output = reading.output.as_str();
        let thought = thought_of(&reading, answer.start());
        let opened = thought
            .as_ref()
            .is_some_and(|thought| thought.open.is_some());
        let before = match thought {
            Some(thought) => {
                if closed_otherwise(layout, &output[thought.close.clone()]) {
                    return Err(NoLayout::Uneven("thought"));
                }
                let after = layout.thought.as_ref().map_or("", |m| &*m.after);
                output[thought.close.end..answer.start()]
                    .strip_prefix(after)
                    .ok_or(NoLayout::Uneven("thought"))?
            }
            None => &output[..answer.start()],
        };
        // The tags write the rows into the answer's region themselves, the terminal that opens it
        // last, so the text the layout writes before that terminal holds none of the markers.
        if answer.open.is_some() && self.marked(before) {
            return Err(NoLayout::Unexplained(reading.case));
        }
        let (after, stops) = match &answer.close {
            Some(close) => (&output[close.end..], false),
            None => (&output[answer.text.end..], true),
        };
        let message = Message {
            before: before.into(),
            after: after.into(),
            stops,
            close: answer
                .close
                .as_ref()
                .map(|close| output[close.clone()].into()),
        };
        Ok(Answered {
            message,
            region: output[answer.start()..].to_owned(),
            plain,
            opened,
        })
    }

    /// Whether `text` holds a terminal of the table: a marker its rows move on, or one it ignores.
    fn marked(&self, text: &str) -> bool {
        self.format
            .terminal_texts()
            .any(|terminal| !terminal.is_empty() && text.contains(terminal))
    }

    /// The turn with the answer and the probe's thought in `field`, where the chat template
    /// renders it with a thought: it reads back as the probe's thought and answer; between the
    /// thought's close and the answer's region it writes the thought's text after and the answer's
    /// text before, as the other turns do; and from there on it writes what the turn without the
    /// thought writes from its answer's region on (`region`).
    fn check_thought_answer(
        &self,
        field: ThoughtField,
        layout: &Layout,
        region: &str,
    ) -> Result<(), NoLayout> {
        let reading = match self.read(Case::ThoughtAnswer(field)) {
            Ok(reading) => reading,
            Err(NoLayout::Unrendered(_) | NoLayout::Prompt(_)) => return Ok(()),
            Err(error) => return Err(error),
        };
        // A chat template that writes the thought before calls but not before an answer leaves
        // nothing to compare; one that writes it there reads back as the thought and the answer.
        if said(&reading).reasoning.is_none() {
            return Ok(());
        }
        expect(&reading, &Said::probe(0, true))?;
        let answer = answer_of(&reading).ok_or(NoLayout::Unexplained(reading.case))?;
        let thought = thought_of(&reading, answer.start()).ok_or(NoLayout::Uneven("thought"))?;
        if closed_otherwise(layout, &reading.output[thought.close.clone()]) {
            return Err(NoLayout::Uneven("thought"));
        }
        let message = |message: Option<&Message>, at: fn(&Message) -> &str| {
            message.map_or(String::new(), |message| at(message).to_owned())
        };
        let written = format!(
            "{}{}",
            message(layout.thought.as_ref(), |m| &m.after),
            message(layout.answer.as_ref(), |m| &m.before),
        );
        if reading.output[thought.close.end..answer.start()] != written {
            return Err(NoLayout::Uneven("thought"));
        }
        if reading.output[answer.start()..] != *region {
            return Err(NoLayout::Uneven("answer"));
        }
        Ok(())
    }
}

/// The layout of a call, the text between two calls and after the last, and the spelling of
/// the values, from the turn with the calls. The thought and the answer come from other turns.
fn call_layout(turn: &Calls) -> Result<Layout, NoLayout> {
    let Calls {
        reading,
        calls,
        thought,
    } = turn;
    let output = reading.output.as_str();
    let first = &calls[0];
    // The lead: what the turn writes before the first call's terminal when nothing comes
    // before the call. After a thought, that text is the thought's.
    let lead = match thought {
        Some(_) => Written::default(),
        None => named(&output[..first.terminal.start], &first.name),
    };
    let text = |range: Range<usize>| output[range].to_owned();
    let header = |call: &Call| {
        let places: Vec<(Range<usize>, Slot)> = call
            .name_at
            .iter()
            .map(|range| (range.clone(), Slot::Name))
            .collect();
        written(output, call.header.clone(), &places)
    };
    let parameter = |param: &Param| {
        let places = [
            (param.key.clone(), Slot::Key),
            (param.value.clone(), Slot::Value),
        ];
        written(output, param.span.clone(), &places)
    };
    let before = |call: &Call| text(call.header.end..call.params[0].span.start);
    let after = |call: &Call| {
        let last = call.params.last().map_or(call.header.end, |p| p.span.end);
        text(last..call.trailer_at)
    };
    let trailer = |call: &Call| text(call.trailer_at..call.close.start);
    let between: Vec<String> = calls
        .iter()
        .flat_map(|call| {
            call.params
                .windows(2)
                .map(|pair| text(pair[0].span.end..pair[1].span.start))
        })
        .collect();
    alike("header", calls.iter().map(header))?;
    alike(
        "parameter",
        calls
            .iter()
            .flat_map(|call| call.params.iter().map(parameter)),
    )?;
    alike("text before the parameters", calls.iter().map(before))?;
    alike("text between two parameters", between.iter())?;
    alike("text after the parameters", calls.iter().map(after))?;
    alike("trailer", calls.iter().map(trailer))?;
    let order: Vec<&str> = first
        .params
        .iter()
        .map(|param| &output[param.key.clone()])
        .collect();
    if order != ALPHA_KEYS {
        return Err(NoLayout::Order);
    }
    // The close: the terminal each call's end took, or none where the stream's stop closed the
    // last; a call before it is then closed by a terminal of the text between two calls.
    let last = &calls[calls.len() - 1];
    let stopped = last.close.is_empty();
    if !stopped {
        alike("close", calls.iter().map(|call| text(call.close.clone())))?;
    }
    let between_calls = match calls.get(1) {
        Some(next) => {
            let from = if stopped {
                first.close.start
            } else {
                first.close.end
            };
            let gap = &output[from..next.terminal.start];
            let lead = fill(&lead, &next.name);
            let between = gap
                .strip_suffix(lead.as_str())
                .ok_or(NoLayout::Uneven("lead"))?;
            Some(Box::from(between))
        }
        None => None,
    };
    let layout = Layout {
        call: CallLayout {
            lead: lead.clone(),
            header: header(first),
            before: before(first).into(),
            parameter: parameter(&first.params[0]),
            between: between.first().cloned().unwrap_or_default().into(),
            after: after(first).into(),
            trailer: trailer(first).into(),
            close: (!stopped).then(|| text(last.close.clone()).into()),
        },
        between_calls,
        after_calls: output[last.close.end..].into(),
        values: spellings(output, first)?,
        thought: None,
        answer: None,
    };
    // The parts write the calls again, from the first call's lead to the end of the turn.
    let from = first.terminal.start - fill(&lead, &first.name).len();
    if write_calls(output, &layout, calls) != output[from..] {
        return Err(NoLayout::Unexplained(reading.case));
    }
    Ok(layout)
}

/// The bytes of the output an event accounts for.
fn bytes_of(event: &Event) -> &str {
    match event {
        Event::Content(text) | Event::Reasoning(text) => &text.text,
        Event::Dropped { text, .. }
        | Event::Malformed { text, .. }
        | Event::ToolCallStart { source: text, .. }
        | Event::ToolCallArguments { source: text, .. }
        | Event::ToolCallEnd { source: text, .. } => &text.text,
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
    }
}

/// What `reading` says, as the Chat API's message holds it.
fn said(reading: &Reading) -> Said {
    let message = adapt::chat::message(0, reading.events.iter().map(|(_, event)| event)).message;
    Said {
        reasoning: message.reasoning_content.filter(|text| !text.is_empty()),
        content: message.content.filter(|text| !text.is_empty()),
        calls: message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|call| {
                let arguments = call
                    .function
                    .arguments
                    .and_then(|text| serde_json::from_str(&text).ok())
                    .unwrap_or(Value::Null);
                (call.function.name, arguments)
            })
            .collect(),
    }
}

/// That `reading` says what `wanted` says.
fn expect(reading: &Reading, wanted: &Said) -> Result<(), NoLayout> {
    let read = said(reading);
    if read == *wanted {
        return Ok(());
    }
    Err(NoLayout::ReadBack {
        case: reading.case,
        read: read.describe(),
    })
}

/// The thought `reading` holds, closed no later than `before` (a place in the output): the
/// terminal that opened it is the bytes dropped just before its start, where the output wrote
/// one, and the terminal that closed it the bytes dropped just before its end.
fn thought_of(reading: &Reading, before: usize) -> Option<Thought> {
    let events = &reading.events;
    let start = events
        .iter()
        .position(|(_, event)| matches!(event, Event::ReasoningStart))?;
    let end = start
        + events[start..]
            .iter()
            .position(|(_, event)| matches!(event, Event::ReasoningEnd))?;
    let marker = |at: Option<usize>| match at.map(|at| &events[at]) {
        Some((range, Event::Dropped { .. })) if !range.is_empty() => Some(range.clone()),
        _ => None,
    };
    let close = marker(end.checked_sub(1))?;
    if close.end > before {
        return None;
    }
    Some(Thought {
        open: marker(start.checked_sub(1)),
        close,
    })
}

/// The text a turn writes around its `thought`, before the first call at `call`: the text before
/// the thought's opener (none where the prompt opened it), and the text from its close to the
/// call.
fn around(output: &str, thought: &Thought, call: usize) -> Message {
    let before = thought
        .open
        .as_ref()
        .map_or("", |open| &output[..open.start]);
    Message {
        before: before.into(),
        after: output[thought.close.end..call].into(),
        stops: false,
        close: Some(output[thought.close.clone()].into()),
    }
}

/// Whether a turn closed its thought with `close`, another terminal than the one the layout's
/// thought was closed with.
fn closed_otherwise(layout: &Layout, close: &str) -> bool {
    layout
        .thought
        .as_ref()
        .and_then(|thought| thought.close.as_deref())
        .is_some_and(|written| written != close)
}

/// The thought the turn with the calls holds before its first call (one the prompt opened, or the
/// probe's), with the text the turn writes around it.
fn held(calls: &Calls) -> Option<Message> {
    let first = calls.calls[0].terminal.start;
    let thought = calls.thought.as_ref()?;
    Some(around(&calls.reading.output, thought, first))
}

/// The answer `reading` holds: its text is its content's bytes; the terminal that opened its
/// region is the last bytes before its text that were dropped as other than whitespace a trimmed
/// state drops, where those come after the turn's other regions; its close the first such bytes
/// after its text.
fn answer_of(reading: &Reading) -> Option<Answer> {
    let events = &reading.events;
    let first = events
        .iter()
        .position(|(_, event)| matches!(event, Event::Content(_)))?;
    let last = events
        .iter()
        .rposition(|(_, event)| matches!(event, Event::Content(_)))?;
    let marker = |(range, event): &(Range<usize>, Event)| match event {
        Event::Dropped { why, .. } if *why != DropReason::Whitespace && !range.is_empty() => {
            Some(Some(range.clone()))
        }
        Event::Dropped { .. } => None,
        _ => Some(None),
    };
    Some(Answer {
        open: events[..first].iter().rev().find_map(marker).flatten(),
        text: events[first].0.start..events[last].0.end,
        close: events[last + 1..].iter().find_map(marker).flatten(),
    })
}

/// That `reading` writes its answer's text right after the terminal that opens the answer's region
/// and right before the one that closes it: a layout has no place for text between them (a
/// trimmed newline), and the tags would not write it.
fn whole(reading: &Reading, answer: &Answer) -> Result<(), NoLayout> {
    let opens = answer
        .open
        .as_ref()
        .is_none_or(|open| open.end == answer.text.start);
    let closes = answer
        .close
        .as_ref()
        .is_none_or(|close| close.start == answer.text.end);
    if opens && closes {
        Ok(())
    } else {
        Err(NoLayout::Unexplained(reading.case))
    }
}

/// What `pattern`, followed by one of `ends`, matches at the start of `text`, `before` being the
/// character before it; `None` without a match.
fn matched(
    pattern: &Pattern,
    ends: &[End],
    word: &Class,
    before: Option<char>,
    text: &str,
) -> Option<Match> {
    let mut matcher = Matcher::new(pattern, ends, word, before).hold_cap(None);
    let verdict = match matcher.feed(text) {
        Verdict::Pending => matcher.finish(),
        decided => decided,
    };
    match verdict {
        Verdict::Matched(found) => Some(found),
        Verdict::Pending | Verdict::NoMatch => None,
    }
}

/// The text of `output` at `range`, with the places at `places`: sorted, inside it, apart.
fn written(output: &str, range: Range<usize>, places: &[(Range<usize>, Slot)]) -> Written {
    let mut segments = Vec::new();
    let mut at = range.start;
    for (place, slot) in places {
        if place.start > at {
            segments.push(Segment::Lit(output[at..place.start].into()));
        }
        segments.push(Segment::Slot(*slot));
        at = place.end;
    }
    if range.end > at {
        segments.push(Segment::Lit(output[at..range.end].into()));
    }
    Written(segments.into())
}

/// `text` with a place for the tool's name wherever `name` stands in it.
fn named(text: &str, name: &str) -> Written {
    let places: Vec<(Range<usize>, Slot)> = text
        .match_indices(name)
        .map(|(at, found)| (at..at + found.len(), Slot::Name))
        .collect();
    written(text, 0..text.len(), &places)
}

/// `written` with `name` in its places.
fn fill(written: &Written, name: &str) -> String {
    written
        .0
        .iter()
        .map(|segment| match segment {
            Segment::Lit(text) => &**text,
            Segment::Slot(_) => name,
        })
        .collect()
}

/// That every part as written is the same.
fn alike<T: PartialEq>(
    part: &'static str,
    parts: impl IntoIterator<Item = T>,
) -> Result<(), NoLayout> {
    let mut parts = parts.into_iter();
    match parts.next() {
        Some(first) if !parts.all(|other| other == first) => Err(NoLayout::Uneven(part)),
        _ => Ok(()),
    }
}

/// The calls the layout writes, with the names, the keys and the values the turn wrote: the turn
/// from the first call's lead to its end, where the parts are right.
fn write_calls(output: &str, layout: &Layout, calls: &[Call]) -> String {
    let call = &layout.call;
    let mut text = String::new();
    for (at, written) in calls.iter().enumerate() {
        if at > 0 {
            text.push_str(layout.between_calls.as_deref().unwrap_or(""));
        }
        text.push_str(&fill(&call.lead, &written.name));
        text.push_str(&output[written.terminal.clone()]);
        text.push_str(&fill(&call.header, &written.name));
        text.push_str(&call.before);
        for (index, param) in written.params.iter().enumerate() {
            if index > 0 {
                text.push_str(&call.between);
            }
            for segment in &*call.parameter.0 {
                text.push_str(match segment {
                    Segment::Lit(lit) => lit,
                    Segment::Slot(Slot::Key) => &output[param.key.clone()],
                    Segment::Slot(Slot::Value) => &output[param.value.clone()],
                    Segment::Slot(Slot::Name) => written.name.as_str(),
                });
            }
        }
        text.push_str(&call.after);
        text.push_str(&call.trailer);
        text.push_str(call.close.as_deref().unwrap_or(""));
    }
    text.push_str(&layout.after_calls);
    text
}

/// How the turn spelled `fn_alpha`'s values that are not strings: its numbers, its list and its
/// object must be JSON, and its `true` and `null` JSON's words or Python's.
fn spellings(output: &str, call: &Call) -> Result<Spellings, NoLayout> {
    let value = |key: &str| {
        call.params
            .iter()
            .find(|param| &output[param.key.clone()] == key)
            .map_or("", |param| &output[param.value.clone()])
    };
    let json = |text: &str| serde_json::from_str::<Value>(text).ok();
    if json(value("i")) != Some(json!(42)) || json(value("n")) != Some(json!(2.5)) {
        return Err(NoLayout::Spelling("number"));
    }
    let words = match (value("b"), value("z")) {
        ("true", "null") => Words::Json,
        ("True", "None") => Words::Python,
        _ => return Err(NoLayout::Spelling("boolean or null")),
    };
    // A list or an object written as other text would not read back as the probe's: its
    // parameter's type casts JSON alone.
    if json(value("a")) != Some(json!([1, 2])) || json(value("o")) != Some(json!({"m": "mu"})) {
        return Err(NoLayout::Spelling("list or object"));
    }
    Ok(Spellings {
        words,
        nested: Nested::Json,
    })
}

/// That no text the layout writes for every tool names one of the probe's tools: each name stands
/// in a place for the name, or the text is the tool's alone.
fn unnamed(layout: &Layout) -> Result<(), NoLayout> {
    let literals = |written: &Written| -> String {
        written
            .0
            .iter()
            .filter_map(|segment| match segment {
                Segment::Lit(text) => Some(&**text),
                Segment::Slot(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\u{0}")
    };
    let call = &layout.call;
    let message = |message: Option<&Message>| {
        message.map_or(String::new(), |m| format!("{}\u{0}{}", m.before, m.after))
    };
    let parts = [
        ("lead", literals(&call.lead)),
        ("header", literals(&call.header)),
        ("parameter", literals(&call.parameter)),
        ("text before the parameters", call.before.to_string()),
        ("text between two parameters", call.between.to_string()),
        ("text after the parameters", call.after.to_string()),
        ("trailer", call.trailer.to_string()),
        ("close", call.close.as_deref().unwrap_or("").to_owned()),
        (
            "text between two calls",
            layout.between_calls.as_deref().unwrap_or("").to_owned(),
        ),
        ("text after the calls", layout.after_calls.to_string()),
        ("text around the thought", message(layout.thought.as_ref())),
        ("text around the answer", message(layout.answer.as_ref())),
    ];
    for (part, text) in parts {
        if ["fn_alpha", "fn_beta"]
            .iter()
            .any(|name| text.contains(name))
        {
            return Err(NoLayout::Named(part));
        }
    }
    Ok(())
}
