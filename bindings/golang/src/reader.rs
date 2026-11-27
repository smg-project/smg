//! The Symphony reader of a request's choices: what the request's output
//! is read with ([`Reading`]), the engine over each choice ([`Reader`]), and
//! the client's delta its events make at every step ([`Step`]).
//!
//! Every choice is read through a table, as the gateway reads every chat
//! response through one. A request that constrains its output through the
//! generic JSON schema (`tool_constraint::tool_constraint_for`) is not in the
//! model's own syntax: a named function yields that function's arguments
//! object alone, which the `forced` table reads as that one call, and
//! `required` yields a list of `{"name": …, "parameters": …}` objects, which
//! the `json` table reads. Every other request is read by the model's own
//! table, the registry's for the model, else `plain`: content as written,
//! what the older passthrough parser gave a model it did not know. The
//! registry lists a checkpoint by its exact stem, so a renamed or quantized
//! one (`Qwen3-8B-FP8`) reads as `plain`, as the gateway reads it without an
//! operator's parser name; the binding takes no such name yet. The
//! model's table reads without its calls where the response is not scanned
//! for calls (no tools declared, on a provider that reads none without
//! them), so a call-shaped text is content while the thought is still
//! reasoning; and without its thought where the request keeps the reasoning
//! in the content (`separate_reasoning: false`). Under `tool_choice: none`
//! over declared tools the table keeps its calls and the reader drops every
//! call, as the engine's own server reads such a call and withholds it. The
//! gateway's streaming path makes the same choices for its readers.

use std::sync::{Arc, LazyLock};

use openai_protocol::{
    chat::{ChatMessageDelta, ChatStreamChoiceRef},
    common::{Tool, ToolCallDelta, ToolChoice, ToolChoiceValue},
    profile::ProviderProfile,
};
use symphony::{
    adapt::chat,
    event::forbidden_calls,
    formats::{for_model_shared, forced, Table},
    CallId, Declared, Engine, EngineFinish, Events, Format, Input, Parser,
};

/// The `forced` table, built once and shared by every request whose forced
/// call the binding reads; the registry shares every other table.
static FORCED: LazyLock<Arc<Format>> = LazyLock::new(|| Arc::new(forced()));

/// What a request's output is read with, decided once per request and shared
/// by the readers of its choices: the table, the tools the calls resolve
/// against (with the function a named choice forces), and whether the
/// readers drop the calls the table reads.
#[derive(Debug, PartialEq)]
pub(crate) struct Reading {
    format: Arc<Format>,
    declared: Declared,
    /// `tool_choice: none` over a response scanned for calls: the table keeps
    /// its calls, so a call the model writes anyway is read whole, and the
    /// reader drops it, as the engine's own server reads such a call and
    /// withholds it.
    forbids_calls: bool,
    /// A forcing choice on a model whose own table keeps the call's header as
    /// its id ([`CallId::Header`], the Kimi K2 line): the `forced` and `json`
    /// tables read no header, so the reader writes the header's shape,
    /// `functions.NAME:INDEX`, which the gateway keeps through the table's own
    /// tag and the older binding minted.
    header_ids: bool,
}

impl Reading {
    /// The reading of a request for `model` with `tools` (the request's
    /// effective tools; `None` when it declares none), its `tool_choice`, and
    /// whether it separates the reasoning from the content.
    ///
    /// The response is scanned for calls when the request declares tools or
    /// the model's provider reads calls without any (a tool the conversation
    /// established comes back as a call), as the gateway decides it.
    pub(crate) fn of(
        model: &str,
        tools: Option<&[Tool]>,
        tool_choice: Option<&ToolChoice>,
        separate_reasoning: bool,
    ) -> Self {
        let scans_for_calls =
            tools.is_some() || ProviderProfile::for_model(model).parses_tool_calls_without_tools();
        let (format, forcing) = table_for(
            model,
            tool_choice,
            tools.is_some(),
            !scans_for_calls,
            !separate_reasoning,
        );
        let declared = Declared::of(tools.unwrap_or_default());
        let declared = match forcing {
            Some(name) => declared.forcing(name),
            None => declared,
        };
        let forbids_calls = scans_for_calls
            && matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None)));
        let forcing = tools.is_some() && tool_choice.is_some_and(ToolChoice::forces_call);
        let header_ids = forcing
            && for_model_shared(model)
                .is_some_and(|table| table.call_id_source() == CallId::Header);
        Self {
            format,
            declared,
            forbids_calls,
            header_ids,
        }
    }

    /// A reader for one choice of the request, at the start of its output,
    /// after `prompt`, the rendered prompt the engine replays the readable
    /// part of ([`Reader::over`]).
    pub(crate) fn reader(&self, prompt: &str) -> Result<Reader, String> {
        Reader::over(
            Engine::shared(Arc::clone(&self.format), self.declared.clone()),
            self.forbids_calls,
            self.header_ids,
            prompt,
        )
    }

    /// The part of a rendered prompt the readers replay ([`Format::replay`]):
    /// from the table's last turn opener, the whole prompt for a table that
    /// names none. A converter keeps this part for its readers, not the
    /// prompt, as the gateway's readers keep it.
    pub(crate) fn replay<'p>(&self, prompt: &'p str) -> &'p str {
        self.format.replay(prompt)
    }

    /// The table's name, for a view of the converter's handle.
    #[cfg(test)]
    pub(crate) fn table(&self) -> &str {
        self.format.name()
    }
}

/// The table that reads a request's output and, when the request names a
/// function, that function: `forced` for a named function, `json` for a
/// choice that forces a call without naming one (`required`, an allowed-tools
/// list in `required` mode), both over declared tools alone, since nothing
/// constrains the output of such a choice without tools (the request path
/// refuses it, `tool_constraint::tool_constraint_for`); else the registry's
/// table for `model` or `plain`, without its calls under `no_calls` and
/// without its thought under `no_reasoning` ([`asked`]).
fn table_for<'c>(
    model: &str,
    tool_choice: Option<&'c ToolChoice>,
    tools_declared: bool,
    no_calls: bool,
    no_reasoning: bool,
) -> (Arc<Format>, Option<&'c str>) {
    match tool_choice {
        Some(ToolChoice::Function { function, .. }) if tools_declared => {
            (Arc::clone(&FORCED), Some(function.name.as_str()))
        }
        Some(choice) if tools_declared && choice.forces_call() => (Table::Json.shared(), None),
        _ => {
            let table = for_model_shared(model).unwrap_or_else(|| Table::Plain.shared());
            (asked(table, no_calls, no_reasoning), None)
        }
    }
}

/// `table` as the request asks for it: itself when it asks for nothing, else
/// a copy without its calls, without its thought, or both. The gateway keeps
/// such variants across its requests; the binding's converter is one
/// request's, so the variant it asks for is built with it.
fn asked(table: Arc<Format>, no_calls: bool, no_reasoning: bool) -> Arc<Format> {
    if !no_calls && !no_reasoning {
        return table;
    }
    let mut variant = (*table).clone();
    if no_calls {
        variant = variant.without_calls();
    }
    if no_reasoning {
        variant = variant.without_reasoning();
    }
    Arc::new(variant)
}

/// The engine's finish reason as Symphony takes it. A blank reason is a stop,
/// as the converter read it before a table read the stream.
pub(crate) fn engine_finish(reason: &str) -> EngineFinish {
    match reason.trim() {
        "" | "stop" => EngineFinish::Stop,
        "length" => EngineFinish::Length,
        "abort" => EngineFinish::Abort,
        other => EngineFinish::Other(other.to_string()),
    }
}

/// What one step of a choice's output came to for the client: the content,
/// the reasoning and the calls the table released, folded into one delta,
/// and the finish reason once the choice ended.
#[derive(Debug, Default)]
pub(crate) struct Step {
    pub(crate) content: String,
    pub(crate) reasoning: String,
    pub(crate) tool_calls: Vec<ToolCallDelta>,
    pub(crate) finish_reason: Option<String>,
}

impl Step {
    /// Whether the step gives the client nothing: no text, no call, no
    /// finish.
    pub(crate) fn is_empty(&self) -> bool {
        self.content.is_empty()
            && self.reasoning.is_empty()
            && self.tool_calls.is_empty()
            && self.finish_reason.is_none()
    }

    /// This step and `later` as one: the later texts and calls after these,
    /// the later finish reason where it has one.
    pub(crate) fn then(mut self, later: Self) -> Self {
        self.content.push_str(&later.content);
        self.reasoning.push_str(&later.reasoning);
        self.tool_calls.extend(later.tool_calls);
        if later.finish_reason.is_some() {
            self.finish_reason = later.finish_reason;
        }
        self
    }

    /// The delta the client gets: the role, and each text and the calls when
    /// the step has them.
    pub(crate) fn into_delta(self) -> ChatMessageDelta {
        ChatMessageDelta {
            role: Some("assistant".to_string()),
            content: (!self.content.is_empty()).then_some(self.content),
            tool_calls: (!self.tool_calls.is_empty()).then_some(self.tool_calls),
            reasoning_content: (!self.reasoning.is_empty()).then_some(self.reasoning),
        }
    }

    /// One stream choice of the chat adapter folded in: its text after the
    /// step's, its calls after the step's (their ids in the header's shape
    /// under `header_ids`, [`Reading::header_ids`]), its finish reason as
    /// the step's. The texts come borrowed from the event
    /// (`chat::into_delta_ref`), so the one copy a step makes of them is the
    /// one into itself.
    fn fold(&mut self, choice: ChatStreamChoiceRef<'_>, header_ids: bool) {
        let delta = choice.delta;
        if let Some(text) = delta.content {
            self.content.push_str(&text);
        }
        if let Some(text) = delta.reasoning_content {
            self.reasoning.push_str(&text);
        }
        if let Some(mut calls) = delta.tool_calls {
            if header_ids {
                calls.iter_mut().for_each(header_shape);
            }
            self.tool_calls.extend(calls);
        }
        if let Some(reason) = choice.finish_reason {
            self.finish_reason = Some(reason.into_owned());
        }
    }
}

/// The id of a call on a table whose header names it ([`CallId::Header`]):
/// `functions.NAME:INDEX`, as the model writes it and its templates render it
/// back, written over the minted id of a call the forced readers read without
/// a header. A delta that carries no id (the arguments after the first) keeps
/// none.
fn header_shape(call: &mut ToolCallDelta) {
    let Some(name) = call
        .function
        .as_ref()
        .and_then(|function| function.name.as_deref())
    else {
        return;
    };
    if call.id.is_some() {
        call.id = Some(format!("functions.{name}:{}", call.index));
    }
}

/// One choice's reader: the engine over the choice's output, fed the text
/// the converter's decoder releases step by step, then the stream's end. Its
/// state (what it holds as a prospective call, the calls it started) belongs
/// to one choice, so the choices of an `n > 1` request never share one, as
/// the gateway's streaming path keeps a reader per choice.
pub(crate) struct Reader {
    engine: Engine,
    /// The events of a step, one buffer reused for every step of the choice.
    events: Events<'static>,
    /// The reading forbids calls ([`Reading::forbids_calls`]): the call
    /// events of every step become dropped text before the adapter sees
    /// them, and a `tool_calls` finish is the engine's `stop` again.
    forbids_calls: bool,
    /// The calls carry the header's id shape ([`Reading::header_ids`]).
    header_ids: bool,
}

impl Reader {
    /// A reader over `engine`, at the start of the output after `prompt`,
    /// dropping the calls the table reads under `forbids_calls` and writing
    /// their ids in the header's shape under `header_ids`.
    ///
    /// The engine replays the readable part of `prompt`, the rendered
    /// prompt, as the gateway's readers do: a thought the template opens
    /// inside the prompt, or the turn opener, puts the output in the right
    /// state. The chat path passes the prompt it rendered; the C entry point
    /// does not receive it yet and passes an empty one, so there the output
    /// starts in the table's first state and such a thought reads as
    /// content. The forced, json and plain tables have nothing to replay.
    pub(crate) fn over(
        mut engine: Engine,
        forbids_calls: bool,
        header_ids: bool,
        prompt: &str,
    ) -> Result<Self, String> {
        let mut events = Events::new();
        engine
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut events,
            )
            .map_err(|error| format!("Symphony refused the prompt: {error}"))?;
        Ok(Self {
            engine,
            events: events.recycle(),
            forbids_calls,
            header_ids,
        })
    }

    /// The text the decoder released for one chunk of the choice at `index`,
    /// read: the step its events make. The converter's usage comes from the
    /// engine's counts, so the delta carries no token spans and the events
    /// count no tokens.
    pub(crate) fn read(&mut self, index: u32, text: &str) -> Result<Step, String> {
        if text.is_empty() {
            return Ok(Step::default());
        }
        self.feed(
            index,
            Input::Delta {
                token_ids: &[],
                text,
                spans: &[],
            },
        )
    }

    /// The choice at `index` ended as the engine said: what the table still
    /// held goes out, with the finish reason the table decides (`tool_calls`
    /// after a call the model stopped on, else the engine's).
    pub(crate) fn end(&mut self, index: u32, finish: EngineFinish) -> Result<Step, String> {
        self.feed(index, Input::End { finish })
    }

    /// One input to the engine, its events rendered as the client's deltas
    /// and folded into one step. The engine refuses only a step out of its
    /// lifecycle order, which the converter never feeds; such a refusal is
    /// the conversion's error.
    fn feed(&mut self, index: u32, input: Input<'_>) -> Result<Step, String> {
        let mut events = std::mem::take(&mut self.events).recycle();
        let fed = self.engine.feed(input, &mut events);
        if self.forbids_calls {
            forbidden_calls(events.as_mut_slice());
        }
        let mut step = Step::default();
        for event in events.consume() {
            if let Some(choice) = chat::into_delta_ref(index, event) {
                step.fold(choice, self.header_ids);
            }
        }
        self.events = events.recycle();
        fed.map_err(|error| format!("Symphony refused the output: {error}"))?;
        Ok(step)
    }
}

#[cfg(test)]
#[path = "../tests/unit/reader/tests.rs"]
mod tests;
