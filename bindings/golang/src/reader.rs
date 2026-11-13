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
mod tests {
    use std::{
        collections::BTreeMap,
        ffi::{CStr, CString},
        os::raw::c_char,
        ptr,
    };

    use llm_tokenizer::traits::Tokenizer;
    use serde_json::{json, Value};
    use smg_grpc_client::sglang_proto as proto;

    use super::*;
    use crate::{
        grpc_converter::{
            convert_proto_chunk_to_openai, sgl_grpc_response_converter_create,
            sgl_grpc_response_converter_free, GrpcResponseConverterHandle,
        },
        tokenizer::TokenizerHandle,
    };

    const TOOLS: &str = r#"[{"type":"function","function":{"name":"get_weather",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]"#;
    const NAMED: &str = r#"{"type":"function","function":{"name":"get_weather"}}"#;

    fn tools() -> Vec<Tool> {
        serde_json::from_str(TOOLS).unwrap()
    }

    fn choice(json: &str) -> ToolChoice {
        serde_json::from_str(json).unwrap()
    }

    /// A reader for one choice of a request for `model` over [`tools`] with
    /// `tool_choice`, the reasoning separated.
    fn reader(model: &str, tool_choice: Option<&ToolChoice>) -> Reader {
        Reading::of(model, Some(&tools()), tool_choice, true)
            .reader("")
            .unwrap()
    }

    /// A thinking model's thought and a call, in Qwen3's syntax.
    const THOUGHT_AND_CALL: &str = concat!(
        "<think>\nParis it is.\n</think>\n\n",
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
    );

    /// The arguments the step's call deltas spell, parsed.
    fn arguments_of(step: &Step) -> Value {
        let arguments: String = step
            .tool_calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        serde_json::from_str(&arguments).unwrap()
    }

    fn names_of(step: &Step) -> Vec<&str> {
        step.tool_calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.name.as_deref())
            .collect()
    }

    /// The table follows the choice: a named function reads through
    /// `forced` with the function's name, a choice that forces a call without
    /// naming one through `json`, and `auto` (or no choice) through the
    /// model's own table, `plain` for a model the registry does not know.
    #[test]
    fn the_table_follows_the_tool_choice_and_the_model() {
        let named = choice(NAMED);
        let (table, forced) = table_for("unmapped-model", Some(&named), true, false, false);
        assert_eq!((table.name(), forced), ("forced", Some("get_weather")));
        let required = choice(r#""required""#);
        let allowed = choice(
            r#"{"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"get_weather"}]}"#,
        );
        for choice in [&required, &allowed] {
            let (table, forced) = table_for("unmapped-model", Some(choice), true, false, false);
            assert_eq!((table.name(), forced), ("json", None), "{choice:?}");
        }
        let auto = choice(r#""auto""#);
        for choice in [Some(&auto), None] {
            let (table, forced) = table_for("unmapped-model", choice, true, false, false);
            assert_eq!((table.name(), forced), ("plain", None), "{choice:?}");
            let (table, forced) = table_for("Qwen/Qwen3-8B", choice, true, false, false);
            assert_eq!((table.name(), forced), ("qwen3", None), "{choice:?}");
        }
        // The reading's asks: a request that asks for nothing reads with the
        // shared table itself; without tools, with the table without its
        // calls; keeping the reasoning in the content, without its thought.
        // Calls are forbidden under `none` over declared tools alone.
        let shared = for_model_shared("Qwen/Qwen3-8B").unwrap();
        let declared = tools();
        let with_tools = Reading::of("Qwen/Qwen3-8B", Some(&declared), None, true);
        assert!(Arc::ptr_eq(&with_tools.format, &shared) && !with_tools.forbids_calls);
        let without_tools = Reading::of("Qwen/Qwen3-8B", None, None, true);
        assert_eq!(*without_tools.format, (*shared).clone().without_calls());
        let kept_in_content = Reading::of("Qwen/Qwen3-8B", Some(&declared), None, false);
        assert_eq!(
            *kept_in_content.format,
            (*shared).clone().without_reasoning()
        );
        let none = choice(r#""none""#);
        assert!(Reading::of("Qwen/Qwen3-8B", Some(&declared), Some(&none), true).forbids_calls);
        assert!(!Reading::of("Qwen/Qwen3-8B", None, Some(&none), true).forbids_calls);
        assert!(!Reading::of("Qwen/Qwen3-8B", Some(&declared), Some(&auto), true).forbids_calls);
        // A forcing choice without tools picks no forcing table: the model's
        // own table reads the output, `plain` for a model the registry does
        // not know.
        for choice in [&named, &required] {
            let (table, forced) = table_for("unmapped-model", Some(choice), false, true, false);
            assert_eq!((table.name(), forced), ("plain", None), "{choice:?}");
            let (table, forced) = table_for("Qwen/Qwen3-8B", Some(choice), false, true, false);
            assert_eq!((table.name(), forced), ("qwen3", None), "{choice:?}");
        }
        // The registry lists a checkpoint by its exact stem: a renamed or
        // quantized one reads as `plain`.
        assert_eq!(
            Reading::of("Qwen/Qwen3-8B-FP8", Some(&declared), None, true).table(),
            "plain"
        );
        assert_eq!(
            Reading::of("qwen/qwen3-8b", Some(&declared), None, true).table(),
            "qwen3"
        );
    }

    /// A forcing choice on a request without tools (the request path refuses
    /// it; a converter may still be created with one): nothing constrains the
    /// output, so the model's table reads it as content, not as a call made
    /// of the prose.
    #[test]
    fn a_forcing_choice_without_tools_reads_the_output_as_content() {
        let prose = "Hello, I cannot check the weather.";
        for tool_choice in [choice(NAMED), choice(r#""required""#)] {
            let mut reader = Reading::of("unmapped-model", None, Some(&tool_choice), true)
                .reader("")
                .unwrap();
            let step = reader
                .read(0, prose)
                .unwrap()
                .then(reader.end(0, EngineFinish::Stop).unwrap());
            assert_eq!(step.content, prose, "{tool_choice:?}");
            assert!(step.tool_calls.is_empty(), "{tool_choice:?}: {step:?}");
            assert_eq!(
                step.finish_reason.as_deref(),
                Some("stop"),
                "{tool_choice:?}"
            );
        }
    }

    /// `required` and an allowed-tools `required` mode: the list the schema
    /// forces reads as the call it names, no content.
    #[test]
    fn a_required_choice_reads_the_forced_list_as_a_call() {
        let output = r#"[{"name":"get_weather","parameters":{"city":"Paris"}}]"#;
        for tool_choice in [
            r#""required""#,
            r#"{"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"get_weather"}]}"#,
        ] {
            let mut reader = reader("unmapped-model", Some(&choice(tool_choice)));
            let step = reader
                .read(0, output)
                .unwrap()
                .then(reader.end(0, EngineFinish::Stop).unwrap());
            assert!(step.content.is_empty(), "{tool_choice}: {step:?}");
            assert_eq!(names_of(&step), ["get_weather"], "{tool_choice}");
            assert_eq!(
                arguments_of(&step),
                json!({"city": "Paris"}),
                "{tool_choice}"
            );
            assert_eq!(step.finish_reason.as_deref(), Some("tool_calls"));
        }
    }

    /// A named function: the arguments object alone is that call, the name
    /// from the request with the first delta, the arguments as they come,
    /// every delta on call index 0.
    #[test]
    fn a_named_function_reads_its_arguments_as_that_call() {
        let mut reader = reader("unmapped-model", Some(&choice(NAMED)));
        let first = reader.read(0, r#"{"city":"#).unwrap();
        let second = reader.read(0, r#""Paris"}"#).unwrap();
        let end = reader.end(0, EngineFinish::Stop).unwrap();
        assert_eq!(names_of(&first), ["get_weather"], "{first:?}");
        assert!(first.tool_calls[0].id.is_some(), "{first:?}");
        assert!(names_of(&second).is_empty(), "{second:?}");
        let whole = first.then(second).then(end);
        assert!(whole.content.is_empty() && whole.reasoning.is_empty());
        assert!(whole.tool_calls.iter().all(|call| call.index == 0));
        assert_eq!(arguments_of(&whole), json!({"city": "Paris"}));
        assert_eq!(whole.finish_reason.as_deref(), Some("tool_calls"));
    }

    /// `auto` on a model the registry does not know: `plain` reads the output
    /// as content, a JSON answer included, and announces no call; the finish
    /// is the engine's.
    #[test]
    fn an_unknown_model_under_auto_is_read_as_content() {
        let answer = r#"[{"name":"get_weather","parameters":{"city":"Paris"}}]"#;
        for tool_choice in [Some(choice(r#""auto""#)), None] {
            let mut reader = reader("unmapped-model", tool_choice.as_ref());
            let step = reader
                .read(0, answer)
                .unwrap()
                .then(reader.end(0, EngineFinish::Length).unwrap());
            assert_eq!(step.content, answer, "{tool_choice:?}");
            assert!(step.tool_calls.is_empty(), "{tool_choice:?}");
            assert_eq!(step.finish_reason.as_deref(), Some("length"));
        }
    }

    /// `auto` on a model the registry knows: its own table reads its own
    /// syntax, the thought as reasoning and the block as a call.
    #[test]
    fn a_known_model_under_auto_is_read_by_its_table() {
        let mut reader = reader("Qwen/Qwen3-8B", Some(&choice(r#""auto""#)));
        let step = reader
            .read(0, THOUGHT_AND_CALL)
            .unwrap()
            .then(reader.end(0, EngineFinish::Stop).unwrap());
        assert_eq!(step.reasoning, "\nParis it is.\n");
        assert_eq!(step.content.trim(), "", "{step:?}");
        assert_eq!(names_of(&step), ["get_weather"]);
        assert_eq!(arguments_of(&step), json!({"city": "Paris"}));
        assert_eq!(step.finish_reason.as_deref(), Some("tool_calls"));
    }

    /// No tools: the model's table reads without its calls, so the thought is
    /// still reasoning and a call block the model writes is content, markers
    /// and all; the finish is the engine's.
    #[test]
    fn without_tools_a_known_models_call_block_is_content_and_its_thought_reasoning() {
        let mut reader = Reading::of("Qwen/Qwen3-8B", None, None, true)
            .reader("")
            .unwrap();
        let step = reader
            .read(0, THOUGHT_AND_CALL)
            .unwrap()
            .then(reader.end(0, EngineFinish::Stop).unwrap());
        assert_eq!(step.reasoning, "\nParis it is.\n");
        assert!(step.tool_calls.is_empty(), "{step:?}");
        assert!(
            step.content.contains("<tool_call>") && step.content.contains(r#""city": "Paris""#),
            "{step:?}"
        );
        assert_eq!(step.finish_reason.as_deref(), Some("stop"));
    }

    /// `tool_choice: none` over declared tools: the table keeps its calls and
    /// the reader drops the call the model writes anyway, neither content nor
    /// call; the finish is `stop`, not `tool_calls`.
    #[test]
    fn under_none_a_call_the_model_writes_is_dropped() {
        let none = choice(r#""none""#);
        let mut reader = reader("Qwen/Qwen3-8B", Some(&none));
        let step = reader
            .read(0, THOUGHT_AND_CALL)
            .unwrap()
            .then(reader.end(0, EngineFinish::Stop).unwrap());
        assert_eq!(step.reasoning, "\nParis it is.\n");
        assert!(step.tool_calls.is_empty(), "{step:?}");
        assert_eq!(step.content.trim(), "", "{step:?}");
        assert_eq!(step.finish_reason.as_deref(), Some("stop"));
    }

    /// A request that keeps the reasoning in the content
    /// (`separate_reasoning: false`): the table reads no thought, the think
    /// block stays content with its markers, and the call still reads.
    #[test]
    fn a_request_that_keeps_the_reasoning_in_the_content_reads_no_thought() {
        let auto = choice(r#""auto""#);
        let mut reader = Reading::of("Qwen/Qwen3-8B", Some(&tools()), Some(&auto), false)
            .reader("")
            .unwrap();
        let step = reader
            .read(0, THOUGHT_AND_CALL)
            .unwrap()
            .then(reader.end(0, EngineFinish::Stop).unwrap());
        assert!(step.reasoning.is_empty(), "{step:?}");
        assert!(
            step.content.contains("<think>") && step.content.contains("Paris it is."),
            "{step:?}"
        );
        assert_eq!(names_of(&step), ["get_weather"]);
        assert_eq!(step.finish_reason.as_deref(), Some("tool_calls"));
    }

    /// A provider that reads calls without a tool inventory (MiniMax): the
    /// request declares none, and the call the model writes is a call.
    #[test]
    fn a_provider_that_reads_calls_without_tools_keeps_the_tables_calls() {
        let mut reader = Reading::of("MiniMaxAI/MiniMax-M3", None, None, true)
            .reader("")
            .unwrap();
        let output = "<tool_call>\n<invoke name=\"get_weather\">\n<city>Paris</city>\n</invoke>\n</tool_call>";
        let step = reader
            .read(0, output)
            .unwrap()
            .then(reader.end(0, EngineFinish::Stop).unwrap());
        assert_eq!(names_of(&step), ["get_weather"], "{step:?}");
        assert_eq!(arguments_of(&step)["city"], json!("Paris"), "{step:?}");
        assert_eq!(step.finish_reason.as_deref(), Some("tool_calls"));
    }

    /// A thought the template opens inside the prompt: the reader replays
    /// the rendered prompt, so the output starts inside the thought, the text
    /// before the closing marker is reasoning and the rest content. Without
    /// the prompt the same output starts in the table's first state.
    #[test]
    fn a_thought_the_template_opens_in_the_prompt_reads_as_reasoning() {
        let prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        let output = "Paris it is.\n</think>\n\nParis.";
        let mut seeded = Reading::of("Qwen/Qwen3-8B", Some(&tools()), None, true)
            .reader(prompt)
            .unwrap();
        let step = seeded
            .read(0, output)
            .unwrap()
            .then(seeded.end(0, EngineFinish::Stop).unwrap());
        assert_eq!(step.reasoning, "Paris it is.\n", "{step:?}");
        assert_eq!(step.content.trim(), "Paris.", "{step:?}");
        assert!(step.tool_calls.is_empty(), "{step:?}");
        assert_eq!(step.finish_reason.as_deref(), Some("stop"));
        let mut unseeded = Reading::of("Qwen/Qwen3-8B", Some(&tools()), None, true)
            .reader("")
            .unwrap();
        let step = unseeded
            .read(0, output)
            .unwrap()
            .then(unseeded.end(0, EngineFinish::Stop).unwrap());
        assert!(step.reasoning.is_empty(), "{step:?}");
        assert!(step.content.contains("Paris it is."), "{step:?}");
    }

    /// The chat path hands the converter the rendered prompt: a stream whose
    /// template opened the thought sends the thought as `reasoning_content`
    /// and the answer as content.
    #[test]
    fn the_chat_path_seeds_its_readers_with_the_rendered_prompt() {
        let tokenizer = byte_tokenizer();
        let request: openai_protocol::chat::ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen3-8B",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        let mut handle = GrpcResponseConverterHandle::for_chat(
            tokenizer.clone(),
            &request,
            "seeded".into(),
            prompt,
        );
        let first = convert_proto_chunk_to_openai(
            chunk_at(0, "Paris it is.\n</think>\n\nParis."),
            &mut handle,
            &tokenizer,
        )
        .unwrap()
        .expect("a delta");
        let delta = &first.choices[0].delta;
        assert_eq!(
            delta.reasoning_content.as_deref(),
            Some("Paris it is.\n"),
            "{first:?}"
        );
        assert_eq!(
            delta.content.as_deref().map(str::trim),
            Some("Paris."),
            "{first:?}"
        );
        let last = convert_proto_chunk_to_openai(complete_at(0, "stop"), &mut handle, &tokenizer)
            .unwrap()
            .expect("the end");
        assert_eq!(last.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    /// The converter keeps the part of the prompt its readers replay, not
    /// the prompt: from the table's last turn opener, the whole prompt for a
    /// table that names none, the bytes the gateway's readers keep.
    #[test]
    fn the_converter_keeps_the_part_of_the_prompt_its_readers_replay() {
        let tokenizer = byte_tokenizer();
        let prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        let request = |model: &str| -> openai_protocol::chat::ChatCompletionRequest {
            serde_json::from_value(json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .unwrap()
        };
        let known = GrpcResponseConverterHandle::for_chat(
            tokenizer.clone(),
            &request("Qwen/Qwen3-8B"),
            "kept".into(),
            prompt,
        );
        assert_eq!(known.prompt, "<|im_start|>assistant\n<think>\n");
        let unknown = GrpcResponseConverterHandle::for_chat(
            tokenizer,
            &request("Qwen/Qwen3-8B-FP8"),
            "kept".into(),
            prompt,
        );
        assert_eq!(unknown.prompt, prompt);
    }

    /// A forcing choice on a model whose table keeps the call's header as
    /// its id (the Kimi K2 line): the forced readers read no header, so the
    /// reader writes the header's shape, `functions.NAME:INDEX`, one index
    /// per call. Under `auto` the model's own table reads the header itself;
    /// a table that mints keeps its `call_` ids.
    #[test]
    fn a_forced_call_on_a_header_id_table_carries_the_header_shape() {
        let mut named = reader("moonshotai/Kimi-K2-Instruct", Some(&choice(NAMED)));
        let step = named
            .read(0, r#"{"city":"Paris"}"#)
            .unwrap()
            .then(named.end(0, EngineFinish::Stop).unwrap());
        assert_eq!(
            step.tool_calls[0].id.as_deref(),
            Some("functions.get_weather:0"),
            "{step:?}"
        );
        let mut required = reader(
            "moonshotai/Kimi-K2-Instruct",
            Some(&choice(r#""required""#)),
        );
        let step = required
            .read(
                0,
                r#"[{"name":"get_weather","parameters":{"city":"Paris"}},{"name":"get_weather","parameters":{"city":"Rome"}}]"#,
            )
            .unwrap()
            .then(required.end(0, EngineFinish::Stop).unwrap());
        let ids: Vec<&str> = step
            .tool_calls
            .iter()
            .filter_map(|call| call.id.as_deref())
            .collect();
        assert_eq!(
            ids,
            ["functions.get_weather:0", "functions.get_weather:1"],
            "{step:?}"
        );
        assert!(!Reading::of("moonshotai/Kimi-K2-Instruct", Some(&tools()), None, true).header_ids);
        let mut minted = reader("unmapped-model", Some(&choice(NAMED)));
        let step = minted
            .read(0, r#"{"city":"Paris"}"#)
            .unwrap()
            .then(minted.end(0, EngineFinish::Stop).unwrap());
        assert!(
            step.tool_calls[0]
                .id
                .as_deref()
                .is_some_and(|id| id.starts_with("call_")),
            "{step:?}"
        );
    }

    /// The engine's finish reasons as the table takes them: a blank one is a
    /// stop.
    #[test]
    fn the_engines_finish_reasons_map_and_a_blank_one_is_a_stop() {
        assert_eq!(engine_finish("stop"), EngineFinish::Stop);
        assert_eq!(engine_finish(""), EngineFinish::Stop);
        assert_eq!(engine_finish("   "), EngineFinish::Stop);
        assert_eq!(engine_finish("length"), EngineFinish::Length);
        assert_eq!(engine_finish("abort"), EngineFinish::Abort);
        assert_eq!(
            engine_finish("stop_sequence"),
            EngineFinish::Other("stop_sequence".to_string())
        );
    }

    /// A tokenizer whose ids are bytes (id = byte + 1), so a test spells the
    /// engine's output as token ids; the decoder releases text byte by byte.
    fn byte_tokenizer() -> Arc<dyn Tokenizer> {
        let path =
            std::env::temp_dir().join(format!("smg-forced-choice-{}.json", uuid::Uuid::now_v7()));
        let mut vocab = serde_json::Map::new();
        vocab.insert("[UNK]".into(), json!(0));
        for byte in 0_u32..=255 {
            vocab.insert(format!("<0x{byte:02X}>"), json!(byte + 1));
        }
        let definition = json!({
            "version":"1.0", "added_tokens":[],
            "model":{"type":"WordLevel", "unk_token":"[UNK]", "vocab":vocab},
            "decoder":{"type":"Sequence", "decoders":[
                {"type":"ByteFallback"}, {"type":"Fuse"}
            ]}
        });
        std::fs::write(&path, serde_json::to_vec(&definition).unwrap()).unwrap();
        let tokenizer =
            llm_tokenizer::HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(path).unwrap();
        Arc::new(tokenizer)
    }

    fn byte_ids(text: &str) -> Vec<u32> {
        text.bytes().map(|byte| u32::from(byte) + 1).collect()
    }

    /// Build an indexed SGLang chunk for forced-tool conversion tests.
    fn chunk_at(index: u32, text: &str) -> proto::GenerateResponse {
        proto::GenerateResponse {
            media_identity: None,
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Chunk(
                proto::GenerateStreamChunk {
                    token_ids: byte_ids(text),
                    index,
                    ..Default::default()
                },
            )),
        }
    }

    /// Build an indexed terminal response without optional media control metadata.
    fn complete_at(index: u32, finish_reason: &str) -> proto::GenerateResponse {
        proto::GenerateResponse {
            media_identity: None,
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Complete(
                proto::GenerateComplete {
                    finish_reason: finish_reason.into(),
                    index,
                    ..Default::default()
                },
            )),
        }
    }

    /// What a streamed choice came to: its content pieces, its tool-call
    /// deltas and its finish reason, with the converter's handle created
    /// the way the client creates it (the tool choice as JSON).
    struct Streamed {
        content: Vec<String>,
        reasoning: Vec<String>,
        names: Vec<String>,
        arguments: String,
        finish_reason: Option<String>,
    }

    fn stream_through_converter(tool_choice_json: Option<&str>, chunks: &[&str]) -> Streamed {
        let steps: Vec<(u32, &str)> = chunks.iter().map(|text| (0, *text)).collect();
        stream_choices_through_converter(tool_choice_json, &steps)
            .remove(&0)
            .expect("choice 0 streamed")
    }

    /// The choices of one request streamed through one converter handle,
    /// their chunks interleaved as given (choice index, text); every choice
    /// ends with a Complete, in index order.
    fn stream_choices_through_converter(
        tool_choice_json: Option<&str>,
        steps: &[(u32, &str)],
    ) -> BTreeMap<u32, Streamed> {
        stream_request_through_converter("unmapped-model", Some(TOOLS), tool_choice_json, steps)
    }

    /// [`stream_choices_through_converter`] for a request of `model` with
    /// `tools_json` (a null argument when `None`) and `tool_choice_json`.
    fn stream_request_through_converter(
        model: &str,
        tools_json: Option<&str>,
        tool_choice_json: Option<&str>,
        steps: &[(u32, &str)],
    ) -> BTreeMap<u32, Streamed> {
        let tokenizer = byte_tokenizer();
        let tokenizer_handle = Box::into_raw(Box::new(TokenizerHandle {
            tokenizer: tokenizer.clone(),
        }));
        let model = CString::new(model).unwrap();
        let request_id = CString::new("forced-choice").unwrap();
        let tools = tools_json.map(|json| CString::new(json).unwrap());
        let tool_choice = tool_choice_json.map(|json| CString::new(json).unwrap());
        let mut error_out: *mut c_char = ptr::null_mut();
        // SAFETY: every pointer is a live C string or null where the function allows null;
        // the tokenizer handle is a live Box released below.
        let handle = unsafe {
            sgl_grpc_response_converter_create(
                tokenizer_handle,
                model.as_ptr(),
                request_id.as_ptr(),
                tools.as_ref().map_or(ptr::null(), |tools| tools.as_ptr()),
                tool_choice
                    .as_ref()
                    .map_or(ptr::null(), |choice| choice.as_ptr()),
                ptr::null(),
                ptr::null(),
                1,
                0,
                ptr::null(),
                &mut error_out,
            )
        };
        if handle.is_null() {
            // SAFETY: a non-null error_out is a C string the create function allocated.
            let error = unsafe { CStr::from_ptr(error_out) }
                .to_string_lossy()
                .into_owned();
            panic!("converter creation failed: {error}");
        }
        let mut streamed: BTreeMap<u32, Streamed> = BTreeMap::new();
        {
            // SAFETY: the handle was just created and is freed once, below.
            let handle: &mut GrpcResponseConverterHandle = unsafe { &mut *handle };
            let mut responses = Vec::new();
            for (index, text) in steps {
                responses.push(convert_proto_chunk_to_openai(
                    chunk_at(*index, text),
                    handle,
                    &tokenizer,
                ));
            }
            let indexes: BTreeMap<u32, ()> = steps.iter().map(|(index, _)| (*index, ())).collect();
            for index in indexes.keys() {
                responses.push(convert_proto_chunk_to_openai(
                    complete_at(*index, "stop"),
                    handle,
                    &tokenizer,
                ));
            }
            for response in responses.into_iter().flat_map(|result| result.unwrap()) {
                let choice = &response.choices[0];
                let streamed = streamed.entry(choice.index).or_insert_with(|| Streamed {
                    content: Vec::new(),
                    reasoning: Vec::new(),
                    names: Vec::new(),
                    arguments: String::new(),
                    finish_reason: None,
                });
                if let Some(content) = &choice.delta.content {
                    streamed.content.push(content.clone());
                }
                if let Some(reasoning) = &choice.delta.reasoning_content {
                    streamed.reasoning.push(reasoning.clone());
                }
                for call in choice.delta.tool_calls.iter().flatten() {
                    if let Some(function) = &call.function {
                        if let Some(name) = &function.name {
                            streamed.names.push(name.clone());
                        }
                        if let Some(arguments) = &function.arguments {
                            streamed.arguments.push_str(arguments);
                        }
                    }
                }
                if let Some(reason) = &choice.finish_reason {
                    streamed.finish_reason = Some(reason.clone());
                }
            }
        }
        // SAFETY: both handles were created above and are released exactly once.
        unsafe {
            sgl_grpc_response_converter_free(handle);
            drop(Box::from_raw(tokenizer_handle));
        }
        streamed
    }

    /// `tool_choice: required` on a model the registry does not know: the
    /// engine's schema-forced list streams as a call named in it, no
    /// content, and the stream ends with `tool_calls`.
    #[test]
    fn a_required_choice_streams_the_forced_array_as_a_tool_call() {
        let streamed = stream_through_converter(
            Some(r#""required""#),
            &[
                r#"[{"name":"get_weather","parameters":{"city":"Pa"#,
                r#"ris"}}]"#,
            ],
        );
        assert!(streamed.content.is_empty(), "{:?}", streamed.content);
        assert_eq!(streamed.names, ["get_weather"]);
        assert_eq!(
            serde_json::from_str::<Value>(&streamed.arguments).unwrap(),
            json!({"city": "Paris"})
        );
        assert_eq!(streamed.finish_reason.as_deref(), Some("tool_calls"));
    }

    /// A named function: the engine emits that function's arguments object
    /// alone; it streams as that call's arguments, the name with the first
    /// delta, and the stream ends with `tool_calls`.
    #[test]
    fn a_named_function_streams_its_arguments_as_that_call() {
        let streamed = stream_through_converter(Some(NAMED), &[r#"{"city":"#, r#""Paris"}"#]);
        assert!(streamed.content.is_empty(), "{:?}", streamed.content);
        assert_eq!(streamed.names, ["get_weather"]);
        assert_eq!(
            serde_json::from_str::<Value>(&streamed.arguments).unwrap(),
            json!({"city": "Paris"})
        );
        assert_eq!(streamed.finish_reason.as_deref(), Some("tool_calls"));
    }

    /// `auto` keeps the model's table, `plain` here: text that is not in a
    /// tool-call syntax stays content and the stream ends as the engine said.
    #[test]
    fn an_auto_choice_keeps_the_models_table() {
        let streamed = stream_through_converter(Some(r#""auto""#), &["hello ", "there"]);
        assert_eq!(streamed.content.concat(), "hello there");
        assert!(streamed.names.is_empty() && streamed.arguments.is_empty());
        assert_eq!(streamed.finish_reason.as_deref(), Some("stop"));
    }

    /// A request without tools on a model the registry knows, through the C
    /// entry point with a null tools argument: the model's table reads the
    /// thought as reasoning and, without its calls, a call block as content;
    /// the stream ends as the engine said.
    #[test]
    fn without_tools_a_known_models_thought_streams_as_reasoning() {
        let streamed = stream_request_through_converter(
            "Qwen/Qwen3-8B",
            None,
            None,
            &[
                (0, "<think>\nParis it is.\n</think>\n\n"),
                (0, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"),
            ],
        )
        .remove(&0)
        .expect("choice 0 streamed");
        assert_eq!(streamed.reasoning.concat(), "\nParis it is.\n");
        let content = streamed.content.concat();
        assert!(
            content.contains("<tool_call>") && content.contains(r#""city": "Paris""#),
            "{content:?}"
        );
        assert!(streamed.names.is_empty() && streamed.arguments.is_empty());
        assert_eq!(streamed.finish_reason.as_deref(), Some("stop"));
    }

    /// Two choices of one request, their chunks interleaved, under a named
    /// function: each choice has its own reader, so each announces the call's
    /// name with its first delta and keeps its own arguments.
    #[test]
    fn each_choice_of_a_named_function_request_announces_its_own_call() {
        let streamed = stream_choices_through_converter(
            Some(NAMED),
            &[
                (0, r#"{"city":"#),
                (1, r#"{"city":"#),
                (0, r#""Paris"}"#),
                (1, r#""Rome"}"#),
            ],
        );
        for (index, city) in [(0, "Paris"), (1, "Rome")] {
            let choice = &streamed[&index];
            assert_eq!(choice.names, ["get_weather"], "choice {index}");
            assert_eq!(
                serde_json::from_str::<Value>(&choice.arguments).unwrap(),
                json!({"city": city}),
                "choice {index}"
            );
            assert_eq!(
                choice.finish_reason.as_deref(),
                Some("tool_calls"),
                "choice {index}"
            );
        }
    }

    /// Two choices under `required`, read by the `json` table: each choice's
    /// call carries its own arguments and no content, the other choice's text
    /// never enters its reader.
    #[test]
    fn each_choice_of_a_required_request_keeps_its_own_arguments() {
        let streamed = stream_choices_through_converter(
            Some(r#""required""#),
            &[
                (0, r#"[{"name":"get_weather","parameters":{"city":"Pa"#),
                (1, r#"[{"name":"get_weather","parameters":{"city":"Ro"#),
                (0, r#"ris"}}]"#),
                (1, r#"me"}}]"#),
            ],
        );
        for (index, city) in [(0, "Paris"), (1, "Rome")] {
            let choice = &streamed[&index];
            assert!(
                choice.content.is_empty(),
                "choice {index}: {:?}",
                choice.content
            );
            assert_eq!(choice.names, ["get_weather"], "choice {index}");
            assert_eq!(
                serde_json::from_str::<Value>(&choice.arguments).unwrap(),
                json!({"city": city}),
                "choice {index}"
            );
            assert_eq!(
                choice.finish_reason.as_deref(),
                Some("tool_calls"),
                "choice {index}"
            );
        }
    }
}
