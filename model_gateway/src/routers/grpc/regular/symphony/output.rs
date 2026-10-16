//! The output grammar for a whole assistant turn, for a model read through the table its response
//! template compiles into ([`super::TEMPLATE_PARSER`]) where that table's calls are addressed to a tool
//! ([`Format::output_spelling`]).
//!
//! A model whose workers all start a request's grammar at its first token
//! (`constrained_decoding_start: first_token`) has its turns constrained whole, from that token
//! ([`OutputSpelling`], [`output_constraint`]): the thought, then the answer (the request's
//! `response_format`, `regex` or `ebnf` in the engine's syntax, or free text) or calls to the tools
//! the choice lets it call, each held to its schema, and under `auto` calls to a tool the request
//! does not declare. Every turn takes it when the workers ask for that policy, the ones that ask
//! for nothing included, except on an engine that applies no grammar (`grammar_backend: none`), where
//! only a request that asks for a constraint gets one ([`Engines`]). A request is refused only
//! where it asks for what the grammar cannot keep: a tool or parameter name a call cannot carry, a
//! `strict` tool whose schema the calls would be held to more loosely, or, under `auto`, a strict
//! tool or `parallel_tool_calls: false` with the request's own `regex` or `ebnf`. Where the prompt
//! leaves the model somewhere no output grammar starts, inside an answer it opened with a format or
//! a forced call asked for, or where the workers start a grammar after their reasoning, the turn
//! takes no grammar of the table's, and the request's own format goes to the engine as it is.
//! Workers that do not say where they start one keep the tags of [`super::constraint`].

use std::sync::Arc;

use axum::response::Response;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, Tool, ToolChoice, ToolChoiceValue},
    messages::{CreateMessageRequest, ToolChoice as MessagesToolChoice},
    worker::ConstrainedDecoding,
};
use serde_json::{json, Value};
use symphony::{Format, Output, OutputAnswer, OutputChoice, OutputPosition, OutputSpelling};
use tool_parser::{GrammarStart, ToolConstraint};

use super::{names_template, reads, template::Templates};
use crate::{
    config::ParserChoice,
    routers::{error, grpc::utils::Names},
    worker::Worker,
};

/// Where the engines of a model's workers start a request's grammar, and whether they apply one,
/// from the workers' labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Engines {
    /// [`GrammarStart::FirstToken`] or [`GrammarStart::AfterReasoning`] when every worker of the
    /// model says so, [`GrammarStart::Unknown`] otherwise.
    pub start: GrammarStart,
    /// No worker says its grammar backend is `none` (a default under which an engine refuses a
    /// constrained request). A worker that does not say is taken to apply grammars.
    pub grammars: bool,
}

/// The worker label that names its engine's grammar backend.
pub(crate) const GRAMMAR_BACKEND_LABEL: &str = "grammar_backend";

impl Engines {
    pub(crate) fn of(workers: &[Arc<dyn Worker>]) -> Self {
        let label = |key: &str| -> Vec<Option<String>> {
            workers
                .iter()
                .map(|worker| worker.metadata().spec.labels.get(key).cloned())
                .collect()
        };
        let starts: Vec<GrammarStart> = label(GrammarStart::LABEL)
            .iter()
            .map(|start| GrammarStart::from_label(start.as_deref()))
            .collect();
        let every = |start: GrammarStart| !starts.is_empty() && starts.iter().all(|s| *s == start);
        let start = [GrammarStart::FirstToken, GrammarStart::AfterReasoning]
            .into_iter()
            .find(|&start| every(start))
            .unwrap_or(GrammarStart::Unknown);
        let grammars = !label(GRAMMAR_BACKEND_LABEL)
            .iter()
            .any(|backend| backend.as_deref() == Some("none"));
        Self { start, grammars }
    }
}

/// Which turns an output grammar constrains: none, every one, or only those whose request asks for
/// a constraint ([`Asked::requested`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputPolicy {
    Off,
    Every,
    Requested,
}

/// The output policy shared by every eligible retry target.
///
/// An unset or mixed policy keeps the existing request path. Debug and test builds accept the
/// `SMG_TEST_OUTPUT_POLICY` harness override.
pub(crate) fn output_policy_for(workers: &[Arc<dyn Worker>]) -> OutputPolicy {
    #[cfg(debug_assertions)]
    match std::env::var("SMG_TEST_OUTPUT_POLICY").ok().as_deref() {
        Some("every") => return OutputPolicy::Every,
        Some("requested") => return OutputPolicy::Requested,
        Some(_) => return OutputPolicy::Off,
        None => {}
    }

    let mut policies = workers.iter().map(|worker| {
        let spec = worker.metadata().spec.as_ref();
        (spec.constrained_decoding.is_some()
            || spec.labels.contains_key(ConstrainedDecoding::LABEL))
        .then(|| ConstrainedDecoding::of(spec))
    });
    let Some(Some(first)) = policies.next() else {
        return OutputPolicy::Off;
    };
    if !policies.all(|policy| policy == Some(first)) {
        return OutputPolicy::Off;
    }
    match first {
        ConstrainedDecoding::Always => OutputPolicy::Every,
        ConstrainedDecoding::Requested => OutputPolicy::Requested,
    }
}

/// What a request asks of an output grammar, in the terms of every API.
pub(crate) struct Asked<'a> {
    /// Every tool the request declares.
    pub declared: &'a [Tool],
    /// The tools its choice lets the model call (none under `none`).
    pub callable: Vec<Tool>,
    pub choice: OutputChoice,
    pub parallel: bool,
    pub answer: Option<OutputAnswer<'a>>,
    /// A call is forced: `required` or a named function, with a tool declared. A forced call
    /// writes no answer, so it leaves out the answer's format.
    pub forced: bool,
    /// The request carries its own grammar: a `regex` or an `ebnf`.
    pub own_grammar: bool,
    /// The request asks for a constraint: a forced call, `allowed_tools`, `none` with tools, a
    /// `strict: true` tool under `auto`, or an answer format the turn composes.
    pub requested: bool,
}

/// The tools a Chat or Responses request lets the model call under its choice, and how.
fn chat_choice(tool_choice: Option<&ToolChoice>, declared: &[Tool]) -> (Vec<Tool>, OutputChoice) {
    let narrowed = || {
        tool_choice
            .and_then(|choice| choice.narrow_tools(declared))
            .unwrap_or_default()
    };
    match tool_choice {
        Some(ToolChoice::Value(ToolChoiceValue::None)) => (Vec::new(), OutputChoice::Auto),
        None | Some(ToolChoice::Value(ToolChoiceValue::Auto)) => {
            (declared.to_vec(), OutputChoice::Auto)
        }
        Some(ToolChoice::Value(ToolChoiceValue::Required)) => {
            (declared.to_vec(), OutputChoice::Required)
        }
        Some(ToolChoice::Function { .. }) => (narrowed(), OutputChoice::One),
        Some(ToolChoice::AllowedTools { mode, .. }) if mode == "required" => {
            (narrowed(), OutputChoice::Required)
        }
        Some(ToolChoice::AllowedTools { .. }) => (narrowed(), OutputChoice::Auto),
    }
}

/// Whether a choice forces a call to one of `declared`: `required`, or a named function.
fn forced(tool_choice: Option<&ToolChoice>, declared: &[Tool]) -> bool {
    !declared.is_empty()
        && matches!(
            tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::Required) | ToolChoice::Function { .. })
        )
}

/// Whether the request asks for a constraint by its tools and choice: `none` with tools, a named,
/// required or allowed-tools choice, or `auto` with a `strict: true` tool.
fn tools_ask(tool_choice: Option<&ToolChoice>, declared: &[Tool]) -> bool {
    !declared.is_empty()
        && match tool_choice {
            None | Some(ToolChoice::Value(ToolChoiceValue::Auto)) => declared
                .iter()
                .any(|tool| tool.function.strict == Some(true)),
            Some(_) => true,
        }
}

/// What a Chat (or Responses) request asks of an output grammar. The protocol layer refuses a request
/// with more than one of `response_format`, `regex` and `ebnf`.
pub(crate) fn chat_asked<'a>(
    request: &'a ChatCompletionRequest,
    declared: &'a [Tool],
) -> Asked<'a> {
    let (callable, choice) = chat_choice(request.tool_choice.as_ref(), declared);
    let answer = match &request.response_format {
        Some(ResponseFormat::JsonObject) => Some(OutputAnswer::Json(&JSON_OBJECT)),
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            Some(OutputAnswer::Json(&json_schema.schema))
        }
        Some(ResponseFormat::Text) | None => None,
    }
    .or_else(|| request.regex.as_deref().map(OutputAnswer::Regex))
    .or_else(|| request.ebnf.as_deref().map(OutputAnswer::Ebnf));
    let forced = forced(request.tool_choice.as_ref(), declared);
    let composed = answer.filter(|answer| !forced && answer.composes());
    Asked {
        declared,
        callable,
        choice,
        parallel: request.parallel_tool_calls != Some(false),
        answer,
        forced,
        own_grammar: request.regex.is_some() || request.ebnf.is_some(),
        requested: tools_ask(request.tool_choice.as_ref(), declared) || composed.is_some(),
    }
}

/// The schema of a `json_object` answer.
static JSON_OBJECT: std::sync::LazyLock<Value> =
    std::sync::LazyLock::new(|| json!({"type": "object"}));

/// What a Messages request asks of an output grammar: no answer format, and its choice as a Chat
/// one (`tool_choice`), with `disable_parallel_tool_use` for parallel calls.
pub(crate) fn messages_asked<'a>(
    request: &CreateMessageRequest,
    declared: &'a [Tool],
    tool_choice: Option<&ToolChoice>,
) -> Asked<'a> {
    let (callable, choice) = chat_choice(tool_choice, declared);
    let parallel = !matches!(
        request.tool_choice,
        Some(
            MessagesToolChoice::Auto {
                disable_parallel_tool_use: Some(true)
            } | MessagesToolChoice::Any {
                disable_parallel_tool_use: Some(true)
            } | MessagesToolChoice::Tool {
                disable_parallel_tool_use: Some(true),
                ..
            }
        )
    );
    Asked {
        declared,
        forced: forced(tool_choice, declared),
        callable,
        choice,
        parallel,
        answer: None,
        own_grammar: false,
        requested: tools_ask(tool_choice, declared),
    }
}

/// The output grammar `asked` gets from a model whose table spells turns `spelling` (`format` its
/// table), served by engines that start grammars at the first token (`grammars` when they apply
/// one), under `policy`; `prompt` is the rendered prompt and `continued` whether it continues the
/// final assistant message.
///
/// - A turn the policy leaves alone, or one that asks for nothing on an engine that applies no
///   grammar, takes none.
/// - Refused: a declared tool or parameter whose name a call cannot carry, whatever the choice;
///   under `auto`, a strict tool or `parallel_tool_calls: false` with the request's own `regex` or
///   `ebnf`, which no grammar of the calls holds; and a callable `strict: true` tool whose calls
///   the grammar would hold to its schema more loosely ([`OutputSpelling::unenforced`]).
/// - An `ebnf` in Lark's syntax does not compose: the answer is free text.
/// - Where the prompt leaves the model no output grammar starts (past text no field holds), and
///   inside an answer it opened when the request asks for a forced call or an answer format, the
///   turn takes none: the request's own format applies as it is. Inside an answer otherwise the
///   rest of a free one is held, under [`OutputPolicy::Every`] only; inside a call its rest is free
///   and the calls after it are held.
/// - A grammar over the parameter counts or [`OutputSpelling::MAX_TAG_BYTES`] is left out; its
///   length is worked out before it is built ([`OutputSpelling::payload_len`]), so an oversized one
///   is never built.
#[expect(
    clippy::too_many_arguments,
    reason = "the request's ask, the table's two views, its engines and the prompt"
)]
pub(crate) fn output_constraint(
    model: &str,
    spelling: &OutputSpelling,
    format: &Format,
    asked: &Asked<'_>,
    grammars: bool,
    policy: OutputPolicy,
    prompt: &str,
    continued: bool,
) -> Result<Option<ToolConstraint>, Response> {
    let wanted = match policy {
        OutputPolicy::Off => false,
        OutputPolicy::Every => true,
        OutputPolicy::Requested => asked.requested,
    };
    if !wanted || !(grammars || asked.requested) {
        return Ok(None);
    }
    spelling
        .check_names(asked.declared)
        .map_err(invalid_tool_configuration)?;
    let strict = asked
        .callable
        .iter()
        .any(|tool| tool.function.strict == Some(true));
    if asked.own_grammar
        && asked.choice == OutputChoice::Auto
        && !asked.callable.is_empty()
        && (strict || !asked.parallel)
    {
        return Err(error::bad_request(
            "invalid_request_parameters",
            format!(
                "Model '{model}' cannot hold a strict tool's calls or parallel_tool_calls: false \
                 under tool_choice auto together with the request's own regex or ebnf"
            ),
        ));
    }
    let answer = asked
        .answer
        .filter(|answer| !asked.forced && answer.composes());
    let position = format.output_position(prompt, continued);
    if position == OutputPosition::InAnswer && (asked.forced || policy != OutputPolicy::Every) {
        return Ok(None);
    }
    let output = Output {
        position,
        declared: asked.declared,
        callable: &asked.callable,
        choice: asked.choice,
        parallel: asked.parallel,
        answer,
    };
    if let Some(why) = spelling.unenforced(&output) {
        return Err(invalid_tool_configuration(why));
    }
    if OutputSpelling::check_parameter_counts(&asked.callable).is_err() {
        return Ok(None);
    }
    // The tag's length comes before the tag: one over the bound is never built.
    match spelling.payload_len(&output) {
        Some(len) if len <= OutputSpelling::MAX_TAG_BYTES => {}
        _ => return Ok(None),
    }
    Ok(spelling
        .grammar(&output)
        .map(|grammar| ToolConstraint::StructuralTag(grammar.payload_text())))
}

/// The refusal of a tool the output grammar cannot hold, `why`.
fn invalid_tool_configuration(why: String) -> Response {
    error::bad_request(
        "invalid_tool_configuration",
        format!("Invalid tool configuration: {why}"),
    )
}

/// The table a request to `model` is read with, when it is the response template's and its calls
/// spell a turn: the parser names say [`super::TEMPLATE_PARSER`] under Symphony, and the model
/// has a table. Anything else leaves the request to [`super::constraint`], which also refuses a
/// model with no table.
fn output_table(
    parser: ParserChoice,
    names: &Names,
    model: &str,
    templates: &Templates,
) -> Option<(Format, OutputSpelling)> {
    if !names_template(names) {
        return None;
    }
    let format = reads(parser, names, model, templates).ok()??;
    let spelling = format.output_spelling().ok()?;
    Some((format, spelling))
}

/// The constraint for a Chat request whose response a table that spells turns reads
/// ([`output_table`]), or `None` when no such table reads it: where the model's `engines` start a
/// grammar at the first token, the output grammar ([`output_constraint`]); where they start it after the
/// reasoning, none; where they do not say, `None`, and [`super::constraint`] decides. `prompt` is
/// the rendered prompt.
pub(crate) fn for_chat(
    parser: ParserChoice,
    names: &Names,
    request: &ChatCompletionRequest,
    workers: &[Arc<dyn Worker>],
    prompt: &str,
    continues: bool,
    templates: &Templates,
) -> Option<Result<Option<ToolConstraint>, Response>> {
    let policy = output_policy_for(workers);
    if policy == OutputPolicy::Off {
        return None;
    }
    let (format, spelling) = output_table(parser, names, &request.model, templates)?;
    let declared: Vec<Tool> = request.effective_tools().cloned().collect();
    let asked = chat_asked(request, &declared);
    for_asked(
        &request.model,
        &format,
        &spelling,
        &asked,
        Engines::of(workers),
        policy,
        prompt,
        continues,
    )
}

/// [`for_chat`] for a Messages request: `tool_choice` is its choice as a Chat one, and
/// `continues` says whether the prompt continues the assistant's message.
#[expect(
    clippy::too_many_arguments,
    reason = "the parser's choice, the request, its engines and the prompt"
)]
pub(crate) fn for_messages(
    parser: ParserChoice,
    names: &Names,
    request: &CreateMessageRequest,
    tool_choice: Option<&ToolChoice>,
    workers: &[Arc<dyn Worker>],
    prompt: &str,
    continues: bool,
    templates: &Templates,
) -> Option<Result<Option<ToolConstraint>, Response>> {
    let policy = output_policy_for(workers);
    if policy == OutputPolicy::Off {
        return None;
    }
    let (format, spelling) = output_table(parser, names, &request.model, templates)?;
    let declared = request.chat_tools();
    let asked = messages_asked(request, &declared, tool_choice);
    for_asked(
        &request.model,
        &format,
        &spelling,
        &asked,
        Engines::of(workers),
        policy,
        prompt,
        continues,
    )
}

/// Apply one already normalized API request to the output grammar.
#[expect(
    clippy::too_many_arguments,
    reason = "the request's ask, the table's views, the workers and the prompt"
)]
fn for_asked(
    model: &str,
    format: &Format,
    spelling: &OutputSpelling,
    asked: &Asked<'_>,
    engines: Engines,
    policy: OutputPolicy,
    prompt: &str,
    continues: bool,
) -> Option<Result<Option<ToolConstraint>, Response>> {
    match engines.start {
        GrammarStart::FirstToken => Some(output_constraint(
            model,
            spelling,
            format,
            asked,
            engines.grammars,
            policy,
            prompt,
            continues,
        )),
        GrammarStart::AfterReasoning => Some(Ok(None)),
        GrammarStart::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use smg_response_template::{load_response_template, to_symphony};
    use OutputPolicy::{Every, Off, Requested};

    use super::*;

    const OPENER: &str = "<|head|>assistant";
    const FIRST_TOKEN: Engines = Engines {
        start: GrammarStart::FirstToken,
        grammars: true,
    };
    type Outcome = Result<Option<Value>, String>;

    fn engines(start: GrammarStart, grammars: bool) -> Engines {
        Engines { start, grammars }
    }

    fn neutral_template() -> Value {
        let source = r#"{
              "start_anchor_pattern": "<\\|head\\|>assistant",
              "fields": {
                "thinking": {"open": "<|via|>notes<|body|>", "close": "<|stop|><|head|>assistant", "content": "text"},
                "content": {"open": "<|via|>reply<|body|>", "close": ["<|done|>", "<|stop|>", "<eos>"], "content": "text"},
                "tool_calls": {
                  "open_pattern": " to=(?P<name>[^\\s<]+)<\\|via\\|>notes <\\|as\\|>xml<\\|body\\|>",
                  "close": ["<|stop|><|head|>assistant", "<eos>"], "repeats": true, "content": "xml-inline",
                  "content_args": {"tag_pattern": "<arg name=\"(?P<key>[^\"]+)\">\\n(?P<value>.*?)\\n(?:</arg>|(?=<arg name=\"))", "value_parser": {"name": "text", "args": {"strip": false}}},
                  "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}}
                }
              }
            }"#;
        serde_json::from_str(source).unwrap()
    }

    fn output_table() -> (Format, OutputSpelling) {
        let template = load_response_template(&neutral_template()).unwrap();
        let stops = ["<|done|>", "<eos>"];
        let format = to_symphony::compile(&template, &stops).unwrap().format;
        let spelling = format.output_spelling().unwrap();
        (format, spelling)
    }

    fn chat(extra: Value) -> ChatCompletionRequest {
        let mut request = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        let request_object = request.as_object_mut().unwrap();
        request_object.extend(extra.as_object().unwrap().clone());
        serde_json::from_value(request).unwrap()
    }

    fn output_of(
        mut extra: Value,
        engines: Engines,
        policy: OutputPolicy,
        continued: Option<&str>,
    ) -> Option<Outcome> {
        let (format, spelling) = output_table();
        if continued.is_some() {
            extra["continue_final_message"] = json!(true);
            extra["messages"] = json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "partial"}
            ]);
        }
        let request = chat(extra);
        let declared: Vec<Tool> = request.effective_tools().cloned().collect();
        let asked = chat_asked(&request, &declared);
        let prompt = format!("user{OPENER}{}", continued.unwrap_or(""));
        Some(
            for_asked(
                &request.model,
                &format,
                &spelling,
                &asked,
                engines,
                policy,
                &prompt,
                request.continues_final_assistant(),
            )?
            .map_err(|response| response.status().to_string())
            .map(|constraint| {
                constraint.map(|constraint| {
                    let payload = constraint.to_tuple().1;
                    serde_json::from_str::<Value>(&payload).unwrap()["format"].clone()
                })
            }),
        )
    }

    fn decided(extra: Value, policy: OutputPolicy, continued: Option<&str>) -> Outcome {
        output_of(extra, FIRST_TOKEN, policy, continued).expect("the turn decides")
    }

    fn every(extra: Value, continued: Option<&str>) -> Outcome {
        decided(extra, Every, continued)
    }

    fn whole(extra: Value, continued: Option<&str>) -> Value {
        every(extra, continued).unwrap().unwrap()
    }

    fn weather() -> Value {
        json!([{"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
    }

    fn answer(format: &Value) -> &Value {
        &format["elements"][1]
    }

    #[test]
    fn every_policy_composes_the_whole_turn() {
        let free = whole(json!({}), None);
        assert_eq!(
            answer(&free)["elements"][1],
            json!({"type": "any_text", "excludes": ["<|done|>", "<|stop|>"]})
        );
        assert_eq!(
            answer(&whole(json!({"tools": weather()}), None))["type"],
            "or"
        );
        let required = whole(json!({"tools": weather(), "tool_choice": "required"}), None);
        assert_eq!(answer(&required)["type"], "sequence");
        let json_answer = whole(json!({"response_format": {"type": "json_object"}}), None);
        assert_eq!(
            answer(&json_answer)["elements"][1],
            json!({"type": "json_schema", "json_schema": {"type": "object"}})
        );
        let regex = whole(json!({"regex": "(yes|no)"}), None);
        assert_eq!(
            answer(&regex)["elements"][1],
            json!({"type": "regex", "pattern": "(yes|no)"})
        );
        assert_eq!(every(json!({"ebnf": "start: \"y\""}), None), Ok(Some(free)));
        assert_eq!(
            every(
                json!({"tools": weather(), "tool_choice": "required",
                "ebnf": "start: \"y\""}),
                None
            ),
            Ok(Some(required))
        );
    }

    #[test]
    fn chat_normalizes_named_and_allowed_narrowing() {
        let tools = json!([
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}
        ]);
        let normalized = |tool_choice: Value| {
            let request = chat(json!({"tools": tools, "tool_choice": tool_choice}));
            let declared: Vec<Tool> = request.effective_tools().cloned().collect();
            let asked = chat_asked(&request, &declared);
            let names: Vec<String> = asked
                .callable
                .iter()
                .map(|tool| tool.function.name.clone())
                .collect();
            (names, asked.choice, asked.forced)
        };
        let named = normalized(json!({"type": "function", "function": {"name": "get_weather"}}));
        assert_eq!(named.0, ["get_weather"]);
        assert_eq!((named.1, named.2), (OutputChoice::One, true));
        let allowed = normalized(json!({"type": "allowed_tools", "mode": "auto",
            "tools": [{"type": "function", "name": "get_time"}]}));
        assert_eq!(allowed.0, ["get_time"]);
        assert_eq!((allowed.1, allowed.2), (OutputChoice::Auto, false));
    }

    #[test]
    fn the_prompt_selects_each_output_position() {
        let thinking = whole(json!({}), Some("<|via|>notes<|body|>Let me"));
        assert_eq!(thinking["elements"][0]["begin"], "");
        let after = whole(
            json!({}),
            Some("<|via|>notes<|body|>Plan.<|stop|><|head|>assistant"),
        );
        assert_eq!(after["elements"][0]["value"], "<|via|>reply<|body|>");

        let answering = Some("<|via|>reply<|body|>The answer");
        let rest = whole(json!({}), answering);
        assert_eq!(rest["elements"][0]["type"], "any_text");
        for extra in [
            json!({"tools": weather(), "tool_choice": "required"}),
            json!({"response_format": {"type": "json_object"}}),
        ] {
            assert_eq!(every(extra, answering), Ok(None));
        }
        let allowed = json!({"tools": weather(), "tool_choice": {"type": "allowed_tools",
            "mode": "required", "tools": [{"type": "function", "name": "get_weather"}]}});
        assert_eq!(every(allowed.clone(), answering), Ok(Some(rest)));
        assert_eq!(decided(allowed, Requested, answering), Ok(None));
        assert_eq!(every(json!({"tools": weather()}), Some("Hello")), Ok(None));

        let calling = " to=get_weather<|via|>notes <|as|>xml<|body|>\
                       <arg name=\"city\">\nPa";
        assert_eq!(
            whole(json!({"tools": weather()}), Some(calling))["elements"][0]["type"],
            "any_text"
        );
    }

    #[test]
    fn refusals_have_focused_counterexamples() {
        let refused = |extra: Value| every(extra, None).is_err();
        let served = |extra: Value| every(extra, None).unwrap().is_some();
        let named = |name: &str| {
            json!([{"type": "function", "function": {"name": name, "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}}}}])
        };
        for choice in [json!("auto"), json!("none"), json!("required")] {
            assert!(refused(
                json!({"tools": named("get weather"), "tool_choice": choice})
            ));
        }
        let key = json!([{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"say \"hi\"": {"type": "string"}}}}}]);
        assert!(refused(json!({"tools": key})));

        let tuple = |strict: bool| {
            json!([{"type": "function", "function": {"name": "f", "strict": strict,
                "parameters": {"type": "object", "properties": {"p": {"type": "tuple"}},
                    "required": ["p"]}}}])
        };
        assert!(refused(json!({"tools": tuple(true)})));
        assert!(served(json!({"tools": tuple(false)})));
        assert!(served(json!({"tools": tuple(true), "tool_choice": "none"})));

        let strict = json!([{"type": "function", "function": {"name": "f", "strict": true,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
        for extra in [
            json!({"tools": strict, "regex": "(a|b)"}),
            json!({"tools": strict, "ebnf": "start: \"y\""}),
            json!({"tools": weather(), "parallel_tool_calls": false, "regex": "(a|b)"}),
        ] {
            assert!(refused(extra));
        }
        for extra in [
            json!({"tools": strict}),
            json!({"tools": strict, "response_format": {"type": "json_object"}}),
            json!({"tools": strict, "tool_choice": "required", "regex": "(a|b)"}),
            json!({"tools": weather(), "regex": "(a|b)"}),
            json!({"tools": weather(), "parallel_tool_calls": false}),
            json!({"regex": "(a|b)", "parallel_tool_calls": false}),
        ] {
            assert!(served(extra.clone()), "{extra}");
        }
    }

    #[test]
    fn policies_and_engine_capabilities_gate_the_grammar() {
        assert_eq!(decided(json!({}), Requested, None), Ok(None));
        let formatted = json!({"response_format": {"type": "json_object"}});
        assert_eq!(decided(formatted.clone(), Off, None), Ok(None));
        for extra in [
            json!({"tools": weather(), "tool_choice": "required"}),
            json!({"tools": weather(), "tool_choice": "none"}),
            json!({"tools": weather(), "tool_choice": {"type": "allowed_tools",
                "mode": "auto", "tools": [{"type": "function", "name": "get_weather"}]}}),
            formatted,
            json!({"regex": "(a|b)"}),
        ] {
            assert!(decided(extra, Requested, None).unwrap().is_some());
        }

        let no_backend = engines(GrammarStart::FirstToken, false);
        assert_eq!(
            output_of(json!({"tools": weather()}), no_backend, Every, None),
            Some(Ok(None))
        );
        let strict = json!([{"type": "function", "function": {"name": "f", "strict": true,
            "parameters": {"type": "object", "properties": {}}}}]);
        for extra in [
            json!({"tools": strict}),
            json!({"tools": weather(), "tool_choice": "required"}),
        ] {
            assert!(matches!(
                output_of(extra, no_backend, Every, None),
                Some(Ok(Some(_)))
            ));
        }
        let after = engines(GrammarStart::AfterReasoning, true);
        assert_eq!(
            output_of(
                json!({"tools": weather(), "tool_choice": "required"}),
                after,
                Every,
                None
            ),
            Some(Ok(None))
        );
        let unknown = engines(GrammarStart::Unknown, true);
        assert!(output_of(json!({"tools": weather()}), unknown, Every, None).is_none());
    }

    #[test]
    fn count_and_size_limits_leave_the_grammar_out() {
        let properties: serde_json::Map<String, Value> = (0..=OutputSpelling::MAX_TOOL_PARAMETERS)
            .map(|i| (format!("p{i}"), json!({"type": "string"})))
            .collect();
        let wide = json!([{"type": "function", "function": {"name": "wide",
            "parameters": {"type": "object", "properties": properties}}}]);
        assert_eq!(every(json!({"tools": wide}), None), Ok(None));

        let huge = "x".repeat(OutputSpelling::MAX_TAG_BYTES);
        let large = json!([{"type": "function", "function": {"name": "large",
        "parameters": {"type": "object", "properties": {
            "p": {"type": "string", "enum": [huge]}
        }}}}]);
        assert_eq!(every(json!({"tools": large}), None), Ok(None));
    }

    #[test]
    fn worker_labels_aggregate_engine_and_output_policy() {
        use crate::worker::{BasicWorkerBuilder, ModelCard, WorkerRegistry, WorkerType};

        let registry = |labels: &[&[(&str, &str)]]| {
            let registry = WorkerRegistry::new();
            for (i, labels) in labels.iter().enumerate() {
                let mut builder = BasicWorkerBuilder::new(format!("http://w{i}:50051"))
                    .model(ModelCard::new("m"))
                    .worker_type(WorkerType::Regular);
                for (key, value) in *labels {
                    builder = builder.label(*key, *value);
                }
                registry.register(Arc::new(builder.build())).unwrap();
            }
            let workers = registry.get_by_model("m");
            (Engines::of(&workers), output_policy_for(&workers))
        };
        let first = (GrammarStart::LABEL, "first_token");
        let requested = (ConstrainedDecoding::LABEL, "requested");
        let always = (ConstrainedDecoding::LABEL, "always");
        let none = (GRAMMAR_BACKEND_LABEL, "none");

        assert_eq!(
            registry(&[&[first, always], &[first, always]]),
            (FIRST_TOKEN, Every)
        );
        assert_eq!(registry(&[&[first, requested]]).1, Requested);
        assert_eq!(registry(&[&[first]]).1, Off);
        assert_eq!(registry(&[&[first, requested], &[first, always]]).1, Off);
        assert_eq!(registry(&[&[first], &[]]).0.start, GrammarStart::Unknown);
        assert!(!registry(&[&[first, none]]).0.grammars);
    }

    #[test]
    fn messages_choices_shape_the_output() {
        use OutputChoice::{Auto, One, Required};

        let tools = json!([{"name": "get_weather", "input_schema": {
            "type": "object", "properties": {"city": {"type": "string"}}}}]);
        let asked = |choice: Value| {
            let request: CreateMessageRequest = serde_json::from_value(json!({
                "model": "m", "max_tokens": 16, "tools": tools,
                "messages": [{"role": "user", "content": "hi"}], "tool_choice": choice
            }))
            .unwrap();
            let choice = request.tool_choice.as_ref().map(ToolChoice::from);
            let declared = request.chat_tools();
            let asked = messages_asked(&request, &declared, choice.as_ref());
            (
                asked.choice,
                asked.parallel,
                asked.forced,
                asked.callable.len(),
            )
        };
        for (choice, expected) in [
            (json!({"type": "auto"}), (Auto, true, false, 1)),
            (json!({"type": "any"}), (Required, true, true, 1)),
            (
                json!({"type": "any", "disable_parallel_tool_use": true}),
                (Required, false, true, 1),
            ),
            (
                json!({"type": "tool", "name": "get_weather"}),
                (One, true, true, 1),
            ),
            (json!({"type": "none"}), (Auto, true, false, 0)),
        ] {
            assert_eq!(asked(choice), expected);
        }
    }
}
