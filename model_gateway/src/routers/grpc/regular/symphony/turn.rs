//! The grammar of a whole turn, for a model read through the table its response template compiles
//! into ([`super::TEMPLATE_PARSER`]) where that table's calls are addressed to a tool
//! ([`Format::turn_spelling`]).
//!
//! A model whose workers all start a request's grammar at its first token
//! (`structured_outputs_start: first_token`) has its turns constrained whole, from that token
//! ([`TurnSpelling`], [`turn_constraint`]): the thought, then the answer (the request's
//! `response_format`, `regex` or `ebnf`, or free text) or calls to the tools the choice lets it
//! call, each held to its schema, and under `auto` calls to a tool the request does not declare.
//! Every turn takes it, the ones that ask for nothing included ([`TURN_POLICY`]), except on an
//! engine that applies no grammar (`grammar_backend: none`), where only a request that asks for a
//! constraint gets one ([`Engines`]). Where the prompt leaves the model somewhere no turn grammar
//! starts, or where the workers start a grammar after their reasoning, a request that asks for a
//! forced call or an answer format is refused, as the table's tags refuse what they cannot hold,
//! and any other goes unconstrained. Workers that do not say where they start one keep the tags
//! of [`super::constraint`].

use axum::response::Response;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, Tool, ToolChoice, ToolChoiceValue},
    messages::{CreateMessageRequest, ToolChoice as MessagesToolChoice},
};
use serde_json::{json, Value};
use symphony::{Format, Turn, TurnAnswer, TurnChoice, TurnPosition, TurnSpelling};
use tool_parser::{GrammarStart, ToolConstraint};

use super::{forces_call, names_template, reads, template::Templates};
use crate::{
    config::ParserChoice,
    routers::{
        error,
        grpc::utils::{continues_final_assistant, message_utils, Names},
    },
    worker::WorkerRegistry,
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
    pub(crate) fn of(registry: &WorkerRegistry, model: &str) -> Self {
        let workers = registry.get_by_model(model);
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

/// Which turns a turn grammar constrains: every one, or only those whose request asks for a
/// constraint ([`Asked::requested`]). The choice is this build's ([`TURN_POLICY`]); nothing at run
/// time switches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TurnPolicy {
    Every,
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the other policy a build may choose, tested here")
    )]
    Requested,
}

/// The turns this gateway constrains: every one.
pub(crate) const TURN_POLICY: TurnPolicy = TurnPolicy::Every;

/// What a request asks of a turn grammar, in the terms of every API.
pub(crate) struct Asked<'a> {
    /// Every tool the request declares.
    pub declared: &'a [Tool],
    /// The tools its choice lets the model call (none under `none`).
    pub callable: Vec<Tool>,
    pub choice: TurnChoice,
    pub parallel: bool,
    pub answer: Option<TurnAnswer<'a>>,
    /// A call is forced: `required`, a named function, or `allowed_tools` in `required` mode, with
    /// a tool to call. A forced call writes no answer, so it leaves out the answer's format.
    pub forced: bool,
    /// The request asks for a constraint: a forced call, `allowed_tools`, `none` with tools, a
    /// `strict: true` tool under `auto`, or an answer format the turn composes.
    pub requested: bool,
}

/// The tools a Chat or Responses request lets the model call under its choice, and how.
fn chat_choice(tool_choice: Option<&ToolChoice>, declared: &[Tool]) -> (Vec<Tool>, TurnChoice) {
    let narrowed = || {
        tool_choice
            .and_then(|choice| choice.narrow_tools(declared))
            .unwrap_or_default()
    };
    match tool_choice {
        Some(ToolChoice::Value(ToolChoiceValue::None)) => (Vec::new(), TurnChoice::Auto),
        None | Some(ToolChoice::Value(ToolChoiceValue::Auto)) => {
            (declared.to_vec(), TurnChoice::Auto)
        }
        Some(ToolChoice::Value(ToolChoiceValue::Required)) => {
            (declared.to_vec(), TurnChoice::Required)
        }
        Some(ToolChoice::Function { .. }) => (narrowed(), TurnChoice::One),
        Some(ToolChoice::AllowedTools { mode, .. }) if mode == "required" => {
            (narrowed(), TurnChoice::Required)
        }
        Some(ToolChoice::AllowedTools { .. }) => (narrowed(), TurnChoice::Auto),
    }
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

/// What a Chat (or Responses) request asks of a turn grammar. The protocol layer refuses a request
/// with more than one of `response_format`, `regex` and `ebnf`.
pub(crate) fn chat_asked<'a>(
    request: &'a ChatCompletionRequest,
    declared: &'a [Tool],
) -> Asked<'a> {
    let (callable, choice) = chat_choice(request.tool_choice.as_ref(), declared);
    let answer = match &request.response_format {
        Some(ResponseFormat::JsonObject) => Some(TurnAnswer::Json(&JSON_OBJECT)),
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            Some(TurnAnswer::Json(&json_schema.schema))
        }
        Some(ResponseFormat::Text) | None => None,
    }
    .or_else(|| request.regex.as_deref().map(TurnAnswer::Regex))
    .or_else(|| request.ebnf.as_deref().map(TurnAnswer::Ebnf));
    let forced = !callable.is_empty() && request.tool_choice.as_ref().is_some_and(forces_call);
    let composed = answer.filter(|answer| !forced && answer.composes());
    Asked {
        declared,
        callable,
        choice,
        parallel: request.parallel_tool_calls != Some(false),
        answer,
        forced,
        requested: tools_ask(request.tool_choice.as_ref(), declared) || composed.is_some(),
    }
}

/// The schema of a `json_object` answer.
static JSON_OBJECT: std::sync::LazyLock<Value> =
    std::sync::LazyLock::new(|| json!({"type": "object"}));

/// What a Messages request asks of a turn grammar: no answer format, and its choice as a Chat
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
        forced: !callable.is_empty() && tool_choice.is_some_and(forces_call),
        callable,
        choice,
        parallel,
        answer: None,
        requested: tools_ask(tool_choice, declared),
    }
}

/// The turn grammar `asked` gets from a model whose table spells turns `spelling` (`format` its
/// table), served by engines that start grammars at the first token (`grammars` when they apply
/// one), under `policy`; `prompt` is the rendered prompt and `continued` whether it continues the
/// final assistant message.
///
/// - An `ebnf` in Lark's syntax does not compose: refused, unless a forced call leaves the answer
///   out anyway.
/// - A turn the policy leaves alone, or one that asks for nothing on an engine that applies no
///   grammar, takes none.
/// - Where the prompt leaves the model no turn grammar starts (inside a call, past text no field
///   holds), or inside an answer it opened, a forced call or an answer format is refused, and
///   anything else goes unconstrained: inside an answer only the rest of a free one is
///   constrained, and under [`TurnPolicy::Every`] only. A continued message whose thought the
///   prompt closed is past the thought.
/// - A forced call to a tool the table cannot read back is refused ([`TurnSpelling::check_tools`]),
///   and so is a forced call or an answer format over the parameter counts, which the backend
///   request no longer carries; other requests go unconstrained there.
/// - A grammar over [`TurnSpelling::MAX_TAG_BYTES`] is refused for a forced call or an answer
///   format, and left out otherwise.
#[expect(
    clippy::too_many_arguments,
    reason = "the request's ask, the table's two views, its engines and the prompt"
)]
pub(crate) fn turn_constraint(
    model: &str,
    spelling: &TurnSpelling,
    format: &Format,
    asked: &Asked<'_>,
    grammars: bool,
    policy: TurnPolicy,
    prompt: &str,
    continued: bool,
) -> Result<Option<ToolConstraint>, Response> {
    if asked
        .answer
        .is_some_and(|answer| !asked.forced && !answer.composes())
    {
        return Err(lark_refused(model));
    }
    let answer = asked.answer.filter(|_| !asked.forced);
    let holds = asked.forced || answer.is_some();
    let wanted = match policy {
        TurnPolicy::Every => true,
        TurnPolicy::Requested => asked.requested,
    };
    if !wanted || !(grammars || asked.requested) {
        return Ok(None);
    }
    let position = format.turn_position(prompt, continued);
    let inside_answer = position == TurnPosition::InAnswer;
    if position == TurnPosition::Elsewhere || (inside_answer && holds) {
        return if holds {
            Err(cannot_continue(model))
        } else {
            Ok(None)
        };
    }
    if inside_answer && policy != TurnPolicy::Every {
        return Ok(None);
    }
    let mut checked = TurnSpelling::check_parameter_counts(&asked.callable);
    if asked.forced {
        checked = checked.and_then(|()| spelling.check_tools(&asked.callable));
    }
    if let Err(e) = checked {
        return if holds {
            Err(error::bad_request(
                "invalid_tool_configuration",
                format!("Invalid tool configuration: {e}"),
            ))
        } else {
            Ok(None)
        };
    }
    let turn = Turn {
        position,
        declared: asked.declared,
        callable: &asked.callable,
        choice: asked.choice,
        parallel: asked.parallel,
        answer,
    };
    let Some(grammar) = spelling.grammar(&turn) else {
        return if holds {
            Err(cannot_continue(model))
        } else {
            Ok(None)
        };
    };
    let payload = grammar.payload().to_string();
    let max = TurnSpelling::MAX_TAG_BYTES;
    if payload.len() > max {
        return if holds {
            Err(error::bad_request(
                "invalid_request_parameters",
                format!(
                    "Model '{model}' would constrain this request's turn with a structural tag \
                     of {} bytes, over its limit of {max} bytes",
                    payload.len()
                ),
            ))
        } else {
            Ok(None)
        };
    }
    Ok(Some(ToolConstraint::StructuralTag(payload)))
}

/// Refuses a forced call or an answer format a model's workers cannot hold: they start a
/// request's grammar after the model's reasoning, where a turn's grammar, which starts with the
/// thought, does not fit; any other request goes unconstrained.
pub(crate) fn after_reasoning(
    model: &str,
    holds: bool,
) -> Result<Option<ToolConstraint>, Response> {
    if !holds {
        return Ok(None);
    }
    Err(error::bad_request(
        "invalid_request_parameters",
        format!(
            "Model '{model}' constrains a forced tool choice or a response format from the start \
             of its turn, and its workers start a grammar only after its reasoning"
        ),
    ))
}

/// The refusal of a forced call or an answer format the prompt leaves no place for.
fn cannot_continue(model: &str) -> Response {
    error::bad_request(
        "invalid_request_parameters",
        format!(
            "Model '{model}' constrains a forced tool choice or a JSON response_format from the \
             start of a new assistant message, so it cannot continue one"
        ),
    )
}

/// The refusal of an `ebnf` the turn's grammar cannot hold (one in Lark's syntax): the engine
/// would apply it from the first token, outside the format the model's table reads.
fn lark_refused(model: &str) -> Response {
    error::bad_request(
        "unsupported_constraint",
        format!(
            "Model '{model}' writes its output in the format its response template declares, \
             and a regex or ebnf constraint would apply outside it"
        ),
    )
}

/// The table a request to `model` is read with, when it is the response template's and its calls
/// spell a turn: the parser names say [`super::TEMPLATE_PARSER`] under Symphony, and the model
/// has a table. Anything else leaves the request to [`super::constraint`], which also refuses a
/// model with no table.
fn turn_table(
    parser: ParserChoice,
    names: &Names,
    model: &str,
    templates: &Templates,
) -> Option<(Format, TurnSpelling)> {
    if !names_template(names) {
        return None;
    }
    let format = reads(parser, names, model, templates).ok()??;
    let spelling = format.turn_spelling().ok()?;
    Some((format, spelling))
}

/// The constraint for a Chat request whose response a table that spells turns reads
/// ([`turn_table`]), or `None` when no such table reads it: where the model's `engines` start a
/// grammar at the first token, the turn's ([`turn_constraint`]); where they start it after the
/// reasoning, none, a forced call and an answer format refused ([`after_reasoning`]); where they
/// do not say, `None`, and [`super::constraint`] decides. `prompt` is the rendered prompt.
pub(crate) fn for_chat(
    parser: ParserChoice,
    names: &Names,
    request: &ChatCompletionRequest,
    engines: Engines,
    prompt: &str,
    templates: &Templates,
) -> Option<Result<Option<ToolConstraint>, Response>> {
    let (format, spelling) = turn_table(parser, names, &request.model, templates)?;
    chat_turn(&format, &spelling, request, engines, prompt)
}

/// [`for_chat`] over `format`, whose calls spell a turn `spelling`.
fn chat_turn(
    format: &Format,
    spelling: &TurnSpelling,
    request: &ChatCompletionRequest,
    engines: Engines,
    prompt: &str,
) -> Option<Result<Option<ToolConstraint>, Response>> {
    let declared: Vec<Tool> = request.effective_tools().cloned().collect();
    let asked = chat_asked(request, &declared);
    match engines.start {
        GrammarStart::FirstToken => Some(turn_constraint(
            &request.model,
            spelling,
            format,
            &asked,
            engines.grammars,
            TURN_POLICY,
            prompt,
            continues_final_assistant(request),
        )),
        GrammarStart::AfterReasoning => Some(after_reasoning(
            &request.model,
            asked.forced || asked.answer.is_some(),
        )),
        GrammarStart::Unknown => None,
    }
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
    engines: Engines,
    prompt: &str,
    continues: bool,
    templates: &Templates,
) -> Option<Result<Option<ToolConstraint>, Response>> {
    let (format, spelling) = turn_table(parser, names, &request.model, templates)?;
    messages_turn(
        &format,
        &spelling,
        request,
        tool_choice,
        engines,
        prompt,
        continues,
    )
}

/// [`for_messages`] over `format`, whose calls spell a turn `spelling`.
fn messages_turn(
    format: &Format,
    spelling: &TurnSpelling,
    request: &CreateMessageRequest,
    tool_choice: Option<&ToolChoice>,
    engines: Engines,
    prompt: &str,
    continues: bool,
) -> Option<Result<Option<ToolConstraint>, Response>> {
    let declared = message_utils::extract_chat_tools(request.tools.as_deref().unwrap_or(&[]));
    let asked = messages_asked(request, &declared, tool_choice);
    match engines.start {
        GrammarStart::FirstToken => Some(turn_constraint(
            &request.model,
            spelling,
            format,
            &asked,
            engines.grammars,
            TURN_POLICY,
            prompt,
            continues,
        )),
        GrammarStart::AfterReasoning => Some(after_reasoning(&request.model, asked.forced)),
        GrammarStart::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use smg_response_template::{load_response_template, to_symphony};

    use super::*;

    /// The turn's anchor of [`neutral_template`].
    const OPENER: &str = "<|head|>assistant";

    /// A response template whose calls are addressed to a tool, with neutral markers: a thought,
    /// an answer and calls, each opened after the turn's anchor; `<eos>` ends a turn that calls.
    fn neutral_template() -> Value {
        json!({
            "start_anchor_pattern": "<\\|head\\|>assistant",
            "fields": {
                "thinking": {
                    "open": "<|via|>notes<|body|>",
                    "close": "<|stop|><|head|>assistant",
                    "content": "text"
                },
                "content": {
                    "open": "<|via|>reply<|body|>",
                    "close": ["<|done|>", "<|stop|>", "<eos>"],
                    "content": "text"
                },
                "tool_calls": {
                    "open_pattern":
                        " to=(?P<name>[^\\s<]+)<\\|via\\|>notes <\\|as\\|>xml<\\|body\\|>",
                    "close": ["<|stop|><|head|>assistant", "<eos>"],
                    "repeats": true,
                    "content": "xml-inline",
                    "content_args": {
                        "tag_pattern":
                            "<arg name=\"(?P<key>[^\"]+)\">\\n(?P<value>.*?)\\n(?:</arg>|(?=<arg name=\"))",
                        "value_parser": {"name": "text", "args": {"strip": false}}
                    },
                    "transform": {"type": "function",
                                  "function": {"name": "{name}", "arguments": "{content}"}}
                }
            }
        })
    }

    /// [`neutral_template`]'s table, `<|done|>` and `<eos>` its stops, as a tokenizer's
    /// end-of-sequence tokens are, and the texts of its turn.
    fn turn_table() -> (Format, TurnSpelling) {
        let template = load_response_template(&neutral_template()).unwrap();
        let format = to_symphony::compile(&template, &["<|done|>", "<eos>"])
            .unwrap()
            .format;
        let spelling = format.turn_spelling().unwrap();
        (format, spelling)
    }

    const FIRST_TOKEN: Engines = Engines {
        start: GrammarStart::FirstToken,
        grammars: true,
    };

    fn chat(extra: Value) -> ChatCompletionRequest {
        let mut request = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(request).unwrap()
    }

    /// The constraint a Chat request takes from [`turn_table`] on `engines`, over a prompt that
    /// starts a new turn, or continues one with `continued`; the refusal's status otherwise.
    /// `None` leaves the request to the table's tags.
    fn turn_of(
        mut extra: Value,
        engines: Engines,
        continued: Option<&str>,
    ) -> Option<Result<Option<Value>, String>> {
        let (format, spelling) = turn_table();
        if continued.is_some() {
            extra["continue_final_message"] = json!(true);
            extra["messages"] = json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "partial"}
            ]);
        }
        let request = chat(extra);
        let prompt = format!("user{OPENER}{}", continued.unwrap_or(""));
        Some(
            match chat_turn(&format, &spelling, &request, engines, &prompt)? {
                Ok(constraint) => Ok(constraint.map(|c| {
                    let (kind, value) = c.to_tuple();
                    assert_eq!(kind, "structural_tag");
                    serde_json::from_str::<Value>(&value).unwrap()["format"].clone()
                })),
                Err(refused) => Err(refused.status().to_string()),
            },
        )
    }

    /// [`turn_of`] where the turn's grammar decides.
    fn decided(
        extra: Value,
        engines: Engines,
        continued: Option<&str>,
    ) -> Result<Option<Value>, String> {
        turn_of(extra, engines, continued).expect("the turn decides")
    }

    fn weather() -> Value {
        json!([{"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
    }

    /// The constraint's answer, past its thought: the sequence of the answer's opening, its body
    /// and its end.
    fn answer_of(format: &Value) -> &Value {
        &format["elements"][1]
    }

    /// The compiled template's table spells the turn the symphony crate's own neutral table
    /// does: the thought, the answer and the calls between its markers.
    #[test]
    fn the_compiled_template_spells_a_turn() {
        let (format, spelling) = turn_table();
        assert_eq!(
            format.turn_position(&format!("user{OPENER}"), false),
            TurnPosition::Start
        );
        let tools: Vec<Tool> = serde_json::from_value(weather()).unwrap();
        assert!(spelling.check_tools(&tools).is_ok());
    }

    /// Where every worker starts a grammar at the first token, every turn takes the whole turn's
    /// grammar: the thought, then the answer (free, or the request's format), or calls.
    #[test]
    fn a_first_token_engine_takes_the_whole_turn() {
        let notools = decided(json!({}), FIRST_TOKEN, None).unwrap().unwrap();
        assert_eq!(notools["elements"][0]["begin"], "<|via|>notes<|body|>");
        assert_eq!(
            answer_of(&notools)["elements"][1],
            json!({"type": "any_text", "excludes": ["<|done|>", "<|stop|>", "<eos>"]})
        );
        let auto = decided(json!({"tools": weather()}), FIRST_TOKEN, None)
            .unwrap()
            .unwrap();
        assert_eq!(answer_of(&auto)["type"], "or", "calls or the answer");
        let required = decided(
            json!({"tools": weather(), "tool_choice": "required"}),
            FIRST_TOKEN,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(answer_of(&required)["type"], "sequence", "calls only");
        let json_answer = decided(
            json!({"response_format": {"type": "json_object"}}),
            FIRST_TOKEN,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            answer_of(&json_answer)["elements"][1],
            json!({"type": "json_schema", "json_schema": {"type": "object"},
                   "excludes": ["<|done|>", "<|stop|>", "<eos>"]})
        );
        let regex = decided(json!({"regex": "(yes|no)"}), FIRST_TOKEN, None)
            .unwrap()
            .unwrap();
        assert_eq!(
            answer_of(&regex)["elements"][1],
            json!({"type": "regex", "pattern": "(yes|no)"})
        );
        // A Lark grammar does not compose: refused, unless a forced call leaves the answer out.
        assert_eq!(
            decided(json!({"ebnf": "start: \"y\""}), FIRST_TOKEN, None).unwrap_err(),
            "400 Bad Request"
        );
        let forced = decided(
            json!({"tools": weather(), "tool_choice": "required", "ebnf": "start: \"y\""}),
            FIRST_TOKEN,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(forced, required);
    }

    /// The prompt says where the turn's grammar starts. Inside a thought it goes on, past a
    /// closed one in a continued message it starts at the answer or the calls, inside an answer
    /// only a free one's rest is held; elsewhere nothing is, and a forced call or an answer
    /// format that cannot be held there is refused.
    #[test]
    fn the_prompt_says_where_the_turn_starts() {
        let thinking = decided(json!({}), FIRST_TOKEN, Some("<|via|>notes<|body|>Let me"))
            .unwrap()
            .unwrap();
        assert_eq!(thinking["elements"][0]["begin"], "");
        let past = decided(
            json!({}),
            FIRST_TOKEN,
            Some("<|via|>notes<|body|>Plan.<|stop|><|head|>assistant"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(past["elements"][0]["value"], "<|via|>reply<|body|>");
        let answering = Some("<|via|>reply<|body|>The answer");
        let rest = decided(json!({}), FIRST_TOKEN, answering).unwrap().unwrap();
        assert_eq!(rest["elements"][0]["type"], "any_text");
        for extra in [
            json!({"tools": weather(), "tool_choice": "required"}),
            json!({"response_format": {"type": "json_object"}}),
        ] {
            assert!(decided(extra.clone(), FIRST_TOKEN, answering).is_err());
            assert!(decided(extra, FIRST_TOKEN, Some("Hello")).is_err());
        }
        assert_eq!(
            decided(json!({"tools": weather()}), FIRST_TOKEN, Some("Hello")),
            Ok(None)
        );
    }

    /// An engine that applies no grammar gets one only for a request that asks for it; a worker
    /// that starts its grammar after the reasoning holds no forced call and no answer format;
    /// one that does not say leaves the request to the table's tags.
    #[test]
    fn the_engines_decide_which_grammar_a_request_gets() {
        let no_backend = Engines {
            start: GrammarStart::FirstToken,
            grammars: false,
        };
        assert_eq!(
            decided(json!({"tools": weather()}), no_backend, None),
            Ok(None)
        );
        let strict = json!([{"type": "function", "function": {"name": "f", "strict": true,
            "parameters": {"type": "object", "properties": {}}}}]);
        assert!(decided(json!({"tools": strict}), no_backend, None)
            .unwrap()
            .is_some());
        let required = json!({"tools": weather(), "tool_choice": "required"});
        assert!(decided(required.clone(), no_backend, None)
            .unwrap()
            .is_some());
        let after = Engines {
            start: GrammarStart::AfterReasoning,
            grammars: true,
        };
        assert!(decided(required.clone(), after, None).is_err());
        assert_eq!(decided(json!({"tools": weather()}), after, None), Ok(None));
        let unknown = Engines {
            start: GrammarStart::Unknown,
            grammars: true,
        };
        assert!(turn_of(required, unknown, None).is_none());
        assert!(turn_of(json!({"tools": weather()}), unknown, None).is_none());
    }

    /// Over the parameter counts an answer format is refused, as a forced call is: the backend
    /// request no longer carries the format, so no grammar would hold it. A request that asks for
    /// neither goes unconstrained.
    #[test]
    fn an_answer_format_over_the_parameter_counts_is_refused() {
        let properties: serde_json::Map<String, Value> = (0..=TurnSpelling::MAX_TOOL_PARAMETERS)
            .map(|i| (format!("p{i}"), json!({"type": "string"})))
            .collect();
        let tools = json!([{"type": "function", "function": {"name": "wide", "parameters": {
            "type": "object", "properties": properties}}}]);
        for extra in [
            json!({"tools": tools, "response_format": {"type": "json_object"}}),
            json!({"tools": tools, "regex": "(yes|no)"}),
            json!({"tools": tools, "tool_choice": "required"}),
        ] {
            assert_eq!(
                decided(extra, FIRST_TOKEN, None).unwrap_err(),
                "400 Bad Request"
            );
        }
        assert_eq!(
            decided(json!({"tools": tools}), FIRST_TOKEN, None),
            Ok(None)
        );
    }

    /// Under the requested-turns policy only a request that asks for a constraint gets one, and
    /// never inside an answer the prompt opened.
    #[test]
    fn the_requested_policy_constrains_only_what_a_request_asks_for() {
        let (format, spelling) = turn_table();
        let prompt = format!("user{OPENER}");
        let constrained = |extra: Value, prompt: &str, continuation: bool| {
            let request = chat(extra);
            let declared: Vec<Tool> = request.effective_tools().cloned().collect();
            let asked = chat_asked(&request, &declared);
            turn_constraint(
                "m",
                &spelling,
                &format,
                &asked,
                true,
                TurnPolicy::Requested,
                prompt,
                continuation,
            )
            .ok()
            .unwrap()
            .is_some()
        };
        assert!(!constrained(json!({}), &prompt, false));
        assert!(!constrained(json!({"tools": weather()}), &prompt, false));
        for extra in [
            json!({"tools": weather(), "tool_choice": "required"}),
            json!({"tools": weather(), "tool_choice": "none"}),
            json!({"response_format": {"type": "json_object"}}),
            json!({"regex": "(a|b)"}),
        ] {
            assert!(constrained(extra, &prompt, false));
        }
        let answering = format!("{prompt}<|via|>reply<|body|>The");
        assert!(!constrained(
            json!({"tools": weather(), "tool_choice": "none"}),
            &answering,
            true
        ));
    }

    /// Every worker of the model says where its engine starts a grammar, or none is taken to;
    /// one that says its grammar backend is `none` makes the model's engines apply none.
    #[test]
    fn the_engines_come_from_the_workers_labels() {
        use crate::worker::{BasicWorkerBuilder, ModelCard, WorkerType};

        let registry = |labels: &[&[(&str, &str)]]| {
            let registry = WorkerRegistry::new();
            for (i, labels) in labels.iter().enumerate() {
                let mut builder = BasicWorkerBuilder::new(format!("http://w{i}:50051"))
                    .model(ModelCard::new("m"))
                    .worker_type(WorkerType::Regular);
                for (key, value) in *labels {
                    builder = builder.label(*key, *value);
                }
                registry
                    .register(std::sync::Arc::new(builder.build()))
                    .expect("register");
            }
            Engines::of(&registry, "m")
        };
        let first = ("structured_outputs_start", "first_token");
        let after = ("structured_outputs_start", "after_reasoning");
        let none = ("grammar_backend", "none");
        assert_eq!(registry(&[&[first], &[first]]), FIRST_TOKEN);
        assert_eq!(registry(&[&[after]]).start, GrammarStart::AfterReasoning);
        for labels in [
            &[][..],
            &[&[first][..], &[]][..],
            &[&[first][..], &[after]][..],
        ] {
            assert_eq!(registry(labels).start, GrammarStart::Unknown);
        }
        assert!(!registry(&[&[first, none]]).grammars);
        assert!(registry(&[&[first, ("grammar_backend", "xgrammar")]]).grammars);
    }

    /// A Messages request's choice shapes its turn as a Chat one's: `any` calls, `tool` one call,
    /// `none` only the answer, and `disable_parallel_tool_use` one call.
    #[test]
    fn a_messages_turn_follows_its_choice() {
        let (format, spelling) = turn_table();
        let prompt = format!("user{OPENER}");
        let tools = json!([{"name": "get_weather", "input_schema": {
            "type": "object", "properties": {"city": {"type": "string"}}}}]);
        let turn = |choice: Value| {
            let request: CreateMessageRequest = serde_json::from_value(json!({
                "model": "m", "max_tokens": 16, "tools": tools,
                "messages": [{"role": "user", "content": "hi"}], "tool_choice": choice
            }))
            .unwrap();
            let tool_choice = request
                .tool_choice
                .as_ref()
                .map(message_utils::convert_message_tool_choice);
            let constraint = messages_turn(
                &format,
                &spelling,
                &request,
                tool_choice.as_ref(),
                FIRST_TOKEN,
                &prompt,
                false,
            )
            .unwrap()
            .ok()
            .unwrap()
            .unwrap();
            serde_json::from_str::<Value>(&constraint.to_tuple().1).unwrap()["format"]["elements"]
                [1]
            .clone()
        };
        assert_eq!(turn(json!({"type": "auto"}))["type"], "or");
        let any = turn(json!({"type": "any"}));
        assert_eq!(
            any["elements"].as_array().unwrap().len(),
            3,
            "call, more, stop"
        );
        let one = turn(json!({"type": "any", "disable_parallel_tool_use": true}));
        assert_eq!(one["elements"].as_array().unwrap().len(), 2);
        let none = turn(json!({"type": "none"}));
        assert_eq!(none["elements"][0]["value"], "<|via|>reply<|body|>");
    }
}
