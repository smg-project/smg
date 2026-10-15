//! Symphony reading a chat response or a message: the opt-in path that parses a model's output
//! with one [`Engine`] per choice in place of the tool and reasoning parser crates.
//!
//! The switch is the gateway's `--parser` flag (`RouterConfig::parser`, `legacy` by default),
//! which a request's `X-SMG-Parser` header overrides for that one request ([`choice`]). With
//! `symphony` on, the parser names set for the model decide ([`reads`]): the
//! `tool_parser` and `reasoning_parser` of the model's card (worker labels), else the
//! `--tool-call-parser` and `--reasoning-parser` flags, resolved as the legacy path resolves them
//! (`ParserResolver`). A name a table answers to (`symphony::formats::Table::answering_to`: its
//! own name, or a legacy parser's that reads its format, `qwen` for `qwen3`, `qwen_xml` for
//! `qwen3_tagged`) reads every request for the model with that table, whatever the model is, the
//! tool-call name deciding; [`TEMPLATE_PARSER`] reads it with the table its tokenizer's
//! `response_template` compiles into ([`template`]), and refuses the request when the model has
//! none; any other legacy tool-call parser's name keeps the legacy parsers, which is the way back
//! for one model when its table misbehaves, `passthrough` among them; no name at all, or a
//! reasoning name alone that no table answers to, other than `passthrough`, takes the registry's
//! table for the model (`symphony::formats::for_model`), and the legacy parsers with their own
//! detection when it has none. The names are checked at startup and at a worker's registration
//! ([`check_names`]). Every request read with `legacy` keeps the older parsers. The response names
//! the parser that read it in `x-smg-parser`, legacy when the choice was Symphony but no table
//! took the request, so a client comparing the two can tell which one read each answer. What a
//! request asks of the channels takes a variant of the table rather than another reader: with no
//! tools declared, or `tool_choice: none`, the table without its calls; with
//! `separate_reasoning: false`, the table without its thought ([`assign`]).
//!
//! Request building assigns Symphony to a request ([`assign`], [`assign_messages`]) and stores
//! the [`Assignment`] in the response spec: the table, and the part of the rendered prompt the
//! engine replays to find the state the generation starts in. Preparation asks the table for the
//! structural tag of a forced or required call, or of a JSON `response_format` ([`constraint`],
//! [`constraint_messages`]) before the older registry. Both decide with one predicate, so the
//! output a table's tag constrains is read by that table and never by the older parsers, which
//! expect another syntax. The streaming and non-streaming drivers run one [`Reader`] per choice
//! over the stop decoder's output and render its events with `symphony::adapt::chat` or
//! `symphony::adapt::messages`, so the client sees the deltas and the message it saw before.

pub(crate) mod template;

use axum::http::HeaderMap;
use llm_tokenizer::stop::{SequenceDecoderOutput, StopSequenceDecoder};
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, Tool, ToolChoice, ToolChoiceValue},
    messages::{CreateMessageRequest, ToolChoice as MessagesToolChoice},
};
use serde_json::{json, Value};
use symphony::{
    formats::{for_model, table_for, Table},
    grammar::Options,
    Declared, Engine, EngineFinish, Event, Events, Format, Grammar, Input, Parser as _, TokenSpan,
};
use tool_parser::ToolConstraint;
use tracing::debug;

use self::template::Templates;
use crate::{
    config::{ParserChoice, TEMPLATE_PARSER},
    routers::grpc::utils::Names,
};

/// The request header that names the parser for one request, `legacy` or `symphony`, over the
/// gateway's `--parser` flag.
pub(crate) const HEADER: &str = "x-smg-parser";

/// The parser that reads a request's response: the one its `X-SMG-Parser` header names, else
/// `default`, the gateway's `--parser` flag. The error names a header value that is neither
/// `legacy` nor `symphony`, for the 400 response.
pub(crate) fn choice(
    default: ParserChoice,
    headers: Option<&HeaderMap>,
) -> Result<ParserChoice, String> {
    let Some(value) = headers.and_then(|headers| headers.get(HEADER)) else {
        return Ok(default);
    };
    let spelled = String::from_utf8_lossy(value.as_bytes());
    ParserChoice::parse(&spelled)
        .ok_or_else(|| format!("{HEADER} must be legacy or symphony, not {spelled:?}"))
}

/// The legacy parser that leaves its channel as written: an operator's choice no table reads
/// for, so it keeps the legacy parsers, alone or beside another name.
const PASSTHROUGH: &str = "passthrough";

fn is_named(name: Option<&str>, what: &str) -> bool {
    name.is_some_and(|name| name.trim().eq_ignore_ascii_case(what))
}

fn is_passthrough(name: Option<&str>) -> bool {
    is_named(name, PASSTHROUGH)
}

/// Whether the parser names select the response template's table: either name is
/// [`TEMPLATE_PARSER`]; [`check_names`] binds the other to it or to nothing.
fn names_template(names: &Names) -> bool {
    is_named(names.tool.as_deref(), TEMPLATE_PARSER)
        || is_named(names.reasoning.as_deref(), TEMPLATE_PARSER)
}

/// The table the parser names set for a model select, when they do: the tool-call name's, else
/// the reasoning name's when no tool-call name is set. A name is a table's own or a legacy
/// parser's that reads its format ([`Table::answering_to`]); any other name selects none,
/// `passthrough` and [`TEMPLATE_PARSER`] among them, and so do no names at all.
pub(crate) fn named_table(names: &Names) -> Option<Table> {
    names
        .tool
        .as_deref()
        .or(names.reasoning.as_deref())
        .and_then(Table::answering_to)
}

/// The table that reads `model`'s responses under `parser`, if one does: the response template's
/// when the `names` say [`TEMPLATE_PARSER`] (an error when the model has none that compiles, which
/// refuses the request, since the names ask for what the model cannot give); else the one the
/// names select ([`named_table`]), whatever the model; else the registry's for the model when no
/// name is set, or when a reasoning name alone selects none, since every table reads the thought
/// channel; `passthrough` alone is the choice to leave the thought as written, which no table
/// does. A tool-call name no table answers to leaves the response to the legacy parsers. What a
/// request asks of the channels takes a variant of the table ([`assign`]). Preparation and
/// request building both decide with this one predicate.
pub(crate) fn reads(
    parser: ParserChoice,
    names: &Names,
    model: &str,
    templates: &Templates,
) -> Result<Option<Format>, String> {
    if parser != ParserChoice::Symphony {
        return Ok(None);
    }
    if names_template(names) {
        return templates.format(model).map(Some);
    }
    let table = match named_table(names) {
        Some(table) => table,
        None if names.tool.is_none() && !is_passthrough(names.reasoning.as_deref()) => {
            return Ok(for_model(model))
        }
        None => return Ok(None),
    };
    // The registry's table comes with the model's own turn opener; a table named for a model
    // the registry reads with another, or does not know, replays from the table's own.
    Ok(match for_model(model) {
        Some(format) if table_for(model) == Some(table) => Some(format),
        _ => Some(table.format()),
    })
}

/// Whether the parser names set for a model (the flags, or one worker's labels) are names the
/// gateway reads with under `parser`: a legacy parser's, or, with Symphony on, a table's or
/// [`TEMPLATE_PARSER`]. `legacy_tool` and `legacy_reasoning` say whether the legacy registries have each
/// name. A table no legacy parser shares, and [`TEMPLATE_PARSER`], bind the other name to the same or to
/// nothing: the legacy parsers read a forced call the table has no tag for by the names they are
/// given, and have no parser of that name. The
/// error says what is wrong with which name, for the startup or registration failure.
pub(crate) fn check_names(
    parser: ParserChoice,
    names: &Names,
    legacy_tool: bool,
    legacy_reasoning: bool,
) -> Result<(), String> {
    let symphony = parser == ParserChoice::Symphony;
    let kinds = [
        ("tool-call", &names.tool, legacy_tool, &names.reasoning),
        ("reasoning", &names.reasoning, legacy_reasoning, &names.tool),
    ];
    for (kind, name, legacy, other) in kinds {
        let Some(name) = name else {
            continue;
        };
        if is_named(Some(name), TEMPLATE_PARSER) {
            if !symphony {
                return Err(format!(
                    "unknown {kind} parser '{name}': the model's response template, which \
                     --parser symphony reads with"
                ));
            }
            if let Some(other) = other
                .as_deref()
                .filter(|other| !is_named(Some(other), TEMPLATE_PARSER))
            {
                return Err(format!(
                    "{kind} parser '{name}' reads with the model's response template, so the \
                     other parser must name '{TEMPLATE_PARSER}' or nothing, not '{other}'"
                ));
            }
            continue;
        }
        if legacy {
            continue;
        }
        match (symphony, Table::named(name)) {
            (true, Some(table)) => {
                if let Some(other) = other {
                    if Table::named(other) != Some(table) {
                        return Err(format!(
                            "{kind} parser '{name}' is a Symphony table's name and no legacy \
                             parser's, so the other parser must name the same table or \
                             nothing, not '{other}'"
                        ));
                    }
                }
            }
            (true, None) => {
                return Err(format!(
                    "unknown {kind} parser '{name}': neither a legacy parser nor a Symphony table"
                ));
            }
            (false, Some(_)) => {
                return Err(format!(
                    "unknown {kind} parser '{name}': a Symphony table's name, which --parser \
                     symphony reads with"
                ));
            }
            (false, None) => return Err(format!("unknown {kind} parser '{name}'")),
        }
    }
    Ok(())
}

/// Whether `tool_choice` leaves the model no choice but a call: a named function, `required`,
/// or allowed tools in `required` mode.
fn forces_call(tool_choice: &ToolChoice) -> bool {
    match tool_choice {
        ToolChoice::Function { .. } => true,
        ToolChoice::Value(ToolChoiceValue::Required) => true,
        ToolChoice::AllowedTools { mode, .. } => mode == "required",
        ToolChoice::Value(_) => false,
    }
}

/// The schema a JSON `response_format` asks the answer to follow: any object for `json_object`.
fn answer_schema(response_format: Option<&ResponseFormat>) -> Option<Value> {
    match response_format? {
        ResponseFormat::JsonObject => Some(json!({"type": "object"})),
        ResponseFormat::JsonSchema { json_schema } => Some(json_schema.schema.clone()),
        ResponseFormat::Text => None,
    }
}

/// The structural tag `format` writes for what the request asks of the whole answer: calls to
/// `tools` when `forced` (at most one when `one_call`), else, for the response template's table
/// alone (`compiled`, [`TEMPLATE_PARSER`]), an answer following `schema` when there is one; the
/// registry's tables leave a JSON `response_format` to the engine's schema, as before.
/// `reasoning_open` when the prompt leaves the model inside its thought. A registry table with
/// no tag for a forced call leaves the older registry to decide, as before; the template's table
/// refuses instead: its output has no other syntax the older path could constrain and the table
/// still read, and so does a prompt that continues the assistant's message, since the tag writes
/// a turn from its start.
/// The constraint `grammar` gives, for the response template's table too (`compiled`,
/// [`TEMPLATE_PARSER`]): a registry table with no tag leaves the older registry to decide, as
/// before; the template's table refuses instead, since its output has no other syntax the older
/// path could constrain and the table still read, and so does a prompt that `continues` the
/// assistant's message, since the tag writes a turn from its start.
fn tagged(
    grammar: Option<Grammar>,
    compiled: bool,
    model: &str,
    asked: &str,
    continues: bool,
) -> Result<Option<ToolConstraint>, String> {
    match grammar {
        Some(_) if compiled && continues => Err(format!(
            "model '{model}' constrains {asked} from the start of a new assistant message, as its \
             response template writes one, so it cannot continue the assistant's message"
        )),
        Some(grammar) => Ok(Some(ToolConstraint::StructuralTag(
            grammar.payload().to_string(),
        ))),
        None if compiled => Err(format!(
            "model '{model}' reads its output with its response template, whose table has no \
             structural tag for {asked}"
        )),
        None => Ok(None),
    }
}

/// The tag of a forced call to one of `tools`, at most one when `one_call`, with
/// `reasoning_open` when the prompt leaves the model inside its thought ([`tagged`]).
fn calls_tag(
    format: &Format,
    compiled: bool,
    model: &str,
    tools: &[Tool],
    one_call: bool,
    reasoning_open: bool,
    continues: bool,
) -> Result<Option<ToolConstraint>, String> {
    let grammar = format.grammar_for(
        tools,
        Options {
            at_least_one: true,
            reasoning_open,
            one_call,
        },
    );
    tagged(grammar, compiled, model, "a forced tool choice", continues)
}

/// The tag of a JSON answer following `schema`, for the response template's table alone: the
/// registry's tables leave a JSON `response_format` to the engine's schema, as before.
fn answer_tag(
    format: &Format,
    model: &str,
    schema: &Value,
    reasoning_open: bool,
    continues: bool,
) -> Result<Option<ToolConstraint>, String> {
    let grammar = format.answer_grammar(schema, reasoning_open);
    tagged(grammar, true, model, "a JSON answer", continues)
}

/// The structural tag the table gives a chat request's forced or required call to one of `tools`,
/// or its JSON `response_format`, with `reasoning_open` when the prompt leaves the model inside
/// its thought ([`calls_tag`], [`answer_tag`]). `Ok(None)` when no table reads the response
/// ([`reads`]), the request asks neither, or the table has no tag for it; the older registry then
/// decides as before. An error refuses the request: the response template's table cannot
/// constrain what is asked, or a `regex` or `ebnf` constraint would apply outside its format.
pub(crate) fn constraint(
    parser: ParserChoice,
    names: &Names,
    request: &ChatCompletionRequest,
    tools: &[Tool],
    reasoning_open: bool,
    templates: &Templates,
) -> Result<Option<ToolConstraint>, String> {
    let Some(format) = reads(parser, names, &request.model, templates)? else {
        return Ok(None);
    };
    let compiled = names_template(names);
    if compiled && (request.regex.is_some() || request.ebnf.is_some()) {
        return Err(format!(
            "model '{}' writes its output in the format its response template declares, and a \
             regex or ebnf constraint would apply outside it",
            request.model
        ));
    }
    let continues = request.continues_final_assistant();
    if !tools.is_empty() && request.tool_choice.as_ref().is_some_and(forces_call) {
        let one_call = matches!(request.tool_choice, Some(ToolChoice::Function { .. }))
            || request.disables_parallel_tool_calls();
        return calls_tag(
            &format,
            compiled,
            &request.model,
            tools,
            one_call,
            reasoning_open,
            continues,
        );
    }
    match answer_schema(request.response_format.as_ref()) {
        Some(schema) if compiled => {
            answer_tag(&format, &request.model, &schema, reasoning_open, continues)
        }
        _ => Ok(None),
    }
}

/// The tag of the calls a chat request's model may make under `auto`, for a worker whose engine
/// takes a grammar on every turn ([`ConstrainedDecoding::Always`]): the table's grammar for the
/// request's callable `tools` with no call required, at most one when parallel calls are off, and
/// `reasoning_open` when the prompt leaves the model inside its thought. The request asked for
/// nothing, so nothing is refused: `Ok(None)` when no table reads the response ([`reads`]), the
/// request has no callable tools or forbids the calls (`tool_choice: none`), it carries a
/// constraint of its own that the tag would displace (a JSON `response_format`, a `regex`, an
/// `ebnf`), it continues the assistant's message (the tag writes a turn from its start), or the
/// table has no tag for the calls. A thought and an answer alone are not constrained today.
pub(crate) fn auto_constraint(
    parser: ParserChoice,
    names: &Names,
    request: &ChatCompletionRequest,
    tools: &[Tool],
    reasoning_open: bool,
    templates: &Templates,
) -> Result<Option<ToolConstraint>, String> {
    let Some(format) = reads(parser, names, &request.model, templates)? else {
        return Ok(None);
    };
    let own_constraint = answer_schema(request.response_format.as_ref()).is_some()
        || request.regex.is_some()
        || request.ebnf.is_some();
    if tools.is_empty()
        || matches!(
            request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        )
        || own_constraint
        || request.continues_final_assistant()
    {
        return Ok(None);
    }
    Ok(auto_tag(
        &format,
        tools,
        reasoning_open,
        request.parallel_tool_calls == Some(false),
    ))
}

/// The tag of the calls `format` lets a model make under `auto`: no call required, one at most
/// when `one_call`; `None` when the table has no tag for them.
fn auto_tag(
    format: &Format,
    tools: &[Tool],
    reasoning_open: bool,
    one_call: bool,
) -> Option<ToolConstraint> {
    format
        .grammar_for(
            tools,
            Options {
                at_least_one: false,
                reasoning_open,
                one_call,
            },
        )
        .map(|grammar| ToolConstraint::StructuralTag(grammar.payload().to_string()))
}

/// [`constraint`] for a Messages request: a `tool_choice` of `any` or a named tool forces a call
/// to one of `tools`, one call when the tool is named or parallel use is disabled. `continues`
/// says whether the prompt continues the assistant's message.
pub(crate) fn constraint_messages(
    parser: ParserChoice,
    names: &Names,
    request: &CreateMessageRequest,
    tools: &[Tool],
    reasoning_open: bool,
    continues: bool,
    templates: &Templates,
) -> Result<Option<ToolConstraint>, String> {
    let Some(format) = reads(parser, names, &request.model, templates)? else {
        return Ok(None);
    };
    let one_call = match &request.tool_choice {
        _ if tools.is_empty() => return Ok(None),
        Some(MessagesToolChoice::Tool { .. }) => true,
        Some(choice @ MessagesToolChoice::Any { .. }) => choice.disables_parallel_tool_use(),
        _ => return Ok(None),
    };
    calls_tag(
        &format,
        names_template(names),
        &request.model,
        tools,
        one_call,
        reasoning_open,
        continues,
    )
}

/// [`auto_constraint`] for a Messages request: the calls it may make under `auto` (or no choice),
/// one at most when parallel use is disabled; nothing when the prompt `continues` the assistant's
/// message. A Messages request carries no constraint of its own (an `output_format` is refused
/// before this).
pub(crate) fn auto_constraint_messages(
    parser: ParserChoice,
    names: &Names,
    request: &CreateMessageRequest,
    tools: &[Tool],
    reasoning_open: bool,
    continues: bool,
    templates: &Templates,
) -> Result<Option<ToolConstraint>, String> {
    let Some(format) = reads(parser, names, &request.model, templates)? else {
        return Ok(None);
    };
    let one_call = match &request.tool_choice {
        _ if tools.is_empty() || continues => return Ok(None),
        Some(MessagesToolChoice::None) => return Ok(None),
        Some(MessagesToolChoice::Auto {
            disable_parallel_tool_use,
        }) => *disable_parallel_tool_use == Some(true),
        // A forced choice has its own tag ([`constraint_messages`]).
        Some(MessagesToolChoice::Any { .. } | MessagesToolChoice::Tool { .. }) => return Ok(None),
        None => false,
    };
    Ok(auto_tag(&format, tools, reasoning_open, one_call))
}

/// Symphony's assignment to a response: the table that reads it, as the request's asks make it
/// ([`assign`]), and the part of the rendered prompt the engine replays.
#[derive(Clone, Debug)]
pub(crate) struct Assignment {
    pub format: Format,
    pub prompt: String,
}

/// The assignment of `format` to a response, with the part of `prompt` the engine replays. Not
/// made when the constraint is a JSON schema: the older path's forced call, whose output has no
/// markers for a table to read, which the older registry writes where the table has no tag.
fn assignment(
    format: Option<Format>,
    model: &str,
    constraint: Option<&(String, String)>,
    prompt: &str,
) -> Option<Assignment> {
    let format = format?;
    if constraint.is_some_and(|(kind, _)| kind == "json_schema") {
        return None;
    }
    debug!(model, table = format.name(), "Symphony reads the response");
    let prompt = format.replay(prompt).to_string();
    Some(Assignment { format, prompt })
}

/// The table as the request's asks make it: without its calls when the request declares no tools
/// or asks for none, as the legacy path runs no tool parser then, and a table that reads calls
/// from the output's first byte (`json`, `xlam`) is content from it; without its thought when the
/// request keeps the reasoning in the content, as the legacy path runs no reasoning parser then.
fn asked(format: Format, no_calls: bool, no_reasoning: bool) -> Format {
    let format = if no_calls {
        format.without_calls()
    } else {
        format
    };
    if no_reasoning {
        format.without_reasoning()
    } else {
        format
    }
}

/// Symphony's assignment to a chat `request`, if a table reads the response ([`reads`]) and the
/// constraint is not a JSON schema ([`assignment`]): the table without its calls for no tools or
/// a `tool_choice` of `none`, without its thought for `separate_reasoning: false` ([`asked`]).
pub(crate) fn assign(
    parser: ParserChoice,
    names: &Names,
    request: &ChatCompletionRequest,
    constraint: Option<&(String, String)>,
    prompt: &str,
    templates: &Templates,
) -> Result<Option<Assignment>, String> {
    let format = reads(parser, names, &request.model, templates)?;
    let no_calls = matches!(
        request.tool_choice,
        Some(ToolChoice::Value(ToolChoiceValue::None))
    ) || request.effective_tools().next().is_none();
    let format = format.map(|format| asked(format, no_calls, !request.separate_reasoning));
    Ok(assignment(format, &request.model, constraint, prompt))
}

/// [`assign`] for a Messages request: its thinking blocks are always its own, so only the
/// calls' ask takes a variant.
pub(crate) fn assign_messages(
    parser: ParserChoice,
    names: &Names,
    request: &CreateMessageRequest,
    constraint: Option<&(String, String)>,
    prompt: &str,
    templates: &Templates,
) -> Result<Option<Assignment>, String> {
    let format = reads(parser, names, &request.model, templates)?;
    let no_calls = matches!(request.tool_choice, Some(MessagesToolChoice::None))
        || request.tools.as_deref().is_none_or(<[_]>::is_empty);
    let format = format.map(|format| asked(format, no_calls, false));
    Ok(assignment(format, &request.model, constraint, prompt))
}

/// The engine's finish reason as Symphony takes it.
pub(crate) fn engine_finish(reason: &str) -> EngineFinish {
    match reason {
        "stop" => EngineFinish::Stop,
        "length" => EngineFinish::Length,
        "abort" => EngineFinish::Abort,
        other => EngineFinish::Other(other.to_string()),
    }
}

/// What the `Finish` event among `events` says.
pub(crate) struct Finished {
    /// The calls made in the output.
    pub tool_calls: u32,
    /// The engine tokens spent on reasoning, markers excluded.
    pub reasoning_tokens: u32,
}

/// The `Finish` event among `events`, if the output ended.
pub(crate) fn finished(events: &[Event]) -> Option<Finished> {
    events.iter().find_map(|event| match event {
        Event::Finish {
            tool_calls,
            reasoning_tokens,
            ..
        } => Some(Finished {
            tool_calls: *tool_calls,
            reasoning_tokens: *reasoning_tokens,
        }),
        _ => None,
    })
}

/// The byte ranges of the output that are the message's content, from the events' sources in
/// order (every byte of the output lands in one event), adjacent runs joined: what the logprobs
/// filter keeps, as the older path keeps the tokens of `message.content` alone. Malformed text
/// is content, as `adapt::chat::message` writes it.
pub(crate) fn content_spans(events: &[Event]) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    for event in events {
        let (bytes, content) = match event {
            Event::Content(text) | Event::Malformed { text, .. } => (text.text.len(), true),
            Event::Reasoning(text) | Event::Dropped { text, .. } => (text.text.len(), false),
            Event::ToolCallStart { source, .. }
            | Event::ToolCallArguments { source, .. }
            | Event::ToolCallEnd { source, .. } => (source.text.len(), false),
            Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => (0, false),
        };
        if content && bytes > 0 {
            match spans.last_mut() {
                Some((_, end)) if *end == at => *end = at + bytes,
                _ => spans.push((at, at + bytes)),
            }
        }
        at += bytes;
    }
    spans
}

/// The engine tokens the reasoning text among `events` cost.
pub(crate) fn reasoning_text_tokens(events: &[Event]) -> u32 {
    events
        .iter()
        .map(|event| match event {
            Event::Reasoning(text) => text.tokens.unwrap_or(0),
            _ => 0,
        })
        .sum()
}

/// One choice's reader: the engine over the choice's output, fed what the stop decoder releases
/// with the span of every token in it.
pub(crate) struct Reader {
    engine: Engine,
    events: Events,
    /// The last token the decoder saw: text it releases later belongs to it.
    last_token: Option<u32>,
    /// Where each token's text ends in the output read so far, in order; the text the decoder
    /// releases after a token extends that token's end, as the older path's `token_ends` do.
    ends: Vec<usize>,
    /// The bytes of the output read so far.
    read: usize,
    ended: bool,
    /// Drop the events of every call after the first.
    first_call_only: bool,
}

impl Reader {
    /// A reader for one choice, seeded with the assignment's prompt.
    pub(crate) fn new(assignment: &Assignment, tools: &[Tool]) -> Result<Self, String> {
        let mut reader = Self {
            engine: Engine::new(assignment.format.clone(), Declared::of(tools)),
            events: Events::new(),
            last_token: None,
            ends: Vec::new(),
            read: 0,
            ended: false,
            first_call_only: false,
        };
        reader.feed(Input::Prompt {
            token_ids: &[],
            text: &assignment.prompt,
        })?;
        Ok(reader)
    }

    /// Under `parallel_tool_calls: false` or `disable_parallel_tool_use`, the
    /// events of every call after the first are dropped: the table's grammar
    /// limits a forced call, the reader limits a free one.
    pub(crate) fn first_call_only(mut self, on: bool) -> Self {
        self.first_call_only = on;
        self
    }

    fn drained(&mut self) -> Vec<Event> {
        let mut events = self.events.drain();
        if self.first_call_only {
            events.retain(|event| {
                !matches!(
                    event,
                    Event::ToolCallStart { index, .. }
                        | Event::ToolCallArguments { index, .. }
                        | Event::ToolCallEnd { index, .. }
                        if *index > 0
                )
            });
            for event in &mut events {
                if let Event::Finish { tool_calls, .. } = event {
                    *tool_calls = (*tool_calls).min(1);
                }
            }
        }
        events
    }

    /// Reads `token_ids` through `decoder` as one delta: the text the decoder releases, with each
    /// token's span in it, zero-width for a token whose bytes it holds back. Gives the events,
    /// and whether the decoder matched a stop, after which the rest of the tokens are not read.
    pub(crate) fn read(
        &mut self,
        decoder: &mut StopSequenceDecoder,
        token_ids: &[u32],
    ) -> Result<(Vec<Event>, bool), String> {
        let mut text = String::new();
        let mut spans = Vec::with_capacity(token_ids.len());
        let mut read = Vec::with_capacity(token_ids.len());
        let mut stopped = false;
        for &token_id in token_ids {
            let output = decoder
                .process_token(token_id)
                .map_err(|e| format!("Stop decoder failed to process token {token_id}: {e}"))?;
            let start = text.len();
            match output {
                SequenceDecoderOutput::Text(released) => text.push_str(&released),
                SequenceDecoderOutput::StoppedWithText(released) => {
                    text.push_str(&released);
                    stopped = true;
                }
                SequenceDecoderOutput::Stopped => stopped = true,
                SequenceDecoderOutput::Held => {}
            }
            spans.push(TokenSpan {
                token_id,
                start,
                end: text.len(),
                continued: false,
            });
            self.ends.push(self.read + text.len());
            read.push(token_id);
            self.last_token = Some(token_id);
            if stopped {
                break;
            }
        }
        self.read += text.len();
        self.feed(Input::Delta {
            token_ids: &read,
            text: &text,
            spans: &spans,
        })?;
        Ok((self.drained(), stopped))
    }

    /// Where each token's text ends in the output read so far, for the logprobs filter the older
    /// path applies: a token belongs to the content when its text does.
    pub(crate) fn token_ends(&self) -> &[usize] {
        &self.ends
    }

    /// Reads `text` the decoder released after its last token, which held it back as a possible
    /// stop: bytes of that token.
    pub(crate) fn tail(&mut self, text: &str) -> Result<Vec<Event>, String> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let spans: Vec<TokenSpan> = self
            .last_token
            .map(|token_id| TokenSpan {
                token_id,
                start: 0,
                end: text.len(),
                continued: true,
            })
            .into_iter()
            .collect();
        self.read += text.len();
        if let Some(last) = self.ends.last_mut() {
            *last = self.read;
        }
        self.feed(Input::Delta {
            token_ids: &[],
            text,
            spans: &spans,
        })?;
        Ok(self.drained())
    }

    /// Ends the output with the engine's finish, once: the events, the last of them the
    /// `Finish`, or nothing when the output already ended.
    pub(crate) fn end(&mut self, finish: EngineFinish) -> Result<Vec<Event>, String> {
        if self.ended {
            return Ok(Vec::new());
        }
        self.ended = true;
        self.feed(Input::End { finish })?;
        Ok(self.drained())
    }

    /// Whether the output ended.
    pub(crate) fn ended(&self) -> bool {
        self.ended
    }

    fn feed(&mut self, input: Input<'_>) -> Result<(), String> {
        self.engine
            .feed(input, &mut self.events)
            .map_err(|e| format!("Symphony failed to read the output: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_tokenizer::registry::TokenizerRegistry;
    use serde_json::{json, Value};
    use symphony::{formats::qwen3, CallSyntax};

    use super::*;

    fn request(extra: Value) -> ChatCompletionRequest {
        let mut request = json!({"model": "Qwen/Qwen3-8B", "messages": [], "stream": true});
        for (key, value) in extra.as_object().into_iter().flatten() {
            request[key] = value.clone();
        }
        serde_json::from_value(request).expect("chat request")
    }

    fn tools() -> Vec<Tool> {
        serde_json::from_value(json!([{
            "type": "function", "function": {
                "name": "lookup", "parameters": {"type": "object", "properties": {}}
            }
        }]))
        .expect("tools")
    }

    const PROMPT: &str = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";

    fn names(tool: Option<&str>, reasoning: Option<&str>) -> Names {
        Names {
            tool: tool.map(str::to_string),
            reasoning: reasoning.map(str::to_string),
        }
    }

    /// No tokenizer registered: the response template's table is never there.
    fn templates() -> Templates {
        Templates::new(Arc::new(TokenizerRegistry::new()))
    }

    /// [`super::assign`] with no templates: the names never say `hf`, so nothing is refused.
    fn assign(
        parser: ParserChoice,
        names: &Names,
        request: &ChatCompletionRequest,
        constraint: Option<&(String, String)>,
        prompt: &str,
    ) -> Option<Assignment> {
        super::assign(parser, names, request, constraint, prompt, &templates()).expect("no hf")
    }

    /// [`super::constraint`] with no templates, as [`assign`].
    fn constraint(
        parser: ParserChoice,
        names: &Names,
        request: &ChatCompletionRequest,
        tools: &[Tool],
        reasoning_open: bool,
    ) -> Option<ToolConstraint> {
        super::constraint(parser, names, request, tools, reasoning_open, &templates())
            .expect("no hf")
    }

    #[test]
    fn the_tool_call_name_selects_the_table_that_answers_to_it() {
        assert_eq!(named_table(&Names::default()), None);
        assert_eq!(
            named_table(&names(Some("kimi_k3"), Some("kimi_k3"))),
            Some(Table::KimiK3)
        );
        assert_eq!(named_table(&names(None, Some("Ling"))), Some(Table::Ling));
        assert_eq!(
            named_table(&names(Some("qwen3_tagged"), None)),
            Some(Table::Qwen3Tagged)
        );
        // A legacy name reads with the table of the format it reads.
        assert_eq!(named_table(&names(Some("qwen"), None)), Some(Table::Qwen3));
        assert_eq!(
            named_table(&names(Some("qwen"), Some("qwen3"))),
            Some(Table::Qwen3)
        );
        assert_eq!(
            named_table(&names(Some(" QWEN_XML "), Some("qwen3"))),
            Some(Table::Qwen3Tagged)
        );
        assert_eq!(
            named_table(&names(Some("glm45_moe"), Some("glm45"))),
            Some(Table::Glm4)
        );
        assert_eq!(
            named_table(&names(Some("glm47_moe"), Some("glm45"))),
            Some(Table::Glm)
        );
        // The reasoning name counts only without a tool-call name.
        assert_eq!(named_table(&names(None, Some("hy_v4"))), Some(Table::Hy4));
        assert_eq!(
            named_table(&names(None, Some("minimax"))),
            Some(Table::MinimaxM2)
        );
        assert_eq!(named_table(&names(None, Some("deepseek_r1"))), None);
        // The tool-call name decides; a legacy name with no alias selects none.
        assert_eq!(
            named_table(&names(Some("kimi_k3"), Some("qwen3"))),
            Some(Table::KimiK3)
        );
        assert_eq!(named_table(&names(Some("llama"), None)), None);
        assert_eq!(named_table(&names(Some("llama"), Some("qwen3"))), None);
        // `passthrough` leaves a channel as written, which no table reads for: the legacy
        // parsers keep it, alone or beside another name.
        assert_eq!(
            named_table(&names(Some("passthrough"), Some(" Passthrough "))),
            None
        );
        assert_eq!(named_table(&names(Some("passthrough"), None)), None);
        assert_eq!(named_table(&names(None, Some("passthrough"))), None);
        assert_eq!(
            named_table(&names(Some("passthrough"), Some("qwen3"))),
            None
        );
    }

    #[test]
    fn every_name_a_table_answers_to_besides_its_own_is_a_legacy_parsers() {
        let tools = tool_parser::ParserFactory::new();
        let reasoning = reasoning_parser::ParserFactory::new();
        for (alias, _) in Table::ALIASES {
            assert!(
                tools.has_parser(alias) || reasoning.registry().has_parser(alias),
                "{alias} is no legacy parser's name"
            );
        }
    }

    #[test]
    fn a_request_without_tools_is_read_by_every_table_and_its_calls_are_content() {
        // `json` and `xlam` read a bare JSON answer as a call: without tools the request asked
        // for none, so the table goes without its calls and the answer is the text it was, as
        // the legacy parsers leave it when they run no tool parser; a table with call markers
        // goes the same way.
        let json = names(Some("json"), None);
        let without = request(json!({}));
        let assigned =
            assign(ParserChoice::Symphony, &json, &without, None, PROMPT).expect("assigned");
        assert_eq!(assigned.format.name(), "json");
        assert_eq!(assigned.format.call_syntax(), None);
        let declared = serde_json::to_value(tools()).expect("tools");
        let with = request(json!({"tools": declared}));
        let assigned =
            assign(ParserChoice::Symphony, &json, &with, None, PROMPT).expect("assigned");
        assert!(assigned.format.call_syntax().is_some());
        let xlam = request(json!({"model": "Salesforce/Llama-xLAM-2-8b-fc-r"}));
        let by_registry = assign(
            ParserChoice::Symphony,
            &Names::default(),
            &xlam,
            None,
            PROMPT,
        )
        .expect("assigned");
        assert_eq!(by_registry.format.name(), "xlam");
        assert_eq!(by_registry.format.call_syntax(), None);
        for named in ["inkling", "llama_nemotron"] {
            let assigned = assign(
                ParserChoice::Symphony,
                &names(Some(named), None),
                &without,
                None,
                PROMPT,
            )
            .expect("assigned");
            assert_eq!(assigned.format.name(), named);
            assert_eq!(assigned.format.call_syntax(), None);
        }
    }

    #[test]
    fn explicit_names_win_over_the_registry() {
        // A table named for a model the registry does not know reads it; a legacy name reads
        // with its alias, or with no alias leaves a model the registry knows to the legacy
        // parsers; a table named for a model the registry reads with another wins, and replays
        // from its own opener.
        let unknown = request(json!({"model": "unknown"}));
        let ling = names(Some("ling"), Some("ling"));
        let assigned = assign(ParserChoice::Symphony, &ling, &unknown, None, PROMPT);
        assert_eq!(assigned.expect("assigned").format.name(), "ling");
        let aliased = names(Some("qwen"), Some("qwen3"));
        let assigned = assign(
            ParserChoice::Symphony,
            &aliased,
            &request(json!({})),
            None,
            PROMPT,
        );
        assert_eq!(assigned.expect("assigned").format.name(), "qwen3");
        let legacy = names(Some("llama"), Some("deepseek_r1"));
        assert!(assign(
            ParserChoice::Symphony,
            &legacy,
            &request(json!({})),
            None,
            PROMPT
        )
        .is_none());
        // A reasoning name alone that selects no table leaves the choice to the registry: a
        // known model takes its table and the table's tag, an unknown one the legacy parsers.
        let required = request(json!({"tool_choice": "required"}));
        let reasoning_only = names(None, Some("deepseek_r1"));
        assert!(constraint(
            ParserChoice::Symphony,
            &reasoning_only,
            &required,
            &tools(),
            false
        )
        .is_some());
        let assigned = assign(
            ParserChoice::Symphony,
            &reasoning_only,
            &request(json!({})),
            None,
            PROMPT,
        );
        assert_eq!(assigned.expect("assigned").format.name(), "qwen3");
        let unknown_required = request(json!({"model": "unknown", "tool_choice": "required"}));
        assert!(constraint(
            ParserChoice::Symphony,
            &reasoning_only,
            &unknown_required,
            &tools(),
            false
        )
        .is_none());
        assert!(assign(
            ParserChoice::Symphony,
            &reasoning_only,
            &unknown,
            None,
            PROMPT
        )
        .is_none());
        // `passthrough` alone is the choice to leave the thought as written: the legacy
        // parsers keep a known model too.
        let passthrough_only = names(None, Some("passthrough"));
        assert!(assign(
            ParserChoice::Symphony,
            &passthrough_only,
            &request(json!({})),
            None,
            PROMPT
        )
        .is_none());
        let kimi = names(Some("kimi_k3"), None);
        let assigned = assign(
            ParserChoice::Symphony,
            &kimi,
            &request(json!({})),
            None,
            PROMPT,
        );
        let assigned = assigned.expect("assigned");
        assert_eq!(assigned.format.name(), "kimi_k3");
        assert_eq!(
            assigned.prompt, PROMPT,
            "no Kimi opener in a Qwen prompt: the whole replays"
        );
        // The registry's own table named for its model keeps the model's turn opener.
        let exaone = request(json!({"model": "LGAI-EXAONE/K-EXAONE-236B-A23B"}));
        let prompt = "<|user|>hi<|endofturn|><|assistant|>\n<think>\n";
        let qwen3 = names(Some("qwen3"), Some("qwen3"));
        let by_name = assign(ParserChoice::Symphony, &qwen3, &exaone, None, prompt);
        let by_registry = assign(
            ParserChoice::Symphony,
            &Names::default(),
            &exaone,
            None,
            prompt,
        );
        assert_eq!(
            by_name.expect("assigned").prompt,
            "<|assistant|>\n<think>\n"
        );
        assert_eq!(
            by_registry.expect("assigned").prompt,
            "<|assistant|>\n<think>\n"
        );
        // Under legacy the names change nothing.
        assert!(assign(ParserChoice::Legacy, &ling, &unknown, None, PROMPT).is_none());
    }

    #[test]
    fn names_are_checked_against_both_registries_by_the_parser_in_force() {
        let legacy = ParserChoice::Legacy;
        let symphony = ParserChoice::Symphony;
        assert_eq!(check_names(legacy, &Names::default(), true, true), Ok(()));
        assert_eq!(
            check_names(legacy, &names(Some("qwen"), Some("qwen3")), true, true),
            Ok(())
        );
        let why = check_names(legacy, &names(Some("nonsense"), None), false, true);
        assert_eq!(
            why.expect_err("unknown"),
            "unknown tool-call parser 'nonsense'"
        );
        let why = check_names(legacy, &names(None, Some("ling")), true, false);
        assert_eq!(
            why.expect_err("a table under legacy"),
            "unknown reasoning parser 'ling': a Symphony table's name, which --parser symphony \
             reads with"
        );
        // Under Symphony a table's name passes alone or doubled, and binds the other name.
        assert_eq!(
            check_names(symphony, &names(Some("ling"), None), false, true),
            Ok(())
        );
        assert_eq!(
            check_names(symphony, &names(Some("ling"), Some("LING")), false, false),
            Ok(())
        );
        assert_eq!(
            check_names(symphony, &names(Some("kimi_k3"), Some("qwen3")), true, true),
            Ok(())
        );
        let why = check_names(symphony, &names(Some("ling"), Some("qwen3")), false, true);
        assert_eq!(
            why.expect_err("bound to the table"),
            "tool-call parser 'ling' is a Symphony table's name and no legacy parser's, so the \
             other parser must name the same table or nothing, not 'qwen3'"
        );
        let why = check_names(symphony, &names(None, Some("nonsense")), true, false);
        assert_eq!(
            why.expect_err("unknown"),
            "unknown reasoning parser 'nonsense': neither a legacy parser nor a Symphony table"
        );
        // A legacy name a table answers to is checked as the legacy name it is: a tool parser's
        // given as the reasoning parser is unknown there, under either parser.
        let why = check_names(symphony, &names(None, Some("qwen_xml")), true, false);
        assert_eq!(
            why.expect_err("wrong kind"),
            "unknown reasoning parser 'qwen_xml': neither a legacy parser nor a Symphony table"
        );
        let why = check_names(legacy, &names(None, Some("qwen_xml")), true, false);
        assert_eq!(
            why.expect_err("wrong kind"),
            "unknown reasoning parser 'qwen_xml'"
        );
    }

    #[test]
    fn the_header_names_the_parser_over_the_flag_and_an_unknown_name_is_refused() {
        assert_eq!(choice(ParserChoice::Legacy, None), Ok(ParserChoice::Legacy));
        assert_eq!(
            choice(ParserChoice::Symphony, Some(&HeaderMap::new())),
            Ok(ParserChoice::Symphony)
        );
        let mut headers = HeaderMap::new();
        headers.insert(HEADER, "symphony".parse().expect("a header value"));
        assert_eq!(
            choice(ParserChoice::Legacy, Some(&headers)),
            Ok(ParserChoice::Symphony)
        );
        headers.insert(HEADER, " Legacy ".parse().expect("a header value"));
        assert_eq!(
            choice(ParserChoice::Symphony, Some(&headers)),
            Ok(ParserChoice::Legacy)
        );
        headers.insert(HEADER, "fast".parse().expect("a header value"));
        let why = choice(ParserChoice::Legacy, Some(&headers)).expect_err("refused");
        assert_eq!(why, "x-smg-parser must be legacy or symphony, not \"fast\"");
    }

    #[test]
    fn the_legacy_choice_assigns_nothing_and_asks_for_no_constraint() {
        assert!(assign(
            ParserChoice::Legacy,
            &Names::default(),
            &request(json!({})),
            None,
            PROMPT
        )
        .is_none());
        let required = request(json!({"tool_choice": "required"}));
        assert!(constraint(
            ParserChoice::Legacy,
            &Names::default(),
            &required,
            &tools(),
            false
        )
        .is_none());
    }

    #[test]
    fn the_requests_asks_take_a_variant_of_the_table_and_do_not_switch_the_reader() {
        // A request that keeps the thought in the content, or asks for no calls, is read by the
        // same table without its thought or its calls; the constraint is the full table's.
        let table = |request: &ChatCompletionRequest| {
            reads(
                ParserChoice::Symphony,
                &Names::default(),
                &request.model,
                &templates(),
            )
            .expect("read")
            .expect("a table")
        };
        let assigned = |request: &ChatCompletionRequest| {
            assign(
                ParserChoice::Symphony,
                &Names::default(),
                request,
                None,
                PROMPT,
            )
            .expect("assigned")
            .format
        };
        let declared = serde_json::to_value(tools()).expect("tools");
        let kept = request(
            json!({"tools": declared, "tool_choice": "required", "separate_reasoning": false}),
        );
        assert_eq!(assigned(&kept), table(&kept).without_reasoning());
        assert!(constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &kept,
            &tools(),
            false
        )
        .is_some());
        let no_calls = request(json!({"tool_choice": "none"}));
        assert_eq!(assigned(&no_calls), table(&no_calls).without_calls());
        assert!(constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &no_calls,
            &tools(),
            false
        )
        .is_none());
        let neither = request(json!({"separate_reasoning": false}));
        assert_eq!(
            assigned(&neither),
            table(&neither).without_calls().without_reasoning()
        );
        let both = request(json!({"tools": declared}));
        assert_eq!(assigned(&both), table(&both));
    }

    #[test]
    fn a_known_model_is_assigned_its_table_and_the_prompts_replay() {
        let assignment = assign(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({})),
            None,
            PROMPT,
        )
        .expect("assigned");
        assert_eq!(assignment.format.name(), "qwen3");
        assert_eq!(assignment.prompt, "<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn an_unknown_model_and_a_json_schema_constraint_are_not_assigned() {
        assert!(assign(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({"model": "unknown"})),
            None,
            PROMPT
        )
        .is_none());
        let schema = ("json_schema".to_string(), "{}".to_string());
        assert!(assign(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({})),
            Some(&schema),
            PROMPT
        )
        .is_none());
        let tag = ("structural_tag".to_string(), "{}".to_string());
        assert!(assign(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({})),
            Some(&tag),
            PROMPT
        )
        .is_some());
    }

    #[test]
    fn a_forced_call_takes_the_tables_structural_tag_and_auto_takes_none() {
        let required = request(json!({"tool_choice": "required"}));
        let Some(ToolConstraint::StructuralTag(tag)) = constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &required,
            &tools(),
            false,
        ) else {
            panic!("a structural tag");
        };
        let tag: Value = serde_json::from_str(&tag).expect("JSON");
        assert!(tag["format"].is_object(), "{tag}");
        let auto = request(json!({"tool_choice": "auto"}));
        assert!(constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &auto,
            &tools(),
            false
        )
        .is_none());
        let unknown = request(json!({"model": "unknown", "tool_choice": "required"}));
        assert!(constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &unknown,
            &tools(),
            false
        )
        .is_none());
    }

    /// For a worker on `always`, the calls a request may make under `auto` take the table's tag
    /// with no call required. Nothing is refused: a request with no tools, `tool_choice: none`,
    /// a constraint of its own (a JSON `response_format`, a `regex`), or a prompt that continues
    /// the assistant's message takes none, and so does a forced Messages choice.
    #[test]
    fn an_always_worker_takes_the_tables_tag_for_the_calls_auto_allows() {
        let auto = request(json!({"tool_choice": "auto"}));
        let tag = auto_constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &auto,
            &tools(),
            false,
            &templates(),
        )
        .expect("no hf");
        let Some(ToolConstraint::StructuralTag(tag)) = tag else {
            panic!("a structural tag");
        };
        let tag: Value = serde_json::from_str(&tag).expect("JSON");
        assert!(tag["format"].is_object(), "{tag}");
        let Some(ToolConstraint::StructuralTag(forced)) = constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({"tool_choice": "required"})),
            &tools(),
            false,
        ) else {
            panic!("a forced tag");
        };
        let forced: Value = serde_json::from_str(&forced).expect("JSON");
        assert_ne!(tag, forced, "the auto tag requires no call");
        assert!(auto_constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &auto,
            &[],
            false,
            &templates(),
        )
        .expect("no hf")
        .is_none());
        assert!(auto_constraint(
            ParserChoice::Legacy,
            &Names::default(),
            &auto,
            &tools(),
            false,
            &templates(),
        )
        .expect("legacy")
        .is_none());
        assert!(auto_constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({"tool_choice": "none"})),
            &tools(),
            false,
            &templates(),
        )
        .expect("no hf")
        .is_none());
        for own in [
            json!({"tool_choice": "auto", "response_format": {"type": "json_object"}}),
            json!({"tool_choice": "auto", "regex": "[a-z]+"}),
            json!({
                "tool_choice": "auto",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "I was saying"}
                ],
                "continue_final_message": true, "add_generation_prompt": false
            }),
        ] {
            assert!(
                auto_constraint(
                    ParserChoice::Symphony,
                    &Names::default(),
                    &request(own.clone()),
                    &tools(),
                    false,
                    &templates(),
                )
                .expect("no hf")
                .is_none(),
                "{own}"
            );
        }
        let messages_auto = message_request(json!({
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
        }));
        let tools: Vec<Tool> = serde_json::from_value(json!([{
            "type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}
        }]))
        .unwrap();
        assert!(matches!(
            auto_constraint_messages(
                ParserChoice::Symphony,
                &Names::default(),
                &messages_auto,
                &tools,
                false,
                false,
                &templates(),
            )
            .expect("no hf"),
            Some(ToolConstraint::StructuralTag(_))
        ));
        assert!(auto_constraint_messages(
            ParserChoice::Symphony,
            &Names::default(),
            &messages_auto,
            &tools,
            false,
            true,
            &templates(),
        )
        .expect("no hf")
        .is_none());
        let messages_any = message_request(json!({
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"}
        }));
        assert!(auto_constraint_messages(
            ParserChoice::Symphony,
            &Names::default(),
            &messages_any,
            &tools,
            false,
            false,
            &templates(),
        )
        .expect("no hf")
        .is_none());
    }

    #[test]
    fn the_engines_finish_reasons_map_and_the_rest_carry_through() {
        assert_eq!(engine_finish("stop"), EngineFinish::Stop);
        assert_eq!(engine_finish("length"), EngineFinish::Length);
        assert_eq!(engine_finish("abort"), EngineFinish::Abort);
        assert_eq!(
            engine_finish("failed"),
            EngineFinish::Other("failed".to_string())
        );
    }

    #[test]
    fn a_reader_ends_once() {
        let assignment = Assignment {
            format: qwen3(CallSyntax::Json),
            prompt: String::new(),
        };
        let mut reader = Reader::new(&assignment, &[]).expect("a reader");
        assert!(!reader.ended());
        let events = reader.end(EngineFinish::Stop).expect("ended");
        assert!(finished(&events).is_some());
        assert!(reader.ended());
        assert!(reader.end(EngineFinish::Stop).expect("ended").is_empty());
    }

    fn message_request(extra: Value) -> CreateMessageRequest {
        let mut request = json!({
            "model": "Qwen/Qwen3-8B", "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]
        });
        for (key, value) in extra.as_object().into_iter().flatten() {
            request[key] = value.clone();
        }
        serde_json::from_value(request).expect("messages request")
    }

    fn tag_of(constraint: Result<Option<ToolConstraint>, String>) -> Value {
        match constraint {
            Ok(Some(ToolConstraint::StructuralTag(tag))) => {
                serde_json::from_str(&tag).expect("a tag as JSON")
            }
            other => panic!("a structural tag, not {other:?}"),
        }
    }

    #[tokio::test]
    async fn hf_reads_with_the_response_templates_table_and_refuses_a_model_without_one() {
        let templates = template::fixtures::templates_with(
            "Qwen/Qwen3-8B",
            template::fixtures::qwen3_5_tokenizer(),
        )
        .await;
        let hf = names(Some("hf"), Some(" HF "));
        let read = |names: &Names, request: &ChatCompletionRequest| {
            reads(ParserChoice::Symphony, names, &request.model, &templates)
        };
        // The template's table reads the model the registry knows with another table, named
        // on both sides or on either alone.
        for names in [&hf, &names(Some("hf"), None), &names(None, Some("hf"))] {
            let format = read(names, &request(json!({})))
                .expect("read")
                .expect("a table");
            assert_eq!(format.name(), "response_template");
        }
        assert!(
            reads(ParserChoice::Legacy, &hf, "Qwen/Qwen3-8B", &templates)
                .expect("read")
                .is_none()
        );
        // What the request asks of the channels takes a variant of the templates' table.
        let kept = request(json!({"separate_reasoning": false}));
        let assigned = super::assign(ParserChoice::Symphony, &hf, &kept, None, PROMPT, &templates)
            .expect("read")
            .expect("assigned");
        assert_eq!(
            assigned.format,
            read(&hf, &kept)
                .expect("read")
                .expect("a table")
                .without_calls()
                .without_reasoning()
        );
        // A model with no table of its own is refused with the reason, in both decisions.
        let unknown = request(json!({"model": "unknown"}));
        assert_eq!(
            read(&hf, &unknown).expect_err("refused"),
            "no tokenizer is registered for model 'unknown'"
        );
        assert!(super::assign(
            ParserChoice::Symphony,
            &hf,
            &unknown,
            None,
            PROMPT,
            &templates
        )
        .is_err());
        let assigned = super::assign(
            ParserChoice::Symphony,
            &hf,
            &request(json!({})),
            None,
            PROMPT,
            &templates,
        )
        .expect("read")
        .expect("assigned");
        assert_eq!(assigned.format.name(), "response_template");
        assert_eq!(assigned.prompt, "<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn hf_is_a_name_under_symphony_alone_and_binds_the_other_name() {
        let symphony = ParserChoice::Symphony;
        assert_eq!(
            check_names(symphony, &names(Some("hf"), None), false, false),
            Ok(())
        );
        assert_eq!(
            check_names(symphony, &names(None, Some("hf")), false, false),
            Ok(())
        );
        assert_eq!(
            check_names(symphony, &names(Some("HF"), Some(" hf ")), false, false),
            Ok(())
        );
        let why = check_names(symphony, &names(Some("hf"), Some("qwen3")), false, true);
        assert_eq!(
            why.expect_err("bound"),
            "tool-call parser 'hf' reads with the model's response template, so the other \
             parser must name 'hf' or nothing, not 'qwen3'"
        );
        let why = check_names(symphony, &names(Some("qwen"), Some("hf")), true, false);
        assert_eq!(
            why.expect_err("bound"),
            "reasoning parser 'hf' reads with the model's response template, so the other \
             parser must name 'hf' or nothing, not 'qwen'"
        );
        let why = check_names(ParserChoice::Legacy, &names(Some("hf"), None), false, true);
        assert_eq!(
            why.expect_err("legacy"),
            "unknown tool-call parser 'hf': the model's response template, which --parser \
             symphony reads with"
        );
    }

    #[tokio::test]
    async fn the_templates_table_gives_a_forced_call_and_a_json_answer_their_tags_or_refuses() {
        let model = "Qwen/Qwen3-8B";
        let with_layout =
            template::fixtures::templates_with(model, template::fixtures::qwen3_5_tokenizer())
                .await;
        let without_layout =
            template::fixtures::templates_with(model, template::fixtures::unrendered_tokenizer())
                .await;
        let hf = names(Some("hf"), Some("hf"));
        let constrain = |templates: &Templates, extra: Value| {
            super::constraint(
                ParserChoice::Symphony,
                &hf,
                &request(extra),
                &tools(),
                false,
                templates,
            )
        };
        // A forced call, one call for a named function, and a JSON answer take the table's tags.
        let required = tag_of(constrain(&with_layout, json!({"tool_choice": "required"})));
        assert!(required["format"].is_object(), "{required}");
        let named = tag_of(constrain(
            &with_layout,
            json!({"tool_choice": {"type": "function", "function": {"name": "lookup"}}}),
        ));
        assert_ne!(named, required, "one call ends where the turn does");
        let answer = tag_of(constrain(
            &with_layout,
            json!({"response_format": {"type": "json_object"}}),
        ));
        assert!(answer["format"].is_object(), "{answer}");
        // Nothing asked, nothing constrained.
        assert!(constrain(&with_layout, json!({"tool_choice": "auto"}))
            .expect("read")
            .is_none());
        // Without a layout the table has no tag for a call: the template's table refuses, where
        // a registry table would leave the older registry to decide. Its content opens with no
        // marker, so a JSON answer is written from the first byte, layout or not.
        let why =
            constrain(&without_layout, json!({"tool_choice": "required"})).expect_err("no tag");
        assert!(
            why.contains("no structural tag for a forced tool choice"),
            "{why}"
        );
        let answer = tag_of(constrain(
            &without_layout,
            json!({"response_format": {"type": "json_object"}}),
        ));
        assert_eq!(answer["format"]["type"], "json_schema", "{answer}");
        let plain = names(None, None);
        assert!(
            super::constraint(
                ParserChoice::Symphony,
                &plain,
                &request(json!({"response_format": {"type": "json_object"}})),
                &tools(),
                false,
                &templates(),
            )
            .expect("read")
            .is_none(),
            "the Qwen3 table has no layout, so the older registry's JSON schema stands"
        );
        // A regex or ebnf constraint would apply outside the template's format.
        let why = constrain(
            &with_layout,
            json!({"tool_choice": "required", "regex": "a+"}),
        )
        .expect_err("refused");
        assert!(why.contains("regex or ebnf"), "{why}");
        // The tag writes a turn from its start: a continued assistant message is refused.
        let continued = json!({
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "I"}],
            "continue_final_message": true, "add_generation_prompt": false,
        });
        assert!(
            request(continued.clone()).continues_final_assistant(),
            "the request continues"
        );
        let why = constrain(&with_layout, continued).expect_err("refused");
        assert!(
            why.contains("cannot continue the assistant's message"),
            "{why}"
        );
    }

    #[tokio::test]
    async fn messages_are_read_by_the_same_tables_and_their_tool_choice_forces_calls() {
        let symphony = ParserChoice::Symphony;
        let plain = Names::default();
        let read = reads(symphony, &plain, "Qwen/Qwen3-8B", &templates())
            .expect("read")
            .expect("the registry's table");
        assert_eq!(read.name(), "qwen3");
        assert!(
            reads(ParserChoice::Legacy, &plain, "Qwen/Qwen3-8B", &templates())
                .expect("read")
                .is_none()
        );
        // A tool_choice of none takes the table without its calls.
        let no_calls = assign_messages(
            symphony,
            &plain,
            &message_request(json!({"tool_choice": {"type": "none"}})),
            None,
            PROMPT,
            &templates(),
        )
        .expect("read")
        .expect("assigned");
        assert_eq!(no_calls.format.name(), "qwen3");
        assert_eq!(no_calls.format.call_syntax(), None);
        let constrain = |extra: Value| {
            constraint_messages(
                symphony,
                &plain,
                &message_request(extra),
                &tools(),
                false,
                false,
                &templates(),
            )
        };
        let any = tag_of(constrain(json!({"tool_choice": {"type": "any"}})));
        let one = tag_of(constrain(
            json!({"tool_choice": {"type": "any", "disable_parallel_tool_use": true}}),
        ));
        let tool = tag_of(constrain(
            json!({"tool_choice": {"type": "tool", "name": "lookup"}}),
        ));
        assert_ne!(any, one, "one call ends where the turn does");
        assert_eq!(tool, one, "a named tool is one call");
        assert!(constrain(json!({"tool_choice": {"type": "auto"}}))
            .expect("read")
            .is_none());
        assert!(constrain(json!({})).expect("read").is_none());
        let assigned = assign_messages(
            symphony,
            &plain,
            &message_request(json!({})),
            None,
            PROMPT,
            &templates(),
        )
        .expect("read")
        .expect("assigned");
        assert_eq!(assigned.format.name(), "qwen3");
        assert_eq!(assigned.prompt, "<|im_start|>assistant\n<think>\n");
        // The template's table reads messages too, and refuses what it cannot tag.
        let hf = names(Some("hf"), None);
        let unrendered = template::fixtures::templates_with(
            "Qwen/Qwen3-8B",
            template::fixtures::unrendered_tokenizer(),
        )
        .await;
        let why = constraint_messages(
            symphony,
            &hf,
            &message_request(json!({"tool_choice": {"type": "any"}})),
            &tools(),
            false,
            false,
            &unrendered,
        )
        .expect_err("no tag");
        assert!(why.contains("no structural tag"), "{why}");
    }

    /// A Qwen3 output: a thought, content, and one call of the declared `lookup` tool.
    const QWEN3_OUTPUT: &str = "<think>plan</think>\n\nHello <tool_call>\n\
                                {\"name\": \"lookup\", \"arguments\": {\"q\": 1}}\n</tool_call>";

    fn read_with(format: Format) -> Vec<Event> {
        let assignment = Assignment {
            format,
            prompt: "<|im_start|>assistant\n".to_string(),
        };
        let mut reader = Reader::new(&assignment, &tools()).expect("a reader");
        let mut events = reader.tail(QWEN3_OUTPUT).expect("read");
        events.extend(reader.end(EngineFinish::Stop).expect("ended"));
        events
    }

    /// Limited to one call, the reader keeps the first call's events and
    /// counts one call; unlimited, both calls come through.
    #[test]
    fn a_reader_limited_to_one_call_drops_the_calls_after_the_first() {
        const TWO_CALLS: &str = "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": 1}}\n</tool_call>\
                                 <tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": 2}}\n</tool_call>";
        let assignment = Assignment {
            format: qwen3(CallSyntax::Json),
            prompt: "<|im_start|>assistant\n".to_string(),
        };
        let calls = |first_only: bool| {
            let mut reader = Reader::new(&assignment, &tools())
                .expect("a reader")
                .first_call_only(first_only);
            let mut events = reader.tail(TWO_CALLS).expect("read");
            events.extend(reader.end(EngineFinish::Stop).expect("ended"));
            let starts = events
                .iter()
                .filter(|event| matches!(event, Event::ToolCallStart { .. }))
                .count();
            let Some(Event::Finish { tool_calls, .. }) = events.last() else {
                panic!("no finish event");
            };
            (starts, *tool_calls)
        };
        assert_eq!(calls(false), (2, 2));
        assert_eq!(calls(true), (1, 1));
    }

    fn content_of(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_bare_answer_the_json_table_reads_as_a_call_comes_back_as_its_text() {
        let assignment = Assignment {
            format: symphony::formats::json().without_calls(),
            prompt: String::new(),
        };
        let mut reader = Reader::new(&assignment, &[]).expect("a reader");
        let answer = "{\"answer\": 42}";
        let mut events = reader.tail(answer).expect("read");
        events.extend(reader.end(EngineFinish::Stop).expect("ended"));
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) | Event::Malformed { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, answer);
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        let Some(Event::Finish { tool_calls, .. }) = events.last() else {
            panic!("a finish");
        };
        assert_eq!(*tool_calls, 0);
    }

    #[test]
    fn a_reader_over_a_tables_variant_gives_the_calls_or_the_thought_as_content() {
        // The table itself: the thought apart, the call a call.
        let events = read_with(qwen3(CallSyntax::Json));
        assert!(events.iter().any(|e| matches!(e, Event::Reasoning(_))));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        assert_eq!(content_of(&events).trim(), "Hello");
        let Some(Event::Finish { tool_calls, .. }) = events.last() else {
            panic!("a finish");
        };
        assert_eq!(*tool_calls, 1);
        // Without its calls: the call and its markers are content, the finish counts no call.
        let events = read_with(qwen3(CallSyntax::Json).without_calls());
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        assert!(events.iter().any(|e| matches!(e, Event::Reasoning(_))));
        let content = content_of(&events);
        assert!(
            content
                .trim_start()
                .starts_with("Hello <tool_call>\n{\"name\": \"lookup\""),
            "{content:?}"
        );
        assert!(content.ends_with("</tool_call>"), "{content:?}");
        assert!(!content.contains("<think>"), "{content:?}");
        let Some(Event::Finish { tool_calls, .. }) = events.last() else {
            panic!("a finish");
        };
        assert_eq!(*tool_calls, 0);
        // Without its thought: the thought and its markers are content, the call stays a call.
        let events = read_with(qwen3(CallSyntax::Json).without_reasoning());
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::Reasoning(_) | Event::ReasoningStart)));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        let content = content_of(&events);
        assert!(content.starts_with("<think>plan</think>"), "{content:?}");
        assert!(content.contains("Hello"), "{content:?}");
        assert!(!content.contains("<tool_call>"), "{content:?}");
        let Some(Event::Finish {
            reasoning_tokens, ..
        }) = events.last()
        else {
            panic!("a finish");
        };
        assert_eq!(*reasoning_tokens, 0);
    }
}
