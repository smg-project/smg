//! Symphony reading a chat response or a message: the opt-in path that parses a model's output
//! with one [`Engine`] per choice in place of the tool and reasoning parser crates.
//!
//! The switch is the gateway's `--parser` flag (`RouterConfig::parser`, `legacy` by default),
//! which a request's `X-SMG-Parser` header overrides for that one request ([`choice`]). With
//! `symphony` on, the parser names set for the model decide ([`reads`]): the
//! `tool_parser` and `reasoning_parser` of the model's card (worker labels), else the
//! `--tool-call-parser` and `--reasoning-parser` flags, resolved as the legacy path resolves them
//! (`ParserResolver`). A name a table answers to (`symphony::formats::Table::answering_to`: its
//! own name, a legacy parser's that reads its format, `qwen` for `qwen3`, `qwen_xml` for
//! `qwen3_tagged`, or a launcher's name for a family whose table has no older parser, `muse` for
//! `muse_spark`) reads every request for the model with that table, whatever the model is, the
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
//! request asks of the channels takes a variant of the table rather than another reader: where the
//! legacy path would run no tool parser (`tool_choice: none`; no tools declared, for a provider
//! that reads calls only with them), the table without its calls; with
//! `separate_reasoning: false`, the table without its thought ([`assign`]).
//!
//! Request building assigns Symphony to a request ([`assign`], [`assign_messages`]) and stores
//! the [`Assignment`] in the response spec: the table, and the part of the rendered prompt the
//! engine replays to find the state the generation starts in. Preparation asks the table for the
//! structural tag of a forced or required call, or of a JSON `response_format` ([`constraint`],
//! [`constraint_messages`]) before the older registry. Both decide with one predicate, so the
//! output a table's tag constrains is read by that table and never by the older parsers, which
//! expect another syntax. Request building decides what constrains the output on the worker
//! selected for the request ([`worker_constraint`]): a worker whose engine takes a grammar on every
//! turn from its first token gets the whole turn's grammar where the response template's table
//! spells one ([`turn_constraint`]); a table's refusal in preparation waits for that decision. The
//! streaming and non-streaming drivers run one [`Reader`] per choice
//! over the stop decoder's output and render its events with `symphony::adapt::chat` or
//! `symphony::adapt::messages`, so the client sees the deltas and the message it saw before.

pub(crate) mod template;

use std::{borrow::Cow, sync::Arc};

use axum::http::HeaderMap;
use llm_tokenizer::{
    stop::{SequenceDecoderOutput, StopSequenceDecoder},
    traits::Tokenizer,
};
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, Tool, ToolChoice, ToolChoiceValue},
    messages::{CreateMessageRequest, ToolChoice as MessagesToolChoice},
    worker::ConstrainedDecoding,
};
use serde_json::{json, Value};
use symphony::{
    formats::{for_model, table_for, Table},
    grammar::{GrammarStart, Options},
    Declared, Engine, EngineFinish, Event, Events, Format, Grammar, Input, OutputAnswer,
    OutputChoice, Parser as _, TokenSpan, Turn,
};
use tool_parser::{GrammarStart as WorkerGrammarStart, ToolConstraint};
use tracing::debug;

use self::template::Templates;
use crate::{
    config::{ParserChoice, TEMPLATE_PARSER},
    routers::grpc::{
        spec::scans_for_tool_calls,
        utils::{Names, WorkerDecoding},
    },
    worker::{Worker, WorkerType},
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
/// the reasoning name's when no tool-call name is set. A name is one a table answers to
/// ([`Table::answering_to`]): its own, a legacy parser's that reads its format, or a launcher's
/// for a family whose table has no older parser; any other name selects none, `passthrough` and
/// [`TEMPLATE_PARSER`] among them, and so do no names at all.
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
/// [`TEMPLATE_PARSER`]. `tools` and `reasoning` are the legacy registries; a name whose registry
/// is absent is not checked (its parsers are unused in that configuration). A name no legacy
/// parser has that a table answers to (the table's own name, or a launcher's name for a family
/// whose table has no older parser), and [`TEMPLATE_PARSER`], bind the other name to the same or
/// to nothing: the legacy parsers read a forced call the table has no tag for by the names they
/// are given, and have no parser of that name. A legacy parser's name a table answers to is
/// checked as the legacy name it is, in its own kind. The error says what is wrong with which
/// name, for the startup or registration failure.
pub(crate) fn check_names(
    parser: ParserChoice,
    names: &Names,
    tools: Option<&tool_parser::ParserFactory>,
    reasoning: Option<&reasoning_parser::ParserFactory>,
) -> Result<(), String> {
    let symphony = parser == ParserChoice::Symphony;
    let legacy_tool = |name: &str| tools.is_some_and(|factory| factory.has_parser(name));
    let legacy_reasoning =
        |name: &str| reasoning.is_some_and(|factory| factory.registry().has_parser(name));
    // The table a name selects when no legacy parser of its kind has it: the table named so,
    // else the one that answers to it when no legacy parser of either kind has the name as the
    // registries spell it (a table answers in any case, the registries in theirs); a legacy
    // parser's name of the other kind selects none, whatever table answers to it.
    let table = |name: &str| {
        Table::named(name).or_else(|| {
            let spelled = name.trim().to_ascii_lowercase();
            (!legacy_tool(&spelled) && !legacy_reasoning(&spelled))
                .then(|| Table::answering_to(name))
                .flatten()
        })
    };
    let kinds = [
        (
            "tool-call",
            names.tool.as_deref(),
            tools.is_none() || names.tool.as_deref().is_some_and(legacy_tool),
            names.reasoning.as_deref(),
        ),
        (
            "reasoning",
            names.reasoning.as_deref(),
            reasoning.is_none() || names.reasoning.as_deref().is_some_and(legacy_reasoning),
            names.tool.as_deref(),
        ),
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
            if let Some(other) = other.filter(|other| !is_named(Some(other), TEMPLATE_PARSER)) {
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
        match (symphony, table(name)) {
            (true, Some(selected)) => {
                if let Some(other) = other {
                    if table(other) != Some(selected) {
                        return Err(format!(
                            "{kind} parser '{name}' selects the Symphony table '{}' and is no \
                             legacy parser's name, so the other parser must name the same \
                             table or nothing, not '{other}'",
                            selected.name()
                        ));
                    }
                }
            }
            (true, None) => {
                return Err(format!(
                    "unknown {kind} parser '{name}': neither a legacy parser nor a Symphony table"
                ));
            }
            (false, Some(selected)) => {
                return Err(format!(
                    "unknown {kind} parser '{name}': the Symphony table '{}', which --parser \
                     symphony reads with",
                    selected.name()
                ));
            }
            (false, None) => return Err(format!("unknown {kind} parser '{name}'")),
        }
    }
    Ok(())
}

/// The schema a JSON `response_format` asks the answer to follow: any object for `json_object`.
fn answer_schema(response_format: Option<&ResponseFormat>) -> Option<Value> {
    match response_format? {
        ResponseFormat::JsonObject => Some(json!({"type": "object"})),
        ResponseFormat::JsonSchema { json_schema } => Some(json_schema.schema.clone()),
        ResponseFormat::Text => None,
    }
}

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
#[expect(
    clippy::too_many_arguments,
    reason = "the table, the request's ask, where the prompt and the engine leave the model"
)]
fn calls_tag(
    format: &Format,
    compiled: bool,
    model: &str,
    tools: &[Tool],
    one_call: bool,
    reasoning_open: bool,
    start: GrammarStart,
    continues: bool,
) -> Result<Option<ToolConstraint>, String> {
    let grammar = format.grammar_for(
        tools,
        Options {
            at_least_one: true,
            reasoning_open,
            one_call,
            start,
        },
    );
    tagged(grammar, compiled, model, "a forced tool choice", continues)
}

/// The tag of a JSON answer following `schema`, with `reasoning_open` when the prompt leaves the
/// model inside its thought ([`Format::answer_grammar`]): the answer written where the table
/// reads one, which the engine's own schema constraint is not for a table whose answer is a
/// message (the bare JSON lands outside every message and the table drops it), nor with the
/// thought open for an engine that runs the grammar from the first token (the JSON lands inside
/// the thought), nor for a table that admits a thought before its answer. A registry table whose
/// tag is the schema alone, the engine's own constraint, leaves it to the engine as before, and
/// so does one with no tag, or a prompt that `continues` the assistant's message, since the tag
/// writes a turn from its start; the template's table takes its tag either way and refuses what
/// it cannot constrain ([`tagged`]).
fn answer_tag(
    format: &Format,
    compiled: bool,
    model: &str,
    schema: &Value,
    reasoning_open: bool,
    start: GrammarStart,
    continues: bool,
) -> Result<Option<ToolConstraint>, String> {
    let grammar = format.answer_grammar_for(schema, reasoning_open, start);
    if !compiled && (continues || matches!(grammar, Some(Grammar::JsonSchema { .. }))) {
        return Ok(None);
    }
    tagged(grammar, compiled, model, "a JSON answer", continues)
}

/// Whether the table that reads `model`'s responses under `parser` ([`reads`]) needs the
/// vocabulary's special tokens kept in the decoded output: one of its markers is, or holds, a
/// token the stop decoder would drop with `skip_special_tokens` (Muse Spark's `<|message|>` and
/// `<|eom|>`, Kimi K3's `<|open|>` and `<|sep|>`). A legacy reasoning parser says so of itself
/// (`requires_special_tokens`); a table says nothing, so the gateway asks the tokenizer: a marker
/// whose tokens decode to its text with the special tokens kept and to something else with them
/// dropped is such a marker. `false` when no table reads the response, or when every marker
/// survives the decoder either way (Qwen3's `<think>` and `<tool_call>` are added tokens the
/// decoder keeps), so a model the older rule already serves is read as before.
pub(crate) fn keeps_special_tokens(
    parser: ParserChoice,
    names: &Names,
    model: &str,
    templates: &Templates,
    tokenizer: &dyn Tokenizer,
) -> bool {
    match reads(parser, names, model, templates) {
        Ok(Some(format)) => markers_hidden_by_the_decoder(&format, tokenizer),
        _ => false,
    }
}

/// Whether one of `format`'s markers loses text when its tokens are decoded with the special
/// tokens dropped: the marker as the tokenizer encodes it decodes back to itself with them kept.
fn markers_hidden_by_the_decoder(format: &Format, tokenizer: &dyn Tokenizer) -> bool {
    format.terminal_texts().any(|marker| {
        let Ok(encoding) = tokenizer.encode(marker, false) else {
            return false;
        };
        let ids = encoding.token_ids();
        let kept = tokenizer.decode(ids, false);
        let dropped = tokenizer.decode(ids, true);
        matches!((kept, dropped), (Ok(kept), Ok(dropped)) if kept == marker && dropped != marker)
    })
}

/// Where a worker's engine starts a request's grammar on a prompt that leaves the model inside
/// its thought, as Symphony's tags take it: the worker's `constrained_decoding_start` label
/// ([`WorkerDecoding::start`]).
pub(crate) fn grammar_start_from(start: WorkerGrammarStart) -> GrammarStart {
    match start {
        WorkerGrammarStart::AfterReasoning => GrammarStart::AfterReasoning,
        WorkerGrammarStart::FirstToken => GrammarStart::FirstToken,
        WorkerGrammarStart::Unknown => GrammarStart::Unknown,
    }
}

/// Where the model's `workers` start a grammar, before one is selected: the start every one of
/// them names in its `constrained_decoding_start` label, a prefill or encode worker left out
/// since it generates no token a grammar constrains (the roles that do are allow-listed, so a
/// role added later is left out until it is named); unknown when none says or they differ, and
/// the tag takes the prefix that fits either kind of engine.
pub(crate) fn grammar_start_of(workers: &[Arc<dyn Worker>]) -> GrammarStart {
    let mut starts = workers
        .iter()
        .filter(|worker| {
            matches!(
                worker.worker_type(),
                WorkerType::Regular | WorkerType::Decode
            )
        })
        .map(|worker| {
            WorkerGrammarStart::from_label(
                worker
                    .metadata()
                    .spec
                    .labels
                    .get(WorkerGrammarStart::LABEL)
                    .map(String::as_str),
            )
        });
    let Some(first) = starts.next() else {
        return GrammarStart::Unknown;
    };
    if starts.all(|start| start == first) {
        grammar_start_from(first)
    } else {
        GrammarStart::Unknown
    }
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
    start: GrammarStart,
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
    if !tools.is_empty()
        && request
            .tool_choice
            .as_ref()
            .is_some_and(ToolChoice::forces_call)
    {
        let one_call = matches!(request.tool_choice, Some(ToolChoice::Function { .. }))
            || request.disables_parallel_tool_calls();
        return calls_tag(
            &format,
            compiled,
            &request.model,
            tools,
            one_call,
            reasoning_open,
            start,
            continues,
        );
    }
    match answer_schema(request.response_format.as_ref()) {
        Some(schema) => answer_tag(
            &format,
            compiled,
            &request.model,
            &schema,
            reasoning_open,
            start,
            continues,
        ),
        None => Ok(None),
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
    start: GrammarStart,
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
        start,
        request.parallel_tool_calls == Some(false),
    ))
}

/// The tag of the calls `format` lets a model make under `auto`: no call required, one at most
/// when `one_call`; `None` when the table has no tag for them.
fn auto_tag(
    format: &Format,
    tools: &[Tool],
    reasoning_open: bool,
    start: GrammarStart,
    one_call: bool,
) -> Option<ToolConstraint> {
    format
        .grammar_for(
            tools,
            Options {
                at_least_one: false,
                reasoning_open,
                one_call,
                start,
            },
        )
        .map(|grammar| ToolConstraint::StructuralTag(grammar.payload().to_string()))
}

/// [`constraint`] for a Messages request: a `tool_choice` of `any` or a named tool forces a call
/// to one of `tools`, one call when the tool is named or parallel use is disabled. `continues`
/// says whether the prompt continues the assistant's message.
#[expect(
    clippy::too_many_arguments,
    reason = "the parser's choice, the request, its tools, where the prompt and the engine leave the model"
)]
pub(crate) fn constraint_messages(
    parser: ParserChoice,
    names: &Names,
    request: &CreateMessageRequest,
    tools: &[Tool],
    reasoning_open: bool,
    start: GrammarStart,
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
        start,
        continues,
    )
}

/// [`auto_constraint`] for a Messages request: the calls it may make under `auto` (or no choice),
/// one at most when parallel use is disabled; nothing when the prompt `continues` the assistant's
/// message. A Messages request carries no constraint of its own (an `output_format` is refused
/// before this).
#[expect(
    clippy::too_many_arguments,
    reason = "the parser's choice, the request, its tools, where the prompt and the engine leave the model"
)]
pub(crate) fn auto_constraint_messages(
    parser: ParserChoice,
    names: &Names,
    request: &CreateMessageRequest,
    tools: &[Tool],
    reasoning_open: bool,
    start: GrammarStart,
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
    Ok(auto_tag(&format, tools, reasoning_open, start, one_call))
}

/// The constraint preparation finds for what a request asks, before its worker is known: the
/// table's tag for a forced call or a JSON answer, else the registry's; or why the table cannot
/// constrain the request ([`constraint`], [`constraint_messages`]). Request building decides on the
/// selected worker ([`worker_constraint`]).
pub(crate) type PreparedConstraint = Result<Option<ToolConstraint>, String>;

/// What constrains a request's output on the worker selected for it.
pub(crate) struct WorkerConstraint {
    pub constraint: Option<ToolConstraint>,
    /// The constraint is the whole turn's grammar, which holds the request's own format: the
    /// engine's request carries none beside it.
    pub whole_turn: bool,
    /// The worker's policy chose the constraint (the whole turn's grammar or the `auto` tag), not
    /// the request: a worker of another policy would not send it.
    pub by_policy: bool,
}

/// What constrains a request's output on the worker selected for it, whose engine takes a grammar
/// on every turn or only when the request asks and starts it where its label says (`decoding`,
/// under PD the decode worker's). The one place a worker's policy decides:
///
/// - `requested`, the default: what preparation found (`prepared`), its refusal included.
/// - `always`, on an engine that starts the grammar at the first token: the whole turn's grammar
///   (`turn`, [`turn_constraint`]) where it starts; its refusal refuses the request.
/// - `always` otherwise, or where the whole turn's grammar does not start: what preparation found,
///   and when that is nothing, the table's tag for the calls `auto` allows (`auto`,
///   [`auto_constraint`]).
///
/// When the policy chose the constraint, a retry re-selects only workers that decide alike
/// ([`WorkerDecoding::decides_alike`]).
pub(crate) fn worker_constraint(
    decoding: WorkerDecoding,
    prepared: PreparedConstraint,
    turn: impl FnOnce() -> Result<Option<ToolConstraint>, String>,
    auto: impl FnOnce() -> Result<Option<ToolConstraint>, String>,
) -> Result<WorkerConstraint, String> {
    let always = decoding.policy == ConstrainedDecoding::Always;
    if decoding.takes_the_whole_turn() {
        if let Some(constraint) = turn()? {
            return Ok(WorkerConstraint {
                constraint: Some(constraint),
                whole_turn: true,
                by_policy: true,
            });
        }
    }
    let (constraint, by_policy) = match prepared? {
        None if always => {
            let tag = auto()?;
            let by_policy = tag.is_some();
            (tag, by_policy)
        }
        prepared => (prepared, false),
    };
    Ok(WorkerConstraint {
        constraint,
        whole_turn: false,
        by_policy,
    })
}

/// The whole turn's grammar for a chat request ([`Format::turn_grammar`]), where the response
/// template's table reads the response ([`TEMPLATE_PARSER`]); `prompt` is the rendered prompt.
/// `Ok(None)` for any other table, and where the turn's grammar does not start; an error refuses
/// what it cannot hold.
pub(crate) fn turn_constraint(
    parser: ParserChoice,
    names: &Names,
    request: &ChatCompletionRequest,
    prompt: &str,
    templates: &Templates,
) -> Result<Option<ToolConstraint>, String> {
    if !names_template(names) {
        return Ok(None);
    }
    match reads(parser, names, &request.model, templates)? {
        Some(format) => chat_turn(&format, request, prompt),
        None => Ok(None),
    }
}

/// [`turn_constraint`] over `format`: the request's tools and choice, and its `response_format`,
/// `regex` or `ebnf` as the answer's (the protocol refuses more than one).
fn chat_turn(
    format: &Format,
    request: &ChatCompletionRequest,
    prompt: &str,
) -> Result<Option<ToolConstraint>, String> {
    let declared = request.effective_tool_list();
    let callable = match request.tool_choice {
        Some(ToolChoice::Value(ToolChoiceValue::None)) => Cow::Borrowed(&[][..]),
        _ => request.callable_tools(),
    };
    let schema = answer_schema(request.response_format.as_ref());
    let answer = schema
        .as_ref()
        .map(OutputAnswer::Json)
        .or_else(|| request.regex.as_deref().map(OutputAnswer::Regex))
        .or_else(|| request.ebnf.as_deref().map(OutputAnswer::Ebnf));
    whole_turn(
        format,
        &Turn {
            prompt,
            continued: request.continues_final_assistant(),
            declared: &declared,
            callable: &callable,
            choice: output_choice(request.tool_choice.as_ref()),
            parallel: !request.disables_parallel_tool_calls(),
            answer,
        },
    )
}

/// [`turn_constraint`] for a Messages request whose tools are `tools` (its chat tools): its
/// `tool_choice` as a Chat one, one call at most when parallel use is disabled; `continues` says
/// whether the prompt continues the assistant's message. A Messages request carries no answer
/// format (an `output_format` is refused before this).
pub(crate) fn turn_constraint_messages(
    parser: ParserChoice,
    names: &Names,
    request: &CreateMessageRequest,
    tools: &[Tool],
    prompt: &str,
    continues: bool,
    templates: &Templates,
) -> Result<Option<ToolConstraint>, String> {
    if !names_template(names) {
        return Ok(None);
    }
    match reads(parser, names, &request.model, templates)? {
        Some(format) => messages_turn(&format, request, tools, prompt, continues),
        None => Ok(None),
    }
}

/// [`turn_constraint_messages`] over `format`.
fn messages_turn(
    format: &Format,
    request: &CreateMessageRequest,
    tools: &[Tool],
    prompt: &str,
    continues: bool,
) -> Result<Option<ToolConstraint>, String> {
    let choice = request.tool_choice.as_ref().map(ToolChoice::from);
    let callable = match &choice {
        Some(ToolChoice::Value(ToolChoiceValue::None)) => Vec::new(),
        Some(choice) => choice.narrow_tools(tools).unwrap_or_else(|| tools.to_vec()),
        None => tools.to_vec(),
    };
    whole_turn(
        format,
        &Turn {
            prompt,
            continued: continues,
            declared: tools,
            callable: &callable,
            choice: output_choice(choice.as_ref()),
            parallel: !request.disables_parallel_tool_use(),
            answer: None,
        },
    )
}

/// How a tool choice shapes a turn: one call for a named function, one or more where the choice
/// forces a call, an answer or calls otherwise.
fn output_choice(choice: Option<&ToolChoice>) -> OutputChoice {
    match choice {
        Some(ToolChoice::Function { .. }) => OutputChoice::One,
        Some(choice) if choice.forces_call() => OutputChoice::Required,
        _ => OutputChoice::Auto,
    }
}

/// `format`'s grammar for `turn` as the engine's structural tag.
fn whole_turn(format: &Format, turn: &Turn<'_>) -> Result<Option<ToolConstraint>, String> {
    Ok(format
        .turn_grammar(turn)?
        .map(|grammar| ToolConstraint::StructuralTag(grammar.payload_text())))
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

/// The table as the request's asks make it: without its calls when the legacy path would run no
/// tool parser (the request asks for none, or its response is not scanned for calls:
/// [`scans_for_tool_calls`]), so a table that reads calls from the output's first byte (`json`,
/// `xlam`) is content from it; without its thought when the request keeps the reasoning in the
/// content, as the legacy path runs no reasoning parser then.
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
/// constraint is not a JSON schema ([`assignment`]): the table without its calls for a
/// `tool_choice` of `none` or a response the legacy path would not scan for calls (no tools
/// declared anywhere, for a provider that reads calls only with them: [`scans_for_tool_calls`]),
/// without its thought for `separate_reasoning: false` ([`asked`]). A call the conversation
/// established without a tool inventory comes back as a call under either parser.
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
    ) || !scans_for_tool_calls(request);
    let format = format.map(|format| asked(format, no_calls, !request.separate_reasoning));
    Ok(assignment(format, &request.model, constraint, prompt))
}

/// [`assign`] for a Messages request: its thinking blocks are always its own, so only the
/// calls' ask takes a variant; the legacy path runs its tool parser for a Messages request that
/// declares tools, an empty list included, and for no other, so the calls stay exactly then.
pub(crate) fn assign_messages(
    parser: ParserChoice,
    names: &Names,
    request: &CreateMessageRequest,
    constraint: Option<&(String, String)>,
    prompt: &str,
    templates: &Templates,
) -> Result<Option<Assignment>, String> {
    let format = reads(parser, names, &request.model, templates)?;
    let no_calls =
        matches!(request.tool_choice, Some(MessagesToolChoice::None)) || request.tools.is_none();
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
    /// A reader for one choice, seeded with the assignment's prompt. `tools` are the request's
    /// as its response spec holds them: every choice's reader shares them, none copies a schema.
    pub(crate) fn new(assignment: &Assignment, tools: Arc<[Tool]>) -> Result<Self, String> {
        let mut reader = Self {
            engine: Engine::new(assignment.format.clone(), Declared::shared(tools)),
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
    use tool_parser::GrammarStart;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

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

    /// [`super::check_names`] against the legacy registries as the gateway builds them.
    fn check(parser: ParserChoice, names: &Names) -> Result<(), String> {
        let tools = tool_parser::ParserFactory::new();
        let reasoning = reasoning_parser::ParserFactory::new();
        check_names(parser, names, Some(&tools), Some(&reasoning))
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
        super::constraint(
            parser,
            names,
            request,
            tools,
            reasoning_open,
            symphony::GrammarStart::Unknown,
            &templates(),
        )
        .expect("no hf")
    }

    /// Where the model's workers start the grammar shapes what precedes a table's answer or its
    /// calls when the prompt leaves the thought open: nothing after the reasoning parser's gate
    /// (a JSON answer then needs no tag; the request's own schema goes to the engine as it is),
    /// the thought closed first from the first token, and the free-text prefix when the workers
    /// do not say.
    #[test]
    fn the_workers_grammar_start_shapes_what_precedes_a_tables_answer_and_calls() {
        let deepseek = names(Some("deepseek_v4"), Some("deepseek_v4"));
        let answer = request(json!({"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "city", "schema": {"type": "object"}},
        }}));
        let tag_of = |start| {
            super::constraint(
                ParserChoice::Symphony,
                &deepseek,
                &answer,
                &[],
                true,
                start,
                &templates(),
            )
            .expect("no hf")
            .map(|constraint| match constraint {
                ToolConstraint::StructuralTag(tag) => {
                    serde_json::from_str::<Value>(&tag).expect("JSON")
                }
                ToolConstraint::JsonSchema(schema) => {
                    panic!("a structural tag, not the JSON schema {schema}")
                }
            })
        };
        let unknown = tag_of(symphony::GrammarStart::Unknown).expect("a tag with the prefix");
        assert_eq!(unknown["format"]["elements"][0]["type"], "any_text");
        assert_eq!(unknown["format"]["elements"][1]["type"], "json_schema");
        assert_eq!(
            tag_of(symphony::GrammarStart::AfterReasoning),
            None,
            "after the gate the schema stands alone, and needs no tag"
        );
        let first = tag_of(symphony::GrammarStart::FirstToken).expect("a tag");
        assert_eq!(first["format"]["elements"][0]["type"], "tag");
        assert_eq!(first["format"]["elements"][0]["begin"], "");
        assert_eq!(first["format"]["elements"][0]["end"], "</think>");
        assert_eq!(first["format"]["elements"][1]["type"], "json_schema");
        let forced = request(json!({"tool_choice": "required"}));
        let call_of = |start| {
            let Some(ToolConstraint::StructuralTag(tag)) = super::constraint(
                ParserChoice::Symphony,
                &deepseek,
                &forced,
                &tools(),
                true,
                start,
                &templates(),
            )
            .expect("no hf") else {
                panic!("a forced tag");
            };
            serde_json::from_str::<Value>(&tag).expect("JSON")
        };
        assert_eq!(
            call_of(symphony::GrammarStart::Unknown)["format"]["elements"][0]["type"],
            "any_text"
        );
        let bare = call_of(symphony::GrammarStart::AfterReasoning);
        assert_eq!(
            bare["format"]["elements"][0]["type"], "const_string",
            "the block's opener comes first at the gate: {bare}"
        );
        let first = call_of(symphony::GrammarStart::FirstToken);
        assert_eq!(first["format"]["elements"][0]["end"], "</think>");
        assert_eq!(first["format"]["elements"][1], bare["format"]);
    }

    /// The model's workers name the start when they all carry the same label; none, a missing
    /// label or a disagreement leaves it unknown. A worker's own label maps one to one.
    #[test]
    fn the_models_workers_name_the_grammar_start_when_they_agree() {
        let worker = |start: Option<&str>| -> Arc<dyn Worker> {
            let mut builder =
                BasicWorkerBuilder::new("grpc://engine:50051").worker_type(WorkerType::Regular);
            if let Some(start) = start {
                builder = builder.label(WorkerGrammarStart::LABEL, start);
            }
            Arc::new(builder.build())
        };
        assert_eq!(grammar_start_of(&[]), symphony::GrammarStart::Unknown);
        assert_eq!(
            grammar_start_of(&[worker(None)]),
            symphony::GrammarStart::Unknown
        );
        assert_eq!(
            grammar_start_of(&[
                worker(Some("after_reasoning")),
                worker(Some("after_reasoning"))
            ]),
            symphony::GrammarStart::AfterReasoning
        );
        assert_eq!(
            grammar_start_of(&[worker(Some("first_token"))]),
            symphony::GrammarStart::FirstToken
        );
        assert_eq!(
            grammar_start_of(&[worker(Some("after_reasoning")), worker(Some("first_token"))]),
            symphony::GrammarStart::Unknown
        );
        assert_eq!(
            grammar_start_of(&[worker(Some("after_reasoning")), worker(None)]),
            symphony::GrammarStart::Unknown
        );
        // A prefill worker generates no token a grammar constrains: its label does not count.
        let prefill = |start: Option<&str>| -> Arc<dyn Worker> {
            let mut builder =
                BasicWorkerBuilder::new("grpc://prefill:50051").worker_type(WorkerType::Prefill);
            if let Some(start) = start {
                builder = builder.label(WorkerGrammarStart::LABEL, start);
            }
            Arc::new(builder.build())
        };
        assert_eq!(
            grammar_start_of(&[
                worker(Some("after_reasoning")),
                prefill(Some("first_token"))
            ]),
            symphony::GrammarStart::AfterReasoning
        );
        // An encode worker (EPD) generates none either: its label does not count.
        let encode = |start: Option<&str>| -> Arc<dyn Worker> {
            let mut builder =
                BasicWorkerBuilder::new("grpc://encode:50051").worker_type(WorkerType::Encode);
            if let Some(start) = start {
                builder = builder.label(WorkerGrammarStart::LABEL, start);
            }
            Arc::new(builder.build())
        };
        assert_eq!(
            grammar_start_of(&[worker(Some("after_reasoning")), encode(Some("first_token"))]),
            symphony::GrammarStart::AfterReasoning
        );
        assert_eq!(
            grammar_start_of(&[encode(Some("first_token"))]),
            symphony::GrammarStart::Unknown
        );
        assert_eq!(
            grammar_start_of(&[worker(Some("after_reasoning")), prefill(None)]),
            symphony::GrammarStart::AfterReasoning
        );
        assert_eq!(
            grammar_start_of(&[prefill(Some("first_token"))]),
            symphony::GrammarStart::Unknown
        );
        assert_eq!(
            grammar_start_from(WorkerGrammarStart::AfterReasoning),
            symphony::GrammarStart::AfterReasoning
        );
        assert_eq!(
            grammar_start_from(WorkerGrammarStart::FirstToken),
            symphony::GrammarStart::FirstToken
        );
        assert_eq!(
            grammar_start_from(WorkerGrammarStart::Unknown),
            symphony::GrammarStart::Unknown
        );
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
        // A launcher's name for a family whose table has no older parser reads with the table:
        // in both flags, in the reasoning flag alone, or as the tool-call name beside another.
        assert_eq!(
            named_table(&names(Some("muse"), Some("muse"))),
            Some(Table::MuseSpark)
        );
        assert_eq!(
            named_table(&names(None, Some("muse"))),
            Some(Table::MuseSpark)
        );
        assert_eq!(
            named_table(&names(Some("muse"), Some("passthrough"))),
            Some(Table::MuseSpark)
        );
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
    fn every_name_a_table_answers_to_besides_its_own_is_a_legacy_parsers_or_a_launchers() {
        let tools = tool_parser::ParserFactory::new();
        let reasoning = reasoning_parser::ParserFactory::new();
        let symphony = ParserChoice::Symphony;
        let mut launchers = Vec::new();
        for (alias, table) in Table::ALIASES {
            if tools.has_parser(alias) || reasoning.registry().has_parser(alias) {
                // A legacy parser's name is checked as the legacy name it is
                // (`names_are_checked_against_both_registries_by_the_parser_in_force`).
                continue;
            }
            // A launcher's name for a family whose table has no older parser passes under
            // Symphony in both flags, in either alone, and beside the table's own name.
            launchers.push(alias);
            let doubled = names(Some(alias), Some(alias));
            assert_eq!(named_table(&doubled), Some(table), "{alias}");
            assert_eq!(check(symphony, &doubled), Ok(()), "{alias}");
            assert_eq!(
                check(symphony, &names(Some(alias), None)),
                Ok(()),
                "{alias}"
            );
            assert_eq!(
                check(symphony, &names(None, Some(alias))),
                Ok(()),
                "{alias}"
            );
            let own = table.name();
            assert_eq!(
                check(symphony, &names(Some(alias), Some(own))),
                Ok(()),
                "{alias}"
            );
            assert_eq!(
                check(symphony, &names(Some(own), Some(alias))),
                Ok(()),
                "{alias}"
            );
        }
        assert_eq!(launchers, ["muse"]);
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
        assert_eq!(check(legacy, &Names::default()), Ok(()));
        assert_eq!(check(legacy, &names(Some("qwen"), Some("qwen3"))), Ok(()));
        let why = check(legacy, &names(Some("nonsense"), None));
        assert_eq!(
            why.expect_err("unknown"),
            "unknown tool-call parser 'nonsense'"
        );
        let why = check(legacy, &names(None, Some("ling")));
        assert_eq!(
            why.expect_err("a table under legacy"),
            "unknown reasoning parser 'ling': the Symphony table 'ling', which --parser symphony \
             reads with"
        );
        // Under Symphony a table's name passes alone or doubled, and binds the other name.
        assert_eq!(check(symphony, &names(Some("ling"), None)), Ok(()));
        assert_eq!(check(symphony, &names(Some("ling"), Some("LING"))), Ok(()));
        // Legacy names pass as the legacy names they are, one a table's own name as well.
        assert_eq!(
            check(symphony, &names(Some("qwen"), Some("kimi_k3"))),
            Ok(())
        );
        let why = check(symphony, &names(Some("ling"), Some("qwen3")));
        assert_eq!(
            why.expect_err("bound to the table"),
            "tool-call parser 'ling' selects the Symphony table 'ling' and is no legacy parser's \
             name, so the other parser must name the same table or nothing, not 'qwen3'"
        );
        let why = check(symphony, &names(None, Some("nonsense")));
        assert_eq!(
            why.expect_err("unknown"),
            "unknown reasoning parser 'nonsense': neither a legacy parser nor a Symphony table"
        );
        // A legacy name a table answers to is checked as the legacy name it is: a tool parser's
        // given as the reasoning parser is unknown there, under either parser.
        let why = check(symphony, &names(None, Some("qwen_xml")));
        assert_eq!(
            why.expect_err("wrong kind"),
            "unknown reasoning parser 'qwen_xml': neither a legacy parser nor a Symphony table"
        );
        let why = check(legacy, &names(None, Some("qwen_xml")));
        assert_eq!(
            why.expect_err("wrong kind"),
            "unknown reasoning parser 'qwen_xml'"
        );
        // The legacy registries spell their names one way: a legacy parser's name in another
        // case or with spaces around it is no legacy parser's, and selects no table either
        // (a table's own name, which reads in any case, excepted).
        let why = check(symphony, &names(Some("QWEN_XML"), None));
        assert_eq!(
            why.expect_err("not as spelled"),
            "unknown tool-call parser 'QWEN_XML': neither a legacy parser nor a Symphony table"
        );
        let why = check(symphony, &names(None, Some(" hy_v4 ")));
        assert_eq!(
            why.expect_err("not as spelled"),
            "unknown reasoning parser ' hy_v4 ': neither a legacy parser nor a Symphony table"
        );
        // A launcher's name for a family whose table has no older parser (`muse`) passes in
        // either flag and binds the other name as the table's own name does; under `legacy` it
        // is unknown, as the table's own name is.
        assert_eq!(check(symphony, &names(Some("muse"), Some("muse"))), Ok(()));
        assert_eq!(
            check(symphony, &names(Some("MUSE"), Some("muse_spark"))),
            Ok(())
        );
        assert_eq!(
            check(symphony, &names(Some("muse_spark"), Some("muse"))),
            Ok(())
        );
        let why = check(symphony, &names(Some("muse"), Some("qwen3")));
        assert_eq!(
            why.expect_err("bound to the table"),
            "tool-call parser 'muse' selects the Symphony table 'muse_spark' and is no legacy \
             parser's name, so the other parser must name the same table or nothing, not 'qwen3'"
        );
        let why = check(symphony, &names(Some("muse"), Some("passthrough")));
        assert_eq!(
            why.expect_err("bound to the table"),
            "tool-call parser 'muse' selects the Symphony table 'muse_spark' and is no legacy \
             parser's name, so the other parser must name the same table or nothing, not \
             'passthrough'"
        );
        let why = check(legacy, &names(None, Some("muse")));
        assert_eq!(
            why.expect_err("a table under legacy"),
            "unknown reasoning parser 'muse': the Symphony table 'muse_spark', which --parser \
             symphony reads with"
        );
        // A registry that is absent checks nothing, as before.
        assert_eq!(
            check_names(symphony, &names(Some("nonsense"), Some("muse")), None, None),
            Ok(())
        );
        assert_eq!(
            check_names(legacy, &names(Some("muse"), Some("nonsense")), None, None),
            Ok(())
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

    /// The conversation of a retry: the user asked for `list_skills`, the assistant called it,
    /// the tool was unavailable, the user asks again.
    fn retry_messages() -> Value {
        json!([
            {"role": "user", "content": "Call the `list_skills` tool once to see what we can do, \
                                         and reply with nothing else."},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": "list_skills", "arguments": "{}"}
            }]},
            {"role": "tool", "tool_call_id": "call_0",
             "content": "`list_skills` is unavailable for now; call it again later."},
            {"role": "user", "content": "The cooldown is over. Try again."}
        ])
    }

    #[test]
    fn the_calls_stay_wherever_the_legacy_path_runs_a_tool_parser() {
        // The legacy path scans a response for calls when the request declares tools, an empty
        // list included, or when the model's provider reads calls without any; the table keeps
        // its calls exactly then, so a call the conversation established comes back as a call
        // under either parser, and a `tool_choice` of `none` drops them under either.
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
        let retry = request(json!({"model": "MiniMax-M3", "messages": retry_messages()}));
        assert!(scans_for_tool_calls(&retry));
        assert_eq!(
            assigned(&retry),
            table(&retry),
            "the provider reads calls without tools"
        );
        let refused = request(json!({
            "model": "MiniMax-M3", "messages": retry_messages(), "tool_choice": "none"
        }));
        assert_eq!(assigned(&refused), table(&refused).without_calls());
        let declared_none = request(json!({"tools": []}));
        assert!(scans_for_tool_calls(&declared_none));
        assert_eq!(
            assigned(&declared_none),
            table(&declared_none),
            "an empty list is a declaration"
        );
        let undeclared = request(json!({"messages": retry_messages()}));
        assert!(!scans_for_tool_calls(&undeclared));
        assert_eq!(
            assigned(&undeclared),
            table(&undeclared).without_calls(),
            "a provider that reads calls only with tools"
        );
        // A Messages request: declared tools, an empty list included, keep the calls; none
        // declared drops them, as the legacy path runs its tool parser on `tools` alone.
        let messages = |extra: Value| {
            assign_messages(
                ParserChoice::Symphony,
                &Names::default(),
                &message_request(extra),
                None,
                PROMPT,
                &templates(),
            )
            .expect("read")
            .expect("assigned")
            .format
        };
        assert!(messages(json!({"tools": []})).call_syntax().is_some());
        assert_eq!(messages(json!({})).call_syntax(), None);
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

    /// A JSON `response_format` takes the table's answer tag whatever names select the table,
    /// unless the tag is the schema alone, the engine's own constraint, which the engine keeps as
    /// before. A table whose answer is written inside a message (Muse Spark's, Kimi K3's,
    /// Inkling's) takes its tag: the engine's schema writes bare JSON from the turn's own state,
    /// which such a table drops as framing, so the content came back empty. A table whose output
    /// starts in content takes the tag with the thought's close before the schema when the prompt
    /// leaves the thought open, and with the thought the model may write first where its table
    /// admits one (MiniMax M3's); a table with no answer state and a prompt that continues the
    /// assistant's message keep the engine's schema.
    #[test]
    fn a_json_response_format_takes_the_tables_answer_tag_unless_it_is_the_schema_alone() {
        let schema = json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
                "population": {"type": "integer"}
            },
            "required": ["city", "country", "population"],
            "additionalProperties": false
        });
        let muse = names(Some("muse_spark"), Some("muse_spark"));
        let any_object = json!({"model": "muse-spark", "response_format": {"type": "json_object"}});
        let answer = |names: &Names, extra: Value, reasoning_open: bool| {
            super::constraint(
                ParserChoice::Symphony,
                names,
                &request(extra),
                &[],
                reasoning_open,
                symphony::GrammarStart::Unknown,
                &templates(),
            )
        };
        // The named table, for a model the registry does not know: `json_object` asks for any
        // object, written in the message to the user.
        let tag = tag_of(answer(&muse, any_object.clone(), false));
        assert_eq!(tag["format"]["type"], "tag", "{tag}");
        assert_eq!(tag["format"]["begin"], " to=user<|message|>", "{tag}");
        assert_eq!(
            tag["format"]["content"],
            json!({"type": "json_schema", "json_schema": {"type": "object"}}),
            "{tag}"
        );
        // `json_schema` asks for the request's schema, strict or not.
        for strict in [true, false] {
            let tag = tag_of(answer(
                &muse,
                json!({
                    "model": "muse-spark",
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "capital", "schema": schema, "strict": strict}
                    }
                }),
                false,
            ));
            assert_eq!(tag["format"]["begin"], " to=user<|message|>", "{tag}");
            assert_eq!(
                tag["format"]["content"]["json_schema"], schema,
                "strict {strict}: {tag}"
            );
        }
        // With the prompt inside the thought, the tag closes it before the message.
        let open = tag_of(answer(&muse, any_object.clone(), true));
        assert_ne!(open, tag, "{open}");
        let begin = open["format"]["begin"].as_str().unwrap_or_default();
        assert!(
            begin.ends_with(" to=user<|message|>") && begin.len() > " to=user<|message|>".len(),
            "{open}"
        );
        // The registry's tables whose answer is a message take theirs too.
        for model in ["moonshotai/Kimi-K3", "thinkingmachines/Inkling"] {
            let tag = tag_of(answer(
                &Names::default(),
                json!({"model": model, "response_format": {"type": "json_object"}}),
                false,
            ));
            assert_eq!(tag["format"]["type"], "tag", "{model}: {tag}");
            assert_eq!(
                tag["format"]["content"]["type"], "json_schema",
                "{model}: {tag}"
            );
        }
        // A table whose output starts in content (the registry's Qwen3 table) has the schema
        // alone for its tag with the thought closed: the engine's own constraint, which stands.
        let content_first = json!({"response_format": {"type": "json_object"}});
        assert!(answer(&Names::default(), content_first.clone(), false)
            .expect("read")
            .is_none());
        // With the prompt inside the thought, the schema follows the prefix a forced call takes,
        // free text that closes the thought, so the JSON is not written inside it.
        let led = tag_of(answer(&Names::default(), content_first, true));
        assert_eq!(led["format"]["type"], "sequence", "{led}");
        assert_eq!(led["format"]["elements"][0]["type"], "any_text", "{led}");
        assert_eq!(
            led["format"]["elements"][1],
            json!({"type": "json_schema", "json_schema": {"type": "object"}}),
            "{led}"
        );
        // A table that admits a thought before its answer (MiniMax M3's) keeps that tag with the
        // thought closed: the thought the model may write first, then the schema.
        let thought_first = tag_of(answer(
            &Names::default(),
            json!({"model": "MiniMaxAI/MiniMax-M3", "response_format": {"type": "json_object"}}),
            false,
        ));
        assert_eq!(
            thought_first["format"]["type"], "sequence",
            "{thought_first}"
        );
        assert_eq!(
            thought_first["format"]["elements"][0]["type"], "optional",
            "{thought_first}"
        );
        assert_eq!(
            thought_first["format"]["elements"][1],
            json!({"type": "json_schema", "json_schema": {"type": "object"}}),
            "{thought_first}"
        );
        // No answer state (xLAM's list), a continued assistant message, or no JSON asked: the
        // engine's schema, or nothing.
        assert!(answer(
            &names(Some("xlam"), None),
            json!({"response_format": {"type": "json_object"}}),
            false
        )
        .expect("read")
        .is_none());
        let continued = json!({
            "model": "muse-spark",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "{"}],
            "continue_final_message": true, "add_generation_prompt": false,
        });
        assert!(request(continued.clone()).continues_final_assistant());
        assert!(answer(&muse, continued, false).expect("read").is_none());
        assert!(answer(
            &muse,
            json!({"model": "muse-spark", "response_format": {"type": "text"}}),
            false
        )
        .expect("read")
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
            symphony::GrammarStart::Unknown,
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
            symphony::GrammarStart::Unknown,
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
            symphony::GrammarStart::Unknown,
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
            symphony::GrammarStart::Unknown,
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
                    symphony::GrammarStart::Unknown,
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
                symphony::GrammarStart::Unknown,
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
            symphony::GrammarStart::Unknown,
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
            symphony::GrammarStart::Unknown,
            false,
            &templates(),
        )
        .expect("no hf")
        .is_none());
    }

    /// A tokenizer whose decoder drops `<eos>` and `<bos>` as special tokens and keeps every
    /// other token, the markers of two tables among them.
    fn marker_tokenizer() -> llm_tokenizer::MockTokenizer {
        llm_tokenizer::MockTokenizer::new().with_tokens(&[
            ("<eos>", 1),
            ("<bos>", 2),
            ("<think>", 3),
            ("</think>", 4),
        ])
    }

    fn table_with_markers(open: &str, close: &str) -> Format {
        Format::new("markers")
            .terminal("open", open)
            .terminal("close", close)
            .state("content", symphony::Emits::Content)
            .state("reasoning", symphony::Emits::Reasoning)
            .transition("content", "open", "reasoning")
            .transition("reasoning", "close", "content")
    }

    #[test]
    fn a_table_whose_marker_the_decoder_drops_keeps_the_special_tokens() {
        let tokenizer = marker_tokenizer();
        // A marker the decoder drops with the special tokens: the table needs them kept.
        assert!(markers_hidden_by_the_decoder(
            &table_with_markers("<bos>", "<eos>"),
            &tokenizer
        ));
        // Markers the decoder keeps either way, and a marker the tokenizer does not know,
        // which decodes to nothing with or without them: read as before.
        assert!(!markers_hidden_by_the_decoder(
            &table_with_markers("<think>", "</think>"),
            &tokenizer
        ));
        assert!(!markers_hidden_by_the_decoder(
            &table_with_markers("<unknown>", "</unknown>"),
            &tokenizer
        ));
        // Through the names: Qwen3's table over this vocabulary needs nothing, and no table
        // (the legacy parsers, or `--parser legacy`) asks for nothing.
        assert!(!keeps_special_tokens(
            ParserChoice::Symphony,
            &names(Some("qwen3"), None),
            "Qwen/Qwen3-8B",
            &templates(),
            &tokenizer,
        ));
        assert!(!keeps_special_tokens(
            ParserChoice::Symphony,
            &names(Some("passthrough"), None),
            "Qwen/Qwen3-8B",
            &templates(),
            &tokenizer,
        ));
        assert!(!keeps_special_tokens(
            ParserChoice::Legacy,
            &names(Some("qwen3"), None),
            "Qwen/Qwen3-8B",
            &templates(),
            &tokenizer,
        ));
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
        let mut reader = Reader::new(&assignment, Arc::default()).expect("a reader");
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
        assert_eq!(check(symphony, &names(Some("hf"), None)), Ok(()));
        assert_eq!(check(symphony, &names(None, Some("hf"))), Ok(()));
        assert_eq!(check(symphony, &names(Some("HF"), Some(" hf "))), Ok(()));
        let why = check(symphony, &names(Some("hf"), Some("qwen3")));
        assert_eq!(
            why.expect_err("bound"),
            "tool-call parser 'hf' reads with the model's response template, so the other \
             parser must name 'hf' or nothing, not 'qwen3'"
        );
        let why = check(symphony, &names(Some("qwen"), Some("hf")));
        assert_eq!(
            why.expect_err("bound"),
            "reasoning parser 'hf' reads with the model's response template, so the other \
             parser must name 'hf' or nothing, not 'qwen'"
        );
        let why = check(ParserChoice::Legacy, &names(Some("hf"), None));
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
                symphony::GrammarStart::Unknown,
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
                symphony::GrammarStart::Unknown,
                &templates(),
            )
            .expect("read")
            .is_none(),
            "the Qwen3 table's answer tag is the schema alone, the engine's own constraint, so the \
             older registry's JSON schema stands"
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
                symphony::GrammarStart::Unknown,
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
            symphony::GrammarStart::Unknown,
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
        let mut reader = Reader::new(&assignment, tools().into()).expect("a reader");
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
            let mut reader = Reader::new(&assignment, tools().into())
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
        let mut reader = Reader::new(&assignment, Arc::default()).expect("a reader");
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

    #[test]
    fn the_selected_workers_policy_decides_once() {
        use ConstrainedDecoding::{Always, Requested};
        use GrammarStart::{AfterReasoning, FirstToken, Unknown};

        let tag = |text: &str| ToolConstraint::StructuralTag(text.to_string());
        let refused = |why: &str| Err::<Option<ToolConstraint>, String>(why.to_string());
        // What one decision gives (the tag's text, and whether it is the whole turn's), and which
        // of the whole turn's grammar and the `auto` tag it built.
        let decide = |policy, start, prepared: PreparedConstraint, turn: PreparedConstraint| {
            let built = std::cell::RefCell::new(Vec::new());
            let decided = worker_constraint(
                WorkerDecoding { policy, start },
                prepared,
                || {
                    built.borrow_mut().push("turn");
                    turn
                },
                || {
                    built.borrow_mut().push("auto");
                    Ok(Some(tag("auto")))
                },
            )
            .map(|decided| {
                let text = decided.constraint.map(|constraint| constraint.to_tuple().1);
                (text, decided.whole_turn, decided.by_policy)
            });
            (decided, built.into_inner())
        };
        // The `auto` tag is the policy's choice; what preparation found is the request's.
        let found = |text: &str| Ok((Some(text.to_string()), false, text == "auto"));

        // `requested`: what preparation found, its refusal included; nothing else is built.
        for start in [FirstToken, AfterReasoning, Unknown] {
            assert_eq!(
                decide(
                    Requested,
                    start,
                    Ok(Some(tag("prepared"))),
                    Ok(Some(tag("turn")))
                ),
                (found("prepared"), vec![])
            );
            assert_eq!(
                decide(Requested, start, refused("no tag"), Ok(Some(tag("turn")))),
                (Err("no tag".to_string()), vec![])
            );
            assert_eq!(
                decide(Requested, start, Ok(None), Ok(Some(tag("turn")))),
                (Ok((None, false, false)), vec![])
            );
        }
        // `always` from the first token: the whole turn's grammar stands over what preparation
        // found or refused, and its refusal refuses the request.
        assert_eq!(
            decide(Always, FirstToken, refused("no tag"), Ok(Some(tag("turn")))),
            (Ok((Some("turn".to_string()), true, true)), vec!["turn"])
        );
        assert_eq!(
            decide(Always, FirstToken, Ok(None), refused("cannot hold")),
            (Err("cannot hold".to_string()), vec!["turn"])
        );
        // Where the whole turn's grammar does not start, and on an engine that starts the grammar
        // later or does not say: what preparation found or refused, else the `auto` tag.
        assert_eq!(
            decide(Always, FirstToken, Ok(Some(tag("prepared"))), Ok(None)),
            (found("prepared"), vec!["turn"])
        );
        assert_eq!(
            decide(Always, FirstToken, refused("no tag"), Ok(None)),
            (Err("no tag".to_string()), vec!["turn"])
        );
        assert_eq!(
            decide(Always, FirstToken, Ok(None), Ok(None)),
            (found("auto"), vec!["turn", "auto"])
        );
        for start in [AfterReasoning, Unknown] {
            assert_eq!(
                decide(Always, start, refused("no tag"), Ok(Some(tag("turn")))),
                (Err("no tag".to_string()), vec![])
            );
            assert_eq!(
                decide(
                    Always,
                    start,
                    Ok(Some(tag("prepared"))),
                    Ok(Some(tag("turn")))
                ),
                (found("prepared"), vec![])
            );
            assert_eq!(
                decide(Always, start, Ok(None), Ok(Some(tag("turn")))),
                (found("auto"), vec!["auto"])
            );
        }
        // Where the table gives no `auto` tag, the policy chose nothing.
        let nothing = worker_constraint(
            WorkerDecoding {
                policy: Always,
                start: AfterReasoning,
            },
            Ok(None),
            || Ok(None),
            || Ok(None),
        )
        .unwrap();
        assert!(nothing.constraint.is_none() && !nothing.by_policy);
    }

    /// A table whose calls spell a whole turn: a call's header names the tool, its parameters
    /// are tags around raw text, and a row out of a call goes on to the next or stops.
    fn turn_table() -> Format {
        let source = json!({
          "start_anchor_pattern": "<\\|head\\|>assistant",
          "fields": {
            "thinking": {"open": "<|via|>notes<|body|>", "close": "<|stop|><|head|>assistant",
                         "content": "text"},
            "content": {"open": "<|via|>reply<|body|>", "close": ["<|done|>", "<|stop|>", "<eos>"],
                        "content": "text"},
            "tool_calls": {
              "open_pattern": " to=(?P<name>[^\\s<]+)<\\|via\\|>notes <\\|as\\|>xml<\\|body\\|>",
              "close": ["<|stop|><|head|>assistant", "<eos>"], "repeats": true,
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
        });
        let template = smg_response_template::load_response_template(&source).expect("template");
        smg_response_template::to_symphony::compile(&template, &["<|done|>", "<eos>"])
            .expect("compiled")
            .format
    }

    fn weather() -> Value {
        json!([{"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
    }

    /// The format of the whole turn's grammar a chat request takes over [`turn_table`], the prompt
    /// ending at the turn's opener, then at `continued` when the request continues an assistant
    /// message; the error is the refusal.
    fn turn(mut extra: Value, continued: Option<&str>) -> Result<Option<Value>, String> {
        if continued.is_some() {
            extra["continue_final_message"] = json!(true);
            extra["messages"] = json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "partial"}
            ]);
        }
        let prompt = format!("user<|head|>assistant{}", continued.unwrap_or(""));
        let constraint = chat_turn(&turn_table(), &request(extra), &prompt)?;
        Ok(constraint.map(|constraint| tag_of(Ok(Some(constraint)))["format"].clone()))
    }

    fn whole(extra: Value, continued: Option<&str>) -> Value {
        turn(extra, continued).expect("held").expect("a grammar")
    }

    /// The part of a turn's format after its thought: the answer, the calls, or either.
    fn after_thought(format: &Value) -> &Value {
        &format["elements"][1]
    }

    #[test]
    fn the_whole_turn_holds_the_answer_and_the_calls() {
        let free = whole(json!({}), None);
        assert_eq!(
            after_thought(&free)["elements"][1],
            json!({"type": "any_text", "excludes": ["<|done|>", "<|stop|>"]})
        );
        assert_eq!(
            after_thought(&whole(json!({"tools": weather()}), None))["type"],
            "or"
        );
        let required = whole(json!({"tools": weather(), "tool_choice": "required"}), None);
        assert_eq!(after_thought(&required)["type"], "sequence");
        let json_answer = whole(json!({"response_format": {"type": "json_object"}}), None);
        assert_eq!(
            after_thought(&json_answer)["elements"][1],
            json!({"type": "json_schema", "json_schema": {"type": "object"}})
        );
        let regex = whole(json!({"regex": "(yes|no)"}), None);
        assert_eq!(
            after_thought(&regex)["elements"][1],
            json!({"type": "regex", "pattern": "(yes|no)"})
        );
        let ebnf = whole(json!({"ebnf": "root ::= \"y\""}), None);
        assert_eq!(after_thought(&ebnf)["elements"][1]["type"], "grammar");
        // A forced call writes no answer, so it leaves the answer's format out, Lark's included.
        assert_eq!(
            turn(
                json!({"tools": weather(), "tool_choice": "required", "ebnf": "start: \"y\""}),
                None
            ),
            Ok(Some(required))
        );
    }

    #[test]
    fn the_prompt_says_where_the_turn_starts() {
        let thinking = whole(json!({}), Some("<|via|>notes<|body|>Let me"));
        assert_eq!(thinking["elements"][0]["begin"], "");
        let after = whole(
            json!({}),
            Some("<|via|>notes<|body|>Plan.<|stop|><|head|>assistant"),
        );
        assert_eq!(after["elements"][0]["value"], "<|via|>reply<|body|>");

        // Inside an answer: the rest of a free one; a forced call or an answer's format starts
        // no grammar there, and the table's tags decide (`worker_constraint`).
        let answering = Some("<|via|>reply<|body|>The answer");
        assert_eq!(
            whole(json!({}), answering)["elements"][0]["type"],
            "any_text"
        );
        for extra in [
            json!({"tools": weather(), "tool_choice": "required"}),
            json!({"tools": weather(), "tool_choice": {"type": "allowed_tools",
                "mode": "required", "tools": [{"type": "function", "name": "get_weather"}]}}),
            json!({"response_format": {"type": "json_object"}}),
        ] {
            assert_eq!(turn(extra, answering), Ok(None));
        }
        // Past text no field holds, no grammar starts.
        assert_eq!(turn(json!({"tools": weather()}), Some("Hello")), Ok(None));
        // Inside a call: the rest of it is free.
        let calling = " to=get_weather<|via|>notes <|as|>xml<|body|><arg name=\"city\">\nPa";
        assert_eq!(
            whole(json!({"tools": weather()}), Some(calling))["elements"][0]["type"],
            "any_text"
        );
    }

    #[test]
    fn the_whole_turn_refuses_what_it_cannot_hold() {
        let refused = |extra: Value| turn(extra, None).is_err();
        let served = |extra: Value| turn(extra, None).expect("held").is_some();
        let named = |name: &str| {
            json!([{"type": "function", "function": {"name": name, "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}}}}])
        };
        // A name a call cannot carry, whatever the choice.
        for choice in [json!("auto"), json!("none"), json!("required")] {
            assert!(refused(
                json!({"tools": named("get weather"), "tool_choice": choice})
            ));
        }
        let key = json!([{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"say \"hi\"": {"type": "string"}}}}}]);
        assert!(refused(json!({"tools": key})));

        // A strict tool whose calls the grammar would hold more loosely, while it is callable.
        let tuple = |strict: bool| {
            json!([{"type": "function", "function": {"name": "f", "strict": strict,
                "parameters": {"type": "object", "properties": {"p": {"type": "tuple"}},
                    "required": ["p"]}}}])
        };
        assert!(refused(json!({"tools": tuple(true)})));
        assert!(served(json!({"tools": tuple(false)})));
        assert!(served(json!({"tools": tuple(true), "tool_choice": "none"})));

        // Under `auto`, a strict tool or one call at most beside the request's own grammar; and
        // an ebnf in Lark's syntax.
        let strict = json!([{"type": "function", "function": {"name": "f", "strict": true,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
        for extra in [
            json!({"tools": strict, "regex": "(a|b)"}),
            json!({"tools": strict, "ebnf": "root ::= \"y\""}),
            json!({"tools": weather(), "parallel_tool_calls": false, "regex": "(a|b)"}),
            json!({"ebnf": "start: \"y\""}),
            json!({"tools": weather(), "ebnf": "start: \"y\""}),
        ] {
            assert!(refused(extra.clone()), "{extra}");
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
        let why = turn(json!({"ebnf": "start: \"y\""}), None).expect_err("refused");
        assert!(why.contains("Lark"), "{why}");
    }

    #[test]
    fn over_its_limits_the_whole_turn_leaves_the_tags_to_decide() {
        let properties: serde_json::Map<String, Value> = (0
            ..=symphony::OutputSpelling::MAX_TOOL_PARAMETERS)
            .map(|i| (format!("p{i}"), json!({"type": "string"})))
            .collect();
        let wide = json!([{"type": "function", "function": {"name": "wide",
            "parameters": {"type": "object", "properties": properties}}}]);
        assert_eq!(turn(json!({"tools": wide}), None), Ok(None));

        let huge = "x".repeat(symphony::OutputSpelling::MAX_TAG_BYTES);
        let large = json!([{"type": "function", "function": {"name": "large",
        "parameters": {"type": "object", "properties": {
            "p": {"type": "string", "enum": [huge]}
        }}}}]);
        assert_eq!(turn(json!({"tools": large}), None), Ok(None));
    }

    #[test]
    fn a_messages_choice_takes_the_turn_its_chat_twin_takes() {
        let tools = json!([{"name": "get_weather", "input_schema": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]);
        let prompt = "user<|head|>assistant";
        for (choice, chat) in [
            (json!({"type": "auto"}), json!({})),
            (json!({"type": "any"}), json!({"tool_choice": "required"})),
            (
                json!({"type": "any", "disable_parallel_tool_use": true}),
                json!({"tool_choice": "required", "parallel_tool_calls": false}),
            ),
            (
                json!({"type": "tool", "name": "get_weather"}),
                json!({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}),
            ),
            (json!({"type": "none"}), json!({"tool_choice": "none"})),
        ] {
            let messages = message_request(json!({"tools": tools, "tool_choice": choice}));
            let mut chat = chat;
            chat["tools"] = weather();
            let tag = |constraint: Result<Option<ToolConstraint>, String>| {
                tag_of(constraint)["format"].clone()
            };
            assert_eq!(
                tag(messages_turn(
                    &turn_table(),
                    &messages,
                    &messages.chat_tools(),
                    prompt,
                    false
                )),
                tag(chat_turn(&turn_table(), &request(chat), prompt)),
                "{choice}"
            );
        }
    }

    #[test]
    fn only_the_response_templates_table_spells_a_whole_turn() {
        // Any other name, and the legacy parsers, leave the turn to the table's tags without
        // looking for a template.
        let request = request(json!({"tools": weather(), "tool_choice": "required"}));
        for (parser, names) in [
            (ParserChoice::Symphony, names(Some("qwen"), None)),
            (ParserChoice::Symphony, names(None, None)),
            (ParserChoice::Legacy, names(Some("hf"), None)),
        ] {
            assert!(matches!(
                turn_constraint(parser, &names, &request, PROMPT, &templates()),
                Ok(None)
            ));
        }
        // The template's names ask for the model's own template, and a model without one is
        // refused, as the table's tags refuse it.
        let why = turn_constraint(
            ParserChoice::Symphony,
            &names(Some("hf"), None),
            &request,
            PROMPT,
            &templates(),
        )
        .expect_err("refused");
        assert_eq!(why, "no tokenizer is registered for model 'Qwen/Qwen3-8B'");
    }
}
