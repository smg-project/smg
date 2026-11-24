//! What the chat and Messages stages decide alike about how the response is
//! read (see `regular::symphony`): in preparation, the constraint the request
//! asks for before its worker is known and whether the decoder keeps the
//! special tokens the reader needs ([`prepare`]); in request building, whether
//! the rendered prompt leaves the model inside its thought
//! ([`reasoning_decision`]), what constrains the output on the selected worker
//! and which table reads the response ([`decide`]). The two stages spell what
//! their request asks ([`Asks`]) and decide through these functions; each
//! keeps what differs between the APIs, the tokenizer-driven reasoning switch
//! and the request the `auto` tag derives for.

use std::sync::Arc;

use ::symphony::formats::json::required_calls_schema;
use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{
    common::{Tool, ToolChoice},
    profile::ProviderProfile,
};
use smg_worker::identity::basic::Worker;
use tracing::{debug, error};

use crate::{
    config::ParserChoice,
    routers::{
        common::snapshot::{ModelLabels, WorkerRegistryExt},
        error,
        grpc::{
            common::stages::helpers::TagForStart,
            context::RequestContext,
            regular::symphony::{
                self, Asks, Assignment, AutoRequest, PreparedConstraint, PromptReplay, Reading,
                ToolConstraint,
            },
            utils::{self, Names},
        },
    },
};

/// What preparation found for the response.
pub(crate) struct Prepared {
    /// The constraint for what the request asks, before its worker is known:
    /// the table's tag for a forced call, the call's JSON schema where that
    /// tag opens with free text, else the older registry's; a JSON
    /// answer's tag as it derives for a worker's start, or its schema itself
    /// where no table reads a Messages request's answer; or why the table
    /// cannot constrain the request. Request building decides on the selected
    /// worker (`regular::symphony::worker_constraint`).
    pub constraint: PreparedConstraint,
    /// The constraint is a JSON schema no table built (a forced call's,
    /// whose JSON has none of the table's markers, or the request's own
    /// answer schema where no table reads), told by its origin: a schema a
    /// table built keeps the table's reading and its markers, whatever its
    /// kind.
    pub json_schema: bool,
    /// The decoder keeps the special tokens the reader of the response needs:
    /// a table's markers, or a legacy reasoning parser's.
    pub keeps_special_tokens: bool,
    /// The parser names the model's cards set, resolved here once for the request: the stage
    /// stores them (`ProcessingState::parser_names`) and request building reads them in
    /// place of a second resolution.
    pub names: Names,
    /// The table's replay of the rendered prompt, asked for the forced call's tag whatever the
    /// request's thinking switch says: the stage stores it (`ProcessingState::prompt_thought`)
    /// and request building's reasoning decision reads it in place of a second replay of the
    /// same prompt.
    pub prompt_thought: PromptReplay,
    /// The table decision for the request (`regular::symphony::reading`: which table reads the
    /// response under the parser, or why the legacy parsers do), read here once for the replay,
    /// the constraint and the decoder's special tokens: the stage stores it
    /// (`ProcessingState::reading`) and request building's assignment reads it in place of a
    /// fourth decision over the same names; the whole turn's grammar and the `auto` tag still
    /// decide for the selected worker. `None` when the names refused (the constraint carries the
    /// refusal).
    pub reading: Option<Reading>,
}

/// The constraint preparation finds for what the request asks, and whether
/// the decoder keeps the special tokens the reader needs. When Symphony reads
/// the response, its table gives a forced call, or a JSON `response_format`,
/// its tag, so the output comes back in the syntax the table parses
/// (`regular::symphony::constraint_for`, over the table decision the
/// reader is assigned by; the answer's tag derives for the selected worker at
/// request building); the older registry decides for every other request
/// (`generate_tool_constraint`, over `tools`, the ones the request's choice
/// lets the model call, and the legacy tool parser's name). A table's tag
/// for a forced call that opens with free text gives way to the forced
/// call's JSON schema (`forced_call_schema`), and stands where the schema
/// cannot be built or loses what a reference in the tools' parameters
/// points at (`regular::symphony::keeps_references`). On a thinking
/// prompt a parser with a reasoning prefix gets its tag wrapped so a forced
/// call follows the reasoning instead of preempting it: `switch` is the API's
/// tokenizer-driven answer, and where a table reads the response its replay
/// of the rendered prompt decides by the rule request building asks
/// (`PromptThought::or_switch` over `regular::symphony::reasoning_open`): a
/// turn the prompt leaves unopened starts inside the thought whatever the
/// switch says, a prompt that closed the thought is past it whatever the
/// switch says, and where the family's rule sets the switch aside the replay
/// decides alone; so the tag reads the prompt as the engine will. The
/// table's refusal waits for request building, which decides
/// on the selected worker; the registry's refuses the request here (a 400,
/// logged under `stage`). A table whose markers the decoder would drop as
/// special tokens needs them kept whatever the request asks of the channels:
/// without them the table reads neither the thought nor the calls
/// (`regular::symphony::keeps_special_tokens_for`), under a JSON-schema
/// constraint too, where the table still reads the thought ahead of the
/// forced call's JSON (`regular::symphony::assign_for`). Where no table reads
/// at all the legacy parsers answer for themselves: a reasoning parser that
/// needs its control tokens, for a request that separates the reasoning.
pub(crate) fn prepare(
    ctx: &RequestContext,
    asks: Asks<'_>,
    tools: &[Tool],
    processed_text: &str,
    switch: bool,
    tokenizer: &Arc<dyn Tokenizer>,
    stage: &'static str,
) -> Result<Prepared, Response> {
    let model = asks.model();
    // Only the chat and messages preparations choose a parser; a request
    // that chose none (transcription shares the chat preparation) keeps the
    // legacy parsers here and at the two sites below, whatever the enum's
    // default.
    let parser = ctx.state.parser.unwrap_or(ParserChoice::Legacy);
    // What the model's workers say together, as the registry's membership
    // snapshot keeps it (`together`): the cards' parser names, and for a
    // forced call's tag the start and the vocabulary they name together; a
    // JSON answer's tag and the `auto` tag take the selected worker's own
    // start at request building.
    let labels = together(ctx, model, ModelLabels::clone);
    let names = ctx.components.parser_resolver.names_of(Names {
        tool: labels.tool_parser,
        reasoning: labels.reasoning_parser,
    });
    // Which table reads the response under the parser, decided once for the
    // request: the replay, the constraint and the decoder's special tokens
    // read it here, the assignment at request building after them. A refusal
    // (the names ask for a response template the model cannot compile) is the
    // constraint's to carry, as before.
    let reading = symphony::reading(parser, &names, model, &ctx.components.templates);
    // The replay is asked whatever the thinking switch says: a template may
    // close the thought for a request that names no switch while the switch's
    // default says thinking, and a prompt that closed the thought is past it.
    let prompt_thought = reading
        .as_ref()
        .ok()
        .and_then(|read| symphony::reasoning_open_for(read, processed_text));
    // The prompt closed the thought itself and wrote the template's text
    // after it (thinking switched off): the calls' tag owes none of it, and
    // no thought stands in front of the calls.
    let thought_closed = prompt_thought.is_some_and(|thought| thought.closed);
    // The one rule request building asks (`reasoning_decision`): the replay's
    // answer, with the switch only where the replay does not decide alone; the
    // switch itself where no table replays the prompt.
    let reasoning = prompt_thought.map_or(switch, |thought| thought.or_switch(|| switch));
    let from_the_table = reading.as_ref().map_err(Clone::clone).and_then(|read| {
        symphony::constraint_for(
            read,
            asks,
            tools,
            reasoning,
            thought_closed,
            labels.grammar_start,
            labels.vocabulary,
        )
    });
    let choice = asks.choice();
    // A table's tag that opens with free text does not hold the model to the
    // call, so the forced call's JSON schema goes out in its place. No
    // parser is named for it: a parser's own tag opens the same way and is
    // not spelled for the workers' vocabulary. A request that schema cannot
    // spell keeps the table's tag: one that cannot be built, or one
    // that loses what a reference in the tools' parameters points at.
    let from_the_table = from_the_table.map(|found| {
        found.held_by(|| {
            let schema = forced_call_schema(tools, choice.as_deref()?)
                .inspect_err(|e| {
                    debug!(
                        function = stage,
                        error = %e,
                        "No JSON schema for the forced call; the table's tag stands"
                    );
                })
                .ok()
                .flatten()?;
            if symphony::keeps_references(&schema, tools) {
                Some(schema)
            } else {
                debug!(
                    function = stage,
                    "The JSON schema for the forced call loses a reference's target; the \
                     table's tag stands"
                );
                None
            }
        })
    });
    let constraint = match (from_the_table, choice.as_deref()) {
        (Ok(symphony::Prepared::None), Some(tool_choice)) if !tools.is_empty() => {
            let constraint = ctx
                .components
                .tool_parser_factory
                .registry()
                .generate_tool_constraint(
                    ctx.components.parser_resolver.legacy_tool(&names),
                    tools,
                    tool_choice,
                    reasoning,
                )
                .map_err(|e| {
                    error!(function = stage, error = %e, "Invalid tool configuration");
                    error::bad_request(
                        "invalid_tool_configuration",
                        format!("Invalid tool configuration: {e}"),
                    )
                })?;
            Ok(symphony::Prepared::from(
                constraint.map(ToolConstraint::from),
            ))
        }
        (prepared, _) => prepared,
    };
    // Where nothing above constrains the answer, a Messages request's answer
    // schema is the engine's own JSON-schema constraint, as a chat request's
    // `response_format` is through its request builder; a table that reads
    // the response gave the schema its tag above (`Prepared::Answer`), and a
    // request whose choice forces a call takes none (the turn has no text
    // answer to format, whether or not the call has a constraint).
    let constraint = match constraint {
        Ok(symphony::Prepared::None) => Ok(symphony::Prepared::from(
            asks.untabled_answer_schema(tools)
                .map(|schema| ToolConstraint::JsonSchema(schema.to_string())),
        )),
        prepared => prepared,
    };
    let json_schema = matches!(
        &constraint,
        Ok(symphony::Prepared::Constraint { constraint, table: false, .. })
            if constraint.is_json_schema()
    );
    let keeps_special_tokens = match reading.as_ref().ok().and_then(|read| {
        symphony::keeps_special_tokens_for(read, model, &ctx.components.templates, tokenizer)
    }) {
        Some(kept) => kept,
        None => {
            !asks.no_reasoning()
                && utils::reasoning_parser_requires_special_tokens(
                    &ctx.components.reasoning_parser_factory,
                    ctx.components
                        .parser_resolver
                        .reasoning_parser_for(Some(&names.labels), model)
                        .as_deref(),
                    model,
                )
        }
    };
    Ok(Prepared {
        constraint,
        json_schema,
        keeps_special_tokens,
        names: names.labels,
        prompt_thought,
        reading: reading.ok(),
    })
}

/// The JSON schema a forced call is written under where no tag holds it: the
/// named function's parameters as they are, or the list a `required` choice
/// asks for. `None` when the choice forces no call.
fn forced_call_schema(
    tools: &[Tool],
    choice: &ToolChoice,
) -> Result<Option<ToolConstraint>, String> {
    let Some(first) = tools.first() else {
        return Ok(None);
    };
    if !choice.forces_call() {
        return Ok(None);
    }
    let schema = match choice {
        ToolChoice::Function { .. } => first.function.parameters_schema().to_string(),
        _ => required_calls_schema(tools)?.to_string(),
    };
    Ok(Some(ToolConstraint::JsonSchema(schema)))
}

/// What the model's workers say together to a request, before one of them is
/// selected: the parser names their cards set
/// (`utils::ParserResolver::card_names_among`), where their engines start a
/// grammar (`regular::symphony::grammar_start_of`) and the tag vocabulary
/// they take (`regular::symphony::vocabulary_of`). A worker's cards and
/// labels stay as they are while it is a member, so the registry's
/// membership snapshot keeps the three answers
/// (`ModelWorkerSnapshot::labels`): the first request after a membership
/// change walks the workers, and every request after it reads the kept
/// answers, whatever the size of the fleet. A request that spells the model
/// otherwise than the kept answers do, or names a model no worker serves,
/// walks the workers itself. `take` reads what its caller needs of the
/// answers.
fn together<T>(ctx: &RequestContext, model: &str, take: impl FnOnce(&ModelLabels) -> T) -> T {
    let read = |workers: &[Arc<dyn Worker>]| {
        let Names { tool, reasoning } = utils::ParserResolver::card_names_among(workers, model);
        ModelLabels {
            tool_parser: tool,
            reasoning_parser: reasoning,
            grammar_start: symphony::grammar_start_of(workers),
            vocabulary: symphony::vocabulary_of(workers),
        }
    };
    let snapshot = ctx
        .components
        .worker_registry
        .model_routing_snapshot(&ctx.input.model_id);
    match snapshot.as_deref() {
        Some(snapshot) => match snapshot.labels(model, read) {
            Some(kept) => take(kept),
            None => take(&read(snapshot)),
        },
        None => take(&read(&[])),
    }
}

/// Whether the rendered prompt leaves the model inside its thought, as the
/// request's grammar, the plan's reasoning state and the response spec all
/// read it: where a Symphony table reads the response, the table's own replay
/// of the prompt (a turn the prompt leaves unopened starts inside the thought:
/// `regular::symphony::reasoning_open`) with the tokenizer-driven `switch`
/// (the API's: `utils::chat_reasoning_starts_in_rendered_prefill`,
/// `utils::messages_reasoning_starts_in_prefill`, over `model`'s tokenizer),
/// or alone where the family's rule sets the switch aside (Gemma 4 prefills
/// an open channel only after a tool response) or the prompt closed the
/// thought itself, and no legacy parser is built to be asked; otherwise
/// native Gemma4 channels follow the rendered prompt
/// when that parser decides, and other parsers keep their tokenizer-driven
/// preference for the engine's reasoning gate. With it, whether the prompt
/// closed the thought itself and wrote the template's text after it
/// (`PromptThought::closed`), which the `auto` tag then owes nothing of.
pub(crate) fn reasoning_decision(
    ctx: &RequestContext,
    model: &str,
    processed_text: &str,
    switch: impl FnOnce(&dyn Tokenizer) -> bool,
) -> (bool, bool) {
    let switch = || {
        ctx.tokenizer_arc()
            .is_some_and(|tokenizer| switch(tokenizer.as_ref()))
    };
    let parser = ctx.state.parser.unwrap_or(ParserChoice::Legacy);
    // Preparation's replay of the same prompt stands where a table reads the response: the
    // names are the request's (`parser_names`), so the table is the same; where the assignment
    // left the response to the legacy parsers, no table replays.
    let replay = match (parser, ctx.state.prompt_thought) {
        (ParserChoice::Symphony, Some(replayed)) => replayed,
        _ => symphony::reasoning_open(
            parser,
            &ctx.components
                .parser_resolver
                .names_for(ctx.state.parser_names.as_ref(), model),
            model,
            &ctx.components.templates,
            processed_text,
        ),
    };
    match replay {
        Some(thought) => (thought.or_switch(switch), thought.closed),
        None => (
            utils::rendered_prompt_reasoning_in_prefill(
                &ctx.components.reasoning_parser_factory,
                ctx.components
                    .parser_resolver
                    .reasoning_parser_for(ctx.state.parser_names.as_ref(), model)
                    .as_deref(),
                model,
                processed_text,
            )
            .unwrap_or_else(switch),
            false,
        ),
    }
}

/// What the table's `auto` tag derives for, when the worker's policy asks for
/// the tag ([`symphony::AutoTag`]): the request, with the tools the tag is
/// over, and whether the prompt leaves the model inside its thought, or
/// closed it itself and wrote the template's text after it.
pub(crate) struct Auto {
    pub request: AutoRequest,
    pub reasoning_open: bool,
    pub thought_closed: bool,
}

/// What [`decide`] decided for the response.
pub(crate) struct Decided {
    /// What derives the table's tag for a worker's start, where the table
    /// shapes it by the start (the `auto` tag the worker's policy chose, a
    /// JSON answer's): the attempt stamp carries it, so a retry gives the
    /// worker it selects the tag shaped for where its own engine starts the
    /// grammar (`helpers::ToolConstraintStamp`).
    pub table_tag: Option<Arc<dyn TagForStart>>,
    /// The constraint as the request carries it, before the plan builder
    /// shapes it for the worker.
    pub tool_constraints: Option<(String, String)>,
    /// The constraint is the whole turn's grammar, which holds the request's
    /// own format: the engine's request carries none beside it.
    pub whole_turn: bool,
    /// The constraint was shaped for an engine that starts the grammar at the
    /// first generated token (the thought, closed, ahead of the answer or the
    /// calls; the whole turn's grammar): the engine's request says so where
    /// its dialect has a switch for the gate behind a reasoning parser
    /// (`WorkerConstraint::first_token`).
    pub first_token: bool,
    /// Symphony's assignment, for the response spec; `None` where the legacy
    /// parsers read the response.
    pub assignment: Option<Assignment>,
}

/// What constrains the output, decided on the worker selected for the
/// request (`regular::symphony::worker_constraint`): the whole turn's grammar
/// on a worker whose engine takes one on every turn from its first token;
/// else what preparation found (`prepared`; a JSON answer's tag derived for
/// the worker's start), and on a worker that takes a grammar on every turn
/// the table's tag for the calls the request allows under `auto` (`auto`,
/// derived for the worker's start). When the worker's
/// policy chose the constraint, the state records the worker's decoding, so a
/// retry re-selects only workers that decide alike
/// (`utils::WorkerDecoding::decides_alike`); what the request asked for suits
/// any worker. Then Symphony's assignment, taken while the rendered prompt is
/// still here (`regular::symphony::assign`): it reads a constraint no table
/// built by its kind, before any shaping for the worker, and the table's
/// by its origin (`WorkerConstraint::table`), and the state records the parser
/// that reads the response, Symphony when a table takes the request and
/// legacy otherwise, for the response's header. `declared` is the tools the
/// template sees (a chat request's effective tool list, a Messages request's
/// chat tools) and `processed_text` the rendered prompt. Either refusal is
/// the 400 the request gets.
pub(crate) fn decide(
    ctx: &mut RequestContext,
    asks: Asks<'_>,
    declared: &[Tool],
    processed_text: &str,
    prepared: PreparedConstraint,
    auto: impl FnOnce(&RequestContext) -> Auto,
) -> Result<Decided, Response> {
    let model = asks.model();
    let names = ctx
        .components
        .parser_resolver
        .names_for(ctx.state.parser_names.as_ref(), model);
    let parser = ctx.state.parser.unwrap_or(ParserChoice::Legacy);
    let decoding = utils::WorkerDecoding::of(ctx.state.workers.as_ref());
    let decided = symphony::worker_constraint(
        decoding,
        prepared,
        || {
            symphony::turn_constraint(
                parser,
                &names,
                asks,
                declared,
                processed_text,
                decoding.vocabulary,
                &ctx.components.templates,
            )
        },
        || {
            let Auto {
                request,
                reasoning_open,
                thought_closed,
            } = auto(ctx);
            symphony::AutoTag {
                parser,
                labels: names.labels.clone(),
                flags: names.flags.clone(),
                request,
                reasoning_open,
                thought_closed,
                templates: Arc::clone(&ctx.components.templates),
            }
        },
    )
    .map_err(|why| error::bad_request("invalid_parser_configuration", why))?;
    if decided.by_policy {
        ctx.state.constraint_decided_on = Some(decoding);
    }
    // The constraint as the request carries it; Symphony's assignment below
    // reads one no table built alone (a `json_schema` pair there is a forced
    // call's schema, the older registry's or `forced_call_schema`'s, which the
    // table reads with its forced call), never the table's, whatever kind the
    // backend's spelling gave it.
    let tabled = decided.constraint.map(|constraint| constraint.to_tuple());
    let untabled = if decided.table { None } else { tabled.as_ref() };
    let provider = ctx
        .state
        .provider
        .unwrap_or_else(|| ProviderProfile::for_model(model));
    // The table decision preparation read for the request stands: the names
    // are the request's and the parser is still preparation's choice until the
    // assignment below records the response's. Where none was kept (another
    // endpoint's preparation; a refusal the constraint carried), the assignment
    // decides itself.
    let read = match ctx.state.reading.clone() {
        Some(read) => read,
        None => symphony::reading(parser, &names, model, &ctx.components.templates)
            .map_err(|why| error::bad_request("invalid_parser_configuration", why))?,
    };
    // A forced call's reader starts where the model's workers start a grammar together, the
    // start preparation built the constraint for, kept with it (`together`); the assignment
    // asks under a forced call's JSON schema alone, so no other request reads it here.
    let assignment = symphony::assign_for(read, asks, untabled, processed_text, provider, || {
        together(ctx, model, |labels| labels.grammar_start)
    });
    ctx.state.parser = Some(match assignment {
        Some(_) => ParserChoice::Symphony,
        None => ParserChoice::Legacy,
    });
    // A Messages answer the table gave this worker no tag for (a registry
    // table leaves the schema alone to the engine, as it does for a chat
    // `response_format`, whose request builder then carries the schema) is
    // the request's own schema, the engine's JSON-schema constraint, on the
    // pair the engine receives alone: the assignment above read the table's,
    // so Symphony still reads the response, and the tag stays on the attempt
    // stamp, from which a retry's worker derives its own
    // (`helpers::ToolConstraintStamp::capture_table`).
    let tool_constraints = tabled.or_else(|| {
        asks.untabled_answer_schema(declared)
            .map(|schema| ToolConstraint::JsonSchema(schema.to_string()).to_tuple())
    });
    Ok(Decided {
        table_tag: decided.table_tag,
        tool_constraints,
        whole_turn: decided.whole_turn,
        first_token: decided.first_token,
        assignment,
    })
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::TokenizerRegistry;
    use openai_protocol::{
        chat::ChatCompletionRequest, messages::CreateMessageRequest, model_card::ModelCard,
        worker::GrammarStart as WorkerGrammarStart,
    };
    use serde_json::json;
    use smg_worker::identity::basic::WorkerType;

    use super::*;
    use crate::routers::{
        common::snapshot::WorkerRegistry,
        grpc::{
            backend_client::BasicWorkerBuilder,
            context::SharedComponents,
            regular::stages::testing::{components, components_over},
            utils::message_utils,
        },
    };

    /// A model the registry has a table for, named by no label or flag.
    const MODEL: &str = "Qwen/Qwen3-8B";
    const PROMPT: &str = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
    /// A model the registry has no table for, with no flag naming one: the legacy parsers read it.
    const NO_TABLE_MODEL: &str = "acme/plain-7b";

    /// What [`decide`] said: the state's parser after, the table assigned with the prompt it
    /// replays, and the constraint.
    #[derive(Debug, PartialEq)]
    struct Outcome {
        parser: Option<ParserChoice>,
        assigned: Option<(String, String)>,
        constraint: Option<(String, String)>,
    }

    /// What [`decide`] says for `asks` on `ctx`, where preparation found nothing and the worker's
    /// policy is `requested`; no `auto` tag is derived.
    fn decided(ctx: &mut RequestContext, asks: Asks<'_>) -> Outcome {
        ctx.state.parser = Some(ParserChoice::Symphony);
        let decided = decide(ctx, asks, &[], PROMPT, Ok(symphony::Prepared::None), |_| {
            panic!("the policy is requested: no auto tag")
        })
        .map_err(|response| response.status())
        .expect("nothing refused");
        assert!(!decided.whole_turn && decided.table_tag.is_none());
        Outcome {
            parser: ctx.state.parser,
            assigned: decided
                .assignment
                .map(|assignment| (assignment.format.name().to_string(), assignment.prompt)),
            constraint: decided.tool_constraints,
        }
    }

    /// A Messages request's answer schema reaches the engine where the table gives the worker no
    /// tag for it: a registry table leaves the schema alone to the engine, as it does for a chat
    /// `response_format` (whose request builder carries the schema), so the decision hands the
    /// engine the request's own JSON-schema constraint, and the table's tag stays for the stamp;
    /// Symphony's assignment read the table's constraint, not the schema (a `json_schema` pair
    /// there is the older registry's, which the table reads without its calls whatever the
    /// request declares), so the table reads the response as the request's asks make it.
    #[test]
    fn a_messages_answer_schema_the_table_leaves_alone_is_the_engines_constraint() {
        let schema = json!({
            "type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]
        });
        let request: Arc<CreateMessageRequest> = Arc::new(
            serde_json::from_value(json!({
                "model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}],
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            }))
            .expect("a Messages request"),
        );
        let mut ctx = RequestContext::for_messages_carrying_its_tools(
            Arc::clone(&request),
            None,
            MODEL.into(),
            components(),
        );
        ctx.state.parser = Some(ParserChoice::Symphony);
        let asks = Asks::for_messages(&request, false);
        let names = ctx.components.parser_resolver.names_for(None, MODEL);
        let read = symphony::reading(
            ParserChoice::Symphony,
            &names,
            MODEL,
            &ctx.components.templates,
        )
        .expect("a table reads the model");
        let prepared = symphony::constraint_for(
            &read,
            asks,
            &[],
            true,
            false,
            symphony::grammar_start_of(&[]),
            symphony::vocabulary_of(&[]),
        );
        assert!(
            matches!(prepared, Ok(symphony::Prepared::Answer(_))),
            "{prepared:?}"
        );
        let decided = decide(&mut ctx, asks, &[], PROMPT, prepared, |_| {
            panic!("the policy is requested: no auto tag")
        })
        .map_err(|response| response.status())
        .expect("nothing refused");
        assert_eq!(
            decided.tool_constraints,
            Some(("json_schema".to_string(), schema.to_string()))
        );
        assert!(decided.table_tag.is_some() && !decided.whole_turn);
        assert!(
            decided.assignment.is_some(),
            "the table reads the response, not the legacy parsers"
        );
        assert_eq!(
            decided
                .assignment
                .as_ref()
                .and_then(|read| read.format.call_syntax()),
            None,
            "a request that declares no tools reads without the table's calls"
        );
        assert!(matches!(ctx.state.parser, Some(ParserChoice::Symphony)));
    }

    /// A constraint the table built keeps the table's own reading whatever its kind: a JSON-schema
    /// pair from the table (a backend's spelling of the table's grammar, where a backend takes a
    /// schema) is not the older registry's schema, which the same pair from the older registry
    /// is. The table reads the response either way: with its calls and their markers under its
    /// own pair, and under the registry's without them, as the JSON alone that pair holds the
    /// output to (an answer here, where the choice forces no call). The engine gets the pair
    /// either way.
    #[test]
    fn a_constraint_the_table_built_keeps_the_table_as_the_reader_whatever_its_kind() {
        let chat: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(json!({
                "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "lookup",
                    "parameters": {"type": "object", "properties": {}}}}],
            }))
            .expect("chat request"),
        );
        for (table, calls) in [(true, true), (false, false)] {
            let mut ctx =
                RequestContext::for_chat(Arc::clone(&chat), None, MODEL.into(), components());
            ctx.state.parser = Some(ParserChoice::Symphony);
            let prepared = Ok(symphony::Prepared::Constraint {
                constraint: ToolConstraint::JsonSchema("{}".to_string()),
                first_token: false,
                table,
            });
            let decided = decide(&mut ctx, Asks::Chat(&chat), &[], PROMPT, prepared, |_| {
                panic!("the policy is requested: no auto tag")
            })
            .map_err(|response| response.status())
            .expect("nothing refused");
            assert_eq!(
                decided.tool_constraints,
                Some(("json_schema".to_string(), "{}".to_string())),
                "table {table}"
            );
            let assignment = decided.assignment.expect("the table reads either way");
            assert_eq!(
                assignment.format.call_syntax().is_some(),
                calls,
                "table {table}"
            );
            assert_eq!(
                ctx.state.parser,
                Some(ParserChoice::Symphony),
                "table {table}"
            );
        }
    }

    /// The decision follows preparation on whether the choice forces a call, over different tool
    /// lists: preparation reads the tools the template renders, narrowed by `tool_choice`
    /// (`message_utils::template_tools`), the decision the request's declared chat tools. A choice
    /// naming a tool of another kind (a server tool, which the custom tools never hold) narrows
    /// preparation's list to nothing and names no declared chat tool at the decision, so both
    /// leave the answer formatted and the engine gets the schema.
    #[test]
    fn a_choice_naming_a_tool_the_model_never_sees_leaves_the_answer_formatted_at_the_decision() {
        let schema = json!({
            "type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]
        });
        let request: Arc<CreateMessageRequest> = Arc::new(
            serde_json::from_value(json!({
                "model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"name": "lookup", "input_schema": {"type": "object", "properties": {}}},
                    {"type": "web_search_20250305", "name": "web_search"}
                ],
                "tool_choice": {"type": "tool", "name": "web_search"},
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            }))
            .expect("a Messages request"),
        );
        let declared = request.chat_tools();
        let narrowed = message_utils::template_tools(&declared, request.tool_choice.as_ref());
        assert_eq!(
            (declared.len(), narrowed.len()),
            (1, 0),
            "the custom tool alone is declared, and the choice narrows it away"
        );
        let mut ctx = RequestContext::for_messages_carrying_its_tools(
            Arc::clone(&request),
            None,
            MODEL.into(),
            components(),
        );
        ctx.state.parser = Some(ParserChoice::Symphony);
        let asks = Asks::for_messages(&request, false);
        let names = ctx.components.parser_resolver.names_for(None, MODEL);
        let read = symphony::reading(
            ParserChoice::Symphony,
            &names,
            MODEL,
            &ctx.components.templates,
        )
        .expect("a table reads the model");
        let prepared = symphony::constraint_for(
            &read,
            asks,
            &narrowed,
            true,
            false,
            symphony::grammar_start_of(&[]),
            symphony::vocabulary_of(&[]),
        );
        assert!(
            matches!(prepared, Ok(symphony::Prepared::Answer(_))),
            "{prepared:?}"
        );
        let decided = decide(&mut ctx, asks, &declared, PROMPT, prepared, |_| {
            panic!("the policy is requested: no auto tag")
        })
        .map_err(|response| response.status())
        .expect("nothing refused");
        assert_eq!(
            decided.tool_constraints,
            Some(("json_schema".to_string(), schema.to_string())),
            "the decision names no forced call over the declared tools either"
        );
        assert!(decided.assignment.is_some());
    }

    /// Request building's reasoning decision reads the replay preparation kept where a table reads
    /// the response (the same prompt, replayed once per request), replays the prompt itself where
    /// none was kept, and leaves a kept replay unread under a legacy assignment, where no table
    /// replays and the legacy parser's reading of the prompt answers. A kept `None` under the
    /// Symphony choice is the table's own answer that no table reads the model (the Messages
    /// stage and the chat `auto` tag decide before [`decide`] records the assignment, so the
    /// choice they see is preparation's): the decision reads it as it reads a fresh replay's
    /// `None`.
    #[test]
    fn the_reasoning_decision_reads_the_replay_preparation_kept() {
        let chat: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(
                json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
            )
            .expect("chat request"),
        );
        let mut ctx = RequestContext::for_chat(Arc::clone(&chat), None, MODEL.into(), components());
        ctx.state.parser = Some(ParserChoice::Symphony);
        // Nothing kept: the decision replays the prompt, which leaves the thought open.
        assert_eq!(
            reasoning_decision(&ctx, MODEL, PROMPT, |_| false),
            (true, false)
        );
        // Preparation's replay stands in place of a second one: its answer, not the prompt's.
        ctx.state.prompt_thought = Some(symphony::reasoning_open(
            ParserChoice::Symphony,
            &ctx.components.parser_resolver.names(MODEL),
            MODEL,
            &ctx.components.templates,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ));
        assert_eq!(
            reasoning_decision(&ctx, MODEL, PROMPT, |_| false),
            (false, true)
        );
        // A prompt that closed the thought overrules a switch that says the model thinks.
        ctx.state.tokenizer = Some(hiding(&["<think>", "</think>"]));
        assert_eq!(
            reasoning_decision(&ctx, MODEL, PROMPT, |_| true),
            (false, true)
        );
        // Under a legacy assignment the kept replay is not read: no closed thought is reported.
        ctx.state.parser = Some(ParserChoice::Legacy);
        ctx.state.prompt_thought = Some(symphony::reasoning_open(
            ParserChoice::Symphony,
            &ctx.components.parser_resolver.names(MODEL),
            MODEL,
            &ctx.components.templates,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ));
        assert!(!reasoning_decision(&ctx, MODEL, PROMPT, |_| false).1);
        // A model no table reads, under the Symphony choice: preparation keeps the table's `None`
        // and the decision reads it as it would read a fresh replay's.
        ctx.state.parser = Some(ParserChoice::Symphony);
        ctx.state.prompt_thought = None;
        let replayed = reasoning_decision(&ctx, NO_TABLE_MODEL, PROMPT, |_| false);
        ctx.state.prompt_thought = Some(symphony::reasoning_open(
            ParserChoice::Symphony,
            &ctx.components.parser_resolver.names(NO_TABLE_MODEL),
            NO_TABLE_MODEL,
            &ctx.components.templates,
            PROMPT,
        ));
        assert_eq!(ctx.state.prompt_thought, Some(None));
        assert_eq!(
            reasoning_decision(&ctx, NO_TABLE_MODEL, PROMPT, |_| false),
            replayed
        );
    }

    /// A chat request and a Messages request that ask the same of the table decide the same,
    /// through the one function: the registry's table reads both (the state records Symphony
    /// for the header), replaying the same part of the prompt, and nothing constrains either.
    #[test]
    fn a_chat_and_a_messages_request_that_ask_the_same_decide_alike() {
        let chat: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(
                json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
            )
            .expect("chat request"),
        );
        let mut ctx = RequestContext::for_chat(Arc::clone(&chat), None, MODEL.into(), components());
        let for_chat = decided(&mut ctx, Asks::Chat(&chat));

        let messages: Arc<CreateMessageRequest> = Arc::new(
            serde_json::from_value(json!({
                "model": MODEL, "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .expect("messages request"),
        );
        let mut ctx = RequestContext::for_messages_carrying_its_tools(
            Arc::clone(&messages),
            None,
            MODEL.into(),
            components(),
        );
        let for_messages = decided(&mut ctx, Asks::for_messages(&messages, false));

        assert_eq!(for_chat, for_messages);
        assert_eq!(for_chat.parser, Some(ParserChoice::Symphony));
        assert_eq!(
            for_chat.assigned.as_ref().map(|(table, _)| table.as_str()),
            Some("qwen3")
        );
        assert!(for_chat.constraint.is_none());
    }

    /// Request building reads the parser names preparation resolved for the request
    /// (`ProcessingState::parser_names`), not the registry again: the names stored name the
    /// Kimi K2 table, so that table reads the response, though no label of the model's workers
    /// and no flag names it (the registry's own table for the model is Qwen3's).
    #[test]
    fn the_decision_reads_the_names_preparation_resolved() {
        let chat: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(
                json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
            )
            .expect("chat request"),
        );
        let mut ctx = RequestContext::for_chat(Arc::clone(&chat), None, MODEL.into(), components());
        ctx.state.parser_names = Some(Names {
            tool: Some("kimi_k2".to_string()),
            reasoning: None,
        });
        let outcome = decided(&mut ctx, Asks::Chat(&chat));
        assert_eq!(outcome.parser, Some(ParserChoice::Symphony));
        assert_eq!(
            outcome.assigned.as_ref().map(|(table, _)| table.as_str()),
            Some("kimi_k2")
        );
    }

    /// Request building's assignment reads the table decision preparation kept for the request
    /// (`ProcessingState::reading`), not the names again: the decision kept names the Kimi K2
    /// table, so that table reads the response, though the names and the registry say Qwen3's; a
    /// kept legacy decision leaves the response to the legacy parsers; with nothing kept the
    /// assignment decides itself.
    #[test]
    fn the_assignment_reads_the_decision_preparation_kept() {
        let chat: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(
                json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
            )
            .expect("chat request"),
        );
        let mut ctx = RequestContext::for_chat(Arc::clone(&chat), None, MODEL.into(), components());
        let table = |outcome: &Outcome| {
            outcome
                .assigned
                .as_ref()
                .map(|(table, _)| table.to_string())
        };
        // Nothing kept: the assignment decides itself, the registry's table for the model.
        assert_eq!(
            table(&decided(&mut ctx, Asks::Chat(&chat))).as_deref(),
            Some("qwen3")
        );
        // The kept decision stands in place of a fourth one over the same names.
        let kimi = Names {
            tool: Some("kimi_k2".to_string()),
            reasoning: None,
        };
        ctx.state.reading = Some(
            symphony::reading(
                ParserChoice::Symphony,
                &ctx.components.parser_resolver.names_for(Some(&kimi), MODEL),
                MODEL,
                &ctx.components.templates,
            )
            .expect("a table the names select"),
        );
        let outcome = decided(&mut ctx, Asks::Chat(&chat));
        assert_eq!(outcome.parser, Some(ParserChoice::Symphony));
        assert_eq!(table(&outcome).as_deref(), Some("kimi_k2"));
        // A kept legacy decision: no table reads the response.
        ctx.state.reading = Some(Reading::Legacy);
        let outcome = decided(&mut ctx, Asks::Chat(&chat));
        assert_eq!(outcome.parser, Some(ParserChoice::Legacy));
        assert!(outcome.assigned.is_none());
    }

    /// A tokenizer whose decoder drops each of `markers` as a special token.
    fn hiding(markers: &[&str]) -> Arc<dyn Tokenizer> {
        let numbered = || markers.iter().zip(1_u32..);
        let added: Vec<serde_json::Value> = numbered()
            .map(|(marker, id)| {
                json!({
                    "id": id, "content": marker, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true
                })
            })
            .collect();
        let mut vocab = serde_json::Map::new();
        vocab.insert("[UNK]".to_string(), json!(0));
        for (marker, id) in numbered() {
            vocab.insert((*marker).to_string(), json!(id));
        }
        let definition = json!({
            "version": "1.0", "added_tokens": added,
            "model": {"type": "WordLevel", "unk_token": "[UNK]", "vocab": vocab}
        });
        let file = tempfile::NamedTempFile::new().expect("a file");
        std::fs::write(file.path(), serde_json::to_vec(&definition).expect("JSON"))
            .expect("the definition written");
        Arc::new(
            llm_tokenizer::HuggingFaceTokenizer::from_file(file.path().to_str().expect("a path"))
                .expect("a tokenizer"),
        )
    }

    /// The gateway's components over one worker of [`SERVED`] that starts a grammar where `start`
    /// says (its label, or none), its card naming `table` as the reasoning parser and no
    /// tool-call parser. With no tool-call name the older registry has no tag of its own for a
    /// forced call, so the call takes its JSON schema where the table has no tag either.
    fn pool(table: &str, start: Option<&str>) -> Arc<SharedComponents> {
        let start = start.map(|start| (WorkerGrammarStart::LABEL, start));
        pool_of(
            ModelCard::new(SERVED).with_reasoning_parser(table),
            start.as_slice(),
        )
    }

    /// The gateway's components over one worker that serves `card` and carries `labels`.
    fn pool_of(card: ModelCard, labels: &[(&str, &str)]) -> Arc<SharedComponents> {
        let registry = Arc::new(WorkerRegistry::new());
        let mut worker = BasicWorkerBuilder::new("grpc://engine:50051")
            .model(card)
            .worker_type(WorkerType::Regular);
        for (label, value) in labels {
            worker = worker.label(*label, *value);
        }
        registry.register(Arc::new(worker.build()));
        let mut components = components_over(
            Arc::new(TokenizerRegistry::new()),
            Arc::clone(&registry),
            ParserChoice::Symphony,
        );
        Arc::get_mut(&mut components)
            .expect("not shared yet")
            .parser_resolver = utils::ParserResolver::with_flags(registry, None, None);
        components
    }

    /// A chat request to [`SERVED`] that forces a call of `lookup` by name.
    fn forced(separate_reasoning: bool) -> Arc<ChatCompletionRequest> {
        Arc::new(
            serde_json::from_value(json!({
                "model": SERVED, "messages": [{"role": "user", "content": "hi"}],
                "separate_reasoning": separate_reasoning,
                "tool_choice": {"type": "function", "function": {"name": "lookup"}},
                "tools": [{"type": "function", "function": {"name": "lookup",
                    "parameters": {"type": "object", "properties": {}}}}],
            }))
            .expect("chat request"),
        )
    }

    /// A model no registry knows by its name: its card names the table.
    const SERVED: &str = "acme/thinker-1";
    /// The prompt's tail with the thought open, as the table whose name the card carries reads it.
    const THINKING: &str = "<|message_model|><|content_thinking|>";

    /// A forced call from a thought the prompt left open: workers that name where they start a
    /// grammar get the table's tag; where they name no start the table's tag would open with
    /// free text, which owes the call nothing, so the call's JSON schema goes out, as on a
    /// table with no tag.
    #[test]
    fn a_forced_call_from_an_open_thought_takes_the_schema_where_the_workers_name_no_start() {
        let tokenizer = hiding(&["<think>", "</think>"]);
        for (start, schema) in [
            (Some("first_token"), false),
            (Some("after_reasoning"), false),
            (None, true),
        ] {
            let chat = forced(true);
            let mut ctx = RequestContext::for_chat(
                Arc::clone(&chat),
                None,
                SERVED.into(),
                pool("qwen3", start),
            );
            ctx.state.parser = Some(ParserChoice::Symphony);
            let tools = chat.tools.clone().unwrap_or_default();
            let prepared = prepare(
                &ctx,
                Asks::Chat(&chat),
                &tools,
                PROMPT,
                true,
                &tokenizer,
                "test",
            )
            .map_err(|response| response.status())
            .expect("nothing refused");
            assert_eq!(
                prepared.json_schema, schema,
                "workers that start at {start:?}: {:?}",
                prepared.constraint
            );
        }
    }

    /// A forced call from an open thought on workers that name no start takes the JSON schema
    /// whatever tool-call parser the card names. Each name here is one a table answers to and
    /// one the older registry builds a tag of its own for. The schema is built with no parser
    /// (`forced_call_schema`), so that tag does not go out in the table's place: it may open
    /// with free text as the table's does, and it is not spelled for workers that take the
    /// older tag vocabulary.
    #[test]
    fn a_forced_call_takes_the_schema_whatever_tool_call_parser_the_card_names() {
        let tokenizer = hiding(&["<think>", "</think>"]);
        let older = [(utils::TagVocabulary::LABEL, utils::TagVocabulary::OLDER)];
        for parser in ["glm47_moe", "minimax_m3", "hy_v4", "deepseek_v41"] {
            for (vocabulary, labels) in [
                (utils::TagVocabulary::Current, &[][..]),
                (utils::TagVocabulary::Older, &older[..]),
            ] {
                let cell = format!("{parser}, the {vocabulary:?} tag vocabulary");
                let chat = forced(true);
                let mut ctx = RequestContext::for_chat(
                    Arc::clone(&chat),
                    None,
                    SERVED.into(),
                    pool_of(ModelCard::new(SERVED).with_tool_parser(parser), labels),
                );
                ctx.state.parser = Some(ParserChoice::Symphony);
                assert!(
                    ctx.components
                        .tool_parser_factory
                        .registry()
                        .has_structural_tag(parser),
                    "{cell}: the older registry builds a tag of its own"
                );
                let workers = ctx
                    .components
                    .worker_registry
                    .get_by_model(&ctx.input.model_id);
                assert_eq!(symphony::vocabulary_of(&workers), vocabulary, "{cell}");
                let tools = chat.tools.clone().unwrap_or_default();
                let prepared = prepare(
                    &ctx,
                    Asks::Chat(&chat),
                    &tools,
                    PROMPT,
                    true,
                    &tokenizer,
                    "test",
                )
                .map_err(|response| response.status())
                .expect("nothing refused");
                assert!(
                    matches!(prepared.reading, Some(Reading::Table(..))),
                    "{cell}: a table reads the response"
                );
                assert!(
                    prepared.json_schema,
                    "{cell}: the schema, not a tag: {:?}",
                    prepared.constraint
                );
            }
        }
    }

    /// A `required` call `forced_call_schema`'s one schema cannot spell keeps the table's tag,
    /// from an open thought on workers that name no start, and is served: two tools that
    /// define one name differently, which that schema refuses to build; a reference to an entry
    /// of `definitions`, which it builds and leaves pointing nowhere, since it lifts the tools'
    /// `$defs` and nothing else. The table's tag reads each tool's parameters apart. Where every
    /// reference keeps its target the schema goes out: one definition under the name, and a
    /// named function, whose parameters are the schema as they are. A member named like a
    /// reference inside the data of `examples` is none, and the schema goes out.
    #[test]
    fn a_required_call_the_schema_cannot_spell_keeps_the_tables_tag() {
        let tokenizer = hiding(&["<think>", "</think>"]);
        let tool = |name: &str, definitions: &str, place: &str| {
            json!({"type": "function", "function": {"name": name, "parameters": {
                "type": "object",
                "properties": {"place": {"$ref": format!("#/{definitions}/place")}},
                definitions: {"place": {"type": place}},
            }}})
        };
        let required = json!("required");
        let named = json!({"type": "function", "function": {"name": "lookup"}});
        for (case, choice, tools, schema) in [
            (
                "one definition under the name",
                &required,
                vec![
                    tool("lookup", "$defs", "string"),
                    tool("route", "$defs", "string"),
                ],
                true,
            ),
            (
                "two definitions under one name",
                &required,
                vec![
                    tool("lookup", "$defs", "string"),
                    tool("route", "$defs", "integer"),
                ],
                false,
            ),
            (
                "a reference to an entry of `definitions`",
                &required,
                vec![tool("lookup", "definitions", "string")],
                false,
            ),
            (
                "a named function, the same reference",
                &named,
                vec![tool("lookup", "definitions", "string")],
                true,
            ),
            (
                "a member named like a reference inside `examples`",
                &required,
                vec![
                    json!({"type": "function", "function": {"name": "lookup", "parameters": {
                        "type": "object",
                        "examples": [{"$ref": "literal"}],
                    }}}),
                ],
                true,
            ),
        ] {
            let chat: Arc<ChatCompletionRequest> = Arc::new(
                serde_json::from_value(json!({
                    "model": SERVED, "messages": [{"role": "user", "content": "hi"}],
                    "tool_choice": choice,
                    "tools": tools,
                }))
                .expect("chat request"),
            );
            let mut ctx = RequestContext::for_chat(
                Arc::clone(&chat),
                None,
                SERVED.into(),
                pool("qwen3", None),
            );
            ctx.state.parser = Some(ParserChoice::Symphony);
            let tools = chat.tools.clone().unwrap_or_default();
            let prepared = prepare(
                &ctx,
                Asks::Chat(&chat),
                &tools,
                PROMPT,
                true,
                &tokenizer,
                "test",
            )
            .map_err(|response| response.status())
            .expect("nothing refused");
            assert_eq!(
                prepared.json_schema, schema,
                "{case}: {:?}",
                prepared.constraint
            );
            assert_eq!(
                matches!(
                    prepared.constraint,
                    Ok(symphony::Prepared::Constraint { table: true, .. })
                ),
                !schema,
                "{case}: the table's own tag where the schema cannot spell the request"
            );
        }
    }

    /// The reader of a forced call under the schema starts the same on a table that has a tag
    /// of its own: in the call on workers that start at the first token, with the prompt's
    /// replay on workers that start once the thought is over, and on workers that do not say
    /// with the replay kept and the JSON's opener named.
    #[test]
    fn a_forced_calls_reader_starts_the_same_on_a_table_that_has_a_tag() {
        const OBJECT: &str = r#"{"type":"object","properties":{}}"#;
        // The replay the reader keeps is the turn the model answers in.
        const TURN: &str = "<|im_start|>assistant\n<think>\n";
        for (start, replayed, opener) in [
            (Some("first_token"), "", None),
            (Some("after_reasoning"), TURN, None),
            (None, TURN, Some(b'{')),
        ] {
            let chat = forced(true);
            let mut ctx = RequestContext::for_chat(
                Arc::clone(&chat),
                None,
                SERVED.into(),
                pool("qwen3", start),
            );
            ctx.state.parser = Some(ParserChoice::Symphony);
            let schema = ToolConstraint::JsonSchema(OBJECT.to_string());
            let decided = decide(
                &mut ctx,
                Asks::Chat(&chat),
                &[],
                PROMPT,
                Ok(symphony::Prepared::from(Some(schema))),
                |_| panic!("the policy is requested: no auto tag"),
            )
            .map_err(|response| response.status())
            .expect("nothing refused");
            let assignment = decided.assignment.expect("the table reads the forced call");
            assert_eq!(assignment.table(), "qwen3", "{start:?}");
            assert_eq!(assignment.forcing.as_deref(), Some("lookup"), "{start:?}");
            assert_eq!(assignment.prompt, replayed, "{start:?}");
            assert_eq!(assignment.opener, opener, "{start:?}");
        }
    }

    /// A forced call the table has no tag for takes the older registry's JSON schema, and the
    /// table still reads the response. Where the table's markers are special tokens of the
    /// vocabulary, the decoder keeps them under the schema too, whether the request separates the
    /// reasoning or keeps it in the content, where the older reasoning parser asked for nothing.
    /// The pool here starts the grammar at the first token and the prompt opened the thought,
    /// where this table has no tag for a forced call.
    #[test]
    fn a_forced_call_under_the_schema_keeps_the_special_tokens_the_table_reads() {
        let components = pool("inkling", Some("first_token"));
        let tokenizer = hiding(&["<|content_thinking|>", "<|end_message|>"]);
        for separate_reasoning in [true, false] {
            let chat = forced(separate_reasoning);
            let mut ctx = RequestContext::for_chat(
                Arc::clone(&chat),
                None,
                SERVED.into(),
                Arc::clone(&components),
            );
            ctx.state.parser = Some(ParserChoice::Symphony);
            let tools = chat.tools.clone().unwrap_or_default();
            let prepared = prepare(
                &ctx,
                Asks::Chat(&chat),
                &tools,
                THINKING,
                false,
                &tokenizer,
                "test",
            )
            .map_err(|response| response.status())
            .expect("nothing refused");
            let table = match &prepared.reading {
                Some(Reading::Table(format, _)) => Some(format.name()),
                _ => None,
            };
            assert_eq!(table, Some("inkling"), "the table the card names reads");
            assert!(
                prepared.json_schema,
                "the older registry's schema, not a tag: {:?}",
                prepared.constraint
            );
            assert!(
                prepared.keeps_special_tokens,
                "separate_reasoning {separate_reasoning}"
            );
        }
    }

    /// Preparation reads the thought by the rule request building asks. On the table whose
    /// replay decides alone, a new turn the prompt leaves outside the thought is not reasoning
    /// whatever the thinking switch says: the forced call's tag is the same with the switch on
    /// and off, wherever the model's workers start a grammar, and request building says the
    /// same of the turn from the replay preparation stored. A channel the prompt opened is
    /// still read as open with the switch off.
    #[test]
    fn preparation_reads_the_thought_by_request_buildings_rule() {
        const NEW_TURN: &str = "<|turn>user\nhi<turn|>\n<|turn>model\n";
        const OPENED: &str = "<|turn>user\nhi<turn|>\n<|turn>model\n<|channel>thought\n";
        let tokenizer = hiding(&["<|channel>", "<channel|>"]);
        for start in [Some("first_token"), Some("after_reasoning"), None] {
            let chat = forced(true);
            let mut ctx = RequestContext::for_chat(
                Arc::clone(&chat),
                None,
                SERVED.into(),
                pool("gemma4", start),
            );
            ctx.state.parser = Some(ParserChoice::Symphony);
            ctx.state.tokenizer = Some(Arc::clone(&tokenizer));
            let tools = chat.tools.clone().unwrap_or_default();
            let prepared_for = |ctx: &RequestContext, prompt: &str, switch: bool| {
                prepare(
                    ctx,
                    Asks::Chat(&chat),
                    &tools,
                    prompt,
                    switch,
                    &tokenizer,
                    "test",
                )
                .map_err(|response| response.status())
                .expect("nothing refused")
            };
            let tag = |prompt: &str, switch: bool| {
                format!("{:?}", prepared_for(&ctx, prompt, switch).constraint)
            };
            let new_turn = tag(NEW_TURN, false);
            assert!(
                new_turn.contains("lookup"),
                "a tag over the forced tool: {new_turn}"
            );
            assert_eq!(
                tag(NEW_TURN, true),
                new_turn,
                "a new turn, workers that start at {start:?}"
            );
            let opened = tag(OPENED, false);
            assert_eq!(
                tag(OPENED, true),
                opened,
                "an opened channel, workers that start at {start:?}"
            );
            if start == Some("first_token") {
                assert_ne!(
                    opened, new_turn,
                    "the open channel's close stands before the call"
                );
            }
            // Request building reads what preparation stored for the request, as the chat and
            // Messages stages store it: the table's replay of the prompt, not one built again.
            for (prompt, open) in [(NEW_TURN, false), (OPENED, true)] {
                for switch in [true, false] {
                    let prepared = prepared_for(&ctx, prompt, switch);
                    let table = match &prepared.reading {
                        Some(Reading::Table(format, _)) => Some(format.name()),
                        _ => None,
                    };
                    assert_eq!(table, Some("gemma4"), "the table the card names reads");
                    ctx.state.parser_names = Some(prepared.names);
                    ctx.state.prompt_thought = Some(prepared.prompt_thought);
                    assert_eq!(
                        reasoning_decision(&ctx, SERVED, prompt, |_| switch),
                        (open, false),
                        "request building over {prompt:?}, the switch {switch}"
                    );
                }
            }
        }
    }

    /// A prompt that closed the thought itself overrules a thinking switch that says the model
    /// thinks: a template may write the closed thought for a request that names no switch while
    /// the switch's default says thinking. The forced call's tag is then the one the same prompt
    /// gets with the switch off, the call with no thought in front of it, wherever the model's
    /// workers start a grammar.
    #[test]
    fn a_prompt_that_closed_the_thought_overrules_the_thinking_switch() {
        const CLOSED: &str =
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let tokenizer = hiding(&["<think>", "</think>"]);
        for start in [Some("first_token"), Some("after_reasoning"), None] {
            let chat = forced(true);
            let mut ctx = RequestContext::for_chat(
                Arc::clone(&chat),
                None,
                SERVED.into(),
                pool("qwen3", start),
            );
            ctx.state.parser = Some(ParserChoice::Symphony);
            let tools = chat.tools.clone().unwrap_or_default();
            let tag = |switch: bool| {
                let prepared = prepare(
                    &ctx,
                    Asks::Chat(&chat),
                    &tools,
                    CLOSED,
                    switch,
                    &tokenizer,
                    "test",
                )
                .map_err(|response| response.status())
                .expect("nothing refused");
                format!("{:?}", prepared.constraint)
            };
            let off = tag(false);
            assert!(off.contains("lookup"), "a tag over the forced tool: {off}");
            assert_eq!(tag(true), off, "workers that start at {start:?}");
        }
    }

    /// The assignment of a forced call under the schema starts the reader where the model's
    /// workers start a grammar together: in the call, with nothing of the prompt replayed, on
    /// workers that start at the first token, where the schema holds the output before the model
    /// writes a thought; with the prompt's replay on workers that start once the thought is
    /// over; and on workers that do not say, with the replay kept and the JSON's opener named,
    /// so that the output's first bytes decide. The schema's own text reaches the assignment:
    /// where the request keeps the reasoning in the content, arguments the schema does not hold
    /// to an object name no opener on such workers, and the reader starts in the call.
    #[test]
    fn a_forced_calls_reader_starts_where_the_models_workers_start_a_grammar() {
        const OBJECT: &str = r#"{"type":"object","properties":{}}"#;
        for (start, separated, schema, replayed, opener) in [
            (Some("first_token"), true, OBJECT, "", None),
            (Some("after_reasoning"), true, OBJECT, THINKING, None),
            (None, true, OBJECT, THINKING, Some(b'{')),
            (None, true, "{}", THINKING, Some(b'{')),
            (None, false, OBJECT, THINKING, Some(b'{')),
            (None, false, "{}", "", None),
        ] {
            let case = format!("{start:?}, separated {separated}, {schema}");
            let chat = forced(separated);
            let mut ctx = RequestContext::for_chat(
                Arc::clone(&chat),
                None,
                SERVED.into(),
                pool("inkling", start),
            );
            ctx.state.parser = Some(ParserChoice::Symphony);
            let schema = ToolConstraint::JsonSchema(schema.to_string());
            let decided = decide(
                &mut ctx,
                Asks::Chat(&chat),
                &[],
                THINKING,
                Ok(symphony::Prepared::from(Some(schema))),
                |_| panic!("the policy is requested: no auto tag"),
            )
            .map_err(|response| response.status())
            .expect("nothing refused");
            let assignment = decided.assignment.expect("the table reads the forced call");
            assert_eq!(assignment.table(), "inkling", "{case}");
            assert_eq!(assignment.forcing.as_deref(), Some("lookup"), "{case}");
            assert_eq!(assignment.prompt, replayed, "{case}");
            assert_eq!(assignment.opener, opener, "{case}");
        }
    }

    /// Preparation reads what the model's workers say together where the registry's membership
    /// snapshot keeps it: the first request after a membership change walks the workers and
    /// leaves its answers on the snapshot, the next request reads them there, and a worker that
    /// joins the model is read by the request after it, its card and its label beside the
    /// first worker's.
    #[test]
    fn preparation_keeps_what_the_models_workers_say_until_the_membership_changes() {
        let components = pool("inkling", Some("first_token"));
        let tokenizer = hiding(&["<|content_thinking|>", "<|end_message|>"]);
        let chat = forced(true);
        let tools = chat.tools.clone().unwrap_or_default();
        let names = || {
            let mut ctx = RequestContext::for_chat(
                Arc::clone(&chat),
                None,
                SERVED.into(),
                Arc::clone(&components),
            );
            ctx.state.parser = Some(ParserChoice::Symphony);
            prepare(
                &ctx,
                Asks::Chat(&chat),
                &tools,
                THINKING,
                false,
                &tokenizer,
                "test",
            )
            .map_err(|response| response.status())
            .expect("nothing refused")
            .names
        };
        // What the snapshot a request draws now holds, read by nobody here: a reader that ran
        // would mean preparation kept nothing.
        let kept = || {
            components
                .worker_registry
                .model_routing_snapshot(SERVED)
                .expect("a served model")
                .labels(SERVED, |_| {
                    panic!("preparation kept nothing on the snapshot")
                })
                .cloned()
                .expect("kept under the request's spelling")
        };
        let inkling = Names {
            tool: None,
            reasoning: Some("inkling".to_string()),
        };
        assert_eq!(names(), inkling);
        assert_eq!(
            kept(),
            ModelLabels {
                tool_parser: None,
                reasoning_parser: Some("inkling".to_string()),
                grammar_start: ::symphony::grammar::GrammarStart::FirstToken,
                vocabulary: utils::TagVocabulary::Current,
            }
        );
        assert_eq!(names(), inkling, "the next request reads the kept answers");

        let joining = BasicWorkerBuilder::new("grpc://engine-2:50051")
            .model(
                ModelCard::new(SERVED)
                    .with_reasoning_parser("inkling")
                    .with_tool_parser("json"),
            )
            .worker_type(WorkerType::Regular)
            .label(WorkerGrammarStart::LABEL, "after_reasoning");
        components
            .worker_registry
            .register(Arc::new(joining.build()));
        let joined = Names {
            tool: Some("json".to_string()),
            ..inkling
        };
        assert_eq!(names(), joined);
        assert_eq!(
            kept(),
            ModelLabels {
                tool_parser: joined.tool,
                reasoning_parser: joined.reasoning,
                grammar_start: ::symphony::grammar::GrammarStart::Either,
                vocabulary: utils::TagVocabulary::Current,
            }
        );
    }
}
