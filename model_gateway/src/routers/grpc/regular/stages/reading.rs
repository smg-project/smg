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

use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::common::Tool;
use tracing::error;

use crate::{
    config::ParserChoice,
    routers::{
        error,
        grpc::{
            common::stages::helpers::TagForStart,
            context::RequestContext,
            regular::symphony::{self, Asks, Assignment, AutoRequest, PreparedConstraint},
            utils::{self, Names},
        },
    },
};

/// What preparation found for the response.
pub(crate) struct Prepared {
    /// The constraint for what the request asks, before its worker is known:
    /// the table's tag for a forced call, else the older registry's; a JSON
    /// answer's tag as it derives for a worker's start; or why the table
    /// cannot constrain the request. Request building decides on the selected
    /// worker (`regular::symphony::worker_constraint`).
    pub constraint: PreparedConstraint,
    /// The constraint is a JSON schema: the older path's forced call, under
    /// which no table reads.
    pub json_schema: bool,
    /// The decoder keeps the special tokens the reader of the response needs:
    /// a table's markers, or a legacy reasoning parser's.
    pub keeps_special_tokens: bool,
    /// The parser names the model's cards set, resolved here once for the request: the stage
    /// stores them (`ProcessingState::parser_names`) and request building reads them in
    /// place of a second resolution.
    pub names: Names,
}

/// The constraint preparation finds for what the request asks, and whether
/// the decoder keeps the special tokens the reader needs. When Symphony reads
/// the response, its table gives a forced call, or a JSON `response_format`,
/// its tag, so the output comes back in the syntax the table parses
/// (`regular::symphony::constraint`, which decides with the predicate the
/// reader is assigned by; the answer's tag derives for the selected worker at
/// request building); the older registry decides for every other request
/// (`generate_tool_constraint`, over `tools`, the ones the request's choice
/// lets the model call, and the legacy tool parser's name). On a thinking
/// prompt a parser with a reasoning prefix gets its tag wrapped so a forced
/// call follows the reasoning instead of preempting it: `switch` is the API's
/// tokenizer-driven answer, and where a table reads the response its replay
/// of the rendered prompt decides beside it (a turn the prompt leaves
/// unopened starts inside the thought whatever the switch says:
/// `regular::symphony::reasoning_open`), so the tag reads the prompt as the
/// engine will. The table's refusal waits for request building, which decides
/// on the selected worker; the registry's refuses the request here (a 400,
/// logged under `stage`). A table whose markers the decoder would drop as
/// special tokens needs them kept whatever the request asks of the channels:
/// without them the table reads neither the thought nor the calls
/// (`regular::symphony::keeps_special_tokens`); under a JSON-schema constraint
/// no table reads (request building makes no assignment), so the legacy
/// parsers answer for themselves, as they do where no table reads at all: a
/// reasoning parser that needs its control tokens, for a request that
/// separates the reasoning.
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
    let names = ctx.components.parser_resolver.names(model);
    // A forced call's tag takes the start the model's workers name together;
    // a JSON answer's tag and the `auto` tag take the selected worker's own
    // start at request building.
    let workers = ctx
        .components
        .worker_registry
        .get_by_model(&ctx.input.model_id);
    let replay = (!switch)
        .then(|| {
            symphony::reasoning_open(
                parser,
                &names,
                model,
                &ctx.components.templates,
                processed_text,
            )
        })
        .flatten();
    let reasoning = switch || replay.is_some_and(|thought| thought.open);
    // The prompt closed the thought itself and wrote the template's text
    // after it (thinking switched off): the calls' tag owes none of it.
    let thought_closed = replay.is_some_and(|thought| thought.closed);
    let from_the_table = symphony::constraint(
        parser,
        &names,
        asks,
        tools,
        reasoning,
        thought_closed,
        symphony::grammar_start_of(&workers),
        symphony::vocabulary_of(&workers),
        &ctx.components.templates,
    );
    let choice = asks.choice();
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
            Ok(symphony::Prepared::from(constraint))
        }
        (prepared, _) => prepared,
    };
    let json_schema = matches!(
        &constraint,
        Ok(symphony::Prepared::Constraint { constraint, .. }) if constraint.is_json_schema()
    );
    let keeps_special_tokens = match symphony::keeps_special_tokens(
        parser,
        &names,
        model,
        &ctx.components.templates,
        tokenizer,
    ) {
        Some(kept) if !json_schema => kept,
        _ => {
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
    })
}

/// Whether the rendered prompt leaves the model inside its thought, as the
/// request's grammar, the plan's reasoning state and the response spec all
/// read it: where a Symphony table reads the response, the table's own replay
/// of the prompt (a turn the prompt leaves unopened starts inside the thought:
/// `regular::symphony::reasoning_open`) with the tokenizer-driven `switch`
/// (the API's: `utils::chat_reasoning_starts_in_rendered_prefill`,
/// `utils::messages_reasoning_starts_in_prefill`, over `model`'s tokenizer),
/// or alone where the family's rule sets the switch aside (Gemma 4 prefills
/// an open channel only after a tool response), and no legacy parser is built
/// to be asked; otherwise native Gemma4 channels follow the rendered prompt
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
    match symphony::reasoning_open(
        ctx.state.parser.unwrap_or(ParserChoice::Legacy),
        &ctx.components
            .parser_resolver
            .names_for(ctx.state.parser_names.as_ref(), model),
        model,
        &ctx.components.templates,
        processed_text,
    ) {
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
/// still here (`regular::symphony::assign`): it reads the constraint's kind
/// alone, before any shaping for the worker, and the state records the parser
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
    let tool_constraints = decided.constraint.map(|constraint| constraint.to_tuple());
    let assignment = symphony::assign(
        parser,
        &names,
        asks,
        tool_constraints.as_ref(),
        processed_text,
        &ctx.components.templates,
    )
    .map_err(|why| error::bad_request("invalid_parser_configuration", why))?;
    ctx.state.parser = Some(match assignment {
        Some(_) => ParserChoice::Symphony,
        None => ParserChoice::Legacy,
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
    use openai_protocol::{chat::ChatCompletionRequest, messages::CreateMessageRequest};
    use serde_json::json;

    use super::*;
    use crate::routers::grpc::regular::stages::testing::components;

    /// A model the registry has a table for, named by no label or flag.
    const MODEL: &str = "Qwen/Qwen3-8B";
    const PROMPT: &str = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";

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
        let mut ctx =
            RequestContext::for_messages(Arc::clone(&messages), None, MODEL.into(), components());
        let for_messages = decided(
            &mut ctx,
            Asks::Messages {
                request: &messages,
                continues: false,
            },
        );

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
}
