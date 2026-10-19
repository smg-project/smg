//! What the chat and Messages request-building stages decide alike once the
//! worker is selected: whether the rendered prompt leaves the model inside its
//! thought, what constrains the output on that worker, and which table reads
//! the response (Symphony's assignment, see `regular::symphony`). The two
//! stages spell what their request asks ([`Asks`]) and decide through these
//! functions; each keeps what differs between the APIs, the tokenizer-driven
//! reasoning switch and the request the `auto` tag derives for.

use std::sync::Arc;

use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::common::Tool;

use crate::{
    config::ParserChoice,
    routers::{
        error,
        grpc::{
            common::stages::helpers::TagForStart,
            context::RequestContext,
            regular::symphony::{self, Asks, Assignment, AutoRequest, PreparedConstraint},
            utils,
        },
    },
};

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
/// preference for the engine's reasoning gate.
pub(crate) fn reasoning_decision(
    ctx: &RequestContext,
    model: &str,
    processed_text: &str,
    switch: impl FnOnce(&dyn Tokenizer) -> bool,
) -> bool {
    let switch = || {
        ctx.tokenizer_arc()
            .is_some_and(|tokenizer| switch(tokenizer.as_ref()))
    };
    match symphony::reasoning_open(
        ctx.state.parser.unwrap_or_default(),
        &ctx.components.parser_resolver.names(model),
        model,
        &ctx.components.templates,
        processed_text,
    ) {
        Some(thought) => thought.or_switch(switch),
        None => utils::rendered_prompt_reasoning_in_prefill(
            &ctx.components.reasoning_parser_factory,
            ctx.components
                .parser_resolver
                .reasoning_parser(model)
                .as_deref(),
            model,
            processed_text,
        )
        .unwrap_or_else(switch),
    }
}

/// What the table's `auto` tag derives for, when the worker's policy asks for
/// the tag ([`symphony::AutoTag`]): the request, with the tools the tag is
/// over, and whether the prompt leaves the model inside its thought.
pub(crate) struct Auto {
    pub request: AutoRequest,
    pub reasoning_open: bool,
}

/// What [`decide`] decided for the response.
pub(crate) struct Decided {
    /// What derives the `auto` tag for a worker's start, when the worker's
    /// policy chose that tag: the attempt stamp carries it, so a retry gives
    /// the worker it selects the tag shaped for where its own engine starts
    /// the grammar (`helpers::ToolConstraintStamp`).
    pub auto_tag: Option<Arc<dyn TagForStart>>,
    /// The constraint as the request carries it, before the plan builder
    /// shapes it for the worker.
    pub tool_constraints: Option<(String, String)>,
    /// The constraint is the whole turn's grammar, which holds the request's
    /// own format: the engine's request carries none beside it.
    pub whole_turn: bool,
    /// Symphony's assignment, for the response spec; `None` where the legacy
    /// parsers read the response.
    pub assignment: Option<Assignment>,
}

/// What constrains the output, decided on the worker selected for the
/// request (`regular::symphony::worker_constraint`): the whole turn's grammar
/// on a worker whose engine takes one on every turn from its first token;
/// else what preparation found (`prepared`), and on a worker that takes a
/// grammar on every turn the table's tag for the calls the request allows
/// under `auto` (`auto`, derived for the worker's start). When the worker's
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
    let names = ctx.components.parser_resolver.names(model);
    let parser = ctx.state.parser.unwrap_or_default();
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
                &ctx.components.templates,
            )
        },
        || {
            let Auto {
                request,
                reasoning_open,
            } = auto(ctx);
            symphony::AutoTag {
                parser,
                labels: names.labels.clone(),
                flags: names.flags.clone(),
                request,
                reasoning_open,
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
        auto_tag: decided.auto.map(|auto| -> Arc<dyn TagForStart> { auto }),
        tool_constraints,
        whole_turn: decided.whole_turn,
        assignment,
    })
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::TokenizerRegistry;
    use openai_protocol::{chat::ChatCompletionRequest, messages::CreateMessageRequest};
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
    use serde_json::json;
    use tool_parser::ParserFactory as ToolParserFactory;

    use super::*;
    use crate::{
        routers::grpc::{
            context::SharedComponents, regular::symphony::template::Templates,
            utils::ParserResolver,
        },
        worker::WorkerRegistry,
    };

    /// A model the registry has a table for, named by no label or flag.
    const MODEL: &str = "Qwen/Qwen3-8B";
    const PROMPT: &str = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";

    /// The gateway with `--parser symphony`, no parser names and no worker.
    fn components() -> Arc<SharedComponents> {
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        Arc::new(SharedComponents {
            templates: Arc::new(Templates::new(Arc::clone(&tokenizer_registry))),
            tokenizer_registry,
            worker_registry: Arc::new(WorkerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: ParserResolver::disabled(),
            multimodal: None,
            tokenizer_autoload: true,
            tokenizer_load_failures: Arc::default(),
            parser: ParserChoice::Symphony,
        })
    }

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
        let decided = decide(ctx, asks, &[], PROMPT, Ok(None), |_| {
            panic!("the policy is requested: no auto tag")
        })
        .map_err(|response| response.status())
        .expect("nothing refused");
        assert!(!decided.whole_turn && decided.auto_tag.is_none());
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
}
