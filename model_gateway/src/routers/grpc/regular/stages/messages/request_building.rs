//! Message request building stage: Build proto GenerateRequest for message requests

use std::{borrow::Cow, sync::Arc};

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::{
    config::ParserChoice,
    routers::{
        error,
        grpc::{
            client::GenerateRequestBuildOptions,
            common::stages::{
                helpers::{self, TagForStart},
                BuildStage,
            },
            context::{
                stage_output, AttemptStamp, BuildOutput, ClientSelection, ExecutionPlan,
                ExecutionPlanKind, PreparationOutput, RequestContext,
            },
            proto_wrapper::MultimodalData,
            regular::symphony,
            spec::ResponseSpec,
            utils,
        },
    },
};

/// Message request building stage
///
/// Builds a backend-specific proto GenerateRequest from the PreparationOutput
/// and CreateMessageRequest sampling parameters.
pub(crate) struct MessageRequestBuildingStage {
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
}

impl MessageRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            inject_pd_metadata,
            plan_kind,
        }
    }
}

/// Whether the rendered prompt leaves the model inside its thought, as the
/// request's grammar, the plan's reasoning state and the response spec all
/// read it: where a Symphony table reads the message, its replay of the
/// prompt (a turn the prompt leaves unopened starts inside the thought
/// whatever the switch says: `regular::symphony::reasoning_open`) with the
/// tokenizer-driven switch, or alone where the family's rule sets the switch
/// aside (Gemma 4 prefills an open channel only after a tool response), and
/// no legacy parser is built to be asked; otherwise native Gemma4 channels
/// follow the rendered prompt when that parser decides, and other parsers
/// keep their tokenizer-driven preference for the engine's reasoning gate.
fn reasoning_decision(
    ctx: &RequestContext,
    messages_request: &openai_protocol::messages::CreateMessageRequest,
    processed_text: &str,
    continued_final_message: bool,
) -> bool {
    let switch = || {
        ctx.tokenizer_arc().is_some_and(|tokenizer| {
            utils::messages_reasoning_starts_in_prefill(
                messages_request,
                continued_final_message,
                tokenizer.as_ref(),
            )
        })
    };
    match symphony::reasoning_open(
        ctx.state.parser.unwrap_or_default(),
        &ctx.components
            .parser_resolver
            .names(&messages_request.model),
        &messages_request.model,
        &ctx.components.templates,
        processed_text,
    ) {
        Some(thought) => thought.or_switch(switch),
        None => utils::rendered_prompt_reasoning_in_prefill(
            &ctx.components.reasoning_parser_factory,
            ctx.components
                .parser_resolver
                .reasoning_parser(&messages_request.model)
                .as_deref(),
            &messages_request.model,
            processed_text,
        )
        .unwrap_or_else(switch),
    }
}

/// What [`constraint_for_worker`] decided: what derives the `auto` tag for a
/// worker's start, the constraint as the request carries it, and the worker
/// decoding the policy chose it on, when it did (the stage records that, so a
/// retry re-selects workers that decide alike).
struct ConstraintForWorker {
    auto_tag: Option<Arc<dyn TagForStart>>,
    tool_constraints: Option<(String, String)>,
    decided_on: Option<utils::WorkerDecoding>,
}

/// What constrains the output, decided on the worker selected for the
/// request (`regular::symphony::worker_constraint`): the whole turn's grammar
/// on a worker whose engine takes one on every turn from its first token;
/// else what preparation found, and on a worker that takes a grammar on
/// every turn the table's tag for the calls the request allows under `auto`.
/// When the worker's policy chose the constraint, a retry re-selects only
/// workers that decide alike (`utils::WorkerDecoding::decides_alike`); what
/// the request asked for suits any worker. The attempt stamp carries what
/// derives the `auto` tag for a worker's start, so the retry's worker gets it
/// shaped for where its own engine starts the grammar.
#[expect(clippy::too_many_arguments)]
fn constraint_for_worker(
    ctx: &RequestContext,
    messages_request: &Arc<openai_protocol::messages::CreateMessageRequest>,
    chat_tools: &Arc<[openai_protocol::common::Tool]>,
    processed_text: &str,
    continued_final_message: bool,
    require_reasoning: bool,
    tool_constraints: symphony::PreparedConstraint,
    names: &utils::ModelNames<'_>,
    parser: ParserChoice,
) -> Result<ConstraintForWorker, Response> {
    let decoding = utils::WorkerDecoding::of(ctx.state.workers.as_ref());
    let decided = symphony::worker_constraint(
        decoding,
        tool_constraints,
        || {
            symphony::turn_constraint(
                parser,
                names,
                symphony::Asks::Messages {
                    request: messages_request,
                    continues: messages_request.continues_final_assistant(),
                },
                chat_tools,
                processed_text,
                &ctx.components.templates,
            )
        },
        || {
            // The tag shares the spec's tools where `tool_choice` narrows
            // nothing (the common case): no copy of the schemas on the
            // first attempt; a narrowed list is the one the narrowing owns.
            let tools: Arc<[_]> = ctx
                .tokenizer_arc()
                .map(|tokenizer| {
                    match utils::template_tools(
                        chat_tools,
                        messages_request.tool_choice.as_ref(),
                        tokenizer.renderer_capabilities(),
                    ) {
                        Cow::Owned(narrowed) => Arc::from(narrowed),
                        Cow::Borrowed([]) => Arc::default(),
                        Cow::Borrowed(_) => Arc::clone(chat_tools),
                    }
                })
                .unwrap_or_default();
            symphony::AutoTag {
                parser,
                labels: names.labels.clone(),
                flags: names.flags.clone(),
                request: symphony::AutoRequest::Messages {
                    request: Arc::clone(messages_request),
                    tools,
                    continues: continued_final_message,
                },
                reasoning_open: require_reasoning,
                templates: Arc::clone(&ctx.components.templates),
            }
        },
    )
    .map_err(|why| error::bad_request("invalid_parser_configuration", why))?;
    Ok(ConstraintForWorker {
        auto_tag: decided.auto.map(|auto| -> Arc<dyn TagForStart> { auto }),
        tool_constraints: decided.constraint.map(|constraint| constraint.to_tuple()),
        decided_on: decided.by_policy.then_some(decoding),
    })
}

/// Build the backend `GenerateRequest` from the Messages request: the
/// request id stamped for the attempt, the forced call's grammar shaped for
/// the worker (the bare calls behind a reasoning parser's gate, the thought
/// and then the calls from the first token: the worker's
/// `constrained_decoding_start` label, `utils::tool_constraints_for_worker`),
/// the sampling defaults, the stops and the PD metadata; returns the plan
/// and the attempt stamp, which carries the unshaped constraint, or what
/// derives the table's `auto` tag for a start, so a retry gives the worker it
/// selects the same shaping (`helpers::ToolConstraintStamp`).
#[expect(clippy::too_many_arguments)]
fn build_messages_backed_plan(
    ctx: &mut RequestContext,
    messages_request: &openai_protocol::messages::CreateMessageRequest,
    processed_text: String,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
    auto_tag: Option<Arc<dyn TagForStart>>,
    require_reasoning: bool,
    multimodal_data: Option<MultimodalData>,
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
) -> Result<(ExecutionPlan, AttemptStamp), Response> {
    let clients = stage_output(
        ctx.state.clients.as_ref(),
        "MessageRequestBuildingStage::build",
    )?;
    let builder_client = clients.builder();

    // Build message request
    let disaggregated = matches!(clients, ClientSelection::Disaggregated { .. });
    let (request_id, id_stamp) = helpers::resolve_request_id_stamp(
        &ctx.input.request_type,
        ctx.input.tenant_request_meta.as_ref(),
        "msg_",
        disaggregated,
    );

    let tool_parser = ctx
        .components
        .parser_resolver
        .tool_parser(&messages_request.model);
    let unshaped_tool_constraints = tool_constraints.clone();
    let tool_constraints = utils::tool_constraints_for_worker(
        &ctx.components.tool_parser_factory,
        tool_parser.as_deref(),
        tool_constraints,
        ctx.state.workers.as_ref(),
    )
    .map_err(|e| {
        error!(function = "MessageRequestBuildingStage::build", error = %e, "Failed to shape the tool constraint for the worker");
        error::internal_error("invalid_tool_constraint", format!("Invalid tool constraint: {e}"))
    })?;

    // The ids are the prompt; the text rides along only when the request
    // carries media (see `helpers::wire_prompt_text`).
    let carries_media = multimodal_data.is_some() || ctx.state.multimodal_refs.is_some();
    let mut proto_request = builder_client
        .build_messages_request(
            request_id,
            messages_request,
            helpers::wire_prompt_text(processed_text, carries_media),
            token_ids,
            GenerateRequestBuildOptions {
                multimodal_inputs: multimodal_data,
                tool_constraints,
                require_reasoning,
            },
        )
        .map_err(|e| {
            error!(function = "MessageRequestBuildingStage::build", error = %e, "Failed to build generate request");
            error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {e}"))
        })?;

    let sampling_mask = helpers::SamplingDefaultsMask::from_request_type(&ctx.input.request_type);
    let sampling_baseline = helpers::apply_sampling_defaults(
        &mut proto_request,
        sampling_mask,
        ctx.state.workers.as_ref(),
    );

    helpers::finish_generate_request(
        ctx,
        &mut proto_request,
        inject_pd_metadata,
        "MessageRequestBuildingStage::build",
    )?;

    let tool_constraint = match auto_tag {
        Some(auto_tag) => helpers::ToolConstraintStamp::capture_table(auto_tag, &proto_request),
        None => helpers::ToolConstraintStamp::capture(
            unshaped_tool_constraints,
            tool_parser.as_deref(),
            &proto_request,
        ),
    };
    Ok((
        ExecutionPlan::generate(plan_kind, proto_request),
        AttemptStamp {
            id: id_stamp,
            sampling_mask,
            sampling_baseline,
            inject_pd_metadata,
            tool_constraint,
        },
    ))
}

#[async_trait]
impl BuildStage for MessageRequestBuildingStage {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        let multimodal_data =
            helpers::assemble_prompt_media(ctx, "MessageRequestBuildingStage::build").await?;

        // Take preparation state (last consumer — worker_selection already ran)
        let prep = stage_output(
            ctx.state.preparation.take(),
            "MessageRequestBuildingStage::build",
        )?;

        let messages_request = ctx.messages_request_arc();

        let PreparationOutput::Messages {
            token_ids,
            processed_messages,
            tool_constraints,
            mut spec,
        } = prep
        else {
            debug_assert!(false, "pipeline guarantees Messages variant");
            return Err(error::internal_error(
                "wrong_preparation_type",
                "Expected Messages preparation output",
            ));
        };

        let continued_final_message = processed_messages.continued_final_message;
        let require_reasoning = reasoning_decision(
            ctx,
            &messages_request,
            &processed_messages.text,
            continued_final_message,
        );

        let names = ctx
            .components
            .parser_resolver
            .names(&messages_request.model);
        let parser = ctx.state.parser.unwrap_or_default();
        let ConstraintForWorker {
            auto_tag,
            tool_constraints,
            decided_on,
        } = constraint_for_worker(
            ctx,
            &messages_request,
            &spec.chat_tools,
            &processed_messages.text,
            continued_final_message,
            require_reasoning,
            tool_constraints,
            &names,
            parser,
        )?;
        if let Some(decoding) = decided_on {
            ctx.state.constraint_decided_on = Some(decoding);
        }

        // Symphony's assignment, taken while the rendered prompt is still here (see
        // `regular::symphony`): the response spec carries it to the drivers, and the state
        // records which parser reads the message for the response's header. It takes the
        // constraint as the request carries it, before the worker's shaping in the plan
        // builder below: the assignment reads the constraint's kind alone, which the shaping
        // leaves as it is, in the order the chat builder has.
        let assignment = symphony::assign(
            parser,
            &names,
            symphony::Asks::Messages {
                request: messages_request.as_ref(),
                continues: messages_request.continues_final_assistant(),
            },
            tool_constraints.as_ref(),
            &processed_messages.text,
            &ctx.components.templates,
        )
        .map_err(|why| error::bad_request("invalid_parser_configuration", why))?;
        ctx.state.parser = Some(match assignment {
            Some(_) => ParserChoice::Symphony,
            None => ParserChoice::Legacy,
        });

        let (plan, stamp) = build_messages_backed_plan(
            ctx,
            &messages_request,
            processed_messages.text,
            token_ids,
            tool_constraints,
            auto_tag,
            require_reasoning,
            multimodal_data,
            self.inject_pd_metadata,
            self.plan_kind,
        )?;
        spec.reasoning_in_prefill = require_reasoning;
        spec.symphony = assignment;
        Ok(BuildOutput {
            plan,
            spec: ResponseSpec::Messages(spec),
            stamp,
        })
    }

    fn name(&self) -> &'static str {
        "MessageRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        format!(
            "MessageRequestBuildingStage(inject_pd_metadata={}, {:?})",
            self.inject_pd_metadata, self.plan_kind
        )
    }
}
