//! Message request building stage: Build proto GenerateRequest for message requests

use std::{borrow::Cow, sync::Arc};

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
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
        regular::{stages::reading, symphony},
        spec::ResponseSpec,
        utils,
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

/// Build the backend `GenerateRequest` from the Messages request: the
/// request id stamped for the attempt, the forced call's grammar shaped for
/// the worker (the bare calls behind a reasoning parser's gate, the thought
/// and then the calls from the first token: the worker's
/// `constrained_decoding_start` label, `utils::tool_constraints_for_worker`),
/// the sampling defaults, the stops and the PD metadata; returns the plan
/// and the attempt stamp, which carries the unshaped constraint, or what
/// derives the table's tag for a start (the `auto` tag, a JSON answer's), so
/// a retry gives the worker it selects the same shaping
/// (`helpers::ToolConstraintStamp`).
#[expect(clippy::too_many_arguments)]
fn build_messages_backed_plan(
    ctx: &mut RequestContext,
    messages_request: &openai_protocol::messages::CreateMessageRequest,
    processed_text: String,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
    table_tag: Option<Arc<dyn TagForStart>>,
    first_token_tag: bool,
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
    let (tool_constraints, led_for_first_token) = utils::tool_constraints_for_worker(
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
                first_token_grammar: first_token_tag || led_for_first_token,
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

    let tool_constraint = match table_tag {
        Some(table_tag) => helpers::ToolConstraintStamp::capture_table(table_tag, &proto_request),
        None => helpers::ToolConstraintStamp::capture(
            unshaped_tool_constraints,
            tool_parser.as_deref(),
            first_token_tag,
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
        let (require_reasoning, thought_closed) = reading::reasoning_decision(
            ctx,
            &messages_request.model,
            &processed_messages.text,
            |tokenizer| {
                utils::messages_reasoning_starts_in_prefill(
                    &messages_request,
                    continued_final_message,
                    tokenizer,
                )
            },
        );

        // What constrains the output on the selected worker, and Symphony's
        // assignment, decided as for a chat request (`reading::decide`), in
        // the order the chat builder has: the assignment takes the constraint
        // as the request carries it, before the worker's shaping in the plan
        // builder below. The `auto` tag derives for the request over the
        // tools the template sees. The state records which parser reads the
        // message, for the response's header.
        let reading::Decided {
            table_tag,
            tool_constraints,
            first_token,
            assignment,
            ..
        } = reading::decide(
            ctx,
            symphony::Asks::Messages {
                request: &messages_request,
                continues: messages_request.continues_final_assistant(),
            },
            &spec.chat_tools,
            &processed_messages.text,
            tool_constraints,
            |_| {
                // The tag shares the spec's tools where `tool_choice` narrows
                // nothing (the common case): no copy of the schemas on the
                // first attempt; a narrowed list is the one the narrowing owns.
                let tools: Arc<[_]> = match utils::template_tools(
                    &spec.chat_tools,
                    messages_request.tool_choice.as_ref(),
                ) {
                    Cow::Owned(narrowed) => Arc::from(narrowed),
                    Cow::Borrowed([]) => Arc::default(),
                    Cow::Borrowed(_) => Arc::clone(&spec.chat_tools),
                };
                reading::Auto {
                    request: symphony::AutoRequest::Messages {
                        request: Arc::clone(&messages_request),
                        tools,
                        continues: continued_final_message,
                    },
                    reasoning_open: require_reasoning,
                    thought_closed,
                }
            },
        )?;

        let (plan, stamp) = build_messages_backed_plan(
            ctx,
            &messages_request,
            processed_messages.text,
            token_ids,
            tool_constraints,
            table_tag,
            first_token,
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
