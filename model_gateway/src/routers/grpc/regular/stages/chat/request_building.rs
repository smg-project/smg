//! Chat request building stage: Build proto GenerateRequest for chat requests

use std::sync::Arc;

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
            regular::symphony,
            spec::{ChatResponseSpec, ResponseSpec},
            utils,
        },
    },
};

/// Chat request building stage
///
/// Extracts chat-specific request building logic from the old unified RequestBuildingStage.
pub(crate) struct ChatRequestBuildingStage {
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
}

impl ChatRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            inject_pd_metadata,
            plan_kind,
        }
    }
}

/// Whether the rendered prompt leaves the model inside its thought, as the
/// request's grammar, the plan's reasoning state and the response spec all
/// read it: native Gemma4 channels follow the rendered prompt when that
/// parser decides; other parsers keep their tokenizer-driven preference for
/// the engine's reasoning gate; and where a Symphony table reads the
/// response, the table's own replay of the prompt (a turn the prompt leaves
/// unopened starts inside the thought: `regular::symphony::reasoning_open`).
fn reasoning_decision(
    ctx: &RequestContext,
    chat_request: &openai_protocol::chat::ChatCompletionRequest,
    processed_text: &str,
    continued_final_message: bool,
) -> bool {
    let switch = utils::rendered_prompt_reasoning_in_prefill(
        &ctx.components.reasoning_parser_factory,
        ctx.components
            .parser_resolver
            .reasoning_parser(&chat_request.model)
            .as_deref(),
        &chat_request.model,
        processed_text,
    )
    .unwrap_or_else(|| {
        ctx.tokenizer_arc().is_some_and(|tokenizer| {
            utils::chat_reasoning_starts_in_rendered_prefill(
                chat_request,
                continued_final_message,
                tokenizer.as_ref(),
            )
        })
    });
    switch
        || symphony::reasoning_open(
            ctx.state.parser.unwrap_or_default(),
            &ctx.components.parser_resolver.names(&chat_request.model),
            &chat_request.model,
            &ctx.components.templates,
            processed_text,
        )
}

/// Build the backend `GenerateRequest` from a chat-shaped request, shared by
/// the chat endpoint and the transcription endpoint. Assembles multimodal
/// data, applies sampling defaults (from the chat request), finalizes stops,
/// and injects PD/EPD metadata — returning the plan, attempt stamp and rendered
/// reasoning state. The caller supplies the `ResponseSpec`.
/// `continued_final_message` is the rendered prompt's
/// (`ProcessedMessages::continued_final_message`). `auto_tag` derives the
/// table's `auto` tag for a worker's grammar start when `tool_constraints` is
/// that tag (`regular::symphony::WorkerConstraint::auto`).
#[expect(clippy::too_many_arguments)]
pub(crate) async fn build_chat_backed_plan(
    ctx: &mut RequestContext,
    chat_request: &openai_protocol::chat::ChatCompletionRequest,
    processed_text: String,
    continued_final_message: bool,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
    auto_tag: Option<Arc<dyn TagForStart>>,
    id_prefix: &'static str,
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
) -> Result<(ExecutionPlan, AttemptStamp, bool), Response> {
    let multimodal_data = helpers::assemble_prompt_media(ctx, "build_chat_backed_plan").await?;

    let clients = stage_output(ctx.state.clients.as_ref(), "build_chat_backed_plan")?;

    let builder_client = clients.builder();
    let disaggregated = matches!(clients, ClientSelection::Disaggregated { .. });
    let (request_id, id_stamp) = helpers::resolve_request_id_stamp(
        &ctx.input.request_type,
        ctx.input.tenant_request_meta.as_ref(),
        id_prefix,
        disaggregated,
    );

    let require_reasoning =
        reasoning_decision(ctx, chat_request, &processed_text, continued_final_message);

    // Now that the worker is known, a forced call's grammar takes the shape
    // its engine needs: the bare calls behind a reasoning parser's gate, the
    // thought and then the calls from the first token (the worker's
    // `constrained_decoding_start` label; see `utils::tool_constraints_for_worker`).
    // The unshaped constraint stays on the attempt stamp, or what derives the
    // table's `auto` tag for a start: a retry gives the worker it selects the
    // same shaping (`helpers::ToolConstraintStamp`).
    let tool_parser = ctx
        .components
        .parser_resolver
        .tool_parser(&chat_request.model);
    let unshaped_tool_constraints = tool_constraints.clone();
    let tool_constraints = utils::tool_constraints_for_worker(
        &ctx.components.tool_parser_factory,
        tool_parser.as_deref(),
        tool_constraints,
        ctx.state.workers.as_ref(),
    )
    .map_err(|e| {
        error!(function = "build_chat_backed_plan", error = %e, "Failed to shape the tool constraint for the worker");
        error::internal_error("invalid_tool_constraint", format!("Invalid tool constraint: {e}"))
    })?;

    // The ids are the prompt; the text rides along only when the request
    // carries media (see `helpers::wire_prompt_text`).
    let carries_media = multimodal_data.is_some() || ctx.state.multimodal_refs.is_some();
    let mut proto_request = builder_client
        .build_chat_request(
            request_id,
            chat_request,
            helpers::wire_prompt_text(processed_text, carries_media),
            token_ids,
            GenerateRequestBuildOptions {
                multimodal_inputs: multimodal_data,
                tool_constraints,
                require_reasoning,
            },
        )
        .map_err(|e| {
            error!(function = "build_chat_backed_plan", error = %e, "Failed to build generate request");
            error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {e}"))
        })?;

    let sampling_mask = Some(helpers::SamplingDefaultsMask::from_chat_request(
        chat_request,
    ));
    let sampling_baseline = helpers::apply_sampling_defaults(
        &mut proto_request,
        sampling_mask,
        ctx.state.workers.as_ref(),
    );

    helpers::finish_generate_request(
        ctx,
        &mut proto_request,
        inject_pd_metadata,
        "build_chat_backed_plan",
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
        require_reasoning,
    ))
}

#[async_trait]
impl BuildStage for ChatRequestBuildingStage {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        // Take preparation state (last consumer — worker_selection already ran)
        let prep = stage_output(
            ctx.state.preparation.take(),
            "ChatRequestBuildingStage::build",
        )?;

        let chat_request = ctx.chat_request_arc();

        let PreparationOutput::Chat {
            token_ids,
            processed_messages,
            tool_constraints,
        } = prep
        else {
            debug_assert!(false, "pipeline guarantees Chat variant");
            return Err(error::internal_error(
                "wrong_preparation_type",
                "Expected Chat preparation output",
            ));
        };

        let unbilled_prompt_tokens = processed_messages.unbilled_prompt_tokens;
        // Symphony's assignment, taken while the rendered prompt is still here (see
        // `regular::symphony`): the response spec carries it to the drivers. The
        // parser asked for is the one chat preparation chose for this request;
        // the one that reads the response is Symphony only when a table takes
        // the request, so the state records that for the response's header.
        let names = ctx.components.parser_resolver.names(&chat_request.model);
        let parser = ctx.state.parser.unwrap_or_default();
        // What constrains the output is decided here, on the worker selected
        // for the request (`regular::symphony::worker_constraint`): the whole
        // turn's grammar on a worker whose engine takes one on every turn from
        // its first token; else what preparation found, and on a worker that
        // takes a grammar on every turn the table's tag for the calls the
        // request allows under `auto`, built on the reasoning state the plan
        // takes (`reasoning_decision`). When the worker's policy chose the
        // constraint, a retry re-selects only workers that decide alike
        // (`utils::WorkerDecoding::decides_alike`); what the request asked for
        // suits any worker. The attempt stamp carries what derives the `auto`
        // tag for a worker's start, so the retry's worker gets it shaped for
        // where its own engine starts the grammar.
        let decoding = utils::WorkerDecoding::of(ctx.state.workers.as_ref());
        let decided = symphony::worker_constraint(
            decoding,
            tool_constraints,
            || {
                symphony::turn_constraint(
                    parser,
                    &names,
                    symphony::Asks::Chat(&chat_request),
                    &chat_request.effective_tool_list(),
                    &processed_messages.text,
                    &ctx.components.templates,
                )
            },
            || symphony::AutoTag {
                parser,
                labels: names.labels.clone(),
                flags: names.flags.clone(),
                request: symphony::AutoRequest::Chat(Arc::clone(&chat_request)),
                reasoning_open: reasoning_decision(
                    ctx,
                    &chat_request,
                    &processed_messages.text,
                    processed_messages.continued_final_message,
                ),
                templates: Arc::clone(&ctx.components.templates),
            },
        )
        .map_err(|why| error::bad_request("invalid_parser_configuration", why))?;
        if decided.by_policy {
            ctx.state.constraint_decided_on = Some(decoding);
        }
        let auto_tag = decided.auto.map(|auto| -> Arc<dyn TagForStart> { auto });
        let tool_constraints = decided.constraint.map(|constraint| constraint.to_tuple());
        let assignment = symphony::assign(
            parser,
            &names,
            symphony::Asks::Chat(chat_request.as_ref()),
            tool_constraints.as_ref(),
            &processed_messages.text,
            &ctx.components.templates,
        )
        .map_err(|why| error::bad_request("invalid_parser_configuration", why))?;
        ctx.state.parser = Some(match assignment {
            Some(_) => ParserChoice::Symphony,
            None => ParserChoice::Legacy,
        });
        // The table's structural tag stands in for a JSON response_format (see
        // `regular::symphony::constraint`), and the whole turn's grammar for a
        // regex or an ebnf too: the engine's request carries the tag alone, so
        // no client reads the two as competing constraints.
        let answered = assignment.is_some()
            && chat_request.response_format.is_some()
            && tool_constraints
                .as_ref()
                .is_some_and(|(kind, _)| kind == "structural_tag");
        let engine_request = (answered || decided.whole_turn).then(|| {
            let mut request = chat_request.as_ref().clone();
            request.response_format = None;
            if decided.whole_turn {
                request.regex = None;
                request.ebnf = None;
            }
            request
        });
        let engine_request = match &engine_request {
            Some(request) => request,
            None => chat_request.as_ref(),
        };
        let continued_final_message = processed_messages.continued_final_message;
        let (plan, stamp, reasoning_in_prefill) = build_chat_backed_plan(
            ctx,
            engine_request,
            processed_messages.text,
            continued_final_message,
            token_ids,
            tool_constraints,
            auto_tag,
            "chatcmpl-",
            self.inject_pd_metadata,
            self.plan_kind,
        )
        .await?;

        // Only the client-facing usage drops them; settlement keeps the engine's count.
        let mut spec = ChatResponseSpec::from(chat_request.as_ref());
        spec.unbilled_prompt_tokens = unbilled_prompt_tokens;
        spec.reasoning_in_prefill = reasoning_in_prefill;
        spec.symphony = assignment;

        Ok(BuildOutput {
            plan,
            spec: ResponseSpec::Chat(Box::new(spec)),
            stamp,
        })
    }

    fn name(&self) -> &'static str {
        "ChatRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        format!(
            "ChatRequestBuildingStage(inject_pd_metadata={}, {:?})",
            self.inject_pd_metadata, self.plan_kind
        )
    }
}
