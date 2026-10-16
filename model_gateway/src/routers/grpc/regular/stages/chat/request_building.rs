//! Chat request building stage: Build proto GenerateRequest for chat requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::{
    config::ParserChoice,
    routers::{
        error,
        grpc::{
            client::GenerateRequestBuildOptions,
            common::stages::{helpers, BuildStage},
            context::{
                AttemptStamp, BuildOutput, ClientSelection, ExecutionPlan, ExecutionPlanKind,
                PreparationOutput, RequestContext,
            },
            multimodal::{
                assemble_media_refs, assemble_multimodal_data,
                assemble_multimodal_data_after_encode, assembly_error_response,
                reserve_multimodal_inflight,
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
/// the engine's reasoning gate.
fn reasoning_decision(
    ctx: &RequestContext,
    chat_request: &openai_protocol::chat::ChatCompletionRequest,
    processed_text: &str,
    continued_final_message: bool,
) -> bool {
    utils::rendered_prompt_reasoning_in_prefill(
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
    })
}

/// Build the backend `GenerateRequest` from a chat-shaped request, shared by
/// the chat endpoint and the transcription endpoint. Assembles multimodal
/// data, applies sampling defaults (from the chat request), finalizes stops,
/// and injects PD/EPD metadata — returning the plan, attempt stamp and rendered
/// reasoning state. The caller supplies the `ResponseSpec`.
/// `continued_final_message` is the rendered prompt's
/// (`ProcessedMessages::continued_final_message`).
#[expect(clippy::too_many_arguments)]
pub(crate) async fn build_chat_backed_plan(
    ctx: &mut RequestContext,
    chat_request: &openai_protocol::chat::ChatCompletionRequest,
    processed_text: String,
    continued_final_message: bool,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
    id_prefix: &'static str,
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
) -> Result<(ExecutionPlan, AttemptStamp, bool), Response> {
    let clients = ctx.state.clients.as_ref().ok_or_else(|| {
        error!(
            function = "build_chat_backed_plan",
            "Client acquisition not completed"
        );
        error::internal_error(
            "client_acquisition_not_completed",
            "Client acquisition not completed",
        )
    })?;

    // Get client for building request (use prefill client in disaggregated mode)
    let builder_client = match clients {
        ClientSelection::Single { client } => client,
        ClientSelection::Disaggregated { prefill, .. } => prefill,
    };

    let disaggregated = matches!(clients, ClientSelection::Disaggregated { .. });
    let (request_id, id_stamp) = helpers::resolve_request_id_stamp(
        &ctx.input.request_type,
        ctx.input.tenant_request_meta.as_ref(),
        id_prefix,
        disaggregated,
    );

    // `encode_outputs` set by EncodeStage selects the pixel-drop assembly path.
    let is_encode_routed = ctx.state.encode_outputs.is_some();

    // Assemble backend-specific multimodal data now that the backend is known;
    // take the intermediate here for the prefill serialization. When
    // encode-routed, drop the prefill pixels.
    let multimodal_data = if let Some(intermediate) = ctx.state.multimodal_intermediate.take() {
        let assembled = if is_encode_routed {
            assemble_multimodal_data_after_encode(
                intermediate,
                builder_client,
                ctx.state.workers.as_ref(),
            )
            .await
        } else {
            assemble_multimodal_data(intermediate, builder_client, ctx.state.workers.as_ref()).await
        };
        Some(assembled.map_err(|e| {
            error!(function = "build_chat_backed_plan", error = %e, "Failed to assemble multimodal request");
            assembly_error_response(&e)
        })?)
    } else {
        None
    };
    if let Some(data) = multimodal_data.as_ref() {
        ctx.state.multimodal_inflight = reserve_multimodal_inflight(
            ctx.components
                .multimodal
                .as_ref()
                .and_then(|multimodal| multimodal.inflight.as_deref()),
            data.inline_bytes(),
        )
        .await?;
    }

    let require_reasoning =
        reasoning_decision(ctx, chat_request, &processed_text, continued_final_message);

    // Now that the worker is known, a forced call's grammar takes the shape
    // its engine needs: the bare calls behind a reasoning parser's gate, the
    // thought and then the calls from the first token (the worker's
    // `constrained_decoding_start` label; see `utils::tool_constraints_for_worker`).
    // The unshaped constraint stays on the attempt stamp: a retry gives the
    // worker it selects the same shaping (`helpers::ToolConstraintStamp`).
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

    // The client resolves string `stop`s its engine can't match and reports
    // the router's residual trim obligation; no transport knowledge here.
    ctx.state.response.router_stop_obligations =
        builder_client.finalize_generate_request(&mut proto_request, ctx.tokenizer_arc().as_ref());

    if inject_pd_metadata {
        if let Some(workers) = ctx.state.workers.as_ref() {
            helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
        }
    }

    // EPD: inject the per-item encode bootstrap info into the prefill request;
    // the dispatch plan stays on `encode_outputs` for request execution to take.
    if let Some(outputs) = ctx.state.encode_outputs.as_mut() {
        proto_request.set_encode_bootstrap_info(std::mem::take(&mut outputs.bootstrap_info));
    }

    // EPD: inject the prefill->decode KV rendezvous for backends that carry it
    // in the request. Runs before execute_parallel_pd clones the request, so
    // both prefill and decode carry the same room.
    if let Some(workers) = ctx.state.workers.as_ref() {
        helpers::maybe_inject_pd_rendezvous(&mut proto_request, workers);
    }

    // Worker-side multimodal processing: attach the media references now that
    // the wire is known, before the PD clone so both legs carry them.
    if let Some(plan) = ctx.state.multimodal_refs.take() {
        if builder_client.is_zmq() {
            return Err(error::bad_request(
                "multimodal_not_supported",
                "media references require a gRPC vLLM worker",
            ));
        }
        let refs =
            assemble_media_refs(plan).map_err(|e| error::bad_request(e.code(), e.to_string()))?;
        proto_request
            .set_vllm_media_refs(refs)
            .map_err(|e| error::bad_request("multimodal_not_supported", e))?;
        ctx.state.media_refs_forwarded = true;
    }

    let tool_constraint = helpers::ToolConstraintStamp::capture(
        unshaped_tool_constraints,
        tool_parser.as_deref(),
        &proto_request,
    );
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
        let prep = ctx.state.preparation.take().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::build",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

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
        // suits any worker. The attempt stamp carries the tag unshaped like a
        // forced call's, so the retry's worker shapes it for where its engine
        // starts the grammar.
        let decoding = utils::WorkerDecoding::of(ctx.state.workers.as_ref());
        let decided = symphony::worker_constraint(
            decoding,
            tool_constraints,
            || {
                symphony::turn_constraint(
                    parser,
                    &names,
                    &chat_request,
                    &processed_messages.text,
                    &ctx.components.templates,
                )
            },
            || {
                let reasoning = reasoning_decision(
                    ctx,
                    &chat_request,
                    &processed_messages.text,
                    processed_messages.continued_final_message,
                );
                symphony::auto_constraint(
                    parser,
                    &names,
                    &chat_request,
                    &chat_request.callable_tools(),
                    reasoning,
                    &ctx.components.templates,
                )
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
            chat_request.as_ref(),
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
            "chatcmpl-",
            self.inject_pd_metadata,
            self.plan_kind,
        )
        .await?;

        // Only the client-facing usage drops them; settlement keeps the engine's count.
        ctx.state.response.unbilled_prompt_tokens = unbilled_prompt_tokens;
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
