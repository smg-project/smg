//! Chat request building stage: Build proto GenerateRequest for chat requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        client::GenerateRequestBuildOptions,
        common::stages::{helpers, BuildStage},
        context::{
            AttemptStamp, BuildOutput, ClientSelection, ExecutionPlan, ExecutionPlanKind,
            PreparationOutput, RequestContext,
        },
        multimodal::{
            assemble_media_refs, assemble_multimodal_data, assemble_multimodal_data_after_encode,
            reserve_multimodal_inflight,
        },
        spec::{ChatResponseSpec, ResponseSpec},
        utils,
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

/// Build the backend `GenerateRequest` from a chat-shaped request, shared by
/// the chat endpoint and the transcription endpoint. Assembles multimodal
/// data, applies sampling defaults (from the chat request), finalizes stops,
/// and injects PD/EPD metadata — returning the plan + attempt stamp. The
/// caller supplies the `ResponseSpec`.
#[expect(clippy::too_many_arguments)]
pub(crate) async fn build_chat_backed_plan(
    ctx: &mut RequestContext,
    chat_request: &openai_protocol::chat::ChatCompletionRequest,
    processed_text: String,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
    id_prefix: &'static str,
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
) -> Result<(ExecutionPlan, AttemptStamp), Response> {
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
            error::bad_request("multimodal_not_supported", format!("{e}"))
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

    // On a thinking prompt the engine defers a grammar past the model's own
    // `</think>` (the request's `require_reasoning`; an engine that runs a
    // reasoning parser does so by itself). A forced-call tag owes nothing before the
    // calls, so it reads the same with or without the deferral.
    let require_reasoning = ctx.tokenizer_arc().is_some_and(|tokenizer| {
        utils::chat_reasoning_starts_in_prefill(chat_request, tokenizer.as_ref())
    });

    // Now that the worker is known, a forced call's grammar takes the shape
    // its engine needs: the bare calls behind a reasoning parser's gate, the
    // thought and then the calls from the first token (the worker's
    // `structured_outputs_start` label; see `utils::tool_constraints_for_worker`).
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
        let vllm_json_tool = tool_constraints
            .as_ref()
            .is_some_and(|(kind, _)| kind == "json_schema")
            && ctx
                .state
                .clients
                .as_ref()
                .is_some_and(|clients| match clients {
                    ClientSelection::Single { client } => {
                        client.runtime_type() == crate::worker::RuntimeType::Vllm
                    }
                    ClientSelection::Disaggregated { prefill, .. } => {
                        prefill.runtime_type() == crate::worker::RuntimeType::Vllm
                    }
                });
        let reasoning_in_prefill = ctx
            .components
            .parser_resolver
            .reasoning_parser(&chat_request.model)
            .filter(|name| name == "gemma4")
            .map(|_| {
                reasoning_parser::parsers::Gemma4Parser::prompt_ends_in_reasoning(
                    &processed_messages.text,
                )
            });
        let (plan, stamp) = build_chat_backed_plan(
            ctx,
            &chat_request,
            processed_messages.text,
            token_ids,
            tool_constraints,
            "chatcmpl-",
            self.inject_pd_metadata,
            self.plan_kind,
        )
        .await?;

        // Only the client-facing usage drops them; settlement keeps the engine's count.
        ctx.state.response.unbilled_prompt_tokens = unbilled_prompt_tokens;
        let spec = chat_response_spec(
            chat_request.as_ref(),
            unbilled_prompt_tokens,
            reasoning_in_prefill,
            vllm_json_tool,
        );

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

pub(crate) fn chat_response_spec(
    request: &openai_protocol::chat::ChatCompletionRequest,
    unbilled_prompt_tokens: u32,
    reasoning_in_prefill: Option<bool>,
    vllm_json_tool: bool,
) -> ChatResponseSpec {
    let mut spec = ChatResponseSpec::from(request);
    spec.unbilled_prompt_tokens = unbilled_prompt_tokens;
    spec.reasoning_in_prefill = reasoning_in_prefill;
    // vLLM's JSON tool grammar applies from token zero and cannot emit a
    // reasoning prefix. Parse its payload as tools, including literal markers
    // inside arguments. SGLang's deferred grammar and native tools are unchanged.
    if vllm_json_tool {
        spec.separate_reasoning = false;
    }
    spec
}

#[cfg(test)]
mod response_spec_tests {
    use super::*;

    #[test]
    fn native_and_deferred_tool_constraints_keep_reasoning_preferences() {
        let request = serde_json::from_value(serde_json::json!({
            "model":"test-model", "messages":[], "separate_reasoning":true,
            "chat_template_kwargs":{"enable_thinking":true}
        }))
        .unwrap();
        // `false` includes auto/native/structural-tag vLLM requests and
        // SGLang, whose require_reasoning defers its JSON constraint.
        for in_prefill in [None, Some(false), Some(true)] {
            let spec = chat_response_spec(&request, 7, in_prefill, false);
            assert!(spec.separate_reasoning);
            assert_eq!(spec.reasoning_in_prefill, in_prefill);
            assert_eq!(spec.unbilled_prompt_tokens, 7);
            assert_eq!(spec.chat_template_kwargs, request.chat_template_kwargs);
        }
    }
}
