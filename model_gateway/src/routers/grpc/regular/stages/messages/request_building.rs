//! Message request building stage: Build proto GenerateRequest for message requests

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

#[async_trait]
impl BuildStage for MessageRequestBuildingStage {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        // Take preparation state (last consumer — worker_selection already ran)
        let prep = ctx.state.preparation.take().ok_or_else(|| {
            error!(
                function = "MessageRequestBuildingStage::build",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "MessageRequestBuildingStage::build",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let messages_request = ctx.messages_request_arc();

        // Get client for building request (use prefill client in disaggregated mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Disaggregated { prefill, .. } => prefill,
        };

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

        // Build message request
        let disaggregated = matches!(clients, ClientSelection::Disaggregated { .. });
        let (request_id, id_stamp) = helpers::resolve_request_id_stamp(
            &ctx.input.request_type,
            ctx.input.tenant_request_meta.as_ref(),
            "msg_",
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
                assemble_multimodal_data(intermediate, builder_client, ctx.state.workers.as_ref())
                    .await
            };
            Some(assembled.map_err(|e| {
                error!(function = "MessageRequestBuildingStage::build", error = %e, "Failed to assemble multimodal request");
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

        let continued_final_message = processed_messages.continued_final_message;
        // Whether the rendered prompt leaves the model inside its thought:
        // native Gemma4 channels follow the rendered prompt when that parser
        // decides; other parsers keep their tokenizer-driven preference for
        // the engine's reasoning gate.
        let require_reasoning = utils::rendered_prompt_reasoning_in_prefill(
            &ctx.components.reasoning_parser_factory,
            ctx.components
                .parser_resolver
                .reasoning_parser(&messages_request.model)
                .as_deref(),
            &messages_request.model,
            &processed_messages.text,
        )
        .unwrap_or_else(|| {
            ctx.tokenizer_arc().is_some_and(|tokenizer| {
                utils::messages_reasoning_starts_in_prefill(
                    &messages_request,
                    continued_final_message,
                    tokenizer.as_ref(),
                )
            })
        });

        // Now that the worker is known, a forced call's grammar takes the
        // shape its engine needs: the bare calls behind a reasoning parser's
        // gate, the thought and then the calls from the first token (the
        // worker's `constrained_decoding_start` label; see
        // `utils::tool_constraints_for_worker`).
        // The unshaped constraint stays on the attempt stamp: a retry gives
        // the worker it selects the same shaping (`helpers::ToolConstraintStamp`).
        let tool_parser = ctx
            .components
            .parser_resolver
            .tool_parser(&messages_request.model);
        let names = ctx
            .components
            .parser_resolver
            .names(&messages_request.model);
        let parser = ctx.state.parser.unwrap_or_default();
        // What constrains the output is decided here, on the worker selected
        // for the request (`regular::symphony::worker_constraint`): the whole
        // turn's grammar on a worker whose engine takes one on every turn from
        // its first token; else what preparation found, and on a worker that
        // takes a grammar on every turn the table's tag for the calls the
        // request allows under `auto`. When the worker's policy chose the
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
                symphony::turn_constraint_messages(
                    parser,
                    &names,
                    &messages_request,
                    &spec.chat_tools,
                    &processed_messages.text,
                    messages_request.continues_final_assistant(),
                    &ctx.components.templates,
                )
            },
            || {
                let tools = ctx
                    .tokenizer_arc()
                    .map(|tokenizer| {
                        utils::template_tools(
                            &spec.chat_tools,
                            messages_request.tool_choice.as_ref(),
                            tokenizer.renderer_capabilities(),
                        )
                    })
                    .unwrap_or_default();
                symphony::auto_constraint_messages(
                    parser,
                    &names,
                    &messages_request,
                    &tools,
                    require_reasoning,
                    symphony::grammar_start_from(decoding.start),
                    continued_final_message,
                    &ctx.components.templates,
                )
            },
        )
        .map_err(|why| error::bad_request("invalid_parser_configuration", why))?;
        if decided.by_policy {
            ctx.state.constraint_decided_on = Some(decoding);
        }
        let tool_constraints = decided.constraint.map(|constraint| constraint.to_tuple());
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
        // Symphony's assignment, taken while the rendered prompt is still here (see
        // `regular::symphony`): the response spec carries it to the drivers, and the state
        // records which parser reads the message for the response's header.
        let assignment = symphony::assign_messages(
            parser,
            &names,
            messages_request.as_ref(),
            tool_constraints.as_ref(),
            &processed_messages.text,
            &ctx.components.templates,
        )
        .map_err(|why| error::bad_request("invalid_parser_configuration", why))?;
        ctx.state.parser = Some(match assignment {
            Some(_) => ParserChoice::Symphony,
            None => ParserChoice::Legacy,
        });
        let carries_media = multimodal_data.is_some() || ctx.state.multimodal_refs.is_some();
        let mut proto_request = builder_client
            .build_messages_request(
                request_id,
                &messages_request,
                helpers::wire_prompt_text(processed_messages.text, carries_media),
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

        let sampling_mask =
            helpers::SamplingDefaultsMask::from_request_type(&ctx.input.request_type);
        let sampling_baseline = helpers::apply_sampling_defaults(
            &mut proto_request,
            sampling_mask,
            ctx.state.workers.as_ref(),
        );

        // The client resolves string `stop`s its engine can't match and
        // reports the router's residual trim obligation; no transport
        // knowledge needed here.
        ctx.state.response.router_stop_obligations = builder_client
            .finalize_generate_request(&mut proto_request, ctx.tokenizer_arc().as_ref());

        if self.inject_pd_metadata {
            if let Some(workers) = ctx.state.workers.as_ref() {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        // EPD: inject the per-item encode bootstrap info into the prefill
        // request; the dispatch plan stays on `encode_outputs` for request
        // execution to take.
        if let Some(outputs) = ctx.state.encode_outputs.as_mut() {
            proto_request.set_encode_bootstrap_info(std::mem::take(&mut outputs.bootstrap_info));
        }

        // EPD: inject the prefill->decode KV rendezvous (mirrors the chat path).
        // No-op unless the backend carries it in the request.
        if let Some(workers) = ctx.state.workers.as_ref() {
            helpers::maybe_inject_pd_rendezvous(&mut proto_request, workers);
        }

        // Worker-side multimodal processing: attach the media references now
        // that the wire is known, before the PD clone so both legs carry them.
        if let Some(plan) = ctx.state.multimodal_refs.take() {
            if builder_client.is_zmq() {
                return Err(error::bad_request(
                    "multimodal_not_supported",
                    "media references require a gRPC vLLM worker",
                ));
            }
            let refs = assemble_media_refs(plan)
                .map_err(|e| error::bad_request(e.code(), e.to_string()))?;
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
        spec.reasoning_in_prefill = require_reasoning;
        spec.symphony = assignment;
        Ok(BuildOutput {
            plan: ExecutionPlan::generate(self.plan_kind, proto_request),
            spec: ResponseSpec::Messages(spec),
            stamp: AttemptStamp {
                id: id_stamp,
                sampling_mask,
                sampling_baseline,
                inject_pd_metadata: self.inject_pd_metadata,
                tool_constraint,
            },
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
