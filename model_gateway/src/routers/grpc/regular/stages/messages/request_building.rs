//! Message request building stage: Build proto GenerateRequest for message requests

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::messages;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        client::GenerateRequestBuildOptions,
        common::stages::{helpers, PipelineStage},
        context::{
            ClientSelection, ExecutionPlan, ExecutionPlanKind, PreparationOutput, RequestContext,
        },
        multimodal::{assemble_multimodal_data, assemble_multimodal_data_after_encode},
        regular::views,
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

#[async_trait]
impl PipelineStage for MessageRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Take preparation state (last consumer — worker_selection already ran)
        let prep = ctx.state.preparation.take().ok_or_else(|| {
            error!(
                function = "MessageRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "MessageRequestBuildingStage::execute",
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
        let request_id = helpers::resolve_request_id(
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
                error!(function = "MessageRequestBuildingStage::execute", error = %e, "Failed to assemble multimodal request");
                error::bad_request("multimodal_not_supported", format!("{e}"))
            })?)
        } else {
            None
        };

        let user_thinking = match &messages_request.thinking {
            Some(messages::ThinkingConfig::Enabled { .. })
            | Some(messages::ThinkingConfig::Adaptive { .. }) => Some(true),
            Some(messages::ThinkingConfig::Disabled) => Some(false),
            None => None,
        };
        let require_reasoning = ctx.tokenizer_arc().is_some_and(|tokenizer| {
            utils::should_mark_reasoning_started(user_thinking, tokenizer.as_ref())
        });

        let mut proto_request = builder_client
            .build_messages_request(
                request_id,
                &messages_request,
                processed_messages.text,
                token_ids,
                GenerateRequestBuildOptions {
                    multimodal_inputs: multimodal_data,
                    tool_constraints,
                    require_reasoning,
                },
            )
            .map_err(|e| {
                error!(function = "MessageRequestBuildingStage::execute", error = %e, "Failed to build generate request");
                error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {e}"))
            })?;

        helpers::apply_sampling_defaults_to_generate_request(
            &mut proto_request,
            &ctx.input.request_type,
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

        // Last request reader before dispatch: extract the response-phase view
        // so response processing never needs the (possibly released) payload.
        ctx.state.response.request_view = Some(views::RequestView::Messages(
            views::MessagesRequestView::from(messages_request.as_ref()),
        ));

        ctx.state.execution_plan = Some(ExecutionPlan::generate(self.plan_kind, proto_request));
        Ok(None)
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
