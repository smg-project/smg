//! Generate request building stage: Build proto GenerateRequest for generate requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, ExecutionPlan, ExecutionPlanKind, RequestContext},
        regular::views,
    },
};

/// Generate request building stage
///
/// Extracts generate-specific request building logic from the old unified RequestBuildingStage.
pub(crate) struct GenerateRequestBuildingStage {
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
}

impl GenerateRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            inject_pd_metadata,
            plan_kind,
        }
    }
}

#[async_trait]
impl PipelineStage for GenerateRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "GenerateRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "GenerateRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let generate_request = ctx.generate_request_arc();

        // Get client for building request (use prefill client in disaggregated mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Disaggregated { prefill, .. } => prefill,
        };

        let disaggregated = matches!(clients, ClientSelection::Disaggregated { .. });
        let request_id = helpers::resolve_request_id(
            &ctx.input.request_type,
            ctx.input.tenant_request_meta.as_ref(),
            "gen-",
            disaggregated,
        );

        // Build proto request using centralized dispatch
        let mut proto_request = builder_client
            .build_generate_request(
                request_id,
                &generate_request,
                prep.routing_text().map(String::from),
                prep.token_ids().to_vec(),
            )
            .map_err(|e| {
                error!(function = "GenerateRequestBuildingStage::execute", error = %e, "Failed to build generate request");
                error::bad_request("build_request_failed", e)
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

        // EPD: inject the prefill->decode KV rendezvous for backends that carry it
        // in the request. No-op unless the selected workers are TokenSpeed EPD.
        if let Some(workers) = ctx.state.workers.as_ref() {
            helpers::maybe_inject_pd_rendezvous(&mut proto_request, workers);
        }

        // Last request reader before dispatch: extract the response-phase view
        // so response processing never needs the (possibly released) payload.
        ctx.state.response.request_view = Some(views::RequestView::Generate(
            views::GenerateRequestView::from(generate_request.as_ref()),
        ));

        ctx.state.execution_plan = Some(ExecutionPlan::generate(self.plan_kind, proto_request));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "GenerateRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        format!(
            "GenerateRequestBuildingStage(inject_pd_metadata={}, {:?})",
            self.inject_pd_metadata, self.plan_kind
        )
    }
}
