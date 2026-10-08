use async_trait::async_trait;
use axum::response::Response;
use smg_grpc_client::SglangSchedulerClient;

use crate::routers::{
    error,
    grpc::{
        backend_client::BackendClient,
        client::GrpcClient,
        common::stages::{helpers, BuildStage},
        context::{
            AttemptStamp, BuildOutput, ClientSelection, ExecutionPlan, ExecutionPlanKind,
            PreparationOutput, RequestContext,
        },
        proto_wrapper::ProtoGenerateRequest,
        spec::{DecisionQuestionSpec, DecisionsResponseSpec, ResponseSpec},
    },
};

pub(crate) struct DecisionsRequestBuildingStage;

#[async_trait]
impl BuildStage for DecisionsRequestBuildingStage {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        if !matches!(
            ctx.state.clients.as_ref(),
            Some(ClientSelection::Single {
                client: BackendClient::Grpc(GrpcClient::Sglang(_))
            })
        ) {
            return Err(error::not_implemented(
                "decisions_not_supported",
                "Decisions scoring requires a regular SGLang gRPC worker",
            ));
        }
        let Some(PreparationOutput::Decisions { items, scoring }) = ctx.state.preparation.take()
        else {
            return Err(error::internal_error(
                "preparation_not_completed",
                "No decision prompts prepared",
            ));
        };
        let max_input_tokens = items
            .iter()
            .map(|item| item.token_ids.len())
            .max()
            .unwrap_or(0);
        let (shared_request_id, batch_stamp) = helpers::resolve_batch_id_stamp(
            &ctx.input.request_type,
            ctx.input.tenant_request_meta.as_ref(),
            "decision_",
            false,
        );
        let mut requests = Vec::with_capacity(items.len());
        let mut questions = Vec::with_capacity(items.len());
        for (index, item) in items.into_iter().enumerate() {
            let request_id =
                helpers::batch_sub_id(&shared_request_id, index, batch_stamp.unique_subs);
            let input_tokens = item.token_ids.len();
            let request = SglangSchedulerClient::build_score_request(
                request_id,
                item.token_ids,
                item.label_ids.clone(),
            );
            questions.push(DecisionQuestionSpec {
                label_ids: item.label_ids,
                input_tokens,
            });
            requests.push(ProtoGenerateRequest::Sglang(Box::new(request)));
        }
        Ok(BuildOutput {
            plan: ExecutionPlan::Batch {
                kind: ExecutionPlanKind::Single,
                shared_request_id,
                requests,
            },
            spec: ResponseSpec::Decisions(DecisionsResponseSpec {
                scoring,
                questions,
                max_input_tokens,
            }),
            stamp: AttemptStamp {
                id: helpers::IdStamp::Batch(batch_stamp),
                sampling_mask: None,
                sampling_baseline: None,
                inject_pd_metadata: false,
            },
        })
    }

    fn name(&self) -> &'static str {
        "DecisionsRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        "DecisionsRequestBuildingStage".into()
    }
}
