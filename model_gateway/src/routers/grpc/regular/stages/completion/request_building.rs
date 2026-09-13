//! Completion request building stage: build proto GenerateRequest(s) from CompletionRequest
//!
//! Stage 4 for the `/v1/completions` pipeline, parallel to `MessageRequestBuildingStage`
//! from the Messages rollout. Builds backend-specific proto `GenerateRequest`s from
//! `PreparationOutput` + `CompletionRequest` sampling parameters — one per prompt.
//!
//! Completions has richer sampling knobs than Messages (frequency_penalty, presence_penalty,
//! repetition_penalty, min_p, n, logprobs, structured output constraints) but no tools
//! and no multimodal.

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::completion::CompletionRequest;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        backend_client::BackendClient,
        common::stages::{helpers, BuildStage},
        context::{
            AttemptStamp, BuildOutput, ClientSelection, CompletionItem, ExecutionPlan,
            ExecutionPlanKind, PreparationOutput, RequestContext, WorkerSelection,
        },
        proto_wrapper::ProtoGenerateRequest,
        spec::{CompletionResponseSpec, ResponseSpec},
    },
};

pub(crate) struct CompletionRequestBuildingStage {
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
}

impl CompletionRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            inject_pd_metadata,
            plan_kind,
        }
    }

    /// Build one backend request for one prompt. PD bootstrap rooms are minted
    /// per call, so injection runs per sub-request rather than
    /// build-once-then-clone.
    fn build_proto_request(
        &self,
        builder_client: &BackendClient,
        request_id: String,
        item: &CompletionItem,
        completion_request: &CompletionRequest,
        sampling_mask: Option<helpers::SamplingDefaultsMask>,
        workers: Option<&WorkerSelection>,
    ) -> Result<(ProtoGenerateRequest, Option<helpers::SamplingBaseline>), Response> {
        let mut proto_request = builder_client
            .build_completion_request(
                request_id,
                completion_request,
                item.text.clone(),
                item.token_ids.clone(),
            )
            .map_err(|e| {
                error!(
                    function = "CompletionRequestBuildingStage::build",
                    error = %e,
                    "Failed to build generate request"
                );
                error::bad_request(
                    "invalid_request_parameters",
                    format!("Invalid request parameters: {e}"),
                )
            })?;

        let sampling_baseline =
            helpers::apply_sampling_defaults(&mut proto_request, sampling_mask, workers);

        if self.inject_pd_metadata {
            if let Some(workers) = workers {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        // EPD: inject the prefill->decode KV rendezvous. Completion EPD is
        // text-only (no encode jobs), so this is the only EPD injection here.
        // No-op unless the backend carries it in the request.
        if let Some(workers) = workers {
            helpers::maybe_inject_pd_rendezvous(&mut proto_request, workers);
        }

        Ok((proto_request, sampling_baseline))
    }
}

#[async_trait]
impl BuildStage for CompletionRequestBuildingStage {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "CompletionRequestBuildingStage::build",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let PreparationOutput::Completion { items, .. } = prep else {
            error!(
                function = "CompletionRequestBuildingStage::build",
                "Preparation output is not a completion"
            );
            return Err(error::internal_error(
                "unexpected_preparation_output",
                "Preparation output is not a completion",
            ));
        };

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "CompletionRequestBuildingStage::build",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let completion_request = ctx.completion_request_arc();

        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Disaggregated { prefill, .. } => prefill,
        };

        let disaggregated = matches!(clients, ClientSelection::Disaggregated { .. });
        let request_type = &ctx.input.request_type;
        let workers = ctx.state.workers.as_ref();
        let sampling_mask = helpers::SamplingDefaultsMask::from_request_type(request_type);
        let mut sampling_baseline = None;
        // Each built request is finalized by the client below: it resolves
        // string `stop`s its engine can't match and reports the router's
        // residual trim obligation.
        let tokenizer = ctx.tokenizer_arc();

        let (plan, id_stamp) = match items.as_slice() {
            [] => {
                return Err(error::internal_error(
                    "preparation_not_completed",
                    "No prompts prepared",
                ))
            }
            [item] => {
                let (request_id, id_stamp) = helpers::resolve_request_id_stamp(
                    request_type,
                    ctx.input.tenant_request_meta.as_ref(),
                    "cmpl_",
                    disaggregated,
                );
                let (mut proto_request, baseline) = self.build_proto_request(
                    builder_client,
                    request_id,
                    item,
                    &completion_request,
                    sampling_mask,
                    workers,
                )?;
                sampling_baseline = baseline;
                ctx.state.response.router_stop_obligations = builder_client
                    .finalize_generate_request(&mut proto_request, tokenizer.as_ref());
                (
                    ExecutionPlan::generate(self.plan_kind, proto_request),
                    id_stamp,
                )
            }
            batch_items => {
                let (shared_request_id, batch_stamp) = helpers::resolve_batch_id_stamp(
                    request_type,
                    ctx.input.tenant_request_meta.as_ref(),
                    "cmpl_",
                    disaggregated,
                );
                let mut requests = Vec::with_capacity(batch_items.len());
                for (i, item) in batch_items.iter().enumerate() {
                    let (mut proto_request, baseline) = self.build_proto_request(
                        builder_client,
                        helpers::batch_sub_id(&shared_request_id, i, batch_stamp.unique_subs),
                        item,
                        &completion_request,
                        sampling_mask,
                        workers,
                    )?;
                    // Every sub-request shares the CompletionRequest's
                    // sampling params, so the first baseline stands for all.
                    if sampling_baseline.is_none() {
                        sampling_baseline = baseline;
                    }
                    // Same CompletionRequest per prompt: every iteration
                    // yields the same residual duty, so keep the last.
                    ctx.state.response.router_stop_obligations = builder_client
                        .finalize_generate_request(&mut proto_request, tokenizer.as_ref());
                    requests.push(proto_request);
                }
                (
                    ExecutionPlan::Batch {
                        kind: self.plan_kind,
                        shared_request_id,
                        requests,
                    },
                    helpers::IdStamp::Batch(batch_stamp),
                )
            }
        };

        Ok(BuildOutput {
            plan,
            spec: ResponseSpec::Completion(CompletionResponseSpec::from(
                completion_request.as_ref(),
            )),
            stamp: AttemptStamp {
                id: id_stamp,
                sampling_mask,
                sampling_baseline,
                inject_pd_metadata: self.inject_pd_metadata,
            },
        })
    }

    fn name(&self) -> &'static str {
        "CompletionRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        format!(
            "CompletionRequestBuildingStage(inject_pd_metadata={}, {:?})",
            self.inject_pd_metadata, self.plan_kind
        )
    }
}
