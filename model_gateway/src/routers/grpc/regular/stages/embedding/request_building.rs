//! Request building stage for embedding and classification requests: one
//! backend request per input of the batch, in the request's order.

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        backend_client::BackendClient,
        client::GrpcClient,
        common::stages::{helpers, BuildStage},
        context::{
            stage_output, AttemptStamp, BuildOutput, EmbeddingInput, ExecutionPlan,
            PreparationOutput, RequestContext, RequestType,
        },
        proto_wrapper::ProtoEmbedRequest,
        spec::ResponseSpec,
    },
};

/// Request building stage for embedding requests
pub(crate) struct EmbeddingRequestBuildingStage;

impl EmbeddingRequestBuildingStage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EmbeddingRequestBuildingStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BuildStage for EmbeddingRequestBuildingStage {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        // Preparation output should have tokenized input
        let prep_output = stage_output(
            ctx.state.preparation.as_ref(),
            "EmbeddingRequestBuildingStage::build",
        )?;
        let PreparationOutput::Embedding { inputs } = prep_output else {
            error!(
                function = "EmbeddingRequestBuildingStage::build",
                "Preparation output is not an embedding's"
            );
            return Err(error::internal_error(
                "preparation_mismatch",
                "Preparation output does not match the request",
            ));
        };

        // Extract client
        let client = ctx
            .state
            .clients
            .as_ref()
            .and_then(|c| c.single())
            .ok_or_else(|| {
                error!(
                    function = "EmbeddingRequestBuildingStage::build",
                    "Client not selected"
                );
                error::internal_error("client_missing", "Client not selected")
            })?;

        // Embeddings/classify are single-worker only (never disaggregated).
        let (prefix, spec) = match &ctx.input.request_type {
            RequestType::Classify(_) => ("classify-", ResponseSpec::Classify),
            RequestType::Embedding(_) => ("embed-", ResponseSpec::Embedding),
            request_type => {
                error!(
                    function = "EmbeddingRequestBuildingStage::build",
                    request_type = %request_type,
                    "{request_type} should not reach this stage"
                );
                return Err(error::internal_error(
                    "wrong_pipeline",
                    format!("{request_type} should use its dedicated pipeline"),
                ));
            }
        };

        // One backend request per input: a single input takes the one id the
        // single-request endpoints mint, a batch the shared id with a
        // per-input sub id, as a batch of completion prompts does.
        let (plan, id_stamp) = match inputs.as_slice() {
            [] => {
                return Err(error::internal_error(
                    "preparation_not_completed",
                    "No inputs prepared",
                ))
            }
            [input] => {
                let (request_id, id_stamp) = helpers::resolve_request_id_stamp(
                    &ctx.input.request_type,
                    ctx.input.tenant_request_meta.as_ref(),
                    prefix,
                    false,
                );
                (
                    ExecutionPlan::embed(build_embed_request(client, request_id, input)?),
                    id_stamp,
                )
            }
            inputs => {
                let (shared_request_id, batch_stamp) = helpers::resolve_batch_id_stamp(
                    &ctx.input.request_type,
                    ctx.input.tenant_request_meta.as_ref(),
                    prefix,
                    false,
                );
                let requests = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, input)| {
                        let request_id =
                            helpers::batch_sub_id(&shared_request_id, i, batch_stamp.unique_subs);
                        build_embed_request(client, request_id, input)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    ExecutionPlan::EmbedBatch {
                        shared_request_id,
                        requests,
                    },
                    helpers::IdStamp::Batch(batch_stamp),
                )
            }
        };

        Ok(BuildOutput {
            plan,
            spec,
            stamp: AttemptStamp {
                id: id_stamp,
                sampling_mask: None,
                sampling_baseline: None,
                inject_pd_metadata: false,
                tool_constraint: None,
            },
        })
    }

    fn name(&self) -> &'static str {
        "EmbeddingRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        "EmbeddingRequestBuildingStage".to_string()
    }
}

/// The backend's embed request for one input under `request_id`: its text
/// beside its ids (the ids alone for an input the client sent as ids), on the
/// wires that serve embeddings.
fn build_embed_request(
    client: &BackendClient,
    request_id: String,
    input: &EmbeddingInput,
) -> Result<ProtoEmbedRequest, Response> {
    let original_text = input.text.clone();
    let token_ids = input.token_ids.clone();
    match client {
        BackendClient::Grpc(GrpcClient::Sglang(c)) => {
            let req = c.build_embed_request(request_id, original_text, token_ids);
            Ok(ProtoEmbedRequest::Sglang(Box::new(req)))
        }
        BackendClient::Grpc(GrpcClient::Vllm(c)) => {
            let req = c.build_embed_request(request_id, original_text, token_ids);
            Ok(ProtoEmbedRequest::Vllm(Box::new(req)))
        }
        BackendClient::Grpc(GrpcClient::Trtllm(_)) => {
            error!(
                function = "EmbeddingRequestBuildingStage::build",
                "TensorRT-LLM embedding not yet supported"
            );
            Err(error::not_implemented(
                "unsupported_backend",
                "TensorRT-LLM embedding is not yet supported via gRPC",
            ))
        }
        BackendClient::Grpc(GrpcClient::Mlx(_)) => {
            error!(
                function = "EmbeddingRequestBuildingStage::build",
                "MLX embedding not supported"
            );
            Err(error::not_implemented(
                "unsupported_backend",
                "MLX embedding is not supported via gRPC",
            ))
        }
        BackendClient::Grpc(GrpcClient::TokenSpeed(_)) => {
            error!(
                function = "EmbeddingRequestBuildingStage::build",
                "TokenSpeed backend does not support embeddings"
            );
            Err(error::not_implemented(
                "unsupported_backend",
                "TokenSpeed backend does not support embeddings",
            ))
        }
        BackendClient::Zmq(_) => Err(error::not_implemented(
            "unsupported_backend",
            "ZMQ backend does not support embeddings yet",
        )),
    }
}
