//! Request building stage for embedding requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        backend_client::BackendClient,
        client::GrpcClient,
        common::stages::{helpers, BuildStage},
        context::{AttemptStamp, BuildOutput, ExecutionPlan, RequestContext, RequestType},
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
        let prep_output = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "EmbeddingRequestBuildingStage::build",
                "Preparation output missing"
            );
            error::internal_error("preparation_missing", "Preparation output missing")
        })?;

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
        let (request_id, id_stamp) = helpers::resolve_request_id_stamp(
            &ctx.input.request_type,
            ctx.input.tenant_request_meta.as_ref(),
            prefix,
            false,
        );

        // Build backend-specific embed request
        let original_text = prep_output.routing_text().map(String::from);
        let token_ids = prep_output.token_ids().to_vec();

        let proto_req = match client {
            BackendClient::Grpc(GrpcClient::Sglang(c)) => {
                let req = c.build_embed_request(request_id.clone(), original_text, token_ids);
                ProtoEmbedRequest::Sglang(Box::new(req))
            }
            BackendClient::Grpc(GrpcClient::Vllm(c)) => {
                let req = c.build_embed_request(request_id.clone(), original_text, token_ids);
                ProtoEmbedRequest::Vllm(Box::new(req))
            }
            BackendClient::Grpc(GrpcClient::Trtllm(_)) => {
                error!(
                    function = "EmbeddingRequestBuildingStage::build",
                    "TensorRT-LLM embedding not yet supported"
                );
                return Err(error::not_implemented(
                    "unsupported_backend",
                    "TensorRT-LLM embedding is not yet supported via gRPC",
                ));
            }
            BackendClient::Grpc(GrpcClient::Mlx(_)) => {
                error!(
                    function = "EmbeddingRequestBuildingStage::build",
                    "MLX embedding not supported"
                );
                return Err(error::not_implemented(
                    "unsupported_backend",
                    "MLX embedding is not supported via gRPC",
                ));
            }
            BackendClient::Grpc(GrpcClient::TokenSpeed(_)) => {
                error!(
                    function = "EmbeddingRequestBuildingStage::build",
                    "TokenSpeed backend does not support embeddings"
                );
                return Err(error::not_implemented(
                    "unsupported_backend",
                    "TokenSpeed backend does not support embeddings",
                ));
            }
            BackendClient::Zmq(_) => {
                return Err(error::not_implemented(
                    "unsupported_backend",
                    "ZMQ backend does not support embeddings yet",
                ));
            }
        };

        Ok(BuildOutput {
            plan: ExecutionPlan::embed(proto_req),
            spec,
            stamp: AttemptStamp {
                id: id_stamp,
                sampling_mask: None,
                sampling_baseline: None,
                inject_pd_metadata: false,
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
