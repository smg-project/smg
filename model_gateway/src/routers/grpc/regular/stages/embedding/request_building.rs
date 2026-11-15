//! Request building stage for embedding and classification requests: one
//! backend request per input of the batch, in the request's order.

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::common::StringOrArray;
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
        // single-request endpoints mint; a classify batch whose `rid` lists
        // one id per input runs each input under its own, as SGLang runs the
        // i-th input under the i-th id; any other batch takes the shared id
        // with a per-input sub id, as a batch of completion prompts does.
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
            inputs => match per_input_ids(&ctx.input.request_type, inputs.len()) {
                // The client's ids are stable across attempts, as a single
                // request's `rid` is, and the batch answers under the first.
                Some(ids @ [first, ..]) => {
                    let requests = inputs
                        .iter()
                        .zip(ids)
                        .map(|(input, id)| build_embed_request(client, id.clone(), input))
                        .collect::<Result<Vec<_>, _>>()?;
                    (
                        ExecutionPlan::EmbedBatch {
                            shared_request_id: first.clone(),
                            requests,
                        },
                        helpers::IdStamp::Exact,
                    )
                }
                _ => {
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
                            let request_id = helpers::batch_sub_id(
                                &shared_request_id,
                                i,
                                batch_stamp.unique_subs,
                            );
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
            },
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

/// The ids of a classify batch whose `rid` is a list of one per input, each
/// input's engine id (`ClassifyRequest`'s rule refuses a list of another
/// length, or one naming an id twice); `None` for any other request.
fn per_input_ids(request_type: &RequestType, inputs: usize) -> Option<&[String]> {
    match request_type {
        RequestType::Classify(request) => match &request.rid {
            Some(StringOrArray::Array(ids)) if ids.len() == inputs => Some(ids),
            _ => None,
        },
        _ => None,
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::classify::ClassifyRequest;
    use serde_json::{json, Value};

    use super::*;
    use crate::routers::grpc::{
        context::ClientSelection,
        regular::stages::testing::{components, mock_backend_client},
    };

    const MODEL: &str = "test-model";

    /// An embed batch's shared id and its engine requests' ids, in order.
    fn embed_ids(plan: &ExecutionPlan) -> (String, Vec<String>) {
        let ExecutionPlan::EmbedBatch {
            shared_request_id,
            requests,
        } = plan
        else {
            panic!("an embed batch")
        };
        let ids = requests
            .iter()
            .map(|request| match request {
                ProtoEmbedRequest::Sglang(request) => request.request_id.clone(),
                ProtoEmbedRequest::Vllm(request) => request.request_id.clone(),
            })
            .collect();
        (shared_request_id.clone(), ids)
    }

    /// A classify batch whose `rid` lists one id per input runs each input
    /// under its own id, as SGLang runs the i-th input under the i-th id, and
    /// answers under the first; a retry keeps the ids. A batch under one id
    /// runs each input under that id's per-input sub id, as before.
    #[tokio::test]
    async fn a_classify_batch_runs_each_input_under_its_listed_id() {
        let cases: [(Value, &str, [&str; 2]); 2] = [
            (json!(["r1", "r2"]), "r1", ["r1", "r2"]),
            (json!("r"), "r", ["r-p0", "r-p1"]),
        ];
        for (rid, shared, subs) in cases {
            let (client, engine) = mock_backend_client(MODEL).await;
            let request: Arc<ClassifyRequest> = Arc::new(
                serde_json::from_value(json!({"model": MODEL, "input": ["a", "b"], "rid": rid}))
                    .expect("a classify request"),
            );
            let mut ctx =
                RequestContext::for_classify(request, None, MODEL.to_string(), components());
            ctx.state.clients = Some(ClientSelection::Single { client });
            ctx.state.preparation = Some(PreparationOutput::Embedding {
                inputs: vec![
                    EmbeddingInput {
                        text: Some("a".to_string()),
                        token_ids: vec![1],
                    },
                    EmbeddingInput {
                        text: Some("b".to_string()),
                        token_ids: vec![2],
                    },
                ],
            });
            let BuildOutput {
                mut plan, stamp, ..
            } = EmbeddingRequestBuildingStage::new()
                .build(&mut ctx)
                .await
                .map_err(|refusal| refusal.status())
                .expect("built");
            engine.abort();
            let built = (shared.to_string(), subs.map(str::to_string).to_vec());
            assert_eq!(embed_ids(&plan), built, "{shared}");

            stamp
                .id
                .restamp(&mut plan)
                .map_err(|refusal| refusal.status())
                .expect("restamped");
            assert_eq!(embed_ids(&plan), built, "a retry keeps the ids: {shared}");
        }
    }
}
