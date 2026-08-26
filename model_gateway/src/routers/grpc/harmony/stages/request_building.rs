//! Harmony Request Building Stage: Build gRPC request from Harmony-encoded tokens

use async_trait::async_trait;
use axum::response::Response;
use smg_grpc_client::{SglangGenerateRequestOptions, TokenSpeedSchedulerClient, VllmEngineClient};
use tracing::{debug, error};

use crate::routers::{
    error,
    grpc::{
        backend_client::BackendClient,
        client::GrpcClient,
        common::stages::{helpers, BuildStage},
        context::{
            AttemptStamp, BuildOutput, ClientSelection, ExecutionPlan, ExecutionPlanKind,
            PreparationOutput, RequestContext, RequestType,
        },
        proto_wrapper::ProtoGenerateRequest,
        spec::{HarmonyResponseSpec, ResponseSpec},
        zmq_client::ZmqDialect,
    },
};

/// Harmony Request Building stage: Convert Harmony tokens to gRPC request
///
/// Takes the Harmony-encoded input_ids from preparation and builds a proto::GenerateRequest.
/// Unlike regular request building, this uses token_ids directly (Harmony encoding handles messages).
pub(crate) struct HarmonyRequestBuildingStage {
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
}

impl HarmonyRequestBuildingStage {
    /// Create a new Harmony request building stage
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            inject_pd_metadata,
            plan_kind,
        }
    }
}

#[async_trait]
impl BuildStage for HarmonyRequestBuildingStage {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        // Take preparation output (last consumer — worker_selection already ran)
        let prep = ctx.state.preparation.take().ok_or_else(|| {
            error!(
                function = "HarmonyRequestBuildingStage::build",
                "Preparation stage not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;
        let PreparationOutput::Harmony {
            token_ids,
            tool_constraints,
            modified_request,
            harmony_stop_ids,
            ..
        } = prep
        else {
            debug_assert!(false, "pipeline guarantees Harmony variant");
            return Err(error::internal_error(
                "wrong_preparation_type",
                "Expected Harmony preparation output",
            ));
        };

        // Get clients
        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "HarmonyRequestBuildingStage::build",
                "Client acquisition stage not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Disaggregated { prefill, .. } => prefill,
        };

        // Generate request_id based on request type. The Harmony spec owns a
        // handle to the request: the tool loop legitimately re-reads it
        // across iterations (the one deliberate post-build request holder).
        let disaggregated = matches!(clients, ClientSelection::Disaggregated { .. });
        let (request_id, id_stamp, harmony_spec) = match &ctx.input.request_type {
            RequestType::Chat(request) => {
                let (id, stamp) = helpers::resolve_request_id_stamp(
                    &ctx.input.request_type,
                    ctx.input.tenant_request_meta.as_ref(),
                    "chatcmpl-",
                    disaggregated,
                );
                (id, stamp, HarmonyResponseSpec::Chat(request.clone()))
            }
            RequestType::Responses(request) => {
                let (id, stamp) = helpers::resolve_request_id_stamp(
                    &ctx.input.request_type,
                    ctx.input.tenant_request_meta.as_ref(),
                    "responses-",
                    disaggregated,
                );
                (id, stamp, HarmonyResponseSpec::Responses(request.clone()))
            }
            request_type @ (RequestType::Generate(_)
            | RequestType::Completion(_)
            | RequestType::Embedding(_)
            | RequestType::Classify(_)
            | RequestType::Messages(_)) => {
                error!(
                    function = "HarmonyRequestBuildingStage::build",
                    request_type = %request_type,
                    "{request_type} request type not supported for Harmony models"
                );
                return Err(error::bad_request(
                    "not_supported_in_harmony",
                    format!("{request_type} requests are not supported with Harmony models"),
                ));
            }
        };

        // Build gRPC request using token_ids directly (Harmony encoding already handled message rendering)
        let placeholder_processed_text = "[harmony]".to_string();

        // Resolve the request kind once; the backend dispatch below is a single
        // compact match with one shared error path. (Non-Chat/Responses kinds
        // were already rejected by the request-id match above; the reject arms
        // here keep this exhaustive without a panic.)
        let body = match &ctx.input.request_type {
            RequestType::Chat(request) => HarmonyBody::Chat(
                modified_request
                    .as_deref()
                    .unwrap_or_else(|| request.as_ref()),
            ),
            RequestType::Responses(request) => HarmonyBody::Responses(request.as_ref()),
            RequestType::Embedding(_) => {
                return Err(error::bad_request(
                    "harmony_embedding_not_supported",
                    "Embedding requests are not supported with Harmony models".to_string(),
                ));
            }
            _ => {
                return Err(error::bad_request(
                    "unsupported_request_type",
                    "Unsupported request type for Harmony models".to_string(),
                ));
            }
        };

        let mut proto_request = build_harmony_proto(
            builder_client,
            body,
            request_id,
            placeholder_processed_text,
            token_ids,
            tool_constraints,
        )
        .map_err(|e| {
            error!(function = "HarmonyRequestBuildingStage::build", error = %e, "Failed to build Harmony generate request");
            error::bad_request(
                "invalid_request_parameters",
                format!("Invalid request parameters: {e}"),
            )
        })?;

        // Inject the Harmony stop ids (<|return|> and <|call|>) so the model
        // cannot generate past a channel boundary.
        if !harmony_stop_ids.is_empty() {
            proto_request.extend_stop_token_ids(&harmony_stop_ids);
            if let ProtoGenerateRequest::Trtllm(req) = &mut proto_request {
                // TRT-LLM strips stop tokens from output by default, but the
                // Harmony parser needs them to detect channel boundaries
                // (e.g. <|call|> marks the tool-call channel transition).
                req.include_stop_token_in_output = true;
            }
            debug!(
                stop_token_count = harmony_stop_ids.len(),
                "Injected Harmony stop tokens"
            );
        }

        // The client resolves string `stop`s its engine can't match and
        // reports the router's residual trim obligation; harmony response
        // processing scans channel text for exactly these strings.
        ctx.state.response.router_stop_obligations = builder_client
            .finalize_generate_request(&mut proto_request, ctx.tokenizer_arc().as_ref());

        if self.inject_pd_metadata {
            if let Some(workers) = ctx.state.workers.as_ref() {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        Ok(BuildOutput {
            plan: ExecutionPlan::generate(self.plan_kind, proto_request),
            spec: ResponseSpec::Harmony(harmony_spec),
            stamp: AttemptStamp {
                id: id_stamp,
                // Harmony builds never applied worker sampling defaults;
                // retries must not start applying them.
                sampling_mask: None,
                sampling_baseline: None,
                inject_pd_metadata: self.inject_pd_metadata,
            },
        })
    }

    fn name(&self) -> &'static str {
        "HarmonyRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        format!(
            "HarmonyRequestBuildingStage(inject_pd_metadata={}, {:?})",
            self.inject_pd_metadata, self.plan_kind
        )
    }
}

/// The two request kinds Harmony serves, with the Chat body already resolved
/// to the (possibly response_format-modified) request.
enum HarmonyBody<'a> {
    Chat(&'a openai_protocol::chat::ChatCompletionRequest),
    Responses(&'a openai_protocol::responses::ResponsesRequest),
}

/// One (backend x request-kind) dispatch for Harmony request building: every
/// arm is just the engine's builder call. vLLM and TokenSpeed build through
/// static translators, so a direct-ZMQ backend routes into the same builders
/// as its gRPC counterpart via its [`ZmqDialect`] — leaving the match
/// exhaustive over transports and engines, with no unrepresentable pairing to
/// error on.
fn build_harmony_proto(
    client: &BackendClient,
    body: HarmonyBody<'_>,
    request_id: String,
    text: String,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
) -> Result<ProtoGenerateRequest, String> {
    use HarmonyBody::{Chat, Responses};
    Ok(match (client, body) {
        (BackendClient::Grpc(GrpcClient::Sglang(c)), Chat(b)) => {
            ProtoGenerateRequest::Sglang(Box::new(c.build_generate_request_from_chat(
                request_id,
                b,
                text,
                token_ids,
                SglangGenerateRequestOptions {
                    multimodal_inputs: None,
                    tool_call_constraint: tool_constraints,
                    require_reasoning: false,
                },
            )?))
        }
        (BackendClient::Grpc(GrpcClient::Sglang(c)), Responses(b)) => {
            ProtoGenerateRequest::Sglang(Box::new(c.build_generate_request_from_responses(
                request_id,
                b,
                text,
                token_ids,
                tool_constraints,
            )?))
        }
        (BackendClient::Grpc(GrpcClient::Trtllm(c)), Chat(b)) => {
            ProtoGenerateRequest::Trtllm(Box::new(c.build_generate_request_from_chat(
                request_id,
                b,
                text,
                token_ids,
                None, // No multimodal in the Harmony pipeline
                tool_constraints,
            )?))
        }
        (BackendClient::Grpc(GrpcClient::Trtllm(c)), Responses(b)) => {
            ProtoGenerateRequest::Trtllm(Box::new(c.build_generate_request_from_responses(
                request_id,
                b,
                text,
                token_ids,
                tool_constraints,
            )?))
        }
        (BackendClient::Grpc(GrpcClient::Mlx(c)), Chat(b)) => ProtoGenerateRequest::Mlx(Box::new(
            c.build_generate_request_from_chat(request_id, b, text, token_ids, tool_constraints)?,
        )),
        (BackendClient::Grpc(GrpcClient::Mlx(c)), Responses(b)) => {
            ProtoGenerateRequest::Mlx(Box::new(c.build_generate_request_from_responses(
                request_id,
                b,
                text,
                token_ids,
                tool_constraints,
            )?))
        }
        (BackendClient::Grpc(GrpcClient::Vllm(_)), body) => {
            build_vllm(body, request_id, text, token_ids, tool_constraints)?
        }
        (BackendClient::Grpc(GrpcClient::TokenSpeed(_)), body) => {
            build_tokenspeed(body, request_id, text, token_ids, tool_constraints)?
        }
        (BackendClient::Zmq(zmq), body) => match zmq.dialect() {
            ZmqDialect::Vllm => build_vllm(body, request_id, text, token_ids, tool_constraints)?,
            ZmqDialect::TokenSpeed => {
                build_tokenspeed(body, request_id, text, token_ids, tool_constraints)?
            }
        },
    })
}

/// vLLM Harmony builders, shared by the gRPC and direct-ZMQ vLLM backends.
fn build_vllm(
    body: HarmonyBody<'_>,
    request_id: String,
    text: String,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
) -> Result<ProtoGenerateRequest, String> {
    let request = match body {
        HarmonyBody::Chat(b) => VllmEngineClient::build_generate_request_from_chat(
            request_id,
            b,
            text,
            token_ids,
            None, // No multimodal in the Harmony pipeline
            tool_constraints,
        )?,
        HarmonyBody::Responses(b) => VllmEngineClient::build_generate_request_from_responses(
            request_id,
            b,
            text,
            token_ids,
            tool_constraints,
        )?,
    };
    Ok(ProtoGenerateRequest::Vllm(Box::new(request)))
}

/// TokenSpeed Harmony builders, shared by the gRPC and direct-ZMQ TokenSpeed
/// backends.
fn build_tokenspeed(
    body: HarmonyBody<'_>,
    request_id: String,
    text: String,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
) -> Result<ProtoGenerateRequest, String> {
    let request = match body {
        HarmonyBody::Chat(b) => TokenSpeedSchedulerClient::build_generate_request_from_chat(
            request_id,
            b,
            text,
            token_ids,
            None, // Harmony path: multimodal not yet wired
            tool_constraints,
        )?,
        HarmonyBody::Responses(b) => {
            TokenSpeedSchedulerClient::build_generate_request_from_responses(
                request_id,
                b,
                text,
                token_ids,
                tool_constraints,
            )?
        }
    };
    Ok(ProtoGenerateRequest::TokenSpeed(Box::new(request)))
}
