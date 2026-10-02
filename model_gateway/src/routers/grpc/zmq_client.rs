//! Gateway seam over the ZMQ engine adapter (`engine-zmq-adapter`).
//!
//! The adapter speaks each engine's own proto; the gateway holds requests in
//! its per-engine wrapper ([`ProtoGenerateRequest`]) and reports metadata in
//! its per-runtime enums, so the variant dispatch lives here, next to the
//! other transport glue, and nowhere else.

use std::sync::Arc;

pub use engine_zmq_adapter::{
    connect_for_worker, zmq_handshake_address, EosTokenIds, ZmqDialect, ZmqEngineClient,
    ZmqGenerateStream, ZmqModelInfo, ZmqServerInfo,
};
use llm_tokenizer::traits::Tokenizer;

use crate::routers::grpc::{
    client::{ModelInfo, ServerInfo},
    proto_wrapper::ProtoGenerateRequest,
};

/// Submit the wrapper's request to the engine it was built for.
pub(crate) async fn generate(
    client: &ZmqEngineClient,
    req: ProtoGenerateRequest,
) -> Result<ZmqGenerateStream, tonic::Status> {
    match req {
        ProtoGenerateRequest::Vllm(req) => client.generate_vllm(*req).await,
        ProtoGenerateRequest::TokenSpeed(req) => client.generate_tokenspeed(*req).await,
        ProtoGenerateRequest::Sglang(_)
        | ProtoGenerateRequest::Trtllm(_)
        | ProtoGenerateRequest::Mlx(_) => Err(tonic::Status::internal(
            "ZMQ backends serve vLLM and TokenSpeed requests only",
        )),
    }
}

/// Model metadata as the gateway's per-runtime variant.
pub(crate) fn model_info(client: &ZmqEngineClient) -> ModelInfo {
    match client.model_info() {
        ZmqModelInfo::Vllm(info) => ModelInfo::Vllm(info),
        ZmqModelInfo::TokenSpeed(info) => ModelInfo::TokenSpeed(info),
    }
}

/// Server metadata as the gateway's per-runtime variant.
pub(crate) fn server_info(client: &ZmqEngineClient) -> ServerInfo {
    match client.server_info() {
        ZmqServerInfo::Vllm(info) => ServerInfo::Vllm(info),
        ZmqServerInfo::TokenSpeed(info) => ServerInfo::TokenSpeed(info),
    }
}

/// Request-time EOS backstop for the tokenizer-less EngineCore, dispatched on
/// the wrapper's variant: only a vLLM request takes it (TokenSpeed's scheduler
/// stops at EOS itself). See `engine_zmq_adapter::fold_tokenizer_eos_backstop`.
pub(crate) fn fold_tokenizer_eos_backstop(
    request: &mut ProtoGenerateRequest,
    tokenizer: Option<&Arc<dyn Tokenizer>>,
) {
    if let ProtoGenerateRequest::Vllm(req) = request {
        engine_zmq_adapter::fold_tokenizer_eos_backstop(req, tokenizer);
    }
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::mock::MockTokenizer;
    use smg_grpc_client::{tokenspeed_proto, vllm_proto as vllm};

    use super::*;

    /// MockTokenizer's EOS set is {999}: a vLLM request gains it as a stop id.
    #[test]
    fn eos_backstop_folds_into_vllm_requests() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let mut req = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams::default()),
            ..Default::default()
        }));
        fold_tokenizer_eos_backstop(&mut req, Some(&tokenizer));
        let ProtoGenerateRequest::Vllm(req) = req else {
            panic!("expected vLLM request");
        };
        assert_eq!(req.sampling_params.unwrap().stop_token_ids, vec![999]);
    }

    /// The variant match is the only gate on the fold: a TokenSpeed request
    /// (whose scheduler stops at EOS itself) is left untouched.
    #[test]
    fn eos_backstop_leaves_tokenspeed_requests_untouched() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let mut req =
            ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
                sampling_params: Some(tokenspeed_proto::SamplingParams {
                    stop_token_ids: vec![7],
                    ..Default::default()
                }),
                ..Default::default()
            }));
        fold_tokenizer_eos_backstop(&mut req, Some(&tokenizer));
        let ProtoGenerateRequest::TokenSpeed(req) = req else {
            panic!("expected TokenSpeed request");
        };
        assert_eq!(req.sampling_params.unwrap().stop_token_ids, vec![7]);
    }
}
