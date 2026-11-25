//! Gateway seam over the ZMQ engine adapter (`engine-zmq-adapter`).
//!
//! The adapter speaks each engine's own proto; the gateway holds requests in
//! its per-engine wrapper (`ProtoGenerateRequest`) and reports metadata in
//! its per-runtime enums, so the variant dispatch lives here, next to the
//! other transport glue, and nowhere else.

use std::{collections::HashMap, sync::Arc};

pub use engine_zmq_adapter::{
    connect_for_worker, zmq_handshake_address, EncodeRendezvous, EngineCount, EosTokenIds,
    SglangProfileStart, ZmqDialect, ZmqEngineClient, ZmqGenerateStream, ZmqModelInfo,
    ZmqServerInfo,
};
use llm_tokenizer::traits::Tokenizer;
use smg_grpc_client::engine_labels::{ModelInfo, ServerInfo};
use smg_worker::identity::basic::CONTEXT_WINDOW_LABELS;

use crate::routers::grpc::proto_wrapper::ProtoGenerateRequest;

/// Submit the wrapper's request to the engine it was built for.
pub(crate) async fn generate(
    client: &ZmqEngineClient,
    req: ProtoGenerateRequest,
) -> Result<ZmqGenerateStream, tonic::Status> {
    match req {
        ProtoGenerateRequest::Vllm(req) => client.generate_vllm(*req).await,
        ProtoGenerateRequest::TokenSpeed(req) => client.generate_tokenspeed(*req).await,
        ProtoGenerateRequest::Sglang(req) => client.generate_sglang(*req).await,
        // A TensorRT-LLM engine is reached through its gRPC servicer (the
        // Rust servicer speaks this wire on the engine's node); the direct
        // lane serves no request of its proto.
        ProtoGenerateRequest::Trtllm(_) | ProtoGenerateRequest::Mlx(_) => Err(
            tonic::Status::internal("ZMQ backends serve vLLM, TokenSpeed and SGLang requests only"),
        ),
    }
}

/// Model metadata as the gateway's per-runtime variant.
pub(crate) fn model_info(client: &ZmqEngineClient) -> ModelInfo {
    match client.model_info() {
        ZmqModelInfo::Vllm(info) => ModelInfo::Vllm(info),
        ZmqModelInfo::TokenSpeed(info) => ModelInfo::TokenSpeed(info),
        ZmqModelInfo::Sglang(info) => ModelInfo::Sglang(info),
        ZmqModelInfo::Trtllm(info) => ModelInfo::Trtllm(*info),
    }
}

/// The labels a ZMQ worker's own link learned at the handshake: the engine's
/// context window, under the key its dialect's model info carries it as
/// (one of the card's window labels), so a refresh after the handshake gives
/// the card the window the gateway's admission reads; and a TokenSpeed
/// group's running window ([`tokenspeed_running_window`]). The rest of the
/// handshake's report stays the link's own.
pub(crate) fn link_labels(client: &ZmqEngineClient) -> HashMap<String, String> {
    let info = model_info(client);
    let tokenspeed = matches!(info, ModelInfo::TokenSpeed(_));
    let mut labels = info.to_labels();
    labels.retain(|key, _| CONTEXT_WINDOW_LABELS.contains(&key.as_str()));
    if tokenspeed {
        let engines = client.engines();
        let max_num_seqs = engines
            .first()
            .map_or(0, |engine| engine.ready_response.max_num_seqs);
        labels.extend(tokenspeed_running_window(max_num_seqs, engines.len()));
    }
    labels
}

/// The labels of [`tokenspeed_running_window`].
const RUNNING_WINDOW_LABELS: [&str; 2] = ["max_num_seqs", "attn_dp_size"];

/// A ZMQ worker's `record` of its engine's labels with what its link reports
/// now. The record holds a rank worker's running window as that rank's
/// share, and a refresh reads the report as the whole group's, so the
/// record's share is dropped: left in place under a link that reports no
/// window, every refresh would divide it by the ranks again.
pub(crate) fn refreshed_labels(
    record: &HashMap<String, String>,
    link: HashMap<String, String>,
) -> HashMap<String, String> {
    let mut fresh = record.clone();
    fresh.retain(|key, _| !RUNNING_WINDOW_LABELS.contains(&key.as_str()));
    fresh.extend(link);
    fresh
}

/// A TokenSpeed link's running window as the engine's report carries it:
/// `max_num_seqs`, which every engine of a group declares as the whole
/// group's, and the number of attention-DP ranks that share it, so a rank
/// worker takes its share (`labels_for_rank`) and a grouped worker the whole.
/// An engine that declares no window gives none.
fn tokenspeed_running_window(max_num_seqs: u64, engines: usize) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    if max_num_seqs == 0 {
        return labels;
    }
    labels.insert("max_num_seqs".to_string(), max_num_seqs.to_string());
    if engines > 1 {
        labels.insert("attn_dp_size".to_string(), engines.to_string());
    }
    labels
}

/// Server metadata as the gateway's per-runtime variant.
pub(crate) fn server_info(client: &ZmqEngineClient) -> ServerInfo {
    match client.server_info() {
        ZmqServerInfo::Vllm(info) => ServerInfo::Vllm(info),
        ZmqServerInfo::TokenSpeed(info) => ServerInfo::TokenSpeed(info),
        ZmqServerInfo::Sglang(info) => ServerInfo::Sglang(info),
        ZmqServerInfo::Trtllm(info) => ServerInfo::Trtllm(*info),
    }
}

/// Request-time EOS backstop for the tokenizer-less EngineCore, dispatched on
/// the wrapper's variant: only a vLLM request takes it (TokenSpeed's and
/// SGLang's schedulers stop at EOS themselves). See `engine_zmq_adapter::fold_tokenizer_eos_backstop`.
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

    /// A group's window comes with the width its ranks share it by; a lone
    /// engine's window is its own, and an engine that declares none gives no
    /// window label.
    #[test]
    fn a_tokenspeed_link_reports_its_running_window_and_the_ranks_that_share_it() {
        assert_eq!(
            tokenspeed_running_window(256, 16),
            HashMap::from([
                ("max_num_seqs".to_string(), "256".to_string()),
                ("attn_dp_size".to_string(), "16".to_string()),
            ])
        );
        assert_eq!(
            tokenspeed_running_window(80, 1),
            HashMap::from([("max_num_seqs".to_string(), "80".to_string())])
        );
        assert!(tokenspeed_running_window(0, 16).is_empty());
    }

    /// A rank worker's record holds its share of the running window. A link
    /// that reports the window replaces the share with the group's; one that
    /// reports none leaves no share behind to be divided again.
    #[test]
    fn a_refresh_drops_the_running_window_its_link_no_longer_reports() {
        let record = HashMap::from([
            ("max_num_seqs".to_string(), "16".to_string()),
            ("attn_dp_size".to_string(), "16".to_string()),
            ("default_temperature".to_string(), "0.6".to_string()),
        ]);
        assert_eq!(
            refreshed_labels(&record, tokenspeed_running_window(256, 16)),
            HashMap::from([
                ("max_num_seqs".to_string(), "256".to_string()),
                ("attn_dp_size".to_string(), "16".to_string()),
                ("default_temperature".to_string(), "0.6".to_string()),
            ])
        );
        assert_eq!(
            refreshed_labels(&record, tokenspeed_running_window(0, 16)),
            HashMap::from([("default_temperature".to_string(), "0.6".to_string())])
        );
    }
}
