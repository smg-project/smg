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
    let mut req = ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
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
