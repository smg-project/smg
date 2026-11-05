//! The unified logprob shapes every wire's response converts into: the
//! sampled tokens' logprobs of an output, the top candidates per position
//! and the prompt's input logprobs, with the one conversion that has no
//! proto-side twin. Moved out of `proto_wrapper.rs` as is; the conversion
//! macros the response wrappers expand stay beside them.

use smg_grpc_client::trtllm_proto as trtllm;

/// Unified output logprobs (backend-agnostic)
#[derive(Clone, Debug)]
pub struct ProtoOutputLogProbs {
    pub token_logprobs: Vec<f32>,
    pub token_ids: Vec<u32>,
    pub top_logprobs: Vec<ProtoTopLogProbs>,
}

/// Unified top logprobs per position
#[derive(Clone, Debug)]
pub struct ProtoTopLogProbs {
    pub values: Vec<f32>,
    pub token_ids: Vec<u32>,
}

/// Unified input (prompt) logprobs
#[derive(Clone, Debug)]
pub struct ProtoInputLogProbs {
    pub token_logprobs: Vec<Option<f32>>, // First token is None
    pub token_ids: Vec<u32>,
    pub top_logprobs: Vec<ProtoTopLogProbs>,
}

/// Convert TRT-LLM TokenLogprob slice to unified ProtoOutputLogProbs.
pub(super) fn convert_trtllm_output_logprobs(
    logprobs: &[trtllm::TokenLogprob],
) -> Option<ProtoOutputLogProbs> {
    if logprobs.is_empty() {
        return None;
    }
    Some(ProtoOutputLogProbs {
        token_logprobs: logprobs.iter().map(|lp| lp.logprob).collect(),
        token_ids: logprobs.iter().map(|lp| lp.token_id).collect(),
        top_logprobs: logprobs
            .iter()
            .map(|lp| ProtoTopLogProbs {
                values: lp.top_logprobs.iter().map(|t| t.logprob).collect(),
                token_ids: lp.top_logprobs.iter().map(|t| t.token_id).collect(),
            })
            .collect(),
    })
}
