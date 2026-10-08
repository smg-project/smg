//! Proto-to-OpenAI logprob conversion functions.

use std::sync::Arc;

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::common::{ChatLogProbs, ChatLogProbsContent, TopLogProb};

use crate::routers::grpc::proto_wrapper::{ProtoInputLogProbs, ProtoOutputLogProbs};

/// Convert OutputLogProbs to OpenAI ChatLogProbs format
///
/// Generic over the token decoding strategy. The `decode_token` closure maps a
/// single token ID to its text representation.
pub(crate) fn convert_proto_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
    decode_token: impl Fn(u32) -> String,
) -> ChatLogProbs {
    let mut content_items = Vec::with_capacity(proto_logprobs.token_logprobs.len());

    for (i, &logprob) in proto_logprobs.token_logprobs.iter().enumerate() {
        let token_id = proto_logprobs.token_ids.get(i).copied().unwrap_or(0);
        let token_text = decode_token(token_id);
        let bytes = Some(token_text.as_bytes().to_vec());

        // Build top_logprobs for this position
        let top_logprobs = if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i) {
            top_logprobs_entry
                .values
                .iter()
                .enumerate()
                .filter_map(|(j, &top_logprob)| {
                    top_logprobs_entry.token_ids.get(j).map(|&tid| {
                        let text = decode_token(tid);
                        let bytes = Some(text.as_bytes().to_vec());
                        TopLogProb {
                            token: text,
                            logprob: top_logprob,
                            bytes,
                        }
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        content_items.push(ChatLogProbsContent {
            token: token_text,
            logprob,
            bytes,
            top_logprobs,
        });
    }

    ChatLogProbs::Detailed {
        content: (!content_items.is_empty()).then_some(content_items),
    }
}

/// Convert OutputLogProbs to OpenAI ChatLogProbs format using a Tokenizer
pub(crate) fn convert_proto_to_openai_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
    tokenizer: &Arc<dyn Tokenizer>,
) -> ChatLogProbs {
    convert_proto_logprobs(proto_logprobs, |token_id| {
        tokenizer
            .decode(&[token_id], false)
            .unwrap_or_else(|_| format!("<token_{token_id}>"))
    })
}

/// Convert OutputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// Each inner vec contains [logprob (f64), token_id (u32), ...]
pub(crate) fn convert_generate_output_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&logprob, &token_id)| vec![Some(logprob as f64), Some(token_id as f64)])
        .collect()
}

/// Convert InputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// First token has null logprob: [[null, token_id], [logprob, token_id], ...]
pub(crate) fn convert_generate_input_logprobs(
    proto_logprobs: &ProtoInputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&token_logprob, &token_id)| {
            // token_logprob is Option<f32> in unified type
            let logprob_value = token_logprob.map(|v| v as f64);
            vec![logprob_value, Some(token_id as f64)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::routers::grpc::proto_wrapper::ProtoTopLogProbs;

    #[test]
    fn chat_logprob_json_retains_sampled_token_with_no_top_candidates() {
        for top in [
            Vec::new(),
            vec![ProtoTopLogProbs {
                values: Vec::new(),
                token_ids: Vec::new(),
            }],
        ] {
            let proto = ProtoOutputLogProbs {
                token_logprobs: vec![-0.5],
                token_ids: vec![10],
                top_logprobs: top,
            };
            let converted = convert_proto_logprobs(&proto, |_| "Hi".into());
            assert_eq!(
                serde_json::to_value(converted).unwrap(),
                json!({"content": [{
                    "token": "Hi", "logprob": -0.5, "bytes": [72, 105], "top_logprobs": []
                }]})
            );
        }
    }

    #[test]
    fn chat_logprob_json_retains_two_explicit_candidates() {
        let proto = ProtoOutputLogProbs {
            token_logprobs: vec![-0.5],
            token_ids: vec![10],
            top_logprobs: vec![ProtoTopLogProbs {
                values: vec![-0.5, -0.125],
                token_ids: vec![10, 20],
            }],
        };
        let converted =
            convert_proto_logprobs(&proto, |id| if id == 10 { "Hi" } else { "hey" }.into());
        assert_eq!(
            serde_json::to_value(converted).unwrap()["content"][0]["top_logprobs"],
            json!([
                {"token": "Hi", "logprob": -0.5, "bytes": [72, 105]},
                {"token": "hey", "logprob": -0.125, "bytes": [104, 101, 121]},
            ])
        );
    }
}
