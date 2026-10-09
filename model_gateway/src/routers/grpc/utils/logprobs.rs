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

/// Convert OutputLogProbs to OpenAI ChatLogProbs format using a Tokenizer.
///
/// `top_logprobs` is the number of alternatives the request asked for per
/// token (`None` when it asked for none). The engine may report more, e.g.
/// the chosen token when only its own logprob was requested; the OpenAI API
/// returns exactly the requested alternatives, an empty list when none were.
pub(crate) fn convert_proto_to_openai_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
    tokenizer: &Arc<dyn Tokenizer>,
    top_logprobs: Option<u32>,
) -> ChatLogProbs {
    let mut logprobs = convert_proto_logprobs(proto_logprobs, |token_id| {
        tokenizer
            .decode(&[token_id], false)
            .unwrap_or_else(|_| format!("<token_{token_id}>"))
    });
    truncate_top_logprobs(&mut logprobs, top_logprobs.unwrap_or(0) as usize);
    logprobs
}

/// Keep at most `requested` alternatives per token.
fn truncate_top_logprobs(logprobs: &mut ChatLogProbs, requested: usize) {
    if let ChatLogProbs::Detailed {
        content: Some(items),
    } = logprobs
    {
        for item in items {
            item.top_logprobs.truncate(requested);
        }
    }
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
    use super::*;
    use crate::routers::grpc::proto_wrapper::ProtoTopLogProbs;

    /// Two tokens; the engine reports the chosen token and one runner-up as
    /// alternatives for each, whether or not alternatives were requested.
    fn proto() -> ProtoOutputLogProbs {
        ProtoOutputLogProbs {
            token_logprobs: vec![-0.1, -0.2],
            token_ids: vec![1, 2],
            top_logprobs: vec![
                ProtoTopLogProbs {
                    values: vec![-0.1, -1.5],
                    token_ids: vec![1, 3],
                },
                ProtoTopLogProbs {
                    values: vec![-0.2, -1.6],
                    token_ids: vec![2, 4],
                },
            ],
        }
    }

    fn alternatives(top_logprobs: Option<u32>) -> Vec<Vec<String>> {
        let mut logprobs = convert_proto_logprobs(&proto(), |id| format!("t{id}"));
        truncate_top_logprobs(&mut logprobs, top_logprobs.unwrap_or(0) as usize);
        let ChatLogProbs::Detailed {
            content: Some(items),
        } = logprobs
        else {
            panic!("detailed logprobs expected");
        };
        assert_eq!(items[0].token, "t1");
        assert_eq!(items[1].logprob, -0.2);
        items
            .iter()
            .map(|item| item.top_logprobs.iter().map(|t| t.token.clone()).collect())
            .collect()
    }

    #[test]
    fn top_logprobs_are_empty_unless_requested() {
        assert_eq!(alternatives(None), vec![Vec::<String>::new(), Vec::new()]);
        assert_eq!(
            alternatives(Some(0)),
            vec![Vec::<String>::new(), Vec::new()]
        );
    }

    #[test]
    fn top_logprobs_are_capped_at_the_requested_count() {
        assert_eq!(alternatives(Some(1)), vec![vec!["t1"], vec!["t2"]]);
        assert_eq!(
            alternatives(Some(2)),
            vec![vec!["t1", "t3"], vec!["t2", "t4"]]
        );
        assert_eq!(
            alternatives(Some(5)),
            vec![vec!["t1", "t3"], vec!["t2", "t4"]]
        );
    }
}
