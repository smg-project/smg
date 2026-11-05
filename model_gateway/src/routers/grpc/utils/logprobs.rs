//! Proto-to-OpenAI logprob conversion functions.

use std::sync::Arc;

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::common::{ChatLogProbs, ChatLogProbsContent, TopLogProb};

use crate::routers::grpc::proto_wrapper::{ProtoInputLogProbs, ProtoOutputLogProbs};

/// Convert an output's logprobs, read in place on the engine message, to the
/// OpenAI ChatLogProbs format.
///
/// Generic over the token decoding strategy. The `decode_token` closure maps a
/// single token ID to its text representation. `requested` is how many
/// candidates per position the answer carries: the engine may report more
/// (the chosen token, its own list's length), and only the first `requested`
/// of them are decoded; `None` keeps every candidate the engine sent.
pub(crate) fn convert_proto_logprobs(
    proto_logprobs: ProtoOutputLogProbs<'_>,
    requested: Option<usize>,
    decode_token: impl Fn(u32) -> String,
) -> ChatLogProbs {
    let mut content_items = Vec::with_capacity(proto_logprobs.len());

    for position in proto_logprobs.positions() {
        let token_text = decode_token(position.token_id);
        let bytes = Some(token_text.as_bytes().to_vec());

        // The position's candidates, in the engine's order, cut to the requested
        // count before any is decoded: the list is sized from the cut.
        let top_logprobs = position
            .top_logprobs
            .iter()
            .take(requested.unwrap_or(usize::MAX))
            .map(|(token_id, logprob)| {
                let text = decode_token(token_id);
                let bytes = Some(text.as_bytes().to_vec());
                TopLogProb {
                    token: text,
                    logprob,
                    bytes,
                }
            })
            .collect();

        content_items.push(ChatLogProbsContent {
            token: token_text,
            logprob: position.logprob,
            bytes,
            top_logprobs,
        });
    }

    ChatLogProbs::Detailed {
        content: (!content_items.is_empty()).then_some(content_items),
        refusal: None,
    }
}

/// Convert OutputLogProbs to OpenAI ChatLogProbs format using a Tokenizer.
///
/// `top_logprobs` is the number of alternatives the request asked for per
/// token (`None` when it asked for none). The engine may report more, e.g.
/// the chosen token when only its own logprob was requested; the OpenAI API
/// returns exactly the requested alternatives, an empty list when none were,
/// and only those are decoded.
pub(crate) fn convert_proto_to_openai_logprobs(
    proto_logprobs: ProtoOutputLogProbs<'_>,
    tokenizer: &Arc<dyn Tokenizer>,
    top_logprobs: Option<u32>,
) -> ChatLogProbs {
    let requested = Some(top_logprobs.unwrap_or(0) as usize);
    convert_proto_logprobs(proto_logprobs, requested, |token_id| {
        tokenizer
            .decode(&[token_id], false)
            .unwrap_or_else(|_| format!("<token_{token_id}>"))
    })
}

/// Whether `logprobs` carries at least one token's entry.
pub(crate) fn has_entries(logprobs: Option<&ChatLogProbs>) -> bool {
    matches!(
        logprobs,
        Some(ChatLogProbs::Detailed {
            content: Some(items),
            ..
        }) if !items.is_empty()
    )
}

/// Convert OutputLogProbs to Generate format `Vec<Vec<Option<f64>>>`
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// Each inner vec contains [logprob (f64), token_id (u32), ...]
pub(crate) fn convert_generate_output_logprobs(
    proto_logprobs: ProtoOutputLogProbs<'_>,
) -> Vec<Vec<Option<f64>>> {
    // `positions()` filters, so its size hint is not exact: the outer list is
    // sized from the position count, as `convert_proto_logprobs` sizes its own.
    let mut rows = Vec::with_capacity(proto_logprobs.len());
    rows.extend(proto_logprobs.positions().map(|position| {
        vec![
            Some(f64::from(position.logprob)),
            Some(f64::from(position.token_id)),
        ]
    }));
    rows
}

/// Convert InputLogProbs to Generate format `Vec<Vec<Option<f64>>>`
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
    use smg_grpc_client::{trtllm_proto as trtllm, vllm_proto as vllm};

    use super::*;

    /// Two tokens; the engine reports the chosen token and one runner-up as
    /// alternatives for each, whether or not alternatives were requested.
    fn proto() -> vllm::OutputLogProbs {
        vllm::OutputLogProbs {
            token_logprobs: vec![-0.1, -0.2],
            token_ids: vec![1, 2],
            top_logprobs: vec![
                vllm::TopLogProbs {
                    values: vec![-0.1, -1.5],
                    token_ids: vec![1, 3],
                },
                vllm::TopLogProbs {
                    values: vec![-0.2, -1.6],
                    token_ids: vec![2, 4],
                },
            ],
        }
    }

    /// The same two tokens on the wire that keeps one record per position.
    fn records() -> Vec<trtllm::TokenLogprob> {
        let candidate = |token_id, logprob| trtllm::TopLogprob { token_id, logprob };
        vec![
            trtllm::TokenLogprob {
                token_id: 1,
                logprob: -0.1,
                top_logprobs: vec![candidate(1, -0.1), candidate(3, -1.5)],
            },
            trtllm::TokenLogprob {
                token_id: 2,
                logprob: -0.2,
                top_logprobs: vec![candidate(2, -0.2), candidate(4, -1.6)],
            },
        ]
    }

    #[test]
    fn the_record_layout_reads_as_the_columns_do() {
        let proto = proto();
        let records = records();
        let columns = convert_proto_logprobs(ProtoOutputLogProbs::Vllm(&proto), None, |id| {
            format!("t{id}")
        });
        let per_position =
            convert_proto_logprobs(ProtoOutputLogProbs::Trtllm(&records), None, |id| {
                format!("t{id}")
            });
        assert_eq!(
            serde_json::to_value(&columns).unwrap(),
            serde_json::to_value(&per_position).unwrap()
        );
        assert_eq!(
            convert_generate_output_logprobs(ProtoOutputLogProbs::Vllm(&proto)),
            convert_generate_output_logprobs(ProtoOutputLogProbs::Trtllm(&records))
        );
        assert_eq!(
            convert_generate_output_logprobs(ProtoOutputLogProbs::Vllm(&proto)),
            vec![
                vec![Some(f64::from(-0.1_f32)), Some(1.0)],
                vec![Some(f64::from(-0.2_f32)), Some(2.0)]
            ]
        );
    }

    /// Two positions with twenty candidates each on the wire; what the
    /// conversion decodes is the chosen token and the requested candidates,
    /// not the engine's list.
    #[test]
    fn only_the_requested_candidates_are_decoded() {
        use std::cell::Cell;
        let wide = vllm::OutputLogProbs {
            token_logprobs: vec![-0.1, -0.2],
            token_ids: vec![1, 2],
            top_logprobs: (0..2u32)
                .map(|position| vllm::TopLogProbs {
                    values: (0..20).map(|k| -(f32::from(k as u8) + 0.5)).collect(),
                    token_ids: (0..20).map(|k| 100 + position * 20 + k).collect(),
                })
                .collect(),
        };
        let decoded_at = |requested: Option<usize>| {
            let decodes = Cell::new(0usize);
            let logprobs =
                convert_proto_logprobs(ProtoOutputLogProbs::Vllm(&wide), requested, |id| {
                    decodes.set(decodes.get() + 1);
                    format!("t{id}")
                });
            let ChatLogProbs::Detailed {
                content: Some(items),
                ..
            } = logprobs
            else {
                panic!("detailed logprobs expected");
            };
            let kept: Vec<Vec<String>> = items
                .iter()
                .map(|item| item.top_logprobs.iter().map(|t| t.token.clone()).collect())
                .collect();
            (decodes.get(), kept)
        };
        let (none, kept) = decoded_at(Some(0));
        assert_eq!(
            (none, kept),
            (2, vec![vec![], vec![]]),
            "the chosen tokens alone"
        );
        let (one, kept) = decoded_at(Some(1));
        assert_eq!(one, 4, "one candidate per position besides the token");
        assert_eq!(
            kept,
            vec![vec!["t100".to_string()], vec!["t120".to_string()]]
        );
        let (five, kept) = decoded_at(Some(5));
        assert_eq!(five, 12, "five candidates per position besides the token");
        assert_eq!(kept[0], ["t100", "t101", "t102", "t103", "t104"]);
        assert_eq!(kept[1][4], "t124");
        let (all, kept) = decoded_at(None);
        assert_eq!(
            all, 42,
            "every candidate the engine sent, when no count is given"
        );
        assert_eq!(kept[1].len(), 20);
        let (more_than_sent, kept) = decoded_at(Some(50));
        assert_eq!(
            (more_than_sent, kept[0].len()),
            (42, 20),
            "a count above the list is the list"
        );
    }

    fn alternatives(top_logprobs: Option<u32>) -> Vec<Vec<String>> {
        let proto = proto();
        let logprobs = convert_proto_logprobs(
            ProtoOutputLogProbs::Vllm(&proto),
            Some(top_logprobs.unwrap_or(0) as usize),
            |id| format!("t{id}"),
        );
        let ChatLogProbs::Detailed {
            content: Some(items),
            ..
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
    fn the_logprobs_object_carries_content_and_refusal() {
        let proto = proto();
        let value = serde_json::to_value(convert_proto_logprobs(
            ProtoOutputLogProbs::Vllm(&proto),
            None,
            |id| format!("t{id}"),
        ))
        .expect("serialize");
        assert_eq!(value["content"][0]["token"], "t1", "{value}");
        assert_eq!(
            value.get("refusal"),
            Some(&serde_json::Value::Null),
            "{value}"
        );
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
