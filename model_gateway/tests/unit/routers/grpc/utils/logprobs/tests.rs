use llm_tokenizer::mock::MockTokenizer;
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
    let per_position = convert_proto_logprobs(ProtoOutputLogProbs::Trtllm(&records), None, |id| {
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
        let logprobs = convert_proto_logprobs(ProtoOutputLogProbs::Vllm(&wide), requested, |id| {
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

/// The chat converter hands the request's `top_logprobs` to the one rule:
/// the candidates per position are the request's count, none when it
/// named none, and the engine's surplus stays behind.
#[test]
fn the_chat_converter_keeps_the_requested_candidates_per_position() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let four_per_position = vllm::OutputLogProbs {
        token_logprobs: vec![-0.1, -0.2],
        token_ids: vec![1, 2],
        top_logprobs: (0..2u32)
            .map(|position| vllm::TopLogProbs {
                values: (0..4).map(|k| -(f32::from(k as u8) + 0.5)).collect(),
                token_ids: (0..4).map(|k| 10 + position * 4 + k).collect(),
            })
            .collect(),
    };
    for (top_logprobs, expected) in [(None, 0), (Some(0), 0), (Some(1), 1), (Some(3), 3)] {
        let logprobs = convert_proto_to_openai_logprobs(
            ProtoOutputLogProbs::Vllm(&four_per_position),
            &tokenizer,
            top_logprobs,
        );
        let ChatLogProbs::Detailed {
            content: Some(items),
            ..
        } = logprobs
        else {
            panic!("detailed logprobs expected");
        };
        let per_position: Vec<usize> = items.iter().map(|item| item.top_logprobs.len()).collect();
        assert_eq!(
            per_position,
            vec![expected; 2],
            "top_logprobs {top_logprobs:?}"
        );
    }
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
    let mut proto = proto();
    proto.top_logprobs[0].values[0] = -0.5;
    let value = serde_json::to_value(convert_proto_logprobs(
        ProtoOutputLogProbs::Vllm(&proto),
        None,
        |id| format!("é{id}"),
    ))
    .expect("serialize");
    assert_eq!(value["content"][0]["token"], "é1", "{value}");
    assert_eq!(
        value["content"][0]["top_logprobs"][0],
        serde_json::json!({"token": "é1", "logprob": -0.5, "bytes": [195, 169, 49]})
    );
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

fn pairs(entries: &[(&str, f32)]) -> HashMap<String, f32> {
    entries
        .iter()
        .map(|(token, logprob)| (token.to_string(), *logprob))
        .collect()
}

/// The legacy shape of the same two tokens: one entry per position in
/// each of the four lists, the candidates keyed by text, the offsets
/// counting the tokens' characters from where the count stood, and the
/// count moved past them for the next chunk.
#[test]
fn completion_logprobs_list_tokens_candidates_and_offsets() {
    let proto = proto();
    let mut offset = 0;
    let logprobs = completion_logprobs(ProtoOutputLogProbs::Vllm(&proto), 1, &mut offset, |id| {
        format!("t{id}")
    });
    assert_eq!(logprobs.tokens, vec!["t1", "t2"]);
    assert_eq!(logprobs.token_logprobs, vec![Some(-0.1), Some(-0.2)]);
    assert_eq!(
        logprobs.top_logprobs,
        vec![
            Some(pairs(&[("t1", -0.1), ("t3", -1.5)])),
            Some(pairs(&[("t2", -0.2), ("t4", -1.6)]))
        ]
    );
    assert_eq!(logprobs.text_offset, vec![0, 2]);
    assert_eq!(offset, 4);
    let next = completion_logprobs(ProtoOutputLogProbs::Vllm(&proto), 1, &mut offset, |id| {
        format!("t{id}")
    });
    assert_eq!(next.text_offset, vec![4, 6]);
    assert_eq!(
        completion_logprobs(ProtoOutputLogProbs::Trtllm(&records()), 1, &mut 0, |id| {
            format!("t{id}")
        })
        .top_logprobs,
        logprobs.top_logprobs
    );
}

/// `logprobs: k` keeps up to `k + 1` candidates per position: the sampled
/// token's entry beside the `k` the request asked for, never more than
/// the engine ranked.
#[test]
fn completion_candidates_are_capped_at_one_more_than_requested() {
    let candidates = |top_logprobs: u32| -> Vec<Vec<String>> {
        let proto = proto();
        completion_logprobs(
            ProtoOutputLogProbs::Vllm(&proto),
            top_logprobs,
            &mut 0,
            |id| format!("t{id}"),
        )
        .top_logprobs
        .iter()
        .map(|top| {
            let mut texts: Vec<String> = top.iter().flatten().map(|(t, _)| t.clone()).collect();
            texts.sort();
            texts
        })
        .collect()
    };
    assert_eq!(candidates(0), vec![vec!["t1"], vec!["t2"]]);
    assert_eq!(candidates(1), vec![vec!["t1", "t3"], vec!["t2", "t4"]]);
    assert_eq!(candidates(5), vec![vec!["t1", "t3"], vec!["t2", "t4"]]);
}

/// A row that repeats the sampled token's id fills no second slot with
/// it, and two ids that decode to one text share the entry, the later
/// value standing, as the engine's own server keys its map.
#[test]
fn completion_candidates_hold_one_entry_per_id_and_per_text() {
    let proto = vllm::OutputLogProbs {
        token_logprobs: vec![-0.1],
        token_ids: vec![1],
        top_logprobs: vec![vllm::TopLogProbs {
            values: vec![-0.1, -0.1, -1.5, -2.5, -3.5],
            token_ids: vec![1, 1, 3, 4, 5],
        }],
    };
    let logprobs = completion_logprobs(ProtoOutputLogProbs::Vllm(&proto), 2, &mut 0, |id| {
        if id == 4 {
            "t3".to_string()
        } else {
            format!("t{id}")
        }
    });
    assert_eq!(
        logprobs.top_logprobs,
        vec![Some(pairs(&[("t1", -0.1), ("t3", -2.5)]))]
    );
}

/// The sampled token's entry is always there, as the public API returns
/// it: a row that does not rank it, or no row at all (`logprobs: 0` on a
/// wire that sends none), still yields it, and it counts toward the cap.
#[test]
fn completion_candidates_always_hold_the_sampled_token() {
    let proto = vllm::OutputLogProbs {
        token_logprobs: vec![-0.1, -0.2],
        token_ids: vec![1, 2],
        top_logprobs: vec![
            vllm::TopLogProbs {
                values: vec![-1.5, -2.5],
                token_ids: vec![3, 4],
            },
            vllm::TopLogProbs {
                values: vec![],
                token_ids: vec![],
            },
        ],
    };
    let one = completion_logprobs(ProtoOutputLogProbs::Vllm(&proto), 1, &mut 0, |id| {
        format!("t{id}")
    });
    assert_eq!(
        one.top_logprobs,
        vec![
            Some(pairs(&[("t1", -0.1), ("t3", -1.5)])),
            Some(pairs(&[("t2", -0.2)]))
        ]
    );
    let none = completion_logprobs(ProtoOutputLogProbs::Vllm(&proto), 0, &mut 0, |id| {
        format!("t{id}")
    });
    assert_eq!(
        none.top_logprobs,
        vec![Some(pairs(&[("t1", -0.1)])), Some(pairs(&[("t2", -0.2)]))]
    );
}
