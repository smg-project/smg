use smg_grpc_client::{tokenspeed_proto as tokenspeed, vllm_proto as vllm};

use super::*;

/// The usage the counters of `responses` make.
fn build_usage(responses: &[ProtoGenerateComplete]) -> Usage {
    TokenCounts::of(responses).usage()
}

fn complete(
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
) -> ProtoGenerateComplete {
    ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete {
        prompt_tokens,
        cached_tokens,
        completion_tokens,
        ..Default::default()
    })
}

#[test]
fn build_usage_charges_the_shared_prompt_once_for_n_greater_than_1() {
    let responses = vec![complete(10, 8, 4), complete(10, 8, 6)];
    let usage = build_usage(&responses);

    assert_eq!(
        usage.prompt_tokens, 10,
        "n>1 choices share one prompt -- must not be multiplied per choice"
    );
    assert_eq!(
        usage
            .prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens),
        Some(8),
        "cached_tokens is a property of the shared prompt too -- must not be multiplied per choice"
    );
    assert_eq!(
        usage.completion_tokens, 10,
        "each choice generates its own completion -- must be summed"
    );
}

#[test]
fn parsed_reasoning_tokens_are_reported_once_the_parser_ran() {
    let reasoning_tokens = |usage: Usage| {
        usage
            .completion_tokens_details
            .and_then(|d| d.reasoning_tokens)
    };
    let usage = build_usage(&[complete(10, 8, 4)]);
    assert_eq!(reasoning_tokens(usage.clone()), None);
    assert_eq!(
        reasoning_tokens(with_parsed_reasoning_tokens(usage.clone(), None)),
        None,
        "no reasoning parser: nothing to report"
    );
    assert_eq!(
        reasoning_tokens(with_parsed_reasoning_tokens(usage.clone(), Some(0))),
        Some(0),
        "the parser ran and found no reasoning: zero, not absent"
    );
    assert_eq!(
        reasoning_tokens(with_parsed_reasoning_tokens(usage.clone(), Some(3))),
        Some(3)
    );
    assert_eq!(
        reasoning_tokens(with_parsed_reasoning_tokens(usage, Some(7))),
        Some(4),
        "the part is not above the whole: four tokens counted, seven read as reasoning"
    );
    let engine_counted = Usage::from_counts(10, 4).with_reasoning_tokens(4);
    assert_eq!(
        reasoning_tokens(with_parsed_reasoning_tokens(engine_counted, Some(3))),
        Some(4),
        "an engine that counts reasoning itself is not overruled"
    );
    let engine_counted_fewer = Usage::from_counts(10, 4).with_reasoning_tokens(2);
    assert_eq!(
        reasoning_tokens(with_parsed_reasoning_tokens(engine_counted_fewer, Some(3))),
        Some(2),
        "nor when its count is lower than the parser's"
    );
}

#[test]
fn an_engines_reasoning_count_above_its_completion_count_is_reported_and_settled_within_it() {
    let counts = TokenCounts {
        completion: 4,
        reasoning: 7,
        ..TokenCounts::default()
    };
    assert_eq!(
        counts
            .usage()
            .completion_tokens_details
            .and_then(|details| details.reasoning_tokens),
        Some(4),
        "four tokens generated, seven counted as reasoning"
    );
    assert_eq!(
        counts.settlement().reasoning_tokens,
        4,
        "the settlement takes the count the usage reports"
    );
}

#[test]
fn token_counts_usage_carries_the_cached_reasoning_and_speculative_details() {
    let usage = TokenCounts {
        prompt: 10,
        completion: 5,
        cached: 4,
        reasoning: 3,
        spec_accepted: 2,
        spec_drafted: Some(3),
    }
    .usage();

    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
    assert_eq!(
        usage
            .prompt_tokens_details
            .as_ref()
            .map(|details| details.cached_tokens),
        Some(4)
    );
    let details = usage
        .completion_tokens_details
        .expect("the reasoning and speculative details");
    assert_eq!(details.reasoning_tokens, Some(3));
    assert_eq!(details.accepted_prediction_tokens, Some(2));
    assert_eq!(details.rejected_prediction_tokens, Some(1));
}

#[test]
fn chat_usage_has_every_detail_counter_and_no_unbilled_prompt_tokens() {
    let usage = serde_json::to_value(chat_usage(
        build_usage(&[complete(10, 8, 4)]),
        None,
        3,
        None,
    ))
    .expect("a JSON usage");
    assert_eq!(
        usage,
        serde_json::json!({
            "prompt_tokens": 7,
            "completion_tokens": 4,
            "total_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": 7, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        })
    );
    let parsed = chat_usage(build_usage(&[complete(10, 8, 4)]), Some(2), 0, None);
    assert_eq!(
        parsed
            .completion_tokens_details
            .and_then(|details| details.reasoning_tokens),
        Some(2)
    );
}

#[test]
fn chat_usage_reports_the_image_tokens_the_gateway_expanded() {
    let with_images = serde_json::to_value(chat_usage(
        build_usage(&[complete(535, 0, 8)]),
        None,
        0,
        Some(324),
    ))
    .expect("a JSON usage");
    assert_eq!(with_images["prompt_tokens"], 535);
    assert_eq!(
        with_images["prompt_tokens_details"],
        serde_json::json!({"cached_tokens": 0, "audio_tokens": 0, "image_tokens": 324})
    );
    let text_only = serde_json::to_value(chat_usage(
        build_usage(&[complete(207, 0, 8)]),
        None,
        0,
        None,
    ))
    .expect("a JSON usage");
    assert_eq!(
        text_only["prompt_tokens_details"],
        serde_json::json!({"cached_tokens": 0, "audio_tokens": 0}),
        "a request without an image the gateway expanded reports no image count"
    );
}

#[test]
fn build_usage_then_unbilled_charges_the_stub_once_for_n_greater_than_1() {
    let usage =
        build_usage(&[complete(10, 10, 4), complete(10, 10, 6)]).with_unbilled_prompt_tokens(3);
    assert_eq!(usage.prompt_tokens, 7);
    assert_eq!(
        usage
            .prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens),
        Some(7)
    );
    assert_eq!(usage.completion_tokens, 10);
    assert_eq!(usage.total_tokens, 17);
}

#[test]
fn completion_token_tracker_follows_chunk_semantics() {
    // Delta stream (the vLLM shape, which the ZMQ lane also emits for
    // TokenSpeed workers): the chunks carry the counts.
    let mut tracker = CompletionTokenTracker::new();
    tracker.record_chunk(&ProtoGenerateStreamChunk::Vllm(vllm::GenerateStreamChunk {
        token_ids: vec![1, 2, 3],
        ..Default::default()
    }));
    tracker.record_complete(&ProtoGenerateComplete::Vllm(Box::new(
        vllm::GenerateComplete {
            completion_tokens: 99,
            ..Default::default()
        },
    )));
    assert_eq!(
        tracker.total(),
        3,
        "delta stream keeps the accumulated count"
    );

    // Cumulative stream: chunks are not accumulated, Complete is the count.
    let mut tracker = CompletionTokenTracker::new();
    tracker.record_chunk(&ProtoGenerateStreamChunk::TokenSpeed(
        tokenspeed::GenerateStreamChunk {
            token_ids: vec![1, 2, 3],
            ..Default::default()
        },
    ));
    tracker.record_complete(&complete(0, 0, 7));
    assert_eq!(
        tracker.total(),
        7,
        "cumulative stream reports it in Complete"
    );
}

#[test]
fn delivered_tokens_follow_a_stream_that_ends_before_complete() {
    // Cumulative stream: a chunk carries the running count; the total is
    // unknown until Complete, the delivered count is not.
    let mut tracker = CompletionTokenTracker::new();
    tracker.record_chunk(&ProtoGenerateStreamChunk::TokenSpeed(
        tokenspeed::GenerateStreamChunk {
            token_ids: vec![5],
            completion_tokens: 41,
            ..Default::default()
        },
    ));
    assert_eq!(tracker.total(), 0, "no authoritative count before Complete");
    assert_eq!(
        tracker.delivered(),
        41,
        "the running count the engine reported"
    );
    tracker.record_complete(&complete(0, 0, 42));
    assert_eq!((tracker.total(), tracker.delivered()), (42, 42));

    // Delta stream: the accumulated chunks are both.
    let mut tracker = CompletionTokenTracker::new();
    tracker.record_chunk(&ProtoGenerateStreamChunk::Vllm(vllm::GenerateStreamChunk {
        token_ids: vec![1, 2, 3],
        ..Default::default()
    }));
    assert_eq!((tracker.total(), tracker.delivered()), (3, 3));
}

#[test]
fn engine_reported_version_beats_dispatch_which_beats_default() {
    assert_eq!(effective_weight_version(Some("v7"), Some("v3")), "v7");
    assert_eq!(effective_weight_version(None, Some("v3")), "v3");
    assert_eq!(effective_weight_version(None, None), "default");
    assert_eq!(
        effective_weight_version(None, Some("")),
        "default",
        "an empty registration label is unset"
    );
}

/// A choice the gateway's own stop ended is settled as its Complete would
/// have settled it: a delta stream keeps what it accumulated, a cumulative
/// stream takes the running count its latest chunk reported; the count is
/// through the step the stop matched in, never the engine's tail.
#[test]
fn a_local_stop_settles_the_count_the_stream_had_reached() {
    // Cumulative stream: the running count of the latest chunk.
    let mut tracker = CompletionTokenTracker::new();
    tracker.record_chunk(&ProtoGenerateStreamChunk::TokenSpeed(
        tokenspeed::GenerateStreamChunk {
            token_ids: vec![5],
            completion_tokens: 41,
            ..Default::default()
        },
    ));
    assert_eq!(tracker.total(), 0, "no settled count before the stop");
    tracker.record_local_stop(0);
    assert_eq!((tracker.total(), tracker.delivered()), (41, 41));

    // Delta stream: the accumulated chunks.
    let mut tracker = CompletionTokenTracker::new();
    tracker.record_chunk(&ProtoGenerateStreamChunk::Vllm(vllm::GenerateStreamChunk {
        token_ids: vec![1, 2, 3],
        ..Default::default()
    }));
    tracker.record_local_stop(0);
    assert_eq!((tracker.total(), tracker.delivered()), (3, 3));

    // A choice that stopped before any chunk settled at zero.
    let mut tracker = CompletionTokenTracker::new();
    tracker.record_local_stop(2);
    assert_eq!((tracker.total(), tracker.delivered()), (0, 0));

    // The Complete of an engine that ended the choice on the stop's step
    // is recorded as usual: the engine's own count of that step.
    let mut tracker = CompletionTokenTracker::new();
    tracker.record_chunk(&ProtoGenerateStreamChunk::TokenSpeed(
        tokenspeed::GenerateStreamChunk {
            token_ids: vec![5],
            completion_tokens: 41,
            ..Default::default()
        },
    ));
    tracker.record_local_stop(0);
    tracker.record_complete(&complete(0, 0, 42));
    assert_eq!(tracker.total(), 42, "the engine's own count of the step");
}

#[cfg(test)]
mod spec_usage_regression {
    use smg_grpc_client::tokenspeed_proto as proto;

    use super::*;

    #[test]
    fn mixed_unknown_proposals_do_not_become_measured_rejections() {
        for (accepted, drafted) in [(2, 0), (4, 2), (0, 0)] {
            let responses: Vec<_> = [(accepted, drafted), (1, 10)]
                .into_iter()
                .map(|(a, d)| {
                    ProtoGenerateComplete::TokenSpeed(proto::GenerateComplete {
                        spec_accepted_tokens: a,
                        spec_draft_tokens: d,
                        ..Default::default()
                    })
                })
                .collect();
            let usage = chat_usage(TokenCounts::of(&responses).usage(), Some(3), 0, None);
            let json = serde_json::to_value(usage).unwrap();
            assert_eq!(
                json["completion_tokens_details"]["accepted_prediction_tokens"],
                accepted + 1
            );
            assert!(
                json["completion_tokens_details"]
                    .get("rejected_prediction_tokens")
                    .is_none(),
                "{json}"
            );
        }
    }
}

#[cfg(test)]
mod zero_accepted_aggregation_regression {
    use smg_grpc_client::tokenspeed_proto as proto;

    use super::*;

    #[test]
    fn partial_zero_accepted_proposals_stay_unknown() {
        let responses: Vec<_> = [(0, 0), (0, 10)]
            .into_iter()
            .map(|(a, d)| {
                ProtoGenerateComplete::TokenSpeed(proto::GenerateComplete {
                    spec_accepted_tokens: a,
                    spec_draft_tokens: d,
                    ..Default::default()
                })
            })
            .collect();
        let usage = chat_usage(TokenCounts::of(&responses).usage(), Some(3), 0, None);
        let details = usage.completion_tokens_details.unwrap();
        assert_eq!(details.accepted_prediction_tokens, Some(0));
        assert_eq!(details.rejected_prediction_tokens, None);
    }
}

#[cfg(test)]
mod default_speculation_control {
    use super::*;

    #[test]
    fn default_counts_keep_legacy_zero_rejected() {
        let details = TokenCounts::default()
            .usage()
            .with_complete_details()
            .completion_tokens_details
            .unwrap();
        assert_eq!(details.accepted_prediction_tokens, Some(0));
        assert_eq!(details.rejected_prediction_tokens, Some(0));
    }
}
