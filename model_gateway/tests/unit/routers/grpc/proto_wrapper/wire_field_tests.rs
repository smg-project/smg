use super::*;

/// The engine's own reasoning count is read no higher than the completion
/// count of the same frame, on a chunk and on a `Complete` alike: a count
/// within it is read as it is, a count past it as the completion count.
#[test]
fn a_reasoning_count_is_read_within_its_frames_completion_count() {
    for (reasoning, read) in [(3, 3), (5, 5), (7, 5)] {
        let chunk = ProtoGenerateStreamChunk::Sglang(sglang::GenerateStreamChunk {
            completion_tokens: 5,
            reasoning_tokens: reasoning,
            ..Default::default()
        });
        assert_eq!(
            chunk.reasoning_tokens(),
            read,
            "a chunk that counts {reasoning} of 5"
        );
        let complete = ProtoGenerateComplete::Sglang(sglang::GenerateComplete {
            completion_tokens: 5,
            reasoning_tokens: reasoning,
            ..Default::default()
        });
        assert_eq!(
            complete.reasoning_tokens(),
            read,
            "a Complete that counts {reasoning} of 5"
        );
    }
}

/// Every wire's counters come from its own fields: the five-wire form on the
/// chunk and complete wrappers, and the complete wrapper's speculative pair,
/// zero on the wire that has none.
#[test]
fn counters_read_each_wires_own_fields() {
    let chunks = [
        ProtoGenerateStreamChunk::Sglang(sglang::GenerateStreamChunk {
            prompt_tokens: 11,
            completion_tokens: 12,
            cached_tokens: 13,
            ..Default::default()
        }),
        ProtoGenerateStreamChunk::Vllm(vllm::GenerateStreamChunk {
            prompt_tokens: 21,
            completion_tokens: 22,
            cached_tokens: 23,
            ..Default::default()
        }),
        ProtoGenerateStreamChunk::Trtllm(trtllm::GenerateStreamChunk {
            prompt_tokens: 31,
            completion_tokens: 32,
            cached_tokens: 33,
            ..Default::default()
        }),
        ProtoGenerateStreamChunk::Mlx(mlx::GenerateStreamChunk {
            prompt_tokens: 41,
            completion_tokens: 42,
            cached_tokens: 43,
            ..Default::default()
        }),
        ProtoGenerateStreamChunk::TokenSpeed(tokenspeed::GenerateStreamChunk {
            prompt_tokens: 51,
            completion_tokens: 52,
            cached_tokens: 53,
            ..Default::default()
        }),
    ];
    for (n, chunk) in (1u32..).zip(chunks.iter()) {
        let base = 10 * n;
        assert_eq!(
            (
                chunk.prompt_tokens(),
                chunk.completion_tokens(),
                chunk.cached_tokens()
            ),
            (base + 1, base + 2, base + 3)
        );
    }
    let completes = [
        ProtoGenerateComplete::Sglang(sglang::GenerateComplete {
            prompt_tokens: 11,
            completion_tokens: 12,
            cached_tokens: 13,
            spec_accepted_tokens: 14,
            spec_draft_tokens: 15,
            ..Default::default()
        }),
        ProtoGenerateComplete::Vllm(Box::new(vllm::GenerateComplete {
            prompt_tokens: 21,
            completion_tokens: 22,
            cached_tokens: 23,
            spec_accepted_tokens: 24,
            spec_draft_tokens: 25,
            ..Default::default()
        })),
        ProtoGenerateComplete::Trtllm(trtllm::GenerateComplete {
            prompt_tokens: 31,
            completion_tokens: 32,
            cached_tokens: 33,
            spec_accepted_tokens: 34,
            spec_draft_tokens: 35,
            ..Default::default()
        }),
        ProtoGenerateComplete::Mlx(mlx::GenerateComplete {
            prompt_tokens: 41,
            completion_tokens: 42,
            cached_tokens: 43,
            ..Default::default()
        }),
        ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete {
            prompt_tokens: 51,
            completion_tokens: 52,
            cached_tokens: 53,
            spec_accepted_tokens: 54,
            spec_draft_tokens: 55,
            ..Default::default()
        }),
    ];
    for (n, complete) in (1u32..).zip(completes.iter()) {
        let base = 10 * n;
        assert_eq!(
            (
                complete.prompt_tokens(),
                complete.completion_tokens(),
                complete.cached_tokens()
            ),
            (base + 1, base + 2, base + 3)
        );
        let speculative = if matches!(complete, ProtoGenerateComplete::Mlx(_)) {
            (0, 0)
        } else {
            (base + 4, base + 5)
        };
        assert_eq!(
            (
                complete.spec_accepted_tokens(),
                complete.spec_draft_tokens()
            ),
            speculative
        );
    }
}
