use super::{trtllm, ProtoGenerateComplete, ProtoGenerateStreamChunk};

fn one_record() -> Vec<trtllm::TokenLogprob> {
    vec![trtllm::TokenLogprob {
        token_id: 7,
        logprob: -0.5,
        top_logprobs: Vec::new(),
    }]
}

#[test]
fn a_trtllm_chunk_reads_an_empty_logprob_list_as_none() {
    // The TRT-LLM wire says "no logprobs" with an empty list where the
    // other wires leave the field absent; one record is one position.
    let none = ProtoGenerateStreamChunk::Trtllm(trtllm::GenerateStreamChunk::default());
    assert!(none.output_logprobs().is_none());
    let one = ProtoGenerateStreamChunk::Trtllm(trtllm::GenerateStreamChunk {
        logprobs: one_record(),
        ..Default::default()
    });
    assert_eq!(one.output_logprobs().map(|lp| lp.len()), Some(1));
}

#[test]
fn a_trtllm_complete_reads_an_empty_logprob_list_as_none() {
    let none = ProtoGenerateComplete::Trtllm(trtllm::GenerateComplete::default());
    assert!(none.output_logprobs().is_none());
    let one = ProtoGenerateComplete::Trtllm(trtllm::GenerateComplete {
        logprobs: one_record(),
        ..Default::default()
    });
    assert_eq!(one.output_logprobs().map(|lp| lp.len()), Some(1));
}
