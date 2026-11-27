use super::*;

/// Both counters of the embed pair come from the wire's own fields, on each wire.
#[test]
fn embed_counters_read_each_wires_own_fields() {
    let one = ProtoEmbedComplete::Sglang(sglang::EmbedResponse {
        embedding_dim: 7,
        prompt_tokens: 3,
        ..Default::default()
    });
    let other = ProtoEmbedComplete::Vllm(vllm::EmbedResponse {
        embedding_dim: 11,
        prompt_tokens: 5,
        ..Default::default()
    });
    assert_eq!((one.prompt_tokens(), one.embedding_dim()), (3, 7));
    assert_eq!((other.prompt_tokens(), other.embedding_dim()), (5, 11));
}

/// The embedding comes from the wire's own vector, on each wire.
#[test]
fn the_embedding_reads_each_wires_own_vector() {
    let one = ProtoEmbedComplete::Sglang(sglang::EmbedResponse {
        embedding: vec![0.5, 0.25],
        ..Default::default()
    });
    let other = ProtoEmbedComplete::Vllm(vllm::EmbedResponse {
        embedding: vec![0.125],
        ..Default::default()
    });
    assert_eq!(one.embedding(), &[0.5, 0.25][..]);
    assert_eq!(other.embedding(), &[0.125][..]);
}
