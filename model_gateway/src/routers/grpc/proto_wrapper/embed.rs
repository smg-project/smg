//! The embedding request and completion wrappers: one variant per wire that
//! serves embeddings, with the typed accessors the embedding stages and the
//! clients read. Moved out of `proto_wrapper.rs` as is.

use smg_grpc_client::{sglang_proto as sglang, vllm_proto as vllm};

/// Unified EmbedRequest that works with all backends
#[derive(Clone)]
pub enum ProtoEmbedRequest {
    Sglang(Box<sglang::EmbedRequest>),
    Vllm(Box<vllm::EmbedRequest>),
}

impl ProtoEmbedRequest {
    /// Serialized wire size, for the release metric.
    pub fn wire_len(&self) -> usize {
        use prost::Message;
        match self {
            Self::Sglang(req) => req.encoded_len(),
            Self::Vllm(req) => req.encoded_len(),
        }
    }

    /// The input's token ids (empty for a text input).
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(req) => req
                .tokenized
                .as_ref()
                .map_or(&[], |input| input.input_ids.as_slice()),
            Self::Vllm(req) => req
                .tokenized
                .as_ref()
                .map_or(&[], |input| input.input_ids.as_slice()),
        }
    }

    /// Get SGLang variant
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang(&self) -> &sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
            Self::Vllm(_) => panic!("Expected SGLang embed request"),
        }
    }

    /// Get mutable SGLang variant
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
            Self::Vllm(_) => panic!("Expected SGLang embed request"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
            Self::Vllm(req) => &req.request_id,
        }
    }

    /// Set request ID.
    pub fn set_request_id(&mut self, request_id: String) {
        match self {
            Self::Sglang(req) => req.request_id = request_id,
            Self::Vllm(req) => req.request_id = request_id,
        }
    }
}

/// Unified embed completion — both backends now use flat EmbedResponse
#[derive(Clone)]
pub enum ProtoEmbedComplete {
    Sglang(sglang::EmbedResponse),
    Vllm(vllm::EmbedResponse),
}

/// The two-wire dispatch of the embed wrapper's readers, written once.
macro_rules! embed_field {
    ($on:expr, $r:ident => $read:expr) => {
        match $on {
            Self::Sglang($r) => $read,
            Self::Vllm($r) => $read,
        }
    };
}

impl ProtoEmbedComplete {
    pub fn embedding(&self) -> &[f32] {
        embed_field!(self, r => &r.embedding)
    }

    pub fn prompt_tokens(&self) -> u32 {
        embed_field!(self, r => r.prompt_tokens)
    }

    pub fn embedding_dim(&self) -> u32 {
        embed_field!(self, r => r.embedding_dim)
    }
}

#[cfg(test)]
mod tests {
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
}
