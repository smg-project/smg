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

impl ProtoEmbedComplete {
    pub fn embedding(&self) -> &[f32] {
        match self {
            Self::Sglang(r) => &r.embedding,
            Self::Vllm(r) => &r.embedding,
        }
    }

    pub fn prompt_tokens(&self) -> u32 {
        match self {
            Self::Sglang(r) => r.prompt_tokens,
            Self::Vllm(r) => r.prompt_tokens,
        }
    }

    pub fn embedding_dim(&self) -> u32 {
        match self {
            Self::Sglang(r) => r.embedding_dim,
            Self::Vllm(r) => r.embedding_dim,
        }
    }
}
