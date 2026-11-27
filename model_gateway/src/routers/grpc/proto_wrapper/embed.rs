//! The embedding request and completion wrappers: one variant per wire that
//! serves embeddings, with the accessors the embedding stages and the clients
//! read. Moved out of `proto_wrapper.rs`.

use smg_grpc_client::{sglang_proto as sglang, vllm_proto as vllm};
use smg_worker::identity::basic::RuntimeType;

/// Unified EmbedRequest that works with all backends
#[derive(Clone)]
pub enum ProtoEmbedRequest {
    Sglang(Box<sglang::EmbedRequest>),
    Vllm(Box<vllm::EmbedRequest>),
}

impl ProtoEmbedRequest {
    /// The runtime whose message type this request is.
    pub fn runtime_type(&self) -> RuntimeType {
        match self {
            Self::Sglang(_) => RuntimeType::Sglang,
            Self::Vllm(_) => RuntimeType::Vllm,
        }
    }

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
#[path = "../../../../tests/unit/routers/grpc/proto_wrapper/embed/tests.rs"]
mod tests;
