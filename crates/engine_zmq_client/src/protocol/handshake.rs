// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/handshake.rs.

//! Startup-handshake messages shared by every engine family. The wire shapes
//! originate from vLLM's EngineCore, but the handshake itself is engine-neutral:
//! TokenSpeed speaks the same HELLO/INIT/READY protocol, so the transport uses
//! these types for both.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::codec::{dtype::ModelDtype, OpaqueValue};

/// Decoded engine startup-handshake payload sent on the handshake socket
/// (`{"status": "HELLO"|"READY", ...}`). Mirrors Python `EngineCoreProc`
/// handshake construction in `vllm/v1/engine/core.py`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadyMessage {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub local: Option<bool>,
    #[serde(default)]
    pub headless: Option<bool>,
    #[serde(default)]
    pub parallel_config_hash: Option<String>,
}

/// KV-event publisher configuration reported by EngineCore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvEventsConfig {
    pub enable_kv_cache_events: bool,
    pub publisher: String,
    pub endpoint: String,
    pub replay_endpoint: Option<String>,
    pub buffer_steps: u32,
    pub hwm: u32,
    pub max_queue_size: u32,
    pub topic: String,
}

/// Post-initialization configuration sent by each engine on the input socket
/// registration message, after the handshake completes.
///
/// Values may differ from the original config (e.g. `max_model_len` after KV
/// cache auto-fitting, `num_gpu_blocks` after profiling), so the frontend must
/// read them rather than assume the CLI values. Mirrors Python
/// `EngineCoreReadyResponse` in `vllm/v1/engine/__init__.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreReadyResponse {
    /// Engine-reported maximum model context length (auto-fitted after KV cache
    /// profiling; may differ from the original config value).
    pub max_model_len: u64,
    /// Number of GPU blocks available for KV cache on this engine.
    pub num_gpu_blocks: u64,
    /// KV cache block size (tokens per block). TokenSpeed renamed the wire
    /// key to `prefix_granularity`; both spellings decode into this field.
    #[serde(alias = "prefix_granularity")]
    pub block_size: u64,
    /// DP coordinator stats publish address, if applicable.
    pub dp_stats_address: Option<String>,
    /// Effective model dtype after Python vLLM resolves `--dtype`.
    pub dtype: ModelDtype,
    /// Python vLLM version reported by the engine process (compatibility gate).
    pub vllm_version: String,
    /// World size (TP * PP) from the parallel config.
    pub world_size: u64,
    /// Data parallelism size from the parallel config.
    pub data_parallel_size: u64,
    // The fields below vary by vLLM version (present on dev/HEAD, absent on the
    // 0.26.0 release). They are `#[serde(default)]` so one struct decodes across
    // versions — the `vllm_version` field above identifies which engine we hit.
    /// Tensor-parallel size of this engine.
    #[serde(default)]
    pub tensor_parallel_size: u32,
    /// Pipeline-parallel size of this engine.
    #[serde(default)]
    pub pipeline_parallel_size: u32,
    /// Decode-context-parallel size of this engine.
    #[serde(default)]
    pub decode_context_parallel_size: u32,
    /// This engine's data-parallel rank. Not sent by all versions; prefer the
    /// engine's ZMQ index for routing (see the connector).
    #[serde(default)]
    pub data_parallel_rank: u32,
    /// Scheduler cap on concurrently running sequences.
    #[serde(default)]
    pub max_num_seqs: u64,
    /// Scheduler cap on batched tokens per step. Signed because TokenSpeed
    /// sends its `chunked_prefill_size` verbatim, where `-1` means "chunked
    /// prefill disabled". SMG does not consume this value.
    #[serde(default)]
    pub max_num_batched_tokens: i64,
    /// Unique identifier for this server instance.
    #[serde(default)]
    pub instance_id: String,
    /// Total KV cache capacity in tokens, if reported.
    #[serde(default)]
    pub kv_cache_size_tokens: Option<u64>,
    /// Maximum achievable request concurrency given the KV cache, if reported.
    #[serde(default)]
    pub kv_cache_max_concurrency: Option<f64>,
    /// KV-event publisher configuration, if configured.
    #[serde(default)]
    pub kv_events_config: Option<KvEventsConfig>,
}

/// Frontend-owned ZMQ addresses sent to the engine during handshake init.
/// Mirrors Python `EngineZmqAddresses` in `vllm/v1/engine/utils.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeAddresses {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub coordinator_input: Option<String>,
    pub coordinator_output: Option<String>,
    pub frontend_stats_publish_address: Option<String>,
}

/// Startup handshake payload sent from the frontend to initialize an engine
/// after receiving `HELLO`. Mirrors Python `EngineHandshakeMetadata`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeInitMessage {
    pub addresses: HandshakeAddresses,
    pub parallel_config: BTreeMap<String, OpaqueValue>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_msgpack, encode_msgpack};

    #[test]
    fn ready_message_roundtrips_hello() {
        let msg = ReadyMessage {
            status: Some("HELLO".to_string()),
            local: Some(true),
            headless: Some(false),
            parallel_config_hash: None,
        };
        let bytes = encode_msgpack(&msg).unwrap();
        let decoded: ReadyMessage = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded.status.as_deref(), Some("HELLO"));
        assert_eq!(decoded.local, Some(true));
        assert_eq!(decoded.headless, Some(false));
    }

    #[test]
    fn ready_response_reads_vllm_version_and_dtype() {
        // A named-field msgpack map (not a positional tuple), like Python sends.
        let json = serde_json::json!({
            "max_model_len": 4096u64,
            "num_gpu_blocks": 1000u64,
            "block_size": 16u64,
            "dp_stats_address": serde_json::Value::Null,
            "dtype": "bfloat16",
            "vllm_version": "0.22.0",
            "world_size": 1u64,
            "data_parallel_size": 1u64,
            "tensor_parallel_size": 1u32,
            "pipeline_parallel_size": 1u32,
            "decode_context_parallel_size": 1u32,
            "data_parallel_rank": 0u32,
            "max_num_seqs": 256u64,
            "max_num_batched_tokens": 8192u64,
            "instance_id": "inst-1",
            "kv_cache_size_tokens": serde_json::Value::Null,
            "kv_cache_max_concurrency": serde_json::Value::Null,
        });
        let resp: EngineCoreReadyResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.vllm_version, "0.22.0");
        assert_eq!(resp.dtype, ModelDtype::BFloat16);
        assert_eq!(resp.data_parallel_rank, 0);
        // Optional trailing field defaults when omitted.
        assert!(resp.kv_events_config.is_none());
    }

    #[test]
    fn ready_response_decodes_negative_max_num_batched_tokens() {
        // TokenSpeed sends chunked_prefill_size verbatim; -1 = disabled. The
        // strict msgpack decoder must accept it (a u64 field would fail here).
        let json = serde_json::json!({
            "max_model_len": 4096u64,
            "num_gpu_blocks": 1000u64,
            "block_size": 16u64,
            "dp_stats_address": serde_json::Value::Null,
            "dtype": "bfloat16",
            "vllm_version": "tokenspeed",
            "world_size": 1u64,
            "data_parallel_size": 1u64,
            "max_num_batched_tokens": -1i64,
        });
        let bytes = rmp_serde::to_vec_named(&json).unwrap();
        let resp: EngineCoreReadyResponse = decode_msgpack(&bytes).unwrap();
        assert_eq!(resp.max_num_batched_tokens, -1);
    }

    #[test]
    fn ready_response_accepts_tokenspeed_prefix_granularity_rename() {
        // TokenSpeed renamed the wire key `block_size` -> `prefix_granularity`;
        // the decoder must accept either spelling.
        let json = serde_json::json!({
            "max_model_len": 131072u64,
            "num_gpu_blocks": 6660u64,
            "prefix_granularity": 64u64,
            "dp_stats_address": serde_json::Value::Null,
            "dtype": "bfloat16",
            "vllm_version": "tokenspeed-0.1.0",
            "world_size": 2u64,
            "data_parallel_size": 2u64,
            "kv_cache_size_tokens": 426176u64,
        });
        let bytes = rmp_serde::to_vec_named(&json).unwrap();
        let resp: EngineCoreReadyResponse = decode_msgpack(&bytes).unwrap();
        assert_eq!(resp.block_size, 64);
    }
}
