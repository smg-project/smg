// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/handshake.rs.

//! Startup-handshake messages shared by every engine family. The wire shapes
//! originate from vLLM's EngineCore, but the handshake itself is engine-neutral:
//! TokenSpeed speaks the same HELLO/INIT/READY protocol, so the transport uses
//! these types for both.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};

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
    /// How many engines dial this frontend as one group, as the engine
    /// declares it (a TokenSpeed attention-DP group says its size); `None`
    /// from an engine that declares none.
    #[serde(default)]
    pub engine_count: Option<usize>,
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
    /// Optional TokenSpeed diagnostic capture identities; absent on older engines.
    #[serde(default)]
    pub cache_trace_epochs: Vec<String>,
    /// The field order of each positional (array-encoded) struct the engine
    /// decodes, by struct name, as the engine's own declaration has it (a
    /// tagged struct's tag is not counted). A protocol whose requests are
    /// positional checks the prefixes it encodes against it once, at connect
    /// time ([`EngineProtocol::check_layouts`](crate::protocol::EngineProtocol::check_layouts)).
    /// Empty from an engine that declares none, whether it leaves the key
    /// out or sends it as nil.
    #[serde(default, deserialize_with = "nil_as_empty_map")]
    pub positional_layouts: BTreeMap<String, Vec<String>>,
    /// Whether the engine's request decoder resolves a tensor the request
    /// addresses as an auxiliary frame of its message (the typed-tensor
    /// wires' convention: the payload is frame 0 and a tensor's slot carries
    /// the one-based index of its frame). Declared by the SGLang plugin,
    /// which decodes with the engine's own hooks and resolves the frames
    /// itself; an SGLang scheduler that declares nothing is sent every tensor
    /// inline. The vLLM and TokenSpeed decoders resolve them on every
    /// release and declare nothing.
    #[serde(default)]
    pub tensor_aux_frames: bool,
    /// The request types this engine role takes, by name (`add`, `abort`,
    /// `control`, `encode`), as the engine's own declaration lists them: a
    /// scheduler lists the first three, a disaggregated encode worker lists
    /// `encode` alone. Empty from an engine that predates the field, which
    /// takes add and abort (and a control call when `positional_layouts`
    /// names its reply).
    #[serde(default)]
    pub request_types: Vec<String>,
    /// The engine's disaggregation role as it declares it (`prefill`,
    /// `decode`, `encode`; `null` or absent for an engine that serves whole
    /// requests). A worker registered for a role checks it here.
    #[serde(default)]
    pub disaggregation_mode: Option<String>,
    /// Where a prefill or encode engine's bootstrap server listens: the
    /// rendezvous a disaggregated request names so the next side can fetch
    /// what this engine produced. `None` from an engine that declares none.
    #[serde(default)]
    pub bootstrap_host: Option<String>,
    /// See `bootstrap_host`.
    #[serde(default)]
    pub bootstrap_port: Option<u16>,
    /// The checkpoint's `model_type`, which (with `architectures`) picks the
    /// rule that computes a multimodal item's MRoPE positions.
    #[serde(default)]
    pub model_type: Option<String>,
    /// The checkpoint's `architectures`.
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Whether the language model's rotary positions are MRoPE, read off the
    /// checkpoint's rope config; `None` when the engine does not say.
    #[serde(default)]
    pub mrope_positions: Option<bool>,
    /// The vision tower's `spatial_merge_size`, an input of the MRoPE rule.
    #[serde(default)]
    pub spatial_merge_size: Option<u32>,
    /// The checkpoint's image placeholder token id, an input of the MRoPE
    /// rule.
    #[serde(default)]
    pub image_token_id: Option<u32>,
    /// The checkpoint's video placeholder token id.
    #[serde(default)]
    pub video_token_id: Option<u32>,
}

/// The request type an engine lists for the generate path (`add`).
pub const REQUEST_KIND_ADD: &str = "add";
/// The request type a disaggregated encode worker lists (`encode`).
pub const REQUEST_KIND_ENCODE: &str = "encode";

impl EngineCoreReadyResponse {
    /// Whether the engine listed `kind` among the request types it takes.
    /// An engine that declared none takes add and abort: `takes_generate`
    /// reads that silence as yes, this reads it as no.
    pub fn lists_request_kind(&self, kind: &str) -> bool {
        self.request_types.iter().any(|listed| listed == kind)
    }

    /// Whether the engine takes generate requests: every engine that declared
    /// nothing, else one that lists `add`.
    pub fn takes_generate(&self) -> bool {
        self.request_types.is_empty() || self.lists_request_kind(REQUEST_KIND_ADD)
    }

    /// The disaggregation role the engine declared, `None` for an engine
    /// that serves whole requests or declares nothing.
    pub fn disaggregation_role(&self) -> Option<&str> {
        self.disaggregation_mode
            .as_deref()
            .filter(|mode| !mode.is_empty() && *mode != "null")
    }

    /// The bootstrap rendezvous the engine declared, `(host, port)`.
    pub fn bootstrap_address(&self) -> Option<(&str, u16)> {
        Some((
            self.bootstrap_host
                .as_deref()
                .filter(|host| !host.is_empty())?,
            self.bootstrap_port?,
        ))
    }
}

/// A map an engine may send as nil (a Python `None` in a struct that omits
/// nothing): the same as leaving the key out.
fn nil_as_empty_map<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<String, Vec<String>>, D::Error> {
    Ok(Option::<BTreeMap<String, Vec<String>>>::deserialize(deserializer)?.unwrap_or_default())
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
            engine_count: Some(2),
        };
        let bytes = encode_msgpack(&msg).unwrap();
        let decoded: ReadyMessage = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded.status.as_deref(), Some("HELLO"));
        assert_eq!(decoded.local, Some(true));
        assert_eq!(decoded.headless, Some(false));
        assert_eq!(decoded.engine_count, Some(2));
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
        assert!(resp.positional_layouts.is_empty());
        assert!(resp.request_types.is_empty());
        assert!(resp.takes_generate());
        assert!(!resp.lists_request_kind(REQUEST_KIND_ENCODE));
        assert_eq!(resp.disaggregation_role(), None);
        assert_eq!(resp.bootstrap_address(), None);
    }

    /// A TokenSpeed engine declares its role, its bootstrap rendezvous and
    /// the checkpoint facts the MRoPE rule reads, by name; a regular one
    /// spells its role `null`.
    #[test]
    fn ready_response_reads_the_engines_role_rendezvous_and_model_facts() {
        let mut json = serde_json::json!({
            "max_model_len": 8192u64,
            "num_gpu_blocks": 0u64,
            "prefix_granularity": 64u64,
            "dp_stats_address": serde_json::Value::Null,
            "dtype": "bfloat16",
            "vllm_version": "tokenspeed-0.1.1",
            "world_size": 1u64,
            "data_parallel_size": 1u64,
            "request_types": ["encode"],
            "disaggregation_mode": "encode",
            "bootstrap_host": "10.0.0.9",
            "bootstrap_port": 8998u64,
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "mrope_positions": true,
            "spatial_merge_size": 2u64,
            "image_token_id": 248056u64,
            "video_token_id": 248057u64,
        });
        let bytes = rmp_serde::to_vec_named(&json).unwrap();
        let resp: EngineCoreReadyResponse = decode_msgpack(&bytes).unwrap();
        assert!(resp.lists_request_kind(REQUEST_KIND_ENCODE));
        assert!(!resp.takes_generate());
        assert_eq!(resp.disaggregation_role(), Some("encode"));
        assert_eq!(resp.bootstrap_address(), Some(("10.0.0.9", 8998)));
        assert_eq!(resp.architectures, ["Qwen3_5ForConditionalGeneration"]);
        assert_eq!(resp.mrope_positions, Some(true));
        assert_eq!(
            (
                resp.spatial_merge_size,
                resp.image_token_id,
                resp.video_token_id
            ),
            (Some(2), Some(248056), Some(248057))
        );

        json["disaggregation_mode"] = "null".into();
        json["bootstrap_host"] = serde_json::Value::Null;
        json["request_types"] = serde_json::json!(["add", "abort", "control"]);
        let bytes = rmp_serde::to_vec_named(&json).unwrap();
        let resp: EngineCoreReadyResponse = decode_msgpack(&bytes).unwrap();
        assert!(resp.takes_generate());
        assert!(!resp.lists_request_kind(REQUEST_KIND_ENCODE));
        assert_eq!(resp.disaggregation_role(), None);
        assert_eq!(resp.bootstrap_address(), None);
    }

    #[test]
    fn ready_response_reads_the_declared_positional_layouts() {
        let json = serde_json::json!({
            "max_model_len": 4096u64,
            "num_gpu_blocks": 1000u64,
            "block_size": 16u64,
            "dp_stats_address": serde_json::Value::Null,
            "dtype": "bfloat16",
            "vllm_version": "engine-1.0",
            "world_size": 1u64,
            "data_parallel_size": 1u64,
            "positional_layouts": {"SamplingParams": ["max_new_tokens", "stop"]},
        });
        let bytes = rmp_serde::to_vec_named(&json).unwrap();
        let resp: EngineCoreReadyResponse = decode_msgpack(&bytes).unwrap();
        assert_eq!(
            resp.positional_layouts["SamplingParams"],
            vec!["max_new_tokens".to_string(), "stop".to_string()]
        );
        // An engine that sends the key as nil declares nothing, like one
        // that leaves it out.
        let mut json = json;
        json["positional_layouts"] = serde_json::Value::Null;
        let bytes = rmp_serde::to_vec_named(&json).unwrap();
        let resp: EngineCoreReadyResponse = decode_msgpack(&bytes).unwrap();
        assert!(resp.positional_layouts.is_empty());
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
            "cache_trace_epochs": ["capture-epoch"],
        });
        let bytes = rmp_serde::to_vec_named(&json).unwrap();
        let resp: EngineCoreReadyResponse = decode_msgpack(&bytes).unwrap();
        assert_eq!(resp.block_size, 64);
        assert_eq!(resp.cache_trace_epochs, vec!["capture-epoch"]);
    }
}
