//! The labels a host keeps of a worker, as its engine reports them: each
//! engine family's model info and server info answers as one flat map of
//! strings. Every host of the worker runtime reads a worker's labels through
//! these conversions, so a worker carries the same labels whichever process
//! registered it.
//!
//! [`ModelInfo`] and [`ServerInfo`] hold one family's answer each. A flat
//! answer is serialised field by field ([`flat_labels`]); an answer that
//! carries the engine's whole configuration gives up a picked list of keys.

use std::collections::HashMap;

use openai_protocol::worker::ConstrainedDecoding;
use smg_worker::labels::{flat_labels, SUPPORTS_VISION_LABEL};

use crate::{
    media_identity::sglang as media_contract, mlx_proto, sglang_proto, tokenspeed_proto,
    trtllm_proto, vllm_proto,
};

// ---------------------------------------------------------------------------
// Metadata wrappers
// ---------------------------------------------------------------------------

/// One engine family's model info answer.
pub enum ModelInfo {
    Sglang(Box<sglang_proto::GetModelInfoResponse>),
    Vllm(vllm_proto::GetModelInfoResponse),
    Trtllm(trtllm_proto::GetModelInfoResponse),
    Mlx(mlx_proto::GetModelInfoResponse),
    TokenSpeed(Box<tokenspeed_proto::GetModelInfoResponse>),
}

/// One engine family's server info answer.
pub enum ServerInfo {
    Sglang(Box<sglang_proto::GetServerInfoResponse>),
    Vllm(Box<vllm_proto::GetServerInfoResponse>),
    Trtllm(trtllm_proto::GetServerInfoResponse),
    Mlx(mlx_proto::GetServerInfoResponse),
    TokenSpeed(Box<tokenspeed_proto::GetServerInfoResponse>),
}

impl ModelInfo {
    /// Convert to labels. A vision field that reads `false` stays a label only
    /// where the family's servicer computes the field itself.
    pub fn to_labels(&self) -> HashMap<String, String> {
        match self {
            // The vLLM servicer computes `supports_vision` itself (the
            // engine's own verdict; `false` = `--language-model-only` or a
            // text model), so its `false` is a fact worth a label. One
            // caveat: when that computation fails on an unexpected config
            // shape, the servicer logs a warning and reports `false` as
            // well, and a vision model in that state is refused images
            // with the label's message instead of being served.
            ModelInfo::Vllm(info) => flat_labels(info),
            ModelInfo::Sglang(info) => without_unset_vision(flat_labels(info)),
            ModelInfo::Trtllm(info) => without_unset_vision(flat_labels(info)),
            ModelInfo::Mlx(info) => without_unset_vision(flat_labels(info)),
            ModelInfo::TokenSpeed(info) => without_unset_vision(flat_labels(info)),
        }
    }
}

/// A proto3 `bool` cannot tell "unset" from `false`. The SGLang Python
/// servicer reads `supports_vision` from its scheduler info with a default
/// of `false` (a vision model served by it was refused images on that
/// default), and a servicer that predates the field sends the proto default;
/// TokenSpeed's servicer derives the field from its static modality list, an
/// accurate answer, but the field cannot tell that derivation from an older
/// servicer's default. For every runtime but vLLM a `false` is therefore
/// dropped as no verdict: no label, so nothing downstream reads a refusal
/// into it; a `true` is kept.
fn without_unset_vision(mut labels: HashMap<String, String>) -> HashMap<String, String> {
    if labels
        .get(SUPPORTS_VISION_LABEL)
        .is_some_and(|value| value == "false")
    {
        labels.remove(SUPPORTS_VISION_LABEL);
    }
    labels
}

impl ServerInfo {
    /// Convert to labels. SGLang needs special handling because its `server_args`
    /// is a `prost_types::Struct` (not Serialize). vLLM/TRT-LLM are plain structs.
    pub fn to_labels(&self) -> HashMap<String, String> {
        match self {
            ServerInfo::Sglang(info) => {
                let mut labels = HashMap::new();
                if let Some(ref args) = info.server_args {
                    pick_prost_fields(&mut labels, args, SGLANG_GRPC_KEYS);
                }
                if !info.sglang_version.is_empty() {
                    labels.insert("version".to_string(), info.sglang_version.clone());
                }
                labels
            }
            ServerInfo::Vllm(info) => flat_labels(info),
            ServerInfo::Trtllm(info) => flat_labels(info),
            ServerInfo::Mlx(info) => flat_labels(info),
            ServerInfo::TokenSpeed(info) => {
                let mut labels = HashMap::new();
                if let Some(ref args) = info.server_args {
                    pick_prost_fields(&mut labels, args, TOKENSPEED_GRPC_KEYS);
                }
                // TokenSpeed's global request window is shared by its attention-DP ranks.
                labels.insert("enable_dp_attention".to_string(), "true".to_string());
                if !info.tokenspeed_version.is_empty() {
                    labels.insert("version".to_string(), info.tokenspeed_version.clone());
                }
                // The engine's declared constrained-decoding policy, a field of
                // its own rather than a `server_args` key: the `constrained_decoding`
                // label the gateway follows per worker. "" is no declaration.
                if !info.constrained_decoding.is_empty() {
                    labels.insert(
                        ConstrainedDecoding::LABEL.to_string(),
                        info.constrained_decoding.clone(),
                    );
                }
                // Carry the worker's /dev/shm namespace identity (advertised in
                // scheduler_info). The router compares it to its own to decide the
                // SHM tensor transport by *verifying* a shared /dev/shm rather than
                // inferring it from the worker URL. See `worker_shares_dev_shm`.
                if let Some(ref sched) = info.scheduler_info {
                    pick_prost_fields(&mut labels, sched, &["shm_namespace_id"]);
                    if let Some(prost_types::Value {
                        kind: Some(prost_types::value::Kind::ListValue(values)),
                    }) = sched.fields.get("cache_trace_epochs")
                    {
                        let epochs: Option<Vec<&str>> = values
                            .values
                            .iter()
                            .map(|value| match &value.kind {
                                Some(prost_types::value::Kind::StringValue(epoch))
                                    if !epoch.is_empty() =>
                                {
                                    Some(epoch.as_str())
                                }
                                _ => None,
                            })
                            .collect();
                        if let Some(epochs) = epochs.filter(|epochs| !epochs.is_empty()) {
                            labels.insert(
                                "cache_trace_epochs".to_string(),
                                serde_json::json!(epochs).to_string(),
                            );
                        }
                    }
                }
                labels
            }
        }
    }
}

/// Keys worth extracting from SGLang gRPC `server_args` (which contains the full config).
const SGLANG_GRPC_KEYS: &[&str] = &[
    "multimodal_encoder_dtype",
    media_contract::LABEL,
    "mm_processor",
    "mm_media_ref_schemes",
    "model_path",
    "served_model_name",
    "tokenizer_path",
    "tp_size",
    "dp_size",
    "pp_size",
    "context_length",
    "max_total_tokens",
    "max_running_requests",
    // Whether the engine splits `max_running_requests` across its
    // attention-DP workers (the flag, or the width its newer releases resolve
    // it into): each rank worker of such an engine then gets its share of
    // the window (see `create_worker::labels_for_rank`).
    "enable_dp_attention",
    "attn_dp_size",
    "load_balance_method",
    "disaggregation_mode",
    "is_embedding",
    "vocab_size",
    "weight_version",
    // PD pairing protocol: the transport and KV layout a prefill and a decode
    // must share for a handoff to work (the engines validate TP themselves).
    "disaggregation_transfer_backend",
    "disaggregation_bootstrap_port",
    "kv_cache_dtype",
    "page_size",
    "attention_backend",
    // The operator's explicit protocol, which the servicer reads from the
    // engine's SMG_PAIRING_PROTOCOL environment.
    "pairing_protocol",
    // The engine's own per-request media limits (`--limit-mm-data-per-request`),
    // flattened by the servicer to `image=1,video=1`: the gateway's own media
    // pipeline holds requests to them, as the engine's front end would.
    "mm_item_limits",
    // The engine's reasoning parser, under a name of its own (`reasoning_parser`
    // is the per-model parser override read off a worker's labels), and where
    // the engine starts a request's grammar on a thinking prompt, which shapes
    // a forced tool call's grammar (`utils::tool_constraints_for_worker`); the
    // servicer writes both.
    "engine_reasoning_parser",
    "constrained_decoding_start",
    // The vocabulary of structural-tag nodes the engine's grammar compiler
    // takes (`older` for an xgrammar that predates the `optional`, `plus`,
    // `star` and `token` nodes), which the servicer reads off the installed
    // xgrammar: the label by which a worker's structural tags are spelled
    // without `optional`, `plus` and `star` (`TagVocabulary::Older`; a `token`
    // node stays as derived).
    "structural_tag_vocabulary",
    // A servicer's own vision verdict. `GetModelInfo.supports_vision` is a
    // proto3 bool, so a `false` there cannot be told from a field the servicer
    // never set and is dropped for this runtime (`ModelInfo::to_labels`); a
    // servicer that knows it carries no media to its engine writes the
    // verdict here on purpose, and the gateway refuses image and video input
    // for the worker at its gate instead of after fetching it.
    "supports_vision",
];

/// Keys worth extracting from TokenSpeed gRPC `server_args` (post-rename: bare
/// names, not `_path` variants — TokenSpeed dropped the legacy suffixes).
/// TokenSpeed spells its parallelism `attn_tp_size` / `pipeline_parallel_size`
/// / `data_parallel_size`; `normalize_grpc_keys` folds those into the
/// canonical `tp_size` / `pp_size` / `dp_size`.
const TOKENSPEED_GRPC_KEYS: &[&str] = &[
    "model",
    "served_model_name",
    "tokenizer",
    "tp_size",
    "attn_tp_size",
    "dp_size",
    "data_parallel_size",
    "pp_size",
    "pipeline_parallel_size",
    "context_length",
    "max_total_tokens",
    // TokenSpeed's spelling of the scheduler's running window; the fleet
    // capacity accounting and the PD admission gate read either name.
    "max_num_seqs",
    "max_running_requests",
    "load_balance_method",
    "is_embedding",
    "vocab_size",
    "weight_version",
    // PD pairing protocol (see SGLANG_GRPC_KEYS). TokenSpeed has no page
    // size: its KV layout is the cache contract exchanged at rendezvous.
    "disaggregation_mode",
    "disaggregation_transfer_backend",
    "disaggregation_bootstrap_port",
    "kv_cache_dtype",
    "attention_backend",
    "pairing_protocol",
    // The engine's own per-request media limits, flattened by the servicer as
    // for SGLang (`image=4`); TokenSpeed's engine has none to advertise today.
    "mm_item_limits",
    // The engine's reasoning parser and where it starts a request's grammar
    // on a thinking prompt, as for SGLang above; the servicer writes both.
    "engine_reasoning_parser",
    "constrained_decoding_start",
    // The vocabulary of structural-tag nodes the engine's grammar compiler
    // takes (`older` for an xgrammar that predates the `optional`, `plus`,
    // `star` and `token` nodes), which the servicer reads off the installed
    // xgrammar: the label by which a worker's structural tags are spelled
    // without `optional`, `plus` and `star` (`TagVocabulary::Older`; a `token`
    // node stays as derived).
    "structural_tag_vocabulary",
    // RL control plane (crates/rl): where the engine's SGLang-compatible
    // control app listens and what it implements. The engine advertises
    // these; SMG's discovery turns them into the labels the RL crate reads.
    "rl.control_url",
    "rl.pause_modes",
    "rl.update_from",
    "rl.abort",
    "rl.flush_cache",
    "rl.sleep_wake",
    "rl.reports_weight_version",
];

// ---------------------------------------------------------------------------
// Label helpers
// ---------------------------------------------------------------------------

/// Pick specific keys from a `prost_types::Struct`.
fn pick_prost_fields(labels: &mut HashMap<String, String>, s: &prost_types::Struct, keys: &[&str]) {
    for key in keys {
        if let Some(val) = s.fields.get(*key) {
            if let Some(ref kind) = val.kind {
                match kind {
                    prost_types::value::Kind::StringValue(s) if !s.is_empty() && s != "null" => {
                        labels.insert((*key).to_string(), s.clone());
                    }
                    prost_types::value::Kind::NumberValue(n) if *n != 0.0 => {
                        let formatted = if *n == (*n as i64) as f64 {
                            (*n as i64).to_string()
                        } else {
                            n.to_string()
                        };
                        labels.insert((*key).to_string(), formatted);
                    }
                    prost_types::value::Kind::BoolValue(b) => {
                        labels.insert((*key).to_string(), b.to_string());
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use super::{pick_prost_fields, ModelInfo, ServerInfo, SUPPORTS_VISION_LABEL};
    use crate::{sglang_proto, tokenspeed_proto, trtllm_proto, vllm_proto};

    /// EC compatibility fields survive discovery as routing labels.
    #[test]
    fn vllm_ec_capabilities_flatten_into_labels() {
        let info = ServerInfo::Vllm(Box::new(vllm_proto::GetServerInfoResponse {
            supports_ec_transfer: true,
            ec_connector: "ECCPUConnector".into(),
            ec_role: "ec_producer".into(),
            ..Default::default()
        }));
        let labels = info.to_labels();
        assert_eq!(labels["ec_connector"], "ECCPUConnector");
        assert_eq!(labels["ec_role"], "ec_producer");
    }

    /// A vLLM worker's `GetServerInfo` fields flatten into labels as they
    /// are: the engine's per-prompt media limits arrive as `mm_item_limits`
    /// and its encoder budget as `mm_encoder_budget`, and an engine without
    /// them (a text model) sets no such labels.
    #[test]
    fn server_info_to_labels_vllm_carries_the_engines_item_limits() {
        let info = ServerInfo::Vllm(Box::new(vllm_proto::GetServerInfoResponse {
            server_type: "vllm-grpc".to_string(),
            mm_item_limits: "image=8,video=2".to_string(),
            mm_encoder_budget: 192_000,
            ..Default::default()
        }));
        let labels = info.to_labels();
        assert_eq!(
            labels.get("mm_item_limits").map(String::as_str),
            Some("image=8,video=2")
        );
        assert_eq!(
            labels.get("mm_encoder_budget").map(String::as_str),
            Some("192000")
        );
        let text_only = ServerInfo::Vllm(Box::default()).to_labels();
        assert!(!text_only.contains_key("mm_item_limits"));
        assert!(!text_only.contains_key("mm_encoder_budget"));
    }

    /// A servicer's reasoning-parser facts flatten into the
    /// `engine_reasoning_parser` and `constrained_decoding_start` labels the
    /// forced tool-call grammar is shaped by, and never into
    /// `reasoning_parser`, the per-model parser override registration
    /// validates against the router's own registry; an engine whose servicer
    /// reports neither (the proto default) sets no such labels.
    #[test]
    fn server_info_to_labels_carries_where_the_engine_starts_a_grammar() {
        let info = ServerInfo::Vllm(Box::new(vllm_proto::GetServerInfoResponse {
            engine_reasoning_parser: "glm45".to_string(),
            constrained_decoding_start: "after_reasoning".to_string(),
            ..Default::default()
        }));
        let labels = info.to_labels();
        assert_eq!(
            labels.get("engine_reasoning_parser").map(String::as_str),
            Some("glm45")
        );
        assert_eq!(
            labels.get("constrained_decoding_start").map(String::as_str),
            Some("after_reasoning")
        );
        assert!(!labels.contains_key("reasoning_parser"));
        let silent = ServerInfo::Vllm(Box::default()).to_labels();
        assert!(!silent.contains_key("engine_reasoning_parser"));
        assert!(!silent.contains_key("constrained_decoding_start"));

        // The TensorRT-LLM server info names the start in a field of its own.
        let trtllm = ServerInfo::Trtllm(trtllm_proto::GetServerInfoResponse {
            constrained_decoding_start: "first_token".to_string(),
            ..Default::default()
        })
        .to_labels();
        assert_eq!(
            trtllm.get("constrained_decoding_start").map(String::as_str),
            Some("first_token")
        );
        assert!(!ServerInfo::Trtllm(Default::default())
            .to_labels()
            .contains_key("constrained_decoding_start"));
    }

    /// The vocabulary of structural-tag nodes a worker's grammar compiler
    /// takes becomes the `structural_tag_vocabulary` label on every runtime
    /// whose servicer reports it: a field of the vLLM and TensorRT-LLM server
    /// info, a `server_args` key of SGLang's and TokenSpeed's; a servicer
    /// that says nothing (the proto default, no key) sets no label.
    #[test]
    fn server_info_to_labels_carries_the_structural_tag_vocabulary() {
        let vllm = ServerInfo::Vllm(Box::new(vllm_proto::GetServerInfoResponse {
            structural_tag_vocabulary: "older".to_string(),
            ..Default::default()
        }))
        .to_labels();
        assert_eq!(
            vllm.get("structural_tag_vocabulary").map(String::as_str),
            Some("older")
        );
        assert!(!ServerInfo::Vllm(Box::default())
            .to_labels()
            .contains_key("structural_tag_vocabulary"));

        let trtllm = ServerInfo::Trtllm(trtllm_proto::GetServerInfoResponse {
            structural_tag_vocabulary: "older".to_string(),
            ..Default::default()
        })
        .to_labels();
        assert_eq!(
            trtllm.get("structural_tag_vocabulary").map(String::as_str),
            Some("older")
        );
        assert!(!ServerInfo::Trtllm(Default::default())
            .to_labels()
            .contains_key("structural_tag_vocabulary"));

        let keyed = prost_types::Struct {
            fields: BTreeMap::from([(
                "structural_tag_vocabulary".to_string(),
                string_value("older"),
            )]),
        };
        let sglang = ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
            server_args: Some(keyed.clone()),
            ..Default::default()
        }))
        .to_labels();
        assert_eq!(
            sglang.get("structural_tag_vocabulary").map(String::as_str),
            Some("older")
        );
        let tokenspeed =
            ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
                server_args: Some(keyed),
                ..Default::default()
            }))
            .to_labels();
        assert_eq!(
            tokenspeed
                .get("structural_tag_vocabulary")
                .map(String::as_str),
            Some("older")
        );
        let silent = prost_types::Struct {
            fields: BTreeMap::from([("grammar_backend".to_string(), string_value("xgrammar"))]),
        };
        assert!(
            !ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
                server_args: Some(silent.clone()),
                ..Default::default()
            }))
            .to_labels()
            .contains_key("structural_tag_vocabulary")
        );
        assert!(
            !ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
                server_args: Some(silent),
                ..Default::default()
            }))
            .to_labels()
            .contains_key("structural_tag_vocabulary")
        );
    }

    /// A TokenSpeed servicer's declared constrained-decoding policy travels as
    /// a field of its own and becomes the `constrained_decoding` label the
    /// gateway follows per worker; an engine that declares none (the proto
    /// default) sets no such label.
    #[test]
    fn server_info_to_labels_carries_the_engines_constrained_decoding_policy() {
        let info = ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
            constrained_decoding: "always".to_string(),
            ..Default::default()
        }));
        assert_eq!(
            info.to_labels()
                .get("constrained_decoding")
                .map(String::as_str),
            Some("always")
        );
        let silent = ServerInfo::TokenSpeed(Box::default()).to_labels();
        assert!(!silent.contains_key("constrained_decoding"));
    }

    /// An SGLang servicer's own vision verdict travels in `server_args`,
    /// where a `false` is deliberate (the proto3 `GetModelInfo` bool cannot
    /// say so): kept as the label either way, absent when the servicer wrote
    /// none.
    #[test]
    fn server_info_to_labels_sglang_keeps_the_servicers_own_vision_verdict() {
        let info = |verdict: Option<bool>| {
            let mut fields = BTreeMap::from([("tp_size".to_string(), number_value(2.0))]);
            if let Some(verdict) = verdict {
                fields.insert(
                    "supports_vision".to_string(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::BoolValue(verdict)),
                    },
                );
            }
            ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
                server_args: Some(prost_types::Struct { fields }),
                ..Default::default()
            }))
        };
        assert_eq!(
            info(Some(false))
                .to_labels()
                .get(SUPPORTS_VISION_LABEL)
                .map(String::as_str),
            Some("false")
        );
        assert_eq!(
            info(Some(true))
                .to_labels()
                .get(SUPPORTS_VISION_LABEL)
                .map(String::as_str),
            Some("true")
        );
        let silent = info(None).to_labels();
        assert!(!silent.contains_key(SUPPORTS_VISION_LABEL));
        assert_eq!(silent.get("tp_size").map(String::as_str), Some("2"));
    }

    /// `supports_vision` is a label only when a servicer set it on purpose:
    /// vLLM's `false` stays (language-model-only / text model); the SGLang
    /// and TokenSpeed protos carry the same `bool`, which their servicers may
    /// leave at the proto default even for a vision model, so their `false`
    /// yields no label (unknown) and their `true` is kept.
    #[test]
    fn model_info_to_labels_keeps_only_a_deliberate_vision_verdict() {
        let vllm_text = ModelInfo::Vllm(vllm_proto::GetModelInfoResponse {
            supports_vision: false,
            ..Default::default()
        });
        assert_eq!(
            vllm_text
                .to_labels()
                .get(SUPPORTS_VISION_LABEL)
                .map(String::as_str),
            Some("false")
        );
        let vllm_vision = ModelInfo::Vllm(vllm_proto::GetModelInfoResponse {
            supports_vision: true,
            ..Default::default()
        });
        assert_eq!(
            vllm_vision
                .to_labels()
                .get(SUPPORTS_VISION_LABEL)
                .map(String::as_str),
            Some("true")
        );

        let sglang_unset = ModelInfo::Sglang(Box::default());
        assert!(!sglang_unset.to_labels().contains_key(SUPPORTS_VISION_LABEL));
        let sglang_vision = ModelInfo::Sglang(Box::new(sglang_proto::GetModelInfoResponse {
            supports_vision: true,
            ..Default::default()
        }));
        assert_eq!(
            sglang_vision
                .to_labels()
                .get(SUPPORTS_VISION_LABEL)
                .map(String::as_str),
            Some("true")
        );

        let tokenspeed_unset = ModelInfo::TokenSpeed(Box::default());
        assert!(!tokenspeed_unset
            .to_labels()
            .contains_key(SUPPORTS_VISION_LABEL));
        // Other booleans keep flattening as before, `false` included.
        assert_eq!(
            ModelInfo::Sglang(Box::new(sglang_proto::GetModelInfoResponse {
                is_generation: false,
                ..Default::default()
            }))
            .to_labels()
            .get("is_generation")
            .map(String::as_str),
            Some("false")
        );
    }

    fn string_value(s: &str) -> prost_types::Value {
        prost_types::Value {
            kind: Some(prost_types::value::Kind::StringValue(s.to_string())),
        }
    }

    fn number_value(n: f64) -> prost_types::Value {
        prost_types::Value {
            kind: Some(prost_types::value::Kind::NumberValue(n)),
        }
    }

    /// The vLLM servicer's running window flattens into the `max_num_seqs`
    /// label the PD admission gate and the capacity tracker read; an engine
    /// that reports none (the proto default) leaves no label, as before.
    #[test]
    fn server_info_to_labels_vllm_carries_the_running_window() {
        let info = ServerInfo::Vllm(Box::new(vllm_proto::GetServerInfoResponse {
            kv_connector: "NixlConnector".to_string(),
            max_num_seqs: 64,
            ..Default::default()
        }));
        let labels = info.to_labels();
        assert_eq!(labels.get("max_num_seqs").map(String::as_str), Some("64"));
        assert_eq!(
            labels.get("kv_connector").map(String::as_str),
            Some("NixlConnector")
        );

        let unknown = ServerInfo::Vllm(Box::default());
        assert!(!unknown.to_labels().contains_key("max_num_seqs"));
    }

    /// The `/workers` metadata path for TokenSpeed: curated `server_args` keys
    /// become labels, everything else (unlisted keys, scheduler_info, runtime
    /// state) stays out.
    #[test]
    fn server_info_to_labels_tokenspeed_picks_curated_keys_and_version() {
        let info = ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("model".to_string(), string_value("Qwen/Qwen3-8B")),
                    ("tokenizer".to_string(), string_value("Qwen/Qwen3-8B")),
                    ("tp_size".to_string(), number_value(2.0)),
                    ("max_total_tokens".to_string(), number_value(8192.0)),
                    ("disaggregation_mode".to_string(), string_value("prefill")),
                    (
                        "disaggregation_transfer_backend".to_string(),
                        string_value("mooncake"),
                    ),
                    (
                        "disaggregation_bootstrap_port".to_string(),
                        number_value(8998.0),
                    ),
                    ("kv_cache_dtype".to_string(), string_value("auto")),
                    ("attention_backend".to_string(), string_value("flashinfer")),
                    ("pipeline_parallel_size".to_string(), number_value(2.0)),
                    (
                        "rl.control_url".to_string(),
                        string_value("http://10.0.0.5:40100"),
                    ),
                    (
                        "rl.pause_modes".to_string(),
                        string_value("wait,abort,keep"),
                    ),
                    (
                        "rl.reports_weight_version".to_string(),
                        string_value("true"),
                    ),
                    ("mm_item_limits".to_string(), string_value("image=4")),
                    // Not in TOKENSPEED_GRPC_KEYS — must not become a label.
                    ("host".to_string(), string_value("127.0.0.1")),
                ]),
            }),
            scheduler_info: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("status".to_string(), string_value("ready")),
                    (
                        "cache_trace_epochs".to_string(),
                        prost_types::Value {
                            kind: Some(prost_types::value::Kind::ListValue(
                                prost_types::ListValue {
                                    values: vec![
                                        string_value("replica-one"),
                                        string_value("replica-two"),
                                    ],
                                },
                            )),
                        },
                    ),
                ]),
            }),
            active_requests: 3,
            uptime_seconds: 12.5,
            max_total_num_tokens: 8192,
            tokenspeed_version: "0.1.0".to_string(),
            ..Default::default()
        }));

        let labels = info.to_labels();

        assert_eq!(
            labels.get("model").map(String::as_str),
            Some("Qwen/Qwen3-8B")
        );
        assert_eq!(
            labels.get("tokenizer").map(String::as_str),
            Some("Qwen/Qwen3-8B")
        );
        // Integral numbers are formatted without a decimal point.
        assert_eq!(labels.get("tp_size").map(String::as_str), Some("2"));
        assert_eq!(
            labels.get("max_total_tokens").map(String::as_str),
            Some("8192")
        );
        assert_eq!(labels.get("version").map(String::as_str), Some("0.1.0"));
        assert_eq!(
            labels.get("cache_trace_epochs").map(String::as_str),
            Some(r#"["replica-one","replica-two"]"#)
        );
        assert!(!labels.contains_key("host"));
        // The pairing-protocol facts survive, under the engine's own names;
        // the discovery step canonicalises the parallelism spellings.
        assert_eq!(
            labels.get("disaggregation_mode").map(String::as_str),
            Some("prefill")
        );
        assert_eq!(
            labels
                .get("disaggregation_transfer_backend")
                .map(String::as_str),
            Some("mooncake")
        );
        assert_eq!(
            labels
                .get("disaggregation_bootstrap_port")
                .map(String::as_str),
            Some("8998")
        );
        assert_eq!(
            labels.get("kv_cache_dtype").map(String::as_str),
            Some("auto")
        );
        assert_eq!(
            labels.get("attention_backend").map(String::as_str),
            Some("flashinfer")
        );
        assert_eq!(
            labels.get("pipeline_parallel_size").map(String::as_str),
            Some("2")
        );
        assert_eq!(
            labels.get("rl.control_url").map(String::as_str),
            Some("http://10.0.0.5:40100")
        );
        assert_eq!(
            labels.get("rl.pause_modes").map(String::as_str),
            Some("wait,abort,keep")
        );
        assert_eq!(
            labels.get("rl.reports_weight_version").map(String::as_str),
            Some("true")
        );
        // The engine's per-request media limits, for the media pipeline.
        assert_eq!(
            labels.get("mm_item_limits").map(String::as_str),
            Some("image=4")
        );
        // scheduler_info and transient runtime state never become labels.
        assert!(!labels.contains_key("status"));
        assert!(!labels.contains_key("active_requests"));
        assert!(!labels.contains_key("uptime_seconds"));
    }

    /// The TokenSpeed servicer's grammar-start facts ride with the server args
    /// and become the `engine_reasoning_parser` and `constrained_decoding_start`
    /// labels, as for SGLang; the engine's own `reasoning_parser` argument is
    /// not the router's parser override and stays out.
    #[test]
    fn server_info_to_labels_tokenspeed_carries_where_the_engine_starts_a_grammar() {
        let info = ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("reasoning_parser".to_string(), string_value("kimi_k25")),
                    (
                        "engine_reasoning_parser".to_string(),
                        string_value("kimi_k25"),
                    ),
                    (
                        "constrained_decoding_start".to_string(),
                        string_value("first_token"),
                    ),
                ]),
            }),
            ..Default::default()
        }));

        let labels = info.to_labels();

        assert_eq!(
            labels.get("engine_reasoning_parser").map(String::as_str),
            Some("kimi_k25")
        );
        assert_eq!(
            labels.get("constrained_decoding_start").map(String::as_str),
            Some("first_token")
        );
        assert!(!labels.contains_key("reasoning_parser"));

        // A servicer that writes neither key: no label, the router keeps the
        // grammar shape that fits either engine.
        let silent = ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([(
                    "reasoning_parser".to_string(),
                    string_value("kimi_k25"),
                )]),
            }),
            ..Default::default()
        }))
        .to_labels();
        assert!(!silent.contains_key("engine_reasoning_parser"));
        assert!(!silent.contains_key("constrained_decoding_start"));
        assert!(!silent.contains_key("reasoning_parser"));
    }

    /// SGLang server facts expose routing capabilities while filtering secrets.
    #[test]
    fn server_info_to_labels_sglang_picks_curated_keys_and_version() {
        let info = ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("model_path".to_string(), string_value("Qwen/Qwen3-8B")),
                    ("dp_size".to_string(), number_value(4.0)),
                    (
                        "is_embedding".to_string(),
                        prost_types::Value {
                            kind: Some(prost_types::value::Kind::BoolValue(false)),
                        },
                    ),
                    (
                        "disaggregation_transfer_backend".to_string(),
                        string_value("nixl"),
                    ),
                    (
                        "disaggregation_bootstrap_port".to_string(),
                        number_value(8998.0),
                    ),
                    ("kv_cache_dtype".to_string(), string_value("fp8_e5m2")),
                    ("page_size".to_string(), number_value(64.0)),
                    ("attention_backend".to_string(), string_value("fa3")),
                    ("pairing_protocol".to_string(), string_value("kv-v1")),
                    (
                        "mm_item_limits".to_string(),
                        string_value("image=1,video=1"),
                    ),
                    (
                        "multimodal_encoder_dtype".to_string(),
                        string_value("bfloat16"),
                    ),
                    // Not in SGLANG_GRPC_KEYS — must not become a label.
                    ("api_key".to_string(), string_value("secret")),
                ]),
            }),
            sglang_version: "0.4.0".to_string(),
            ..Default::default()
        }));

        let labels = info.to_labels();

        assert_eq!(
            labels.get("model_path").map(String::as_str),
            Some("Qwen/Qwen3-8B")
        );
        assert_eq!(labels.get("dp_size").map(String::as_str), Some("4"));
        // Booleans are kept even when false (embedding detection relies on it).
        assert_eq!(
            labels.get("is_embedding").map(String::as_str),
            Some("false")
        );
        assert_eq!(labels.get("version").map(String::as_str), Some("0.4.0"));
        assert!(!labels.contains_key("api_key"));
        assert_eq!(
            labels.get("multimodal_encoder_dtype").map(String::as_str),
            Some("bfloat16")
        );
        // The pairing-protocol facts survive: transport, bootstrap port, KV layout.
        assert_eq!(
            labels
                .get("disaggregation_transfer_backend")
                .map(String::as_str),
            Some("nixl")
        );
        assert_eq!(
            labels
                .get("disaggregation_bootstrap_port")
                .map(String::as_str),
            Some("8998")
        );
        assert_eq!(
            labels.get("kv_cache_dtype").map(String::as_str),
            Some("fp8_e5m2")
        );
        assert_eq!(labels.get("page_size").map(String::as_str), Some("64"));
        assert_eq!(
            labels.get("attention_backend").map(String::as_str),
            Some("fa3")
        );
        // The explicit protocol the servicer read from the engine's environment.
        assert_eq!(
            labels.get("pairing_protocol").map(String::as_str),
            Some("kv-v1")
        );
        // The engine's per-request media limits, for the media pipeline.
        assert_eq!(
            labels.get("mm_item_limits").map(String::as_str),
            Some("image=1,video=1")
        );
    }

    /// The servicer's grammar-start facts ride with the server args and become
    /// the `engine_reasoning_parser` and `constrained_decoding_start` labels. The
    /// engine's own `reasoning_parser` argument, which the server args carry
    /// as well, is not the router's parser override and stays out.
    #[test]
    fn server_info_to_labels_sglang_carries_where_the_engine_starts_a_grammar() {
        let info = ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("reasoning_parser".to_string(), string_value("qwen3")),
                    ("engine_reasoning_parser".to_string(), string_value("qwen3")),
                    (
                        "constrained_decoding_start".to_string(),
                        string_value("after_reasoning"),
                    ),
                ]),
            }),
            ..Default::default()
        }));

        let labels = info.to_labels();

        assert_eq!(
            labels.get("engine_reasoning_parser").map(String::as_str),
            Some("qwen3")
        );
        assert_eq!(
            labels.get("constrained_decoding_start").map(String::as_str),
            Some("after_reasoning")
        );
        assert!(
            !labels.contains_key("reasoning_parser"),
            "the engine's parser argument is not the router's parser override"
        );

        // A servicer that writes neither key (one from before the facts): no
        // label, so the router keeps the grammar shape that fits either engine.
        let silent = ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([("reasoning_parser".to_string(), string_value("qwen3"))]),
            }),
            ..Default::default()
        }))
        .to_labels();
        assert!(!silent.contains_key("engine_reasoning_parser"));
        assert!(!silent.contains_key("constrained_decoding_start"));
        assert!(!silent.contains_key("reasoning_parser"));
    }

    /// `GetModelInfoResponse` is flat for every backend, so it serializes via
    /// `flat_labels`: empty strings and zero numbers are skipped, booleans are
    /// kept (except a non-vLLM `supports_vision: false`, which is no verdict),
    /// arrays are JSON-encoded.
    #[test]
    fn model_info_to_labels_tokenspeed_flat_serializes_skipping_empty_and_zero() {
        let info = ModelInfo::TokenSpeed(Box::new(tokenspeed_proto::GetModelInfoResponse {
            model_path: "Qwen/Qwen3-8B".to_string(),
            tokenizer_path: String::new(), // empty — skipped
            served_model_name: "qwen3-8b".to_string(),
            architectures: vec!["Qwen3ForCausalLM".to_string()],
            max_context_length: 32768,
            vocab_size: 151_936,
            pad_token_id: 0, // zero — skipped
            supports_vision: false,
            supports_multimodal: false,
            ..Default::default()
        }));

        let labels = info.to_labels();

        assert_eq!(
            labels.get("model_path").map(String::as_str),
            Some("Qwen/Qwen3-8B")
        );
        assert_eq!(
            labels.get("served_model_name").map(String::as_str),
            Some("qwen3-8b")
        );
        assert_eq!(
            labels.get("max_context_length").map(String::as_str),
            Some("32768")
        );
        assert_eq!(labels.get("vocab_size").map(String::as_str), Some("151936"));
        assert_eq!(
            labels.get("architectures").map(String::as_str),
            Some(r#"["Qwen3ForCausalLM"]"#)
        );
        // A proto-default `false` is no vision verdict on this runtime...
        assert!(!labels.contains_key("supports_vision"));
        // ...while other booleans flatten as before, `false` included.
        assert_eq!(
            labels.get("supports_multimodal").map(String::as_str),
            Some("false")
        );
        assert!(!labels.contains_key("tokenizer_path"));
        assert!(!labels.contains_key("pad_token_id"));
    }

    /// A picked number that is zero is left out; a whole number is written with
    /// no point and a fraction in full.
    #[test]
    fn a_picked_zero_is_left_out_and_a_fraction_is_kept_whole() {
        let number = |n: f64| prost_types::Value {
            kind: Some(prost_types::value::Kind::NumberValue(n)),
        };
        let report = prost_types::Struct {
            fields: [
                ("unset".to_string(), number(0.0)),
                ("whole".to_string(), number(8.0)),
                ("part".to_string(), number(0.5)),
            ]
            .into_iter()
            .collect(),
        };
        let mut labels = HashMap::new();
        pick_prost_fields(&mut labels, &report, &["unset", "whole", "part"]);
        assert_eq!(labels.get("unset"), None);
        assert_eq!(labels.get("whole").map(String::as_str), Some("8"));
        assert_eq!(labels.get("part").map(String::as_str), Some("0.5"));
    }
}
