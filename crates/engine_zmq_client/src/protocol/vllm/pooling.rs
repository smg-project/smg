//! Pooling (embedding) requests on the EngineCore wire: the `PoolingParams` a
//! pooling request carries in place of sampling params, and the pooled tensor
//! the engine returns for it.
//!
//! Mirrors `vllm/pooling_params.py` (`PoolingParams`, `LateInteractionParams`:
//! `array_like` msgspec structs, so positional arrays whose field order is the
//! wire contract) and the `pooling_output: torch.Tensor | None` field of
//! `EngineCoreOutput` (`vllm/v1/engine/__init__.py`), which vLLM's
//! `MsgpackEncoder._encode_tensor` (`vllm/v1/serial_utils.py`) writes as
//! `(torch dtype name, shape, raw-view ext | aux frame index)`.

use std::collections::HashMap;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::{
    codec::{
        dtype::decode_error,
        tensor::{decode_tensor_f32, resolve_array_bytes, DecodedFloatTensor, WireTensor},
    },
    error::Result,
};

/// Which outputs a request streams. Mirrors Python `RequestOutputKind`;
/// pooling accepts only `FinalOnly` (`PoolingParams.__post_init__`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum RequestOutputKind {
    Cumulative = 0,
    Delta = 1,
    #[default]
    FinalOnly = 2,
}

/// The pooling task. Mirrors Python `PoolingTask` (`vllm/tasks.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingTask {
    #[serde(rename = "embed")]
    Embed,
    #[serde(rename = "classify")]
    Classify,
    #[serde(rename = "token_embed")]
    TokenEmbed,
    #[serde(rename = "token_classify")]
    TokenClassify,
    #[serde(rename = "plugin")]
    Plugin,
    #[serde(rename = "embed&token_classify")]
    EmbedAndTokenClassify,
}

/// Every pooling task name (`vllm.tasks.POOLING_TASKS`); what an engine's
/// `get_supported_tasks` answer is filtered by to find its pooling tasks.
pub const POOLING_TASKS: [&str; 6] = [
    "embed",
    "classify",
    "token_embed",
    "token_classify",
    "plugin",
    "embed&token_classify",
];

/// Worker-side late-interaction scoring metadata. Mirrors Python
/// `LateInteractionParams` (`array_like`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
pub struct LateInteractionParams {
    pub mode: String,
    pub query_key: String,
    #[serde(default)]
    pub query_uses: Option<u32>,
}

/// Parameters of a pooling request. Mirrors Python `PoolingParams`, an
/// `array_like` msgspec struct: a positional array in this field order (the
/// wire contract, do not reorder). vLLM encodes all ten positions, defaults
/// included, and so does this struct; a shorter array still decodes.
#[derive(Debug, Clone, PartialEq, Default, Serialize_tuple, Deserialize_tuple)]
pub struct PoolingParams {
    /// Apply the pooler's activation (L2 normalization for `embed`). The
    /// engine's heads treat `None` as off; vLLM's frontend resolves it to the
    /// pooler config's value, else `true`, before the engine sees it.
    #[serde(default)]
    pub use_activation: Option<bool>,
    /// Matryoshka truncation of the embedding.
    #[serde(default)]
    pub dimensions: Option<u32>,
    #[serde(default)]
    pub step_tag_id: Option<u32>,
    #[serde(default)]
    pub returned_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub task: Option<PoolingTask>,
    #[serde(default)]
    pub requires_token_ids: bool,
    #[serde(default)]
    pub skip_reading_prefix_cache: Option<bool>,
    #[serde(default)]
    pub late_interaction_params: Option<LateInteractionParams>,
    #[serde(default)]
    pub extra_kwargs: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub output_kind: RequestOutputKind,
}

/// The model's pooler config fields vLLM's frontend merges into a pooling
/// request's unset parameters (`PoolingParams._merge_default_parameters`):
/// `use_activation` and the Matryoshka `dimensions`, from `--pooler-config`
/// or the model's own pooling config (a sentence-transformers model without
/// a `Normalize` module turns the activation off).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolerDefaults {
    pub use_activation: Option<bool>,
    pub dimensions: Option<u32>,
}

impl PoolingParams {
    /// What vLLM's frontend hands the engine for `PoolingParams(task="embed")`
    /// on a model with these pooler defaults: they fill the unset
    /// `use_activation` and `dimensions` (`_merge_default_parameters`), then
    /// an activation still unset defaults to on (`_set_default_parameters`);
    /// prefix-cache reads stay enabled for sequence pooling.
    pub fn embed_for(pooler: PoolerDefaults) -> Self {
        Self {
            use_activation: pooler.use_activation.or(Some(true)),
            dimensions: pooler.dimensions,
            task: Some(PoolingTask::Embed),
            skip_reading_prefix_cache: Some(false),
            ..Self::default()
        }
    }

    /// What vLLM's frontend hands the engine for an embedding request:
    /// `PoolingParams(task="embed")` after `InputProcessor._validate_params`
    /// ran `verify(model_config)` on it, which turns the activation on
    /// (`_set_default_parameters`) and prefix-cache reads on
    /// (`_merge_default_parameters`). A pooler config overriding
    /// `use_activation` or `dimensions` is the caller's to apply on top.
    pub fn embed() -> Self {
        Self::embed_for(PoolerDefaults::default())
    }
}

/// The pooled tensor a finished pooling request carries in `pooling_output`,
/// as it came off the wire: dtype and shape as the engine wrote them, with an
/// aux-frame payload already resolved to its bytes (shared with the frame,
/// not copied).
#[derive(Debug, Clone, PartialEq)]
pub struct PoolingOutput {
    tensor: WireTensor,
}

impl PoolingOutput {
    /// Wrap an already-built wire tensor (the mock engine's send side).
    pub fn new(tensor: WireTensor) -> Self {
        Self { tensor }
    }

    /// Resolve a wire tensor against the message's aux `frames`.
    pub(crate) fn resolve(tensor: WireTensor, frames: &[Bytes]) -> Result<Self> {
        let WireTensor { dtype, shape, data } = tensor;
        let bytes = resolve_array_bytes(data, "pooling_output", frames)?;
        Ok(Self {
            tensor: WireTensor::from_raw_bytes(dtype, shape, bytes),
        })
    }

    /// The tensor's dtype, as the engine named it (`float32`, `bfloat16`, ...).
    pub fn dtype(&self) -> &str {
        &self.tensor.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.tensor.shape
    }

    /// The tensor in its wire form (the send side's encoding input).
    pub fn as_wire(&self) -> &WireTensor {
        &self.tensor
    }

    /// The values as `f32`, in row-major order with the shape. Any floating
    /// dtype decodes; an integer dtype is an error.
    pub fn decode_f32(&self) -> Result<DecodedFloatTensor> {
        decode_tensor_f32(self.tensor.clone(), "pooling_output", &[])
    }

    /// The embedding vector of a rank-1 tensor, as `f32`. Any other rank is
    /// an error: an `embed` pooler yields one vector per request.
    pub fn to_vector(&self) -> Result<Vec<f32>> {
        let decoded = self.decode_f32()?;
        if decoded.shape.len() != 1 {
            return Err(decode_error(
                "pooling_output",
                &format!("expected a rank-1 embedding, got shape {:?}", decoded.shape),
            ));
        }
        Ok(decoded.data)
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::codec::{decode_msgpack, decode_value, encode_msgpack, unhex};

    /// `msgspec.msgpack.encode(PoolingParams(task="embed", use_activation=True,
    /// skip_reading_prefix_cache=False))` on vLLM 0.30.1rc1: the form the
    /// frontend's `verify` leaves for an embedding request.
    const VLLM_EMBED_PARAMS: &str = "9ac3c0c0c0a5656d626564c2c2c0c002";
    /// `msgspec.msgpack.encode(PoolingParams(task="embed"))`: the raw form,
    /// every default still encoded.
    const VLLM_RAW_EMBED_PARAMS: &str = "9ac0c0c0c0a5656d626564c2c0c0c002";

    #[test]
    fn embed_params_encode_byte_for_byte_like_vllm() {
        assert_eq!(
            encode_msgpack(&PoolingParams::embed()).unwrap(),
            unhex(VLLM_EMBED_PARAMS)
        );
        let raw = PoolingParams {
            task: Some(PoolingTask::Embed),
            ..PoolingParams::default()
        };
        assert_eq!(encode_msgpack(&raw).unwrap(), unhex(VLLM_RAW_EMBED_PARAMS));
        // Spot-check the positions against the Python field order.
        let Value::Array(array) = decode_value(&unhex(VLLM_EMBED_PARAMS)).unwrap() else {
            panic!("expected array");
        };
        assert_eq!(array.len(), 10);
        assert_eq!(array[0], Value::from(true)); // use_activation
        assert_eq!(array[4], Value::from("embed")); // task
        assert_eq!(array[5], Value::from(false)); // requires_token_ids
        assert_eq!(array[6], Value::from(false)); // skip_reading_prefix_cache
        assert_eq!(array[9], Value::from(2)); // output_kind FINAL_ONLY
    }

    /// The pooler config fills what the request left unset, as vLLM's
    /// `_merge_default_parameters` does, before the `use_activation` default.
    #[test]
    fn embed_params_take_the_pooler_defaults() {
        assert_eq!(
            PoolingParams::embed_for(PoolerDefaults::default()),
            PoolingParams::embed()
        );
        let params = PoolingParams::embed_for(PoolerDefaults {
            use_activation: Some(false),
            dimensions: Some(256),
        });
        assert_eq!(params.use_activation, Some(false));
        assert_eq!(params.dimensions, Some(256));
        assert_eq!(params.task, Some(PoolingTask::Embed));
        assert_eq!(params.skip_reading_prefix_cache, Some(false));
        // Dimensions alone leave the activation at its default.
        let params = PoolingParams::embed_for(PoolerDefaults {
            use_activation: None,
            dimensions: Some(64),
        });
        assert_eq!(params.use_activation, Some(true));
        assert_eq!(params.dimensions, Some(64));
    }

    #[test]
    fn pooling_params_decode_full_and_truncated_arrays() {
        assert_eq!(
            decode_msgpack::<PoolingParams>(&unhex(VLLM_EMBED_PARAMS)).unwrap(),
            PoolingParams::embed()
        );
        // A shorter array (older encoder) falls back to the Python defaults.
        let short = Value::Array(vec![Value::Nil, Value::from(64), Value::Nil, Value::Nil]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &short).unwrap();
        let decoded: PoolingParams = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded.dimensions, Some(64));
        assert_eq!(decoded.task, None);
        assert_eq!(decoded.output_kind, RequestOutputKind::FinalOnly);
    }

    #[test]
    fn pooling_output_decodes_to_a_vector_and_rejects_other_ranks() {
        let vector =
            PoolingOutput::new(WireTensor::from_f32(vec![3], vec![0.25, -1.5, 3.0]).unwrap());
        assert_eq!(vector.dtype(), "float32");
        assert_eq!(vector.shape(), &[3]);
        assert_eq!(vector.to_vector().unwrap(), vec![0.25, -1.5, 3.0]);

        let matrix =
            PoolingOutput::new(WireTensor::from_f32(vec![1, 3], vec![0.25, -1.5, 3.0]).unwrap());
        assert_eq!(matrix.decode_f32().unwrap().shape, vec![1, 3]);
        let error = matrix.to_vector().unwrap_err();
        assert!(error.to_string().contains("rank-1"), "{error}");

        let ints = PoolingOutput::new(WireTensor::from_i64(vec![2], vec![1, 2]).unwrap());
        assert!(ints.to_vector().is_err());
    }
}
