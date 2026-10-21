// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/multimodal.rs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::codec::tensor::{TensorFields, WireArrayData, WireTensor};

/// Multimodal feature payload carried at `EngineCoreRequest.mm_features`.
///
/// Python: `list[MultiModalFeatureSpec] | None` (`vllm/v1/engine/__init__.py`).
pub type MmFeatures = Vec<MmFeatureSpec>;

/// A single multimodal input with its processed data and metadata. A request
/// containing multiple multimodal items carries one `MmFeatureSpec` per item.
///
/// Python: `MultiModalFeatureSpec` (`vllm/multimodal/inputs.py`), a dataclass —
/// encodes as a string-keyed msgpack map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MmFeatureSpec {
    /// Processed multimodal data for this item. `None` only when the engine's
    /// receiver cache already holds `identifier` — an external frontend that
    /// does not mirror that cache protocol must always send the full data.
    pub data: Option<MmKwargsItem>,

    /// The input modality, e.g. `"image"`, `"audio"`, `"video"`.
    pub modality: String,

    /// The hash for caching encoder outputs (with LoRA prefix if applicable).
    pub identifier: String,

    /// The location of the `modality` tokens corresponding to this item in
    /// the prompt.
    pub mm_position: PlaceholderRange,

    /// The hash for caching processor outputs (without LoRA prefix).
    #[serde(default)]
    pub mm_hash: Option<String>,
}

impl TensorFields for MmFeatureSpec {
    /// The item's data (its fields in key order), then its embedding mask.
    fn for_each_tensor(&self, visit: &mut dyn FnMut(&WireArrayData)) {
        self.data.for_each_tensor(visit);
        self.mm_position.is_embed.for_each_tensor(visit);
    }

    fn for_each_tensor_mut(&mut self, visit: &mut dyn FnMut(&mut WireArrayData)) {
        self.data.for_each_tensor_mut(visit);
        self.mm_position.is_embed.for_each_tensor_mut(visit);
    }
}

/// Placeholder location information for one multimodal item.
///
/// Python: `PlaceholderRange` (`vllm/multimodal/inputs.py`), a dataclass —
/// encodes as a string-keyed msgpack map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaceholderRange {
    /// The start index of the placeholder in the prompt.
    pub offset: usize,

    /// The length of the placeholder.
    pub length: usize,

    /// A boolean mask of shape `(length,)` indicating which positions between
    /// `offset` and `offset + length` receive embeddings. `None` means all.
    #[serde(default)]
    pub is_embed: Option<WireTensor>,
}

/// Processed keyword arguments for a single multimodal item, keyed by model
/// kwarg name (e.g. `pixel_values`).
///
/// Python: `MultiModalKwargsItem` (`vllm/multimodal/inputs.py`) — encoded by
/// the serializer hooks as a string-keyed map.
pub type MmKwargsItem = BTreeMap<String, MmFieldElem>;

/// One processed keyword argument of a `MmKwargsItem`.
///
/// Python: `MultiModalFieldElem` (`vllm/multimodal/inputs.py`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MmFieldElem {
    /// The keyword argument value passed to the model. `None` only when the
    /// item is cached engine-side (see [`MmFeatureSpec::data`]).
    pub data: Option<MmKwargValue>,

    /// How this field's values combine with other items' for batching.
    pub field: MmField,
}

/// Processed multimodal keyword argument value (Python `NestedTensors`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MmKwargValue {
    Tensor(WireTensor),
    Int(i64),
    Float(f64),
    List(Vec<MmKwargValue>),
}

impl TensorFields for MmFieldElem {
    fn for_each_tensor(&self, visit: &mut dyn FnMut(&WireArrayData)) {
        self.data.for_each_tensor(visit);
    }

    fn for_each_tensor_mut(&mut self, visit: &mut dyn FnMut(&mut WireArrayData)) {
        self.data.for_each_tensor_mut(visit);
    }
}

impl TensorFields for MmKwargValue {
    /// Nested lists are descended in order; scalars carry no tensor.
    fn for_each_tensor(&self, visit: &mut dyn FnMut(&WireArrayData)) {
        match self {
            Self::Tensor(tensor) => tensor.for_each_tensor(visit),
            Self::List(values) => values.for_each_tensor(visit),
            Self::Int(_) | Self::Float(_) => {}
        }
    }

    fn for_each_tensor_mut(&mut self, visit: &mut dyn FnMut(&mut WireArrayData)) {
        match self {
            Self::Tensor(tensor) => tensor.for_each_tensor_mut(visit),
            Self::List(values) => values.for_each_tensor_mut(visit),
            Self::Int(_) | Self::Float(_) => {}
        }
    }
}

/// How to interpret tensor data belonging to a keyword argument.
///
/// Wire form is a 2-tuple `(factory_name, kwargs_map)` with factory names
/// `"batched"`, `"flat"`, `"shared"` — the serializer's
/// `MMF_CLASS_TO_FACTORY` encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "MmFieldWire", into = "MmFieldWire")]
pub enum MmField {
    Batched(MmBatchedField),
    Flat(MmFlatField),
    Shared(MmSharedField),
}

/// Python `MultiModalFieldConfig.batched`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmBatchedField {
    /// If `true`, this field is excluded from being moved to the accelerator
    /// when multimodal items are grouped and batched.
    pub keep_on_cpu: bool,
}

/// Python `MultiModalFieldConfig.flat` / `flat_from_sizes`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmFlatField {
    /// For each multimodal item, a slice (`dim=0`) or a tuple of slices
    /// (`dim>0`) that extracts the data corresponding to it.
    pub slices: Vec<MmSlice>,

    /// The dimension to extract data from, default 0.
    pub dim: i32,

    /// If `true`, this field is excluded from being moved to the accelerator
    /// when multimodal items are grouped and batched.
    pub keep_on_cpu: bool,
}

/// Python `MultiModalFieldConfig.shared`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmSharedField {
    pub batch_size: usize,

    /// If `true`, this field is excluded from being moved to the accelerator
    /// when multimodal items are grouped and batched.
    pub keep_on_cpu: bool,
}

/// Python slice encoded as `(start, stop, step)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
pub struct SliceSpec {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: Option<isize>,
}

/// A single slice or a tuple of slices used by [`MmFlatField`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MmSlice {
    Slice(SliceSpec),
    Slices(Vec<SliceSpec>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
struct MmFieldWire {
    name: String,
    inner: MmFieldWireInner,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
enum MmFieldWireInner {
    Batched(MmBatchedField),
    Flat(MmFlatField),
    Shared(MmSharedField),
}

impl TryFrom<MmFieldWire> for MmField {
    type Error = String;

    fn try_from(value: MmFieldWire) -> Result<Self, Self::Error> {
        match (value.name.as_str(), value.inner) {
            ("batched", MmFieldWireInner::Batched(kwargs)) => Ok(Self::Batched(kwargs)),
            ("flat", MmFieldWireInner::Flat(kwargs)) => Ok(Self::Flat(kwargs)),
            ("shared", MmFieldWireInner::Shared(kwargs)) => Ok(Self::Shared(kwargs)),
            (name, _) => Err(format!(
                "mismatched or unknown multimodal field factory {name:?}"
            )),
        }
    }
}

impl From<MmField> for MmFieldWire {
    fn from(value: MmField) -> Self {
        match value {
            MmField::Batched(kwargs) => Self {
                name: "batched".to_string(),
                inner: MmFieldWireInner::Batched(kwargs),
            },
            MmField::Flat(kwargs) => Self {
                name: "flat".to_string(),
                inner: MmFieldWireInner::Flat(kwargs),
            },
            MmField::Shared(kwargs) => Self {
                name: "shared".to_string(),
                inner: MmFieldWireInner::Shared(kwargs),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use rmpv::Value;

    use super::*;
    use crate::codec::encode_msgpack;

    fn encode_value<T: Serialize + std::fmt::Debug>(value: &T) -> Value {
        let bytes = encode_msgpack(value).expect("encode value");
        rmpv::decode::read_value(&mut Cursor::new(bytes)).expect("decode value")
    }

    #[test]
    fn field_serializes_to_python_factory_tuple() {
        let field = MmField::Flat(MmFlatField {
            slices: vec![MmSlice::Slice(SliceSpec {
                start: Some(0),
                stop: Some(1200),
                step: None,
            })],
            dim: 0,
            keep_on_cpu: false,
        });

        let value = encode_value(&field);
        let Value::Array(items) = value else {
            panic!("field should encode as a 2-tuple array");
        };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].as_str(), Some("flat"));
        let Value::Map(kwargs) = &items[1] else {
            panic!("field kwargs should encode as a map");
        };
        for key in ["slices", "dim", "keep_on_cpu"] {
            assert!(
                kwargs.iter().any(|(k, _)| k.as_str() == Some(key)),
                "missing kwarg {key}"
            );
        }
    }

    #[test]
    fn field_round_trips_python_factory_tuple() {
        for field in [
            MmField::Batched(MmBatchedField { keep_on_cpu: true }),
            MmField::Shared(MmSharedField {
                batch_size: 4,
                keep_on_cpu: false,
            }),
        ] {
            let encoded = encode_msgpack(&field).expect("encode field");
            let decoded: MmField = rmp_serde::from_slice(&encoded).expect("decode field");
            assert_eq!(decoded, field);
        }
    }

    #[test]
    fn feature_spec_serializes_as_named_map_with_tensor_ext() {
        let mut item = MmKwargsItem::new();
        item.insert(
            "pixel_values".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Tensor(
                    WireTensor::from_f32(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                        .expect("tensor built"),
                )),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
            },
        );
        let spec = MmFeatureSpec {
            data: Some(item),
            modality: "image".to_string(),
            identifier: "abc123".to_string(),
            mm_position: PlaceholderRange {
                offset: 5,
                length: 6,
                is_embed: None,
            },
            mm_hash: Some("abc123".to_string()),
        };

        let value = encode_value(&spec);
        let Value::Map(entries) = value else {
            panic!("feature spec should encode as a map");
        };
        for key in ["data", "modality", "identifier", "mm_position", "mm_hash"] {
            assert!(
                entries.iter().any(|(k, _)| k.as_str() == Some(key)),
                "missing key {key}"
            );
        }

        // The tensor payload must reach the wire as the 3-tuple
        // (dtype, shape, ext-3 raw view).
        let data = entries
            .iter()
            .find(|(k, _)| k.as_str() == Some("data"))
            .map(|(_, v)| v)
            .expect("data present");
        let Value::Map(kwargs) = data else {
            panic!("kwargs item should encode as a map");
        };
        let Value::Map(elem) = &kwargs[0].1 else {
            panic!("field elem should encode as a map");
        };
        let tensor = elem
            .iter()
            .find(|(k, _)| k.as_str() == Some("data"))
            .map(|(_, v)| v)
            .expect("elem data present");
        let Value::Array(tuple) = tensor else {
            panic!("tensor should encode as (dtype, shape, data)");
        };
        assert_eq!(tuple[0].as_str(), Some("float32"));
        assert!(matches!(&tuple[2], Value::Ext(3, _)));
    }
}
