//! Shared encoder-input types for all encoder-backed modalities.

use std::{borrow::Cow, collections::HashMap};

use anyhow::{Context, Result as AnyhowResult};
use ndarray::{Array, ArrayD, Axis, Dimension};

use crate::types::FieldLayout;

/// Model-specific auxiliary output values.
#[derive(Debug, Clone)]
pub enum ModelSpecificValue {
    /// A tensor with shape information (data as flat vec, shape as dims)
    Tensor { data: Vec<f32>, shape: Vec<usize> },

    /// A tensor of integers (e.g., aspect_ratio_ids)
    IntTensor { data: Vec<i64>, shape: Vec<usize> },

    /// A tensor of unsigned integers (e.g., image_grid_thw)
    UintTensor { data: Vec<u32>, shape: Vec<usize> },

    /// Simple integer value
    Int(i64),

    /// Simple float value
    Float(f64),

    /// List of integers
    IntVec(Vec<i64>),

    /// List of unsigned integers
    UintVec(Vec<u32>),

    /// List of floats
    FloatVec(Vec<f32>),

    /// List of tuples (e.g., media item sizes)
    TupleVec(Vec<(u32, u32)>),

    /// Boolean flag
    Bool(bool),
}

impl ModelSpecificValue {
    /// Create a 1D uint tensor from a vector.
    pub fn uint_1d(data: Vec<u32>) -> Self {
        let len = data.len();
        Self::UintTensor {
            data,
            shape: vec![len],
        }
    }

    /// Create a 2D uint tensor.
    pub fn uint_2d(data: Vec<u32>, rows: usize, cols: usize) -> Self {
        Self::UintTensor {
            data,
            shape: vec![rows, cols],
        }
    }

    /// Create a 1D int tensor from a vector.
    pub fn int_1d(data: Vec<i64>) -> Self {
        let len = data.len();
        Self::IntTensor {
            data,
            shape: vec![len],
        }
    }

    /// Create a 2D int tensor.
    pub fn int_2d(data: Vec<i64>, rows: usize, cols: usize) -> Self {
        Self::IntTensor {
            data,
            shape: vec![rows, cols],
        }
    }

    /// Interpret this value as per-item flat sizes.
    pub fn as_flat_sizes(&self) -> AnyhowResult<Vec<usize>> {
        match self {
            Self::IntTensor { data, .. } => data
                .iter()
                .map(|&v| usize::try_from(v).context("negative flat size"))
                .collect(),
            Self::UintTensor { data, .. } => Ok(data.iter().map(|&v| v as usize).collect()),
            Self::IntVec(values) => values
                .iter()
                .map(|&v| usize::try_from(v).context("negative flat size"))
                .collect(),
            Self::UintVec(values) => Ok(values.iter().map(|&v| v as usize).collect()),
            _ => Err(anyhow::anyhow!("unsupported flat sizes value type")),
        }
    }

    /// Slice item-batched metadata along the first dimension.
    pub fn slice_first_dim(&self, start: usize, len: usize) -> AnyhowResult<Self> {
        match self {
            Self::Tensor { data, shape } => {
                let (data, shape) = slice_tensor_first_dim(data, shape, start, len)?;
                Ok(Self::Tensor { data, shape })
            }
            Self::IntTensor { data, shape } => {
                let (data, shape) = slice_tensor_first_dim(data, shape, start, len)?;
                Ok(Self::IntTensor { data, shape })
            }
            Self::UintTensor { data, shape } => {
                let (data, shape) = slice_tensor_first_dim(data, shape, start, len)?;
                Ok(Self::UintTensor { data, shape })
            }
            Self::IntVec(values) => Ok(Self::IntVec(slice_1d(values, start, len)?.to_vec())),
            Self::UintVec(values) => Ok(Self::UintVec(slice_1d(values, start, len)?.to_vec())),
            Self::FloatVec(values) => Ok(Self::FloatVec(slice_1d(values, start, len)?.to_vec())),
            Self::TupleVec(values) => Ok(Self::TupleVec(slice_1d(values, start, len)?.to_vec())),
            _ => Ok(self.clone()),
        }
    }

    /// Stack per-item values along their first dimension. A scalar has no
    /// item dimension: it must be the same in every part and is kept once.
    pub fn concat_first_dim(parts: &[&Self]) -> AnyhowResult<Self> {
        let first = parts.first().context("cannot join zero values")?;
        match first {
            Self::Tensor { .. } => {
                let (data, shape) = concat_tensors(parts, |part| match part {
                    Self::Tensor { data, shape } => Some((data.as_slice(), shape.as_slice())),
                    _ => None,
                })?;
                Ok(Self::Tensor { data, shape })
            }
            Self::IntTensor { .. } => {
                let (data, shape) = concat_tensors(parts, |part| match part {
                    Self::IntTensor { data, shape } => Some((data.as_slice(), shape.as_slice())),
                    _ => None,
                })?;
                Ok(Self::IntTensor { data, shape })
            }
            Self::UintTensor { .. } => {
                let (data, shape) = concat_tensors(parts, |part| match part {
                    Self::UintTensor { data, shape } => Some((data.as_slice(), shape.as_slice())),
                    _ => None,
                })?;
                Ok(Self::UintTensor { data, shape })
            }
            Self::IntVec(_) => Ok(Self::IntVec(concat_vecs(parts, |part| match part {
                Self::IntVec(values) => Some(values.as_slice()),
                _ => None,
            })?)),
            Self::UintVec(_) => Ok(Self::UintVec(concat_vecs(parts, |part| match part {
                Self::UintVec(values) => Some(values.as_slice()),
                _ => None,
            })?)),
            Self::FloatVec(_) => Ok(Self::FloatVec(concat_vecs(parts, |part| match part {
                Self::FloatVec(values) => Some(values.as_slice()),
                _ => None,
            })?)),
            Self::TupleVec(_) => Ok(Self::TupleVec(concat_vecs(parts, |part| match part {
                Self::TupleVec(values) => Some(values.as_slice()),
                _ => None,
            })?)),
            Self::Int(_) | Self::Float(_) | Self::Bool(_) => {
                anyhow::ensure!(
                    parts.iter().all(|part| part.same_scalar(first)),
                    "scalar model-specific value differs between batch parts"
                );
                Ok((*first).clone())
            }
        }
    }

    fn same_scalar(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Bool(a), Self::Bool(b)) => a == b,
            _ => false,
        }
    }
}

/// Concatenate tensors along the first dimension; the other dimensions must agree.
fn concat_tensors<'a, T: Clone + 'a>(
    parts: &[&'a ModelSpecificValue],
    view: impl Fn(&'a ModelSpecificValue) -> Option<(&'a [T], &'a [usize])>,
) -> AnyhowResult<(Vec<T>, Vec<usize>)> {
    let mut data = Vec::new();
    let mut shape: Option<Vec<usize>> = None;
    for part in parts {
        let (part_data, part_shape) =
            view(part).context("model-specific value kinds differ between batch parts")?;
        let (&first_dim, rest) = part_shape
            .split_first()
            .context("cannot join scalar tensors along an item dimension")?;
        match &mut shape {
            None => shape = Some(part_shape.to_vec()),
            Some(shape) => {
                anyhow::ensure!(
                    shape[1..] == *rest,
                    "model-specific tensor shapes differ beyond the item dimension"
                );
                shape[0] += first_dim;
            }
        }
        data.extend_from_slice(part_data);
    }
    Ok((data, shape.context("cannot join zero tensors")?))
}

fn concat_vecs<'a, T: Clone + 'a>(
    parts: &[&'a ModelSpecificValue],
    view: impl Fn(&'a ModelSpecificValue) -> Option<&'a [T]>,
) -> AnyhowResult<Vec<T>> {
    let mut joined = Vec::new();
    for part in parts {
        joined.extend_from_slice(
            view(part).context("model-specific value kinds differ between batch parts")?,
        );
    }
    Ok(joined)
}

fn slice_tensor_first_dim<T: Clone>(
    data: &[T],
    shape: &[usize],
    start: usize,
    len: usize,
) -> AnyhowResult<(Vec<T>, Vec<usize>)> {
    let first_dim = *shape
        .first()
        .ok_or_else(|| anyhow::anyhow!("cannot slice scalar tensor"))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("tensor slice range overflow"))?;
    anyhow::ensure!(
        end <= first_dim,
        "tensor first-dimension slice {start}..{end} exceeds {first_dim}"
    );
    let row_width = shape[1..]
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| anyhow::anyhow!("tensor row width overflow"))?;
    let data_start = start
        .checked_mul(row_width)
        .ok_or_else(|| anyhow::anyhow!("tensor data start overflow"))?;
    let data_len = len
        .checked_mul(row_width)
        .ok_or_else(|| anyhow::anyhow!("tensor data length overflow"))?;
    let data_end = data_start
        .checked_add(data_len)
        .ok_or_else(|| anyhow::anyhow!("tensor data end overflow"))?;
    anyhow::ensure!(
        data_end <= data.len(),
        "tensor slice data range {data_start}..{data_end} exceeds {}",
        data.len()
    );
    let mut new_shape = shape.to_vec();
    new_shape[0] = len;
    Ok((data[data_start..data_end].to_vec(), new_shape))
}

fn slice_1d<T>(values: &[T], start: usize, len: usize) -> AnyhowResult<&[T]> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("slice range overflow"))?;
    values
        .get(start..end)
        .ok_or_else(|| anyhow::anyhow!("slice range {start}..{end} exceeds {}", values.len()))
}

/// Preprocessed encoder inputs ready for model consumption.
#[derive(Debug, Clone)]
pub struct PreprocessedEncoderInputs {
    /// Primary encoder input as a dynamic-dimensional float32 tensor.
    pub encoder_input: ArrayD<f32>,

    /// Number of encoder feature tokens per media item in the batch.
    pub feature_token_counts: Vec<usize>,

    /// Modality-specific item size metadata before preprocessing.
    ///
    /// The exact tuple order follows each processor/model contract. Auxiliary
    /// shape tensors that need a fixed order should be emitted in
    /// `model_specific`.
    pub item_sizes: Vec<(u32, u32)>,

    /// Model-specific auxiliary outputs.
    pub model_specific: HashMap<String, ModelSpecificValue>,
}

impl PreprocessedEncoderInputs {
    /// Create encoder inputs backed by a tensor of any dimensionality.
    pub fn new<D: Dimension>(
        encoder_input: Array<f32, D>,
        feature_token_counts: Vec<usize>,
        item_sizes: Vec<(u32, u32)>,
    ) -> Self {
        Self {
            encoder_input: encoder_input.into_dyn(),
            feature_token_counts,
            item_sizes,
            model_specific: HashMap::new(),
        }
    }

    /// Add a model-specific value.
    pub fn with_extra(mut self, key: impl Into<String>, value: ModelSpecificValue) -> Self {
        self.model_specific.insert(key.into(), value);
        self
    }

    /// Get the number of media items represented by this preprocessed batch.
    pub fn batch_size(&self) -> usize {
        self.item_sizes.len()
    }

    /// Get the number of dimensions of encoder_input.
    pub fn ndim(&self) -> usize {
        self.encoder_input.ndim()
    }

    /// Get total number of encoder feature tokens across all media items.
    pub fn total_feature_tokens(&self) -> usize {
        self.feature_token_counts.iter().sum()
    }

    /// Get the primary encoder input as a flat f32 slice without copying if possible.
    pub fn encoder_input_flat(&self) -> Cow<'_, [f32]> {
        match self.encoder_input.as_slice() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(self.encoder_input.iter().copied().collect()),
        }
    }

    /// Get the shape of the primary encoder input as a vector.
    pub fn encoder_input_shape(&self) -> Vec<usize> {
        self.encoder_input.shape().to_vec()
    }

    /// Join batches processed one item at a time into the batch a processor
    /// would have produced from all items at once: the encoder input and every
    /// per-item value are stacked along their first dimension. `layouts` names
    /// the values the model reads per item; a value without a layout is still
    /// stacked when it has an item dimension, and must be identical in every
    /// part when it is a scalar.
    pub fn concat(parts: Vec<Self>, layouts: &HashMap<String, FieldLayout>) -> AnyhowResult<Self> {
        anyhow::ensure!(!parts.is_empty(), "cannot join zero batches");
        if parts.len() == 1 {
            return parts.into_iter().next().context("cannot join zero batches");
        }
        let views = parts
            .iter()
            .map(|part| part.encoder_input.view())
            .collect::<Vec<_>>();
        let encoder_input = ndarray::concatenate(Axis(0), &views)
            .context("encoder inputs of the batch parts do not stack")?;

        let keys = parts[0].model_specific.keys().cloned().collect::<Vec<_>>();
        let mut model_specific = HashMap::with_capacity(keys.len());
        for key in keys {
            let values = parts
                .iter()
                .map(|part| {
                    part.model_specific.get(&key).with_context(|| {
                        format!("batch parts disagree on model-specific key {key}")
                    })
                })
                .collect::<AnyhowResult<Vec<_>>>()?;
            let joined = ModelSpecificValue::concat_first_dim(&values)
                .with_context(|| format!("failed to join model-specific value {key}"))?;
            debug_assert!(
                !matches!(
                    layouts.get(&key),
                    Some(FieldLayout::Batched | FieldLayout::Flat { .. })
                ) || !matches!(
                    joined,
                    ModelSpecificValue::Int(_)
                        | ModelSpecificValue::Float(_)
                        | ModelSpecificValue::Bool(_)
                ),
                "a per-item layout was declared for scalar value {key}"
            );
            model_specific.insert(key, joined);
        }
        anyhow::ensure!(
            parts
                .iter()
                .all(|part| part.model_specific.len() == model_specific.len()),
            "batch parts disagree on model-specific keys"
        );

        let mut feature_token_counts = Vec::new();
        let mut item_sizes = Vec::new();
        for part in parts {
            feature_token_counts.extend(part.feature_token_counts);
            item_sizes.extend(part.item_sizes);
        }
        Ok(Self {
            encoder_input,
            feature_token_counts,
            item_sizes,
            model_specific,
        })
    }

    /// Extract batched tensor keys from explicit field layout declarations.
    pub fn batched_keys(layouts: &HashMap<String, FieldLayout>) -> Vec<String> {
        layouts
            .iter()
            .filter(|(_, layout)| matches!(layout, FieldLayout::Batched))
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Extract flat-slicing tensor keys from explicit field layout declarations.
    ///
    /// Returns a map of tensor name to sizes tensor name.
    pub fn flat_keys(layouts: &HashMap<String, FieldLayout>) -> HashMap<String, String> {
        layouts
            .iter()
            .filter_map(|(key, layout)| match layout {
                FieldLayout::Flat { sizes_key } => Some((key.clone(), sizes_key.clone())),
                FieldLayout::Batched => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array4;

    use super::*;

    #[test]
    fn encoder_input_accessors_are_modality_neutral() {
        let inputs = PreprocessedEncoderInputs::new(
            Array4::<f32>::zeros((2, 3, 4, 5)),
            vec![6, 7],
            vec![(4, 5), (8, 9)],
        );

        assert_eq!(inputs.batch_size(), 2);
        assert_eq!(inputs.ndim(), 4);
        assert_eq!(inputs.total_feature_tokens(), 13);
        assert_eq!(inputs.encoder_input_shape(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn encoder_inputs_accept_model_specific_values() {
        let inputs = PreprocessedEncoderInputs::new(
            Array4::<f32>::zeros((1, 3, 224, 224)),
            vec![196],
            vec![(224, 224)],
        )
        .with_extra(
            "image_grid_thw",
            ModelSpecificValue::uint_1d(vec![1, 16, 16]),
        )
        .with_extra("aspect_ratio_id", ModelSpecificValue::Int(0));

        assert!(inputs.model_specific.contains_key("image_grid_thw"));
        assert!(inputs.model_specific.contains_key("aspect_ratio_id"));
    }

    #[test]
    fn model_specific_value_tensor_constructors_set_shapes() {
        assert!(matches!(
            ModelSpecificValue::uint_1d(vec![1, 2, 3]),
            ModelSpecificValue::UintTensor { data, shape }
                if data == vec![1, 2, 3] && shape == vec![3]
        ));
        assert!(matches!(
            ModelSpecificValue::uint_2d(vec![1, 2, 3, 4], 2, 2),
            ModelSpecificValue::UintTensor { data, shape }
                if data == vec![1, 2, 3, 4] && shape == vec![2, 2]
        ));
        assert!(matches!(
            ModelSpecificValue::int_1d(vec![1, 2, 3]),
            ModelSpecificValue::IntTensor { data, shape }
                if data == vec![1, 2, 3] && shape == vec![3]
        ));
        assert!(matches!(
            ModelSpecificValue::int_2d(vec![1, 2, 3, 4], 2, 2),
            ModelSpecificValue::IntTensor { data, shape }
                if data == vec![1, 2, 3, 4] && shape == vec![2, 2]
        ));
    }

    #[test]
    fn encoder_input_flat_preserves_values() {
        let encoder_input = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let inputs = PreprocessedEncoderInputs::new(encoder_input, vec![4], vec![(2, 2)]);

        assert_eq!(inputs.encoder_input_flat(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    fn clip(patches: usize, grid_t: i64, fps_seconds: f32) -> PreprocessedEncoderInputs {
        PreprocessedEncoderInputs::new(
            ndarray::Array2::<f32>::from_elem((patches, 3), patches as f32),
            vec![patches / 4],
            vec![(64, 64)],
        )
        .with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(vec![grid_t, 4, 4], 1, 3),
        )
        .with_extra(
            "patches_per_video",
            ModelSpecificValue::int_1d(vec![patches as i64]),
        )
        .with_extra(
            "video_second_per_grid",
            ModelSpecificValue::Tensor {
                data: vec![fps_seconds],
                shape: vec![1],
            },
        )
        .with_extra("temporal_patch_size", ModelSpecificValue::Int(2))
    }

    #[test]
    fn concat_stacks_clips_the_way_one_batch_would_be_laid_out() {
        let layouts = HashMap::from([
            ("video_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_video".to_string(), FieldLayout::Batched),
        ]);

        let joined =
            PreprocessedEncoderInputs::concat(vec![clip(16, 1, 1.0), clip(32, 2, 0.5)], &layouts)
                .unwrap();

        assert_eq!(joined.encoder_input_shape(), vec![48, 3]);
        // Rows of the first clip come first, then the second clip's.
        assert_eq!(joined.encoder_input_flat()[16 * 3 - 1], 16.0);
        assert_eq!(joined.encoder_input_flat()[16 * 3], 32.0);
        assert_eq!(joined.feature_token_counts, vec![4, 8]);
        assert_eq!(joined.item_sizes, vec![(64, 64), (64, 64)]);
        assert!(matches!(
            &joined.model_specific["video_grid_thw"],
            ModelSpecificValue::IntTensor { data, shape }
                if data == &vec![1, 4, 4, 2, 4, 4] && shape == &vec![2, 3]
        ));
        assert!(matches!(
            &joined.model_specific["patches_per_video"],
            ModelSpecificValue::IntTensor { data, shape } if data == &vec![16, 32] && shape == &vec![2]
        ));
        assert!(matches!(
            &joined.model_specific["video_second_per_grid"],
            ModelSpecificValue::Tensor { data, shape } if data == &vec![1.0, 0.5] && shape == &vec![2]
        ));
        assert!(matches!(
            joined.model_specific["temporal_patch_size"],
            ModelSpecificValue::Int(2)
        ));
    }

    #[test]
    fn concat_keeps_a_single_clip_untouched_and_rejects_mismatched_parts() {
        let layouts = HashMap::new();
        let single = PreprocessedEncoderInputs::concat(vec![clip(16, 1, 1.0)], &layouts).unwrap();
        assert_eq!(single.encoder_input_shape(), vec![16, 3]);

        let mut other_scalar = clip(16, 1, 1.0);
        other_scalar.model_specific.insert(
            "temporal_patch_size".to_string(),
            ModelSpecificValue::Int(3),
        );
        assert!(
            PreprocessedEncoderInputs::concat(vec![clip(16, 1, 1.0), other_scalar], &layouts)
                .is_err()
        );

        let mut extra_key = clip(16, 1, 1.0);
        extra_key
            .model_specific
            .insert("only_here".to_string(), ModelSpecificValue::Bool(true));
        assert!(
            PreprocessedEncoderInputs::concat(vec![clip(16, 1, 1.0), extra_key], &layouts).is_err()
        );

        let mut wrong_width = clip(16, 1, 1.0);
        wrong_width.encoder_input = ndarray::Array2::<f32>::zeros((16, 5)).into_dyn();
        assert!(
            PreprocessedEncoderInputs::concat(vec![clip(16, 1, 1.0), wrong_width], &layouts)
                .is_err()
        );
    }
}
