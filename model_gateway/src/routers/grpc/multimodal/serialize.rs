//! Tensor serialization helpers: the encoder input and model-specific values
//! to raw little-endian bytes in the requested wire dtype.

use std::{collections::HashMap, time::Instant};

use llm_multimodal::{
    EncoderDtype, EncoderInputView, ModelSpecificValue, PreprocessedEncoderInputs,
};
use tracing::{info, warn};

use super::log_mm_timing_enabled;
use crate::routers::grpc::proto_wrapper::{
    write_tokenspeed_shm_with, TensorBytes, TokenSpeedTensor,
};

/// Serialize the primary encoder input to raw little-endian bytes in `dtype`,
/// returning the shape and the dtype the bytes were actually written in,
/// which is what the receiver must be told to read them at.
pub(super) fn serialize_encoder_input(
    preprocessed: &PreprocessedEncoderInputs,
    dtype: &str,
) -> (Vec<u8>, Vec<u32>, String) {
    let input = preprocessed.encoder_input.view();
    let dtype = wire_dtype(dtype, true);
    (input.write_as(dtype), shape_u32(&input), dtype.to_string())
}

/// One item's encoder input for TokenSpeed: written straight into shared
/// memory when enabled and large enough, inline otherwise.
pub(super) fn serialize_tokenspeed_encoder_input(
    input: &EncoderInputView<'_>,
    dtype: &str,
    shm_enabled: bool,
    shm_min_bytes: usize,
) -> TokenSpeedTensor {
    // TokenSpeed reads float widths only.
    let dtype = wire_dtype(dtype, false);
    let shape = shape_u32(input);
    let nbytes = input.len() * dtype.element_size();

    if shm_enabled && nbytes >= shm_min_bytes {
        let started = Instant::now();
        match write_tokenspeed_shm_with(nbytes, |output| {
            input
                .write_into(dtype, output)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
        }) {
            Ok(handle) => {
                if log_mm_timing_enabled() {
                    info!(
                        nbytes,
                        elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
                        "smg_mm_timing tokenspeed_shm_write_direct"
                    );
                }
                return TokenSpeedTensor::shm(handle, shape, dtype.to_string());
            }
            Err(error) => {
                use crate::observability::metrics::Metrics;
                warn!(
                    ?error,
                    nbytes,
                    %dtype,
                    "Failed to write TokenSpeed encoder input directly to SHM; falling back to bytes path"
                );
                Metrics::record_mm_shm_write_failure("tokenspeed");
            }
        }
    }

    TokenSpeedTensor::inline(input.write_as(dtype), shape, dtype.to_string())
}

/// The dtype `asked` names, or `float32` for one the receiver may not share:
/// a name it does not know, and `uint8` where only float widths are read.
fn wire_dtype(asked: &str, uint8_allowed: bool) -> EncoderDtype {
    match asked.parse::<EncoderDtype>() {
        Ok(EncoderDtype::Uint8) if !uint8_allowed => {
            warn!(
                dtype = asked,
                "TokenSpeed encoder inputs are floats; writing float32 instead of uint8"
            );
            EncoderDtype::Float32
        }
        Ok(dtype) => dtype,
        Err(_) => {
            warn!(
                dtype = asked,
                "Unsupported encoder input dtype; falling back to float32"
            );
            EncoderDtype::Float32
        }
    }
}

fn shape_u32(input: &EncoderInputView<'_>) -> Vec<u32> {
    input.shape().iter().map(|&d| d as u32).collect()
}

/// Serialize model-specific values to TensorBytes, consuming the map to avoid key clones.
pub(super) fn serialize_model_specific(
    model_specific: HashMap<String, ModelSpecificValue>,
) -> HashMap<String, TensorBytes> {
    model_specific
        .into_iter()
        .filter_map(|(key, value)| match model_specific_to_tensor_bytes(&value) {
            Some(tensor) => Some((key, tensor)),
            None => {
                warn!(tensor_key = %key, "Dropping unsupported model_specific value during multimodal serialization");
                None
            }
        })
        .collect()
}

/// Convert a model-specific value to backend-agnostic TensorBytes.
pub(super) fn model_specific_to_tensor_bytes(value: &ModelSpecificValue) -> Option<TensorBytes> {
    match value {
        ModelSpecificValue::Tensor { data, shape } => Some(TensorBytes {
            data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.iter().map(|&d| d as u32).collect(),
            dtype: "float32".to_string(),
        }),
        ModelSpecificValue::IntTensor { data, shape } => Some(TensorBytes {
            data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.iter().map(|&d| d as u32).collect(),
            dtype: "int64".to_string(),
        }),
        ModelSpecificValue::UintTensor { data, shape } => Some(TensorBytes {
            data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.iter().map(|&d| d as u32).collect(),
            dtype: "uint32".to_string(),
        }),
        ModelSpecificValue::UintVec(v) => Some(TensorBytes {
            data: v.iter().flat_map(|val| val.to_le_bytes()).collect(),
            shape: vec![v.len() as u32],
            dtype: "uint32".to_string(),
        }),
        ModelSpecificValue::IntVec(v) => Some(TensorBytes {
            data: v.iter().flat_map(|val| val.to_le_bytes()).collect(),
            shape: vec![v.len() as u32],
            dtype: "int64".to_string(),
        }),
        ModelSpecificValue::FloatVec(v) => Some(TensorBytes {
            data: v.iter().flat_map(|val| val.to_le_bytes()).collect(),
            shape: vec![v.len() as u32],
            dtype: "float32".to_string(),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use llm_multimodal::{f32_to_bf16_bits, f32_to_f16_bits, EncoderInput, PixelNorm};
    use ndarray::{ArrayD, IxDyn, ShapeBuilder};

    use super::*;

    fn floats(shape: &[usize], values: Vec<f32>) -> PreprocessedEncoderInputs {
        PreprocessedEncoderInputs::new(
            ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// Three 2-element channel blocks of pixel bytes with a normalization
    /// (`v / 2 - 1`) every byte maps through exactly in `f32`.
    fn pixel_bytes() -> PreprocessedEncoderInputs {
        PreprocessedEncoderInputs::new(
            EncoderInput::bytes(
                ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![0u8, 255, 51, 102, 153, 204]).unwrap(),
                PixelNorm {
                    scale: [0.5; 3],
                    bias: [-1.0; 3],
                    channel_block: 2,
                },
            ),
            Vec::new(),
            Vec::new(),
        )
    }

    fn le_bytes<T: IntoIterator<Item = u16>>(values: T) -> Vec<u8> {
        values.into_iter().flat_map(u16::to_le_bytes).collect()
    }

    #[test]
    fn narrow_dtypes_round_like_the_scalar_conversion() {
        let values: Vec<f32> = (0..300_000)
            .map(|index| (index as f32 - 150_000.0) / 257.0)
            .collect();
        let preprocessed = floats(&[values.len()], values.clone());

        for (dtype, convert) in [
            ("bfloat16", f32_to_bf16_bits as fn(f32) -> u16),
            ("float16", f32_to_f16_bits as fn(f32) -> u16),
        ] {
            let (actual, _, written) = serialize_encoder_input(&preprocessed, dtype);
            assert_eq!(written, dtype);
            assert_eq!(actual, le_bytes(values.iter().map(|&value| convert(value))));
        }
    }

    /// Every wire writes its encoder input through this one function, so the
    /// dtype it reports has to be the dtype it wrote: the receiver has
    /// nothing else to read the bytes by.
    #[test]
    fn serialized_bytes_and_reported_dtype_agree() {
        let preprocessed = floats(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);

        for (asked, expected_dtype, element_size) in [
            ("float32", "float32", 4),
            ("bfloat16", "bfloat16", 2),
            ("float16", "float16", 2),
            ("uint8", "uint8", 1),
            // Nothing sensible can be written for a name the receiver may not
            // share, so the widest dtype is used and reported as such.
            ("float64", "float32", 4),
        ] {
            let (data, shape, dtype) = serialize_encoder_input(&preprocessed, asked);

            assert_eq!(dtype, expected_dtype, "asked for {asked}");
            assert_eq!(shape, vec![2, 2]);
            assert_eq!(data.len(), 4 * element_size, "asked for {asked}");
        }
    }

    /// Pixel bytes are the wire's `uint8` as they are, and are normalized on
    /// the way to a float dtype; TokenSpeed, which reads floats only, gets
    /// them normalized even when `uint8` is asked for.
    #[test]
    fn pixel_bytes_are_raw_for_uint8_and_normalized_for_floats() {
        let preprocessed = pixel_bytes();
        let normalized: Vec<f32> = [0u8, 255, 51, 102, 153, 204]
            .iter()
            .map(|&v| v as f32 * 0.5 - 1.0)
            .collect();

        let (data, shape, dtype) = serialize_encoder_input(&preprocessed, "uint8");
        assert_eq!(
            (data, shape, dtype.as_str()),
            (vec![0, 255, 51, 102, 153, 204], vec![1, 6], "uint8")
        );

        let (data, _, dtype) = serialize_encoder_input(&preprocessed, "float32");
        assert_eq!(dtype, "float32");
        let floats: Vec<f32> = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(floats, normalized);

        let tensor = serialize_tokenspeed_encoder_input(
            &preprocessed.encoder_input.view(),
            "uint8",
            false,
            usize::MAX,
        );
        assert_eq!(tensor.dtype, "float32");
        assert_eq!(tensor.shape, vec![1, 6]);
    }

    /// An item sliced out of a batch is written in logical (row-major) order
    /// whatever the batch's memory layout.
    #[test]
    fn item_slices_serialize_in_logical_order() {
        let c_order = floats(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let item = c_order.encoder_input.view().slice_axis0(1, 1).unwrap();
        assert_eq!(item.shape(), &[1, 2]);
        let expected: Vec<u8> = [3.0_f32, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(item.write_as(EncoderDtype::Float32), expected);

        let fortran = PreprocessedEncoderInputs::new(
            ArrayD::from_shape_vec(IxDyn(&[3, 2]).f(), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap(),
            Vec::new(),
            Vec::new(),
        );
        let item = fortran.encoder_input.view().slice_axis0(1, 1).unwrap();
        let expected: Vec<u8> = [2.0_f32, 5.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(item.write_as(EncoderDtype::Float32), expected);
        let mut direct = vec![0; expected.len()];
        item.write_into(EncoderDtype::Float32, &mut direct).unwrap();
        assert_eq!(direct, expected);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn shm_min_bytes_threshold_gates_inline_vs_shm() {
        use std::path::Path;

        use crate::routers::grpc::proto_wrapper::TokenSpeedTensorStorage;

        // float32 = 4 bytes/elem; threshold of 16 bytes falls at 4 elements.
        let below = floats(&[3], vec![1.0, 2.0, 3.0]);
        let at = floats(&[4], vec![1.0, 2.0, 3.0, 4.0]);

        // Below the threshold: stays inline even with SHM enabled.
        let tensor =
            serialize_tokenspeed_encoder_input(&below.encoder_input.view(), "float32", true, 16);
        assert!(matches!(tensor.storage, TokenSpeedTensorStorage::Inline(_)));

        // At/above the threshold: uses SHM (/dev/shm is writable on Linux CI).
        let tensor =
            serialize_tokenspeed_encoder_input(&at.encoder_input.view(), "float32", true, 16);
        match tensor.storage {
            TokenSpeedTensorStorage::Shm(handle) => {
                let path = Path::new("/dev/shm").join(&handle.name);
                assert!(path.exists());
                let _ = std::fs::remove_file(&path);
            }
            TokenSpeedTensorStorage::Inline(_) => {
                panic!("expected SHM at/above the threshold when /dev/shm is writable")
            }
            TokenSpeedTensorStorage::Remote(_) => {
                panic!("unexpected remote payload in the SHM threshold test")
            }
        }
    }
}
