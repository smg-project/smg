//! Tensor serialization helpers: encoder input and model-specific values to
//! raw little-endian bytes in the requested wire dtype (f32/bf16/f16).

use std::{collections::HashMap, mem::size_of, time::Instant};

use anyhow::Result;
use llm_multimodal::{ModelSpecificValue, PreprocessedEncoderInputs};
use ndarray::{ArrayD, ArrayViewD, Axis, Slice};
use rayon::prelude::*;
use tracing::{info, warn};

use super::log_mm_timing_enabled;
use crate::routers::grpc::proto_wrapper::{
    write_tokenspeed_shm_with, TensorBytes, TokenSpeedTensor,
};

/// Serialize the primary encoder input ndarray to raw little-endian f32 bytes + shape.
pub(super) fn serialize_encoder_input(
    preprocessed: &PreprocessedEncoderInputs,
) -> (Vec<u8>, Vec<u32>) {
    serialize_array(&preprocessed.encoder_input.view())
}

fn serialize_array(encoder_input: &ArrayViewD<'_, f32>) -> (Vec<u8>, Vec<u32>) {
    let encoder_bytes: Vec<u8> = if let Some(encoder_slice) = encoder_input
        // Fast path only for C-contiguous arrays, whose memory order equals
        // logical (row-major) order. A non-C-contiguous array (e.g. a
        // Fortran-contiguous view) falls through to logical `.iter()` below;
        // `as_slice_memory_order()` is deliberately NOT used as a fallback
        // because it would serialize such arrays in the wrong dimension order.
        .as_slice()
    {
        // Zero-copy reinterpret: &[f32] → &[u8] on little-endian (x86).
        // This replaces the per-element flat_map(to_le_bytes) which was the
        // #1 CPU hotspot (13% of SMG CPU in profiling).
        #[cfg(target_endian = "little")]
        {
            let byte_slice: &[u8] = bytemuck::cast_slice(encoder_slice);
            byte_slice.to_vec()
        }
        #[cfg(not(target_endian = "little"))]
        {
            encoder_slice.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
    } else {
        // Non-C-contiguous array: `.iter()` walks in logical (row-major) order,
        // which matches the shape.
        encoder_input.iter().flat_map(|v| v.to_le_bytes()).collect()
    };
    (encoder_bytes, array_shape(encoder_input))
}

/// Serialize encoder input to the requested wire dtype.
pub(super) fn serialize_array_as_tokenspeed_tensor(
    encoder_input: &ArrayViewD<'_, f32>,
    dtype: &str,
    shm_enabled: bool,
    shm_min_bytes: usize,
) -> TokenSpeedTensor {
    let dtype = match canonical_float_dtype(dtype).as_deref() {
        Some("float32") => "float32".to_string(),
        Some("bfloat16") => "bfloat16".to_string(),
        Some("float16") => "float16".to_string(),
        _ => {
            warn!(
                dtype,
                "Unsupported TokenSpeed encoder input dtype; falling back to float32"
            );
            "float32".to_string()
        }
    };
    let shape = array_shape(encoder_input);
    let element_size = if dtype == "bfloat16" || dtype == "float16" {
        size_of::<u16>()
    } else {
        size_of::<f32>()
    };
    let nbytes = encoder_input.len() * element_size;

    if shm_enabled && nbytes >= shm_min_bytes {
        let started = Instant::now();
        match write_tokenspeed_shm_with(nbytes, |output| {
            fill_array_as_dtype(output, encoder_input, &dtype)
        }) {
            Ok(handle) => {
                if log_mm_timing_enabled() {
                    info!(
                        nbytes,
                        elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
                        "smg_mm_timing tokenspeed_shm_write_direct"
                    );
                }
                return TokenSpeedTensor::shm(handle, shape, dtype);
            }
            Err(error) => {
                use crate::observability::metrics::Metrics;
                warn!(
                    ?error,
                    nbytes,
                    dtype = %dtype,
                    "Failed to write TokenSpeed encoder input directly to SHM; falling back to bytes path"
                );
                Metrics::record_mm_shm_write_failure("tokenspeed");
            }
        }
    }

    let (data, shape, dtype) = serialize_array_as_dtype(encoder_input, &dtype);
    TokenSpeedTensor::inline(data, shape, dtype)
}

fn fill_array_as_dtype(
    output: &mut [u8],
    encoder_input: &ArrayViewD<'_, f32>,
    dtype: &str,
) -> std::io::Result<()> {
    let element_size = if dtype == "bfloat16" || dtype == "float16" {
        size_of::<u16>()
    } else {
        size_of::<f32>()
    };
    if output.len() != encoder_input.len() * element_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "encoder input output buffer has an unexpected byte length",
        ));
    }

    match dtype {
        "float32" => {
            fill_array_as_f32_bytes(output, encoder_input);
            Ok(())
        }
        "bfloat16" => {
            fill_array_as_u16_bytes(output, encoder_input, f32_to_bf16_bits);
            Ok(())
        }
        "float16" => {
            fill_array_as_u16_bytes(output, encoder_input, f32_to_f16_bits);
            Ok(())
        }
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unsupported TokenSpeed encoder input dtype: {other}"),
        )),
    }
}

fn fill_array_as_f32_bytes(output: &mut [u8], encoder_input: &ArrayViewD<'_, f32>) {
    if let Some(encoder_slice) = encoder_input
        // Fast path only for C-contiguous arrays, whose memory order equals
        // logical (row-major) order. A non-C-contiguous array (e.g. a
        // Fortran-contiguous view) falls through to logical `.iter()` below;
        // `as_slice_memory_order()` is deliberately NOT used as a fallback
        // because it would serialize such arrays in the wrong dimension order.
        .as_slice()
    {
        #[cfg(target_endian = "little")]
        output.copy_from_slice(bytemuck::cast_slice(encoder_slice));
        #[cfg(not(target_endian = "little"))]
        fill_f32_values_as_bytes(output, encoder_slice.iter().copied());
        return;
    }

    fill_f32_values_as_bytes(output, encoder_input.iter().copied());
}

fn fill_f32_values_as_bytes(output: &mut [u8], values: impl IntoIterator<Item = f32>) {
    for (output, value) in output
        .as_chunks_mut::<{ size_of::<f32>() }>()
        .0
        .iter_mut()
        .zip(values)
    {
        output.copy_from_slice(&value.to_le_bytes());
    }
}

fn fill_array_as_u16_bytes<F>(output: &mut [u8], encoder_input: &ArrayViewD<'_, f32>, convert: F)
where
    F: Fn(f32) -> u16 + Copy + Send + Sync,
{
    if let Some(encoder_slice) = encoder_input
        // Fast path only for C-contiguous arrays, whose memory order equals
        // logical (row-major) order. A non-C-contiguous array (e.g. a
        // Fortran-contiguous view) falls through to logical `.iter()` below;
        // `as_slice_memory_order()` is deliberately NOT used as a fallback
        // because it would serialize such arrays in the wrong dimension order.
        .as_slice()
    {
        fill_f32_slice_as_u16_bytes(output, encoder_slice, convert);
    } else {
        fill_f32_values_as_u16_bytes(output, encoder_input.iter().copied(), convert);
    }
}

fn serialize_array_as_dtype(
    encoder_input: &ArrayViewD<'_, f32>,
    dtype: &str,
) -> (Vec<u8>, Vec<u32>, String) {
    match canonical_float_dtype(dtype).as_deref() {
        Some("float32") => {
            let (data, shape) = serialize_array(encoder_input);
            (data, shape, "float32".to_string())
        }
        Some("bfloat16") => (
            serialize_array_as_u16_bytes(encoder_input, f32_to_bf16_bits),
            array_shape(encoder_input),
            "bfloat16".to_string(),
        ),
        Some("float16") => (
            serialize_array_as_u16_bytes(encoder_input, f32_to_f16_bits),
            array_shape(encoder_input),
            "float16".to_string(),
        ),
        _ => {
            warn!(
                dtype,
                "Unsupported TokenSpeed encoder input dtype; falling back to float32"
            );
            let (data, shape) = serialize_array(encoder_input);
            (data, shape, "float32".to_string())
        }
    }
}

fn serialize_array_as_u16_bytes<F>(encoder_input: &ArrayViewD<'_, f32>, convert: F) -> Vec<u8>
where
    F: Fn(f32) -> u16 + Copy + Send + Sync,
{
    let element_count = encoder_input.len();
    let mut bytes = vec![0u8; element_count * size_of::<u16>()];
    fill_array_as_u16_bytes(&mut bytes, encoder_input, convert);
    bytes
}

fn fill_f32_slice_as_u16_bytes<F>(bytes: &mut [u8], values: &[f32], convert: F)
where
    F: Fn(f32) -> u16 + Copy + Send + Sync,
{
    debug_assert_eq!(bytes.len(), values.len() * size_of::<u16>());
    const MIN_OUTPUT_BYTES: usize = 1 << 19;
    const MIN_VALUES_PER_TASK: usize = 32;
    const MAX_TASKS: usize = 8;
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let tasks = if bytes.len() < MIN_OUTPUT_BYTES {
        1
    } else {
        (values.len() / MIN_VALUES_PER_TASK)
            .min(available)
            .clamp(1, MAX_TASKS)
    };
    if tasks == 1 {
        fill_f32_values_as_u16_bytes(bytes, values.iter().copied(), convert);
        return;
    }

    let chunk_values = values.len().div_ceil(tasks);
    bytes
        .par_chunks_mut(chunk_values * size_of::<u16>())
        .zip(values.par_chunks(chunk_values))
        .for_each(|(output, values)| {
            fill_f32_values_as_u16_bytes(output, values.iter().copied(), convert);
        });
}

fn fill_f32_values_as_u16_bytes<I, F>(bytes: &mut [u8], values: I, convert: F)
where
    I: IntoIterator<Item = f32>,
    F: Fn(f32) -> u16 + Copy,
{
    for (output, value) in bytes
        .as_chunks_mut::<{ size_of::<u16>() }>()
        .0
        .iter_mut()
        .zip(values)
    {
        output.copy_from_slice(&convert(value).to_le_bytes());
    }
}

fn canonical_float_dtype(dtype: &str) -> Option<String> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "float32" | "fp32" | "f32" => Some("float32".to_string()),
        "bfloat16" | "bf16" => Some("bfloat16".to_string()),
        "float16" | "fp16" | "f16" | "half" => Some("float16".to_string()),
        _ => None,
    }
}

fn array_shape(encoder_input: &ArrayViewD<'_, f32>) -> Vec<u32> {
    encoder_input.shape().iter().map(|&d| d as u32).collect()
}

#[inline]
fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

#[inline]
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 0xff {
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x800000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = (mantissa >> shift) as u16;
        let round_bit = (mantissa >> (shift - 1)) & 1;
        let sticky = mantissa & ((1u32 << (shift - 1)) - 1);
        if round_bit != 0 && (sticky != 0 || (half_mant & 1) != 0) {
            half_mant += 1;
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    let round = mant & 0x1fff;
    if round > 0x1000 || (round == 0x1000 && (half & 1) != 0) {
        half += 1;
    }
    half
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

pub(super) fn slice_array_axis0(
    array: &ArrayD<f32>,
    start: usize,
    len: usize,
) -> Result<ArrayViewD<'_, f32>> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("array slice range overflow"))?;
    let rows = array.shape().first().copied().unwrap_or(0);
    anyhow::ensure!(
        end <= rows,
        "array first-dimension slice {start}..{end} exceeds {rows}"
    );
    Ok(array.slice_axis(Axis(0), Slice::from(start..end)))
}

#[cfg(test)]
mod tests {
    use ndarray::{IxDyn, ShapeBuilder};

    use super::*;

    #[test]
    fn parallel_u16_serialization_matches_scalar_conversion() {
        let values: Vec<f32> = (0..300_000)
            .map(|index| (index as f32 - 150_000.0) / 257.0)
            .collect();
        let array = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.clone()).unwrap();

        for (dtype, convert) in [
            ("bfloat16", f32_to_bf16_bits as fn(f32) -> u16),
            ("float16", f32_to_f16_bits as fn(f32) -> u16),
        ] {
            let actual = serialize_array_as_u16_bytes(&array.view(), convert);
            let expected: Vec<u8> = values
                .iter()
                .flat_map(|&value| convert(value).to_le_bytes())
                .collect();
            assert_eq!(actual, expected);

            let mut direct = vec![0; expected.len()];
            fill_array_as_dtype(&mut direct, &array.view(), dtype).unwrap();
            assert_eq!(direct, expected);
        }
    }

    #[test]
    fn encoder_input_slice_is_borrowed_and_serializes_in_logical_order() {
        let array =
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let item = slice_array_axis0(&array, 1, 1).unwrap();

        assert_eq!(item.as_ptr(), array.as_ptr().wrapping_add(2));
        assert_eq!(item.shape(), &[1, 2]);
        let expected = [3.0_f32, 4.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(serialize_array(&item), (expected, vec![1, 2]));

        let fortran_array =
            ArrayD::from_shape_vec(IxDyn(&[3, 2]).f(), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let fortran_item = slice_array_axis0(&fortran_array, 1, 1).unwrap();
        let expected: Vec<u8> = [2.0_f32, 5.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert!(fortran_item.as_slice().is_none());
        let mut direct = vec![0; expected.len()];
        fill_array_as_dtype(&mut direct, &fortran_item, "float32").unwrap();
        assert_eq!(direct, expected);
        assert_eq!(serialize_array(&fortran_item), (expected, vec![1, 2]));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn shm_min_bytes_threshold_gates_inline_vs_shm() {
        use std::path::Path;

        use crate::routers::grpc::proto_wrapper::TokenSpeedTensorStorage;

        // float32 = 4 bytes/elem; threshold of 16 bytes falls at 4 elements.
        let below = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0_f32, 2.0, 3.0]).unwrap();
        let at = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();

        // Below the threshold: stays inline even with SHM enabled.
        let tensor = serialize_array_as_tokenspeed_tensor(&below.view(), "float32", true, 16);
        assert!(matches!(tensor.storage, TokenSpeedTensorStorage::Inline(_)));

        // At/above the threshold: uses SHM (/dev/shm is writable on Linux CI).
        let tensor = serialize_array_as_tokenspeed_tensor(&at.view(), "float32", true, 16);
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
