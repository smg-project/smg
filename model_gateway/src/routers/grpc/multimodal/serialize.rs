//! Tensor serialization helpers: the encoder input and model-specific values
//! to raw little-endian bytes in the requested wire dtype.

use std::{collections::HashMap, time::Instant};

use llm_multimodal::{EncoderDtype, EncoderInput, EncoderInputView, ModelSpecificValue, WireBytes};
use tracing::{info, warn};

use super::log_mm_timing_enabled;
use crate::routers::grpc::proto_wrapper::{
    write_tokenspeed_shm_with, TensorBytes, TokenSpeedTensor,
};

/// Serialize the primary encoder input to raw little-endian bytes in `dtype`,
/// returning the shape and the dtype the bytes were actually written in,
/// which is what the receiver must be told to read them at. Bytes the
/// preprocessor already laid out in `dtype` move into the result.
pub(super) fn serialize_encoder_input_owned(
    encoder_input: EncoderInput,
    dtype: &str,
) -> anyhow::Result<(WireBytes, Vec<u32>, String)> {
    let dtype = wire_dtype(dtype, true);
    let shape = shape_u32(&encoder_input.view());
    let bytes = encoder_input
        .into_wire_bytes_as(dtype)
        .map_err(|message| anyhow::Error::new(WireMismatch(message)))?;
    Ok((bytes, shape, dtype.to_string()))
}

/// An encoder input laid out for one wire dtype and asked for in another at
/// assembly: the dtype was resolved over the pool the selection draws from, so
/// only a change of that pool between the two steps produces it. It is the
/// gateway's own transient error, not the worker's and not the client's: the
/// request builders answer it with a retryable status.
#[derive(Debug)]
pub(crate) struct WireMismatch(String);

impl std::fmt::Display for WireMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the images were prepared for another encoder dtype than the selected worker reads: {}",
            self.0
        )
    }
}

impl std::error::Error for WireMismatch {}

impl WireMismatch {
    /// Whether an assembly error is a wire mismatch (anywhere in its chain).
    pub(crate) fn is(error: &anyhow::Error) -> bool {
        error.chain().any(|cause| cause.is::<WireMismatch>())
    }
}

/// One item's encoder input for TokenSpeed: written straight into shared
/// memory when enabled and large enough, inline otherwise.
pub(super) fn serialize_tokenspeed_encoder_input(
    input: &EncoderInputView<'_>,
    dtype: &str,
    shm_enabled: bool,
    shm_min_bytes: usize,
) -> anyhow::Result<TokenSpeedTensor> {
    let (dtype, shape) = tokenspeed_wire_shape(input, dtype)?;
    if let Some(tensor) = tokenspeed_shm_tensor(input, dtype, &shape, shm_enabled, shm_min_bytes) {
        return Ok(tensor);
    }
    let bytes = input
        .try_write_as(dtype)
        .map_err(|message| anyhow::Error::new(WireMismatch(message)))?;
    Ok(TokenSpeedTensor::inline(bytes, shape, dtype.to_string()))
}

/// [`serialize_tokenspeed_encoder_input`] for an input the caller hands over
/// whole: bytes already in the wire dtype move into the payload instead of
/// being written again.
pub(super) fn serialize_tokenspeed_encoder_input_owned(
    input: EncoderInput,
    dtype: &str,
    shm_enabled: bool,
    shm_min_bytes: usize,
) -> anyhow::Result<TokenSpeedTensor> {
    let view = input.view();
    let (dtype, shape) = match tokenspeed_wire_shape(&view, dtype) {
        Ok(wire) => wire,
        Err(error) => {
            recycle_encoder_input(input);
            return Err(error);
        }
    };
    if let Some(tensor) = tokenspeed_shm_tensor(&view, dtype, &shape, shm_enabled, shm_min_bytes) {
        recycle_encoder_input(input);
        return Ok(tensor);
    }
    let bytes = input
        .into_wire_bytes_as(dtype)
        .map_err(|message| anyhow::Error::new(WireMismatch(message)))?;
    Ok(TokenSpeedTensor::inline(bytes, shape, dtype.to_string()))
}

/// An encoder input whose bytes went elsewhere (written from a view, or into
/// shared memory) or are left out of the request: its wire bytes or patches
/// go back to the buffer pool, where a drop would free them.
pub(super) fn recycle_encoder_input(input: EncoderInput) {
    match input {
        EncoderInput::Wire { bytes, .. } => llm_multimodal::buffer_pool::recycle(bytes),
        EncoderInput::U8 { patches, .. } => {
            llm_multimodal::buffer_pool::recycle(patches.into_raw_vec_and_offset().0);
        }
        EncoderInput::F32(_) | EncoderInput::SharedWire { .. } => {}
    }
}

/// The dtype TokenSpeed reads for `dtype` and the input's shape in it,
/// checked before any segment is created for it.
fn tokenspeed_wire_shape(
    input: &EncoderInputView<'_>,
    dtype: &str,
) -> anyhow::Result<(EncoderDtype, Vec<u32>)> {
    // TokenSpeed reads float widths only.
    let dtype = wire_dtype(dtype, false);
    input
        .writable_as(dtype)
        .map_err(|message| anyhow::Error::new(WireMismatch(message)))?;
    Ok((dtype, shape_u32(input)))
}

/// The input written straight into a shared-memory segment when the transport
/// takes one of its size; `None` leaves it to the bytes path.
fn tokenspeed_shm_tensor(
    input: &EncoderInputView<'_>,
    dtype: EncoderDtype,
    shape: &[u32],
    shm_enabled: bool,
    shm_min_bytes: usize,
) -> Option<TokenSpeedTensor> {
    let nbytes = input.len() * dtype.element_size();
    if !shm_enabled || nbytes < shm_min_bytes {
        return None;
    }
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
            Some(TokenSpeedTensor::shm(
                handle,
                shape.to_vec(),
                dtype.to_string(),
            ))
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
            None
        }
    }
}

/// The dtype `asked` names, or `float32` for one the receiver may not share:
/// a name it does not know, and `uint8` where only float widths are read.
pub(super) fn wire_dtype(asked: &str, uint8_allowed: bool) -> EncoderDtype {
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
#[path = "../../../../tests/unit/routers/grpc/multimodal/serialize/tests.rs"]
mod tests;
