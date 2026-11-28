use llm_multimodal::{
    f32_to_bf16_bits, f32_to_f16_bits, EncoderInput, PixelNorm, PreprocessedEncoderInputs,
};
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
                scale: 1.0,
                offset: [2.0; 3],
                divisor: [2.0; 3],
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

/// A wire written for another dtype than the worker reads is the typed
/// mismatch, on the owned path and on the per-item path alike.
#[test]
fn a_wire_dtype_mismatch_is_the_typed_error() {
    let norm = PixelNorm {
        scale: 1.0,
        offset: [0.0; 3],
        divisor: [1.0; 3],
        channel_block: 1,
    };
    let wire = EncoderInput::wire(vec![0u8; 12], EncoderDtype::BFloat16, vec![2, 3], norm).unwrap();
    let error = serialize_encoder_input_owned(wire.clone(), "float32").unwrap_err();
    assert!(WireMismatch::is(&error), "{error:#}");
    let error =
        serialize_tokenspeed_encoder_input(&wire.view(), "float32", false, usize::MAX).unwrap_err();
    assert!(WireMismatch::is(&error), "{error:#}");
    assert!(serialize_encoder_input_owned(wire, "bfloat16").is_ok());
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
        let (actual, _, written) =
            serialize_encoder_input_owned(preprocessed.encoder_input.clone(), dtype).unwrap();
        assert_eq!(written, dtype);
        assert_eq!(
            actual.as_slice(),
            le_bytes(values.iter().map(|&value| convert(value)))
        );
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
        let (data, shape, dtype) =
            serialize_encoder_input_owned(preprocessed.encoder_input.clone(), asked).unwrap();

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

    let (data, shape, dtype) =
        serialize_encoder_input_owned(preprocessed.encoder_input.clone(), "uint8").unwrap();
    assert_eq!(
        (data.as_slice(), shape, dtype.as_str()),
        ([0, 255, 51, 102, 153, 204].as_slice(), vec![1, 6], "uint8")
    );

    let (data, _, dtype) =
        serialize_encoder_input_owned(preprocessed.encoder_input.clone(), "float32").unwrap();
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
    )
    .unwrap();
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
        ArrayD::from_shape_vec(IxDyn(&[3, 2]).f(), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
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
        serialize_tokenspeed_encoder_input(&below.encoder_input.view(), "float32", true, 16)
            .unwrap();
    assert!(matches!(tensor.storage, TokenSpeedTensorStorage::Inline(_)));

    // At/above the threshold: uses SHM (/dev/shm is writable on Linux CI).
    let tensor =
        serialize_tokenspeed_encoder_input(&at.encoder_input.view(), "float32", true, 16).unwrap();
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
