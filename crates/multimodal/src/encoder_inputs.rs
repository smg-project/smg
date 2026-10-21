//! Shared encoder-input types for all encoder-backed modalities.
//!
//! The primary encoder input comes in two forms. Most processors produce it
//! as `f32`, normalized. The processors that patchify pixels (the Qwen-VL
//! family) produce the image's own bytes in the model's patch layout plus the
//! normalization a float destination applies ([`EncoderInput::U8`]): the
//! destination's dtype, or no normalization at all for an engine that
//! normalizes on device, is then one pass at serialization
//! ([`EncoderInput::write_as`]), and a cache of preprocessed inputs holds
//! bytes, a quarter of the floats, valid for every destination.

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt,
    mem::{size_of, size_of_val},
    str::FromStr,
};

use anyhow::{Context, Result as AnyhowResult};
use ndarray::{Array, ArrayD, ArrayViewD, Axis, Dimension, IxDyn, Slice};

use crate::{pixel_convert, types::FieldLayout, vision::execution};

/// The dtypes an encoder input is written in for the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderDtype {
    Float32,
    BFloat16,
    Float16,
    /// The image's own bytes, for an engine that rescales and normalizes on
    /// device. Only an [`EncoderInput::U8`] can be written this way exactly.
    Uint8,
}

impl EncoderDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::BFloat16 => "bfloat16",
            Self::Float16 => "float16",
            Self::Uint8 => "uint8",
        }
    }

    pub fn element_size(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::BFloat16 | Self::Float16 => 2,
            Self::Uint8 => 1,
        }
    }
}

impl fmt::Display for EncoderDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EncoderDtype {
    type Err = String;

    /// The wire names and their common spellings (`fp32`, `bf16`, `half`, ...).
    fn from_str(dtype: &str) -> Result<Self, Self::Err> {
        match dtype.trim().to_ascii_lowercase().as_str() {
            "float32" | "fp32" | "f32" => Ok(Self::Float32),
            "bfloat16" | "bf16" => Ok(Self::BFloat16),
            "float16" | "fp16" | "f16" | "half" => Ok(Self::Float16),
            "uint8" | "u8" => Ok(Self::Uint8),
            other => Err(format!("unsupported encoder input dtype {other:?}")),
        }
    }
}

/// Round an `f32` to the nearest `bfloat16` (ties to even), as the bits.
#[inline]
pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Round an `f32` to the nearest `float16` (ties to even), as the bits.
#[inline]
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 0xff {
        // Infinity or NaN; keep a NaN a NaN.
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -25 {
        return sign;
    }
    if unbiased < -14 {
        // Subnormal: shift the implicit one into the mantissa.
        let full = mant | 0x800000;
        let shift = (-14 - unbiased) as u32;
        let shifted = full >> (13 + shift);
        let remainder = full & ((1u32 << (13 + shift)) - 1);
        let halfway = 1u32 << (12 + shift);
        let rounded = match remainder.cmp(&halfway) {
            std::cmp::Ordering::Greater => shifted + 1,
            std::cmp::Ordering::Equal => shifted + (shifted & 1),
            std::cmp::Ordering::Less => shifted,
        };
        return sign | rounded as u16;
    }
    let half_exp = ((unbiased + 15) as u32) << 10;
    let half_mant = mant >> 13;
    let remainder = mant & 0x1fff;
    let mut result = half_exp | half_mant;
    if remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1) == 1) {
        result += 1;
    }
    sign | result as u16
}

/// The per-channel map from a pixel byte to the normalized value a float
/// destination takes: `(value * scale - offset[c]) / divisor[c]` in `f32`,
/// the arithmetic of the reference image processors (the torchvision-backed
/// ones the serving engines run). They fuse rescale and normalize by scaling
/// the mean and std into byte units and then subtract once and divide once,
/// so a byte maps to the same `f32` here as there, and so to the same
/// narrower float when the engine casts it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelNorm {
    /// Multiplies the byte first: the rescale factor when the bytes are
    /// rescaled but not normalized, else 1.
    pub scale: f32,
    /// Subtracted next, per channel: the mean, in byte units when the bytes
    /// are rescaled as well (`f32(mean) * f32(1 / rescale_factor)`, the
    /// reference's own rounding, as `pixel_norm` in the Qwen base builds it);
    /// 0 without normalization.
    pub offset: [f32; 3],
    /// Divides last, per channel: the std in the same units and the same
    /// rounding as `offset`; 1 without normalization.
    pub divisor: [f32; 3],
    /// Consecutive elements of one channel within a row of the encoder
    /// input: for a patch layout `[C, temporal, patch_h, patch_w]` the
    /// temporal × patch_h × patch_w product. Channel `c` owns elements
    /// `[c * channel_block, (c + 1) * channel_block)` of every row.
    pub channel_block: usize,
}

impl PixelNorm {
    /// The `f32` each byte maps to, per channel.
    pub fn table_f32(&self) -> [[f32; 256]; 3] {
        std::array::from_fn(|channel| {
            std::array::from_fn(|value| {
                (value as f32 * self.scale - self.offset[channel]) / self.divisor[channel]
            })
        })
    }

    fn table_u16(&self, convert: fn(f32) -> u16) -> [[u16; 256]; 3] {
        let table = self.table_f32();
        std::array::from_fn(|channel| std::array::from_fn(|value| convert(table[channel][value])))
    }
}

/// The primary encoder input, in the form its processor produced it.
#[derive(Debug, Clone, PartialEq)]
pub enum EncoderInput {
    /// Normalized floats, the form of most processors.
    F32(ArrayD<f32>),
    /// The image's own bytes in the model's patch layout (rows of
    /// `channel_block × 3` elements per channel block), with the
    /// normalization a float destination applies.
    U8 {
        patches: ArrayD<u8>,
        norm: PixelNorm,
    },
    /// The encoder input already in a wire dtype: `shape`'s elements as
    /// little-endian `dtype` values, written from pixel bytes with `norm` in
    /// one pass (the patch layout and the normalization together, for a
    /// destination whose dtype was known before preprocessing). A
    /// destination of that dtype reads it as it is; a narrower float dtype
    /// is derived exactly from a `float32` wire; nothing else is.
    Wire {
        bytes: Vec<u8>,
        dtype: EncoderDtype,
        shape: Vec<usize>,
        norm: PixelNorm,
    },
}

impl<D: Dimension> From<Array<f32, D>> for EncoderInput {
    fn from(array: Array<f32, D>) -> Self {
        Self::F32(array.into_dyn())
    }
}

impl EncoderInput {
    /// Pixel bytes in the model's patch layout with their normalization.
    pub fn bytes<D: Dimension>(patches: Array<u8, D>, norm: PixelNorm) -> Self {
        Self::U8 {
            patches: patches.into_dyn(),
            norm,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(array) => array.shape(),
            Self::U8 { patches, .. } => patches.shape(),
            Self::Wire { shape, .. } => shape,
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Elements in all.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A [`Wire`](Self::Wire) whose byte count is checked against its shape
    /// and dtype, the invariant every reader of the variant relies on.
    pub fn wire(
        bytes: Vec<u8>,
        dtype: EncoderDtype,
        shape: Vec<usize>,
        norm: PixelNorm,
    ) -> Result<Self, String> {
        let expected = shape.iter().product::<usize>() * dtype.element_size();
        if bytes.len() != expected {
            return Err(format!(
                "wire holds {} bytes, {expected} needed for {shape:?} in {dtype}",
                bytes.len()
            ));
        }
        Ok(Self::Wire {
            bytes,
            dtype,
            shape,
            norm,
        })
    }

    /// Bytes per element as stored.
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32(_) => size_of::<f32>(),
            Self::U8 { .. } => size_of::<u8>(),
            Self::Wire { dtype, .. } => dtype.element_size(),
        }
    }

    /// Bytes as stored.
    pub fn nbytes(&self) -> usize {
        self.len() * self.element_size()
    }

    /// The normalization bytes carry; `None` for floats, already normalized.
    pub fn pixel_norm(&self) -> Option<PixelNorm> {
        match self {
            Self::F32(_) => None,
            Self::U8 { norm, .. } | Self::Wire { norm, .. } => Some(*norm),
        }
    }

    /// The input as normalized `f32`: borrowed when it is one, materialized
    /// through the normalization table when it is bytes (the same values the
    /// processors produced when they normalized while patchifying), and the
    /// wire's own values widened when it is already a wire (exact for a
    /// `float32` wire; a narrower wire holds its rounded values).
    pub fn as_f32(&self) -> Cow<'_, ArrayD<f32>> {
        match self {
            Self::F32(array) => Cow::Borrowed(array),
            Self::Wire {
                bytes,
                dtype,
                shape,
                norm,
            } => Cow::Owned(
                ArrayD::from_shape_vec(IxDyn(shape), wire_to_f32(bytes, *dtype, norm))
                    .unwrap_or_else(|_| ArrayD::zeros(IxDyn(shape))),
            ),
            Self::U8 { patches, norm } => {
                let table = norm.table_f32();
                let block = norm.channel_block.max(1);
                let row = row_len(patches.shape());
                let mut out = vec![0f32; patches.len()];
                let standard = patches.as_standard_layout();
                let src = standard.as_slice().unwrap_or(&[]);
                fill_rows(&mut out, src, row, |dst, src| {
                    for (i, (d, &v)) in dst.iter_mut().zip(src).enumerate() {
                        *d = table[(i / block) % 3][v as usize];
                    }
                });
                Cow::Owned(
                    ArrayD::from_shape_vec(patches.raw_dim(), out)
                        .unwrap_or_else(|_| ArrayD::zeros(patches.raw_dim())),
                )
            }
        }
    }

    /// The input as normalized `f32`, owned: a move for floats.
    pub fn into_f32(self) -> ArrayD<f32> {
        match self {
            Self::F32(array) => array,
            other @ (Self::U8 { .. } | Self::Wire { .. }) => other.as_f32().into_owned(),
        }
    }

    /// The input as little-endian bytes of `dtype`, consuming it: bytes
    /// already in `dtype` move out; anything else is written as
    /// [`EncoderInputView::try_write_as`] writes it, with its errors.
    pub fn into_bytes_as(self, dtype: EncoderDtype) -> Result<Vec<u8>, String> {
        match self {
            Self::Wire {
                bytes,
                dtype: own,
                shape,
                ..
            } if own == dtype => {
                let expected = shape.iter().product::<usize>() * dtype.element_size();
                if bytes.len() != expected {
                    return Err(format!(
                        "wire holds {} bytes, {expected} needed for {shape:?} in {dtype}",
                        bytes.len()
                    ));
                }
                Ok(bytes)
            }
            other => other.view().try_write_as(dtype),
        }
    }

    /// The normalized values flattened in row-major order.
    pub fn flat_f32(&self) -> Cow<'_, [f32]> {
        match self {
            Self::F32(array) => match array.as_slice() {
                Some(slice) => Cow::Borrowed(slice),
                None => Cow::Owned(array.iter().copied().collect()),
            },
            Self::U8 { .. } | Self::Wire { .. } => {
                Cow::Owned(self.as_f32().into_owned().into_raw_vec_and_offset().0)
            }
        }
    }

    /// This input, borrowed: what the per-destination writers take.
    pub fn view(&self) -> EncoderInputView<'_> {
        match self {
            Self::F32(array) => EncoderInputView::F32(array.view()),
            Self::U8 { patches, norm } => EncoderInputView::U8 {
                patches: patches.view(),
                norm: *norm,
            },
            Self::Wire {
                bytes,
                dtype,
                shape,
                norm,
            } => EncoderInputView::Wire {
                bytes,
                dtype: *dtype,
                shape: shape.clone(),
                norm: *norm,
            },
        }
    }

    /// [`EncoderInputView::write_as`] for the whole input.
    pub fn write_as(&self, dtype: EncoderDtype) -> Vec<u8> {
        self.view().write_as(dtype)
    }

    /// [`EncoderInputView::write_into`] for the whole input.
    pub fn write_into(&self, dtype: EncoderDtype, out: &mut [u8]) -> Result<(), String> {
        self.view().write_into(dtype, out)
    }

    /// Rows `[start, start + len)` along the first axis, as an owned input.
    pub fn slice_axis0(&self, start: usize, len: usize) -> AnyhowResult<Self> {
        Ok(self.view().slice_axis0(start, len)?.to_owned())
    }

    /// Stack inputs along the first axis. Bytes stack with bytes of the same
    /// normalization, floats with floats; the two forms do not mix.
    pub fn concat(parts: &[&Self]) -> AnyhowResult<Self> {
        let first = parts.first().context("cannot join zero encoder inputs")?;
        match first {
            Self::Wire {
                dtype, norm, shape, ..
            } => {
                let mut bytes = Vec::new();
                let mut rows = 0usize;
                for part in parts {
                    match part {
                        Self::Wire {
                            bytes: part_bytes,
                            dtype: part_dtype,
                            shape: part_shape,
                            norm: part_norm,
                        } if part_dtype == dtype
                            && part_norm == norm
                            && part_shape.get(1..) == shape.get(1..) =>
                        {
                            bytes.extend_from_slice(part_bytes);
                            rows += part_shape.first().copied().unwrap_or(0);
                        }
                        Self::Wire { .. } => anyhow::bail!(
                            "encoder inputs of the batch parts disagree on wire dtype, normalization or shape"
                        ),
                        Self::F32(_) | Self::U8 { .. } => anyhow::bail!(
                            "encoder inputs of the batch parts mix wire bytes with other forms"
                        ),
                    }
                }
                let mut joined = shape.clone();
                if let Some(first_dim) = joined.first_mut() {
                    *first_dim = rows;
                }
                Self::wire(bytes, *dtype, joined, *norm).map_err(anyhow::Error::msg)
            }
            Self::F32(_) => {
                let views = parts
                    .iter()
                    .map(|part| match part {
                        Self::F32(array) => Ok(array.view()),
                        Self::U8 { .. } | Self::Wire { .. } => Err(anyhow::anyhow!(
                            "encoder inputs of the batch parts mix bytes and floats"
                        )),
                    })
                    .collect::<AnyhowResult<Vec<_>>>()?;
                Ok(Self::F32(ndarray::concatenate(Axis(0), &views).context(
                    "encoder inputs of the batch parts do not stack",
                )?))
            }
            Self::U8 { norm, .. } => {
                let views = parts
                    .iter()
                    .map(|part| match part {
                        Self::U8 {
                            patches,
                            norm: other,
                        } if other == norm => Ok(patches.view()),
                        Self::U8 { .. } => Err(anyhow::anyhow!(
                            "encoder inputs of the batch parts disagree on pixel normalization"
                        )),
                        Self::F32(_) | Self::Wire { .. } => Err(anyhow::anyhow!(
                            "encoder inputs of the batch parts mix bytes and floats"
                        )),
                    })
                    .collect::<AnyhowResult<Vec<_>>>()?;
                Ok(Self::U8 {
                    patches: ndarray::concatenate(Axis(0), &views)
                        .context("encoder inputs of the batch parts do not stack")?,
                    norm: *norm,
                })
            }
        }
    }
}

/// A borrowed [`EncoderInput`]: the whole, or rows of it. Per-destination
/// materialization lives here so an item sliced out of a batch is written
/// without first being copied.
#[derive(Debug, Clone)]
pub enum EncoderInputView<'a> {
    F32(ArrayViewD<'a, f32>),
    U8 {
        patches: ArrayViewD<'a, u8>,
        norm: PixelNorm,
    },
    /// Rows of a [`EncoderInput::Wire`]: `shape`'s elements as `dtype` bytes.
    Wire {
        bytes: &'a [u8],
        dtype: EncoderDtype,
        shape: Vec<usize>,
        norm: PixelNorm,
    },
}

impl<'a> EncoderInputView<'a> {
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(array) => array.shape(),
            Self::U8 { patches, .. } => patches.shape(),
            Self::Wire { shape, .. } => shape,
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Elements in all.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The normalization bytes carry; `None` for floats, already normalized.
    pub fn pixel_norm(&self) -> Option<PixelNorm> {
        match self {
            Self::F32(_) => None,
            Self::U8 { norm, .. } | Self::Wire { norm, .. } => Some(*norm),
        }
    }

    /// An owned copy, in the same form.
    pub fn to_owned(&self) -> EncoderInput {
        match self {
            Self::F32(array) => EncoderInput::F32(array.to_owned()),
            Self::U8 { patches, norm } => EncoderInput::U8 {
                patches: patches.to_owned(),
                norm: *norm,
            },
            Self::Wire {
                bytes,
                dtype,
                shape,
                norm,
            } => EncoderInput::Wire {
                bytes: bytes.to_vec(),
                dtype: *dtype,
                shape: shape.clone(),
                norm: *norm,
            },
        }
    }

    /// Rows `[start, start + len)` along the first axis, still borrowed.
    pub fn slice_axis0(&self, start: usize, len: usize) -> AnyhowResult<Self> {
        let rows = self.shape().first().copied().unwrap_or(0);
        anyhow::ensure!(
            start.checked_add(len).is_some_and(|end| end <= rows),
            "rows {start}..{} are out of bounds for {rows} rows",
            start.saturating_add(len)
        );
        let range = Slice::from(start..start + len);
        Ok(match self {
            Self::F32(array) => Self::F32(array.clone().slice_axis_move(Axis(0), range)),
            Self::U8 { patches, norm } => Self::U8 {
                patches: patches.clone().slice_axis_move(Axis(0), range),
                norm: *norm,
            },
            Self::Wire {
                bytes,
                dtype,
                shape,
                norm,
            } => {
                let row_bytes = row_len(shape) * dtype.element_size();
                let mut sliced = shape.clone();
                if let Some(first_dim) = sliced.first_mut() {
                    *first_dim = len;
                }
                Self::Wire {
                    bytes: bytes
                        .get(start * row_bytes..(start + len) * row_bytes)
                        .context("wire bytes shorter than their shape")?,
                    dtype: *dtype,
                    shape: sliced,
                    norm: *norm,
                }
            }
        })
    }

    /// The input as little-endian bytes of `dtype`, in row-major order.
    ///
    /// Bytes to a float dtype go through the normalization table once;
    /// floats to a narrower dtype are rounded to nearest even. `Uint8` from
    /// bytes is the bytes themselves; from floats it is the rounded, clamped
    /// value, for a caller that knows its floats are whole numbers.
    pub fn write_as(&self, dtype: EncoderDtype) -> Vec<u8> {
        // A wire that cannot be written in the dtype asked for leaves nothing
        // behind (an empty tensor the destination rejects) rather than values
        // that are not the reference's; the paths that build a request use
        // [`try_write_as`](Self::try_write_as) and report it instead.
        self.try_write_as(dtype).unwrap_or_else(|message| {
            tracing::error!(%message, "encoder input not written");
            Vec::new()
        })
    }

    /// [`write_as`](Self::write_as) with its error: a wire written in another
    /// dtype than its own (a `float32` wire narrows to the half widths; no
    /// other conversion is exact), or a wire whose bytes do not match its
    /// shape.
    pub fn try_write_as(&self, dtype: EncoderDtype) -> Result<Vec<u8>, String> {
        let mut out = vec![0u8; self.len() * dtype.element_size()];
        self.write_into(dtype, &mut out)?;
        Ok(out)
    }

    /// Whether [`write_into`](Self::write_into) can write this input in
    /// `dtype`, decided before a destination (a shared-memory segment) is
    /// prepared for it.
    pub fn writable_as(&self, dtype: EncoderDtype) -> Result<(), String> {
        match self {
            Self::Wire { dtype: own, .. }
                if *own != dtype
                    && !(*own == EncoderDtype::Float32
                        && matches!(dtype, EncoderDtype::BFloat16 | EncoderDtype::Float16)) =>
            {
                Err(format!(
                    "encoder input holds {own} wire values, which cannot be written as {dtype}"
                ))
            }
            _ => Ok(()),
        }
    }

    /// [`write_as`](Self::write_as) into a buffer the caller owns (a
    /// shared-memory mapping), which must hold exactly `len() *
    /// dtype.element_size()` bytes.
    pub fn write_into(&self, dtype: EncoderDtype, out: &mut [u8]) -> Result<(), String> {
        let expected = self.len() * dtype.element_size();
        if out.len() != expected {
            return Err(format!(
                "encoder input buffer holds {} bytes, {expected} needed for {dtype}",
                out.len()
            ));
        }
        let row = row_len(self.shape());
        match self {
            Self::Wire {
                bytes, dtype: own, ..
            } => write_wire(bytes, *own, dtype, out),
            Self::U8 { patches, norm } => {
                write_u8(patches, norm, row, dtype, out);
                Ok(())
            }
            Self::F32(array) => {
                write_f32(array, row, dtype, out);
                Ok(())
            }
        }
    }
}

/// A wire's bytes written as `asked`: copied in its own dtype, rounded to a
/// narrower float from a `float32` wire (the normalized values exactly, so
/// the halves round as the float path rounds), nothing else.
fn write_wire(
    bytes: &[u8],
    own: EncoderDtype,
    asked: EncoderDtype,
    out: &mut [u8],
) -> Result<(), String> {
    match (own, asked) {
        (own, asked) if own == asked => {
            if bytes.len() != out.len() {
                return Err(format!(
                    "wire holds {} bytes, {} needed for its shape in {own}",
                    bytes.len(),
                    out.len()
                ));
            }
            out.copy_from_slice(bytes);
            Ok(())
        }
        (EncoderDtype::Float32, EncoderDtype::BFloat16) => {
            narrow_wire(bytes, out, f32_to_bf16_bits)
        }
        (EncoderDtype::Float32, EncoderDtype::Float16) => narrow_wire(bytes, out, f32_to_f16_bits),
        (own, asked) => Err(format!(
            "encoder input holds {own} wire values, which cannot be written as {asked}"
        )),
    }
}

/// A `float32` wire narrowed to a half width, `convert` fixed at compile time
/// so the conversion inlines into the loop whichever half it is.
fn narrow_wire(bytes: &[u8], out: &mut [u8], convert: impl Fn(f32) -> u16) -> Result<(), String> {
    if bytes.len() != 2 * out.len() {
        return Err(format!(
            "wire holds {} bytes, {} needed for its shape in float32",
            bytes.len(),
            2 * out.len()
        ));
    }
    for (d, word) in out
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(bytes.as_chunks::<4>().0)
    {
        *d = convert(f32::from_le_bytes(*word)).to_le_bytes();
    }
    Ok(())
}

/// Pixel bytes in their patch layout (rows of `row` elements) written as
/// `dtype`: the bytes themselves for `Uint8`, the normalization table's
/// values for a float dtype.
fn write_u8(
    patches: &ArrayViewD<'_, u8>,
    norm: &PixelNorm,
    row: usize,
    dtype: EncoderDtype,
    out: &mut [u8],
) {
    let standard = patches.as_standard_layout();
    let src = standard.as_slice().unwrap_or(&[]);
    let block = norm.channel_block.max(1);
    match dtype {
        EncoderDtype::Uint8 => out.copy_from_slice(src),
        EncoderDtype::Float32 => {
            let table = norm.table_f32().map(|channel| channel.map(f32::to_bits));
            fill_rows_bytes(out, src, row, 4, |dst, src| {
                pixel_convert::write_f32(dst, src, block, norm, &table);
            });
        }
        EncoderDtype::BFloat16 => {
            let table = norm.table_u16(f32_to_bf16_bits);
            fill_rows_bytes(out, src, row, 2, |dst, src| {
                pixel_convert::write_bf16(dst, src, block, norm, &table);
            });
        }
        EncoderDtype::Float16 => {
            let table = norm.table_u16(f32_to_f16_bits);
            fill_rows_bytes(out, src, row, 2, |dst, src| {
                pixel_convert::write_u16(dst, src, block, &table);
            });
        }
    }
}

/// Normalized floats (rows of `row` elements) written as `dtype`: their own
/// bytes for `Float32`, rounded to nearest even for the half widths, the
/// rounded and clamped value for `Uint8`.
fn write_f32(array: &ArrayViewD<'_, f32>, row: usize, dtype: EncoderDtype, out: &mut [u8]) {
    let standard = array.as_standard_layout();
    let src = standard.as_slice().unwrap_or(&[]);
    match dtype {
        EncoderDtype::Float32 => {
            fill_rows_bytes(out, src, row, 4, |dst, src| {
                for (d, &v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                    *d = v.to_le_bytes();
                }
            });
        }
        EncoderDtype::BFloat16 => write_f32_halves(out, src, row, f32_to_bf16_bits),
        EncoderDtype::Float16 => write_f32_halves(out, src, row, f32_to_f16_bits),
        EncoderDtype::Uint8 => {
            for (d, &v) in out.iter_mut().zip(src) {
                *d = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Floats (rows of `row` elements) written as a half width, `convert` fixed
/// at compile time so the conversion inlines into the row loop whichever
/// half it is.
fn write_f32_halves(out: &mut [u8], src: &[f32], row: usize, convert: impl Fn(f32) -> u16 + Sync) {
    fill_rows_bytes(out, src, row, 2, |dst, src| {
        for (d, &v) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
            *d = convert(v).to_le_bytes();
        }
    });
}

/// A wire's values as `f32`: exact from a `float32` wire, the rounded values
/// widened from the narrower floats, the normalization applied to bytes.
fn wire_to_f32(bytes: &[u8], dtype: EncoderDtype, norm: &PixelNorm) -> Vec<f32> {
    match dtype {
        EncoderDtype::Float32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect(),
        EncoderDtype::BFloat16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|half| f32::from_bits(u32::from(u16::from_le_bytes(*half)) << 16))
            .collect(),
        EncoderDtype::Float16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|half| f16_bits_to_f32(u16::from_le_bytes(*half)))
            .collect(),
        EncoderDtype::Uint8 => {
            let table = norm.table_f32();
            let block = norm.channel_block.max(1);
            bytes
                .iter()
                .enumerate()
                .map(|(i, &v)| table[(i / block) % 3][v as usize])
                .collect()
        }
    }
}

/// The `f32` a `float16` bit pattern denotes (exact).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x3ff);
    let value = match exp {
        0 => {
            if mant == 0 {
                0
            } else {
                // Subnormal: normalize the mantissa.
                let shift = mant.leading_zeros() - 21;
                let mant = (mant << shift) & 0x3ff;
                ((113 - shift) << 23) | (mant << 13)
            }
        }
        0x1f => 0x7f80_0000 | (mant << 13),
        _ => ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(sign | value)
}

/// Elements per row: everything but the first axis (the whole input when it
/// has one axis).
fn row_len(shape: &[usize]) -> usize {
    match shape {
        [] => 0,
        [n] => *n,
        [_, rest @ ..] => rest.iter().product::<usize>().max(1),
    }
}

/// Run `fill(dst_rows, src_rows)` over row bands, concurrently when the
/// executor allows it: each band is a whole number of rows so a channel
/// block never straddles two bands.
fn fill_rows<T: Copy + Send + Sync, S: Copy + Send + Sync>(
    out: &mut [T],
    src: &[S],
    row: usize,
    fill: impl Fn(&mut [T], &[S]) + Sync,
) {
    let rows = src.len().checked_div(row).unwrap_or(0);
    let tasks = execution::task_count(size_of_val(out), rows, 32);
    if tasks <= 1 {
        fill(out, src);
        return;
    }
    let band_rows = rows.div_ceil(tasks);
    execution::scope(|spawner| {
        let fill = &fill;
        let (mut out_rest, mut src_rest) = (out, src);
        while !out_rest.is_empty() {
            let n = band_rows.min(out_rest.len() / row.max(1));
            let (dst, dst_tail) = out_rest.split_at_mut(n * row);
            let (band_src, src_tail) = src_rest.split_at(n * row);
            out_rest = dst_tail;
            src_rest = src_tail;
            spawner.spawn(move |_| fill(dst, band_src));
        }
    });
}

/// [`fill_rows`] for a byte output of `width` bytes per element.
fn fill_rows_bytes<S: Copy + Send + Sync>(
    out: &mut [u8],
    src: &[S],
    row: usize,
    width: usize,
    fill: impl Fn(&mut [u8], &[S]) + Sync,
) {
    let rows = src.len().checked_div(row).unwrap_or(0);
    let tasks = execution::task_count(out.len(), rows, 32);
    if tasks <= 1 {
        fill(out, src);
        return;
    }
    let band_rows = rows.div_ceil(tasks);
    execution::scope(|spawner| {
        let fill = &fill;
        let (mut out_rest, mut src_rest) = (out, src);
        while !out_rest.is_empty() {
            let n = band_rows.min(src_rest.len() / row.max(1));
            let (dst, dst_tail) = out_rest.split_at_mut(n * row * width);
            let (band_src, src_tail) = src_rest.split_at(n * row);
            out_rest = dst_tail;
            src_rest = src_tail;
            spawner.spawn(move |_| fill(dst, band_src));
        }
    });
}

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

    /// Whether two values carry the same thing. Floats compare by bits so that
    /// two values that were written from the same source always match.
    fn same_value(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Tensor {
                    data: a,
                    shape: a_shape,
                },
                Self::Tensor {
                    data: b,
                    shape: b_shape,
                },
            ) => a_shape == b_shape && same_floats(a, b),
            (
                Self::IntTensor {
                    data: a,
                    shape: a_shape,
                },
                Self::IntTensor {
                    data: b,
                    shape: b_shape,
                },
            ) => a_shape == b_shape && a == b,
            (
                Self::UintTensor {
                    data: a,
                    shape: a_shape,
                },
                Self::UintTensor {
                    data: b,
                    shape: b_shape,
                },
            ) => a_shape == b_shape && a == b,
            (Self::IntVec(a), Self::IntVec(b)) => a == b,
            (Self::UintVec(a), Self::UintVec(b)) => a == b,
            (Self::FloatVec(a), Self::FloatVec(b)) => same_floats(a, b),
            (Self::TupleVec(a), Self::TupleVec(b)) => a == b,
            (Self::Int(_) | Self::Float(_) | Self::Bool(_), _) => self.same_scalar(other),
            _ => false,
        }
    }
}

/// Compare float sequences by bits, so two values written the same way match
/// and neither NaN nor a signed zero makes them differ by accident.
fn same_floats(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
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
    /// Primary encoder input: normalized floats, or pixel bytes with their
    /// normalization (see [`EncoderInput`]).
    pub encoder_input: EncoderInput,

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
    pub fn new(
        encoder_input: impl Into<EncoderInput>,
        feature_token_counts: Vec<usize>,
        item_sizes: Vec<(u32, u32)>,
    ) -> Self {
        Self {
            encoder_input: encoder_input.into(),
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
        self.encoder_input.flat_f32()
    }

    /// Get the shape of the primary encoder input as a vector.
    pub fn encoder_input_shape(&self) -> Vec<usize> {
        self.encoder_input.shape().to_vec()
    }

    /// Join batches processed one item at a time into the batch a processor
    /// would have produced from all items at once.
    ///
    /// The encoder input always stacks along its first dimension. For the other
    /// values `layouts` decides: a value the model reads per item is stacked
    /// the same way, and a value with no layout is one the backend shares
    /// across the whole batch, so it is kept once and every part has to agree
    /// on it. Stacking a shared value instead would hand the joined sequence to
    /// each item.
    pub fn concat(parts: Vec<Self>, layouts: &HashMap<String, FieldLayout>) -> AnyhowResult<Self> {
        anyhow::ensure!(!parts.is_empty(), "cannot join zero batches");
        if parts.len() == 1 {
            return parts.into_iter().next().context("cannot join zero batches");
        }
        let inputs = parts
            .iter()
            .map(|part| &part.encoder_input)
            .collect::<Vec<_>>();
        let encoder_input = EncoderInput::concat(&inputs)?;

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
            let per_item = matches!(
                layouts.get(&key),
                Some(FieldLayout::Batched | FieldLayout::Flat { .. })
            );
            let joined = if per_item {
                ModelSpecificValue::concat_first_dim(&values)
                    .with_context(|| format!("failed to join model-specific value {key}"))?
            } else {
                let first = values
                    .first()
                    .copied()
                    .context("cannot join zero batch parts")?;
                anyhow::ensure!(
                    values.iter().all(|value| value.same_value(first)),
                    "model-specific value {key} is shared across the batch but the parts disagree on it"
                );
                first.clone()
            };
            debug_assert!(
                !per_item
                    || !matches!(
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

    fn video_layouts() -> HashMap<String, FieldLayout> {
        HashMap::from([
            ("video_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_video".to_string(), FieldLayout::Batched),
            ("video_second_per_grid".to_string(), FieldLayout::Batched),
        ])
    }

    #[test]
    fn concat_stacks_clips_the_way_one_batch_would_be_laid_out() {
        let layouts = video_layouts();

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

    /// A value the backend shares across the batch must not be stacked: the
    /// whole joined sequence would then stand for every item. Parts that
    /// disagree on such a value cannot be joined at all.
    #[test]
    fn a_value_without_a_layout_is_kept_once_and_must_agree() {
        let mut layouts = video_layouts();
        layouts.remove("video_second_per_grid");

        let same =
            PreprocessedEncoderInputs::concat(vec![clip(16, 1, 0.5), clip(32, 2, 0.5)], &layouts)
                .unwrap();
        assert!(matches!(
            &same.model_specific["video_second_per_grid"],
            ModelSpecificValue::Tensor { data, shape } if data == &vec![0.5] && shape == &vec![1]
        ));

        assert!(PreprocessedEncoderInputs::concat(
            vec![clip(16, 1, 1.0), clip(32, 2, 0.5)],
            &layouts
        )
        .is_err());
    }

    #[test]
    fn concat_keeps_a_single_clip_untouched_and_rejects_mismatched_parts() {
        let layouts = video_layouts();
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
        wrong_width.encoder_input = ndarray::Array2::<f32>::zeros((16, 5)).into();
        assert!(
            PreprocessedEncoderInputs::concat(vec![clip(16, 1, 1.0), wrong_width], &layouts)
                .is_err()
        );
    }

    /// Every pixel byte of every channel, written to each float wire dtype,
    /// equals the per-element arithmetic of the reference processors,
    /// `(byte * scale - offset[c]) / divisor[c]` in `f32`, rounded to the
    /// narrower floats as the scalar conversions round: the table lookups
    /// move those values and nothing else, whatever the channel block.
    #[test]
    fn pixel_bytes_write_the_per_element_normalization_for_every_byte_and_channel() {
        let into_bytes = (1.0_f64 / (1.0 / 255.0)) as f32;
        let norms = [
            // mean 0.5 / std 0.5 with rescale, in byte units
            PixelNorm {
                scale: 1.0,
                offset: [0.5_f64 as f32 * into_bytes; 3],
                divisor: [0.5_f64 as f32 * into_bytes; 3],
                channel_block: 512,
            },
            // the CLIP mean / std with rescale, in byte units
            PixelNorm {
                scale: 1.0,
                offset: [
                    0.481_454_66_f64 as f32 * into_bytes,
                    0.457_827_5_f64 as f32 * into_bytes,
                    0.408_210_73_f64 as f32 * into_bytes,
                ],
                divisor: [
                    0.268_629_54_f64 as f32 * into_bytes,
                    0.261_302_58_f64 as f32 * into_bytes,
                    0.275_777_11_f64 as f32 * into_bytes,
                ],
                channel_block: 392,
            },
            // rescale only
            PixelNorm {
                scale: (1.0_f64 / 255.0) as f32,
                offset: [0.0; 3],
                divisor: [1.0; 3],
                channel_block: 17,
            },
            // normalize only, blocks shorter than a vector
            PixelNorm {
                scale: 1.0,
                offset: [0.5, 0.4, 0.3],
                divisor: [0.25, 0.2, 0.3],
                channel_block: 7,
            },
        ];
        for norm in norms {
            let block = norm.channel_block;
            // 256 rows of three channel blocks: row r holds the byte
            // (r + 85 c) % 256 in channel c, so each channel sees every byte.
            let bytes: Vec<u8> = (0..256)
                .flat_map(|r| {
                    (0..3).flat_map(move |c| std::iter::repeat_n(((r + 85 * c) % 256) as u8, block))
                })
                .collect();
            let patches = ndarray::Array2::from_shape_vec((256, 3 * block), bytes.clone())
                .unwrap()
                .into_dyn();
            let input = EncoderInput::bytes(patches, norm);
            let value = |i: usize, v: u8| {
                let c = (i / block) % 3;
                (f32::from(v) * norm.scale - norm.offset[c]) / norm.divisor[c]
            };
            let f32_bytes: Vec<u8> = bytes
                .iter()
                .enumerate()
                .flat_map(|(i, &v)| value(i, v).to_le_bytes())
                .collect();
            assert_eq!(
                input.write_as(EncoderDtype::Float32),
                f32_bytes,
                "f32, block {block}"
            );
            let bf16_bytes: Vec<u8> = bytes
                .iter()
                .enumerate()
                .flat_map(|(i, &v)| f32_to_bf16_bits(value(i, v)).to_le_bytes())
                .collect();
            assert_eq!(
                input.write_as(EncoderDtype::BFloat16),
                bf16_bytes,
                "bf16, block {block}"
            );
            let f16_bytes: Vec<u8> = bytes
                .iter()
                .enumerate()
                .flat_map(|(i, &v)| f32_to_f16_bits(value(i, v)).to_le_bytes())
                .collect();
            assert_eq!(
                input.write_as(EncoderDtype::Float16),
                f16_bytes,
                "f16, block {block}"
            );
            assert_eq!(
                input.write_as(EncoderDtype::Uint8),
                bytes,
                "u8, block {block}"
            );
        }
    }

    /// Every `float16` bit pattern decodes to its IEEE value: subnormals as
    /// `mantissa * 2^-24`, normals as `(1024 + mantissa) * 2^(exponent - 25)`,
    /// the top exponent as infinities and NaNs.
    #[test]
    fn every_half_float_pattern_decodes_exactly() {
        for bits in 0..=u16::MAX {
            let got = f16_bits_to_f32(bits);
            let sign = if bits >> 15 == 1 { -1.0f64 } else { 1.0 };
            let exp = i32::from((bits >> 10) & 0x1f);
            let mant = f64::from(bits & 0x3ff);
            match exp {
                0 => assert_eq!(got, (sign * mant * 2f64.powi(-24)) as f32, "{bits:#06x}"),
                0x1f if mant == 0.0 => {
                    assert!(
                        got.is_infinite() && (got < 0.0) == (sign < 0.0),
                        "{bits:#06x}"
                    );
                }
                0x1f => assert!(got.is_nan(), "{bits:#06x}"),
                _ => assert_eq!(
                    got,
                    (sign * (1024.0 + mant) * 2f64.powi(exp - 25)) as f32,
                    "{bits:#06x}"
                ),
            }
        }
    }

    /// A wire's bytes are checked against its shape and dtype by the
    /// constructor, and a malformed one built by hand is refused at the
    /// write, not a panic or a silent truncation.
    #[test]
    fn a_wire_is_checked_against_its_shape_and_dtype() {
        let norm = PixelNorm {
            scale: 1.0,
            offset: [0.0; 3],
            divisor: [1.0; 3],
            channel_block: 1,
        };
        assert!(
            EncoderInput::wire(vec![0u8; 12], EncoderDtype::BFloat16, vec![2, 3], norm).is_ok()
        );
        assert!(
            EncoderInput::wire(vec![0u8; 11], EncoderDtype::BFloat16, vec![2, 3], norm).is_err()
        );
        let short = EncoderInput::Wire {
            bytes: vec![0u8; 10],
            dtype: EncoderDtype::BFloat16,
            shape: vec![2, 3],
            norm,
        };
        let mut out = vec![0u8; 12];
        assert!(short
            .view()
            .write_into(EncoderDtype::BFloat16, &mut out)
            .is_err());
        assert!(short.view().try_write_as(EncoderDtype::BFloat16).is_err());
        assert!(short.into_bytes_as(EncoderDtype::BFloat16).is_err());
        let wide =
            EncoderInput::wire(vec![0u8; 24], EncoderDtype::Float32, vec![2, 3], norm).unwrap();
        assert!(wide.view().writable_as(EncoderDtype::BFloat16).is_ok());
        assert!(wide.view().try_write_as(EncoderDtype::Float16).is_ok());
        let half =
            EncoderInput::wire(vec![0u8; 12], EncoderDtype::BFloat16, vec![2, 3], norm).unwrap();
        assert!(half.view().writable_as(EncoderDtype::Float32).is_err());
        assert!(half.view().try_write_as(EncoderDtype::Float32).is_err());
        assert!(half.into_bytes_as(EncoderDtype::Float32).is_err());
    }

    /// A wire input built from a byte input's own write: the same shape and
    /// bytes, its rows sliced and stacked by byte ranges, written as itself
    /// without a conversion, as a narrower float only from `float32`.
    #[test]
    fn a_wire_input_carries_its_bytes_and_slices_by_rows() {
        let norm = PixelNorm {
            scale: 1.0,
            offset: [127.5; 3],
            divisor: [127.5; 3],
            channel_block: 4,
        };
        let bytes: Vec<u8> = (0..3 * 12 * 2).map(|i| (i * 37 % 256) as u8).collect();
        let patches = EncoderInput::bytes(
            ndarray::Array2::from_shape_vec((6, 12), bytes).unwrap(),
            norm,
        );
        for dtype in [
            EncoderDtype::BFloat16,
            EncoderDtype::Float16,
            EncoderDtype::Float32,
        ] {
            let written = patches.write_as(dtype);
            let wire = EncoderInput::Wire {
                bytes: written.clone(),
                dtype,
                shape: vec![6, 12],
                norm,
            };
            assert_eq!(wire.shape(), &[6, 12]);
            assert_eq!(wire.len(), 72);
            assert_eq!(wire.nbytes(), written.len());
            assert_eq!(wire.pixel_norm(), Some(norm));
            assert_eq!(wire.write_as(dtype), written);
            // rows 2..4 of the wire are the bytes of rows 2..4 of the patches
            let rows = wire.slice_axis0(2, 2).unwrap();
            assert_eq!(rows.shape(), &[2, 12]);
            assert_eq!(
                rows.write_as(dtype),
                patches.slice_axis0(2, 2).unwrap().write_as(dtype)
            );
            let stacked = EncoderInput::concat(&[&rows, &wire.slice_axis0(0, 1).unwrap()]).unwrap();
            assert_eq!(stacked.shape(), &[3, 12]);
            assert_eq!(stacked.nbytes(), 3 * 12 * dtype.element_size());
            if dtype == EncoderDtype::Float16 {
                // the wire's own rounded values, decoded exactly
                let rounded: Vec<f32> = patches
                    .as_f32()
                    .iter()
                    .map(|value| f16_bits_to_f32(f32_to_f16_bits(*value)))
                    .collect();
                assert_eq!(wire.as_f32().as_slice().unwrap(), &rounded[..]);
            }
            if dtype == EncoderDtype::Float32 {
                // the normalized values themselves, so the narrower floats derive exactly
                assert_eq!(
                    wire.as_f32().as_slice().unwrap(),
                    patches.as_f32().as_slice().unwrap()
                );
                assert_eq!(
                    wire.write_as(EncoderDtype::BFloat16),
                    patches.write_as(EncoderDtype::BFloat16)
                );
                assert_eq!(
                    wire.write_as(EncoderDtype::Float16),
                    patches.write_as(EncoderDtype::Float16)
                );
            } else {
                // a narrower wire cannot be widened into the reference's float32: nothing is written
                assert!(wire
                    .write_into(EncoderDtype::Float32, &mut vec![0u8; 72 * 4])
                    .is_err());
                assert!(wire.write_as(EncoderDtype::Float32).is_empty());
            }
        }
    }
}
