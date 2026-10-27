//! DeepSeek-V4.1 image layout shared by identity validation and ZMQ assembly.
use crate::sglang_proto::{MultimodalInputs, TensorData};

pub const HASHES: &str = "smg_feature_hashes";

/// One image's patch interval and the native encoder's spatial dimensions.
#[derive(Debug)]
pub struct ImageLayout {
    pub height: i64,
    pub width: i64,
    pub patches: std::ops::Range<usize>,
}

/// Read an exact int64 tensor, rejecting malformed dimensions or byte lengths.
pub fn integers(tensor: &TensorData, shape: &[u32]) -> Result<Vec<i64>, String> {
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d as usize));
    if tensor.dtype != "int64"
        || tensor.shape != shape
        || count.and_then(|n| n.checked_mul(8)) != Some(tensor.data.len())
    {
        return Err("invalid DeepSeek image metadata tensor".into());
    }
    Ok(tensor
        .data
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| i64::from_le_bytes(*v))
        .collect())
}

/// Validate per-image grids, patch counts and placeholder lengths before slicing.
/// None means another processor's layout; malformed DeepSeek input is an error.
pub fn layout(mm: &MultimodalInputs) -> Result<Option<Vec<ImageLayout>>, String> {
    if !mm.model_specific_tensors.contains_key("vit_grid") {
        return Ok(None);
    }
    let n = u32::try_from(mm.mm_placeholders.len()).map_err(|_| "too many images")?;
    if n == 0 || mm.modalities.is_empty() || mm.modalities.iter().any(|m| m != "image") {
        return Err("DeepSeek requires image placeholders".into());
    }
    let read = |name: &str, shape: &[u32]| {
        integers(
            mm.model_specific_tensors
                .get(name)
                .ok_or_else(|| format!("missing {name}"))?,
            shape,
        )
    };
    let vit = read("vit_grid", &[n, 2])?;
    let llm = read("llm_grid", &[n, 2])?;
    let patches = read("patches_per_image", &[n])?;
    let counts = read("types_per_image", &[n])?;
    let mut start = 0usize;
    let mut result = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        let (h, w, lh, lw) = (vit[2 * i], vit[2 * i + 1], llm[2 * i], llm[2 * i + 1]);
        if h <= 0
            || w <= 0
            || lh <= 0
            || lw <= 0
            || lh > h
            || lw > w
            || h.checked_mul(w) != Some(patches[i])
            || lw
                .checked_add(1)
                .and_then(|v| lh.checked_mul(v))
                .and_then(|v| v.checked_add(2))
                != Some(counts[i])
            || counts[i] != i64::from(mm.mm_placeholders[i].length)
        {
            return Err("inconsistent DeepSeek image grid or placeholder length".into());
        }
        let end = start
            .checked_add(usize::try_from(patches[i]).map_err(|_| "image too large")?)
            .ok_or("image too large")?;
        result.push(ImageLayout {
            height: h,
            width: w,
            patches: start..end,
        });
        start = end;
    }
    if let Some(pixels) = &mm.pixel_values {
        if pixels.shape.len() != 4
            || pixels.shape[0] as usize != start
            || pixels.shape[1] != 3
            || pixels.shape[2] == 0
            || pixels.shape[2] != pixels.shape[3]
        {
            return Err("invalid DeepSeek patch tensor shape".into());
        }
    }
    Ok(Some(result))
}

/// Exact byte stride of a contiguous patch tensor, including its declared dtype.
pub fn patch_stride(pixels: &TensorData) -> Result<usize, String> {
    let width = match pixels.dtype.as_str() {
        "float32" => 4usize,
        "float16" | "bfloat16" => 2,
        _ => return Err("unsupported DeepSeek pixel dtype".into()),
    };
    let stride = pixels
        .shape
        .iter()
        .skip(1)
        .try_fold(width, |n, &d| n.checked_mul(d as usize))
        .ok_or("patch shape overflow")?;
    if pixels.shape.len() != 4
        || stride.checked_mul(pixels.shape[0] as usize) != Some(pixels.data.len())
    {
        return Err("DeepSeek patch byte length mismatch".into());
    }
    Ok(stride)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sglang_proto::PlaceholderRange;

    /// Create two images with uneven patch counts to exercise independent native item ranges.
    fn fixture() -> MultimodalInputs {
        let mut mm = MultimodalInputs {
            modalities: vec!["image".into()],
            mm_placeholders: vec![
                PlaceholderRange {
                    offset: 1,
                    length: 4,
                },
                PlaceholderRange {
                    offset: 6,
                    length: 5,
                },
            ],
            ..Default::default()
        };
        for (name, shape, values) in [
            ("vit_grid", vec![2, 2], vec![3i64, 3, 3, 6]),
            ("llm_grid", vec![2, 2], vec![1, 1, 1, 2]),
            ("patches_per_image", vec![2], vec![9, 18]),
            ("types_per_image", vec![2], vec![4, 5]),
        ] {
            mm.model_specific_tensors.insert(
                name.into(),
                TensorData {
                    dtype: "int64".into(),
                    shape,
                    data: values
                        .into_iter()
                        .flat_map(i64::to_le_bytes)
                        .collect::<Vec<_>>()
                        .into(),
                },
            );
        }
        mm
    }

    /// Image boundaries must remain independent for uneven multi-image batches.
    #[test]
    fn validates_multi_image_patch_ranges_and_metadata() {
        let mm = fixture();
        let images = layout(&mm).unwrap().unwrap();
        assert_eq!(images[0].patches, 0..9);
        assert_eq!(images[1].patches, 9..27);
        assert_eq!((images[1].height, images[1].width), (3, 6));
        assert!(layout(&MultimodalInputs::default()).unwrap().is_none());
        for case in 0..5 {
            let mut bad = fixture();
            match case {
                0 => {
                    bad.mm_placeholders[1].length = 4;
                }
                1 => {
                    bad.model_specific_tensors.remove("llm_grid");
                }
                2 => {
                    bad.model_specific_tensors.get_mut("vit_grid").unwrap().data =
                        vec![0; 32].into();
                }
                3 => {
                    bad.model_specific_tensors
                        .get_mut("patches_per_image")
                        .unwrap()
                        .shape = vec![1];
                }
                _ => {
                    bad.pixel_values = Some(TensorData {
                        shape: vec![26, 3, 14, 14],
                        ..Default::default()
                    });
                }
            }
            assert!(layout(&bad).is_err(), "case {case}");
        }
    }

    /// Incorrect wire lengths and unsupported pixel dtypes cannot reach slicing.
    #[test]
    fn patch_stride_checks_dtype_shape_and_exact_length() {
        let mut pixels = TensorData {
            dtype: "bfloat16".into(),
            shape: vec![2, 3, 14, 14],
            data: vec![0; 2352].into(),
        };
        assert_eq!(patch_stride(&pixels).unwrap(), 1176);
        pixels.data = vec![0; 2351].into();
        assert!(patch_stride(&pixels).is_err());
        pixels.dtype = "int64".into();
        assert!(patch_stride(&pixels).is_err());
    }
}
