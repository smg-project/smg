//! DeepSeek-V4.1 image batches as SGLang's own processor lays them out.
//!
//! The Router sends a V4.1 image batch as one tensor of every image's
//! patches, with each image's patch grid (`vit_grid`), patch count
//! (`patches_per_image`) and token count (`types_per_image`) beside it: the
//! layout a vLLM engine splits by the batch's field layouts. SGLang's V4.1
//! model encodes one image per item and reads its grid as `n_vit_h` /
//! `n_vit_w` (its `multimodal/processors/deepseek_v41.py` and
//! `models/deepseek_v4.py`), so the batch is cut into those items here, by
//! the Python servicer's rule (`_deepseek_v41_items`). A batch whose counts
//! disagree is refused before the scheduler, where it would stop the engine.

use std::collections::BTreeMap;

use bytes::Bytes;
use engine_zmq_client::{
    codec::tensor::WireArrayData,
    protocol::{
        handshake::EngineCoreReadyResponse,
        sglang::multimodal::{SglangWireMmItem, SglangWireModality, SglangWireTensor},
    },
    ConnectedEngine,
};
use smg_grpc_client::sglang_proto;

const PREFIX: &str = "DeepSeek-V4.1 image batch:";

/// The grid only the Router's DeepSeek-V4.1 preprocessor sends with an image
/// batch; GLM-5.3 Flash sends `patches_per_image` as well.
const MARKER: &str = "vit_grid";

/// A batch that goes to the scheduler as one item an image, as one
/// connection takes it: what its ranks declared to check such a batch
/// against. Read from the handshakes once ([`Self::declared`]); every
/// translated batch is then put to it ([`Self::cut`]). Which batch is such
/// a batch is read off the batch alone ([`Self::claims`]), whatever the
/// ranks declared.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PerImageItems {
    vision: Option<V41Vision>,
}

impl PerImageItems {
    pub(crate) fn declared(ranks: &[ConnectedEngine]) -> Self {
        Self {
            vision: V41Vision::from_handshakes(ranks),
        }
    }

    /// Whether a payload's model tensors lay its images out one by one. Its
    /// other fields are then required, so a batch missing one is refused, not
    /// sent on as one item.
    pub(crate) fn claims(payload: &sglang_proto::MultimodalInputs) -> bool {
        payload.model_specific_tensors.contains_key(MARKER)
    }

    /// The items a translated `batch` is cut into, one an image; `None` for
    /// a batch that is not laid out so, which stays one item.
    pub(crate) fn cut(
        self,
        batch: &SglangWireMmItem,
        im_token_id: Option<u32>,
    ) -> Result<Option<Vec<SglangWireMmItem>>, String> {
        if !batch.model_specific_data.contains_key(MARKER) {
            return Ok(None);
        }
        // Only an image batch is cut into V4.1 items. Under the video label the
        // batch would go on as a video item, which SGLang's V4.1 model has no
        // embedder for.
        if batch.modality != SglangWireModality::Image {
            return Err(format!("{PREFIX} vit_grid on a video batch"));
        }
        // SGLang pads the image positions only when the id is set. Without it
        // the model finds no position for the items' embeddings and fails in
        // the scheduler.
        if im_token_id.is_none() {
            return Err(format!("{PREFIX} no im_token_id"));
        }
        deepseek_v41_items(
            batch.feature.as_ref(),
            &batch.model_specific_data,
            batch.offsets.as_deref().unwrap_or_default(),
            self.vision,
        )
        .map(Some)
    }
}

/// The served DeepSeek-V4.1 model's patch size and downsample ratio, as its
/// SGLang model reads them (`vision_patch_size`, `vision_downsample_ratio`).
/// The scheduler declares them at its handshake, off the model config it
/// loads, so model overrides are in them; the Python servicer loads that
/// config the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct V41Vision {
    patch_size: u64,
    downsample_ratio: u64,
}

impl V41Vision {
    /// From a scheduler's handshake: `None` unless it declared both values,
    /// each above zero. A scheduler declares none when its model is not a
    /// V4.1 or its config does not hold both, or when its plugin predates the
    /// fields.
    fn from_handshake(ready: &EngineCoreReadyResponse) -> Option<Self> {
        let declared = |value: Option<u64>| value.filter(|&value| value > 0);
        Some(Self {
            patch_size: declared(ready.vision_patch_size)?,
            downsample_ratio: declared(ready.vision_downsample_ratio)?,
        })
    }

    /// The geometry every rank of a connection declared: `None` unless each
    /// declared the same one. A request without a pin is placed on a rank at
    /// submit, by load, after its batch is checked, so a batch is checked
    /// only against a geometry all the ranks share.
    fn from_handshakes(ranks: &[ConnectedEngine]) -> Option<Self> {
        let mut declared = ranks
            .iter()
            .map(|rank| Self::from_handshake(&rank.ready_response));
        let first = declared.next()??;
        declared.all(|rank| rank == Some(first)).then_some(first)
    }

    /// The values one patch row holds: three channels of a square patch. The
    /// model's patch embedding is a `Linear(3 * patch_size^2, ..)` over each
    /// flattened row.
    fn row_values(self) -> u64 {
        3u64.saturating_mul(self.patch_size)
            .saturating_mul(self.patch_size)
    }

    /// How many prompt positions SGLang's V4.1 model fills for a patch grid:
    /// the length of its `image_token_types` for the downsampled grid, a start
    /// token, each line's tokens and a newline, and an end token.
    fn image_tokens(self, height: u64, width: u64) -> u64 {
        let lines = height.div_ceil(self.downsample_ratio);
        let tokens_per_line = width.div_ceil(self.downsample_ratio);
        lines
            .saturating_mul(tokens_per_line.saturating_add(1))
            .saturating_add(2)
    }
}

/// One item per image: its rows of the feature (viewed, not copied), its
/// placeholder range, and its grid as the 0-d integers `n_vit_h` and
/// `n_vit_w`. The batch's hash named every image's bytes at once, so the items
/// carry none: the scheduler, or the opt-in hasher, hashes each. With the
/// served model's `vision`, a batch whose patch rows or spans do not fit that
/// model is refused as well.
fn deepseek_v41_items(
    feature: Option<&SglangWireTensor>,
    tensors: &BTreeMap<String, SglangWireTensor>,
    offsets: &[(u64, u64)],
    vision: Option<V41Vision>,
) -> Result<Vec<SglangWireMmItem>, String> {
    let feature = feature
        .ok_or_else(|| format!("{PREFIX} no pixel_values to cut its per-image items from"))?;
    let grid_values = ints(tensors, "vit_grid")?;
    let patches = ints(tensors, "patches_per_image")?;
    let tokens = tensors
        .contains_key("types_per_image")
        .then(|| ints(tensors, "types_per_image"))
        .transpose()?;
    let (grids, odd) = grid_values.as_chunks::<2>();
    if !odd.is_empty() {
        return Err(format!(
            "{PREFIX} vit_grid holds {} values, not (height, width) pairs",
            grid_values.len()
        ));
    }
    let images = grids.len();
    if images == 0 {
        return Err(format!("{PREFIX} no images"));
    }
    for (name, count) in [
        ("placeholders", offsets.len()),
        ("patch counts", patches.len()),
        ("token counts", tokens.as_ref().map_or(images, Vec::len)),
    ] {
        if count != images {
            return Err(format!("{PREFIX} {images} images but {count} {name}"));
        }
    }
    // A patch row holds at least one value.
    let (rows, row_dims) = match feature.shape.split_first() {
        Some((&rows, row_dims)) if !row_dims.is_empty() && !row_dims.contains(&0) => {
            (rows, row_dims)
        }
        _ => {
            return Err(format!(
                "{PREFIX} pixel_values has shape {:?}, not rows of patches",
                feature.shape
            ))
        }
    };
    let total = patches
        .iter()
        .try_fold(0u64, |sum, &count| sum.checked_add(count));
    if total != Some(rows) {
        return Err(format!(
            "{PREFIX} {} patches but the pixel tensor has {rows} rows",
            total.map_or_else(|| "more than u64".to_string(), |sum| sum.to_string())
        ));
    }
    if let Some(vision) = vision {
        let values = row_dims
            .iter()
            .try_fold(1u64, |product, &dim| product.checked_mul(dim));
        if values != Some(vision.row_values()) {
            let size = vision.patch_size;
            return Err(format!(
                "{PREFIX} each patch row holds {} values, but the model's {size}x{size} patches \
                 take 3x{size}x{size} = {}",
                values.map_or_else(|| "more than u64".to_string(), |n| n.to_string()),
                vision.row_values()
            ));
        }
    }
    let bytes = feature
        .data
        .as_raw_view()
        .ok_or_else(|| format!("{PREFIX} pixel_values are not inline"))?;
    let row_bytes = match rows {
        0 => 0,
        rows => bytes.len() / rows as usize,
    };
    let mut items = Vec::with_capacity(images);
    let mut start = 0usize;
    let mut previous_last = None;
    for (index, ((&[height, width], &count), &(first, last))) in
        grids.iter().zip(&patches).zip(offsets).enumerate()
    {
        if height == 0 || width == 0 || height.checked_mul(width) != Some(count) {
            return Err(format!(
                "{PREFIX} image {index} has a {height}x{width} grid but {count} patches"
            ));
        }
        // SGLang writes the items' embeddings over the placeholders in prompt
        // order, so the ranges must be in prompt order and must not overlap.
        if let Some(previous) = previous_last.filter(|&previous| first <= previous) {
            return Err(format!(
                "{PREFIX} image {index}'s placeholder {first}..={last} does not start after \
                 image {}'s, which ends at {previous}",
                index - 1
            ));
        }
        previous_last = Some(last);
        let span = last - first + 1;
        if let Some(tokens) = &tokens {
            if span != tokens[index] {
                return Err(format!(
                    "{PREFIX} image {index} fills {span} prompt positions but has {} tokens",
                    tokens[index]
                ));
            }
        }
        if let Some(vision) = vision {
            let grid_tokens = vision.image_tokens(height, width);
            if span != grid_tokens {
                return Err(format!(
                    "{PREFIX} image {index} fills {span} prompt positions but its \
                     {height}x{width} grid makes {grid_tokens} tokens"
                ));
            }
        }
        let end = start + count as usize * row_bytes;
        let shape = std::iter::once(count)
            .chain(row_dims.iter().copied())
            .collect();
        let pixels = SglangWireTensor::new(feature.dtype.clone(), shape, bytes.slice(start..end))?;
        start = end;
        items.push(SglangWireMmItem {
            modality: SglangWireModality::Image,
            hash: None,
            offsets: Some(vec![(first, last)]),
            feature: Some(pixels),
            model_specific_data: BTreeMap::from([
                ("n_vit_h".to_string(), scalar(height)),
                ("n_vit_w".to_string(), scalar(width)),
            ]),
        });
    }
    Ok(items)
}

/// An integer tensor's values, as the Router writes its grids (`int64`, or
/// `uint32`).
fn ints(tensors: &BTreeMap<String, SglangWireTensor>, name: &str) -> Result<Vec<u64>, String> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| format!("{PREFIX} no {name}"))?;
    let bytes = tensor
        .data
        .as_raw_view()
        .ok_or_else(|| format!("{PREFIX} {name} is not inline"))?;
    match tensor.dtype.as_str() {
        "int64" => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&value| {
                u64::try_from(i64::from_le_bytes(value))
                    .map_err(|_| format!("{PREFIX} {name} holds a negative value"))
            })
            .collect(),
        "uint32" => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&value| u64::from(u32::from_le_bytes(value)))
            .collect()),
        dtype => Err(format!(
            "{PREFIX} {name} is {dtype}, not an int64 or uint32 tensor"
        )),
    }
}

/// A 0-d `int64` tensor: the scheduler reads it with `int(item.n_vit_h)`, as
/// it reads the plain integer SGLang's own processor stores.
fn scalar(value: u64) -> SglangWireTensor {
    SglangWireTensor {
        dtype: "int64".to_string(),
        shape: Vec::new(),
        data: WireArrayData::RawView(Bytes::copy_from_slice(&(value as i64).to_le_bytes())),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use engine_zmq_client::{
        codec::decode_msgpack,
        mock_engine::{connect_to_frontend, default_ready_response},
        protocol::{
            handshake::EngineCoreReadyResponse,
            sglang::{
                multimodal::SglangWireMmInputs, request::TokenizedGenerateReqInput as WireRequest,
            },
        },
        ConnectedEngine, EngineId,
    };
    use openai_protocol::worker::RuntimeType;
    use smg_grpc_client::sglang_proto;

    use super::{PerImageItems, V41Vision};
    use crate::{
        client::ZmqEngineClient, eos::EosTokenIds, sglang::translate_sglang_multimodal,
        test_support::connected_pair_declaring,
    };

    /// Two images: a 2x3 and a 1x2 patch grid, 6 + 2 patches of 3x2x2 floats,
    /// filling token spans of 5 and 4 positions in a 12-token prompt. The
    /// served model takes 2x2 patches and merges 2x2 of them per token
    /// (`VISION`): the grids make 1*(2+1)+2 = 5 and 1*(1+1)+2 = 4 tokens.
    const GRIDS: [(i64, i64); 2] = [(2, 3), (1, 2)];
    const PATCHES: [i64; 2] = [6, 2];
    const SPANS: [(u32, u32); 2] = [(1, 5), (7, 4)];
    const PROMPT_LEN: usize = 12;
    const ROW_FLOATS: usize = 3 * 2 * 2;
    const VISION: V41Vision = V41Vision {
        patch_size: 2,
        downsample_ratio: 2,
    };

    /// The batch alone, translated for the served model `VISION`.
    fn translate(batch: sglang_proto::MultimodalInputs) -> Result<SglangWireMmInputs, String> {
        translate_sglang_multimodal(
            batch,
            Vec::new(),
            PROMPT_LEN,
            PerImageItems {
                vision: Some(VISION),
            },
        )
    }

    /// `batch` with `rows` patch rows of zeros, each of `row_shape`.
    fn with_pixel_rows(
        mut batch: sglang_proto::MultimodalInputs,
        rows: u32,
        row_shape: &[u32],
    ) -> sglang_proto::MultimodalInputs {
        let shape: Vec<u32> = std::iter::once(rows)
            .chain(row_shape.iter().copied())
            .collect();
        let floats: u32 = shape.iter().product();
        batch.pixel_values = Some(tensor("float32", shape, vec![0; floats as usize * 4]));
        batch
    }

    fn tensor(dtype: &str, shape: Vec<u32>, data: Vec<u8>) -> sglang_proto::TensorData {
        sglang_proto::TensorData {
            data: data.into(),
            shape,
            dtype: dtype.to_string(),
        }
    }

    fn int64(values: &[i64], shape: Vec<u32>) -> sglang_proto::TensorData {
        tensor(
            "int64",
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }

    /// A V4.1 image batch as the Router assembles it: every float of patch
    /// row `i` is `i`.
    fn v41_batch(
        grids: &[(i64, i64)],
        patches: &[i64],
        spans: &[(u32, u32)],
    ) -> sglang_proto::MultimodalInputs {
        let rows: i64 = PATCHES.iter().sum();
        let pixels: Vec<u8> = (0..rows)
            .flat_map(|row| std::iter::repeat_n(row as f32, ROW_FLOATS))
            .flat_map(f32::to_le_bytes)
            .collect();
        let flat_grids: Vec<i64> = grids.iter().flat_map(|&(h, w)| [h, w]).collect();
        let tokens: Vec<i64> = spans.iter().map(|&(_, n)| i64::from(n)).collect();
        let types = vec![0; tokens.iter().sum::<i64>() as usize];
        sglang_proto::MultimodalInputs {
            modalities: vec!["image".to_string()],
            pixel_values: Some(tensor("float32", vec![rows as u32, 3, 2, 2], pixels)),
            model_specific_tensors: [
                ("vit_grid", int64(&flat_grids, vec![grids.len() as u32, 2])),
                ("llm_grid", int64(&flat_grids, vec![grids.len() as u32, 2])),
                (
                    "patches_per_image",
                    int64(patches, vec![patches.len() as u32]),
                ),
                ("types_per_image", int64(&tokens, vec![tokens.len() as u32])),
                ("types", int64(&types, vec![types.len() as u32])),
            ]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
            im_token_id: Some(9),
            mm_placeholders: spans
                .iter()
                .map(|&(offset, length)| sglang_proto::PlaceholderRange { offset, length })
                .collect(),
            ..Default::default()
        }
    }

    /// A video batch, the primary batch of a mixed request.
    fn video_batch() -> sglang_proto::MultimodalInputs {
        sglang_proto::MultimodalInputs {
            modalities: vec!["video".to_string()],
            im_token_id: Some(8),
            pixel_values: Some(tensor("float32", vec![2], vec![0; 8])),
            mm_placeholders: vec![sglang_proto::PlaceholderRange {
                offset: 1,
                length: 3,
            }],
            ..Default::default()
        }
    }

    /// A generate request of `PROMPT_LEN` tokens carrying `batch`.
    fn generate_request(batch: sglang_proto::MultimodalInputs) -> sglang_proto::GenerateRequest {
        sglang_proto::GenerateRequest {
            request_id: "v41".into(),
            tokenized: Some(sglang_proto::TokenizedInput {
                input_ids: (0..PROMPT_LEN as u32).collect(),
                original_text: String::new(),
            }),
            mm_inputs: Some(Box::new(batch)),
            ..Default::default()
        }
    }

    fn scalar_of(item: &super::SglangWireMmItem, key: &str) -> i64 {
        let tensor = &item.model_specific_data[key];
        assert_eq!((tensor.dtype.as_str(), tensor.shape.len()), ("int64", 0));
        let bytes = tensor.data.as_raw_view().unwrap();
        i64::from_le_bytes(bytes.as_ref().try_into().unwrap())
    }

    /// Each image's patches, span and grid make one item SGLang's V4.1 model
    /// can encode; the feature bytes are views of the request's, not copies.
    #[test]
    fn each_image_is_its_own_item() {
        let batch = v41_batch(&GRIDS, &PATCHES, &SPANS);
        let pixels_ptr = batch.pixel_values.as_ref().unwrap().data.as_ptr();
        let output = translate(batch).unwrap();
        assert_eq!(output.im_token_id, Some(9));
        assert_eq!(output.mm_items.len(), 2);
        let row_bytes = ROW_FLOATS * 4;
        for (index, item) in output.mm_items.iter().enumerate() {
            let (first, length) = SPANS[index];
            let (first, length) = (u64::from(first), u64::from(length));
            assert_eq!(item.offsets, Some(vec![(first, first + length - 1)]));
            assert_eq!(item.hash, None);
            assert_eq!(scalar_of(item, "n_vit_h"), GRIDS[index].0);
            assert_eq!(scalar_of(item, "n_vit_w"), GRIDS[index].1);
            assert_eq!(item.model_specific_data.len(), 2);
            let feature = item.feature.as_ref().unwrap();
            assert_eq!(feature.dtype, "float32");
            assert_eq!(feature.shape, vec![PATCHES[index] as u64, 3, 2, 2]);
            let bytes = feature.data.as_raw_view().unwrap();
            let first_row = if index == 0 { 0 } else { PATCHES[0] as usize };
            assert_eq!(
                bytes.as_ptr(),
                pixels_ptr.wrapping_add(first_row * row_bytes)
            );
            let values: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .step_by(ROW_FLOATS)
                .map(|&value| f32::from_le_bytes(value))
                .collect();
            let expected: Vec<f32> = (first_row..first_row + PATCHES[index] as usize)
                .map(|row| row as f32)
                .collect();
            assert_eq!(values, expected);
        }
    }

    /// A batch whose grids, patch counts, token counts and placeholders
    /// disagree never reaches the scheduler.
    #[test]
    fn counts_that_disagree_are_refused() {
        let refused = |batch: sglang_proto::MultimodalInputs, needle: &str| {
            let error = translate(batch).unwrap_err();
            assert!(error.contains(needle), "{error:?} does not name {needle:?}");
        };
        refused(
            v41_batch(&GRIDS, &PATCHES, &SPANS[..1]),
            "2 images but 1 placeholders",
        );
        refused(
            v41_batch(&GRIDS, &[6], &SPANS),
            "2 images but 1 patch counts",
        );
        refused(
            v41_batch(&GRIDS, &[5, 2], &SPANS),
            "7 patches but the pixel tensor has 8 rows",
        );
        refused(
            v41_batch(&[(2, 3), (1, 3)], &PATCHES, &SPANS),
            "image 1 has a 1x3 grid but 2 patches",
        );
        refused(
            v41_batch(&[(2, 3), (-1, -2)], &PATCHES, &SPANS),
            "vit_grid holds a negative value",
        );
        refused(
            v41_batch(
                &[(2, 3), (1, 2), (0, 5)],
                &[6, 2, 0],
                &[(1, 5), (7, 4), (11, 1)],
            ),
            "image 2 has a 0x5 grid but 0 patches",
        );
        let mut batch = v41_batch(&GRIDS, &PATCHES, &SPANS);
        batch
            .model_specific_tensors
            .insert("types_per_image".into(), int64(&[5, 3], vec![2]));
        refused(batch, "image 1 fills 4 prompt positions but has 3 tokens");
        let mut pixel_less = v41_batch(&GRIDS, &PATCHES, &SPANS);
        pixel_less.pixel_values = None;
        pixel_less.feature_hash = Some(7);
        refused(pixel_less.clone(), "no pixel_values");
        pixel_less
            .model_specific_tensors
            .remove("patches_per_image");
        refused(pixel_less, "no pixel_values");
        let mut no_patch_counts = v41_batch(&GRIDS, &PATCHES, &SPANS);
        no_patch_counts
            .model_specific_tensors
            .remove("patches_per_image");
        refused(
            no_patch_counts,
            "DeepSeek-V4.1 image batch: no patches_per_image",
        );
        let mut flat_pixels = v41_batch(&GRIDS, &PATCHES, &SPANS);
        let floats: Vec<u8> = (0..8).flat_map(|row| (row as f32).to_le_bytes()).collect();
        flat_pixels.pixel_values = Some(tensor("float32", vec![8], floats));
        refused(
            flat_pixels,
            "pixel_values has shape [8], not rows of patches",
        );
    }

    /// SGLang writes the items' embeddings over the placeholders in prompt
    /// order: a shared position, or a range before the one of the image
    /// ahead of it, would put one image's embeddings on another's positions.
    /// A range may start right after the one before it ends.
    #[test]
    fn placeholders_out_of_prompt_order_are_refused() {
        for (spans, needle) in [
            (
                [(1, 5), (1, 5)],
                "image 1's placeholder 1..=5 does not start after image 0's, which ends at 5",
            ),
            (
                [(1, 5), (5, 4)],
                "image 1's placeholder 5..=8 does not start after image 0's, which ends at 5",
            ),
            (
                [(7, 5), (1, 4)],
                "image 1's placeholder 1..=4 does not start after image 0's, which ends at 11",
            ),
        ] {
            let batch = v41_batch(&GRIDS, &PATCHES, &spans);
            let error = translate(batch).unwrap_err();
            assert!(error.contains(needle), "{error:?} does not name {needle:?}");
        }
        let touching = v41_batch(&GRIDS, &PATCHES, &[(1, 5), (6, 4)]);
        let output = translate(touching).unwrap();
        let offsets: Vec<_> = output.mm_items.iter().map(|item| &item.offsets).collect();
        assert_eq!(offsets, [&Some(vec![(1, 5)]), &Some(vec![(6, 9)])]);
    }

    /// SGLang's V4.1 patch embedding takes rows of 3 x patch_size x
    /// patch_size values and fails on any other width, in the scheduler.
    #[test]
    fn patch_rows_must_fit_the_models_patches() {
        for (row_shape, values) in [(&[3, 2, 3][..], 18), (&[1][..], 1)] {
            let batch = with_pixel_rows(v41_batch(&GRIDS, &PATCHES, &SPANS), 8, row_shape);
            let error = translate(batch).unwrap_err();
            let needle = format!(
                "each patch row holds {values} values, but the model's 2x2 patches take 3x2x2 = 12"
            );
            assert!(
                error.contains(&needle),
                "{error:?} does not name {needle:?}"
            );
        }
    }

    /// SGLang sizes an image's embedding from its grid alone. A span of
    /// another length is refused, also when the token counts agree with the
    /// spans or are absent. A 1x1 grid makes a start, one token, a newline
    /// and an end.
    #[test]
    fn span_length_must_match_the_grid() {
        let twins = with_pixel_rows(
            v41_batch(&[(2, 3), (2, 3)], &[6, 6], &SPANS),
            12,
            &[3, 2, 2],
        );
        let mut without_counts = twins.clone();
        without_counts
            .model_specific_tensors
            .remove("types_per_image");
        for batch in [twins, without_counts] {
            let error = translate(batch).unwrap_err();
            let needle = "image 1 fills 4 prompt positions but its 2x3 grid makes 5 tokens";
            assert!(error.contains(needle), "{error:?} does not name {needle:?}");
        }
        let one_patch = with_pixel_rows(v41_batch(&[(1, 1)], &[1], &[(2, 4)]), 1, &[3, 2, 2]);
        let output = translate(one_patch).unwrap();
        assert_eq!(output.mm_items[0].offsets, Some(vec![(2, 5)]));
    }

    /// Without the served model's geometry (its scheduler declared none: it
    /// serves another model, or its plugin predates the two fields) the width
    /// and grid checks are skipped.
    #[test]
    fn without_the_models_vision_its_checks_are_skipped() {
        let batch = with_pixel_rows(
            v41_batch(&[(2, 3), (2, 3)], &[6, 6], &SPANS),
            12,
            &[3, 2, 3],
        );
        let output =
            translate_sglang_multimodal(batch, Vec::new(), PROMPT_LEN, PerImageItems::default())
                .unwrap();
        assert_eq!(output.mm_items.len(), 2);
    }

    /// The geometry is what the scheduler declared at its handshake: both
    /// values, each above zero; none when one is missing or zero.
    #[test]
    fn the_models_vision_is_what_the_scheduler_declared() {
        let declared = |patch_size, downsample_ratio| {
            V41Vision::from_handshake(&EngineCoreReadyResponse {
                vision_patch_size: patch_size,
                vision_downsample_ratio: downsample_ratio,
                ..default_ready_response()
            })
        };
        assert_eq!(
            declared(Some(14), Some(3)),
            Some(V41Vision {
                patch_size: 14,
                downsample_ratio: 3,
            })
        );
        for (patch_size, downsample_ratio) in [
            (None, None),
            (Some(14), None),
            (None, Some(3)),
            (Some(0), Some(3)),
            (Some(14), Some(0)),
        ] {
            assert_eq!(
                declared(patch_size, downsample_ratio),
                None,
                "{patch_size:?} {downsample_ratio:?}"
            );
        }
    }

    /// A connection's geometry is the one every rank declared: none when two
    /// ranks declare different values, or one declares none.
    #[test]
    fn the_connections_vision_is_what_every_rank_declared() {
        let vision_of = |declared: &[Option<(u64, u64)>]| {
            let ranks: Vec<_> = declared
                .iter()
                .zip(0..)
                .map(|(geometry, index)| ConnectedEngine {
                    engine_id: EngineId::from_engine_index(index),
                    ready_response: EngineCoreReadyResponse {
                        vision_patch_size: geometry.map(|(patch_size, _)| patch_size),
                        vision_downsample_ratio: geometry.map(|(_, ratio)| ratio),
                        ..default_ready_response()
                    },
                })
                .collect();
            V41Vision::from_handshakes(&ranks)
        };
        assert_eq!(vision_of(&[Some((2, 2))]), Some(VISION));
        assert_eq!(vision_of(&[Some((2, 2)), Some((2, 2))]), Some(VISION));
        for differing in [
            &[Some((2, 2)), Some((4, 2))][..],
            &[Some((2, 2)), Some((2, 4))],
            &[Some((2, 2)), None],
            &[None, Some((2, 2))],
            &[],
        ] {
            assert_eq!(vision_of(differing), None, "{differing:?}");
        }
    }

    /// The geometry a batch is checked against is the scheduler's own,
    /// whatever the model id is (here not a local directory): a scheduler
    /// that declared 2x2 patches has a batch of wider rows refused before it
    /// is sent, and a fitting batch sent on as two items; a scheduler that
    /// declared nothing has the wider rows sent on.
    #[tokio::test]
    async fn a_batch_is_checked_against_the_geometry_its_scheduler_declared() {
        let request = generate_request;
        let wide = || with_pixel_rows(v41_batch(&GRIDS, &PATCHES, &SPANS), 8, &[3, 2, 3]);
        let declared = EngineCoreReadyResponse {
            vision_patch_size: Some(2),
            vision_downsample_ratio: Some(2),
            ..default_ready_response()
        };
        for (ready, checked) in [(declared, true), (default_ready_response(), false)] {
            let (client, engine, _dir) = connected_pair_declaring(RuntimeType::Sglang, ready).await;
            let (mut input, _output) = engine.split();
            let sent = if checked {
                let refused = client
                    .generate_sglang(request(wide()))
                    .await
                    .err()
                    .expect("rows the declared patches do not take are refused");
                assert_eq!(refused.code(), tonic::Code::InvalidArgument);
                assert!(
                    refused.message().contains(
                        "each patch row holds 18 values, but the model's 2x2 patches take 3x2x2 = \
                         12"
                    ),
                    "{refused:?}"
                );
                v41_batch(&GRIDS, &PATCHES, &SPANS)
            } else {
                wide()
            };
            let (stream, frames) =
                tokio::join!(client.generate_sglang(request(sent)), input.recv_frames());
            let stream = stream.expect("sent on");
            let wire: WireRequest = decode_msgpack(frames.unwrap()[1].as_ref()).unwrap();
            let items = wire.mm_inputs.expect("the processor output").mm_items;
            assert_eq!(items.len(), 2, "checked={checked}");
            drop(stream);
        }
    }

    /// Ranks that declare different geometry leave a batch's rows and spans
    /// unchecked, whichever rank takes it: rows the first rank's patches do
    /// not take are sent on, here to the second rank, which declared nothing.
    #[tokio::test]
    async fn ranks_that_declare_different_geometry_skip_the_geometry_checks() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let declared = EngineCoreReadyResponse {
            vision_patch_size: Some(2),
            vision_downsample_ratio: Some(2),
            ..default_ready_response()
        };
        let (client, _first, second) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                2,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Sglang,
                Duration::from_secs(10)
            ),
            connect_to_frontend(&handshake, EngineId::from_engine_index(0), declared),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(1),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let (mut input, _output) = second.expect("mock engine").split();
        let wide = with_pixel_rows(v41_batch(&GRIDS, &PATCHES, &SPANS), 8, &[3, 2, 3]);
        let request = sglang_proto::GenerateRequest {
            data_parallel_rank: Some(1),
            ..generate_request(wide)
        };
        let (stream, frames) = tokio::join!(client.generate_sglang(request), input.recv_frames());
        let _stream = stream.expect("sent on");
        let wire: WireRequest = decode_msgpack(frames.unwrap()[1].as_ref()).unwrap();
        assert_eq!(wire.mm_inputs.expect("the items").mm_items.len(), 2);
    }

    /// A patch row holds at least one value. A pixel tensor with a row
    /// dimension of zero is refused when the model's geometry is not known,
    /// where an empty tensor would otherwise reach the scheduler.
    #[test]
    fn patch_rows_without_values_are_refused() {
        for row_shape in [&[0][..], &[3, 0, 2]] {
            let batch = with_pixel_rows(v41_batch(&GRIDS, &PATCHES, &SPANS), 8, row_shape);
            let error = translate_sglang_multimodal(
                batch,
                Vec::new(),
                PROMPT_LEN,
                PerImageItems::default(),
            )
            .unwrap_err();
            assert!(
                error.contains("not rows of patches"),
                "{row_shape:?}: {error:?}"
            );
        }
    }

    /// Only an image batch is cut into V4.1 items. A batch that carries
    /// `vit_grid` under the video label would go on as a video item, which
    /// SGLang's V4.1 model has no embedder for, so it is refused: alone, and
    /// as the extra batch after an image batch.
    #[test]
    fn a_grid_on_a_video_batch_is_refused() {
        let mut video = v41_batch(&GRIDS, &PATCHES, &SPANS);
        video.modalities = vec!["video".to_string()];
        let needle = "DeepSeek-V4.1 image batch: vit_grid on a video batch";
        let error = translate(video.clone()).unwrap_err();
        assert!(error.contains(needle), "{error:?}");
        let image = v41_batch(&GRIDS, &PATCHES, &SPANS);
        let error = translate_sglang_multimodal(
            image,
            vec![video],
            PROMPT_LEN,
            PerImageItems {
                vision: Some(VISION),
            },
        )
        .unwrap_err();
        assert!(error.contains(needle), "{error:?}");
    }

    /// SGLang pads the image positions only when the request names the image
    /// token id. Without it the model finds no position for the items'
    /// embeddings and fails in the scheduler, so the batch is refused. A
    /// batch without `vit_grid` is sent on without one, as before.
    #[test]
    fn a_batch_without_its_image_token_id_is_refused() {
        let mut batch = v41_batch(&GRIDS, &PATCHES, &SPANS);
        batch.im_token_id = None;
        let needle = "DeepSeek-V4.1 image batch: no im_token_id";
        let error = translate(batch.clone()).unwrap_err();
        assert!(error.contains(needle), "{error:?}");
        // Also as the extra batch of a mixed request, after a video batch.
        let extras = vec![batch.clone()];
        let error = translate_sglang_multimodal(
            video_batch(),
            extras,
            PROMPT_LEN,
            PerImageItems {
                vision: Some(VISION),
            },
        )
        .unwrap_err();
        assert!(error.contains(needle), "{error:?}");
        batch.model_specific_tensors.remove("vit_grid");
        let output = translate(batch).unwrap();
        assert_eq!((output.mm_items.len(), output.im_token_id), (1, None));
    }

    /// Empty grids make no items. Refused alone, and refused, not a panic,
    /// as the extra batch after a video batch, where the caller reads the
    /// first item of each batch.
    #[test]
    fn a_batch_without_images_is_refused() {
        let empty = v41_batch(&[], &[], &[]);
        let error = translate(empty.clone()).unwrap_err();
        assert!(
            error.contains("DeepSeek-V4.1 image batch: no images"),
            "{error:?}"
        );
        let error = translate_sglang_multimodal(
            video_batch(),
            vec![empty],
            PROMPT_LEN,
            PerImageItems {
                vision: Some(VISION),
            },
        )
        .unwrap_err();
        assert!(error.contains("no images"), "{error:?}");
    }

    /// The token counts are optional, as in the Python servicer; a batch hash
    /// is not passed on to items it no longer names.
    #[test]
    fn without_token_counts_or_with_a_batch_hash_the_items_are_the_same() {
        let mut batch = v41_batch(&GRIDS, &PATCHES, &SPANS);
        batch.model_specific_tensors.remove("types_per_image");
        batch.feature_hash = Some(7);
        let output = translate(batch).unwrap();
        assert_eq!(output.mm_items.len(), 2);
        assert!(output.mm_items.iter().all(|item| item.hash.is_none()));
    }

    /// A batch without `vit_grid` stays one item: a Qwen-style
    /// `image_grid_thw`, or GLM-5.3 Flash's, which carries `patches_per_image`
    /// as well.
    #[test]
    fn other_families_keep_one_item_per_batch() {
        let qwen = [(
            "image_grid_thw".to_string(),
            int64(&[1, 2, 2, 1, 2, 2], vec![2, 3]),
        )];
        let mut glm53_flash = qwen.to_vec();
        glm53_flash.push(("patches_per_image".to_string(), int64(&[4, 4], vec![2])));
        for tensors in [qwen.to_vec(), glm53_flash] {
            let mut batch = v41_batch(&GRIDS, &PATCHES, &SPANS);
            batch.model_specific_tensors = tensors.iter().cloned().collect();
            let output = translate(batch).unwrap();
            assert_eq!(output.mm_items.len(), 1);
            assert_eq!(output.mm_items[0].offsets, Some(vec![(1, 5), (7, 10)]));
            assert!(tensors
                .iter()
                .all(|(key, _)| output.mm_items[0].model_specific_data.contains_key(key)));
        }
    }
}
