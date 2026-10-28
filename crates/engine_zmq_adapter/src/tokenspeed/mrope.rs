//! The MRoPE position table of the Qwen3.5 lineage, computed as the engine's
//! own input processor computes it on the path this wire bypasses.
//!
//! An MRoPE language model (a text rope config with an `mrope_section`) reads
//! three position axes per token: text runs advance all three linearly, an
//! image or video frame lays its merged patch grid out over (time, height,
//! width). The engine derives the table in its frontend from the un-padded
//! prompt, the items' `image_grid_thw` / `video_grid_thw` rows and their
//! placeholder spans (`runtime/multimodal/mrope.py`,
//! `_compute_qwen35_mrope_positions_from_offsets`): nothing between the
//! msgpack wire and the executor repeats that step, and an empty table means
//! the executor falls back to one-dimensional positions without a word. This
//! module is that step, rule for rule, so the table on the wire is the one the
//! engine would have written.

use engine_zmq_client::codec::tensor::WireTensor;
use smg_grpc_client::tokenspeed_proto;

/// The checkpoint facts the rule needs besides the request: the vision
/// tower's spatial merge (patches per position along each axis) and the two
/// placeholder token ids the engine checks each span against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MropeLayout {
    pub spatial_merge_size: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
}

/// Which grid an item carries, by the key the engine looks for first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GridKind {
    Image,
    Video,
}

impl GridKind {
    pub(crate) const KEYS: [(&'static str, GridKind); 2] = [
        ("image_grid_thw", GridKind::Image),
        ("video_grid_thw", GridKind::Video),
    ];
}

/// One item's inputs to the rule: its grid rows `(t, h, w)` as the tensor
/// carries them (`None` for an item without a grid, which contributes
/// nothing) and its inclusive placeholder spans in prompt order.
#[derive(Debug)]
pub(crate) struct GridItem<'a> {
    pub grid: Option<(GridKind, Vec<[i64; 3]>)>,
    pub offsets: &'a [(u64, u64)],
}

/// The three fields the engine's input processor writes: the table `(3, L)`
/// int64, the delta `(1, 1)` int64 and the delta as an integer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MropeTable {
    pub positions: WireTensor,
    pub delta: WireTensor,
    pub scalar: i64,
}

/// The architectures the engine dispatches to the Qwen3.5 offsets rule
/// (its `_MROPE_ARCHITECTURES` minus Qwen3-Omni, whose rule differs), with
/// the model types that rule accepts.
const QWEN35_ARCHITECTURES: [&str; 3] = [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen4ExpForConditionalGeneration",
];
const QWEN35_MODEL_TYPES: [&str; 3] = ["qwen3_5", "qwen3_5_moe", "qwen4_exp"];

/// Whether the engine computes this checkpoint's positions with the Qwen3.5
/// offsets rule: one of its architectures with one of its model types.
pub fn qwen35_lineage(architectures: &[String], model_type: &str) -> bool {
    architectures
        .iter()
        .any(|arch| QWEN35_ARCHITECTURES.contains(&arch.as_str()))
        && QWEN35_MODEL_TYPES.contains(&model_type)
}

/// Read a grid tensor's `(t, h, w)` rows: shape `[n, 3]` or `[3]`, an
/// integer dtype of either width. The engine reads the rows with `tolist()`,
/// whatever the integer dtype.
pub(crate) fn grid_rows(
    key: &str,
    tensor: &tokenspeed_proto::TensorData,
) -> Result<Vec<[i64; 3]>, String> {
    let shape: Vec<usize> = tensor.shape.iter().map(|&d| d as usize).collect();
    let rows = match shape.as_slice() {
        [3] => 1,
        [n, 3] => *n,
        other => {
            return Err(format!(
                "multimodal item's {key} has shape {other:?}; a grid is [n, 3] rows of (t, h, w)"
            ))
        }
    };
    let bytes = match &tensor.payload {
        Some(tokenspeed_proto::tensor_data::Payload::Inline(bytes)) => bytes,
        _ => return Err(format!("multimodal item's {key} carries no inline payload")),
    };
    let values: Vec<i64> = match tensor.dtype.as_str() {
        "int64" => elements::<8>(bytes, key)?
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect(),
        "uint64" => elements::<8>(bytes, key)?
            .iter()
            .map(|c| i64::try_from(u64::from_le_bytes(*c)))
            .collect::<Result<_, _>>()
            .map_err(|_| format!("multimodal item's {key} holds a value above i64"))?,
        "int32" => elements::<4>(bytes, key)?
            .iter()
            .map(|c| i64::from(i32::from_le_bytes(*c)))
            .collect(),
        "uint32" => elements::<4>(bytes, key)?
            .iter()
            .map(|c| i64::from(u32::from_le_bytes(*c)))
            .collect(),
        other => {
            return Err(format!(
                "multimodal item's {key} has dtype {other:?}; a grid is an integer tensor"
            ))
        }
    };
    if values.len() != rows * 3 {
        return Err(format!(
            "multimodal item's {key} holds {} values for shape {shape:?}",
            values.len()
        ));
    }
    Ok(values.as_chunks::<3>().0.to_vec())
}

/// The tensor's bytes as `WIDTH`-byte elements; a length that is not a whole
/// number of elements is a malformed tensor.
fn elements<'a, const WIDTH: usize>(
    bytes: &'a [u8],
    key: &str,
) -> Result<&'a [[u8; WIDTH]], String> {
    let (elements, rest) = bytes.as_chunks::<WIDTH>();
    if !rest.is_empty() {
        return Err(format!(
            "multimodal item's {key} byte length {} is not a multiple of its element width {WIDTH}",
            bytes.len()
        ));
    }
    Ok(elements)
}

/// A placeholder span with the merged grid it covers.
#[derive(Debug, Clone, Copy)]
struct Segment {
    start: usize,
    end: usize,
    t: i64,
    h: i64,
    w: i64,
}

/// The Qwen3.5 rule over the un-padded prompt: the engine's
/// `_compute_qwen35_mrope_positions_from_offsets` followed by
/// `_build_qwen35_mrope_positions`. Every precondition under which the engine
/// would abandon this rule (and rescan the prompt) is an error naming it: this
/// path serves the spans the gateway's media pipeline builds, which meet them,
/// and refuses anything else rather than guess. A request whose items carry no
/// grid gets what the engine's rescan gives such a prompt: linear positions on
/// all three axes and a zero delta.
pub(crate) fn qwen35_positions(
    layout: MropeLayout,
    input_ids: &[u32],
    items: &[GridItem<'_>],
) -> Result<MropeTable, String> {
    let seq_len = input_ids.len();
    if seq_len == 0 {
        return Err("MRoPE positions need a non-empty prompt".to_string());
    }
    let merge = i64::from(layout.spatial_merge_size);
    if merge <= 0 {
        return Err("MRoPE positions need a positive spatial merge size".to_string());
    }
    let mut segments: Vec<Segment> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let Some((kind, rows)) = &item.grid else {
            continue;
        };
        if item.offsets.is_empty() {
            return Err(format!(
                "multimodal item {index} carries a grid but no placeholder spans"
            ));
        }
        // A video's (T, H, W) row stands for T frames, one span each.
        let (rows, placeholder): (Vec<[i64; 3]>, u32) = match kind {
            GridKind::Image => (rows.clone(), layout.image_token_id),
            GridKind::Video => {
                // The frame count first, so a malformed row cannot ask for a
                // frame list the size of its `t`: the frames must equal the
                // spans before any list exists.
                let frames = rows.iter().enumerate().try_fold(0usize, |sum, (row, &[t, ..])| {
                    let t = usize::try_from(t).map_err(|_| {
                        format!(
                            "multimodal item {index} video grid row {row} has a negative frame count {t}"
                        )
                    })?;
                    sum.checked_add(t).ok_or_else(|| {
                        format!("multimodal item {index} video grid frame count overflows")
                    })
                })?;
                if frames != item.offsets.len() {
                    return Err(format!(
                        "multimodal item {index} has {frames} grid row(s) for {} placeholder span(s)",
                        item.offsets.len()
                    ));
                }
                (
                    rows.iter()
                        .flat_map(|&[t, h, w]| {
                            std::iter::repeat_n([1, h, w], usize::try_from(t).unwrap_or(0))
                        })
                        .collect(),
                    layout.video_token_id,
                )
            }
        };
        if rows.len() != item.offsets.len() {
            return Err(format!(
                "multimodal item {index} has {} grid row(s) for {} placeholder span(s)",
                rows.len(),
                item.offsets.len()
            ));
        }
        for (&(start, end), [t, h, w]) in item.offsets.iter().zip(rows) {
            let (start, end) = (start as usize, end as usize);
            if end < start || end >= seq_len {
                return Err(format!(
                    "multimodal item {index} placeholder span [{start}, {end}] does not fit the \
                     {seq_len}-token prompt"
                ));
            }
            if input_ids[start..=end].iter().any(|&id| id != placeholder) {
                return Err(format!(
                    "multimodal item {index} placeholder span [{start}, {end}] holds a token other \
                     than the placeholder id {placeholder}"
                ));
            }
            let (lh, lw) = (h / merge, w / merge);
            if t <= 0 || lh <= 0 || lw <= 0 {
                return Err(format!(
                    "multimodal item {index} grid ({t}, {h}, {w}) is empty under spatial merge {merge}"
                ));
            }
            let expected = t
                .checked_mul(lh)
                .and_then(|n| n.checked_mul(lw))
                .ok_or_else(|| format!("multimodal item {index} grid ({t}, {h}, {w}) overflows"))?;
            if (end - start + 1) as i64 != expected {
                return Err(format!(
                    "multimodal item {index} placeholder span [{start}, {end}] has {} tokens; its \
                     grid ({t}, {h}, {w}) under spatial merge {merge} needs {expected}",
                    end - start + 1
                ));
            }
            segments.push(Segment {
                start,
                end,
                t,
                h: lh,
                w: lw,
            });
        }
    }
    segments.sort_by_key(|segment| segment.start);
    let mut prev_end: Option<usize> = None;
    for segment in &segments {
        if prev_end.is_some_and(|prev| segment.start <= prev) {
            return Err("multimodal placeholder spans overlap".to_string());
        }
        prev_end = Some(segment.end);
    }
    build_table(seq_len, &segments)
}

/// `_build_qwen35_mrope_positions`: text runs linear from the running
/// position, a grid laid out row-major over (t, h, w) from it, the running
/// position advanced by the grid's largest side.
fn build_table(seq_len: usize, segments: &[Segment]) -> Result<MropeTable, String> {
    /// A text run: the same linear positions on all three axes.
    fn fill_text(axes: &mut [&mut [i64]; 3], from: usize, to: usize, next_pos: &mut i64) {
        for pos in from..to {
            let value = *next_pos + (pos - from) as i64;
            for axis in axes.iter_mut() {
                axis[pos] = value;
            }
        }
        *next_pos += (to - from) as i64;
    }
    let mut positions = vec![0i64; 3 * seq_len];
    let (t_axis, rest) = positions.split_at_mut(seq_len);
    let (h_axis, w_axis) = rest.split_at_mut(seq_len);
    let mut axes = [t_axis, h_axis, w_axis];
    let mut next_pos: i64 = 0;
    let mut cursor = 0usize;
    for segment in segments {
        if cursor < segment.start {
            fill_text(&mut axes, cursor, segment.start, &mut next_pos);
        }
        let (h, w) = (segment.h as usize, segment.w as usize);
        for (offset, pos) in (segment.start..=segment.end).enumerate() {
            axes[0][pos] = next_pos + (offset / (h * w)) as i64;
            axes[1][pos] = next_pos + ((offset / w) % h) as i64;
            axes[2][pos] = next_pos + (offset % w) as i64;
        }
        next_pos += segment.t.max(segment.h).max(segment.w);
        cursor = segment.end + 1;
    }
    if cursor < seq_len {
        fill_text(&mut axes, cursor, seq_len, &mut next_pos);
    }
    let scalar = next_pos - seq_len as i64;
    Ok(MropeTable {
        positions: WireTensor::from_i64(vec![3, seq_len], positions)?,
        delta: WireTensor::from_i64(vec![1, 1], vec![scalar])?,
        scalar,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The engine's own test checkpoint (`test/runtime/test_multimodal_mrope.py`).
    pub(crate) const LAYOUT: MropeLayout = MropeLayout {
        spatial_merge_size: 2,
        image_token_id: 151655,
        video_token_id: 151656,
    };
    const VISION_START: u32 = 151652;
    const VISION_END: u32 = 151653;

    pub(crate) fn rows_of(table: &MropeTable) -> Vec<Vec<i64>> {
        let bytes = table.positions.data.as_raw_view().expect("inline table");
        let values: Vec<i64> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect();
        let len = table.positions.shape[1];
        assert_eq!(table.positions.shape, vec![3, len]);
        assert_eq!(table.positions.dtype, "int64");
        values.chunks_exact(len).map(<[i64]>::to_vec).collect()
    }

    fn delta_of(table: &MropeTable) -> i64 {
        assert_eq!(table.delta.shape, vec![1, 1]);
        assert_eq!(table.delta.dtype, "int64");
        let bytes = table.delta.data.as_raw_view().expect("inline delta");
        i64::from_le_bytes(bytes.as_chunks::<8>().0[0])
    }

    /// An item's grid (if any) and its spans, owned, for the cases below.
    type Spec = (Option<(GridKind, Vec<[i64; 3]>)>, Vec<(u64, u64)>);

    fn image(rows: &[[i64; 3]], offsets: &[(u64, u64)]) -> Spec {
        (Some((GridKind::Image, rows.to_vec())), offsets.to_vec())
    }

    fn video(rows: &[[i64; 3]], offsets: &[(u64, u64)]) -> Spec {
        (Some((GridKind::Video, rows.to_vec())), offsets.to_vec())
    }

    fn table(input_ids: &[u32], items: &[Spec]) -> Result<MropeTable, String> {
        let items: Vec<GridItem<'_>> = items
            .iter()
            .map(|(grid, offsets)| GridItem {
                grid: grid.clone(),
                offsets,
            })
            .collect();
        qwen35_positions(LAYOUT, input_ids, &items)
    }

    fn repeated(id: u32, n: usize) -> Vec<u32> {
        vec![id; n]
    }

    /// The engine's image case: a (1, 4, 6) grid over six tokens after a
    /// text token and the vision start; the engine's function gives this
    /// table (its test pins the fast path to the rescan on it).
    #[test]
    fn an_image_grid_lays_out_over_time_height_and_width() {
        let mut ids = vec![10, VISION_START];
        ids.extend(repeated(LAYOUT.image_token_id, 6));
        ids.push(11);
        let table = table(&ids, &[image(&[[1, 4, 6]], &[(2, 7)])]).unwrap();
        assert_eq!(
            rows_of(&table),
            vec![
                vec![0, 1, 2, 2, 2, 2, 2, 2, 5],
                vec![0, 1, 2, 2, 2, 3, 3, 3, 5],
                vec![0, 1, 2, 3, 4, 2, 3, 4, 5],
            ]
        );
        assert_eq!(delta_of(&table), -3);
        assert_eq!(table.scalar, -3);
    }

    /// The engine's timestamped-video case, its expected table verbatim: a
    /// (2, 4, 4) video is two frames of (1, 4, 4), each its own span.
    #[test]
    fn a_video_row_is_one_frame_per_temporal_step() {
        let mut ids = vec![VISION_START, 42];
        ids.extend(repeated(LAYOUT.video_token_id, 4));
        ids.extend([VISION_END, 42, VISION_START]);
        ids.extend(repeated(LAYOUT.video_token_id, 4));
        ids.push(VISION_END);
        let table = table(&ids, &[video(&[[2, 4, 4]], &[(2, 5), (9, 12)])]).unwrap();
        assert_eq!(
            rows_of(&table),
            vec![
                vec![0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 7, 7, 7, 9],
                vec![0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 8, 9],
                vec![0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 7, 8, 9],
            ]
        );
        assert_eq!(delta_of(&table), -4);
    }

    /// The engine's mixed case: an image then a two-frame video; the running
    /// position advances by each grid's largest side.
    #[test]
    fn mixed_items_advance_by_each_grids_largest_side() {
        let mut ids = vec![10, VISION_START];
        ids.extend(repeated(LAYOUT.image_token_id, 4));
        ids.extend([11, VISION_START]);
        ids.extend(repeated(LAYOUT.video_token_id, 4));
        ids.extend([12, VISION_START]);
        ids.extend(repeated(LAYOUT.video_token_id, 4));
        ids.push(13);
        let table = table(
            &ids,
            &[
                image(&[[1, 4, 4]], &[(2, 5)]),
                video(&[[2, 4, 4]], &[(8, 11), (14, 17)]),
            ],
        )
        .unwrap();
        let rows = rows_of(&table);
        // Text before the image: 0, 1; the image at 2 (2x2); text resumes at 4.
        assert_eq!(&rows[0][..8], &[0, 1, 2, 2, 2, 2, 4, 5]);
        assert_eq!(&rows[2][..8], &[0, 1, 2, 3, 2, 3, 4, 5]);
        // The first frame at 6, text at 8, the second frame at 10, text at 12.
        assert_eq!(&rows[0][8..], &[6, 6, 6, 6, 8, 9, 10, 10, 10, 10, 12]);
        assert_eq!(&rows[1][8..], &[6, 6, 7, 7, 8, 9, 10, 10, 11, 11, 12]);
        assert_eq!(delta_of(&table), 13 - 19);
    }

    /// Items in reverse prompt order are sorted, as the engine sorts them.
    #[test]
    fn items_out_of_prompt_order_are_sorted() {
        let mut ids = repeated(LAYOUT.image_token_id, 4);
        ids.push(7);
        ids.extend(repeated(LAYOUT.image_token_id, 4));
        let sorted = table(
            &ids,
            &[
                image(&[[1, 4, 4]], &[(0, 3)]),
                image(&[[1, 4, 4]], &[(5, 8)]),
            ],
        )
        .unwrap();
        let reversed = table(
            &ids,
            &[
                image(&[[1, 4, 4]], &[(5, 8)]),
                image(&[[1, 4, 4]], &[(0, 3)]),
            ],
        )
        .unwrap();
        assert_eq!(sorted, reversed);
        assert_eq!(rows_of(&sorted)[0], vec![0, 0, 0, 0, 2, 3, 3, 3, 3]);
    }

    /// No grid on any item: the engine's rescan of such a prompt gives linear
    /// positions and a zero delta, and so does this.
    #[test]
    fn a_prompt_without_grids_gets_linear_positions_and_a_zero_delta() {
        let table = table(&[5, 6, 7], &[(None, vec![(1, 1)])]).unwrap();
        assert_eq!(rows_of(&table), vec![vec![0, 1, 2]; 3]);
        assert_eq!(delta_of(&table), 0);
        assert_eq!(table.scalar, 0);
    }

    /// Each precondition under which the engine abandons the rule is a
    /// refusal naming it.
    #[test]
    fn the_rules_preconditions_are_refused_by_name() {
        let mut ids = vec![10];
        ids.extend(repeated(LAYOUT.image_token_id, 4));
        ids.push(11);
        let err = |items: &[Spec]| table(&ids, items).unwrap_err();
        assert!(err(&[(Some((GridKind::Image, vec![[1, 4, 4]])), vec![])])
            .contains("no placeholder spans"));
        assert!(err(&[image(&[[1, 4, 4], [1, 4, 4]], &[(1, 4)])]).contains("2 grid row(s) for 1"));
        assert!(err(&[image(&[[1, 4, 4]], &[(1, 6)])]).contains("does not fit"));
        assert!(err(&[image(&[[1, 4, 4]], &[(0, 3)])]).contains("other than the placeholder id"));
        assert!(err(&[image(&[[1, 4, 6]], &[(1, 4)])]).contains("needs 6"));
        assert!(err(&[image(&[[1, 1, 4]], &[(1, 4)])]).contains("is empty under spatial merge"));
        assert!(err(&[
            image(&[[1, 2, 4]], &[(1, 2)]),
            image(&[[1, 2, 4]], &[(2, 3)])
        ])
        .contains("overlap"));
        assert!(table(&[], &[]).unwrap_err().contains("non-empty prompt"));
        // A video frame count of zero leaves no rows for its span.
        assert!(err(&[video(&[[0, 4, 4]], &[(1, 4)])]).contains("0 grid row(s) for 1"));
        // A video row asking for more frames than there are spans is refused
        // before any frame list exists (a count, not an allocation).
        assert!(err(&[video(&[[1 << 40, 4, 4]], &[(1, 4)])]).contains("grid row(s) for 1"));
        // A negative frame count is named with its row, not read as an overflow.
        assert!(
            err(&[video(&[[-1, 4, 4]], &[(1, 4)])]).contains("row 0 has a negative frame count -1")
        );
        assert!(err(&[video(&[[1, 4, 4], [-2, 4, 4]], &[(1, 4), (6, 9)])])
            .contains("video grid row 1 has a negative frame count -2"));
        // A grid whose token count overflows is refused, not wrapped.
        assert!(
            err(&[image(&[[1, 22, 2 * 0x2E8B_A2E8_BA2E_8BA3]], &[(1, 4)])]).contains("overflows")
        );
    }

    #[test]
    fn grid_rows_read_either_integer_width_and_either_rank() {
        let tensor = |dtype: &str, shape: Vec<u32>, bytes: Vec<u8>| tokenspeed_proto::TensorData {
            shape,
            dtype: dtype.to_string(),
            payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(bytes.into())),
        };
        let i64_bytes: Vec<u8> = [1i64, 4, 6, 2, 8, 8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            grid_rows("image_grid_thw", &tensor("int64", vec![2, 3], i64_bytes)).unwrap(),
            vec![[1, 4, 6], [2, 8, 8]]
        );
        let u32_bytes: Vec<u8> = [1u32, 4, 6].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            grid_rows(
                "image_grid_thw",
                &tensor("uint32", vec![3], u32_bytes.clone())
            )
            .unwrap(),
            vec![[1, 4, 6]]
        );
        assert!(grid_rows(
            "image_grid_thw",
            &tensor("uint32", vec![1, 2], u32_bytes.clone())
        )
        .unwrap_err()
        .contains("shape"));
        assert!(grid_rows(
            "image_grid_thw",
            &tensor("float32", vec![1, 3], u32_bytes.clone())
        )
        .unwrap_err()
        .contains("dtype"));
        assert!(grid_rows(
            "image_grid_thw",
            &tensor("uint32", vec![1, 3], u32_bytes[..7].to_vec())
        )
        .unwrap_err()
        .contains("multiple"));
        assert!(grid_rows(
            "image_grid_thw",
            &tensor("uint32", vec![1, 3], u32_bytes[..8].to_vec())
        )
        .unwrap_err()
        .contains("holds 2 values"));
    }

    #[test]
    fn the_lineage_is_the_engines_dispatch() {
        let arch = |a: &str| vec![a.to_string()];
        assert!(qwen35_lineage(
            &arch("Qwen3_5ForConditionalGeneration"),
            "qwen3_5"
        ));
        assert!(qwen35_lineage(
            &arch("Qwen3_5MoeForConditionalGeneration"),
            "qwen3_5_moe"
        ));
        assert!(qwen35_lineage(
            &arch("Qwen4ExpForConditionalGeneration"),
            "qwen4_exp"
        ));
        // Qwen3-Omni has its own rule; a Qwen3-VL checkpoint is outside the
        // engine's set; an architecture without its model type is not the rule.
        assert!(!qwen35_lineage(
            &arch("Qwen3OmniMoeForConditionalGeneration"),
            "qwen3_omni_moe"
        ));
        assert!(!qwen35_lineage(
            &arch("Qwen3VLForConditionalGeneration"),
            "qwen3_vl"
        ));
        assert!(!qwen35_lineage(
            &arch("Qwen3_5ForConditionalGeneration"),
            "qwen3_vl"
        ));
        assert!(!qwen35_lineage(&[], "qwen3_5"));
    }

    /// The cost of the rule per request on one core, printed as a table
    /// (`--ignored --nocapture`, a release build, the core pinned): the
    /// shapes the gateway's media pipeline builds, from one small image to a
    /// prompt that fills a window with a large image.
    #[test]
    #[ignore = "a timing table, run alone on a pinned core"]
    #[expect(clippy::print_stdout)]
    fn speed_table() {
        let shapes: [(&str, usize, Vec<[i64; 3]>, usize); 5] = [
            (
                "1 image 16x16 merged (256 tokens), 64 text",
                64,
                vec![[1, 32, 32]],
                64,
            ),
            (
                "1 image 36x36 merged (1296 tokens), 256 text",
                256,
                vec![[1, 72, 72]],
                256,
            ),
            (
                "4 images 16x16 merged, 128 text between",
                128,
                vec![[1, 32, 32]; 4],
                128,
            ),
            (
                "1 grid of 8 temporal steps x 12x12 merged (1152 tokens), 16 text",
                16,
                vec![[8, 24, 24]],
                16,
            ),
            (
                "1 image 60x60 merged (3600 tokens) in a 7.6k prompt",
                2000,
                vec![[1, 120, 120]],
                2000,
            ),
        ];
        println!("| shape | prompt tokens | per request |");
        println!("|---|---|---|");
        for (name, lead, rows, tail) in shapes {
            let mut ids: Vec<u32> = (0..lead as u32).map(|i| 1000 + i).collect();
            let mut items = Vec::new();
            for row in &rows {
                let [t, h, w] = *row;
                let n = (t * (h / 2) * (w / 2)) as usize;
                let start = ids.len();
                ids.extend(std::iter::repeat_n(LAYOUT.image_token_id, n));
                ids.extend((0..64u32).map(|i| 2000 + i));
                items.push((vec![*row], vec![(start as u64, (start + n - 1) as u64)]));
            }
            ids.extend((0..tail as u32).map(|i| 3000 + i));
            let grid_items: Vec<GridItem<'_>> = items
                .iter()
                .map(|(rows, offsets)| GridItem {
                    grid: Some((GridKind::Image, rows.clone())),
                    offsets,
                })
                .collect();
            let iterations = 2000;
            let started = std::time::Instant::now();
            for _ in 0..iterations {
                let table = qwen35_positions(LAYOUT, &ids, &grid_items).unwrap();
                std::hint::black_box(table);
            }
            let per = started.elapsed() / iterations;
            println!(
                "| {name} | {} | {:.1} us |",
                ids.len(),
                per.as_secs_f64() * 1e6
            );
        }
    }

    /// The parity corpus: a JSON file of cases with the engine's own tables
    /// (`SMG_TOKENSPEED_MROPE_CORPUS=<file>`, produced by running the engine's
    /// `compute_mrope_positions` from its source on CPU). Every table this
    /// rule produces must match the engine's bit for bit; a case it refuses
    /// must be one the engine answers by rescanning the prompt (its fast path
    /// abandoned).
    #[test]
    #[ignore = "set SMG_TOKENSPEED_MROPE_CORPUS to the engine-oracle corpus file"]
    #[expect(clippy::print_stderr)]
    fn the_engine_oracle_corpus_matches_bit_for_bit() {
        let path = std::env::var("SMG_TOKENSPEED_MROPE_CORPUS").expect("corpus path");
        let corpus: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("corpus file"))
                .expect("corpus json");
        let config = &corpus["config"];
        let layout = MropeLayout {
            spatial_merge_size: config["spatial_merge_size"].as_u64().unwrap() as u32,
            image_token_id: config["image_token_id"].as_u64().unwrap() as u32,
            video_token_id: config["video_token_id"].as_u64().unwrap() as u32,
        };
        let mut checked = 0;
        let mut refused = 0;
        for case in corpus["cases"].as_array().unwrap() {
            let ids: Vec<u32> = case["input_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let offsets: Vec<Vec<(u64, u64)>> = case["items"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| {
                    item["offsets"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|o| (o[0].as_u64().unwrap(), o[1].as_u64().unwrap()))
                        .collect()
                })
                .collect();
            let items: Vec<GridItem<'_>> = case["items"]
                .as_array()
                .unwrap()
                .iter()
                .zip(&offsets)
                .map(|(item, offsets)| {
                    let kind = match item["modality"].as_str().unwrap() {
                        "image" => GridKind::Image,
                        _ => GridKind::Video,
                    };
                    let rows = item["grid"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|r| {
                            [
                                r[0].as_i64().unwrap(),
                                r[1].as_i64().unwrap(),
                                r[2].as_i64().unwrap(),
                            ]
                        })
                        .collect();
                    GridItem {
                        grid: Some((kind, rows)),
                        offsets,
                    }
                })
                .collect();
            let expected = &case["expected"];
            let id = case["id"].as_str().unwrap_or("?");
            match qwen35_positions(layout, &ids, &items) {
                Ok(table) => {
                    let rows: Vec<Vec<i64>> = expected["positions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|row| {
                            row.as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_i64().unwrap())
                                .collect()
                        })
                        .collect();
                    assert_eq!(rows_of(&table), rows, "case {id}: positions");
                    assert_eq!(
                        delta_of(&table),
                        expected["delta"].as_i64().unwrap(),
                        "case {id}: delta"
                    );
                    assert_eq!(
                        table.scalar,
                        expected["scalar"].as_i64().unwrap(),
                        "case {id}: scalar"
                    );
                    checked += 1;
                }
                Err(reason) => {
                    assert_eq!(
                        expected["fast_path"].as_bool(),
                        Some(false),
                        "case {id}: refused ({reason}) where the engine's rule served it"
                    );
                    refused += 1;
                }
            }
        }
        eprintln!("corpus {path}: {checked} tables identical, {refused} rescan cases refused");
        assert!(checked > 0);
    }
}
