//! The MRoPE position tables of the families whose rule the engine's own
//! input processor has, computed as it computes them on the path this wire
//! bypasses.
//!
//! An MRoPE language model (a text rope config with an `mrope_section`) reads
//! three position axes per token: text runs advance all three linearly, an
//! image or video frame lays its merged patch grid out over (time, height,
//! width). The engine derives the table in its frontend from the un-padded
//! prompt, the items' `image_grid_thw` / `video_grid_thw` rows and their
//! placeholder spans (`runtime/multimodal/mrope.py`): the Qwen3.5 lineage by
//! `_compute_qwen35_mrope_positions_from_offsets`, Qwen3-Omni by
//! `_compute_qwen3_omni_mrope_positions`, and no other architecture (its
//! dispatch, `_MROPE_ARCHITECTURES`, writes no table for the rest). Nothing
//! between the msgpack wire and the executor repeats that step, and an empty
//! table means the executor falls back to one-dimensional positions without a
//! word. This module is that step, rule for rule, so the table on the wire is
//! the one the engine would have written.

use engine_zmq_client::{
    codec::tensor::WireTensor, protocol::tokenspeed::multimodal::TokenSpeedWireModality,
};
use smg_grpc_client::tokenspeed_proto;

/// The checkpoint facts the Qwen3.5 rule needs besides the request: the vision
/// tower's spatial merge (patches per position along each axis) and the two
/// placeholder token ids the engine checks each span against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MropeLayout {
    pub spatial_merge_size: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
}

/// The checkpoint facts the Qwen3-Omni rule needs besides the request, which
/// its checkpoint keeps under `thinker_config`: the vision tower's spatial
/// merge and the model's clock. The clock is read where the engine reads it
/// (`getattr(thinker_config, ..., default)`: the launcher ships the engine's
/// defaults, 13 and 2.0, for a thinker that leaves them unsaid); the rule
/// itself defaults nothing, so a source that carries no clock gets no rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniLayout {
    pub spatial_merge_size: u32,
    /// `position_id_per_seconds`: the positions one second of media spans on
    /// the time axis.
    pub position_id_per_seconds: f64,
    /// `seconds_per_chunk`: the seconds one temporal grid of a video spans
    /// when its item carries no `video_second_per_grid`.
    pub seconds_per_chunk: f64,
}

/// The checkpoint facts a launcher, or the engine's handshake, ships for the
/// position rules, each `None` when the checkpoint has none; which of them a
/// rule needs is the rule's ([`RuleFamily::rule`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PositionFacts {
    pub spatial_merge_size: Option<u32>,
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub position_id_per_seconds: Option<f64>,
    pub seconds_per_chunk: Option<f64>,
}

/// The families whose table the engine's frontend computes, hence the rules
/// this module has: its dispatch, the architecture first (Qwen3-Omni has its
/// own rule), then the Qwen3.5 model types of the remaining architectures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleFamily {
    Qwen35,
    Qwen3Omni,
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
const QWEN3_OMNI_ARCHITECTURES: [&str; 1] = ["Qwen3OmniMoeForConditionalGeneration"];

impl RuleFamily {
    /// The family the engine computes this checkpoint's table with; `None`
    /// for a checkpoint it writes no table for.
    pub fn of(architectures: &[String], model_type: &str) -> Option<Self> {
        let named = |names: &[&str]| {
            architectures
                .iter()
                .any(|arch| names.contains(&arch.as_str()))
        };
        if named(&QWEN3_OMNI_ARCHITECTURES) {
            return Some(Self::Qwen3Omni);
        }
        if named(&QWEN35_ARCHITECTURES) && QWEN35_MODEL_TYPES.contains(&model_type) {
            return Some(Self::Qwen35);
        }
        None
    }

    /// The family's rule with the facts it reads, `None` when a fact it needs
    /// is missing: the Qwen3.5 rule reads the merge size and both placeholder
    /// ids, the Omni rule the merge size and both clock facts (a source that
    /// does not carry the clock, such as the engine's handshake today, gets no
    /// rule rather than a guessed clock).
    pub fn rule(self, facts: PositionFacts) -> Option<PositionRule> {
        match self {
            Self::Qwen35 => match (
                facts.spatial_merge_size,
                facts.image_token_id,
                facts.video_token_id,
            ) {
                (Some(spatial_merge_size), Some(image_token_id), Some(video_token_id)) => {
                    Some(PositionRule::Qwen35(MropeLayout {
                        spatial_merge_size,
                        image_token_id,
                        video_token_id,
                    }))
                }
                _ => None,
            },
            Self::Qwen3Omni => match (
                facts.spatial_merge_size,
                facts.position_id_per_seconds,
                facts.seconds_per_chunk,
            ) {
                (
                    Some(spatial_merge_size),
                    Some(position_id_per_seconds),
                    Some(seconds_per_chunk),
                ) => Some(PositionRule::Qwen3Omni(OmniLayout {
                    spatial_merge_size,
                    position_id_per_seconds,
                    seconds_per_chunk,
                })),
                _ => None,
            },
        }
    }

    /// The facts the family's rule cannot run without, by their config names,
    /// for the refusal that names them.
    pub fn needs(self) -> &'static str {
        match self {
            Self::Qwen35 => "spatial_merge_size, image_token_id, video_token_id",
            Self::Qwen3Omni => {
                "thinker_config.vision_config.spatial_merge_size, \
                 thinker_config.position_id_per_seconds, thinker_config.seconds_per_chunk"
            }
        }
    }
}

/// Whether the engine computes this checkpoint's positions with the Qwen3.5
/// offsets rule: one of its architectures with one of its model types.
pub fn qwen35_lineage(architectures: &[String], model_type: &str) -> bool {
    RuleFamily::of(architectures, model_type) == Some(RuleFamily::Qwen35)
}

/// A family's rule with the checkpoint facts it reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PositionRule {
    Qwen35(MropeLayout),
    Qwen3Omni(OmniLayout),
}

impl PositionRule {
    /// The table the engine's frontend would write for this prompt and these
    /// items, or the precondition it would fail on, by name.
    pub(crate) fn positions(
        self,
        input_ids: &[u32],
        items: &[GridItem<'_>],
    ) -> Result<MropeTable, String> {
        match self {
            Self::Qwen35(layout) => qwen35_positions(layout, input_ids, items),
            Self::Qwen3Omni(layout) => qwen3_omni_positions(layout, input_ids.len(), items),
        }
    }

    /// Whether the rule reads a video item's clock (`video_second_per_grid`)
    /// and its `use_audio_in_video` flag: the Omni rule does.
    pub(crate) fn reads_video_timing(self) -> bool {
        matches!(self, Self::Qwen3Omni(_))
    }
}

/// Which grid an item carries: the Qwen3.5 rule takes the first key an item
/// carries, image before video, as the engine's fast path does; the Omni rule
/// the modality's own key, as the engine's `_omni_media_segments` does.
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

    /// The grid a modality owns (`{modality}_grid_thw`); audio has none.
    pub(crate) fn of_modality(modality: TokenSpeedWireModality) -> Option<Self> {
        match modality {
            TokenSpeedWireModality::Image => Some(Self::Image),
            TokenSpeedWireModality::Video => Some(Self::Video),
            TokenSpeedWireModality::Audio => None,
        }
    }

    /// The tensor key the grid rides under.
    pub(crate) fn key(self) -> &'static str {
        match self {
            Self::Image => "image_grid_thw",
            Self::Video => "video_grid_thw",
        }
    }
}

/// The refusal for a running position no `i64` holds: never a wrapped table.
const POSITIONS_OVERFLOW: &str =
    "the position table overflows: a multimodal item's time axis climbs past what a position holds";

/// Below 2^62 a float32 time product casts to an `i64` without saturating;
/// the checked additions decide the rest.
const TIME_POSITION_LIMIT: f32 = (1u64 << 62) as f32;

/// One item's inputs to the rules: its modality, its grid rows `(t, h, w)`
/// as the tensor carries them (`None` for an item without a grid, which
/// contributes nothing to the Qwen3.5 rule), its inclusive placeholder spans
/// in prompt order, and what the Omni rule reads of a video: its
/// `video_second_per_grid` values (`None` when the item carries none) and
/// whether it asked for its audio interleaved.
#[derive(Debug)]
pub(crate) struct GridItem<'a> {
    pub modality: TokenSpeedWireModality,
    pub grid: Option<(GridKind, Vec<[i64; 3]>)>,
    pub offsets: &'a [(u64, u64)],
    pub seconds_per_grid: Option<Vec<f32>>,
    pub audio_in_video: bool,
}

/// The three fields the engine's input processor writes: the table `(3, L)`
/// int64, the delta `(1, 1)` int64 and the delta as an integer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MropeTable {
    pub positions: WireTensor,
    pub delta: WireTensor,
    pub scalar: i64,
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
    let values = integers(key, tensor)?;
    if values.len() != rows * 3 {
        return Err(format!(
            "multimodal item's {key} holds {} values for shape {shape:?}",
            values.len()
        ));
    }
    Ok(values.as_chunks::<3>().0.to_vec())
}

/// The inline payload of a tensor, or the refusal for one that is not inline.
fn inline_bytes<'a>(
    key: &str,
    tensor: &'a tokenspeed_proto::TensorData,
) -> Result<&'a [u8], String> {
    match &tensor.payload {
        Some(tokenspeed_proto::tensor_data::Payload::Inline(bytes)) => Ok(bytes),
        _ => Err(format!("multimodal item's {key} carries no inline payload")),
    }
}

/// An integer tensor's values, whatever their width.
fn integers(key: &str, tensor: &tokenspeed_proto::TensorData) -> Result<Vec<i64>, String> {
    let bytes = inline_bytes(key, tensor)?;
    match tensor.dtype.as_str() {
        "int64" => Ok(elements::<8>(bytes, key)?
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect()),
        "uint64" => elements::<8>(bytes, key)?
            .iter()
            .map(|c| i64::try_from(u64::from_le_bytes(*c)))
            .collect::<Result<_, _>>()
            .map_err(|_| format!("multimodal item's {key} holds a value above i64")),
        "int32" => Ok(elements::<4>(bytes, key)?
            .iter()
            .map(|c| i64::from(i32::from_le_bytes(*c)))
            .collect()),
        "uint32" => Ok(elements::<4>(bytes, key)?
            .iter()
            .map(|c| i64::from(u32::from_le_bytes(*c)))
            .collect()),
        other => Err(format!(
            "multimodal item's {key} has dtype {other:?}; a grid is an integer tensor"
        )),
    }
}

/// A video item's `video_second_per_grid` values as the engine reads them:
/// the tensor flattened and taken as float32 (a float64 or integer value
/// rounds to the nearest float32, as `torch.as_tensor(dtype=float32)` rounds
/// it); the gateway's media pipeline sends one float32 per clip.
pub(crate) fn seconds_per_grid(
    key: &str,
    tensor: &tokenspeed_proto::TensorData,
) -> Result<Vec<f32>, String> {
    let bytes = inline_bytes(key, tensor)?;
    match tensor.dtype.as_str() {
        "float32" => Ok(elements::<4>(bytes, key)?
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()),
        "float64" => Ok(elements::<8>(bytes, key)?
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect()),
        "int64" | "uint64" | "int32" | "uint32" => {
            Ok(integers(key, tensor)?.iter().map(|&v| v as f32).collect())
        }
        other => Err(format!(
            "multimodal item's {key} has dtype {other:?}; a clip's seconds per grid are a float tensor"
        )),
    }
}

/// Whether a flag tensor holds any true value, as `torch.as_tensor(..).any()`
/// reads it: a bool or integer tensor by its bytes (a non-zero element has a
/// non-zero byte), a float tensor by its values.
pub(crate) fn flag_set(key: &str, tensor: &tokenspeed_proto::TensorData) -> Result<bool, String> {
    let bytes = inline_bytes(key, tensor)?;
    match tensor.dtype.as_str() {
        "bool" | "uint8" | "int8" | "int16" | "uint16" | "int32" | "uint32" | "int64"
        | "uint64" => Ok(bytes.iter().any(|&b| b != 0)),
        "float32" => Ok(elements::<4>(bytes, key)?
            .iter()
            .any(|c| f32::from_le_bytes(*c) != 0.0)),
        "float64" => Ok(elements::<8>(bytes, key)?
            .iter()
            .any(|c| f64::from_le_bytes(*c) != 0.0)),
        other => Err(format!(
            "multimodal item's {key} has dtype {other:?}; a flag is a bool tensor"
        )),
    }
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
            let (start, end) = span_in(index, start, end, seq_len)?;
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

/// An inclusive span as prompt indices, or the refusal for one outside the
/// prompt.
fn span_in(index: usize, start: u64, end: u64, seq_len: usize) -> Result<(usize, usize), String> {
    let fits = end >= start && usize::try_from(end).is_ok_and(|end| end < seq_len);
    if !fits {
        return Err(format!(
            "multimodal item {index} placeholder span [{start}, {end}] does not fit the \
             {seq_len}-token prompt"
        ));
    }
    Ok((start as usize, end as usize))
}

/// A text run: the same linear positions on all three axes; a running
/// position that cannot take the run (a time axis the Omni rule let climb
/// that far) is a refusal, never a wrap.
fn fill_text(
    axes: &mut [&mut [i64]; 3],
    from: usize,
    to: usize,
    next_pos: &mut i64,
) -> Result<(), String> {
    let after = next_pos
        .checked_add((to - from) as i64)
        .ok_or_else(|| POSITIONS_OVERFLOW.to_string())?;
    for pos in from..to {
        let value = *next_pos + (pos - from) as i64;
        for axis in axes.iter_mut() {
            axis[pos] = value;
        }
    }
    *next_pos = after;
    Ok(())
}

/// The three axes of a table over `seq_len` tokens, and the table they make.
fn table_of(
    seq_len: usize,
    fill: impl FnOnce(&mut [&mut [i64]; 3]) -> Result<i64, String>,
) -> Result<MropeTable, String> {
    let mut positions = vec![0i64; 3 * seq_len];
    let (t_axis, rest) = positions.split_at_mut(seq_len);
    let (h_axis, w_axis) = rest.split_at_mut(seq_len);
    let mut axes = [t_axis, h_axis, w_axis];
    let next_pos = fill(&mut axes)?;
    let scalar = next_pos - seq_len as i64;
    Ok(MropeTable {
        positions: WireTensor::from_i64(vec![3, seq_len], positions)?,
        delta: WireTensor::from_i64(vec![1, 1], vec![scalar])?,
        scalar,
    })
}

/// `_build_qwen35_mrope_positions`: text runs linear from the running
/// position, a grid laid out row-major over (t, h, w) from it, the running
/// position advanced by the grid's largest side.
fn build_table(seq_len: usize, segments: &[Segment]) -> Result<MropeTable, String> {
    table_of(seq_len, |axes| {
        let mut next_pos: i64 = 0;
        let mut cursor = 0usize;
        for segment in segments {
            if cursor < segment.start {
                fill_text(axes, cursor, segment.start, &mut next_pos)?;
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
            fill_text(axes, cursor, seq_len, &mut next_pos)?;
        }
        Ok(next_pos)
    })
}

/// A span of the Omni rule: an audio run, or a vision grid with the clock its
/// time axis ticks on.
#[derive(Debug, Clone, Copy)]
struct OmniSegment {
    item: usize,
    start: usize,
    end: usize,
    media: OmniMedia,
}

#[derive(Debug, Clone, Copy)]
enum OmniMedia {
    Audio,
    /// The grid as the tensor carries it and the float32 factor one temporal
    /// step spans on the time axis.
    Vision {
        t: i64,
        h: i64,
        w: i64,
        clock: f32,
    },
}

/// The Qwen3-Omni rule over the prompt: the engine's
/// `_compute_qwen3_omni_mrope_positions` with its `_omni_media_segments`.
/// The items' spans are authoritative (sorted by start, never checked
/// against the ids); an audio span advances all three axes linearly; an image
/// or video item lays each grid row out over one span, row-major over
/// (t, h/m, w/m), its time axis in steps of the model's clock: an image
/// `position_id_per_seconds` per step, a video that times its seconds per
/// grid, truncated to an integer as torch truncates a float32 (the arithmetic
/// is float32, as the engine's is); the running position resumes one past
/// the span's largest position. Every precondition the engine raises on is
/// an error naming it.
pub(crate) fn qwen3_omni_positions(
    layout: OmniLayout,
    seq_len: usize,
    items: &[GridItem<'_>],
) -> Result<MropeTable, String> {
    if seq_len == 0 {
        return Err("MRoPE positions need a non-empty prompt".to_string());
    }
    let merge = i64::from(layout.spatial_merge_size);
    if merge <= 0 {
        return Err("MRoPE positions need a positive spatial merge size".to_string());
    }
    let mut segments: Vec<OmniSegment> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        // An item without spans contributes nothing, as the engine skips it.
        if item.offsets.is_empty() {
            continue;
        }
        let modality = item.modality;
        if modality == TokenSpeedWireModality::Audio {
            for &(start, end) in item.offsets {
                let (start, end) = span_in(index, start, end, seq_len)?;
                segments.push(OmniSegment {
                    item: index,
                    start,
                    end,
                    media: OmniMedia::Audio,
                });
            }
            continue;
        }
        let Some((_, rows)) = &item.grid else {
            let key = GridKind::of_modality(modality).map_or("grid", GridKind::key);
            return Err(format!(
                "multimodal item {index} ({modality:?}) carries no {key}, which the engine's \
                 rule lays its span out over"
            ));
        };
        if modality == TokenSpeedWireModality::Video && item.audio_in_video {
            return Err(format!(
                "multimodal item {index} asks for its audio interleaved with the video \
                 (use_audio_in_video), which the engine's rule does not serve"
            ));
        }
        for (row, &[t, h, w]) in rows.iter().enumerate() {
            if t <= 0 || h <= 0 || w <= 0 {
                return Err(format!(
                    "multimodal item {index} grid row {row} ({t}, {h}, {w}) is not positive"
                ));
            }
        }
        if rows.len() != item.offsets.len() {
            return Err(format!(
                "multimodal item {index} has {} grid row(s) for {} placeholder span(s)",
                rows.len(),
                item.offsets.len()
            ));
        }
        // A video's seconds per grid: the item's one value for every row, or
        // one per row, else the checkpoint's; an image ticks the bare clock.
        let seconds: Vec<Option<f64>> = if modality == TokenSpeedWireModality::Video {
            if let Some(bad) = item
                .seconds_per_grid
                .as_deref()
                .and_then(|values| values.iter().find(|v| !v.is_finite() || **v < 0.0))
            {
                return Err(format!(
                    "multimodal item {index} video_second_per_grid {bad} is not a finite, \
                     non-negative number of seconds"
                ));
            }
            match &item.seconds_per_grid {
                Some(values) if values.len() == 1 => vec![Some(f64::from(values[0])); rows.len()],
                Some(values) if values.len() == rows.len() => {
                    values.iter().map(|&v| Some(f64::from(v))).collect()
                }
                Some(values) => {
                    return Err(format!(
                        "multimodal item {index} carries {} video_second_per_grid value(s) for \
                         {} grid row(s); one, or one per row",
                        values.len(),
                        rows.len()
                    ))
                }
                None => vec![Some(layout.seconds_per_chunk); rows.len()],
            }
        } else {
            vec![None; rows.len()]
        };
        for ((&(start, end), &[t, h, w]), seconds) in item.offsets.iter().zip(rows).zip(seconds) {
            let (start, end) = span_in(index, start, end, seq_len)?;
            // The engine scales a float32 `arange` by a Python float: the
            // product of the seconds and the clock, in double, narrowed to
            // float32 at the multiply.
            let clock = match seconds {
                Some(seconds) => (seconds * layout.position_id_per_seconds) as f32,
                None => layout.position_id_per_seconds as f32,
            };
            segments.push(OmniSegment {
                item: index,
                start,
                end,
                media: OmniMedia::Vision { t, h, w, clock },
            });
        }
    }
    segments.sort_by_key(|segment| segment.start);
    table_of(seq_len, |axes| {
        let mut next_pos: i64 = 0;
        let mut cursor = 0usize;
        for segment in &segments {
            if segment.start < cursor {
                return Err("multimodal placeholder spans overlap".to_string());
            }
            fill_text(axes, cursor, segment.start, &mut next_pos)?;
            match segment.media {
                OmniMedia::Audio => fill_text(axes, segment.start, segment.end + 1, &mut next_pos)?,
                OmniMedia::Vision { t, h, w, clock } => {
                    let index = segment.item;
                    if h % merge != 0 || w % merge != 0 {
                        return Err(format!(
                            "multimodal item {index} grid ({t}, {h}, {w}) does not divide by the \
                             spatial merge {merge}"
                        ));
                    }
                    let (lh, lw) = (h / merge, w / merge);
                    let expected = t
                        .checked_mul(lh)
                        .and_then(|n| n.checked_mul(lw))
                        .ok_or_else(|| {
                            format!("multimodal item {index} grid ({t}, {h}, {w}) overflows")
                        })?;
                    let span = segment.end - segment.start + 1;
                    if span as i64 != expected {
                        return Err(format!(
                            "multimodal item {index} placeholder span [{}, {}] has {span} tokens; \
                             its grid ({t}, {h}, {w}) under spatial merge {merge} needs {expected}",
                            segment.start, segment.end
                        ));
                    }
                    // One time position per temporal step, as the engine's
                    // float32 `arange * clock` truncated; the span check above
                    // bounds `t` by the prompt. A product the request's
                    // seconds push past what a position holds is refused
                    // (the engine's cast would wrap it into a garbage table).
                    let steps = (0..t)
                        .map(|step| {
                            let product = step as f32 * clock;
                            (product.is_finite() && (0.0..TIME_POSITION_LIMIT).contains(&product))
                                .then_some(product as i64)
                                .ok_or_else(|| {
                                    format!(
                                        "multimodal item {index} video_second_per_grid puts time \
                                         step {step} at {product}, past what a position holds"
                                    )
                                })
                        })
                        .collect::<Result<Vec<i64>, String>>()?;
                    let (lh, lw) = (lh as usize, lw as usize);
                    // Every position the span writes is at most `next_pos +
                    // top`, so the one checked advance bounds all three axes
                    // before a cell is written; the fills stay plain.
                    let top = steps
                        .iter()
                        .copied()
                        .max()
                        .unwrap_or(0)
                        .max(lh as i64 - 1)
                        .max(lw as i64 - 1);
                    let after = top
                        .checked_add(1)
                        .and_then(|advance| next_pos.checked_add(advance))
                        .ok_or_else(|| POSITIONS_OVERFLOW.to_string())?;
                    for (offset, pos) in (segment.start..=segment.end).enumerate() {
                        let (step, cell) = (offset / (lh * lw), offset % (lh * lw));
                        axes[0][pos] = next_pos + steps[step];
                        axes[1][pos] = next_pos + (cell / lw) as i64;
                        axes[2][pos] = next_pos + (cell % lw) as i64;
                    }
                    next_pos = after;
                }
            }
            cursor = segment.end + 1;
        }
        fill_text(axes, cursor, seq_len, &mut next_pos)?;
        Ok(next_pos)
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

    /// The engine's own Omni test facts (`test/runtime/models/test_qwen3_omni.py`).
    pub(crate) const OMNI: OmniLayout = OmniLayout {
        spatial_merge_size: 2,
        position_id_per_seconds: 13.0,
        seconds_per_chunk: 2.0,
    };

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

    /// A rule's item from a spec: the modality follows the grid's kind.
    fn item_of<'a>(
        grid: Option<&(GridKind, Vec<[i64; 3]>)>,
        offsets: &'a [(u64, u64)],
    ) -> GridItem<'a> {
        GridItem {
            modality: match grid {
                Some((GridKind::Video, _)) => TokenSpeedWireModality::Video,
                _ => TokenSpeedWireModality::Image,
            },
            grid: grid.cloned(),
            offsets,
            seconds_per_grid: None,
            audio_in_video: false,
        }
    }

    fn table(input_ids: &[u32], items: &[Spec]) -> Result<MropeTable, String> {
        let items: Vec<GridItem<'_>> = items
            .iter()
            .map(|(grid, offsets)| item_of(grid.as_ref(), offsets))
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

    /// A clip's seconds per grid read as float32 whatever the tensor's
    /// width, and a flag by any true value; a dtype neither is, is refused.
    #[test]
    fn video_timing_tensors_read_as_the_engine_reads_them() {
        let tensor = |dtype: &str, shape: Vec<u32>, bytes: Vec<u8>| tokenspeed_proto::TensorData {
            shape,
            dtype: dtype.to_string(),
            payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(bytes.into())),
        };
        let key = "video_second_per_grid";
        let f32_bytes: Vec<u8> = [0.5f32, 0.1].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            seconds_per_grid(key, &tensor("float32", vec![2], f32_bytes)).unwrap(),
            vec![0.5, 0.1]
        );
        // A float64 rounds to the nearest float32, an integer converts.
        assert_eq!(
            seconds_per_grid(
                key,
                &tensor("float64", vec![1], 0.1f64.to_le_bytes().to_vec())
            )
            .unwrap(),
            vec![0.1f32]
        );
        assert_eq!(
            seconds_per_grid(key, &tensor("int64", vec![1], 2i64.to_le_bytes().to_vec())).unwrap(),
            vec![2.0]
        );
        assert!(
            seconds_per_grid(key, &tensor("bfloat16", vec![1], vec![0, 0]))
                .unwrap_err()
                .contains("dtype")
        );
        let flag = "use_audio_in_video";
        assert!(flag_set(flag, &tensor("bool", vec![1], vec![1])).unwrap());
        assert!(!flag_set(flag, &tensor("bool", vec![2], vec![0, 0])).unwrap());
        assert!(flag_set(flag, &tensor("int64", vec![1], 1i64.to_le_bytes().to_vec())).unwrap());
        assert!(!flag_set(
            flag,
            &tensor("float32", vec![1], 0.0f32.to_le_bytes().to_vec())
        )
        .unwrap());
        assert!(flag_set(flag, &tensor("bfloat16", vec![1], vec![0, 0]))
            .unwrap_err()
            .contains("dtype"));
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
        // Qwen3-Omni has its own rule, picked by its architecture whatever
        // the model type says; a Qwen3-VL checkpoint is outside the engine's
        // set; an architecture without its model type is not the rule.
        assert_eq!(
            RuleFamily::of(
                &arch("Qwen3OmniMoeForConditionalGeneration"),
                "qwen3_omni_moe"
            ),
            Some(RuleFamily::Qwen3Omni)
        );
        assert!(!qwen35_lineage(
            &arch("Qwen3OmniMoeForConditionalGeneration"),
            "qwen3_omni_moe"
        ));
        assert_eq!(
            RuleFamily::of(&arch("Qwen3VLForConditionalGeneration"), "qwen3_vl"),
            None
        );
        assert!(!qwen35_lineage(
            &arch("Qwen3_5ForConditionalGeneration"),
            "qwen3_vl"
        ));
        assert!(!qwen35_lineage(&[], "qwen3_5"));
    }

    /// Each family's rule needs its own facts: the Qwen3.5 rule all three
    /// layout facts, the Omni rule the merge size with the engine's defaults
    /// for a clock the checkpoint leaves unsaid.
    #[test]
    fn each_family_reads_the_facts_its_rule_needs() {
        let qwen35 = PositionFacts {
            spatial_merge_size: Some(2),
            image_token_id: Some(LAYOUT.image_token_id),
            video_token_id: Some(LAYOUT.video_token_id),
            ..PositionFacts::default()
        };
        assert_eq!(
            RuleFamily::Qwen35.rule(qwen35),
            Some(PositionRule::Qwen35(LAYOUT))
        );
        assert_eq!(
            RuleFamily::Qwen35.rule(PositionFacts {
                video_token_id: None,
                ..qwen35
            }),
            None
        );
        // The Omni rule reads the merge size and both clock facts; a source
        // without the clock (the engine's handshake today) gets no rule, not
        // the engine's defaults guessed in.
        let merge_only = PositionFacts {
            spatial_merge_size: Some(2),
            ..PositionFacts::default()
        };
        assert_eq!(RuleFamily::Qwen3Omni.rule(merge_only), None);
        let omni = PositionFacts {
            position_id_per_seconds: Some(13.0),
            seconds_per_chunk: Some(2.0),
            ..merge_only
        };
        assert_eq!(
            RuleFamily::Qwen3Omni.rule(omni),
            Some(PositionRule::Qwen3Omni(OMNI))
        );
        assert_eq!(
            RuleFamily::Qwen3Omni.rule(PositionFacts {
                position_id_per_seconds: Some(25.0),
                seconds_per_chunk: Some(0.5),
                ..omni
            }),
            Some(PositionRule::Qwen3Omni(OmniLayout {
                spatial_merge_size: 2,
                position_id_per_seconds: 25.0,
                seconds_per_chunk: 0.5,
            }))
        );
        assert_eq!(
            RuleFamily::Qwen3Omni.rule(PositionFacts {
                seconds_per_chunk: None,
                ..omni
            }),
            None
        );
        assert_eq!(RuleFamily::Qwen3Omni.rule(PositionFacts::default()), None);
        let needs = RuleFamily::Qwen3Omni.needs();
        for fact in [
            "spatial_merge_size",
            "position_id_per_seconds",
            "seconds_per_chunk",
        ] {
            assert!(needs.contains(fact), "{needs}");
        }
    }

    /// An Omni item: its modality, its spans, its grid rows and a video's
    /// seconds per grid.
    struct OmniSpec {
        modality: TokenSpeedWireModality,
        offsets: Vec<(u64, u64)>,
        rows: Vec<[i64; 3]>,
        seconds: Option<Vec<f32>>,
        audio_in_video: bool,
    }

    fn omni_audio(offsets: &[(u64, u64)]) -> OmniSpec {
        OmniSpec {
            modality: TokenSpeedWireModality::Audio,
            offsets: offsets.to_vec(),
            rows: Vec::new(),
            seconds: None,
            audio_in_video: false,
        }
    }

    fn omni_image(rows: &[[i64; 3]], offsets: &[(u64, u64)]) -> OmniSpec {
        OmniSpec {
            modality: TokenSpeedWireModality::Image,
            offsets: offsets.to_vec(),
            rows: rows.to_vec(),
            seconds: None,
            audio_in_video: false,
        }
    }

    fn omni_video(rows: &[[i64; 3]], offsets: &[(u64, u64)], seconds: Option<&[f32]>) -> OmniSpec {
        OmniSpec {
            modality: TokenSpeedWireModality::Video,
            offsets: offsets.to_vec(),
            rows: rows.to_vec(),
            seconds: seconds.map(<[f32]>::to_vec),
            audio_in_video: false,
        }
    }

    fn omni_table(seq_len: usize, specs: &[OmniSpec]) -> Result<MropeTable, String> {
        let items: Vec<GridItem<'_>> = specs
            .iter()
            .map(|spec| GridItem {
                modality: spec.modality,
                grid: (spec.modality != TokenSpeedWireModality::Audio && !spec.rows.is_empty())
                    .then(|| {
                        let kind = if spec.modality == TokenSpeedWireModality::Video {
                            GridKind::Video
                        } else {
                            GridKind::Image
                        };
                        (kind, spec.rows.clone())
                    }),
                offsets: &spec.offsets,
                seconds_per_grid: spec.seconds.clone(),
                audio_in_video: spec.audio_in_video,
            })
            .collect();
        qwen3_omni_positions(OMNI, seq_len, &items)
    }

    fn linear(from: i64, n: usize) -> Vec<i64> {
        (from..from + n as i64).collect()
    }

    /// The engine's mixed case (`test_mixed_modalities_follow_authored_offsets`):
    /// an audio run, an image and a one-second-per-grid video over 25 tokens,
    /// the items handed over out of order; its expected table verbatim.
    #[test]
    fn omni_mixed_modalities_follow_the_authored_offsets() {
        let table = omni_table(
            25,
            &[
                omni_video(&[[2, 4, 4]], &[(15, 22)], Some(&[1.0])),
                omni_audio(&[(2, 4)]),
                omni_image(&[[1, 4, 4]], &[(8, 11)]),
            ],
        )
        .unwrap();
        let rows = rows_of(&table);
        for axis in &rows {
            assert_eq!(&axis[..8], &linear(0, 8)[..]);
            assert_eq!(&axis[12..15], &[10, 11, 12]);
            assert_eq!(&axis[23..], &[27, 28]);
        }
        assert_eq!(&rows[0][8..12], &[8, 8, 8, 8]);
        assert_eq!(&rows[1][8..12], &[8, 8, 9, 9]);
        assert_eq!(&rows[2][8..12], &[8, 9, 8, 9]);
        assert_eq!(&rows[0][15..23], &[13, 13, 13, 13, 26, 26, 26, 26]);
        assert_eq!(&rows[1][15..23], &[13, 13, 14, 14, 13, 13, 14, 14]);
        assert_eq!(&rows[2][15..23], &[13, 14, 13, 14, 13, 14, 13, 14]);
        assert_eq!(delta_of(&table), 4);
        assert_eq!(table.scalar, 4);
    }

    /// The engine's fractional case (`test_multiple_videos_support_fractional_seconds_per_grid`):
    /// two videos timed at half and one and a half seconds per grid; the
    /// time axis truncates as torch truncates a float32.
    #[test]
    fn omni_videos_tick_their_own_seconds_per_grid() {
        let table = omni_table(
            22,
            &[
                omni_video(&[[2, 4, 4]], &[(12, 19)], Some(&[1.5])),
                omni_video(&[[2, 4, 4]], &[(2, 9)], Some(&[0.5])),
            ],
        )
        .unwrap();
        let rows = rows_of(&table);
        assert_eq!(&rows[0][2..10], &[2, 2, 2, 2, 8, 8, 8, 8]);
        assert_eq!(&rows[0][12..20], &[11, 11, 11, 11, 30, 30, 30, 30]);
        assert_eq!(delta_of(&table), 11);
        // A video without its seconds ticks the checkpoint's seconds per
        // chunk (two seconds: 26 positions a step); an image ticks the bare
        // clock, which a one-step image never shows.
        let table = omni_table(10, &[omni_video(&[[2, 4, 4]], &[(1, 8)], None)]).unwrap();
        assert_eq!(&rows_of(&table)[0][1..9], &[1, 1, 1, 1, 27, 27, 27, 27]);
        assert_eq!(delta_of(&table), 28 + 1 - 10);
        // One value for one row is the broadcast arm again.
        let table = omni_table(10, &[omni_video(&[[2, 4, 4]], &[(1, 8)], Some(&[0.5]))]).unwrap();
        assert_eq!(&rows_of(&table)[0][1..9], &[1, 1, 1, 1, 7, 7, 7, 7]);
        // One value per grid row: each row ticks its own clock (the engine's
        // table for two rows at 0.5 and 1.5 seconds per grid: the second row
        // starts where the first's top left off, at 8, and its second step
        // is 1.5 * 13 = 19.5 positions on, truncated).
        let table = omni_table(
            18,
            &[omni_video(
                &[[2, 4, 4], [2, 4, 4]],
                &[(1, 8), (9, 16)],
                Some(&[0.5, 1.5]),
            )],
        )
        .unwrap();
        let rows = rows_of(&table);
        assert_eq!(
            rows[0],
            vec![0, 1, 1, 1, 1, 7, 7, 7, 7, 8, 8, 8, 8, 27, 27, 27, 27, 28]
        );
        assert_eq!(
            rows[1],
            vec![0, 1, 1, 2, 2, 1, 1, 2, 2, 8, 8, 9, 9, 8, 8, 9, 9, 28]
        );
        assert_eq!(
            rows[2],
            vec![0, 1, 2, 1, 2, 1, 2, 1, 2, 8, 9, 8, 9, 8, 9, 8, 9, 28]
        );
        assert_eq!(delta_of(&table), 11);
    }

    /// Every precondition the engine's Omni rule raises on is a refusal
    /// naming it (`test_rejects_audio_in_video_interleaving`,
    /// `test_rejects_placeholder_grid_mismatch` and the rest of its checks).
    #[test]
    fn the_omni_rules_preconditions_are_refused_by_name() {
        let err = |seq_len: usize, specs: &[OmniSpec]| omni_table(seq_len, specs).unwrap_err();
        let mut interleaved = omni_video(&[[1, 4, 4]], &[(1, 4)], None);
        interleaved.audio_in_video = true;
        assert!(err(6, &[interleaved]).contains("use_audio_in_video"));
        assert!(err(5, &[omni_image(&[[1, 4, 4]], &[(1, 3)])]).contains("needs 4"));
        assert!(err(6, &[omni_image(&[], &[(1, 4)])]).contains("carries no image_grid_thw"));
        assert!(err(6, &[omni_image(&[[1, 4, 4], [1, 4, 4]], &[(1, 4)])])
            .contains("2 grid row(s) for 1"));
        assert!(err(6, &[omni_image(&[[1, 0, 4]], &[(1, 4)])]).contains("is not positive"));
        assert!(err(6, &[omni_image(&[[1, 3, 4]], &[(1, 2)])]).contains("does not divide"));
        assert!(err(6, &[omni_image(&[[1, 4, 4]], &[(3, 6)])]).contains("does not fit"));
        assert!(err(6, &[omni_audio(&[(5, 7)])]).contains("does not fit"));
        assert!(err(
            9,
            &[
                omni_image(&[[1, 4, 4]], &[(1, 4)]),
                omni_image(&[[1, 4, 4]], &[(4, 7)])
            ]
        )
        .contains("overlap"));
        assert!(err(
            10,
            &[omni_video(&[[2, 4, 4]], &[(1, 8)], Some(&[0.5, 1.5]))]
        )
        .contains("2 video_second_per_grid value(s) for 1"));
        assert!(err(0, &[]).contains("non-empty prompt"));
        // The clock side refuses what the engine's cast would wrap into a
        // garbage table: a non-finite or negative seconds value, a step whose
        // product leaves the range a position holds (the engine serves
        // `[2, 2, 2]` at 1e18 seconds per grid with i64::MIN + 1 in its
        // table), and a running position no `i64` takes.
        assert!(err(
            4,
            &[omni_video(&[[2, 2, 2]], &[(1, 2)], Some(&[f32::INFINITY]))]
        )
        .contains("video_second_per_grid inf is not a finite"));
        assert!(
            err(4, &[omni_video(&[[2, 2, 2]], &[(1, 2)], Some(&[-1.0]))])
                .contains("video_second_per_grid -1 is not a finite")
        );
        assert!(
            err(4, &[omni_video(&[[2, 2, 2]], &[(1, 2)], Some(&[1e18]))])
                .contains("past what a position holds")
        );
        // Three such videos: each advances the running position by 3.9e18,
        // and the third's step no longer fits beside it.
        assert!(err(
            10,
            &[
                omni_video(&[[2, 2, 2]], &[(1, 2)], Some(&[3e17])),
                omni_video(&[[2, 2, 2]], &[(4, 5)], Some(&[3e17])),
                omni_video(&[[2, 2, 2]], &[(7, 8)], Some(&[3e17]))
            ]
        )
        .contains("overflows"));
        // An item without spans contributes nothing, as the engine skips it;
        // a prompt of audio alone is linear with a zero delta.
        let table = omni_table(4, &[omni_audio(&[(1, 2)]), omni_image(&[[1, 4, 4]], &[])]).unwrap();
        assert_eq!(rows_of(&table), vec![linear(0, 4); 3]);
        assert_eq!(table.scalar, 0);
    }

    /// The cost of the rules per request on one core, printed as a table
    /// (`--ignored --nocapture`, a release build, the core pinned): the
    /// shapes the gateway's media pipeline builds, from one small image to a
    /// prompt that fills a window with a large image, for each rule.
    /// `SMG_MROPE_SPEED_ITERATIONS` sets the repetitions (2000 by default) and
    /// `SMG_MROPE_SPEED_SHAPE` keeps one shape by its index and
    /// `SMG_MROPE_SPEED_RULE` one rule by its name, so a counter run (`perf
    /// stat` on this binary) reads one cell, and two runs at different
    /// repetition counts separate the rule's cost from the process's.
    #[test]
    #[ignore = "a timing table, run alone on a pinned core"]
    #[expect(clippy::print_stdout)]
    fn speed_table() {
        type Shape = (
            &'static str,
            usize,
            Vec<(TokenSpeedWireModality, [i64; 3], usize)>,
            usize,
        );
        /// An item of a shape as the rule reads it: its modality, grid rows
        /// and spans.
        type ShapeItem = (TokenSpeedWireModality, Vec<[i64; 3]>, Vec<(u64, u64)>);
        let image = |rows: [i64; 3]| (TokenSpeedWireModality::Image, rows, 0);
        let video = |rows: [i64; 3]| (TokenSpeedWireModality::Video, rows, 0);
        let audio = |n: usize| (TokenSpeedWireModality::Audio, [0, 0, 0], n);
        let shapes: [Shape; 8] = [
            (
                "1 image 16x16 merged (256 tokens), 64 text",
                64,
                vec![image([1, 32, 32])],
                64,
            ),
            (
                "1 image 36x36 merged (1296 tokens), 256 text",
                256,
                vec![image([1, 72, 72])],
                256,
            ),
            (
                "4 images 16x16 merged, 128 text between",
                128,
                vec![image([1, 32, 32]); 4],
                128,
            ),
            (
                "1 grid of 8 temporal steps x 12x12 merged (1152 tokens), 16 text",
                16,
                vec![video([8, 24, 24])],
                16,
            ),
            (
                "1 image 60x60 merged (3600 tokens) in a 7.6k prompt",
                2000,
                vec![image([1, 120, 120])],
                2000,
            ),
            (
                "1 video 8 steps x 12x12 merged as one span (1152 tokens), 64 text",
                64,
                vec![video([8, 24, 24])],
                64,
            ),
            (
                "1 audio run of 300 tokens, 64 text",
                64,
                vec![audio(300)],
                64,
            ),
            (
                "audio 300 + image 16x16 + video 4 steps x 8x8, 64 text between",
                64,
                vec![audio(300), image([1, 32, 32]), video([4, 16, 16])],
                64,
            ),
        ];
        let iterations: u32 = std::env::var("SMG_MROPE_SPEED_ITERATIONS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(2000);
        let only: Option<usize> = std::env::var("SMG_MROPE_SPEED_SHAPE")
            .ok()
            .and_then(|value| value.parse().ok());
        let only_rule = std::env::var("SMG_MROPE_SPEED_RULE").ok();
        println!("| rule | shape | prompt tokens | per request |");
        println!("|---|---|---|---|");
        for (rule_name, rule) in [
            ("Qwen3.5", PositionRule::Qwen35(LAYOUT)),
            ("Qwen3-Omni", PositionRule::Qwen3Omni(OMNI)),
        ] {
            if only_rule.as_deref().is_some_and(|keep| keep != rule_name) {
                continue;
            }
            for (index, (name, lead, specs, tail)) in shapes.iter().enumerate() {
                if only.is_some_and(|keep| keep != index) {
                    continue;
                }
                let omni = rule.reads_video_timing();
                let mut ids: Vec<u32> = (0..*lead as u32).map(|i| 1000 + i).collect();
                let mut items: Vec<ShapeItem> = Vec::new();
                for &(modality, [t, h, w], audio_len) in specs {
                    let (grid_rows, spans): (Vec<[i64; 3]>, Vec<(u64, u64)>) =
                        if modality == TokenSpeedWireModality::Audio {
                            if !omni {
                                continue;
                            }
                            let start = ids.len();
                            ids.extend(std::iter::repeat_n(151646, audio_len));
                            (
                                Vec::new(),
                                vec![(start as u64, (start + audio_len - 1) as u64)],
                            )
                        } else if modality == TokenSpeedWireModality::Video && !omni {
                            // The Qwen3.5 rule reads a video as one span per
                            // temporal step.
                            let per_step = ((h / 2) * (w / 2)) as usize;
                            let mut spans = Vec::new();
                            for _ in 0..t {
                                let start = ids.len();
                                ids.extend(std::iter::repeat_n(LAYOUT.video_token_id, per_step));
                                spans.push((start as u64, (start + per_step - 1) as u64));
                                ids.extend((0..8u32).map(|i| 4000 + i));
                            }
                            (vec![[t, h, w]], spans)
                        } else {
                            let n = (t * (h / 2) * (w / 2)) as usize;
                            let start = ids.len();
                            let id = if modality == TokenSpeedWireModality::Video {
                                LAYOUT.video_token_id
                            } else {
                                LAYOUT.image_token_id
                            };
                            ids.extend(std::iter::repeat_n(id, n));
                            (
                                vec![[t, h, w]],
                                vec![(start as u64, (start + n - 1) as u64)],
                            )
                        };
                    ids.extend((0..64u32).map(|i| 2000 + i));
                    items.push((modality, grid_rows, spans));
                }
                ids.extend((0..*tail as u32).map(|i| 3000 + i));
                let grid_items: Vec<GridItem<'_>> = items
                    .iter()
                    .map(|(modality, rows, offsets)| GridItem {
                        modality: *modality,
                        grid: (!rows.is_empty()).then(|| {
                            let kind = if *modality == TokenSpeedWireModality::Video {
                                GridKind::Video
                            } else {
                                GridKind::Image
                            };
                            (kind, rows.clone())
                        }),
                        offsets,
                        seconds_per_grid: (*modality == TokenSpeedWireModality::Video)
                            .then(|| vec![2.0]),
                        audio_in_video: false,
                    })
                    .collect();
                let started = std::time::Instant::now();
                for _ in 0..iterations {
                    let table = rule.positions(&ids, &grid_items).unwrap();
                    std::hint::black_box(table);
                }
                let per = started.elapsed() / iterations;
                println!(
                    "| {rule_name} | {name} | {} | {:.1} us |",
                    ids.len(),
                    per.as_secs_f64() * 1e6
                );
            }
        }
    }

    /// The parity corpus: a JSON file of cases with the engine's own tables
    /// (`SMG_TOKENSPEED_MROPE_CORPUS=<file>`, produced by running the engine's
    /// `compute_mrope_positions` from its source on CPU; `config.rule` names
    /// the family, the Qwen3.5 one when absent). Every table a rule produces
    /// must match the engine's bit for bit; a case it refuses must be one the
    /// engine refused too, or (the Qwen3.5 rule) one the engine answered by
    /// rescanning the prompt with its fast path abandoned.
    #[test]
    #[ignore = "set SMG_TOKENSPEED_MROPE_CORPUS to the engine-oracle corpus file"]
    #[expect(clippy::print_stderr)]
    fn the_engine_oracle_corpus_matches_bit_for_bit() {
        let path = std::env::var("SMG_TOKENSPEED_MROPE_CORPUS").expect("corpus path");
        let corpus: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("corpus file"))
                .expect("corpus json");
        let config = &corpus["config"];
        let merge = config["spatial_merge_size"].as_u64().unwrap() as u32;
        let rule = match config["rule"].as_str().unwrap_or("qwen3_5") {
            "qwen3_omni" => PositionRule::Qwen3Omni(OmniLayout {
                spatial_merge_size: merge,
                // The engine's `getattr` defaults, which the launcher ships
                // for a thinker that leaves them unsaid.
                position_id_per_seconds: config["position_id_per_seconds"].as_f64().unwrap_or(13.0),
                seconds_per_chunk: config["seconds_per_chunk"].as_f64().unwrap_or(2.0),
            }),
            _ => PositionRule::Qwen35(MropeLayout {
                spatial_merge_size: merge,
                image_token_id: config["image_token_id"].as_u64().unwrap() as u32,
                video_token_id: config["video_token_id"].as_u64().unwrap() as u32,
            }),
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
                    let modality = match item["modality"].as_str().unwrap() {
                        "image" => TokenSpeedWireModality::Image,
                        "audio" => TokenSpeedWireModality::Audio,
                        _ => TokenSpeedWireModality::Video,
                    };
                    let kind = if modality == TokenSpeedWireModality::Video {
                        GridKind::Video
                    } else {
                        GridKind::Image
                    };
                    let grid = item["grid"].as_array().map(|rows| {
                        let rows = rows
                            .iter()
                            .map(|r| {
                                [
                                    r[0].as_i64().unwrap(),
                                    r[1].as_i64().unwrap(),
                                    r[2].as_i64().unwrap(),
                                ]
                            })
                            .collect();
                        (kind, rows)
                    });
                    let seconds_per_grid = item["seconds"]
                        .as_array()
                        .map(|values| values.iter().map(|v| v.as_f64().unwrap() as f32).collect());
                    GridItem {
                        modality,
                        grid,
                        offsets,
                        seconds_per_grid,
                        audio_in_video: item["audio_in_video"].as_bool().unwrap_or(false),
                    }
                })
                .collect();
            let expected = &case["expected"];
            let id = case["id"].as_str().unwrap_or("?");
            match rule.positions(&ids, &items) {
                Ok(table) => {
                    assert_ne!(
                        expected["refused"].as_bool(),
                        Some(true),
                        "case {id}: served where the engine refused ({})",
                        expected["error"]
                    );
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
                    assert!(
                        expected["refused"].as_bool() == Some(true)
                            || expected["fast_path"].as_bool() == Some(false),
                        "case {id}: refused ({reason}) where the engine's rule served it"
                    );
                    refused += 1;
                }
            }
        }
        eprintln!("corpus {path}: {checked} tables identical, {refused} refusals matched");
        assert!(checked > 0);
    }
}
