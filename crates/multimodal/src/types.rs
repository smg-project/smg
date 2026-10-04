use std::{
    collections::{HashMap, HashSet},
    fmt,
    path::PathBuf,
    sync::{Arc, OnceLock},
};

use image::{DynamicImage, RgbImage};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    audio::DecodedAudio, error::MediaConnectorError, media::decode_image_bytes, vision::execution,
};

/// Supported multimodal modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Image,
    ImageEmbeds,
    Audio,
    Video,
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Modality::Image => write!(f, "image"),
            Modality::ImageEmbeds => write!(f, "image_embeds"),
            Modality::Audio => write!(f, "audio"),
            Modality::Video => write!(f, "video"),
        }
    }
}

/// Detail level passed by OpenAI style APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    #[default]
    Auto,
    Low,
    High,
}

/// A normalized content part understood by the tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
        /// MiniMax-M3 extension: cap the image's long side before preprocessing.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_long_side_pixel: Option<u32>,
    },
    ImageData {
        data: Vec<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    ImageEmbeds {
        payload: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
    AudioUrl {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
    AudioData {
        data: Vec<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
    VideoUrl {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
        /// MiniMax-M3 extension: frames per second to sample at.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fps: Option<f64>,
        /// MiniMax-M3 extension: cap each sampled frame's long side.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_long_side_pixel: Option<u32>,
    },
    VideoData {
        data: Vec<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
}

/// Image source metadata (useful for hashing & tracing).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    Url { url: String },
    DataUrl,
    InlineBytes,
    File { path: PathBuf },
}

/// Audio source metadata (useful for hashing & tracing).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AudioSource {
    Url { url: String },
    DataUrl,
    InlineBytes,
    File { path: PathBuf },
}

/// Video source metadata (useful for hashing & tracing).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VideoSource {
    Url { url: String },
    DataUrl,
    InlineBytes,
    File { path: PathBuf },
}

/// Concrete image payload captured by the media connector: the encoded bytes
/// with their identity and size, decoded on first use, so a request whose
/// pixels are already cached never pays for the decode.
#[derive(Debug, Clone)]
pub struct ImageFrame {
    decoded: OnceCell<DynamicImage>,
    /// Width and height the decode produces, from the header and the cap.
    size: ImageSize,
    /// The long-side cap the decode applies; `size` already reflects it.
    max_long_side_pixel: Option<u32>,
    pub raw_bytes: bytes::Bytes,
    pub detail: ImageDetail,
    pub source: ImageSource,
    /// Blake3 hex-digest of `raw_bytes` and the resolution cap.
    pub hash: String,
}

/// Decoded audio payload captured by the media connector.
#[derive(Debug, Clone)]
pub struct AudioClip {
    pub raw_bytes: bytes::Bytes,
    pub decoded: DecodedAudio,
    pub source: AudioSource,
    /// Blake3 hex-digest of raw_bytes, computed at decode time.
    pub hash: String,
}

/// How the decoded frames of a clip were sampled from the source stream.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoSamplingInfo {
    /// Source stream frame rate, as decoded.
    pub source_fps: f64,
    /// Source frame index behind each decoded frame, in order; one entry per decoded frame.
    pub frame_indices: Vec<usize>,
}

/// Decoded video payload captured by the media connector.
#[derive(Debug, Clone)]
pub struct VideoClip {
    pub frames: Vec<DynamicImage>,
    pub rgb_video: Option<DecodedRgbVideo>,
    /// Effective frame rate after connector-side sampling and frame-count clamps.
    pub sample_fps: f32,
    /// Source fps and sampled frame indices; `None` when the decoder could not recover them.
    pub sampling: Option<VideoSamplingInfo>,
    pub raw_bytes: bytes::Bytes,
    pub source: VideoSource,
    /// Blake3 hex-digest of raw_bytes, computed at decode time.
    pub hash: String,
    /// The long-side cap the caller asked for (MiniMax `max_long_side_pixel`);
    /// the frames were already scaled to it when set.
    pub max_long_side_pixel: Option<u32>,
}

/// Borrowed RGB frame data for video preprocessors.
#[derive(Debug, Clone, Copy)]
pub struct RgbFrameRef<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
}

/// One decoded RGB frame inside a shared decoded-video byte buffer.
#[derive(Debug, Clone)]
pub struct DecodedRgbFrame {
    pub width: u32,
    pub height: u32,
    pub offset: usize,
    pub len: usize,
}

/// Decoded RGB video frames backed by one shared byte buffer.
#[derive(Debug, Clone)]
pub struct DecodedRgbVideo {
    pub data: bytes::Bytes,
    pub frames: Vec<DecodedRgbFrame>,
}

impl DecodedRgbVideo {
    pub fn new(data: bytes::Bytes, frames: Vec<DecodedRgbFrame>) -> Self {
        Self { data, frames }
    }

    pub fn frame_refs(&self) -> Result<Vec<RgbFrameRef<'_>>, String> {
        self.frames
            .iter()
            .map(|frame| {
                let end = frame
                    .offset
                    .checked_add(frame.len)
                    .ok_or_else(|| "decoded RGB frame offset overflow".to_string())?;
                let data = self
                    .data
                    .get(frame.offset..end)
                    .ok_or_else(|| "decoded RGB frame range is out of bounds".to_string())?;
                Ok(RgbFrameRef {
                    width: frame.width,
                    height: frame.height,
                    data,
                })
            })
            .collect()
    }

    pub fn to_dynamic_images(&self) -> Result<Vec<DynamicImage>, String> {
        let mut images = Vec::with_capacity(self.frames.len());
        for frame in &self.frames {
            let end = frame
                .offset
                .checked_add(frame.len)
                .ok_or_else(|| "decoded RGB frame offset overflow".to_string())?;
            let data = self
                .data
                .get(frame.offset..end)
                .ok_or_else(|| "decoded RGB frame range is out of bounds".to_string())?;
            let image =
                RgbImage::from_raw(frame.width, frame.height, data.to_vec()).ok_or_else(|| {
                    format!(
                        "failed to build RGB frame from {} bytes for {}x{} video",
                        frame.len, frame.width, frame.height
                    )
                })?;
            images.push(DynamicImage::ImageRgb8(image));
        }
        Ok(images)
    }
}

impl VideoClip {
    pub fn new(
        frames: Vec<DynamicImage>,
        raw_bytes: bytes::Bytes,
        source: VideoSource,
        hash: String,
    ) -> Self {
        Self::new_with_sample_fps(frames, raw_bytes, source, hash, 2.0)
    }

    pub fn new_with_sample_fps(
        frames: Vec<DynamicImage>,
        raw_bytes: bytes::Bytes,
        source: VideoSource,
        hash: String,
        sample_fps: f32,
    ) -> Self {
        Self {
            frames,
            rgb_video: None,
            sample_fps,
            sampling: None,
            raw_bytes,
            source,
            hash,
            max_long_side_pixel: None,
        }
    }

    pub fn new_rgb(
        rgb_video: DecodedRgbVideo,
        raw_bytes: bytes::Bytes,
        source: VideoSource,
        hash: String,
    ) -> Self {
        Self::new_rgb_with_sample_fps(rgb_video, raw_bytes, source, hash, 2.0)
    }

    pub fn new_rgb_with_sample_fps(
        rgb_video: DecodedRgbVideo,
        raw_bytes: bytes::Bytes,
        source: VideoSource,
        hash: String,
        sample_fps: f32,
    ) -> Self {
        Self {
            frames: Vec::new(),
            rgb_video: Some(rgb_video),
            sample_fps,
            sampling: None,
            raw_bytes,
            source,
            hash,
            max_long_side_pixel: None,
        }
    }

    pub fn with_sampling(mut self, sampling: Option<VideoSamplingInfo>) -> Self {
        self.sampling = sampling;
        self
    }

    /// Record the long-side cap the frames were decoded under.
    pub fn with_max_long_side_pixel(mut self, max_long_side_pixel: Option<u32>) -> Self {
        self.max_long_side_pixel = max_long_side_pixel;
        self
    }

    pub fn frames(&self) -> &[DynamicImage] {
        &self.frames
    }

    pub fn max_long_side_pixel(&self) -> Option<u32> {
        self.max_long_side_pixel
    }

    pub fn rgb_video(&self) -> Option<&DecodedRgbVideo> {
        self.rgb_video.as_ref()
    }

    pub fn sample_fps(&self) -> f32 {
        self.sample_fps
    }

    pub fn sampling(&self) -> Option<&VideoSamplingInfo> {
        self.sampling.as_ref()
    }

    pub fn materialized_frames(&self) -> Result<Vec<DynamicImage>, String> {
        if !self.frames.is_empty() {
            return Ok(self.frames.clone());
        }
        self.rgb_video
            .as_ref()
            .ok_or_else(|| "video clip has no decoded frames".to_string())?
            .to_dynamic_images()
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw_bytes
    }

    pub fn source(&self) -> &VideoSource {
        &self.source
    }
}

impl AudioClip {
    pub fn new(
        raw_bytes: bytes::Bytes,
        decoded: DecodedAudio,
        source: AudioSource,
        hash: String,
    ) -> Self {
        Self {
            raw_bytes,
            decoded,
            source,
            hash,
        }
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw_bytes
    }

    pub fn decoded(&self) -> &DecodedAudio {
        &self.decoded
    }

    pub fn source(&self) -> &AudioSource {
        &self.source
    }
}

impl ImageFrame {
    /// A frame from pixels already decoded.
    pub fn new(
        image: DynamicImage,
        raw_bytes: bytes::Bytes,
        detail: ImageDetail,
        source: ImageSource,
        hash: String,
    ) -> Self {
        let size = ImageSize::new(image.width(), image.height());
        Self {
            decoded: OnceCell::with_value(image),
            size,
            max_long_side_pixel: None,
            raw_bytes,
            detail,
            source,
            hash,
        }
    }

    /// A frame from encoded bytes whose header gave `size` (after the cap);
    /// the pixels are decoded, and capped, when first asked for.
    pub fn encoded(
        raw_bytes: bytes::Bytes,
        detail: ImageDetail,
        source: ImageSource,
        hash: String,
        size: ImageSize,
        max_long_side_pixel: Option<u32>,
    ) -> Self {
        Self {
            decoded: OnceCell::new(),
            size,
            max_long_side_pixel,
            raw_bytes,
            detail,
            source,
            hash,
        }
    }

    /// The decoded pixels, decoding on the first call. A body its header did
    /// not vouch for fails here, as the caller's input.
    pub fn image(&self) -> Result<&DynamicImage, MediaConnectorError> {
        self.decoded
            .get_or_try_init(|| decode_image_bytes(&self.raw_bytes, self.max_long_side_pixel))
    }

    /// Decode the pixels of every frame that is still encoded, in parallel on
    /// the preprocessing pool (inline under `Parallelism::Inline`), instead of
    /// one photo after another on the calling thread. [`image`](Self::image)
    /// is a lookup for each of them afterwards.
    pub fn decode_all(frames: &[Arc<Self>]) -> Result<(), MediaConnectorError> {
        let pending = Self::undecoded(frames);
        if pending.len() < 2 {
            return pending
                .into_iter()
                .try_for_each(|frame| frame.image().map(|_| ()));
        }
        let failures: Vec<OnceLock<MediaConnectorError>> =
            pending.iter().map(|_| OnceLock::new()).collect();
        execution::scope(|spawner| {
            for (frame, failure) in pending.iter().copied().zip(&failures) {
                spawner.spawn(move |_| {
                    if let Err(error) = frame.image() {
                        // One slot, one attempt: the cell is empty here.
                        let _ = failure.set(error);
                    }
                });
            }
        });
        failures
            .into_iter()
            .find_map(OnceLock::into_inner)
            .map_or(Ok(()), Err)
    }

    /// The distinct frames of `frames` that still have to be decoded. A
    /// repeated part is the same `Arc` several times over; it decodes once,
    /// and the repeats must not become tasks that park on its cell.
    fn undecoded(frames: &[Arc<Self>]) -> Vec<&Self> {
        let mut seen = HashSet::with_capacity(frames.len());
        frames
            .iter()
            .map(Arc::as_ref)
            .filter(|frame| {
                frame.decoded.get().is_none() && seen.insert(std::ptr::from_ref(*frame))
            })
            .collect()
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw_bytes
    }

    pub fn source(&self) -> &ImageSource {
        &self.source
    }

    pub fn size(&self) -> ImageSize {
        ImageSize::new(self.size.width, self.size.height)
    }
}

/// Container for all supported multimodal media objects.
#[derive(Debug, Clone)]
pub enum TrackedMedia {
    Image(Arc<ImageFrame>),
    Audio(Arc<AudioClip>),
    Video(Arc<VideoClip>),
    /// Placeholder variants for future modalities.
    Embeddings,
}

pub type MultiModalData = HashMap<Modality, Vec<TrackedMedia>>;
pub type MultiModalUUIDs = HashMap<Modality, Vec<Option<String>>>;

pub type TokenId = i32;

/// Declares how a multimodal tensor's first dimension maps to media items.
///
/// Used by [`crate::registry::ModelProcessorSpec::encoder_field_layouts_for`] to tell the backend
/// how to split tensors for per-item scheduling (vLLM `MultiModalFieldConfig`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldLayout {
    /// First dimension equals number of media items (one slice per item).
    Batched,
    /// Variable-length slices per item. The sizes are stored in the tensor
    /// named by `sizes_key` (e.g. `"patches_per_image"` or `"patches_per_video"`).
    Flat { sizes_key: String },
}

impl FieldLayout {
    /// Convenience constructor for `Flat`.
    pub fn flat(sizes_key: impl Into<String>) -> Self {
        Self::Flat {
            sizes_key: sizes_key.into(),
        }
    }
}

/// Layout contract for one modality's encoder inputs.
///
/// The primary encoder input is transported independently from named,
/// model-specific side tensors. Keeping its layout typed avoids leaking a
/// vision-specific field name into audio and other modality processors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderFieldLayouts {
    pub encoder_input: FieldLayout,
    pub model_specific: HashMap<String, FieldLayout>,
}

impl EncoderFieldLayouts {
    pub fn new(encoder_input: FieldLayout, model_specific: HashMap<String, FieldLayout>) -> Self {
        Self {
            encoder_input,
            model_specific,
        }
    }

    /// Convert the legacy HF/vLLM-shaped field map into the neutral contract.
    ///
    /// Existing vision specs use `pixel_values` for the primary encoder input.
    /// New specs should construct [`Self`] directly instead.
    pub fn from_legacy_fields(mut fields: HashMap<String, FieldLayout>) -> Self {
        let encoder_input = fields
            .remove("pixel_values")
            .unwrap_or(FieldLayout::Batched);
        Self::new(encoder_input, fields)
    }
}

impl Default for EncoderFieldLayouts {
    fn default() -> Self {
        Self::new(FieldLayout::Batched, HashMap::new())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

impl ImageSize {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct PlaceholderRange {
    pub offset: usize,
    pub length: usize,
}

#[derive(Debug, Clone)]
pub struct PromptReplacement {
    pub modality: Modality,
    pub placeholder_token: String,
    pub tokens: Vec<TokenId>,
    /// Feature-token ranges relative to the start of `tokens`.
    ///
    /// Most model specs leave this unset and let the gateway discover feature
    /// ranges by scanning for the configured placeholder token id. Specs whose
    /// transport fill token is also a structural token can set explicit ranges
    /// so the structural header is not mistaken for an encoder position.
    pub feature_ranges: Option<Vec<PlaceholderRange>>,
    /// Number of structural tokens the chat template emits *immediately before*
    /// this placeholder (e.g. Qwen's leading `<|vision_start|>`) that belong to
    /// the placeholder's range. `expand_tokens` folds them into the reported
    /// [`PlaceholderRange`] without re-emitting them, so backends that scan the
    /// range for structural markers see the leading marker. vLLM's video mrope
    /// walks each frame from `<|vision_start|>` starting at the range offset, so
    /// the offset must sit on (or before) the first marker. 0 for the common
    /// case where the range is exactly the replacement.
    pub structural_prefix: usize,
}

impl PromptReplacement {
    pub fn repeated(
        modality: Modality,
        placeholder_token: &str,
        token_id: TokenId,
        count: usize,
    ) -> Self {
        Self {
            modality,
            placeholder_token: placeholder_token.to_string(),
            tokens: vec![token_id; count],
            feature_ranges: None,
            structural_prefix: 0,
        }
    }

    pub fn sequence(modality: Modality, placeholder_token: &str, sequence: Vec<TokenId>) -> Self {
        Self {
            modality,
            placeholder_token: placeholder_token.to_string(),
            tokens: sequence,
            feature_ranges: None,
            structural_prefix: 0,
        }
    }

    /// Declare the encoder-feature ranges inside the replacement sequence.
    /// Offsets are relative to the first replacement token.
    #[must_use]
    pub fn with_feature_ranges(mut self, ranges: Vec<PlaceholderRange>) -> Self {
        self.feature_ranges = Some(ranges);
        self
    }

    /// Declare one contiguous encoder-feature span inside the replacement.
    #[must_use]
    pub fn with_feature_span(self, offset: usize, length: usize) -> Self {
        self.with_feature_ranges(vec![PlaceholderRange { offset, length }])
    }

    /// Declare that `n` template-emitted structural tokens precede this
    /// placeholder and should be included in its reported range. See
    /// [`Self::structural_prefix`].
    #[must_use]
    pub fn with_structural_prefix(mut self, n: usize) -> Self {
        self.structural_prefix = n;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_range_serializes() {
        let range = PlaceholderRange {
            offset: 10,
            length: 4,
        };
        let json = serde_json::to_string(&range).unwrap();
        assert!(json.contains("offset"));
    }

    #[test]
    fn prompt_replacement_builders() {
        let rep = PromptReplacement::repeated(Modality::Image, "<image>", 100, 3);
        assert_eq!(rep.tokens, vec![100, 100, 100]);
        assert!(rep.feature_ranges.is_none());

        let rep = rep.with_feature_span(1, 2);
        assert_eq!(
            rep.feature_ranges,
            Some(vec![PlaceholderRange {
                offset: 1,
                length: 2
            }])
        );
    }

    #[test]
    fn video_clip_sampling_defaults_to_none_and_follows_the_builder() {
        let clip = VideoClip::new(
            Vec::new(),
            bytes::Bytes::new(),
            VideoSource::InlineBytes,
            "hash".to_string(),
        );
        assert!(clip.sampling().is_none());

        let rgb = VideoClip::new_rgb(
            DecodedRgbVideo::new(bytes::Bytes::new(), Vec::new()),
            bytes::Bytes::new(),
            VideoSource::InlineBytes,
            "hash".to_string(),
        );
        assert!(rgb.sampling().is_none());

        let sampling = VideoSamplingInfo {
            source_fps: 30.0,
            frame_indices: vec![0, 15, 30],
        };
        let clip = clip.with_sampling(Some(sampling.clone()));
        assert_eq!(clip.sampling(), Some(&sampling));
        assert!(clip.with_sampling(None).sampling().is_none());
    }

    #[test]
    fn legacy_encoder_fields_are_split_into_typed_layouts() {
        let layouts = EncoderFieldLayouts::from_legacy_fields(HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
        ]));

        assert_eq!(
            layouts.encoder_input,
            FieldLayout::flat("patches_per_image")
        );
        assert_eq!(
            layouts.model_specific,
            HashMap::from([("image_grid_thw".to_string(), FieldLayout::Batched)])
        );
    }

    fn encoded_fixture() -> ImageFrame {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/images/tiny.jpg"
        ))
        .expect("fixture");
        let (width, height) = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .expect("format")
            .into_dimensions()
            .expect("header");
        ImageFrame::encoded(
            bytes::Bytes::from(bytes),
            ImageDetail::Auto,
            ImageSource::InlineBytes,
            String::new(),
            ImageSize::new(width, height),
            None,
        )
    }

    #[test]
    fn decode_all_decodes_a_repeated_frame_once() {
        let first = Arc::new(encoded_fixture());
        let second = Arc::new(encoded_fixture());
        let mut frames: Vec<Arc<ImageFrame>> = (0..70).map(|_| Arc::clone(&first)).collect();
        frames.push(Arc::clone(&second));

        // 70 repeats of one frame plus one other: two decodes, not 71 tasks.
        let pending = ImageFrame::undecoded(&frames);
        assert_eq!(pending.len(), 2);
        assert!(std::ptr::eq(pending[0], &*first));
        assert!(std::ptr::eq(pending[1], &*second));

        ImageFrame::decode_all(&frames).expect("decode");
        let image = first.image().expect("decoded");
        for frame in &frames[..70] {
            assert!(std::ptr::eq(frame.image().expect("decoded"), image));
        }
        assert!(ImageFrame::undecoded(&frames).is_empty());
    }

    #[test]
    fn decode_all_reports_a_bad_repeated_frame_once_and_decodes_the_rest() {
        let bad = Arc::new(ImageFrame::encoded(
            bytes::Bytes::from_static(b"not an image"),
            ImageDetail::Auto,
            ImageSource::InlineBytes,
            String::new(),
            ImageSize::new(1, 1),
            None,
        ));
        let good = Arc::new(encoded_fixture());
        let frames = vec![Arc::clone(&bad), Arc::clone(&good), Arc::clone(&bad)];
        assert_eq!(ImageFrame::undecoded(&frames).len(), 2);
        assert!(ImageFrame::decode_all(&frames).is_err());
        assert!(good.image().is_ok());
    }
}
