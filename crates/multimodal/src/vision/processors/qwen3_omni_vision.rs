//! Qwen3-Omni image and video preprocessing.
//!
//! This processor shares patchification and normalization through
//! `QwenVLProcessorBase`, but cannot use `Qwen3VLProcessor` directly. Omni
//! applies video pixel limits per frame rather than across the sampled clip,
//! gives its video preprocessor config precedence for video-specific limits,
//! and emits the timing metadata mixed-modality M-RoPE needs at the rate its
//! processor fixes rather than the clip's sampled rate.

use image::DynamicImage;

use super::qwen_vl_base::{
    QwenVLConfig, QwenVLProcessorBase, QwenVideoResizeMode, QwenVideoTiming,
};
use crate::{
    encoder_inputs::EncoderDtype,
    types::{Modality, RgbFrameRef},
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{PreprocessedEncoderInputs, VideoClipOptions, VisionPreProcessor},
        transforms::TransformError,
    },
};

pub(crate) const QWEN3_OMNI_MEAN: [f64; 3] = [0.5; 3];
pub(crate) const QWEN3_OMNI_STD: [f64; 3] = [0.5; 3];
pub(crate) const DEFAULT_IMAGE_MIN_PIXELS: usize = 3_136;
pub(crate) const DEFAULT_IMAGE_MAX_PIXELS: usize = 12_845_056;
pub(crate) const DEFAULT_VIDEO_MIN_PIXELS: usize = 128 * 32 * 32;
pub(crate) const DEFAULT_VIDEO_MAX_PIXELS: usize = 768 * 32 * 32;
pub(crate) const DEFAULT_PATCH_SIZE: usize = 16;
pub(crate) const DEFAULT_MERGE_SIZE: usize = 2;
pub(crate) const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;

/// The `fps` argument of the checkpoint's processor. It computes a clip's
/// `video_second_per_grid`, the time one temporal patch spans in the
/// position ids, as the temporal patch size over this argument, which
/// defaults to 1.0 and which the serving path leaves at its default: two
/// seconds a temporal patch, whatever rate the clip was sampled at.
const PROCESSOR_FPS_ARGUMENT: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct Qwen3OmniVisionProcessor {
    inner: QwenVLProcessorBase,
}

impl Default for Qwen3OmniVisionProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen3OmniVisionProcessor {
    pub(crate) fn new() -> Self {
        Self::with_limits(
            DEFAULT_IMAGE_MIN_PIXELS,
            DEFAULT_IMAGE_MAX_PIXELS,
            DEFAULT_VIDEO_MIN_PIXELS,
            DEFAULT_VIDEO_MAX_PIXELS,
            DEFAULT_PATCH_SIZE,
            DEFAULT_MERGE_SIZE,
            DEFAULT_TEMPORAL_PATCH_SIZE,
        )
    }

    fn with_limits(
        image_min_pixels: usize,
        image_max_pixels: usize,
        video_min_pixels: usize,
        video_max_pixels: usize,
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
    ) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels: image_min_pixels,
                max_pixels: image_max_pixels,
                video_min_pixels,
                video_max_pixels,
                video_resize_mode: QwenVideoResizeMode::PerFrame,
                temporal_patch_size,
                mean: QWEN3_OMNI_MEAN,
                std: QWEN3_OMNI_STD,
                model_name: "qwen3-omni",
            })
            .with_video_timing(QwenVideoTiming::FixedRate(PROCESSOR_FPS_ARGUMENT)),
        }
    }

    fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        let configured_min = config.min_pixels.or_else(|| config.get_shortest_edge());
        let configured_max = config.max_pixels.or_else(|| config.get_longest_edge());
        Self::with_limits(
            configured_min.unwrap_or(DEFAULT_IMAGE_MIN_PIXELS),
            configured_max.unwrap_or(DEFAULT_IMAGE_MAX_PIXELS),
            DEFAULT_VIDEO_MIN_PIXELS,
            DEFAULT_VIDEO_MAX_PIXELS,
            config.get_patch_size(DEFAULT_PATCH_SIZE),
            config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
            config
                .temporal_patch_size
                .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
        )
        .with_transform_config(config)
    }

    fn from_video_preprocessor_config(config: &PreProcessorConfig) -> Self {
        let configured_min = config.min_pixels.or_else(|| config.get_shortest_edge());
        let configured_max = config.max_pixels.or_else(|| config.get_longest_edge());
        Self::with_limits(
            DEFAULT_IMAGE_MIN_PIXELS,
            DEFAULT_IMAGE_MAX_PIXELS,
            configured_min.unwrap_or(DEFAULT_VIDEO_MIN_PIXELS),
            configured_max.unwrap_or(DEFAULT_VIDEO_MAX_PIXELS),
            config.get_patch_size(DEFAULT_PATCH_SIZE),
            config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
            config
                .temporal_patch_size
                .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
        )
        .with_transform_config(config)
    }

    fn with_transform_config(mut self, config: &PreProcessorConfig) -> Self {
        self.inner = self.inner.with_preprocessor_config(config);
        self
    }

    /// See [`QwenVLProcessorBase::validate`].
    pub(crate) fn validate(&self) -> Result<(), TransformError> {
        self.inner.validate()
    }

    /// Build a processor for a resolved image or video configuration.
    pub(crate) fn from_config_for(config: &PreProcessorConfig, modality: Modality) -> Self {
        if modality == Modality::Video && !config.is_image_only_processor_type() {
            Self::from_video_preprocessor_config(config)
        } else {
            // The checkpoint's one preprocessor_config.json carries the image
            // bounds. The family's composite processor hands its video
            // processor a size of its own at call time, the module's video
            // defaults (128 and 768 merged patches of 32 x 32 pixels per
            // frame), over whatever that processor read from the file, so
            // the file's image bounds must not become the per-frame video
            // budget here.
            Self::from_preprocessor_config(config)
        }
    }
}

impl VisionPreProcessor for Qwen3OmniVisionProcessor {
    fn emits_pixel_bytes(&self) -> bool {
        true
    }

    fn default_mean(&self) -> [f64; 3] {
        self.inner.default_mean()
    }

    fn default_std(&self) -> [f64; 3] {
        self.inner.default_std()
    }

    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.inner.preprocess(images)
    }

    fn preprocess_wire(
        &self,
        images: &[&DynamicImage],
        dtype: EncoderDtype,
    ) -> Option<Result<PreprocessedEncoderInputs, TransformError>> {
        self.inner.preprocess_wire(images, dtype)
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.inner.preprocess_video(frames, clip)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.inner.preprocess_video_rgb(frames, clip)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        self.inner.calculate_num_tokens(width, height)
    }

    /// Number of stored patch elements for one image.
    fn image_input_elements(&self, width: u32, height: u32) -> Option<usize> {
        self.inner.image_input_elements(width, height)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        self.inner.get_processed_size()
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, RgbImage};

    use super::*;
    use crate::{
        test_support::{hex, lcg_rgb_image},
        vision::{processor::ModelSpecificValue, processors::Qwen3VLProcessor},
    };

    #[test]
    fn omni_multiframe_resize_differs_from_qwen3_volume_budget() {
        let omni = Qwen3OmniVisionProcessor::new();
        let qwen3_vl = Qwen3VLProcessor::with_config(
            DEFAULT_PATCH_SIZE,
            DEFAULT_MERGE_SIZE,
            DEFAULT_VIDEO_MIN_PIXELS,
            DEFAULT_VIDEO_MAX_PIXELS,
            DEFAULT_TEMPORAL_PATCH_SIZE,
        );

        assert_eq!(omni.inner.min_pixels(), DEFAULT_IMAGE_MIN_PIXELS);
        assert_eq!(omni.inner.max_pixels(), DEFAULT_IMAGE_MAX_PIXELS);
        assert_eq!(omni.inner.video_min_pixels(), DEFAULT_VIDEO_MIN_PIXELS);
        assert_eq!(omni.inner.video_max_pixels(), DEFAULT_VIDEO_MAX_PIXELS);
        assert_eq!(
            omni.inner.video_resize_mode(),
            QwenVideoResizeMode::PerFrame
        );

        let omni_size = omni.inner.smart_resize_video(16, 720, 1280).unwrap();
        let volume_size = qwen3_vl.smart_resize_video(16, 720, 1280).unwrap();

        assert_eq!(omni_size, (640, 1152));
        assert_eq!(volume_size, (160, 288));
        assert_ne!(omni_size, volume_size);
    }

    #[test]
    fn video_preprocessor_config_overrides_per_frame_limits() {
        let config = PreProcessorConfig::from_json(
            r#"{"size":{"shortest_edge":65536,"longest_edge":262144},"temporal_patch_size":2}"#,
        )
        .unwrap();
        let processor = Qwen3OmniVisionProcessor::from_video_preprocessor_config(&config);

        assert_eq!(processor.inner.min_pixels(), DEFAULT_IMAGE_MIN_PIXELS);
        assert_eq!(processor.inner.max_pixels(), DEFAULT_IMAGE_MAX_PIXELS);
        assert_eq!(processor.inner.video_min_pixels(), 65_536);
        assert_eq!(processor.inner.video_max_pixels(), 262_144);
        assert_eq!(
            processor.inner.smart_resize_video(32, 720, 1280).unwrap(),
            (384, 672)
        );
    }

    /// The checkpoint ships one `preprocessor_config.json` for its image and
    /// its video processor, but the family's composite processor passes its
    /// own video size at call time, so a clip keeps the per-frame video
    /// defaults and never takes the file's image bounds: a 720p frame is
    /// sized to 1152x640 (36 x 20 patches), not kept at 1280x720.
    #[test]
    fn shared_image_config_keeps_omni_video_defaults() {
        let config = PreProcessorConfig::from_json(
            r#"{"image_processor_type":"Qwen2VLImageProcessor","min_pixels":3136,"max_pixels":12845056,"patch_size":16,"merge_size":2,"temporal_patch_size":2}"#,
        )
        .unwrap();
        let processor = Qwen3OmniVisionProcessor::from_config_for(&config, Modality::Video);

        assert_eq!(processor.inner.video_min_pixels(), DEFAULT_VIDEO_MIN_PIXELS);
        assert_eq!(processor.inner.video_max_pixels(), DEFAULT_VIDEO_MAX_PIXELS);
        assert_eq!(
            processor.inner.smart_resize_video(16, 720, 1280).unwrap(),
            (640, 1152)
        );
    }

    #[test]
    fn empty_config_uses_omni_half_normalization() {
        let image = DynamicImage::ImageRgb8(RgbImage::new(32, 32));
        let output = Qwen3OmniVisionProcessor::new()
            .preprocess(&[&image])
            .unwrap();

        assert!(output
            .encoder_input
            .as_f32()
            .iter()
            .all(|value| (*value + 1.0).abs() < 1e-6));
    }

    /// The checkpoint's processor computes `video_second_per_grid` as the
    /// temporal patch size over its own `fps` argument, 1.0 unless a caller
    /// passes one, and the serving path passes none: a two-frame temporal
    /// patch spans two seconds whatever rate the clip was sampled at, where
    /// the Qwen2-VL line's processors report the sampled rate's share (0.5
    /// at 4 frames a second).
    #[test]
    fn video_second_per_grid_is_the_processors_constant() {
        let config = PreProcessorConfig::from_json(
            r#"{"size":{"shortest_edge":1024,"longest_edge":65536}}"#,
        )
        .unwrap();
        let frames = vec![
            DynamicImage::ImageRgb8(RgbImage::new(32, 32)),
            DynamicImage::ImageRgb8(RgbImage::new(32, 32)),
        ];
        let processor = Qwen3OmniVisionProcessor::from_config_for(&config, Modality::Video);

        for sampled_fps in [0.8, 2.0, 4.0] {
            let output = processor
                .preprocess_video(&frames, &VideoClipOptions::sampled_at(sampled_fps))
                .unwrap();
            assert!(
                matches!(
                    output.model_specific.get("video_second_per_grid"),
                    Some(ModelSpecificValue::Tensor { data, shape })
                        if data == &vec![2.0] && shape == &vec![1]
                ),
                "sampled at {sampled_fps}"
            );
        }
    }

    #[test]
    fn the_element_estimate_of_an_image_is_the_bases() {
        let omni = Qwen3OmniVisionProcessor::new();
        // 4096x3136 is exactly the default max_pixels and a multiple of the
        // 32-pixel factor, so it is kept as is: 3136 * 4096 * 3 * 2.
        assert_eq!(omni.image_input_elements(4096, 3136), Some(77_070_336));
        assert_eq!(
            omni.image_input_elements(4096, 3136),
            omni.inner.image_input_elements(4096, 3136)
        );
    }

    /// `output` against one run of the reference on the same bytes: the
    /// shape and the grid of its tensor, four of its f32 values by index and
    /// the SHA-256 of them all.
    fn assert_reference(
        output: &PreprocessedEncoderInputs,
        grid_key: &str,
        shape: [usize; 2],
        grid: [i64; 3],
        witnesses: [(usize, f32); 4],
        digest: &str,
    ) {
        use sha2::{Digest, Sha256};

        assert_eq!(output.encoder_input.shape(), &shape, "{grid_key}");
        assert!(
            matches!(
                output.model_specific.get(grid_key),
                Some(ModelSpecificValue::IntTensor { data, shape })
                    if data[..] == grid && shape[..] == [1, 3]
            ),
            "{grid_key} is not {grid:?}"
        );
        let bytes = output.encoder_input.write_as(EncoderDtype::Float32);
        let bytes: &[u8] = &bytes;
        for (index, reference) in witnesses {
            let value = f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap());
            assert!(
                value.to_bits() == reference.to_bits(),
                "{grid_key} {grid:?} element {index}: {value} (the reference {reference})"
            );
        }
        assert_eq!(hex(&Sha256::digest(bytes)), digest, "{grid_key} {grid:?}");
    }

    /// Two seeded images and one clip of three seeded frames through the
    /// processor as it is built without a config, against the reference
    /// image and video processors of the family on the same bytes under the
    /// family's own bounds: an image 3,136 to 12,845,056 pixels, a clip
    /// frame 128 to 768 merged patches, sized frame by frame. Each input
    /// comes out at another size than under the bounds of
    /// `Qwen3VLProcessor`, whose fixtures pin the shared base: 37x23, under
    /// the image floor, at 64x96; 259x194, above it, at 192x256; a 70x100
    /// frame, under the frame floor, at 448x320, the third frame repeated to
    /// fill the second temporal patch.
    #[test]
    fn images_and_a_clip_match_the_reference_bit_for_bit() {
        let omni = Qwen3OmniVisionProcessor::new();

        let image = lcg_rgb_image(37, 23, 7);
        assert_reference(
            &omni.preprocess(&[&image]).unwrap(),
            "image_grid_thw",
            [24, 1536],
            [1, 4, 6],
            [
                (0, -0.2627451),
                (1, -0.19215687),
                (18_509, 0.62352943),
                (36_863, 0.56078434),
            ],
            "6e6980c78b7555ec04d617ddadc4065c392edc8f21c768efa8da2b2c13d9806f",
        );

        let image = lcg_rgb_image(259, 194, 11);
        assert_reference(
            &omni.preprocess(&[&image]).unwrap(),
            "image_grid_thw",
            [192, 1536],
            [1, 12, 16],
            [
                (0, 0.043137256),
                (1, 0.34901962),
                (147_533, 0.44313726),
                (294_911, -0.96862745),
            ],
            "2e54dfc29ab8569c631a2305df465c2d10e5c35e07b47a89fc6f12eee146539b",
        );

        let frames: Vec<DynamicImage> = (21..24).map(|seed| lcg_rgb_image(70, 100, seed)).collect();
        assert_reference(
            &omni
                .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
                .unwrap(),
            "video_grid_thw",
            [1120, 1536],
            [2, 28, 20],
            [
                (0, -0.6313726),
                (1, -0.56078434),
                (860_237, -0.25490198),
                (1_720_319, -0.654902),
            ],
            "9742fb8bb1ba4e768a46becee972a412660931e11b81ad011e4b400db0347e5e",
        );
    }
}
