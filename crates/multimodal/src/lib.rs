pub mod audio;
pub mod buffer_pool;
pub mod data_url;
pub mod encoder_inputs;
pub mod error;
mod fetch_policy;
pub use fetch_policy::effective_allowed_domains;
pub mod hasher;
#[cfg(feature = "hf-hub")]
pub mod hub;
pub mod jpeg_turbo;
pub mod media;
pub mod minimax_resize;
mod mp4_metadata;
#[cfg(feature = "opencv-video")]
mod opencv_buffer;
mod pixel_convert;
pub mod registry;
pub mod tracker;
pub mod types;
mod video_seek;
pub mod vision;
pub mod webp_native;
mod wire_bytes;

pub use audio::AudioPreProcessor;
pub use encoder_inputs::{
    f32_to_bf16_bits, f32_to_f16_bits, EncoderDtype, EncoderInput, EncoderInputView,
    ModelSpecificValue, PixelNorm, PreprocessedEncoderInputs,
};
pub use error::{
    MediaConnectorError, MediaErrorCode, MediaFault, MultiModalError, MultiModalResult,
    TransformError,
};
pub use media::{
    init_log_video_decode_timing, FrameSampling, ImageFetchConfig, MediaConnector,
    MediaConnectorConfig, MediaSource, VideoFetchConfig,
};
pub use registry::{
    EncoderInputFormat, MediaItemInfo, MediaPartOrder, ModelMetadata, ModelProcessorSpec,
    ModelRegistry, Tokenizer, DEEPSEEK_V41_IMAGE_PLACEHOLDER,
};
pub use tracker::{
    AsyncMultiModalTracker, FetchObserver, FetchOutcome, FetchReport, FetchSourceKind,
    TrackerOutput,
};
pub use types::{
    AudioClip, AudioSource, EncoderFieldLayouts, FieldLayout, ImageDetail, ImageFrame, ImageSize,
    ImageSource, MediaContentPart, Modality, MultiModalData, MultiModalUUIDs, PlaceholderRange,
    PromptReplacement, RgbFrameRef, TokenId, TrackedMedia, TransparencyPolicy, VideoClip,
    VideoSamplingInfo, VideoSource,
};
// Re-export vision processing components
pub use vision::execution::{configure_parallelism, parallelism, Parallelism, POOL_THREADS_ENV};
pub use vision::{
    DeepseekV41Processor, LlavaNextProcessor, LlavaProcessor, PreProcessorConfig, VideoClipOptions,
    VisionPreProcessor, VisionPreprocessingContext,
};
pub use wire_bytes::WireBytes;

/// Support for this crate's unit tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::process::Command;

    use image::{DynamicImage, Rgb, RgbImage};
    use rustls::crypto::ring;

    use crate::vision::processor::{ModelSpecificValue, PreprocessedEncoderInputs};

    /// Set to a non-empty value where the video tools must be installed (CI
    /// sets it on the shards that install ffmpeg): the tests that decode or
    /// write a clip then fail, instead of skipping, when ffmpeg, ffprobe or
    /// an encoder they use is missing.
    pub(crate) const REQUIRE_VIDEO_TOOLS: &str = "SMG_REQUIRE_VIDEO_TOOLS";

    /// Reports that `missing` keeps a test or a case from running: a failure
    /// where [`REQUIRE_VIDEO_TOOLS`] is set, a skip said on stderr elsewhere.
    #[expect(
        clippy::print_stderr,
        reason = "a skip says so; the harness shows it under --nocapture"
    )]
    pub(crate) fn skip_for(missing: &str) {
        let required = std::env::var_os(REQUIRE_VIDEO_TOOLS).is_some_and(|value| !value.is_empty());
        assert!(!required, "{missing}, and {REQUIRE_VIDEO_TOOLS} is set");
        eprintln!("{missing}: skipping; the tests that need no clip still run");
    }

    /// Whether ffmpeg and ffprobe run on this host; says so when they do not.
    pub(crate) fn video_tools_available() -> bool {
        let runs = |tool: &str| {
            Command::new(tool)
                .arg("-version")
                .output()
                .is_ok_and(|out| out.status.success())
        };
        let available = runs("ffmpeg") && runs("ffprobe");
        if !available {
            skip_for("ffmpeg or ffprobe is not installed");
        }
        available
    }

    /// The HTTP client builder for the tests' media connectors. The library
    /// takes its client's builder from the caller; the tests make it here,
    /// with the process-level TLS crypto provider installed first (reqwest's
    /// `rustls-no-provider` client panics at build time without one).
    #[expect(
        clippy::disallowed_methods,
        reason = "a test binary's client factory; the process TLS crypto provider is installed first"
    )]
    pub(crate) fn http_client_builder() -> reqwest::ClientBuilder {
        let _ = ring::default_provider().install_default();
        reqwest::Client::builder()
    }

    /// A solid image of the given size and colour: what the vision tests
    /// build their inputs with.
    pub(crate) fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    /// `len` bytes from a linear congruential generator seeded with `seed`
    /// (`state * 1103515245 + 12345`, masked to 31 bits, byte = bits 16..24):
    /// the pixels the kernel tests feed the reference libraries to pin their
    /// outputs, reproducible in any language.
    pub(crate) fn lcg_bytes(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
                ((state >> 16) & 0xff) as u8
            })
            .collect()
    }

    /// A `width` by `height` RGB image of [`lcg_bytes`], interleaved.
    pub(crate) fn lcg_rgb_image(width: u32, height: u32, seed: u32) -> DynamicImage {
        let bytes = lcg_bytes((width * height * 3) as usize, seed);
        DynamicImage::ImageRgb8(RgbImage::from_raw(width, height, bytes).expect("image size"))
    }

    /// `w` by `h` RGB bytes that vary with the position and `seed`: the
    /// resamplers' parity inputs.
    pub(crate) fn gradient_rgb(w: usize, h: usize, seed: usize) -> Vec<u8> {
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                data[i] = ((seed + x * 7 + y * 3) % 256) as u8;
                data[i + 1] = ((seed + x * 5 + y * 11) % 256) as u8;
                data[i + 2] = ((seed + x + y * 2) % 256) as u8;
            }
        }
        data
    }

    /// `actual` is `expected` byte for byte; otherwise the test fails naming
    /// `what`, the first differing byte and the two values there.
    pub(crate) fn assert_same_bytes(actual: &[u8], expected: &[u8], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        if let Some(first) = actual.iter().zip(expected).position(|(a, b)| a != b) {
            panic!(
                "{what}: first difference at byte {first}: {} vs {}",
                actual[first], expected[first]
            );
        }
    }

    /// The assertions the processors' two-image tests share: both images
    /// present, a 2-D encoder input, a grid of two rows of (T, H, W) and a
    /// per-image patch count that sums to the encoder input's rows.
    pub(crate) fn assert_two_image_batch(result: &PreprocessedEncoderInputs) {
        assert_eq!(result.item_sizes.len(), 2);
        assert_eq!(result.feature_token_counts.len(), 2);
        assert_eq!(result.encoder_input.ndim(), 2);
        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("image_grid_thw")
        {
            assert_eq!(shape, &[2, 3]);
            assert_eq!(data.len(), 6);
        } else {
            panic!("Expected image_grid_thw to be IntTensor");
        }
        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            result.model_specific.get("patches_per_image")
        {
            assert_eq!(shape, &[2]);
            assert_eq!(data.len(), 2);
            let total: i64 = data.iter().sum();
            assert_eq!(total as usize, result.encoder_input.shape()[0]);
        } else {
            panic!("Expected patches_per_image to be IntTensor");
        }
    }
}
