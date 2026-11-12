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
    init_log_video_decode_timing, missing_video_decoder, FrameSampling, ImageFetchConfig,
    MediaConnector, MediaConnectorConfig, MediaSource, VideoFetchConfig,
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

    use crate::{
        audio::DecodedAudio,
        vision::processor::{ModelSpecificValue, PreprocessedEncoderInputs},
    };

    /// Set to a non-empty value where the video tools must be installed (CI
    /// sets it on the shards that install ffmpeg): the tests that decode or
    /// write a clip then fail, instead of skipping, when ffmpeg, ffprobe or
    /// an encoder they use is missing.
    pub(crate) const REQUIRE_VIDEO_TOOLS: &str = "SMG_REQUIRE_VIDEO_TOOLS";

    /// Set to a non-empty value where libjpeg-turbo must be installed (CI
    /// sets it on the jobs that install the library): the TurboJPEG decode
    /// tests then fail, instead of skipping, when the library does not load.
    pub(crate) const REQUIRE_TURBOJPEG: &str = "SMG_REQUIRE_TURBOJPEG";

    /// Whether the TurboJPEG library loads, for a test that needs it: where
    /// it does not, a failure where [`REQUIRE_TURBOJPEG`] is set, else a
    /// skip said on stderr, so a run that requires the library cannot pass
    /// `test` without running it.
    #[expect(
        clippy::print_stderr,
        reason = "a skip says so; the harness shows it under --nocapture"
    )]
    pub(crate) fn turbojpeg_or_skip(test: &str) -> bool {
        if crate::jpeg_turbo::turbojpeg_available() {
            return true;
        }
        let required = std::env::var_os(REQUIRE_TURBOJPEG).is_some_and(|value| !value.is_empty());
        assert!(
            !required,
            "libturbojpeg not found, and {REQUIRE_TURBOJPEG} is set: {test}"
        );
        eprintln!("libturbojpeg not found: skipping {test}");
        false
    }

    /// Whether the TurboJPEG library loads and reads a call's warning
    /// (`tjGetErrorCode`, TurboJPEG 2.0 and later), for a test of a frame
    /// kept after a warning: a 1.x library decodes the same files and keeps
    /// none, so where the reading is missing, a failure where
    /// [`REQUIRE_TURBOJPEG`] is set, else a skip said on stderr.
    #[expect(
        clippy::print_stderr,
        reason = "a skip says so; the harness shows it under --nocapture"
    )]
    pub(crate) fn warned_decode_or_skip(test: &str) -> bool {
        if !turbojpeg_or_skip(test) {
            return false;
        }
        if crate::jpeg_turbo::turbojpeg_reads_warnings() {
            return true;
        }
        let required = std::env::var_os(REQUIRE_TURBOJPEG).is_some_and(|value| !value.is_empty());
        assert!(
            !required,
            "libturbojpeg reads no warning (TurboJPEG 1.x), and {REQUIRE_TURBOJPEG} is set: {test}"
        );
        eprintln!("libturbojpeg reads no warning: skipping {test}");
        false
    }

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

    /// `bytes` as lowercase hex, two digits a byte: the form the kernel tests
    /// pin their reference digests in.
    pub(crate) fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
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

    /// The audio processors' frame-geometry corpus: three clips per count in
    /// `sample_counts` at 16 kHz (a sine with LCG noise, quiet at 0.001, mid at
    /// 0.5, loud at 1.0; named `<samples> samples <label>`), then one 8 kHz clip
    /// of 8,000 samples through the resampler.
    pub(crate) fn waveform_corpus(sample_counts: &[usize]) -> Vec<(String, DecodedAudio)> {
        let mut corpus = Vec::new();
        for &samples in sample_counts {
            for (scale, label) in [(0.001_f32, "quiet"), (0.5, "mid"), (1.0, "loud")] {
                let mut state: u32 = 0x9E37_79B9 ^ samples as u32;
                let wave: Vec<f32> = (0..samples)
                    .map(|i| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let noise = (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
                        ((i as f32 * 0.03).sin() * 0.7 + noise * 0.3) * scale
                    })
                    .collect();
                corpus.push((
                    format!("{samples} samples {label}"),
                    DecodedAudio {
                        samples: wave,
                        sample_rate: 16_000,
                    },
                ));
            }
        }
        let wave: Vec<f32> = (0..8_000).map(|i| (i as f32 * 0.05).sin() * 0.4).collect();
        corpus.push((
            "8000 samples at 8 kHz".to_string(),
            DecodedAudio {
                samples: wave,
                sample_rate: 8_000,
            },
        ));
        corpus
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
