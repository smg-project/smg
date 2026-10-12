pub mod audio;
pub mod encoder_inputs;
pub mod error;
mod fetch_policy;
pub use fetch_policy::effective_allowed_domains;
pub mod hasher;
pub mod hub;
pub mod jpeg_turbo;
pub mod media;
mod mp4_metadata;
#[cfg(feature = "opencv-video")]
mod opencv_buffer;
mod pixel_convert;
pub mod registry;
pub mod tracker;
pub mod types;
pub mod vision;
pub mod webp_native;

pub use audio::AudioPreProcessor;
pub use encoder_inputs::{
    f32_to_bf16_bits, f32_to_f16_bits, EncoderDtype, EncoderInput, EncoderInputView,
    ModelSpecificValue, PixelNorm, PreprocessedEncoderInputs,
};
pub use error::{
    MediaConnectorError, MediaFault, MultiModalError, MultiModalResult, TransformError,
};
pub use media::{
    init_log_video_decode_timing, FrameSampling, ImageFetchConfig, MediaConnector,
    MediaConnectorConfig, MediaSource, VideoFetchConfig,
};
pub use registry::{
    MediaItemInfo, MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistry, Tokenizer,
    DEEPSEEK_V41_IMAGE_PLACEHOLDER,
};
pub use tracker::{AsyncMultiModalTracker, TrackerOutput};
pub use types::{
    AudioClip, AudioSource, EncoderFieldLayouts, FieldLayout, ImageDetail, ImageFrame, ImageSize,
    ImageSource, MediaContentPart, Modality, MultiModalData, MultiModalUUIDs, PlaceholderRange,
    PromptReplacement, RgbFrameRef, TokenId, TrackedMedia, VideoClip, VideoSamplingInfo,
    VideoSource,
};
// Re-export vision processing components
pub use vision::execution::{configure_parallelism, parallelism, Parallelism, POOL_THREADS_ENV};
pub use vision::{
    DeepseekV41Processor, LlavaNextProcessor, LlavaProcessor, PreProcessorConfig,
    VisionPreProcessor, VisionProcessorRegistry,
};

/// Support for this crate's unit tests.
#[cfg(test)]
pub(crate) mod test_support {
    use rustls::crypto::ring;

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
}
