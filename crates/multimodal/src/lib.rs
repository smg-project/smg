pub mod audio;
pub mod encoder_inputs;
pub mod error;
pub mod hasher;
pub mod hub;
pub mod jpeg_turbo;
pub mod media;
#[cfg(feature = "opencv-video")]
mod opencv_buffer;
pub mod registry;
pub mod tracker;
pub mod types;
pub mod vision;

pub use audio::AudioPreProcessor;
pub use encoder_inputs::{
    f32_to_bf16_bits, f32_to_f16_bits, EncoderDtype, EncoderInput, EncoderInputView,
    ModelSpecificValue, PixelNorm, PreprocessedEncoderInputs,
};
pub use error::{MediaConnectorError, MultiModalError, MultiModalResult, TransformError};
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
