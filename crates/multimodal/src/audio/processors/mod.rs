//! Model-specific audio preprocessing implementations.

mod inkling;
mod muse_spark;
mod parakeet;
mod qwen3_audio;

pub use inkling::{InklingAudioEncoderParams, InklingAudioProcessor};
pub use muse_spark::{
    MuseSparkAudioConfigError, MuseSparkAudioParams, MuseSparkAudioProcessor,
    MUSE_SPARK_AUDIO_SAMPLES_PER_TOKEN, MUSE_SPARK_AUDIO_SAMPLE_RATE,
    MUSE_SPARK_SAMPLES_PER_AUDIO_KEY,
};
pub use parakeet::{ParakeetAudioParams, ParakeetAudioProcessor, ParakeetConfigError};
pub use qwen3_audio::{Qwen3AudioParams, Qwen3AudioProcessor};
