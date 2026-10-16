//! Audio preprocessing implementations.

pub mod decode;
pub mod processor;
pub mod processors;
pub(crate) mod transforms;

pub use decode::{decode_audio_mono_f32, DecodedAudio};
pub use processor::AudioPreProcessor;
pub use processors::{
    InklingAudioEncoderParams, InklingAudioProcessor, MuseSparkAudioConfigError,
    MuseSparkAudioParams, MuseSparkAudioProcessor, ParakeetAudioParams, ParakeetAudioProcessor,
    ParakeetConfigError, Qwen3AudioParams, Qwen3AudioProcessor, MUSE_SPARK_AUDIO_SAMPLES_PER_TOKEN,
    MUSE_SPARK_AUDIO_SAMPLE_RATE, MUSE_SPARK_SAMPLES_PER_AUDIO_KEY,
};
