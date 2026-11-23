//! Audio preprocessing implementations.

pub(crate) mod decode;
pub(crate) mod processor;
pub(crate) mod processors;
pub(crate) mod transforms;

pub use decode::{decode_audio_mono_f32, DecodedAudio};
pub use processor::AudioPreProcessor;
pub use processors::{
    InklingAudioEncoderParams, InklingAudioProcessor, MuseSparkAudioConfigError,
    MuseSparkAudioParams, MuseSparkAudioProcessor, NemotronTranscribeAudioParams,
    NemotronTranscribeAudioProcessor, NemotronTranscribeConfigError, ParakeetAudioParams,
    ParakeetAudioProcessor, ParakeetConfigError, Qwen3AudioParams, Qwen3AudioProcessor,
    MUSE_SPARK_AUDIO_SAMPLES_PER_TOKEN, MUSE_SPARK_AUDIO_SAMPLE_RATE,
    MUSE_SPARK_SAMPLES_PER_AUDIO_KEY, NEMOTRON_AUDIO_LENGTH_KEY, NEMOTRON_MAX_PADDED_AUDIO_BYTES,
};
