//! Preprocessing integration tests: the image and audio processors against the reference
//! processors' goldens, the bit-identity fingerprints of the resize and preprocess kernels, and the
//! wire layout.

mod audio_mel_parity;
mod decoded_rgb_reference;
mod deepseek_v41_preprocess_golden;
mod gemma4_preprocess_golden;
mod inkling_preprocess_golden;
mod muse_spark_audio_golden;
mod muse_spark_preprocess_golden;
mod nemotron_omni_preprocess_golden;
mod preprocess_fingerprint;
mod qwen_pixel_norm_golden;
mod qwen_preprocess_golden;
mod qwen_smart_resize_golden;
mod resize_fingerprint;
mod vision_golden_tests;
mod wire_parity;
