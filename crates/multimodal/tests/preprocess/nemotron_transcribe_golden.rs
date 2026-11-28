//! Full-waveform parity with the pinned native vLLM Nemotron Transcribe processor.
//!
//! The generator executes the original native processor and audio parser. These
//! tests use the public processor entry point, including the production decoder
//! for stereo arrays and PCM WAVs, and compare every row of a native batch.
#![allow(clippy::expect_used, clippy::panic)]

use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
    sync::Arc,
};

use llm_multimodal::{
    audio::{
        decode_audio_mono_f32, AudioPreProcessor, DecodedAudio, NemotronTranscribeAudioProcessor,
        NEMOTRON_AUDIO_LENGTH_KEY,
    },
    registry::{ModelMetadata, ModelRegistry, Tokenizer},
    types::{AudioClip, AudioSource, Modality},
    vision::PreProcessorConfig,
    EncoderInput, ModelSpecificValue, PreprocessedEncoderInputs,
};
use serde::Deserialize;
use serde_json::{json, Value};

const GOLDEN: &str = include_str!("../fixtures/nemotron_transcribe_native.json");

#[derive(Deserialize)]
struct Golden {
    provenance: Provenance,
    config: Value,
    cases: Vec<Case>,
    batches: Vec<Batch>,
}

#[derive(Deserialize)]
struct Provenance {
    native_vllm_commit: String,
    model_config_sha256: String,
    executed_native_symbols: BTreeMap<String, NativeSource>,
    torch: String,
    torchaudio: String,
    numpy: String,
    transformers: String,
    method: String,
}

#[derive(Deserialize)]
struct NativeSource {
    sha256: String,
    symbols: Vec<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    Wav {
        input_wav: String,
    },
    Constant {
        input_num_samples: usize,
        constant_sample: f32,
    },
    Samples {
        input_samples: Samples,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Samples {
    Mono(Vec<f32>),
    Stereo(Vec<Vec<f32>>),
}

#[derive(Deserialize)]
struct Case {
    name: String,
    sample_rate: usize,
    #[serde(flatten)]
    input: Input,
    expected_length: usize,
    expected_tokens: usize,
    expected_dtype: String,
    length_dtype: String,
    length_device: String,
    expected_samples: Option<Vec<f32>>,
    expected_constant_samples: Option<f32>,
    expected_zero_padding: Option<usize>,
}

#[derive(Deserialize)]
struct Batch {
    name: String,
    case_names: Vec<String>,
    input_lengths: Vec<usize>,
    expected_lengths: Vec<usize>,
    expected_shapes: Vec<Vec<usize>>,
    expected_tokens: Vec<usize>,
    expected_samples: Vec<Vec<f32>>,
    expected_dtype: String,
    length_dtype: String,
    length_device: String,
}

fn assert_dtypes(waveform: &str, lengths: &str, device: &str) {
    assert_eq!(waveform, "torch.float32");
    assert_eq!(lengths, "torch.int64");
    assert_eq!(device, "cpu");
}

fn parse_fixture(source: &str) -> Golden {
    let golden: Golden = serde_json::from_str(source).expect("invalid native Transcribe fixture");
    let provenance = &golden.provenance;
    assert_eq!(
        provenance.native_vllm_commit,
        "a900ac810ecaedc0c2c5e9a8887e353cf1a5c350"
    );
    assert_eq!(
        provenance.model_config_sha256,
        "671882895b3711f93b2e88149e11271f4591680e2756976a43a2967992c946c7"
    );
    for (path, expected_sha256, required_symbols) in [
        (
            "transformers_utils/configs/nemotron3_5_transcribe.py",
            "56efdb3efef3d8d3b4ace79821c4263a4119094700b207cca023939fe13a07c8",
            &["resolve_audio_config"][..],
        ),
        (
            "utils/torch_utils.py",
            "9f39635fcfc84fcd0d2f517d84af29b7eb1cde8db0eeae2fe14ebe505ed163a3",
            &["set_default_torch_num_threads"][..],
        ),
        (
            "multimodal/audio.py",
            "5a3f5c492e5ec6f9c213d0c2898b83573e39179a8cb8b6c9b920347e55ebf476",
            &[
                "normalize_audio",
                "resample_audio_torchaudio",
                "AudioResampler",
            ][..],
        ),
        (
            "multimodal/parse.py",
            "0b7d5db662eb466e580708138cbf804ae894d770fba370b1159803c0af55ffff",
            &["MultiModalDataParser"][..],
        ),
        (
            "model_executor/models/nemotron3_5_transcribe.py",
            "e9f8023f9d5c6412f5a503c1fca508e32927010d7e3f079e2eec22fa203f3e09",
            &[
                "_num_audio_embeddings",
                "Nemotron3_5TranscribeProcessingInfo",
                "Nemotron3_5TranscribeMultiModalProcessor",
            ][..],
        ),
    ] {
        let source = provenance
            .executed_native_symbols
            .get(path)
            .unwrap_or_else(|| panic!("missing native oracle source {path}"));
        assert_eq!(
            source.sha256, expected_sha256,
            "{path}: native source identity"
        );
        for symbol in required_symbols {
            assert!(
                source.symbols.iter().any(|value| value == symbol),
                "{path}: missing {symbol}"
            );
        }
    }
    for version in [
        &provenance.torch,
        &provenance.torchaudio,
        &provenance.numpy,
        &provenance.transformers,
    ] {
        assert!(
            !version.is_empty(),
            "reference dependency version must be recorded"
        );
    }
    assert!(provenance.method.contains("compiled unchanged"));
    assert_eq!(
        golden.cases.len(),
        30,
        "the complete native case matrix is required"
    );
    let names: HashSet<_> = golden.cases.iter().map(|case| case.name.as_str()).collect();
    assert_eq!(names.len(), golden.cases.len(), "case names must be unique");
    for name in [
        "mono16k_constant_1",
        "mono16k_constant_159",
        "mono16k_constant_160",
        "mono16k_constant_161",
        "mono16k_constant_1279",
        "mono16k_constant_1280",
        "mono16k_constant_1281",
        "mono16k_constant_1439",
        "mono16k_constant_1440",
        "mono16k_constant_2559",
        "mono16k_constant_2560",
        "mono16k_constant_2561",
        "mono16k_constant_16000",
        "mono16k_signed_pcm",
        "stereo16k_channels_time",
        "stereo16k_time_channels",
        "pcm16wav_16000_1ch_159",
        "pcm16wav_16000_1ch_1440",
        "pcm16wav_16000_2ch_161",
        "pcm16wav_8000_1ch_97",
        "pcm16wav_44100_2ch_317",
        "pcm16wav_48000_1ch_481",
    ] {
        assert!(names.contains(name), "missing required native case {name}");
    }
    for rate in [8_000, 16_000, 22_050, 44_100, 48_000] {
        assert!(
            golden.cases.iter().any(|case| case.sample_rate == rate),
            "missing {rate} Hz coverage"
        );
    }
    assert!(
        !golden.batches.is_empty(),
        "a native variable-length batch is required"
    );
    for case in &golden.cases {
        assert!(
            case.sample_rate > 0 && case.expected_length > 0 && case.expected_tokens > 0,
            "{}: empty metadata",
            case.name
        );
        assert_dtypes(
            &case.expected_dtype,
            &case.length_dtype,
            &case.length_device,
        );
        // Check the complete expected vector before any zipped element comparison.
        assert_eq!(
            expected_samples(case).len(),
            case.expected_length,
            "{}: truncated waveform oracle",
            case.name
        );
    }
    golden
}

fn fixture() -> Golden {
    parse_fixture(GOLDEN)
}

#[test]
fn fixture_guards_reject_missing_coverage_and_truncated_waveforms() {
    #[derive(Debug)]
    enum Corruption {
        EmptyCases,
        EmptyBatches,
        TruncatedWaveform,
    }
    for corruption in [
        Corruption::EmptyCases,
        Corruption::EmptyBatches,
        Corruption::TruncatedWaveform,
    ] {
        let mut fixture: Value = serde_json::from_str(GOLDEN).expect("valid checked-in JSON");
        match corruption {
            Corruption::EmptyCases => fixture["cases"] = json!([]),
            Corruption::EmptyBatches => fixture["batches"] = json!([]),
            Corruption::TruncatedWaveform => {
                let case = fixture["cases"]
                    .as_array_mut()
                    .expect("case array")
                    .iter_mut()
                    .find(|case| case["name"] == "mono16k_signed_pcm")
                    .expect("explicit waveform case");
                case["expected_samples"]
                    .as_array_mut()
                    .expect("waveform array")
                    .pop();
            }
        }
        assert!(
            std::panic::catch_unwind(|| parse_fixture(&fixture.to_string())).is_err(),
            "fixture guard accepted {corruption:?}"
        );
    }
}

fn expected_samples(case: &Case) -> Vec<f32> {
    match (
        &case.expected_samples,
        case.expected_constant_samples,
        case.expected_zero_padding,
    ) {
        (Some(samples), None, None) => samples.clone(),
        (None, Some(value), Some(padding)) => {
            let count = match case.input {
                Input::Constant {
                    input_num_samples, ..
                } => input_num_samples,
                _ => panic!("{}: constant oracle requires constant input", case.name),
            };
            assert_eq!(
                count + padding,
                case.expected_length,
                "{}: oracle padding",
                case.name
            );
            let mut samples = vec![value; count];
            samples.resize(case.expected_length, 0.0);
            samples
        }
        _ => panic!(
            "{}: expected exactly one complete waveform oracle",
            case.name
        ),
    }
}

/// Encode the two reference array layouts as lossless float WAV, so channel
/// reduction is performed by the public production decoder.
fn stereo_wav(samples: &[Vec<f32>], rate: usize) -> Vec<u8> {
    assert!(!samples.is_empty(), "stereo input must contain frames");
    let interleaved: Vec<f32> = if samples.len() == 2 {
        assert!(!samples[0].is_empty());
        assert_eq!(samples[0].len(), samples[1].len());
        samples[0]
            .iter()
            .zip(&samples[1])
            .flat_map(|(&left, &right)| [left, right])
            .collect()
    } else {
        assert!(samples.iter().all(|frame| frame.len() == 2));
        samples.iter().flatten().copied().collect()
    };
    let data_bytes =
        u32::try_from(interleaved.len() * size_of::<f32>()).expect("small fixture WAV");
    let rate = u32::try_from(rate).expect("fixture sample rate fits WAV");
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&3_u16.to_le_bytes()); // IEEE float PCM.
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&(rate * 8).to_le_bytes());
    wav.extend_from_slice(&8_u16.to_le_bytes());
    wav.extend_from_slice(&32_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes.to_le_bytes());
    for sample in interleaved {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

async fn clip(case: &Case) -> Arc<AudioClip> {
    let decoded = match &case.input {
        Input::Wav { input_wav } => {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/nemotron_transcribe")
                .join(input_wav);
            let bytes = std::fs::read(path).expect("checked-in PCM WAV fixture");
            decode_audio_mono_f32(bytes.into())
                .await
                .expect("decode PCM WAV fixture")
        }
        Input::Samples {
            input_samples: Samples::Stereo(samples),
        } => decode_audio_mono_f32(stereo_wav(samples, case.sample_rate).into())
            .await
            .expect("decode lossless stereo fixture"),
        Input::Samples {
            input_samples: Samples::Mono(samples),
        } => DecodedAudio {
            samples: samples.clone(),
            sample_rate: case.sample_rate,
        },
        Input::Constant {
            input_num_samples,
            constant_sample,
        } => DecodedAudio {
            samples: vec![*constant_sample; *input_num_samples],
            sample_rate: case.sample_rate,
        },
    };
    assert_eq!(
        decoded.sample_rate, case.sample_rate,
        "{}: decoded rate",
        case.name
    );
    assert!(
        !decoded.samples.is_empty(),
        "{}: empty decoded input",
        case.name
    );
    Arc::new(AudioClip::new(
        bytes::Bytes::new(),
        decoded,
        AudioSource::InlineBytes,
        String::new(),
    ))
}

fn assert_lengths(output: &PreprocessedEncoderInputs, expected: &[usize], name: &str) {
    // The concrete variant gates signed i64 storage, rather than coercing a
    // differently typed output through a float conversion.
    match output.model_specific.get(NEMOTRON_AUDIO_LENGTH_KEY) {
        Some(ModelSpecificValue::IntTensor { data, shape }) => {
            assert_eq!(shape, &[expected.len()], "{name}: length shape");
            assert_eq!(
                data,
                &expected
                    .iter()
                    .map(|&length| length as i64)
                    .collect::<Vec<_>>(),
                "{name}: lengths"
            );
        }
        other => panic!("{name}: expected CPU int64 lengths, got {other:?}"),
    }
    assert!(
        matches!(output.encoder_input, EncoderInput::F32(_)),
        "{name}: waveform must be float32"
    );
    assert_eq!(
        output.item_sizes,
        expected
            .iter()
            .map(|&length| (length as u32, 1))
            .collect::<Vec<_>>(),
        "{name}: item sizes"
    );
}

fn assert_waveform(actual: &[f32], expected: &[f32], rate: usize, name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}: waveform value count");
    let tolerance = if rate == 16_000 { 0.0 } else { 1e-5 };
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name} sample {index}: {actual} != {expected} (tolerance {tolerance})"
        );
    }
}

#[tokio::test]
async fn every_clip_matches_the_native_waveform_oracle() {
    let golden = fixture();
    let processor =
        NemotronTranscribeAudioProcessor::from_config(&json!({"audio_config": golden.config}))
            .expect("native fixture geometry");
    for case in &golden.cases {
        let output = processor
            .preprocess(&[clip(case).await])
            .unwrap_or_else(|error| panic!("{}: preprocessing failed: {error}", case.name));
        assert_eq!(
            output.encoder_input.shape(),
            &[1, case.expected_length],
            "{}: shape",
            case.name
        );
        assert_eq!(
            output.feature_token_counts,
            &[case.expected_tokens],
            "{}: tokens",
            case.name
        );
        assert_lengths(&output, &[case.expected_length], &case.name);
        assert_waveform(
            &output.encoder_input.flat_f32(),
            &expected_samples(case),
            case.sample_rate,
            &case.name,
        );
    }
}

struct NoTokenizer;
impl Tokenizer for NoTokenizer {
    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }
    fn id_to_token(&self, _id: u32) -> Option<String> {
        None
    }
    fn encode_text(&self, _text: &str) -> Option<Vec<u32>> {
        None
    }
}

#[tokio::test]
async fn native_batches_preserve_every_row_length_order_and_padding() {
    let golden = fixture();
    let config = json!({"model_type": "nemotron3_5_transcribe", "audio_config": golden.config});
    let registry = ModelRegistry::new();
    let metadata = ModelMetadata {
        model_id: "native-transcribe",
        tokenizer: &NoTokenizer,
        config: &config,
    };
    let spec = registry
        .lookup(&metadata)
        .expect("public registry resolves native Transcribe");
    assert_eq!(spec.name(), "nemotron_transcribe");
    assert_eq!(
        spec.keep_on_cpu_keys_for(Modality::Audio),
        &[NEMOTRON_AUDIO_LENGTH_KEY]
    );
    assert_eq!(
        spec.encoder_input_key_for(Modality::Audio).as_deref(),
        Some("audio_signal")
    );
    let processor = spec
        .audio_processor(&config, &PreProcessorConfig::default())
        .expect("registry processor construction")
        .expect("audio processor exists");
    for batch in &golden.batches {
        let count = batch.case_names.len();
        assert!(
            count >= 2,
            "{}: actual native batch is required",
            batch.name
        );
        assert_dtypes(
            &batch.expected_dtype,
            &batch.length_dtype,
            &batch.length_device,
        );
        for length in [
            batch.input_lengths.len(),
            batch.expected_lengths.len(),
            batch.expected_shapes.len(),
            batch.expected_tokens.len(),
            batch.expected_samples.len(),
        ] {
            assert_eq!(length, count, "{}: incomplete batch oracle", batch.name);
        }
        assert!(
            batch
                .input_lengths
                .windows(2)
                .any(|pair| pair[0] != pair[1]),
            "{}: unequal clip lengths required",
            batch.name
        );
        let mut clips = Vec::with_capacity(count);
        let mut rates = Vec::with_capacity(count);
        for (index, name) in batch.case_names.iter().enumerate() {
            let case = golden
                .cases
                .iter()
                .find(|case| &case.name == name)
                .unwrap_or_else(|| panic!("{}: unknown input case {name}", batch.name));
            let input = clip(case).await;
            assert_eq!(
                input.decoded().samples.len(),
                batch.input_lengths[index],
                "{}: row {index} input length",
                batch.name
            );
            assert_eq!(
                batch.expected_shapes[index],
                &[batch.expected_lengths[index]],
                "{}: native row {index} shape",
                batch.name
            );
            assert_eq!(
                batch.expected_samples[index].len(),
                batch.expected_lengths[index],
                "{}: truncated native row {index}",
                batch.name
            );
            assert_eq!(
                batch.expected_lengths[index], case.expected_length,
                "{}: row {index} length",
                batch.name
            );
            assert_eq!(
                batch.expected_tokens[index], case.expected_tokens,
                "{}: row {index} tokens",
                batch.name
            );
            assert_waveform(
                &batch.expected_samples[index],
                &expected_samples(case),
                case.sample_rate,
                &batch.name,
            );
            clips.push(input);
            rates.push(case.sample_rate);
        }
        let output = processor
            .preprocess(&clips)
            .unwrap_or_else(|error| panic!("{}: batch preprocessing failed: {error}", batch.name));
        let width = *batch.expected_lengths.iter().max().expect("nonempty batch");
        assert_eq!(
            output.encoder_input.shape(),
            &[count, width],
            "{}: padded batch shape",
            batch.name
        );
        assert_eq!(
            output.feature_token_counts, batch.expected_tokens,
            "{}: batch tokens",
            batch.name
        );
        assert_lengths(&output, &batch.expected_lengths, &batch.name);
        let flat = output.encoder_input.flat_f32();
        assert_eq!(
            flat.len(),
            count * width,
            "{}: batch value count",
            batch.name
        );
        for index in 0..count {
            let length = batch.expected_lengths[index];
            let row = &flat[index * width..(index + 1) * width];
            assert_waveform(
                &row[..length],
                &batch.expected_samples[index],
                rates[index],
                &format!("{} row {index}", batch.name),
            );
            assert!(
                row[length..].iter().all(|&sample| sample == 0.0),
                "{}: row {index} batch padding",
                batch.name
            );
        }
    }
}
