//! Bit-level identity gate for the Qwen3 audio log-mel frontend.
//!
//! [`super::audio_mel_parity`] checks the frontend against
//! `transformers.WhisperFeatureExtractor` on a **1e-3 tolerance**, because a
//! pure-Rust `rustfft` and resampler cannot be bitwise equal to torch. Observed
//! diffs there are 2e-5 to 5e-5, so that test carries 20-50x of slack: it is the
//! right oracle for "have we drifted from the reference" and the wrong one for
//! "is this byte-for-byte what the previous commit produced".
//!
//! This file is the second oracle. It pins an FNV-1a fingerprint of the raw
//! `f32` bits of `encoder_input` for a fixed, adversarial corpus, so a change
//! that is meant to be bit-identical is proven bit-identical rather than merely
//! close. A refactor that alters the last mantissa bit fails here and passes
//! there, which is the point.
//!
//! **The float digest is compared on `x86_64` only.** It is FNV-1a over raw `f32`
//! bits, and `rustfft` makes no promise that the last mantissa bit is the same on
//! every architecture, so the recorded value describes the machine that recorded it
//! as well as this crate. The integer assertions -- `shape` and
//! `feature_token_counts` -- run everywhere, because those drive prompt length and
//! tensor slicing and an off-by-one in either is a wrong answer rather than a drift.
//!
//! The portable form of the bit-identity claim is not here: it is in
//! `audio::processors::qwen3_audio::tests`
//! (`the_batched_frontend_equals_the_frame_by_frame_computation_bit_for_bit`), where
//! the batched frontend is compared against the frame-by-frame computation in the same
//! process. That is what actually proves a change altered nothing, on whatever target
//! runs it. This file adds a second, stricter signal on one architecture -- it is not
//! the load-bearing gate.
//!
//! The fingerprints were generated from the unmodified frontend, so they cannot
//! have been fitted to a later change. To regenerate after a deliberate
//! numerical change:
//!
//! ```text
//! SMG_WRITE_QWEN3_MEL_BITS=1 cargo test -p llm-multimodal --test preprocess_tests \
//!     qwen3_audio_mel_bits -- --nocapture
//! ```
//!
//! and replace `tests/fixtures/golden/qwen3_audio_mel_bits.json` with the
//! printed document. Regenerating is a deliberate act: it asserts that the
//! numbers were *supposed* to move.

#![allow(clippy::expect_used, clippy::panic)]
#![expect(
    clippy::print_stdout,
    reason = "fixture regeneration prints the document"
)]

use std::sync::Arc;

use llm_multimodal::{
    audio::{AudioPreProcessor, DecodedAudio, Qwen3AudioProcessor},
    types::{AudioClip, AudioSource},
};
use serde::Deserialize;

use super::fnv1a_f32;

const GOLDEN: &str = include_str!("../fixtures/golden/qwen3_audio_mel_bits.json");

#[derive(Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    sample_rate: usize,
    samples: usize,
    shape: Vec<usize>,
    feature_token_counts: Vec<usize>,
    fnv1a_f32: String,
}

/// One corpus entry: a deterministic waveform, no RNG seeded from the clock.
struct Signal {
    name: &'static str,
    rate: usize,
    samples: Vec<f32>,
}

/// The corpus, chosen to exercise the paths a numerical change is most likely
/// to disturb: a smooth tone, broadband noise, an isolated impulse, amplitudes
/// at both ends of the `f32` range, a signal that is silent until its tail, and
/// alternating samples at Nyquist. Plus three length edges: shorter than one
/// FFT window, exactly one hop, and a clip whose length is not a hop multiple.
fn corpus() -> Vec<Signal> {
    const N: usize = 48_000;
    // xorshift32 with a fixed constant: reproducible on every platform.
    let mut state = 0x243f_6a88_u32;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    vec![
        Signal {
            name: "sine",
            rate: 16_000,
            samples: (0..N).map(|i| (i as f32 * 0.001).sin() * 0.5).collect(),
        },
        Signal {
            name: "white_noise",
            rate: 16_000,
            samples: (0..N).map(|_| noise()).collect(),
        },
        Signal {
            name: "impulse",
            rate: 16_000,
            samples: {
                let mut v = vec![0.0_f32; N];
                v[7] = 1.0;
                v[N / 2] = -1.0;
                v
            },
        },
        Signal {
            name: "very_quiet",
            rate: 16_000,
            samples: (0..N).map(|i| (i as f32 * 0.01).sin() * 1e-30).collect(),
        },
        Signal {
            name: "loud_square",
            rate: 16_000,
            samples: (0..N)
                .map(|i| {
                    if (i / 37) % 2 == 0 {
                        30_000.0
                    } else {
                        -30_000.0
                    }
                })
                .collect(),
        },
        Signal {
            name: "silence_then_tone",
            rate: 16_000,
            samples: {
                let mut v = vec![0.0_f32; N];
                for sample in v.iter_mut().skip(N - 1_000) {
                    *sample = 0.9;
                }
                v
            },
        },
        Signal {
            name: "alternating_nyquist",
            rate: 16_000,
            samples: (0..N)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        },
        Signal {
            name: "silence",
            rate: 16_000,
            samples: vec![0.0; 4_800],
        },
        // Shorter than n_fft (400): exercises the reflect-pad edge.
        Signal {
            name: "sub_window",
            rate: 16_000,
            samples: (0..300).map(|i| (i as f32 * 0.05).sin() * 0.3).collect(),
        },
        // Exactly one hop, the minimum the processor accepts.
        Signal {
            name: "single_hop",
            rate: 16_000,
            samples: (0..160).map(|i| (i as f32 * 0.1).cos() * 0.4).collect(),
        },
        // Not a whole number of hops: the trailing partial frame is dropped.
        Signal {
            name: "ragged_length",
            rate: 16_000,
            samples: (0..1_237).map(|i| (i as f32 * 0.02).sin() * 0.6).collect(),
        },
        // Resample paths: these also gate `bandlimited_resample`, since a change
        // there moves the mel bits downstream of it.
        Signal {
            name: "resample_44100",
            rate: 44_100,
            samples: (0..44_100)
                .map(|i| (i as f32 * 0.0004).sin() * 0.5)
                .collect(),
        },
        Signal {
            name: "resample_48000",
            rate: 48_000,
            samples: (0..48_000)
                .map(|i| (i as f32 * 0.0003).sin() * 0.5)
                .collect(),
        },
        Signal {
            name: "resample_8000",
            rate: 8_000,
            samples: (0..8_000).map(|i| (i as f32 * 0.002).sin() * 0.5).collect(),
        },
    ]
}

/// The processor's output for one signal, through the public batched entry
/// point production uses.
fn fingerprint(signal: &Signal) -> (Vec<usize>, Vec<usize>, String) {
    let clip = Arc::new(AudioClip::new(
        bytes::Bytes::new(),
        DecodedAudio {
            samples: signal.samples.clone(),
            sample_rate: signal.rate,
        },
        AudioSource::InlineBytes,
        String::new(),
    ));
    let output = Qwen3AudioProcessor::new()
        .preprocess(&[clip])
        .unwrap_or_else(|error| panic!("{}: preprocessing failed: {error}", signal.name));
    let encoder = output.encoder_input.as_f32();
    let values = encoder
        .as_slice_memory_order()
        .unwrap_or_else(|| panic!("{}: encoder_input must be contiguous", signal.name));
    (
        output.encoder_input.shape().to_vec(),
        output.feature_token_counts.clone(),
        fnv1a_f32(values),
    )
}

#[test]
fn qwen3_audio_log_mel_is_bit_identical_to_the_recorded_fingerprints() {
    if std::env::var_os("SMG_WRITE_QWEN3_MEL_BITS").is_some() {
        let cases: Vec<String> = corpus()
            .iter()
            .map(|signal| {
                let (shape, counts, digest) = fingerprint(signal);
                format!(
                    "    {{\n      \"name\": {:?},\n      \"sample_rate\": {},\n      \
                     \"samples\": {},\n      \"shape\": {:?},\n      \
                     \"feature_token_counts\": {:?},\n      \"fnv1a_f32\": {:?}\n    }}",
                    signal.name,
                    signal.rate,
                    signal.samples.len(),
                    shape,
                    counts,
                    digest
                )
            })
            .collect();
        println!(
            "{{\n  \"digest\": \"FNV-1a over the little-endian f32 bits of encoder_input\",\n  \
             \"purpose\": \"bit-identity gate; audio_mel_parity is a 1e-3 tolerance test and \
             cannot distinguish bit-identical from merely close\",\n  \"cases\": [\n{}\n  ]\n}}",
            cases.join(",\n")
        );
        return;
    }

    let golden: Golden =
        serde_json::from_str(GOLDEN).expect("invalid checked-in qwen3 audio mel bit fixture");
    assert_eq!(
        golden.cases.len(),
        corpus().len(),
        "fixture case count does not match the corpus; regenerate deliberately"
    );

    for (signal, case) in corpus().iter().zip(&golden.cases) {
        assert_eq!(
            signal.name, case.name,
            "corpus and fixture are out of order"
        );
        assert_eq!(
            signal.samples.len(),
            case.samples,
            "{}: the corpus waveform changed length",
            case.name
        );
        assert_eq!(
            signal.rate, case.sample_rate,
            "{}: the corpus sample rate changed",
            case.name
        );

        let (shape, counts, digest) = fingerprint(signal);
        assert_eq!(shape, case.shape, "{}: encoder_input shape", case.name);
        assert_eq!(
            counts, case.feature_token_counts,
            "{}: feature_token_counts",
            case.name
        );
        // Compared on x86_64 only, and the asymmetry is deliberate.
        //
        // This digest is FNV-1a over raw `f32` bits. `rustfft` does not promise the
        // same last mantissa bit on every architecture, so a recorded digest is a
        // property of the machine that recorded it as much as of this crate --
        // checking it on aarch64 tests the host, not the code.
        //
        // What survives on every target is above: `shape` and `feature_token_counts`
        // are integers, they drive prompt length and tensor slicing, an off-by-one in
        // either is a wrong answer rather than a drift, and they are asserted
        // unconditionally.
        //
        // The portable form of the bit-identity claim lives in
        // `audio::processors::qwen3_audio::tests`, where the batched frontend is
        // compared against the frame-by-frame computation in the *same process*. That
        // proves "this change altered nothing" on whatever target runs it, which is
        // the property this fixture was added to defend.
        //
        // `cfg!` rather than `#[cfg]` so the comparison still type-checks everywhere.
        if cfg!(target_arch = "x86_64") {
            assert_eq!(
                digest, case.fnv1a_f32,
                "{}: log-mel bits differ from the recorded fingerprint. A change meant to be \
                 bit-identical is not. If the numbers were supposed to move, regenerate with \
                 SMG_WRITE_QWEN3_MEL_BITS=1 and say so in the commit message.",
                case.name
            );
        }
    }
}
