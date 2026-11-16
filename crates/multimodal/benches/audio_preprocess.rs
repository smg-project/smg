//! Benchmark: Qwen3 audio preprocessing — decode-side resample plus the
//! Whisper-compatible log-mel frontend.
//!
//! Run:  cargo bench -p llm-multimodal --bench audio_preprocess
//!
//! # What this bench is for, and what it is not for
//!
//! It is for **absolute regression tracking**: record a baseline with
//! `--save-baseline <name>`, compare a later run with `--baseline <name>`, and
//! notice when the cost of this path moves.
//!
//! It is **not** a tool for comparing two implementations of the same function.
//! Criterion runs benchmark ids sequentially, so a "before" group and an
//! "after" group are measured minutes apart; on a shared host, drift in load or
//! memory-bandwidth pressure between the two lands entirely in the ratio. That
//! is not hypothetical — while this path was being optimised, sequential arms
//! put a guard's overhead at +7.7% while an interleaved A/B/A/B estimator and
//! an order-swapped control both put it at +0.9%. Taking the sequential figure
//! would have rejected a change that was in fact free.
//!
//! For a before/after question, interleave the two arms sample-by-sample in one
//! process and report the distribution of per-pair ratios. Do not reach for
//! this file.
//!
//! Pin to one core (`taskset -c 0`) when the host is shared; wall time on a
//! pinned core tracked process CPU time to within 0.6% on the machine this was
//! developed on.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use llm_multimodal::audio::{DecodedAudio, Qwen3AudioProcessor};

/// A deterministic waveform: a tone plus a little broadband content, so the
/// mel filterbank sees energy across its bins rather than one spike. No RNG
/// seeded from the clock — two runs benchmark the same bytes.
fn waveform(sample_rate: usize, seconds: usize) -> Vec<f32> {
    let n = sample_rate * seconds;
    let mut state = 0x243f_6a88_u32;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            (i as f32 * 0.001).sin() * 0.45 + noise * 0.05
        })
        .collect()
}

/// The mel frontend alone: clips already at the processor's rate skip
/// `bandlimited_resample` entirely (`qwen3_audio.rs` compares the rates before
/// resampling), so this isolates the STFT and the mel projection.
fn bench_log_mel(c: &mut Criterion) {
    let processor = Qwen3AudioProcessor::new();
    let rate = processor.params().sample_rate;
    let mut group = c.benchmark_group("qwen3_audio_log_mel_16khz");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));

    for seconds in [1_usize, 10] {
        let clip = DecodedAudio {
            samples: waveform(rate, seconds),
            sample_rate: rate,
        };
        group.throughput(Throughput::Elements(clip.samples.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{seconds}s")),
            &clip,
            |b, clip| {
                b.iter(|| processor.preprocess_decoded_clips(&[clip]).unwrap());
            },
        );
    }
    group.finish();
}

/// Resample plus mel: the cost a clip at a non-native rate actually pays. The
/// gap against `qwen3_audio_log_mel_16khz` at the same duration is the
/// resampler's share.
fn bench_resample_and_log_mel(c: &mut Criterion) {
    let processor = Qwen3AudioProcessor::new();
    let mut group = c.benchmark_group("qwen3_audio_resample_and_log_mel");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(5));

    for (rate, seconds) in [(44_100_usize, 1_usize), (44_100, 10), (48_000, 10)] {
        let clip = DecodedAudio {
            samples: waveform(rate, seconds),
            sample_rate: rate,
        };
        group.throughput(Throughput::Elements(clip.samples.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}khz_{seconds}s", rate / 1000)),
            &clip,
            |b, clip| {
                b.iter(|| processor.preprocess_decoded_clips(&[clip]).unwrap());
            },
        );
    }
    group.finish();
}

/// A batch of unequal clips: every clip is padded to the longest before the
/// STFT, so the batch costs the longest clip times the batch size regardless of
/// what the short ones hold. Tracks that the padding behaviour does not regress.
fn bench_ragged_batch(c: &mut Criterion) {
    let processor = Qwen3AudioProcessor::new();
    let rate = processor.params().sample_rate;
    let clips: Vec<DecodedAudio> = [5_usize, 2, 1]
        .iter()
        .map(|seconds| DecodedAudio {
            samples: waveform(rate, *seconds),
            sample_rate: rate,
        })
        .collect();

    let mut group = c.benchmark_group("qwen3_audio_ragged_batch");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    let borrowed: Vec<&DecodedAudio> = clips.iter().collect();
    group.bench_function("5s_2s_1s", |b| {
        b.iter(|| processor.preprocess_decoded_clips(&borrowed).unwrap());
    });
    group.finish();
}

criterion_group!(
    audio,
    bench_log_mel,
    bench_resample_and_log_mel,
    bench_ragged_batch
);
criterion_main!(audio);
