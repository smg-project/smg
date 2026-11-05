//! Benchmark: a 30 s audio clip through the request path: the clip's bytes decoded to mono
//! samples, and the samples through the audio frontend's batch entry, the one the gateway calls
//! with a request's clips (it takes the decoded samples by copy today). One clip alone, both
//! steps together, and eight clips at once on the runtime, as a loaded gateway runs them. Beside
//! the rows, the allocation calls one clip makes on each step, counted by this binary's own
//! allocator.
//!
//! The clip is a 16 kHz PCM WAV written in memory (a tone with a little broadband content, the
//! same bytes every run), so the decode row prices the container walk and the sample conversion,
//! not a codec; a compressed clip costs more inside the decoder and the same after it. The
//! frontend's inner loops (the STFT and the mel projection on decoded samples) have rows of their
//! own in the frontend bench; this file prices the clip's path around them.
//!
//! Run:  cargo bench -p llm-multimodal --bench audio_clip

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator reports the allocation calls of a clip beside its time"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use llm_multimodal::{
    audio::{
        decode_audio_mono_f32, AudioPreProcessor, DecodedAudio, InklingAudioProcessor,
        ParakeetAudioProcessor, Qwen3AudioProcessor,
    },
    AudioClip, AudioSource, PreprocessedEncoderInputs,
};

/// Allocation and reallocation calls made through the global allocator so far.
static CALLS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the layout is the caller's, handed to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocation calls `work` makes, on whichever thread (the runtime is idle otherwise).
fn allocations_during<T>(work: impl FnOnce() -> T) -> usize {
    let before = CALLS.load(Ordering::Relaxed);
    black_box(work());
    CALLS.load(Ordering::Relaxed) - before
}

const SAMPLE_RATE: u32 = 16_000;
const SECONDS: u32 = 30;

/// A 30 s mono 16-bit PCM WAV: a tone plus a little broadband content, so the mel filter bank
/// sees energy across its bins; no clock-seeded randomness, two runs decode the same bytes.
fn wav() -> Vec<u8> {
    let count = SAMPLE_RATE * SECONDS;
    let data_bytes = count * 2;
    let mut bytes = Vec::with_capacity(44 + data_bytes as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_bytes.to_le_bytes());
    let mut state = 0x243f_6a88_u32;
    for i in 0..count {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let sample = (i as f32 * 0.001).sin() * 0.45 + noise * 0.05;
        bytes.extend_from_slice(&((sample * i16::MAX as f32) as i16).to_le_bytes());
    }
    bytes
}

/// The bytes decoded as the media connector decodes them: the clip's buffer handed over by
/// reference count, read in place.
fn decode(runtime: &tokio::runtime::Runtime, bytes: &Bytes) -> DecodedAudio {
    runtime
        .block_on(decode_audio_mono_f32(bytes.clone()))
        .unwrap()
}

/// The decoded clip as the gateway holds it for the frontend.
fn clip(decoded: DecodedAudio) -> Arc<AudioClip> {
    Arc::new(AudioClip::new(
        Bytes::new(),
        decoded,
        AudioSource::InlineBytes,
        String::new(),
    ))
}

/// A frontend's batch entry over the clips, the one the gateway calls.
fn mel(processor: &dyn AudioPreProcessor, clips: &[Arc<AudioClip>]) -> PreprocessedEncoderInputs {
    processor.preprocess(clips).unwrap()
}

/// `count` clips at once: each decoded on the runtime and run through the frontend on its
/// blocking pool, as a gateway under load runs them.
fn concurrent(
    runtime: &tokio::runtime::Runtime,
    processor: &Arc<Qwen3AudioProcessor>,
    bytes: &Bytes,
    count: usize,
) -> Vec<PreprocessedEncoderInputs> {
    runtime.block_on(async {
        let tasks = (0..count).map(|_| {
            let processor = Arc::clone(processor);
            let bytes = bytes.clone();
            async move {
                let decoded = decode_audio_mono_f32(bytes).await.unwrap();
                tokio::task::spawn_blocking(move || mel(&*processor, &[clip(decoded)]))
                    .await
                    .unwrap()
            }
        });
        futures::future::join_all(tasks).await
    })
}

#[expect(
    clippy::print_stdout,
    reason = "the allocation calls of a clip are printed beside the timing rows"
)]
fn bench_audio_clip(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();
    let processor = Arc::new(Qwen3AudioProcessor::new());
    assert_eq!(
        processor.params().sample_rate,
        SAMPLE_RATE as usize,
        "the clip is at the frontend's rate, so no resample runs"
    );
    let bytes = Bytes::from(wav());
    let decoded = decode(&runtime, &bytes);
    assert_eq!(decoded.samples.len(), (SAMPLE_RATE * SECONDS) as usize);
    let clips = vec![clip(decoded)];
    println!(
        "audio_clip/30s: {} bytes of WAV, {} samples; decode: {} allocation calls; mel through \
         the batch entry: {} allocation calls",
        bytes.len(),
        clips[0].decoded().samples.len(),
        allocations_during(|| decode(&runtime, &bytes)),
        allocations_during(|| mel(&*processor, &clips)),
    );

    let mut group = c.benchmark_group("audio_clip");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    group.bench_with_input(BenchmarkId::new("decode", "wav_30s"), &bytes, |b, bytes| {
        b.iter_with_large_drop(|| decode(&runtime, bytes));
    });
    group.bench_with_input(BenchmarkId::new("mel", "30s"), &clips, |b, clips| {
        b.iter_with_large_drop(|| mel(&*processor, clips));
    });
    group.bench_with_input(
        BenchmarkId::new("decode_and_mel", "30s"),
        &bytes,
        |b, bytes| {
            b.iter_with_large_drop(|| mel(&*processor, &[clip(decode(&runtime, bytes))]));
        },
    );
    group.throughput(Throughput::Elements(8));
    group.measurement_time(Duration::from_secs(10));
    group.bench_with_input(
        BenchmarkId::new("decode_and_mel", "30s_x8_concurrent"),
        &bytes,
        |b, bytes| {
            b.iter_with_large_drop(|| concurrent(&runtime, &processor, bytes, 8));
        },
    );
    group.finish();

    // The sibling frontends on the same clip: the dMel bins of the Inkling
    // family and the log-mel frames of the Parakeet family.
    let inkling = InklingAudioProcessor::new();
    let parakeet = ParakeetAudioProcessor::default();
    println!(
        "audio_clip/30s through the sibling frontends: inkling dmel {} allocation calls; parakeet \
         log-mel {} allocation calls",
        allocations_during(|| mel(&inkling, &clips)),
        allocations_during(|| mel(&parakeet, &clips)),
    );
    let mut group = c.benchmark_group("audio_clip_siblings");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    group.bench_with_input(
        BenchmarkId::new("inkling_dmel", "30s"),
        &clips,
        |b, clips| {
            b.iter_with_large_drop(|| mel(&inkling, clips));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("parakeet_log_mel", "30s"),
        &clips,
        |b, clips| {
            b.iter_with_large_drop(|| mel(&parakeet, clips));
        },
    );
    group.finish();
}

criterion_group!(audio, bench_audio_clip);
criterion_main!(audio);
