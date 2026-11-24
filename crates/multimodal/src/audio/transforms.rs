//! Shared digital-signal-processing transforms for audio frontends.

use std::{
    f32::consts::PI as PI_F32,
    f64::consts::PI,
    mem::size_of,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use crate::error::TransformError;

pub(super) fn hann_window(window_size: usize) -> Vec<f32> {
    (0..window_size)
        .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / window_size as f64).cos()) as f32)
        .collect()
}

fn hz_to_mel(frequency: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f64.ln() / 27.0;
    if frequency >= min_log_hz {
        min_log_mel + (frequency / min_log_hz).ln() / logstep
    } else {
        frequency / f_sp
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * f_sp
    }
}

/// Per filter of `filters` (`n_mels` rows of `fft_bins` weights), the half-open
/// bin range `[first, last)` outside which its weight is zero. A filter is a
/// triangle over the bins, so its weight is one contiguous band and a bin
/// carries weight in at most two filters; the band is read off the weights
/// rather than assumed, and an all-zero filter gets an empty one. A product
/// over the band is the product over the row, bit for bit, when every power
/// is finite: a zero weight adds nothing to a finite value.
pub(super) fn mel_bands(filters: &[f32], n_mels: usize, fft_bins: usize) -> Vec<(usize, usize)> {
    (0..n_mels)
        .map(|mel| {
            let row = &filters[mel * fft_bins..(mel + 1) * fft_bins];
            match row.iter().position(|weight| *weight != 0.0) {
                Some(first) => {
                    let last = row
                        .iter()
                        .rposition(|weight| *weight != 0.0)
                        .map_or(first, |index| index + 1);
                    (first, last)
                }
                None => (0, 0),
            }
        })
        .collect()
}

/// Build a Slaney-normalized mel filter bank.
pub(super) fn mel_basis(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let fft_bins = n_fft / 2 + 1;
    let mut fft_freqs = Vec::with_capacity(fft_bins);
    for bin in 0..fft_bins {
        fft_freqs.push(bin as f64 * sample_rate as f64 / n_fft as f64);
    }

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate as f64 / 2.0);
    let mut mel_edges = Vec::with_capacity(n_mels + 2);
    for i in 0..n_mels + 2 {
        let t = i as f64 / (n_mels + 1) as f64;
        mel_edges.push(mel_to_hz(mel_min + (mel_max - mel_min) * t));
    }

    let mel_widths: Vec<f64> = mel_edges.windows(2).map(|w| w[1] - w[0]).collect();
    let mut weights = vec![0.0_f32; n_mels * fft_bins];
    for mel in 0..n_mels {
        let enorm = 2.0 / (mel_edges[mel + 2] - mel_edges[mel]);
        for (bin, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_edges[mel]) / mel_widths[mel];
            let upper = (mel_edges[mel + 2] - freq) / mel_widths[mel + 1];
            weights[mel * fft_bins + bin] = lower.min(upper).max(0.0).mul_add(enorm, 0.0) as f32;
        }
    }
    weights
}

/// The two bounds that keep a resampling kernel cheap, shared by the
/// single-precision and the exact resampler and applied before anything is
/// built. Before each phase trims its zero taps, a rate sharing little with the
/// frontend's swells both `kernel_len` and `new_freq * kernel_len`, and a corrupt
/// header can name any rate: 2 147 491 648 Hz (8 kHz with one bit flipped) to
/// 16 kHz reduces to 33 554 557 : 250, a table of 8.8 G coefficients that
/// took minutes to build for the two samples it yields. The rates audio is
/// stored at reduce to small ratios: to 16 kHz the longest kernel among the
/// common ones is 176.4 kHz's 575 coefficients and the largest table
/// 11.025 kHz's 291 200; to 24 kHz, 176.4 kHz's 237 and 11.025 kHz's 51 520.
/// The odd ones real headers carry still fit: 5 512 Hz (689 : 2 000 to
/// 16 kHz, 703 coefficients, a 1.4 M table), 47 952 Hz, the 48 kHz NTSC
/// pull-down (2 997 : 1 000 to 16 kHz, 3 035 coefficients, a 3.0 M table;
/// 999 : 500 to 24 kHz, 1 025), and the hi-res rates up to DSD512 as the
/// external decoder renders it (2 822 400 Hz: 882 : 5 to 16 kHz, 3 022
/// coefficients). A kernel of up to 4 096 coefficients and a
/// table of up to 2^22 (16 MiB of f32, 32 MiB of i64) admit all of those;
/// past either bound the rate is refused, deliberately: 44 056 Hz (5 507 :
/// 2 000 to 16 kHz, 5 541 coefficients, an 11 M table), 23 994 Hz (3 999 :
/// 4 000 to 24 kHz, a 16 M table) and 23 999 Hz (24 013 coefficients, over
/// half a billion) would each take seconds per clip. The kernel bound is also
/// what keeps the exact resampler's accumulator in range: `kernel_len`
/// products of 2^40 by 2^62 stay under 2^114 in i128.
const MAX_RESAMPLE_KERNEL_LEN: usize = 1 << 12;
const MAX_RESAMPLE_KERNEL_VALUES: usize = 1 << 22;

const _: () = assert!(MAX_RESAMPLE_KERNEL_LEN <= u16::MAX as usize);
const _: () = assert!(MAX_RESAMPLE_KERNEL_VALUES <= u32::MAX as usize);

const MAX_CACHED_KERNELS: usize = 16;
const MAX_CACHED_KERNEL_BYTES: usize = 4 << 20;

#[derive(Clone, Copy)]
struct PhaseSpan {
    tap_start: u32,
    window_offset: u16,
    tap_len: u16,
}

struct PolyphaseKernel<T> {
    key: (usize, usize),
    kernel_len: usize,
    taps: Vec<T>,
    phases: Vec<PhaseSpan>,
}

impl<T> PolyphaseKernel<T> {
    fn byte_size(&self) -> usize {
        self.taps.capacity() * size_of::<T>() + self.phases.capacity() * size_of::<PhaseSpan>()
    }
}

struct KernelCache<T>(Mutex<Vec<Arc<PolyphaseKernel<T>>>>);

impl<T> KernelCache<T> {
    const fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    fn entries(&self) -> MutexGuard<'_, Vec<Arc<PolyphaseKernel<T>>>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn lookup(
        entries: &mut Vec<Arc<PolyphaseKernel<T>>>,
        key: (usize, usize),
    ) -> Option<Arc<PolyphaseKernel<T>>> {
        let pos = entries.iter().rposition(|kernel| kernel.key == key)?;
        let hit = Arc::clone(&entries[pos]);
        if pos + 1 < entries.len() {
            let entry = entries.remove(pos);
            entries.push(entry);
        }
        Some(hit)
    }

    fn get_or_build(
        &self,
        key: (usize, usize),
        build: impl FnOnce() -> Result<PolyphaseKernel<T>, TransformError>,
    ) -> Result<Arc<PolyphaseKernel<T>>, TransformError> {
        if let Some(hit) = Self::lookup(&mut self.entries(), key) {
            return Ok(hit);
        }

        let built = Arc::new(build()?);
        let cost = built.byte_size();
        if cost > MAX_CACHED_KERNEL_BYTES {
            return Ok(built);
        }

        let mut evicted: [Option<Arc<PolyphaseKernel<T>>>; MAX_CACHED_KERNELS] =
            [const { None }; MAX_CACHED_KERNELS];
        let out = {
            let mut entries = self.entries();
            if let Some(existing) = Self::lookup(&mut entries, key) {
                existing
            } else {
                let mut held: usize = entries.iter().map(|kernel| kernel.byte_size()).sum();
                let mut slot = 0;
                while !entries.is_empty()
                    && (entries.len() >= MAX_CACHED_KERNELS
                        || held + cost > MAX_CACHED_KERNEL_BYTES)
                {
                    let old = entries.remove(0);
                    held -= old.byte_size();
                    evicted[slot] = Some(old);
                    slot += 1;
                }
                entries.push(Arc::clone(&built));
                built
            }
        };
        drop(evicted);
        Ok(out)
    }
}

static SINGLE_KERNELS: KernelCache<f32> = KernelCache::new();
static EXACT_KERNELS: KernelCache<i64> = KernelCache::new();

/// Support for half-width `width` falls in `shift + 1..=shift + 2 * width`.
/// The extra taps on each end cover `f32` rounding of the tap coordinate.
fn build_polyphase_kernel<T: Copy + PartialEq>(
    key: (usize, usize),
    width: usize,
    kernel_len: usize,
    zero: T,
    mut tap_at: impl FnMut(usize, usize) -> T,
) -> Result<PolyphaseKernel<T>, TransformError> {
    let (orig_freq, new_freq) = key;
    let tap_budget = new_freq.saturating_mul(2 * width).saturating_add(5);
    let mut taps = Vec::new();
    taps.try_reserve_exact(tap_budget).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate audio resample kernel: {error}"))
    })?;
    let mut phases = Vec::new();
    phases.try_reserve_exact(new_freq).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate audio resample phases: {error}"))
    })?;

    for phase in 0..new_freq {
        let start = taps.len();
        let mut end = start;
        let mut window_offset = 0_u16;
        let shift = ((phase as u64 * orig_freq as u64) / new_freq as u64) as usize;
        let k_lo = shift.saturating_sub(2);
        let k_hi = shift.saturating_add(2 * width + 3).min(kernel_len);
        for kernel_index in k_lo..k_hi {
            let weight = tap_at(phase, kernel_index);
            if taps.len() == start {
                if weight == zero {
                    continue;
                }
                window_offset = kernel_index as u16;
            }
            taps.push(weight);
            if weight != zero {
                end = taps.len();
            }
        }
        taps.truncate(end);
        phases.push(PhaseSpan {
            tap_start: start as u32,
            window_offset,
            tap_len: (end - start) as u16,
        });
    }

    Ok(PolyphaseKernel {
        key,
        kernel_len,
        taps,
        phases,
    })
}

/// The size of the band-limited sinc kernel from `src_sample_rate` to
/// `dst_sample_rate` over the reduced ratio `orig_freq : new_freq` with the
/// half width `width` in input samples: the coefficients per output and in
/// the whole table, refused past [`MAX_RESAMPLE_KERNEL_LEN`] or
/// [`MAX_RESAMPLE_KERNEL_VALUES`] before either is allocated.
fn resample_kernel_size(
    src_sample_rate: usize,
    dst_sample_rate: usize,
    orig_freq: usize,
    new_freq: usize,
    width: usize,
) -> Result<(usize, usize), TransformError> {
    let kernel_len = width
        .checked_mul(2)
        .and_then(|value| value.checked_add(orig_freq))
        .ok_or_else(|| TransformError::ShapeError("audio resample kernel is too large".into()))?;
    let kernel_values = new_freq
        .checked_mul(kernel_len)
        .ok_or_else(|| TransformError::ShapeError("audio resample kernel size overflow".into()))?;
    if kernel_len > MAX_RESAMPLE_KERNEL_LEN || kernel_values > MAX_RESAMPLE_KERNEL_VALUES {
        return Err(TransformError::ShapeError(format!(
            "audio sample rate {src_sample_rate} Hz has no resampling kernel of a practical \
             size for {dst_sample_rate} Hz ({kernel_len} coefficients per output, \
             {kernel_values} in the table)"
        )));
    }
    Ok((kernel_len, kernel_values))
}

pub(super) fn resampled_len(
    samples: usize,
    src_sample_rate: usize,
    dst_sample_rate: usize,
) -> Result<usize, TransformError> {
    if src_sample_rate == 0 || dst_sample_rate == 0 {
        return Err(TransformError::ShapeError(
            "audio resampling rates must be positive".to_string(),
        ));
    }
    if samples == 0 || src_sample_rate == dst_sample_rate {
        return Ok(samples);
    }
    let len = (samples as u128 * dst_sample_rate as u128).div_ceil(src_sample_rate as u128);
    usize::try_from(len)
        .map_err(|_| TransformError::ShapeError("resampled audio length exceeds usize".to_string()))
}

/// Match torchaudio's default `functional.resample`: band-limited sinc
/// interpolation with a Hann window, filter width 6, and rolloff 0.99.
pub(super) fn bandlimited_resample(
    samples: &[f32],
    src_sample_rate: usize,
    dst_sample_rate: usize,
) -> Result<Vec<f32>, TransformError> {
    const LOWPASS_FILTER_WIDTH: f32 = 6.0;
    const ROLLOFF: f32 = 0.99;

    if src_sample_rate == 0 || dst_sample_rate == 0 {
        return Err(TransformError::ShapeError(
            "audio resampling rates must be positive".to_string(),
        ));
    }
    if samples.is_empty() || src_sample_rate == dst_sample_rate {
        return Ok(samples.to_vec());
    }

    let gcd = greatest_common_divisor(src_sample_rate, dst_sample_rate);
    let orig_freq = src_sample_rate / gcd;
    let new_freq = dst_sample_rate / gcd;
    let base_freq = orig_freq.min(new_freq) as f32 * ROLLOFF;
    let width = (LOWPASS_FILTER_WIDTH * orig_freq as f32 / base_freq).ceil() as usize;
    let (kernel_len, _) =
        resample_kernel_size(src_sample_rate, dst_sample_rate, orig_freq, new_freq, width)?;
    let kernel = SINGLE_KERNELS.get_or_build((orig_freq, new_freq), || {
        let orig_freq_f32 = orig_freq as f32;
        let new_freq_f32 = new_freq as f32;
        let scale = base_freq / orig_freq_f32;
        build_polyphase_kernel(
            (orig_freq, new_freq),
            width,
            kernel_len,
            0.0_f32,
            |phase, kernel_index| {
                let idx = (kernel_index as f32 - width as f32) / orig_freq_f32;
                let t = (idx - phase as f32 / new_freq_f32) * base_freq;
                // Clamping `t` to `LOWPASS_FILTER_WIDTH` keeps dead end taps non-zero:
                // `(PI_F32 / 2.0).cos().powi(2)` is 1.9e-15 in `f32`.
                if t.abs() < LOWPASS_FILTER_WIDTH {
                    let window = (t * PI_F32 / LOWPASS_FILTER_WIDTH / 2.0).cos().powi(2);
                    let radians = t * PI_F32;
                    let sinc = if radians == 0.0 {
                        1.0
                    } else {
                        radians.sin() / radians
                    };
                    sinc * window * scale
                } else {
                    0.0
                }
            },
        )
    })?;

    let target_len = resampled_len(samples.len(), src_sample_rate, dst_sample_rate)?;
    let mut output = Vec::new();
    output.try_reserve_exact(target_len).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate resampled audio: {error}"))
    })?;

    #[cfg(target_arch = "x86_64")]
    if convolve_rows(&mut output, samples, &kernel, orig_freq, width, target_len)? {
        return Ok(output);
    }
    convolve_blocks(&mut output, samples, &kernel, orig_freq, width, target_len)?;
    Ok(output)
}

/// [`bandlimited_resample`]'s convolution one output sample after the other,
/// appended to `output`: the loop of a processor without AVX2 and FMA, and
/// the bits [`convolve_rows`] must give.
fn convolve_blocks(
    output: &mut Vec<f32>,
    samples: &[f32],
    kernel: &PolyphaseKernel<f32>,
    orig_freq: usize,
    width: usize,
    target_len: usize,
) -> Result<(), TransformError> {
    let kernel_len = kernel.kernel_len;
    // The convolution reads `width` samples before the signal and up to a
    // kernel beyond its end, where it previously contributed nothing: a tap
    // whose index fell outside the input was skipped. Materialising those
    // zeros once turns the inner loop into a flat dot product over two slices,
    // with no bounds test per tap. The result is unchanged bit for bit --
    // `0.0f32.mul_add(coefficient, acc)` is `acc` exactly for the finite
    // coefficients this kernel holds, and the accumulator starts at `+0.0`, so
    // no signed-zero difference arises either.
    // The last block starts at most `orig_freq - 1` samples before the input's end, so the
    // last window ends before `len + kernel_len` of the padded input.
    let blocks = samples.len().div_ceil(orig_freq);
    let padded_len = samples.len().checked_add(kernel_len).ok_or_else(|| {
        TransformError::ShapeError("padded resample input size overflow".to_string())
    })?;
    let mut padded: Vec<f32> = Vec::new();
    padded.try_reserve_exact(padded_len).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate padded resample input: {error}"))
    })?;
    padded.resize(width, 0.0);
    padded.extend_from_slice(samples);
    padded.resize(padded_len, 0.0);

    'blocks: for block in 0..blocks {
        let input_start = block * orig_freq;
        let window = &padded[input_start..input_start + kernel_len];
        for span in &kernel.phases {
            if output.len() == target_len {
                break 'blocks;
            }
            let (ts, wo, len) = (
                span.tap_start as usize,
                span.window_offset as usize,
                span.tap_len as usize,
            );
            let mut value = 0.0_f32;
            for (&coefficient, &sample) in
                kernel.taps[ts..ts + len].iter().zip(&window[wo..wo + len])
            {
                value = sample.mul_add(coefficient, value);
            }
            output.push(value);
        }
    }

    Ok(())
}

/// Blocks of output whose chains [`convolve_rows`] steps together.
#[cfg(target_arch = "x86_64")]
const RESAMPLE_ROW_BLOCKS: usize = 128;

/// The running values of a row's chains on a 64-byte boundary, which keeps
/// each group of eight consecutive values that starts at index 0, 8 and so
/// on inside one cache line. The taps store and reload them up to eight at a
/// time. On a plain array the processor counted 3.12 million blocked
/// store-to-load forwards a call in the gateway instead of 0.53 million, on a
/// 30 s clip at 44.1 kHz, and a 30 s clip at 22.05 kHz cost the gateway
/// 59.1 ms of CPU instead of 41.5 ms.
#[cfg(target_arch = "x86_64")]
#[repr(align(64))]
struct RowValues([f32; RESAMPLE_ROW_BLOCKS]);

/// [`bandlimited_resample`]'s convolution on a processor with AVX2 and FMA:
/// fills `output` and returns `true`. On any other processor it returns
/// `false` and [`convolve_blocks`] runs.
///
/// Every output sample stays the chain `value = sample.mul_add(coefficient,
/// value)` over its taps in order from `+0.0`, as [`convolve_blocks`] runs it;
/// only which chain steps next differs, so no bit does. The chains of one
/// phase over a row of blocks take each tap together, eight to an
/// instruction. Their samples lie `orig_freq` apart, so the row's input is
/// laid out by input phase first: `strands[strand * strand_len + position]`
/// is sample `(first_block + position) * orig_freq + strand` of the input
/// with `width` zeros in front and zeros behind.
#[cfg(target_arch = "x86_64")]
#[multiversion::multiversion(targets("x86_64+avx2+fma"))]
fn convolve_rows(
    output: &mut Vec<f32>,
    samples: &[f32],
    kernel: &PolyphaseKernel<f32>,
    orig_freq: usize,
    width: usize,
    target_len: usize,
) -> Result<bool, TransformError> {
    let fused = multiversion::target::match_target! { "x86_64+avx2+fma" => true, _ => false };
    if !fused {
        return Ok(false);
    }
    let (_, new_freq) = kernel.key;
    let kernel_len = kernel.kernel_len;
    let blocks = samples.len().div_ceil(orig_freq);
    let strand_len = blocks.min(RESAMPLE_ROW_BLOCKS) + (kernel_len - 1) / orig_freq;
    let mut strands: Vec<f32> = Vec::new();
    strands
        .try_reserve_exact(orig_freq * strand_len)
        .map_err(|error| {
            TransformError::ShapeError(format!("failed to allocate resample input rows: {error}"))
        })?;
    strands.resize(orig_freq * strand_len, 0.0);
    output.resize(target_len, 0.0);
    let mut values = RowValues([0.0; RESAMPLE_ROW_BLOCKS]);
    for first_block in (0..blocks).step_by(RESAMPLE_ROW_BLOCKS) {
        let values = &mut values.0[..(blocks - first_block).min(RESAMPLE_ROW_BLOCKS)];
        strands.fill(0.0);
        for (strand, positions) in strands.chunks_exact_mut(strand_len).enumerate() {
            let start = first_block * orig_freq + strand;
            let lead = width.saturating_sub(start).div_ceil(orig_freq);
            let source = samples
                .iter()
                .skip(start + lead * orig_freq - width)
                .step_by(orig_freq);
            for (position, &sample) in positions.iter_mut().skip(lead).zip(source) {
                *position = sample;
            }
        }
        for (phase, span) in kernel.phases.iter().enumerate() {
            values.fill(0.0);
            let (ts, len) = (span.tap_start as usize, span.tap_len as usize);
            let mut shift = span.window_offset as usize / orig_freq;
            let mut strand = span.window_offset as usize % orig_freq;
            for &coefficient in &kernel.taps[ts..ts + len] {
                let row = &strands[strand * strand_len..(strand + 1) * strand_len];
                for (value, &sample) in values.iter_mut().zip(&row[shift..]) {
                    *value = sample.mul_add(coefficient, *value);
                }
                strand += 1;
                if strand == orig_freq {
                    strand = 0;
                    shift += 1;
                }
            }
            let slots = output
                .iter_mut()
                .skip(first_block * new_freq + phase)
                .step_by(new_freq);
            for (slot, &value) in slots.zip(values.iter()) {
                *slot = value;
            }
        }
    }
    Ok(true)
}

/// Fractional bits of the sample grid: samples are placed on a grid of
/// `2^-SAMPLE_GRID_BITS` of the clip's peak magnitude bound, which keeps
/// every `f32` sample within `2^-16` of the peak exact (all 16- and 24-bit
/// PCM, and float audio down to -96 dB of its own peak).
const EXACT_RESAMPLE_SAMPLE_BITS: i32 = 40;
/// Fractional bits of the filter coefficient grid (`|coefficient| <= 1`).
const EXACT_RESAMPLE_COEFFICIENT_BITS: i32 = 62;

/// [`bandlimited_resample`]'s filter (Hann-windowed sinc, width 6, rolloff
/// 0.99, torchaudio's definition) evaluated exactly: the samples and the
/// coefficients are placed on fixed-point grids and convolved in integer
/// arithmetic, and each output is rounded to `f32` once.
///
/// For a model whose encoder consumes the resampled samples themselves the
/// rounding of this step is the feature. A floating-point evaluation depends
/// on the summation order (which differs between implementations and CPU
/// kernels) and leaves exact cancellations as rounding noise; the integer
/// convolution computes one rational number for each output, so every
/// implementation of the same grids rounds to the same `f32`.
pub(super) fn bandlimited_resample_exact(
    samples: &[f32],
    src_sample_rate: usize,
    dst_sample_rate: usize,
) -> Result<Vec<f32>, TransformError> {
    const LOWPASS_FILTER_WIDTH: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;

    if src_sample_rate == 0 || dst_sample_rate == 0 {
        return Err(TransformError::ShapeError(
            "audio resampling rates must be positive".to_string(),
        ));
    }
    if samples.is_empty() || src_sample_rate == dst_sample_rate {
        return Ok(samples.to_vec());
    }

    let gcd = greatest_common_divisor(src_sample_rate, dst_sample_rate);
    let orig_freq = src_sample_rate / gcd;
    let new_freq = dst_sample_rate / gcd;
    let base_freq = orig_freq.min(new_freq) as f64 * ROLLOFF;
    let width = (LOWPASS_FILTER_WIDTH * orig_freq as f64 / base_freq).ceil() as usize;
    // The shared bounds refuse an impractical rate before anything is built;
    // the kernel bound is also what keeps the accumulator below in range.
    let (kernel_len, _) =
        resample_kernel_size(src_sample_rate, dst_sample_rate, orig_freq, new_freq, width)?;

    let kernel = EXACT_KERNELS.get_or_build((orig_freq, new_freq), || {
        let orig_freq_f64 = orig_freq as f64;
        let new_freq_f64 = new_freq as f64;
        let scale = base_freq / orig_freq_f64;
        let coefficient_scale = 2f64.powi(EXACT_RESAMPLE_COEFFICIENT_BITS);
        build_polyphase_kernel(
            (orig_freq, new_freq),
            width,
            kernel_len,
            0_i64,
            |phase, kernel_index| {
                let idx = (kernel_index as f64 - width as f64) / orig_freq_f64;
                let mut t = (idx - phase as f64 / new_freq_f64) * base_freq;
                // `(PI / 2.0).cos().powi(2)` is 3.75e-33 (~2^-108), rounding to `0_i64` at 2^-62.
                t = t.clamp(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH);
                let window = (t * PI / LOWPASS_FILTER_WIDTH / 2.0).cos().powi(2);
                let radians = t * PI;
                let sinc = if radians == 0.0 {
                    1.0
                } else {
                    radians.sin() / radians
                };
                ((sinc * window * scale) * coefficient_scale).round_ties_even() as i64
            },
        )
    })?;

    // Samples on the 2^-40 grid of the peak bound 2^peak_exponent.
    let peak = samples
        .iter()
        .fold(0.0f32, |peak, &sample| peak.max(sample.abs()));
    if !peak.is_finite() {
        return Err(TransformError::ShapeError(
            "audio samples must be finite".to_string(),
        ));
    }
    let peak_exponent = frexp_exponent(peak);
    let sample_scale = 2f64.powi(EXACT_RESAMPLE_SAMPLE_BITS - peak_exponent);
    let mut quantized: Vec<i64> = Vec::new();
    quantized
        .try_reserve_exact(width + samples.len() + kernel_len)
        .map_err(|error| {
            TransformError::ShapeError(format!("failed to allocate resample input: {error}"))
        })?;
    quantized.resize(width, 0);
    quantized.extend(
        samples
            .iter()
            .map(|&sample| (f64::from(sample) * sample_scale).round_ties_even() as i64),
    );
    quantized.resize(width + samples.len() + kernel_len, 0);

    let target_len_u128 = (samples.len() as u128 * new_freq as u128).div_ceil(orig_freq as u128);
    let target_len = usize::try_from(target_len_u128).map_err(|_| {
        TransformError::ShapeError("resampled audio length exceeds usize".to_string())
    })?;
    let mut output = Vec::new();
    output.try_reserve_exact(target_len).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate resampled audio: {error}"))
    })?;
    let output_scale =
        2f64.powi(peak_exponent - EXACT_RESAMPLE_SAMPLE_BITS - EXACT_RESAMPLE_COEFFICIENT_BITS);
    'blocks: for block in 0..samples.len().div_ceil(orig_freq) {
        let input_start = block * orig_freq;
        let window = &quantized[input_start..input_start + kernel_len];
        for span in &kernel.phases {
            if output.len() == target_len {
                break 'blocks;
            }
            let (ts, wo, len) = (
                span.tap_start as usize,
                span.window_offset as usize,
                span.tap_len as usize,
            );
            let mut acc: i128 = 0;
            for (&coefficient, &sample) in
                kernel.taps[ts..ts + len].iter().zip(&window[wo..wo + len])
            {
                acc += i128::from(coefficient) * i128::from(sample);
            }
            // The exact sum to f64 (one rounding), the power-of-two scale
            // (exact), then f32 (one rounding).
            output.push(((acc as f64) * output_scale) as f32);
        }
    }

    Ok(output)
}

/// `e` with `2^(e-1) <= |value| < 2^e` (C's `frexp` exponent); 0 for zero.
fn frexp_exponent(value: f32) -> i32 {
    if value == 0.0 {
        return 0;
    }
    let bits = f64::from(value).to_bits();
    ((bits >> 52) & 0x7ff) as i32 - 1022
}

fn greatest_common_divisor(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        (lhs, rhs) = (rhs, lhs % rhs);
    }
    lhs
}

#[cfg(test)]
mod tests {
    use super::{bandlimited_resample, bandlimited_resample_exact, frexp_exponent};
    use crate::error::TransformError;

    #[test]
    fn bandlimited_resample_matches_torchaudio_golden_vector() {
        let input = [0.0, 0.25, -0.5, 0.75, -1.0, 0.5, 0.125, -0.25];
        // torchaudio 2.11 functional.resample(input, 8000, 16000), using
        // the default Hann-windowed sinc kernel.
        let expected = [
            0.012_859_77,
            0.347_330_45,
            0.230_903_15,
            -0.388_803_3,
            -0.476_089_15,
            0.325_961_17,
            0.724_112_45,
            -0.117_642_37,
            -0.975_495_4,
            -0.540_876_27,
            0.479_874_4,
            0.698_470_23,
            0.138_904_54,
            -0.281_137_47,
            -0.257_479_25,
            -0.095_786_646,
        ];

        let actual = bandlimited_resample(&input, 8_000, 16_000).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-5,
                "resampled sample {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn bandlimited_resample_uses_torchaudio_output_length_formula() {
        for (input_len, source_rate, destination_rate) in [
            (17, 16_000, 8_000),
            (8, 8_000, 11_025),
            (161, 48_000, 44_100),
        ] {
            let input = vec![0.0; input_len];
            let output = bandlimited_resample(&input, source_rate, destination_rate).unwrap();
            let expected_len =
                (input_len as u128 * destination_rate as u128).div_ceil(source_rate as u128);

            assert_eq!(
                output.len(),
                expected_len as usize,
                "unexpected output length for {source_rate} Hz to {destination_rate} Hz"
            );
        }
    }

    #[test]
    fn bandlimited_resample_handles_empty_and_passthrough_inputs() {
        assert_eq!(
            bandlimited_resample(&[], 8_000, 16_000).unwrap(),
            Vec::<f32>::new()
        );

        let input = [0.25, -0.5, 1.0];
        assert_eq!(bandlimited_resample(&input, 16_000, 16_000).unwrap(), input);
    }

    #[test]
    fn bandlimited_resample_rejects_zero_rates() {
        for (source_rate, destination_rate) in [(0, 16_000), (16_000, 0)] {
            assert!(matches!(
                bandlimited_resample(&[0.0], source_rate, destination_rate),
                Err(TransformError::ShapeError(message))
                    if message == "audio resampling rates must be positive"
            ));
        }
    }

    /// A header's rate sizes the single-precision kernel too, and a corrupt
    /// one can size it as it likes: 2 147 491 648 Hz (8 kHz with one bit
    /// flipped) to 16 kHz reduces to 33 554 557 : 250, a table of 250 phases
    /// by 35 M coefficients that took minutes to build before the two samples
    /// it yields were refused. The bound refuses the rate before anything is
    /// built; the odd rates refused for 24 kHz are refused here too (44 056 Hz:
    /// 5 507 : 2 000, 5 541 coefficients; 15 999 Hz: 16 013 coefficients, a
    /// 256 M table).
    #[test]
    fn a_rate_with_an_impractical_single_precision_kernel_is_refused_before_it_is_built() {
        let started = std::time::Instant::now();
        for rate in [2_147_491_648_usize, 44_056, 15_999] {
            let error = bandlimited_resample(&[0.25; 48], rate, 16_000).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("no resampling kernel of a practical size"),
                "{rate} Hz: {error}"
            );
        }
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "the refusals took {elapsed:?}"
        );
        // The rates audio is stored at resample to the 16 kHz frontends as
        // before; the odd ones real headers carry fit too, 47 952 Hz (the
        // 48 kHz NTSC pull-down, 2 997 : 1 000) with the longest kernel of
        // them, 3 035 coefficients, and the hi-res rates up to DSD512 as the
        // external decoder renders it (705 600, 768 000, 1 411 200 and
        // 2 822 400 Hz: 977, 630, 1 511 and 3 022 coefficients).
        for rate in [
            4_000, 5_512, 8_000, 11_025, 12_000, 22_050, 24_000, 32_000, 37_800, 44_100, 47_952,
            48_000, 64_000, 88_200, 96_000, 176_400, 192_000, 384_000, 705_600, 768_000, 1_411_200,
            2_822_400,
        ] {
            bandlimited_resample(&[0.25; 48], rate, 16_000)
                .unwrap_or_else(|error| panic!("{rate} Hz: {error}"));
        }
    }

    #[test]
    fn a_rate_with_an_impractical_kernel_is_refused_before_it_is_built() {
        // 23 999 Hz shares no factor with 24 000 (24 000 phases of 24 013
        // coefficients, over half a billion in the table); 23 994 Hz reduces
        // to 3 999 : 4 000 (4 013 coefficients, a 16 M table); 44 056 Hz, the
        // 44.1 kHz NTSC pull-down, to 5 507 : 3 000 (5 531 coefficients, a
        // 16.6 M table). All three are refused before anything is built.
        for rate in [23_999, 23_994, 44_056] {
            let error = bandlimited_resample_exact(&[0.25; 48], rate, 24_000).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("no resampling kernel of a practical size"),
                "{rate} Hz: {error}"
            );
        }
        // The rates audio is stored at stay inside both bounds: the longest
        // kernel among the common ones is 176.4 kHz's and the largest table
        // 11.025 kHz's; the odd ones real headers carry, 5 512 Hz (legacy
        // 5.5 kHz) and 47 952 Hz (the 48 kHz NTSC pull-down, 1 025
        // coefficients), fit too, as do the hi-res rates up to DSD512 as the
        // external decoder renders it (505, 420, 1 008 and 2 014 coefficients).
        for rate in [
            4_000, 5_512, 8_000, 11_025, 12_000, 16_000, 22_050, 32_000, 37_800, 44_100, 47_952,
            48_000, 64_000, 88_200, 96_000, 176_400, 192_000, 384_000, 705_600, 768_000, 1_411_200,
            2_822_400,
        ] {
            bandlimited_resample_exact(&[0.25; 48], rate, 24_000)
                .unwrap_or_else(|error| panic!("{rate} Hz: {error}"));
        }
    }

    #[test]
    fn exact_resample_matches_the_single_precision_kernel_closely() {
        let samples: Vec<f32> = (0..4_800)
            .map(|i| ((i * 73 + 977) % 65_536 - 32_768) as f32 / 32_768.0)
            .collect();
        for (src, dst) in [
            (16_000, 24_000),
            (44_100, 24_000),
            (8_000, 24_000),
            (48_000, 24_000),
        ] {
            let single = bandlimited_resample(&samples, src, dst).unwrap();
            let exact = bandlimited_resample_exact(&samples, src, dst).unwrap();
            assert_eq!(single.len(), exact.len(), "{src}->{dst}");
            let max_diff = single
                .iter()
                .zip(&exact)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // The single-precision kernel's own rounding reaches 1e-5 on the
            // long 44.1 kHz kernel; the two agree to that.
            assert!(max_diff < 2e-5, "{src}->{dst}: {max_diff}");
        }
        assert_eq!(
            bandlimited_resample_exact(&samples, 24_000, 24_000).unwrap(),
            samples
        );
        assert!(bandlimited_resample_exact(&samples, 0, 24_000).is_err());
        assert!(bandlimited_resample_exact(&[], 16_000, 24_000)
            .unwrap()
            .is_empty());
        // Scaling the input scales the output exactly: the grid follows the peak.
        let loud: Vec<f32> = samples.iter().map(|s| s * 1024.0).collect();
        let quiet: Vec<f32> = samples.iter().map(|s| s / 1024.0).collect();
        let base = bandlimited_resample_exact(&samples, 16_000, 24_000).unwrap();
        assert_eq!(
            bandlimited_resample_exact(&loud, 16_000, 24_000).unwrap(),
            base.iter().map(|v| v * 1024.0).collect::<Vec<_>>()
        );
        assert_eq!(
            bandlimited_resample_exact(&quiet, 16_000, 24_000).unwrap(),
            base.iter().map(|v| v / 1024.0).collect::<Vec<_>>()
        );
        assert_eq!(frexp_exponent(1.0), 1);
        assert_eq!(frexp_exponent(0.5), 0);
        assert_eq!(frexp_exponent(0.75), 0);
        assert_eq!(frexp_exponent(3.0), 2);
        assert_eq!(frexp_exponent(0.0), 0);
    }

    /// The rows of [`convolve_rows`] against the plain loop over one table:
    /// the same bits, for the ratios and half widths of 44.1, 48 and 8 kHz to
    /// 16 kHz and of 16 to 24 kHz, at lengths around one block and one row,
    /// each with a cut last block. A processor without AVX2 and FMA runs the
    /// plain loop alone and has nothing to compare.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_rows_convolve_like_the_plain_loop() {
        use super::{build_polyphase_kernel, convolve_blocks, convolve_rows, RESAMPLE_ROW_BLOCKS};

        for (orig_freq, new_freq, width) in [(441, 160, 17), (3, 1, 19), (1, 2, 7), (2, 3, 7)] {
            let kernel_len = 2 * width + orig_freq;
            let kernel = build_polyphase_kernel(
                (orig_freq, new_freq),
                width,
                kernel_len,
                0.0_f32,
                |p, k| {
                    let shift = (p * orig_freq) / new_freq;
                    if k <= shift || k > shift + 2 * width {
                        0.0
                    } else {
                        (((p * kernel_len + k) * 37 + 11) % 251) as f32 / 251.0 - 0.5
                    }
                },
            )
            .unwrap();
            let row = RESAMPLE_ROW_BLOCKS * orig_freq;
            for len in [1, orig_freq + 1, row - 1, row + 1] {
                let samples: Vec<f32> = (0..len)
                    .map(|i| ((i * 73 + 977) % 65_536) as f32 / 32_768.0 - 1.0)
                    .collect();
                let target_len = (len * new_freq).div_ceil(orig_freq);
                let (mut rows, mut plain) = (Vec::new(), Vec::new());
                let by_rows =
                    convolve_rows(&mut rows, &samples, &kernel, orig_freq, width, target_len)
                        .unwrap();
                if !by_rows {
                    return;
                }
                convolve_blocks(&mut plain, &samples, &kernel, orig_freq, width, target_len)
                    .unwrap();
                assert!(
                    rows.iter()
                        .map(|value| value.to_bits())
                        .eq(plain.iter().map(|value| value.to_bits())),
                    "{orig_freq}:{new_freq}, {len} samples"
                );
            }
        }
    }

    #[test]
    fn compacted_resample_matches_full_block_rows_bit_for_bit() {
        use super::{
            greatest_common_divisor, EXACT_RESAMPLE_COEFFICIENT_BITS, EXACT_RESAMPLE_SAMPLE_BITS,
            PI, PI_F32,
        };

        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let pairs = [
            (1, 16_000),
            (11_025, 16_000),
            (44_100, 16_000),
            (47_952, 24_000),
            (768_000, 16_000),
        ];
        for (src, dst) in pairs {
            let gcd = greatest_common_divisor(src, dst);
            let (orig, new) = (src / gcd, dst / gcd);
            let sig: Vec<f32> = (0..orig + 2)
                .map(|i| (i as f32 * 0.37).sin() * 0.8 + (i as f32 * 1.13).cos() * 0.15)
                .collect();
            let out_len = ((sig.len() as u128 * new as u128).div_ceil(orig as u128)) as usize;

            let base32 = orig.min(new) as f32 * 0.99;
            let w32 = (6.0 * orig as f32 / base32).ceil() as usize;
            let len32 = 2 * w32 + orig;
            let mut pad32 = vec![0.0_f32; sig.len() + len32];
            pad32[w32..w32 + sig.len()].copy_from_slice(&sig);
            let ref32: Vec<f32> = (0..out_len)
                .map(|i| {
                    let (win, p) = (&pad32[(i / new) * orig..][..len32], (i % new) as f32);
                    win.iter().enumerate().fold(0.0_f32, |val, (k, &s)| {
                        let t = ((k as f32 - w32 as f32) / orig as f32 - p / new as f32) * base32;
                        let r = t * PI_F32;
                        let c = if t.abs() < 6.0 {
                            (if r == 0.0 { 1.0 } else { r.sin() / r })
                                * (t * PI_F32 / 6.0 / 2.0).cos().powi(2)
                                * (base32 / orig as f32)
                        } else {
                            0.0
                        };
                        s.mul_add(c, val)
                    })
                })
                .collect();
            assert_eq!(
                bits(&bandlimited_resample(&sig, src, dst).unwrap()),
                bits(&ref32),
                "{src}->{dst}"
            );

            let base64 = orig.min(new) as f64 * 0.99;
            let w64 = (6.0 * orig as f64 / base64).ceil() as usize;
            let len64 = 2 * w64 + orig;
            let exp = frexp_exponent(sig.iter().fold(0.0f32, |p, &s| p.max(s.abs())));
            let s_scale = 2f64.powi(EXACT_RESAMPLE_SAMPLE_BITS - exp);
            let out_scale =
                2f64.powi(exp - EXACT_RESAMPLE_SAMPLE_BITS - EXACT_RESAMPLE_COEFFICIENT_BITS);
            let c_scale = 2f64.powi(EXACT_RESAMPLE_COEFFICIENT_BITS);
            let mut pad64 = vec![0_i64; sig.len() + len64];
            for (slot, &s) in pad64[w64..w64 + sig.len()].iter_mut().zip(&sig) {
                *slot = (f64::from(s) * s_scale).round_ties_even() as i64;
            }
            let ref64: Vec<f32> = (0..out_len)
                .map(|i| {
                    let (win, p) = (&pad64[(i / new) * orig..][..len64], (i % new) as f64);
                    let acc: i128 = win
                        .iter()
                        .enumerate()
                        .map(|(k, &s)| {
                            let t = (((k as f64 - w64 as f64) / orig as f64 - p / new as f64)
                                * base64)
                                .clamp(-6.0, 6.0);
                            let r = t * PI;
                            let sinc = if r == 0.0 { 1.0 } else { r.sin() / r };
                            let w = (t * PI / 6.0 / 2.0).cos().powi(2);
                            let c = (sinc * w * (base64 / orig as f64) * c_scale).round_ties_even();
                            i128::from(c as i64) * i128::from(s)
                        })
                        .sum();
                    ((acc as f64) * out_scale) as f32
                })
                .collect();
            assert_eq!(
                bits(&bandlimited_resample_exact(&sig, src, dst).unwrap()),
                bits(&ref64),
                "{src}->{dst}"
            );
        }
    }

    #[test]
    fn kernel_cache_promotes_recent_hits_and_evicts_past_either_bound() {
        use std::sync::Arc;

        use super::{
            build_polyphase_kernel, KernelCache, MAX_CACHED_KERNELS, MAX_CACHED_KERNEL_BYTES,
        };

        let cache: KernelCache<i64> = KernelCache::new();
        let build = |o, n, w| {
            build_polyphase_kernel((o, n), w, 2 * w + o, 0_i64, |_, k| i64::from(k < 2 * w))
        };
        let get = |o, n, w| cache.get_or_build((o, n), || build(o, n, w)).unwrap();
        let hit = |o, n| cache.get_or_build((o, n), || panic!("miss")).unwrap();
        let hot = get(3, 1, 7);
        let cold = get(5, 2, 7);
        assert!(Arc::ptr_eq(&hot, &hit(3, 1)));

        for n in 1..MAX_CACHED_KERNELS {
            get(n * 2 + 7, 2, 7);
        }
        assert_eq!(cache.entries().len(), MAX_CACHED_KERNELS);
        assert!(Arc::ptr_eq(&hot, &hit(3, 1)));
        assert!(!Arc::ptr_eq(&cold, &get(5, 2, 7)));

        for n in 1..=4 {
            get(n, 4_000, 24);
        }
        let held: usize = cache.entries().iter().map(|k| k.byte_size()).sum();
        assert!(cache.entries().len() < 4 && held <= MAX_CACHED_KERNEL_BYTES);
        assert!(get(1, 8_000, 40).byte_size() > MAX_CACHED_KERNEL_BYTES);
    }
}
