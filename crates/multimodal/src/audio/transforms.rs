//! Shared digital-signal-processing transforms for audio frontends.

use std::{f32::consts::PI as PI_F32, f64::consts::PI};

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
    let kernel_len = width
        .checked_mul(2)
        .and_then(|value| value.checked_add(orig_freq))
        .ok_or_else(|| TransformError::ShapeError("audio resample kernel is too large".into()))?;
    let kernel_values = new_freq
        .checked_mul(kernel_len)
        .ok_or_else(|| TransformError::ShapeError("audio resample kernel size overflow".into()))?;
    let mut kernels = Vec::new();
    kernels.try_reserve_exact(kernel_values).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate audio resample kernel: {error}"))
    })?;

    let orig_freq_f32 = orig_freq as f32;
    let new_freq_f32 = new_freq as f32;
    let scale = base_freq / orig_freq_f32;
    for phase in 0..new_freq {
        for kernel_index in 0..kernel_len {
            let idx = (kernel_index as f32 - width as f32) / orig_freq_f32;
            let mut t = (idx - phase as f32 / new_freq_f32) * base_freq;
            t = t.clamp(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH);
            let window = (t * PI_F32 / LOWPASS_FILTER_WIDTH / 2.0).cos().powi(2);
            let radians = t * PI_F32;
            let sinc = if radians == 0.0 {
                1.0
            } else {
                radians.sin() / radians
            };
            kernels.push(sinc * window * scale);
        }
    }

    let target_len_u128 = (samples.len() as u128 * new_freq as u128).div_ceil(orig_freq as u128);
    let target_len = usize::try_from(target_len_u128).map_err(|_| {
        TransformError::ShapeError("resampled audio length exceeds usize".to_string())
    })?;
    let mut output = Vec::new();
    output.try_reserve_exact(target_len).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate resampled audio: {error}"))
    })?;

    for block in 0..samples.len().div_ceil(orig_freq) {
        let input_start = block * orig_freq;
        for phase in 0..new_freq {
            if output.len() == target_len {
                return Ok(output);
            }
            let kernel = &kernels[phase * kernel_len..(phase + 1) * kernel_len];
            let mut value = 0.0_f32;
            for (kernel_index, &coefficient) in kernel.iter().enumerate() {
                let padded_index = input_start + kernel_index;
                if padded_index >= width {
                    let sample_index = padded_index - width;
                    if let Some(&sample) = samples.get(sample_index) {
                        value = sample.mul_add(coefficient, value);
                    }
                }
            }
            output.push(value);
        }
    }

    Ok(output)
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
    let kernel_len = width
        .checked_mul(2)
        .and_then(|value| value.checked_add(orig_freq))
        .ok_or_else(|| TransformError::ShapeError("audio resample kernel is too large".into()))?;
    let kernel_values = new_freq
        .checked_mul(kernel_len)
        .ok_or_else(|| TransformError::ShapeError("audio resample kernel size overflow".into()))?;
    // Two bounds keep the exact evaluation cheap for every rate audio is
    // stored at and refuse the rest before anything is built. Each output
    // costs a `kernel_len` dot product, and the table holds `new_freq` phases
    // of `kernel_len` coefficients, so a rate that shares little with 24 kHz
    // is expensive on both counts: 23 999 Hz needs 24 000 phases of 24 013
    // coefficients (a multi-gigabyte table), and 23 994 Hz still 4 000 phases
    // of 4 013 (a 122 MiB table, 16 M sin/cos to fill it, 4 k multiply-adds
    // per output: seconds per clip). The rates in use peak at 237 coefficients
    // per output (176.4 kHz) and 51 520 in the table (11.025 kHz), so a kernel
    // of up to 1 024 coefficients and a table of up to 2^20 (8 MiB of i64)
    // leave 4x and 20x headroom; past either, the rate is refused.
    const MAX_KERNEL_LEN: usize = 1 << 10;
    const MAX_KERNEL_VALUES: usize = 1 << 20;
    if kernel_len > MAX_KERNEL_LEN || kernel_values > MAX_KERNEL_VALUES {
        return Err(TransformError::ShapeError(format!(
            "audio sample rate {src_sample_rate} Hz has no exact resampling kernel of a \
             practical size for {dst_sample_rate} Hz ({kernel_len} coefficients per output, \
             {kernel_values} in the table)"
        )));
    }

    // Coefficients on the 2^-62 grid, from the filter evaluated in f64 as
    // torchaudio writes it: t = (idx - phase / new) * base, clamped to the
    // width, a squared cosine window, sin(pi t) / (pi t), scaled.
    let mut kernels: Vec<i64> = Vec::new();
    kernels.try_reserve_exact(kernel_values).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate audio resample kernel: {error}"))
    })?;
    let orig_freq_f64 = orig_freq as f64;
    let new_freq_f64 = new_freq as f64;
    let scale = base_freq / orig_freq_f64;
    let coefficient_scale = 2f64.powi(EXACT_RESAMPLE_COEFFICIENT_BITS);
    for phase in 0..new_freq {
        for kernel_index in 0..kernel_len {
            let idx = (kernel_index as f64 - width as f64) / orig_freq_f64;
            let mut t = (idx - phase as f64 / new_freq_f64) * base_freq;
            t = t.clamp(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH);
            let window = (t * PI / LOWPASS_FILTER_WIDTH / 2.0).cos().powi(2);
            let radians = t * PI;
            let sinc = if radians == 0.0 {
                1.0
            } else {
                radians.sin() / radians
            };
            let coefficient = sinc * window * scale;
            kernels.push((coefficient * coefficient_scale).round_ties_even() as i64);
        }
    }

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
        for phase in 0..new_freq {
            if output.len() == target_len {
                break 'blocks;
            }
            let kernel = &kernels[phase * kernel_len..(phase + 1) * kernel_len];
            let mut acc: i128 = 0;
            for (&coefficient, &sample) in kernel.iter().zip(window) {
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

    #[test]
    fn a_rate_with_an_impractical_kernel_is_refused_before_it_is_built() {
        // 23 999 Hz shares no factor with 24 000 (24 000 phases of 24 013
        // coefficients, over half a billion in the table); 23 994 Hz reduces
        // to 3 999 : 4 000 (4 000 phases of 4 013, a 16 M table and 4 k
        // multiply-adds per output). Both are refused before anything is built.
        for rate in [23_999, 23_994] {
            let error = bandlimited_resample_exact(&[0.25; 48], rate, 24_000).unwrap_err();
            assert!(
                error.to_string().contains("no exact resampling kernel"),
                "{rate} Hz: {error}"
            );
        }
        // The rates audio is stored at stay well inside both bounds: the
        // longest kernel per output is 176.4 kHz's and the largest table is
        // 11.025 kHz's.
        for rate in [
            4_000, 8_000, 11_025, 12_000, 16_000, 22_050, 32_000, 37_800, 44_100, 48_000, 64_000,
            88_200, 96_000, 176_400, 192_000,
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
}
