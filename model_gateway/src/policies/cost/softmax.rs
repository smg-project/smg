//! Shared selection arithmetic: the softmax draw over scores.

use rand::RngExt;

/// Softmax selection over min-max normalised *scores* (higher is better): the draw cache-aware
/// routing has always used. The normalisation makes temperature scale-free, the best candidate's
/// exponent is exactly 0 (overflow-safe), a degenerate spread is a uniform draw, and the
/// inverse-CDF walk falls back to the last row against floating-point drift. Returns a position.
pub fn sample_by_score_temperature(scores: &[f64], temperature: f32) -> Option<usize> {
    let first = *scores.first()?;
    let (min, max) = scores
        .iter()
        .fold((first, first), |(min, max), &s| (min.min(s), max.max(s)));
    let range = max - min;
    if range <= 0.0 {
        return Some(rand::rng().random_range(0..scores.len()));
    }
    let weights: Vec<f64> = scores
        .iter()
        .map(|&s| (((s - min) / range - 1.0) / f64::from(temperature)).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let draw = rand::rng().random::<f64>() * total;
    let mut cumulative = 0.0;
    for (position, weight) in weights.iter().enumerate() {
        cumulative += weight;
        if cumulative >= draw {
            return Some(position);
        }
    }
    Some(scores.len() - 1)
}
