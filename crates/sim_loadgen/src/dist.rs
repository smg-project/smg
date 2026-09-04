//! Deterministic randomness for the whole generator: a splitmix64 generator
//! plus the helpers derived from it (exponential arrivals, piecewise-linear
//! inverse CDFs, token-id and base64 streams). One algorithm everywhere makes
//! every run reproducible from `--seed` alone.

/// Salts naming the independent streams derived from `--seed`. Values are
/// arbitrary but fixed; [`mix`] spreads them apart.
pub const SALT_ARRIVAL: u64 = 1;
pub const SALT_SESSION: u64 = 2;
pub const SALT_PREFIX: u64 = 3;
pub const SALT_PAD: u64 = 4;
pub const SALT_SUFFIX: u64 = 5;
pub const SALT_IMAGE: u64 = 6;

/// Generated token ids stay below this bound — under the default image
/// placeholder id (151655), so padding never fakes an image run.
const TOKEN_ID_SPACE: u64 = 150_000;

const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// The splitmix64 output mixer.
pub fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Derive an independent stream seed from a parent seed and a salt.
pub fn sub_seed(seed: u64, salt: u64) -> u64 {
    mix(seed ^ mix(salt))
}

/// FNV-1a over the key bytes, finished with the splitmix64 mixer so
/// consecutive keys spread across a small modulus.
pub fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in s.as_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    mix(h)
}

/// splitmix64 sequence generator.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        mix(self.state)
    }

    /// Uniform in [0, 1): the top 53 bits as a double.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Exponential with the given mean via inverse CDF; `1 - u` keeps the
    /// argument of `ln` in (0, 1].
    pub fn next_exp(&mut self, mean: f64) -> f64 {
        -(1.0 - self.next_f64()).ln() * mean
    }

    /// Uniform index in [0, n). `n` must be non-zero.
    pub fn next_index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Piecewise-linear inverse CDF through (tokens, cumulative) anchors, with a
/// fixed low anchor at cumulative 0.0 and a tail anchor at 1.0.
pub struct PiecewiseCdf {
    /// (cumulative, tokens) points, strictly increasing in both.
    points: Vec<(f64, f64)>,
}

impl PiecewiseCdf {
    pub fn new(low_tokens: u32, anchors: &[(u32, f64)], max_tokens: u32) -> Self {
        let mut points = vec![(0.0, f64::from(low_tokens))];
        for &(tokens, cum) in anchors {
            points.push((cum, f64::from(tokens)));
        }
        // The tail segment reaches max_tokens at cumulative 1.0 unless the
        // caller's last anchor is already there.
        if points[points.len() - 1].0 < 1.0 {
            points.push((1.0, f64::from(max_tokens)));
        }
        Self { points }
    }

    /// Sample with `u` uniform in [0, 1), linearly interpolating between the
    /// two anchors bracketing `u`.
    pub fn sample(&self, u: f64) -> u32 {
        for pair in self.points.windows(2) {
            let (c0, t0) = pair[0];
            let (c1, t1) = pair[1];
            if u <= c1 {
                let width = c1 - c0;
                let t = if width > 0.0 {
                    t0 + (t1 - t0) * (u - c0) / width
                } else {
                    t1
                };
                return (t.round() as u32).max(1);
            }
        }
        (self.points[self.points.len() - 1].1.round() as u32).max(1)
    }
}

/// A deterministic run of token ids for the given stream seed.
pub fn token_ids(seed: u64, count: usize) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|_| (rng.next_u64() % TOKEN_ID_SPACE) as u32)
        .collect()
}

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// A deterministic base64-alphabet string of exactly `len` characters, so the
/// same (session, image) seed reproduces byte-identical payloads across turns.
pub fn base64_blob(seed: u64, len: usize) -> String {
    let mut rng = Rng::new(seed);
    let mut out = String::with_capacity(len);
    'fill: loop {
        // Ten 6-bit symbols per 64-bit draw; the top 4 bits are discarded.
        let mut word = rng.next_u64();
        for _ in 0..10 {
            if out.len() == len {
                break 'fill;
            }
            out.push(char::from(BASE64_ALPHABET[(word & 63) as usize]));
            word >>= 6;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_hits_its_anchors_exactly_and_stays_monotonic() {
        let cdf = PiecewiseCdf::new(
            256,
            &[(5000, 0.216), (10_000, 0.530), (20_000, 0.994)],
            32_000,
        );
        // Inverse-CDF at each anchor's cumulative probability returns the
        // anchor's token count — the property that makes the sampled
        // distribution match the production percentiles.
        assert_eq!(cdf.sample(0.216), 5000);
        assert_eq!(cdf.sample(0.530), 10_000);
        assert_eq!(cdf.sample(0.994), 20_000);
        assert_eq!(cdf.sample(0.0), 256);
        assert_eq!(cdf.sample(1.0), 32_000);

        let mut prev = 0;
        for i in 0..=1000 {
            let v = cdf.sample(f64::from(i) / 1000.0);
            assert!(v >= prev, "inverse CDF must be monotonic");
            assert!((256..=32_000).contains(&v));
            prev = v;
        }
    }

    #[test]
    fn derived_streams_are_deterministic_and_independent() {
        // Same seed → identical stream (turn-2 image regeneration and
        // cross-run reproducibility depend on this).
        assert_eq!(token_ids(7, 32), token_ids(7, 32));
        assert_eq!(base64_blob(7, 64), base64_blob(7, 64));
        // Different salts under one seed give unrelated streams.
        assert_ne!(
            token_ids(sub_seed(42, SALT_PREFIX), 32),
            token_ids(sub_seed(42, SALT_PAD), 32)
        );
        // Blob is exactly the requested length and base64-alphabet only.
        let blob = base64_blob(9, 257);
        assert_eq!(blob.len(), 257);
        assert!(blob
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/'));
    }
}
