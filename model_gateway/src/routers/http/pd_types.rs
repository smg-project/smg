//! Types and utilities for the prefill-decode (PD) disaggregated router.

use serde::Serialize;

/// Optimized bootstrap wrapper for single requests.
#[derive(Serialize)]
pub struct RequestWithBootstrap<'a, T: Serialize> {
    #[serde(flatten)]
    pub original: &'a T,
    pub bootstrap_host: String,
    pub bootstrap_port: Option<u16>,
    pub bootstrap_room: u64,
}

/// Generate a random bootstrap room ID.
pub fn generate_room_id() -> u64 {
    // Generate a value in the range [0, 2^63 - 1] to match Python's random.randint(0, 2**63 - 1)
    rand::random::<u64>() & (i64::MAX as u64)
}

/// Generate a random bootstrap room ID in the same range, congruent to `rank`
/// modulo `dp_size`: SGLang's prefill side, under its default
/// `follow_bootstrap_room` placement, fails a request whose
/// `bootstrap_room % dp_size` is not the rank it runs on. The caller passes
/// `dp_size > 1` and `rank < dp_size`.
pub fn generate_room_id_for_rank(rank: usize, dp_size: usize) -> u64 {
    let dp_size = dp_size as u64;
    let base = rand::random_range(0..i64::MAX as u64 - dp_size);
    base - base % dp_size + rank as u64
}

/// PD-specific routing policies.
#[derive(Debug, Clone, PartialEq)]
pub enum PDSelectionPolicy {
    Random,
    PowerOfTwo,
    CacheAware {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
    },
    Bucket {
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        bucket_adjust_interval_secs: usize,
    },
}
