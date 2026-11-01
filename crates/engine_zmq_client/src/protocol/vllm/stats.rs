// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/stats.rs.

use serde::{Deserialize, Serialize};

/// Breakdown of a scheduled prefill computation.
///
/// Python models this as a plain `@dataclass`, so msgspec serializes it as a
/// map (named fields), not the array-like form used by `EngineCoreOutput`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillStats {
    #[serde(default)]
    pub num_prompt_tokens: u32,
    #[serde(default)]
    pub num_computed_tokens: u32,
    #[serde(default)]
    pub num_cached_tokens: u32,
    #[serde(default)]
    pub num_local_cached_tokens: u32,
    #[serde(default)]
    pub num_external_cached_tokens: u32,
    #[serde(default)]
    pub num_cache_creation_tokens: u32,
}

/// The scheduler stats piggybacked on every output batch, as far as this
/// client reads them: the per-rank load signal. Mirrors the load fields of
/// Python `SchedulerStats`.
///
/// `num_running_reqs` / `num_waiting_reqs` are SMG's per-rank DP load signal —
/// fresher than any polled `GetLoads`, and the reason the connector consumes
/// these instead of the reference's client-side load balancer. The engine's
/// map carries more (prefix-cache and connector-cache hit counts, eviction
/// samples, speculative-decoding, connector, graph and performance stats);
/// nothing on the receiving side reads those, so the decoder skips them and
/// the stats ride the batch unboxed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Number of requests in model execution batches.
    pub num_running_reqs: u64,
    /// Length of the "waiting" request queue.
    pub num_waiting_reqs: u64,
    /// Length of the "skipped waiting" queue.
    #[serde(default)]
    pub num_skipped_waiting_reqs: u64,
    /// KV-cache usage. `1.0` means 100% usage.
    pub kv_cache_usage: f64,
}
