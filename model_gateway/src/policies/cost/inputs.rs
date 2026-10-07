//! Per-request and per-worker inputs a selection policy sees.
//!
//! The host (cache-aware routing) gathers these once per request from the index or tree and the
//! optimistic accounting, so a policy never touches a lock or a tree itself; nothing a policy
//! must not do is reachable from them (no handles to workers, no mutable shared state).

/// The request being routed.
#[derive(Debug, Clone, Copy)]
pub struct RequestInputs<'a> {
    /// Prompt length in tokens (the routing key's length for text routing).
    pub prompt_tokens: usize,
    /// Cache block size in tokens, at least 1.
    pub block_size: usize,
    /// Prompt length in blocks, at least 1 (an empty prompt still costs one block of accounting).
    pub request_blocks: usize,
    /// Mean in-flight request count over the healthy fleet.
    pub avg_load: f64,
    /// Chain hash of the prompt's blocks, position `i` covering blocks `0..=i`, when the host
    /// computed them (event-driven and token routing). Policies that key on prefixes need them;
    /// without them they fall back to load-only behaviour.
    pub prefix_hashes: Option<&'a [u64]>,
}

/// One eligible worker as the policy sees it.
#[derive(Debug, Clone)]
pub struct CandidateInputs<'a> {
    /// Position in the host's worker slice; policies return picks by position in the candidate
    /// slice, the host maps them back.
    pub idx: usize,
    /// Worker URL: the stable identity for deterministic tie-breaking and hashing.
    pub url: &'a str,
    /// Prefix blocks this worker holds on the device tier (GPU), undecayed, or the deeper
    /// placement the optimistic accounting predicts for it.
    pub device_blocks: f64,
    /// The host's decayed affinity score (device overlap after the waiting-prefill decay); what
    /// the cache-aware decision ranks on.
    pub effective_score: f64,
}

impl CandidateInputs<'_> {
    /// Prompt tokens this worker would still have to prefill after its device-resident prefix.
    pub fn uncached_prompt_tokens(&self, request: &RequestInputs<'_>) -> usize {
        let cached = (self.device_blocks.max(0.0) * request.block_size as f64) as usize;
        request.prompt_tokens.saturating_sub(cached)
    }
}
