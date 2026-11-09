//! The bounded view: the per-decision, handle-free picture of one request and
//! its candidates. Built once per decision by the host into a reused buffer;
//! every row is `Copy`, and nothing here reaches back into the host.

/// The stable identity of a candidate: a hash of the registration key and the
/// incarnation, never a URL. A same-URL replacement is a new id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub u64);

/// A worker's role in the fleet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Role {
    #[default]
    Single,
    Prefill,
    Decode,
    Encode,
}

/// Which phase of a request a decision is for.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    #[default]
    Single,
    Prefill,
    Decode,
    Encode,
}

/// What the request's block hashes were computed over.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HashKind {
    /// Token blocks of the model's block size of record.
    #[default]
    Token,
    /// Fixed character windows over the text, for a host without a tokenizer.
    Text,
}

/// Where a candidate's load row came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SignalSource {
    #[default]
    None,
    Events,
    Scrape,
    Relay,
    Poll,
}

/// The pool's overload regime, as the host reads it per decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OverloadRegime {
    #[default]
    None,
    Partial,
    Saturated,
}

/// Health as the pool reads it. Each flag is one bit of the host's routing
/// state, read once per decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Health {
    /// The worker's status is ready.
    pub healthy: bool,
    /// The pool's circuit breaker refuses execution.
    pub breaker_open: bool,
    /// The liveness veto: unreachable or wedged.
    pub stalled: bool,
    /// Being drained; nothing new goes there.
    pub draining: bool,
    /// The latched overload verdict of the last load report.
    pub overloaded: bool,
}

impl Health {
    /// Every routing veto but the overload one.
    pub const fn routable_ignoring_overload(self) -> bool {
        self.healthy && !self.breaker_open && !self.stalled && !self.draining
    }

    /// Every routing veto.
    pub const fn routable(self) -> bool {
        self.routable_ignoring_overload() && !self.overloaded
    }
}

/// One load report as the view carries it. Narrow fields keep the row small;
/// the host rounds its wider counters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LoadReport {
    /// Requests running on the engine.
    pub running: u32,
    /// Requests waiting in the engine's queue.
    pub waiting: u32,
    /// Token-work of the waiting queue, when the engine reports it (`0` when
    /// it reports a depth alone).
    pub waiting_uncached_tokens: u32,
    /// KV-cache pressure, `0.0` to `1.0`.
    pub kv_pressure: f32,
    /// Resident token usage, `0.0` to `1.0`.
    pub token_usage: f32,
    /// The worker's live drain rate in tokens per second, `0.0` for none.
    pub live_rate: f32,
    /// Age of the report's sample at the decision, in microseconds.
    pub age_micros: u32,
    /// The report's sample instant on the host's clock, in microseconds since
    /// the host's origin: dispatches made at or before it are the engine's by
    /// now and leave a load scorer's credit.
    pub sampled_at_micros: u64,
}

/// One candidate row: a worker, or one DP rank of a data-parallel worker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Candidate {
    /// The row's position in the host's candidate slice; targets name it.
    pub position: u32,
    pub id: WorkerId,
    pub role: Role,
    pub dp_rank: u16,
    pub dp_size: u16,
    pub health: Health,
    /// Requests the host holds on the worker right now.
    pub in_flight: u32,
    /// The latest load report, when the host has one for the worker.
    pub report: Option<LoadReport>,
    pub signal: SignalSource,
    /// Prefix blocks of the request this worker holds, as the host's index
    /// or predictor reads them.
    pub overlap_blocks: u32,
    /// The session key of the request is pinned to this worker.
    pub session_hit: bool,
    /// Requests of the request's session key in flight on this worker.
    pub session_in_flight: u32,
    /// The PD partner group: a decode may pair with a prefill of the same
    /// group. `0` for a worker outside any pairing.
    pub partner_group: u32,
    /// Host-supplied admission bit: the prefill capacity window has a slot.
    pub has_capacity: bool,
    /// Host-supplied admission bit: the decode's running window has a room
    /// nobody waiting at its gate is in line for.
    pub has_room: bool,
    /// Milliseconds since the worker was admitted to the pool.
    pub admitted_age_ms: u32,
    /// Blocks the index learned about the worker since admission.
    pub indexed_growth: u32,
    pub zone: u16,
}

/// Media on the request: counts and bytes, for workers that take media.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MediaCounts {
    pub images: u16,
    pub audio: u16,
    pub video: u16,
    pub bytes: u32,
}

/// One request as a decision sees it. Borrowed from the host for the
/// duration of the decision.
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestFacts<'a> {
    /// The resolved model id, after aliasing.
    pub model: &'a str,
    /// The cache namespace folded into a seed (salt, adapter, extra key).
    pub namespace_seed: u64,
    /// The tenant, hashed.
    pub tenant: u64,
    /// Priority as a signed band; `0` without a scheduler.
    pub priority: i32,
    pub stream: bool,
    pub max_output_tokens: u32,
    /// The host's token count or estimate.
    pub prompt_tokens: u32,
    /// Prefix block hashes, capped by the host.
    pub block_hashes: &'a [u64],
    pub hash_kind: HashKind,
    /// The routing key as derived today, lineage-stripped.
    pub session_key: Option<&'a str>,
    pub phase: Phase,
    /// The retry number, `0` for the first attempt.
    pub attempt: u8,
    /// Candidates earlier attempts of this request already tried.
    pub tried: &'a [WorkerId],
    /// Time left for the decision, relative to the host's receipt.
    pub budget_micros: u32,
    /// Allow-listed headers only.
    pub headers: &'a [(&'a str, &'a str)],
    pub media: MediaCounts,
    /// The per-decision seed, recorded in the trace.
    pub seed: u64,
    /// The host's clock at the decision, in microseconds since its origin.
    pub now_micros: u64,
}

/// What only the deciding host knows now: its standing vetoes and the
/// request-level lifts around the profile.
#[derive(Clone, Copy, Debug, Default)]
pub struct Constraints<'a> {
    /// Candidates this host just failed to reach.
    pub excluded: &'a [WorkerId],
    /// The overload veto does not apply to this decision (a pool whose
    /// policy balances load against affinity itself, a session owner).
    pub overload_lifted: bool,
    /// A subset hint: decide over at most this many candidates.
    pub subset_hint: Option<u32>,
}

/// Pool-wide facts beside the candidates.
#[derive(Clone, Copy, Debug, Default)]
pub struct PoolFacts {
    /// Pool size before any narrowing.
    pub pool_size: u32,
    pub membership_epoch: u64,
    pub overload_regime: OverloadRegime,
    /// The spec generation the decision ran under.
    pub generation: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_flags_compose_into_the_two_routability_reads() {
        let ready = Health {
            healthy: true,
            ..Health::default()
        };
        assert!(ready.routable());
        let overloaded = Health {
            overloaded: true,
            ..ready
        };
        assert!(!overloaded.routable());
        assert!(overloaded.routable_ignoring_overload());
        let open = Health {
            breaker_open: true,
            ..ready
        };
        assert!(!open.routable_ignoring_overload());
    }

    #[test]
    fn a_candidate_row_stays_small() {
        // The row is copied once per candidate per decision; a change that
        // grows it past two cache lines is a design change, not a field.
        assert!(size_of::<Candidate>() <= 128);
    }
}
