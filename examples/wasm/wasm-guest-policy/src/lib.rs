//! An example policy plugin for the policy controller: a filter, a scorer and
//! a picker in one WebAssembly component of the `smg:policy` record world.
//!
//! The policy: among the healthy workers that have capacity, prefer the one
//! with the fewest requests in flight, with a bonus for the worker that
//! already holds the request's session.
//!
//! The host builds one `decision` per request, calls the stages the profile
//! binds and drops it. Nothing here outlives a decision but the configuration
//! `init` read: a plugin that keeps state across decisions is another example.

wit_bindgen::generate!({
    path: "../../../crates/policy_controller/contracts/smg-policy.wit",
    world: "policy",
});

use std::sync::atomic::{AtomicU32, Ordering};

use exports::smg::policy::plugin::{
    Candidate, Decline, Guest, GuestDecision, LifecycleEvent, Pool, Request,
};
use smg::policy::{host, types::Health};

/// The session bonus, in thousandths of a score. `init` sets it from the
/// plugin's configuration; the default is a fifth of the score range.
static SESSION_BONUS_MILLIS: AtomicU32 = AtomicU32::new(200);

/// The position, in the metrics the spec declares for this plugin, of the
/// counter of completed dispatches.
const COMPLETED: u32 = 0;

/// The host's log levels: 0 error, 1 warn, 2 info, 3 debug, 4 trace.
const INFO: u8 = 2;

struct Plugin;

impl Guest for Plugin {
    type Decision = Decision;

    /// Once per instance, with the plugin's configuration bytes: here one
    /// byte, the session bonus in percent. An error fails the apply.
    fn init(config: Vec<u8>) -> Result<(), String> {
        match config.as_slice() {
            [] => {}
            [percent @ 0..=100] => {
                SESSION_BONUS_MILLIS.store(u32::from(*percent) * 10, Ordering::Relaxed);
                host::log(INFO, &format!("session bonus {percent}%"));
            }
            [percent] => {
                return Err(format!(
                    "the session bonus is 0 to 100 percent, got {percent}"
                ))
            }
            _ => {
                return Err(format!(
                    "the configuration is one byte, got {}",
                    config.len()
                ))
            }
        }
        Ok(())
    }

    /// The lifecycle of the decisions the engine issued, outside any
    /// decision: this plugin counts the dispatches that completed.
    fn observe(event: LifecycleEvent) {
        if let LifecycleEvent::Completed(_) = event {
            host::counter(COMPLETED, 1);
        }
    }
}

/// What one decision keeps of its candidates between the stage calls: a row
/// per candidate, at the candidate's position.
struct Decision {
    rows: Vec<Row>,
}

struct Row {
    usable: bool,
    in_flight: u32,
    session_hit: bool,
}

/// The three stages. A stage that cannot judge a request answers
/// `Err(Decline::NotApplicable)` and the profile's fallback stage runs in its
/// place. With none bound, the stage and the plugin's failure mode decide: a
/// declined `filter` keeps every candidate under `open` and sheds the request
/// under `closed`; a declined `pick` ranks by the combined scores under `open`
/// and sheds under `closed`; a declined `score` is a flat score under either
/// mode, since a score is a preference and has no veto. This policy judges
/// every request, so its stages never decline.
impl GuestDecision for Decision {
    /// The view of one request: the request, the candidate rows and the
    /// pool's facts. A stage answers in positions of `cands`.
    fn new(_req: Request, cands: Vec<Candidate>, _pool: Pool) -> Self {
        let rows = cands
            .iter()
            .map(|candidate| Row {
                // Healthy and no other health flag (not draining, stalled,
                // overloaded or behind an open breaker), with capacity left.
                usable: candidate.health == Health::HEALTHY && candidate.has_capacity,
                in_flight: candidate.in_flight,
                session_hit: candidate.session_hit,
            })
            .collect();
        Self { rows }
    }

    /// The surviving positions, unique and in range. With no usable
    /// candidate the list is empty: a filter that empties the pool sheds
    /// the request, whatever the plugin's failure mode. (Bound as a soft
    /// filter it steps aside instead, and the profile's other stages
    /// decide.)
    fn filter(&self) -> Result<Vec<u32>, Decline> {
        let kept: Vec<u32> = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.usable)
            .map(|(position, _)| position as u32)
            .collect();
        Ok(kept)
    }

    /// One finite score per candidate (a NaN is a trap), 0 to 1: idle
    /// workers score high, the session's worker higher. The host clamps a
    /// score to that range, which would level an idle worker with the
    /// session's idle worker, so the sum is scaled back into it.
    fn score(&self) -> Result<Vec<f32>, Decline> {
        let bonus = SESSION_BONUS_MILLIS.load(Ordering::Relaxed) as f32 / 1000.0;
        Ok(self
            .rows
            .iter()
            .map(|row| {
                let idle = 1.0 / (1.0 + row.in_flight as f32);
                let held = if row.session_hit { bonus } else { 0.0 };
                (idle + held) / (1.0 + bonus)
            })
            .collect())
    }

    /// A position within the surviving set. The host passes the combined
    /// scores back, and a row the filters removed carries a score below
    /// zero, so the highest score is a surviving row's; a picker that
    /// prefers anything else skips those rows itself. The lower position
    /// wins a tie; `None` declines the pick.
    fn pick(&self, scores: Vec<f32>) -> Result<Option<u32>, Decline> {
        let mut best: Option<(u32, f32)> = None;
        for (position, score) in scores.iter().enumerate() {
            if best.is_none_or(|(_, top)| *score > top) {
                best = Some((position as u32, *score));
            }
        }
        Ok(best.map(|(position, _)| position))
    }
}

export!(Plugin);
