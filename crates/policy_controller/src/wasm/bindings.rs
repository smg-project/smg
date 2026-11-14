//! The host side of the committed `smg:policy` package, generated from
//! `contracts/smg-policy.wit` at compile time: the record world and the
//! packed world share one `types` interface and one `host` interface, so a
//! guest value has one Rust type whichever world its module targets. The
//! borrowing form lowers the host's slices and strings without a copy.

use std::{sync::Arc, time::Instant};

use wasmtime::StoreLimits;

use super::{config::LogLimit, counters::Counters};
use crate::Draw;

/// The record world, `smg:policy/policy`.
pub(crate) mod record {
    wasmtime::component::bindgen!({
        path: "contracts/smg-policy.wit",
        world: "policy",
        ownership: Borrowing { duplicate_if_necessary: false },
        additional_derives: [PartialEq],
    });
}

/// The packed world, `smg:policy/policy-packed`, over the record world's
/// `types` and `host`.
pub(crate) mod packed {
    wasmtime::component::bindgen!({
        path: "contracts/smg-policy.wit",
        world: "policy-packed",
        ownership: Borrowing { duplicate_if_necessary: false },
        additional_derives: [PartialEq],
        with: {
            "smg:policy/types@1.0.0": super::record::smg::policy::types,
            "smg:policy/host@1.0.0": super::record::smg::policy::host,
        },
    });
}

pub(crate) use record::smg::policy::types as wit;

/// What a store carries beside the instance: the memory limiter and the
/// state the four host imports read.
pub(crate) struct HostState {
    pub(crate) limits: StoreLimits,
    /// The plugin's name, for its log lines.
    pub(crate) name: &'static str,
    /// The decision's clock inside the stages (fixed for the whole decision,
    /// so a replay reads the same instant); `None` outside a decision, when
    /// `now-micros` reads the live clock.
    pub(crate) decision_micros: Option<u64>,
    /// The draw sequence of the decision's seed.
    pub(crate) draw: Draw,
    /// The counters the spec declared for the plugin, by id.
    pub(crate) counters: Counters,
    pub(crate) log: LogGate,
    /// The host's clock, for `now-micros` outside a decision.
    pub(crate) clock: Arc<dyn Fn() -> u64 + Send + Sync>,
}

/// The per-plugin log gate: a level cap and a token bucket refilled once a
/// second.
pub(crate) struct LogGate {
    limit: LogLimit,
    tokens: u32,
    refilled_at: Instant,
}

impl LogGate {
    pub(crate) fn new(limit: LogLimit, now: Instant) -> Self {
        Self {
            tokens: limit.per_second,
            limit,
            refilled_at: now,
        }
    }

    /// Takes one token, refilling the bucket when a second has passed.
    fn admit(&mut self, now: Instant) -> bool {
        if now.duration_since(self.refilled_at).as_secs() >= 1 {
            self.tokens = self.limit.per_second;
            self.refilled_at = now;
        }
        if self.tokens == 0 {
            return false;
        }
        self.tokens -= 1;
        true
    }
}

/// The guest's log levels: 0 error, 1 warn, 2 info, 3 debug, 4 and above
/// trace. A level below the plugin's cap is raised to the cap.
pub(crate) fn capped_level(level: u8, cap: u8) -> u8 {
    level.max(cap)
}

impl record::smg::policy::host::Host for HostState {
    fn log(&mut self, level: u8, msg: String) {
        let now = Instant::now();
        if !self.log.admit(now) {
            return;
        }
        let plugin = self.name;
        match capped_level(level, self.log.limit.level_cap) {
            0 => tracing::error!(plugin, "{msg}"),
            1 => tracing::warn!(plugin, "{msg}"),
            2 => tracing::info!(plugin, "{msg}"),
            3 => tracing::debug!(plugin, "{msg}"),
            _ => tracing::trace!(plugin, "{msg}"),
        }
    }

    fn now_micros(&mut self) -> u64 {
        match self.decision_micros {
            Some(micros) => micros,
            None => (self.clock)(),
        }
    }

    fn counter(&mut self, id: u32, by: u64) {
        self.counters.add(id, by);
    }

    fn random_u64(&mut self) -> u64 {
        self.draw.next_u64()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{capped_level, LogGate};
    use crate::wasm::config::LogLimit;

    #[test]
    fn the_log_gate_admits_its_per_second_budget_and_refills_after_a_second() {
        let start = Instant::now();
        let mut gate = LogGate::new(
            LogLimit {
                level_cap: 1,
                per_second: 2,
            },
            start,
        );
        assert!(gate.admit(start));
        assert!(gate.admit(start));
        assert!(!gate.admit(start + Duration::from_millis(999)));
        assert!(gate.admit(start + Duration::from_secs(1)));
    }

    #[test]
    fn a_level_below_the_cap_is_raised_to_it() {
        assert_eq!(capped_level(0, 2), 2);
        assert_eq!(capped_level(3, 2), 3);
    }
}
