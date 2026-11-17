//! The parts of a plugin's row that are not about its kind: what of the
//! request it may read, what a stage does when it cannot answer, when the
//! breaker bypasses it, and the frame budget of one call.

/// How a tenant or a session key reaches the plugin: withheld, as its
/// fingerprint (the default), or in the clear when the host holds it so.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reveal {
    None,
    #[default]
    Hash,
    Plain,
}

/// The allow-listed headers a plugin sees, as bits over the spec's header
/// positions (at most 256 names). Empty by default: a plugin names what it
/// reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeaderSet([u64; 4]);

impl HeaderSet {
    /// No header.
    pub const NONE: Self = Self([0; 4]);

    /// Every position the view can carry.
    pub const ALL: Self = Self([u64::MAX; 4]);

    /// The set of these positions.
    pub fn of(positions: impl IntoIterator<Item = u8>) -> Self {
        let mut set = Self::NONE;
        for p in positions {
            set.0[usize::from(p >> 6)] |= 1 << (p & 63);
        }
        set
    }

    pub const fn contains(&self, position: u8) -> bool {
        self.0[(position >> 6) as usize] & (1 << (position & 63)) != 0
    }
}

/// The exposure of one plugin: what of the request's identities and headers
/// it may read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Exposure {
    pub tenant: Reveal,
    pub session_key: Reveal,
    pub headers: HeaderSet,
}

/// What a stage does when its plugin cannot answer (a trap, an exhausted
/// budget, a decline without a fallback stage, an open breaker).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Failure {
    /// The stage steps aside: a filter keeps every row, a scorer writes a
    /// flat score, a picker ranks by the combined scores.
    #[default]
    Open,
    /// The decision sheds, naming the stage: a filter empties the pool, a
    /// picker declines. A scorer has no shed path and writes a flat score
    /// under either policy.
    Closed,
}

/// When the plugin is bypassed to its fallbacks for a cool-down: this many
/// failures (traps, invalid answers, the budget hits the plugin ran out of
/// under its own bounds; one a decision, however many of its opens fail)
/// within the window, or a moving p50 of its decisions past half the budget
/// or a p99 (the second-slowest sample of the ring, each the plugin's own
/// charge in a decision over that decision's budget) past it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BreakerSpec {
    pub failures: u32,
    pub window_micros: u64,
    pub cooldown_micros: u64,
}

impl Default for BreakerSpec {
    fn default() -> Self {
        Self {
            failures: 5,
            window_micros: 10_000_000,
            cooldown_micros: 30_000_000,
        }
    }
}

/// A plugin's frame budget per stage call (`0` takes the request's) and the
/// fuel per microsecond it is metered at: the one place the frame is clamped
/// to the request's and turned into fuel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameBudget {
    pub micros: u32,
    pub fuel_per_micro: u64,
}

impl FrameBudget {
    /// The frame of one call under `request_budget_micros` left: the
    /// plugin's own, clamped to the request's.
    pub fn frame_micros(self, request_budget_micros: u32) -> u32 {
        if self.micros == 0 {
            request_budget_micros
        } else {
            self.micros.min(request_budget_micros)
        }
    }

    /// The fuel of one call's frame under `request_budget_micros` left; `0`
    /// means no call.
    pub fn fuel_for(self, request_budget_micros: u32) -> u64 {
        self.fuel_of(self.frame_micros(request_budget_micros))
    }

    /// The fuel of `micros` of wall time at this calibration.
    pub fn fuel_of(self, micros: u32) -> u64 {
        u64::from(micros).saturating_mul(self.fuel_per_micro)
    }
}

#[cfg(test)]
mod tests {
    use super::HeaderSet;

    #[test]
    fn a_header_set_holds_positions_over_the_whole_byte_range() {
        let set = HeaderSet::of([0, 63, 64, 255]);
        assert!(set.contains(0) && set.contains(63) && set.contains(64) && set.contains(255));
        assert!(!set.contains(1) && !set.contains(200));
        assert!(!HeaderSet::NONE.contains(0));
        assert!(HeaderSet::ALL.contains(129));
    }
}
