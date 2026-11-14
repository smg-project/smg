//! What a plugin row resolves to before the host loads it. The spec's
//! `plugins` section (its names, its memory budget over every row, its
//! `max_candidates`) deserializes into these; until that row lands a host
//! builds them directly.

use std::fmt;

use crate::spec::Needs;

/// The `smg:policy` package versions the host accepts: one major, every
/// minor up to [`Abi::SUPPORTED_MINOR`]. A 1.y release adds host functions,
/// interfaces or worlds a module may leave unused; a changed shape is the
/// next major.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Abi {
    pub major: u32,
    pub minor: u32,
}

impl Abi {
    /// The major the host speaks.
    pub const MAJOR: u32 = 1;
    /// The newest minor the host speaks.
    pub const SUPPORTED_MINOR: u32 = 0;

    /// The one committed version today.
    pub const V1_0: Self = Self { major: 1, minor: 0 };

    /// Whether a module of this version runs on this host: the major must
    /// match and the minor lie within the supported range.
    pub fn accepted(self) -> bool {
        self.major == Self::MAJOR && (..=Self::SUPPORTED_MINOR).contains(&self.minor)
    }

    /// `major.minor` of a `major.minor.patch` suffix; the patch is dropped.
    pub(crate) fn parse(version: &str) -> Option<Self> {
        let mut parts = version.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        Some(Self { major, minor })
    }
}

impl fmt::Display for Abi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

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

/// The plugin's log allowance: a level cap (0 error, 1 warn, 2 info, 3
/// debug, 4 trace; a lower level is raised to the cap) and lines per
/// second.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogLimit {
    pub level_cap: u8,
    pub per_second: u32,
}

impl Default for LogLimit {
    fn default() -> Self {
        Self {
            level_cap: 2,
            per_second: 10,
        }
    }
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

/// One plugin as the host loads it.
#[derive(Clone, Debug)]
pub struct PluginSpec {
    /// The name the trace and the shed carry as the stage.
    pub name: String,
    /// The component's bytes. Sourcing (a file or an image reference pulled
    /// at validation and cached by digest) hands them over; the host reads
    /// no path itself.
    pub component: Vec<u8>,
    /// The spec's `abi` pin; `None` accepts any version in the host's range.
    pub abi: Option<Abi>,
    /// Passed to `init` once per instance.
    pub config: Vec<u8>,
    /// The linear-memory cap of one instance, in MiB.
    pub memory_mib: u32,
    /// The plugin's own frame budget per stage call, in microseconds; `0`
    /// takes the decision's remaining budget. The host clamps it to at most
    /// the decision's, and meters it as fuel through `fuel_per_micro`.
    pub budget_micros: u32,
    /// The host's calibration of fuel to wall time on the reference host.
    pub fuel_per_micro: u64,
    pub view: Exposure,
    /// What the host must feed for the plugin's stages to mean anything
    /// (hashes, loads, headers, media); the plugin declares it, the host
    /// cannot read it off a module.
    pub needs: Needs,
    /// The counters the plugin may add to, by id; a name per id.
    pub metrics: Vec<String>,
    pub failure: Failure,
    pub log: LogLimit,
    pub breaker: BreakerSpec,
}

impl PluginSpec {
    /// The default memory cap, in MiB.
    pub const DEFAULT_MEMORY_MIB: u32 = 16;

    /// The default calibration, fuel units per microsecond of wall time:
    /// the test guests measured 524 (the record form's decision, host calls
    /// among it), 1,190 (a tight loop) and 1,290 (the packed form's
    /// decision) on the reference host; 1,000 is the rounded middle, so a
    /// frame ends at or before its wall budget on a busy loop.
    pub const DEFAULT_FUEL_PER_MICRO: u64 = 1_000;

    /// A plugin of `name` over `component` with every default.
    pub fn new(name: impl Into<String>, component: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            component,
            abi: None,
            config: Vec::new(),
            memory_mib: Self::DEFAULT_MEMORY_MIB,
            budget_micros: 0,
            fuel_per_micro: Self::DEFAULT_FUEL_PER_MICRO,
            view: Exposure::default(),
            needs: Needs::default(),
            metrics: Vec::new(),
            failure: Failure::default(),
            log: LogLimit::default(),
            breaker: BreakerSpec::default(),
        }
    }

    /// The plugin's frame budget with its calibration: the owner of the
    /// frame rules the host applies per call.
    pub fn frame(&self) -> FrameBudget {
        FrameBudget {
            micros: self.budget_micros,
            fuel_per_micro: self.fuel_per_micro,
        }
    }

    /// The fuel of one stage call under `request_budget_micros` of decision
    /// budget left: the plugin's frame budget clamped to the decision's,
    /// times the calibration. `0` means no call: the budget is gone.
    pub fn fuel_for(&self, request_budget_micros: u32) -> u64 {
        self.frame().fuel_for(request_budget_micros)
    }

    /// The frame budget of one stage call in microseconds, for the latency
    /// breaker.
    pub fn frame_micros(&self, request_budget_micros: u32) -> u32 {
        self.frame().frame_micros(request_budget_micros)
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
    use super::{Abi, HeaderSet, PluginSpec};

    #[test]
    fn the_abi_range_is_one_major_up_to_the_supported_minor() {
        assert!(Abi::V1_0.accepted());
        assert!(!Abi { major: 2, minor: 0 }.accepted());
        assert!(!Abi {
            major: 1,
            minor: Abi::SUPPORTED_MINOR + 1
        }
        .accepted());
        assert_eq!(Abi::parse("1.0.0"), Some(Abi::V1_0));
        assert_eq!(Abi::parse("1.2"), Some(Abi { major: 1, minor: 2 }));
        assert_eq!(Abi::parse("one"), None);
        assert_eq!(Abi::V1_0.to_string(), "1.0");
    }

    #[test]
    fn a_header_set_holds_positions_over_the_whole_byte_range() {
        let set = HeaderSet::of([0, 63, 64, 255]);
        assert!(set.contains(0) && set.contains(63) && set.contains(64) && set.contains(255));
        assert!(!set.contains(1) && !set.contains(200));
        assert!(!HeaderSet::NONE.contains(0));
        assert!(HeaderSet::ALL.contains(129));
    }

    #[test]
    fn the_frame_budget_is_the_plugins_clamped_to_the_decisions() {
        let mut spec = PluginSpec::new("p", Vec::new());
        spec.fuel_per_micro = 10;
        assert_eq!(spec.fuel_for(500), 5_000);
        assert_eq!(spec.frame_micros(500), 500);
        spec.budget_micros = 200;
        assert_eq!(spec.fuel_for(500), 2_000);
        assert_eq!(spec.fuel_for(100), 1_000);
        assert_eq!(spec.frame_micros(100), 100);
        assert_eq!(spec.fuel_for(0), 0);
    }
}
