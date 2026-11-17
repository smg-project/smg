//! What a plugin row resolves to before the host loads it. The spec's
//! `plugins` section (its names, its memory budget over every row, its
//! `max_candidates`) deserializes into these; until that row lands a host
//! builds them directly.

use std::fmt;

pub use crate::plugins::config::{BreakerSpec, Exposure, Failure, FrameBudget, HeaderSet, Reveal};
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
    /// (hashes, loads, headers, media, the affinity column behind a
    /// candidate's five affinity facts); the plugin declares it, the host
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

#[cfg(test)]
mod tests {
    use super::{Abi, PluginSpec};

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
