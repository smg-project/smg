//! The bounds of the router configuration's free-standing numeric knobs.
//!
//! One table row per knob: the field an operator sets, the bound the value
//! must satisfy and the reason the error reports when it does not.
//! `ConfigValidator::validate` runs the table for every start-up (command
//! line, configuration file and the bindings alike), so a value outside its
//! bound is a named configuration error before anything binds, instead of a
//! NaN no comparison ever catches, a share above one or a zero that sizes a
//! budget away. The knobs a policy owns keep their rules next to that policy
//! (`validate_policy`); this table holds the knobs the gateway reads whatever
//! the policy.

use std::fmt;

use super::{ConfigError, ConfigResult, RouterConfig};
use crate::routers::grpc::multimodal::INFLIGHT_UNIT_BYTES;

/// The bound one knob must satisfy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bound {
    /// A finite share from 0.0 to 1.0, both included.
    UnitFraction,
    /// A count or a duration of at least 1 when it is set at all; unset
    /// keeps the documented default.
    AtLeastOneWhenSet,
    /// A count of at least this many when it is set at all (a budget with a
    /// granularity: below one unit it rounds to nothing); unset keeps the
    /// documented default.
    AtLeastWhenSet(u64),
}

/// The value a row checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Knob {
    /// The configuration's float knobs are `f32`; kept as such, so the error
    /// prints the value the operator wrote (`1.1`, not its `f64` widening).
    Float(f32),
    OptionalCount(Option<u64>),
}

/// One row of the table.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Row {
    /// The field, as the configuration and the error name it.
    pub field: &'static str,
    pub value: Knob,
    pub bound: Bound,
    /// What the knob means, for the error.
    pub reason: &'static str,
}

impl Bound {
    /// Whether `value` satisfies the bound. A NaN satisfies none: every
    /// comparison with it is false, which is how it passed before.
    fn holds(self, value: Knob) -> bool {
        match (self, value) {
            (Bound::UnitFraction, Knob::Float(v)) => v.is_finite() && (0.0..=1.0).contains(&v),
            (Bound::AtLeastOneWhenSet, Knob::OptionalCount(v)) => v.is_none_or(|n| n >= 1),
            (Bound::AtLeastWhenSet(least), Knob::OptionalCount(v)) => v.is_none_or(|n| n >= least),
            // A row pairs a bound with the kind of value it is written for;
            // a mismatch is a table bug, refused loudly rather than passed.
            (Bound::UnitFraction, Knob::OptionalCount(_))
            | (Bound::AtLeastOneWhenSet | Bound::AtLeastWhenSet(_), Knob::Float(_)) => false,
        }
    }

    /// The bound, as the error states it.
    fn requirement(self) -> String {
        match self {
            Bound::UnitFraction => "Must be a finite value from 0.0 to 1.0".to_string(),
            Bound::AtLeastOneWhenSet => "Must be at least 1 when set".to_string(),
            Bound::AtLeastWhenSet(least) => format!("Must be at least {least} when set"),
        }
    }
}

impl fmt::Display for Knob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Knob::Float(v) => write!(f, "{v}"),
            Knob::OptionalCount(Some(n)) => write!(f, "{n}"),
            Knob::OptionalCount(None) => f.write_str("unset"),
        }
    }
}

/// The table for `config`.
pub(crate) fn rows(config: &RouterConfig) -> [Row; 4] {
    [
        Row {
            field: "worker_warmup_share",
            value: Knob::Float(config.worker_warmup_share),
            bound: Bound::UnitFraction,
            reason: "the share of cache misses offered to warming workers",
        },
        Row {
            field: "worker_warmup_thin_ratio",
            value: Knob::Float(config.worker_warmup_thin_ratio),
            bound: Bound::UnitFraction,
            reason: "the share of the fleet's index level under which a worker is thin; \
                     0 keeps the age rule alone",
        },
        Row {
            field: "multimodal_max_inflight_bytes",
            value: Knob::OptionalCount(
                config
                    .multimodal_max_inflight_bytes
                    .map(|bytes| u64::try_from(bytes).unwrap_or(u64::MAX)),
            ),
            bound: Bound::AtLeastWhenSet(INFLIGHT_UNIT_BYTES as u64),
            reason: "the budget of media bytes in flight, counted in units of 1024 bytes; \
                     below one unit it rounds to 0 and would refuse every request carrying \
                     media, unset holds an unbounded amount",
        },
        Row {
            field: "rdma_slot_ttl_s",
            value: Knob::OptionalCount(config.rdma_slot_ttl_s),
            bound: Bound::AtLeastOneWhenSet,
            reason: "the seconds a leased RDMA pixel slot lives; unset derives the TTL from \
                     the worker's hold",
        },
    ]
}

/// Every row holds, or the first that does not as the configuration error.
pub(crate) fn validate(config: &RouterConfig) -> ConfigResult<()> {
    rows(config)
        .into_iter()
        .find(|row| !row.bound.holds(row.value))
        .map_or(Ok(()), |row| {
            Err(ConfigError::InvalidValue {
                field: row.field.to_string(),
                value: row.value.to_string(),
                reason: format!("{} ({})", row.bound.requirement(), row.reason),
            })
        })
}

#[cfg(test)]
#[path = "../../tests/unit/config/bounds/tests.rs"]
mod tests;
