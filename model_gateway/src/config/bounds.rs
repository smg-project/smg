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

/// The bound one knob must satisfy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bound {
    /// A finite share from 0.0 to 1.0, both included.
    UnitFraction,
    /// A count or a duration of at least 1 when it is set at all; unset
    /// keeps the documented default.
    AtLeastOneWhenSet,
}

/// The value a row checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Knob {
    Float(f64),
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
            // A row pairs a bound with the kind of value it is written for;
            // a mismatch is a table bug, refused loudly rather than passed.
            (Bound::UnitFraction, Knob::OptionalCount(_))
            | (Bound::AtLeastOneWhenSet, Knob::Float(_)) => false,
        }
    }

    /// The bound, as the error states it.
    fn requirement(self) -> &'static str {
        match self {
            Bound::UnitFraction => "Must be a finite value from 0.0 to 1.0",
            Bound::AtLeastOneWhenSet => "Must be at least 1 when set",
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
            value: Knob::Float(f64::from(config.worker_warmup_share)),
            bound: Bound::UnitFraction,
            reason: "the share of cache misses offered to warming workers",
        },
        Row {
            field: "worker_warmup_thin_ratio",
            value: Knob::Float(f64::from(config.worker_warmup_thin_ratio)),
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
            bound: Bound::AtLeastOneWhenSet,
            reason: "the budget of media bytes in flight; 0 would refuse every request \
                     carrying media, unset holds an unbounded amount",
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
mod tests {
    use super::*;

    /// A setter for a float knob of the configuration.
    type FloatSetter = fn(&mut RouterConfig, f32);
    /// A setter for an optional count knob of the configuration.
    type CountSetter = fn(&mut RouterConfig, Option<u64>);

    /// The field, value and reason of the error `config` gets.
    fn refusal(config: &RouterConfig) -> (String, String, String) {
        match validate(config) {
            Err(ConfigError::InvalidValue {
                field,
                value,
                reason,
            }) => (field, value, reason),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn the_default_configuration_is_within_every_bound() {
        assert!(validate(&RouterConfig::default()).is_ok());
    }

    #[test]
    fn every_row_names_a_distinct_field() {
        let rows = rows(&RouterConfig::default());
        let mut fields: Vec<_> = rows.iter().map(|row| row.field).collect();
        fields.sort_unstable();
        fields.dedup();
        assert_eq!(fields.len(), rows.len());
    }

    /// Each unit-fraction knob: NaN, the infinities and values outside
    /// 0.0..=1.0 are refused naming the field; the ends and a value between
    /// them pass.
    #[test]
    fn unit_fractions_refuse_nan_infinities_and_values_outside_zero_to_one() {
        let knobs: [(&str, FloatSetter); 2] = [
            ("worker_warmup_share", |config, value| {
                config.worker_warmup_share = value;
            }),
            ("worker_warmup_thin_ratio", |config, value| {
                config.worker_warmup_thin_ratio = value;
            }),
        ];
        for (field, set) in knobs {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.5, 1e30] {
                let mut config = RouterConfig::default();
                set(&mut config, bad);
                let (named, _, reason) = refusal(&config);
                assert_eq!(named, field, "{field} = {bad}");
                assert!(
                    reason.starts_with("Must be a finite value from 0.0 to 1.0"),
                    "{field} = {bad}: {reason}"
                );
            }
            for good in [0.0, 0.25, 1.0] {
                let mut config = RouterConfig::default();
                set(&mut config, good);
                assert!(validate(&config).is_ok(), "{field} = {good}");
            }
        }
        let config = RouterConfig {
            worker_warmup_share: f32::NAN,
            ..RouterConfig::default()
        };
        assert_eq!(refusal(&config).1, "NaN");
    }

    /// Each optional count: 0 is refused naming the field; unset, 1 and a
    /// large value pass.
    #[test]
    fn optional_counts_refuse_zero_and_take_unset_or_one() {
        let knobs: [(&str, CountSetter); 2] = [
            ("multimodal_max_inflight_bytes", |config, value| {
                config.multimodal_max_inflight_bytes =
                    value.map(|n| usize::try_from(n).expect("fits"));
            }),
            ("rdma_slot_ttl_s", |config, value| {
                config.rdma_slot_ttl_s = value;
            }),
        ];
        for (field, set) in knobs {
            let mut config = RouterConfig::default();
            set(&mut config, Some(0));
            let (named, value, reason) = refusal(&config);
            assert_eq!(named, field);
            assert_eq!(value, "0", "{field}");
            assert!(
                reason.starts_with("Must be at least 1 when set"),
                "{field}: {reason}"
            );
            for good in [None, Some(1), Some(1 << 30)] {
                set(&mut config, good);
                assert!(validate(&config).is_ok(), "{field} = {good:?}");
            }
        }
    }
}
