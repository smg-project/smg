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

/// Each optional count: a value under its floor is refused naming the
/// field (1 for a plain count, one 1024-byte unit for the media budget,
/// which rounds anything smaller to 0); unset, the floor and a large value
/// pass.
#[test]
fn optional_counts_refuse_values_under_their_floor_and_take_unset_or_the_floor() {
    let knobs: [(&str, CountSetter, u64); 2] = [
        (
            "multimodal_max_inflight_bytes",
            |config, value| {
                config.multimodal_max_inflight_bytes =
                    value.map(|n| usize::try_from(n).expect("fits"));
            },
            INFLIGHT_UNIT_BYTES as u64,
        ),
        (
            "rdma_slot_ttl_s",
            |config, value| {
                config.rdma_slot_ttl_s = value;
            },
            1,
        ),
    ];
    for (field, set, floor) in knobs {
        let mut config = RouterConfig::default();
        for bad in [0, floor - 1] {
            set(&mut config, Some(bad));
            let (named, value, reason) = refusal(&config);
            assert_eq!(named, field, "{field} = {bad}");
            assert_eq!(value, bad.to_string(), "{field}");
            assert!(
                reason.starts_with(&format!("Must be at least {floor} when set")),
                "{field}: {reason}"
            );
        }
        for good in [None, Some(floor), Some(1 << 30)] {
            set(&mut config, good);
            assert!(validate(&config).is_ok(), "{field} = {good:?}");
        }
    }
}

/// The refused value is printed as the operator wrote it: an `f32` knob
/// is not widened to `f64` on the way to the error.
#[test]
fn a_refused_float_prints_as_written() {
    let config = RouterConfig {
        worker_warmup_share: 1.1,
        ..RouterConfig::default()
    };
    assert_eq!(refusal(&config).1, "1.1");
}
