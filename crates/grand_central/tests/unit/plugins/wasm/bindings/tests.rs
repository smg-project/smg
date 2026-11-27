use std::time::{Duration, Instant};

use super::{capped_level, LogGate};
use crate::plugins::wasm::config::LogLimit;

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
