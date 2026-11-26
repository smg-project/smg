use super::{Abi, PluginSpec};

#[test]
fn the_abi_range_is_one_major_from_the_floor_to_the_supported_minor() {
    assert!(Abi::V1_2.accepted());
    assert!(!Abi { major: 2, minor: 2 }.accepted());
    // Under the floor: the minors a shape was changed in place after.
    assert!(!Abi { major: 1, minor: 0 }.accepted());
    assert!(!Abi::V1_1.accepted());
    assert_eq!(Abi::accepted_range(), "1.2");
    assert!(!Abi {
        major: 1,
        minor: Abi::SUPPORTED_MINOR + 1
    }
    .accepted());
    assert_eq!(Abi::parse("1.2.0"), Some(Abi::V1_2));
    assert_eq!(Abi::parse("1.1"), Some(Abi::V1_1));
    assert_eq!(Abi::parse("one"), None);
    assert_eq!(Abi::V1_2.to_string(), "1.2");
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
