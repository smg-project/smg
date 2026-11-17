//! The cases of a generation that moves the pairing mode and the overload
//! block at once: the membership is derived again under the new mode, and
//! every engine is judged under the new block by the same apply.

use std::time::{Duration, Instant};

use super::{block, member, overloaded, pressure};
use crate::{
    spec::{BindingSpec, OverloadSpec, Pairing},
    store::{LoadSink, LoadSource, MemoryStore},
    AppliedSpec, PolicySpec,
};

/// A generation whose pair plan names the pairing `mode` and whose one
/// binding hands every pool the block `overload`.
fn generation_of(number: u64, mode: Pairing, overload: OverloadSpec) -> AppliedSpec {
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = number;
    document.plans.pd_pair.pairing = mode;
    let profile = document
        .profiles
        .keys()
        .next()
        .expect("the preset's profile")
        .clone();
    document.bindings = vec![BindingSpec {
        profile: Some(profile),
        overload: Some(overload),
        ..BindingSpec::default()
    }];
    AppliedSpec::validate(&document).expect("the document validates")
}

#[test]
fn a_generation_of_another_mode_judges_every_engine_under_its_own_block() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let a = member(&store, "http://a:8000");
    // On file before any threshold.
    store.record(a, 0, pressure(5, 0), t0, LoadSource::Stream);
    assert_eq!(overloaded(&store, t0), []);
    // A generation of another mode derives the membership again, and the
    // block it brings is the one every engine is judged under.
    let apply = |number: u64, mode: Pairing, overload: OverloadSpec| {
        store
            .apply(generation_of(number, mode, overload))
            .expect("applied");
        overloaded(&store, t0)
    };
    assert_eq!(apply(1, Pairing::Strict, block(5, 1.0)), [a]);
    // Another mode again with a higher count: the verdict is taken back,
    // and the count the record reaches raises it once more.
    assert_eq!(apply(2, Pairing::Off, block(6, 1.0)), []);
    assert_eq!(apply(3, Pairing::Strict, block(5, 1.0)), [a]);
    // A mode's change that switches the block off clears the verdict.
    let off = OverloadSpec {
        enabled: false,
        ..block(1, 0.1)
    };
    assert_eq!(apply(4, Pairing::Lenient, off), []);
    assert_eq!(
        store.overload_flips(),
        0,
        "no write of the load table moved one"
    );
}
