//! The cases of a generation that moves the pairing mode and the overload
//! block at once: the membership is derived again under the new mode, and
//! every engine is judged under the new block by the same apply.

use std::time::{Duration, Instant};

use super::{
    ask, block, facts, member, overloaded, pressure, rank_of, view, MODEL, PUBLISH_WINDOW,
    REPORT_PERIOD,
};
use crate::{
    spec::{BindingSpec, OverloadSpec, Pairing},
    store::{LoadSink, LoadSource, MemoryStore, WorkersStore},
    view::Role,
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

/// A generation of another mode derives the membership again, and the
/// write that does reads each rank's receipt from its sample on file: a
/// rank that fell silent long before the generation is not counted under
/// its block, and is counted again once it is heard.
#[test]
fn a_generation_of_another_mode_ages_each_sample_from_its_own_receipt() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let at = |secs: u64| t0 + Duration::from_secs(secs);
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    assert!(store.heard_every(REPORT_PERIOD));
    // No threshold stands yet. One rank falls silent with a queue on file,
    // and the other is heard long after it.
    store.record(one, 1, pressure(9, 0), at(0), LoadSource::Stream);
    store.record(zero, 0, pressure(0, 0), at(0), LoadSource::Stream);
    member(&store, "http://b:8000");
    store.record(zero, 0, pressure(0, 0), at(100), LoadSource::Stream);
    assert_eq!(overloaded(&store, at(100)), []);
    // The generation's block is met by the silent rank's queue alone, and
    // that count ended long ago.
    store
        .apply(generation_of(1, Pairing::Strict, block(8, 1.0)))
        .expect("applied");
    assert_eq!(overloaded(&store, at(100)), []);
    store.record(one, 1, pressure(9, 0), at(101), LoadSource::Stream);
    assert_eq!(overloaded(&store, at(101)), [zero, one]);
}

/// A generation of another mode builds the membership as it stands, at the
/// store's epoch: a registration whose build was owed is in the world the
/// apply publishes, and nothing is left for the window's end to build.
#[test]
fn a_generation_of_another_mode_pays_the_build_a_registration_owes() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    member(&store, "http://a:8000");
    store.upsert(facts("http://b:8000", Role::Single, &[MODEL]));
    let builds = store.membership_builds();
    assert!(store.publish_due().is_some());
    assert!(store.world().epoch() < WorkersStore::epoch(&store));
    store
        .apply(generation_of(1, Pairing::Strict, block(8, 1.0)))
        .expect("applied");
    assert_eq!(store.world().epoch(), WorkersStore::epoch(&store));
    assert_eq!(view(&store, &ask(t0)).0.len(), 2);
    assert_eq!(
        (store.membership_builds(), store.publish_due()),
        (builds + 1, None)
    );
    store.publish(t0 + PUBLISH_WINDOW);
    assert_eq!(store.membership_builds(), builds + 1);
}
