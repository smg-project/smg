//! Which binding's overload block judges a pool: the binding the pool is
//! decided under.

use std::time::{Duration, Instant};

use super::{ask, block, facts, pressure, MODEL};
use crate::{
    spec::{BindingSpec, Match, OverloadSpec},
    store::{world::ViewBuffers, LoadSink, LoadSource, MemoryStore, WorkersStore},
    view::{Phase, Role},
    AppliedSpec, PolicySpec,
};

/// The judge follows the decision: a pair's decode side runs under the
/// model's pair binding, so that binding's block judges the decode pool
/// though a decode-phase binding stands before it. A pair binding with its
/// block off leaves the pool unjudged, and a model the pair binding does
/// not match keeps the decode-phase binding's.
#[test]
fn a_pair_s_decode_pool_is_judged_under_the_pair_binding_s_block() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let paired = store
        .upsert(facts("http://d:8000", Role::Decode, &[MODEL]))
        .id();
    let alone = store
        .upsert(facts("http://o:8000", Role::Decode, &["n"]))
        .id();
    store.publish_now();
    // A decode-phase binding for every model with a count of 2, then the
    // pair binding of one model with `pair_block`.
    let document = |generation: u64, pair_block: OverloadSpec| {
        let mut document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("presets of this build");
        document.generation = generation;
        let plain = PolicySpec::preset("round_robin").expect("a preset of this build");
        let (_, profile) = plain
            .profiles
            .into_iter()
            .next()
            .expect("the preset's profile");
        document.profiles.insert("alone".to_string(), profile);
        let mut pair = document.bindings.remove(0);
        pair.matcher.model = MODEL.to_string();
        pair.overload = Some(pair_block);
        let decode_first = BindingSpec {
            matcher: Match {
                phase: vec![Phase::Decode],
                ..Match::default()
            },
            profile: Some("alone".to_string()),
            overload: Some(block(2, 1.0)),
            ..BindingSpec::default()
        };
        document.bindings = vec![decode_first, pair];
        AppliedSpec::validate(&document).expect("the document validates")
    };
    let over = |model: &str| {
        let mut out = ViewBuffers::default();
        store
            .world()
            .build(model, Role::Decode, &ask(t0), &mut out)
            .expect("the pool");
        out.candidates
            .iter()
            .map(|row| row.health.overloaded)
            .collect::<Vec<_>>()
    };
    store.record(paired, 0, pressure(5, 0), t0, LoadSource::Stream);
    store.record(alone, 0, pressure(5, 0), t0, LoadSource::Stream);
    // One queue of five on both: under the pair's nine, over the phase's
    // two.
    store.apply(document(1, block(9, 1.0))).expect("applied");
    assert_eq!((over(MODEL), over("n")), (vec![false], vec![true]));
    // The pair's block names a count the queue is over.
    store.apply(document(2, block(4, 1.0))).expect("applied");
    assert_eq!((over(MODEL), over("n")), (vec![true], vec![true]));
    // Switched off, it leaves the pool unjudged: the phase binding's block
    // is not read for it.
    let off = OverloadSpec {
        enabled: false,
        ..block(4, 1.0)
    };
    store.apply(document(3, off)).expect("applied");
    assert_eq!((over(MODEL), over("n")), (vec![false], vec![true]));
}
