//! The store's answer to a collector that fetches loads on demand alone
//! ([`LoadSink::wants_loads`]): a reader wants them where a stage of the
//! standing generation reads a load, or a load write is judged.

use std::time::{Duration, Instant};

use super::{block, facts, generation, generation_with, load, member, rank_of, MODEL};
use crate::{
    spec::{BindingSpec, OverloadSpec},
    store::{LoadSink, LoadSource, MemoryStore, Overload, WorkerFacts, WorkersStore},
    view::Role,
    AppliedSpec, PolicySpec,
};

const ENGINE: &str = "grpc://e:9000";
const URL: &str = "http://a:8000";

/// The preset `name` as the generation `number`, with its binding's
/// overload block switched off: no pool has a threshold.
fn unprotected(name: &str, number: u64) -> AppliedSpec {
    let mut document = PolicySpec::preset(name).expect("a preset of this build");
    document.generation = number;
    let profile = document.profiles.keys().next().expect("a profile");
    document.bindings = vec![BindingSpec {
        profile: Some(profile.clone()),
        overload: Some(OverloadSpec {
            enabled: false,
            ..OverloadSpec::default()
        }),
        ..BindingSpec::default()
    }];
    AppliedSpec::validate(&document).expect("the document validates")
}

fn store_under(spec: AppliedSpec) -> MemoryStore {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    store.apply(spec).expect("a newer generation");
    store
}

#[test]
fn no_reader_wants_loads_where_no_stage_reads_one_and_no_threshold_stands() {
    let store = store_under(unprotected("round_robin", 1));
    member(&store, URL);
    assert!(!store.wants_loads());
    // A document that says nothing of overload keeps its block on.
    store.apply(generation(2)).expect("a newer generation");
    assert!(store.wants_loads());
}

#[test]
fn a_stage_that_reads_a_load_wants_them_under_its_generation() {
    let reads = unprotected("power_of_two", 1);
    assert!(reads.needs().loads, "the preset's scorer reads a load");
    let store = store_under(reads);
    member(&store, URL);
    assert!(store.wants_loads());
    let next = unprotected("round_robin", 2);
    store.apply(next).expect("a newer generation");
    assert!(!store.wants_loads());
}

#[test]
fn a_pools_threshold_wants_loads_while_an_engine_stands_under_it() {
    let store = store_under(generation_with(1, block(4, 0.9)));
    assert!(!store.wants_loads(), "no engine to judge");
    let id = member(&store, URL);
    assert!(store.wants_loads());
    let revision = WorkersStore::get(&store, id).expect("the row").revision;
    store.remove(id, revision).expect("removed");
    store.publish_now();
    assert!(!store.wants_loads(), "the engine left");
}

#[test]
fn a_rows_own_threshold_wants_loads_from_the_build_its_write_owes() {
    let store = store_under(unprotected("round_robin", 1));
    store.upsert(WorkerFacts {
        overload: Overload {
            waiting_requests: Some(2),
            kv_usage_permille: None,
        },
        ..facts(URL, Role::Single, &[MODEL])
    });
    assert!(!store.wants_loads(), "the build is owed");
    store.publish_now();
    assert!(store.wants_loads());
}

#[test]
fn a_rank_its_engine_stops_naming_wants_loads_with_no_threshold() {
    let store = store_under(unprotected("round_robin", 1));
    let at = Instant::now();
    let zero = rank_of(&store, ENGINE, 0);
    let one = rank_of(&store, ENGINE, 1);
    let both = [(zero, 0, Some(load(1, 0))), (one, 1, Some(load(2, 0)))];
    store.report(&both, at, LoadSource::Poll);
    assert!(!store.wants_loads(), "both ranks are named");
    // The answer leaves rank zero out. Only a later answer or record names
    // it again, so the collector goes on asking.
    let one_alone = [(zero, 0, None), (one, 1, Some(load(2, 0)))];
    store.report(&one_alone, at, LoadSource::Poll);
    assert!(store.wants_loads());
}
