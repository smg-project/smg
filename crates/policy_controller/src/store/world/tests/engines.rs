//! The cases of a pool's engines: the list a membership build derives
//! beside the rows, each engine's rows by position, and where a decision
//! is handed it.

use std::time::Instant;

use super::{ask, facts, member, rank_among, ASKER, MODEL};
use crate::{
    engine::view::{Role, WorkerId},
    store::{GatewayId, MemoryStore, Reachability, ViewBuffers, WorkersStore},
};

const E: &str = "grpc://e:9000";
const F: &str = "grpc://f:9000";

/// The engines the lent view hands, each as its rows' workers and ranks,
/// held to the build door's list on the way.
fn listed(store: &MemoryStore) -> Vec<Vec<(WorkerId, u16)>> {
    // The build the case's writes owe, asked: the list is read at once.
    store.publish_now();
    let ask = ask(Instant::now());
    let (mut lent, mut built) = (ViewBuffers::default(), ViewBuffers::default());
    store.with_world(|world| {
        let view = world
            .view(MODEL, Role::Single, &ask, false, &mut lent)
            .expect("the pool");
        let read = world
            .build(MODEL, Role::Single, &ask, &mut built)
            .expect("the pool");
        assert_eq!(world.engines(&read), view.engines, "one list at both doors");
        assert_eq!(view.pool_facts().engines, view.engines);
        (0..view.engines.len())
            .map(|engine| {
                let rows = view.engines.rows(engine).iter();
                rows.map(|&row| {
                    let row = &view.candidates[row as usize];
                    (row.id, row.dp_rank)
                })
                .collect()
            })
            .collect()
    })
}

fn remove(store: &MemoryStore, id: WorkerId) {
    let revision = WorkersStore::get(store, id).expect("the row").revision;
    store.remove(id, revision).expect("removed");
}

#[test]
fn a_pool_of_single_rank_workers_lists_no_engine() {
    let store = MemoryStore::new();
    for key in ["http://a:8000", "http://b:8000", "http://c:8000"] {
        member(&store, key);
    }
    assert_eq!(listed(&store), Vec::<Vec<_>>::new());
}

#[test]
fn an_engines_ranks_stand_together_in_their_ranks_order_whatever_their_ids() {
    let store = MemoryStore::new();
    // The ranks register out of order and between other members, so
    // neither engine's rows stand together by id.
    let e2 = rank_among(&store, E, 2, 3);
    let alone = member(&store, "http://a:8000");
    let f1 = rank_among(&store, F, 1, 2);
    let e0 = rank_among(&store, E, 0, 3);
    let f0 = rank_among(&store, F, 0, 2);
    let e1 = rank_among(&store, E, 1, 3);
    let mut engines = listed(&store);
    assert_eq!(engines.len(), 3, "two engines and a worker alone");
    assert_eq!(engines.iter().map(Vec::len).sum::<usize>(), 6, "a row once");
    // The engines stand as their first rows do by id: the order is the
    // list's own, and a case reads them by their rows.
    engines.sort_by_key(Vec::len);
    assert_eq!(engines[0], [(alone, 0)]);
    assert_eq!(engines[1], [(f0, 0), (f1, 1)]);
    assert_eq!(engines[2], [(e0, 0), (e1, 1), (e2, 2)]);
}

#[test]
fn a_rank_alone_of_its_engine_is_an_engine_of_one_row_and_lists_nothing() {
    let store = MemoryStore::new();
    member(&store, "http://a:8000");
    let e0 = rank_among(&store, E, 0, 2);
    assert_eq!(
        listed(&store),
        Vec::<Vec<_>>::new(),
        "no engine of two rows"
    );
    // Its second rank makes the engine one of two rows, and its leaving
    // one of one again: the list follows each membership write.
    let e1 = rank_among(&store, E, 1, 2);
    assert_eq!(listed(&store).len(), 2);
    remove(&store, e0);
    assert_eq!(listed(&store), Vec::<Vec<_>>::new());
    let e0 = rank_among(&store, E, 0, 2);
    let engines = listed(&store);
    assert!(engines.contains(&vec![(e0, 0), (e1, 1)]), "{engines:?}");
}

#[test]
fn a_pool_that_holds_a_host_local_row_hands_no_engines() {
    let store = MemoryStore::new();
    rank_among(&store, E, 0, 2);
    rank_among(&store, E, 1, 2);
    assert_eq!(listed(&store).len(), 1);
    // A host-local row gives each gateway a list of its own rows: the
    // positions the engines name are no longer every gateway's.
    for gateway in [ASKER, GatewayId(2)] {
        store.upsert(crate::store::WorkerFacts {
            reachable_from: Reachability::Gateway(gateway),
            ..facts(
                &format!("ipc:///tmp/{}.ipc", gateway.0),
                Role::Single,
                &[MODEL],
            )
        });
    }
    assert_eq!(listed(&store), Vec::<Vec<_>>::new());
}

#[test]
fn an_engine_is_listed_in_each_pool_by_that_pools_rows() {
    let store = MemoryStore::new();
    // Rank 0 serves two models and rank 1 one: the engine has two rows in
    // the first model's pool and one in the second's.
    let other = "n";
    let e0 = store
        .upsert(crate::store::WorkerFacts {
            engine: E.to_string(),
            dp_rank: 0,
            dp_size: 2,
            ..facts(&format!("{E}#0"), Role::Single, &[MODEL, other])
        })
        .id();
    let e1 = rank_among(&store, E, 1, 2);
    member(&store, "http://a:8000");
    let engines = listed(&store);
    assert!(engines.contains(&vec![(e0, 0), (e1, 1)]), "{engines:?}");
    let ask = ask(Instant::now());
    let mut out = ViewBuffers::default();
    store.with_world(|world| {
        let view = world
            .view(other, Role::Single, &ask, false, &mut out)
            .expect("the second pool");
        assert_eq!(view.candidates.len(), 1);
        assert!(view.engines.is_empty(), "one row of the engine serves it");
    });
}

#[test]
fn each_pools_list_carries_the_pools_number() {
    let store = MemoryStore::new();
    // Two models, an engine of two ranks each: two pools that list one.
    let other = "n";
    for (engine, model) in [(E, MODEL), (F, other)] {
        for rank in 0..2 {
            store.upsert(crate::store::WorkerFacts {
                engine: engine.to_string(),
                dp_rank: rank,
                dp_size: 2,
                ..facts(&format!("{engine}#{rank}"), Role::Single, &[model])
            });
        }
    }
    store.publish_now();
    let ask = ask(Instant::now());
    let (mut lent, mut built) = (ViewBuffers::default(), ViewBuffers::default());
    store.with_world(|world| {
        let numbers = [MODEL, other].map(|model| {
            let view = world
                .view(model, Role::Single, &ask, false, &mut lent)
                .expect("the pool");
            assert_eq!(view.engines.len(), 1);
            assert_eq!(view.engines.pool(), view.built.pool, "the pool's index");
            let read = world
                .build(model, Role::Single, &ask, &mut built)
                .expect("the pool");
            assert_eq!(world.engines(&read).pool(), read.pool);
            view.engines.pool()
        });
        assert_ne!(numbers[0], numbers[1], "a number a pool");
    });
}
