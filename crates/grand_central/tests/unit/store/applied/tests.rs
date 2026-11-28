use std::{
    future::{poll_fn, Future},
    pin::pin,
    sync::Arc,
    task::Poll,
    time::{Duration, Instant},
};

use smg_worker::{
    identity::basic::Worker,
    registry::{event::WorkerEvent, WorkerId as ObjectId},
};
use tokio::sync::broadcast::Receiver;

use crate::{
    document::{AppliedSpec, PolicySpec},
    engine::{
        decision::Targets,
        decision_engine::{DecisionRefused, Request, RequestMaker},
        framework::{Picker, TieBand},
        in_process::{LendsLists, Loaded, WorkerLists},
        view::{Candidate, RequestFacts},
        InProcess, Members,
    },
    stages::{registry::StageRegistry, Lifecycle},
    store::{MemoryStore, WorkerFacts},
    tracking::lease::Membership,
};

/// A request whose host's own code panics where the engine makes it:
/// inside its decision.
struct Unmade;

impl RequestMaker for Unmade {
    fn model(&self) -> &str {
        "model-a"
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        panic!("the case's own panic, inside a decision")
    }
}

impl LendsLists for Unmade {
    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::None
    }
}

/// A picker of a host's own build that panics where it hears of a join.
#[derive(Debug)]
struct Deaf;

impl Lifecycle for Deaf {
    fn on_membership(&self, change: &Membership<'_>) {
        assert!(
            !matches!(change, Membership::Added(_)),
            "the case's own panic, at a word of membership"
        );
    }
}

impl Picker for Deaf {
    fn name(&self) -> &'static str {
        "deaf"
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        _: &[f32],
        _: TieBand,
        _: &mut Targets,
    ) {
    }
}

/// A registry of a host whose own code panics where the engine's task
/// asks it for its events.
struct Unread;

impl Members for Unread {
    fn events(&self) -> Receiver<WorkerEvent> {
        panic!("the case's own panic, in a step of the engine's task")
    }

    fn standing(&self) -> Vec<(ObjectId, Arc<dyn Worker>)> {
        Vec::new()
    }
}

/// The registration of one worker under `key`.
fn member(key: &str) -> WorkerFacts {
    WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation: 1,
        models: vec!["model-a".to_owned()],
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    }
}

/// One decision under `front`'s document that panics and is held:
/// refused by name.
fn decision(front: Loaded<'_, '_>) {
    let refused = front.decide_without_lease(&Unmade, Instant::now());
    assert_eq!(refused.err(), Some(DecisionRefused::Faulted));
}

/// One step of the engine's own task that panics and is held: the
/// first poll of the task, over a registry that panics. The task goes
/// on.
async fn step(engine: &InProcess) {
    engine.set_registry(Arc::new(Unread));
    let mut task = pin!(engine.run());
    let goes_on = poll_fn(|cx| Poll::Ready(task.as_mut().poll(cx).is_pending()));
    assert!(goes_on.await, "the task waits for its next tick");
}

/// The store's counts of the panics its engine held never fall. The
/// store keeps one engine for its life, built at its unbound start: a
/// count stands through an apply, and a panic held under a replaced
/// document that a reader's world still holds is counted where the
/// store reads, also behind the apply that hands that document back.
/// Were an engine built for each document to count for itself, the
/// store's read would start again at an apply.
///
/// A decision panics here in its host's own code, where the engine makes
/// the request. The word is a registration's, which the store tells the
/// engine. The task's step runs under the catch a controller's tick
/// runs the upkeep under ([`super::Applied::sweep`]).
#[tokio::test(start_paused = true)]
async fn the_panics_a_stores_engine_held_are_counted_past_their_documents() {
    let store = MemoryStore::new();
    let mut stages = StageRegistry::builtin();
    let added = stages.picker("deaf", || -> Box<dyn Picker> { Box::new(Deaf) });
    added.expect("a free name");
    let apply = |number| {
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = number;
        for profile in document.profiles.values_mut() {
            profile.pick = "deaf".into();
        }
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        let spec = spec.expect("the stages hold the picker");
        store.apply(spec).expect("above the standing one");
    };
    let held = || {
        let held = store.stats().engine_faults;
        [held.decisions, held.events, held.upkeep]
    };
    assert_eq!(held(), [0, 0, 0]);
    // Under the unbound start too.
    store.deciding(|world| decision(world.front()));
    assert_eq!(held(), [1, 0, 0]);
    apply(1);
    assert_eq!(held(), [1, 0, 0], "a document starts no count again");

    // A reader keeps the first document's world.
    let first = store.snapshot();
    let engine = &first.applied().engine;
    decision(first.front());
    store.upsert(member("http://worker-a:8000"));
    step(engine).await;
    assert_eq!(held(), [2, 1, 1]);
    apply(2);
    assert_eq!(held(), [2, 1, 1], "behind an apply");
    store.deciding(|world| decision(world.front()));
    store.upsert(member("http://worker-b:8000"));
    assert_eq!(held(), [3, 2, 1]);

    // The third apply hands the first document back: a decision under
    // the reader's world still counts where the store reads.
    apply(3);
    assert_eq!(held(), [3, 2, 1], "behind the hand-back");
    decision(first.front());
    step(engine).await;
    step(engine).await;
    assert_eq!(held(), [4, 2, 3]);
    drop(first);
    assert_eq!(held(), [4, 2, 3], "behind the reader's world");
}

/// A store's engine reads the store's own clock: an instant reads alike
/// in the store and under each document it applies. A stage reads the
/// stamp of a lease the store dated on the clock it was written on. An
/// engine that started a clock of its own would read a send of the last
/// second as one of the future, by the store's age.
#[test]
fn a_stores_engine_reads_the_stores_clock() {
    let store = MemoryStore::new();
    let at = Instant::now() + Duration::from_secs(5);
    let engines = |store: &MemoryStore| store.deciding(|world| world.front().clock().micros(at));
    for number in 1..=3 {
        let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
        document.generation = number;
        let spec = AppliedSpec::validate(&document).expect("a preset validates");
        store.apply(spec).expect("above the standing one");
        assert_eq!(
            engines(&store),
            store.micros(at),
            "under generation {number}"
        );
    }
}
