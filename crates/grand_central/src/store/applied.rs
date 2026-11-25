//! The document a store publishes its worlds under, as its host decides by
//! it.
//!
//! A host that decides over worker lists decides by the engine's in-process
//! front ([`InProcess`]), which owns its document and the state of the
//! document's stages. A store tells those stages every word of a lease it
//! files: its issue, its dispatch and its end. So a build that holds that
//! front keeps a store's applied document as that front's engine
//! ([`Applied`]). One engine object stands for a document: the stages that
//! hear of a lease are the ones that decided it, and the document is
//! validated once.
//!
//! A new document is a new engine object, built by the front's constructor.
//! Its session table and its record of sent prompts start empty.
//!
//! The front a store builds runs no task of its own. The store tells it
//! each word of its membership ([`Applied::on_membership`]), and the store's
//! host runs its upkeep on a tick of its own ([`Applied::sweep`]).
//!
//! Every engine a store builds reads the store's own clock: the store hands
//! it its origin. An apply hands a profile's count of sends from the
//! standing engine to the next, and a lease is dated where the store reads
//! its file: one instant reads alike under every document and in the store.
//!
//! Every engine a store builds counts the panics it holds into the store's
//! one set of counts ([`EngineCounts`]): a count outlasts the document it
//! rose under, and a panic held by an engine that a decision still holds
//! past its document is counted where the store reads.

#[cfg(feature = "in-process")]
use std::sync::Arc;
use std::{ops::Deref, time::Instant};

#[cfg(feature = "in-process")]
use crate::engine::in_process::{HeldFaults, InProcess, SpecHandle};
#[cfg(not(feature = "in-process"))]
use crate::stages::Lifecycle;
use crate::{document::spec::AppliedSpec, tracking::lease::Membership};

/// The panics the engines of one store held under their own catches, as
/// one read of the store answers them ([`MemoryStore::stats`]). They are
/// counted over every engine the store built, whichever document each
/// decided under, and none starts again at a new document. All three are
/// zero in a build without the in-process front, whose store builds no
/// such engine.
///
/// [`MemoryStore::stats`]: crate::store::MemoryStore::stats
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct EngineFaults {
    /// Whole decisions that panicked, and shadow profiles that did beside
    /// a live decision that stood.
    pub decisions: u64,
    /// Words to the stages that panicked in a stage that heard them, under
    /// the engine's catch: a store's words of membership. A lease's words
    /// are told under their host's catch and counted there
    /// ([`MemoryStore::lifecycle_faults`](crate::store::MemoryStore::lifecycle_faults)).
    pub events: u64,
    /// Runs of the engine's upkeep that panicked.
    pub upkeep: u64,
}

/// The counts every engine of one store counts its held panics into: the
/// store makes them once and hands them to each engine it builds
/// ([`Applied::new`]). Nothing where the build holds no in-process front.
#[derive(Debug, Default)]
pub(crate) struct EngineCounts {
    #[cfg(feature = "in-process")]
    held: Arc<HeldFaults>,
}

impl EngineCounts {
    /// The three counts as they stand, of every engine handed them.
    #[cfg_attr(
        not(feature = "in-process"),
        expect(
            clippy::unused_self,
            reason = "a build without the front holds no count"
        )
    )]
    pub(crate) fn read(&self) -> EngineFaults {
        #[cfg(feature = "in-process")]
        let [decisions, events, upkeep] = self.held.read();
        #[cfg(not(feature = "in-process"))]
        let [decisions, events, upkeep] = [0; 3];
        EngineFaults {
            decisions,
            events,
            upkeep,
        }
    }
}

/// One applied document as its store holds it: the engine that decides
/// under it, where the build holds the in-process front, and the document
/// alone otherwise. It reads as the document ([`Deref`]).
#[derive(Debug)]
pub(crate) struct Applied {
    #[cfg(feature = "in-process")]
    engine: InProcess,
    /// The engine's document, held for as long as the engine: a store
    /// puts no second document in force on it.
    #[cfg(feature = "in-process")]
    spec: SpecHandle,
    #[cfg(not(feature = "in-process"))]
    spec: AppliedSpec,
}

impl Applied {
    /// `spec` as a store holds it from its apply on. `origin` is the origin
    /// of the store's clock: the engine that decides under the document
    /// reads that clock. `counts` are the store's: the engine counts the
    /// panics it holds there.
    pub(crate) fn new(spec: AppliedSpec, origin: Instant, counts: &EngineCounts) -> Self {
        #[cfg(feature = "in-process")]
        let applied = {
            let engine = InProcess::with_origin(spec, origin, Arc::clone(&counts.held));
            let spec = engine.spec();
            Self { engine, spec }
        };
        #[cfg(not(feature = "in-process"))]
        let applied = {
            // A build without the front keeps no clock beside the store's,
            // and builds no engine that counts.
            let _ = (origin, counts);
            Self { spec }
        };
        applied
    }

    /// The front that decides under the document: the one engine object
    /// whose stages hear the store's words of a lease.
    #[cfg(feature = "in-process")]
    #[cfg_attr(
        not(any(feature = "controller", test)),
        expect(dead_code, reason = "a controller's deciding thread asks it")
    )]
    pub(crate) fn front(&self) -> &InProcess {
        &self.engine
    }

    /// One word of the store's membership to the engine that decides under
    /// the document, said under the store's lock of its workers. Where the
    /// build holds the front, the front hears it: its records of sent
    /// prefixes, then the document's engine
    /// ([`InProcess::on_membership`]). The front holds a stage's panic
    /// at the word and counts it ([`InProcess::unheard`]): the store's
    /// write goes on. Else the document's engine hears it alone, under no
    /// catch.
    pub(crate) fn on_membership(&self, word: &Membership<'_>) {
        #[cfg(feature = "in-process")]
        self.engine.on_membership(word);
        #[cfg(not(feature = "in-process"))]
        self.spec.engine().on_membership(word);
    }

    /// The upkeep of the engine that decides under the document, at `now`,
    /// under the engine's own catch ([`InProcess::upkeep`]): the session
    /// keys past their end leave its table, and its records of sent prefixes
    /// take a step of the work they left waiting. A controller asks it at
    /// each pass of its presence sweep, a tenth of the grace period: once a
    /// second, the period of the engine's own task in a gateway's process.
    #[cfg(feature = "in-process")]
    #[cfg_attr(
        not(feature = "controller"),
        expect(dead_code, reason = "a controller's tick asks it")
    )]
    pub(crate) fn sweep(&self, now: Instant) {
        self.engine.upkeep(now);
    }
}

impl Deref for Applied {
    type Target = AppliedSpec;

    fn deref(&self) -> &AppliedSpec {
        &self.spec
    }
}

// The cases drive a store's engines, which stand under the feature
// `in-process`. The feature's gate is an attribute of its own: the lints
// read a module of cases by its plain `cfg(test)`.
#[cfg(test)]
#[cfg(feature = "in-process")]
mod tests {
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
            decision_engine::{DecisionRefused, Request, RequestMaker, WorkerLists},
            framework::{Picker, TieBand},
            view::{Candidate, RequestFacts},
            InProcess, Members,
        },
        stages::{registry::StageRegistry, Lifecycle},
        store::{MemoryStore, WorkerFacts, WorkersStore},
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

    /// One decision of `front` that panics and is held: refused by name.
    fn decision(front: &InProcess) {
        let refused = front.decide_without_lease(&Unmade, Instant::now());
        assert_eq!(refused.err(), Some(DecisionRefused::Faulted));
    }

    /// One step of `front`'s own task that panics and is held: the first
    /// poll of the task, over a registry that panics. The task goes on.
    async fn step(front: &InProcess) {
        front.set_registry(Arc::new(Unread));
        let mut task = pin!(front.run());
        let goes_on = poll_fn(|cx| Poll::Ready(task.as_mut().poll(cx).is_pending()));
        assert!(goes_on.await, "the task waits for its next tick");
    }

    /// The store's counts of the panics its engines held never fall. Every
    /// engine the store builds counts into the store's one set: the unbound
    /// start's and each document's. A count stands through an apply, and a
    /// panic held by a replaced engine that a reader's world still holds is
    /// counted where the store reads, also behind the apply that hands its
    /// document back. Were each engine to count for itself, the store's read
    /// would start again at an apply, or leave out an engine that neither
    /// stands nor was let go.
    ///
    /// A decision panics here in its host's own code, where the engine makes
    /// the request. The word is a registration's, which the store tells the
    /// engine that stands. The task's step runs under the catch a
    /// controller's tick runs the upkeep under ([`super::Applied::sweep`]).
    #[tokio::test(start_paused = true)]
    async fn the_panics_a_stores_engines_held_are_counted_past_their_documents() {
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
        // The engine of the unbound start counts there too.
        store.deciding(|world| decision(world.front()));
        assert_eq!(held(), [1, 0, 0]);
        apply(1);
        assert_eq!(held(), [1, 0, 0], "a document starts no count again");

        // A reader keeps the first document's world.
        let first = store.snapshot();
        decision(first.front());
        store.upsert(member("http://worker-a:8000"));
        step(first.front()).await;
        assert_eq!(held(), [2, 1, 1]);
        apply(2);
        assert_eq!(held(), [2, 1, 1], "behind an apply");
        store.deciding(|world| decision(world.front()));
        store.upsert(member("http://worker-b:8000"));
        assert_eq!(held(), [3, 2, 1]);

        // The third apply hands the first document back: its engine goes
        // on with the reader's world, and still counts where the store
        // reads.
        apply(3);
        assert_eq!(held(), [3, 2, 1], "behind the hand-back");
        decision(first.front());
        step(first.front()).await;
        step(first.front()).await;
        assert_eq!(held(), [4, 2, 3]);
        drop(first);
        assert_eq!(held(), [4, 2, 3], "behind the reader's world");
    }

    /// Every engine a store builds reads the store's own clock: an instant
    /// reads alike in the store and under each document it applies. An apply
    /// hands a profile's count of sends to the next document's engine, which
    /// reads their stamps on the clock they were written on. An engine that
    /// started a clock of its own would read a send of the last second as one
    /// of the future, by the standing document's age.
    #[test]
    fn every_engine_a_store_builds_reads_the_stores_clock() {
        let store = MemoryStore::new();
        let at = Instant::now() + Duration::from_secs(5);
        let engines =
            |store: &MemoryStore| store.deciding(|world| world.front().clock().micros(at));
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
}
