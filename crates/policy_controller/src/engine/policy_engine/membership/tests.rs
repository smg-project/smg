//! The cases of the engine's follow of a registry: each over the worker
//! crate's own registry, whose events the follow reads as the engine's task
//! reads them, with a picker that keeps what the stages are told.

use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use smg_worker::{
    identity::basic::Worker,
    registry::{event::WorkerEvent, Generation, WorkerId as ObjectId, WorkerRegistry},
    test_support::ready_worker,
    worker::WorkerStatus,
};
use tokio::{
    sync::broadcast::{
        self,
        error::{RecvError, TryRecvError},
        Receiver, Sender,
    },
    task::JoinHandle,
    time::sleep,
};

use super::{Follow, Members};
use crate::engine::{
    decision_engine::DecisionEngine,
    policy_engine::{
        single::tests::{hearing, Heard},
        PolicyEngine,
    },
    records::worker_id,
    view::{FixedFacts, WorkerId},
};

const MODEL: &str = "model-a";

/// The plainest generation of a registry: a list's members and nothing
/// beside them.
struct Listed(Arc<[Arc<dyn Worker>]>);

impl fmt::Debug for Listed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Listed").field(&self.0.len()).finish()
    }
}

impl Generation for Listed {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        Self(members)
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.0
    }
}

type Registry = WorkerRegistry<Listed>;

/// What the stages were told: the word and the worker's id.
type Words = Vec<(&'static str, WorkerId)>;

fn url(at: usize) -> String {
    format!("http://worker-{at}:8000")
}

/// The fingerprint of the address `url(at)`: the id of the worker there
/// where its host states none.
fn id(at: usize) -> WorkerId {
    worker_id(ready_worker(&url(at), MODEL).as_ref())
}

/// A registry in which the workers at `url(0)` to `url(workers - 1)` stand.
fn registry(workers: usize) -> Arc<Registry> {
    let registry = Arc::new(Registry::new());
    for at in 0..workers {
        register(&registry, at);
    }
    registry
}

/// Registers a worker at `url(at)`, which holds none.
fn register(registry: &Registry, at: usize) {
    let registered = registry.register(ready_worker(&url(at), MODEL));
    registered.expect("a free address");
}

/// The registry's own id of the worker at `url(at)`.
fn object(registry: &Registry, at: usize) -> ObjectId {
    registry.get_id_by_url(&url(at)).expect("a worker there")
}

/// Removes the worker at `url(at)` from the registry.
fn remove(registry: &Registry, at: usize) {
    let removed = registry.remove(&object(registry, at));
    removed.expect("the worker was removed");
}

/// Replaces the object of the worker at `url(at)` under its id.
fn replace(registry: &Registry, at: usize) {
    let standing = object(registry, at);
    assert!(registry.replace(&standing, ready_worker(&url(at), MODEL)));
}

/// More changes of the status of the worker at `url(at)` than the
/// registry's channel holds: a reader that waited through them lost the
/// events ahead of them.
fn flood(registry: &Registry, at: usize) {
    let mut witness = registry.subscribe_events();
    let standing = object(registry, at);
    for turn in 0..1_100 {
        let status = if turn % 2 == 0 {
            WorkerStatus::NotReady
        } else {
            WorkerStatus::Ready
        };
        let moved = registry.transition_status(&standing, status);
        assert!(moved.is_some(), "turn {turn}");
    }
    let dropped = matches!(witness.try_recv(), Err(TryRecvError::Lagged(_)));
    assert!(dropped, "the channel dropped its oldest events");
}

/// `words` in the order of their ids: a whole read tells in no order.
fn in_id_order(mut words: Words) -> Words {
    words.sort_by_key(|&(word, id)| (id.0, word));
    words
}

/// What the stages were told of single workers since the last call, in the
/// order told: the joins and the leaves, without a whole read's snapshot.
fn told(heard: &Heard) -> Words {
    told_and_standing(heard).0
}

/// What the stages were told since the last call: the joins and the leaves
/// in the order told, and the ids its snapshots named, in their order.
fn told_and_standing(heard: &Heard) -> (Words, Vec<WorkerId>) {
    let words = std::mem::take(&mut *heard.lock());
    let words = words.into_iter().map(|(word, id, _)| (word, id));
    let (standing, single): (Words, Words) = words.partition(|&(word, _)| word == "standing");
    let standing = standing.into_iter().map(|(_, id)| id).collect();
    (single, sorted(standing))
}

/// `ids` in their order.
fn sorted(mut ids: Vec<WorkerId>) -> Vec<WorkerId> {
    ids.sort_unstable();
    ids
}

/// The ids the follow holds as told, in their order.
fn held(follow: &Follow) -> Vec<WorkerId> {
    sorted(follow.told.values().copied().collect())
}

/// Every word that waits on the follow's subscription, told as the
/// engine's task tells it.
fn drain(engine: &PolicyEngine, follow: &mut Follow) {
    loop {
        let events = follow.events.as_mut().expect("a subscription");
        let word = match events.try_recv() {
            Ok(event) => Ok(event),
            Err(TryRecvError::Lagged(skipped)) => Err(RecvError::Lagged(skipped)),
            Err(TryRecvError::Closed) => Err(RecvError::Closed),
            Err(TryRecvError::Empty) => return,
        };
        if !engine.tells(follow, word) {
            return;
        }
    }
}

/// `registry` as a host hands it to the engine.
fn handed(registry: &Arc<impl Members + 'static>) -> Arc<dyn Members> {
    Arc::clone(registry) as Arc<dyn Members>
}

/// The follow the engine's task starts at an upkeep: none of an engine
/// handed no registry.
fn started(engine: &PolicyEngine) -> Option<Follow> {
    let mut members = None;
    engine.follow_registry(&mut members);
    members
}

/// An engine that keeps what its stages are told, handed `registry`, and
/// the follow its task would start.
fn following(registry: &Arc<impl Members + 'static>) -> (PolicyEngine, Heard, Follow) {
    let (engine, heard) = hearing(false);
    engine.set_registry(handed(registry));
    let follow = started(&engine).expect("a registry was handed");
    (engine, heard, follow)
}

/// The engine's task, spawned as a host spawns it.
fn running(engine: &Arc<PolicyEngine>) -> JoinHandle<()> {
    let engine = Arc::clone(engine);
    #[expect(
        clippy::disallowed_methods,
        reason = "the case holds the task's handle and ends the task"
    )]
    tokio::spawn(async move { engine.run().await })
}

/// Lets the task take what stands ready for it: no second passes.
async fn settle() {
    sleep(Duration::from_millis(1)).await;
}

/// A host that keeps ids of its own: its registry, and the id it states of
/// the worker at each address. `panics` makes its answer panic.
struct Stating {
    registry: Arc<Registry>,
    ids: Mutex<FxHashMap<String, WorkerId>>,
    panics: AtomicBool,
}

impl Stating {
    fn over(registry: &Arc<Registry>) -> Arc<Self> {
        Arc::new(Self {
            registry: Arc::clone(registry),
            ids: Mutex::default(),
            panics: AtomicBool::new(false),
        })
    }

    /// States `id` of the worker at `url(at)` from here on.
    fn state(&self, at: usize, id: u64) {
        self.ids.lock().insert(url(at), WorkerId(id));
    }
}

impl Members for Stating {
    fn events(&self) -> Receiver<WorkerEvent> {
        self.registry.subscribe_events()
    }

    fn standing(&self) -> Vec<(ObjectId, Arc<dyn Worker>)> {
        self.registry.get_all_with_ids()
    }

    fn facts_of(&self, worker: &Arc<dyn Worker>) -> FixedFacts {
        assert!(!self.panics.load(Ordering::Relaxed), "the host's own code");
        FixedFacts {
            id: self.ids.lock().get(worker.url()).copied(),
            ..FixedFacts::default()
        }
    }
}

/// The follow starts from the workers that stand, each told as joined, and
/// tells the stages each registration as a join and each removal as a
/// leave, in the registry's order. A worker removed and another registered
/// at its address are a leave and then a join under one id. An engine
/// handed no registry follows none.
#[test]
fn a_registration_and_a_removal_of_the_registry_are_told_to_the_stages() {
    let (unhanded, _) = hearing(false);
    assert!(started(&unhanded).is_none());

    let registry = registry(2);
    let (engine, heard, mut follow) = following(&registry);
    let stand = vec![("joined", id(0)), ("joined", id(1))];
    assert_eq!(in_id_order(told(&heard)), in_id_order(stand));

    register(&registry, 2);
    remove(&registry, 0);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("joined", id(2)), ("left", id(0))]);

    remove(&registry, 1);
    register(&registry, 1);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("left", id(1)), ("joined", id(1))]);
}

/// The registry's replace of a worker's object at its address tells the
/// stages nothing, and neither does a change of a worker's status: the
/// worker stands under its id, and the follow still holds it. Classic's
/// policies hear a removal in the removal workflow alone
/// (`model_gateway/src/workflow/steps/local/remove_from_policy_registry.rs`),
/// and nothing at a replacement.
#[test]
fn a_replaced_object_and_a_status_change_tell_the_stages_nothing() {
    let registry = registry(2);
    let (engine, heard, mut follow) = following(&registry);
    let _ = told(&heard);

    replace(&registry, 0);
    let moved = registry.transition_status(&object(&registry, 0), WorkerStatus::NotReady);
    assert!(moved.is_some());
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), []);
    assert_eq!(held(&follow), sorted(vec![id(0), id(1)]));

    // The worker is still followed: its removal is told.
    remove(&registry, 0);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("left", id(0))]);
}

/// A host that states ids of its own: a worker is told under the id its
/// host states, not the fingerprint of its address. A replacement under
/// which the host states another id is a leave of the id told and a join
/// of the new one: the registry keeps its own id of a worker whose host
/// gives it another life. A replacement under which the stated id stands
/// tells nothing. A removal is told under the id told at the join: the
/// host is not asked of a worker that is gone.
#[test]
fn a_replacement_that_changes_the_stated_id_is_a_leave_and_a_join() {
    let registry = registry(1);
    let host = Stating::over(&registry);
    host.state(0, 7);
    let (engine, heard, mut follow) = following(&host);
    assert_eq!(told(&heard), [("joined", WorkerId(7))]);

    host.state(0, 8);
    replace(&registry, 0);
    drain(&engine, &mut follow);
    let another_life = [("left", WorkerId(7)), ("joined", WorkerId(8))];
    assert_eq!(told(&heard), another_life);

    replace(&registry, 0);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), []);

    // The host knows the worker no more by the time its removal is heard.
    host.ids.lock().clear();
    remove(&registry, 0);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("left", WorkerId(8))]);
    assert_eq!(held(&follow), []);
}

/// A follow that fell behind reads the registry whole: the channel dropped
/// the removal of one worker and the registration of another, and the
/// stages are still told that leave and that join, once each. The events
/// the channel kept say nothing (changes of status), so nothing else is
/// told.
#[test]
fn a_follow_that_fell_behind_reads_the_registry_whole() {
    let registry = registry(2);
    let (engine, heard, mut follow) = following(&registry);
    let _ = told(&heard);

    remove(&registry, 0);
    register(&registry, 2);
    flood(&registry, 1);
    drain(&engine, &mut follow);
    let moved = vec![("left", id(0)), ("joined", id(2))];
    assert_eq!(in_id_order(told(&heard)), in_id_order(moved));
    assert_eq!(held(&follow), sorted(vec![id(1), id(2)]));
}

/// The events the channel kept come behind a whole read and may be older
/// than it. A worker registered and removed at an address ahead of the
/// read, where the read found the next one, is heard behind the read: its
/// registration reads as the id's next life, a leave and a join, and its
/// removal tells no leave of the id the standing worker is held under.
#[test]
fn a_removal_heard_behind_a_whole_read_tells_no_leave_of_a_standing_worker() {
    let registry = registry(1);
    let (engine, heard, mut follow) = following(&registry);
    let _ = told(&heard);

    // The channel keeps its newest events: these three, behind the flood.
    flood(&registry, 0);
    register(&registry, 1);
    remove(&registry, 1);
    register(&registry, 1);
    drain(&engine, &mut follow);
    let read_then_heard = [("joined", id(1)), ("left", id(1)), ("joined", id(1))];
    assert_eq!(told(&heard), read_then_heard);
    assert_eq!(held(&follow), sorted(vec![id(0), id(1)]));
}

/// A worker whose whole life the channel dropped is in no list of a whole
/// read: the follow never held it, and it stands no more. The read tells
/// the stages the ids that stand, so a stage drops what a request left
/// under any other, and tells nothing else. The follow's first read tells
/// them too.
#[test]
fn a_whole_read_tells_the_stages_the_ids_that_stand() {
    let registry = registry(2);
    let (engine, heard, mut follow) = following(&registry);
    let (joined, standing) = told_and_standing(&heard);
    assert_eq!(joined.len(), 2);
    assert_eq!(standing, sorted(vec![id(0), id(1)]), "at its start");

    register(&registry, 2);
    remove(&registry, 2);
    flood(&registry, 1);
    drain(&engine, &mut follow);
    let (single, standing) = told_and_standing(&heard);
    assert_eq!(single, [], "the follow never held the worker");
    assert_eq!(standing, sorted(vec![id(0), id(1)]));
}

/// A registry can say a registration at an address ahead of the removal it
/// follows. A worker registered under an id another is held under is the
/// id's next life: the stages are told a leave of the id and then its
/// join, so they keep nothing of the earlier life, and the earlier
/// worker's removal then tells nothing. The removal of the last worker
/// under the id is its leave.
#[test]
fn a_registration_heard_ahead_of_the_removal_it_follows_is_the_ids_next_life() {
    let registry = registry(0);
    let (engine, heard, mut follow) = following(&registry);
    let (first, next) = (ObjectId::new(), ObjectId::new());
    let registered = |object: &ObjectId| WorkerEvent::Registered {
        worker_id: object.clone(),
        worker: ready_worker(&url(0), MODEL),
    };
    let removed = |object: &ObjectId| WorkerEvent::Removed {
        worker_id: object.clone(),
        worker: ready_worker(&url(0), MODEL),
    };

    assert!(engine.tells(&mut follow, Ok(registered(&first))));
    assert_eq!(told(&heard), [("joined", id(0))]);

    assert!(engine.tells(&mut follow, Ok(registered(&next))));
    assert_eq!(told(&heard), [("left", id(0)), ("joined", id(0))]);

    assert!(engine.tells(&mut follow, Ok(removed(&first))));
    assert_eq!(told(&heard), [], "the id's next life stands");

    assert!(engine.tells(&mut follow, Ok(removed(&next))));
    assert_eq!(told(&heard), [("left", id(0))]);
    assert_eq!(held(&follow), []);
}

/// The engine's own task follows the registry its host handed: the workers
/// that stand are told at its first upkeep, a removal and a registration
/// behind it as they come, with no call of the host. A registry handed
/// behind the spawn is followed within a second.
#[tokio::test(start_paused = true)]
async fn the_engines_task_tells_the_stages_what_the_registry_says() {
    let (engine, heard) = hearing(false);
    let engine = Arc::new(engine);
    let registry = registry(2);
    let task = running(&engine);
    settle().await;
    assert_eq!(told(&heard), [], "no registry is handed yet");

    engine.set_registry(handed(&registry));
    sleep(Duration::from_secs(1)).await;
    let stand = vec![("joined", id(0)), ("joined", id(1))];
    assert_eq!(in_id_order(told(&heard)), in_id_order(stand));

    remove(&registry, 0);
    register(&registry, 2);
    settle().await;
    assert_eq!(told(&heard), [("left", id(0)), ("joined", id(2))]);
    task.abort();
}

/// A panic of the host's own code at a word is held and counted: the task
/// goes on. The word is lost, so the task reads the registry whole at its
/// next upkeep, and at the one behind it where the host's code panics
/// again. Once the host answers, the worker whose registration was lost is
/// told as joined, and no other worker is told again.
#[tokio::test(start_paused = true)]
async fn a_panic_of_the_hosts_code_does_not_end_the_engines_task() {
    let registry = registry(1);
    let host = Stating::over(&registry);
    host.state(0, 7);
    let (engine, heard) = hearing(false);
    let engine = Arc::new(engine);
    engine.set_registry(handed(&host));
    let task = running(&engine);
    settle().await;
    assert_eq!(told(&heard), [("joined", WorkerId(7))]);

    host.panics.store(true, Ordering::Relaxed);
    register(&registry, 1);
    settle().await;
    assert_eq!(told(&heard), [], "the host's answer panicked");
    assert_eq!(engine.upkeep_faults(), 1);

    // The read owed at the next upkeep panics too, and is owed again.
    sleep(Duration::from_secs(1)).await;
    assert_eq!(engine.upkeep_faults(), 2);
    assert!(!task.is_finished(), "the task goes on");

    host.panics.store(false, Ordering::Relaxed);
    host.state(1, 8);
    sleep(Duration::from_secs(1)).await;
    assert_eq!(told(&heard), [("joined", WorkerId(8))]);
    assert_eq!(engine.upkeep_faults(), 2);
    assert_eq!(engine.faults(), 0, "no decision panicked");
    task.abort();
}

/// A host whose channel the case closes: the registry it is read whole
/// from, and the sender of the channel it handed last. It says no event:
/// its follow learns by whole reads alone. `shut` makes every channel it
/// hands a closed one.
struct Closing {
    registry: Arc<Registry>,
    sender: Mutex<Option<Sender<WorkerEvent>>>,
    shut: AtomicBool,
    subscriptions: AtomicUsize,
}

impl Closing {
    fn over(registry: &Arc<Registry>) -> Arc<Self> {
        Arc::new(Self {
            registry: Arc::clone(registry),
            sender: Mutex::default(),
            shut: AtomicBool::new(false),
            subscriptions: AtomicUsize::new(0),
        })
    }

    /// Closes the channel handed last.
    fn close(&self) {
        drop(self.sender.lock().take());
    }

    fn subscriptions(&self) -> usize {
        self.subscriptions.load(Ordering::Relaxed)
    }
}

impl Members for Closing {
    fn events(&self) -> Receiver<WorkerEvent> {
        self.subscriptions.fetch_add(1, Ordering::Relaxed);
        let (sender, events) = broadcast::channel(8);
        if !self.shut.load(Ordering::Relaxed) {
            *self.sender.lock() = Some(sender);
        }
        events
    }

    fn standing(&self) -> Vec<(ObjectId, Arc<dyn Worker>)> {
        self.registry.get_all_with_ids()
    }
}

/// A channel that closed takes nothing the follow holds. The close itself
/// tells the stages nothing. At its next upkeep the task subscribes again
/// and reads the registry whole: a worker removed and one registered
/// through the gap are told, and the worker that stood through it is not
/// told again. A host that hands closed channels is read whole once a
/// second, tells nothing more, and counts no fault.
#[tokio::test(start_paused = true)]
async fn a_closed_channel_is_taken_again_and_no_standing_worker_is_told_twice() {
    let registry = registry(2);
    let host = Closing::over(&registry);
    let (engine, heard) = hearing(false);
    let engine = Arc::new(engine);
    engine.set_registry(handed(&host));
    let task = running(&engine);
    settle().await;
    let stand = vec![("joined", id(0)), ("joined", id(1))];
    assert_eq!(in_id_order(told(&heard)), in_id_order(stand));
    assert_eq!(host.subscriptions(), 1);

    remove(&registry, 0);
    register(&registry, 2);
    host.close();
    settle().await;
    assert_eq!(told(&heard), [], "the close tells nothing");
    assert_eq!(host.subscriptions(), 1, "taken again at an upkeep alone");

    sleep(Duration::from_secs(1)).await;
    let moved = vec![("left", id(0)), ("joined", id(2))];
    assert_eq!(in_id_order(told(&heard)), in_id_order(moved));
    assert_eq!(host.subscriptions(), 2);

    // Every channel the host hands from here on is a closed one.
    host.shut.store(true, Ordering::Relaxed);
    host.close();
    sleep(Duration::from_secs(3)).await;
    assert_eq!(told(&heard), [], "no standing worker is told again");
    assert_eq!(host.subscriptions(), 5, "one subscription a second");
    assert_eq!(engine.upkeep_faults(), 0, "a close is no fault");
    assert!(!task.is_finished(), "the task goes on");
    task.abort();
}
