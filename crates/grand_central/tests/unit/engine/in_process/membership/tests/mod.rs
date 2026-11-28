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
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    registry::{event::WorkerEvent, Generation, WorkerId as ObjectId, WorkerRegistry},
    test_support::{no_health_check, ready_worker},
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
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision_engine::DecisionEngine,
        in_process::{
            single::tests::{decided, hearing, pool, request, Asks, Heard},
            InProcess,
        },
        list_cell::PublishedList,
        prompt_hashes::HashSeeds,
        records::worker_id,
        view::{FixedFacts, WorkerId},
    },
};

pub(in crate::engine::in_process) mod fills;

const MODEL: &str = "model-a";

/// The plainest generation of a registry: a list's members and nothing
/// beside them.
pub(in crate::engine::in_process) struct Listed(Arc<[Arc<dyn Worker>]>);

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

pub(in crate::engine::in_process) type Registry = WorkerRegistry<Listed>;

/// What the stages were told: the word and the worker's id.
type Words = Vec<(&'static str, WorkerId)>;

fn url(at: usize) -> String {
    format!("http://worker-{at}:8000")
}

/// The fingerprint of the address `url(at)`: the id of the worker there
/// where its host states none.
pub(in crate::engine::in_process) fn id(at: usize) -> WorkerId {
    worker_id(ready_worker(&url(at), MODEL).as_ref())
}

/// A registry in which the workers at `url(0)` to `url(workers - 1)` stand.
pub(in crate::engine::in_process) fn registry(workers: usize) -> Arc<Registry> {
    let registry = Arc::new(Registry::new());
    for at in 0..workers {
        register(&registry, at);
    }
    registry
}

/// Registers a worker at `url(at)`, which holds none.
pub(in crate::engine::in_process) fn register(registry: &Registry, at: usize) {
    let registered = registry.register(ready_worker(&url(at), MODEL));
    registered.expect("a free address");
}

/// The registry's own id of the worker at `url(at)`.
fn object(registry: &Registry, at: usize) -> ObjectId {
    registry.get_id_by_url(&url(at)).expect("a worker there")
}

/// Removes the worker at `url(at)` from the registry.
pub(in crate::engine::in_process) fn remove(registry: &Registry, at: usize) {
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
pub(in crate::engine::in_process) fn in_id_order(mut words: Words) -> Words {
    words.sort_by_key(|&(word, id)| (id.0, word));
    words
}

/// What the stages were told of single workers since the last call, in the
/// order told: the joins and the leaves, without a whole read's snapshot.
pub(in crate::engine::in_process) fn told(heard: &Heard) -> Words {
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
    sorted(follow.told.values().map(|held| held.id).collect())
}

/// Every word that waits on the follow's subscription, told as the
/// engine's task tells it.
fn drain(engine: &InProcess, follow: &mut Follow) {
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

/// The registry's word of a registration of `worker` under its id
/// `object`, as the engine's task hears it.
fn registered(object: &ObjectId, worker: &Arc<dyn Worker>) -> WorkerEvent {
    WorkerEvent::Registered {
        worker_id: object.clone(),
        worker: Arc::clone(worker),
    }
}

/// The registry's word of the removal of `worker`, which it knew as
/// `object`.
fn removed(object: &ObjectId, worker: &Arc<dyn Worker>) -> WorkerEvent {
    WorkerEvent::Removed {
        worker_id: object.clone(),
        worker: Arc::clone(worker),
    }
}

/// The registry's word that `new` replaces `old` under its id `object`.
fn replaced(object: &ObjectId, old: &Arc<dyn Worker>, new: &Arc<dyn Worker>) -> WorkerEvent {
    WorkerEvent::Replaced {
        worker_id: object.clone(),
        old: Arc::clone(old),
        new: Arc::clone(new),
    }
}

/// `registry` as a host hands it to the engine.
pub(in crate::engine::in_process) fn handed(
    registry: &Arc<impl Members + 'static>,
) -> Arc<dyn Members> {
    Arc::clone(registry) as Arc<dyn Members>
}

/// The follow the engine's task starts at an upkeep: none of an engine
/// handed no registry.
fn started(engine: &InProcess) -> Option<Follow> {
    let mut members = None;
    engine.follow_registry(&mut members);
    members
}

/// An engine that keeps what its stages are told, handed `registry`, and
/// the follow its task would start.
fn following(registry: &Arc<impl Members + 'static>) -> (InProcess, Heard, Follow) {
    let (engine, heard) = hearing(false);
    engine.set_registry(handed(registry));
    let follow = started(&engine).expect("a registry was handed");
    (engine, heard, follow)
}

/// The engine's task, spawned as a host spawns it.
fn running(engine: &Arc<InProcess>) -> JoinHandle<()> {
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
/// the worker at each address. `panics` makes its answer panic, and its
/// hand-over of a subscription.
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
        assert!(!self.panics.load(Ordering::Relaxed), "the host's own code");
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
    // A later hand-over is ignored.
    engine.set_registry(handed(&Arc::new(Registry::new())));

    register(&registry, 2);
    remove(&registry, 0);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("joined", id(2)), ("left", id(0))]);
    assert!(
        !follow.unread,
        "a word of a host that states no id owes no read"
    );

    remove(&registry, 1);
    register(&registry, 1);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("left", id(1)), ("joined", id(1))]);
}

/// The registry's replace of a worker's object at its address tells the
/// stages nothing, and neither does a change of a worker's status: the
/// worker stands under its id, and the follow still holds it. Classic's
/// policies hear a removal in the removal workflow alone
/// (`model_gateway/src/workflow/steps/local/remove_from_policy_registry.rs`).
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
    assert!(
        !follow.unread,
        "a replacement of a worker whose host states no id owes no read"
    );

    // The worker is still followed: its removal is told.
    remove(&registry, 0);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("left", id(0))]);
}

/// A worker replaced under its id by a registration of another model
/// serves the first model no more: the stages are told the model it left,
/// and a stage that lists workers by model unlists it there. The worker
/// stands, so no leave of its id is told.
#[test]
fn a_replacement_that_takes_a_model_tells_the_stages_the_model_left() {
    let registry = registry(2);
    let (engine, heard, mut follow) = following(&registry);
    let _ = told(&heard);

    let standing = object(&registry, 0);
    assert!(registry.replace(&standing, ready_worker(&url(0), "model-b")));
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("left a model", id(0))]);
    assert_eq!(held(&follow), sorted(vec![id(0), id(1)]));
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
    // A read the follow owed already is this one.
    follow.unread = true;
    drain(&engine, &mut follow);
    let moved = vec![("left", id(0)), ("joined", id(2))];
    assert_eq!(in_id_order(told(&heard)), in_id_order(moved));
    assert_eq!(held(&follow), sorted(vec![id(1), id(2)]));
    assert!(!follow.unread, "the lag's read is the read owed");
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
    let (earlier, later) = (ready_worker(&url(0), MODEL), ready_worker(&url(0), MODEL));

    assert!(engine.tells(&mut follow, Ok(registered(&first, &earlier))));
    assert_eq!(told(&heard), [("joined", id(0))]);

    assert!(engine.tells(&mut follow, Ok(registered(&next, &later))));
    assert_eq!(told(&heard), [("left", id(0)), ("joined", id(0))]);

    assert!(engine.tells(&mut follow, Ok(removed(&first, &earlier))));
    assert_eq!(told(&heard), [], "the id's next life stands");

    assert!(engine.tells(&mut follow, Ok(removed(&next, &later))));
    assert_eq!(told(&heard), [("left", id(0))]);
    assert_eq!(held(&follow), []);
}

/// A replacement names the object it replaces. One of an object the follow
/// does not hold tells nothing and asks the host nothing: it is older than
/// a read that found a later object, or its worker's removal was heard
/// ahead of it. So a host that states another id by then is told no leave
/// of the id that stands.
#[test]
fn a_replacement_of_an_object_the_follow_does_not_hold_tells_nothing() {
    let registry = registry(0);
    let host = Stating::over(&registry);
    host.state(0, 7);
    let (engine, heard, mut follow) = following(&host);
    let object = ObjectId::new();
    let worker = || ready_worker(&url(0), MODEL);
    let (earlier, standing, later) = (worker(), worker(), worker());

    // No worker is held under the registry's id.
    assert!(engine.tells(&mut follow, Ok(replaced(&object, &earlier, &later))));
    assert_eq!(told(&heard), []);
    assert_eq!(held(&follow), []);

    assert!(engine.tells(&mut follow, Ok(registered(&object, &standing))));
    assert_eq!(told(&heard), [("joined", WorkerId(7))]);

    // The host states another id by now, and the word is of another object.
    host.state(0, 8);
    assert!(engine.tells(&mut follow, Ok(replaced(&object, &earlier, &later))));
    assert_eq!(told(&heard), [], "the worker held stands under its id");
    assert_eq!(held(&follow), [WorkerId(7)]);

    assert!(engine.tells(&mut follow, Ok(replaced(&object, &standing, &later))));
    let rekeyed = [("left", WorkerId(7)), ("joined", WorkerId(8))];
    assert_eq!(told(&heard), rekeyed, "the replacement of the object held");
}

/// A replacement inherits its worker's state. Where the follow never heard
/// the replacement, the worker's removal names an object the follow does
/// not hold: it is the worker held all the same, by that state, and the
/// removal is its leave. The removal of a worker with a state of its own
/// is another life's, and tells nothing.
#[test]
fn the_removal_of_a_replacement_the_follow_never_heard_is_the_workers_leave() {
    let registry = registry(0);
    let (engine, heard, mut follow) = following(&registry);
    let object = ObjectId::new();
    let worker = || ready_worker(&url(0), MODEL);
    let (first, second, stranger) = (worker(), worker(), worker());
    let inherited = second.inherit_shared_state_from(first.as_ref());
    assert!(inherited, "a replacement takes its worker's state");

    assert!(engine.tells(&mut follow, Ok(registered(&object, &first))));
    assert_eq!(told(&heard), [("joined", id(0))]);

    assert!(engine.tells(&mut follow, Ok(removed(&object, &stranger))));
    assert_eq!(told(&heard), [], "another life's removal");

    assert!(engine.tells(&mut follow, Ok(removed(&object, &second))));
    assert_eq!(told(&heard), [("left", id(0))]);
    assert_eq!(held(&follow), []);
}

/// A registry can say a registration under its own id of a worker ahead of
/// that worker's removal: it frees the id ahead of the removal's word.
/// Another object registered under a registry id the follow holds is the
/// worker's next life, a leave and a join. The earlier object's removal
/// then tells nothing and the worker stands in the follow: a removal tells
/// a leave only of the object held. The object held, registered again,
/// tells nothing.
#[test]
fn a_registry_id_registered_again_ahead_of_its_removal_is_the_workers_next_life() {
    let registry = registry(0);
    let (engine, heard, mut follow) = following(&registry);
    let object = ObjectId::new();
    let (earlier, later) = (ready_worker(&url(0), MODEL), ready_worker(&url(0), MODEL));

    assert!(engine.tells(&mut follow, Ok(registered(&object, &earlier))));
    assert_eq!(told(&heard), [("joined", id(0))]);

    assert!(engine.tells(&mut follow, Ok(registered(&object, &later))));
    assert_eq!(told(&heard), [("left", id(0)), ("joined", id(0))]);
    assert!(engine.tells(&mut follow, Ok(registered(&object, &later))));
    assert_eq!(told(&heard), [], "the object held, said again");

    assert!(engine.tells(&mut follow, Ok(removed(&object, &earlier))));
    assert_eq!(told(&heard), [], "the removal of the earlier life");
    assert_eq!(held(&follow), [id(0)]);

    assert!(engine.tells(&mut follow, Ok(removed(&object, &later))));
    assert_eq!(told(&heard), [("left", id(0))]);
    assert_eq!(held(&follow), []);
}

/// The registration of a worker a whole read found is heard behind the
/// read where the channel kept it. It names the object the follow holds,
/// so it tells nothing, and the worker's own removal is then its leave.
#[test]
fn a_registration_heard_behind_the_read_that_found_its_worker_tells_nothing() {
    let registry = registry(1);
    let (engine, heard, mut follow) = following(&registry);
    let _ = told(&heard);

    // The channel keeps its newest events: the registration, behind the flood.
    flood(&registry, 0);
    register(&registry, 1);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("joined", id(1))], "told by the read alone");

    remove(&registry, 1);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("left", id(1))]);
    assert_eq!(held(&follow), [id(0)]);
}

/// A worker the registry replaced inside the dropped events keeps its
/// state, which a replacement inherits: the whole read finds another
/// object under the registry id the follow holds, knows the worker by that
/// state, and tells nothing. Its removal behind the read is its leave.
#[test]
fn a_worker_replaced_inside_dropped_events_is_told_nothing_by_the_read() {
    let registry = registry(2);
    let (engine, heard, mut follow) = following(&registry);
    let _ = told(&heard);

    replace(&registry, 0);
    flood(&registry, 1);
    drain(&engine, &mut follow);
    assert_eq!(
        told(&heard),
        [],
        "the worker stands, with what the stages keep"
    );
    assert_eq!(held(&follow), sorted(vec![id(0), id(1)]));

    remove(&registry, 0);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("left", id(0))]);
}

/// A host that keeps ids by address answers of the worker that stands, so
/// an id that stood only between two replacements heard together is told
/// to no stage. A replacement of a host that states ids owes a whole read
/// at the next upkeep, with no lag: its snapshot names the ids that stand,
/// and a stage drops what requests left under any other.
#[test]
fn a_replacement_that_changes_the_id_told_owes_a_whole_read() {
    let registry = registry(1);
    let host = Stating::over(&registry);
    host.state(0, 7);
    let (engine, heard, follow) = following(&host);
    let mut members = Some(follow);
    let _ = told(&heard);

    // The host stated 8 and then 9 of the worker ahead of the first word.
    replace(&registry, 0);
    replace(&registry, 0);
    host.state(0, 9);
    drain(&engine, members.as_mut().expect("the follow"));
    let rekeyed = [("left", WorkerId(7)), ("joined", WorkerId(9))];
    assert_eq!(told_and_standing(&heard), (rekeyed.to_vec(), vec![]));

    engine.follow_registry(&mut members);
    assert_eq!(told_and_standing(&heard), (vec![], vec![WorkerId(9)]));

    // No read is owed behind it.
    engine.follow_registry(&mut members);
    assert_eq!(told_and_standing(&heard), (vec![], vec![]));
}

/// A host that keeps ids by address can state another id of a worker and
/// the first again between two replacements heard together: both words
/// name the id told, so the stages are told nothing, and what requests
/// left under the id between would stay. A replacement of a worker whose
/// id its host states owes a whole read at the next upkeep, with no lag:
/// its snapshot names the ids that stand.
#[test]
fn a_replacement_under_a_stated_id_that_stands_owes_a_whole_read() {
    let registry = registry(1);
    let host = Stating::over(&registry);
    host.state(0, 7);
    let (engine, heard, follow) = following(&host);
    let mut members = Some(follow);
    let _ = told(&heard);

    // The host states 8 and then 7 again of the worker ahead of the first
    // word.
    host.state(0, 8);
    replace(&registry, 0);
    host.state(0, 7);
    replace(&registry, 0);
    drain(&engine, members.as_mut().expect("the follow"));
    assert_eq!(told_and_standing(&heard), (vec![], vec![]));

    engine.follow_registry(&mut members);
    assert_eq!(told_and_standing(&heard), (vec![], vec![WorkerId(7)]));
}

/// A host that keeps ids by address can state another id of a worker and
/// take every statement of it back ahead of the word: the word then names
/// the fingerprint of the worker's address, and what requests left under
/// the id between would stay. The follow knows the host as one that
/// states ids since the read at its start, so the replacement owes a
/// whole read at the next upkeep: its snapshot names the ids that stand.
#[test]
fn a_replacement_that_changes_the_id_told_to_one_not_stated_owes_a_whole_read() {
    let registry = registry(1);
    let host = Stating::over(&registry);
    host.state(0, 7);
    let (engine, heard, follow) = following(&host);
    let mut members = Some(follow);
    let _ = told(&heard);

    // The host states 8 of the worker, and then nothing, ahead of the
    // word.
    host.state(0, 8);
    replace(&registry, 0);
    host.ids.lock().remove(&url(0));
    drain(&engine, members.as_mut().expect("the follow"));
    let rekeyed = [("left", WorkerId(7)), ("joined", id(0))];
    assert_eq!(told_and_standing(&heard), (rekeyed.to_vec(), vec![]));

    engine.follow_registry(&mut members);
    assert_eq!(told_and_standing(&heard), (vec![], vec![id(0)]));
}

/// A host that keeps ids by address can state an id of a worker and take
/// it back between the registry's write and the follow's ask, at a
/// registration and at a removal as at a replacement: the word then names
/// another id, and what requests left under the stated one would stay. So
/// every word of membership of a host that states ids owes a whole read
/// at the next upkeep: its snapshot names the ids that stand. A change of
/// status is no such word, and owes none.
#[test]
fn a_registration_and_a_removal_under_a_host_that_states_ids_owe_a_whole_read() {
    let registry = registry(1);
    let host = Stating::over(&registry);
    host.state(0, 7);
    let (engine, heard, follow) = following(&host);
    let mut members = Some(follow);
    let _ = told(&heard);

    // The host states nothing of the worker it registers, at the ask.
    register(&registry, 1);
    drain(&engine, members.as_mut().expect("the follow"));
    assert_eq!(told_and_standing(&heard), (vec![("joined", id(1))], vec![]));
    engine.follow_registry(&mut members);
    let both = sorted(vec![WorkerId(7), id(1)]);
    assert_eq!(told_and_standing(&heard), (vec![], both));

    remove(&registry, 1);
    drain(&engine, members.as_mut().expect("the follow"));
    assert_eq!(told_and_standing(&heard), (vec![("left", id(1))], vec![]));
    engine.follow_registry(&mut members);
    assert_eq!(told_and_standing(&heard), (vec![], vec![WorkerId(7)]));

    let moved = registry.transition_status(&object(&registry, 0), WorkerStatus::NotReady);
    assert!(moved.is_some());
    drain(&engine, members.as_mut().expect("the follow"));
    assert!(
        !members.as_ref().is_some_and(|follow| follow.unread),
        "a change of status owes no read"
    );
}

/// The follow knows a host as one that states ids from the first ask the
/// host answers with one, and the word of that ask owes the read too. A
/// host that has stated nothing owes none at a registration. Then it
/// states an id of a second worker and registers it: the follow owes a
/// whole read behind that very word.
#[test]
fn the_word_whose_ask_first_answers_with_an_id_owes_a_whole_read() {
    let registry = registry(0);
    let host = Stating::over(&registry);
    let (engine, heard, mut follow) = following(&host);
    let _ = told(&heard);

    register(&registry, 0);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("joined", id(0))]);
    assert!(!follow.unread, "a host that has stated no id owes no read");

    host.state(1, 9);
    register(&registry, 1);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("joined", WorkerId(9))]);
    assert!(
        follow.unread,
        "the word whose ask first answers with an id owes the read"
    );
}

/// Two workers stand under one stated id, and the host gives one of them
/// another: the id still names the other worker, so no leave of it is
/// told. The other worker's removal is the id's leave.
#[test]
fn a_worker_given_another_id_leaves_the_id_it_shared_standing() {
    let registry = registry(2);
    let host = Stating::over(&registry);
    host.state(0, 7);
    host.state(1, 7);
    let (engine, heard, mut follow) = following(&host);
    let one_id = WorkerId(7);
    let shared = [("joined", one_id), ("left", one_id), ("joined", one_id)];
    assert_eq!(
        told(&heard),
        shared,
        "the second worker is the id's next life"
    );

    host.state(0, 8);
    replace(&registry, 0);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("joined", WorkerId(8))]);

    remove(&registry, 1);
    drain(&engine, &mut follow);
    assert_eq!(told(&heard), [("left", one_id)]);
    assert_eq!(held(&follow), [WorkerId(8)]);
}

/// The records of sent prefixes forget a worker's id where the follow tells
/// its join with no leave ahead, and where it tells its leave: a prompt
/// the records name a worker for is then placed by its request's seed
/// again. A worker so inherits nothing of a life the channel dropped whole.
/// The lists name eight workers by their addresses, whatever the registry
/// holds.
#[test]
fn a_join_and_a_leave_of_the_follow_forget_the_ids_sent_prefixes() {
    let mut document = PolicySpec::preset("cache_aware").expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = InProcess::new(spec);
    let registry = registry(0);
    engine.set_registry(handed(&registry));
    let mut follow = started(&engine).expect("a registry was handed");
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let sent = |prompt: u32, seed: u64| {
        let tokens: Vec<u32> = (prompt * 100_000..prompt * 100_000 + 1_024).collect();
        let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.seed = seed;
        hashes.write_to(&mut asked.facts);
        let asks = Asks::of(asked, Some(&members));
        let decided = decided(&engine, &asks);
        let pick = decided.decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position
    };
    // Sixteen prompts, each asked under a seed of its own.
    let places = |seeds: u64| -> Vec<u32> {
        (0..16)
            .map(|prompt| sent(prompt, seeds + u64::from(prompt)))
            .collect()
    };
    let first = places(0);
    assert_eq!(places(100), first, "a prompt stays where it was sent");

    // Eight joins no leave came ahead of.
    for at in 0..8 {
        register(&registry, at);
    }
    drain(&engine, &mut follow);
    let second = places(200);
    assert_ne!(second, first, "a join starts with no sent prefixes");
    assert_eq!(places(300), second);

    for at in 0..8 {
        remove(&registry, at);
    }
    drain(&engine, &mut follow);
    assert_ne!(places(400), second, "a leave forgets the sent prefixes");
}

/// A whole read tells the records of sent prefixes the ids that stand. A
/// worker that was sent prefixes and stands no more is forgotten there,
/// though the follow never told its leave: a prompt the records name it
/// for is placed by its request's seed again. A worker that stands keeps
/// what it was sent.
#[test]
fn a_whole_read_forgets_the_sent_prefixes_of_an_id_that_stands_no_more() {
    let mut document = PolicySpec::preset("cache_aware").expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = InProcess::new(spec);
    let registry = registry(0);
    engine.set_registry(handed(&registry));
    let mut follow = started(&engine).expect("a registry was handed");
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let sent = |prompt: u32, seed: u64| {
        let tokens: Vec<u32> = (prompt * 100_000..prompt * 100_000 + 1_024).collect();
        let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.seed = seed;
        hashes.write_to(&mut asked.facts);
        let asks = Asks::of(asked, Some(&members));
        let decided = decided(&engine, &asks);
        let pick = decided.decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position
    };
    let places = |seeds: u64| -> Vec<u32> {
        (0..16)
            .map(|prompt| sent(prompt, seeds + u64::from(prompt)))
            .collect()
    };
    let first = places(0);
    assert_eq!(places(100), first, "a prompt stays where it was sent");

    // None of the eight stands in the registry.
    engine.heard(Some(&mut follow), Err(RecvError::Lagged(1)));
    let second = places(200);
    assert_ne!(
        second, first,
        "the read forgets a worker that stands no more"
    );

    // All eight stand: the read forgets none.
    for at in 0..8 {
        register(&registry, at);
    }
    drain(&engine, &mut follow);
    let third = places(300);
    engine.heard(Some(&mut follow), Err(RecvError::Lagged(1)));
    assert_eq!(places(400), third, "the read keeps a worker that stands");
}

/// A worker replaced under its id by a registration of another model
/// serves the first model no more: the records of sent prefixes forget it,
/// as at a leave. A prompt of the first model the records named it for is
/// placed by its request's seed again. A replacement that keeps the model
/// keeps what the worker was sent. One the channel dropped is judged the
/// same way by the whole read that finds its object. The lists name eight
/// workers by their addresses, whatever the registry holds.
#[test]
fn a_replacement_that_takes_a_model_forgets_the_workers_sent_prefixes() {
    let mut document = PolicySpec::preset("cache_aware").expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = InProcess::new(spec);
    let registry = registry(8);
    engine.set_registry(handed(&registry));
    let mut follow = started(&engine).expect("a registry was handed");
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let sent = |prompt: u32, seed: u64| {
        let tokens: Vec<u32> = (prompt * 100_000..prompt * 100_000 + 1_024).collect();
        let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.seed = seed;
        hashes.write_to(&mut asked.facts);
        let asks = Asks::of(asked, Some(&members));
        let decided = decided(&engine, &asks);
        let pick = decided.decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position
    };
    let places = |seeds: u64| -> Vec<u32> {
        (0..16)
            .map(|prompt| sent(prompt, seeds + u64::from(prompt)))
            .collect()
    };
    let first = places(0);
    assert_eq!(places(100), first, "a prompt stays where it was sent");

    // Each worker is replaced under its id by a registration of the same
    // model: the records keep what it was sent.
    for at in 0..8 {
        replace(&registry, at);
    }
    drain(&engine, &mut follow);
    assert_eq!(
        places(150),
        first,
        "a replacement that keeps the model keeps its sent prefixes"
    );

    // Each worker registers another model under its id.
    for at in 0..8 {
        let standing = object(&registry, at);
        assert!(registry.replace(&standing, ready_worker(&url(at), "model-b")));
    }
    drain(&engine, &mut follow);
    assert_ne!(
        places(200),
        first,
        "a worker that left the model keeps no sent prefix of it"
    );

    // Each worker serves the first model again. Then the channel drops its
    // replacement, and a whole read finds the new object: one of the model
    // held keeps what the worker was sent, one of another model does not.
    for at in 0..8 {
        replace(&registry, at);
    }
    drain(&engine, &mut follow);
    let second = places(250);
    for (model, kept) in [(MODEL, true), ("model-b", false)] {
        for at in 0..8 {
            let standing = object(&registry, at);
            assert!(registry.replace(&standing, ready_worker(&url(at), model)));
        }
        flood(&registry, 0);
        drain(&engine, &mut follow);
        assert_eq!(places(300) == second, kept, "found serving {model}");
    }
}

/// A list keeps its record of sent prefixes under the model its first
/// worker states. Where every worker is replaced under its id by one that
/// states another model and still answers to the first by an alias, no
/// list reads the first record again: it leaves at the task's sweep with
/// its places. A record whose model a worker still states stays where its
/// list is published again and no request follows.
#[tokio::test(start_paused = true)]
async fn a_record_leaves_once_no_worker_states_its_model() {
    let mut document = PolicySpec::preset("cache_aware").expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = Arc::new(InProcess::new(spec));
    let registry = registry(2);
    engine.set_registry(handed(&registry));
    let task = running(&engine);
    settle().await;
    let sent = |list: &PublishedList| {
        for prompt in 0..16u32 {
            let tokens: Vec<u32> = (prompt * 100_000..prompt * 100_000 + 1_024).collect();
            let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            asked.facts.seed = u64::from(prompt);
            hashes.write_to(&mut asked.facts);
            let decided = decided(&engine, &Asks::of(asked, Some(list)));
            assert!(decided.decision.pick().is_some());
        }
    };
    let models = || {
        let held = engine.affinity.held();
        let mut models: Vec<Box<str>> = held.into_iter().map(|(model, _)| model).collect();
        models.sort();
        models
    };
    let list = PublishedList::new(pool(2));
    sent(&list);
    drop(list);
    sleep(Duration::from_secs(1)).await;
    assert_eq!(models(), [Box::from(MODEL)], "its workers state the model");

    let renamed: Arc<[Arc<dyn Worker>]> = (0..2)
        .map(|at| {
            let card = ModelCard::new("model-b").with_alias(MODEL);
            let built = <BasicWorkerBuilder>::new(url(at)).models(vec![card]);
            Arc::new(built.health_config(no_health_check()).build()) as Arc<dyn Worker>
        })
        .collect();
    assert!(registry.replace(&object(&registry, 0), Arc::clone(&renamed[0])));
    sleep(Duration::from_secs(1)).await;
    assert_eq!(models(), [Box::from(MODEL)], "one worker states it yet");
    assert!(registry.replace(&object(&registry, 1), Arc::clone(&renamed[1])));
    let list = PublishedList::new(Arc::clone(&renamed));
    sent(&list);
    sleep(Duration::from_secs(1)).await;
    assert_eq!(
        models(),
        [Box::from("model-b")],
        "no list reads the first record"
    );
    task.abort();
}

/// A list keeps the record of a model its workers list behind their first
/// under that model's own id. No worker states that id as its first model,
/// and the record stays all the same while a worker that stands lists the
/// model: the follow asks each standing worker for the id of every card it
/// lists, where no list holds the record and another worker joins.
#[tokio::test(start_paused = true)]
async fn a_record_of_a_model_listed_behind_the_first_stays() {
    const SHARED: &str = "model-s";
    let mut document = PolicySpec::preset("cache_aware").expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = Arc::new(InProcess::new(spec));
    let registry = registry(2);
    engine.set_registry(handed(&registry));
    let task = running(&engine);
    settle().await;
    // Each worker lists the shared model behind a first model of its own.
    let shared: Arc<[Arc<dyn Worker>]> = [MODEL, "model-b"]
        .into_iter()
        .enumerate()
        .map(|(at, first)| {
            let cards = vec![ModelCard::new(first), ModelCard::new(SHARED)];
            let built = <BasicWorkerBuilder>::new(url(at)).models(cards);
            Arc::new(built.health_config(no_health_check()).build()) as Arc<dyn Worker>
        })
        .collect();
    for at in 0..2 {
        assert!(registry.replace(&object(&registry, at), Arc::clone(&shared[at])));
    }
    let list = PublishedList::new(Arc::clone(&shared));
    for prompt in 0..16u32 {
        let tokens: Vec<u32> = (prompt * 100_000..prompt * 100_000 + 1_024).collect();
        let hashes = engine.prompt_hashes(SHARED, Some(&tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.model = SHARED;
        asked.facts.seed = u64::from(prompt);
        hashes.write_to(&mut asked.facts);
        let decided = decided(&engine, &Asks::of(asked, Some(&list)));
        assert!(decided.decision.pick().is_some());
    }
    drop(list);
    register(&registry, 2);
    sleep(Duration::from_secs(1)).await;
    let held = engine.affinity.held();
    let models: Vec<Box<str>> = held.into_iter().map(|(model, _)| model).collect();
    assert_eq!(
        models,
        [Box::from(SHARED)],
        "a worker that stands lists the model"
    );
    task.abort();
}

/// While no list holds the record of a model listed behind the first, the
/// upkeep asks the registry of it once: from then on it asks the worker
/// found listing the model alone, and the registry again behind a word of
/// membership. A card that worker drops in place is read at the next
/// upkeep: the registry is asked once more, and the record leaves where no
/// worker that stands lists the model.
#[test]
fn the_registry_is_asked_once_of_a_listed_model_no_list_holds() {
    /// A host that counts how often its standing workers are asked for.
    struct Counting {
        registry: Arc<Registry>,
        asks: AtomicUsize,
    }
    impl Members for Counting {
        fn events(&self) -> Receiver<WorkerEvent> {
            self.registry.subscribe_events()
        }

        fn standing(&self) -> Vec<(ObjectId, Arc<dyn Worker>)> {
            self.asks.fetch_add(1, Ordering::Relaxed);
            self.registry.get_all_with_ids()
        }
    }
    const SHARED: &str = "model-s";
    let mut document = PolicySpec::preset("cache_aware").expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = InProcess::new(spec);
    let registry = registry(2);
    let host = Arc::new(Counting {
        registry: Arc::clone(&registry),
        asks: AtomicUsize::new(0),
    });
    engine.set_registry(handed(&host));
    let mut members = started(&engine);
    // Each worker lists the shared model behind a first model of its own.
    let firsts = [MODEL, "model-b"];
    let shared: Arc<[Arc<dyn Worker>]> = firsts
        .into_iter()
        .enumerate()
        .map(|(at, first)| {
            let cards = vec![ModelCard::new(first), ModelCard::new(SHARED)];
            let built = <BasicWorkerBuilder>::new(url(at)).models(cards);
            Arc::new(built.health_config(no_health_check()).build()) as Arc<dyn Worker>
        })
        .collect();
    for at in 0..2 {
        assert!(registry.replace(&object(&registry, at), Arc::clone(&shared[at])));
    }
    engine.follow_so_far(&mut members);
    let list = PublishedList::new(Arc::clone(&shared));
    for prompt in 0..16u32 {
        let tokens: Vec<u32> = (prompt * 100_000..prompt * 100_000 + 1_024).collect();
        let hashes = engine.prompt_hashes(SHARED, Some(&tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.model = SHARED;
        asked.facts.seed = u64::from(prompt);
        hashes.write_to(&mut asked.facts);
        let decided = decided(&engine, &Asks::of(asked, Some(&list)));
        assert!(decided.decision.pick().is_some());
    }
    drop(list);
    let records = || engine.affinity.held().len();
    let asked = || host.asks.load(Ordering::Relaxed);
    let before = asked();
    for _ in 0..3 {
        engine.follow_registry(&mut members);
    }
    assert_eq!(
        asked() - before,
        1,
        "the registry is asked once of the record"
    );
    assert_eq!(records(), 1, "a worker that stands lists the model");
    register(&registry, 2);
    engine.follow_so_far(&mut members);
    for _ in 0..2 {
        engine.follow_registry(&mut members);
    }
    assert_eq!(
        asked() - before,
        2,
        "and once more behind a word of membership"
    );
    assert_eq!(records(), 1, "the worker asked anew lists the model");
    for (at, first) in firsts.into_iter().enumerate() {
        assert!(registry.refresh_models(&shared[at], vec![ModelCard::new(first)]));
    }
    engine.follow_registry(&mut members);
    assert_eq!(asked() - before, 3, "a card dropped in place is read");
    assert_eq!(records(), 0, "no worker that stands lists the model");
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

/// A panic of the host's own code is held and counted, and the task goes
/// on: where the follow takes its subscription, which it takes again at
/// the next upkeep, and at a word. The word is lost, so the task reads the
/// registry whole at its next upkeep, and at the one behind it where the
/// host's code panics again. Once the host answers, the worker whose
/// registration was lost is told as joined, and no other worker is told
/// again.
#[tokio::test(start_paused = true)]
async fn a_panic_of_the_hosts_code_does_not_end_the_engines_task() {
    let registry = registry(1);
    let host = Stating::over(&registry);
    host.state(0, 7);
    let (engine, heard) = hearing(false);
    let engine = Arc::new(engine);
    host.panics.store(true, Ordering::Relaxed);
    engine.set_registry(handed(&host));
    let task = running(&engine);
    settle().await;
    assert_eq!(told(&heard), [], "the host handed no subscription");
    assert_eq!(engine.upkeep_faults(), 1);
    assert!(!task.is_finished(), "the task goes on");

    host.panics.store(false, Ordering::Relaxed);
    sleep(Duration::from_secs(1)).await;
    assert_eq!(told(&heard), [("joined", WorkerId(7))]);
    assert_eq!(engine.upkeep_faults(), 1);

    host.panics.store(true, Ordering::Relaxed);
    register(&registry, 1);
    settle().await;
    assert_eq!(told(&heard), [], "the host's answer panicked");
    assert_eq!(engine.upkeep_faults(), 2);

    // The read owed at the next upkeep panics too, and is owed again.
    sleep(Duration::from_secs(1)).await;
    assert_eq!(engine.upkeep_faults(), 3);
    assert!(!task.is_finished(), "the task goes on");

    host.panics.store(false, Ordering::Relaxed);
    host.state(1, 8);
    sleep(Duration::from_secs(1)).await;
    assert_eq!(told(&heard), [("joined", WorkerId(8))]);
    assert_eq!(engine.upkeep_faults(), 3);
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
