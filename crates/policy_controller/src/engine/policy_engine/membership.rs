//! The engine follows its host's worker registry.
//!
//! A host hands the engine its registry once
//! ([`PolicyEngine::set_registry`], whose doc is the rule a host reads).
//! The engine's own task ([`PolicyEngine::run`]) then reads the registry's
//! events and tells the document's stages a worker that joined and a
//! worker that was removed, as a host did by hand
//! ([`PolicyEngine::joined`], [`PolicyEngine::forget`]).
//!
//! A worker is told under the id its host states of it
//! ([`Members::facts_of`], [`FixedFacts::id_or`]). The follow keeps the id
//! it told each worker as joined under, by the registry's own id of the
//! worker ([`Follow`]): a leave names the id told at the join, and the
//! host is not asked of a worker that is gone. Beside the id it keeps weak
//! pointers to the worker's object and to its state: each event carries
//! the object it speaks of, and the follow reads by them which life of the
//! registry's id the event is of. A replacement inherits its worker's
//! state, and a registration builds its own.
//!
//! The events are a broadcast, and a reader that falls behind loses the
//! oldest. So the follow reads the registry whole at its start and where
//! the channel says it dropped events: each worker it holds that stands no
//! more is told as at its removal, then each worker that stands as at its
//! registration or its replacement. Last the stages are told the ids that
//! stand: a worker whose whole life the channel dropped is in neither
//! list, and a stage drops there what requests left under it. The records
//! of sent prefixes forget there every id that is not among them, and an
//! id at its join, so a worker inherits nothing of a life the channel
//! dropped.
//!
//! The events the channel kept come behind that read and may be older than
//! it, and a registry can say a registration ahead of the removal it
//! follows. So a removal tells a leave only where it names the worker the
//! follow holds under the registry's id: the removal of an earlier worker,
//! heard behind a read or a registration that found the next one, tells
//! nothing. A removal tells an id as left only once the follow holds no
//! worker under it. A worker registered under an id that another worker is
//! held under is the id's next life: a leave and a join of the id, while
//! the earlier worker is still held. So is another worker registered, or
//! found by a whole read, under a registry id the follow holds.
//!
//! A channel that closed takes nothing the follow holds. The follow
//! subscribes again at the task's next upkeep and reads the registry whole
//! behind it, so a worker that stood through the gap is not told again.
//!
//! A host that states ids of its own can state one of a worker and take
//! it back ahead of the follow's ask, and the word then names another id.
//! From the first ask such a host answers with an id, the follow reads the
//! registry whole at the task's next upkeep behind every registration,
//! replacement and removal: the read ends with the ids that stand.
//!
//! A panic of the host's own code at a word is held and counted
//! ([`PolicyEngine::upkeep_faults`]). The word is lost, so the follow reads
//! the registry whole again at the task's next upkeep.

use std::{
    fmt,
    future::pending,
    ptr,
    sync::{Arc, OnceLock, Weak},
};

use rustc_hash::FxHashMap;
use smg_worker::{
    identity::basic::Worker,
    registry::{event::WorkerEvent, Generation, WorkerId as ObjectId, WorkerRegistry},
    runtime::WorkerRuntime,
};
use tokio::sync::broadcast::{error::RecvError, Receiver};

use super::PolicyEngine;
use crate::{
    engine::{
        records::worker_id,
        view::{FixedFacts, WorkerId},
    },
    tracking::lease::Membership,
};

/// A host's worker registry, as far as the engine follows it: its events,
/// the workers that stand, and what the host says of a worker beside its
/// object. The worker crate's registry is one, of a host that states no id
/// of its own.
///
/// The engine's own task calls all three. A call that waits holds back the
/// task's other work: the upkeep, and the newest load snapshot.
///
/// The follow knows one life of a worker by the object an event or
/// [`Self::standing`] hands it, and by the state that object keeps
/// ([`Worker::runtime_state`]). A host that wraps a worker in another
/// object before it hands it on must hand on that state: else a removal
/// is read as an earlier life's and tells no leave, and each whole read
/// tells every worker a leave and a join. A replacement is read by its
/// object alone: it tells something only where its `old` is the object
/// the follow was handed of the worker. A host that wraps each event
/// anew so has every replacement tell nothing, and an id a replacement
/// changes is told at the next whole read alone. A registry registers a
/// new object at each registration: an object it removed and registers
/// again is read as the life that was removed.
pub trait Members: Send + Sync {
    /// A subscription to the registry's events from this call on.
    fn events(&self) -> Receiver<WorkerEvent>;

    /// Every worker that stands, whatever its status, with the registry's
    /// own id of it.
    fn standing(&self) -> Vec<(ObjectId, Arc<dyn Worker>)>;

    /// What the host says of `worker` beside its object: what it says
    /// where a list's records are written
    /// ([`PolicyEngine::pool_members_with_fixed_facts`]). The follow reads
    /// the worker's id of it ([`FixedFacts::id_or`]), so the stages are
    /// told a worker under the id its lists name it by. The default says
    /// nothing: the id is the fingerprint of the worker's address. An id
    /// the host's lists name that this states of no standing worker is in
    /// no whole read's snapshot: the stages drop what they keep under it,
    /// a ring or a bucket fences it for a while, and the records of sent
    /// prefixes forget it.
    ///
    /// Asked on the engine's task, some time behind the registry's write:
    /// the host may have replaced the object since. A host that keeps a
    /// worker's id with its object answers of the object it is asked of,
    /// and each replacement is then told under its own id. A host that
    /// keeps the id by the worker's address answers of the worker that
    /// stands there. An id that stood only between two replacements heard
    /// together is then told to no stage. So is an id stated of a worker
    /// and taken back ahead of the word of its registration or its
    /// removal. The follow knows a host as one that states ids from the
    /// first ask this answers with one. From then on every registration,
    /// replacement and removal owes a whole read at the task's next
    /// upkeep, and its snapshot drops what requests left under such an id
    /// ([`PolicyEngine::set_registry`]). Ahead of that first answer, what
    /// requests left under an id taken back waits for the next whole read.
    fn facts_of(&self, _worker: &Arc<dyn Worker>) -> FixedFacts {
        FixedFacts::default()
    }
}

impl<G: Generation> Members for WorkerRegistry<G> {
    fn events(&self) -> Receiver<WorkerEvent> {
        self.subscribe_events()
    }

    fn standing(&self) -> Vec<(ObjectId, Arc<dyn Worker>)> {
        self.get_all_with_ids()
    }
}

/// The registry a host handed the engine ([`PolicyEngine::set_registry`]).
#[derive(Default)]
pub(super) struct Handed(OnceLock<Arc<dyn Members>>);

impl Handed {
    /// The registry the host handed: none until it hands one.
    pub(super) fn get(&self) -> Option<&Arc<dyn Members>> {
        self.0.get()
    }
}

impl fmt::Debug for Handed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handed")
            .field("registry", &self.get().is_some())
            .finish()
    }
}

/// What the follow told of one worker: the id it told the worker as joined
/// under, the worker's object, and the state the object keeps
/// ([`Worker::runtime_state`]). The pointers are weak ones: each names
/// what it points at, and keeps its address from another's use while the
/// follow holds it.
struct Held {
    id: WorkerId,
    object: Weak<dyn Worker>,
    state: Option<Weak<WorkerRuntime>>,
}

impl Held {
    fn of(id: WorkerId, worker: &Arc<dyn Worker>) -> Self {
        let state = worker.runtime_state();
        Self {
            id,
            object: Arc::downgrade(worker),
            state: state.as_ref().map(Arc::downgrade),
        }
    }

    /// Whether `worker` is the object held.
    fn holds(&self, worker: &Arc<dyn Worker>) -> bool {
        ptr::addr_eq(self.object.as_ptr(), Arc::as_ptr(worker))
    }

    /// Whether `worker` is of the life held: the object held, or one that
    /// keeps its state. A replacement inherits its worker's state, and a
    /// registration builds its own.
    fn is(&self, worker: &Arc<dyn Worker>) -> bool {
        let inherited = || match (&self.state, worker.runtime_state()) {
            (Some(held), Some(state)) => ptr::eq(held.as_ptr(), Arc::as_ptr(&state)),
            _ => false,
        };
        self.holds(worker) || inherited()
    }
}

/// The engine's follow of its host's registry: the registry, the
/// subscription, and what it told of each worker, by the registry's own id
/// of the worker. Of a worker it keeps weak pointers alone: the registry
/// holds the workers.
pub(super) struct Follow {
    registry: Arc<dyn Members>,
    /// The subscription. None at the follow's start and behind a channel
    /// that closed, until the task's next upkeep takes one.
    events: Option<Receiver<WorkerEvent>>,
    told: FxHashMap<ObjectId, Held>,
    /// How many workers the follow holds under each id it told: the ids
    /// that stand.
    under: FxHashMap<WorkerId, u32>,
    /// Whether the follow owes a whole read of the registry at the task's
    /// next upkeep: at its start, behind a channel that closed, behind a
    /// word lost to a panic of the host's own code, and behind a word of
    /// membership of a host that states ids.
    unread: bool,
    /// Whether the host stated an id of a worker at any ask so far. Such a
    /// host can state an id and take it back between the registry's write
    /// and the follow's ask.
    stating: bool,
}

impl Follow {
    /// The id the stages are told `worker` under: the one its host states,
    /// or the fingerprint of its address. A host that states one is known
    /// from here on as one that states ids.
    fn id_of(&mut self, worker: &Arc<dyn Worker>) -> WorkerId {
        let facts = self.registry.facts_of(worker);
        self.stating |= facts.id.is_some();
        facts.id_or(worker_id(worker.as_ref()))
    }
}

/// The registry's next word on the follow's subscription. Never ready
/// while the engine follows none, or its follow holds no subscription.
pub(super) async fn next(follow: Option<&mut Follow>) -> Result<WorkerEvent, RecvError> {
    match follow.and_then(|follow| follow.events.as_mut()) {
        Some(events) => events.recv().await,
        None => pending().await,
    }
}

impl PolicyEngine {
    /// Hands the engine its host's worker registry, once, at the host's
    /// start. A later call is ignored.
    ///
    /// The engine's task follows it ([`Self::run`]) and tells the
    /// document's stages a worker that joined or was removed, so the host
    /// calls neither [`Self::joined`] nor [`Self::forget`]. A worker is
    /// told under the id the host states of it ([`Members::facts_of`]),
    /// which is the id a list's records name it by:
    ///
    /// | The registry's event | The stages are told |
    /// |---|---|
    /// | `Registered` | a join |
    /// | `Removed` | a leave, under the id told at the worker's join |
    /// | `Replaced` | nothing where the host states the id told of the new object: the worker stands, with what the stages keep for it. Where it states another, a leave of the id told and a join of the new one. Nothing of an object the follow does not hold |
    /// | `StatusChanged` | nothing: a decision reads a worker's status on its object |
    ///
    /// A worker removed and another registered at its address come as a
    /// `Removed` and a `Registered`: a leave, then a join. The records of
    /// sent prefixes forget the worker's id at a leave, at a join no leave
    /// came ahead of, and behind a whole read where it stands no more. The
    /// session table is told nothing, as at the two calls.
    ///
    /// An id names one worker. A worker registered under an id that
    /// another worker is held under is read as the id's next life: a leave
    /// and a join of the id, and the earlier worker's removal then tells
    /// nothing. So is another worker registered under the registry's own
    /// id of a worker that is held. A registry can say a registration
    /// ahead of the removal it follows, and a removal tells a leave only
    /// of the worker the follow holds. A worker is the one held where it
    /// is the object held or keeps its state, which a replacement inherits
    /// and a registration builds anew ([`Worker::runtime_state`]).
    ///
    /// The events are a broadcast, and a reader that falls behind loses
    /// the oldest: the task then reads the registry whole and tells what
    /// changed, and last the ids that stand ([`Membership::Snapshot`]),
    /// so a stage keeps nothing of a worker whose whole life the channel
    /// dropped. A worker that keeps no state and was replaced inside the
    /// dropped events is told there as its next life. It reads whole
    /// at its next upkeep behind every registration, replacement and
    /// removal of a host that states ids too: such a host may have stated
    /// an id of a worker and taken it back ahead of the word, and the read
    /// drops what requests left under it. It does the same behind a
    /// channel that closed, on another
    /// subscription, once a second for as long as the host hands a closed
    /// one. The engine keeps the registry for as long as it stands, and
    /// reads it off a request's path alone.
    pub fn set_registry(&self, registry: Arc<dyn Members>) {
        // The first hand-over stands.
        drop(self.members.0.set(registry));
    }

    /// The follow's part of the task's upkeep. It starts the follow of the
    /// registry the host handed, where the task follows none yet. It takes
    /// a subscription where the follow holds none: at its start, and
    /// behind a channel that closed. Then it reads the registry whole
    /// where the follow owes a read: the subscription stands ahead of the
    /// read, so a worker that changes during the read is heard behind it.
    /// A panic of the host's own code in either step is held and counted
    /// ([`Self::upkeep_faults`]), and the step is owed again at the next
    /// upkeep.
    pub(super) fn follow_registry(&self, members: &mut Option<Follow>) {
        let follow = match members {
            Some(follow) => follow,
            None => {
                let Some(registry) = self.members.get() else {
                    return;
                };
                members.insert(Follow {
                    registry: Arc::clone(registry),
                    events: None,
                    told: FxHashMap::default(),
                    under: FxHashMap::default(),
                    unread: true,
                    stating: false,
                })
            }
        };
        if follow.events.is_none() {
            follow.events = self.held(|| follow.registry.events());
        }
        if follow.unread && follow.events.is_some() {
            let read = self.held(|| self.read_whole(follow));
            follow.unread = read.is_none();
        }
    }

    /// One word of the registry's channel, heard by the engine's task. A
    /// channel that closed takes nothing the follow holds: the follow lets
    /// the subscription go, and owes another and a whole read at the
    /// task's next upkeep. A panic of the host's own code at the word is
    /// held and counted ([`Self::upkeep_faults`]): the word is lost, so
    /// the follow owes a whole read then too.
    pub(super) fn heard(&self, follow: Option<&mut Follow>, word: Result<WorkerEvent, RecvError>) {
        let Some(follow) = follow else {
            return;
        };
        match self.held(|| self.tells(follow, word)) {
            Some(true) => {}
            Some(false) => {
                follow.events = None;
                follow.unread = true;
            }
            None => follow.unread = true,
        }
    }

    /// Tells the document's stages one word of the registry's channel, by
    /// the rule of this file. Whether the channel still stands.
    fn tells(&self, follow: &mut Follow, word: Result<WorkerEvent, RecvError>) -> bool {
        let membership = matches!(
            word,
            Ok(WorkerEvent::Registered { .. }
                | WorkerEvent::Replaced { .. }
                | WorkerEvent::Removed { .. })
        );
        match word {
            Ok(WorkerEvent::Registered { worker_id, worker }) => {
                let id = follow.id_of(&worker);
                self.stands(follow, worker_id, &worker, id, false);
            }
            Ok(WorkerEvent::Replaced {
                worker_id,
                old,
                new,
            }) => {
                // A replacement of another object than the one held is
                // older than a read that found a later one, or its worker's
                // removal was heard ahead: it tells nothing.
                let held = follow.told.get(&worker_id);
                if held.is_some_and(|held| held.holds(&old)) {
                    let id = follow.id_of(&new);
                    self.stands(follow, worker_id, &new, id, true);
                }
            }
            Ok(WorkerEvent::Removed { worker_id, worker }) => {
                self.gone(follow, &worker_id, Some(&worker));
            }
            Ok(WorkerEvent::StatusChanged { .. }) => {}
            Err(RecvError::Lagged(_)) => {
                self.read_whole(follow);
                // The read is the one the follow owed, where it owed one.
                follow.unread = false;
            }
            Err(RecvError::Closed) => return false,
        }
        // A host that states ids may have stated one of the word's worker
        // and taken it back ahead of the ask: the read owed here ends with
        // the ids that stand.
        follow.unread |= membership && follow.stating;
        true
    }

    /// The registry knows `worker` as `object`, and its host names it
    /// `id`. Nothing where the follow holds that worker under that id, and
    /// nothing where the registry says `worker` replaces the object held
    /// (`replaced`) and the id stands. Else a join of `id`, behind a leave
    /// of the id the follow held under `object`: another worker under a
    /// registry id that is held, with no word of a replacement, is the
    /// worker's next life, heard ahead of the earlier one's removal or
    /// found by a whole read. Where another worker is held under `id`,
    /// this one is the id's next life too: a leave of `id` comes ahead of
    /// its join, so the stages keep nothing of the earlier life.
    ///
    /// A join no leave comes ahead of starts with no sent prefixes all the
    /// same: the records forget the id first, so the worker inherits
    /// nothing of a life the channel dropped whole.
    fn stands(
        &self,
        follow: &mut Follow,
        object: ObjectId,
        worker: &Arc<dyn Worker>,
        id: WorkerId,
        replaced: bool,
    ) {
        if let Some(was) = follow.told.insert(object, Held::of(id, worker)) {
            if was.id == id && (replaced || was.is(worker)) {
                return;
            }
            self.unnamed(follow, was.id);
        }
        let held = follow.under.entry(id).or_insert(0);
        *held += 1;
        if *held > 1 {
            self.left(id);
        } else {
            self.affinity.forget(id);
        }
        self.came(id);
    }

    /// The worker the registry knew as `object` stands no more: a leave of
    /// the id it was held under, and nothing where the follow held none. A
    /// removal names the object it removed (`removed`): where the follow
    /// holds a worker of another life under `object`, the removal is of an
    /// earlier life, and tells nothing. A whole read names none: what the
    /// follow holds and the read found no more is gone.
    fn gone(&self, follow: &mut Follow, object: &ObjectId, removed: Option<&Arc<dyn Worker>>) {
        let earlier = |held: &Held| removed.is_some_and(|removed| !held.is(removed));
        if follow.told.get(object).is_some_and(earlier) {
            return;
        }
        if let Some(was) = follow.told.remove(object) {
            self.unnamed(follow, was.id);
        }
    }

    /// One worker fewer is held under `was`: a leave of it, once the
    /// follow holds no worker under it.
    fn unnamed(&self, follow: &mut Follow, was: WorkerId) {
        match follow.under.get_mut(&was) {
            Some(held) if *held > 1 => *held -= 1,
            _ => {
                follow.under.remove(&was);
                self.left(was);
            }
        }
    }

    /// Reads the registry whole and tells the stages what the follow does
    /// not hold as it stands: each worker held that stands no more first,
    /// then each worker that stands. Last it tells them the ids that stand
    /// ([`Membership::Snapshot`]): a worker whose whole life the channel
    /// dropped is in neither list, and a stage drops there what requests
    /// left under it. The records of sent prefixes forget every id that
    /// is not among them ([`super::overlap::Affinity::keep`]). A worker
    /// found under a
    /// registry id the follow holds is the worker held where it is the
    /// object held or keeps its state, and the worker's next life
    /// otherwise: a worker that keeps no state and was replaced inside the
    /// dropped events is told as a next life. The host is asked of every
    /// worker ahead of the first word.
    fn read_whole(&self, follow: &mut Follow) {
        let standing = follow.registry.standing().into_iter();
        let stands: FxHashMap<ObjectId, (WorkerId, Arc<dyn Worker>)> = standing
            .map(|(object, worker)| (object, (follow.id_of(&worker), worker)))
            .collect();
        let held = follow.told.keys();
        let gone: Vec<ObjectId> = held
            .filter(|object| !stands.contains_key(*object))
            .cloned()
            .collect();
        for object in &gone {
            self.gone(follow, object, None);
        }
        for (object, (id, worker)) in stands {
            self.stands(follow, object, &worker, id, false);
        }
        let standing: Vec<WorkerId> = follow.under.keys().copied().collect();
        self.affinity.keep(&standing);
        self.told(&Membership::Snapshot(&standing));
    }
}

#[cfg(test)]
pub(super) mod tests;
