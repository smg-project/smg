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
//! host is not asked of a worker that is gone.
//!
//! The events are a broadcast, and a reader that falls behind loses the
//! oldest. So the follow reads the registry whole at its start and where
//! the channel says it dropped events: each worker it holds that stands no
//! more is told as at its removal, then each worker that stands as at its
//! registration or its replacement. Last the stages are told the ids that
//! stand: a worker whose whole life the channel dropped is in neither
//! list, and a stage drops there what requests left under it.
//!
//! The events the channel kept come behind that read and may be older than
//! it. So an id is told as left only once the follow holds no worker under
//! it: the removal of an earlier worker at an address, heard behind a read
//! that found the next one there, tells nothing. A worker registered under
//! an id that another worker is held under is the id's next life: a leave
//! and a join of the id.
//!
//! A channel that closed takes nothing the follow holds. The follow
//! subscribes again at the task's next upkeep and reads the registry whole
//! behind it, so a worker that stood through the gap is not told again.
//!
//! A panic of the host's own code at a word is held and counted
//! ([`PolicyEngine::upkeep_faults`]). The word is lost, so the follow reads
//! the registry whole again at the task's next upkeep.

use std::{
    fmt,
    future::pending,
    sync::{Arc, OnceLock},
};

use rustc_hash::FxHashMap;
use smg_worker::{
    identity::basic::Worker,
    registry::{event::WorkerEvent, Generation, WorkerId as ObjectId, WorkerRegistry},
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
    /// nothing: the id is the fingerprint of the worker's address.
    ///
    /// Asked on the engine's task, some time behind the registry's write:
    /// the host may have replaced the object since. A host that keeps a
    /// worker's id with its object answers of the object it is asked of,
    /// and each replacement is then told under its own id. A host that
    /// keeps the id by the worker's address answers of the worker that
    /// stands there. An id that stood only between two replacements heard
    /// together is then told to no stage: what requests left under it
    /// stays until a whole read's snapshot
    /// ([`PolicyEngine::set_registry`]).
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

impl fmt::Debug for Handed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handed")
            .field("registry", &self.0.get().is_some())
            .finish()
    }
}

/// The engine's follow of its host's registry: the subscription, and the id
/// each worker was told as joined under, by the registry's own id of the
/// worker. It holds no worker.
pub(super) struct Follow {
    registry: Arc<dyn Members>,
    /// The subscription. None at the follow's start and behind a channel
    /// that closed, until the task's next upkeep takes one.
    events: Option<Receiver<WorkerEvent>>,
    told: FxHashMap<ObjectId, WorkerId>,
    /// How many workers the follow holds under each id it told: the ids
    /// that stand.
    under: FxHashMap<WorkerId, u32>,
    /// Whether the follow owes a whole read of the registry at the task's
    /// next upkeep: at its start, behind a channel that closed, and behind
    /// a word lost to a panic of the host's own code.
    unread: bool,
}

impl Follow {
    /// The id the stages are told `worker` under: the one its host states,
    /// or the fingerprint of its address.
    fn id_of(&self, worker: &Arc<dyn Worker>) -> WorkerId {
        let facts = self.registry.facts_of(worker);
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
    /// | `Replaced` | nothing where the host states the id told of the new object: the worker stands, with what the stages keep for it. Where it states another, a leave of the id told and a join of the new one |
    /// | `StatusChanged` | nothing: a decision reads a worker's status on its object |
    ///
    /// A worker removed and another registered at its address come as a
    /// `Removed` and a `Registered`: a leave, then a join. At a leave the
    /// records of sent prefixes forget the worker too. The session table
    /// is told nothing, as at the two calls.
    ///
    /// An id names one worker. A worker registered under an id that
    /// another worker is held under is read as the id's next life: a leave
    /// and a join of the id, and the earlier worker's removal then tells
    /// nothing. A registry can say a registration at an address ahead of
    /// the removal it follows.
    ///
    /// The events are a broadcast, and a reader that falls behind loses
    /// the oldest: the task then reads the registry whole and tells what
    /// changed, and last the ids that stand ([`Membership::Snapshot`]),
    /// so a stage keeps nothing of a worker whose whole life the channel
    /// dropped. It does the same behind a channel that closed, on another
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
                let Some(registry) = self.members.0.get() else {
                    return;
                };
                members.insert(Follow {
                    registry: Arc::clone(registry),
                    events: None,
                    told: FxHashMap::default(),
                    under: FxHashMap::default(),
                    unread: true,
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
        match word {
            Ok(WorkerEvent::Registered { worker_id, worker }) => {
                let id = follow.id_of(&worker);
                self.stands(follow, worker_id, id);
            }
            Ok(WorkerEvent::Replaced { worker_id, new, .. }) => {
                let id = follow.id_of(&new);
                self.stands(follow, worker_id, id);
            }
            Ok(WorkerEvent::Removed { worker_id, .. }) => self.gone(follow, &worker_id),
            Ok(WorkerEvent::StatusChanged { .. }) => {}
            Err(RecvError::Lagged(_)) => self.read_whole(follow),
            Err(RecvError::Closed) => return false,
        }
        true
    }

    /// The worker the registry knows as `object` stands, and its host
    /// names it `id`: nothing where the follow holds it under that id.
    /// Else a join of `id`, behind a leave of the id it was held under.
    /// Where another worker is held under `id`, this one is the id's next
    /// life, heard ahead of the earlier one's removal: a leave of `id`
    /// comes ahead of its join, so the stages keep nothing of the earlier
    /// life.
    fn stands(&self, follow: &mut Follow, object: ObjectId, id: WorkerId) {
        let was = follow.told.insert(object, id);
        if was == Some(id) {
            return;
        }
        if let Some(was) = was {
            self.unnamed(follow, was);
        }
        let held = follow.under.entry(id).or_insert(0);
        *held += 1;
        if *held > 1 {
            self.left(id);
        }
        self.came(id);
    }

    /// The worker the registry knew as `object` stands no more: a leave of
    /// the id it was held under, and nothing where the follow held none.
    fn gone(&self, follow: &mut Follow, object: &ObjectId) {
        if let Some(was) = follow.told.remove(object) {
            self.unnamed(follow, was);
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
    /// left under it. The records of sent prefixes are told the single
    /// leaves alone. The host is asked of every worker ahead of the first
    /// word.
    fn read_whole(&self, follow: &mut Follow) {
        let standing = follow.registry.standing().into_iter();
        let stands: FxHashMap<ObjectId, WorkerId> = standing
            .map(|(object, worker)| (object, follow.id_of(&worker)))
            .collect();
        let held = follow.told.keys();
        let gone: Vec<ObjectId> = held
            .filter(|object| !stands.contains_key(*object))
            .cloned()
            .collect();
        for object in &gone {
            self.gone(follow, object);
        }
        for (object, id) in stands {
            self.stands(follow, object, id);
        }
        let standing: Vec<WorkerId> = follow.under.keys().copied().collect();
        self.told(&Membership::Snapshot(&standing));
    }
}

#[cfg(test)]
mod tests;
