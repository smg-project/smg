//! Who can dial each worker of one list.
//!
//! A worker a controller registers states two facts that no request
//! changes: the wires it speaks, and whether every gateway dials it or one
//! gateway alone holds its host-local socket
//! ([`WorkerFacts::wire_modes`], [`WorkerFacts::reachable_from`]). A worker
//! object speaks one wire of the stated set, so the object alone cannot
//! say which gateways dial its worker.
//!
//! [`ListReach`] holds the two facts for one list of worker objects, in
//! the list's order, with what holds for the list as a whole. It is made
//! once a list, and read by each decision over the list for the places its
//! asking gateway may take.
//!
//! **A list's reach is as of its generation.** The wires a worker states do
//! not move within its life: a registration that states others starts
//! another life (`WorkerFacts::compare_to_standing`), and a holder lays
//! another life's object, which the registry publishes. Who dials a worker
//! moves within its life. A holder that builds its objects from the rows
//! lays a new object at each move of a row, so the registry publishes a
//! generation and the list's reach is made again. A holder that keeps the
//! objects its registrations hand in moves its entry alone and publishes
//! nothing ([`Objects::with_built_kept`](super::Objects::with_built_kept)):
//! there a list's reach stands as it was made until the model's next
//! membership change. No served controller is such a holder.

use crate::{
    engine::view::WireModes,
    store::{Reachability, WorkerFacts},
};

/// One worker's reach: the wires it states, and who dials it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reach {
    pub(crate) wires: WireModes,
    pub(crate) from: Reachability,
}

impl Reach {
    /// The reach a worker's registration states.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the holder of a controller's worker lists reads it where it makes a list's reach"
        )
    )]
    pub(crate) const fn of(facts: &WorkerFacts) -> Self {
        Self {
            wires: facts.wire_modes,
            from: facts.reachable_from,
        }
    }
}

/// The reach of every worker of one list, in the list's order, with two
/// facts of the whole list that answer most decisions at two tests
/// ([`Self::whole`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ListReach {
    workers: Box<[Reach]>,
    /// Whether a worker of the list is host-local: one gateway alone dials
    /// it, so another gateway's places leave it out.
    locals: bool,
    /// For each set of wires a gateway can state, by its bits, whether
    /// every worker of the list shares a wire with it.
    dialed_whole: u8,
}

impl ListReach {
    /// The reach of a list whose workers state `workers`, in the list's
    /// order. Its cost grows with the list, once a list.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the holder of a controller's worker lists makes one beside each list it lends"
        )
    )]
    pub(crate) fn new(workers: impl IntoIterator<Item = Reach>) -> Self {
        let workers: Box<[Reach]> = workers.into_iter().collect();
        let locals = workers
            .iter()
            .any(|worker| worker.from != Reachability::Any);
        let dialed_whole = (0..8_u8)
            .filter(|&stated| {
                let wires = WireModes::from_bits(stated);
                workers.iter().all(|worker| worker.wires.intersects(wires))
            })
            .fold(0, |whole, stated| whole | (1 << stated));
        Self {
            workers,
            locals,
            dialed_whole,
        }
    }

    /// Whether a gateway that states `wires` may take every place of the
    /// list, whichever gateway it is: every gateway dials every worker,
    /// and each worker shares a wire with it. A gateway that states no
    /// wires is held to none.
    pub(crate) fn whole(&self, wires: Option<WireModes>) -> bool {
        !self.locals && wires.is_none_or(|wires| self.dialed_whole & (1 << wires.bits()) != 0)
    }

    /// Each worker's reach, by its place in the list.
    pub(crate) fn workers(&self) -> &[Reach] {
        &self.workers
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smg_worker::registry::{Generation as _, WorkerRegistry};

    use super::{ListReach, Reach};
    use crate::{
        engine::view::WireModes,
        store::{GatewayId, MemoryStore, Reachability, WorkerFacts, WorkersStore as _},
        worker_host::{Members, Objects},
    };

    fn any(wires: WireModes) -> Reach {
        Reach {
            wires,
            from: Reachability::Any,
        }
    }

    /// A worker's reach is the two facts its registration states.
    #[test]
    fn a_workers_reach_is_what_its_registration_states() {
        let facts = WorkerFacts {
            wire_modes: WireModes::HTTP.union(WireModes::GRPC),
            reachable_from: Reachability::Gateway(GatewayId(7)),
            ..WorkerFacts::default()
        };
        let reach = Reach::of(&facts);
        assert_eq!(reach.wires, WireModes::HTTP.union(WireModes::GRPC));
        assert_eq!(reach.from, Reachability::Gateway(GatewayId(7)));
    }

    /// A list every gateway dials is whole for a gateway that states no
    /// wires, and for one whose wires every worker shares one of. One
    /// worker that shares none, or that states none, ends that.
    #[test]
    fn a_list_is_whole_for_wires_every_worker_shares_one_of() {
        let both = WireModes::HTTP.union(WireModes::GRPC);
        let list = ListReach::new([any(WireModes::HTTP), any(both), any(WireModes::HTTP)]);
        assert!(list.whole(None));
        assert!(list.whole(Some(WireModes::HTTP)));
        assert!(list.whole(Some(both)));
        assert!(!list.whole(Some(WireModes::GRPC)));
        assert!(!list.whole(Some(WireModes::ZMQ)));
        // A gateway that states only wires this build does not know
        // states wires, and shares one with nobody.
        assert!(!list.whole(Some(WireModes::NONE)));

        let silent = ListReach::new([any(WireModes::HTTP), any(WireModes::NONE)]);
        assert!(silent.whole(None));
        assert!(!silent.whole(Some(WireModes::HTTP)));
        assert_eq!(silent.workers().len(), 2);
    }

    /// One host-local worker ends the list's wholeness for every gateway,
    /// its own included: the places are then read worker by worker.
    #[test]
    fn a_host_local_worker_ends_the_lists_wholeness() {
        let local = Reach {
            wires: WireModes::HTTP,
            from: Reachability::Gateway(GatewayId(1)),
        };
        let list = ListReach::new([any(WireModes::HTTP), local]);
        assert!(!list.whole(None));
        assert!(!list.whole(Some(WireModes::HTTP)));
    }

    /// Who dials a worker moves within the worker's life. A holder that
    /// builds its objects from the rows lays a new object at that move, so
    /// the registry publishes another generation, and the reach made of
    /// that generation's list names the gateway the row then states: a
    /// reach kept with a generation is never behind its list there.
    #[test]
    fn a_built_holders_next_generation_states_a_reach_that_moved() {
        const URL: &str = "http://w:8000";
        const MODEL: &str = "model-a";
        let store = MemoryStore::new();
        let registry = Arc::new(WorkerRegistry::new());
        let objects: Objects<Members> = Objects::new(Arc::clone(&registry));
        let stated = |reachable_from: Reachability| WorkerFacts {
            key: URL.to_owned(),
            models: vec![MODEL.to_owned()],
            wire_modes: WireModes::HTTP,
            reachable_from,
            ..WorkerFacts::default()
        };
        let id = store.upsert(stated(Reachability::Any)).id();
        // The model's list as the registry publishes it now, and its reach
        // by what the holder states of the object's row.
        let published = || {
            objects.follow(&store);
            let generation = registry.generation(MODEL);
            let list = Arc::clone(generation.members());
            let object = objects.object(id).expect("the row's object");
            let laid = registry.get(&object).expect("the object");
            assert!(matches!(&*list, [one] if Arc::ptr_eq(one, &laid)));
            let row = objects.row_of(&object).expect("the object's row");
            (list, ListReach::new([Reach::of(&row.facts)]))
        };
        let (first, before) = published();
        assert_eq!(before.workers(), [any(WireModes::HTTP)]);
        assert!(before.whole(Some(WireModes::HTTP)));

        // The same life, dialed by one gateway alone from here on.
        let only = Reachability::Gateway(GatewayId(7));
        assert_eq!(
            store.upsert(stated(only)).id(),
            id,
            "the row moves in place"
        );
        let (second, after) = published();
        assert!(!Arc::ptr_eq(&first, &second), "another generation");
        let local = Reach {
            wires: WireModes::HTTP,
            from: only,
        };
        assert_eq!(after.workers(), [local]);
        assert!(!after.whole(None));
    }
}
