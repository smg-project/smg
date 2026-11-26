use std::{
    ptr,
    sync::{Arc, Barrier},
    thread,
};

use smg_worker::{
    registry::{Generation as _, WorkerRegistry},
    test_support::ready_worker,
};

use super::{ListReach, Reach};
use crate::{
    engine::{view::WireModes, Role},
    store::{GatewayId, MemoryStore, Reachability, Registrant, WorkerFacts, WorkersStore as _},
    worker_host::{remainder::RowRemainder, Members, Objects},
};

const MODEL: &str = "model-a";
const SINGLE_0: &str = "http://single-0:8000";
const SINGLE_1: &str = "http://single-1:8000";
const SINGLE_2: &str = "http://single-2:8000";
const SINGLE_3: &str = "http://single-3:8000";
const PREFILL: &str = "http://prefill-0:8000";
const DECODE: &str = "http://decode-0:8000";
const ENCODE: &str = "http://encode-0:8000";
const STRAY: &str = "http://stray-0:8000";
const HTTP: WireModes = WireModes::HTTP;
const GRPC: WireModes = WireModes::GRPC;

/// A worker every gateway dials on `wires`.
fn any(wires: WireModes) -> Reach {
    Reach {
        wires,
        from: Reachability::Any,
    }
}

/// A worker gateway `gateway` alone dials, on `wires`.
fn local(gateway: u64, wires: WireModes) -> Reach {
    Reach {
        wires,
        from: Reachability::Gateway(GatewayId(gateway)),
    }
}

/// A holder that builds its objects from the rows, with the store its
/// rows are written to.
struct Replica {
    store: MemoryStore,
    objects: Objects<Members>,
}

impl Replica {
    fn new() -> Self {
        Self {
            store: MemoryStore::new(),
            objects: Objects::new(Arc::new(WorkerRegistry::new())),
        }
    }

    /// Write the row at `url`, of `role` and stating `reach`, through
    /// the holder.
    fn write(&self, url: &str, role: Role, reach: Reach) {
        let facts = WorkerFacts {
            key: url.to_owned(),
            engine: url.to_owned(),
            incarnation: 1,
            role,
            models: vec![MODEL.to_owned()],
            dp_size: 1,
            wire_modes: reach.wires,
            reachable_from: reach.from,
            ..WorkerFacts::default()
        };
        let written = self.objects.upsert(&self.store, facts, Registrant::Control);
        written.expect("the store takes the write");
    }

    /// Remove the row at `url` through the holder.
    fn remove(&self, url: &str) {
        let row = self.store.by_key(url).expect("a row");
        let removed = self.objects.remove(&self.store, row.id, row.revision);
        assert!(removed.is_some());
    }

    /// A worker another writer registers: the holder lays no row for
    /// it.
    fn register_stray(&self) {
        let stray = ready_worker(STRAY, MODEL);
        assert!(self.objects.registry().register(stray).is_some());
    }

    /// The model's generation, as the registry holds it.
    fn generation(&self) -> Arc<Members> {
        let registry = self.objects.registry();
        registry.model_generation(MODEL).expect("a generation")
    }

    /// The address of each of `generation`'s workers of `role`, as the
    /// holder lends them, with the reach lent at the worker's place.
    fn lent<'a>(&self, generation: &'a Members, role: Role) -> Vec<(&'a str, Reach)> {
        let list = self.objects.lent(generation, role);
        let reach = self.objects.reach(generation, role).workers();
        assert_eq!(reach.len(), list.workers.len(), "one reach a worker");
        let urls = list.workers.iter().map(|worker| worker.url());
        urls.zip(reach.iter().copied()).collect()
    }
}

/// A worker's reach is the two facts its row states, read from what the
/// holder keeps beside the row's object.
#[test]
fn a_workers_reach_is_what_its_row_states() {
    let replica = Replica::new();
    let both = HTTP.union(GRPC);
    replica.write(SINGLE_0, Role::Single, local(7, both));
    let row = replica.store.by_key(SINGLE_0).expect("a row");
    assert_eq!(Reach::of(&RowRemainder::of(&row)), local(7, both));
}

/// A list every gateway dials is whole for a gateway that states no
/// wires, and for one whose wires every worker shares one of. One
/// worker that shares none, or that states none, ends that.
#[test]
fn a_list_is_whole_for_wires_every_worker_shares_one_of() {
    let both = HTTP.union(GRPC);
    let list = ListReach::new([any(HTTP), any(both), any(HTTP)]);
    assert!(list.whole(None));
    assert!(list.whole(Some(HTTP)));
    assert!(list.whole(Some(both)));
    assert!(!list.whole(Some(GRPC)));
    assert!(!list.whole(Some(WireModes::ZMQ)));
    // The last of the sets a gateway can state: every wire.
    let all = both.union(WireModes::ZMQ);
    assert!(list.whole(Some(all)));
    // A gateway that states only wires this build does not know
    // states wires, and shares one with nobody.
    assert!(!list.whole(Some(WireModes::NONE)));

    let silent = ListReach::new([any(HTTP), any(WireModes::NONE)]);
    assert!(silent.whole(None));
    assert!(!silent.whole(Some(HTTP)));
    assert!(!silent.whole(Some(all)));
    assert_eq!(silent.workers().len(), 2);
}

/// One host-local worker ends the list's wholeness for every gateway,
/// its own included: the places are then read worker by worker.
#[test]
fn a_host_local_worker_ends_the_lists_wholeness() {
    let list = ListReach::new([any(HTTP), local(1, HTTP)]);
    assert!(!list.whole(None));
    assert!(!list.whole(Some(HTTP)));
}

/// The holder lends the reach of each role's list beside the list: each
/// lent worker's stated wires and who dials it, at the worker's place,
/// and as many as the list. Two workers side by side that state the
/// same reach are two places, and a worker that states no wire states
/// none here. A worker with no row is in neither.
#[test]
fn each_roles_reach_states_each_lent_worker_at_its_place() {
    let replica = Replica::new();
    let both = HTTP.union(GRPC);
    let silent = any(WireModes::NONE);
    replica.register_stray();
    replica.write(SINGLE_0, Role::Single, any(HTTP));
    replica.write(SINGLE_1, Role::Single, any(HTTP));
    replica.write(PREFILL, Role::Prefill, local(7, GRPC));
    replica.write(SINGLE_2, Role::Single, local(3, both));
    replica.write(DECODE, Role::Decode, any(HTTP));
    replica.write(SINGLE_3, Role::Single, silent);
    replica.write(ENCODE, Role::Encode, any(GRPC));
    let generation = replica.generation();
    assert_eq!(generation.members().len(), 8);

    let singles = [
        (SINGLE_0, any(HTTP)),
        (SINGLE_1, any(HTTP)),
        (SINGLE_2, local(3, both)),
        (SINGLE_3, silent),
    ];
    assert_eq!(replica.lent(&generation, Role::Single), singles);
    let prefills = [(PREFILL, local(7, GRPC))];
    assert_eq!(replica.lent(&generation, Role::Prefill), prefills);
    assert_eq!(
        replica.lent(&generation, Role::Decode),
        [(DECODE, any(HTTP))]
    );
    assert_eq!(
        replica.lent(&generation, Role::Encode),
        [(ENCODE, any(GRPC))]
    );
    // The rows were read once for the four lists and their reach.
    assert_eq!(replica.objects.unstated(), 1, "one read of the rows");

    // What holds for each list as a whole.
    let single = replica.objects.reach(&generation, Role::Single);
    assert!(!single.whole(None), "one gateway alone dials a single");
    let decode = replica.objects.reach(&generation, Role::Decode);
    assert!(decode.whole(None) && decode.whole(Some(HTTP)));
    assert!(!decode.whole(Some(GRPC)));
    let again = replica.objects.reach(&generation, Role::Single);
    assert!(ptr::eq(single, again), "cut once");
}

/// A list and its reach are cut by one pass over the generation's
/// members, whichever is asked first, and the generation keeps both as
/// that pass read them. So a decision that holds one generation reads
/// a list and a reach that agree place by place, whatever left between
/// its two asks: a worker whose row left before the first ask is in
/// neither, and one whose row left behind it is in both.
#[test]
fn a_list_and_its_reach_are_of_one_pass_whichever_is_asked_first() {
    let replica = Replica::new();
    replica.write(SINGLE_0, Role::Single, any(HTTP));
    replica.write(SINGLE_1, Role::Single, local(3, HTTP));
    let before = replica.generation();
    assert_eq!(before.members().len(), 2);

    // The reach is asked first, once the first worker's row left.
    replica.remove(SINGLE_0);
    let reach = replica.objects.reach(&before, Role::Single);
    assert_eq!(reach.workers(), [local(3, HTTP)]);
    assert_eq!(replica.objects.unstated(), 1);
    // The list asked behind it is of the same pass.
    let stood = [(SINGLE_1, local(3, HTTP))];
    assert_eq!(replica.lent(&before, Role::Single), stood);
    assert_eq!(replica.objects.unstated(), 1, "one read of the rows");

    // The other order over the next generation: the list first, then
    // its one worker's row leaves, then the reach.
    let next = replica.generation();
    let list = replica.objects.lent(&next, Role::Single);
    assert!(matches!(&**list.workers, [one] if one.url() == SINGLE_1));
    replica.remove(SINGLE_1);
    let reach = replica.objects.reach(&next, Role::Single);
    assert_eq!(reach.workers(), [local(3, HTTP)]);
    assert_eq!(replica.objects.unstated(), 1, "one read of its rows");
}

/// Who dials a worker moves within the worker's life. A holder that
/// builds its objects from the rows lays a new object at that move, so
/// the registry publishes another generation, whose reach names the
/// gateway the row then states. The generation cut before the move
/// keeps the reach it was cut with.
#[test]
fn a_built_holders_next_generation_states_a_reach_that_moved() {
    let replica = Replica::new();
    replica.write(SINGLE_0, Role::Single, local(3, HTTP));
    let row = replica.store.by_key(SINGLE_0).expect("a row").id;
    let before = replica.generation();
    let stood = [(SINGLE_0, local(3, HTTP))];
    assert_eq!(replica.lent(&before, Role::Single), stood);

    // The same life, dialed by another gateway alone from here on.
    replica.write(SINGLE_0, Role::Single, local(9, HTTP));
    let moved = replica.store.by_key(SINGLE_0).expect("a row");
    assert_eq!(moved.id, row, "the row moves in place");
    let after = replica.generation();
    assert!(!Arc::ptr_eq(&before, &after), "another generation");
    let states = [(SINGLE_0, local(9, HTTP))];
    assert_eq!(replica.lent(&after, Role::Single), states);
    assert_eq!(replica.lent(&before, Role::Single), stood);
}

/// Asks of a generation's list and of its reach that meet at its first
/// ask cut both once: each reads a reach of the list's length, and the
/// holder's rows are read once.
#[test]
fn asks_of_a_list_and_of_its_reach_that_meet_cut_both_once() {
    let replica = Replica::new();
    replica.register_stray();
    replica.write(SINGLE_0, Role::Single, local(3, HTTP));
    let generation = replica.generation();
    let start = Barrier::new(8);
    let ask = |reach_first: bool| {
        start.wait();
        if reach_first {
            let reach = replica.objects.reach(&generation, Role::Single);
            let list = replica.objects.lent(&generation, Role::Single);
            (reach.workers().len(), list.workers.len())
        } else {
            let list = replica.objects.lent(&generation, Role::Single);
            let reach = replica.objects.reach(&generation, Role::Single);
            (reach.workers().len(), list.workers.len())
        }
    };
    let lengths: Vec<(usize, usize)> = thread::scope(|scope| {
        let asks: Vec<_> = (0..8)
            .map(|n| scope.spawn(move || ask(n % 2 == 0)))
            .collect();
        let read = asks.into_iter().map(|ask| ask.join().expect("an ask"));
        read.collect()
    });
    assert_eq!(lengths, [(1, 1); 8]);
    assert_eq!(replica.objects.unstated(), 1, "one read of the rows");
}
