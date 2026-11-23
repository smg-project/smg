//! The cases of one gateway's places of one worker list.
//!
//! Each case names the place the controller on main answers by today: the
//! engine over stored records, which cuts and masks its rows for the
//! asking gateway (`StoreSnapshot::view` and `written` in
//! `store/snapshot.rs`, `narrow` there, and `emptied` in
//! `engine/store_backed_engine.rs`). One case holds the two against each
//! other over two fleets: the rows the store builds for an ask, and the
//! places this module names for it. The last case asks over a list its
//! holder lends.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use smg_worker::{identity::basic::Worker, registry::WorkerRegistry, test_support::ready_worker};

use super::{Asking, GatewayPlaces, Held, Taken};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{SUBSET_HINT, WIRES},
        decision_engine::{Decided, DecisionEngine, WorkerLists},
        list_cell::tests::PublishedList,
        policy_engine::{PolicyEngine, PoolMembers},
        view::{Candidate, FixedFacts, Role, WireModes, WorkerId},
    },
    service::{
        controller::{
            frame_request::FrameRequest,
            wire::{Checked, Lists},
        },
        proto,
    },
    store::{
        snapshot::{narrow, StandingVeto, ViewBuffers, ViewQuery},
        GatewayId, HealthSink, MemoryStore, Reachability, VetoWrite, WorkerFacts, WorkersStore,
    },
    worker_host::{
        reach::{ListReach, Reach},
        Members, Objects,
    },
};

const MODEL: &str = "model-a";
const ASKER: GatewayId = GatewayId(1);
const OTHER: GatewayId = GatewayId(2);
const HTTP: WireModes = WireModes::HTTP;
const GRPC: WireModes = WireModes::GRPC;

/// An engine under the preset `name`.
fn engine(name: &str) -> PolicyEngine {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    PolicyEngine::new(AppliedSpec::validate(&document).expect("it validates"))
}

/// A list of `workers` ready worker objects.
fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
    (0..workers)
        .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
        .collect()
}

/// An engine and its value of a list of `workers` worker objects.
fn list(workers: usize) -> (PolicyEngine, PoolMembers) {
    let engine = engine("round_robin");
    let members = engine.pool_members(&pool(workers));
    (engine, members)
}

/// A worker every gateway dials on `wires`.
fn any(wires: WireModes) -> Reach {
    Reach {
        wires,
        from: Reachability::Any,
    }
}

/// A worker `gateway` alone dials, on `wires`.
fn local(gateway: GatewayId, wires: WireModes) -> Reach {
    Reach {
        wires,
        from: Reachability::Gateway(gateway),
    }
}

/// A gateway that states no wires and no hint.
fn plain(gateway: GatewayId) -> Asking {
    Asking {
        gateway,
        wires: None,
        subset: None,
    }
}

fn stating(gateway: GatewayId, wires: WireModes) -> Asking {
    Asking {
        wires: Some(wires),
        ..plain(gateway)
    }
}

fn none_of(asking: Asking) -> Asking {
    Asking {
        subset: Some(0),
        ..asking
    }
}

/// A case's answer, owned: what [`Taken`] lends from the buffers.
#[derive(Debug, PartialEq, Eq)]
enum Answer {
    Held {
        places: Option<Vec<u32>>,
        stalled: Vec<u32>,
        held: Vec<u32>,
    },
    Emptied(&'static str),
    Unreached,
}

impl From<Taken<'_>> for Answer {
    fn from(taken: Taken<'_>) -> Self {
        match taken {
            Taken::Held(Held {
                places,
                stalled,
                held,
            }) => Self::Held {
                places: places.map(<[u32]>::to_vec),
                stalled: stalled.to_vec(),
                held: held.to_vec(),
            },
            Taken::Emptied(stage) => Self::Emptied(stage),
            Taken::Unreached => Self::Unreached,
        }
    }
}

/// An answer that holds the request to `places` with `stalled` and `held`
/// held back.
fn held(places: Option<&[u32]>, stalled: &[u32], held: &[u32]) -> Answer {
    Answer::Held {
        places: places.map(<[u32]>::to_vec),
        stalled: stalled.to_vec(),
        held: held.to_vec(),
    }
}

/// The whole list with nothing held back.
fn whole() -> Answer {
    held(None, &[], &[])
}

/// What `asking` may take of a list whose workers state `reach`, under
/// `vetoes` on the workers at those places: each a place and whether its
/// hold stands.
fn taken(reach: &[Reach], asking: Asking, vetoes: &[(usize, bool)]) -> Answer {
    let (engine, members) = list(reach.len());
    let records = members.records(engine.spec(), engine.clock());
    let ids = records.ids();
    let vetoes = vetoes.iter().map(|&(place, held)| StandingVeto {
        id: ids[place],
        held,
    });
    let reach = ListReach::new(reach.iter().copied());
    let mut buffers = GatewayPlaces::default();
    buffers.taken(&reach, records, asking, vetoes).into()
}

/// Every gateway dials the list whole, no veto stands and the stated wires
/// share one with every worker: no place is stated and no position is held
/// back, as the engine over stored records lends its rows whole
/// (`StoreSnapshot::view`).
#[test]
fn nothing_narrows_a_list_every_gateway_dials_whole() {
    let reach = [any(HTTP), any(HTTP.union(GRPC)), any(HTTP)];
    assert_eq!(taken(&reach, plain(ASKER), &[]), whole());
    assert_eq!(taken(&reach, stating(ASKER, HTTP), &[]), whole());
    assert_eq!(
        taken(&reach, stating(ASKER, HTTP.union(GRPC)), &[]),
        whole()
    );
    // A positive hint narrows nothing: no stage reads one.
    let some_of = Asking {
        subset: Some(1),
        ..plain(ASKER)
    };
    assert_eq!(taken(&reach, some_of, &[]), whole());
}

/// What a frame says of its gateway is read off the frame: the wires it
/// states, `None` where it states none, and its subset hint. The gateway
/// is the one whose stream carried the frame.
#[test]
fn a_frames_wires_and_hint_are_what_its_gateway_asks_by() {
    let frame = |constraints: proto::Constraints| {
        Checked::new(proto::DecideRequest {
            request_id: vec![1],
            facts: Some(proto::RequestFacts {
                model: MODEL.to_owned(),
                ..proto::RequestFacts::default()
            }),
            constraints: Some(constraints),
            ..proto::DecideRequest::default()
        })
        .expect("a frame the record holds")
    };
    let silent = Asking::of(OTHER, &frame(proto::Constraints::default()));
    assert_eq!(
        (silent.gateway, silent.wires, silent.subset),
        (OTHER, None, None)
    );
    let stated = Asking::of(
        ASKER,
        &frame(proto::Constraints {
            wire_modes: 0b011,
            subset_hint: Some(0),
            ..proto::Constraints::default()
        }),
    );
    assert_eq!(stated.gateway, ASKER);
    assert_eq!(stated.wires, Some(WireModes::from_bits(0b011)));
    assert_eq!(stated.subset, Some(0));
}

/// A worker another gateway alone dials is no place, and the asking
/// gateway's own is one, as the rows of another gateway's host-local
/// workers are left out of a gateway's list (`Segments::of`). A list whose
/// host-local workers are all the asking gateway's own states no place.
#[test]
fn another_gateways_host_local_worker_is_no_place() {
    let reach = [any(HTTP), local(OTHER, HTTP), local(ASKER, HTTP), any(HTTP)];
    let places = |gateway: GatewayId| taken(&reach, plain(gateway), &[]);
    assert_eq!(places(ASKER), held(Some(&[0, 2, 3]), &[], &[]));
    assert_eq!(places(OTHER), held(Some(&[0, 1, 3]), &[], &[]));
    assert_eq!(places(GatewayId(3)), held(Some(&[0, 3]), &[], &[]));
    let own = [any(HTTP), local(ASKER, HTTP)];
    assert_eq!(taken(&own, plain(ASKER), &[]), whole());
}

/// A gateway that states wires takes the workers that state one of them:
/// a worker that shares none is no place, and neither is one that states
/// no wire (`narrow`, by `wire_modes.intersects`). The stated set decides:
/// a worker that states two wires is a place for a gateway that states
/// either.
#[test]
fn a_worker_that_shares_no_wire_with_the_gateway_is_no_place() {
    let reach = [
        any(HTTP),
        any(GRPC),
        any(HTTP.union(GRPC)),
        any(WireModes::NONE),
    ];
    let over = |wires: WireModes| taken(&reach, stating(ASKER, wires), &[]);
    assert_eq!(over(HTTP), held(Some(&[0, 2]), &[], &[]));
    assert_eq!(over(GRPC), held(Some(&[1, 2]), &[], &[]));
    // A gateway that states no wires is held to none, the silent worker
    // included.
    assert_eq!(taken(&reach, plain(ASKER), &[]), whole());
    let mixed = [any(HTTP), local(OTHER, GRPC), any(GRPC)];
    assert_eq!(
        taken(&mixed, stating(ASKER, HTTP), &[]),
        held(Some(&[0]), &[], &[])
    );
}

/// Where the gateway dials a worker and its wires take every one, the
/// frame is shed under `wires`. A hint of 0 is shed under `subset-hint`,
/// ahead of the wires. Both answer as `emptied` names the stage. A gateway
/// that states only wires this build does not know states wires, and
/// shares one with nobody.
#[test]
fn a_list_the_wires_or_a_hint_of_0_empty_is_shed_under_that_stage() {
    let reach = [any(HTTP), any(HTTP)];
    let over = |asking: Asking| taken(&reach, asking, &[]);
    assert_eq!(over(stating(ASKER, GRPC)), Answer::Emptied(WIRES));
    assert_eq!(
        over(stating(ASKER, WireModes::NONE)),
        Answer::Emptied(WIRES)
    );
    assert_eq!(over(none_of(plain(ASKER))), Answer::Emptied(SUBSET_HINT));
    assert_eq!(
        over(none_of(stating(ASKER, GRPC))),
        Answer::Emptied(SUBSET_HINT)
    );
}

/// Where the gateway dials no worker of the list nothing is shed here,
/// whatever its wires and its hint: the decision is one over no worker, as
/// the engine over stored records decides over a pool of no row for it
/// (the test of `pool_size` ahead of `emptied`).
#[test]
fn a_gateway_that_dials_no_worker_of_the_list_decides_over_no_worker() {
    let reach = [local(OTHER, HTTP), local(OTHER, HTTP)];
    let over = |asking: Asking| taken(&reach, asking, &[]);
    assert_eq!(over(plain(ASKER)), Answer::Unreached);
    assert_eq!(over(stating(ASKER, GRPC)), Answer::Unreached);
    assert_eq!(over(none_of(plain(ASKER))), Answer::Unreached);
    // A list of no worker: every gateway takes all of it, and a hint of 0
    // finds no worker to shed.
    assert_eq!(taken(&[], plain(ASKER), &[]), whole());
    assert_eq!(taken(&[], none_of(plain(ASKER)), &[]), Answer::Unreached);
}

/// A standing veto holds its worker's position back, and holds it for
/// good while its hold stands, as the view names the row stalled and held.
/// The positions ascend whatever the order the vetoes come in.
#[test]
fn a_standing_veto_holds_its_workers_position_back() {
    let reach = [any(HTTP), any(HTTP), any(HTTP), any(HTTP)];
    // Neither the held-back positions nor the held ones come in order.
    let vetoes = [(3, false), (2, true), (0, true)];
    assert_eq!(
        taken(&reach, plain(ASKER), &vetoes),
        held(None, &[0, 2, 3], &[0, 2])
    );
}

/// A veto on a worker that is no place says nothing, as a veto on a row
/// the gateway's list leaves out masks nothing (`mask`, by the row's id).
/// Neither does a veto on a worker the list does not hold.
#[test]
fn a_veto_on_a_worker_that_is_no_place_says_nothing() {
    let reach = [any(HTTP), local(OTHER, HTTP), any(GRPC), any(HTTP)];
    let vetoes = [(1, true), (2, false), (3, true)];
    assert_eq!(
        taken(&reach, stating(ASKER, HTTP), &vetoes),
        held(Some(&[0, 3]), &[3], &[3])
    );

    let (engine, members) = list(2);
    let records = members.records(engine.spec(), engine.clock());
    let reach = ListReach::new([any(HTTP), any(HTTP)]);
    let elsewhere = StandingVeto {
        id: WorkerId(u64::MAX),
        held: true,
    };
    let mut buffers = GatewayPlaces::default();
    let answer = buffers.taken(&reach, records, plain(ASKER), [elsewhere].into_iter());
    assert_eq!(Answer::from(answer), whole());
}

/// A deciding thread asks frame after frame on the same buffers: a later
/// frame's answer holds nothing of an earlier frame's positions, where it
/// narrows too and where it does not.
#[test]
fn a_later_frames_places_hold_nothing_of_an_earlier_frames() {
    let (engine, members) = list(3);
    let records = members.records(engine.spec(), engine.clock());
    let ids = records.ids();
    let mixed = ListReach::new([any(HTTP), any(GRPC), any(HTTP)]);
    let every = ListReach::new([any(HTTP), any(HTTP), any(HTTP)]);
    let vetoed = std::iter::once(StandingVeto {
        id: ids[2],
        held: true,
    });
    let mut buffers = GatewayPlaces::default();
    let first = buffers.taken(&mixed, records, stating(ASKER, HTTP), vetoed);
    assert_eq!(Answer::from(first), held(Some(&[0, 2]), &[2], &[2]));
    // Another gateway's frame over the same list: other places, no veto.
    let second = buffers.taken(&mixed, records, stating(OTHER, GRPC), std::iter::empty());
    assert_eq!(Answer::from(second), held(Some(&[1]), &[], &[]));
    let third = buffers.taken(&every, records, plain(ASKER), std::iter::empty());
    assert_eq!(Answer::from(third), whole());
}

/// What the engine over stored records decides an ask over: the rows the
/// store builds for the asking gateway, narrowed by its wires and its hint
/// (`MemoryStore::build_view`, then `narrow`). Each is a worker's id, with
/// the two bits the build writes on a row its gateway vetoes.
struct Built {
    /// How many rows the build held before the wires and the hint.
    reached: usize,
    rows: Vec<WorkerId>,
    stalled: Vec<WorkerId>,
    held: Vec<WorkerId>,
}

fn built(store: &MemoryStore, asking: Asking, now: Instant) -> Built {
    let mut out = ViewBuffers::default();
    let ask = ViewQuery {
        gateway: asking.gateway,
        now,
        pinned: None,
        affinity: false,
        signal_cells: false,
    };
    let pool = store.build_view(MODEL, Role::Single, &ask, &mut out);
    assert!(pool.is_some(), "the model has a pool");
    let reached = out.candidates.len();
    narrow(&mut out, asking.wires, asking.subset);
    let ids = |keep: fn(&Candidate) -> bool| {
        let mut ids: Vec<WorkerId> = out
            .candidates
            .iter()
            .filter(|row| keep(row))
            .map(|row| row.id)
            .collect();
        ids.sort_unstable();
        ids
    };
    Built {
        reached,
        rows: ids(|_| true),
        stalled: ids(|row| row.health.stalled),
        held: ids(|row| row.health.breaker_open),
    }
}

/// How many asks over one fleet met each answer: held to a part of the
/// list, handed the list whole, shed, or over no row. `decided` counts the
/// frames of those asks that were answered a pick, eight frames an ask.
#[derive(Debug, Default)]
struct Met {
    narrowed: usize,
    whole: usize,
    shed: usize,
    unreached: usize,
    decided: usize,
}

/// Over one fleet, every ask is answered the same workers both ways. The
/// first fleet holds seven rows: two host-local, to different gateways,
/// workers on one wire, on the other and on both, and one that states no
/// wire. Vetoes stand through the door a controller writes
/// by: a held one, a doubt, one on a worker its gateway does not dial, and
/// another gateway's. The second fleet is one gateway's alone, so the two
/// other gateways reach no row of it. For each of three gateways, six sets
/// of wires a frame can state and three hints:
///
/// - the places this module names are the rows the store builds for the
///   ask, read as worker ids, the positions held back are the rows that
///   build marks stalled, and those held for good the rows whose breaker
///   it reads open;
/// - an ask the wires or a hint of 0 leave no row of a pool that has rows
///   is shed under that stage, and an ask of a gateway that reaches no
///   row is over no worker both ways, whatever it states;
/// - a frame of that ask, made by the maker held to those places, is
///   answered workers among those rows alone: never a held one, and no
///   stalled one while a row stands that is not.
#[test]
fn every_ask_takes_the_rows_the_store_builds_for_it() {
    let both = HTTP.union(GRPC);
    let mixed = [
        any(HTTP),
        local(OTHER, HTTP),
        any(GRPC),
        any(both),
        local(ASKER, HTTP),
        any(HTTP),
        any(WireModes::NONE),
    ];
    let vetoes = [
        (ASKER, 3, true),
        (ASKER, 5, false),
        (ASKER, 1, true),
        (OTHER, 0, false),
    ];
    let met = compared(&mixed, &vetoes);
    // The fleet met asks that narrow, sheds and decisions. Every gateway
    // dials a worker of it, and none dials every worker.
    assert!(
        met.narrowed > 0 && met.shed > 0 && met.decided > 0,
        "{met:?}"
    );
    assert_eq!((met.whole, met.unreached), (0, 0), "{met:?}");

    // A fleet one gateway alone dials: that gateway is handed it whole
    // where its wires allow, and each ask of the two others, 18 a
    // gateway, is over no row.
    let locals = [local(OTHER, HTTP), local(OTHER, both)];
    let met = compared(&locals, &[(OTHER, 1, false)]);
    assert_eq!(met.unreached, 2 * 18, "{met:?}");
    assert!(met.whole > 0 && met.shed > 0 && met.decided > 0, "{met:?}");
}

/// Every ask of three gateways over `fleet`, one row a worker in its
/// order, under `vetoes` (a gateway, the place of the worker it vetoes,
/// whether the veto holds): the store's build against this module's
/// places, and the frames of each ask that takes places decided over the
/// list.
fn compared(fleet: &[Reach], vetoes: &[(GatewayId, usize, bool)]) -> Met {
    const S: Duration = Duration::from_secs(1);
    let both = HTTP.union(GRPC);
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let ids: Vec<WorkerId> = (0..)
        .zip(fleet.iter().copied())
        .map(|(n, reach): (u32, Reach)| {
            let key = format!("http://worker-{n}:8000");
            let stated = WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                role: Role::Single,
                models: vec![MODEL.to_owned()],
                dp_size: 1,
                wire_modes: reach.wires,
                reachable_from: reach.from,
                ..WorkerFacts::default()
            };
            store.upsert(stated).id()
        })
        .collect();
    store.publish_now();

    // The list: one worker object a row, in the rows' order, under records
    // that state the store's ids.
    let engine = engine("random");
    let workers = pool(fleet.len());
    let mut place = 0;
    let members = PublishedList::with_facts(Arc::clone(&workers), |_| {
        place += 1;
        FixedFacts {
            id: ids.get(place - 1).copied(),
            ..FixedFacts::default()
        }
    });
    let made = engine.members_of(members.lent());
    let records = made.records(engine.spec(), engine.clock());
    assert_eq!(records.ids(), ids.as_slice());
    let reach = ListReach::new(fleet.iter().copied());

    let until = Some(t0 + 60 * S);
    for &(gateway, place, holds) in vetoes {
        let hold = until.filter(|_| holds);
        let written = store.gateway_veto_noted(gateway, ids[place], until, hold, false, t0);
        assert_eq!(written, VetoWrite::Written);
    }

    let by_id = |places: &[u32]| {
        let mut named: Vec<WorkerId> = places.iter().map(|&place| ids[place as usize]).collect();
        named.sort_unstable();
        named
    };
    let every: Vec<u32> = (0_u32..).take(fleet.len()).collect();
    let mut buffers = GatewayPlaces::default();
    let mut lists = Lists::default();
    let mut met = Met::default();
    // The wires a frame states, as its field carries them: none, each
    // kind, two, and a bit this build does not know.
    let stated = [HTTP, GRPC, both, WireModes::ZMQ].map(|wires| u32::from(wires.bits()));
    for gateway in [ASKER, OTHER, GatewayId(3)] {
        for wire_modes in [0, stated[0], stated[1], stated[2], stated[3], 0b1000] {
            for subset in [None, Some(0), Some(2)] {
                let frame_of = |seed: u64| {
                    Checked::new(proto::DecideRequest {
                        request_id: vec![1],
                        facts: Some(proto::RequestFacts {
                            model: MODEL.to_owned(),
                            seed,
                            ..proto::RequestFacts::default()
                        }),
                        constraints: Some(proto::Constraints {
                            wire_modes,
                            subset_hint: subset,
                            ..proto::Constraints::default()
                        }),
                        ..proto::DecideRequest::default()
                    })
                    .expect("a frame the record holds")
                };
                let asking = Asking::of(gateway, &frame_of(0));
                let case = format!("{asking:?}");
                let built = built(&store, asking, t0);
                let vetoes: Vec<StandingVeto> = store.with_snapshot(|world| {
                    world
                        .gateway_vetoes(MODEL, Role::Single, gateway)
                        .standing_at(t0)
                        .collect()
                });
                let held = match buffers.taken(&reach, records, asking, vetoes.into_iter()) {
                    Taken::Unreached => {
                        assert_eq!(built.reached, 0, "{case}");
                        met.unreached += 1;
                        continue;
                    }
                    Taken::Emptied(stage) => {
                        assert!(built.reached > 0 && built.rows.is_empty(), "{case}");
                        let by = if subset == Some(0) {
                            SUBSET_HINT
                        } else {
                            WIRES
                        };
                        assert_eq!(stage, by, "{case}");
                        met.shed += 1;
                        continue;
                    }
                    Taken::Held(held) => held,
                };
                let places = held.places.unwrap_or(every.as_slice());
                assert_eq!(by_id(places), built.rows, "{case}");
                assert!(!built.rows.is_empty(), "{case}");
                assert_eq!(by_id(held.stalled), built.stalled, "{case}");
                assert_eq!(by_id(held.held), built.held, "{case}");
                met.narrowed += usize::from(held.places.is_some());
                met.whole += usize::from(held.places.is_none());

                // The ask's frames, decided over the list.
                let free = built.rows.len() > built.stalled.len();
                for seed in 0..8 {
                    let frame = frame_of(seed);
                    let read = lists.of(&frame);
                    let maker = FrameRequest::new(
                        &frame,
                        &read,
                        Duration::from_millis(2),
                        WorkerLists::Single(members.lent()),
                    )
                    .held_to(held.places, held.stalled, held.held);
                    let answer = engine.decide_under(gateway, &maker, t0).expect("an answer");
                    let Decided::Whole(whole, _) = answer else {
                        panic!("{case}: a whole decision");
                    };
                    let Some(pick) = whole.decision.pick() else {
                        // Every row of the ask is held: nobody is picked.
                        assert_eq!(built.rows, built.held, "{case}: {:?}", whole.decision);
                        continue;
                    };
                    for target in pick.targets.iter() {
                        assert!(built.rows.contains(&target.id), "{case}: {target:?}");
                        assert!(!built.held.contains(&target.id), "{case}: {target:?}");
                        if free {
                            assert!(!built.stalled.contains(&target.id), "{case}: {target:?}");
                        }
                    }
                    met.decided += 1;
                }
            }
        }
    }
    met
}

/// Over a list its holder lends, a veto written through the door a
/// controller writes by holds its worker's position back. The veto names
/// its worker by the row's id, which the lent list's records state, and
/// the reach lent beside the list is of that list's places: a worker of
/// another role stands in neither.
#[test]
fn a_veto_holds_its_workers_position_back_over_a_lent_list() {
    let store = MemoryStore::new();
    let objects: Objects<Members> = Objects::new(Arc::new(WorkerRegistry::new()));
    let t0 = Instant::now();
    let row = |n: u32, role: Role, reach: Reach| {
        let key = format!("http://worker-{n}:8000");
        let stated = WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            role,
            models: vec![MODEL.to_owned()],
            dp_size: 1,
            wire_modes: reach.wires,
            reachable_from: reach.from,
            ..WorkerFacts::default()
        };
        store.upsert(stated).id()
    };
    let first = row(0, Role::Single, any(HTTP));
    row(1, Role::Prefill, any(HTTP));
    let second = row(2, Role::Single, local(ASKER, HTTP));
    let third = row(3, Role::Single, any(HTTP));
    objects.follow(&store);
    store.publish_now();

    let registry = objects.registry();
    let generation = registry.model_generation(MODEL).expect("a generation");
    let engine = engine("round_robin");
    let lent = objects.lent(&generation, Role::Single);
    let records = engine
        .members_of(lent)
        .records(engine.spec(), engine.clock());
    assert_eq!(records.ids(), [first, second, third]);
    let reach = objects.reach(&generation, Role::Single);
    let stated = [any(HTTP), local(ASKER, HTTP), any(HTTP)];
    assert_eq!(reach.workers(), stated);

    let until = Some(t0 + Duration::from_secs(60));
    let holds = store.gateway_veto_noted(ASKER, third, until, until, false, t0);
    let doubts = store.gateway_veto_noted(OTHER, first, until, None, false, t0);
    assert_eq!((holds, doubts), (VetoWrite::Written, VetoWrite::Written));
    let standing = |gateway: GatewayId| -> Vec<StandingVeto> {
        store.with_snapshot(|world| {
            world
                .gateway_vetoes(MODEL, Role::Single, gateway)
                .standing_at(t0)
                .collect()
        })
    };
    let mut buffers = GatewayPlaces::default();
    // The asking gateway dials all three, and its hold stands on the third.
    let vetoes = standing(ASKER).into_iter();
    let asker = buffers.taken(reach, records, plain(ASKER), vetoes);
    assert_eq!(Answer::from(asker), held(None, &[2], &[2]));
    // Another gateway does not dial the second, and doubts the first.
    let vetoes = standing(OTHER).into_iter();
    let other = buffers.taken(reach, records, plain(OTHER), vetoes);
    assert_eq!(Answer::from(other), held(Some(&[0, 2]), &[0], &[]));
}

/// A measurement for the Numbers table, not a check: what one ask costs,
/// in release under Callgrind, by the difference of two runs that differ
/// in their asks alone. The run is named in the environment,
/// `PLACES_COUNT=<form>:<workers>:<asks>`, and with none nothing is asked.
/// The forms: `whole`, a list nothing narrows; `wires`, a list whose every
/// third worker shares no wire with the gateway; `vetoed`, a whole list
/// under four standing vetoes.
#[test]
#[ignore = "a measurement for the Numbers table, run by name"]
fn measure_one_ask() {
    let Some(run) = std::env::var_os("PLACES_COUNT") else {
        return;
    };
    let run = run.to_string_lossy();
    let mut named = run.split(':');
    let form = named.next().unwrap_or_default();
    let mut number = || named.next().and_then(|n| n.parse::<usize>().ok());
    let (workers, asks) = (number().expect("the workers"), number().expect("the asks"));
    let (engine, members) = list(workers);
    let records = members.records(engine.spec(), engine.clock());
    let reach = ListReach::new((0..workers).map(|at| {
        if form == "wires" && at % 3 == 2 {
            any(GRPC)
        } else {
            any(HTTP)
        }
    }));
    let vetoed = if form == "vetoed" { 4 } else { 0 };
    let vetoes: Vec<StandingVeto> = records
        .ids()
        .iter()
        .take(vetoed)
        .map(|&id| StandingVeto { id, held: false })
        .collect();
    let asking = stating(ASKER, HTTP);
    let mut buffers = GatewayPlaces::default();
    let mut read = 0;
    for _ in 0..asks {
        let reach = std::hint::black_box(&reach);
        if let Taken::Held(held) = buffers.taken(reach, records, asking, vetoes.iter().copied()) {
            read += held.places.map_or(workers, <[u32]>::len) + held.stalled.len();
        }
    }
    assert_eq!(std::hint::black_box(read) % asks.max(1), 0);
}
