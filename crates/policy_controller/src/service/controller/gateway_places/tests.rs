//! The cases of one gateway's places of one worker list.
//!
//! Each case names the place the controller on main answers by today: the
//! engine over stored records, which cuts and masks its rows for the
//! asking gateway (`StoreSnapshot::view` and `written` in
//! `store/snapshot.rs`, `narrow` there, and `emptied` in
//! `engine/store_backed_engine.rs`). The last case holds the two against
//! each other over one fleet: the rows the store builds for an ask, and
//! the places this module names for it.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use smg_worker::{identity::basic::Worker, test_support::ready_worker};

use super::{Asking, GatewayPlaces, Held, Taken};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{SUBSET_HINT, WIRES},
        decision_engine::{Decided, DecisionEngine, WorkerLists},
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
    worker_host::reach::{ListReach, Reach},
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
    let records = engine.with_spec(|spec| members.records(spec, engine.clock()));
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
    let records = engine.with_spec(|spec| members.records(spec, engine.clock()));
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
    let records = engine.with_spec(|spec| members.records(spec, engine.clock()));
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

/// Over one fleet, every ask is answered the same workers both ways. The
/// store holds seven rows: two host-local, to different gateways, and
/// wires of every kind. Vetoes stand through the door a controller writes
/// by: a held one, a doubt, one on a worker its gateway does not dial, and
/// another gateway's. For each of three gateways, six sets of wires a
/// frame can state and three hints:
///
/// - the places this module names are the rows the store builds for the
///   ask, read as worker ids, the positions held back are the rows that
///   build marks stalled, and those held for good the rows whose breaker
///   it reads open;
/// - an ask the wires or a hint of 0 leave no row of a pool that has rows
///   is shed under that stage, and an ask over no row is no shed here;
/// - a frame of that ask, made by the maker held to those places, is
///   answered workers among those rows alone: never a held one, and no
///   stalled one while a row stands that is not.
#[test]
fn every_ask_takes_the_rows_the_store_builds_for_it() {
    const S: Duration = Duration::from_secs(1);
    let both = HTTP.union(GRPC);
    let fleet = [
        any(HTTP),
        local(OTHER, HTTP),
        any(GRPC),
        any(both),
        local(ASKER, HTTP),
        any(HTTP),
        any(WireModes::NONE),
    ];
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let ids: Vec<WorkerId> = (0..)
        .zip(fleet)
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
            assert_eq!(Reach::of(&stated), reach);
            store.upsert(stated).id()
        })
        .collect();
    store.publish_now();

    // The list: one worker object a row, in the rows' order, under records
    // that state the store's ids.
    let engine = engine("random");
    let workers = pool(fleet.len());
    let mut place = 0;
    let members = engine.pool_members_with_fixed_facts(&workers, |_| {
        place += 1;
        FixedFacts {
            id: ids.get(place - 1).copied(),
            ..FixedFacts::default()
        }
    });
    let records = engine.with_spec(|spec| members.records(spec, engine.clock()));
    assert_eq!(records.ids(), ids.as_slice());
    let reach = ListReach::new(fleet);

    let until = Some(t0 + 60 * S);
    let veto = |gateway: GatewayId, place: usize, hold: Option<Instant>| {
        let written = store.gateway_veto_noted(gateway, ids[place], until, hold, false, t0);
        assert_eq!(written, VetoWrite::Written);
    };
    veto(ASKER, 3, until);
    veto(ASKER, 5, None);
    veto(ASKER, 1, until);
    veto(OTHER, 0, None);

    let by_id = |places: &[u32]| {
        let mut named: Vec<WorkerId> = places.iter().map(|&place| ids[place as usize]).collect();
        named.sort_unstable();
        named
    };
    let every: Vec<u32> = (0_u32..).take(fleet.len()).collect();
    let mut buffers = GatewayPlaces::default();
    let mut lists = Lists::default();
    let (mut narrowed, mut shed, mut unreached, mut decided) = (0, 0, 0, 0);
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
                        unreached += 1;
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
                        shed += 1;
                        continue;
                    }
                    Taken::Held(held) => held,
                };
                let places = held.places.unwrap_or(every.as_slice());
                assert_eq!(by_id(places), built.rows, "{case}");
                assert!(!built.rows.is_empty(), "{case}");
                assert_eq!(by_id(held.stalled), built.stalled, "{case}");
                assert_eq!(by_id(held.held), built.held, "{case}");
                narrowed += usize::from(held.places.is_some());

                // The ask's frames, decided over the list.
                let free = built.rows.len() > built.stalled.len();
                for seed in 0..8 {
                    let frame = frame_of(seed);
                    let read = lists.of(&frame);
                    let maker = FrameRequest::new(
                        &frame,
                        &read,
                        Duration::from_millis(2),
                        WorkerLists::Single(&members),
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
                    decided += 1;
                }
            }
        }
    }
    // The fleet met every answer: asks that narrow and asks that do not,
    // sheds, a gateway with no row, and decisions.
    assert!(
        narrowed > 0 && shed > 0 && decided > 0,
        "{narrowed} narrowed, {shed} shed, {unreached} unreached, {decided} decided"
    );
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
    let records = engine.with_spec(|spec| members.records(spec, engine.clock()));
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
