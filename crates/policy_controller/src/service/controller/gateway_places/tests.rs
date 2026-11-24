//! The cases of one gateway's places of one worker list.
//!
//! `GatewayPlaces::taken` names the places the asking gateway can use.
//! The last case asks over a list its holder lends.

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
        policy_engine::{PolicyEngine, PoolMembers},
        view::{Role, WireModes, WorkerId},
    },
    service::{controller::wire::Checked, proto},
    store::{
        snapshot::StandingVeto, GatewayId, HealthSink, MemoryStore, Reachability, VetoWrite,
        WorkerFacts, WorkersStore,
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
/// back.
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
/// gateway's own is one. A list whose host-local workers are all the asking
/// gateway's own states no place.
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
/// no wire (`GatewayPlaces::taken`). The stated set decides:
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
/// ahead of the wires. `Taken::Emptied` names the stage. A gateway
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
/// whatever its wires and its hint: the decision is one over no worker.
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
/// good while its hold stands. The positions ascend whatever the order the
/// vetoes come in.
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

/// A veto on a worker that is no place says nothing. Neither does a veto
/// on a worker the list does not hold.
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
