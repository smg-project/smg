//! The cases of the worker objects behind a call: a decision behind an
//! acknowledged call reads the fleet the call left, a call builds a row's
//! object at its write, its status behind it, and the holder pairs two
//! registered rows by the worker crate's reading of their runtime names.

use super::*;

/// The replica's worker objects hold a call's write when the call
/// answers. No task follows the store here, and a decision follows
/// nothing: each is the in-process front's over the list the holder
/// lends as the door's last call left it. A worker registered through
/// the door is named by a decision behind the call. One whose
/// deregistration was acknowledged is named by none behind it, and
/// neither is one whose drain was acknowledged: a drain is written on
/// the worker object first. The replica's own writer shares the door's
/// turn: a worker it drains is named by none behind its answer.
#[tokio::test]
async fn a_decision_behind_an_acknowledged_call_reads_the_fleet_the_call_left() {
    use std::collections::BTreeSet;

    use crate::service::controller::decision_slots::tests::{frame, lent_pick, preset};

    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let objects = holder(&store);
    let door = MembershipCalls::based(Arc::clone(&store), feed.log, 0, Arc::clone(&objects));
    let door = Arc::new(door);
    store
        .apply(preset("round_robin", 1))
        .expect("above the unbound start");
    // The workers twelve decisions name, with no follow of the case's.
    let named = || -> BTreeSet<WorkerId> {
        (1..=12)
            .filter_map(|seed| lent_pick(&store, &objects, frame(seed, u64::from(seed))))
            .collect()
    };
    let ids: Vec<WorkerId> = ["w0", "w1", "w2"]
        .iter()
        .map(|name| {
            let asked = ask(unchecked(&format!("grpc://{name}:8000")), Mode::Add);
            let told = door
                .register(&operator(), under(TERM), &asked)
                .expect("the door takes the worker");
            WorkerId(row_of(&told).id)
        })
        .collect();
    assert_eq!(named(), ids.iter().copied().collect());
    let leave = |id: WorkerId, drain: bool| {
        let row = store.get(id).expect("the row stands");
        let request = proto::DeregisterWorkerRequest {
            worker_id: id.0,
            revision: row.revision,
            drain,
        };
        door.deregister(&operator(), under(TERM), &request)
            .expect("the door acknowledges");
    };
    // An acknowledged deregistration: no decision behind it names the
    // worker.
    leave(ids[0], false);
    assert_eq!(named(), BTreeSet::from([ids[1], ids[2]]));
    // And an acknowledged drain.
    leave(ids[1], true);
    assert_eq!(named(), BTreeSet::from([ids[2]]));
    // The last worker is drained through the replica's own writer,
    // over the same door: no decision behind its answer names a worker.
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    let last = store.get(ids[2]).expect("the row stands");
    let by = Remover::Registrant(&last.maker);
    assert_eq!(
        hosted.drained(by, "grpc://w2:8000", standing(&last)),
        Ok(MembershipWriteOutcome::Draining(ids[2]))
    );
    assert_eq!(named(), BTreeSet::new());
}

/// A call's write of a row builds the row's worker object where it is
/// written, and what the table still leads reaches the object behind
/// the write: the registry tells a registration, then the status the
/// connect signal moved, as the log tells the two. So it is for a call
/// of one row, for each rank of a wide call, and for a row the
/// replica's own writer lays. A door that wrote the table alone would
/// build each object behind its connect signal, ready from the start,
/// and the registry would tell no status.
#[test]
fn a_call_builds_a_rows_object_at_its_write_and_its_status_follows() {
    use smg_worker::registry::event::WorkerEvent;

    use crate::store::HealthPolicy;

    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let objects = holder(&store);
    let mut events = objects.registry().subscribe_events();
    let door = Arc::new(MembershipCalls::based(store, feed.log, 0, objects));
    // What the registry told since the last read, in its order.
    let mut told = move || -> Vec<&'static str> {
        let mut told = Vec::new();
        while let Ok(event) = events.try_recv() {
            told.push(match event {
                WorkerEvent::Registered { .. } => "registered",
                WorkerEvent::Replaced { .. } => "replaced",
                WorkerEvent::Removed { .. } => "removed",
                WorkerEvent::StatusChanged { .. } => "status",
            });
        }
        told
    };
    door.register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
        .expect("one row");
    assert_eq!(told(), ["registered", "status"]);
    let ranks = proto::WorkerSpec {
        dp_size: 2,
        ..unchecked("grpc://wide:8000")
    };
    door.register(&operator(), under(TERM), &ask(ranks, Mode::Add))
        .expect("two ranks");
    // A rank's connect signal is written behind its own row: the first
    // rank's status reaches its object ahead of the second rank's write.
    assert_eq!(told(), ["registered", "status", "registered", "status"]);
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    let own = "grpc://own:8000";
    let wrote = hosted.found_one(&found("pods/a"), own, &|_| WorkerFacts {
        key: own.to_owned(),
        engine: own.to_owned(),
        health: HealthPolicy {
            probes: false,
            ..HealthPolicy::default()
        },
        ..found_facts()
    });
    assert!(
        matches!(wrote, Ok(MembershipWriteOutcome::Registered(_))),
        "{wrote:?}"
    );
    assert_eq!(told(), ["registered", "status"]);
}

/// A registration states its runtime's name as its host writes it, and
/// the row keeps that spelling. The replica's holder hands the pair rule
/// the worker crate's reading of the name, so a prefill and a decode whose
/// registrations spell one runtime two ways pair.
#[test]
fn two_registrations_that_spell_one_runtime_two_ways_pair() {
    use smg_worker::worker::RuntimeType;

    use crate::document::spec::Pairing;

    let name = RuntimeType::Generic.as_str();
    let (store, door) = door();
    let sides = [
        (
            "grpc://10.0.3.1:8000",
            proto::Role::Prefill,
            name.to_uppercase(),
        ),
        ("grpc://10.0.3.2:8000", proto::Role::Decode, name.to_owned()),
    ];
    for (url, role, runtime) in &sides {
        let stated = proto::WorkerSpec {
            role: *role as i32,
            runtime: runtime.clone(),
            ..unchecked(url)
        };
        door.register(&operator(), under(TERM), &ask(stated, Mode::Add))
            .expect("one row");
        let row = store.by_key(url).expect("the registered row");
        assert_eq!(&row.facts.engine_kind, runtime, "{url}");
    }
    let registry = door.objects.registry();
    let generation = registry
        .model_generation("model-a")
        .expect("the model's generation");
    for mode in [Pairing::Off, Pairing::Lenient] {
        let classes = door.objects.pair_classes(&generation, mode);
        assert_eq!(
            (classes.prefills(), classes.decodes()),
            (&[1][..], &[1][..]),
            "{mode:?}"
        );
    }
}
