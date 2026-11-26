use std::{
    collections::BTreeSet,
    sync::{atomic::Ordering, Arc},
    thread,
    time::{Duration, Instant},
};

use http::uri::PathAndQuery;
use prost::Message as _;
use tokio::{
    net::TcpListener,
    sync::{mpsc, watch},
    task::JoinSet,
};
use tonic::{
    client::Grpc,
    codegen::tokio_stream::StreamExt as _,
    transport::{server::TcpIncoming, Channel, Server},
    Code, Request,
};
use tonic_prost::ProstCodec;

use super::{unanswered, Call, RpcStatus, Service, Shared};
use crate::{
    service::{
        controller,
        controller::{
            config::Scope,
            decision_slots::{tests as bed, Job, TERM_MARK},
            election::{self, ElectionObservation, LeasePlace},
            gateways::tests::PATIENT,
            generations::{tests::LEAST_LOAD, Format, Unanswered},
            principal::{
                gateway_id,
                tests::{stating, told, NO_PRINCIPAL},
                Layer,
            },
            replication::{journaled_at, tests::journaled_from},
            spec_store::{tests::names, SpecStore},
            wire::Checked,
            PROTOCOL_FLOOR, PROTOCOL_VERSION,
        },
        elector::{
            ConditionalWriteOutcome, ElectError, ElectionSettings, Elector, FileLease,
            LeaseBackend, LeaseRecord, MemoryLease, TickOutcome,
        },
        proto::{self, busy::Reason, grand_central_server::GrandCentral},
    },
    stages::registry::StageRegistry,
    store::{LeasesStore, Position, WorkersStore},
    tracking::lease::{LeaseId, ReleaseReason},
};

impl Call {
    /// Every call, in the contract's order.
    const ALL: [Call; 14] = [
        Call::Session,
        Call::Decide,
        Call::WatchWorkers,
        Call::RegisterWorker,
        Call::PatchWorker,
        Call::DeregisterWorker,
        Call::GetSnapshot,
        Call::Explain,
        Call::ApplyPolicySpec,
        Call::GetPolicySpec,
        Call::RollbackPolicySpec,
        Call::StepDown,
        Call::WatchRegime,
        Call::Replicate,
    ];
}

const ID: &str = "controller-0";
const ADDRESS: &str = "10.0.0.7:30100";
const IDENTITY: &str = "controller-0@10.0.0.7:30100";

/// A candidate over a lease of its own, the election's writer and one
/// listener's service over that candidate's fence.
fn service(scopes: &[Scope]) -> (Elector, watch::Sender<ElectionObservation>, Service) {
    let elector = Elector::new(
        Box::new(MemoryLease::new()),
        IDENTITY,
        ElectionSettings::default(),
    )
    .expect("the default settings are in order");
    over(elector, scopes)
}

/// The same over a candidate alone over its lease: its take leads at
/// once, by the clock a call reads.
fn service_alone(scopes: &[Scope]) -> (Elector, watch::Sender<ElectionObservation>, Service) {
    let elector = Elector::alone(IDENTITY, PATIENT).expect("the settings are in order");
    over(elector, scopes)
}

/// The election's writer and one listener's service over the fence of
/// `elector`, handed back beside them.
fn over(
    elector: Elector,
    scopes: &[Scope],
) -> (Elector, watch::Sender<ElectionObservation>, Service) {
    over_in(elector, scopes, StageRegistry::builtin())
}

/// [`over`]'s service, its documents naming the stages of `catalog`.
fn over_in(
    elector: Elector,
    scopes: &[Scope],
    catalog: StageRegistry,
) -> (Elector, watch::Sender<ElectionObservation>, Service) {
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let shared = Arc::new(
        Shared::new(
            ID.to_owned(),
            ADDRESS.to_owned(),
            &elector,
            bed::unserved(stood),
            bed::followed(journaled_at(&LeasePlace::Process, 0, 2)),
            (Arc::new(catalog), &LeasePlace::Process),
            2,
        )
        .expect("the deciders start"),
    );
    (elector, standing, Service::new(scopes.to_vec(), shared))
}

/// Polls the answer until `met`, bounded: every answer's term is at or
/// above the one before it, `last` carried from poll to poll.
fn answers_until(
    service: &Service,
    last: &mut u64,
    what: &str,
    met: impl Fn(&proto::LeaderInfo) -> bool,
) -> proto::LeaderInfo {
    let started = Instant::now();
    loop {
        let info = service.leader_info(Instant::now());
        assert!(
            info.term >= *last,
            "{what}: an answer under term {} after one under {last}",
            info.term
        );
        *last = info.term;
        if met(&info) {
            return info;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "{what} did not happen in 20 s: {info:?}"
        );
        thread::sleep(Duration::from_millis(2));
    }
}

/// A store that was given a word to the engine before its replica's
/// shared state is built keeps that one, and the replica's count of
/// sends would then ask another's worker objects or none: a debug
/// build stops where the shared state is built.
#[cfg(debug_assertions)]
#[tokio::test]
#[should_panic(expected = "before its replica's")]
async fn a_word_given_before_the_replicas_stops_a_debug_build() {
    let elector = Elector::alone(IDENTITY, PATIENT).expect("the settings are in order");
    let (_standing, stood) = watch::channel(ElectionObservation::default());
    let journaled = journaled_from(&LeasePlace::Process, 0);
    let (store, feed, objects, loads) = bed::followed(journaled);
    store.tell_through(|_, _, _, _, _| {});
    let catalog = Arc::new(StageRegistry::builtin());
    let state = (store, feed, objects, loads);
    let _ = Shared::new(
        ID.to_owned(),
        ADDRESS.to_owned(),
        &elector,
        bed::unserved(stood),
        state,
        (catalog, &LeasePlace::Process),
        1,
    );
}

/// The service's own calls object writes the worker objects its
/// decisions read: the one holder the service is built with. Workers
/// registered through that object are named by a decision over the
/// holder, with no follow of the case's and no deciding thread's. One
/// whose deregistration the object acknowledged is named by none
/// behind it, and neither is one whose drain it acknowledged. A service
/// that handed its calls another holder would name none from the first
/// registration on.
#[tokio::test]
async fn the_services_calls_write_the_worker_objects_its_decisions_read() {
    use crate::{
        engine::view::WorkerId,
        service::proto::{register_worker_request::Mode, worker_event::Event},
    };

    const TERM: u64 = 3;
    let elector = Elector::alone(IDENTITY, PATIENT).expect("the settings are in order");
    let (_standing, stood) = watch::channel(ElectionObservation::default());
    let journaled = journaled_from(&LeasePlace::Process, 0);
    let (store, feed, objects, loads) = bed::followed(journaled);
    store.set_term(TERM);
    let shared = Shared::new(
        ID.to_owned(),
        ADDRESS.to_owned(),
        &elector,
        bed::unserved(stood),
        (Arc::clone(&store), feed, Arc::clone(&objects), loads),
        (Arc::new(StageRegistry::builtin()), &LeasePlace::Process),
        1,
    )
    .expect("the decider starts");
    let applied = store.apply(bed::preset("round_robin", 1));
    applied.expect("above the unbound start");
    let by = told("operator-a");
    // The workers twelve decisions name over the service's holder.
    let named = || -> BTreeSet<WorkerId> {
        let frames = (1..=12).map(|seed| bed::frame(seed, u64::from(seed)));
        frames
            .filter_map(|frame| bed::lent_pick(&store, &objects, frame))
            .collect()
    };
    let register = |name: &str| {
        let spec = proto::WorkerSpec {
            url: format!("grpc://{name}:8000"),
            wire: proto::WireMode::Grpc as i32,
            role: proto::Role::Single as i32,
            models: vec![bed::MODEL.to_owned()],
            health_policy: Some(proto::HealthPolicy {
                disabled: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let asked = proto::RegisterWorkerRequest {
            spec: Some(spec),
            mode: Mode::Add as i32,
            wait_ready: false,
        };
        let told = shared.door.register(&by, || Ok(TERM), &asked);
        match told.expect("the service's door takes the worker").event {
            Some(Event::Registered(row)) => WorkerId(row.id),
            other => panic!("no registration in {other:?}"),
        }
    };
    let ids = ["w0", "w1", "w2"].map(register);
    assert_eq!(named(), BTreeSet::from(ids));
    let leave = |id: WorkerId, drain: bool| {
        let row = WorkersStore::get(&*store, id).expect("the row stands");
        let request = proto::DeregisterWorkerRequest {
            worker_id: id.0,
            revision: row.revision,
            drain,
        };
        let told = shared.door.deregister(&by, || Ok(TERM), &request);
        told.expect("the service's door acknowledges");
    };
    leave(ids[0], false);
    assert_eq!(named(), BTreeSet::from([ids[1], ids[2]]));
    leave(ids[1], true);
    assert_eq!(named(), BTreeSet::from([ids[2]]));
}

/// A leader whose term's lease ids pass their mark leaves and decides on
/// under the next term, over the election's own thread as the controller
/// starts it. Every request sent across the hand-over is answered, by a
/// pick or by `Busy` for no leader; no lease id is minted twice; the
/// leases of the term before stay on file for their frames.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_leader_at_its_terms_mark_leaves_and_decides_on_under_the_next_term() {
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(400),
    };
    let elector = Elector::alone(IDENTITY, settings).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    let shared = Shared::new(
        ID.to_owned(),
        ADDRESS.to_owned(),
        &elector,
        bed::unserved(stood),
        bed::followed((Arc::clone(&store), feed)),
        (Arc::new(StageRegistry::builtin()), &LeasePlace::Process),
        1,
    )
    .expect("the decider starts");
    let mut running = shared.elect(elector, standing).expect("a thread");
    let wait = Duration::from_secs(5);

    // The first take, then a fleet, a profile and the ids just under
    // the term's mark.
    let begun = Instant::now();
    while shared.fence.leader_term(Instant::now()).is_none() {
        assert!(begun.elapsed() < wait, "no first take");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    bed::fleet(&store, &shared.objects);
    let applied = store.apply(bed::preset("random", 1));
    applied.expect("above the unbound start");
    store.spend_ids(u32::MAX - TERM_MARK + 4);

    let (sender, mut answers) = mpsc::channel(64);
    let (mut picks, mut busies, mut sent) = (Vec::<u64>::new(), Vec::new(), 0);
    let begun = Instant::now();
    while picks.iter().filter(|id| **id >> 32 == 2).count() < 32 {
        assert!(
            begun.elapsed() < wait,
            "no pick under the next term: {} picks, {} Busy",
            picks.len(),
            busies.len()
        );
        sent += 16;
        for answer in bed::burst(&shared.deciders, &sender, &mut answers).await {
            match answer {
                Ok(id) => picks.push(id),
                Err(reason) => busies.push(reason),
            }
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    // Every request sent was answered, a `Busy` for no leader alone,
    // and the counter never reached its end.
    assert_eq!(picks.len() + busies.len(), sent);
    assert!(
        busies.iter().all(|reason| *reason == Reason::NotLeader),
        "{busies:?}"
    );
    let unleased = &shared.deciders.counts().unleased;
    assert_eq!(unleased.load(Ordering::Relaxed), 0);
    // The term before decided up to its mark and past it, the next one
    // from its first id on, and no id came twice.
    let old = picks.iter().take_while(|id| **id >> 32 == 1).count();
    let (old, new) = picks.split_at(old);
    assert!(old.contains(&(1 << 32 | u64::from(TERM_MARK))));
    assert_eq!(new.first(), Some(&(2 << 32 | 1)));
    assert!(new.iter().all(|id| id >> 32 == 2), "{new:?}");
    assert_eq!(picks.iter().collect::<BTreeSet<_>>().len(), picks.len());
    // A lease of the term before stays on file, and its frame is taken.
    let last = LeaseId(*old.last().expect("a pick under the first term"));
    let leases: &dyn LeasesStore = store.as_ref();
    assert!(leases.get(bed::GATEWAY, last).is_some());
    assert!(leases.released(bed::GATEWAY, last, None, ReleaseReason::ClientGone));

    running.stop();
    tokio::time::timeout(Duration::from_secs(20), running.ended())
        .await
        .expect("the election thread ends within 20 s of its stop");
}

/// The deciders hold the election's door, and the door holds its words
/// weakly: an election started for this state and dropped with no stop
/// ends and leads no more, while the state and its deciders live on.
#[tokio::test]
async fn an_election_dropped_without_a_stop_ends_while_the_deciders_hold_its_door() {
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(400),
    };
    let elector = Elector::alone(IDENTITY, settings).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    let shared = Shared::new(
        ID.to_owned(),
        ADDRESS.to_owned(),
        &elector,
        bed::unserved(stood),
        bed::followed((store, feed)),
        (Arc::new(StageRegistry::builtin()), &LeasePlace::Process),
        1,
    )
    .expect("the decider starts");
    let running = shared.elect(elector, standing).expect("a thread");
    let patience = Duration::from_secs(20);
    let mut looks = shared.standing.clone();
    {
        let led = looks.wait_for(|standing| standing.holder == IDENTITY);
        let led = tokio::time::timeout(patience, led).await;
        assert!(matches!(led, Ok(Ok(_))), "the first take");
    }

    // The thread's end drops the standing's sender: no look follows it.
    drop(running);
    let ended = async { while looks.changed().await.is_ok() {} };
    tokio::time::timeout(patience, ended)
        .await
        .expect("the election thread ends within 20 s of its handle's drop");
    assert_eq!(shared.fence.leader_term(Instant::now()), None);
    assert!(looks.borrow().holder.is_empty(), "the lease is given up");
}

/// A replica that deferred its start's document runs it a moment
/// behind its first term: the take applies it behind the stamp, under
/// the fence, so a take that never leads publishes nothing.
#[tokio::test]
async fn a_deferred_document_stands_a_moment_behind_the_first_term() {
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(400),
    };
    let elector = Elector::alone(IDENTITY, settings).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    let shared = Shared::new(
        ID.to_owned(),
        ADDRESS.to_owned(),
        &elector,
        bed::unserved(stood),
        bed::followed((store, feed)),
        // The door stands under the fence, as under a lease another
        // replica can take: its start passes the document.
        (
            Arc::new(StageRegistry::builtin()),
            &LeasePlace::File("shared".into()),
        ),
        1,
    )
    .expect("the decider starts");
    shared
        .generations
        .start_with(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    assert_eq!(shared.store.snapshot().spec().generation(), 0);

    let mut running = shared.elect(elector, standing).expect("a thread");
    let stands = async {
        while shared.store.snapshot().spec().generation() == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(20), stands)
        .await
        .expect("the first take applies the start's document within 20 s");
    assert_eq!(shared.store.snapshot().spec().generation(), 1);
    assert_eq!(shared.fence.term(), 1, "under its first term");

    running.stop();
    tokio::time::timeout(Duration::from_secs(20), running.ended())
        .await
        .expect("the election thread ends within 20 s of its stop");
}

/// Who may take the lease a leader hands over at its term's mark.
#[derive(Clone, Copy, Debug)]
enum Place {
    /// The leader is alone over a lease of its own.
    Alone,
    /// One replica over a shared lease, and no standby.
    Shared,
    /// A shared lease a standby looks at.
    Standby,
}

/// What a term's mark costs, by the clock: how long the leader decides
/// on under the term it leaves, past the burst that carried the mark's
/// pick; then the time with no leader, to the next leader's open fence,
/// this replica's under its next term or the standby's; and the `Busy`
/// answers this replica gave from the mark on.
async fn left_with_no_leader(
    place: Place,
    settings: ElectionSettings,
) -> (Duration, Duration, usize) {
    let dir = tempfile::tempdir().expect("a temp dir");
    let lease = dir.path().join("lease");
    let elector = match place {
        Place::Alone => Elector::alone(IDENTITY, settings),
        Place::Shared | Place::Standby => {
            Elector::new(Box::new(FileLease::new(&lease)), IDENTITY, settings)
        }
    }
    .expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    let shared = Shared::new(
        ID.to_owned(),
        ADDRESS.to_owned(),
        &elector,
        bed::unserved(stood),
        bed::followed((Arc::clone(&store), feed)),
        (Arc::new(StageRegistry::builtin()), &LeasePlace::Process),
        1,
    )
    .expect("the decider starts");
    let mut running = shared.elect(elector, standing).expect("a thread");
    let (wait, begun) = (Duration::from_secs(60), Instant::now());
    while shared.fence.leader_term(Instant::now()).is_none() {
        assert!(begun.elapsed() < wait, "no first take");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    // The standby starts under a leader and has read its record: it
    // follows.
    let mut standby = None;
    if matches!(place, Place::Standby) {
        let other = "controller-1@10.0.0.8:30100";
        let other = Elector::new(Box::new(FileLease::new(&lease)), other, settings)
            .expect("the settings are in order");
        let fence = other.fence();
        let (standing, mut stood) = watch::channel(ElectionObservation::default());
        let store = journaled_from(&LeasePlace::Process, 0).0;
        let running = election::start(other, store, standing).expect("a thread");
        let follows = stood.wait_for(|standing| standing.holder == IDENTITY);
        let followed = tokio::time::timeout(wait, follows).await;
        assert!(
            matches!(followed, Ok(Ok(_))),
            "the standby reads the leader's record"
        );
        standby = Some((fence, running));
    }
    bed::fleet(&store, &shared.objects);
    let applied = store.apply(bed::preset("random", 1));
    applied.expect("above the unbound start");
    store.spend_ids(u32::MAX - TERM_MARK + 4);

    let (sender, mut answers) = mpsc::channel(64);
    let mark = Ok(1 << 32 | u64::from(TERM_MARK));
    // The burst of the mark's pick, then the last burst a pick of the
    // term it leaves came in: the time with no leader starts there.
    let (mut marked, mut last, mut busy) = (None, begun, 0);
    let led = loop {
        assert!(begun.elapsed() < wait, "no leader after the mark");
        let read = bed::burst(&shared.deciders, &sender, &mut answers).await;
        let now = Instant::now();
        if marked.is_some() {
            busy += read.iter().filter(|answer| answer.is_err()).count();
            let led = match &standby {
                Some((fence, _)) => fence.leader_term(now).is_some(),
                None => read
                    .iter()
                    .any(|answer| matches!(answer, Ok(id) if id >> 32 == 2)),
            };
            if led {
                break now;
            }
        } else if read.contains(&mark) {
            marked = Some(now);
        }
        if read
            .iter()
            .any(|answer| matches!(answer, Ok(id) if id >> 32 == 1))
        {
            last = now;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    };
    let marked = marked.expect("the mark's pick came before the next leader");

    let patience = Duration::from_secs(20);
    running.stop();
    tokio::time::timeout(patience, running.ended())
        .await
        .expect("the election thread ends within 20 s of its stop");
    if let Some((_, other)) = standby.as_mut() {
        other.stop();
        tokio::time::timeout(patience, other.ended())
            .await
            .expect("the standby's thread ends within 20 s of its stop");
    }
    (last.duration_since(marked), led.duration_since(last), busy)
}

/// What the page's Numbers say of a hand-over at a term's mark under
/// the default settings, by who may take the lease: the time the leader
/// still decides past the mark, the time with no leader and the `Busy`
/// answers. A measurement, run by name with `--ignored`: it prints, and
/// asserts only that a leader follows.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn the_time_a_terms_mark_leaves_with_no_leader() {
    use std::io::Write as _;

    let mut out = std::io::stdout();
    for place in [Place::Alone, Place::Shared, Place::Standby] {
        let (decided, none, busy) = left_with_no_leader(place, ElectionSettings::default()).await;
        writeln!(
            out,
            "MARK {place:?}: decided on for {decided:?}, no leader for {none:?}, {busy} Busy"
        )
        .expect("stdout");
    }
}

/// The promise at the answer, over the election's own thread: the lease
/// file goes back to this run's first record while the run leads under a
/// later term. Nobody is named under that later term while the record
/// stands still for its duration, then this run leads above it: no
/// answer's term is under the one before it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_put_back_to_this_runs_older_record_never_lowers_an_answers_term() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("lease");
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(400),
    };
    let elector = Elector::new(Box::new(FileLease::new(&path)), IDENTITY, settings)
        .expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let (store, feed) = journaled_at(&LeasePlace::Process, 0, 2);
    let shared = Arc::new(
        Shared::new(
            ID.to_owned(),
            ADDRESS.to_owned(),
            &elector,
            bed::unserved(stood),
            bed::followed((Arc::clone(&store), feed)),
            (Arc::new(StageRegistry::builtin()), &LeasePlace::Process),
            2,
        )
        .expect("the deciders start"),
    );
    let service = Service::new(vec![Scope::Gateway], shared);
    let mut running = election::start(elector, store, standing).expect("a thread");
    let mut last = 0;

    let first = answers_until(&service, &mut last, "the first take", |info| info.is_leader);
    assert_eq!(first, answer(ID, ADDRESS, 1, true));
    let kept = dir.path().join("kept");
    std::fs::copy(&path, &kept).expect("the copy");
    // The case moves the file under the store's own lock, as a writer of
    // the store does: no renewal stands between its compare and its
    // rename at a move, so none puts back what the move took.
    let lock = dir.path().join("lease.lock");
    let locked = |moved: &dyn Fn()| {
        let held = std::fs::File::create(&lock).expect("the lock file");
        held.lock().expect("the store's lock");
        moved();
    };

    // The record is gone, by no release: this run waits its own lost
    // record's duration out, as a candidate that saw it does, and takes
    // again under the next term.
    let removed = Instant::now();
    locked(&|| std::fs::remove_file(&path).expect("the removal"));
    answers_until(&service, &mut last, "the take after the removal", |info| {
        info.is_leader && info.term == 2
    });
    assert!(
        removed.elapsed() >= settings.duration,
        "the take came {:?} after the removal, sooner than the lost record's duration",
        removed.elapsed()
    );

    // The store goes back to the first record, whole, as a restore puts
    // it. The run's own record under an older term: nobody leads under
    // term two, and the take that follows is above it.
    locked(&|| std::fs::rename(&kept, &path).expect("the restore"));
    let fenced = answers_until(
        &service,
        &mut last,
        "the look at the restored record",
        |info| !info.is_leader,
    );
    assert_eq!(fenced, answer("", "", 2, false));
    // A renewal that landed before the restore can open the fence once
    // more under term two, where the host held the election thread: the
    // take above is the first leader's answer past that term.
    let third = answers_until(&service, &mut last, "the take above", |info| {
        info.is_leader && info.term > 2
    });
    assert_eq!(third, answer(ID, ADDRESS, 3, true));

    running.stop();
    // Bounded like every wait here: a loop that ignored its stop fails
    // this case by name.
    tokio::time::timeout(Duration::from_secs(20), running.ended())
        .await
        .expect("the election thread ends within 20 s of its stop");
}

fn saw(holder: &str, term: u64) -> ElectionObservation {
    ElectionObservation {
        holder: holder.to_owned(),
        term,
        looked: true,
    }
}

/// An answer that names `controller_id` at `address` under `term`.
fn answer(controller_id: &str, address: &str, term: u64, is_leader: bool) -> proto::LeaderInfo {
    proto::LeaderInfo {
        controller_id: controller_id.to_owned(),
        address: address.to_owned(),
        term,
        is_leader,
        synced: false,
        sync: None,
        groups: Vec::new(),
        protocol_min: PROTOCOL_FLOOR,
        protocol_max: PROTOCOL_VERSION,
    }
}

/// The table a second time, copied by hand from the contract's header:
/// a change of a call's scopes is made in both.
#[test]
fn every_call_keeps_the_scopes_copied_here_from_the_contracts_header() {
    let expected: [(&str, &[Scope]); 14] = [
        ("Session", &[Scope::Gateway]),
        ("Decide", &[Scope::Gateway]),
        ("WatchWorkers", &[Scope::Gateway]),
        ("RegisterWorker", &[Scope::Control]),
        ("PatchWorker", &[Scope::Control]),
        ("DeregisterWorker", &[Scope::Control]),
        ("GetSnapshot", &[Scope::Control]),
        ("Explain", &[Scope::Control]),
        ("ApplyPolicySpec", &[Scope::Control]),
        ("GetPolicySpec", &[Scope::Control]),
        ("RollbackPolicySpec", &[Scope::Control]),
        ("StepDown", &[Scope::Control]),
        ("WatchRegime", &[Scope::Control]),
        ("Replicate", &[Scope::Replication]),
    ];
    assert_eq!(Call::ALL.map(|call| (call.name(), call.scopes())), expected);
}

/// `message` as a call the listener's layer placed `subject` on.
fn from<T>(subject: &str, message: T) -> Request<T> {
    let mut request = Request::new(message);
    request.extensions_mut().insert(told(subject));
    request
}

/// Admission hands back the principal the layer placed, on the scopes
/// the listener mounts. Off them the call is denied whoever asks; on
/// them a call with no principal is refused as unauthenticated.
#[test]
fn a_listener_admits_the_calls_of_its_scopes_from_a_caller_with_a_principal() {
    let listeners: [&[Scope]; 4] = [
        &[Scope::Gateway],
        &[Scope::Control],
        &[Scope::Replication],
        &Scope::ALL,
    ];
    for scopes in listeners {
        let (_elector, _standing, service) = service(scopes);
        for call in Call::ALL {
            let mounted = call.scopes().iter().any(|scope| scopes.contains(scope));
            let placed = from("gateway-a", ());
            match service.admit(call, &placed) {
                Ok(principal) => {
                    assert!(mounted, "{} admitted on {scopes:?}", call.name());
                    assert_eq!(principal, &told("gateway-a"));
                }
                Err(status) => {
                    assert!(!mounted, "{} refused on {scopes:?}", call.name());
                    assert_eq!(status.code(), Code::PermissionDenied);
                    assert_eq!(
                        status.message(),
                        format!("{} is not a call of this listener's scopes", call.name())
                    );
                }
            }
            let status = service
                .admit(call, &Request::new(()))
                .expect_err("no principal stands on the call");
            let code = if mounted {
                Code::Unauthenticated
            } else {
                Code::PermissionDenied
            };
            assert_eq!(status.code(), code, "{} on {scopes:?}", call.name());
            if mounted {
                assert_eq!(status.message(), NO_PRINCIPAL);
            }
        }
    }
}

#[tokio::test]
async fn an_unserved_call_is_denied_off_its_scope_and_unimplemented_to_a_principal_on_it() {
    let (_elector, _standing, gateway) = service(&[Scope::Gateway]);
    let (_elector, _standing, control) = service(&[Scope::Control]);
    let (_elector, _standing, replication) = service(&[Scope::Replication]);

    let read = || from("operator-a", proto::GetPolicySpecRequest::default());
    let denied = gateway.get_policy_spec(read()).await.expect_err("refused");
    assert_eq!(denied.code(), Code::PermissionDenied);
    let unserved = control.get_policy_spec(read()).await.expect_err("refused");
    assert_eq!(unserved.code(), Code::Unimplemented);
    assert_eq!(
        unserved.message(),
        "GetPolicySpec is not served by a controller started with no spec source"
    );
    // A caller no layer placed is refused before the call is named
    // unserved: no call body is reached without a principal.
    let unplaced = control
        .get_policy_spec(Request::new(proto::GetPolicySpecRequest::default()))
        .await
        .expect_err("refused");
    assert_eq!(unplaced.code(), Code::Unauthenticated);
    assert_eq!(unplaced.message(), NO_PRINCIPAL);

    // The snapshot is the operators' read and the leader's to serve:
    // this replica does not lead. A standby is restored on its
    // Replicate stream: the replication scope does not make the call.
    let snapshot = || from("standby-a", proto::SnapshotRequest::default());
    for (listener, code) in [
        (&gateway, Code::PermissionDenied),
        (&control, Code::Unavailable),
        (&replication, Code::PermissionDenied),
    ] {
        let status = listener
            .get_snapshot(snapshot())
            .await
            .expect_err("refused");
        assert_eq!(status.code(), code);
    }
}

#[tokio::test]
async fn a_watch_past_its_principals_ceiling_is_refused_by_name() {
    let (mut elector, _standing, service) = service_alone(&[Scope::Gateway]);
    elector.lead_at(Instant::now());
    service.shared.store.set_term(1);
    let watch = |by| from(by, proto::WatchWorkersRequest::default());
    // A request refused by its field takes no seat.
    let needs = proto::WatchWorkersRequest {
        with_needs: true,
        ..Default::default()
    };
    let refused = service
        .watch_workers(from("gateway-a", needs))
        .await
        .expect_err("refused");
    assert_eq!(refused.code(), Code::InvalidArgument);
    let mut held = Vec::new();
    for _ in 0..super::watch_workers::WATCHES_MAX {
        let opened = service.watch_workers(watch("gateway-a")).await;
        held.push(opened.expect("a seat"));
    }
    let refused = service
        .watch_workers(watch("gateway-a"))
        .await
        .expect_err("one watch past the ceiling");
    assert_eq!(
        (refused.code(), refused.message()),
        (
            Code::ResourceExhausted,
            "this principal holds its ceiling of 4 open watches: end one, then watch again"
        )
    );
    // Another principal's watches are its own.
    service
        .watch_workers(watch("gateway-b"))
        .await
        .expect("another principal's seat");
}

#[tokio::test]
async fn a_watch_is_served_by_the_leader_once_its_store_stands_in_the_term() {
    let (mut elector, _standing, service) = service_alone(&[Scope::Gateway, Scope::Control]);
    let watch = || from("gateway-a", proto::WatchWorkersRequest::default());
    let register = || {
        let spec = proto::WorkerSpec {
            url: "grpc://10.0.0.5:8000".to_owned(),
            wire: proto::WireMode::Grpc as i32,
            role: proto::Role::Single as i32,
            models: vec!["model-a".to_owned()],
            ..Default::default()
        };
        let request = proto::RegisterWorkerRequest {
            spec: Some(spec),
            ..Default::default()
        };
        from("operator-a", request)
    };
    // A replica that does not lead follows no log in this build: a
    // standby not yet synced, with the `NotLeader` a write's refusal
    // carries.
    let refused = service.watch_workers(watch()).await.expect_err("refused");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::FOLLOWS_NO_LOG)
    );
    let details = RpcStatus::decode(refused.details()).expect("a google.rpc.Status");
    assert_eq!(
        (details.code, details.message.as_str()),
        (Code::Unavailable as i32, super::FOLLOWS_NO_LOG)
    );
    let [detail] = details.details.as_slice() else {
        panic!("one detail: {details:?}");
    };
    let named = proto::NotLeader::decode(detail.value.as_slice()).expect("a NotLeader");
    assert_eq!(named, proto::NotLeader::default());

    // The lease is taken and its fence is open, at the take of a
    // candidate alone; the store has not entered the term: no watch is
    // cut and no write is numbered under the term before it.
    elector.lead_at(Instant::now());
    let early = service.watch_workers(watch()).await.expect_err("refused");
    assert_eq!(
        (early.code(), early.message()),
        (Code::Unavailable, super::NOT_IN_TERM)
    );
    let early = service
        .register_worker(register())
        .await
        .expect_err("refused");
    assert_eq!(
        (early.code(), early.message()),
        (Code::Unavailable, super::NOT_IN_TERM)
    );
    assert_eq!(service.shared.store.epoch(), 0);

    // In the term: the watch is cut at the term's start, and the answer
    // of a registration and the watch tell one event at one place.
    service.shared.store.set_term(1);
    let mut stream = service
        .watch_workers(watch())
        .await
        .expect("opened")
        .into_inner();
    let mut next = async || {
        tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("an event within five seconds")
            .expect("the stream is open")
            .expect("an event, not a status")
    };
    let page = next().await;
    assert_eq!((page.seq, page.term, page.membership_epoch), (1, 1, 0));
    let answered = service
        .register_worker(register())
        .await
        .expect("registered")
        .into_inner();
    assert_eq!(
        (answered.seq, answered.term, answered.membership_epoch),
        (2, 1, 1)
    );
    assert_eq!(next().await, answered);

    // A request this build cannot serve is refused by its field.
    let needs = proto::WatchWorkersRequest {
        with_needs: true,
        ..Default::default()
    };
    let refused = service
        .watch_workers(from("gateway-a", needs))
        .await
        .expect_err("refused");
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert!(
        refused.message().starts_with("with_needs: "),
        "{}",
        refused.message()
    );
}

/// A watch hears the role's word from its open on. The word that
/// nobody leads here, told before a take, stands until the take is
/// told: the fence admits a watch under it, and it ends no such watch.
/// Told again, it ends the watch whatever the fence reads.
#[tokio::test]
async fn a_watch_ends_at_a_role_told_after_its_open_and_at_none_before() {
    let mut elector = Elector::alone(IDENTITY, PATIENT).expect("the settings are in order");
    let (_standing, stood) = watch::channel(ElectionObservation::default());
    // The role's word is the case's to tell, as the serve's one task
    // tells it.
    let (told, role) = watch::channel(None);
    let (stood, stop, _untold, lag) = bed::unserved(stood);
    let shared = Shared::new(
        ID.to_owned(),
        ADDRESS.to_owned(),
        &elector,
        (stood, stop, role, lag),
        bed::followed(journaled_from(&LeasePlace::Process, 0)),
        (Arc::new(StageRegistry::builtin()), &LeasePlace::Process),
        1,
    )
    .expect("the decider starts");
    let service = Service::new(vec![Scope::Gateway], Arc::new(shared));
    told.send_replace(None);
    elector.lead_at(Instant::now());
    service.shared.store.set_term(1);
    let mut stream = service
        .watch_workers(from("gateway-a", proto::WatchWorkersRequest::default()))
        .await
        .expect("opened")
        .into_inner();
    let mut next = async || {
        tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("a word within five seconds")
            .expect("the stream is open")
    };
    let page = next()
        .await
        .expect("a page: the word before the open ends nothing");
    assert_eq!((page.seq, page.term, page.membership_epoch), (1, 1, 0));
    // The same word again, while the fence reads this replica's lead.
    told.send_replace(None);
    let ended = next().await.expect_err("the word told after the open");
    assert_eq!(ended.code(), Code::Aborted);
    assert_eq!(service.shared.fence.leader_term(Instant::now()), Some(1));
}

/// The path the transport routes `Decide` by is the contract's: its
/// service's name, then its method's.
#[test]
fn the_decide_calls_path_is_the_contracts_services_and_its_methods() {
    assert!(methods().contains(&"Decide"));
    assert_eq!(
        super::DECIDE_PATH,
        format!("/{}/Decide", controller::SERVICE_NAME)
    );
}

/// The methods of the contract's service, by the contract's own text.
fn methods() -> Vec<&'static str> {
    const CONTRACT: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/contracts/grand_central.proto"
    ));
    CONTRACT
        .lines()
        .skip_while(|line| !line.starts_with("service GrandCentral"))
        .take_while(|line| !line.starts_with('}'))
        .filter_map(|line| line.trim_start().strip_prefix("rpc "))
        .filter_map(|rest| rest.split('(').next())
        .collect()
}

/// The calls a step serves, beside `Leader`, each with its refusal by a
/// replica that never took the lease: a write is the leader's alone, as
/// are Explain and a standby's batches. A watch of the rows and the operators'
/// snapshot are a standby's once it follows the leader's log, which
/// none does in this build.
fn refused_off_the_leader(call: Call) -> Option<(Code, &'static str)> {
    const NOT_LEADING: (Code, &str) = (Code::FailedPrecondition, "this replica does not lead");
    match call {
        Call::RegisterWorker
        | Call::PatchWorker
        | Call::DeregisterWorker
        | Call::Explain
        | Call::Replicate => Some(NOT_LEADING),
        Call::WatchWorkers | Call::GetSnapshot => Some((Code::Unavailable, super::FOLLOWS_NO_LOG)),
        _ => None,
    }
}

/// Every method of the contract, called by its own name over a socket
/// on a listener of each scope set, three ways: a name stated to a
/// listener with no layer, a name stated to one that takes identities
/// as told, and no name stated to that one. `Leader` is served every
/// way; any other answers in its own name: denied off the listener's
/// scopes, and on them unauthenticated to a caller the layer did not
/// place. To the caller it placed, a call no step serves is
/// unimplemented, `Session` is reached (the empty message is no
/// `Hello`), and registration, watching rows, snapshots and Explain
/// are refused by a replica that does not lead. The names are read
/// from the contract, so a method added there joins the case. A name
/// the contract dropped is no method: whoever calls it, the transport
/// answers unimplemented and no body admits the call first.
#[tokio::test]
async fn every_method_of_the_contract_answers_by_its_calls_scopes_and_its_callers_principal() {
    /// The relays' two calls, which left the contract.
    const DROPPED: [&str; 2] = ["PushLoad", "PushKvEvents"];
    let methods = methods();
    assert_eq!(
        methods.len(),
        Call::ALL.len() + 1,
        "one call a method, and Leader: {methods:?}"
    );
    let listeners = [
        &[Scope::Gateway][..],
        &[Scope::Control],
        &[Scope::Replication],
    ];
    let ways = [
        (Layer::Off, true),
        (Layer::AsTold, true),
        (Layer::AsTold, false),
    ];
    for (scopes, (layer, stated)) in listeners
        .into_iter()
        .flat_map(|scopes| ways.map(|way| (scopes, way)))
    {
        let (_elector, _standing, service) = service(scopes);
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a loopback port");
        let addr = listener.local_addr().expect("the bound address");
        // Aborted at its drop, at the end of this listener's round.
        let mut serving = JoinSet::new();
        serving.spawn(
            Server::builder()
                .add_routes(service.routes(layer))
                .serve_with_incoming(TcpIncoming::from(listener)),
        );
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("a uri")
            .connect()
            .await
            .expect("the listener accepts");
        let mut client = Grpc::new(channel);
        for method in methods.iter().chain(&DROPPED) {
            let path = PathAndQuery::try_from(format!("/smg.policy.v1.GrandCentral/{method}"))
                .expect("a path");
            client.ready().await.expect("the channel is ready");
            // An empty message is every request's default.
            let request = if stated {
                stating("gateway-a")
            } else {
                Request::new(())
            };
            let answered = client
                .unary(request, path, ProstCodec::<(), ()>::default())
                .await;
            let round = format!("{method} on {scopes:?}, {layer:?}, a name stated: {stated}");
            if DROPPED.contains(method) {
                let status = answered.expect_err("the contract holds no such method");
                assert_eq!(
                    (status.code(), status.message()),
                    (Code::Unimplemented, ""),
                    "{round}"
                );
                continue;
            }
            if *method == "Leader" {
                assert!(answered.is_ok(), "{round}: {answered:?}");
                continue;
            }
            let call = Call::ALL.into_iter().find(|call| call.name() == *method);
            assert!(call.is_some(), "{method} is no call of the table");
            let mounted = call
                .into_iter()
                .flat_map(Call::scopes)
                .any(|scope| scopes.contains(scope));
            let expected = if !mounted {
                (
                    Code::PermissionDenied,
                    format!("{method} is not a call of this listener's scopes"),
                )
            } else if layer == Layer::AsTold && stated && *method == "Session" {
                (
                        Code::FailedPrecondition,
                        "no session stands on this stream: Hello is the first frame, a frame of no kind came first"
                            .to_owned(),
                    )
            } else if layer == Layer::AsTold
                && stated
                && matches!(
                    *method,
                    "ApplyPolicySpec" | "GetPolicySpec" | "RollbackPolicySpec"
                )
            {
                (
                    Code::Unimplemented,
                    format!("{method} is not served by a controller started with no spec source"),
                )
            } else if layer == Layer::AsTold && stated && *method == "Decide" {
                // The stream opens, and the call itself refuses the
                // case's one frame: it carries no id.
                (
                    Code::InvalidArgument,
                    "request_id is out of its range in this protocol version".to_owned(),
                )
            } else if layer == Layer::AsTold && stated {
                match call.and_then(refused_off_the_leader) {
                    Some((code, why)) => (code, why.to_owned()),
                    None => (
                        Code::Unimplemented,
                        format!("{method} is not served by this build"),
                    ),
                }
            } else {
                (Code::Unauthenticated, NO_PRINCIPAL.to_owned())
            };
            let status = answered.expect_err("no listener here serves the call");
            assert_eq!(
                (status.code(), status.message().to_owned()),
                expected,
                "{round}"
            );
        }
    }
}

/// A lease whose every write takes `slow`.
struct WritesSlowly {
    lease: MemoryLease,
    slow: Duration,
}

impl LeaseBackend for WritesSlowly {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        thread::sleep(self.slow);
        self.lease.write(expected, record)
    }

    fn place(&self) -> String {
        self.lease.place()
    }
}

/// A take the election thread has not published yet: the fence holds
/// the new term while the last look still names the holder before it.
#[test]
fn a_take_newer_than_the_last_look_names_nobody() {
    const OTHER: &str = "controller-1@10.0.0.8:30100";
    let settings = ElectionSettings {
        duration: Duration::from_millis(60),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(40),
    };
    let lease = MemoryLease::new();
    let mut other =
        Elector::new(Box::new(lease.clone()), OTHER, settings).expect("the settings are in order");
    // This replica's writes outlive its renew deadline.
    let mut elector = Elector::new(
        Box::new(WritesSlowly {
            lease: lease.clone(),
            slow: settings.renew_deadline + Duration::from_millis(20),
        }),
        IDENTITY,
        settings,
    )
    .expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let (store, feed) = journaled_at(&LeasePlace::Process, 0, 2);
    let service = Service::new(
        vec![Scope::Gateway],
        Arc::new(
            Shared::new(
                ID.to_owned(),
                ADDRESS.to_owned(),
                &elector,
                bed::unserved(stood),
                bed::followed((store, feed)),
                (Arc::new(StageRegistry::builtin()), &LeasePlace::Process),
                2,
            )
            .expect("the deciders start"),
        ),
    );

    // The other replica holds the lease under term 1; this one's look
    // saw it and is published.
    assert_eq!(
        other.tick(Instant::now()).expect("the take"),
        TickOutcome::Acquired { term: 1 }
    );
    assert_eq!(
        elector.tick(Instant::now()).expect("a look"),
        TickOutcome::Following {
            holder: OTHER.to_owned(),
            term: 1
        }
    );
    standing.send_replace(saw(OTHER, 1));
    assert_eq!(
        service.leader_info(Instant::now()),
        answer("controller-1", "10.0.0.8:30100", 1, false)
    );

    // The other renews no more. This replica's take lands later than
    // the renew deadline after its step began: the fence holds term 2,
    // closed at birth. Until the look is published the answer names
    // nobody: the other never held term 2.
    thread::sleep(settings.duration + Duration::from_millis(20));
    assert_eq!(
        elector.tick(Instant::now()).expect("the take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(
        service.leader_info(Instant::now()),
        answer("", "", 2, false)
    );

    // Published, the look names this replica behind its closed fence:
    // nobody still, under the same term.
    standing.send_replace(saw(IDENTITY, 2));
    assert_eq!(
        service.leader_info(Instant::now()),
        answer("", "", 2, false)
    );
}

#[test]
fn a_replica_whose_fence_is_open_names_itself_under_the_fences_term() {
    let (mut elector, standing, service) = service(&[Scope::Gateway]);
    // Before any look: nobody, at no term.
    assert_eq!(
        service.leader_info(Instant::now()),
        answer("", "", 0, false)
    );

    // The lease held no record: the take opens no fence, and the answer
    // names nobody under the take's term until the first renewal a renew
    // deadline later. The instants are ahead of the clock: the wait is
    // dated by the later of the two.
    let took = Instant::now() + ElectionSettings::default().renew_period;
    assert_eq!(
        elector.tick(took).expect("the take"),
        TickOutcome::Acquired { term: 1 }
    );
    standing.send_replace(saw(IDENTITY, 1));
    assert_eq!(service.leader_info(took), answer("", "", 1, false));
    let taken = took + ElectionSettings::default().renew_deadline;
    assert_eq!(
        elector.tick(taken).expect("the renewal"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(service.leader_info(taken), answer(ID, ADDRESS, 1, true));

    // Past the renew deadline the fence is closed: the record still
    // names this replica, the answer names nobody and keeps the term.
    let late = taken + ElectionSettings::default().renew_deadline + Duration::from_millis(1);
    assert_eq!(service.leader_info(late), answer("", "", 1, false));

    // The answer reads the fence, never the standing: a standing that
    // lags the take still answers as the leader.
    standing.send_replace(ElectionObservation::default());
    assert_eq!(service.leader_info(taken), answer(ID, ADDRESS, 1, true));
}

#[test]
fn a_replica_that_does_not_lead_refuses_a_write_in_the_form_every_client_decodes() {
    let (_elector, standing, service) = service(&[Scope::Control]);
    standing.send_replace(saw("controller-1@10.0.0.8:30100", 4));
    let refused = service
        .leading(Instant::now())
        .expect_err("this replica follows");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::FailedPrecondition, "this replica does not lead")
    );
    // The details are a `google.rpc.Status`: the code and the message
    // again, and the `NotLeader` under its type's URL.
    let details = RpcStatus::decode(refused.details()).expect("a google.rpc.Status");
    assert_eq!(
        (details.code, details.message.as_str()),
        (
            Code::FailedPrecondition as i32,
            "this replica does not lead"
        )
    );
    let [detail] = details.details.as_slice() else {
        panic!("one detail: {details:?}");
    };
    assert_eq!(
        detail.type_url,
        "type.googleapis.com/smg.policy.v1.NotLeader"
    );
    let named = proto::NotLeader::decode(detail.value.as_slice()).expect("a NotLeader");
    assert_eq!(
        named,
        proto::NotLeader {
            leader_address: "10.0.0.8:30100".to_owned(),
            successor_id: String::new(),
        }
    );
}

#[test]
fn a_follower_names_the_holder_its_last_look_saw() {
    let (_elector, standing, service) = service(&[Scope::Control]);
    let now = Instant::now();

    standing.send_replace(saw("controller-1@10.0.0.8:30100", 4));
    assert_eq!(
        service.leader_info(now),
        answer("controller-1", "10.0.0.8:30100", 4, false)
    );

    // A holder that advertises no address is named by its id alone.
    standing.send_replace(saw("controller-9", 5));
    assert_eq!(
        service.leader_info(now),
        answer("controller-9", "", 5, false)
    );

    // A free lease, and a look that failed: nobody, under the highest
    // term seen.
    standing.send_replace(saw("", 5));
    assert_eq!(service.leader_info(now), answer("", "", 5, false));
    standing.send_replace(ElectionObservation {
        holder: String::new(),
        term: 5,
        looked: false,
    });
    assert_eq!(service.leader_info(now), answer("", "", 5, false));
}

#[test]
fn a_follower_says_its_lag_and_reads_synced_under_the_bound() {
    let (mut elector, standing, service) = service(&[Scope::Replication]);
    standing.send_replace(saw("controller-1@10.0.0.8:30100", 4));
    let at = Instant::now();
    // No frame applied yet: no sync state.
    let info = service.leader_info(at);
    assert!(!info.is_leader && !info.synced && info.sync.is_none());
    service.shared.lag.applied(at);
    // The lag holds the frame's instant at a whole microsecond: read
    // from that instant, the bound is met to the nanosecond.
    let at = at - service.shared.lag.age(at).expect("a frame applied");
    let within = service.shared.lag.within();
    let lag = |info: &proto::LeaderInfo| info.sync.map(|sync| sync.replication_lag_ms);
    let under = service.leader_info(at + within - Duration::from_millis(1));
    assert!(under.synced);
    assert_eq!(lag(&under), u32::try_from(within.as_millis() - 1).ok());
    // The bound itself is not under it.
    let at_it = service.leader_info(at + within);
    assert!(!at_it.synced);
    assert_eq!(lag(&at_it), u32::try_from(within.as_millis()).ok());

    // A replica that leads follows nobody, whatever frame it applied
    // before its take: it is not synced and says no sync state. A take
    // of a lease nobody held leads from its first renewal one renew
    // deadline later.
    let took = Instant::now();
    elector.tick(took).expect("the take");
    // The wait runs from the take's own read, a moment after `took`.
    let wait = ElectionSettings::default().renew_deadline + Duration::from_secs(1);
    let now = took + wait;
    elector.tick(now).expect("the renewal");
    service.shared.lag.applied(now);
    let led = service.leader_info(now);
    assert!(led.is_leader && !led.synced && led.sync.is_none());
}

#[tokio::test]
async fn leader_answers_on_every_listener_and_for_every_model() {
    for scopes in [
        &[Scope::Gateway][..],
        &[Scope::Control],
        &[Scope::Replication],
    ] {
        let (_elector, standing, service) = service(scopes);
        standing.send_replace(saw("controller-1@10.0.0.8:30100", 4));
        for model in ["", "model-a"] {
            let asked = Request::new(proto::LeaderRequest {
                model: model.to_owned(),
            });
            let info = service.leader(asked).await.expect("served").into_inner();
            assert_eq!(
                info,
                answer("controller-1", "10.0.0.8:30100", 4, false),
                "{scopes:?} {model:?}"
            );
        }
    }
}

/// A controller with a spec source serves the read call on the control
/// scope: the source's bytes under the standing number, to a caller with
/// a principal. A number that does not stand is not found, by name, and
/// the scope's rules stand as for every call.
#[tokio::test]
async fn the_read_call_answers_the_standing_document_once_a_source_applied_one() {
    let (_elector, _standing, control) = service(&[Scope::Control]);
    let (_elector, _standing, gateway) = service(&[Scope::Gateway]);
    for service in [&control, &gateway] {
        service
            .shared
            .generations
            .apply(LEAST_LOAD.into(), Format::Yaml)
            .expect("a valid document");
    }
    let read = |generation| from("operator-a", proto::GetPolicySpecRequest { generation });

    let answer = control
        .get_policy_spec(read(0))
        .await
        .expect("the standing document")
        .into_inner();
    assert_eq!(answer.generation, 1);
    assert_eq!(answer.document, LEAST_LOAD.as_bytes());
    assert_eq!(answer.format, "yaml");
    assert!(answer.digest.starts_with("sha256:"), "{}", answer.digest);

    let older = control.get_policy_spec(read(4)).await.expect_err("refused");
    assert_eq!(older.code(), Code::NotFound);
    assert_eq!(
            older.message(),
            "no document is held for generation 4: the spec source's document is generation 1, generation 1 stands, and the standing generation's document alone is retained"
        );

    let unplaced = control
        .get_policy_spec(Request::new(proto::GetPolicySpecRequest::default()))
        .await
        .expect_err("refused");
    assert_eq!(unplaced.code(), Code::Unauthenticated);
    let denied = gateway.get_policy_spec(read(0)).await.expect_err("refused");
    assert_eq!(denied.code(), Code::PermissionDenied);
}

/// A control listener's service over an election no case outlasts (an
/// hour's deadline), with no spec source: the leader where `leads`,
/// else a follower whose last look saw `controller-1` hold the lease.
fn electing(leads: bool) -> (Elector, watch::Sender<ElectionObservation>, Service) {
    electing_in(leads, StageRegistry::builtin())
}

/// [`electing`]'s service, its documents naming the stages of `catalog`.
fn electing_in(
    leads: bool,
    catalog: StageRegistry,
) -> (Elector, watch::Sender<ElectionObservation>, Service) {
    let hour = Duration::from_secs(3600);
    let settings = ElectionSettings {
        duration: 2 * hour,
        renew_period: Duration::from_secs(1),
        renew_deadline: hour,
    };
    let elector = Elector::alone(IDENTITY, settings).expect("the settings are in order");
    let (mut elector, standing, control) = over_in(elector, &[Scope::Control], catalog);
    if leads {
        elector.lead_at(Instant::now());
        // The store enters the term, as the election has it at a take: a
        // write call is answered under a term the store stands in.
        control.shared.store.set_term(1);
        standing.send_replace(saw(IDENTITY, 1));
    } else {
        standing.send_replace(saw("controller-1@10.0.0.8:30100", 4));
    }
    (elector, standing, control)
}

const EXPLAIN_GATEWAY: &str = "gateway-a";

fn explain_request(id: u8) -> Request<proto::ExplainRequest> {
    from(
        "operator-a",
        proto::ExplainRequest {
            key: Some(proto::explain_request::Key::RequestId(vec![id])),
            gateway_id: EXPLAIN_GATEWAY.to_owned(),
        },
    )
}

fn explain_spec(service: &Service, profile: &str, sample: u32) {
    let document = format!(
            "apiVersion: smg.policy/v1\nprofiles:\n  a-other: {{pick: random}}\n  {profile}: {{filter: [healthy], score: [least-load], pick: best}}\nbindings:\n  - match: {{model: other}}\n    profile: a-other\n  - profile: {profile}\nexplain: {{sample: {sample}}}"
        );
    service
        .shared
        .generations
        .apply(document.into_bytes(), Format::Yaml)
        .expect("a valid document");
}

async fn decide_for_explain(service: &Service, request: proto::DecideRequest) -> proto::Decision {
    let (answers, mut received) = mpsc::channel(1);
    let job = Job {
        request: Checked::new(request).expect("a frame"),
        gateway: gateway_id(EXPLAIN_GATEWAY),
        received: Instant::now(),
        answers,
    };
    assert!(service.shared.deciders.submit(job).is_none());
    let answered = tokio::time::timeout(Duration::from_secs(5), received.recv())
        .await
        .expect("an answer in time")
        .expect("an answer")
        .frame();
    match answered.answer {
        Some(proto::decide_answer::Answer::Decision(decision)) => decision,
        other => panic!("a decision: {other:?}"),
    }
}

#[tokio::test]
async fn explain_echoes_the_key_and_decision_and_names_only_the_standing_generation() {
    let (_elector, _standing, control) = electing(true);
    let _ = bed::fleet(&control.shared.store, &control.shared.objects);
    explain_spec(&control, "z-first", 1);

    for (model, id) in [(bed::MODEL, 1), ("empty", 2)] {
        let frame = bed::frame_for(model, id, u64::from(id));
        let mut decided = decide_for_explain(&control, frame).await;
        assert_eq!(!decided.leases.is_empty(), model == bed::MODEL);
        decided.leases.clear();
        let record = control
            .explain(explain_request(id))
            .await
            .expect("a hit")
            .into_inner();
        assert_eq!(record.request_id, [id]);
        assert_eq!(record.gateway_id, EXPLAIN_GATEWAY);
        assert_eq!((record.generation, record.binding), (1, 1));
        assert_eq!(record.decision, Some(decided));
        assert!(record.at_unix_micros > 0);
        // A model no worker serves is shed before a profile is chosen:
        // its record names none.
        let served = model == bed::MODEL;
        let (profile, picker): (&str, &str) = if served {
            ("z-first", "best")
        } else {
            ("", "")
        };
        let (filters, scorers): (&[&str], &[&str]) = if served {
            (&["healthy"], &["least-load"])
        } else {
            (&[], &[])
        };
        assert_eq!(record.profile_name, profile);
        assert_eq!(record.picker_name, picker);
        assert_eq!(record.filter_names, filters);
        assert_eq!(record.scorer_names, scorers);
        if served {
            assert_eq!(record.plan_name, "single");
        }
        assert_eq!((record.decision_id, record.view_digest), (0, 0));
        assert!(record.candidates.is_empty() && record.projections.is_empty());
        assert!(record.request.is_none() && record.outcome.is_none());
    }

    let mut wrong_gateway = explain_request(1);
    "gateway-b".clone_into(&mut wrong_gateway.get_mut().gateway_id);
    let miss = control
        .explain(wrong_gateway)
        .await
        .expect_err("a different gateway");
    assert_eq!(miss.code(), Code::NotFound);

    explain_spec(&control, "z-second", 0);
    let _ = decide_for_explain(&control, bed::frame(3, 3)).await;
    let miss = control
        .explain(explain_request(3))
        .await
        .expect_err("sampling off");
    assert_eq!(miss.code(), Code::NotFound);
    let old = control
        .explain(explain_request(1))
        .await
        .expect("an older hit")
        .into_inner();
    assert_eq!((old.generation, old.binding), (1, 1));
    assert_eq!(
        old.decision
            .expect("the decision")
            .trace
            .expect("the trace")
            .profile,
        1
    );
    assert!(old.profile_name.is_empty() && old.plan_name.is_empty() && old.picker_name.is_empty());
    assert!(old.filter_names.is_empty() && old.scorer_names.is_empty());
}

#[tokio::test]
async fn explain_names_only_the_plan_for_a_shed_ahead_of_the_first_profile() {
    let (_elector, _standing, control) = electing(true);
    let _ = bed::fleet(&control.shared.store, &control.shared.objects);
    explain_spec(&control, "a-first", 1);

    let mut frame = bed::frame(1, 1);
    frame.constraints = Some(proto::Constraints {
        subset_hint: Some(0),
        ..proto::Constraints::default()
    });
    let decision = decide_for_explain(&control, frame).await;
    assert!(matches!(
        decision.outcome,
        Some(proto::decision::Outcome::Shed(shed)) if shed.emptied_by == "subset-hint"
    ));

    let record = control
        .explain(explain_request(1))
        .await
        .expect("the sampled shed")
        .into_inner();
    assert_eq!(record.plan_name, "single");
    assert!(record.profile_name.is_empty(), "{}", record.profile_name);
    assert!(record.picker_name.is_empty());
    assert!(record.filter_names.is_empty());
    assert!(record.scorer_names.is_empty());
    let trace = record
        .decision
        .and_then(|decision| decision.trace)
        .expect("the trace");
    assert_eq!((trace.profile, trace.no_profile), (0, true));
}

#[tokio::test]
async fn explain_refuses_unsupported_or_incomplete_keys_by_field() {
    use proto::explain_request::Key;

    let (_elector, _standing, control) = electing(true);
    for (key, gateway, field) in [
        (Some(Key::DecisionId(0)), "", "decision_id"),
        (None, EXPLAIN_GATEWAY, "request_id"),
        (Some(Key::RequestId(vec![])), EXPLAIN_GATEWAY, "request_id"),
        (
            Some(Key::RequestId(vec![0; 257])),
            EXPLAIN_GATEWAY,
            "request_id",
        ),
        (Some(Key::RequestId(vec![1])), "", "gateway_id"),
    ] {
        let request = from(
            "operator-a",
            proto::ExplainRequest {
                key,
                gateway_id: gateway.to_owned(),
            },
        );
        let refused = control.explain(request).await.expect_err("an invalid key");
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert_eq!(refused.message().split(':').next(), Some(field));
    }
}

#[tokio::test]
async fn explain_is_for_the_leader_and_a_new_term_starts_empty() {
    let (mut elector, standing, control) = electing(true);
    explain_spec(&control, "z-first", 1);
    let _ = decide_for_explain(&control, bed::frame_for("empty", 1, 1)).await;
    assert!(control.explain(explain_request(1)).await.is_ok());
    let no_principal = control
        .explain(Request::new(explain_request(1).into_inner()))
        .await;
    assert_eq!(
        no_principal.expect_err("the call needs a principal").code(),
        Code::Unauthenticated
    );
    let gateway = Service::new(vec![Scope::Gateway], Arc::clone(&control.shared));
    assert_eq!(
        gateway
            .explain(explain_request(1))
            .await
            .expect_err("control scope only")
            .code(),
        Code::PermissionDenied
    );

    assert!(elector.release().expect("the release"));
    standing.send_replace(saw("controller-1@10.0.0.8:30100", 2));
    let mut invalid = explain_request(1);
    invalid.get_mut().key = None;
    for request in [explain_request(1), invalid] {
        let lost = control
            .explain(request)
            .await
            .expect_err("this replica follows");
        assert_eq!(lost.code(), Code::FailedPrecondition);
        let details = RpcStatus::decode(lost.details()).expect("a google.rpc.Status");
        let [detail] = details.details.as_slice() else {
            panic!("one detail: {details:?}");
        };
        let named = proto::NotLeader::decode(detail.value.as_slice()).expect("a NotLeader");
        assert_eq!(named.leader_address, "10.0.0.8:30100");
    }

    elector.lead_at(Instant::now());
    control.shared.store.set_term(2);
    let miss = control
        .explain(explain_request(1))
        .await
        .expect_err("a new lead");
    assert_eq!(miss.code(), Code::NotFound);
    let _ = decide_for_explain(&control, bed::frame_for("empty", 2, 2)).await;
    assert!(control.explain(explain_request(2)).await.is_ok());
    let miss = control
        .explain(explain_request(1))
        .await
        .expect_err("the old record stayed gone");
    assert_eq!(miss.code(), Code::NotFound);
}

/// [`electing`]'s service, its spec source a file's document:
/// `LEAST_LOAD`, applied as generation 1.
fn sourced(leads: bool) -> (Elector, watch::Sender<ElectionObservation>, Service) {
    let (elector, standing, control) = electing(leads);
    control
        .shared
        .generations
        .apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    (elector, standing, control)
}

/// An `ApplyPolicySpec` call from `operator-a`.
fn apply(
    mode: impl Into<i32>,
    document: impl Into<Vec<u8>>,
    format: &str,
    expected_digest: &str,
) -> Request<proto::ApplyRequest> {
    from(
        "operator-a",
        proto::ApplyRequest {
            document: document.into(),
            format: format.to_owned(),
            mode: mode.into(),
            expected_digest: expected_digest.to_owned(),
        },
    )
}

/// A document that would run: valid, and not the standing one.
const RANDOM: &str =
    "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    pick: random\n";
/// The same under a filter no catalog holds: the validator refuses it.
const NOWHERE: &str =
    "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [nowhere]\n    pick: random\n";

/// A dry run on the leader answers its document's verdict and applies
/// nothing: accepted, with the number the document would run under now
/// and its digest, or refused, with the standing number and digest and
/// one issue by its field. Either answer reads `read_only`: the source
/// is a file.
#[tokio::test]
async fn a_dry_run_answers_its_documents_verdict_and_applies_nothing() {
    let (_elector, _standing, control) = sourced(true);
    let standing = control.shared.generations.standing();
    let standing_digest = standing.1.expect("a digest").to_string();

    let would = control
        .apply_policy_spec(apply(proto::ApplyMode::DryRun, RANDOM, "yaml", ""))
        .await
        .expect("a verdict")
        .into_inner();
    assert!(would.accepted && would.read_only, "{would:?}");
    assert_eq!(would.generation, 2);
    assert!(would.digest.starts_with("sha256:"), "{would:?}");
    assert_ne!(would.digest, standing_digest);
    assert!(would.issues.is_empty(), "{would:?}");
    // The feed before and after: the standing document reads the loads
    // in every role, and the offered one reads nothing.
    let feed = would.needs.expect("the needs diff");
    let before = feed.before.expect("the feed before");
    assert_eq!(before.models.len(), 4, "{before:?}");
    assert!(
        before
            .models
            .iter()
            .all(|row| row.model == "*" && row.loads),
        "{before:?}"
    );
    assert_eq!(feed.after, Some(proto::NeedsSummary::default()));

    // A guard that names the standing digest passes, in JSON as in YAML.
    let json = br#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"pick":"random"}}}"#;
    let guarded = control
        .apply_policy_spec(apply(
            proto::ApplyMode::DryRun,
            json.as_slice(),
            "json",
            &standing_digest,
        ))
        .await
        .expect("a verdict")
        .into_inner();
    assert!(guarded.accepted, "{guarded:?}");
    assert_eq!(guarded.generation, 2, "the first dry run reserved nothing");

    // A document sent as JSON is read as JSON: its refusal says so.
    let unread = control
        .apply_policy_spec(apply(proto::ApplyMode::DryRun, "{", "json", ""))
        .await
        .expect("a verdict")
        .into_inner();
    let [issue] = unread.issues.as_slice() else {
        panic!("one issue: {unread:?}")
    };
    assert!(
        issue.message.starts_with("not a policy spec in json: "),
        "{issue:?}"
    );

    let refused = control
        .apply_policy_spec(apply(proto::ApplyMode::DryRun, NOWHERE, "yaml", ""))
        .await
        .expect("a verdict")
        .into_inner();
    assert!(!refused.accepted && refused.read_only, "{refused:?}");
    assert!(refused.needs.is_none(), "{refused:?}");
    assert_eq!(
        (refused.generation, refused.digest.as_str()),
        (1, standing_digest.as_str())
    );
    let [issue] = refused.issues.as_slice() else {
        panic!("one issue: {refused:?}")
    };
    assert_eq!(issue.field_path, "profiles.default.filter");
    assert_eq!(
        issue.message,
        "profile \"default\": unknown filter \"nowhere\""
    );

    assert_eq!(control.shared.generations.standing(), standing);
}

/// A dry run a registered filter's factory refuses names the parameter's
/// own field: the path an operator edits.
#[tokio::test]
async fn a_dry_run_names_the_field_a_registered_filters_factory_refuses() {
    /// Keeps the workers with `slots` of their eight request slots free.
    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct MinFree {
        slots: u32,
    }

    impl crate::engine::Filter for MinFree {
        fn name(&self) -> &'static str {
            "min-free"
        }

        fn keep(
            &self,
            _: &crate::engine::RequestFacts<'_>,
            candidate: &crate::engine::Candidate,
            _: &crate::engine::Constraints<'_>,
        ) -> bool {
            candidate.in_flight.saturating_add(self.slots) <= 8
        }
    }

    const FIELD: &str = "profiles.default.filter[1].min-free.slots";
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter_with("min-free", |build| {
            let rule: MinFree = build.params()?;
            if rule.slots == 0 {
                return Err(crate::stages::StageError::bad(
                    "slots",
                    rule.slots,
                    "above zero",
                ));
            }
            Ok(Box::new(rule))
        })
        .expect("a free name");
    let (_elector, _standing, control) = electing_in(true, catalog);
    control
        .shared
        .generations
        .apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    let standing = control.shared.generations.standing();
    let dry_run = |slots: u32| {
        let document = format!(
                "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy, {{ min-free: {{ slots: {slots} }} }}]\n    pick: random\n"
            );
        apply(proto::ApplyMode::DryRun, document, "yaml", "")
    };

    let would = control
        .apply_policy_spec(dry_run(2))
        .await
        .expect("a verdict")
        .into_inner();
    assert!(would.accepted, "{would:?}");

    let refused = control
        .apply_policy_spec(dry_run(0))
        .await
        .expect("a verdict")
        .into_inner();
    assert!(!refused.accepted, "{refused:?}");
    let [issue] = refused.issues.as_slice() else {
        panic!("one issue: {refused:?}")
    };
    assert_eq!(issue.field_path, FIELD);
    assert_eq!(issue.message, format!("{FIELD} = 0 is not above zero"));
    assert_eq!(control.shared.generations.standing(), standing);
}

/// The write call refuses a document whose bindings differ on the
/// overload switch, at a dry run and at an enforce alike: one issue by
/// the field of the first binding that differs, in the refusal's words,
/// and the standing generation stays. Bindings that agree are taken.
#[tokio::test]
async fn the_write_call_names_the_binding_that_differs_on_the_overload_switch() {
    const FIELD: &str = "bindings[1].overload.enabled";
    const OFF: &str = "    overload:\n      enabled: false\n";
    let (_dir, _elector, _standing, control) = stored(true);
    let write = |mode: proto::ApplyMode, second: &str| {
        let head = "apiVersion: smg.policy/v1\nprofiles:\n  p:\n    pick: random\nbindings:\n";
        let first = format!("  - match:\n      model: chat/*\n    profile: p\n{OFF}");
        let document = format!("{head}{first}  - profile: p\n{second}");
        control.apply_policy_spec(apply(mode, document, "yaml", ""))
    };
    let agreed = write(proto::ApplyMode::Enforce, OFF).await;
    let agreed = agreed.expect("an answer").into_inner();
    assert!(agreed.accepted, "{agreed:?}");
    assert_eq!(agreed.generation, 1);
    for mode in [proto::ApplyMode::DryRun, proto::ApplyMode::Enforce] {
        let refused = write(mode, "").await.expect("an answer").into_inner();
        assert!(!refused.accepted && !refused.read_only, "{refused:?}");
        assert_eq!(refused.generation, 1, "the standing generation stays");
        let [issue] = refused.issues.as_slice() else {
            panic!("one issue: {refused:?}")
        };
        assert_eq!(issue.field_path, FIELD);
        assert_eq!(
                issue.message,
                format!("{FIELD} = true is not the false bindings[0] says, by its block or the block's default: a controller builds its workers with one default set of overload thresholds, so the bindings of its document say one word")
            );
    }
    assert_eq!(control.shared.generations.generation(), 1);
}

/// The spec's source is a file, which no call writes: an enforce and a
/// rollback answer `read_only` with the standing number and digest,
/// whatever the document, and validate nothing.
#[tokio::test]
async fn a_write_under_a_file_source_answers_read_only() {
    let (_elector, _standing, control) = sourced(true);
    let (generation, digest) = control.shared.generations.standing();
    let read_only = proto::ApplyResult {
        accepted: false,
        generation,
        digest: digest.expect("a digest").to_string(),
        read_only: true,
        ..proto::ApplyResult::default()
    };
    for document in [RANDOM, "no document at all: ["] {
        let answer = control
            .apply_policy_spec(apply(proto::ApplyMode::Enforce, document, "yaml", ""))
            .await
            .expect("an answer")
            .into_inner();
        assert_eq!(answer, read_only);
    }
    let rollback = control
        .rollback_policy_spec(from("operator-a", proto::RollbackRequest { generation: 1 }))
        .await
        .expect("an answer")
        .into_inner();
    assert_eq!(rollback, read_only);
    assert_eq!(control.shared.generations.standing().0, 1);
}

/// The lease is taken before the store is told its term. In between a
/// spec write, its dry run and a rollback are refused as a registration
/// is, `UNAVAILABLE`, and the standing generation does not move; each
/// answers once the store stands in the term.
#[tokio::test]
async fn a_spec_call_is_refused_until_the_store_stands_in_its_term() {
    // The dry run's kit before the store enters the term: the take of a
    // candidate alone, its fence open.
    let (mut elector, standing, control) = sourced(false);
    elector.lead_at(Instant::now());
    standing.send_replace(saw(IDENTITY, 1));
    let stands = control.shared.generations.standing();
    let calls = async || {
        [
            control
                .apply_policy_spec(apply(proto::ApplyMode::Enforce, RANDOM, "yaml", ""))
                .await,
            control
                .apply_policy_spec(apply(proto::ApplyMode::DryRun, RANDOM, "yaml", ""))
                .await,
            control
                .rollback_policy_spec(from("operator-a", proto::RollbackRequest { generation: 1 }))
                .await,
        ]
    };
    for early in calls().await {
        let early = early.expect_err("refused");
        assert_eq!(
            (early.code(), early.message()),
            (Code::Unavailable, super::NOT_IN_TERM)
        );
    }
    assert_eq!(control.shared.generations.standing(), stands);

    control.shared.store.set_term(1);
    for answer in calls().await {
        let answer = answer.expect("an answer").into_inner();
        assert!(answer.read_only, "{answer:?}");
    }
    assert_eq!(control.shared.generations.standing(), stands);
}

/// A take at or below a term the log followed starts no term in it: the
/// replica holds the lease and its store's writes stand under no term.
/// A watch's open, a registration, a patch and a removal are refused
/// then, `UNAVAILABLE`, and the words say that it lasts the lead. The
/// operators' snapshot is refused the same way: its header would name a
/// term the store does not lead. A standby's restore rides its stream,
/// refused below.
#[tokio::test]
async fn a_take_that_started_no_term_serves_no_watch_or_membership_call() {
    for followed in [1, 5] {
        let scopes = [Scope::Gateway, Scope::Control];
        let (mut elector, _standing, service) = service_alone(&scopes);
        // The log followed a term: a restart at a snapshot's place of
        // it. The take is of term 1, and the store is told it.
        let place = Position {
            seq: 40,
            term: followed,
        };
        service
            .shared
            .feed
            .log
            .restart(place)
            .expect("a place the log takes");
        elector.lead_at(Instant::now());
        service.shared.store.set_term(1);
        assert_eq!(service.shared.feed.log.followed(), Some(followed));

        let refused = |answer: Option<tonic::Status>, call: &str| {
            let Some(status) = answer else {
                panic!("{call} was served where the log followed term {followed}");
            };
            assert_eq!(
                (status.code(), status.message()),
                (Code::Unavailable, super::NO_TERM_LED),
                "{call}, the log followed term {followed}"
            );
        };
        let watch = from("gateway-a", proto::WatchWorkersRequest::default());
        refused(service.watch_workers(watch).await.err(), "a watch's open");
        let register = from("operator-a", proto::RegisterWorkerRequest::default());
        refused(
            service.register_worker(register).await.err(),
            "a registration",
        );
        let patch = from("operator-a", proto::PatchWorkerRequest::default());
        refused(service.patch_worker(patch).await.err(), "a patch");
        let removal = from("operator-a", proto::DeregisterWorkerRequest::default());
        refused(service.deregister_worker(removal).await.err(), "a removal");
        let snapshot = from("operator-a", proto::SnapshotRequest::default());
        refused(service.get_snapshot(snapshot).await.err(), "a snapshot");
        assert_eq!(service.shared.feed.log.head(), place);
    }
}

/// A spec write, its dry run and a rollback are refused the same way,
/// and the standing generation does not move.
#[tokio::test]
async fn a_take_that_started_no_term_serves_no_spec_call() {
    let (mut elector, standing, control) = sourced(false);
    let place = Position { seq: 40, term: 1 };
    control
        .shared
        .feed
        .log
        .restart(place)
        .expect("a place the log takes");
    elector.lead_at(Instant::now());
    control.shared.store.set_term(1);
    standing.send_replace(saw(IDENTITY, 1));
    let stands = control.shared.generations.standing();
    let answers = [
        control
            .apply_policy_spec(apply(proto::ApplyMode::Enforce, RANDOM, "yaml", ""))
            .await,
        control
            .apply_policy_spec(apply(proto::ApplyMode::DryRun, RANDOM, "yaml", ""))
            .await,
        control
            .rollback_policy_spec(from("operator-a", proto::RollbackRequest { generation: 1 }))
            .await,
    ];
    for answer in answers {
        let refused = answer.expect_err("refused");
        assert_eq!(
            (refused.code(), refused.message()),
            (Code::Unavailable, super::NO_TERM_LED)
        );
    }
    assert_eq!(control.shared.generations.standing(), stands);
}

/// A standby's stream is refused the same way. Its handler takes the
/// request as a stream of frames, so the call goes over a socket.
#[tokio::test]
async fn a_take_that_started_no_term_serves_no_standbys_stream() {
    let (mut elector, _standing, service) = service_alone(&[Scope::Replication]);
    let place = Position { seq: 40, term: 1 };
    service
        .shared
        .feed
        .log
        .restart(place)
        .expect("a place the log takes");
    elector.lead_at(Instant::now());
    service.shared.store.set_term(1);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a loopback port");
    let addr = listener.local_addr().expect("the bound address");
    // Aborted at its drop, at the end of the case.
    let mut serving = JoinSet::new();
    serving.spawn(
        Server::builder()
            .add_routes(service.routes(Layer::AsTold))
            .serve_with_incoming(TcpIncoming::from(listener)),
    );
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("a uri")
        .connect()
        .await
        .expect("the listener accepts");
    let mut client = Grpc::new(channel);
    client.ready().await.expect("the channel is ready");
    let path = PathAndQuery::from_static("/smg.policy.v1.GrandCentral/Replicate");
    // An empty message opens the stream: the lead is read before a frame.
    let refused = client
        .unary(stating("standby-a"), path, ProstCodec::<(), ()>::default())
        .await
        .expect_err("no stream is served");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::NO_TERM_LED)
    );
}

/// A call admitted under a take its store has since passed is told to
/// retry: the fence read the earlier term, the log leads the later one.
#[tokio::test]
async fn a_call_under_a_take_its_store_has_passed_is_told_to_retry() {
    let (mut elector, _standing, service) = service_alone(&[Scope::Gateway]);
    elector.lead_at(Instant::now());
    service.shared.store.set_term(1);
    // The store is told a later take than the fence reads.
    service.shared.store.set_term(2);
    let watch = from("gateway-a", proto::WatchWorkersRequest::default());
    let Err(refused) = service.watch_workers(watch).await else {
        panic!("a watch was opened under a take that is over");
    };
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::TAKE_OVER)
    );
}

/// The checks of a write call run in one order, each neighbour pair by
/// a call that fails both: the scope before the lead, the lead before
/// the mode, the mode before the source, the source before the guard,
/// the guard before the document.
#[tokio::test]
async fn the_checks_of_a_write_call_run_in_one_order() {
    let (_elector, _standing, follower) = sourced(false);
    let (_elector, _standing, leader) = sourced(true);
    let unstated = proto::ApplyMode::Unspecified;

    // The scope before the lead: a follower refuses a caller with no
    // principal as it refuses one everywhere.
    let unplaced = follower
        .apply_policy_spec(Request::new(proto::ApplyRequest::default()))
        .await
        .expect_err("refused");
    assert_eq!(unplaced.code(), Code::Unauthenticated);

    // The lead before the mode: a follower refuses a write that states
    // no mode by the leader's name, and a rollback with it.
    let not_leading = [
        follower
            .apply_policy_spec(apply(unstated, RANDOM, "yaml", ""))
            .await
            .expect_err("refused"),
        follower
            .rollback_policy_spec(from("operator-a", proto::RollbackRequest::default()))
            .await
            .expect_err("refused"),
    ];
    for refused in not_leading {
        assert_eq!(refused.code(), Code::FailedPrecondition);
        assert_eq!(refused.message(), "this replica does not lead");
    }

    // The mode before the source: a write states its mode, and audit is
    // not served; neither reads the read-only answer of the source.
    for mode in [unstated.into(), 99] {
        let refused = leader
            .apply_policy_spec(apply(mode, RANDOM, "yaml", ""))
            .await
            .expect_err("refused");
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert_eq!(
            refused.message(),
            "mode: a write states its mode, enforce, audit or dry-run"
        );
    }
    let audit = leader
        .apply_policy_spec(apply(proto::ApplyMode::Audit, RANDOM, "yaml", ""))
        .await
        .expect_err("refused");
    assert_eq!(audit.code(), Code::Unimplemented);
    assert_eq!(
        audit.message(),
        "ApplyPolicySpec in audit mode is not served by this build"
    );

    // The source before the guard: an enforce whose guard names another
    // digest is answered read-only; no write could follow it.
    let enforced = leader
        .apply_policy_spec(apply(
            proto::ApplyMode::Enforce,
            RANDOM,
            "yaml",
            "sha256:another",
        ))
        .await
        .expect("an answer")
        .into_inner();
    assert!(enforced.read_only && !enforced.accepted, "{enforced:?}");

    // The guard before the document: a dry run whose guard names
    // another digest is refused whatever its document and its form.
    for (document, format) in [(RANDOM, "yaml"), (NOWHERE, "yaml"), (RANDOM, "toml")] {
        let guarded = leader
            .apply_policy_spec(apply(
                proto::ApplyMode::DryRun,
                document,
                format,
                "sha256:another",
            ))
            .await
            .expect_err("refused");
        assert_eq!(guarded.code(), Code::FailedPrecondition);
        assert!(
            guarded
                .message()
                .starts_with("expected_digest: generation 1 stands with digest sha256:"),
            "{}",
            guarded.message()
        );
    }

    // The document last: its form, then its verdict.
    let form = leader
        .apply_policy_spec(apply(proto::ApplyMode::DryRun, RANDOM, "toml", ""))
        .await
        .expect_err("refused");
    assert_eq!(form.code(), Code::InvalidArgument);
    assert_eq!(form.message(), "format: name yaml or json");
}

/// The call asks through the door's turn: while a dry run's pass stands
/// in the validator a second call's pass does not enter it, and both
/// are answered once the first ends.
#[tokio::test]
async fn a_second_call_of_a_dry_run_waits_its_turn() {
    use crate::service::controller::generations::tests::{gate, gated, Gate, Gated};

    gate!(Here);
    let gate = Here::gate();
    let (_elector, _standing, mut control) = sourced(true);
    let door = gated::<Here>();
    door.apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    Arc::get_mut(&mut control.shared)
        .expect("the service alone holds it")
        .generations = door;
    let run = |document: &'static str| {
        control.apply_policy_spec(apply(proto::ApplyMode::DryRun, document, "yaml", ""))
    };
    gate.arm();
    let beside = async {
        let entered = gate.entered_a_pace_on().await;
        gate.release();
        entered
    };
    let (first, second, entered) = tokio::join!(run(RANDOM), run(NOWHERE), beside);
    assert_eq!(entered, 1, "a second call's pass ran beside the first");
    assert!(first.expect("an answer").into_inner().accepted);
    assert!(!second.expect("an answer").into_inner().accepted);
}

/// A dry run that no place is left for is told to ask again.
#[test]
fn a_dry_run_let_in_nowhere_is_told_to_ask_again() {
    use crate::service::controller::generations::Unanswered;

    let full = unanswered(Unanswered::Full);
    assert_eq!(full.code(), Code::ResourceExhausted);
    assert_eq!(
        full.message(),
        "32 dry runs wait their turn: ask again later"
    );
}

/// [`electing`]'s service, its spec source a store over a new, empty
/// directory, handed back beside it.
fn stored(
    leads: bool,
) -> (
    tempfile::TempDir,
    Elector,
    watch::Sender<ElectionObservation>,
    Service,
) {
    let dir = tempfile::tempdir().expect("a temp dir");
    let (elector, standing, control) = electing(leads);
    let opened = SpecStore::open(dir.path().to_owned()).expect("the directory opens");
    control
        .shared
        .generations
        .bind_store(opened)
        .expect("the store binds");
    (dir, elector, standing, control)
}

/// An enforce under a spec store, on the leader: the document is
/// persisted and runs as the next generation, and the answer says so:
/// accepted, the number, the digest, the feed before and after, and not
/// `read_only`. The read call answers the bytes. A document that does
/// not run answers the standing number and one issue, a guard that
/// names another digest is refused, and a dry run answers what an
/// enforce would with nothing applied.
#[tokio::test]
async fn an_enforce_under_a_spec_store_runs_as_the_next_generation() {
    let (dir, _elector, _standing, control) = stored(true);
    let on_file = || names(dir.path()).len();
    let read = || {
        let standing = proto::GetPolicySpecRequest::default();
        control.get_policy_spec(from("operator-a", standing))
    };
    let write = |mode: proto::ApplyMode, document: &'static str, expected: &str| {
        control.apply_policy_spec(apply(mode, document, "yaml", expected))
    };
    // No generation yet: the spec calls are served, and the read has
    // no document to answer.
    let none = read().await.expect_err("no document yet");
    assert_eq!(none.code(), Code::NotFound);

    let first = write(proto::ApplyMode::Enforce, RANDOM, "")
        .await
        .expect("an answer")
        .into_inner();
    assert!(first.accepted && !first.read_only, "{first:?}");
    assert_eq!(first.generation, 1);
    assert!(first.digest.starts_with("sha256:"), "{first:?}");
    assert!(first.issues.is_empty(), "{first:?}");
    let nothing = Some(proto::NeedsSummary::default());
    let feed = first.needs.clone().expect("the needs diff");
    assert_eq!((&feed.before, &feed.after), (&nothing, &nothing));
    assert_eq!((control.shared.generations.generation(), on_file()), (1, 1));
    let standing = read().await.expect("the standing document").into_inner();
    assert_eq!(
        (standing.generation, standing.document.as_slice()),
        (1, RANDOM.as_bytes())
    );
    assert_eq!(standing.digest, first.digest);

    // A document that does not run: the standing number and digest,
    // and one issue by its field.
    let refused = write(proto::ApplyMode::Enforce, NOWHERE, &first.digest)
        .await
        .expect("an answer")
        .into_inner();
    assert!(!refused.accepted && !refused.read_only, "{refused:?}");
    assert_eq!((refused.generation, &refused.digest), (1, &first.digest));
    let [issue] = refused.issues.as_slice() else {
        panic!("one issue: {refused:?}")
    };
    assert!(!issue.field_path.is_empty(), "{issue:?}");
    assert!(issue.message.contains("nowhere"), "{issue:?}");
    assert!(refused.needs.is_none(), "{refused:?}");

    // A guard that names another digest than the standing one: another
    // apply landed first.
    let other = format!("sha256:{}", "0".repeat(64));
    let late = write(proto::ApplyMode::Enforce, LEAST_LOAD, &other)
        .await
        .expect_err("guarded");
    assert_eq!(late.code(), Code::FailedPrecondition);
    assert_eq!(
        late.message(),
        format!(
            "expected_digest: generation 1 stands with digest {}",
            first.digest
        )
    );

    // A dry run answers what an enforce would and applies nothing.
    let dry = write(proto::ApplyMode::DryRun, LEAST_LOAD, &first.digest)
        .await
        .expect("a verdict")
        .into_inner();
    assert!(dry.accepted && !dry.read_only, "{dry:?}");
    assert_eq!(dry.generation, 2);
    assert_eq!((control.shared.generations.generation(), on_file()), (1, 1));

    let second = write(proto::ApplyMode::Enforce, LEAST_LOAD, &first.digest)
        .await
        .expect("an answer")
        .into_inner();
    assert!(second.accepted, "{second:?}");
    assert_eq!((second.generation, on_file()), (2, 2));
    let feed = second.needs.expect("the needs diff");
    assert_eq!(feed.before, nothing);
    assert!(
        feed.after.is_some_and(|after| !after.models.is_empty()),
        "the standing document reads the loads"
    );
}

/// A follower takes no write under a spec store either: it refuses an
/// enforce and a rollback by the leader's name, and nothing is on file.
#[tokio::test]
async fn a_follower_refuses_a_write_under_a_spec_store() {
    let (dir, _elector, _standing, control) = stored(false);
    let refused = control
        .apply_policy_spec(apply(proto::ApplyMode::Enforce, RANDOM, "yaml", ""))
        .await
        .expect_err("a follower");
    assert_eq!(refused.code(), Code::FailedPrecondition);
    assert_eq!(refused.message(), "this replica does not lead");
    let rollback = from("operator-a", proto::RollbackRequest { generation: 1 });
    let refused = control
        .rollback_policy_spec(rollback)
        .await
        .expect_err("a follower");
    assert_eq!(refused.code(), Code::FailedPrecondition);
    assert_eq!(refused.message(), "this replica does not lead");
    assert_eq!(names(dir.path()).len(), 0);
    assert_eq!(control.shared.generations.generation(), 0);
}

/// A rollback under a spec store, on the leader: the document of a
/// generation on file runs again as the next generation, and the answer
/// is an enforce's. The read call answers each generation on file by
/// its number. A rollback that names no generation is refused by the
/// field, and a generation that is not on file is not found, by a
/// rollback and by the read call alike.
#[tokio::test]
async fn a_rollback_under_a_spec_store_runs_a_generation_on_file_again() {
    let (dir, _elector, _standing, control) = stored(true);
    let write = |document: &'static str| {
        control.apply_policy_spec(apply(proto::ApplyMode::Enforce, document, "yaml", ""))
    };
    let rollback = |generation: u64| {
        let request = proto::RollbackRequest { generation };
        control.rollback_policy_spec(from("operator-a", request))
    };
    let read = |generation: u64| {
        let request = proto::GetPolicySpecRequest { generation };
        control.get_policy_spec(from("operator-a", request))
    };
    let first = write(RANDOM).await.expect("an answer").into_inner();
    let second = write(LEAST_LOAD).await.expect("an answer").into_inner();
    assert_eq!((first.generation, second.generation), (1, 2));

    let back = rollback(1).await.expect("an answer").into_inner();
    assert!(back.accepted && !back.read_only, "{back:?}");
    assert_eq!((back.generation, &back.digest), (3, &first.digest));
    assert!(back.issues.is_empty(), "{back:?}");
    let feed = back.needs.expect("the needs diff");
    assert!(
        feed.before.is_some_and(|before| !before.models.is_empty()),
        "the document rolled away from reads the loads"
    );
    assert_eq!(feed.after, Some(proto::NeedsSummary::default()));
    assert_eq!(control.shared.generations.generation(), 3);
    assert_eq!(names(dir.path()).len(), 3);

    for (asked, answered, document, digest) in [
        (1, 1, RANDOM, &first.digest),
        (2, 2, LEAST_LOAD, &second.digest),
        (3, 3, RANDOM, &first.digest),
        (0, 3, RANDOM, &first.digest),
    ] {
        let held = read(asked).await.expect("a document").into_inner();
        assert_eq!(held.generation, answered, "asked for {asked}");
        assert_eq!(
            (held.document.as_slice(), &held.digest, held.format.as_str()),
            (document.as_bytes(), digest, "yaml")
        );
    }

    let unnamed = rollback(0).await.expect_err("no generation is named");
    assert_eq!(unnamed.code(), Code::InvalidArgument);
    assert_eq!(
        unnamed.message(),
        "generation: a rollback names a generation on file"
    );
    let gone = [
        rollback(9).await.expect_err("not on file"),
        read(9).await.expect_err("not on file"),
    ];
    for refused in gone {
        assert_eq!(refused.code(), Code::NotFound);
        assert_eq!(
                refused.message(),
                "no document is held for generation 9: the spec store has 3 on file, from generation 1 to generation 3"
            );
    }
    assert_eq!(control.shared.generations.generation(), 3);
}

/// A `GetSnapshot` call from `operator-a` for `slices`.
fn snapshot(slices: &[proto::snapshot_request::Slice]) -> Request<proto::SnapshotRequest> {
    from(
        "operator-a",
        proto::SnapshotRequest {
            slices: slices.iter().map(|slice| i32::from(*slice)).collect(),
            ..Default::default()
        },
    )
}

/// The operators' snapshot on the leader, under a spec store: the
/// header, the rows as an operator reads them and the spec's status. A
/// row registered under a URL with userinfo is told by the id and the
/// revision a patch names, its address without the userinfo. The status
/// lists the generation a rollback can name. A slice that was not asked
/// for is absent, and the answer names no place of the log.
#[tokio::test]
async fn the_leader_serves_the_snapshot_of_its_rows_and_its_spec() {
    use proto::snapshot_request::Slice;

    let (_dir, _elector, _standing, control) = stored(true);
    for document in [LEAST_LOAD, RANDOM] {
        let ran = control
            .apply_policy_spec(apply(proto::ApplyMode::Enforce, document, "yaml", ""))
            .await
            .expect("a verdict")
            .into_inner();
        assert!(ran.accepted, "{ran:?}");
    }
    let url = "grpc://operator-name:s3cret-word@10.0.0.5:8000";
    let spec = proto::WorkerSpec {
        url: url.to_owned(),
        wire: proto::WireMode::Grpc as i32,
        role: proto::Role::Single as i32,
        models: vec!["model-a".to_owned()],
        ..Default::default()
    };
    let request = proto::RegisterWorkerRequest {
        spec: Some(spec),
        ..Default::default()
    };
    let registered = control
        .register_worker(from("operator-a", request))
        .await
        .expect("registered")
        .into_inner();
    let Some(proto::worker_event::Event::Registered(stated)) = registered.event else {
        panic!("a registration's event: {registered:?}");
    };
    // The registration's answer tells its caller the URL it stated.
    assert_eq!(stated.url, url);

    let answer = control
        .get_snapshot(snapshot(&[Slice::Workers, Slice::Spec]))
        .await
        .expect("a snapshot")
        .into_inner();
    assert_eq!(
        (
            answer.term,
            answer.generation,
            answer.controller_id.as_str()
        ),
        (1, 2, ID)
    );
    let [row] = answer.workers.as_slice() else {
        panic!("one row: {answer:?}");
    };
    // The snapshot tells the row to any operator: by what a patch names
    // it, and without the userinfo.
    assert_eq!((row.id, row.revision), (stated.id, stated.revision));
    let clean = "grpc://10.0.0.5:8000";
    assert_eq!(
        (
            row.url.as_str(),
            row.registration_key.as_str(),
            row.engine.as_str()
        ),
        (clean, clean, clean)
    );
    assert!(!answer.truncated);
    let status = answer.spec.expect("the spec slice");
    let standing = control.shared.generations.standing();
    assert_eq!(status.generation, 2);
    assert_eq!(status.digest, standing.1.expect("a digest").to_string());
    assert_eq!(status.retained, [1]);
    assert_eq!(status.source, "api");
    assert_eq!(status.mode, proto::ApplyMode::Enforce as i32);
    assert!(status.applied_unix_micros > 0, "{status:?}");
    // Redacted, and no base a standby starts from.
    assert_eq!(
        (
            answer.replication_seq,
            answer.replication_term,
            answer.membership_epoch
        ),
        (0, 0, 0)
    );
    assert!(answer.ids.is_none() && answer.documents.is_empty());

    // One slice asked for: the other is absent.
    let rows = control
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect("the rows")
        .into_inner();
    assert_eq!(
        (rows.workers.len(), rows.spec, rows.generation),
        (1, None, 2)
    );
    let status = control
        .get_snapshot(snapshot(&[Slice::Spec]))
        .await
        .expect("the status")
        .into_inner();
    assert!(status.workers.is_empty() && status.spec.is_some());
}

/// A replica that does not lead serves no snapshot: it refuses as it
/// refuses a watch's open, with the `NotLeader` that names the holder,
/// before it reads the request's fields.
#[tokio::test]
async fn a_replica_that_does_not_lead_serves_no_snapshot() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = sourced(false);
    for request in [
        snapshot(&[Slice::Workers, Slice::Spec]),
        // A request no leader serves either: the lead is read first.
        from("operator-a", proto::SnapshotRequest::default()),
    ] {
        let refused = control.get_snapshot(request).await.expect_err("refused");
        assert_eq!(
            (refused.code(), refused.message()),
            (Code::Unavailable, super::FOLLOWS_NO_LOG)
        );
        let details = RpcStatus::decode(refused.details()).expect("a google.rpc.Status");
        let [detail] = details.details.as_slice() else {
            panic!("one detail: {details:?}");
        };
        let named = proto::NotLeader::decode(detail.value.as_slice()).expect("a NotLeader");
        assert_eq!(named.leader_address, "10.0.0.8:30100");
    }
}

/// On the leader a snapshot is refused by its request's fields: one
/// that names no slice, a slice this build does not serve, identifiers
/// in clear. A leader with no spec source serves its rows, and refuses
/// the spec slice by its name.
#[tokio::test]
async fn the_leader_refuses_a_snapshot_by_its_requests_field() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = electing(true);
    let every = from("operator-a", proto::SnapshotRequest::default());
    let refused = control.get_snapshot(every).await.expect_err("refused");
    assert_eq!(refused.code(), Code::Unimplemented);
    assert!(
        refused
            .message()
            .starts_with("slices: none named is every slice"),
        "{}",
        refused.message()
    );
    let health = control
        .get_snapshot(snapshot(&[Slice::Workers, Slice::Health]))
        .await
        .expect_err("refused");
    assert_eq!(
        (health.code(), health.message()),
        (
            Code::Unimplemented,
            "slices: this build does not serve SLICE_HEALTH: it serves SLICE_WORKERS, SLICE_SPEC"
        )
    );
    let plain = proto::SnapshotRequest {
        plain_identifiers: true,
        ..snapshot(&[Slice::Workers]).into_inner()
    };
    let refused = control
        .get_snapshot(from("operator-a", plain))
        .await
        .expect_err("refused");
    assert_eq!(refused.code(), Code::Unimplemented);
    assert!(
        refused.message().starts_with("plain_identifiers: "),
        "{}",
        refused.message()
    );

    // No spec source: the rows alone.
    let spec = control
        .get_snapshot(snapshot(&[Slice::Workers, Slice::Spec]))
        .await
        .expect_err("refused");
    assert_eq!(
        (spec.code(), spec.message()),
        (
            Code::Unimplemented,
            "slices: SLICE_SPEC is not served by a controller started with no spec source"
        )
    );
    let rows = control
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect("the rows")
        .into_inner();
    assert_eq!((rows.term, rows.generation, rows.workers.len()), (1, 0, 0));
    assert_eq!(rows.spec, None);
}

/// One snapshot is taken at a time, and no more calls are let in than
/// the turn has places: the call past them is told to ask again. A call
/// refused by the lead or by its request's field takes no place.
#[tokio::test]
async fn a_snapshot_past_the_ones_that_wait_is_told_to_ask_again() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = electing(true);
    let places = control.shared.readers.snapshots().fill();
    assert_eq!(places.len(), super::snapshot::SNAPSHOTS_WAITING + 1);
    let past = control
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect_err("no place is left");
    assert_eq!(
        (past.code(), past.message()),
        (
            Code::ResourceExhausted,
            "32 snapshots wait their turn: ask again later"
        )
    );
    // A request refused by its field is refused by it, with no place.
    let health = control
        .get_snapshot(snapshot(&[Slice::Health]))
        .await
        .expect_err("refused");
    assert_eq!(health.code(), Code::Unimplemented);
    // So is the spec slice of a controller with no spec source: the
    // source is read ahead of the turn.
    let spec = control
        .get_snapshot(snapshot(&[Slice::Spec]))
        .await
        .expect_err("refused");
    assert_eq!(
        (spec.code(), spec.message()),
        (
            Code::Unimplemented,
            "slices: SLICE_SPEC is not served by a controller started with no spec source"
        )
    );
    drop(places);
    let rows = control
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect("a place is free again");
    assert_eq!(rows.into_inner().term, 1);
    assert_eq!(control.shared.readers.snapshots().let_in(), 0);

    // A replica that does not lead refuses by its lead, places or none.
    let (_elector, _standing, follower) = electing(false);
    let _places = follower.shared.readers.snapshots().fill();
    let refused = follower
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect_err("refused");
    assert_eq!(refused.code(), Code::Unavailable);
}

/// A snapshot that states no limit and no models reads the rows as a
/// watch's start does. It shares the cut the feed keeps at the log's
/// head and keeps the one it takes, so calls one behind another copy
/// the rows once a write, and the call behind a write tells it. The
/// last call out lets the cut go where no watch is open: a lone call's
/// cut goes with it, and a call beside another that is let in keeps it
/// for that one.
#[tokio::test]
async fn a_snapshot_shares_the_feeds_cut_and_keeps_none_with_no_watch_open() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = electing(true);
    let feed = control.shared.feed.clone();
    let told = async || {
        control
            .get_snapshot(snapshot(&[Slice::Workers]))
            .await
            .expect("the rows")
            .into_inner()
            .workers
            .len()
    };
    // No watch is open: the lone call's cut goes with it.
    assert_eq!(told().await, 0);
    assert!(feed.kept().is_none(), "no reader can share the cut");

    // Another call is let in beside it: the cut stays for that one, the
    // next call shares it, and the last call out lets it go.
    let beside = control
        .shared
        .readers
        .snapshots()
        .place()
        .expect("a place is free");
    assert_eq!(told().await, 0);
    let cut = feed.kept().expect("a call is let in to share the cut");
    assert_eq!(told().await, 0);
    assert!(
        Arc::ptr_eq(&cut, &feed.kept().expect("still kept")),
        "one copy for the calls let in"
    );
    drop(beside);
    assert_eq!(told().await, 0);
    assert!(feed.kept().is_none(), "the last call out let it go");

    // A watch is open: the snapshot shares the cut its start took.
    let gateways = Service::new(vec![Scope::Gateway], Arc::clone(&control.shared));
    let mut watching = gateways
        .watch_workers(from("gateway-a", proto::WatchWorkersRequest::default()))
        .await
        .expect("opened")
        .into_inner();
    tokio::time::timeout(Duration::from_secs(5), watching.next())
        .await
        .expect("a page within five seconds")
        .expect("the stream is open")
        .expect("a page, not a status");
    let cut = feed.kept().expect("the watch's cut is kept");
    assert_eq!(told().await, 0);
    assert!(
        Arc::ptr_eq(&cut, &feed.kept().expect("still kept")),
        "the snapshot shared the watch's cut"
    );

    // A write moves the log past that cut: the next snapshot tells the
    // row and keeps the cut it took, and the call behind it shares it.
    let request = proto::RegisterWorkerRequest {
        spec: Some(proto::WorkerSpec {
            url: "grpc://10.0.0.5:8000".to_owned(),
            wire: proto::WireMode::Grpc as i32,
            role: proto::Role::Single as i32,
            models: vec!["model-a".to_owned()],
            ..Default::default()
        }),
        ..Default::default()
    };
    control
        .register_worker(from("operator-a", request))
        .await
        .expect("registered");
    assert_eq!(told().await, 1);
    let later = feed.kept().expect("the snapshot's cut is kept");
    assert!(!Arc::ptr_eq(&cut, &later), "a cut of the later place");
    assert_eq!(told().await, 1);
    assert!(
        Arc::ptr_eq(&later, &feed.kept().expect("still kept")),
        "one copy a write"
    );

    // The last watch leaves, and the next snapshot keeps no cut.
    drop(watching);
    assert_eq!(told().await, 1);
    assert!(feed.kept().is_none(), "no watch can share the cut");
}

/// A lone call with no watch open that states a limit or its models
/// takes no cut of every row: it reads the rows it may tell. One that
/// states neither would copy every row anyway, so it takes the feed's
/// cut, which a call let in while it runs shares. Beside another call
/// each takes the cut the two share, once: the first of them takes it,
/// whatever it states. So does a call beside an open watch, where no cut
/// stands at the log's head: the watch can share it.
#[tokio::test]
async fn a_lone_snapshot_takes_a_cut_only_where_it_tells_every_row() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = electing(true);
    let feed = control.shared.feed.clone();
    let models = ["model-a", "model-a", "model-a", "model-b"];
    for (worker, model) in models.into_iter().enumerate() {
        let request = proto::RegisterWorkerRequest {
            spec: Some(proto::WorkerSpec {
                url: format!("grpc://10.0.0.{worker}:8000"),
                wire: proto::WireMode::Grpc as i32,
                role: proto::Role::Single as i32,
                models: vec![model.to_owned()],
                ..Default::default()
            }),
            ..Default::default()
        };
        control
            .register_worker(from("operator-a", request))
            .await
            .expect("registered");
    }
    let told = async |limit| {
        let mut request = snapshot(&[Slice::Workers]);
        request.get_mut().limit = limit;
        let answer = control
            .get_snapshot(request)
            .await
            .expect("the rows")
            .into_inner();
        (answer.workers.len(), answer.truncated)
    };
    assert_eq!(told(1).await, (1, true));
    assert_eq!(feed.cuts(), 0, "a lone call for one row copied every row");
    assert_eq!(told(0).await, (4, false));
    assert_eq!(feed.cuts(), 1, "a call for every row takes the cut");
    assert!(feed.kept().is_none(), "no reader can share the cut");

    // A lone call for one model with no limit reads that model's rows
    // alone, and takes no cut.
    let mut request = snapshot(&[Slice::Workers]);
    request.get_mut().models = vec!["model-a".to_owned()];
    let answer = control
        .get_snapshot(request)
        .await
        .expect("the rows")
        .into_inner();
    assert_eq!((answer.workers.len(), answer.truncated), (3, false));
    assert_eq!(feed.cuts(), 1, "a lone call for its models took the cut");

    // Another call is let in beside it: the call for one row takes the
    // cut the two share, and the call behind it shares that one.
    let beside = control
        .shared
        .readers
        .snapshots()
        .place()
        .expect("a place is free");
    assert_eq!(told(1).await, (1, true));
    assert_eq!(feed.cuts(), 2, "a call beside another takes their cut");
    assert_eq!(told(0).await, (4, false));
    assert_eq!(feed.cuts(), 2, "one copy for the calls let in");
    drop(beside);

    // A watch is open, and no cut stands at the log's head: the call
    // for one row takes the cut the watch can share, and keeps it.
    assert_eq!(told(0).await, (4, false));
    assert!(feed.kept().is_none(), "the last call out let the cut go");
    let gateways = Service::new(vec![Scope::Gateway], Arc::clone(&control.shared));
    let watching = gateways
        .watch_workers(from("gateway-a", proto::WatchWorkersRequest::default()))
        .await
        .expect("opened");
    assert_eq!(told(1).await, (1, true));
    assert_eq!(feed.cuts(), 3, "a call beside an open watch takes its cut");
    assert!(feed.kept().is_some(), "kept for the watch");
    drop(watching);
}

/// A snapshot that waited its turn through a later take of the lease is
/// told to retry: the lead is read again behind the wait, and no answer
/// is told under a term the store has since passed.
#[tokio::test]
async fn a_snapshot_that_waited_through_a_later_take_is_told_to_retry() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = electing(true);
    let control = Arc::new(control);
    let held = control.shared.readers.snapshots().hold().await;
    let mut calls = JoinSet::new();
    calls.spawn({
        let control = Arc::clone(&control);
        async move { control.get_snapshot(snapshot(&[Slice::Workers])).await }
    });
    // The call is let in and waits behind the held turn.
    let began = Instant::now();
    while control.shared.readers.snapshots().let_in() < 1 {
        assert!(
            began.elapsed() < Duration::from_secs(20),
            "the call took no place"
        );
        tokio::task::yield_now().await;
    }
    // The store is told a later take than the call was admitted under.
    control.shared.store.set_term(2);
    drop(held);
    let refused = calls
        .join_next()
        .await
        .expect("the one call")
        .expect("the call's task")
        .expect_err("a later take");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::TAKE_OVER)
    );
}

/// A snapshot admitted under one term waits its turn while the replica
/// gives its lease up and takes it again, its store told the new term.
/// The fence and the store both stand in the later term: the lead read
/// again names a term, and it is not the call's, so the compare itself
/// tells the call to retry.
#[tokio::test]
async fn a_snapshot_that_waited_through_a_take_its_store_follows_is_told_to_retry() {
    use proto::snapshot_request::Slice;

    let (mut elector, _standing, control) = electing(true);
    let control = Arc::new(control);
    let held = control.shared.readers.snapshots().hold().await;
    let mut calls = JoinSet::new();
    calls.spawn({
        let control = Arc::clone(&control);
        async move { control.get_snapshot(snapshot(&[Slice::Workers])).await }
    });
    let began = Instant::now();
    while control.shared.readers.snapshots().let_in() < 1 {
        assert!(
            began.elapsed() < Duration::from_secs(20),
            "the call took no place"
        );
        tokio::task::yield_now().await;
    }
    let now = Instant::now();
    assert!(elector.step_down(None, now).expect("the release"));
    assert_eq!(
        elector.tick(now).expect("the take"),
        TickOutcome::Acquired { term: 2 }
    );
    control.shared.store.set_term(2);
    assert_eq!(control.watched(Instant::now()).ok(), Some(2));
    drop(held);
    let refused = calls
        .join_next()
        .await
        .expect("the one call")
        .expect("the call's task")
        .expect_err("a later take");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::TAKE_OVER)
    );
}

#[test]
fn a_write_call_that_found_no_place_is_told_to_ask_again() {
    let full = unanswered(Unanswered::Waiting);
    assert_eq!(full.code(), Code::ResourceExhausted);
    assert_eq!(
        full.message(),
        "32 applies wait their turn: ask again later"
    );
}
