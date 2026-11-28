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

mod admission;
mod explain;
mod policy_spec;
mod snapshots;

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
        let row = store.get(id).expect("the row stands");
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
    while shared.leading(Instant::now()).is_none() {
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
    let mut looks = shared.steps();
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
    assert_eq!(shared.leading(Instant::now()), None);
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
    assert_eq!(shared.held_term(), 1, "under its first term");

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
    while shared.leading(Instant::now()).is_none() {
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

/// `message` as a call the listener's layer placed `subject` on.
fn from<T>(subject: &str, message: T) -> Request<T> {
    let mut request = Request::new(message);
    request.extensions_mut().insert(told(subject));
    request
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
    assert_eq!(service.shared.leading(Instant::now()), Some(1));
}

/// A watch that names two models is told the rows of both and no third
/// model's: every name of its request reaches the watch's open.
#[tokio::test]
async fn a_watch_of_two_models_is_told_the_rows_of_both() {
    let (mut elector, _standing, service) = service_alone(&[Scope::Gateway, Scope::Control]);
    elector.lead_at(Instant::now());
    service.shared.store.set_term(1);
    for (port, model) in [(8000, "model-a"), (8001, "model-b"), (8002, "model-c")] {
        let spec = proto::WorkerSpec {
            url: format!("grpc://10.0.0.5:{port}"),
            wire: proto::WireMode::Grpc as i32,
            role: proto::Role::Single as i32,
            models: vec![model.to_owned()],
            ..Default::default()
        };
        let request = proto::RegisterWorkerRequest {
            spec: Some(spec),
            ..Default::default()
        };
        service
            .register_worker(from("operator-a", request))
            .await
            .expect("registered");
    }
    let asked = proto::WatchWorkersRequest {
        models: vec!["model-a".to_owned(), "model-b".to_owned()],
        ..Default::default()
    };
    let mut stream = service
        .watch_workers(from("gateway-a", asked))
        .await
        .expect("opened")
        .into_inner();
    let page = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("a page within five seconds")
        .expect("the stream is open")
        .expect("a page, not a status");
    let Some(proto::worker_event::Event::Snapshot(rows)) = page.event else {
        panic!("a snapshot's page: {page:?}");
    };
    let told: BTreeSet<&str> = rows
        .workers
        .iter()
        .flat_map(|row| row.models.iter().map(String::as_str))
        .collect();
    assert_eq!(told, BTreeSet::from(["model-a", "model-b"]));
    assert!(rows.last, "one page holds the two rows");
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

/// The shared state says the identity its elector elects under, whole: a
/// standby draws its wait between two streams by it. An identity that
/// carries no address is said as it stands.
#[test]
fn the_shared_state_says_the_identity_its_elector_elects_under() {
    let (_elector, _standing, service) = service(&[Scope::Replication]);
    assert_eq!(service.shared.identity(), IDENTITY);
    let alone = Elector::alone("controller-9", PATIENT).expect("the settings are in order");
    let (_elector, _standing, service) = over(alone, &[Scope::Replication]);
    assert_eq!(service.shared.identity(), "controller-9");
}

/// A refused call names its holder by the fence at the call's own
/// instant. A nanosecond ahead of the lead's opening and a nanosecond past
/// its end, the record names this replica behind a closed fence and the
/// refusal names nobody: the instant beside either would name this
/// replica.
#[test]
fn a_refused_call_names_its_holder_by_the_fence_at_the_calls_own_instant() {
    const NANOSECOND: Duration = Duration::from_nanos(1);
    // A candidate alone over its lease: its take opens the lead at the
    // take's own instant, for one renew deadline.
    let (mut elector, standing, service) = service_alone(&[Scope::Control]);
    let opens = Instant::now();
    assert_eq!(
        elector.tick(opens).expect("the take"),
        TickOutcome::Acquired { term: 1 }
    );
    standing.send_replace(saw(IDENTITY, 1));
    let ends = elector.fence().turns_at(opens).expect("the lead ends");
    for (outside, beside) in [(opens - NANOSECOND, opens), (ends + NANOSECOND, ends)] {
        assert_eq!(service.leader_info(beside), answer(ID, ADDRESS, 1, true));
        let refused = service
            .leading(outside)
            .expect_err("the fence is closed at that instant");
        let details = RpcStatus::decode(refused.details()).expect("a google.rpc.Status");
        let [detail] = details.details.as_slice() else {
            panic!("one detail: {details:?}");
        };
        let named = proto::NotLeader::decode(detail.value.as_slice()).expect("a NotLeader");
        assert_eq!(named.leader_address, "", "at {outside:?}");
    }
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

/// The presence table's task runs on with no stop said, and ends at the
/// stop it is handed: the shared state's own word is not said here. The
/// case's clock is paused. An hour of it passes with no stop said: a task
/// that ends by itself inside that hour fails the case. Behind the stop
/// the task has ended with no more of that clock passed: a task that holds
/// its stop unread and ends by itself at a later instant fails there.
#[tokio::test(start_paused = true)]
async fn the_presence_tables_task_ends_at_the_stop_it_is_handed() {
    let (_elector, _standing, service) = service_alone(&[Scope::Gateway]);
    let (stop, stopped) = watch::channel(false);
    let mut task = JoinSet::new();
    task.spawn(service.shared.sweep(stopped));
    let early = tokio::time::timeout(Duration::from_secs(3_600), task.join_next()).await;
    assert!(
        early.is_err(),
        "the task ended with no stop said: {early:?}"
    );
    let said = tokio::time::Instant::now();
    stop.send(true).expect("the task holds the receiver");
    tokio::time::timeout(Duration::from_secs(20), task.join_next())
        .await
        .expect("the task ends within 20 s of its stop")
        .expect("one task")
        .expect("it ended by itself");
    assert_eq!(
        said.elapsed(),
        Duration::ZERO,
        "the task ended in the instant its stop was said"
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
