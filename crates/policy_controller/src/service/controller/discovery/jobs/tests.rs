use std::{sync::Arc, time::Duration};

use smg_worker::{
    jobs::{
        queue::{JobOutcomeError, JobQueue, JobQueueConfig},
        WorkerJob, WorkerJobAsker, WorkerRegistrationMode,
    },
    worker::{HealthCheckConfig, WorkerSpec, WorkerUpdateRequest},
};
use tonic::Status;

use super::{
    super::tests::grpc_engine, Jobs, Registration, WorkerKind, WorkflowId, ENDED_RUN_KEPT,
    WORKER_REGISTRATION_WORKFLOW,
};
use crate::{
    service::controller::{
        decision_slots::tests::holder,
        election::LeasePlace,
        membership::{MembershipCalls, ReplicaMembershipWriter},
        principal::Principal,
        replication::journaled_from,
    },
    store::{self, MemoryStore, Registrant, WorkerStatus, WorkersStore},
};

const TERM: u64 = 3;
/// Past a removal's settle window, which is the worker crate's default.
const WAIT: Duration = Duration::from_secs(30);

/// A queue over a host as a controller builds it ([`host`]).
fn queue(leads: bool) -> (Arc<MemoryStore>, Arc<Jobs>, Arc<JobQueue<Jobs>>) {
    let (store, jobs) = host(leads, ENDED_RUN_KEPT);
    let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&jobs));
    (store, jobs, queue)
}

/// A host whose jobs write at the replica's own door, over a store that
/// stands in [`TERM`]. The door leads that term where `leads`, and refuses
/// as a replica that does not lead otherwise. An ended run's state stands
/// for `kept` ahead of its sweep.
fn host(leads: bool, kept: Duration) -> (Arc<MemoryStore>, Arc<Jobs>) {
    let (store, feed) = journaled_from(&LeasePlace::Process, 0);
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let objects = holder(&store);
    let calls = MembershipCalls::based(Arc::clone(&store), feed.log, 0, Arc::clone(&objects));
    let leading = move || {
        if leads {
            Ok(TERM)
        } else {
            Err(Status::failed_precondition("this replica does not lead"))
        }
    };
    let door =
        ReplicaMembershipWriter::new(Arc::new(calls), Principal::replica("controller-a"), leading);
    let jobs = Jobs::swept(&objects, Arc::new(door), kept).expect("the two definitions");
    (store, jobs)
}

/// The registration a discovery pass asks for the worker at `url`.
fn found(url: &str) -> WorkerJob {
    WorkerJob::Add {
        spec: Box::new(WorkerSpec::new(url)),
        mode: WorkerRegistrationMode::Upsert,
        asker: WorkerJobAsker::Discovery,
    }
}

/// A registration dials the worker and writes its row at the replica's own
/// door, made by the provider `discovery`. The replica's worker objects
/// hold the row when the job answers. A removal marks the row draining and
/// takes it out behind the worker's settle window.
#[tokio::test]
async fn a_registration_writes_the_workers_row_and_a_removal_takes_it_out() {
    let (url, _engine_stop, _engine) = grpc_engine("served").await;
    let (store, jobs, queue) = queue(true);

    queue
        .submit_and_wait(found(&url), WAIT)
        .await
        .expect("the registration");
    let row = store.by_key(&url).expect("the worker's row");
    assert!(
        matches!(&row.maker, Registrant::Provider(by) if by.provider() == "discovery"),
        "{:?}",
        row.maker
    );
    assert_eq!(row.facts.models, ["served"]);
    assert!(jobs.registry.get_by_url(&url).is_some(), "the row's object");

    let gone = WorkerJob::Remove {
        url: url.clone(),
        expected_revisions: None,
        asker: WorkerJobAsker::Discovery,
    };
    let leaving = queue.submit_and_wait(gone, WAIT);
    let draining = async {
        loop {
            match store.by_key(&url).map(|row| row.status) {
                Some(WorkerStatus::Draining) => break true,
                None => break false,
                Some(_) => tokio::time::sleep(Duration::from_millis(10)).await,
            }
        }
    };
    let (left, drained) = tokio::join!(leaving, draining);
    left.expect("the removal");
    assert!(drained, "the row left with no drain");
    assert!(store.by_key(&url).is_none());
    assert!(jobs.registry.get_by_url(&url).is_none());
}

/// The door reads the term at the write. A job whose replica leads no more
/// by then fails with the door's refusal, and no row is written.
#[tokio::test]
async fn a_registration_behind_a_lost_lead_writes_nothing() {
    let (url, _engine_stop, _engine) = grpc_engine("served").await;
    let (store, _jobs, queue) = queue(false);
    let written = store.epoch();

    let refused = queue.submit_and_wait(found(&url), WAIT).await;
    assert!(
        matches!(&refused, Err(JobOutcomeError::Failed(why)) if why.contains("does not lead")),
        "{refused:?}"
    );
    assert!(store.by_key(&url).is_none());
    assert_eq!(store.epoch(), written);
}

/// No asker of a controller sends an update or a labels refresh. A queue
/// that is sent one answers that it runs none, and writes nothing.
#[tokio::test]
async fn an_update_and_a_labels_refresh_are_refused() {
    const URL: &str = "http://worker:8000";
    let (store, _jobs, queue) = queue(true);
    let written = store.epoch();
    let asker = WorkerJobAsker::Health;
    for (job, kind) in [
        (
            WorkerJob::Update {
                url: URL.to_owned(),
                update: Box::new(WorkerUpdateRequest {
                    priority: Some(1),
                    cost: None,
                    labels: None,
                    api_key: None,
                    health: None,
                }),
                asker: asker.clone(),
            },
            WorkerJob::UPDATE_TYPE,
        ),
        (
            WorkerJob::RefreshLabels {
                url: URL.to_owned(),
                asker: asker.clone(),
            },
            WorkerJob::REFRESH_LABELS_TYPE,
        ),
    ] {
        let refused = queue.submit_and_wait(job, WAIT).await;
        let said = format!("a controller's queue runs no {kind} job");
        assert!(
            matches!(&refused, Err(JobOutcomeError::Failed(why)) if why.contains(&said)),
            "{refused:?}"
        );
    }
    assert_eq!(store.epoch(), written);
}

/// The README names what a controller fixes for a registration: a starting
/// worker is polled every 30 seconds for 1800 seconds from the job's start,
/// over HTTP/1.1, and its row states the worker crate's default checks.
#[test]
fn a_registrations_settings_are_the_ones_the_readme_names() {
    let fixed = super::settings();
    let timing = (
        fixed.startup_timeout_secs,
        fixed.startup_delay_secs,
        fixed.startup_check_interval_secs,
    );
    assert_eq!(timing, (1800, 0, 30));
    assert!(!fixed.upstream_http2);
    assert_eq!(fixed.health, HealthCheckConfig::default());
}

/// A job's wait can end ahead of its run, and nobody then takes the run's
/// state at its end. The host's engines sweep it. Here no job waits for a
/// registration of a worker nothing serves: the run fails once its probes
/// are spent, and its state is gone at the next sweep.
#[tokio::test(start_paused = true)]
async fn the_state_of_a_run_no_job_waits_for_is_swept_behind_its_end() {
    // An ended run's state is swept at the first sweep behind its end.
    let (_store, jobs) = host(true, Duration::ZERO);
    let free = std::net::TcpListener::bind("127.0.0.1:0").expect("a port");
    let port = free.local_addr().expect("the port's address").port();
    drop(free);
    let mut data = Registration::new(
        WorkerSpec::new(format!("grpc://127.0.0.1:{port}")),
        WorkerRegistrationMode::Upsert,
        Arc::clone(&jobs),
        WorkerJobAsker::Discovery,
    );
    data.worker_kind = Some(WorkerKind::Local);
    let run = jobs
        .registration
        .start_workflow(WorkflowId::new(WORKER_REGISTRATION_WORKFLOW), data)
        .await
        .expect("the run starts");
    assert!(jobs.registration.get_status(run).await.is_ok());

    // The probes are spent well inside two hours of the runtime's clock.
    let swept = async {
        while jobs.registration.get_status(run).await.is_ok() {
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    };
    let gone = tokio::time::timeout(Duration::from_secs(7200), swept).await;
    assert!(gone.is_ok(), "no sweep took the state of the ended run");
}
