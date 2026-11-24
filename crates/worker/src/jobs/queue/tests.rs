use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use super::*;
use crate::{
    jobs::{WorkerJob, WorkerJobs, WorkerRegistrationMode},
    worker::WorkerSpec,
};

#[derive(Clone, Debug)]
enum Work {
    Other(&'static str),
    Removal(&'static str),
}

impl QueueJob for Work {
    const REGISTRATION_TYPE: &'static str = "Other";

    fn job_type(&self) -> &'static str {
        match self {
            Self::Other(_) => "Other",
            Self::Removal(_) => "Removal",
        }
    }

    fn key(&self) -> &str {
        "one-key"
    }

    fn shown_key(_job_type: &str, key: &str) -> String {
        key.to_string()
    }

    fn is_removal(&self) -> bool {
        matches!(self, Self::Removal(_))
    }

    fn merge_removals(held: Self, newer: Self) -> Result<Self, (Self, Self)> {
        Err((held, newer))
    }
}

struct Script<J: QueueJob> {
    started: mpsc::UnboundedSender<J>,
    hold: Mutex<Option<oneshot::Receiver<()>>>,
    fail: bool,
    failures: Mutex<Vec<Option<String>>>,
    queue: OnceLock<Weak<JobQueue<Self>>>,
}

#[async_trait]
impl<J: QueueJob> JobHost for Script<J> {
    type Job = J;
    const NAME: &'static str = "Script";

    async fn execute(self: &Arc<Self>, job: &J, _run: &JobRun<'_>) -> Result<String, String> {
        let hold = self.hold.lock().unwrap().take();
        self.started.send(job.clone()).unwrap();
        if let Some(hold) = hold {
            hold.await.unwrap();
        }
        if self.fail {
            Err("job refused".to_string())
        } else {
            Ok("job done".to_string())
        }
    }

    fn failed(&self, job: &J) {
        let status = self
            .queue
            .get()
            .and_then(Weak::upgrade)
            .and_then(|queue| queue.get_status(job.key()))
            .map(|status| status.status);
        self.failures.lock().unwrap().push(status);
    }
}

struct ScriptQueue<J: QueueJob> {
    host: Arc<Script<J>>,
    queue: Arc<JobQueue<Script<J>>>,
    started: mpsc::UnboundedReceiver<J>,
    release: oneshot::Sender<()>,
}

impl<J: QueueJob> ScriptQueue<J> {
    fn new(fail: bool) -> Self {
        let (events, started) = mpsc::unbounded_channel();
        let (release, hold) = oneshot::channel();
        let host = Arc::new(Script {
            started: events,
            hold: Mutex::new(Some(hold)),
            fail,
            failures: Mutex::default(),
            queue: OnceLock::new(),
        });
        let queue = JobQueue::new(
            JobQueueConfig {
                queue_capacity: 1,
                ..JobQueueConfig::default()
            },
            Arc::downgrade(&host),
        );
        assert!(host.queue.set(Arc::downgrade(&queue)).is_ok());
        Self {
            host,
            queue,
            started,
            release,
        }
    }
}

async fn next<J>(started: &mut mpsc::UnboundedReceiver<J>) -> J {
    tokio::time::timeout(Duration::from_secs(1), started.recv())
        .await
        .expect("the job starts")
        .expect("the host sent a job")
}

#[tokio::test(start_paused = true)]
async fn a_second_job_of_one_key_waits_its_turn() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    assert_eq!(queue.submit_or_defer(Work::Other("first")).await, Ok(true));
    assert!(matches!(next(&mut started).await, Work::Other("first")));
    assert_eq!(queue.submit_or_defer(Work::Other("next")).await, Ok(false));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err()
    );
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Other("next")));
}

#[tokio::test(start_paused = true)]
async fn a_registration_is_refused_while_one_is_in_flight() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    let jobs: &dyn WorkerJobs = &*queue;
    let url = "http://worker:8000";
    let register = || {
        jobs.add_worker(
            WorkerSpec::new(url),
            WorkerRegistrationMode::Upsert,
            WorkerJobAsker::Discovery,
        )
    };
    assert_eq!(register().await, Ok(true));
    assert!(matches!(next(&mut started).await, WorkerJob::Add { .. }));
    assert_eq!(register().await, Ok(false));
    release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while jobs.job_in_flight(url) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("the first registration finished");
    assert_eq!(register().await, Ok(true));
    assert!(matches!(next(&mut started).await, WorkerJob::Add { .. }));
}

#[tokio::test(start_paused = true)]
async fn the_hosts_registration_type_name_guards_its_key() {
    let script = ScriptQueue::new(false);
    let job = Work::Other("registration");
    assert!(script.queue.claim_if_idle(&job));
    assert_eq!(script.queue.submit_unless_registering(job).await, Ok(false));
}

#[tokio::test(start_paused = true)]
async fn a_label_re_read_is_kept_behind_a_job_in_flight_and_runs_after_it() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    let jobs: &dyn WorkerJobs = &*queue;
    let url = "http://worker:8000";
    assert_eq!(
        jobs.add_worker(
            WorkerSpec::new(url),
            WorkerRegistrationMode::Upsert,
            WorkerJobAsker::Discovery,
        )
        .await,
        Ok(true)
    );
    assert!(matches!(next(&mut started).await, WorkerJob::Add { .. }));
    assert_eq!(
        jobs.refresh_worker_labels(url.to_string(), WorkerJobAsker::Health)
            .await,
        Ok(false)
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err()
    );
    release.send(()).unwrap();
    assert!(matches!(
        next(&mut started).await,
        WorkerJob::RefreshLabels { url: seen, .. } if seen == url
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err()
    );
}

#[tokio::test(start_paused = true)]
async fn a_kept_removal_runs_before_a_kept_job_of_another_kind() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    assert_eq!(queue.submit_or_defer(Work::Other("holder")).await, Ok(true));
    assert!(matches!(next(&mut started).await, Work::Other("holder")));
    assert_eq!(queue.submit_or_defer(Work::Other("kept")).await, Ok(false));
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept removal")).await,
        Ok(false)
    );
    release.send(()).unwrap();
    assert!(matches!(
        next(&mut started).await,
        Work::Removal("kept removal")
    ));
    assert!(matches!(next(&mut started).await, Work::Other("kept")));
}

#[tokio::test(start_paused = true)]
async fn a_claim_dropped_before_its_slot_leaves_no_pending_record() {
    let script = ScriptQueue::new(false);
    let queue = &script.queue;
    let held = queue.tx.reserve().await.unwrap();
    let job = Work::Other("blocked");
    let key = job.key().to_string();
    assert!(queue.claim_if_idle(&job));
    let mut submit = Box::pin(queue.submit_claimed(job));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut submit)
        .await
        .is_err());
    assert_eq!(queue.get_status(&key).unwrap().status, "pending");
    drop(submit);
    assert!(queue.get_status(&key).is_none());
    drop(held);
}

#[tokio::test(start_paused = true)]
async fn a_failed_job_calls_the_hosts_failed_hook_before_its_status_turns_terminal() {
    let ScriptQueue {
        host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(true);
    let job = Work::Other("failure");
    let waiting = queue.submit_and_wait(job.clone(), Duration::from_secs(1));
    let keep_a_second_failure = async {
        assert!(matches!(next(&mut started).await, Work::Other("failure")));
        assert_eq!(
            queue.submit_or_defer(Work::Removal("kept failure")).await,
            Ok(false)
        );
        release.send(()).unwrap();
        assert!(matches!(
            next(&mut started).await,
            Work::Removal("kept failure")
        ));
    };
    let (result, ()) = tokio::join!(waiting, keep_a_second_failure);
    assert_eq!(
        result,
        Err(JobOutcomeError::Failed("job refused".to_string()))
    );
    assert_eq!(
        *host.failures.lock().unwrap(),
        [
            Some("processing".to_string()),
            Some("processing".to_string())
        ]
    );
    assert_eq!(queue.get_status(job.key()).unwrap().status, "failed");
    assert_eq!(queue.get_status(job.key()).unwrap().job_type, "Removal");
}

#[tokio::test]
async fn a_mark_takes_the_failure_kept_under_its_id() {
    type Queue = JobQueue<Script<Work>>;
    let statuses = DashMap::new();
    let queue = ScriptQueue::<Work>::new(false).queue;
    let marks = &queue.cancelled_before_start;
    let id = WorkerId::new();
    // A mark set before the job's end: no failure is kept.
    queue.set_cancel_mark(&id);
    Queue::keep_own_failure(&statuses, marks, &id, false, "Other", "one-key", "refused");
    assert!(statuses.is_empty());
    // A mark set after the failure was kept, by a `DELETE` answered while
    // the job ends: the job's end takes the failure with the mark. A second
    // `DELETE` beside it that takes its own mark back leaves the first one's.
    queue.clear_cancel_mark(&id);
    Queue::keep_own_failure(&statuses, marks, &id, false, "Other", "one-key", "refused");
    assert!(statuses.contains_key(id.as_str()));
    queue.set_cancel_mark(&id);
    queue.set_cancel_mark(&id);
    queue.clear_cancel_mark(&id);
    Queue::end_mark(&statuses, marks, &id);
    assert!(statuses.is_empty() && marks.is_empty());
    // The count of a mark that is gone goes when the next mark is set.
    queue.set_cancel_mark(&WorkerId::new());
    assert_eq!(queue.mark_holders.len(), 1);
}

#[tokio::test(start_paused = true)]
async fn the_port_states_the_callers_asker_on_the_job() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    let jobs: &dyn WorkerJobs = &*queue;
    let id = WorkerId::from_string("one".to_string());
    release.send(()).unwrap();
    jobs.add_worker(
        WorkerSpec::new("http://add:8000"),
        WorkerRegistrationMode::Upsert,
        WorkerJobAsker::Startup,
    )
    .await
    .unwrap();
    assert!(matches!(
        next(&mut started).await,
        WorkerJob::Add {
            asker: WorkerJobAsker::Startup,
            ..
        }
    ));

    jobs.refresh_worker_labels("http://refresh:8000".to_string(), WorkerJobAsker::Health)
        .await
        .unwrap();
    assert!(matches!(
        next(&mut started).await,
        WorkerJob::RefreshLabels {
            asker: WorkerJobAsker::Health,
            ..
        }
    ));

    jobs.remove_worker(
        "http://remove:8000".to_string(),
        &id,
        7,
        WorkerJobAsker::Api,
    )
    .await
    .unwrap();
    assert!(matches!(next(&mut started).await,
        WorkerJob::Remove { url, asker: WorkerJobAsker::Api, expected_revisions: Some(pins) }
            if url == "http://remove:8000" && pins == HashMap::from([("one".to_string(), Some(7))])
    ));

    let remove_many = "http://remove-many:8000";
    assert!(queue.claim_if_idle(&WorkerJob::Add {
        spec: Box::new(WorkerSpec::new(remove_many)),
        mode: WorkerRegistrationMode::Upsert,
        asker: WorkerJobAsker::Discovery,
    }));
    jobs.remove_workers_at(
        remove_many.to_string(),
        vec![(id, 8)],
        WorkerJobAsker::Discovery,
    )
    .await
    .unwrap();
    assert!(matches!(next(&mut started).await,
        WorkerJob::Remove { url, asker: WorkerJobAsker::Discovery, expected_revisions: Some(pins) }
            if url == "http://remove-many:8000" && pins == HashMap::from([("one".to_string(), Some(8))])
    ));
}

#[test]
fn worker_jobs_keep_their_status_names_and_address_keys() {
    let url = "http://user:password@worker:8000";
    let jobs = [
        WorkerJob::Add {
            spec: Box::new(WorkerSpec::new(url)),
            mode: WorkerRegistrationMode::Upsert,
            asker: WorkerJobAsker::Api,
        },
        WorkerJob::Update {
            url: url.to_string(),
            update: Box::new(serde_json::from_str("{}").unwrap()),
            asker: WorkerJobAsker::Api,
        },
        WorkerJob::RefreshLabels {
            url: url.to_string(),
            asker: WorkerJobAsker::Health,
        },
        WorkerJob::Remove {
            url: url.to_string(),
            expected_revisions: None,
            asker: WorkerJobAsker::Api,
        },
    ];
    for (job, name) in jobs.into_iter().zip([
        "AddWorker",
        "UpdateWorker",
        "RefreshWorkerLabels",
        "RemoveWorker",
    ]) {
        assert_eq!(job.job_type(), name);
        assert_eq!(job.key(), url);
        assert_eq!(job.is_removal(), name == "RemoveWorker");
        assert_eq!(WorkerJob::shown_key(name, url), "http://***@worker:8000");
    }
}

#[test]
fn a_merged_removal_pins_the_later_revision_of_a_worker_both_pin() {
    type Pins<'a> = &'a [(&'a str, Option<u64>)];
    fn pins(pins: Pins<'_>) -> HashMap<String, Option<u64>> {
        pins.iter()
            .map(|(id, pin)| ((*id).to_string(), *pin))
            .collect()
    }
    fn removal(held: Pins<'_>) -> WorkerJob {
        WorkerJob::Remove {
            url: "http://worker:8000".to_string(),
            expected_revisions: Some(pins(held)),
            asker: WorkerJobAsker::Health,
        }
    }

    // The held job's pins, the newer job's, the merged job's. A revision
    // only grows, so the earlier pin of a replaced worker matches nothing.
    let rows: [(Pins<'_>, Pins<'_>, Pins<'_>); 4] = [
        (
            &[("one", Some(7))],
            &[("one", Some(8))],
            &[("one", Some(8))],
        ),
        (
            &[("one", Some(8))],
            &[("one", Some(7))],
            &[("one", Some(8))],
        ),
        (&[("one", Some(7))], &[("one", None)], &[("one", None)]),
        (
            &[("one", Some(7))],
            &[("two", Some(3))],
            &[("one", Some(7)), ("two", Some(3))],
        ),
    ];
    for (held, newer, merged) in rows {
        let Ok(WorkerJob::Remove {
            expected_revisions: Some(kept),
            ..
        }) = WorkerJob::merge_removals(removal(held), removal(newer))
        else {
            panic!("two pinned removals of one asker merge into a pinned one");
        };
        assert_eq!(kept, pins(merged), "{held:?} then {newer:?}");
    }
}
