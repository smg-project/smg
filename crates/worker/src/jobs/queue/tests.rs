use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use super::*;
use crate::{
    jobs::{WorkerJob, WorkerJobs, WorkerRegistrationMode},
    worker::WorkerSpec,
};

thread_local! {
    /// What a test runs once where the queue weighs two removals of a key
    /// (`merge_removals`): when a second one is kept, and when one whose turn
    /// was refused is put back.
    static AT_MERGE: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
}

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
        if let Some(at_merge) = AT_MERGE.take() {
            at_merge();
        }
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

/// A removal kept behind a claim still runs when the claim's caller leaves
/// while it waits for a slot of a full queue: the claim goes back without a
/// job, so no completion is left to run what was kept behind it.
#[tokio::test(start_paused = true)]
async fn a_job_kept_behind_a_claim_dropped_before_its_slot_still_runs() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release: _release,
    } = ScriptQueue::new(false);
    let held = queue.tx.reserve().await.unwrap();
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    drop(given_up);
    drop(held);
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
    // The claim's own job left with its caller: nothing else starts.
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err()
    );
}

/// A claim that goes back after the runtime's end starts nothing and does
/// not panic: the job kept behind it stays kept, as before.
#[test]
fn a_claim_dropped_after_the_runtimes_end_leaves_the_kept_job_in_place() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap();
    let (script, held, given_up) = runtime.block_on(async {
        let script = ScriptQueue::new(false);
        let held = script.queue.tx.clone().reserve_owned().await.unwrap();
        let queue = script.queue.clone();
        let mut given_up =
            Box::pin(async move { queue.submit_or_defer(Work::Removal("given up")).await });
        assert!(
            tokio::time::timeout(Duration::from_millis(1), &mut given_up)
                .await
                .is_err()
        );
        assert_eq!(
            script.queue.submit_or_defer(Work::Removal("kept")).await,
            Ok(false)
        );
        (script, held, given_up)
    });
    drop(runtime);
    drop(given_up);
    assert!(script.queue.kept.contains_key("one-key"));
    assert!(script.queue.get_status("one-key").is_none());
    drop(held);
}

/// A claim made beside a job in flight (`submit_unless_registering` takes the
/// key over that job's entry) starts nothing when it goes back: the job kept
/// behind the one in flight runs after it, never beside it.
#[tokio::test(start_paused = true)]
async fn a_job_kept_behind_a_job_in_flight_waits_for_it_when_another_claim_is_given_up() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    assert_eq!(
        queue.submit_or_defer(Work::Removal("in flight")).await,
        Ok(true)
    );
    assert!(matches!(
        next(&mut started).await,
        Work::Removal("in flight")
    ));
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    let held = queue.tx.reserve().await.unwrap();
    let mut given_up = Box::pin(queue.submit_unless_registering(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    drop(given_up);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "beside"
    );
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
    drop(held);
}

/// A plain removal that waited for a slot ahead of a claim holds the key
/// from its submission. The claim that goes back starts nothing: the job
/// kept behind the claim runs after the job in flight, never beside it.
#[tokio::test(start_paused = true)]
async fn a_claim_given_up_under_a_job_that_took_its_key_starts_nothing_beside_it() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    let held = queue.tx.reserve().await.unwrap();
    let mut plain = Box::pin(queue.submit(Work::Removal("in flight")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut plain)
        .await
        .is_err());
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    drop(held);
    plain.await.unwrap();
    assert!(matches!(
        next(&mut started).await,
        Work::Removal("in flight")
    ));
    drop(given_up);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "beside"
    );
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
}

/// The same order one instant earlier: the plain removal holds its slot and
/// its status reads pending, but no concurrency permit lets it start. The
/// claim that goes back starts nothing here either (the job holds the
/// key), so the kept job runs after the accepted one, never ahead of it.
#[tokio::test(start_paused = true)]
async fn a_claim_given_up_under_an_accepted_job_that_took_its_key_starts_nothing_ahead_of_it() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    let held = queue.tx.reserve().await.unwrap();
    let mut plain = Box::pin(queue.submit(Work::Removal("in flight")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut plain)
        .await
        .is_err());
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    let permits = u32::try_from(queue.concurrency_limit.available_permits()).unwrap();
    let all = queue
        .concurrency_limit
        .clone()
        .acquire_many_owned(permits)
        .await
        .unwrap();
    drop(held);
    plain.await.unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err()
    );
    assert_eq!(queue.get_status("one-key").unwrap().status, "pending");
    drop(given_up);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "ahead"
    );
    drop(all);
    assert!(matches!(
        next(&mut started).await,
        Work::Removal("in flight")
    ));
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
}

/// One step earlier still: the plain removal still waits for its slot. It
/// holds the key from its submission, so the claim that goes back starts
/// nothing ahead of it; it runs once it has its slot, and its end runs the
/// kept job.
#[tokio::test(start_paused = true)]
async fn a_claim_given_up_under_a_plain_job_that_waits_for_its_slot_starts_nothing_ahead_of_it() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    let held = queue.tx.reserve().await.unwrap();
    let mut plain = Box::pin(queue.submit(Work::Removal("waits")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut plain)
        .await
        .is_err());
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    drop(given_up);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "ahead"
    );
    drop(held);
    plain.await.unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("waits")));
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
}

/// The plain removal is dropped before its slot too, behind the claim that
/// went back. It was the last holder of the key and leaves no completion, so
/// the kept job gets its turn behind it.
#[tokio::test(start_paused = true)]
async fn a_plain_job_dropped_before_its_slot_behind_a_claim_given_up_starts_the_kept_job() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release: _release,
    } = ScriptQueue::new(false);
    let held = queue.tx.reserve().await.unwrap();
    let mut plain = Box::pin(queue.submit(Work::Removal("waits")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut plain)
        .await
        .is_err());
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    drop(given_up);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "beside the plain removal that still waits"
    );
    drop(plain);
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
    drop(held);
}

/// Only a claim that held the key alone starts the kept jobs. A claim of
/// another type takes the first removal's entry and goes back, so a later
/// removal claims the vacant URL while the first still waits, and a third is
/// kept behind it. The first claim, given up, starts nothing (the later
/// claim holds the key): the kept removal runs after the later one, never
/// beside it.
#[tokio::test(start_paused = true)]
async fn a_claim_given_up_under_a_later_claim_of_its_type_starts_nothing_beside_it() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    let held = queue.tx.reserve().await.unwrap();
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    let mut other = Box::pin(queue.submit_unless_registering(Work::Other("other")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut other)
        .await
        .is_err());
    drop(other);
    assert!(queue.get_status("one-key").is_none());
    let mut later = Box::pin(queue.submit_or_defer(Work::Removal("later")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut later)
        .await
        .is_err());
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    drop(given_up);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "ahead"
    );
    drop(held);
    assert_eq!(later.await, Ok(true));
    assert!(matches!(next(&mut started).await, Work::Removal("later")));
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
}

/// The same for a later claim that can stand beside a job
/// (`submit_unless_registering`): written over the first claim's entry, it
/// holds the key. The first claim, given up, starts nothing: the kept
/// removal runs after the later claim's job, never ahead of it.
#[tokio::test(start_paused = true)]
async fn a_claim_given_up_under_a_later_claim_beside_it_starts_nothing_ahead_of_it() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    let held = queue.tx.reserve().await.unwrap();
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    let mut later = Box::pin(queue.submit_unless_registering(Work::Removal("later")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut later)
        .await
        .is_err());
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    drop(given_up);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "ahead"
    );
    drop(held);
    assert_eq!(later.await, Ok(true));
    assert!(matches!(next(&mut started).await, Work::Removal("later")));
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
}

/// The same for later claims made idle (`claim_if_idle`): one of another
/// type takes the first removal's entry and goes back, then one of the
/// removal's own type claims the vacant URL and holds it. The
/// first claim, given up, starts nothing: the kept removal runs after the
/// later claim's job, never ahead of it.
#[tokio::test(start_paused = true)]
async fn a_claim_given_up_under_a_later_claim_made_idle_starts_nothing_ahead_of_it() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    let held = queue.tx.reserve().await.unwrap();
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert!(queue.claim_if_idle(&Work::Other("other")));
    let mut other = Box::pin(queue.submit_claimed(Work::Other("other")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut other)
        .await
        .is_err());
    drop(other);
    assert!(queue.get_status("one-key").is_none());
    assert!(queue.claim_if_idle(&Work::Removal("later")));
    let mut later = Box::pin(queue.submit_claimed(Work::Removal("later")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut later)
        .await
        .is_err());
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    drop(given_up);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "ahead"
    );
    drop(held);
    assert_eq!(later.await, Ok(()));
    assert!(matches!(next(&mut started).await, Work::Removal("later")));
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
}

/// A kept job that a job's end starts holds the URL while it runs. A removal
/// runs unseen (a job of another type, submitted plainly under its URL, took
/// the entry and ended),
/// so a later removal claims the URL and waits for a slot, and two jobs are
/// kept behind it. The first removal's end takes the waiting claim's entry
/// and starts the kept removal. The waiting claim, given up, starts nothing
/// beside it: the other kept job runs after the kept removal.
#[tokio::test(start_paused = true)]
async fn a_claim_given_up_beside_a_kept_job_a_jobs_end_started_starts_nothing_beside_it() {
    let ScriptQueue {
        host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    queue.submit(Work::Removal("unseen")).await.unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("unseen")));
    queue.submit(Work::Other("through")).await.unwrap();
    assert!(matches!(next(&mut started).await, Work::Other("through")));
    tokio::time::sleep(Duration::from_millis(1)).await;
    assert!(queue.get_status("one-key").is_none());
    let held = queue.tx.reserve().await.unwrap();
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    assert_eq!(
        queue.submit_or_defer(Work::Other("kept beside")).await,
        Ok(false)
    );
    let (release_kept, hold) = oneshot::channel();
    *host.hold.lock().unwrap() = Some(hold);
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
    drop(given_up);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "beside"
    );
    release_kept.send(()).unwrap();
    assert!(matches!(
        next(&mut started).await,
        Work::Other("kept beside")
    ));
    drop(held);
}

/// On a many-thread runtime the kept jobs' turn can begin on another thread
/// the moment it is spawned, so the claim's entry must be gone by then. The
/// claim goes back on a second thread while this one holds the entry's lock:
/// no turn begins under the entry (a turn holds the kept jobs' lock while it
/// claims the URL), and once the entry is free the kept removal runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_job_kept_behind_a_claim_dropped_on_another_thread_starts_once_the_claims_entry_is_gone()
{
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release: _release,
    } = ScriptQueue::new(false);
    let held = queue.tx.clone().reserve_owned().await.unwrap();
    let mut given_up = Box::pin({
        let queue = queue.clone();
        async move { queue.submit_or_defer(Work::Removal("given up")).await }
    });
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    let entry = queue.status_map.get("one-key").expect("the claim's entry");
    let dropped = tokio::task::spawn_blocking(move || drop(given_up));
    let turn_began = (0..50).any(|_| {
        std::thread::sleep(Duration::from_millis(2));
        queue.kept.try_get("one-key").is_locked()
    });
    drop(entry);
    assert!(!turn_began, "a turn began under the claim's entry");
    dropped.await.unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
    drop(held);
}

/// The count of holders shows a job the status entry no longer shows.
/// A removal runs unseen (a job of another type took its entry and ended). A
/// later removal claims the URL and waits, a removal is kept behind it and
/// the claim is given up: the unseen removal still holds the URL, so nothing
/// starts beside it, and its end runs the kept removal.
#[tokio::test(start_paused = true)]
async fn a_claim_given_up_beside_an_unseen_job_starts_nothing_beside_it() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    queue.submit(Work::Removal("unseen")).await.unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("unseen")));
    queue.submit(Work::Other("through")).await.unwrap();
    assert!(matches!(next(&mut started).await, Work::Other("through")));
    tokio::time::sleep(Duration::from_millis(1)).await;
    assert!(queue.get_status("one-key").is_none());
    let held = queue.tx.reserve().await.unwrap();
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    drop(given_up);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "beside"
    );
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
    drop(held);
}

/// An older claim still waits for a slot, its entry taken and given back by
/// a claim of another type. A later claim of its type claims the URL and
/// waits, a removal is kept and the later claim is given up: the older claim
/// still holds the URL, so nothing starts ahead of its job, and that job's
/// end runs the kept removal.
#[tokio::test(start_paused = true)]
async fn a_later_claim_given_up_starts_nothing_ahead_of_an_older_claim_that_waits() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    let held = queue.tx.reserve().await.unwrap();
    let mut older = Box::pin(queue.submit_or_defer(Work::Removal("older")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut older)
        .await
        .is_err());
    let mut other = Box::pin(queue.submit_unless_registering(Work::Other("other")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut other)
        .await
        .is_err());
    drop(other);
    assert!(queue.get_status("one-key").is_none());
    let mut later = Box::pin(queue.submit_or_defer(Work::Removal("later")));
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut later)
        .await
        .is_err());
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    drop(later);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "ahead of the older claim"
    );
    drop(held);
    assert_eq!(older.await, Ok(true));
    assert!(matches!(next(&mut started).await, Work::Removal("older")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), started.recv())
            .await
            .is_err(),
        "beside"
    );
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
}

/// A claim that can stand beside a job holds the URL while it waits, as one
/// made with no job in flight does. An older claim made idle, and then one
/// made unless a registration holds the URL, waits for a slot, its entry
/// taken and given back by a claim of another type. A later claim is given
/// up with a removal kept behind it: nothing starts ahead of the older
/// claim's job.
#[tokio::test(start_paused = true)]
async fn a_later_claim_given_up_starts_nothing_ahead_of_an_older_claim_that_can_stand_beside_a_job()
{
    type Waiting<'a> =
        std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool, String>> + 'a>>;
    async fn made_idle(queue: &JobQueue<Script<Work>>, job: Work) -> Result<bool, String> {
        assert!(queue.claim_if_idle(&job));
        queue.submit_claimed(job).await.map(|()| true)
    }
    for idle in [true, false] {
        let ScriptQueue {
            host: _host,
            queue,
            mut started,
            release,
        } = ScriptQueue::new(false);
        let held = queue.tx.reserve().await.unwrap();
        let (mut older, mut other): (Waiting<'_>, Waiting<'_>) = if idle {
            (
                Box::pin(made_idle(&queue, Work::Removal("older"))),
                Box::pin(queue.submit_unless_registering(Work::Other("other"))),
            )
        } else {
            (
                Box::pin(queue.submit_unless_registering(Work::Other("older"))),
                Box::pin(made_idle(&queue, Work::Removal("other"))),
            )
        };
        for claim in [&mut older, &mut other] {
            assert!(tokio::time::timeout(Duration::from_millis(1), claim)
                .await
                .is_err());
        }
        drop(other);
        assert!(queue.get_status("one-key").is_none());
        let mut later = Box::pin(queue.submit_or_defer(Work::Removal("later")));
        assert!(tokio::time::timeout(Duration::from_millis(1), &mut later)
            .await
            .is_err());
        assert_eq!(
            queue.submit_or_defer(Work::Removal("kept")).await,
            Ok(false)
        );
        drop(later);
        assert!(
            tokio::time::timeout(Duration::from_millis(1), started.recv())
                .await
                .is_err(),
            "ahead of the older claim (made idle: {idle})"
        );
        drop(held);
        assert_eq!(older.await, Ok(true));
        assert!(matches!(
            next(&mut started).await,
            Work::Removal("older") | Work::Other("older")
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(1), started.recv())
                .await
                .is_err(),
            "beside"
        );
        release.send(()).unwrap();
        assert!(matches!(next(&mut started).await, Work::Removal("kept")));
    }
}

/// A kept job holds the URL only while it runs. A removal kept behind a job
/// in flight runs at that job's end and ends. A later claim under the URL
/// that is given up then finds no holder left, and the removal kept behind
/// it runs.
#[tokio::test(start_paused = true)]
async fn a_job_kept_behind_a_claim_given_up_runs_after_an_earlier_kept_job_ended() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    queue.submit(Work::Removal("first")).await.unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("first")));
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept first")).await,
        Ok(false)
    );
    release.send(()).unwrap();
    assert!(matches!(
        next(&mut started).await,
        Work::Removal("kept first")
    ));
    tokio::time::sleep(Duration::from_millis(1)).await;
    assert!(queue.get_status("one-key").is_none());
    let held = queue.tx.reserve().await.unwrap();
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    drop(given_up);
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
    drop(held);
}

/// The same for the claim of `submit_unless_in_flight`.
#[tokio::test(start_paused = true)]
async fn a_job_kept_behind_a_claim_of_submit_unless_in_flight_dropped_before_its_slot_still_runs() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release: _release,
    } = ScriptQueue::new(false);
    let held = queue.tx.reserve().await.unwrap();
    let mut given_up = Box::pin(queue.submit_unless_in_flight(Work::Other("given up")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    drop(given_up);
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
    drop(held);
}

/// The same for the claim `submit_or_defer` makes for a kept job. The holder
/// leaves while a second removal is being kept behind it, so that caller
/// claims the URL for the first kept removal and waits for a slot. The caller
/// leaves too, with the removal it carried (as on main): the second removal
/// runs.
#[tokio::test(start_paused = true)]
async fn a_job_kept_behind_the_claim_made_for_a_kept_job_dropped_before_its_slot_still_runs() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release: _release,
    } = ScriptQueue::new(false);
    let held = queue.tx.clone().reserve_owned().await.unwrap();
    let mut holder = Box::pin({
        let queue = queue.clone();
        async move { queue.submit_or_defer(Work::Removal("holder")).await }
    });
    assert!(tokio::time::timeout(Duration::from_millis(1), &mut holder)
        .await
        .is_err());
    assert_eq!(
        queue.submit_or_defer(Work::Removal("kept")).await,
        Ok(false)
    );
    AT_MERGE.set(Some(Box::new(move || drop(holder))));
    let mut given_up = Box::pin(queue.submit_or_defer(Work::Removal("kept too")));
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    assert_eq!(queue.get_status("one-key").unwrap().status, "pending");
    drop(given_up);
    assert!(matches!(
        next(&mut started).await,
        Work::Removal("kept too")
    ));
    drop(held);
}

/// A claim keeps the kept jobs' turn when another claim took its entry and
/// gave it back. A claim waits for a slot; a claim that can stand beside a
/// job (`submit_unless_registering`) takes its entry and goes back, with a
/// removal kept behind it or, on the URL it freed, behind a later claim that
/// is given up too. Nothing starts while the first claim holds the URL; it
/// goes back with no holder left, and the kept removal runs.
#[tokio::test(start_paused = true)]
async fn a_claim_whose_entry_was_taken_and_given_back_starts_the_kept_job_when_it_goes_back() {
    for later_claim in [false, true] {
        let ScriptQueue {
            host: _host,
            queue,
            mut started,
            release: _release,
        } = ScriptQueue::new(false);
        let held = queue.tx.reserve().await.unwrap();
        let mut first = Box::pin(queue.submit_or_defer(Work::Removal("first")));
        assert!(tokio::time::timeout(Duration::from_millis(1), &mut first)
            .await
            .is_err());
        let mut beside = Box::pin(queue.submit_unless_registering(Work::Other("beside")));
        assert!(tokio::time::timeout(Duration::from_millis(1), &mut beside)
            .await
            .is_err());
        if later_claim {
            drop(beside);
            let mut later = Box::pin(queue.submit_or_defer(Work::Removal("later")));
            assert!(tokio::time::timeout(Duration::from_millis(1), &mut later)
                .await
                .is_err());
            assert_eq!(
                queue.submit_or_defer(Work::Removal("kept")).await,
                Ok(false)
            );
            drop(later);
        } else {
            assert_eq!(
                queue.submit_or_defer(Work::Removal("kept")).await,
                Ok(false)
            );
            drop(beside);
        }
        assert!(queue.get_status("one-key").is_none());
        assert!(
            tokio::time::timeout(Duration::from_millis(1), started.recv())
                .await
                .is_err(),
            "while the first claim holds the URL (a later claim: {later_claim})"
        );
        drop(first);
        assert!(matches!(next(&mut started).await, Work::Removal("kept")));
        drop(held);
    }
}

/// A job's hold ends before the kept jobs' turn at its end. A removal runs
/// unseen, a claim of another type waits for a slot and two removals are kept
/// behind it. The removal's end is refused the turn by the claim's entry, and
/// the claim goes back right there (where the refused removal is put back
/// beside the other one): it finds no holder left and gives the kept removals
/// their turn. With the ended job's hold still counted, nobody would.
#[tokio::test(start_paused = true)]
async fn a_claim_given_up_right_after_it_refused_a_jobs_end_the_turn_starts_the_kept_jobs() {
    let ScriptQueue {
        host: _host,
        queue,
        mut started,
        release,
    } = ScriptQueue::new(false);
    queue.submit(Work::Removal("unseen")).await.unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("unseen")));
    queue.submit(Work::Other("through")).await.unwrap();
    assert!(matches!(next(&mut started).await, Work::Other("through")));
    tokio::time::sleep(Duration::from_millis(1)).await;
    assert!(queue.get_status("one-key").is_none());
    let held = queue.tx.clone().reserve_owned().await.unwrap();
    let mut given_up = Box::pin({
        let queue = queue.clone();
        async move { queue.submit_or_defer(Work::Other("given up")).await }
    });
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut given_up)
            .await
            .is_err()
    );
    for kept in ["kept", "kept too"] {
        assert_eq!(queue.submit_or_defer(Work::Removal(kept)).await, Ok(false));
    }
    AT_MERGE.set(Some(Box::new(move || drop(given_up))));
    release.send(()).unwrap();
    assert!(matches!(next(&mut started).await, Work::Removal("kept")));
    assert!(matches!(
        next(&mut started).await,
        Work::Removal("kept too")
    ));
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
