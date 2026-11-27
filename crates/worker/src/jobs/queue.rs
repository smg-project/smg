//! Dispatches jobs, keeps work behind an in-flight job, and hands outcomes
//! to callers waiting for them. The host supplies the jobs and runs them.

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Weak,
    },
    time::{Duration, SystemTime},
};

use async_trait::async_trait;
use dashmap::{mapref::entry::Entry, DashMap, DashSet};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::{debug, error, field, warn};

use self::slots::{Kept, Outcomes, Waiter, WaiterSlot};
use super::WorkerJobAsker;
use crate::{registry::WorkerId, worker::JobStatus};

#[cfg(feature = "test-util")]
pub mod testing;
#[cfg(test)]
mod tests;

/// What the queue reads of a job.
pub trait QueueJob: Clone + Send + Sync + 'static {
    /// The type name that marks a registration in the status records.
    const REGISTRATION_TYPE: &'static str;

    /// The kind's name in a status record and a log line.
    fn job_type(&self) -> &'static str;
    /// The address or name the job is recorded under.
    fn key(&self) -> &str;
    /// The key as a log line may show it.
    fn shown_key(job_type: &str, key: &str) -> String;
    /// Whether the job is a removal, which the kept slot takes first.
    fn is_removal(&self) -> bool;
    /// One job for two kept removals that can run together, or both back.
    /// The bound on the removals kept for a key is the job type's: a type
    /// that merges none keeps every one.
    fn merge_removals(held: Self, newer: Self) -> Result<Self, (Self, Self)>;
    /// The asker shown in logs and used to identify a kept removal.
    fn asker(&self) -> Option<&WorkerJobAsker> {
        None
    }
}

/// The host of a queue. It brings the jobs and runs them.
#[async_trait]
pub trait JobHost: Send + Sync + 'static {
    /// The jobs this host runs.
    type Job: QueueJob;
    /// The host's name in errors when it has shut down.
    const NAME: &'static str;

    /// Runs the job to its end. `run` gives the job its cancel mark.
    async fn execute(self: &Arc<Self>, job: &Self::Job, run: &JobRun<'_>)
        -> Result<String, String>;
    /// The id whose cancel mark the job owns, if any.
    fn mark_id(&self, _job: &Self::Job) -> Option<WorkerId> {
        None
    }
    /// Called after a dispatcher slot is held, just before the job is sent.
    fn accepted(&self, _job: &Self::Job) {}
    /// Called on failure before the job's status turns terminal.
    fn failed(&self, _job: &Self::Job) {}
}

/// The cancel marks available to a running job.
pub struct JobRun<'a> {
    /// Cancel marks keyed by worker id.
    pub cancelled_before_start: &'a DashSet<WorkerId>,
    /// The id whose mark this job may take.
    pub own_id: Option<&'a WorkerId>,
    /// Set by the job when a `DELETE` ended it: its id keeps no failed status.
    pub ended_by_delete: &'a AtomicBool,
}

/// The queue capacity and the number of dispatcher permits.
#[derive(Clone, Debug)]
pub struct JobQueueConfig {
    /// Number of slots in the submission channel.
    pub queue_capacity: usize,
    /// Number of jobs the dispatcher can hand off at once.
    pub max_concurrent_jobs: usize,
}

impl Default for JobQueueConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1000,
            max_concurrent_jobs: 200,
        }
    }
}

/// Queues jobs for a host and tracks their status and outcomes.
pub struct JobQueue<H: JobHost> {
    /// Channel for submitting jobs, each with the id of the waiter for its
    /// outcome when a caller waits on it
    tx: mpsc::Sender<Queued<H::Job>>,
    /// The host owns the queue. The weak reference lets it shut down.
    host: Weak<H>,
    /// Job status tracking by worker URL
    status_map: Arc<DashMap<String, JobStatus>>,
    /// Semaphore to limit concurrent job execution
    concurrency_limit: Arc<Semaphore>,
    /// The waiters of [`Self::submit_and_wait`], by job key: a job's outcome
    /// is handed to its waiter when the job ends.
    outcomes: Outcomes,
    /// The next waiter's id: what tells a caller's waiter from a later one
    /// under the same key.
    next_waiter_id: AtomicU64,
    /// The jobs kept per key while another job holds it (`submit_or_defer`).
    /// That job's completion runs them in turn (`run_kept`).
    kept: Arc<DashMap<String, Kept<H::Job>>>,
    /// Cancel marks by id. A job reads and clears only the mark it owns.
    cancelled_before_start: Arc<DashSet<WorkerId>>,
    /// How many `DELETE`s hold the mark of an id: one that takes its own
    /// mark back leaves it while another holds it. A count ends with its
    /// mark: the next mark set drops the count of a mark that is gone.
    mark_holders: DashMap<WorkerId, usize>,
}

/// A job handed to the dispatcher with the id of the waiter for its outcome,
/// when a caller waits on it: the outcome goes to that waiter alone, never to
/// a later caller's under the same key.
struct Queued<J> {
    job: J,
    waiter: Option<u64>,
}

/// Why [`JobQueue::submit_and_wait`] returned no success message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobOutcomeError {
    /// The queue took no job: it is closed, or its host is gone.
    NotQueued(String),
    /// The job ran and failed, with its reason.
    Failed(String),
    /// The job did not end within the wait; it keeps running.
    TimedOut(Duration),
    /// A caller is waiting on a job under this key already: the job runs
    /// once and its outcome goes to that caller. The key is written as a
    /// log line writes it.
    InFlight(String),
}

impl std::fmt::Display for JobOutcomeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotQueued(reason) => write!(f, "Failed to submit job: {reason}"),
            Self::Failed(reason) => f.write_str(reason),
            Self::TimedOut(wait) => write!(f, "Job timeout after {}s", wait.as_secs()),
            Self::InFlight(key) => write!(f, "A job for {key} is in flight; wait for its outcome"),
        }
    }
}

impl std::error::Error for JobOutcomeError {}

impl<H: JobHost> std::fmt::Debug for JobQueue<H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobQueue")
            .field("status_count", &self.status_map.len())
            .finish()
    }
}

mod slots {
    use super::*;

    /// The caller waiting on a job's outcome under the job's key, one per key
    /// at a time; `id` is the caller's own, so a caller leaving takes its own
    /// entry alone.
    pub struct Waiter {
        pub id: u64,
        pub sender: oneshot::Sender<Result<String, String>>,
    }

    /// The waiters for a job's outcome, by job key.
    pub type Outcomes = Arc<DashMap<String, Waiter>>;

    /// Removes the caller's waiter when the caller leaves
    /// [`JobQueue::submit_and_wait`] by any path: a submit refused, a wait
    /// cancelled while the queue is full, a timeout, or the outcome delivered
    /// (which removed it already). The entry goes only while it is still this
    /// caller's, so a later waiter under the same key stays.
    pub struct WaiterSlot {
        pub outcomes: Outcomes,
        pub key: String,
        pub id: u64,
    }

    impl Drop for WaiterSlot {
        fn drop(&mut self) {
            self.outcomes
                .remove_if(&self.key, |_, waiter| waiter.id == self.id);
        }
    }

    /// Removals kept for a key run in the order first kept, before the latest
    /// job of any other kind. The job type decides which removals can merge.
    pub struct Kept<J> {
        pub removals: VecDeque<J>,
        pub other: Option<J>,
    }

    impl<J> Default for Kept<J> {
        fn default() -> Self {
            Self {
                removals: VecDeque::new(),
                other: None,
            }
        }
    }

    impl<J: QueueJob> Kept<J> {
        pub(super) fn is_empty(&self) -> bool {
            self.removals.is_empty() && self.other.is_none()
        }

        /// The next job to run, taking removals first.
        pub(super) fn take_next(&mut self) -> Option<J> {
            self.removals.pop_front().or_else(|| self.other.take())
        }

        /// Keeps a job or puts one with a refused claim back at the front.
        pub(super) fn place(&mut self, mut job: J, newer_wins: bool) {
            if job.is_removal() {
                // Check each kept removal once. A merge takes its held spot.
                for checked in 0..self.removals.len() {
                    let Some(held) = self.removals.pop_front() else {
                        break;
                    };
                    match J::merge_removals(held, job) {
                        Ok(merged) => {
                            self.removals.push_back(merged);
                            self.removals.rotate_right(checked + 1);
                            return;
                        }
                        Err((held, newer)) => {
                            self.removals.push_back(held);
                            job = newer;
                        }
                    }
                }
                if newer_wins {
                    self.removals.push_back(job);
                } else {
                    self.removals.push_front(job);
                }
            } else if newer_wins || self.other.is_none() {
                self.other = Some(job);
            }
        }
    }
}

/// A claimed submission's hold on its own status entry while it waits for a
/// slot: dropped with its caller before the slot is held (a cancelled request
/// on a full queue) or ended by a refused submission, it takes the claim back
/// out, so no pending entry outlives the submission it was made for.
struct Claim<'a, H: JobHost> {
    queue: &'a JobQueue<H>,
    url: &'a str,
    job_type: &'static str,
    held: bool,
}

impl<H: JobHost> Drop for Claim<'_, H> {
    fn drop(&mut self) {
        if !self.held {
            JobQueue::<H>::record_own_status(&self.queue.status_map, self.url, self.job_type, None);
        }
    }
}

impl<H: JobHost> JobQueue<H> {
    /// Starts a dispatcher and a task that expires old statuses. The queue
    /// keeps a weak reference to the host. Requires a running Tokio runtime.
    pub fn new(config: JobQueueConfig, host: Weak<H>) -> Arc<Self> {
        let (tx, mut rx) = mpsc::channel(config.queue_capacity);

        debug!(
            "Initializing job queue: capacity={}, max_concurrent={}",
            config.queue_capacity, config.max_concurrent_jobs
        );

        let status_map = Arc::new(DashMap::new());
        let concurrency_limit = Arc::new(Semaphore::new(config.max_concurrent_jobs));
        let outcomes: Outcomes = Arc::new(DashMap::new());
        let kept: Arc<DashMap<String, Kept<H::Job>>> = Arc::new(DashMap::new());
        let cancelled_before_start = Arc::new(DashSet::new());

        let queue = Arc::new(Self {
            tx,
            host: host.clone(),
            status_map: status_map.clone(),
            concurrency_limit: concurrency_limit.clone(),
            outcomes: outcomes.clone(),
            next_waiter_id: AtomicU64::new(1),
            kept: kept.clone(),
            cancelled_before_start: cancelled_before_start.clone(),
            mark_holders: DashMap::new(),
        });

        // Single dispatcher task
        let host = host.clone();
        let status = status_map.clone();
        let sem = concurrency_limit.clone();
        let waiters = outcomes.clone();
        let marks = cancelled_before_start.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "The dispatcher drains until the channel sender is dropped"
        )]
        tokio::spawn(async move {
            while let Some(queued) = rx.recv().await {
                // Acquire permit (blocks if at concurrency limit)
                let Ok(permit) = sem.clone().acquire_owned().await else {
                    error!("Semaphore closed, stopping dispatcher");
                    break;
                };

                let host_clone = host.clone();
                let status_clone = status.clone();
                let waiters_clone = waiters.clone();
                let kept_clone = kept.clone();
                let marks_clone = marks.clone();

                #[expect(
                    clippy::disallowed_methods,
                    reason = "The dispatcher hands a permit to each task and the task releases it before running the job"
                )]
                tokio::spawn(async move {
                    Self::process_job(
                        queued,
                        host_clone,
                        status_clone,
                        waiters_clone,
                        kept_clone,
                        marks_clone,
                        permit,
                    )
                    .await;
                });
            }

            debug!("Job dispatcher stopped");
        });

        // Spawn cleanup task for old job statuses (TTL 5 minutes)
        let cleanup_status_map = status_map.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "Status cleanup expires terminal records at regular intervals"
        )]
        tokio::spawn(async move {
            Self::cleanup_old_statuses(cleanup_status_map).await;
        });

        queue
    }

    /// Get current queue and concurrency status
    pub fn get_load_info(&self) -> (usize, usize) {
        let queue_depth = self.tx.max_capacity() - self.tx.capacity();
        let available_permits = self.concurrency_limit.available_permits();
        (queue_depth, available_permits)
    }

    /// Claims the key unless a job of its type is pending or processing.
    /// The check and claim are one step, so only the caller that holds the
    /// claim can prepare the job and pass it to [`Self::submit_claimed`].
    pub fn claim_if_idle(&self, job: &H::Job) -> bool {
        Self::claim(&self.status_map, job.key(), job.job_type())
    }

    /// Submit a job whose URL the caller claimed with [`Self::claim_if_idle`].
    /// A submission refused or dropped before it holds a slot owns no job, so
    /// its claim goes with it (`submit_inner`'s hold on the entry).
    pub async fn submit_claimed(&self, job: H::Job) -> Result<(), String> {
        self.submit_inner(job, true, None).await
    }

    /// A claim blocks jobs of the same type. Different types can share a key.
    fn claim(status_map: &DashMap<String, JobStatus>, url: &str, job_type: &'static str) -> bool {
        match status_map.entry(url.to_string()) {
            Entry::Occupied(e)
                if e.get().job_type == job_type
                    && matches!(e.get().status.as_str(), "pending" | "processing") =>
            {
                false
            }
            entry => {
                entry.insert(JobStatus::pending(job_type, url));
                true
            }
        }
    }

    /// Submit unless a job of any type is pending or processing for this URL.
    /// For a best-effort submission that must not take an in-flight job's
    /// status under the key: a label refresh while the worker's registration
    /// is still processing would take that job's entry (and let the in-flight
    /// guard admit a second registration). Returns `Ok(false)` when one is in
    /// flight; the caller's next occasion submits again. A job of another type
    /// submitted after the claim keeps its entry too: the pending status
    /// written once a slot is held, the claim's return on a refused or dropped
    /// submission and a job's later status writes touch only this job's own
    /// entry (`record_own_status`).
    pub async fn submit_unless_in_flight(&self, job: H::Job) -> Result<bool, String> {
        let worker_url = job.key().to_string();
        let job_type = job.job_type();
        if !Self::claim_unless_in_flight(&self.status_map, &worker_url, job_type) {
            return Ok(false);
        }
        self.submit_inner(job, true, None).await.map(|()| true)
    }

    /// Submit `job` unless a registration is pending or processing under its
    /// URL: the check and the job's claim are one entry operation, and the
    /// job's pending status is then written over its own entry only, once it
    /// holds a slot, so a registration that claims the URL during the wait
    /// keeps its entry. `Ok(false)` when a registration holds the URL; any
    /// other entry (another type's, or an ended job's) is taken.
    pub async fn submit_unless_registering(&self, job: H::Job) -> Result<bool, String> {
        let worker_url = job.key().to_string();
        let job_type = job.job_type();
        match self.status_map.entry(worker_url.clone()) {
            Entry::Occupied(e)
                if e.get().job_type == H::Job::REGISTRATION_TYPE
                    && matches!(e.get().status.as_str(), "pending" | "processing") =>
            {
                return Ok(false);
            }
            entry => {
                entry.insert(JobStatus::pending(job_type, &worker_url));
            }
        }
        self.submit_inner(job, true, None).await.map(|()| true)
    }

    /// Keeps a job that finds its key held. The holder's completion runs the
    /// kept jobs in order. If the holder ended before the job was kept, this
    /// call tries to submit the next kept job. Returns `Ok(true)` when it
    /// submits from the same group it kept. Non-removals share a group and
    /// removals are grouped by asker for this answer.
    pub async fn submit_or_defer(&self, job: H::Job) -> Result<bool, String> {
        let worker_url = job.key().to_string();
        if Self::claim_unless_in_flight(&self.status_map, &worker_url, job.job_type()) {
            return self.submit_inner(job, true, None).await.map(|()| true);
        }
        let removal_asker = job.is_removal().then(|| job.asker().cloned());
        Self::keep(&self.kept, &worker_url, job);
        let Some(next) = Self::take_next(&self.kept, &worker_url, |next| {
            Self::claim_unless_in_flight(&self.status_map, &worker_url, next.job_type())
        }) else {
            return Ok(false);
        };
        let own = if next.is_removal() {
            removal_asker
                .as_ref()
                .is_some_and(|asker| asker.as_ref() == next.asker())
        } else {
            removal_asker.is_none()
        };
        self.submit_inner(next, true, None).await.map(|()| own)
    }

    /// Keeps `job` for `url` under the rules of [`Kept`].
    fn keep(kept: &DashMap<String, Kept<H::Job>>, url: &str, job: H::Job) {
        kept.entry(url.to_string()).or_default().place(job, true);
    }

    /// Takes the next kept job and claims the URL under the same lock, so
    /// two runners cannot take jobs out of order. A refused job goes back
    /// in place. The entry goes when it has no jobs left.
    fn take_next(
        kept: &DashMap<String, Kept<H::Job>>,
        url: &str,
        claim: impl FnOnce(&H::Job) -> bool,
    ) -> Option<H::Job> {
        let mut entry = kept.get_mut(url)?;
        let job = entry.take_next()?;
        if !claim(&job) {
            entry.place(job, false);
            return None;
        }
        let emptied = entry.is_empty();
        drop(entry);
        if emptied {
            kept.remove_if(url, |_, slots| slots.is_empty());
        }
        Some(job)
    }

    /// Runs the kept jobs in order, each under its own status. If another
    /// job has taken the URL, its completion runs the jobs still kept.
    async fn run_kept(
        worker_url: &str,
        host: &Arc<H>,
        status_map: &Arc<DashMap<String, JobStatus>>,
        kept: &DashMap<String, Kept<H::Job>>,
        cancelled_before_start: &DashSet<WorkerId>,
    ) {
        while let Some(job) = Self::take_next(kept, worker_url, |next| {
            Self::claim_unless_in_flight(status_map, worker_url, next.job_type())
        }) {
            let job_type = job.job_type();
            debug!(
                asker = job.asker().map(field::debug),
                "Running the {} job kept for worker={} while another job held the URL",
                job_type,
                H::Job::shown_key(job_type, worker_url)
            );
            Self::record_own_status(
                status_map,
                worker_url,
                job_type,
                Some(JobStatus::processing(job_type, worker_url)),
            );
            let start = std::time::Instant::now();
            let own_id = host.mark_id(&job);
            let ended_by_delete = AtomicBool::new(false);
            let run = JobRun {
                cancelled_before_start,
                own_id: own_id.as_ref(),
                ended_by_delete: &ended_by_delete,
            };
            let result = host.execute(&job, &run).await;
            if let (Some(id), Err(error)) = (&own_id, &result) {
                Self::keep_own_failure(
                    status_map,
                    cancelled_before_start,
                    id,
                    ended_by_delete.load(Ordering::Relaxed),
                    job_type,
                    worker_url,
                    error,
                );
            }
            if result.is_err() {
                host.failed(&job);
            }
            Self::record_job_completion(job_type, worker_url, start.elapsed(), &result, status_map);
            if let Some(id) = own_id {
                Self::end_mark(status_map, cancelled_before_start, &id);
            }
        }
    }

    /// Any in-flight job under the key blocks the claim, whatever its type.
    fn claim_unless_in_flight(
        status_map: &DashMap<String, JobStatus>,
        url: &str,
        job_type: &'static str,
    ) -> bool {
        match status_map.entry(url.to_string()) {
            Entry::Occupied(e) if matches!(e.get().status.as_str(), "pending" | "processing") => {
                false
            }
            entry => {
                entry.insert(JobStatus::pending(job_type, url));
                true
            }
        }
    }

    /// Submits a job without first claiming its key.
    pub async fn submit(&self, job: H::Job) -> Result<(), String> {
        self.submit_inner(job, false, None).await
    }

    /// `submit`, with the pending status written over this job's own claim
    /// alone when `claimed` ([`Self::submit_claimed`],
    /// [`Self::submit_unless_in_flight`]): a job of another type that took the
    /// URL during the wait for a slot keeps its entry, where a plain submission
    /// writes its own over whatever is there. `waiter` is the id of the caller
    /// waiting on the outcome, [`Self::submit_and_wait`]'s own; none otherwise.
    async fn submit_inner(
        &self,
        job: H::Job,
        claimed: bool,
        waiter: Option<u64>,
    ) -> Result<(), String> {
        // Extract values before moving job
        let job_type = job.job_type();
        let worker_url = job.key().to_string();
        // A claimed submission holds its entry only as long as it lives:
        // ended or dropped before a slot is held, the claim goes with it. A
        // plain one builds no guard (built eagerly, a guard would drop at once
        // and take another job's same-type entry with it).
        let mut claim = claimed.then(|| Claim {
            queue: self,
            url: &worker_url,
            job_type,
            held: false,
        });
        let Some(host) = self.host() else {
            return Err(format!("Job queue shutting down: {} dropped", H::NAME));
        };

        // A slot in the queue first: the wait for one (a full queue) can be
        // cancelled with the caller and must leave nothing behind, so nothing
        // is recorded before a slot is held; from the slot on, the hand-over
        // is synchronous and cannot fail.
        let Ok(permit) = self.tx.reserve().await else {
            // The channel refuses a slot only once its receiver is gone.
            return Err("Job queue closed: its dispatcher stopped".to_string());
        };

        if let Some(claim) = claim.as_mut() {
            claim.held = true;
        }
        // Record pending status
        Self::write_pending(&self.status_map, &worker_url, job_type, claimed);
        host.accepted(&job);
        permit.send(Queued { job, waiter });

        let (queue_depth, available_permits) = self.get_load_info();
        debug!(
            "Job submitted: type={}, worker={}, queue_depth={}, available_slots={}",
            job_type,
            H::Job::shown_key(job_type, &worker_url),
            queue_depth,
            available_permits
        );
        Ok(())
    }

    /// Submit `job` and wait for its outcome: the job's own success message
    /// or its failure, as the status map records them, without a poll.
    /// `timeout` bounds the wait; a job that outlives it keeps running and
    /// its outcome is dropped.
    pub async fn submit_and_wait(
        &self,
        job: H::Job,
        timeout: Duration,
    ) -> Result<String, JobOutcomeError> {
        let key = job.key().to_string();
        let (tx, rx) = oneshot::channel();
        let id = self.next_waiter_id.fetch_add(1, Ordering::Relaxed);
        // One waiter per key: a second caller for a job in flight is refused,
        // not handed the first caller's outcome (and the first its loss).
        match self.outcomes.entry(key.clone()) {
            Entry::Occupied(_) => {
                let shown = H::Job::shown_key(job.job_type(), &key);
                return Err(JobOutcomeError::InFlight(shown));
            }
            Entry::Vacant(entry) => {
                entry.insert(Waiter { id, sender: tx });
            }
        }
        // The waiter leaves with the caller, whichever way the caller leaves.
        let _slot = WaiterSlot {
            outcomes: Arc::clone(&self.outcomes),
            key,
            id,
        };
        if let Err(reason) = self.submit_inner(job, false, Some(id)).await {
            return Err(JobOutcomeError::NotQueued(reason));
        }
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(Ok(message))) => Ok(message),
            Ok(Ok(Err(reason))) => Err(JobOutcomeError::Failed(reason)),
            Ok(Err(_)) => Err(JobOutcomeError::Failed(
                "the job ended without an outcome".to_string(),
            )),
            Err(_) => Err(JobOutcomeError::TimedOut(timeout)),
        }
    }

    /// Hands a finished job's outcome to the waiter the job was queued for,
    /// while that waiter is still under the key. A job queued without a
    /// waiter, or whose waiter left and a later caller took the key, delivers
    /// nothing; a waiter that gave up (its receiver dropped) is a no-op.
    fn deliver_outcome(
        outcomes: &Outcomes,
        key: &str,
        waiter: Option<u64>,
        outcome: Result<String, String>,
    ) {
        let Some(id) = waiter else { return };
        if let Some((_, waiter)) = outcomes.remove_if(key, |_, waiter| waiter.id == id) {
            let _ = waiter.sender.send(outcome);
        }
    }

    /// Reads the status of the job recorded under the key.
    pub fn get_status(&self, worker_url: &str) -> Option<JobStatus> {
        self.status_map.get(worker_url).map(|entry| entry.clone())
    }

    /// Gets the host if it is still alive.
    pub fn host(&self) -> Option<Arc<H>> {
        self.host.upgrade()
    }

    /// Sets the cancel mark a job of this id reads when it starts.
    pub fn set_cancel_mark(&self, id: &WorkerId) {
        let marks = &self.cancelled_before_start;
        // A mark its job or the job's end took leaves its count behind.
        self.mark_holders.retain(|held, _| marks.contains(held));
        // The set changes under the count's lock, here and in a take-back.
        let mut count = self.mark_holders.entry(id.clone()).or_insert(0);
        *count = if marks.insert(id.clone()) {
            1
        } else {
            *count + 1
        };
    }

    /// Takes back the mark one `DELETE` set for this id. The mark stays
    /// while another `DELETE` holds it.
    pub fn clear_cancel_mark(&self, id: &WorkerId) {
        if let Entry::Occupied(mut count) = self.mark_holders.entry(id.clone()) {
            *count.get_mut() -= 1;
            if *count.get() == 0 {
                self.cancelled_before_start.remove(id);
                count.remove();
            }
        }
    }

    /// Process a single job with status tracking and error handling
    async fn process_job(
        queued: Queued<H::Job>,
        host: Weak<H>,
        status_map: Arc<DashMap<String, JobStatus>>,
        outcomes: Outcomes,
        kept: Arc<DashMap<String, Kept<H::Job>>>,
        cancelled_before_start: Arc<DashSet<WorkerId>>,
        permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        let Queued { job, waiter } = queued;
        let job_type = job.job_type();
        let worker_url = job.key().to_string();
        let start = std::time::Instant::now();

        // Update to processing: this job's entry only (a job of another type
        // submitted for the URL since this one was queued keeps its status).
        Self::record_own_status(
            &status_map,
            &worker_url,
            job_type,
            Some(JobStatus::processing(job_type, &worker_url)),
        );

        debug!(
            asker = job.asker().map(field::debug),
            "Processing job: type={}, worker={}",
            job_type,
            H::Job::shown_key(job_type, &worker_url)
        );

        // Release concurrency slot immediately. The semaphore bounds how many
        // jobs can be dequeued concurrently (preventing thundering herd), but
        // long-running waits (e.g. 30-min worker health checks) must not hold
        // a slot or they starve the queue for all other job types.
        drop(permit);

        match host.upgrade() {
            Some(host) => {
                // Read the id first. The host can release it when the job fails.
                let own_id = host.mark_id(&job);
                let ended_by_delete = AtomicBool::new(false);
                let run = JobRun {
                    cancelled_before_start: &cancelled_before_start,
                    own_id: own_id.as_ref(),
                    ended_by_delete: &ended_by_delete,
                };
                let result = host.execute(&job, &run).await;
                let duration = start.elapsed();
                if let (Some(id), Err(error)) = (&own_id, &result) {
                    Self::keep_own_failure(
                        &status_map,
                        &cancelled_before_start,
                        id,
                        ended_by_delete.load(Ordering::Relaxed),
                        job_type,
                        &worker_url,
                        error,
                    );
                }
                if result.is_err() {
                    host.failed(&job);
                }
                Self::record_job_completion(job_type, &worker_url, duration, &result, &status_map);
                if let Some(id) = own_id {
                    Self::end_mark(&status_map, &cancelled_before_start, &id);
                }
                Self::deliver_outcome(&outcomes, &worker_url, waiter, result);
                Self::run_kept(
                    &worker_url,
                    &host,
                    &status_map,
                    &kept,
                    &cancelled_before_start,
                )
                .await;
            }
            None => {
                let error_msg = format!("{} dropped", H::NAME);
                status_map.insert(
                    worker_url.clone(),
                    JobStatus::failed(job_type, &worker_url, error_msg.clone()),
                );
                Self::deliver_outcome(&outcomes, &worker_url, waiter, Err(error_msg));
                error!(
                    "{} dropped, cannot process job: type={}, worker={}",
                    H::NAME,
                    job_type,
                    H::Job::shown_key(job_type, &worker_url)
                );
            }
        }
    }

    /// Keeps the failure of a job that owned an id under that id as well as
    /// under its URL, before the host releases the id: a reader of the id
    /// meets no gap between pending and failed. A URL keeps one such failure:
    /// a later one replaces it. A job a `DELETE` ended (its own word, or a
    /// mark still set) keeps none: its id reads no status.
    fn keep_own_failure(
        status_map: &DashMap<String, JobStatus>,
        marks: &DashSet<WorkerId>,
        id: &WorkerId,
        ended_by_delete: bool,
        job_type: &'static str,
        worker_url: &str,
        error: &str,
    ) {
        if !ended_by_delete && !marks.contains(id) {
            status_map.retain(|key, kept| key == worker_url || kept.worker_url != worker_url);
            status_map.insert(
                id.as_str().to_string(),
                JobStatus::failed(job_type, worker_url, error.to_string()),
            );
        }
    }

    /// Ends the mark of a job's id with the job. A mark still set was set
    /// after the job's reads, by a `DELETE` that was told it ended the
    /// registration: the failure kept under the id goes with the mark.
    fn end_mark(status_map: &DashMap<String, JobStatus>, marks: &DashSet<WorkerId>, id: &WorkerId) {
        if marks.remove(id).is_some() {
            status_map.remove(id.as_str());
        }
    }

    /// Update job status on completion
    fn record_job_completion(
        job_type: &'static str,
        worker_url: &str,
        _duration: Duration,
        result: &Result<String, String>,
        status_map: &Arc<DashMap<String, JobStatus>>,
    ) {
        match result {
            Ok(message) => {
                Self::record_own_status(status_map, worker_url, job_type, None);
                debug!(
                    "Completed job: type={}, worker={}, result={}",
                    job_type,
                    H::Job::shown_key(job_type, worker_url),
                    message
                );
            }
            Err(error) => {
                Self::record_own_status(
                    status_map,
                    worker_url,
                    job_type,
                    Some(JobStatus::failed(job_type, worker_url, error.clone())),
                );
                warn!(
                    "Failed job: type={}, worker={}, error={}",
                    job_type,
                    H::Job::shown_key(job_type, worker_url),
                    error
                );
            }
        }
    }

    /// The pending status under `url` once a slot is held: written over
    /// whatever is there, or, for a claimed submission (`own_only`), over this
    /// job's own claim alone (`record_own_status`), so a job of another type
    /// that took the URL during the wait keeps its entry and the in-flight
    /// guard that reads it.
    fn write_pending(
        status_map: &DashMap<String, JobStatus>,
        url: &str,
        job_type: &'static str,
        own_only: bool,
    ) {
        if own_only {
            Self::record_own_status(
                status_map,
                url,
                job_type,
                Some(JobStatus::pending(job_type, url)),
            );
        } else {
            status_map.insert(url.to_string(), JobStatus::pending(job_type, url));
        }
    }

    /// Writes a job's status under its URL (`None` removes it) unless the
    /// entry there is another job's: a job of another type submitted for the
    /// URL after this one was queued keeps its status for its lifetime, since
    /// the in-flight guards (the reconciler's, `submit_unless_in_flight`) read
    /// that entry. A vacant entry is written as before.
    fn record_own_status(
        status_map: &DashMap<String, JobStatus>,
        url: &str,
        job_type: &'static str,
        status: Option<JobStatus>,
    ) {
        match status_map.entry(url.to_string()) {
            Entry::Occupied(entry) if entry.get().job_type != job_type => {
                debug!(
                    "Job status of type={} for worker={} left to the {} job under the key",
                    job_type,
                    H::Job::shown_key(job_type, url),
                    entry.get().job_type
                );
            }
            Entry::Occupied(mut entry) => match status {
                Some(status) => {
                    entry.insert(status);
                }
                None => {
                    entry.remove();
                }
            },
            Entry::Vacant(entry) => {
                if let Some(status) = status {
                    entry.insert(status);
                }
            }
        }
    }

    /// A status is reclaimed only once terminal and older than the TTL.
    /// Pending/processing entries are the reconciler's in-flight guard: a
    /// long registration wave holds jobs in those states well past any
    /// fixed TTL, and purging them makes every discovery pass resubmit
    /// duplicate jobs for workers already mid-registration.
    fn status_expired(status: &JobStatus, now: u64, ttl: u64) -> bool {
        !matches!(status.status.as_str(), "pending" | "processing")
            && now.saturating_sub(status.timestamp) >= ttl
    }

    /// Cleanup old terminal job statuses (TTL 5 minutes)
    async fn cleanup_old_statuses(status_map: Arc<DashMap<String, JobStatus>>) {
        const CLEANUP_INTERVAL: Duration = Duration::from_secs(60); // Run every minute
        const STATUS_TTL: u64 = 300; // 5 minutes in seconds

        loop {
            tokio::time::sleep(CLEANUP_INTERVAL).await;

            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            status_map.retain(|_key, value| !Self::status_expired(value, now, STATUS_TTL));

            debug!(
                "Cleaned up old job statuses, remaining: {}",
                status_map.len()
            );
        }
    }
}
