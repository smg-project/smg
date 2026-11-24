//! Fixtures for exercising a host's queue without waiting for the dispatcher.

use std::{sync::Arc, time::Duration};

use dashmap::{DashMap, DashSet};
use tokio::sync::{mpsc, Semaphore};

pub use super::slots::{Kept, Outcomes, Waiter, WaiterSlot};
use super::{JobHost, JobQueue, Queued};
use crate::{registry::WorkerId, worker::JobStatus};

/// The queue's slots, records and limits, borrowed by a host's tests.
pub struct QueueTest<'a, H: JobHost> {
    pub status_map: &'a Arc<DashMap<String, JobStatus>>,
    pub concurrency_limit: &'a Arc<Semaphore>,
    pub outcomes: &'a Outcomes,
    pub kept: &'a DashMap<String, Kept<H::Job>>,
    pub cancelled_before_start: &'a DashSet<WorkerId>,
}

impl<'a, H: JobHost> QueueTest<'a, H> {
    /// Borrows the state of a queue.
    pub fn state(queue: &'a JobQueue<H>) -> Self {
        Self {
            status_map: &queue.status_map,
            concurrency_limit: &queue.concurrency_limit,
            outcomes: &queue.outcomes,
            kept: &queue.kept,
            cancelled_before_start: &queue.cancelled_before_start,
        }
    }

    /// Starts a queue that accepts submissions and lets the caller read them.
    pub fn recording(host: &Arc<H>) -> (Arc<JobQueue<H>>, impl FnMut() -> Option<H::Job>) {
        let config = super::JobQueueConfig::default();
        let (tx, mut rx) = mpsc::channel::<Queued<H::Job>>(config.queue_capacity);
        let queue = Arc::new(JobQueue {
            tx,
            host: Arc::downgrade(host),
            status_map: Arc::default(),
            concurrency_limit: Arc::new(Semaphore::new(config.max_concurrent_jobs)),
            outcomes: Arc::default(),
            next_waiter_id: 1.into(),
            kept: Arc::default(),
            cancelled_before_start: Arc::default(),
            mark_holders: DashMap::new(),
        });
        (queue, move || rx.try_recv().ok().map(|queued| queued.job))
    }

    /// Holds the key as if the dispatcher were running a job there.
    pub fn hold_url_for_tests(queue: &JobQueue<H>, url: &str, job_type: &'static str) {
        queue
            .status_map
            .insert(url.to_string(), JobStatus::processing(job_type, url));
    }

    /// Reads whether a job is kept for the key.
    pub fn kept_for(queue: &JobQueue<H>, url: &str) -> bool {
        queue.kept.get(url).is_some_and(|slots| !slots.is_empty())
    }

    /// Holds one channel slot until the caller drops it.
    pub async fn reserve_slot(queue: &JobQueue<H>) -> Option<impl Sized + '_> {
        queue.tx.reserve().await.ok()
    }

    /// Reads the configured channel capacity.
    pub fn capacity(queue: &JobQueue<H>) -> usize {
        queue.tx.max_capacity()
    }

    /// Claims a key for a job type.
    pub fn claim(
        status_map: &DashMap<String, JobStatus>,
        url: &str,
        job_type: &'static str,
    ) -> bool {
        JobQueue::<H>::claim(status_map, url, job_type)
    }

    /// Claims a key if no job at all is in flight.
    pub fn claim_unless_in_flight(
        status_map: &DashMap<String, JobStatus>,
        url: &str,
        job_type: &'static str,
    ) -> bool {
        JobQueue::<H>::claim_unless_in_flight(status_map, url, job_type)
    }

    /// Keeps a job behind a holder.
    pub fn keep(kept: &DashMap<String, Kept<H::Job>>, url: &str, job: H::Job) {
        JobQueue::<H>::keep(kept, url, job);
    }

    /// Takes the next kept job if the claim succeeds.
    pub fn take_next(
        kept: &DashMap<String, Kept<H::Job>>,
        url: &str,
        claim: impl FnOnce(&H::Job) -> bool,
    ) -> Option<H::Job> {
        JobQueue::<H>::take_next(kept, url, claim)
    }

    /// Returns a job whose claim did not succeed.
    pub fn put_back(kept: &DashMap<String, Kept<H::Job>>, url: &str, job: H::Job) {
        kept.entry(url.to_string()).or_default().place(job, false);
    }

    /// Runs the jobs kept behind a holder.
    pub async fn run_kept(
        url: &str,
        host: &Arc<H>,
        status_map: &Arc<DashMap<String, JobStatus>>,
        kept: &DashMap<String, Kept<H::Job>>,
        marks: &DashSet<WorkerId>,
    ) {
        JobQueue::<H>::run_kept(url, host, status_map, kept, marks).await;
    }

    /// Delivers an outcome to its own waiter.
    pub fn deliver_outcome(
        outcomes: &Outcomes,
        key: &str,
        waiter: Option<u64>,
        outcome: Result<String, String>,
    ) {
        JobQueue::<H>::deliver_outcome(outcomes, key, waiter, outcome);
    }

    /// Records the job's success or failure.
    pub fn record_job_completion(
        job_type: &'static str,
        url: &str,
        duration: Duration,
        result: &Result<String, String>,
        status_map: &Arc<DashMap<String, JobStatus>>,
    ) {
        JobQueue::<H>::record_job_completion(job_type, url, duration, result, status_map);
    }

    /// Writes a pending record after a slot is held.
    pub fn write_pending(
        status_map: &DashMap<String, JobStatus>,
        url: &str,
        job_type: &'static str,
        own_only: bool,
    ) {
        JobQueue::<H>::write_pending(status_map, url, job_type, own_only);
    }

    /// Writes a status without taking another type's record.
    pub fn record_own_status(
        status_map: &DashMap<String, JobStatus>,
        url: &str,
        job_type: &'static str,
        status: Option<JobStatus>,
    ) {
        JobQueue::<H>::record_own_status(status_map, url, job_type, status);
    }

    /// Reads whether cleanup would expire a status.
    pub fn status_expired(status: &JobStatus, now: u64, ttl: u64) -> bool {
        JobQueue::<H>::status_expired(status, now, ttl)
    }
}
