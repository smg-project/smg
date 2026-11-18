//! Local, per-attempt prefill reservations. Selection and booking share one lock.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

/// A frozen backend estimate, combined with reservations under the ledger lock.
#[derive(Debug, Clone)]
pub struct PrefillCandidate<'a> {
    pub idx: usize,
    pub worker: &'a str,
    pub uncached_tokens: u64,
    /// Running-request work estimate that remains after the first output token.
    pub active_request_tokens: u64,
    /// A router-local count has no backend sample to reconcile with reservations.
    pub local_count_mean: Option<u64>,
    pub backend_work_tokens: u64,
    pub sampled_at: Option<Instant>,
    pub tokens_per_second: f64,
    pub kv_penalty_seconds: f64,
}

#[derive(Debug)]
struct Entry {
    dispatched_at: Option<Instant>,
    pending_tokens: u64,
    active_tokens: u64,
    counted_requests: u64,
}

#[derive(Debug, Default)]
struct Book {
    next_id: u64,
    workers: HashMap<String, HashMap<u64, Entry>>,
}

/// Shared by the registry's model policies, so aliases do not hide GPU work.
#[derive(Debug, Default)]
pub struct ReservationLedger {
    book: Mutex<Book>,
}

impl ReservationLedger {
    fn lock(&self) -> std::sync::MutexGuard<'_, Book> {
        self.book
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Returns the selected candidate position, its estimated seconds, and its lease.
    pub fn select_and_reserve(
        self: &Arc<Self>,
        candidates: &[PrefillCandidate<'_>],
    ) -> Option<(usize, f64, Reservation)> {
        self.select_and_reserve_with_local_counts(candidates, |_| 0)
    }

    /// Read local worker counters under the lock used to attach their load guards.
    pub(crate) fn select_and_reserve_with_local_counts(
        self: &Arc<Self>,
        candidates: &[PrefillCandidate<'_>],
        local_running: impl Fn(usize) -> usize,
    ) -> Option<(usize, f64, Reservation)> {
        let mut book = self.lock();
        let mut best: Option<(usize, f64)> = None;
        for (position, candidate) in candidates.iter().enumerate() {
            if !candidate.tokens_per_second.is_finite() || candidate.tokens_per_second <= 0.0 {
                continue;
            }
            let (mut reflected, mut newer, mut counted_requests) = (0u64, 0u64, 0u64);
            if let Some(entries) = book.workers.get(candidate.worker) {
                for entry in entries.values() {
                    counted_requests = counted_requests.saturating_add(entry.counted_requests);
                    let local_work = entry.pending_tokens.saturating_add(entry.active_tokens);
                    if entry.dispatched_at.is_none()
                        || candidate
                            .sampled_at
                            .is_some_and(|at| entry.dispatched_at.is_some_and(|sent| sent > at))
                    {
                        newer = newer.saturating_add(local_work);
                    } else {
                        reflected = reflected.saturating_add(local_work);
                    }
                }
            }
            // Undispatched reservations and dispatches newer than the sample cannot
            // be reflected. Older sent requests may overlap the report: max is an
            // estimate, not proof that the engine has received those requests.
            let existing_work = if let Some(mean) = candidate.local_count_mean {
                let uncovered =
                    (local_running(candidate.idx) as u64).saturating_sub(counted_requests);
                reflected
                    .saturating_add(newer)
                    .saturating_add(uncovered.saturating_mul(mean))
            } else {
                candidate
                    .backend_work_tokens
                    .max(reflected)
                    .saturating_add(newer)
            };
            let work = existing_work
                .saturating_add(candidate.uncached_tokens)
                .saturating_add(candidate.active_request_tokens);
            let cost = work as f64 / candidate.tokens_per_second + candidate.kv_penalty_seconds;
            if cost.is_finite() && best.is_none_or(|(_, minimum)| cost < minimum) {
                best = Some((position, cost));
            }
        }
        let (position, cost) = best?;
        let candidate = &candidates[position];
        let id = book.next_id.checked_add(1)?;
        book.next_id = id;
        book.workers
            .entry(candidate.worker.to_owned())
            .or_default()
            .insert(
                id,
                Entry {
                    dispatched_at: None,
                    pending_tokens: candidate.uncached_tokens,
                    active_tokens: candidate.active_request_tokens,
                    counted_requests: 0,
                },
            );
        drop(book);
        Some((
            candidate.idx,
            cost,
            Reservation {
                owner: Arc::new(ReservationOwner {
                    ledger: Arc::clone(self),
                    worker: candidate.worker.to_owned(),
                    id,
                    parts: AtomicUsize::new(1),
                    dispatch_parts: AtomicUsize::new(1),
                }),
                pending: true,
                dispatch_pending: true,
            },
        ))
    }

    fn prefill_complete(&self, worker: &str, id: u64) {
        if let Some(entry) = self
            .lock()
            .workers
            .get_mut(worker)
            .and_then(|entries| entries.get_mut(&id))
        {
            entry.pending_tokens = 0;
        }
    }

    fn dispatched(&self, worker: &str, id: u64) {
        if let Some(entry) = self
            .lock()
            .workers
            .get_mut(worker)
            .and_then(|entries| entries.get_mut(&id))
        {
            entry.dispatched_at.get_or_insert_with(Instant::now);
        }
    }

    fn release(&self, worker: &str, id: u64) {
        let mut book = self.lock();
        if let Some(entries) = book.workers.get_mut(worker) {
            entries.remove(&id);
            if entries.is_empty() {
                book.workers.remove(worker);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn pending(&self, worker: &str) -> u64 {
        self.lock().workers.get(worker).map_or(0, |entries| {
            entries
                .values()
                .fold(0u64, |sum, entry| sum.saturating_add(entry.pending_tokens))
        })
    }

    #[cfg(test)]
    pub(crate) fn active(&self, worker: &str) -> usize {
        self.lock().workers.get(worker).map_or(0, HashMap::len)
    }
}

#[derive(Debug)]
struct ReservationOwner {
    ledger: Arc<ReservationLedger>,
    worker: String,
    id: u64,
    parts: AtomicUsize,
    dispatch_parts: AtomicUsize,
}

impl Drop for ReservationOwner {
    fn drop(&mut self) {
        self.ledger.release(&self.worker, self.id);
    }
}

/// Owned by one dispatch future, then its raw response stream. Drop rolls back.
#[derive(Debug)]
pub struct Reservation {
    owner: Arc<ReservationOwner>,
    pending: bool,
    dispatch_pending: bool,
}

/// A worker's local load count and its coverage in the reservation ledger.
#[derive(Debug)]
pub(crate) struct CountedLoad<T> {
    guard: Option<T>,
    ledger: Arc<ReservationLedger>,
    worker: String,
    id: u64,
}

impl<T> Drop for CountedLoad<T> {
    fn drop(&mut self) {
        let mut book = self.ledger.lock();
        drop(self.guard.take());
        if let Some(entry) = book
            .workers
            .get_mut(&self.worker)
            .and_then(|entries| entries.get_mut(&self.id))
        {
            entry.counted_requests = 0;
        }
    }
}

impl Reservation {
    /// The worker this attempt's work is charged to.
    pub fn worker(&self) -> &str {
        &self.owner.worker
    }

    /// Attach synchronous worker load guards without exposing an unpaired count.
    pub(crate) fn counted_load<T>(&self, count: usize, make: impl FnOnce() -> T) -> CountedLoad<T> {
        let mut book = self.owner.ledger.lock();
        let guard = make();
        if let Some(entry) = book
            .workers
            .get_mut(&self.owner.worker)
            .and_then(|entries| entries.get_mut(&self.owner.id))
        {
            entry.counted_requests = count as u64;
        }
        drop(book);
        CountedLoad {
            guard: Some(guard),
            ledger: Arc::clone(&self.owner.ledger),
            worker: self.owner.worker.clone(),
            id: self.owner.id,
        }
    }

    /// Mark this branch sent once all its sibling dispatches have started or ended.
    pub fn dispatched(&mut self) {
        if self.dispatch_pending {
            self.dispatch_pending = false;
            if self.owner.dispatch_parts.fetch_sub(1, Ordering::AcqRel) == 1 {
                self.owner
                    .ledger
                    .dispatched(&self.owner.worker, self.owner.id);
            }
        }
    }

    /// Clear pending input work once every branch has produced output or ended.
    pub fn prefill_complete(&mut self) {
        if self.pending {
            self.pending = false;
            if self.owner.parts.fetch_sub(1, Ordering::AcqRel) == 1 {
                self.owner
                    .ledger
                    .prefill_complete(&self.owner.worker, self.owner.id);
            }
        }
    }

    /// A batched logical request keeps its work until every sibling starts or ends.
    /// Share one attempt across independently owned batch branches.
    pub fn split(mut self, parts: usize) -> Vec<Self> {
        let parts = parts.max(1);
        let pending = self.pending;
        if pending {
            self.owner.parts.fetch_add(parts - 1, Ordering::AcqRel);
        }
        let dispatch_pending = self.dispatch_pending;
        if dispatch_pending {
            self.owner
                .dispatch_parts
                .fetch_add(parts - 1, Ordering::AcqRel);
        }
        self.pending = false;
        self.dispatch_pending = false;
        (0..parts)
            .map(|_| Self {
                owner: Arc::clone(&self.owner),
                pending,
                dispatch_pending,
            })
            .collect()
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.prefill_complete();
        self.dispatched();
    }
}

/// An attempt-local handoff, borrowed only during synchronous worker selection.
#[derive(Debug)]
pub struct ReservationSlot {
    pub ledger: Arc<ReservationLedger>,
    inner: Mutex<SlotState>,
}

#[derive(Debug, Default)]
struct SlotState {
    request_count: usize,
    active_tokens: u64,
    total_tokens: Option<u64>,
    required_tokens: Option<u64>,
    lease: Option<Reservation>,
}

impl ReservationSlot {
    /// Create an attempt-local handoff over a shared worker ledger.
    pub fn new(ledger: Arc<ReservationLedger>) -> Self {
        Self {
            ledger,
            inner: Mutex::new(SlotState::default()),
        }
    }

    /// Affinity can use the first prompt, while admission prices every batch item.
    pub fn with_total_tokens(mut self, tokens: usize) -> Self {
        self.inner
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .total_tokens = Some(tokens as u64);
        self
    }

    /// Price the running allowance for every input in a batch.
    pub fn with_request_count(mut self, requests: usize) -> Self {
        self.inner
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .request_count = requests.max(1);
        self
    }

    /// Number of backend inputs represented by this attempt.
    pub fn request_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .request_count
            .max(1)
    }

    /// Set the per-input allowance retained while the attempt is alive.
    pub fn set_active_tokens(&self, per_request_tokens: u64) {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.active_tokens = per_request_tokens.saturating_mul(state.request_count.max(1) as u64);
    }

    /// Use the full batch size when known, at least as large as its first prompt.
    pub fn total_tokens_or(&self, first_prompt_tokens: usize) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .total_tokens
            .unwrap_or(first_prompt_tokens as u64)
            .max(first_prompt_tokens as u64)
    }

    /// Keep enough work to book a target that bypasses normal policy selection.
    pub fn require(&self, prompt_tokens: u64) {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.required_tokens = Some(prompt_tokens.max(state.total_tokens.unwrap_or(0)));
    }

    /// Replace the provisional choice, releasing the previous attempt's work.
    pub fn store(&self, lease: Reservation) {
        let old = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .lease
            .replace(lease);
        // Release outside both the slot lock and the selection's ledger lock.
        drop(old);
    }

    /// Sticky overrides and veto fallbacks can change or bypass a provisional pick.
    /// They keep their target; conservatively reserve its full prompt if necessary.
    pub fn take_for(&self, worker: &str) -> Option<Reservation> {
        let (lease, required, active_tokens) = {
            let mut state = self
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                state.lease.take(),
                state.required_tokens.take(),
                state.active_tokens,
            )
        };
        if lease.as_ref().is_some_and(|lease| lease.worker() == worker) {
            return lease;
        }
        drop(lease);
        let tokens = required?;
        self.ledger
            .select_and_reserve(&[PrefillCandidate {
                idx: 0,
                worker,
                uncached_tokens: tokens,
                active_request_tokens: active_tokens,
                local_count_mean: None,
                backend_work_tokens: 0,
                sampled_at: None,
                tokens_per_second: 1.0,
                kv_penalty_seconds: 0.0,
            }])
            .map(|(_, _, lease)| lease)
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Barrier, thread, time::Duration};

    use super::*;

    fn candidates() -> [PrefillCandidate<'static>; 2] {
        [
            PrefillCandidate {
                idx: 0,
                worker: "holder",
                uncached_tokens: 12288,
                active_request_tokens: 0,
                local_count_mean: None,
                backend_work_tokens: 0,
                sampled_at: None,
                tokens_per_second: 2000.0,
                kv_penalty_seconds: 0.0,
            },
            PrefillCandidate {
                idx: 1,
                worker: "cold",
                uncached_tokens: 16384,
                active_request_tokens: 0,
                local_count_mean: None,
                backend_work_tokens: 0,
                sampled_at: None,
                tokens_per_second: 2000.0,
                kv_penalty_seconds: 0.0,
            },
        ]
    }

    #[test]
    fn a_second_long_prompt_prices_the_reservation_before_choosing() {
        let ledger = Arc::new(ReservationLedger::default());
        let (first, _, first_lease) = ledger.select_and_reserve(&candidates()).unwrap();
        let (second, _, second_lease) = ledger.select_and_reserve(&candidates()).unwrap();
        assert_eq!((first, second), (0, 1));
        drop((first_lease, second_lease));
        assert!(ledger.lock().workers.is_empty());
    }

    #[test]
    fn simultaneous_choices_book_before_the_other_thread_scores() {
        for _ in 0..100 {
            let ledger = Arc::new(ReservationLedger::default());
            let barrier = Arc::new(Barrier::new(2));
            let workers: Vec<_> = (0..2)
                .map(|_| {
                    let (ledger, barrier) = (Arc::clone(&ledger), Arc::clone(&barrier));
                    thread::spawn(move || {
                        barrier.wait();
                        ledger.select_and_reserve(&candidates()).unwrap()
                    })
                })
                .collect();
            let picks: Vec<_> = workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect();
            assert_ne!(picks[0].0, picks[1].0);
            assert_eq!(ledger.active("holder") + ledger.active("cold"), 2);
        }
    }

    #[test]
    fn a_tiny_suffix_keeps_its_expensive_cached_prefix() {
        let ledger = Arc::new(ReservationLedger::default());
        let mut rows = candidates();
        rows[0].uncached_tokens = 64;
        let (_, _, first) = ledger.select_and_reserve(&rows).unwrap();
        let (second, _, second_lease) = ledger.select_and_reserve(&rows).unwrap();
        assert_eq!(second, 0);
        drop((first, second_lease));
    }

    #[test]
    fn first_token_releases_work_but_holds_the_attempt_until_drop() {
        let ledger = Arc::new(ReservationLedger::default());
        let (_, _, mut lease) = ledger.select_and_reserve(&candidates()).unwrap();
        lease.prefill_complete();
        lease.prefill_complete();
        assert_eq!(ledger.pending("holder"), 0);
        assert_eq!(ledger.active("holder"), 1);
        drop(lease);
        assert_eq!(ledger.active("holder"), 0);
    }

    #[test]
    fn out_of_order_cancellation_releases_that_attempt_only() {
        let ledger = Arc::new(ReservationLedger::default());
        let a = [candidates()[0].clone()];
        let (_, _, first) = ledger.select_and_reserve(&a).unwrap();
        let mut b = a.clone();
        b[0].uncached_tokens = 7;
        let (_, _, second) = ledger.select_and_reserve(&b).unwrap();
        drop(first);
        assert_eq!(ledger.pending("holder"), 7);
        drop(second);
        assert_eq!(ledger.active("holder"), 0);
    }

    #[test]
    fn a_report_does_not_double_charge_an_older_reservation() {
        let ledger = Arc::new(ReservationLedger::default());
        let (_, _, mut first) = ledger.select_and_reserve(&candidates()[..1]).unwrap();
        first.dispatched();
        let mut rows = candidates();
        rows[0].sampled_at = Some(Instant::now());
        rows[0].backend_work_tokens = 12288;
        let (_, cost, second) = ledger.select_and_reserve(&rows[..1]).unwrap();
        assert_eq!(cost, 24576.0 / 2000.0);
        drop((first, second));
    }

    #[test]
    fn a_report_cannot_hide_dispatches_newer_than_its_sample() {
        let ledger = Arc::new(ReservationLedger::default());
        let sample = Instant::now() - Duration::from_secs(1);
        let (_, _, mut first) = ledger.select_and_reserve(&candidates()[..1]).unwrap();
        first.dispatched();
        let mut rows = candidates();
        rows[0].sampled_at = Some(sample);
        rows[0].backend_work_tokens = 5000;
        let (_, cost, second) = ledger.select_and_reserve(&rows[..1]).unwrap();
        assert_eq!(cost, (5000.0 + 24576.0) / 2000.0);
        drop((first, second));
    }

    #[test]
    fn a_local_count_only_adds_requests_not_covered_by_the_ledger() {
        struct Guard<'a>(&'a AtomicUsize);
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::AcqRel);
            }
        }
        for (untracked, counted, dispatched) in
            [(0, true, false), (2, true, true), (2, false, false)]
        {
            let ledger = Arc::new(ReservationLedger::default());
            let live = AtomicUsize::new(untracked);
            let mut row = candidates()[0].clone();
            row.active_request_tokens = 1_024;
            row.local_count_mean = Some(1_024);
            let (_, _, mut first) = ledger
                .select_and_reserve_with_local_counts(&[row.clone()], |_| {
                    live.load(Ordering::Acquire)
                })
                .unwrap();
            let guard = counted.then(|| {
                first.counted_load(1, || {
                    live.fetch_add(1, Ordering::AcqRel);
                    Guard(&live)
                })
            });
            if dispatched {
                first.dispatched();
            }
            row.uncached_tokens = 0;
            let (_, cost, second) = ledger
                .select_and_reserve_with_local_counts(&[row], |_| live.load(Ordering::Acquire))
                .unwrap();
            assert_eq!(cost, (14_336 + untracked * 1_024) as f64 / 2_000.0);
            drop((second, guard, first));
        }
    }

    #[test]
    fn a_batched_attempt_waits_for_all_siblings_or_their_cancellation() {
        let ledger = Arc::new(ReservationLedger::default());
        let (_, _, lease) = ledger.select_and_reserve(&candidates()).unwrap();
        let mut parts = lease.split(2);
        parts[0].prefill_complete();
        assert_eq!(ledger.pending("holder"), 12288);
        parts.pop();
        assert_eq!(ledger.pending("holder"), 0);
        drop(parts);
        assert_eq!(ledger.active("holder"), 0);
    }

    #[test]
    fn a_sticky_retarget_rolls_back_the_provisional_worker() {
        let ledger = Arc::new(ReservationLedger::default());
        let slot = ReservationSlot::new(Arc::clone(&ledger));
        slot.require(16384);
        let (_, _, lease) = ledger.select_and_reserve(&candidates()).unwrap();
        slot.store(lease);
        let fixed = slot.take_for("pinned").unwrap();
        assert_eq!(ledger.pending("holder"), 0);
        assert_eq!(ledger.pending("pinned"), 16384);
        drop(fixed);
        assert!(ledger.lock().workers.is_empty());
    }

    #[test]
    fn dropping_selection_before_dispatch_rolls_back_its_work() {
        let ledger = Arc::new(ReservationLedger::default());
        {
            let slot = ReservationSlot::new(Arc::clone(&ledger));
            let (_, _, lease) = ledger.select_and_reserve(&candidates()).unwrap();
            slot.store(lease);
            assert_eq!(ledger.pending("holder"), 12288);
        }
        assert_eq!(ledger.pending("holder"), 0);
    }

    #[test]
    fn a_later_report_cannot_absorb_work_not_yet_dispatched() {
        let ledger = Arc::new(ReservationLedger::default());
        let (_, _, first) = ledger.select_and_reserve(&candidates()[..1]).unwrap();
        let mut rows = candidates();
        rows[0].sampled_at = Some(Instant::now());
        rows[0].backend_work_tokens = 10000;
        rows[0].uncached_tokens = 2000;
        let (_, cost, second) = ledger.select_and_reserve(&rows[..1]).unwrap();
        assert_eq!(cost, (10000.0 + 12288.0 + 2000.0) / 2000.0);
        drop((first, second));
    }

    #[test]
    fn splitting_a_sibling_keeps_all_other_siblings_pending() {
        let ledger = Arc::new(ReservationLedger::default());
        let (_, _, lease) = ledger.select_and_reserve(&candidates()).unwrap();
        let mut parts = lease.split(2);
        let mut nested = parts.pop().unwrap().split(2);
        parts[0].prefill_complete();
        nested[0].prefill_complete();
        assert_eq!(ledger.pending("holder"), 12288);
        nested[1].prefill_complete();
        assert_eq!(ledger.pending("holder"), 0);
        drop((parts, nested));
        assert_eq!(ledger.active("holder"), 0);
    }

    #[test]
    fn fully_cached_owners_are_still_priced_until_their_last_drop() {
        let ledger = Arc::new(ReservationLedger::default());
        let mut rows = candidates();
        for row in &mut rows {
            row.uncached_tokens = 0;
            row.active_request_tokens = 1024;
        }
        let (first, _, mut lease) = ledger.select_and_reserve(&rows).unwrap();
        lease.dispatched();
        lease.prefill_complete();
        let (second, _, other) = ledger.select_and_reserve(&rows).unwrap();
        assert_ne!(first, second);
        assert_eq!(ledger.pending("holder") + ledger.pending("cold"), 0);
        drop((lease, other));
        assert!(ledger.lock().workers.is_empty());
    }

    #[test]
    fn fixed_target_batch_preserves_its_running_cost_after_the_first_token() {
        let ledger = Arc::new(ReservationLedger::default());
        let slot = ReservationSlot::new(Arc::clone(&ledger))
            .with_total_tokens(16384)
            .with_request_count(3);
        slot.require(64);
        slot.set_active_tokens(1024);
        let mut lease = slot.take_for("holder").unwrap();
        lease.prefill_complete();
        let mut rows = candidates();
        for row in &mut rows {
            row.uncached_tokens = 0;
            row.active_request_tokens = 1024;
        }
        let (next, _, other) = ledger.select_and_reserve(&rows).unwrap();
        assert_eq!(next, 1, "the fixed target still owns three running items");
        drop((lease, other));
        assert!(ledger.lock().workers.is_empty());
    }

    #[test]
    fn a_partial_batch_dispatch_is_not_absorbed_by_a_later_report() {
        let ledger = Arc::new(ReservationLedger::default());
        let (_, _, lease) = ledger.select_and_reserve(&candidates()[..1]).unwrap();
        let mut parts = lease.split(2);
        parts[0].dispatched();
        let mut rows = candidates();
        rows[0].sampled_at = Some(Instant::now());
        rows[0].backend_work_tokens = 10000;
        rows[0].uncached_tokens = 2000;
        let (_, cost, other) = ledger.select_and_reserve(&rows[..1]).unwrap();
        assert_eq!(cost, (10000.0 + 12288.0 + 2000.0) / 2000.0);
        drop((parts, other));
    }
}
