//! The monitor's own sink of the load port: the receipts of the pushed
//! records and the poll decision they give, the overload verdict written on
//! the registry's worker, the cache of rank loads, and the trim of what the
//! routing side booked. [`WorkerMonitor::new`](super::WorkerMonitor::new)
//! builds it over the registry and the routing side it is handed.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use tracing::warn;

use super::{dp_rank_token_loads, LoadConsumer, WorkerLoadManager};
use crate::{
    health::overload::{log_overload_edge, OverloadReading},
    identity::basic::Worker,
    load::port::{GroupTick, LoadSink, PollMode, PolledMember, PolledRanks},
    registry::{Generation, WorkerRegistry},
    worker::WorkerLoadResponse,
};

pub(super) struct ClassicLoadSink<G> {
    worker_registry: Arc<WorkerRegistry<G>>,
    load_consumer: Arc<dyn LoadConsumer>,
    worker_load_manager: Arc<WorkerLoadManager>,
    /// When each rank of a worker last pushed a load record, by receipt:
    /// the tick's poll decision reads it (see [`PollMode`]). Keyed by the
    /// worker's URL, a string the host registered, and every record looks
    /// its worker up: it hashes with FxHash.
    pushed_at: Mutex<FxHashMap<String, BTreeMap<i32, Instant>>>,
}

impl<G: Generation> ClassicLoadSink<G> {
    pub(super) fn new(
        worker_registry: Arc<WorkerRegistry<G>>,
        load_consumer: Arc<dyn LoadConsumer>,
        worker_load_manager: Arc<WorkerLoadManager>,
    ) -> Self {
        Self {
            worker_registry,
            load_consumer,
            worker_load_manager,
            pushed_at: Mutex::new(FxHashMap::default()),
        }
    }

    /// The absolute overload verdict of one load report (`None`: a poll that
    /// brought none, which clears the veto; absent means no opinion),
    /// applied to `worker` under its effective thresholds (resolved at
    /// registration; nothing happens while they are off). An edge is an
    /// INFO line with the report's figures, the thresholds and `source`:
    /// a worker vetoed by overload reads as "unavailable" in a refusal line
    /// exactly like one with an open breaker, and the breaker and the
    /// liveness veto log their edges already. Re-asserting the verdict the
    /// worker carries writes nothing; the counters and the gauge move once
    /// per edge, in the registry.
    fn apply_overload_verdict(
        &self,
        worker: &Arc<dyn Worker>,
        load: Option<&WorkerLoadResponse>,
        source: &'static str,
    ) {
        let overload = worker.metadata().overload;
        if !overload.is_enabled() {
            return;
        }
        let verdict = load.is_some_and(|load| overload.is_overloaded(load));
        if !self.worker_registry.set_worker_overloaded(worker, verdict) {
            return;
        }
        let reading = OverloadReading {
            waiting_requests: load.map_or(0, WorkerLoadResponse::total_waiting_reqs),
            kv_pressure: load.map_or(0.0, WorkerLoadResponse::effective_kv_pressure),
            thresholds: overload,
        };
        log_overload_edge(worker.as_ref(), verdict, &reading, source);
    }

    /// The poll decision for `url` at `now` from the receipt times of its
    /// pushed records: none on file polls, a record within `interval` on
    /// every pushing rank suppresses the poll, an older one on any rank
    /// falls back to it.
    pub(super) fn poll_mode_of(&self, url: &str, now: Instant, interval: Duration) -> PollMode {
        let pushed = self.pushed_at.lock();
        match pushed.get(url) {
            Some(ranks) if !ranks.is_empty() => {
                let fresh = ranks
                    .values()
                    .all(|&at| now.saturating_duration_since(at) < interval);
                if fresh {
                    PollMode::Suppressed
                } else {
                    PollMode::Fallback
                }
            }
            _ => PollMode::Poll,
        }
    }

    /// A poll's report of `url` (`load`, asked at `polled_at`) is the newer
    /// figure for every rank that pushed no record since the poll began.
    /// Under data parallelism the ranks' streams are not equally busy: a
    /// rank whose working set is cached, or that has nothing in flight,
    /// stores no blocks, and the heartbeats that keep the worker's stream
    /// alive name the last active rank, so a quiet rank's own record stops
    /// arriving while the worker's report keeps, for that rank, the record
    /// it last sent. The tick's decision ([`Self::poll_mode_of`]) falls back
    /// to the poll for such a worker; the poll's entry for the quiet rank
    /// replaces the one on file, so the report the next push publishes
    /// carries it and the rank's figures are at most a tick interval old,
    /// the poll decision's own bound. A rank whose record arrived since the
    /// poll began keeps it: the stream's is the newer. The receipt of a rank
    /// that stays is not touched: a poll is not the stream working, and the
    /// worker stays polled until the rank's records resume. Only a rank on
    /// file is replaced: a rank the stream never named has no receipt for the
    /// poll decision to age, so a poll's figure for it would sit in the
    /// report unrefreshed once the pushing ranks are fresh again; it stays
    /// out of the pushed report, and the poll's report stands for it while
    /// the worker is polled. A worker with no pushed record on file has no
    /// report to merge into; the poll's report stands on its own, as always.
    /// A rank on file that pushed nothing since the poll began and that the
    /// answer omits leaves the file, its row and its receipt, as after an
    /// empty answer: the file and the poll's publish say one thing of a rank,
    /// and the worker's next pushed record does not publish the old row
    /// again. The answer that may retract a rank from the file is one whose
    /// entries the rank cache takes as the worker's ranks, and no other: its
    /// entries are ranks and it carries token capacity, which stands in for
    /// the engine's own list of its ranks. An answer the cache does not take
    /// leaves the cache nothing of the worker and takes nothing from the
    /// file: the old row stays. A retraction reads one more word of the
    /// answer, where the two tests part: an answer that counts more ranks
    /// than it lists is short by its own count, so it says nothing of a rank
    /// it leaves out and retracts none, though the cache takes its rows as
    /// the counts to pick by. A rank worker's file holds its own rank and its
    /// answer is scoped to that rank, so an answer with rows never omits it.
    /// Names, in `ranks`, the ranks on file that pushed since the poll began
    /// (the stream's, the newer), named by the poll's answer or not, and the
    /// ranks that leave the file; the rows are the collector's to keep, lay
    /// or drop.
    pub(super) fn merge(
        &self,
        url: &str,
        load: &WorkerLoadResponse,
        on_file: Option<&WorkerLoadResponse>,
        polled_at: Instant,
        ranks: &mut PolledRanks,
    ) {
        // A rollup from a gateway registered as a worker is keyed by
        // downstream worker, not by rank: its entries all say rank 0 and
        // would collapse into one slot (the DP cache applies the same
        // guard).
        if !load.ranks_are_dp_ranks() {
            return;
        }
        let Some(report) = on_file else {
            return;
        };
        let mut pushed_at = self.pushed_at.lock();
        let mut receipts = pushed_at.get_mut(url);
        // The rank cache's test: the entries are ranks (the guard above)
        // and the answer carries token capacity. The capacity stands in for
        // the engine's own list of its ranks, so a rank such an answer omits
        // is retracted. An answer with none retracts nothing: the row
        // synthesized from backend-wide gauges, which names one rank and
        // says nothing of the others, and the answer of an engine that
        // reports no capacity, or none yet. A retraction reads the answer's
        // own count too, and here it parts from the cache's test: an answer
        // that counts more ranks than it lists says ranks exist that it
        // does not list, so it retracts none. A count of zero says nothing
        // against the list.
        let lists_ranks = load.has_absolute_token_data() && load.lists_every_rank();
        // Every rank on file: one that pushed since the poll began keeps its
        // row and hands it to the publish, whether the poll's answer names
        // the rank or not; any other takes the answer's entry for it, and
        // one a listing omits leaves the file, row and receipt, as after an
        // empty answer: what a poll retracts it retracts from the file, so
        // no later record of another rank publishes the old row again.
        for row in &report.loads {
            let pushed_since = receipts
                .as_deref()
                .and_then(|times| times.get(&row.dp_rank))
                .is_some_and(|&at| at >= polled_at);
            if pushed_since {
                ranks.newer.push(row.dp_rank);
                continue;
            }
            if load.loads.iter().any(|entry| entry.dp_rank == row.dp_rank) || !lists_ranks {
                continue;
            }
            if let Some(times) = receipts.as_deref_mut() {
                times.remove(&row.dp_rank);
            }
            ranks.leaving.push(row.dp_rank);
        }
    }

    /// Retract pre-poll pushed ranks after an authoritative empty response.
    /// Records received since the poll began remain available for the next
    /// push. Names, in `leaving`, the rows on file whose receipt is gone.
    fn retract(
        &self,
        url: &str,
        rank: Option<usize>,
        on_file: Option<&WorkerLoadResponse>,
        polled_at: Instant,
        leaving: &mut Vec<i32>,
    ) {
        let mut receipts = self.pushed_at.lock();
        let Some(times) = receipts.get_mut(url) else {
            return;
        };
        times.retain(|&dp_rank, at| {
            rank.is_some_and(|rank| usize::try_from(dp_rank) != Ok(rank)) || *at >= polled_at
        });
        if let Some(report) = on_file {
            leaving.extend(
                report
                    .loads
                    .iter()
                    .map(|load| load.dp_rank)
                    .filter(|dp_rank| !times.contains_key(dp_rank)),
            );
        }
    }

    /// Whether a receipt of `rank` stands for `url`, for the cases.
    #[cfg(test)]
    pub(super) fn has_receipt(&self, url: &str, rank: i32) -> bool {
        self.pushed_at
            .lock()
            .get(url)
            .is_some_and(|ranks| ranks.contains_key(&rank))
    }

    /// One polled member of an engine's answer: rows are merged, an answer
    /// without a row retracts unless it counts ranks, and either that or a
    /// failed fetch clears the overload veto, since absent means no opinion.
    fn member(&self, member: &PolledMember<'_>, polled_at: Instant, ranks: &mut PolledRanks) {
        let worker = member.worker;
        let url = worker.url();
        match member.answer {
            Some(load) if !load.loads.is_empty() => {
                self.merge(url, load, member.on_file, polled_at, ranks);
            }
            Some(load) => {
                self.apply_overload_verdict(worker, None, "poll");
                // A successful native poll can retract a rank. A failed poll
                // leaves the last-known DP cache alone; an explicit omission
                // of a rank that pushed nothing since the tick began must not
                // keep that rank selectable or be published as an idle load.
                // An answer that counts ranks and lists none is short by its
                // own count, as in `merge`: it retracts no rank from the file.
                if load.dp_rank_count <= 0 {
                    self.retract(
                        url,
                        worker.dp_rank(),
                        member.on_file,
                        polled_at,
                        &mut ranks.leaving,
                    );
                }
                if let Some(rank) = worker.dp_rank() {
                    self.worker_load_manager.remove_rank(url, rank);
                } else {
                    self.worker_load_manager.remove_worker(url);
                }
            }
            // A failed fetch means no fresh signal, which clears the flag.
            None => self.apply_overload_verdict(worker, None, "poll"),
        }
    }
}

impl<G: Generation> LoadSink for ClassicLoadSink<G> {
    fn record(
        &self,
        worker: &Arc<dyn Worker>,
        rank: i32,
        report: &WorkerLoadResponse,
        received_at: Instant,
        _sampled_at: Instant,
    ) {
        let url = worker.url();
        // The poll decision reads the receipt, not the sample's age: a record
        // that arrives is a stream that works, whatever the engine's clock.
        // A worker's entry is made on its first record; the lookup that
        // finds it copies no key.
        {
            let mut receipts = self.pushed_at.lock();
            if let Some(ranks) = receipts.get_mut(url) {
                ranks.insert(rank, received_at);
            } else {
                receipts.insert(url.to_string(), BTreeMap::from([(rank, received_at)]));
            }
        }
        // What a poll does with a fresh report, read off the worker's own.
        self.apply_overload_verdict(worker, Some(report), "pushed record");
    }

    fn poll_mode(&self, worker: &Arc<dyn Worker>, now: Instant, interval: Duration) -> PollMode {
        self.poll_mode_of(worker.url(), now, interval)
    }

    fn report(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> PolledRanks {
        let mut ranks = PolledRanks::default();
        for member in answer {
            self.member(member, polled_at, &mut ranks);
        }
        ranks
    }

    fn report_empty(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> Vec<i32> {
        self.report(answer, polled_at).leaving
    }

    fn fetch_failed(&self, members: &[Arc<dyn Worker>], _polled_at: Instant) {
        for worker in members {
            self.apply_overload_verdict(worker, None, "poll");
        }
    }

    fn forget(&self, worker: &Arc<dyn Worker>) {
        let url = worker.url();
        // No load feed, no verdict: a flag left set would strand the worker
        // out of routing with nothing to ever clear it.
        self.worker_registry.set_worker_overloaded(worker, false);
        self.worker_load_manager.remove_worker(url);
        // The receipts go with the worker: a replacement is polled until
        // its own stream pushes.
        self.pushed_at.lock().remove(url);
    }

    fn reset(&self) {
        self.worker_load_manager.clear();
        // The feed that would clear the vetoes is being torn down: fail open,
        // but say so — a wiped gauge is otherwise indistinguishable from a
        // genuine recovery. With no thresholds anywhere no flag was ever
        // written, so this clears nothing and stays silent. The line keeps
        // the monitor's log target. The receipts stay: the collector said
        // `forget` for every pusher an event it lost would have evicted,
        // and a worker that stays is polled again once its receipt is
        // older than a tick.
        let cleared = self.worker_registry.clear_all_overload_flags();
        if cleared > 0 {
            warn!(
                target: "smg_worker::load::monitor",
                vetoes_cleared = cleared,
                "Overload vetoes cleared with the load feed; every worker is routable again \
                 until its next poll"
            );
        }
    }

    fn needs_loads(&self) -> bool {
        self.load_consumer.needs_loads()
    }

    fn polls(&self, worker: &dyn Worker) -> bool {
        self.load_consumer.polls(worker)
    }

    fn tick(&self, tick: &GroupTick<'_>) {
        // The overload predicate runs exactly here, once per report, never
        // on a request path, against the worker's effective thresholds
        // (resolved at registration).
        for polled in tick.reports {
            self.apply_overload_verdict(polled.worker, Some(polled.report), "poll");
        }
        // The poll is also the reconciliation tick: each policy trims what it
        // booked on a worker to the requests the router still holds there,
        // the safety net behind the load guard's completion report. It runs
        // for every worker, polled or served by its stream, whether or not a
        // fetch succeeded; the live count is the router's.
        for worker in tick.workers {
            self.load_consumer.reconcile_in_flight(worker.as_ref());
        }
        // The DP cache deliberately keeps last-known-good entries so routing
        // decisions still have a hint to fall back to when the upstream is
        // briefly unreachable.
        if tick.reports.is_empty() {
            return;
        }
        let mut group_dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
        let mut dp_evict: Vec<String> = Vec::new();
        for polled in tick.reports {
            let url = polled.worker.url().to_string();
            // Only feed the DP-rank cache from responses that carry real
            // absolute per-rank token counts. Ratio-only snapshots,
            // which would otherwise poison with a fake `{0: 0}`
            // entry and collapse DP routing onto rank 0.
            //
            // A fleet rollup from a gateway worker is keyed by downstream
            // worker, not by rank, so its repeated `dp_rank: 0` entries
            // would overwrite each other down to a single bogus rank.
            if polled.carries_rank_capacity {
                group_dp_loads.insert(url, dp_rank_token_loads(polled.report));
            } else {
                dp_evict.push(url);
            }
        }
        self.worker_load_manager.update_dp_loads(&group_dp_loads);
        if !dp_evict.is_empty() {
            self.worker_load_manager.remove_workers(&dp_evict);
        }
    }
}
