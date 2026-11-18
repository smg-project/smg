//! All-fleet prefill placement with native KV overlap and per-attempt work leases.
//!
//! Input-token capacity prices input work; output-token throughput is never a
//! substitute. A missing KV index contributes zero cache credit, and selection
//! still uses the shared reservation ledger across every eligible worker.

pub mod reservation;

use std::{any::Any, collections::HashSet, sync::Arc};

use kv_index::{compute_request_content_hashes, salt::request_content_hashes_with_seed};
use openai_protocol::worker::WorkerLoadResponse;
use parking_lot::RwLock;

use super::{
    expected_wait::{
        ExpectedWait, DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT,
    },
    get_healthy_worker_indices, normalize_model_key,
    prefill_cost::reservation::PrefillCandidate,
    LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::worker::{
    load::state::{LoadReceiver, LoadSnapshot},
    KvEventMonitor, Worker,
};

/// Prefill placement using native KV overlap and outstanding input work.
#[derive(Debug)]
pub struct PrefillCostPolicy {
    block_size: usize,
    kv_pressure_weight: f64,
    mean_prefill_tokens: u32,
    default_throughput: f64,
    kv_monitor: RwLock<Option<Arc<KvEventMonitor>>>,
    load_rx: RwLock<Option<LoadReceiver>>,
}

impl Default for PrefillCostPolicy {
    fn default() -> Self {
        Self::with_params(
            16,
            DEFAULT_KV_PRESSURE_WEIGHT,
            DEFAULT_MEAN_PREFILL_TOKENS,
            DEFAULT_THROUGHPUT,
        )
    }
}

impl PrefillCostPolicy {
    /// Use the default block size, work allowance and input-rate fallback.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure input-work scoring; throughput is measured in input tokens/s.
    pub fn with_params(
        block_size: usize,
        kv_pressure_weight: f64,
        mean_prefill_tokens: u32,
        default_throughput: f64,
    ) -> Self {
        Self {
            block_size: block_size.max(1),
            kv_pressure_weight: if kv_pressure_weight.is_finite() && kv_pressure_weight >= 0.0 {
                kv_pressure_weight
            } else {
                DEFAULT_KV_PRESSURE_WEIGHT
            },
            mean_prefill_tokens: mean_prefill_tokens.max(1),
            default_throughput: if default_throughput.is_finite() && default_throughput > 0.0 {
                default_throughput
            } else {
                DEFAULT_THROUGHPUT
            },
            kv_monitor: RwLock::new(None),
            load_rx: RwLock::new(None),
        }
    }

    /// Read prefix reuse from this monitor, or count the full prompt without one.
    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        *self.kv_monitor.write() = monitor;
    }

    fn snapshot(&self) -> Option<Arc<LoadSnapshot>> {
        self.load_rx.read().as_ref().map(|rx| rx.borrow().clone())
    }

    /// Partial DP reports and invalid input rates use nominal peer capacity.
    fn input_rate(worker: &dyn Worker, load: &WorkerLoadResponse) -> Option<f64> {
        if let Some(rank) = worker.dp_rank() {
            if load.loads.len() != 1 || load.loads[0].dp_rank != i32::try_from(rank).ok()? {
                return None;
            }
        } else {
            let labels = &worker.metadata().spec.labels;
            let dimension = |name: &str| -> Option<usize> {
                labels
                    .get(name)
                    .map_or(Some(1), |value| value.parse().ok())
                    .filter(|value| *value > 0)
            };
            let expected = dimension("dp_size")?
                .checked_mul(dimension("attn_dp_size")?)?
                .max(worker.dp_size().unwrap_or(1))
                .max(load.dp_rank_count.max(1) as usize);
            if load.loads.len() != expected {
                return None;
            }
            let mut ranks = HashSet::with_capacity(load.loads.len());
            for row in &load.loads {
                let rank = usize::try_from(row.dp_rank).ok()?;
                if rank >= expected || !ranks.insert(rank) {
                    return None;
                }
            }
        }
        if load.loads.is_empty() {
            return None;
        }
        let rate = load.loads.iter().try_fold(0.0, |sum, rank| {
            rank.prefill_throughput
                .filter(|rate| rate.is_finite() && *rate > 0.0)
                .map(|rate| sum + rate)
        })?;
        (rate.is_finite() && rate > 0.0).then_some(rate)
    }

    fn cached_tokens(
        &self,
        workers: &[Arc<dyn Worker>],
        eligible: &[usize],
        info: &SelectWorkerInfo,
    ) -> Vec<u64> {
        let mut cached = vec![0; workers.len()];
        let Some(tokens) = info.tokens else {
            return cached;
        };
        let Some(monitor) = self.kv_monitor.read().clone() else {
            return cached;
        };
        let Some(&first) = eligible.first() else {
            return cached;
        };
        let model = normalize_model_key(workers[first].model_id());
        let Some(index) = monitor.indexers.get(model).map(|entry| Arc::clone(&entry)) else {
            return cached;
        };
        // Engine-reported block size takes precedence over the fallback.
        let block_size = monitor.block_size(model).unwrap_or(self.block_size).max(1);
        let hashes = match info.cache_namespace {
            Some(namespace) => {
                request_content_hashes_with_seed(tokens, block_size, namespace.event_seed())
            }
            None => compute_request_content_hashes(tokens, block_size),
        };
        if hashes.is_empty() {
            return cached;
        }
        let mut overlap = index.find_matches(&hashes, false);
        index.cap_by_groups(&hashes, block_size, &mut overlap);
        for &idx in eligible {
            if let Some(blocks) = index
                .worker_id(workers[idx].url())
                .and_then(|id| overlap.scores.get(&id))
            {
                cached[idx] = (*blocks as u64)
                    .saturating_mul(block_size as u64)
                    .min(tokens.len() as u64);
            }
        }
        cached
    }

    fn backend_work(&self, load: &WorkerLoadResponse) -> u64 {
        let waiting = load.total_waiting_reqs().max(0) as u64;
        let reported_queue = load.total_waiting_uncached_tokens().max(0) as u64;
        let queued = if reported_queue == 0 && waiting > 0 {
            waiting.saturating_mul(self.mean_prefill_tokens as u64)
        } else {
            reported_queue
        };
        // No remaining-running-prefill gauge is available. The configured mean
        // estimates running work and includes decode requests on regular workers.
        queued.saturating_add(
            (load.total_running_reqs().max(0) as u64)
                .saturating_mul(self.mean_prefill_tokens as u64),
        )
    }
}

impl LoadBalancingPolicy for PrefillCostPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        // Never silently preview with an isolated ledger. Validation rejects
        // transports that cannot hand a lease to their dispatch context.
        let slot = info.reservation?;
        let first_prompt = info
            .tokens
            .map_or(self.mean_prefill_tokens as usize, <[u32]>::len);
        let total_prompt = slot.total_tokens_or(first_prompt);
        slot.require(total_prompt);
        slot.set_active_tokens(u64::from(self.mean_prefill_tokens));
        let active_request_tokens =
            u64::from(self.mean_prefill_tokens).saturating_mul(slot.request_count() as u64);
        let eligible = get_healthy_worker_indices(workers, info);
        if eligible.is_empty() {
            return None;
        }
        let snapshot = self.snapshot();
        let (mut rate_sum, mut rate_count) = (0.0, 0);
        for &idx in &eligible {
            if let Some(rate) = snapshot
                .as_ref()
                .and_then(|loads| loads.get(workers[idx].url()))
                .and_then(|load| Self::input_rate(workers[idx].as_ref(), load))
            {
                rate_sum += rate;
                rate_count += 1;
            }
        }
        let nominal = if rate_count > 0 && rate_sum.is_finite() {
            rate_sum / rate_count as f64
        } else {
            self.default_throughput
        };
        let cached = self.cached_tokens(workers, &eligible, info);
        let candidates: Vec<_> = eligible
            .iter()
            .map(|&idx| {
                let worker = &workers[idx];
                let (backend_work_tokens, sampled_at, rate, pressure, local_count_mean) =
                    match snapshot
                        .as_ref()
                        .and_then(|loads| loads.report(worker.url()))
                    {
                        Some((load, sampled_at)) => (
                            self.backend_work(load),
                            Some(sampled_at),
                            Self::input_rate(worker.as_ref(), load).unwrap_or(nominal),
                            load.effective_kv_pressure(),
                            None,
                        ),
                        None => (
                            0,
                            None,
                            nominal,
                            0.0,
                            Some(u64::from(self.mean_prefill_tokens)),
                        ),
                    };
                let pressure = if pressure.is_nan() { 0.0 } else { pressure };
                PrefillCandidate {
                    idx,
                    worker: worker.url(),
                    // Native overlap covers the first prompt only; other batch
                    // prompts are conservatively charged in full.
                    uncached_tokens: total_prompt.saturating_sub(cached[idx]),
                    active_request_tokens,
                    local_count_mean,
                    backend_work_tokens,
                    sampled_at,
                    tokens_per_second: rate,
                    kv_penalty_seconds: ExpectedWait::new(
                        0.0,
                        rate,
                        pressure,
                        self.kv_pressure_weight,
                    )
                    .seconds(0),
                }
            })
            .collect();
        let (idx, _, lease) = slot
            .ledger
            .select_and_reserve_with_local_counts(&candidates, |idx| workers[idx].load())?;
        slot.store(lease);
        workers[idx].increment_processed();
        Some(idx)
    }

    fn name(&self) -> &'static str {
        "prefill_cost"
    }
    fn reservation_active_tokens(&self) -> Option<u64> {
        Some(u64::from(self.mean_prefill_tokens))
    }
    fn needs_backend_loads(&self) -> bool {
        true
    }
    fn needs_kv_events(&self) -> bool {
        true
    }
    fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        *self.load_rx.write() = rx;
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests;
