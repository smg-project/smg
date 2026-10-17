use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::{Duration, SystemTime},
};

use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use rand::RngExt;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::{
    get_healthy_worker_indices, normalize_model_key, utils::PeriodicTask, BucketConfig,
    LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::worker::Worker;

#[derive(Debug)]
pub struct BucketPolicy {
    config: BucketConfig,
    buckets: Arc<DashMap<String, Arc<RwLock<Bucket>>>>,
    _adjustment_task: Option<PeriodicTask>,
}

impl Default for BucketPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl BucketPolicy {
    pub fn new() -> Self {
        Self::with_config(BucketConfig::default())
    }

    pub fn with_config(config: BucketConfig) -> Self {
        let buckets = Arc::new(DashMap::<String, Arc<RwLock<Bucket>>>::new());

        let adjustment_task = {
            let buckets_clone = Arc::clone(&buckets);

            Some(PeriodicTask::spawn(
                config.bucket_adjust_interval_secs as u64,
                "BucketAdjustment",
                move || {
                    for bucket_ref in buckets_clone.iter() {
                        let bucket = bucket_ref.value();
                        let mut bucket_guard = bucket.write();
                        bucket_guard.adjust_boundary();
                    }
                },
            ))
        };

        Self {
            config,
            buckets,
            _adjustment_task: adjustment_task,
        }
    }

    pub fn init_prefill_worker_urls(&self, prefill_workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        let mut model_workers: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in prefill_workers {
            let model_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(model_key.to_string())
                .or_default()
                .push(worker);
        }
        // Initialize bucket for each model
        for (model_key, model_workers) in model_workers {
            let bucket = self
                .buckets
                .entry(model_key)
                .or_insert_with(|| {
                    Arc::new(RwLock::new(Bucket::new(
                        self.config.bucket_adjust_interval_secs * 1000,
                    )))
                })
                .clone();

            let worker_urls: Vec<String> = model_workers
                .iter()
                .map(|worker| worker.url().to_string())
                .collect();

            let mut bucket_guard = bucket.write();
            bucket_guard.init_prefill_worker_urls(worker_urls);
        }
    }
}

impl LoadBalancingPolicy for BucketPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Request-size signal: token count for pre-tokenized requests, else
        // chars. Boundaries adapt to whichever unit the traffic carries.
        let char_count = match (info.tokens, info.request_text) {
            (Some(tokens), _) => tokens.len(),
            (None, Some(text)) => text.chars().count(),
            (None, None) => 0,
        };

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let model_key = normalize_model_key(workers[healthy_indices[0]].model_id());

        let bucket = self
            .buckets
            .get(model_key)
            .map(|entry| entry.value().clone());
        if let Some(bucket) = bucket {
            let (choiced_url, chars_per_url_snapshot) = {
                let buc = bucket.read();
                let chars_per_url_snapshot = buc.chars_per_url.lock().clone();
                let choiced_url = buc.find_boundary(char_count);
                (choiced_url, chars_per_url_snapshot)
            };
            // The imbalance test and the spill weigh the offered healthy
            // workers alone: a listed worker nobody offers drains to zero and
            // would otherwise drag the minimum down and draw every spill to a
            // worker that cannot serve.
            let offered: Vec<(&String, usize)> = chars_per_url_snapshot
                .iter()
                .filter(|(url, _)| {
                    healthy_indices
                        .iter()
                        .any(|&i| workers[i].url() == url.as_str())
                })
                .map(|(url, &chars)| (url, chars))
                .collect();
            let max_load = offered.iter().map(|&(_, chars)| chars).max().unwrap_or(0);
            let min_load = offered.iter().map(|&(_, chars)| chars).min().unwrap_or(0);
            let abs_diff = max_load.saturating_sub(min_load);
            let rel_threshold = self.config.balance_rel_threshold * min_load as f32;
            let is_imbalanced =
                abs_diff > self.config.balance_abs_threshold && max_load as f32 > rel_threshold;
            debug!(
                "Current PD instance status | is_imbalanced={}",
                is_imbalanced
            );

            let mut rng = rand::rng();
            let prefill_url = if is_imbalanced {
                debug!("select prefill instance by Load Balance policy");
                let min_url = offered
                    .iter()
                    .min_by_key(|&&(_, chars)| chars)
                    .map(|&(url, _)| url.clone())
                    .unwrap_or_else(|| {
                        let idx = rng.random_range(0..healthy_indices.len());
                        let url = workers[healthy_indices[idx]].url();
                        warn!("No URL found, randomly selecting: {}", url);
                        url.to_string()
                    });
                min_url
            } else {
                debug!("select prefill instance by Bucket policy");
                match choiced_url {
                    Some(url) if !url.is_empty() => url,
                    _ => {
                        let idx = rng.random_range(0..healthy_indices.len());
                        let selected_url = workers[healthy_indices[idx]].url();
                        warn!("Boundary not found, randomly selection: {}", selected_url);
                        selected_url.to_string()
                    }
                }
            };

            // The bucket chooses among the workers it lists; the caller offers
            // the ones available now (on a retry, the ones not yet tried). A
            // listed worker that is not offered, or not healthy, gives way to
            // an offered one, and the load is booked to the worker that serves
            // the request.
            let position = healthy_indices
                .iter()
                .copied()
                .find(|&i| workers[i].url() == prefill_url)
                .unwrap_or_else(|| {
                    let fallback = healthy_indices[rng.random_range(0..healthy_indices.len())];
                    warn!(
                        "Bucket worker {} is not among the offered workers, randomly selecting: {}",
                        prefill_url,
                        workers[fallback].url()
                    );
                    fallback
                });

            {
                let mut buc = bucket.write();
                buc.post_process_request(char_count, workers[position].url().to_string());
            }

            return Some(position);
        }

        warn!(
            "No bucket found for model {}, randomly selecting healthy worker",
            model_key
        );
        let mut rng = rand::rng();
        Some(healthy_indices[rng.random_range(0..healthy_indices.len())])
    }

    /// A worker that left the fleet leaves every bucket that listed it; a
    /// neighbour takes its size range over (see
    /// [`Bucket::remove_prefill_worker_url`]). A URL names one worker across
    /// models, so every bucket is asked.
    fn remove_worker(&self, url: &str) {
        let buckets: Vec<(String, Arc<RwLock<Bucket>>)> = self
            .buckets
            .iter()
            .map(|entry| (entry.key().clone(), Arc::clone(entry.value())))
            .collect();
        for (model_key, bucket) in buckets {
            let mut bucket = bucket.write();
            if bucket.remove_prefill_worker_url(url) {
                info!(
                    "Removed worker {} from bucket for model {} (remaining workers: {})",
                    url, model_key, bucket.bucket_cnt
                );
            }
        }
    }

    fn name(&self) -> &'static str {
        "bucket"
    }

    fn needs_request_text(&self) -> bool {
        true // Bucket policy needs request text
    }

    /// The size signal is the token count when the ids are there and the
    /// text's length only without them, so a tokenized request keeps no
    /// copy of its prompt for a retry's re-selection.
    fn needs_request_text_with_tokens(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The declared span of request sizes a listing of the workers splits evenly
/// among them; the last range is open-ended regardless. A constant, so that no
/// path can widen it: the adjustment learns the real sizes from the requests
/// seen.
const L_MAX: usize = 4096;

#[derive(Debug, Clone)]
pub struct Bucket {
    bucket_cnt: usize,
    pub prefill_worker_urls: Arc<Mutex<Vec<String>>>,
    load_total: usize,
    pub period: usize,
    bucket_load: usize,
    boundary: Vec<Boundary>,
    request_list: VecDeque<SequencerRequest>,
    t_req_loads: HashMap<String, usize>,
    pub chars_per_url: Arc<Mutex<HashMap<String, usize>>>,
}

#[derive(Debug, Clone)]
pub struct SequencerRequest {
    pub id: String,
    pub char_cnt: usize,
    pub timestamp: SystemTime,
    pub prefill_worker_url: String,
}

#[derive(Debug, Clone)]
pub struct Boundary {
    pub url: String,
    pub range: [usize; 2],
}

impl Boundary {
    pub fn new(url: String, range: [usize; 2]) -> Self {
        Boundary { url, range }
    }
}

impl Bucket {
    pub fn new(period: usize) -> Self {
        let bucket_cnt = 0;

        let load_total = 0;
        let bucket_load = 0;

        let t_req_loads = HashMap::new();
        let request_list = VecDeque::new();

        let initial_map = HashMap::new();

        let boundary = Vec::new();

        let prefill_worker_urls = Arc::new(Mutex::new(Vec::new()));

        Bucket {
            bucket_cnt,
            prefill_worker_urls,
            load_total,
            period,
            bucket_load,
            boundary,
            request_list,
            t_req_loads,
            chars_per_url: Arc::new(Mutex::new(initial_map)),
        }
    }

    /// List the prefill workers the bucket partitions the sizes over, as the
    /// registration step does whenever a worker joins (it hands the bucket
    /// every prefill worker again). The same set again, in any order, is a
    /// no-op: the partition and the load counters stay. A changed set keeps
    /// the counters of the workers that stay, starts the newcomers at zero
    /// and splits the declared span [`L_MAX`] evenly among all of them with
    /// the last range open-ended, exactly as the first split does; the span
    /// is a constant, so no later split divides the open end. The even split
    /// is provisional: the next adjustment rebuilds it from the sizes seen
    /// without waiting for the per-worker load to double
    /// ([`Self::adjust_boundary`]).
    pub fn init_prefill_worker_urls(&mut self, prefill_worker_urls: Vec<String>) {
        {
            let mut urls = self.prefill_worker_urls.lock();
            let listed: HashSet<&str> = urls.iter().map(String::as_str).collect();
            let incoming: HashSet<&str> = prefill_worker_urls.iter().map(String::as_str).collect();
            if listed == incoming {
                return;
            }
            urls.clone_from(&prefill_worker_urls);
        }
        {
            let mut chars = self.chars_per_url.lock();
            chars.retain(|url, _| prefill_worker_urls.contains(url));
            for url in &prefill_worker_urls {
                chars.entry(url.clone()).or_insert(0);
            }
        }
        let worker_cnt = prefill_worker_urls.len();
        self.bucket_cnt = worker_cnt;
        self.boundary = match L_MAX.checked_div(worker_cnt) {
            None => Vec::new(),
            Some(gap) => {
                let gap = gap.max(1);
                prefill_worker_urls
                    .iter()
                    .enumerate()
                    .map(|(i, url)| {
                        let min = i * gap;
                        let max = if i == worker_cnt - 1 {
                            usize::MAX
                        } else {
                            (i + 1) * gap - 1
                        };
                        Boundary::new(url.clone(), [min, max])
                    })
                    .collect()
            }
        };
        self.bucket_load = 0;
        info!("Init boundary:{:?}", self.boundary);
    }

    pub fn post_process_request(&mut self, char_cnt: usize, prefill_url: String) {
        {
            // Only a worker the bucket counts takes the load: a selection that
            // overlapped the worker's removal must not put its counter back,
            // where it would draw the spill towards a URL no caller offers.
            let mut map = self.chars_per_url.lock();
            if let Some(count) = map.get_mut(&prefill_url) {
                *count += char_cnt;
            }
        }

        let now = SystemTime::now();
        let time_window_duration = Duration::from_millis(self.period as u64);
        let mut removed_load = 0;

        while let Some(req) = self.request_list.front() {
            let expired = match now.duration_since(req.timestamp) {
                Ok(duration) => duration > time_window_duration,
                Err(_) => true,
            };

            if !expired {
                break;
            }

            if let Some(removed_req) = self.request_list.pop_front() {
                self.t_req_loads.remove(&removed_req.id);
                removed_load += removed_req.char_cnt;

                let mut map = self.chars_per_url.lock();
                if let Some(count) = map.get_mut(&removed_req.prefill_worker_url) {
                    *count = count.saturating_sub(removed_req.char_cnt);
                }
            }
        }

        self.load_total = self.load_total.saturating_sub(removed_load);

        let id = Uuid::now_v7().to_string();

        self.t_req_loads.insert(id.clone(), char_cnt);

        self.request_list.push_back(SequencerRequest {
            id,
            char_cnt,
            timestamp: now,
            prefill_worker_url: prefill_url,
        });

        self.load_total = self.load_total.saturating_add(char_cnt);
    }

    /// Forget `url`: its place among the workers, its load counter and its
    /// range, which the previous neighbour takes over (the next one when it
    /// was first). The partition stays contiguous until the next adjustment
    /// rebuilds it from the request sizes seen: the reference load of the
    /// last rebuild is cleared, so that tick re-partitions among the workers
    /// left instead of holding a partition whose per-worker load moved by
    /// less than the factor of two it otherwise waits for. Whether the bucket
    /// held it.
    pub fn remove_prefill_worker_url(&mut self, url: &str) -> bool {
        let listed = {
            let mut urls = self.prefill_worker_urls.lock();
            let before = urls.len();
            urls.retain(|u| u != url);
            self.bucket_cnt = urls.len();
            urls.len() != before
        };
        if listed {
            self.bucket_load = 0;
        }
        let counted = self.chars_per_url.lock().remove(url).is_some();
        let Some(i) = self.boundary.iter().position(|b| b.url == url) else {
            return listed || counted;
        };
        let gone = self.boundary.remove(i);
        if let Some(previous) = i.checked_sub(1) {
            self.boundary[previous].range[1] = gone.range[1];
        } else if let Some(next) = self.boundary.first_mut() {
            next.range[0] = gone.range[0];
        }
        true
    }

    pub fn find_boundary(&self, char_count: usize) -> Option<String> {
        let mut left = 0;
        let mut right = self.boundary.len();
        let mut _steps = 0;

        while left < right {
            _steps += 1;
            let mid = left + (right - left) / 2;
            let range = self.boundary[mid].range;

            if char_count < range[0] {
                right = mid;
            } else if char_count > range[1] {
                left = mid + 1;
            } else {
                return Some(self.boundary[mid].url.clone());
            }
        }
        None
    }

    pub fn get_total_load(&self) -> usize {
        self.load_total
    }

    fn update_workers_cnt(&mut self) {
        let pwu = self.prefill_worker_urls.lock();
        self.bucket_cnt = pwu.len();

        let mut char_map = self.chars_per_url.lock();
        let current_urls: HashSet<_> = char_map.keys().cloned().collect();
        let new_urls: HashSet<_> = pwu.iter().cloned().collect();

        for url in new_urls.difference(&current_urls) {
            char_map.insert(url.clone(), 0);
        }

        for url in current_urls.difference(&new_urls) {
            if char_map.get(url) == Some(&0) {
                char_map.remove(url);
            }
        }
    }

    pub fn adjust_boundary(&mut self) {
        if self.t_req_loads.is_empty() {
            return;
        }

        self.update_workers_cnt();
        let worker_cnt = self.bucket_cnt;
        if worker_cnt == 0 {
            return;
        }
        let new_single_bucket_load = self.get_total_load() / worker_cnt;
        let old_single_bucket_load = self.bucket_load;

        if new_single_bucket_load <= 2 * old_single_bucket_load
            && (old_single_bucket_load <= 2 * new_single_bucket_load && old_single_bucket_load != 0)
        {
            info!("No need to adjust the bucket boundaries.");
            return;
        }

        info!("Before adjusting boundary | {:?}", self.boundary);
        self.bucket_load = new_single_bucket_load;
        let mut new_boundary = Vec::new();
        let mut hist_load: Vec<usize> = self.t_req_loads.values().copied().collect();
        hist_load.sort_unstable();
        let mut upper_bound: usize = 0;
        let mut last_load_index: usize = 0;
        let max_value = usize::MAX;

        let worker_url = {
            let guard = self.prefill_worker_urls.lock();
            (*guard).clone()
        };

        let mut iter = worker_url.iter().peekable();
        // let mut curr_worker_id = 0;
        while let Some(url) = iter.next() {
            if last_load_index >= hist_load.len() && iter.peek().is_none() {
                new_boundary.push(Boundary::new(url.clone(), [upper_bound, max_value]));
                break;
            }
            let mut load_accumulator = 0;
            let mut break_flag = false;
            for &load in &hist_load[last_load_index..] {
                load_accumulator += load;
                if load_accumulator >= new_single_bucket_load {
                    if iter.peek().is_none() {
                        new_boundary.push(Boundary::new(url.clone(), [upper_bound, max_value]));
                        break_flag = true;
                        break;
                    }
                    let real_load = upper_bound + new_single_bucket_load;
                    if load <= upper_bound {
                        new_boundary.push(Boundary::new(url.clone(), [upper_bound, real_load]));
                        upper_bound = real_load + 1;
                    } else {
                        new_boundary.push(Boundary::new(url.clone(), [upper_bound, load]));
                        upper_bound = load + 1;
                    }
                    last_load_index += 1;
                    break_flag = true;
                    break;
                } else {
                    last_load_index += 1;
                }
            }
            if !break_flag {
                let mut right_bound_value = upper_bound + new_single_bucket_load;
                if iter.peek().is_none() {
                    right_bound_value = max_value;
                    new_boundary.push(Boundary::new(url.clone(), [upper_bound, right_bound_value]));
                    break;
                }
                new_boundary.push(Boundary::new(url.clone(), [upper_bound, right_bound_value]));
                upper_bound = right_bound_value + 1;
            }
        }
        self.boundary = new_boundary;
        info!("After adjusting boundary | {:?}", self.boundary);
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::HealthCheckConfig;
    use tracing_test::traced_test;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_drop_stops_adjustment_thread() {
        let config = BucketConfig {
            // Long interval so the thread is parked in its sleep loop at drop time.
            bucket_adjust_interval_secs: 3600,
            ..Default::default()
        };
        let policy = BucketPolicy::with_config(config);

        // Clones alive here: the policy's own, the adjustment thread's, and this
        // one. If the thread leaks on drop, its clone outlives the policy and the
        // count stays above 1.
        let buckets = Arc::clone(&policy.buckets);
        assert_eq!(Arc::strong_count(&buckets), 3);

        drop(policy);

        assert_eq!(
            Arc::strong_count(&buckets),
            1,
            "adjustment thread should be joined on drop, releasing its Arc clone"
        );
    }

    #[tokio::test]
    async fn test_load_balancing_conditions() {
        // Test 1: Basic load balancing trigger
        let config = BucketConfig {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 10,
        };
        let policy = BucketPolicy::with_config(config);
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // Initialize the policy with prefill_workers
        policy.init_prefill_worker_urls(&prefill_workers);

        // === Phase S1: Construct bucket boundaries ===
        // Requests len =33 -> Bucket 1(expected range: 0-33)
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(33)),
                    ..Default::default()
                },
            )
            .unwrap();
        // Two requests len =34 ->load balancing
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(34)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(34)),
                    ..Default::default()
                },
            )
            .unwrap();

        tokio::time::sleep(Duration::from_secs(11)).await;
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let bucket_guard = bucket.write();
                {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 33);
                    assert_eq!(bucket_guard.boundary[1].range[1], 67);
                }
            }
        }
        // === Phase S2: Validate load balancing ===
        // Three consecutive len=33 requests (Should route to different buckets)
        let idx_1 = policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(33)),
                    ..Default::default()
                },
            )
            .unwrap();
        let idx_2 = policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(33)),
                    ..Default::default()
                },
            )
            .unwrap();
        let idx_3 = policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(33)),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx_1, 0, "Should not trigger load balancing");
        assert_ne!(idx_2, idx_3, "Should trigger load balancing");
        assert_ne!(idx_2, 0, "Should trigger load balancing");
        assert_ne!(idx_3, 0, "Should trigger load balancing");

        // Test 2: Not triggering when absolute threshold not met
        let config = BucketConfig {
            balance_abs_threshold: 30,
            balance_rel_threshold: 2.0,
            ..Default::default()
        };
        let policy = BucketPolicy::with_config(config);
        policy.init_prefill_worker_urls(&prefill_workers);

        // Create load difference below absolute threshold(20 + 8 = 28 < 30)
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(20)),
                    ..Default::default()
                },
            )
            .unwrap(); // worker1: 20
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(8)),
                    ..Default::default()
                },
            )
            .unwrap(); // worker1: 8

        // Next request should not use bucket scheduling (no load balancing)
        let idx = policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some("request"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            idx, 0,
            "Should not trigger load balancing when relative threshold not met"
        );

        // Test 3: Not triggering when relative threshold not met
        let config = BucketConfig {
            balance_abs_threshold: 5,
            balance_rel_threshold: 3.0,
            ..Default::default()
        };
        let policy = BucketPolicy::with_config(config);
        policy.init_prefill_worker_urls(&prefill_workers);

        // Create load difference (but relative threshold not met)
        // Max/Min ratio = 15/5 = 3.0
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(15)),
                    ..Default::default()
                },
            )
            .unwrap(); // worker1: 15
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some("short"),
                    ..Default::default()
                },
            )
            .unwrap(); // worker2: 5
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(10)),
                    ..Default::default()
                },
            )
            .unwrap(); // worker3: 10

        // Next request should use bucket scheduling (load balancing)
        let idx = policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some("request"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            idx, 0,
            "Should not trigger load balancing when relative threshold not met"
        );
    }

    #[tokio::test]
    async fn test_adjust_boundary_1() {
        // Test configuration: Set high threshold to prevent load balancing policy.
        let config = BucketConfig {
            balance_abs_threshold: 300,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 3,
        };
        let policy = BucketPolicy::with_config(config);
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // Initialize the policy with prefill_workers
        policy.init_prefill_worker_urls(&prefill_workers);

        // Initial boundary
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let bucket_guard = bucket.write();
                {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 1364);
                    assert_eq!(bucket_guard.boundary[1].range[1], 2729);
                }
            }
        }

        // ===Phase S1: Initial requests to trigger boundary adjustment ===
        // Send requests with lengths: [5, 10, 15, 20, 24, 26] (total = 100)
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(5)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(10)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(15)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(20)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(24)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(26)),
                    ..Default::default()
                },
            )
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        // Verify boundaries adjusted to: [0, 20], [21, 26], [27, MAX]
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let bucket_guard = bucket.write();
                {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 20);
                    assert_eq!(bucket_guard.boundary[1].range[1], 26);
                }
            }
        }

        // ===Phase S2: Second set of  requests to trigger boundary adjustment ===
        // Send requests with lengths: [10, 20, 30, 40, 45, 57] (total = 202)
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(10)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(20)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(30)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(40)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(45)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(57)),
                    ..Default::default()
                },
            )
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        // Verify boundaries adjusted to: [0, 40], [41, 57], [58, MAX]
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let bucket_guard = bucket.write();
                {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 40);
                    assert_eq!(bucket_guard.boundary[1].range[1], 57);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_adjust_boundary_2() {
        let config = BucketConfig {
            balance_abs_threshold: 300,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 3,
        };
        let policy = BucketPolicy::with_config(config);
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // Initialize the policy with prefill_workers
        policy.init_prefill_worker_urls(&prefill_workers);

        // Initial boundary
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let bucket_guard = bucket.write();
                {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 1364);
                    assert_eq!(bucket_guard.boundary[1].range[1], 2729);
                }
            }
        }

        // Send requests with char_count 20
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(20)),
                    ..Default::default()
                },
            )
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let bucket_guard = bucket.write();
                {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 20);
                    assert_eq!(bucket_guard.boundary[1].range[1], 27);
                }
            }
        }

        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(7)),
                    ..Default::default()
                },
            )
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let bucket_guard = bucket.write();
                {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 7);
                    assert_eq!(bucket_guard.boundary[1].range[1], 10);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_not_adjust_boundary() {
        let config = BucketConfig {
            balance_abs_threshold: 300,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 3,
        };
        let policy = BucketPolicy::with_config(config);
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // Initialize the policy with prefill_workers
        policy.init_prefill_worker_urls(&prefill_workers);

        // Initial boundary
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let bucket_guard = bucket.write();
                {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 1364);
                    assert_eq!(bucket_guard.boundary[1].range[1], 2729);
                }
            }
        }

        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(5)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(10)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(15)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(20)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(24)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(26)),
                    ..Default::default()
                },
            )
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let bucket_guard = bucket.write();
                {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 20);
                    assert_eq!(bucket_guard.boundary[1].range[1], 26);
                }
            }
        }

        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(10)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(20)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(30)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(32)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(45)),
                    ..Default::default()
                },
            )
            .unwrap();
        policy
            .select_worker(
                &prefill_workers,
                &SelectWorkerInfo {
                    request_text: Some(&*"a".repeat(55)),
                    ..Default::default()
                },
            )
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        {
            let model_key = "default";

            let bucket = policy
                .buckets
                .get(model_key)
                .map(|entry| entry.value().clone());
            if let Some(bucket) = bucket {
                let bucket_guard = bucket.write();
                {
                    // Expected Boundary: [0, 33] [34, 67] [68, MAX]
                    assert_eq!(bucket_guard.boundary[0].range[1], 20);
                    assert_eq!(bucket_guard.boundary[1].range[1], 26);
                }
            }
        }
    }

    fn prefill_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .worker_type(WorkerType::Prefill)
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    /// A policy whose selection follows the boundaries alone: the load spill
    /// never fires and the periodic adjustment never runs within a test.
    fn boundary_only_policy() -> BucketPolicy {
        BucketPolicy::with_config(BucketConfig {
            balance_abs_threshold: usize::MAX,
            bucket_adjust_interval_secs: 3600,
            ..Default::default()
        })
    }

    /// The boundaries of the bucket `workers` share, as (url, range).
    fn boundaries(policy: &BucketPolicy, workers: &[Arc<dyn Worker>]) -> Vec<(String, [usize; 2])> {
        let entry = policy
            .buckets
            .get(normalize_model_key(workers[0].model_id()))
            .expect("the workers' model has a bucket");
        let bucket = entry.read();
        bucket
            .boundary
            .iter()
            .map(|b| (b.url.clone(), b.range))
            .collect()
    }

    fn select(policy: &BucketPolicy, workers: &[Arc<dyn Worker>], size: usize) -> Option<usize> {
        let text = "a".repeat(size);
        policy.select_worker(
            workers,
            &SelectWorkerInfo {
                request_text: Some(&text),
                ..Default::default()
            },
        )
    }

    /// A removed worker leaves the bucket with its range: a request of a size
    /// it owned goes to the neighbour that took the range over, not to no
    /// worker because a boundary still named a departed URL.
    #[test]
    fn a_removed_worker_leaves_the_boundaries_and_a_neighbour_takes_its_range() {
        let policy = boundary_only_policy();
        let workers = prefill_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        policy.init_prefill_worker_urls(&workers);
        // The first split hands 4096 out evenly: w2 owns [1365, 2729].
        assert_eq!(
            boundaries(&policy, &workers),
            [
                ("http://w1:8000".to_string(), [0, 1364]),
                ("http://w2:8000".to_string(), [1365, 2729]),
                ("http://w3:8000".to_string(), [2730, usize::MAX]),
            ]
        );

        policy.remove_worker("http://w2:8000");

        let remaining = [Arc::clone(&workers[0]), Arc::clone(&workers[2])];
        let selected = select(&policy, &remaining, 2000)
            .expect("a request in the removed worker's range selects a worker");
        assert_eq!(remaining[selected].url(), "http://w1:8000");
        assert_eq!(
            boundaries(&policy, &workers),
            [
                ("http://w1:8000".to_string(), [0, 2729]),
                ("http://w3:8000".to_string(), [2730, usize::MAX]),
            ]
        );
        let entry = policy
            .buckets
            .get(normalize_model_key(workers[0].model_id()))
            .unwrap();
        let bucket = entry.read();
        assert_eq!(
            *bucket.prefill_worker_urls.lock(),
            ["http://w1:8000", "http://w3:8000"]
        );
        assert!(!bucket.chars_per_url.lock().contains_key("http://w2:8000"));
        assert_eq!(bucket.bucket_cnt, 2);
    }

    /// At either end the one neighbour takes the range; the last worker
    /// standing owns every size; an emptied bucket still selects among the
    /// workers the caller offers; a URL the bucket never held is a no-op.
    #[test]
    fn removal_at_either_end_hands_the_range_to_the_one_neighbour() {
        let policy = boundary_only_policy();
        let workers = prefill_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        policy.init_prefill_worker_urls(&workers);

        policy.remove_worker("http://w1:8000");
        assert_eq!(
            boundaries(&policy, &workers),
            [
                ("http://w2:8000".to_string(), [0, 2729]),
                ("http://w3:8000".to_string(), [2730, usize::MAX]),
            ]
        );
        policy.remove_worker("http://w3:8000");
        assert_eq!(
            boundaries(&policy, &workers),
            [("http://w2:8000".to_string(), [0, usize::MAX])]
        );
        let only = [Arc::clone(&workers[1])];
        for size in [0, 1365, 4096, 100_000] {
            assert_eq!(select(&policy, &only, size), Some(0), "size {size}");
        }

        policy.remove_worker("http://w2:8000");
        policy.remove_worker("http://w9:8000");
        assert!(boundaries(&policy, &workers).is_empty());
        let newcomer = prefill_workers(&["http://w4:8000"]);
        assert_eq!(select(&policy, &newcomer, 2000), Some(0));
    }

    /// After a removal the next adjustment re-partitions among the workers
    /// left: the reference load of the last rebuild is cleared, so the skip
    /// for a per-worker load that moved by less than a factor of two (three
    /// workers to two is a factor of one and a half) does not hold the merged
    /// range in place.
    #[test]
    fn a_removal_lets_the_next_adjustment_re_partition_among_the_survivors() {
        let policy = boundary_only_policy();
        let workers = prefill_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        policy.init_prefill_worker_urls(&workers);
        for size in [100, 200, 3000] {
            select(&policy, &workers, size).expect("a worker serves the request");
        }
        let bucket = {
            let entry = policy
                .buckets
                .get(normalize_model_key(workers[0].model_id()))
                .expect("the workers' model has a bucket");
            Arc::clone(entry.value())
        };
        bucket.write().adjust_boundary();
        assert_eq!(
            bucket.read().bucket_load,
            1100,
            "the first rebuild keeps the per-worker load as its reference"
        );

        policy.remove_worker("http://w2:8000");
        assert_eq!(bucket.read().bucket_load, 0);
        let merged = boundaries(&policy, &workers);
        assert_eq!(
            merged,
            [
                ("http://w1:8000".to_string(), [0, 4101]),
                ("http://w3:8000".to_string(), [4102, usize::MAX]),
            ]
        );

        bucket.write().adjust_boundary();
        assert_eq!(
            boundaries(&policy, &workers),
            [
                ("http://w1:8000".to_string(), [0, 3000]),
                ("http://w3:8000".to_string(), [3001, usize::MAX]),
            ],
            "the tick after a removal re-partitions among the survivors"
        );
        assert_eq!(bucket.read().bucket_load, 1650);
    }

    /// A selection that chose a worker before its removal and books its load
    /// after it does not bring the worker's counter back.
    #[test]
    fn a_selection_overlapping_a_removal_does_not_resurrect_the_departed_worker() {
        let policy = boundary_only_policy();
        let workers = prefill_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_prefill_worker_urls(&workers);
        policy.remove_worker("http://w2:8000");

        let entry = policy
            .buckets
            .get(normalize_model_key(workers[0].model_id()))
            .expect("the workers' model has a bucket");
        let mut bucket = entry.write();
        bucket.post_process_request(500, "http://w2:8000".to_string());
        assert!(!bucket.chars_per_url.lock().contains_key("http://w2:8000"));
        bucket.post_process_request(500, "http://w1:8000".to_string());
        assert_eq!(
            bucket.chars_per_url.lock().get("http://w1:8000"),
            Some(&500)
        );
    }

    /// A worker the bucket lists but the caller does not offer (not available
    /// now, or already tried on a retry) gives way to an offered worker, and
    /// the load is booked to the worker that serves the request.
    #[test]
    fn a_listed_worker_the_caller_does_not_offer_gives_way_to_an_offered_one() {
        let policy = boundary_only_policy();
        let workers = prefill_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        policy.init_prefill_worker_urls(&workers);

        // 2000 is in w2's range [1365, 2729]; w2 is listed but not offered.
        let offered = [Arc::clone(&workers[0]), Arc::clone(&workers[2])];
        let selected =
            select(&policy, &offered, 2000).expect("an offered worker serves the request");
        let served = offered[selected].url().to_string();

        let entry = policy
            .buckets
            .get(normalize_model_key(workers[0].model_id()))
            .expect("the workers' model has a bucket");
        let bucket = entry.read();
        let counted = bucket.chars_per_url.lock();
        assert_eq!(counted.get("http://w2:8000"), Some(&0));
        assert_eq!(counted.get(served.as_str()), Some(&2000));
    }

    /// The imbalance test and the spill weigh the offered workers alone: a
    /// listed worker nobody offers drains to zero and must neither trip the
    /// spill on every request nor receive it.
    #[test]
    fn the_spill_weighs_the_offered_workers_alone() {
        for _ in 0..8 {
            let policy = BucketPolicy::with_config(BucketConfig {
                balance_abs_threshold: 100,
                balance_rel_threshold: 1.1,
                bucket_adjust_interval_secs: 3600,
            });
            let workers = prefill_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
            policy.init_prefill_worker_urls(&workers);
            // w2 is listed but never offered; w1 takes 50 chars, then w3 3000.
            let offered = [Arc::clone(&workers[0]), Arc::clone(&workers[2])];
            assert_eq!(select(&policy, &offered, 50), Some(0));
            assert_eq!(select(&policy, &offered, 3000), Some(1));
            // Among the offered workers w3 is 2950 chars ahead: the spill sends
            // the next request to w1, never to the unoffered w2.
            assert_eq!(select(&policy, &offered, 3000), Some(0));
            let entry = policy
                .buckets
                .get(normalize_model_key(workers[0].model_id()))
                .expect("the workers' model has a bucket");
            let bucket = entry.read();
            let counted = bucket.chars_per_url.lock();
            assert_eq!(counted.get("http://w2:8000"), Some(&0));
            assert_eq!(counted.get("http://w1:8000"), Some(&3050));
        }
    }

    /// The ranges of `bucket`'s partition, as (url, range).
    fn bucket_ranges(bucket: &Bucket) -> Vec<(String, [usize; 2])> {
        bucket
            .boundary
            .iter()
            .map(|b| (b.url.clone(), b.range))
            .collect()
    }

    /// A prefill worker that joins after the first split is re-listed with
    /// the others (the registration step hands the bucket every prefill
    /// worker again): the sizes split among all of them as the first split
    /// did. A mid-size request reaches the second worker and a large one the
    /// newcomer; the first worker does not take every size.
    #[test]
    fn a_prefill_worker_re_listed_after_the_first_split_takes_its_share_of_the_sizes() {
        let policy = boundary_only_policy();
        let two = prefill_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_prefill_worker_urls(&two);
        assert_eq!(
            boundaries(&policy, &two),
            [
                ("http://w1:8000".to_string(), [0, 2047]),
                ("http://w2:8000".to_string(), [2048, usize::MAX]),
            ]
        );

        let three = prefill_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        policy.init_prefill_worker_urls(&three);

        assert_eq!(
            boundaries(&policy, &three),
            [
                ("http://w1:8000".to_string(), [0, 1364]),
                ("http://w2:8000".to_string(), [1365, 2729]),
                ("http://w3:8000".to_string(), [2730, usize::MAX]),
            ]
        );
        for (size, owner) in [
            (100, "http://w1:8000"),
            (2000, "http://w2:8000"),
            (3000, "http://w3:8000"),
            (100_000, "http://w3:8000"),
        ] {
            let selected = select(&policy, &three, size)
                .unwrap_or_else(|| panic!("a request of {size} chars selects no worker"));
            assert_eq!(three[selected].url(), owner, "size {size}");
        }
    }

    /// A registration pass that changed nothing about the prefill workers
    /// (a decode worker joined, say) re-lists the same URLs, in whatever
    /// order the registry returns them: the partition and the load counters
    /// stay as they were.
    #[test]
    fn re_listing_the_same_prefill_workers_keeps_the_partition_and_the_counters() {
        let policy = boundary_only_policy();
        let workers = prefill_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        policy.init_prefill_worker_urls(&workers);
        assert_eq!(select(&policy, &workers, 2000), Some(1));
        let partition = boundaries(&policy, &workers);

        let same_in_another_order =
            prefill_workers(&["http://w3:8000", "http://w1:8000", "http://w2:8000"]);
        policy.init_prefill_worker_urls(&same_in_another_order);

        assert_eq!(boundaries(&policy, &workers), partition);
        let entry = policy
            .buckets
            .get(normalize_model_key(workers[0].model_id()))
            .unwrap();
        let bucket = entry.read();
        assert_eq!(
            bucket.chars_per_url.lock().get("http://w2:8000"),
            Some(&2000)
        );
        assert_eq!(bucket.bucket_cnt, 3);
    }

    /// A changed membership leaves the even split provisional: the next
    /// adjustment rebuilds the partition from the sizes seen, as the first
    /// adjustment does, without waiting for the per-worker load to double.
    /// The sizes seen are 100, 200, 300, 2000, 2100 and 2200; with three
    /// workers a third of their sum, 2300, is one worker's share.
    #[test]
    fn a_changed_membership_lets_the_next_adjustment_re_split_from_the_sizes_seen() {
        let url = |n: usize| format!("http://w{n}:8000");
        let mut bucket = Bucket::new(3_600_000);
        bucket.init_prefill_worker_urls(vec![url(1), url(2)]);
        for size in [100, 200, 300, 2000, 2100, 2200] {
            let owner = bucket.find_boundary(size).expect("every size has an owner");
            bucket.post_process_request(size, owner);
        }
        bucket.adjust_boundary();
        assert_eq!(
            bucket_ranges(&bucket),
            [(url(1), [0, 2100]), (url(2), [2101, usize::MAX])]
        );
        // Steady load: the second adjustment leaves the partition alone.
        bucket.adjust_boundary();
        assert_eq!(
            bucket_ranges(&bucket),
            [(url(1), [0, 2100]), (url(2), [2101, usize::MAX])]
        );

        bucket.init_prefill_worker_urls(vec![url(1), url(2), url(3)]);
        assert_eq!(
            bucket_ranges(&bucket),
            [
                (url(1), [0, 1364]),
                (url(2), [1365, 2729]),
                (url(3), [2730, usize::MAX]),
            ]
        );

        bucket.adjust_boundary();
        assert_eq!(
            bucket_ranges(&bucket),
            [
                (url(1), [0, 2000]),
                (url(2), [2001, 2200]),
                (url(3), [2201, usize::MAX]),
            ]
        );
    }

    /// A listed worker the caller does not offer (draining before its removal,
    /// or already tried on a retry) hands its range to the neighbour the
    /// removal will hand it to: the previous range's owner, the next one when
    /// it was first. Every request of the range goes there, none at random.
    #[test]
    #[traced_test]
    fn an_unoffered_workers_range_goes_to_the_neighbour_the_removal_would_pick() {
        let policy = boundary_only_policy();
        let workers = prefill_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        policy.init_prefill_worker_urls(&workers);

        // w2 owns [1365, 2729] and is not offered: w1, the previous neighbour,
        // serves its range, request after request.
        let without_w2 = [Arc::clone(&workers[0]), Arc::clone(&workers[2])];
        for _ in 0..32 {
            assert_eq!(select(&policy, &without_w2, 2000), Some(0));
        }
        // w1 owns the first range and is not offered: w2, the next, takes it.
        let without_w1 = [Arc::clone(&workers[1]), Arc::clone(&workers[2])];
        for _ in 0..32 {
            assert_eq!(select(&policy, &without_w1, 100), Some(0));
        }
        // Only w3 is offered: it serves every size.
        let only_w3 = [Arc::clone(&workers[2])];
        for size in [100, 2000, 3000] {
            assert_eq!(select(&policy, &only_w3, size), Some(0), "size {size}");
        }
        // A normal drain is not worth a warning per request.
        assert!(!logs_contain("randomly selecting"));
    }
}
