//! Bounded managed worker discovery and local reservations, independent of legacy policies.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use sha2::{Digest, Sha256};
use smg_grpc_client::sglang_managed::{
    clock,
    proto::{
        self, managed_encoder_client::ManagedEncoderClient,
        managed_sglang_client::ManagedSglangClient,
    },
    LAYOUT, VERSION,
};
use tonic::{transport::Channel, Status};

use super::{
    executor::AttemptObserver,
    planner::{Contract, EncodeCost, EncoderCandidate, PdCandidate, PdCost, Worker},
    router::Config,
};

struct Reservation {
    operation: proto::OperationRef,
    encoder: bool,
}

#[derive(Default)]
struct State {
    encoders: HashMap<String, LoadTracker>,
    capacities: HashMap<String, u64>,
    records: HashMap<String, Reservation>,
    generation: u64,
    watched: HashMap<String, (proto::Capabilities, Instant)>,
}

pub struct Registry {
    config: Config,
    state: Mutex<State>,
    channels: HashMap<String, Channel>,
}

impl Registry {
    /// This registry never mutates worker objects or any legacy policy during candidate scoring.
    pub fn new(config: Config) -> Result<Arc<Self>, Status> {
        let channels = config
            .encoders
            .iter()
            .map(|e| &e.endpoint)
            .chain(config.pairs.iter().flat_map(|p| [&p.prefill, &p.decode]))
            .map(|endpoint| {
                Channel::from_shared(endpoint.clone())
                    .map(|channel| (endpoint.clone(), channel.connect_lazy()))
                    .map_err(|error| Status::invalid_argument(error.to_string()))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let registry = Arc::new(Self {
            config,
            state: Mutex::new(State::default()),
            channels,
        });
        let endpoints =
            registry
                .config
                .encoders
                .iter()
                .map(|e| (e.endpoint.clone(), "encoder"))
                .chain(
                    registry.config.pairs.iter().flat_map(|p| {
                        [(p.prefill.clone(), "prefill"), (p.decode.clone(), "decode")]
                    }),
                )
                .collect::<std::collections::HashSet<_>>();
        for (endpoint, role) in endpoints {
            let weak = Arc::downgrade(&registry);
            #[expect(
                clippy::disallowed_methods,
                reason = "one bounded monitor per configured endpoint; weak ownership stops it on router drop"
            )]
            tokio::spawn(async move {
                watch_worker(weak, endpoint, role).await;
            });
        }
        Ok(registry)
    }

    /// Readiness follows compatible live pools, independently of temporary admission saturation.
    pub(crate) fn readiness(&self) -> crate::health::ReadinessSnapshot {
        self.readiness_with_clock(clock::check(self.config.max_clock_error_ms).is_ok())
    }

    /// Clock health is sampled once for the entire pool snapshot.
    fn readiness_with_clock(&self, clock_healthy: bool) -> crate::health::ReadinessSnapshot {
        use crate::health::{ModelReadiness, ReadinessSnapshot};
        let now = Instant::now();
        let age = Duration::from_millis(self.config.snapshot_max_age_ms);
        let endpoints: std::collections::HashSet<_> = self
            .config
            .pairs
            .iter()
            .flat_map(|pair| [&pair.prefill, &pair.decode])
            .chain(self.config.encoders.iter().map(|encoder| &encoder.endpoint))
            .collect();
        let total_workers = endpoints.len();
        let Ok(state) = self.state.lock() else {
            return ReadinessSnapshot::default();
        };
        let live = |endpoint: &str| {
            state
                .watched
                .get(endpoint)
                .filter(|(_, observed)| now.saturating_duration_since(*observed) <= age)
                .map(|(capability, _)| capability)
        };
        let healthy_workers = endpoints
            .iter()
            .filter(|endpoint| live(endpoint).is_some())
            .count();
        let ready = clock_healthy
            && self.config.pairs.iter().any(|pair| {
                let Some(p) = live(&pair.prefill).and_then(|cap| worker(&pair.prefill, cap).ok())
                else {
                    return false;
                };
                let Some(d) = live(&pair.decode).and_then(|cap| worker(&pair.decode, cap).ok())
                else {
                    return false;
                };
                p.contract.accepts_decode(&d.contract)
                    && (self.config.encoders.is_empty()
                        || self.config.encoders.iter().any(|encoder| {
                            live(&encoder.endpoint)
                                .and_then(|cap| worker(&encoder.endpoint, cap).ok())
                                .is_some_and(|e| {
                                    p.contract.accepts_encoder(&e.contract)
                                        && state
                                            .encoders
                                            .get(&instance_key(&e.instance))
                                            .is_some_and(|tracker| tracker.is_available(now, age))
                                })
                        }))
            });
        ReadinessSnapshot {
            workers_ready: ready,
            tokenizers_ready: true,
            healthy_workers,
            total_workers,
            models: vec![ModelReadiness {
                model: self.config.model.clone(),
                ready,
                reason: (!ready).then_some("no_compatible_managed_pipeline"),
                healthy_workers,
                total_workers,
            }],
            ..ReadinessSnapshot::default()
        }
    }

    /// Read pool-level observations; requests never open per-encoder load streams.
    pub fn candidates(
        &self,
        request_id: &str,
    ) -> Result<(Vec<PdCandidate>, Vec<proto::Capabilities>, u64), Status> {
        clock::check(self.config.max_clock_error_ms)?;
        let now = Instant::now();
        let age = Duration::from_millis(self.config.snapshot_max_age_ms);
        let mut state = self
            .state
            .lock()
            .map_err(|_| Status::internal("managed registry poisoned"))?;
        state.generation = state.generation.wrapping_add(1);
        let mut encoders = Vec::new();
        for configured in &self.config.encoders {
            let Some((capability, observed)) = state.watched.get(&configured.endpoint).cloned()
            else {
                continue;
            };
            if now.saturating_duration_since(observed) > age {
                continue;
            }
            let instance = capability
                .instance
                .clone()
                .ok_or_else(|| Status::data_loss("E instance missing"))?;
            let key = instance_key(&instance);
            if !state.encoders.contains_key(&key) && state.encoders.len() >= self.config.max_records
            {
                continue;
            }
            let tracker = state
                .encoders
                .entry(key)
                .or_insert_with(|| LoadTracker::new(instance, self.config.max_records));

            let mut cost = EncodeCost::cold();
            // Stages overlap: GPU occupancy must never become serial CPU waiting.
            let load = tracker.stage_work(now, age);
            let (cpu, gpu, send_bytes) = load.unwrap_or((u64::MAX, u64::MAX, u64::MAX));
            cost.cpu_wait_ms = cost.cpu_wait_ms.saturating_add(cpu);
            cost.gpu_available_ms = cost.gpu_available_ms.saturating_add(gpu);
            cost.send_available_ms = cost
                .send_available_ms
                .saturating_add(send_bytes.div_ceil(cost.bytes_per_ms));
            encoders.push(EncoderCandidate {
                worker: worker(&configured.endpoint, &capability)?,
                cost,
                fresh: load.is_some(),
                has_capacity: tracker.has_capacity(now, age),
                stable_tie_break: stable_tie(
                    request_id,
                    &capability
                        .instance
                        .as_ref()
                        .map(instance_key)
                        .unwrap_or_default(),
                ),
            });
        }
        let mut candidates = Vec::new();
        let mut prefills = Vec::new();
        for configured in &self.config.pairs {
            let Some((prefill, p_observed)) = state.watched.get(&configured.prefill).cloned()
            else {
                continue;
            };
            let Some((decode, d_observed)) = state.watched.get(&configured.decode).cloned() else {
                continue;
            };
            if now.saturating_duration_since(p_observed) > age
                || now.saturating_duration_since(d_observed) > age
            {
                continue;
            }
            let p = worker(&configured.prefill, &prefill)?;
            let d = worker(&configured.decode, &decode)?;
            for (instance, capacity) in [
                (&p.instance, prefill.capacity),
                (&d.instance, decode.capacity),
            ] {
                let key = instance_key(instance);
                if !state.capacities.contains_key(&key)
                    && state.capacities.len() >= self.config.max_records
                {
                    return Err(Status::resource_exhausted("managed instance history full"));
                }
                state.capacities.insert(key, capacity);
            }
            let has_capacity = active(&state, &p.instance) < prefill.capacity
                && active(&state, &d.instance) < decode.capacity;
            let prior = PdCost::default();
            candidates.push(PdCandidate {
                prefill: p,
                decode: d,
                encoders: Vec::new(),
                has_capacity,
                layout_ms: prior.layout_ms,
                manifest_delivery_ms: prior.manifest_delivery_ms,
                decode_registration_ms: prior.decode_registration_ms,
                bind_submit_ms: prior.bind_submit_ms,
                native_pd_ms: prior.native_pd_ms,
            });
            prefills.push(prefill);
        }
        let mut ranked: Vec<_> = candidates.into_iter().zip(prefills).collect();
        ranked.sort_by_cached_key(|(pair, _)| {
            (
                !pair.has_capacity,
                active(&state, &pair.prefill.instance)
                    .saturating_add(active(&state, &pair.decode.instance)),
                stable_tie(
                    request_id,
                    &format!(
                        "{}\0{}",
                        instance_key(&pair.prefill.instance),
                        instance_key(&pair.decode.instance)
                    ),
                ),
            )
        });
        ranked.truncate(self.config.max_pd_candidates);
        // Rank E candidates only for shortlisted P/D pairs, never the complete E×P×D product.
        for (pair, _) in &mut ranked {
            pair.encoders.clone_from(&encoders);
            pair.encoders.sort_by_key(|e| {
                (
                    !e.has_capacity,
                    e.cost.media_ready_ms().unwrap_or(u64::MAX),
                    e.stable_tie_break,
                )
            });
            pair.encoders.truncate(self.config.max_encoder_candidates);
        }
        let (candidates, prefills) = ranked.into_iter().unzip();
        Ok((candidates, prefills, state.generation))
    }

    /// One ordered observer publishes a complete capability/load snapshot atomically.
    fn publish(
        &self,
        endpoint: &str,
        capability: proto::Capabilities,
        snapshot: Option<proto::LoadSnapshot>,
    ) -> Result<(), Status> {
        let now = Instant::now();
        let mut state = self
            .state
            .lock()
            .map_err(|_| Status::internal("managed registry poisoned"))?;
        if let Some(snapshot) = snapshot {
            let instance = capability
                .instance
                .clone()
                .ok_or_else(|| Status::data_loss("E instance missing"))?;
            let key = instance_key(&instance);
            if !state.encoders.contains_key(&key) && state.encoders.len() >= self.config.max_records
            {
                return Err(Status::resource_exhausted("managed instance history full"));
            }
            state
                .encoders
                .entry(key)
                .or_insert_with(|| LoadTracker::new(instance, self.config.max_records))
                .observe(snapshot, now)?;
        }
        state
            .watched
            .insert(endpoint.to_string(), (capability, now));
        Ok(())
    }

    /// Reserve all selected operations atomically before any control request can be sent.
    pub fn reserve(
        &self,
        operations: &[(proto::OperationRef, bool)],
        work: u64,
    ) -> Result<(), Status> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Status::internal("managed registry poisoned"))?;
        if state.records.len().saturating_add(operations.len()) > self.config.max_records
            || operations
                .iter()
                .any(|(op, _)| state.records.contains_key(&op.operation_id))
        {
            return Err(Status::resource_exhausted(
                "managed local reservations full/conflicting",
            ));
        }
        for (operation, encoder) in operations {
            if !*encoder {
                let target = operation
                    .target
                    .as_ref()
                    .ok_or_else(|| Status::invalid_argument("missing managed target"))?;
                let capacity = state
                    .capacities
                    .get(&instance_key(target))
                    .copied()
                    .unwrap_or(0);
                if active(&state, target) >= capacity {
                    return Err(Status::resource_exhausted(
                        "P/D capacity changed during planning",
                    ));
                }
            }
        }
        for (operation, encoder) in operations {
            if *encoder {
                let target = operation
                    .target
                    .as_ref()
                    .ok_or_else(|| Status::invalid_argument("missing E target"))?;
                let tracker = state
                    .encoders
                    .get_mut(&instance_key(target))
                    .ok_or_else(|| Status::unavailable("E snapshot disappeared"))?;
                if !tracker.has_capacity(
                    Instant::now(),
                    Duration::from_millis(self.config.snapshot_max_age_ms),
                ) {
                    return Err(Status::resource_exhausted(
                        "E capacity changed during planning",
                    ));
                }
                tracker.reserve(operation.operation_id.clone(), work)?;
            }
        }
        for (operation, encoder) in operations {
            state.records.insert(
                operation.operation_id.clone(),
                Reservation {
                    operation: operation.clone(),
                    encoder: *encoder,
                },
            );
        }
        Ok(())
    }

    /// Capabilities and load must come from the same current instance and a synchronized clock.
    async fn probe(
        &self,
        endpoint: &str,
        role: &str,
    ) -> Result<(proto::Capabilities, Option<proto::LoadSnapshot>), Status> {
        let started = wall_ms()?;
        let query = async {
            let channel = self.channel(endpoint)?;
            let (capability, snapshot) = if role == "encoder" {
                let mut client = ManagedEncoderClient::new(channel);
                let capability = client
                    .get_capabilities(proto::CapabilityQuery {})
                    .await?
                    .into_inner();
                let instance = capability
                    .instance
                    .clone()
                    .ok_or_else(|| Status::data_loss("E lacks instance"))?;
                let snapshot = client
                    .watch_load(instance)
                    .await?
                    .into_inner()
                    .message()
                    .await?
                    .ok_or_else(|| Status::unavailable("E load stream ended"))?;
                (capability, Some(snapshot))
            } else {
                (
                    ManagedSglangClient::new(channel)
                        .get_capabilities(proto::CapabilityQuery {})
                        .await?
                        .into_inner(),
                    None,
                )
            };
            let finished = wall_ms()?;
            if capability.version != VERSION
                || capability.role != role
                || !capability.ready
                || capability.layout_contract != LAYOUT
                || capability.model_revision != self.config.model_revision
                || capability.processor_contract != self.config.processor_contract
                || capability.capacity == 0
                || capability
                    .wall_time_unix_ms
                    .saturating_add(self.config.max_clock_error_ms)
                    < started
                || capability.wall_time_unix_ms
                    > finished.saturating_add(self.config.max_clock_error_ms)
            {
                return Err(Status::failed_precondition(
                    "managed capability/clock mismatch",
                ));
            }
            Ok((capability, snapshot))
        };
        tokio::time::timeout(Duration::from_millis(self.config.probe_timeout_ms), query)
            .await
            .map_err(|_| Status::unavailable("managed capability probe timed out"))?
    }
}

/// Reconnect bounded pool monitors; E load remains a long-lived ordered stream.
async fn watch_worker(registry: std::sync::Weak<Registry>, endpoint: String, role: &'static str) {
    loop {
        let Some(owner) = registry.upgrade() else {
            return;
        };
        let age = Duration::from_millis(owner.config.snapshot_max_age_ms);
        let interval = age / 3;
        let result = owner.probe(&endpoint, role).await;
        let capability = match result {
            Ok((capability, snapshot)) => {
                if let Err(error) = owner.publish(&endpoint, capability.clone(), snapshot) {
                    tracing::warn!(role, %error, "managed pool publication refused; retrying");
                    None
                } else {
                    Some(capability)
                }
            }
            Err(error) => {
                tracing::debug!(role, %error, "managed pool probe unavailable");
                None
            }
        };
        drop(owner);
        if let Some(capability) = capability.clone().filter(|_| role == "encoder") {
            let follow = async {
                let channel = registry
                    .upgrade()
                    .ok_or_else(|| Status::cancelled("managed registry dropped"))?
                    .channel(&endpoint)?;
                let mut client = ManagedEncoderClient::new(channel);
                let mut stream = client
                    .watch_load(
                        capability
                            .instance
                            .clone()
                            .ok_or_else(|| Status::data_loss("E instance missing"))?,
                    )
                    .await?
                    .into_inner();
                loop {
                    let snapshot = tokio::time::timeout(age, stream.message())
                        .await
                        .map_err(|_| Status::unavailable("E load snapshot expired"))??
                        .ok_or_else(|| Status::unavailable("E load stream ended"))?;
                    let Some(owner) = registry.upgrade() else {
                        return Ok::<(), Status>(());
                    };
                    owner.publish(&endpoint, capability.clone(), Some(snapshot))?;
                }
            };
            if let Err(error) = follow.await {
                tracing::debug!(%error, "managed E load stream reconnecting");
            }
        }
        let Some(owner) = registry.upgrade() else {
            return;
        };
        if role == "encoder" || capability.is_none() {
            if let Ok(mut state) = owner.state.lock() {
                state.watched.remove(&endpoint);
            }
        }
        drop(owner);
        tokio::time::sleep(interval.max(Duration::from_millis(50))).await;
    }
}

impl AttemptObserver for Registry {
    fn channel(&self, endpoint: &str) -> Result<Channel, Status> {
        self.channels
            .get(endpoint)
            .cloned()
            .ok_or_else(|| Status::failed_precondition("endpoint is outside managed deployment"))
    }

    /// Reconcile ordered acknowledgements against snapshot coverage without releasing unknown work.
    fn event(&self, event: &proto::ControlEvent) -> Result<(), Status> {
        let operation = event
            .operation
            .as_ref()
            .ok_or_else(|| Status::data_loss("event lacks operation"))?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| Status::internal("managed registry poisoned"))?;
        let Some(reservation) = state.records.get(&operation.operation_id) else {
            return Ok(());
        };
        if &reservation.operation != operation {
            return Err(Status::data_loss("local operation fence mismatch"));
        }
        if reservation.encoder {
            let target = operation
                .target
                .as_ref()
                .ok_or_else(|| Status::data_loss("event lacks target"))?;
            if let Some(tracker) = state.encoders.get_mut(&instance_key(target)) {
                if matches!(event.event, Some(proto::control_event::Event::Stage(stage)) if stage == proto::Stage::Accepted as i32)
                {
                    tracker.accepted(&operation.operation_id, target, event.admission_seq)?;
                }
                if matches!(event.event, Some(proto::control_event::Event::Stage(stage)) if stage == proto::Stage::Quiesced as i32)
                {
                    tracker.rejected_or_quiesced(&operation.operation_id);
                    state.records.remove(&operation.operation_id);
                }
            }
        }
        Ok(())
    }

    /// Only a matching confirmed safe terminal receipt removes an outstanding reservation.
    fn termination(&self, operation: &proto::OperationRef, quiesced: bool) {
        if !quiesced {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state
            .records
            .get(&operation.operation_id)
            .is_some_and(|record| record.operation == *operation)
        {
            state.records.remove(&operation.operation_id);
            if let Some(tracker) = operation
                .target
                .as_ref()
                .and_then(|target| state.encoders.get_mut(&instance_key(target)))
            {
                tracker.rejected_or_quiesced(&operation.operation_id);
            }
        }
    }
}

/// Instance generations are a separate namespace for occupancy and snapshot coverage.
fn instance_key(instance: &proto::Instance) -> String {
    format!("{}\0{}", instance.worker_id, instance.generation)
}

/// Unknown operations remain in this count until a real quiescence receipt arrives.
fn active(state: &State, instance: &proto::Instance) -> u64 {
    state
        .records
        .values()
        .filter(|record| record.operation.target.as_ref() == Some(instance))
        .count() as u64
}

/// Convert the advertised complete contract into the planner's equality key.
fn worker(endpoint: &str, capability: &proto::Capabilities) -> Result<Worker, Status> {
    Ok(Worker {
        instance: capability
            .instance
            .clone()
            .ok_or_else(|| Status::data_loss("managed instance missing"))?,
        control_endpoint: endpoint.to_owned(),
        receive_endpoint: capability.receive_endpoint.clone(),
        contract: Contract {
            model_revision: capability.model_revision.clone(),
            processor: capability.processor_contract.clone(),
            layout: capability.layout_contract.clone(),
            embedding_dtype: capability.embedding_dtype.clone(),
            kv: capability.kv_backend.clone(),
            transfer: capability.embedding_backend.clone(),
            topology: capability.topology_contract.clone(),
        },
    })
}

/// Wall timestamps are for wire expiry only; execution retains a separate monotonic deadline.
pub fn wall_ms() -> Result<u64, Status> {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Status::unavailable("invalid wall clock"))?
            .as_millis(),
    )
    .map_err(|_| Status::unavailable("wall timestamp overflow"))
}

/// Stable request/worker hashing is a tie breaker, never a latency estimate.
fn stable_tie(request: &str, worker: &str) -> u64 {
    let digest = Sha256::digest(format!("{request}\0{worker}"));
    let mut bytes = [0; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(bytes)
}

struct Pending {
    work: u64,
    admission_seq: Option<u64>,
}

pub struct LoadTracker {
    instance: proto::Instance,
    snapshot: Option<(proto::LoadSnapshot, Instant)>,
    pending: HashMap<String, Pending>,
    max_pending: usize,
}

impl LoadTracker {
    /// A restarted worker gets a new tracker; old unknown work belongs to its old instance.
    pub fn new(instance: proto::Instance, max_pending: usize) -> Self {
        Self {
            instance,
            snapshot: None,
            pending: HashMap::new(),
            max_pending,
        }
    }

    /// Register in-flight work before sending Encode, including unknown acceptance outcomes.
    pub fn reserve(&mut self, operation_id: String, work: u64) -> Result<(), Status> {
        if self.pending.contains_key(&operation_id) {
            return Err(Status::already_exists("encode reservation exists"));
        }
        if self.pending.len() >= self.max_pending {
            return Err(Status::resource_exhausted("encode reservations full"));
        }
        self.pending.insert(
            operation_id,
            Pending {
                work,
                admission_seq: None,
            },
        );
        Ok(())
    }

    /// Accepted/GetOperation reconcile with an already-arrived covering snapshot too.
    pub fn accepted(
        &mut self,
        operation_id: &str,
        instance: &proto::Instance,
        seq: u64,
    ) -> Result<(), Status> {
        if instance != &self.instance || seq == 0 {
            return Err(Status::failed_precondition(
                "stale admission acknowledgement",
            ));
        }
        if self
            .snapshot
            .as_ref()
            .is_some_and(|(s, _)| seq <= s.covered_admission_seq)
        {
            self.pending.remove(operation_id);
            return Ok(());
        }
        let pending = self
            .pending
            .get_mut(operation_id)
            .ok_or_else(|| Status::not_found("unknown encode reservation"))?;
        if pending.admission_seq.is_some_and(|old| old != seq) {
            return Err(Status::failed_precondition("admission sequence changed"));
        }
        pending.admission_seq = Some(seq);
        Ok(())
    }

    /// Snapshot order and admission coverage cannot regress within an instance.
    pub fn observe(&mut self, snapshot: proto::LoadSnapshot, now: Instant) -> Result<bool, Status> {
        if snapshot.instance.as_ref() != Some(&self.instance) {
            return Err(Status::failed_precondition("stale load instance"));
        }
        if let Some((old, _)) = &self.snapshot {
            if snapshot.snapshot_seq <= old.snapshot_seq {
                return Ok(false);
            }
            if snapshot.covered_admission_seq < old.covered_admission_seq {
                return Err(Status::failed_precondition("load coverage regressed"));
            }
        }
        self.pending.retain(|_, p| {
            p.admission_seq
                .is_none_or(|seq| seq > snapshot.covered_admission_seq)
        });
        self.snapshot = Some((snapshot, now));
        Ok(true)
    }

    /// Unknown acceptance stays charged; cancellation alone must never call this.
    pub fn rejected_or_quiesced(&mut self, operation_id: &str) {
        self.pending.remove(operation_id);
    }

    /// Keep CPU/GPU availability separate; unknown admissions conservatively charge both.
    /// No fresh snapshot means no normal candidate, including a just-restarted instance.
    pub fn stage_work(&self, now: Instant, max_age: Duration) -> Option<(u64, u64, u64)> {
        let (snapshot, observed) = self.snapshot.as_ref()?;
        if now.saturating_duration_since(*observed) > max_age {
            return None;
        }
        let pending = self
            .pending
            .values()
            .try_fold(0u64, |total, p| total.checked_add(p.work))?;
        Some((
            snapshot.cpu_work.checked_add(pending)?,
            snapshot.gpu_work.checked_add(pending)?,
            snapshot.send_bytes,
        ))
    }

    /// A draining or clock-unhealthy encoder publishes zero admission capacity.
    pub fn is_available(&self, now: Instant, max_age: Duration) -> bool {
        self.snapshot.as_ref().is_some_and(|(snapshot, observed)| {
            snapshot.capacity > 0 && now.saturating_duration_since(*observed) <= max_age
        })
    }

    /// Count only local operations not covered by the same instance's snapshot.
    pub fn has_capacity(&self, now: Instant, max_age: Duration) -> bool {
        self.snapshot.as_ref().is_some_and(|(snapshot, observed)| {
            now.saturating_duration_since(*observed) <= max_age
                && snapshot.active.saturating_add(self.pending.len() as u64) < snapshot.capacity
        })
    }
}

#[cfg(test)]
mod readiness_tests {
    use super::*;

    /// Saturation is an admission outcome; stale or incompatible workers remove readiness.
    #[test]
    fn readiness_requires_fresh_compatible_pools_but_not_free_slots() {
        let config: Config = serde_json::from_value(serde_json::json!({
            "model": "m", "model_path": "/model", "model_revision": "m", "processor_contract": "p",
            "prefill_urls": ["http://p:1"], "decode_urls": ["http://d:1"],
            "encoders": [{"endpoint": "http://e:1"}]
        }))
        .unwrap();
        let config = config.resolve().unwrap();
        let registry = Registry {
            config,
            state: Mutex::new(State::default()),
            channels: HashMap::new(),
        };
        assert!(!registry.readiness_with_clock(true).workers_ready);
        let now = Instant::now();
        {
            let mut state = registry.state.lock().unwrap();
            for (endpoint, role) in [
                ("http://e:1", "encoder"),
                ("http://p:1", "prefill"),
                ("http://d:1", "decode"),
            ] {
                let instance = proto::Instance {
                    worker_id: role.into(),
                    generation: "g".into(),
                };
                state.watched.insert(
                    endpoint.into(),
                    (
                        proto::Capabilities {
                            instance: Some(instance.clone()),
                            ready: true,
                            capacity: 1,
                            model_revision: "m".into(),
                            processor_contract: "p".into(),
                            layout_contract: LAYOUT.into(),
                            ..Default::default()
                        },
                        now,
                    ),
                );
                if role == "encoder" {
                    let mut load = LoadTracker::new(instance.clone(), 3);
                    load.observe(
                        proto::LoadSnapshot {
                            instance: Some(instance.clone()),
                            snapshot_seq: 1,
                            capacity: 1,
                            active: 1,
                            ..Default::default()
                        },
                        now,
                    )
                    .unwrap();
                    state.encoders.insert(instance_key(&instance), load);
                }
            }
        }
        assert!(registry.readiness_with_clock(true).workers_ready);
        assert_eq!(registry.readiness_with_clock(true).healthy_workers, 3);
        assert!(!registry.readiness_with_clock(false).workers_ready);
        registry
            .state
            .lock()
            .unwrap()
            .watched
            .get_mut("http://d:1")
            .unwrap()
            .1 = now - Duration::from_millis(registry.config.snapshot_max_age_ms + 1);
        assert!(!registry.readiness_with_clock(true).workers_ready);
        let mut state = registry.state.lock().unwrap();
        let (decode, observed) = state.watched.get_mut("http://d:1").unwrap();
        *observed = now;
        decode.model_revision = "different".into();
        drop(state);
        assert!(!registry.readiness_with_clock(true).workers_ready);
    }
}
