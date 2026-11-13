//! Bounded managed worker discovery and local reservations, independent of legacy policies.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::Deserialize;
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
    load_tracker::LoadTracker,
    planner::{Contract, EncodeCost, EncoderCandidate, PdCandidate, Worker},
};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderEndpoint {
    pub endpoint: String,
    /// Conservative cold/unknown-media prior, calibrated for the deployment's model and links.
    pub prior: EncodeCost,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PdEndpoints {
    pub prefill: String,
    pub decode: String,
    pub prior: PdCost,
}

/// Conservative measured deployment priors, in milliseconds from dispatch.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PdCost {
    pub layout_ms: u64,
    pub manifest_delivery_ms: u64,
    pub decode_registration_ms: u64,
    pub bind_submit_ms: u64,
    pub native_pd_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model: String,
    pub model_path: String,
    pub model_revision: String,
    pub processor_contract: String,
    pub pairs: Vec<PdEndpoints>,
    pub encoders: Vec<EncoderEndpoint>,
    pub max_attempts: usize,
    pub max_records: usize,
    pub timeout_ms: u64,
    pub max_clock_error_ms: u64,
    pub probe_timeout_ms: u64,
    pub snapshot_max_age_ms: u64,
    pub max_images: usize,
    #[serde(default = "default_candidates")]
    pub max_pd_candidates: usize,
    #[serde(default = "default_candidates")]
    pub max_encoder_candidates: usize,
}

/// Bound the default search independently of pool size.
fn default_candidates() -> usize {
    4
}

impl Config {
    /// Invalid bounds fail startup; an absent configuration keeps this entire mode disabled.
    pub fn validate(&self) -> Result<(), Status> {
        if self.model.is_empty()
            || self.model_path.is_empty()
            || self.model_revision.is_empty()
            || self.processor_contract.is_empty()
            || self.pairs.is_empty()
            || self.pairs.len() > 64
            || self.encoders.len() > 64
            || !(1..=64).contains(&self.max_pd_candidates)
            || !(1..=64).contains(&self.max_encoder_candidates)
            || self.max_attempts == 0
            || self.max_attempts > 1024
            || self.max_records < self.max_attempts.saturating_mul(3)
            || self.max_records > 65536
            || self.timeout_ms == 0
            || self.timeout_ms > 120_000
            || self.probe_timeout_ms == 0
            || self.max_clock_error_ms == 0
            || self.snapshot_max_age_ms == 0
            || self.max_images == 0
            || self.max_images > 32
            || self.encoders.iter().any(|encoder| {
                encoder.prior.media_ready_ms().is_none()
                    || encoder.prior.verified_cache_saving_ms != 0
            })
        {
            return Err(Status::invalid_argument(
                "invalid managed gateway deployment bounds",
            ));
        }
        Ok(())
    }
}

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
}

impl Registry {
    /// This registry never mutates worker objects or any legacy policy during candidate scoring.
    pub fn new(config: Config) -> Arc<Self> {
        let registry = Arc::new(Self {
            config,
            state: Mutex::new(State::default()),
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
        registry
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

            let mut cost = configured.prior;
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
        encoders.sort_by_key(|e| {
            (
                !e.has_capacity,
                e.cost.media_ready_ms().unwrap_or(u64::MAX),
                e.stable_tie_break,
            )
        });
        encoders.truncate(self.config.max_encoder_candidates);
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
            candidates.push(PdCandidate {
                prefill: p,
                decode: d,
                encoders: encoders.clone(),
                has_capacity,
                layout_ms: configured.prior.layout_ms,
                manifest_delivery_ms: configured.prior.manifest_delivery_ms,
                decode_registration_ms: configured.prior.decode_registration_ms,
                bind_submit_ms: configured.prior.bind_submit_ms,
                native_pd_ms: configured.prior.native_pd_ms,
            });
            prefills.push(prefill);
        }
        let mut ranked: Vec<_> = candidates.into_iter().zip(prefills).collect();
        ranked.sort_by_key(|(pair, _)| {
            (
                !pair.has_capacity,
                active(&state, &pair.prefill.instance)
                    .saturating_add(active(&state, &pair.decode.instance)),
                stable_tie(request_id, &instance_key(&pair.prefill.instance)),
            )
        });
        ranked.truncate(self.config.max_pd_candidates);
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
            let channel = Channel::from_shared(endpoint.to_string())
                .map_err(|e| Status::invalid_argument(e.to_string()))?
                .connect()
                .await
                .map_err(|e| Status::unavailable(e.to_string()))?;
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
                if owner
                    .publish(&endpoint, capability.clone(), snapshot)
                    .is_err()
                {
                    return;
                }
                Some(capability)
            }
            Err(error) => {
                tracing::debug!(role, %error, "managed pool probe unavailable");
                None
            }
        };
        drop(owner);
        if let Some(capability) = capability.clone().filter(|_| role == "encoder") {
            let follow = async {
                let channel = Channel::from_shared(endpoint.clone())
                    .map_err(|e| Status::invalid_argument(e.to_string()))?
                    .connect()
                    .await
                    .map_err(|e| Status::unavailable(e.to_string()))?;
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
