//! Bounded, read-only candidate scoring followed by one immutable plan.

#[cfg(test)]
use std::sync::atomic::Ordering;
use std::time::Duration;

use smg_grpc_client::sglang_managed::{proto, validate_binding, validate_external_plan};
use tonic::Status;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contract {
    pub model_revision: String,
    pub processor: String,
    pub layout: String,
    pub embedding_dtype: String,
    pub kv: String,
    pub transfer: String,
    pub topology: String,
}

impl Contract {
    /// Match the complete prepared-input representation independently of parallelism.
    fn same_input(&self, peer: &Self) -> bool {
        self.model_revision == peer.model_revision
            && self.processor == peer.processor
            && self.layout == peer.layout
            && self.embedding_dtype == peer.embedding_dtype
    }

    /// A single native TP group excludes attention-DP, pipeline and unified-memory state.
    fn single_tp_group(&self) -> bool {
        matches!(
            self.topology.as_str(),
            "single-host-tp1" | "single-host-tp2" | "single-host-tp4" | "single-host-tp8"
        )
    }

    /// Native E sends complete embeddings independently of P's KV sharding.
    pub fn accepts_encoder(&self, encoder: &Self) -> bool {
        self.same_input(encoder)
            && self.transfer == encoder.transfer
            && (self.topology == encoder.topology
                || (self.single_tp_group() && encoder.single_tp_group()))
    }

    /// Mooncake reslices the supported Qwen3-VL KV layout across native TP groups.
    pub fn accepts_decode(&self, decode: &Self) -> bool {
        self.same_input(decode)
            && self.kv == decode.kv
            && (self.topology == decode.topology
                || (self.kv == "mooncake" && self.single_tp_group() && decode.single_tp_group()))
    }
}

#[derive(Debug, Clone)]
pub struct Worker {
    pub instance: proto::Instance,
    pub control_endpoint: String,
    pub receive_endpoint: String,
    pub contract: Contract,
}

/// Milliseconds from the same planning origin; cache-hit prediction is outside v1.
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeCost {
    pub cpu_wait_ms: u64,
    pub preprocess_ms: u64,
    pub gpu_available_ms: u64,
    pub encode_ms: u64,
    pub receive_ready_ms: u64,
    pub send_available_ms: u64,
    pub transfer_startup_ms: u64,
    pub embedding_bytes: u64,
    pub bytes_per_ms: u64,
    pub error_ms: u64,
}

impl EncodeCost {
    /// Fixed Qwen3-VL stage priors; deadlines remain independently enforced at every owner.
    pub fn cold() -> Self {
        Self {
            // Warm H200 mixed-image measurements stayed below these stage estimates.
            // Multi-second per-request priors falsely exhausted the deadline in a full queue.
            preprocess_ms: 100,
            encode_ms: 250,
            receive_ready_ms: 100,
            transfer_startup_ms: 100,
            embedding_bytes: 32 * 1024 * 1024,
            bytes_per_ms: 100_000,
            error_ms: 1000,
            ..Self::default()
        }
    }

    /// CPU/GPU/receiver/send readiness overlap; transfer bytes are not a request count.
    pub fn media_ready_ms(self) -> Option<u64> {
        if self.bytes_per_ms == 0 {
            return None;
        }
        let cpu = self.cpu_wait_ms.checked_add(self.preprocess_ms)?;
        let encoded = cpu.max(self.gpu_available_ms).checked_add(self.encode_ms)?;
        encoded
            .max(self.receive_ready_ms)
            .max(self.send_available_ms)
            .checked_add(self.transfer_startup_ms)?
            .checked_add(self.embedding_bytes.div_ceil(self.bytes_per_ms))
    }
}

#[derive(Debug, Clone)]
pub struct EncoderCandidate {
    pub worker: Worker,
    pub cost: EncodeCost,
    pub fresh: bool,
    pub has_capacity: bool,
    pub stable_tie_break: u64,
}

#[derive(Debug, Clone)]
pub struct PdCandidate {
    pub prefill: Worker,
    pub decode: Worker,
    pub encoders: Vec<EncoderCandidate>,
    pub has_capacity: bool,
    pub layout_ms: u64,
    pub manifest_delivery_ms: u64,
    pub decode_registration_ms: u64,
    pub bind_submit_ms: u64,
    /// Measured native critical path after complete input; not a new execution barrier.
    pub native_pd_ms: u64,
}

/// Private fields prevent rebinding a submitted attempt; runtime ownership is elsewhere.
#[derive(Debug)]
pub struct Plan {
    binding: proto::InputBinding,
    encoder: Option<Worker>,
    prefill: Worker,
    decode: Worker,
    external: proto::ExternalEncodePlan,
    predicted_ttft_ms: u64,
    registry_generation: u64,
}

impl Plan {
    /// Binding includes both instance generations and the attempt's unique PD room.
    pub fn binding(&self) -> &proto::InputBinding {
        &self.binding
    }
    /// Text plans have no encoder; all image occurrences share the one returned worker.
    pub fn encoder(&self) -> Option<&Worker> {
        self.encoder.as_ref()
    }
    /// The chosen prefill cannot be replaced after dispatch.
    pub fn prefill(&self) -> &Worker {
        &self.prefill
    }
    /// The chosen decode cannot be replaced after dispatch.
    pub fn decode(&self) -> &Worker {
        &self.decode
    }
    /// The receiver's assignment and encoder's part are the same frozen value.
    pub fn external(&self) -> &proto::ExternalEncodePlan {
        &self.external
    }
    /// Prediction is diagnostic, not an observation of GPU time.
    pub fn predicted_ttft_ms(&self) -> u64 {
        self.predicted_ttft_ms
    }
    /// Every candidate came from this registry snapshot.
    pub fn registry_generation(&self) -> u64 {
        self.registry_generation
    }
}

pub struct Planner {
    max_pd: usize,
    max_e: usize,
}

impl Default for Planner {
    fn default() -> Self {
        Self {
            max_pd: 4,
            max_e: 4,
        }
    }
}

impl Planner {
    /// Deployment bounds cap evaluation without changing any legacy router policy.
    pub fn new(max_pd: usize, max_e: usize) -> Result<Self, Status> {
        if !(1..=64).contains(&max_pd) || !(1..=64).contains(&max_e) {
            return Err(Status::invalid_argument(
                "managed candidate bounds must be in 1..=64",
            ));
        }
        Ok(Self { max_pd, max_e })
    }

    /// Candidate vectors are pre-ranked by a read-only topology/load index, not legacy policy.
    pub fn choose(
        &self,
        pairs: &[PdCandidate],
        images: usize,
        remaining: Duration,
    ) -> Result<(usize, Option<usize>, u64), Status> {
        let mut best: Option<(usize, Option<usize>, u64, u64, u64)> = None;
        for (p, pair) in pairs.iter().take(self.max_pd).enumerate() {
            if !pair.has_capacity || !pair.prefill.contract.accepts_decode(&pair.decode.contract) {
                continue;
            }
            let evaluate = |media_ms: u64| -> Option<u64> {
                media_ms
                    .checked_add(pair.layout_ms)?
                    .checked_add(pair.manifest_delivery_ms)?
                    .max(pair.decode_registration_ms)
                    .checked_add(pair.bind_submit_ms)?
                    .checked_add(pair.native_pd_ms)
            };
            let candidates: Vec<_> = if images == 0 {
                vec![(None, 0, 0, 0)]
            } else {
                pair.encoders
                    .iter()
                    .take(self.max_e)
                    .enumerate()
                    .filter_map(|(e, candidate)| {
                        if !candidate.fresh
                            || !candidate.has_capacity
                            || !pair
                                .prefill
                                .contract
                                .accepts_encoder(&candidate.worker.contract)
                        {
                            return None;
                        }
                        Some((
                            Some(e),
                            candidate.cost.media_ready_ms()?,
                            candidate.cost.error_ms,
                            candidate.stable_tie_break,
                        ))
                    })
                    .collect()
            };
            for (e, media, error, tie) in candidates {
                let Some(score) = evaluate(media) else {
                    continue;
                };
                if u128::from(score.saturating_add(error)) >= remaining.as_millis() {
                    continue;
                }
                let better = best.is_none_or(|(_, _, old, old_error, old_tie)| {
                    // The ignored real-worker benchmark changes only the final ranking;
                    // compatibility, capacity and deadline eligibility remain identical.
                    #[cfg(test)]
                    if super::tests::BENCHMARK_HASH.load(Ordering::Relaxed) {
                        return tie < old_tie;
                    }
                    let band = error.max(old_error);
                    if score.abs_diff(old) <= band {
                        tie < old_tie
                    } else {
                        score < old
                    }
                });
                if better {
                    best = Some((p, e, score, error, tie));
                }
            }
        }
        best.map(|(p, e, score, ..)| (p, e, score)).ok_or_else(|| {
            Status::unavailable("no compatible managed EPD plan within deadline/capacity")
        })
    }

    /// Freeze a pure selection; the executor must reserve its budget before any dispatch.
    pub fn freeze(
        &self,
        pair: &PdCandidate,
        encoder_index: Option<usize>,
        images: usize,
        binding: proto::InputBinding,
        predicted_ttft_ms: u64,
        registry_generation: u64,
    ) -> Result<Plan, Status> {
        validate_binding(&binding)?;
        if !pair.has_capacity
            || !pair.prefill.contract.accepts_decode(&pair.decode.contract)
            || encoder_index.is_some_and(|index| {
                index >= self.max_e
                    || pair.encoders.get(index).is_none_or(|candidate| {
                        !candidate.fresh
                            || !candidate.has_capacity
                            || !pair
                                .prefill
                                .contract
                                .accepts_encoder(&candidate.worker.contract)
                    })
            })
        {
            return Err(Status::failed_precondition(
                "managed selection lost capacity or compatibility",
            ));
        }
        let encoder = match encoder_index {
            Some(index) => Some(
                pair.encoders
                    .get(index)
                    .ok_or_else(|| Status::invalid_argument("invalid encoder index"))?
                    .worker
                    .clone(),
            ),
            None => None,
        };
        if binding.prefill.as_ref() != Some(&pair.prefill.instance)
            || binding.decode.as_ref().and_then(|op| op.target.as_ref())
                != Some(&pair.decode.instance)
            || (images == 0) != encoder.is_none()
            || binding.model_revision != pair.prefill.contract.model_revision
            || binding.processor_contract != pair.prefill.contract.processor
            || binding.layout_contract != pair.prefill.contract.layout
        {
            return Err(Status::invalid_argument(
                "managed plan does not match selection",
            ));
        }
        let mut external = proto::ExternalEncodePlan {
            receiver_rid: binding.prefill_rid.clone(),
            ..Default::default()
        };
        if let Some(worker) = &encoder {
            external
                .encoder_receive_endpoints
                .push(worker.receive_endpoint.clone());
            external.image_counts.push(
                u32::try_from(images).map_err(|_| Status::invalid_argument("too many images"))?,
            );
            external.parts.push(proto::EncodePart {
                native_part_rid: format!("{}_local_part_0", binding.prefill_rid),
                part_idx: 0,
                ordered_item_ids: (0..images as u32).collect(),
            });
        }
        validate_external_plan(&external, images)?;
        Ok(Plan {
            binding,
            encoder,
            prefill: pair.prefill.clone(),
            decode: pair.decode.clone(),
            external,
            predicted_ttft_ms,
            registry_generation,
        })
    }
}

/// Conservative native PD critical-path costs, shared across candidates until calibration exists.
#[derive(Clone, Copy)]
pub struct PdCost {
    pub layout_ms: u64,
    pub manifest_delivery_ms: u64,
    pub decode_registration_ms: u64,
    pub bind_submit_ms: u64,
    pub native_pd_ms: u64,
}
impl Default for PdCost {
    fn default() -> Self {
        Self {
            layout_ms: 100,
            manifest_delivery_ms: 100,
            decode_registration_ms: 100,
            bind_submit_ms: 100,
            native_pd_ms: 5000,
        }
    }
}
