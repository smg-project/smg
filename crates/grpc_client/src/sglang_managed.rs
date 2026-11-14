//! Strict, versioned managed EPD control contract; independent of MediaIdentity v1.

use prost::Message;
use sha2::{Digest, Sha256};
use tonic::Status;

pub mod budget {
    //! Original-expiry admission and a local monotonic budget.

    use std::time::{Duration, Instant};

    use tonic::Status;

    use super::proto;

    #[derive(Debug, Clone, Copy)]
    pub struct Deadline {
        expiry_unix_ms: u64,
        local: Instant,
    }

    impl Deadline {
        /// Remote timestamps never lengthen the caller's remaining budget.
        pub fn admit(
            budget: &proto::Budget,
            wall_now_ms: u64,
            monotonic_now: Instant,
            clock_offset_ms: i64,
            max_clock_skew_ms: u64,
            max_duration_ms: u64,
        ) -> Result<Self, Status> {
            if clock_offset_ms.unsigned_abs() > max_clock_skew_ms {
                return Err(Status::unavailable(
                    "managed worker clock is outside tolerance",
                ));
            }
            let remaining = budget
                .original_expiry_unix_ms
                .saturating_sub(wall_now_ms)
                .min(budget.remaining_timeout_ms);
            if remaining == 0 {
                return Err(Status::deadline_exceeded("managed operation expired"));
            }
            if remaining > max_duration_ms
                || budget.original_expiry_unix_ms.saturating_sub(wall_now_ms) > max_duration_ms
            {
                return Err(Status::invalid_argument(
                    "managed deadline exceeds retention bound",
                ));
            }
            let local = monotonic_now
                .checked_add(Duration::from_millis(remaining))
                .ok_or_else(|| Status::invalid_argument("managed deadline overflow"))?;
            Ok(Self {
                expiry_unix_ms: budget.original_expiry_unix_ms,
                local,
            })
        }

        /// Bind, query and reconnect use this same deadline, never a fresh timeout.
        pub fn remaining(self, now: Instant) -> Result<Duration, Status> {
            self.local
                .checked_duration_since(now)
                .filter(|d| !d.is_zero())
                .ok_or_else(|| Status::deadline_exceeded("managed operation expired"))
        }

        /// Tombstones remain until the latest legitimate start could arrive.
        pub fn expiry_unix_ms(self) -> u64 {
            self.expiry_unix_ms
        }
    }
}

pub mod clock {
    //! Read-only kernel synchronization health for absolute managed admission deadlines.

    use tonic::Status;

    /// Fail closed when the OS cannot bound wall-clock error; this never adjusts the clock.
    #[cfg(target_os = "linux")]
    #[expect(
        unsafe_code,
        reason = "read-only adjtimex query with modes=0 and owned initialized storage"
    )]
    pub fn check(max_error_ms: u64) -> Result<(), Status> {
        // SAFETY: all-zero timex has modes=0 (query only). The syscall receives writable
        // storage of the platform libc ABI size; no pointer or storage escapes the call.
        let (state, sample) = unsafe {
            let mut sample: libc::timex = std::mem::zeroed();
            let state = libc::adjtimex(&raw mut sample);
            (state, sample)
        };
        if max_error_ms == 0
            || state < 0
            || state == libc::TIME_ERROR
            || sample.status & libc::STA_UNSYNC != 0
            || sample.maxerror < 0
            || sample.maxerror as u64 > max_error_ms.saturating_mul(1000)
        {
            return Err(Status::unavailable(
                "managed clock synchronization outside configured tolerance",
            ));
        }
        Ok(())
    }

    /// Managed absolute deadlines require an implemented synchronization health source.
    #[cfg(not(target_os = "linux"))]
    pub fn check(_max_error_ms: u64) -> Result<(), Status> {
        Err(Status::unimplemented(
            "managed clock health currently requires Linux",
        ))
    }
}

#[expect(
    clippy::allow_attributes,
    reason = "generated tonic code controls these lint attributes"
)]
pub mod proto {
    #![allow(
        clippy::all,
        clippy::absolute_paths,
        clippy::trivially_copy_pass_by_ref,
        unused_qualifications
    )]
    tonic::include_proto!("smg.sglang.managed.v1");
}

pub const VERSION: u32 = 1;
pub const LAYOUT: &str = "sglang_native_manifest_v1";

/// Limits apply before allocation/submission and never enable new modalities.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub manifest_bytes: usize,
    pub tokens: usize,
    pub images: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            manifest_bytes: 4 * 1024 * 1024,
            tokens: 131_072,
            images: 32,
        }
    }
}

/// Check a complete process fence, rather than a stable worker name alone.
pub fn validate_instance(instance: &proto::Instance) -> Result<(), Status> {
    if instance.worker_id.is_empty()
        || instance.generation.is_empty()
        || instance.worker_id.len() > 256
        || instance.generation.len() > 256
    {
        return Err(Status::invalid_argument("invalid managed worker instance"));
    }
    Ok(())
}

/// Validate operation identity before using it as a registry key.
pub fn validate_operation(op: &proto::OperationRef) -> Result<(), Status> {
    let identity = op
        .identity
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("missing identity"))?;
    if identity.logical_request_id.is_empty()
        || identity.attempt_id.is_empty()
        || identity.branch_id != 0
        || op.operation_id.is_empty()
        || identity.logical_request_id.len() > 256
        || identity.attempt_id.len() > 256
        || op.operation_id.len() > 256
        || op.request_digest.len() != 32
        || op.plan_digest.len() != 32
    {
        return Err(Status::invalid_argument(
            "invalid managed operation identity/digest",
        ));
    }
    validate_instance(
        op.target
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing target"))?,
    )
}

/// Require all source, destination, PD and model contracts, even for text input.
pub fn validate_binding(binding: &proto::InputBinding) -> Result<(), Status> {
    validate_operation(
        binding
            .decode
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing decode operation"))?,
    )?;
    validate_instance(
        binding
            .prefill
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing prefill instance"))?,
    )?;
    let bootstrap = binding
        .bootstrap
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("missing bootstrap"))?;
    if binding.prefill_rid.is_empty()
        || binding.prefill_rid.len() > 256
        || binding.layout_contract != LAYOUT
        || binding.processor_contract.is_empty()
        || binding.model_revision.is_empty()
        || bootstrap.host.is_empty()
        || bootstrap.port == 0
        || bootstrap.port > u32::from(u16::MAX)
        || bootstrap.room < 0
        || binding.media_routing_labels.len() > 32
        || binding
            .media_routing_labels
            .iter()
            .any(|label| label.len() > 256)
    {
        return Err(Status::invalid_argument("incomplete managed input binding"));
    }
    Ok(())
}

/// Hash deterministic protobuf fields; no maps or self-referential digest.
pub fn manifest_digest(manifest: &proto::PreparedDecodeInput) -> Vec<u8> {
    let mut unsigned = manifest.clone();
    unsigned.integrity_digest.clear();
    Sha256::digest(unsigned.encode_to_vec()).to_vec()
}

/// Refuse malformed/foreign layouts before D constructs any native request.
/// This initial typed manifest supports image layouts with explicit M-RoPE.
pub fn validate_manifest(
    manifest: &proto::PreparedDecodeInput,
    expected: &proto::InputBinding,
    limits: Limits,
) -> Result<(), Status> {
    validate_binding(expected)?;
    if manifest.encoded_len() > limits.manifest_bytes {
        return Err(Status::resource_exhausted(
            "managed manifest exceeds byte limit",
        ));
    }
    if manifest.version != VERSION
        || manifest.binding.as_ref() != Some(expected)
        || manifest.integrity_digest.len() != 32
        || manifest.integrity_digest != manifest_digest(manifest)
    {
        return Err(Status::invalid_argument(
            "managed manifest binding/digest mismatch",
        ));
    }
    let count = manifest.unpadded_ids.len();
    if count == 0
        || count > limits.tokens
        || manifest.padded_ids.len() != count
        || manifest.images.is_empty()
        || manifest.images.len() > limits.images
        || manifest.image_token_id == 0
    {
        return Err(Status::invalid_argument("incomplete managed image layout"));
    }
    let mut last_end = None;
    for (occurrence, item) in manifest.images.iter().enumerate() {
        if item.occurrence as usize != occurrence
            || item.full_identity.len() != 32
            || item.processor_fingerprint.as_ref().is_some_and(|value| value.is_empty() || value.len() > 1024)
            || item.pad_values.is_empty()
            // Native EPD may consume the grid into final M-RoPE and retain no grid.
            || !matches!(item.grid_thw.len(), 0 | 3)
            || item.grid_thw.contains(&0)
            || item.offsets.is_empty()
            || !(1..=3).contains(&item.native_format)
        {
            return Err(Status::invalid_argument(
                "incomplete managed per-image identity/layout",
            ));
        }
        let hash_bytes: [u8; 8] = item.full_identity[..8]
            .try_into()
            .map_err(|_| Status::invalid_argument("invalid native identity"))?;
        if item.native_hash != u64::from_be_bytes(hash_bytes)
            || item
                .routing_label
                .as_ref()
                .is_some_and(|label| label.len() > 256)
            || (!expected.media_routing_labels.is_empty()
                && item.routing_label.as_ref() != expected.media_routing_labels.get(occurrence))
        {
            return Err(Status::invalid_argument(
                "native hash/routing identity mismatch",
            ));
        }
        let mut hasher = Sha256::new();
        hasher.update(b"sglang-mm-wide-pad-v1");
        hasher.update(&item.full_identity);
        let digest = hasher.finalize();
        let expected_pads: Vec<i64> = digest
            .as_slice()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| 1_000_000 + (u64::from_be_bytes(*chunk) & ((1 << 62) - 1)) as i64)
            .collect();
        if item.pad_values != expected_pads {
            return Err(Status::invalid_argument(
                "native full identity/padding mismatch",
            ));
        }
        for offset in &item.offsets {
            let start = offset.start as usize;
            let end = offset.end as usize;
            if start > end
                || end >= count
                || last_end.is_some_and(|previous| start <= previous)
                || !manifest.padded_ids[start..=end]
                    .iter()
                    .all(|id| item.pad_values.contains(id))
            {
                return Err(Status::invalid_argument(
                    "invalid managed image offset/padding",
                ));
            }
            let text_start = last_end.map_or(0, |end| end + 1);
            if !manifest.unpadded_ids[text_start..start]
                .iter()
                .map(|&id| i64::from(id))
                .eq(manifest.padded_ids[text_start..start].iter().copied())
            {
                return Err(Status::invalid_argument(
                    "managed layout changed a non-image token",
                ));
            }
            last_end = Some(end);
        }
    }
    let text_start = last_end.map_or(0, |end| end + 1);
    if !manifest.unpadded_ids[text_start..]
        .iter()
        .map(|&id| i64::from(id))
        .eq(manifest.padded_ids[text_start..].iter().copied())
    {
        return Err(Status::invalid_argument(
            "managed layout changed a non-image token",
        ));
    }
    let positions = manifest
        .mrope_positions
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("missing native positions"))?;
    validate_integer_array(positions, limits.tokens.saturating_mul(3))?;
    if positions.shape != [3, count as u32] {
        return Err(Status::invalid_argument("invalid native position shape"));
    }
    let delta = manifest
        .mrope_delta
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("missing native position delta"))?;
    validate_integer_array(delta, 1)?;
    if delta.values.len() != 1 {
        return Err(Status::invalid_argument("invalid native position delta"));
    }
    if let Some(types) = &manifest.token_types {
        validate_integer_array(types, limits.tokens)?;
        if types.values.len() != count {
            return Err(Status::invalid_argument("invalid native token types"));
        }
    }
    Ok(())
}

/// Bound integer metadata and check tensor shape without allowing tensor handles.
fn validate_integer_array(array: &proto::IntegerArray, max_elements: usize) -> Result<(), Status> {
    let elements = array
        .shape
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim as usize));
    if array.shape.len() > 3
        || elements != Some(array.values.len())
        || array.values.len() > max_elements
    {
        return Err(Status::invalid_argument(
            "invalid integer metadata shape/size",
        ));
    }
    Ok(())
}

/// Native receiver derives this identifier; it is scoped to the fresh attempt rid.
pub fn validate_external_plan(
    plan: &proto::ExternalEncodePlan,
    images: usize,
) -> Result<(), Status> {
    if plan.receiver_rid.is_empty() {
        return Err(Status::invalid_argument("missing receiver rid"));
    }
    if images == 0 {
        if !plan.parts.is_empty()
            || !plan.image_counts.is_empty()
            || !plan.encoder_receive_endpoints.is_empty()
        {
            return Err(Status::invalid_argument(
                "text request cannot have encode work",
            ));
        }
        return Ok(());
    }
    if plan.encoder_receive_endpoints.len() != 1
        || plan.image_counts != [images as u32]
        || plan.parts.len() != 1
        || plan.encoder_receive_endpoints[0].is_empty()
    {
        return Err(Status::invalid_argument(
            "managed images require exactly one encoder and part",
        ));
    }
    let part = &plan.parts[0];
    if part.part_idx != 0
        || part.native_part_rid != format!("{}_local_part_0", plan.receiver_rid)
        || part.ordered_item_ids.len() != images
        || !part.ordered_item_ids.iter().copied().eq(0..images as u32)
    {
        return Err(Status::invalid_argument(
            "managed part does not match receiver mapping",
        ));
    }
    Ok(())
}
