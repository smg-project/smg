//! Strict, versioned managed EPD control contract; independent of MediaIdentity v1.

use prost::Message;
use tonic::Status;

pub mod proto {
    #![allow(
        clippy::all,
        clippy::absolute_paths,
        clippy::trivially_copy_pass_by_ref,
        unused_qualifications
    )]
    tonic::include_proto!("smg.sglang.managed.v1");
}

/// Wire protocol version for managed SGLang EPD control envelopes.
pub const VERSION: u32 = 1;
/// Layout contract identifier exchanged between managed E, P, and D workers.
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
            manifest_bytes: 32 * 1024 * 1024,
            tokens: 1_048_576,
            images: 1024,
        }
    }
}

/// Zero-allocation check for supported single-host TP topologies.
pub fn is_valid_topology(tp_size: u32, topology_contract: &str) -> bool {
    matches!(
        (tp_size, topology_contract),
        (1, "single-host-tp1")
            | (2, "single-host-tp2")
            | (4, "single-host-tp4")
            | (8, "single-host-tp8")
    )
}

/// Stamp the shared wire version, layout contract, and readiness onto capabilities.
pub fn normalized_capabilities(mut cap: proto::Capabilities, ready: bool) -> proto::Capabilities {
    cap.version = VERSION;
    cap.layout_contract = LAYOUT.into();
    cap.ready = ready;
    cap
}

/// Validate worker-advertised managed capability fields before startup.
pub fn validate_capabilities(cap: &proto::Capabilities) -> Result<(), Status> {
    if cap.model_revision.is_empty()
        || cap.processor_contract.is_empty()
        || cap.topology_contract.is_empty()
        || cap.tp_size == 0
        || !matches!(
            cap.embedding_dtype.as_str(),
            "bfloat16" | "float16" | "float32"
        )
        || !matches!(
            cap.embedding_backend.as_str(),
            "mooncake" | "zmq_to_scheduler"
        )
        || cap.kv_backend.is_empty()
        || (cap.role == "prefill"
            && (cap.bootstrap_host.is_empty()
                || cap.bootstrap_port == 0
                || cap.bootstrap_port > u32::from(u16::MAX)))
    {
        return Err(Status::invalid_argument(
            "incomplete managed capability contract",
        ));
    }
    Ok(())
}

/// Check contract compatibility between an encoder and a prefill/decode worker.
pub fn capabilities_compatible(
    encoder: &proto::Capabilities,
    worker: &proto::Capabilities,
    expected_worker_role: &str,
) -> bool {
    encoder.version == VERSION
        && worker.version == VERSION
        && is_valid_topology(encoder.tp_size, &encoder.topology_contract)
        && is_valid_topology(worker.tp_size, &worker.topology_contract)
        && encoder.role == "encoder"
        && worker.role == expected_worker_role
        && encoder.model_revision == worker.model_revision
        && encoder.processor_contract == worker.processor_contract
        && encoder.layout_contract == LAYOUT
        && worker.layout_contract == LAYOUT
        && encoder.embedding_dtype == worker.embedding_dtype
        && (worker.role == "decode" || encoder.embedding_backend == worker.embedding_backend)
}

/// Stable PD pairing key so managed and unmanaged workers never mix even when pairing is off.
pub fn prepared_pairing_key(cap: &proto::Capabilities) -> String {
    let topology = if cap.kv_backend == "mooncake" {
        "single-host"
    } else {
        cap.topology_contract.as_str()
    };
    serde_json::to_string(&(
        &cap.model_revision,
        &cap.processor_contract,
        &cap.layout_contract,
        &cap.embedding_dtype,
        &cap.kv_backend,
        topology,
    ))
    .unwrap_or_default()
}

/// Require complete PD rendezvous and model contracts.
pub fn validate_binding(binding: &proto::InputBinding) -> Result<(), Status> {
    let bootstrap = binding
        .bootstrap
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("missing bootstrap"))?;
    if binding.prefill_rid.is_empty()
        || binding.prefill_rid.len() > 256
        || binding.decode_rid.is_empty()
        || binding.decode_rid.len() > 256
        || binding.layout_contract != LAYOUT
        || binding.processor_contract.is_empty()
        || binding.model_revision.is_empty()
        || bootstrap.host.is_empty()
        || bootstrap.port == 0
        || bootstrap.port > u32::from(u16::MAX)
        || bootstrap.room < 0
    {
        return Err(Status::invalid_argument("incomplete managed input binding"));
    }
    Ok(())
}

/// Refuse malformed/foreign layouts before D constructs any native request.
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
    if manifest.version != VERSION || manifest.binding.as_ref() != Some(expected) {
        return Err(Status::invalid_argument(
            "managed manifest binding mismatch",
        ));
    }
    let count = manifest.unpadded_ids.len();
    if count == 0
        || count > limits.tokens
        || manifest.padded_ids.len() != count
        || manifest.images.is_empty()
        || manifest.images.len() > limits.images
        || (manifest.image_token_id == 0 && manifest.video_token_id == 0)
    {
        return Err(Status::invalid_argument("incomplete managed media layout"));
    }
    let mut last_end = None;
    let mut cursor = 0usize;
    let mut has_image = false;
    let mut has_video = false;
    for (occurrence, item) in manifest.images.iter().enumerate() {
        if item.occurrence as usize != occurrence
            || item.full_identity.len() != 32
            || item
                .processor_fingerprint
                .as_ref()
                .is_some_and(|value| value.is_empty() || value.len() > 1024)
            || item.pad_values.is_empty()
            // Native EPD may consume the grid into final M-RoPE and retain no grid.
            || !matches!(item.grid_thw.len(), 0 | 3)
            || item.grid_thw.contains(&0)
            || item.offsets.is_empty()
            || !(1..=3).contains(&item.native_format)
            || item.modality > 2
            || item.second_per_grid_ts.len() > 1024
            || item
                .second_per_grid_ts
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || item
                .routing_label
                .as_ref()
                .is_some_and(|label| label.len() > 256)
        {
            return Err(Status::invalid_argument(
                "incomplete managed per-item identity/layout",
            ));
        }
        if item.modality == 2 {
            has_video = true;
        } else {
            has_image = true;
        }
        for offset in &item.offsets {
            let start = offset.start as usize;
            let end = offset.end as usize;
            if start > end || end >= count || last_end.is_some_and(|prev| start <= prev) {
                return Err(Status::invalid_argument("invalid managed media offset"));
            }
            if manifest.padded_ids[cursor..start]
                .iter()
                .zip(&manifest.unpadded_ids[cursor..start])
                .any(|(&padded, &unpadded)| padded != i64::from(unpadded))
            {
                return Err(Status::invalid_argument(
                    "managed padded_ids diverged from unpadded_ids outside media spans",
                ));
            }
            cursor = end + 1;
            last_end = Some(end);
        }
    }
    if manifest.padded_ids[cursor..count]
        .iter()
        .zip(&manifest.unpadded_ids[cursor..count])
        .any(|(&padded, &unpadded)| padded != i64::from(unpadded))
    {
        return Err(Status::invalid_argument(
            "managed padded_ids diverged from unpadded_ids outside media spans",
        ));
    }
    if (has_image && manifest.image_token_id == 0)
        || (has_video && manifest.video_token_id == 0 && manifest.image_token_id == 0)
    {
        return Err(Status::invalid_argument(
            "missing placeholder token id for managed media modality",
        ));
    }
    match (&manifest.mrope_positions, &manifest.mrope_delta) {
        (Some(positions), Some(delta)) => {
            validate_integer_array(positions, limits.tokens.saturating_mul(3))?;
            if positions.shape != [3, count as u32] {
                return Err(Status::invalid_argument("invalid native position shape"));
            }
            validate_integer_array(delta, 1)?;
            if delta.values.len() != 1 {
                return Err(Status::invalid_argument("invalid native position delta"));
            }
        }
        (None, None) => {}
        _ => {
            return Err(Status::invalid_argument(
                "incomplete native M-RoPE positions",
            ))
        }
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

#[cfg(test)]
mod tests {
    use super::{proto::*, *};

    fn sample_binding() -> InputBinding {
        InputBinding {
            prefill_rid: "req-p".into(),
            decode_rid: "req-d".into(),
            bootstrap: Some(Bootstrap {
                host: "10.0.0.1".into(),
                port: 8998,
                room: 42,
            }),
            layout_contract: LAYOUT.into(),
            processor_contract: "smg-image-grid-v1".into(),
            model_revision: "rev1".into(),
        }
    }

    fn sample_manifest(binding: &InputBinding) -> PreparedDecodeInput {
        PreparedDecodeInput {
            version: VERSION,
            binding: Some(binding.clone()),
            unpadded_ids: vec![1, 151_655, 151_655, 2],
            padded_ids: vec![1, 1_000_001, 1_000_001, 2],
            images: vec![ImageMetadata {
                occurrence: 0,
                full_identity: vec![7; 32],
                pad_values: vec![1_000_001],
                processor_fingerprint: None,
                offsets: vec![Offset { start: 1, end: 2 }],
                native_format: 1,
                grid_thw: vec![1, 2, 2],
                routing_label: None,
                native_hash: 0x0707_0707_0707_0707,
                modality: 1,
                second_per_grid_ts: Vec::new(),
            }],
            mrope_positions: Some(IntegerArray {
                shape: vec![3, 4],
                values: vec![0; 12],
            }),
            mrope_delta: Some(IntegerArray {
                shape: vec![1],
                values: vec![-1],
            }),
            token_types: None,
            image_token_id: 151_655,
            image_start_id: 0,
            image_end_id: 0,
            video_token_id: 0,
        }
    }

    #[test]
    fn manifest_validates_with_and_without_mrope_and_rejects_bad_bounds() {
        let binding = sample_binding();
        let mut manifest = sample_manifest(&binding);
        validate_manifest(&manifest, &binding, Limits::default()).unwrap();

        manifest.mrope_positions = None;
        manifest.mrope_delta = None;
        manifest.images[0].grid_thw.clear();
        validate_manifest(&manifest, &binding, Limits::default()).unwrap();

        manifest.mrope_positions = Some(IntegerArray {
            shape: vec![3, 4],
            values: vec![0; 12],
        });
        assert!(validate_manifest(&manifest, &binding, Limits::default()).is_err());

        let mut bad = sample_manifest(&binding);
        bad.images[0].offsets = vec![Offset { start: 2, end: 1 }];
        assert!(validate_manifest(&bad, &binding, Limits::default()).is_err());
    }

    #[test]
    fn manifest_validates_videos_over_32_items_and_rejects_invalid_video_fields() {
        let binding = sample_binding();
        let mut video = sample_manifest(&binding);
        video.image_token_id = 0;
        video.video_token_id = 151_656;
        video.images[0].modality = 2;
        video.images[0].second_per_grid_ts = vec![1.0, 1.0];
        validate_manifest(&video, &binding, Limits::default()).unwrap();

        // Reject unsupported modality and non-finite/non-positive video timings.
        let mut bad_mod = video.clone();
        bad_mod.images[0].modality = 3;
        assert!(validate_manifest(&bad_mod, &binding, Limits::default()).is_err());

        for bad_ts in [vec![0.0], vec![-1.0], vec![f32::NAN], vec![f32::INFINITY]] {
            let mut bad_video = video.clone();
            bad_video.images[0].second_per_grid_ts = bad_ts;
            assert!(validate_manifest(&bad_video, &binding, Limits::default()).is_err());
        }

        // Default limits permit >32 items (e.g. 64 items) to match high-item-count VLM PD specs.
        let mut many = sample_manifest(&binding);
        many.unpadded_ids = vec![151_655; 64];
        many.padded_ids = vec![1_000_001; 64];
        many.images = (0..64u32)
            .map(|idx| ImageMetadata {
                occurrence: idx,
                offsets: vec![Offset {
                    start: idx,
                    end: idx,
                }],
                ..many.images[0].clone()
            })
            .collect();
        many.mrope_positions = None;
        many.mrope_delta = None;
        validate_manifest(&many, &binding, Limits::default()).unwrap();

        // Reject padded_ids mutating text tokens outside image/video offset spans.
        let mut mutated_prefix = sample_manifest(&binding);
        mutated_prefix.padded_ids[0] = 999_999;
        assert!(validate_manifest(&mutated_prefix, &binding, Limits::default()).is_err());

        let mut mutated_suffix = sample_manifest(&binding);
        let last_idx = mutated_suffix.padded_ids.len() - 1;
        mutated_suffix.padded_ids[last_idx] = 999_999;
        assert!(validate_manifest(&mutated_suffix, &binding, Limits::default()).is_err());
    }
}
