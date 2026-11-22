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
        || manifest.image_token_id == 0
    {
        return Err(Status::invalid_argument("incomplete managed image layout"));
    }
    let mut last_end = None;
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
            || item
                .routing_label
                .as_ref()
                .is_some_and(|label| label.len() > 256)
        {
            return Err(Status::invalid_argument(
                "incomplete managed per-image identity/layout",
            ));
        }
        for offset in &item.offsets {
            let start = offset.start as usize;
            let end = offset.end as usize;
            if start > end || end >= count || last_end.is_some_and(|prev| start <= prev) {
                return Err(Status::invalid_argument("invalid managed image offset"));
            }
            last_end = Some(end);
        }
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
}
