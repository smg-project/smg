use std::time::{Duration, Instant};

use prost::Message;
use sha2::{Digest, Sha256};
use smg_grpc_client::sglang_managed::{
    budget::Deadline, manifest_digest, proto::*, validate_external_plan, validate_manifest, Limits,
    LAYOUT,
};

fn fixture() -> PreparedDecodeInput {
    let identity = vec![7; 32];
    let digest = Sha256::digest([b"sglang-mm-wide-pad-v1".as_slice(), &identity].concat());
    let pads: Vec<_> = digest
        .as_slice()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| 1_000_000 + (u64::from_be_bytes(*v) & ((1 << 62) - 1)) as i64)
        .collect();
    let binding = InputBinding {
        decode: Some(OperationRef {
            identity: Some(Identity {
                logical_request_id: "request".into(),
                attempt_id: "attempt".into(),
                branch_id: 0,
            }),
            target: Some(Instance {
                worker_id: "d".into(),
                generation: "d-generation".into(),
            }),
            operation_id: "decode".into(),
            request_digest: vec![1; 32],
            plan_digest: vec![2; 32],
        }),
        prefill: Some(Instance {
            worker_id: "p".into(),
            generation: "p-generation".into(),
        }),
        prefill_rid: "attempt-p".into(),
        bootstrap: Some(Bootstrap {
            host: "p".into(),
            port: 8998,
            room: 17,
        }),
        layout_contract: LAYOUT.into(),
        processor_contract: "qwen2_5_vl-rust-test".into(),
        model_revision: "weights-revision".into(),
        media_routing_labels: vec![],
    };
    let mut input = PreparedDecodeInput {
        version: 1,
        binding: Some(binding),
        unpadded_ids: vec![1, 151655, 2],
        padded_ids: vec![1, pads[0], 2],
        images: vec![ImageMetadata {
            occurrence: 0,
            full_identity: identity,
            pad_values: pads,
            processor_fingerprint: Some("native-fingerprint".into()),
            offsets: vec![Offset { start: 1, end: 1 }],
            native_format: 3,
            grid_thw: vec![1, 2, 2],
            routing_label: None,
            native_hash: u64::from_be_bytes([7; 8]),
        }],
        mrope_positions: Some(IntegerArray {
            shape: vec![3, 3],
            values: vec![0; 9],
        }),
        mrope_delta: Some(IntegerArray {
            shape: vec![1],
            values: vec![0],
        }),
        image_token_id: 151655,
        ..Default::default()
    };
    input.integrity_digest = manifest_digest(&input);
    input
}

#[test]
fn native_wide_padding_roundtrips_without_uint32_truncation() {
    let input = fixture();
    assert!(input.padded_ids[1] > i64::from(u32::MAX));
    let decoded = PreparedDecodeInput::decode(input.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded, input);
    validate_manifest(&decoded, input.binding.as_ref().unwrap(), Limits::default()).unwrap();
}

#[test]
fn rejects_foreign_instances_branches_and_self_consistent_but_incomplete_layout() {
    let original = fixture();
    for change in 0..7 {
        let mut input = original.clone();
        match change {
            0 => {
                input
                    .binding
                    .as_mut()
                    .unwrap()
                    .prefill
                    .as_mut()
                    .unwrap()
                    .generation = "restarted".into();
            }
            1 => {
                input
                    .binding
                    .as_mut()
                    .unwrap()
                    .decode
                    .as_mut()
                    .unwrap()
                    .identity
                    .as_mut()
                    .unwrap()
                    .branch_id = 1;
            }
            2 => input.images[0].full_identity.pop().map(|_| ()).unwrap(),
            3 => input.mrope_positions = None,
            4 => input.images[0].offsets[0].end = 99,
            5 => input.padded_ids[1] = 0,
            _ => input.images[0].pad_values[0] += 1,
        }
        input.integrity_digest = manifest_digest(&input);
        assert!(
            validate_manifest(
                &input,
                original.binding.as_ref().unwrap(),
                Limits::default()
            )
            .is_err(),
            "change {change}"
        );
    }
    let mut input = original.clone();
    input.padded_ids[0] += 1;
    assert!(validate_manifest(
        &input,
        original.binding.as_ref().unwrap(),
        Limits::default()
    )
    .is_err());
    assert!(validate_manifest(
        &original,
        original.binding.as_ref().unwrap(),
        Limits {
            manifest_bytes: 1,
            ..Limits::default()
        }
    )
    .is_err());
}

#[test]
fn all_image_occurrences_form_one_native_part() {
    let plan = ExternalEncodePlan {
        receiver_rid: "attempt".into(),
        encoder_receive_endpoints: vec!["http://e:30000".into()],
        image_counts: vec![3],
        parts: vec![EncodePart {
            native_part_rid: "attempt_local_part_0".into(),
            part_idx: 0,
            ordered_item_ids: vec![0, 1, 2],
        }],
    };
    validate_external_plan(&plan, 3).unwrap();
    let mut split = plan.clone();
    split
        .encoder_receive_endpoints
        .push("http://e2:30000".into());
    assert!(validate_external_plan(&split, 3).is_err());
    let mut reordered = plan;
    reordered.parts[0].ordered_item_ids.swap(0, 1);
    assert!(validate_external_plan(&reordered, 3).is_err());
}

#[test]
fn original_expiry_and_monotonic_budget_cannot_be_extended() {
    let now = Instant::now();
    let budget = Budget {
        original_expiry_unix_ms: 2000,
        remaining_timeout_ms: 300,
        ..Default::default()
    };
    let deadline = Deadline::admit(&budget, 1000, now, 0, 20, 3000).unwrap();
    assert_eq!(deadline.remaining(now).unwrap(), Duration::from_millis(300));
    assert!(deadline
        .remaining(now + Duration::from_millis(300))
        .is_err());
    assert!(Deadline::admit(&budget, 2000, now, 0, 20, 3000).is_err());
    assert!(Deadline::admit(&budget, 1000, now, 21, 20, 3000).is_err());
}

/// Native Qwen EPD consumes image grids into M-RoPE and may leave the per-item namespace unset.
#[test]
fn preserves_native_epd_absent_grid_and_processor_fingerprint() {
    let mut input = fixture();
    input.images[0].grid_thw.clear();
    input.images[0].processor_fingerprint = None;
    input.integrity_digest = manifest_digest(&input);
    validate_manifest(&input, input.binding.as_ref().unwrap(), Limits::default()).unwrap();
    input.images[0].grid_thw = vec![1, 2];
    input.integrity_digest = manifest_digest(&input);
    assert!(validate_manifest(&input, input.binding.as_ref().unwrap(), Limits::default()).is_err());
}
