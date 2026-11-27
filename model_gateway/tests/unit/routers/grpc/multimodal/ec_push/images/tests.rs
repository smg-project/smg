use super::*;
use crate::routers::grpc::backend_client::BasicWorkerBuilder;

fn fixture() -> (ProtoGenerateRequest, Vec<EncodeWorkerAssignment>) {
    let request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                input_ids: vec![11, 9, 12, 9, 13, 9, 14],
                original_text: "original".into(),
            },
        )),
        media_refs: Some(vllm::MediaRefs {
            items: ["a", "b", "a"]
                .into_iter()
                .map(|url| vllm::MediaRef {
                    modality: Modality::Image as i32,
                    url: url.into(),
                    ..Default::default()
                })
                .collect(),
        }),
        ..Default::default()
    }));
    let workers: Vec<Arc<dyn Worker>> = ["grpc://e0:9000", "grpc://e1:9000"]
        .into_iter()
        .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
        .collect();
    let assignments = [0, 1, 1]
        .into_iter()
        .enumerate()
        .map(|(item_index, worker)| EncodeWorkerAssignment {
            item_index,
            worker: workers[worker].clone(),
        })
        .collect();
    (request, assignments)
}

/// Complete each E group with different image lengths and per-occurrence PUSH ids.
fn complete(groups: &mut [ImageGroup]) {
    for group in groups {
        let ProtoGenerateRequest::Vllm(child) = &group.request else {
            panic!()
        };
        let refs = &child.media_refs.as_ref().unwrap().items;
        let mut mm = vllm::MultimodalInputs {
            modality: Modality::Image as i32,
            batched_keys: vec!["image_grid_thw".into()],
            ..Default::default()
        };
        let mut expanded = Vec::new();
        let mut local = 0;
        let mut grid = Vec::new();
        let mut handles = serde_json::Map::new();
        for &token in tokens(child).unwrap() {
            if token != 9 {
                expanded.push(token);
                continue;
            }
            let hash = &refs[local].url;
            let length = if hash == "a" { 2 } else { 3 };
            mm.mm_hashes.push(hash.clone());
            mm.mm_placeholders.push(vllm::PlaceholderRange {
                offset: expanded.len() as u32,
                length,
            });
            expanded.extend(std::iter::repeat_n(9, length as usize));
            grid.extend(
                [1_i64, length as i64, 4]
                    .into_iter()
                    .flat_map(i64::to_le_bytes),
            );
            handles.insert(hash.clone(), serde_json::json!({"size_bytes": 128}));
            local += 1;
        }
        mm.model_specific_tensors.insert(
            "image_grid_thw".into(),
            vllm::TensorData {
                shape: vec![local as u32, 3],
                dtype: "int64".into(),
                payload: Some(vllm::tensor_data::Payload::Inline(grid.into())),
            },
        );
        let mut params = serde_json::json!({"ec_items": group.indices.iter().map(|i| {
        serde_json::json!({"transfer_id": format!("t{i}")})
    }).collect::<Vec<_>>()});
        super::super::apply_completion(
            &mut group.request,
            vllm::GenerateComplete {
                media_identity: Some(vllm::MediaIdentity {
                    prompt_token_ids: expanded,
                    mm_inputs: Some(mm),
                    extra_mm_inputs: vec![],
                }),
                ec_transfer_params_json: Some(serde_json::Value::Object(handles).to_string()),
                ..Default::default()
            },
            &mut params,
            "ECMooncakeConnector",
        )
        .unwrap();
    }
}

#[test]
fn grouped_images_restore_text_order_grids_and_duplicate_transfers() {
    let (request, assignments) = fixture();
    let mut groups = split(&request, &assignments, 9).unwrap();
    assert_eq!(
        groups.iter().map(|g| g.indices.clone()).collect::<Vec<_>>(),
        vec![vec![0], vec![1, 2]]
    );
    complete(&mut groups);
    groups.reverse(); // Encoder completion order must not affect prompt order.
    let ProtoGenerateRequest::Vllm(merged) = merge(&request, &groups, Some(9)).unwrap() else {
        panic!()
    };
    assert_eq!(
        tokens(&merged).unwrap(),
        &[11, 9, 9, 12, 9, 9, 9, 13, 9, 9, 14]
    );
    assert!(merged.media_refs.is_none());
    let mm = merged.mm_inputs.unwrap();
    assert_eq!(mm.mm_hashes, ["a", "b", "a"]);
    assert_eq!(
        mm.mm_placeholders
            .iter()
            .map(|p| (p.offset, p.length))
            .collect::<Vec<_>>(),
        [(1, 2), (4, 3), (8, 2)]
    );
    let grid = &mm.model_specific_tensors["image_grid_thw"];
    assert_eq!(grid.shape, [3, 3]);
    let Some(vllm::tensor_data::Payload::Inline(bytes)) = &grid.payload else {
        panic!()
    };
    let values: Vec<_> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .copied()
        .map(i64::from_le_bytes)
        .collect();
    assert_eq!(values, [1, 2, 4, 1, 3, 4, 1, 2, 4]);
    let handles: serde_json::Value =
        serde_json::from_str(merged.ec_transfer_params_json.as_ref().unwrap()).unwrap();
    assert_eq!(
        handles["ec_items"],
        serde_json::json!([
            {"mm_hash":"a", "transfer_id":"t0"},
            {"mm_hash":"b", "transfer_id":"t1"},
            {"mm_hash":"a", "transfer_id":"t2"},
        ])
    );
}

#[test]
fn pull_handles_from_different_encoders_are_merged_without_push_ids() {
    let (request, assignments) = fixture();
    let mut groups = split(&request, &assignments, 9).unwrap();
    complete(&mut groups);
    for group in &mut groups {
        let ProtoGenerateRequest::Vllm(child) = &mut group.request else {
            panic!()
        };
        let mut handles: serde_json::Value =
            serde_json::from_str(child.ec_transfer_params_json.as_ref().unwrap()).unwrap();
        handles.as_object_mut().unwrap().remove("ec_items");
        child.ec_transfer_params_json = Some(handles.to_string());
    }
    let ProtoGenerateRequest::Vllm(merged) = merge(&request, &groups, Some(9)).unwrap() else {
        panic!()
    };
    let handles: serde_json::Value =
        serde_json::from_str(merged.ec_transfer_params_json.as_ref().unwrap()).unwrap();
    assert!(handles.get("ec_items").is_none());
    assert_eq!(handles["a"]["size_bytes"], 128);
    assert_eq!(handles["b"]["size_bytes"], 128);
}

#[test]
fn incompatible_prompt_edits_leave_the_original_request_intact() {
    let (request, assignments) = fixture();
    let mut groups = split(&request, &assignments, 9).unwrap();
    complete(&mut groups);
    let ProtoGenerateRequest::Vllm(child) = &mut groups[0].request else {
        panic!()
    };
    let Some(vllm::generate_request::Input::Tokenized(input)) = child.input.as_mut() else {
        panic!()
    };
    input.input_ids[0] = 77;
    assert!(merge(&request, &groups, Some(9)).is_none());
    assert!(request.has_vllm_media_refs());
}

#[test]
fn single_encoder_and_mixed_modalities_keep_the_whole_prompt() {
    let (request, mut assignments) = fixture();
    assignments[0].worker = assignments[1].worker.clone();
    assert!(split(&request, &assignments, 9).is_none());
    let (mut request, assignments) = fixture();
    let ProtoGenerateRequest::Vllm(original) = &mut request else {
        panic!()
    };
    original.media_refs.as_mut().unwrap().items[1].modality = Modality::Video as i32;
    assert!(split(&request, &assignments, 9).is_none());
}
