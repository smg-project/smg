mod engine_error;
mod fanout;
mod watched;

use prost::Message;
use smg_grpc_client::common_proto as common;

#[test]
/// Keep PD references intact and reject metadata the SGLang wire cannot express.
fn sglang_media_refs_preserve_video_and_refuse_lossy_conversion() {
    let mut request = ProtoGenerateRequest::Sglang(Box::default());
    let refs = vllm::MediaRefs {
        items: vec![vllm::MediaRef {
            modality: common::Modality::Video as i32,
            url: "data:video/mp4;base64,AA==".into(),
        }],
    };
    request.set_media_refs(refs.clone()).unwrap();
    let decode = request.clone_without_mm_pixels();
    let ProtoGenerateRequest::Sglang(decode) = decode else {
        panic!("expected SGLang decode request")
    };
    assert_eq!(
        decode.mm_inputs.as_ref().unwrap().video_urls,
        ["data:video/mp4;base64,AA=="],
        "PD decode needs references to expand the same prompt as Prefill"
    );
    let ProtoGenerateRequest::Sglang(inner) = &request else {
        panic!("expected SGLang request")
    };
    let mm = inner.mm_inputs.as_ref().unwrap();
    assert_eq!(mm.video_urls, ["data:video/mp4;base64,AA=="]);
    assert!(mm.image_urls.is_empty());
    assert!(mm.pixel_values.is_none());
    let mut mixed = refs;
    mixed.items.push(vllm::MediaRef {
        modality: common::Modality::Image as i32,
        url: "https://example.test/image.png".into(),
    });
    request.set_media_refs(mixed).unwrap();
    let mm = request.as_sglang().mm_inputs.as_ref().unwrap();
    assert_eq!(mm.image_urls, ["https://example.test/image.png"]);
    assert_eq!(mm.video_urls, ["data:video/mp4;base64,AA=="]);
}

/// The raw-file request's wire bytes are the files, length-delimited in
/// order, whether the generated field holds them as shared or as owned
/// bytes: what the TRT-LLM servicer reads does not change with the type.
#[test]
fn trtllm_raw_images_encode_as_the_files_length_delimited_in_order() {
    let files: [&[u8]; 2] = [b"abc", b"de"];
    let data = TrtllmMultimodalData {
        image_data: files
            .iter()
            .map(|file| bytes::Bytes::from_static(file))
            .collect(),
    };
    let mut expected = Vec::new();
    for file in files {
        // field 1, wire type 2 (length-delimited): the tag, the length, the bytes
        expected.push(0x0a);
        expected.push(u8::try_from(file.len()).unwrap());
        expected.extend_from_slice(file);
    }
    assert_eq!(data.into_proto().encode_to_vec(), expected);
}

#[test]
/// Account for all inline pixels and raw media across supported engine payloads.
fn inline_bytes_counts_every_inline_tensor_and_raw_image() {
    let trtllm = MultimodalData::Trtllm(TrtllmMultimodalData {
        image_data: vec![
            bytes::Bytes::from(vec![0; 3]),
            bytes::Bytes::from(vec![0; 5]),
        ],
    });
    assert_eq!(trtllm.inline_bytes(), 8);

    let sglang = MultimodalData::Sglang(SglangMultimodalData {
        extra_batches: vec![SglangMultimodalData {
            extra_batches: Vec::new(),
            modality: "video".to_string(),
            image_data: Vec::new(),
            pixel_values: vec![0; 8],
            pixel_values_shape: vec![2],
            pixel_values_dtype: "float32".to_string(),
            model_specific_tensors: HashMap::from([(
                "video_grid_thw".to_string(),
                TensorBytes {
                    data: vec![0; 24],
                    shape: vec![1, 3],
                    dtype: "int64".to_string(),
                },
            )]),
            im_token_id: Some(8),
            mm_placeholders: vec![(4, 2)],
        }],
        modality: "image".to_string(),
        image_data: vec![vec![0; 2]],
        pixel_values: vec![0; 16],
        pixel_values_shape: vec![4, 4],
        pixel_values_dtype: "float32".to_string(),
        model_specific_tensors: HashMap::from([(
            "grid".to_string(),
            TensorBytes {
                data: vec![0; 24],
                shape: vec![1, 3],
                dtype: "int64".to_string(),
            },
        )]),
        im_token_id: None,
        mm_placeholders: vec![(0, 4)],
    });
    assert_eq!(sglang.inline_bytes(), 42 + 8 + 24);
}

#[test]
fn inline_bytes_counts_every_batch_of_a_mixed_request() {
    let alone = MultimodalData::Vllm(vllm_mm_data(common::Modality::Image));
    assert_eq!(alone.inline_bytes(), 16);

    let mut video = vllm_mm_data(common::Modality::Video);
    video.model_specific_tensors.insert(
        "video_grid_thw".to_string(),
        TensorBytes {
            data: vec![0; 24],
            shape: vec![1, 3],
            dtype: "int64".to_string(),
        },
    );
    let mut image = vllm_mm_data(common::Modality::Image);
    image.extra_batches = vec![video];
    assert_eq!(MultimodalData::Vllm(image).inline_bytes(), 56);
}

use super::{multimodal::tensor_bytes_to_tokenspeed, *};

#[test]
fn tokenspeed_image_into_proto_uses_itemized_payload() {
    let mut model_specific_tensors = HashMap::new();
    model_specific_tensors.insert(
        "image_grid_thw".to_string(),
        TensorBytes {
            data: vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
            shape: vec![1, 3],
            dtype: "uint32".to_string(),
        },
    );

    let proto = TokenSpeedMultimodalData {
        items: vec![TokenSpeedMultimodalItem {
            modality: TokenSpeedModality::Image,
            encoder_input: TokenSpeedTensor::inline(vec![42; 8], vec![1, 2], "float32".to_string()),
            model_specific_tensors,
            placeholder_token_id: Some(151655),
            mm_placeholders: vec![(4, 2)],
            content_hash: vec![7; 32],
        }],
        shm_enabled: false,
        shm_min_bytes: 0,
    }
    .into_proto(false);

    assert_eq!(proto.items.len(), 1);
    let item = &proto.items[0];
    assert_eq!(item.modality, common::Modality::Image as i32);
    assert_eq!(item.placeholder_token_id, Some(151655));
    assert_eq!(item.placeholders[0].offset, 4);
    assert_eq!(item.placeholders[0].length, 2);
    assert_eq!(
        inline_tensor_data(item.encoder_input.as_ref().unwrap()),
        &[42; 8]
    );
    assert!(item.model_specific_tensors.contains_key("image_grid_thw"));
}

#[test]
fn tokenspeed_video_into_proto_uses_itemized_payload() {
    let mut model_specific_tensors = HashMap::new();
    model_specific_tensors.insert(
        "video_grid_thw".to_string(),
        TensorBytes {
            data: vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
            shape: vec![1, 3],
            dtype: "uint32".to_string(),
        },
    );

    let proto = TokenSpeedMultimodalData {
        items: vec![TokenSpeedMultimodalItem {
            modality: TokenSpeedModality::Video,
            encoder_input: TokenSpeedTensor::inline(vec![42; 8], vec![1, 2], "float32".to_string()),
            model_specific_tensors,
            placeholder_token_id: Some(151656),
            mm_placeholders: vec![(4, 2)],
            content_hash: vec![7; 32],
        }],
        shm_enabled: false,
        shm_min_bytes: 0,
    }
    .into_proto(false);

    assert_eq!(proto.items.len(), 1);
    let item = &proto.items[0];
    assert_eq!(item.modality, common::Modality::Video as i32);
    assert_eq!(item.placeholder_token_id, Some(151656));
    assert_eq!(item.placeholders[0].offset, 4);
    assert_eq!(item.placeholders[0].length, 2);
    assert_eq!(
        inline_tensor_data(item.encoder_input.as_ref().unwrap()),
        &[42; 8]
    );
    assert!(item.model_specific_tensors.contains_key("video_grid_thw"));
}

#[test]
fn tokenspeed_tensor_data_uses_clean_payload_tags() {
    let tensor = tensor_bytes_to_tokenspeed(
        TensorBytes {
            data: vec![0xaa, 0xbb],
            shape: vec![2, 3],
            dtype: "uint32".to_string(),
        },
        false,
        0,
    );

    assert_eq!(
        tensor.encode_to_vec(),
        vec![
            0x0a, 0x02, 0x02, 0x03, // shape = 1, packed uint32 [2, 3]
            0x12, 0x06, b'u', b'i', b'n', b't', b'3', b'2', // dtype = 2
            0x1a, 0x02, 0xaa, 0xbb, // inline = 3
        ]
    );
}

#[test]
fn tokenspeed_shm_encoder_input_into_proto_uses_shm_payload() {
    let proto = TokenSpeedMultimodalData {
        items: vec![TokenSpeedMultimodalItem {
            modality: TokenSpeedModality::Image,
            encoder_input: TokenSpeedTensor::shm(
                common::ShmHandle {
                    name: "smg-test-shm".to_string(),
                    offset: 0,
                    nbytes: 8,
                    owner_id: "smg:test".to_string(),
                },
                vec![1, 2],
                "bfloat16".to_string(),
            ),
            model_specific_tensors: HashMap::new(),
            placeholder_token_id: Some(151655),
            mm_placeholders: vec![(4, 2)],
            content_hash: vec![7; 32],
        }],
        shm_enabled: true,
        shm_min_bytes: 0,
    }
    .into_proto(false);

    let tensor = proto.items[0].encoder_input.as_ref().unwrap();
    assert_eq!(tensor.shape, vec![1, 2]);
    assert_eq!(tensor.dtype, "bfloat16");
    match tensor.payload.as_ref() {
        Some(tokenspeed::tensor_data::Payload::Shm(handle)) => {
            assert_eq!(handle.name, "smg-test-shm");
            assert_eq!(handle.nbytes, 8);
        }
        _ => panic!("expected shm TensorData payload"),
    }
}

#[test]
#[cfg(target_os = "linux")]
fn tokenspeed_shm_writer_exposes_complete_mapped_payload() {
    use super::shm::tokenspeed_shm_path;

    let expected = [0x12, 0x34, 0x56, 0x78];
    let handle = write_tokenspeed_shm_with(expected.len(), |output| {
        output.copy_from_slice(&expected);
        Ok(())
    })
    .unwrap();
    let payload = std::fs::read(tokenspeed_shm_path(&handle.name));
    cleanup_mm_shm_handles(std::slice::from_ref(&handle));

    assert_eq!(payload.unwrap(), expected);
    assert!(!tokenspeed_shm_path(&handle.name).exists());
}

#[test]
fn tokenspeed_remote_encoder_input_into_proto_uses_remote_payload() {
    let proto = TokenSpeedMultimodalData {
        items: vec![TokenSpeedMultimodalItem {
            modality: TokenSpeedModality::Image,
            encoder_input: TokenSpeedTensor::remote(
                common::RemoteTensorHandle {
                    transport: "nixl".to_string(),
                    descriptor: vec![1, 2, 3],
                    nbytes: 8,
                },
                vec![1, 2],
                "bfloat16".to_string(),
            ),
            model_specific_tensors: HashMap::new(),
            placeholder_token_id: Some(151655),
            mm_placeholders: vec![(4, 2)],
            content_hash: vec![7; 32],
        }],
        shm_enabled: true,
        shm_min_bytes: 0,
    }
    .into_proto(false);

    let tensor = proto.items[0].encoder_input.as_ref().unwrap();
    assert_eq!(tensor.shape, vec![1, 2]);
    assert_eq!(tensor.dtype, "bfloat16");
    match tensor.payload.as_ref() {
        Some(tokenspeed::tensor_data::Payload::Remote(handle)) => {
            assert_eq!(handle.transport, "nixl");
            assert_eq!(handle.descriptor, vec![1, 2, 3]);
            assert_eq!(handle.nbytes, 8);
        }
        _ => panic!("expected remote TensorData payload"),
    }
}

fn inline_tensor_data(tensor: &tokenspeed::TensorData) -> &[u8] {
    match tensor.payload.as_ref() {
        Some(tokenspeed::tensor_data::Payload::Inline(data)) => data.as_ref(),
        _ => panic!("expected inline TensorData payload"),
    }
}

#[test]
fn set_data_parallel_rank_per_engine() {
    let mut vllm_req = ProtoGenerateRequest::Vllm(Box::default());
    vllm_req.set_data_parallel_rank(2);
    assert!(matches!(
        &vllm_req,
        ProtoGenerateRequest::Vllm(req) if req.data_parallel_rank == Some(2)
    ));

    let mut sglang_req = ProtoGenerateRequest::Sglang(Box::default());
    sglang_req.set_data_parallel_rank(3);
    assert!(matches!(
        &sglang_req,
        ProtoGenerateRequest::Sglang(req) if req.data_parallel_rank == Some(3)
    ));

    let mut ts_req = ProtoGenerateRequest::TokenSpeed(Box::default());
    ts_req.set_data_parallel_rank(5);
    assert!(matches!(
        &ts_req,
        ProtoGenerateRequest::TokenSpeed(req) if req.data_parallel_rank == Some(5)
    ));

    // Engines without the proto field ignore the pin
    let mut mlx_req = ProtoGenerateRequest::Mlx(Box::default());
    mlx_req.set_data_parallel_rank(1);
}

fn vllm_mm_data(modality: common::Modality) -> VllmMultimodalData {
    let is_video = modality == common::Modality::Video;
    VllmMultimodalData {
        pixel_values: vec![0u8; 16],
        pixel_values_shape: vec![1, 4],
        pixel_values_dtype: "float32".to_string(),
        model_specific_tensors: HashMap::new(),
        im_token_id: Some(if is_video { 151656 } else { 151655 }),
        mm_placeholders: vec![(3, 4)],
        mm_hashes: vec!["h0".to_string()],
        batched_keys: vec![],
        flat_keys: HashMap::new(),
        keep_on_cpu_keys: vec![],
        encoder_input_key: None,
        modality,
        shm_enabled: false,
        shm_min_bytes: 0,
        rdma_enabled: false,
        extra_batches: Vec::new(),
    }
}

#[test]
fn vllm_modality_round_trips_into_proto() {
    // The video path must set the proto `modality` so the servicer routes to
    // vLLM's video modality; the image path must set image.
    let video = vllm_mm_data(common::Modality::Video).into_proto();
    assert_eq!(video.modality, common::Modality::Video as i32);
    assert_eq!(video.im_token_id, Some(151656));

    let image = vllm_mm_data(common::Modality::Image).into_proto();
    assert_eq!(image.modality, common::Modality::Image as i32);
}
#[test]
fn extend_stop_token_ids_reaches_every_variant() {
    let ids = [7u32, 8];
    let mut req = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
        sampling_params: Some(vllm::SamplingParams::default()),
        ..Default::default()
    }));
    req.extend_stop_token_ids(&ids);
    match &req {
        ProtoGenerateRequest::Vllm(r) => {
            assert_eq!(r.sampling_params.as_ref().unwrap().stop_token_ids, ids);
        }
        _ => panic!("variant changed"),
    }

    // TRT-LLM keeps ids on the request itself.
    let mut req = ProtoGenerateRequest::Trtllm(Box::default());
    req.extend_stop_token_ids(&ids);
    match &req {
        ProtoGenerateRequest::Trtllm(r) => assert_eq!(r.stop_token_ids, ids),
        _ => panic!("variant changed"),
    }

    // Missing sampling params: untouched, no panic.
    let mut req = ProtoGenerateRequest::TokenSpeed(Box::default());
    req.extend_stop_token_ids(&ids);
}

#[test]
fn chunk_semantics_follow_the_response_shape() {
    // The vLLM shape is the delta contract — the ZMQ lane emits it for
    // every engine it fronts, TokenSpeed included.
    assert!(
        ProtoGenerateStreamChunk::Vllm(vllm::GenerateStreamChunk::default())
            .chunk_semantics()
            .is_delta()
    );
    assert!(ProtoGenerateComplete::Vllm(Box::default())
        .chunk_semantics()
        .is_delta());

    // Every other shape reports running totals.
    for chunk in [
        ProtoGenerateStreamChunk::Sglang(sglang::GenerateStreamChunk::default()),
        ProtoGenerateStreamChunk::Trtllm(trtllm::GenerateStreamChunk::default()),
        ProtoGenerateStreamChunk::Mlx(mlx::GenerateStreamChunk::default()),
        ProtoGenerateStreamChunk::TokenSpeed(tokenspeed::GenerateStreamChunk::default()),
    ] {
        assert_eq!(chunk.chunk_semantics(), ChunkSemantics::Cumulative);
    }
    for complete in [
        ProtoGenerateComplete::Sglang(sglang::GenerateComplete::default()),
        ProtoGenerateComplete::Trtllm(trtllm::GenerateComplete::default()),
        ProtoGenerateComplete::Mlx(mlx::GenerateComplete::default()),
        ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete::default()),
    ] {
        assert_eq!(complete.chunk_semantics(), ChunkSemantics::Cumulative);
    }
}

#[test]
fn tokenspeed_generate_messages_expose_the_engine_weight_version() {
    let complete = ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete {
        weight_version: Some("v7".to_string()),
        ..Default::default()
    });
    assert_eq!(complete.weight_version(), Some("v7"));
    let empty = ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete {
        weight_version: Some(String::new()),
        ..Default::default()
    });
    assert_eq!(empty.weight_version(), None, "empty is the same as unset");
    let placeholder = ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete {
        weight_version: Some("default".to_string()),
        ..Default::default()
    });
    assert_eq!(
        placeholder.weight_version(),
        None,
        "the engine's own placeholder is not a version and must not shadow a label"
    );
    let unset = ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete::default());
    assert_eq!(unset.weight_version(), None);

    let chunk = ProtoGenerateStreamChunk::TokenSpeed(tokenspeed::GenerateStreamChunk {
        weight_version: Some("v7".to_string()),
        ..Default::default()
    });
    assert_eq!(chunk.weight_version(), Some("v7"));
    let sglang = ProtoGenerateComplete::Sglang(sglang::GenerateComplete::default());
    assert_eq!(
        sglang.weight_version(),
        None,
        "no other proto carries the field"
    );
}
