#[cfg(target_os = "linux")]
use std::path::Path;
use std::{mem::size_of, sync::Arc};

use llm_multimodal::{
    audio::DecodedAudio, AudioClip, AudioSource, ImageDetail, ImageFrame, VideoClip,
};
use ndarray::{ArrayD, IxDyn};

use super::*;
#[cfg(target_os = "linux")]
use crate::routers::grpc::proto_wrapper::write_tokenspeed_shm_with;

#[cfg(target_os = "linux")]
fn pending_tokenspeed_shm_assembly() -> (PendingTokenSpeedAssembly, std::path::PathBuf) {
    let handle = write_tokenspeed_shm_with(4, |output| {
        output.copy_from_slice(&[1, 2, 3, 4]);
        Ok(())
    })
    .unwrap();
    let path = Path::new("/dev/shm").join(&handle.name);
    let data = TokenSpeedMultimodalData {
        items: vec![TokenSpeedMultimodalItem {
            modality: TokenSpeedModality::Image,
            encoder_input: TokenSpeedTensor::shm(handle, vec![2], "bfloat16".to_string()),
            model_specific_tensors: HashMap::new(),
            placeholder_token_id: None,
            mm_placeholders: vec![],
            content_hash: vec![],
        }],
        shm_enabled: true,
        shm_min_bytes: 0,
    };
    (PendingTokenSpeedAssembly::new(data), path)
}

#[test]
#[cfg(target_os = "linux")]
fn pending_tokenspeed_assembly_cleans_shm_when_dropped() {
    let (pending, path) = pending_tokenspeed_shm_assembly();
    assert!(path.exists());

    drop(pending);

    assert!(!path.exists());
}

#[test]
#[cfg(target_os = "linux")]
fn pending_tokenspeed_assembly_transfers_shm_ownership() {
    let (pending, path) = pending_tokenspeed_shm_assembly();
    let data = pending.into_inner().unwrap();
    assert!(path.exists());

    cleanup_tokenspeed_items_encoder_shm(&data.items, None);

    assert!(!path.exists());
}

/// One image whose flat span is the whole encoder input: the tensor
/// handed over by value moves into the item's payload, and the item is
/// the same as the one written from a view of it.
fn single_image_intermediate() -> PrecomputedMultimodalIntermediate {
    let mut model_specific = HashMap::new();
    model_specific.insert(
        "patches_per_image".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![4],
            shape: vec![1],
        },
    );
    let preprocessed = PreprocessedEncoderInputs {
        encoder_input: ArrayD::from_shape_vec(
            IxDyn(&[4, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap()
        .into(),
        feature_token_counts: vec![4],
        item_sizes: vec![(1, 1)],
        model_specific,
    };
    PrecomputedMultimodalIntermediate {
        preprocessed,
        media: MediaBatch::Images(vec![Arc::new(ImageFrame::new(
            image::DynamicImage::new_rgb8(1, 1),
            bytes::Bytes::from_static(b"a"),
            ImageDetail::Auto,
            llm_multimodal::ImageSource::InlineBytes,
            "hash-a".to_string(),
        ))]),
        bindings: vec![PromptBinding {
            item_index: 0,
            prompt_ordinal: 0,
            structural: PlaceholderRange {
                offset: 10,
                length: 4,
            },
            patches: vec![PlaceholderRange {
                offset: 10,
                length: 4,
            }],
        }],
        placeholder_token_id: Some(151655),
        field_layouts: EncoderFieldLayouts::new(
            FieldLayout::flat("patches_per_image"),
            HashMap::from([("patches_per_image".to_string(), FieldLayout::Batched)]),
        ),
        keep_on_cpu_keys: vec![],
        encoder_input_key: None,
    }
}

#[test]
fn a_single_whole_item_moves_its_encoder_input_into_the_same_payload() {
    use crate::routers::grpc::proto_wrapper::TokenSpeedTensorStorage;

    let borrowed = assemble_tokenspeed(&single_image_intermediate(), None, false).unwrap();
    let mut taken = single_image_intermediate();
    assert!(take_whole_encoder_input(&mut taken).is_some());
    assert_eq!(taken.preprocessed.encoder_input.view().len(), 0);

    let options = vec![tokenspeed_assembly_options(Modality::Image, None, false)];
    let owned =
        assemble_tokenspeed_batches_owned(vec![single_image_intermediate()], options).unwrap();
    assert_eq!(owned.items.len(), 1);
    let (a, b) = (
        &borrowed.items[0].encoder_input,
        &owned.items[0].encoder_input,
    );
    assert_eq!(a.shape, b.shape);
    assert_eq!(a.dtype, b.dtype);
    match (&a.storage, &b.storage) {
        (TokenSpeedTensorStorage::Inline(x), TokenSpeedTensorStorage::Inline(y)) => {
            assert_eq!(x, y);
            // Eight bfloat16 values, two bytes each.
            assert_eq!(x.len(), 16);
        }
        other => panic!("inline payloads expected, got {other:?}"),
    }
    assert_eq!(
        owned.items[0].mm_placeholders,
        borrowed.items[0].mm_placeholders
    );
    assert_eq!(owned.items[0].content_hash, borrowed.items[0].content_hash);
}

/// The production case of the move: the one-pass wire buffer, already
/// in the dtype the worker reads, handed over whole.
fn single_image_wire_intermediate() -> PrecomputedMultimodalIntermediate {
    use llm_multimodal::PixelNorm;
    let mut intermediate = single_image_intermediate();
    // Eight bfloat16 values with the shape [4, 2]: 1.0, 2.0, ... 8.0.
    let bytes: Vec<u8> = [
        0x3f80u16, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
    intermediate.preprocessed.encoder_input = EncoderInput::wire(
        bytes,
        EncoderDtype::BFloat16,
        vec![4, 2],
        PixelNorm {
            scale: 1.0,
            offset: [0.0; 3],
            divisor: [1.0; 3],
            channel_block: 512,
        },
    )
    .expect("a wire of the shape's length");
    intermediate
}

#[test]
fn a_single_whole_wire_item_moves_its_own_bytes_into_the_payload() {
    use crate::routers::grpc::proto_wrapper::TokenSpeedTensorStorage;

    let wire_bytes = match &single_image_wire_intermediate().preprocessed.encoder_input {
        EncoderInput::Wire { bytes, .. } => bytes.clone(),
        other => panic!("a wire input expected, got {other:?}"),
    };
    let borrowed = assemble_tokenspeed(&single_image_wire_intermediate(), None, false).unwrap();
    let options = vec![tokenspeed_assembly_options(Modality::Image, None, false)];
    let owned =
        assemble_tokenspeed_batches_owned(vec![single_image_wire_intermediate()], options).unwrap();
    let (a, b) = (
        &borrowed.items[0].encoder_input,
        &owned.items[0].encoder_input,
    );
    assert_eq!(a.shape, vec![4, 2]);
    assert_eq!(a.shape, b.shape);
    assert_eq!(a.dtype, "bfloat16");
    assert_eq!(a.dtype, b.dtype);
    match (&a.storage, &b.storage) {
        (TokenSpeedTensorStorage::Inline(x), TokenSpeedTensorStorage::Inline(y)) => {
            assert_eq!(
                x, &wire_bytes,
                "the view-written item carries the wire's bytes"
            );
            assert_eq!(y, &wire_bytes, "the moved item carries the wire's bytes");
        }
        other => panic!("inline payloads expected, got {other:?}"),
    }
}

/// Two items share one tensor: both are written from views, nothing moves.
#[test]
fn a_batch_of_two_items_keeps_its_encoder_input() {
    let mut two = two_image_intermediate();
    assert!(take_whole_encoder_input(&mut two).is_none());
    assert_eq!(two.preprocessed.encoder_input.view().len(), 8);
}

fn two_image_intermediate() -> PrecomputedMultimodalIntermediate {
    let mut model_specific = HashMap::new();
    model_specific.insert(
        "patches_per_image".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![2, 2],
            shape: vec![2],
        },
    );
    model_specific.insert(
        "image_grid_thw".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![1, 2, 3, 4, 5, 6],
            shape: vec![2, 3],
        },
    );

    let preprocessed = PreprocessedEncoderInputs {
        encoder_input: ArrayD::from_shape_vec(
            IxDyn(&[4, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap()
        .into(),
        feature_token_counts: vec![2, 2],
        item_sizes: vec![(1, 1), (1, 1)],
        model_specific,
    };

    let images = vec![
        Arc::new(ImageFrame::new(
            image::DynamicImage::new_rgb8(1, 1),
            bytes::Bytes::from_static(b"a"),
            ImageDetail::Auto,
            llm_multimodal::ImageSource::InlineBytes,
            "hash-a".to_string(),
        )),
        Arc::new(ImageFrame::new(
            image::DynamicImage::new_rgb8(1, 1),
            bytes::Bytes::from_static(b"b"),
            ImageDetail::Auto,
            llm_multimodal::ImageSource::InlineBytes,
            "hash-b".to_string(),
        )),
    ];

    PrecomputedMultimodalIntermediate {
        preprocessed,
        media: MediaBatch::Images(images),
        bindings: vec![
            PromptBinding {
                item_index: 0,
                prompt_ordinal: 0,
                structural: PlaceholderRange {
                    offset: 10,
                    length: 2,
                },
                patches: vec![PlaceholderRange {
                    offset: 10,
                    length: 2,
                }],
            },
            PromptBinding {
                item_index: 1,
                prompt_ordinal: 1,
                structural: PlaceholderRange {
                    offset: 20,
                    length: 2,
                },
                patches: vec![PlaceholderRange {
                    offset: 20,
                    length: 2,
                }],
            },
        ],
        placeholder_token_id: Some(151655),
        field_layouts: EncoderFieldLayouts::new(
            FieldLayout::flat("patches_per_image"),
            HashMap::from([
                ("patches_per_image".to_string(), FieldLayout::Batched),
                ("image_grid_thw".to_string(), FieldLayout::Batched),
            ]),
        ),
        keep_on_cpu_keys: vec![],
        encoder_input_key: None,
    }
}

#[test]
fn assemble_tokenspeed_splits_image_items() {
    let intermediate = two_image_intermediate();
    let assembled = assemble_tokenspeed(&intermediate, None, false).unwrap();
    assert_eq!(assembled.items.len(), 2);

    let first = &assembled.items[0];
    assert_eq!(first.modality, TokenSpeedModality::Image);
    assert_eq!(first.encoder_input.shape, vec![2, 2]);
    // bf16 is the default TokenSpeed encoder_input wire dtype (2 bytes/elem).
    assert_eq!(first.encoder_input.nbytes(), 4 * size_of::<u16>());
    assert_eq!(first.mm_placeholders, vec![(10, 2)]);
    assert_eq!(
        first.content_hash,
        hash_hex_strings(std::iter::once("hash-a"))
    );
    assert_eq!(
        first.model_specific_tensors["image_grid_thw"].shape,
        vec![1, 3]
    );
    assert_eq!(
        first.model_specific_tensors["patches_per_image"].shape,
        vec![1]
    );

    let second = &assembled.items[1];
    assert_eq!(second.encoder_input.shape, vec![2, 2]);
    assert_eq!(second.mm_placeholders, vec![(20, 2)]);
    assert_eq!(
        second.content_hash,
        hash_hex_strings(std::iter::once("hash-b"))
    );
    assert_eq!(
        second.model_specific_tensors["image_grid_thw"].shape,
        vec![1, 3]
    );
}

/// One image with two encoder features, bound to a structural range of
/// `structural_length` and the given patch ranges.
fn one_image_intermediate(
    structural_length: usize,
    patches: Vec<PlaceholderRange>,
) -> PrecomputedMultimodalIntermediate {
    let mut model_specific = HashMap::new();
    model_specific.insert(
        "patches_per_image".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![2],
            shape: vec![1],
        },
    );

    PrecomputedMultimodalIntermediate {
        preprocessed: PreprocessedEncoderInputs {
            encoder_input: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0])
                .unwrap()
                .into(),
            feature_token_counts: vec![2],
            item_sizes: vec![(1, 1)],
            model_specific,
        },
        media: MediaBatch::Images(vec![Arc::new(ImageFrame::new(
            image::DynamicImage::new_rgb8(1, 1),
            bytes::Bytes::from_static(b"a"),
            ImageDetail::Auto,
            llm_multimodal::ImageSource::InlineBytes,
            "hash-a".to_string(),
        ))]),
        bindings: vec![PromptBinding {
            item_index: 0,
            prompt_ordinal: 0,
            structural: PlaceholderRange {
                offset: 10,
                length: structural_length,
            },
            patches,
        }],
        placeholder_token_id: Some(151655),
        field_layouts: EncoderFieldLayouts::new(
            FieldLayout::flat("patches_per_image"),
            HashMap::from([("patches_per_image".to_string(), FieldLayout::Batched)]),
        ),
        keep_on_cpu_keys: vec![],
        encoder_input_key: None,
    }
}

/// An item with no patches is sent as its whole structural range, so a
/// range that does not match the item's feature count would hand the
/// backend the wrong prompt positions.
#[test]
fn a_patchless_item_is_rejected_when_its_range_misses_the_features() {
    let good = one_image_intermediate(2, vec![]);
    let assembled = assemble_tokenspeed(&good, None, false).unwrap();
    assert_eq!(assembled.items[0].mm_placeholders, vec![(10, 2)]);

    let bad = one_image_intermediate(3, vec![]);
    let error = assemble_tokenspeed(&bad, None, false)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("3 prompt positions for 2 encoder features"),
        "{error}"
    );
}

/// The same mismatch is fine once the item declares patches: those are the
/// ranges that get sent, and they already carry the feature count.
#[test]
fn a_patched_item_may_span_a_wider_structural_range() {
    let intermediate = one_image_intermediate(
        5,
        vec![PlaceholderRange {
            offset: 11,
            length: 2,
        }],
    );
    let assembled = assemble_tokenspeed(&intermediate, None, false).unwrap();
    assert_eq!(assembled.items[0].mm_placeholders, vec![(11, 2)]);
}

#[test]
fn assemble_tokenspeed_v41_layouts_stay_inline_when_shm_disabled() {
    use crate::routers::grpc::proto_wrapper::TokenSpeedTensorStorage;

    // DeepSeek-V4.1-shaped batch: flat primary sliced by patches_per_image,
    // batched int grids, and a second flat tensor (`types`) sliced by
    // types_per_image. shm_min_bytes=0 would push every tensor to /dev/shm
    // if SHM were on; the ZMQ arm assembles with SHM off and must keep all
    // payloads inline.
    let mut model_specific = HashMap::new();
    model_specific.insert(
        "patches_per_image".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![2, 2],
            shape: vec![2],
        },
    );
    model_specific.insert(
        "types_per_image".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![3, 3],
            shape: vec![2],
        },
    );
    model_specific.insert(
        "vit_grid".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![1, 2, 3, 4, 5, 6],
            shape: vec![2, 3],
        },
    );
    model_specific.insert(
        "llm_grid".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![6, 5, 4, 3, 2, 1],
            shape: vec![2, 3],
        },
    );
    model_specific.insert(
        "types".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![0, 1, 0, 1, 0, 1],
            shape: vec![6],
        },
    );

    let preprocessed = PreprocessedEncoderInputs {
        encoder_input: ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0; 8])
            .unwrap()
            .into(),
        feature_token_counts: vec![2, 2],
        item_sizes: vec![(1, 1), (1, 1)],
        model_specific,
    };

    let images = vec![
        Arc::new(ImageFrame::new(
            image::DynamicImage::new_rgb8(1, 1),
            bytes::Bytes::from_static(b"a"),
            ImageDetail::Auto,
            llm_multimodal::ImageSource::InlineBytes,
            "hash-a".to_string(),
        )),
        Arc::new(ImageFrame::new(
            image::DynamicImage::new_rgb8(1, 1),
            bytes::Bytes::from_static(b"b"),
            ImageDetail::Auto,
            llm_multimodal::ImageSource::InlineBytes,
            "hash-b".to_string(),
        )),
    ];

    let intermediate = PrecomputedMultimodalIntermediate {
        preprocessed,
        media: MediaBatch::Images(images),
        bindings: vec![
            PromptBinding {
                item_index: 0,
                prompt_ordinal: 0,
                structural: PlaceholderRange {
                    offset: 10,
                    length: 2,
                },
                patches: vec![PlaceholderRange {
                    offset: 10,
                    length: 2,
                }],
            },
            PromptBinding {
                item_index: 1,
                prompt_ordinal: 1,
                structural: PlaceholderRange {
                    offset: 20,
                    length: 2,
                },
                patches: vec![PlaceholderRange {
                    offset: 20,
                    length: 2,
                }],
            },
        ],
        placeholder_token_id: Some(129264),
        field_layouts: EncoderFieldLayouts::new(
            FieldLayout::flat("patches_per_image"),
            HashMap::from([
                ("vit_grid".to_string(), FieldLayout::Batched),
                ("llm_grid".to_string(), FieldLayout::Batched),
                ("types".to_string(), FieldLayout::flat("types_per_image")),
                ("patches_per_image".to_string(), FieldLayout::Batched),
                ("types_per_image".to_string(), FieldLayout::Batched),
            ]),
        ),
        keep_on_cpu_keys: vec![
            "vit_grid".to_string(),
            "llm_grid".to_string(),
            "types".to_string(),
        ],
        encoder_input_key: Some("patches".to_string()),
    };

    let assembled = assemble_tokenspeed_with_options(
        &intermediate,
        TokenSpeedAssemblyOptions {
            shm_enabled: false,
            shm_min_bytes: 0,
            encoder_input_dtype: "bfloat16".to_string(),
            skip_pixel_values: false,
        },
        None,
    )
    .unwrap();

    assert_eq!(assembled.items.len(), 2);
    assert!(!assembled.shm_enabled);
    for item in &assembled.items {
        assert_eq!(item.encoder_input.shape, vec![2, 2]);
        assert!(matches!(
            item.encoder_input.storage,
            TokenSpeedTensorStorage::Inline(_)
        ));
        assert_eq!(item.model_specific_tensors["vit_grid"].shape, vec![1, 3]);
        assert_eq!(item.model_specific_tensors["llm_grid"].shape, vec![1, 3]);
        assert_eq!(item.model_specific_tensors["types"].shape, vec![3]);
        // Grid/type metadata must keep the exact integer wire dtype that
        // `model_specific_to_tensor_bytes` assigns to `UintTensor`; casting
        // to the encoder dtype would corrupt the values. (Side tensors are
        // `TensorBytes` — always inline by construction.)
        for key in ["vit_grid", "llm_grid", "types"] {
            assert_eq!(item.model_specific_tensors[key].dtype, "uint32", "{key}");
        }
    }
}

/// A video batch of the Qwen3-VL family, from the family's processor to the
/// vLLM wire with the layout its spec declares for video: validation must
/// find the batch's own sizes tensor (`patches_per_video`), and the wire
/// must name it, not the image one. The spec once answered every modality
/// with the image layout, and every video failed here with
/// "missing flat sizes tensor patches_per_image".
#[test]
fn a_qwen3_vl_video_batch_assembles_with_its_own_sizes_key() {
    use llm_multimodal::{
        vision::Qwen3VLProcessor, ModelMetadata, ModelRegistry, PreProcessorConfig, Tokenizer,
        VideoClipOptions, VideoSource, VisionPreProcessor,
    };

    struct NoTokenizer;
    impl Tokenizer for NoTokenizer {
        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }
        fn encode_text(&self, _text: &str) -> Option<Vec<u32>> {
            None
        }
    }

    let model_config = serde_json::json!({
        "model_type": "qwen3_vl",
        "image_token_id": 151655,
        "video_token_id": 151656,
    });
    let registry = ModelRegistry::new();
    let spec = registry
        .lookup(&ModelMetadata {
            model_id: "qwen3-vl",
            tokenizer: &NoTokenizer,
            config: &model_config,
        })
        .expect("the family's spec");

    let config = PreProcessorConfig::from_json(
        r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "size": {"shortest_edge": 4096, "longest_edge": 25165824}}"#,
    )
    .expect("preprocessor config");
    let processor = Qwen3VLProcessor::from_preprocessor_config(&config);
    let frames: Vec<image::DynamicImage> = (0..4u8)
        .map(|t| {
            image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(64, 64, |x, y| {
                image::Rgb([x as u8 * 4, y as u8 * 4, t * 60])
            }))
        })
        .collect();
    let preprocessed = processor
        .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
        .expect("video preprocess");
    let rows = preprocessed.encoder_input.shape()[0];
    let features = preprocessed.encoder_input.shape()[1];
    let tokens = preprocessed.feature_token_counts[0];
    assert_eq!(
        (rows, tokens),
        (32, 8),
        "4 frames of 64x64: 2 x 4 x 4 patches, merged 2x2"
    );

    let intermediate = PrecomputedMultimodalIntermediate {
        preprocessed,
        media: MediaBatch::Videos(vec![Arc::new(VideoClip::new(
            frames,
            bytes::Bytes::from_static(b"clip"),
            VideoSource::InlineBytes,
            "video-hash".to_string(),
        ))]),
        bindings: vec![PromptBinding {
            item_index: 0,
            prompt_ordinal: 0,
            structural: PlaceholderRange {
                offset: 10,
                length: tokens,
            },
            patches: vec![],
        }],
        placeholder_token_id: Some(151656),
        field_layouts: spec.encoder_field_layouts_for(Modality::Video),
        keep_on_cpu_keys: spec.keep_on_cpu_keys_for(Modality::Video),
        encoder_input_key: spec.encoder_input_key_for(Modality::Video),
    };

    validate_precomputed_batch(&intermediate)
        .expect("a video batch validates against the spec's video layout");
    let assembled = assemble_vllm(
        intermediate,
        &VllmAssembly {
            encoder_dtype: "bfloat16".to_string(),
            device_normalizes: false,
            shm_enabled: false,
            shm_min_bytes: 0,
        },
    )
    .expect("vLLM assembly of a video batch");

    assert_eq!(assembled.modality, common::Modality::Video);
    assert_eq!(
        assembled.flat_keys,
        HashMap::from([("pixel_values".to_string(), "patches_per_video".to_string())])
    );
    for key in [
        "video_grid_thw",
        "patches_per_video",
        "video_second_per_grid",
    ] {
        assert!(
            assembled.batched_keys.iter().any(|k| k == key),
            "{key} is batched"
        );
        assert!(
            assembled.model_specific_tensors.contains_key(key),
            "{key} is sent"
        );
    }
    assert!(
        !assembled
            .batched_keys
            .iter()
            .any(|k| k.starts_with("image_")),
        "no image tensor is announced for a video batch: {:?}",
        assembled.batched_keys
    );
    assert_eq!(
        assembled.pixel_values_shape,
        vec![rows as u32, features as u32]
    );
    assert_eq!(assembled.mm_placeholders, vec![(10, tokens as u32)]);
}

#[test]
fn assemble_tokenspeed_splits_video_items() {
    let mut model_specific = HashMap::new();
    model_specific.insert(
        "patches_per_video".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![2, 2],
            shape: vec![2],
        },
    );
    model_specific.insert(
        "video_grid_thw".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![1, 2, 3, 4, 5, 6],
            shape: vec![2, 3],
        },
    );

    let preprocessed = PreprocessedEncoderInputs {
        encoder_input: ArrayD::from_shape_vec(
            IxDyn(&[4, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap()
        .into(),
        feature_token_counts: vec![2, 2],
        item_sizes: vec![(1, 1), (1, 1)],
        model_specific,
    };

    let videos = vec![
        Arc::new(VideoClip::new(
            vec![image::DynamicImage::new_rgb8(1, 1)],
            bytes::Bytes::from_static(b"a"),
            llm_multimodal::VideoSource::InlineBytes,
            "video-hash-a".to_string(),
        )),
        Arc::new(VideoClip::new(
            vec![image::DynamicImage::new_rgb8(1, 1)],
            bytes::Bytes::from_static(b"b"),
            llm_multimodal::VideoSource::InlineBytes,
            "video-hash-b".to_string(),
        )),
    ];

    let intermediate = PrecomputedMultimodalIntermediate {
        preprocessed,
        media: MediaBatch::Videos(videos),
        bindings: vec![
            PromptBinding {
                item_index: 0,
                prompt_ordinal: 0,
                structural: PlaceholderRange {
                    offset: 30,
                    length: 2,
                },
                patches: vec![PlaceholderRange {
                    offset: 30,
                    length: 2,
                }],
            },
            PromptBinding {
                item_index: 1,
                prompt_ordinal: 1,
                structural: PlaceholderRange {
                    offset: 40,
                    length: 2,
                },
                patches: vec![PlaceholderRange {
                    offset: 40,
                    length: 2,
                }],
            },
        ],
        placeholder_token_id: Some(151656),
        field_layouts: EncoderFieldLayouts::new(
            FieldLayout::flat("patches_per_video"),
            HashMap::from([
                ("patches_per_video".to_string(), FieldLayout::Batched),
                ("video_grid_thw".to_string(), FieldLayout::Batched),
            ]),
        ),
        keep_on_cpu_keys: vec![],
        encoder_input_key: None,
    };

    let assembled = assemble_tokenspeed(&intermediate, None, false).unwrap();
    assert_eq!(assembled.items.len(), 2);

    let first = &assembled.items[0];
    assert_eq!(first.modality, TokenSpeedModality::Video);
    assert_eq!(first.encoder_input.shape, vec![2, 2]);
    // bf16 is the default TokenSpeed encoder_input wire dtype (2 bytes/elem).
    assert_eq!(first.encoder_input.nbytes(), 4 * size_of::<u16>());
    assert_eq!(first.mm_placeholders, vec![(30, 2)]);
    assert_eq!(
        first.content_hash,
        hash_hex_strings(std::iter::once("video-hash-a"))
    );
    assert_eq!(
        first.model_specific_tensors["video_grid_thw"].shape,
        vec![1, 3]
    );
    assert_eq!(
        first.model_specific_tensors["patches_per_video"].shape,
        vec![1]
    );

    let second = &assembled.items[1];
    assert_eq!(second.encoder_input.shape, vec![2, 2]);
    assert_eq!(second.mm_placeholders, vec![(40, 2)]);
    assert_eq!(
        second.content_hash,
        hash_hex_strings(std::iter::once("video-hash-b"))
    );
    assert_eq!(
        second.model_specific_tensors["video_grid_thw"].shape,
        vec![1, 3]
    );
}

/// One clip, two encoder rows, sized and bound however the caller asks.
fn one_video_intermediate(
    item_size: u32,
    prompt_positions: usize,
) -> PrecomputedMultimodalIntermediate {
    let model_specific = HashMap::from([(
        "patches_per_video".to_string(),
        ModelSpecificValue::UintTensor {
            data: vec![item_size],
            shape: vec![1],
        },
    )]);

    PrecomputedMultimodalIntermediate {
        preprocessed: PreprocessedEncoderInputs {
            encoder_input: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0])
                .unwrap()
                .into(),
            feature_token_counts: vec![2],
            item_sizes: vec![(1, 1)],
            model_specific,
        },
        media: MediaBatch::Videos(vec![Arc::new(VideoClip::new(
            vec![image::DynamicImage::new_rgb8(1, 1)],
            bytes::Bytes::from_static(b"a"),
            llm_multimodal::VideoSource::InlineBytes,
            "video-hash-a".to_string(),
        ))]),
        bindings: vec![PromptBinding {
            item_index: 0,
            prompt_ordinal: 0,
            structural: PlaceholderRange {
                offset: 30,
                length: prompt_positions,
            },
            patches: vec![PlaceholderRange {
                offset: 30,
                length: prompt_positions,
            }],
        }],
        placeholder_token_id: Some(151656),
        field_layouts: EncoderFieldLayouts::new(
            FieldLayout::flat("patches_per_video"),
            HashMap::from([("patches_per_video".to_string(), FieldLayout::Batched)]),
        ),
        keep_on_cpu_keys: vec![],
        encoder_input_key: None,
    }
}

#[test]
fn a_prompt_that_reserves_more_room_than_the_media_fills_is_refused() {
    assert!(validate_precomputed_batch(&one_video_intermediate(2, 2)).is_ok());

    let error = validate_precomputed_batch(&one_video_intermediate(2, 3))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("reserves 3 prompt positions for 2"),
        "{error}"
    );
}

fn intermediate_with_features(
    features: usize,
    prompt_positions: usize,
) -> PrecomputedMultimodalIntermediate {
    let mut intermediate = one_video_intermediate(2, prompt_positions);
    intermediate.preprocessed.feature_token_counts = vec![features];
    intermediate
}

/// A model that packs several encoder outputs into each placed embedding
/// reserves fewer positions than it has features, and must still be sent.
/// The ratios here are the ones a 336px tile produces at the shuffle
/// settings a shipped vision model uses.
#[test]
fn a_prompt_that_packs_several_features_into_each_position_is_sent() {
    for (features, reserved) in [(576, 144), (288, 144), (2, 2)] {
        assert!(
            validate_precomputed_batch(&intermediate_with_features(features, reserved)).is_ok(),
            "{reserved} positions for {features} features"
        );
    }

    // Part of a feature cannot be placed anywhere, so a count that does not
    // divide is still the mismatch this guards against.
    let error = validate_precomputed_batch(&intermediate_with_features(500, 144))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("reserves 144 prompt positions for 500"),
        "{error}"
    );
}

#[test]
fn media_the_engine_would_not_read_in_full_is_refused() {
    let error = validate_precomputed_batch(&one_video_intermediate(1, 2))
        .unwrap_err()
        .to_string();
    assert!(error.contains("cover 1 of 2 encoder rows"), "{error}");
}

#[test]
fn assemble_tokenspeed_splits_audio_items_as_bfloat16_by_default() {
    let mut model_specific = HashMap::new();
    model_specific.insert(
        "row_lengths".to_string(),
        ModelSpecificValue::IntTensor {
            data: vec![2, 2],
            shape: vec![2],
        },
    );

    let preprocessed = PreprocessedEncoderInputs {
        encoder_input: ArrayD::from_shape_vec(
            IxDyn(&[4, 2]),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )
        .unwrap()
        .into(),
        feature_token_counts: vec![2, 2],
        item_sizes: vec![(2, 2), (2, 2)],
        model_specific,
    };

    let audios = vec![
        Arc::new(AudioClip::new(
            bytes::Bytes::from_static(b"a"),
            DecodedAudio {
                samples: Vec::new(),
                sample_rate: 16_000,
            },
            AudioSource::InlineBytes,
            "audio-hash-a".to_string(),
        )),
        Arc::new(AudioClip::new(
            bytes::Bytes::from_static(b"b"),
            DecodedAudio {
                samples: Vec::new(),
                sample_rate: 16_000,
            },
            AudioSource::InlineBytes,
            "audio-hash-b".to_string(),
        )),
    ];

    let intermediate = PrecomputedMultimodalIntermediate {
        preprocessed,
        media: MediaBatch::Audios(audios),
        bindings: vec![
            PromptBinding {
                item_index: 0,
                prompt_ordinal: 0,
                structural: PlaceholderRange {
                    offset: 30,
                    length: 2,
                },
                patches: vec![PlaceholderRange {
                    offset: 30,
                    length: 2,
                }],
            },
            PromptBinding {
                item_index: 1,
                prompt_ordinal: 1,
                structural: PlaceholderRange {
                    offset: 40,
                    length: 2,
                },
                patches: vec![PlaceholderRange {
                    offset: 40,
                    length: 2,
                }],
            },
        ],
        placeholder_token_id: Some(42),
        field_layouts: EncoderFieldLayouts::new(
            FieldLayout::flat("row_lengths"),
            HashMap::from([("row_lengths".to_string(), FieldLayout::Batched)]),
        ),
        keep_on_cpu_keys: vec![],
        encoder_input_key: None,
    };

    let assembled = assemble_tokenspeed(&intermediate, None, false).unwrap();
    assert_eq!(assembled.items.len(), 2);

    let first = &assembled.items[0];
    assert_eq!(first.modality, TokenSpeedModality::Audio);
    assert_eq!(first.encoder_input.dtype, "bfloat16");
    assert_eq!(first.encoder_input.shape, vec![2, 2]);
    assert_eq!(first.encoder_input.nbytes(), 4 * size_of::<u16>());
    assert_eq!(first.mm_placeholders, vec![(30, 2)]);
    assert_eq!(
        first.content_hash,
        hash_hex_strings(std::iter::once("audio-hash-a"))
    );
    assert_eq!(first.model_specific_tensors["row_lengths"].shape, vec![1]);

    let second = &assembled.items[1];
    assert_eq!(second.encoder_input.dtype, "bfloat16");
    assert_eq!(second.encoder_input.shape, vec![2, 2]);
    assert_eq!(second.mm_placeholders, vec![(40, 2)]);
    assert_eq!(
        second.content_hash,
        hash_hex_strings(std::iter::once("audio-hash-b"))
    );
}

#[test]
fn tokenspeed_epd_items_and_hashes_follow_prompt_binding_order() {
    let one_item_inputs = || PreprocessedEncoderInputs {
        encoder_input: ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0])
            .unwrap()
            .into(),
        feature_token_counts: vec![1],
        item_sizes: vec![(1, 1)],
        model_specific: HashMap::new(),
    };
    let image_batch = PrecomputedMultimodalIntermediate {
        preprocessed: one_item_inputs(),
        media: MediaBatch::Images(vec![Arc::new(ImageFrame::new(
            image::DynamicImage::new_rgb8(1, 1),
            bytes::Bytes::from_static(b"image"),
            ImageDetail::Auto,
            llm_multimodal::ImageSource::InlineBytes,
            "image-hash".to_string(),
        ))]),
        // Deliberately earlier by offset but later by explicit ordinal.
        bindings: vec![PromptBinding {
            item_index: 0,
            prompt_ordinal: 2,
            structural: PlaceholderRange {
                offset: 1,
                length: 1,
            },
            patches: vec![],
        }],
        placeholder_token_id: Some(10),
        field_layouts: EncoderFieldLayouts::default(),
        keep_on_cpu_keys: vec![],
        encoder_input_key: None,
    };
    let video_batch = PrecomputedMultimodalIntermediate {
        preprocessed: one_item_inputs(),
        media: MediaBatch::Videos(vec![Arc::new(VideoClip::new(
            vec![image::DynamicImage::new_rgb8(1, 1)],
            bytes::Bytes::from_static(b"video"),
            llm_multimodal::VideoSource::InlineBytes,
            "video-hash".to_string(),
        ))]),
        bindings: vec![PromptBinding {
            item_index: 0,
            prompt_ordinal: 1,
            structural: PlaceholderRange {
                offset: 50,
                length: 1,
            },
            patches: vec![],
        }],
        placeholder_token_id: Some(30),
        field_layouts: EncoderFieldLayouts::default(),
        keep_on_cpu_keys: vec![],
        encoder_input_key: None,
    };
    let audio_batch = PrecomputedMultimodalIntermediate {
        preprocessed: one_item_inputs(),
        media: MediaBatch::Audios(vec![Arc::new(AudioClip::new(
            bytes::Bytes::from_static(b"audio"),
            DecodedAudio {
                samples: Vec::new(),
                sample_rate: 16_000,
            },
            AudioSource::InlineBytes,
            "audio-hash".to_string(),
        ))]),
        bindings: vec![PromptBinding {
            item_index: 0,
            prompt_ordinal: 0,
            structural: PlaceholderRange {
                offset: 100,
                length: 1,
            },
            patches: vec![],
        }],
        placeholder_token_id: Some(20),
        field_layouts: EncoderFieldLayouts::default(),
        keep_on_cpu_keys: vec![],
        encoder_input_key: None,
    };
    let intermediate =
        MultimodalIntermediate::try_new(vec![image_batch, video_batch, audio_batch]).unwrap();
    let routing_hashes = encode_routing_hashes(&intermediate).unwrap();
    let assembled = assemble_tokenspeed_for_encode(&intermediate, None).unwrap();

    assert_eq!(assembled.items.len(), 3);
    assert_eq!(assembled.items[0].modality, TokenSpeedModality::Audio);
    assert_eq!(assembled.items[0].mm_placeholders, vec![(100, 1)]);
    assert_eq!(assembled.items[1].modality, TokenSpeedModality::Video);
    assert_eq!(assembled.items[1].mm_placeholders, vec![(50, 1)]);
    assert_eq!(assembled.items[2].modality, TokenSpeedModality::Image);
    assert_eq!(assembled.items[2].mm_placeholders, vec![(1, 1)]);
    assert_eq!(
        routing_hashes,
        assembled
            .items
            .iter()
            .map(|item| item.content_hash.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn qwen_audio_batched_layout_slices_items_not_mel_bins() {
    let mut model_specific = HashMap::new();
    model_specific.insert(
        "feature_attention_mask".to_string(),
        ModelSpecificValue::int_2d(vec![1, 1, 1, 1, 1, 1, 0, 0], 2, 4),
    );
    model_specific.insert(
        "audio_feature_lengths".to_string(),
        ModelSpecificValue::int_1d(vec![4, 2]),
    );
    let preprocessed = PreprocessedEncoderInputs {
        encoder_input: ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 4]),
            (0..24).map(|value| value as f32).collect(),
        )
        .unwrap()
        .into(),
        feature_token_counts: vec![1, 1],
        item_sizes: vec![(3, 4), (3, 2)],
        model_specific,
    };
    let layouts = EncoderFieldLayouts::new(
        FieldLayout::Batched,
        HashMap::from([
            ("feature_attention_mask".to_string(), FieldLayout::Batched),
            ("audio_feature_lengths".to_string(), FieldLayout::Batched),
        ]),
    );

    let second = encoder_input_for_item(&preprocessed, &layouts.encoder_input, 1).unwrap();
    assert_eq!(second.shape(), &[1, 3, 4]);
    assert_eq!(
        second.to_owned().flat_f32().into_owned(),
        (12..24).map(|v| v as f32).collect::<Vec<_>>()
    );

    let extras =
        serialize_model_specific_for_item(&preprocessed.model_specific, &layouts.model_specific, 1)
            .unwrap();
    assert_eq!(extras["feature_attention_mask"].shape, vec![1, 4]);
    assert_eq!(extras["audio_feature_lengths"].shape, vec![1]);
}

#[test]
fn vllm_layout_adapter_restores_legacy_primary_field_name() {
    let flat = EncoderFieldLayouts::new(
        FieldLayout::flat("patches_per_image"),
        HashMap::from([("image_grid_thw".to_string(), FieldLayout::Batched)]),
    );
    let (batched_keys, flat_keys) = vllm_field_layout_keys(&flat, "pixel_values");
    assert_eq!(batched_keys, vec!["image_grid_thw"]);
    assert_eq!(
        flat_keys.get("pixel_values").map(String::as_str),
        Some("patches_per_image")
    );

    let batched = EncoderFieldLayouts::default();
    let (batched_keys, flat_keys) = vllm_field_layout_keys(&batched, "pixel_values");
    assert_eq!(batched_keys, vec!["pixel_values"]);
    assert!(flat_keys.is_empty());
}

/// DeepSeek-V4.1's forward pops `patches`: the layout keys and the wire
/// key name the primary tensor by the spec's `encoder_input_key_for`.
#[test]
fn vllm_layout_adapter_names_the_primary_tensor_by_the_spec_key() {
    let layouts = EncoderFieldLayouts::new(
        FieldLayout::flat("patches_per_image"),
        HashMap::from([
            ("vit_grid".to_string(), FieldLayout::Batched),
            ("types".to_string(), FieldLayout::flat("types_per_image")),
        ]),
    );
    let (batched_keys, flat_keys) = vllm_field_layout_keys(&layouts, "patches");
    assert_eq!(batched_keys, vec!["vit_grid"]);
    assert_eq!(
        flat_keys.get("patches").map(String::as_str),
        Some("patches_per_image")
    );
    assert_eq!(
        flat_keys.get("types").map(String::as_str),
        Some("types_per_image")
    );
    assert!(!flat_keys.contains_key("pixel_values"));
}

/// The direct-ZMQ SGLang arm reaches the SGLang assembly (the function the
/// gRPC arm calls) instead of the "not supported" refusal, and what it
/// assembles crosses the wire: sent through the client to a mock scheduler,
/// the request the scheduler decodes carries one image item whose feature
/// bytes, grid tensor, offsets and image token are the assembled payload's.
#[tokio::test]
async fn the_zmq_sglang_arm_assembles_the_grpc_paths_payload_and_the_wire_carries_it() {
    use std::time::Duration;

    use engine_zmq_client::{
        codec::decode_msgpack,
        mock_engine::{connect_to_frontend, default_ready_response},
        protocol::sglang::request::{SglangRequestType, TokenizedGenerateReqInput},
        EngineId,
    };
    use smg_grpc_client::sglang_proto;

    use crate::routers::grpc::zmq_client::{EosTokenIds, ZmqEngineClient};

    let dir = tempfile::tempdir().unwrap();
    let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
    let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
    let (client, engine) = tokio::join!(
        ZmqEngineClient::connect(
            &handshake,
            &input,
            &output,
            1,
            "m".to_string(),
            EosTokenIds::default(),
            RuntimeType::Sglang,
            Duration::from_secs(10)
        ),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    let engine = engine.expect("mock scheduler");
    let client = BackendClient::Zmq(client.expect("adapter connect"));

    let intermediate = MultimodalIntermediate::try_new(vec![single_image_intermediate()]).unwrap();
    let MultimodalData::Sglang(assembled) = assemble_multimodal_data(intermediate, &client, None)
        .await
        .expect("the SGLang arm")
    else {
        panic!("the SGLang runtime over ZMQ assembles SGLang data");
    };
    assert_eq!(assembled.mm_placeholders, vec![(10, 4)]);
    assert_eq!(assembled.im_token_id, Some(151655));
    assert!(!assembled.pixel_values.is_empty());
    let pixel_values = assembled.pixel_values.clone();
    let pixel_values_shape = assembled.pixel_values_shape.clone();
    let grid = assembled.model_specific_tensors["patches_per_image"].clone();

    // The prompt the placeholders describe: the four image tokens at 10..14.
    let mut input_ids: Vec<u32> = (1..=16).collect();
    for id in &mut input_ids[10..14] {
        *id = 151655;
    }
    let request = sglang_proto::GenerateRequest {
        request_id: "mm-over-zmq".into(),
        tokenized: Some(sglang_proto::TokenizedInput {
            input_ids: input_ids.clone(),
            original_text: String::new(),
        }),
        mm_inputs: Some(assembled.into_proto()),
        ..Default::default()
    };
    let BackendClient::Zmq(zmq) = &client else {
        panic!("built as the ZMQ backend above");
    };
    let (mut scheduler_input, _scheduler_output) = engine.split();
    let (sent, received) =
        tokio::join!(zmq.generate_sglang(request), scheduler_input.recv_frames());
    let _stream = sent.expect("the request is admitted and sent");
    let frames = received.expect("the scheduler receives the request");
    assert_eq!(
        SglangRequestType::from_frame(frames[0].as_ref()),
        Some(SglangRequestType::Add)
    );
    let decoded: TokenizedGenerateReqInput =
        decode_msgpack(frames[1].as_ref()).expect("the scheduler's own struct");
    assert_eq!(decoded.input_ids, input_ids);
    let media = decoded.mm_inputs.expect("the processor output");
    assert_eq!(media.im_token_id, Some(151655));
    assert_eq!(media.mm_items.len(), 1);
    let item = &media.mm_items[0];
    assert_eq!(item.offsets, Some(vec![(10, 13)]));
    let feature = item.feature.as_ref().expect("the pixel tensor");
    assert_eq!(feature.dtype, "float32");
    assert_eq!(feature.data.as_ref(), pixel_values.as_slice());
    let widened = |shape: &[u32]| shape.iter().map(|&d| u64::from(d)).collect::<Vec<u64>>();
    assert_eq!(feature.shape, widened(&pixel_values_shape));
    let wire_grid = &item.model_specific_data["patches_per_image"];
    assert_eq!(wire_grid.dtype, grid.dtype);
    assert_eq!(wire_grid.shape, widened(&grid.shape));
    assert_eq!(wire_grid.data.as_ref(), grid.data.as_slice());
}
