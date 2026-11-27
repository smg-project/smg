use bytes::Bytes;
use llm_multimodal::{
    ImageDetail, ImageSource, ModelRegistry, Tokenizer, VideoSamplingInfo, VideoSource,
};

use super::*;
use crate::observability::metrics::test_support::global_recorder;

/// The queue wait is measured on the pool thread before the work and
/// recorded there: a task behind the one held thread records at least
/// the time from its queueing to the hold's release and at most the time
/// from just before its queueing to the drain's end, once every record is
/// in (the test's own instants bracket each wait from both sides, so a
/// wait measured from an instant taken too early fails too), a task whose
/// awaiting future was dropped while it queued records its wait all the
/// same, and the record is taken before the work runs.
#[test]
fn the_pool_queue_wait_is_recorded_on_the_pool_thread() {
    use std::{
        collections::BTreeMap,
        future::Future,
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc as std_mpsc,
        },
        task::Poll,
    };

    use tokio::sync::{mpsc, oneshot};

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .expect("a runtime");
    runtime.block_on(async {
        // Each record: who, the wait, and whether the work had run.
        let (tx, mut rx) = mpsc::unbounded_channel::<(&'static str, f64, bool)>();
        // The hold takes the pool's one thread, says so, and keeps it
        // until the test releases it once both tasks are queued behind.
        let (taken_tx, taken_rx) = oneshot::channel();
        let (release_tx, release_rx) = std_mpsc::channel::<()>();
        let hold = tokio::task::spawn_blocking(move || {
            let _ = taken_tx.send(());
            let _ = release_rx.recv();
        });
        taken_rx.await.expect("the hold took the thread");

        // Polled once, so its task is queued behind the hold, then dropped.
        let dropped_tx = tx.clone();
        let mut dropped = Box::pin(on_pool_recording(
            || (),
            move |waited| {
                let _ = dropped_tx.send(("dropped", waited, false));
            },
        ));
        // Stamped before the poll, so the task's own clock starts after it.
        let dropped_before = Instant::now();
        std::future::poll_fn(|cx| {
            let _ = dropped.as_mut().poll(cx);
            Poll::Ready(())
        })
        .await;
        // Stamped after the poll, so the task's own clock started first.
        let dropped_queued = Instant::now();
        drop(dropped);

        // Queued behind both, then awaited; its work sets a flag the
        // record closure reads, so the order of the two is recorded.
        let worked = Arc::new(AtomicBool::new(false));
        let worked_by_work = Arc::clone(&worked);
        let worked_by_record = Arc::clone(&worked);
        let awaited_tx = tx.clone();
        let mut awaited = Box::pin(on_pool_recording(
            move || {
                worked_by_work.store(true, Ordering::SeqCst);
                7
            },
            move |waited| {
                let _ = awaited_tx.send((
                    "awaited",
                    waited,
                    worked_by_record.load(Ordering::SeqCst),
                ));
            },
        ));
        let awaited_before = Instant::now();
        std::future::poll_fn(|cx| {
            let _ = awaited.as_mut().poll(cx);
            Poll::Ready(())
        })
        .await;
        let awaited_queued = Instant::now();

        // Stamped before the release, so no record can precede it.
        let released = Instant::now();
        release_tx.send(()).expect("the hold is waiting");
        let value = awaited.await.expect("the work ran");
        assert_eq!(value, 7);
        hold.await.expect("the hold ran");
        drop(tx);

        let mut seen = BTreeMap::new();
        while let Some((who, waited, worked_first)) = rx.recv().await {
            seen.insert(who, (waited, worked_first));
        }
        // Stamped once every sender is gone: each record was sent before
        // its sender dropped, whatever order the pool ran the tasks in.
        let done = Instant::now();
        for (who, before, queued) in [
            ("dropped", dropped_before, dropped_queued),
            ("awaited", awaited_before, awaited_queued),
        ] {
            let (waited, _) = seen
                .get(who)
                .unwrap_or_else(|| panic!("{who}: no wait recorded"));
            let held = released.duration_since(queued).as_secs_f64();
            assert!(
                *waited >= held,
                "{who}: waited {waited} s, queued {held} s before the release"
            );
            let bracket = done.duration_since(before).as_secs_f64();
            assert!(
                *waited <= bracket,
                "{who}: waited {waited} s, but {bracket} s passed from before its queueing to the drain's end"
            );
        }
        let (_, worked_first) = seen["awaited"];
        assert!(!worked_first, "the wait was recorded after the work ran");
    });
}

/// The production wiring: `on_pool` records the wait as the `pool_queue`
/// stage of the model's histogram. The record happens on the pool thread,
/// so the test reads the binary's one process-wide recorder
/// (`test_support::global_recorder`), which the local recorders of the
/// other tests take precedence over on their threads.
#[test]
fn the_pool_queue_stage_is_recorded_under_the_models_label() {
    let handle = global_recorder();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime");
    let value = runtime
        .block_on(on_pool("the-pool-queue-model", || 11))
        .expect("the work ran");
    assert_eq!(value, 11);
    let rendered = handle.render();
    let counted = rendered.lines().any(|line| {
        line.starts_with("smg_mm_processing_seconds_count{")
            && line.contains("model=\"the-pool-queue-model\"")
            && line.contains("stage=\"pool_queue\"")
            && line.rsplit(' ').next().and_then(|n| n.parse::<u64>().ok()) >= Some(1)
    });
    assert!(
        counted,
        "no pool_queue count under the model in:\n{rendered}"
    );
}

/// A processor whose RGB path is refused its output; its frame path
/// must not run.
struct RefusedRgb;

impl VisionPreProcessor for RefusedRgb {
    fn default_mean(&self) -> [f64; 3] {
        [0.0; 3]
    }
    fn default_std(&self) -> [f64; 3] {
        [1.0; 3]
    }
    fn calculate_num_tokens(&self, _: u32, _: u32) -> Result<usize, TransformError> {
        Ok(1)
    }
    fn model_name(&self) -> &'static str {
        "refused"
    }
    fn preprocess(
        &self,
        _: &[&image::DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError("images".to_string()))
    }
    fn preprocess_video(
        &self,
        _: &[image::DynamicImage],
        _: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        panic!("the frame path ran")
    }
    fn preprocess_video_rgb(
        &self,
        _: &[llm_multimodal::RgbFrameRef<'_>],
        _: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ReserveRefused("no room".to_string()))
    }
}

#[test]
fn a_refused_rgb_clip_is_answered_without_the_frame_path() {
    let frame = llm_multimodal::types::DecodedRgbFrame {
        width: 1,
        height: 1,
        offset: 0,
        len: 3,
    };
    let video = VideoClip::new_rgb(
        llm_multimodal::types::DecodedRgbVideo::new(Bytes::from_static(&[0; 3]), vec![frame]),
        Bytes::new(),
        VideoSource::InlineBytes,
        "video-hash".to_string(),
    );

    let error = preprocess_clip(&RefusedRgb, &video).expect_err("refused");

    assert_eq!(
        error.to_string(),
        "Video preprocessing failed: Shape error: no room"
    );
}

#[test]
fn decoded_video_sample_fps_overrides_processor_default() {
    let video = VideoClip::new_with_sample_fps(
        Vec::new(),
        Bytes::new(),
        VideoSource::InlineBytes,
        "video-hash".to_string(),
        0.8,
    );

    let clip = video_clip_options(&video);

    assert_eq!(clip.sample_fps, 0.8);
    assert!(clip.max_long_side_pixel.is_none());
}

#[test]
fn decoded_video_tier_reaches_the_processor() {
    let video = VideoClip::new_with_sample_fps(
        Vec::new(),
        Bytes::new(),
        VideoSource::InlineBytes,
        "video-hash".to_string(),
        2.0,
    )
    .with_max_long_side_pixel(Some(1008));

    let clip = video_clip_options(&video);

    assert_eq!(clip.max_long_side_pixel, Some(1008));
}

#[test]
fn media_item_infos_carry_video_sampling_per_clip() {
    let sampling = VideoSamplingInfo {
        source_fps: 30.0,
        frame_indices: vec![0, 15, 30],
    };
    let sampled = VideoClip::new(
        Vec::new(),
        Bytes::new(),
        VideoSource::InlineBytes,
        "sampled".to_string(),
    )
    .with_sampling(Some(sampling.clone()));
    let unsampled = VideoClip::new(
        Vec::new(),
        Bytes::new(),
        VideoSource::InlineBytes,
        "unsampled".to_string(),
    );

    let infos = media_item_infos(&MediaBatch::Videos(vec![
        Arc::new(sampled),
        Arc::new(unsampled),
    ]));

    assert_eq!(
        infos,
        vec![
            MediaItemInfo {
                video_sampling: Some(sampling)
            },
            MediaItemInfo::default()
        ]
    );
}

#[test]
fn media_item_infos_default_for_every_image() {
    let image = || {
        Arc::new(ImageFrame::new(
            image::DynamicImage::new_rgb8(1, 1),
            Bytes::new(),
            ImageDetail::Auto,
            ImageSource::InlineBytes,
            "hash".to_string(),
        ))
    };

    let infos = media_item_infos(&MediaBatch::Images(vec![image(), image(), image()]));

    assert_eq!(infos, vec![MediaItemInfo::default(); 3]);
}

#[test]
fn preprocessor_config_for_video_is_the_video_config_when_shipped() {
    let image_config = PreProcessorConfig {
        temporal_patch_size: Some(2),
        ..Default::default()
    };
    let video_config = PreProcessorConfig {
        temporal_patch_size: Some(4),
        ..Default::default()
    };
    let with_video = MultimodalModelConfig::new(
        serde_json::json!({}),
        image_config.clone(),
        Some(video_config),
    );
    let without_video = MultimodalModelConfig::new(serde_json::json!({}), image_config, None);

    let temporal = |config: &MultimodalModelConfig, modality| {
        preprocessor_config_for(config, modality).temporal_patch_size
    };
    assert_eq!(temporal(&with_video, Modality::Video), Some(4));
    assert_eq!(temporal(&with_video, Modality::Image), Some(2));
    assert_eq!(temporal(&with_video, Modality::Audio), Some(2));
    assert_eq!(temporal(&without_video, Modality::Video), Some(2));
}

/// The samplers read the config the video processor is built from: the
/// checkpoint's video config when it ships one, else its shared one, so a
/// family whose reference reads its rate and frame budget from that config
/// samples a clip the way its processor groups it.
#[test]
fn video_sampling_reads_the_processors_config() {
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
    let model = serde_json::json!({
        "model_type": "muse_spark", "image_token_id": 1, "video_token_id": 2
    });
    let registry = ModelRegistry::new();
    let spec = registry
        .lookup(&ModelMetadata {
            model_id: "muse-spark",
            tokenizer: &NoTokenizer,
            config: &model,
        })
        .expect("a family whose sampler reads its processor config");
    let config = |fps: f64, num_frames: usize| {
        PreProcessorConfig::from_value(serde_json::json!({"fps": fps, "num_frames": num_frames}))
            .unwrap()
    };
    let shared_only = MultimodalModelConfig::new(serde_json::json!({}), config(1.0, 8), None);
    let with_video = MultimodalModelConfig::new(
        serde_json::json!({}),
        config(1.0, 8),
        Some(config(2.0, 600)),
    );

    assert_eq!(
        video_sampling(spec, &shared_only, None),
        (
            Some(1.0),
            FrameSampling::TemporalGroups {
                frame_factor: 2,
                max_frames: 8,
            }
        )
    );
    assert_eq!(
        video_sampling(spec, &with_video, None),
        (
            Some(2.0),
            FrameSampling::TemporalGroups {
                frame_factor: 2,
                max_frames: 600,
            }
        )
    );
}

#[test]
fn test_explicit_feature_ranges_keep_non_uniform_strides() {
    let replacements = vec![
        PromptReplacement::sequence(Modality::Video, "<video>", vec![7; 48]).with_feature_ranges(
            vec![
                PlaceholderRange {
                    offset: 5,
                    length: 4,
                },
                PlaceholderRange {
                    offset: 19,
                    length: 4,
                },
                PlaceholderRange {
                    offset: 40,
                    length: 4,
                },
            ],
        ),
    ];
    let expansion = ModalityExpansion {
        modality: Modality::Video,
        search_token_id: Some(100),
        placeholder_token_id: Some(7),
        replacements: &replacements,
    };

    let result = expand_tokens_for_modalities(&[1, 2, 100, 3], &[expansion]).unwrap();

    assert_eq!(result.token_ids.len(), 3 + 48);
    assert_eq!(result.bindings[0][0].structural.offset, 2);
    assert_eq!(result.bindings[0][0].structural.length, 48);
    assert_eq!(
        result.bindings[0][0].patches,
        vec![
            PlaceholderRange {
                offset: 7,
                length: 4,
            },
            PlaceholderRange {
                offset: 21,
                length: 4,
            },
            PlaceholderRange {
                offset: 42,
                length: 4,
            },
        ]
    );
}

#[test]
fn test_expand_tokens_basic() {
    let token_ids = vec![1, 2, 100, 3, 4]; // 100 is the placeholder
    let replacements = vec![PromptReplacement {
        modality: Modality::Image,
        placeholder_token: "<image>".to_string(),
        tokens: vec![50, 50, 50, 50], // Expand to 4 tokens
        feature_ranges: None,
        structural_prefix: 0,
    }];

    let expansion = ModalityExpansion {
        modality: Modality::Image,
        search_token_id: Some(100),
        placeholder_token_id: None,
        replacements: &replacements,
    };
    let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

    assert_eq!(result.token_ids, vec![1, 2, 50, 50, 50, 50, 3, 4]);
    assert_eq!(result.bindings[0].len(), 1);
    assert_eq!(result.bindings[0][0].structural.offset, 2);
    assert_eq!(result.bindings[0][0].structural.length, 4);
    assert!(result.bindings[0][0].patches.is_empty());
}

#[test]
fn test_expand_tokens_structural_prefix_folds_leading_marker() {
    // Qwen video: the chat template emits <VS><video_pad><VE> (777, 100, 778).
    // The replacement declares structural_prefix=1, so the reported range must
    // start on the leading <VS> (offset 2, not 3) and grow by one, letting a
    // backend that scans the range for <VS> (vLLM's per-frame video mrope) find
    // it. The token stream is unchanged — the marker is not re-emitted.
    let token_ids = vec![1, 2, 777, 100, 778, 3]; // 100 is the placeholder
    let replacements = vec![PromptReplacement {
        modality: Modality::Video,
        placeholder_token: "<|video_pad|>".to_string(),
        tokens: vec![50, 50, 50], // expands to 3 video tokens
        feature_ranges: None,
        structural_prefix: 1,
    }];

    let expansion = ModalityExpansion {
        modality: Modality::Video,
        search_token_id: Some(100),
        placeholder_token_id: None,
        replacements: &replacements,
    };
    let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

    assert_eq!(result.token_ids, vec![1, 2, 777, 50, 50, 50, 778, 3]);
    assert_eq!(result.bindings[0].len(), 1);
    // Range starts on <VS> (index 2) and covers it + the 3 video tokens.
    assert_eq!(result.bindings[0][0].structural.offset, 2);
    assert_eq!(result.bindings[0][0].structural.length, 4);
}

#[test]
fn test_expand_tokens_no_placeholder() {
    let token_ids = vec![1, 2, 3];
    let replacements = Vec::new();
    let expansion = ModalityExpansion {
        modality: Modality::Image,
        search_token_id: None,
        placeholder_token_id: None,
        replacements: &replacements,
    };
    let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

    assert_eq!(result.token_ids, vec![1, 2, 3]);
    assert!(result.bindings[0].is_empty());
}

#[test]
fn test_expand_tokens_multiple_images() {
    let token_ids = vec![1, 100, 2, 100, 3]; // Two placeholder tokens
    let replacements = vec![
        PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![50, 50], // 2 tokens for first image
            feature_ranges: None,
            structural_prefix: 0,
        },
        PromptReplacement {
            modality: Modality::Image,
            placeholder_token: "<image>".to_string(),
            tokens: vec![60, 60, 60], // 3 tokens for second image
            feature_ranges: None,
            structural_prefix: 0,
        },
    ];

    let expansion = ModalityExpansion {
        modality: Modality::Image,
        search_token_id: Some(100),
        placeholder_token_id: None,
        replacements: &replacements,
    };
    let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

    assert_eq!(result.token_ids, vec![1, 50, 50, 2, 60, 60, 60, 3]);
    assert_eq!(result.bindings[0].len(), 2);
    assert_eq!(result.bindings[0][0].structural.offset, 1);
    assert_eq!(result.bindings[0][0].structural.length, 2);
    assert_eq!(result.bindings[0][1].structural.offset, 4);
    assert_eq!(result.bindings[0][1].structural.length, 3);
}

#[test]
fn image_prompt_tokens_are_the_image_replacements_alone() {
    let images = PromptReplacement::repeated_per_item(Modality::Image, "<image>", 7, &[324, 576]);
    let video = PromptReplacement::repeated_per_item(Modality::Video, "<video>", 8, &[48]);
    assert_eq!(
        image_prompt_tokens([
            (Modality::Video, video.as_slice()),
            (Modality::Image, images.as_slice()),
        ]),
        Some(900)
    );
    assert_eq!(
        image_prompt_tokens([(Modality::Video, video.as_slice())]),
        None,
        "a video's placeholders are not image tokens"
    );
    assert_eq!(image_prompt_tokens(std::iter::empty()), None);
}

#[test]
fn test_expand_tokens_patch_offsets_with_structural() {
    // Simulates Llama-4: placeholder expands to structural + patch tokens
    // 88=image_start, 92=patch(im_token_id), 93=separator, 89=image_end
    let token_ids = vec![1, 100, 2]; // 100 is the placeholder
    let replacements = vec![PromptReplacement {
        modality: Modality::Image,
        placeholder_token: "<image>".to_string(),
        tokens: vec![88, 92, 92, 92, 93, 92, 92, 92, 89], // start + patches + sep + patches + end
        feature_ranges: None,
        structural_prefix: 0,
    }];

    let expansion = ModalityExpansion {
        modality: Modality::Image,
        search_token_id: Some(100),
        placeholder_token_id: Some(92),
        replacements: &replacements,
    };
    let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

    // Full structural range
    assert_eq!(result.bindings[0].len(), 1);
    assert_eq!(result.bindings[0][0].structural.offset, 1);
    assert_eq!(result.bindings[0][0].structural.length, 9);

    // Patch-only offsets: two runs of token 92
    let patch = &result.bindings[0][0].patches;
    assert_eq!(patch.len(), 2);
    assert_eq!((patch[0].offset, patch[0].length), (2, 3));
    assert_eq!((patch[1].offset, patch[1].length), (6, 3));
}

#[test]
fn test_explicit_feature_range_excludes_same_id_structural_header() {
    let replacements = vec![PromptReplacement::sequence(
        Modality::Image,
        "<|content_image|>",
        vec![200005, 200005, 200005, 200005],
    )
    .with_feature_span(1, 3)];
    let expansion = ModalityExpansion {
        modality: Modality::Image,
        search_token_id: Some(200005),
        placeholder_token_id: Some(200005),
        replacements: &replacements,
    };

    let result = expand_tokens_for_modalities(&[1, 200005, 2], &[expansion]).unwrap();

    assert_eq!(result.token_ids, vec![1, 200005, 200005, 200005, 200005, 2]);
    assert_eq!(result.bindings[0][0].structural.offset, 1);
    assert_eq!(result.bindings[0][0].structural.length, 4);
    assert_eq!(result.bindings[0][0].patches.len(), 1);
    assert_eq!(result.bindings[0][0].patches[0].offset, 2);
    assert_eq!(result.bindings[0][0].patches[0].length, 3);
}

#[test]
fn test_explicit_feature_range_must_fit_replacement() {
    let replacements = vec![
        PromptReplacement::sequence(Modality::Image, "<image>", vec![50, 50])
            .with_feature_span(1, 2),
    ];
    let expansion = ModalityExpansion {
        modality: Modality::Image,
        search_token_id: Some(100),
        placeholder_token_id: Some(50),
        replacements: &replacements,
    };

    let error = expand_tokens_for_modalities(&[100], &[expansion]).unwrap_err();
    assert!(format!("{error:#}").contains("exceeds replacement length"));
}

#[test]
fn test_expand_tokens_preserves_template_owned_audio_end() {
    // The template owns the audio end marker. Expansion preserves the
    // anchor, adds feature tokens, and must not inject another end marker.
    let audio_anchor: u32 = 100;
    let audio_placeholder: u32 = 101;
    let audio_end: u32 = 102;
    let message_end: u32 = 103;
    let token_ids = vec![1, audio_anchor, audio_end, message_end];
    let replacements = vec![PromptReplacement {
        modality: Modality::Audio,
        placeholder_token: "<audio>".to_string(),
        tokens: vec![
            audio_anchor as i32,
            audio_placeholder as i32,
            audio_placeholder as i32,
        ],
        feature_ranges: None,
        structural_prefix: 0,
    }];

    let expansion = ModalityExpansion {
        modality: Modality::Audio,
        search_token_id: Some(audio_anchor),
        placeholder_token_id: Some(audio_placeholder),
        replacements: &replacements,
    };
    let result = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap();

    assert_eq!(
        result.token_ids,
        vec![
            1,
            audio_anchor,
            audio_placeholder,
            audio_placeholder,
            audio_end,
            message_end
        ]
    );
    assert_eq!(
        result
            .token_ids
            .iter()
            .filter(|&&token| token == audio_end)
            .count(),
        1
    );
    assert_eq!(result.bindings[0][0].patches.len(), 1);
    assert_eq!(result.bindings[0][0].patches[0].offset, 2);
    assert_eq!(result.bindings[0][0].patches[0].length, 2);
}

#[test]
fn test_expand_tokens_mixed_image_audio_offsets_follow_final_prompt() {
    let audio_anchor: u32 = 100;
    let audio_placeholder: u32 = 101;
    let audio_end: u32 = 102;
    let image_anchor: u32 = 200;
    let image_placeholder: u32 = 201;
    let message_end: u32 = 103;
    let token_ids = vec![
        1,
        audio_anchor,
        audio_end,
        message_end,
        2,
        image_anchor,
        message_end,
    ];
    let audio_replacements = vec![PromptReplacement {
        modality: Modality::Audio,
        placeholder_token: "<audio>".to_string(),
        tokens: vec![
            audio_anchor as i32,
            audio_placeholder as i32,
            audio_placeholder as i32,
        ],
        feature_ranges: None,
        structural_prefix: 0,
    }];
    let image_replacements = vec![PromptReplacement {
        modality: Modality::Image,
        placeholder_token: "<image>".to_string(),
        tokens: vec![
            image_anchor as i32,
            image_placeholder as i32,
            image_placeholder as i32,
            image_placeholder as i32,
        ],
        feature_ranges: None,
        structural_prefix: 0,
    }];
    let expansions = vec![
        ModalityExpansion {
            modality: Modality::Image,
            search_token_id: Some(image_anchor),
            placeholder_token_id: Some(image_placeholder),
            replacements: &image_replacements,
        },
        ModalityExpansion {
            modality: Modality::Audio,
            search_token_id: Some(audio_anchor),
            placeholder_token_id: Some(audio_placeholder),
            replacements: &audio_replacements,
        },
    ];

    let result = expand_tokens_for_modalities(&token_ids, &expansions).unwrap();

    assert_eq!(
        result.token_ids,
        vec![
            1,
            audio_anchor,
            audio_placeholder,
            audio_placeholder,
            audio_end,
            message_end,
            2,
            image_anchor,
            image_placeholder,
            image_placeholder,
            image_placeholder,
            message_end,
        ]
    );
    assert_eq!(result.bindings[0][0].structural.offset, 7);
    assert_eq!(result.bindings[0][0].structural.length, 4);
    assert_eq!(result.bindings[0][0].patches[0].offset, 8);
    assert_eq!(result.bindings[0][0].patches[0].length, 3);
    assert_eq!(result.bindings[0][0].prompt_ordinal, 1);
    assert_eq!(result.bindings[1][0].structural.offset, 1);
    assert_eq!(result.bindings[1][0].structural.length, 3);
    assert_eq!(result.bindings[1][0].patches[0].offset, 2);
    assert_eq!(result.bindings[1][0].patches[0].length, 2);
    assert_eq!(result.bindings[1][0].prompt_ordinal, 0);
}

#[test]
fn test_expand_tokens_three_modalities_follow_final_prompt() {
    let image_replacements = vec![PromptReplacement {
        modality: Modality::Image,
        placeholder_token: "<image>".to_string(),
        tokens: vec![110, 101, 111],
        feature_ranges: None,
        structural_prefix: 0,
    }];
    let video_replacements = vec![PromptReplacement {
        modality: Modality::Video,
        placeholder_token: "<video>".to_string(),
        tokens: vec![210, 201, 201, 211],
        feature_ranges: None,
        structural_prefix: 0,
    }];
    let audio_replacements = vec![PromptReplacement {
        modality: Modality::Audio,
        placeholder_token: "<audio>".to_string(),
        tokens: vec![310, 301, 301],
        feature_ranges: None,
        structural_prefix: 0,
    }];
    let expansions = vec![
        ModalityExpansion {
            modality: Modality::Image,
            search_token_id: Some(100),
            placeholder_token_id: Some(101),
            replacements: &image_replacements,
        },
        ModalityExpansion {
            modality: Modality::Video,
            search_token_id: Some(200),
            placeholder_token_id: Some(201),
            replacements: &video_replacements,
        },
        ModalityExpansion {
            modality: Modality::Audio,
            search_token_id: Some(300),
            placeholder_token_id: Some(301),
            replacements: &audio_replacements,
        },
    ];

    let result = expand_tokens_for_modalities(&[300, 9, 100, 8, 200], &expansions).unwrap();

    assert_eq!(
        result.token_ids,
        vec![310, 301, 301, 9, 110, 101, 111, 8, 210, 201, 201, 211]
    );
    let image = &result.bindings[0][0];
    let video = &result.bindings[1][0];
    let audio = &result.bindings[2][0];
    assert_eq!(image.prompt_ordinal, 1);
    assert_eq!((image.structural.offset, image.patches[0].offset), (4, 5));
    assert_eq!(video.prompt_ordinal, 2);
    assert_eq!((video.structural.offset, video.patches[0].offset), (8, 9));
    assert_eq!(audio.prompt_ordinal, 0);
    assert_eq!((audio.structural.offset, audio.patches[0].offset), (0, 1));
}

#[test]
fn test_expand_tokens_more_placeholders_than_replacements() {
    // Two prompt anchors for one replacement must fail instead of silently
    // binding user/template text to the wrong media item.
    let token_ids = vec![1, 100, 2, 100, 3];
    let replacements = vec![PromptReplacement {
        modality: Modality::Image,
        placeholder_token: "<image>".to_string(),
        tokens: vec![50, 50],
        feature_ranges: None,
        structural_prefix: 0,
    }];

    let expansion = ModalityExpansion {
        modality: Modality::Image,
        search_token_id: Some(100),
        placeholder_token_id: None,
        replacements: &replacements,
    };
    let error = expand_tokens_for_modalities(&token_ids, &[expansion]).unwrap_err();
    assert!(error.to_string().contains("Extra prompt anchor"));
}

#[test]
fn test_expand_tokens_rejects_missing_anchor() {
    let replacements = vec![PromptReplacement {
        modality: Modality::Image,
        placeholder_token: "<image>".to_string(),
        tokens: vec![50, 50],
        feature_ranges: None,
        structural_prefix: 0,
    }];

    let expansion = ModalityExpansion {
        modality: Modality::Image,
        search_token_id: Some(100),
        placeholder_token_id: None,
        replacements: &replacements,
    };
    let error = expand_tokens_for_modalities(&[1, 2, 3], &[expansion]).unwrap_err();
    assert!(error.to_string().contains("Missing prompt anchors"));
}

#[test]
fn test_expand_tokens_rejects_shared_anchor_id() {
    let image_replacements = vec![PromptReplacement {
        modality: Modality::Image,
        placeholder_token: "<image>".to_string(),
        tokens: vec![50],
        feature_ranges: None,
        structural_prefix: 0,
    }];
    let audio_replacements = vec![PromptReplacement {
        modality: Modality::Audio,
        placeholder_token: "<audio>".to_string(),
        tokens: vec![60],
        feature_ranges: None,
        structural_prefix: 0,
    }];
    let expansions = [
        ModalityExpansion {
            modality: Modality::Image,
            search_token_id: Some(100),
            placeholder_token_id: Some(50),
            replacements: &image_replacements,
        },
        ModalityExpansion {
            modality: Modality::Audio,
            search_token_id: Some(100),
            placeholder_token_id: Some(60),
            replacements: &audio_replacements,
        },
    ];

    let error = expand_tokens_for_modalities(&[100, 100], &expansions).unwrap_err();
    assert!(error.to_string().contains("anchors must be unique"));
}

#[test]
fn test_expand_tokens_rejects_negative_replacement_token() {
    let replacements = vec![PromptReplacement {
        modality: Modality::Image,
        placeholder_token: "<image>".to_string(),
        tokens: vec![50, -1],
        feature_ranges: None,
        structural_prefix: 0,
    }];

    let expansion = ModalityExpansion {
        modality: Modality::Image,
        search_token_id: Some(100),
        placeholder_token_id: Some(50),
        replacements: &replacements,
    };
    let error = expand_tokens_for_modalities(&[100], &[expansion]).unwrap_err();
    assert!(error.to_string().contains("Invalid negative token ID"));
}
