use std::sync::Mutex;

use llm_multimodal::{
    ImageSource, Tokenizer, TransformError, VisionPreProcessor, VisionProcessorRegistry,
};
use ndarray::Array2;

use super::*;
use crate::routers::grpc::multimodal::{
    config::MultimodalConfigRegistry, pixel_cache::PixelCache, settings::MultimodalSettings,
};

#[derive(Default)]
struct Calls {
    batches: Mutex<Vec<Vec<u32>>>,
}

struct Processor {
    calls: Arc<Calls>,
    independent: bool,
}

impl VisionPreProcessor for Processor {
    fn supports_per_image_preprocessing(&self) -> bool {
        self.independent
    }
    fn default_mean(&self) -> [f64; 3] {
        [0.0; 3]
    }
    fn default_std(&self) -> [f64; 3] {
        [1.0; 3]
    }
    fn calculate_num_tokens(&self, _: u32, _: u32, _: &PreProcessorConfig) -> usize {
        1
    }
    fn model_name(&self) -> &'static str {
        "cache-test"
    }
    fn preprocess(
        &self,
        images: &[image::DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let ids: Vec<_> = images.iter().map(image::DynamicImage::width).collect();
        self.calls.batches.lock().unwrap().push(ids.clone());
        Ok(PreprocessedEncoderInputs::new(
            Array2::from_shape_vec(
                (ids.len(), 1),
                ids.iter()
                    .map(|&id| id as f32 * config.rescale_factor.unwrap_or(1.0) as f32)
                    .collect(),
            )
            .unwrap(),
            ids.iter().map(|&id| id as usize).collect(),
            ids.iter().map(|&id| (id, 1)).collect(),
        )
        .with_extra(
            "patches_per_image",
            llm_multimodal::ModelSpecificValue::int_1d(vec![
                if config.extra.contains_key("invalid_layout") {
                    0
                } else {
                    1
                };
                ids.len()
            ]),
        )
        .with_extra(
            "image_grid_thw",
            llm_multimodal::ModelSpecificValue::int_2d(vec![1; ids.len() * 3], ids.len(), 3),
        ))
    }
}

struct NoTokenizer;
impl Tokenizer for NoTokenizer {
    fn token_to_id(&self, _: &str) -> Option<u32> {
        None
    }
    fn id_to_token(&self, _: u32) -> Option<String> {
        None
    }
    fn encode_text(&self, _: &str) -> Option<Vec<u32>> {
        None
    }
}

fn setup(independent: bool) -> (MultimodalComponents, Arc<Calls>, MultimodalModelConfig) {
    let calls = Arc::new(Calls::default());
    let mut registry = VisionProcessorRegistry::new();
    registry.register(
        "qwen2-vl",
        Box::new(Processor {
            calls: calls.clone(),
            independent,
        }),
    );
    let mut components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .unwrap();
    components.vision_processor_registry = Arc::new(registry);
    components.pixel_cache = Some(Arc::new(PixelCache::new(1024 * 1024)));
    (
        components,
        calls,
        MultimodalModelConfig {
            config: serde_json::json!({"model_type": "qwen2_vl"}),
            preprocessor_config: PreProcessorConfig::default(),
            video_preprocessor_config: None,
        },
    )
}

async fn run(
    components: &MultimodalComponents,
    config: &MultimodalModelConfig,
    ids: &[u32],
    tokenizer_id: &str,
) -> PreprocessedEncoderInputs {
    let media = MediaBatch::Images(
        ids.iter()
            .map(|&id| {
                Arc::new(ImageFrame::new(
                    image::DynamicImage::new_rgb8(id, 1),
                    Default::default(),
                    Default::default(),
                    ImageSource::InlineBytes,
                    id.to_string(),
                ))
            })
            .collect(),
    );
    let metadata = ModelMetadata {
        model_id: "qwen2-vl",
        tokenizer: &NoTokenizer,
        config: &config.config,
    };
    let spec = components.model_registry.lookup(&metadata).unwrap();
    preprocess_modality(
        &media,
        components,
        metadata.model_id,
        Some("qwen2_vl"),
        spec,
        tokenizer_id,
        config,
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn mixed_hits_and_batched_misses_preserve_order() {
    let (components, calls, config) = setup(true);
    run(&components, &config, &[3], "tok").await;
    calls.batches.lock().unwrap().clear();
    let result = run(&components, &config, &[1, 3, 2, 3], "tok").await;
    assert_eq!(*calls.batches.lock().unwrap(), vec![vec![1, 2]]);
    assert_eq!(result.feature_token_counts, [1, 3, 2, 3]);
    assert_eq!(result.item_sizes, [(1, 1), (3, 1), (2, 1), (3, 1)]);
    assert_eq!(
        result.encoder_input.flat_f32().into_owned(),
        [1.0, 3.0, 2.0, 3.0]
    );
    calls.batches.lock().unwrap().clear();
    run(&components, &config, &[3, 2, 1], "tok").await;
    assert!(calls.batches.lock().unwrap().is_empty());
}

#[tokio::test]
async fn repeated_cold_images_keep_every_occurrence() {
    for budget in [0, 1024 * 1024] {
        let (mut components, calls, config) = setup(true);
        components.pixel_cache = Some(Arc::new(PixelCache::new(budget)));
        let result = run(&components, &config, &[1, 2, 1, 2, 1], "tok").await;
        assert_eq!(*calls.batches.lock().unwrap(), vec![vec![1, 2]]);
        assert_eq!(
            result.encoder_input.flat_f32().into_owned(),
            [1.0, 2.0, 1.0, 2.0, 1.0]
        );
        assert_eq!(result.feature_token_counts, [1, 2, 1, 2, 1]);
        assert_eq!(result.item_sizes, [(1, 1), (2, 1), (1, 1), (2, 1), (1, 1)]);
        calls.batches.lock().unwrap().clear();
        run(&components, &config, &[2; 31], "tok").await;
        let expected = if budget == 0 { vec![vec![2]] } else { vec![] };
        assert_eq!(*calls.batches.lock().unwrap(), expected);
    }
}

#[tokio::test]
async fn cache_separates_preprocessing_model_and_tokenizer_parameters() {
    let (components, calls, mut config) = setup(true);
    run(&components, &config, &[3, 4], "tok").await;
    config.preprocessor_config.rescale_factor = Some(2.0);
    calls.batches.lock().unwrap().clear();
    let result = run(&components, &config, &[3, 4], "tok").await;
    assert_eq!(result.encoder_input.flat_f32().into_owned(), [6.0, 8.0]);
    assert_eq!(calls.batches.lock().unwrap().len(), 1);
    config.config["revision"] = serde_json::json!(2);
    calls.batches.lock().unwrap().clear();
    run(&components, &config, &[3, 4], "tok").await;
    assert_eq!(calls.batches.lock().unwrap().len(), 1);
    calls.batches.lock().unwrap().clear();
    run(&components, &config, &[3, 4], "other-tokenizer").await;
    assert_eq!(calls.batches.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn large_batches_cache_items_but_unconfirmed_processors_and_disabled_cache_bypass() {
    for (independent, count, cached) in [
        (true, 31, true),
        (true, 32, true),
        (true, 70, true),
        (false, 2, true),
        (true, 2, false),
    ] {
        let (mut components, calls, config) = setup(independent);
        if !cached {
            components.pixel_cache = None;
        }
        let ids: Vec<_> = (1..=count).collect();
        run(&components, &config, &ids, "tok").await;
        assert_eq!(*calls.batches.lock().unwrap(), vec![ids.clone()]);
        calls.batches.lock().unwrap().clear();
        let out = run(&components, &config, &ids, "tok").await;
        assert_eq!(
            out.feature_token_counts,
            ids.iter().map(|&id| id as usize).collect::<Vec<_>>()
        );
        if cached && independent {
            assert!(calls.batches.lock().unwrap().is_empty());
        } else {
            assert_eq!(*calls.batches.lock().unwrap(), vec![ids]);
        }
    }
}

/// Hits and misses per model, as `(hits, misses)`; `None` until both series
/// are rendered.
fn lookups(
    handle: &metrics_exporter_prometheus::PrometheusHandle,
    model: &str,
) -> Option<(f64, f64)> {
    let body = handle.render();
    let sample = |family: &str| -> Option<f64> {
        let prefix = format!("{family}{{model=\"{model}\"}} ");
        body.lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .and_then(|value| value.trim().parse().ok())
    };
    Some((
        sample("smg_mm_pixel_cache_hits_total")?,
        sample("smg_mm_pixel_cache_misses_total")?,
    ))
}

/// Every lookup is a hit or a miss on `/metrics`, by model, with both series
/// present from the first lookup. An image repeated within one request is
/// deduplicated before the cache and is neither.
#[test]
fn lookups_are_counted_as_hits_and_misses_by_model() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let (components, _calls, config) = setup(true);
            run(&components, &config, &[1, 2], "tok").await;
            assert_eq!(lookups(&handle, "qwen2-vl"), Some((0.0, 2.0)));

            run(&components, &config, &[1, 2, 1], "tok").await;
            assert_eq!(lookups(&handle, "qwen2-vl"), Some((2.0, 2.0)));

            run(&components, &config, &[3], "tok").await;
            assert_eq!(lookups(&handle, "qwen2-vl"), Some((2.0, 3.0)));
        });
    });
}

#[tokio::test]
async fn large_mixed_batch_reuses_hits_and_deduplicates_misses() {
    let (components, calls, config) = setup(true);
    run(&components, &config, &[3, 5], "tok").await;
    calls.batches.lock().unwrap().clear();
    let ids: Vec<_> = [5, 2, 3, 4, 2].into_iter().cycle().take(70).collect();
    let output = run(&components, &config, &ids, "tok").await;
    assert_eq!(*calls.batches.lock().unwrap(), vec![vec![2, 4]]);
    assert_eq!(
        output.encoder_input_flat().as_ref(),
        ids.iter().map(|&id| id as f32).collect::<Vec<_>>()
    );
    assert_eq!(
        output.feature_token_counts,
        ids.iter().map(|&id| id as usize).collect::<Vec<_>>()
    );
    assert_eq!(
        output.item_sizes,
        ids.iter().map(|&id| (id, 1)).collect::<Vec<_>>()
    );
    calls.batches.lock().unwrap().clear();
    run(&components, &config, &ids, "tok").await;
    assert!(calls.batches.lock().unwrap().is_empty());
}

#[tokio::test]
async fn invalid_cache_layout_falls_back_without_publishing_entries() {
    let (components, calls, mut config) = setup(true);
    config
        .preprocessor_config
        .extra
        .insert("invalid_layout".into(), serde_json::json!(true));
    for _ in 0..2 {
        calls.batches.lock().unwrap().clear();
        let output = run(&components, &config, &[3, 4], "tok").await;
        assert_eq!(output.encoder_input_flat().as_ref(), &[3.0, 4.0]);
        // The original cold batch remains usable even when it cannot be split.
        // A later request must retry preprocessing, not consume a poisoned hit.
        assert_eq!(*calls.batches.lock().unwrap(), vec![vec![3, 4]]);
    }
}

#[tokio::test]
async fn minimax_mixed_hits_reordering_and_concurrent_requests_match_batch() {
    use llm_multimodal::vision::processors::MiniMaxM3VisionProcessor;

    let mut components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .unwrap();
    components.pixel_cache = Some(Arc::new(PixelCache::new(64 * 1024 * 1024)));
    let config = MultimodalModelConfig {
        config: serde_json::json!({"model_type":"minimax_m3_vl"}),
        preprocessor_config: PreProcessorConfig {
            max_pixels: Some(112 * 112),
            ..Default::default()
        },
        video_preprocessor_config: None,
    };
    let metadata = ModelMetadata {
        model_id: "minimax-cache-review",
        tokenizer: &NoTokenizer,
        config: &config.config,
    };
    let spec = components.model_registry.lookup(&metadata).unwrap();
    let frames: Vec<_> = [(224, 224), (400, 40), (40, 400), (224, 112)]
        .into_iter()
        .enumerate()
        .map(|(id, (w, h))| {
            Arc::new(ImageFrame::new(
                image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| {
                    image::Rgb([(x % 251) as u8, (y % 239) as u8, ((x + y) % 253) as u8])
                })),
                Default::default(),
                Default::default(),
                ImageSource::InlineBytes,
                id.to_string(),
            ))
        })
        .collect();
    let invoke = async |media: &MediaBatch| {
        preprocess_modality(
            media,
            &components,
            metadata.model_id,
            Some("minimax_m3_vl"),
            spec,
            "review",
            &config,
        )
        .await
    };
    invoke(&MediaBatch::Images(vec![frames[0].clone()]))
        .await
        .unwrap();
    let processor = MiniMaxM3VisionProcessor::new();
    for order in [vec![3, 0, 2, 1, 3], vec![1, 2, 0, 3, 1]] {
        let images: Vec<_> = order
            .into_iter()
            .cycle()
            .take(70)
            .map(|i| frames[i].clone())
            .collect();
        let raw: Vec<_> = images.iter().map(|f| f.image().unwrap().clone()).collect();
        let expected = processor
            .preprocess(&raw, &config.preprocessor_config)
            .unwrap();
        let media = MediaBatch::Images(images);
        let (a, b) = tokio::join!(invoke(&media), invoke(&media));
        for got in [a.unwrap(), b.unwrap()] {
            assert_eq!(got.encoder_input.shape(), expected.encoder_input.shape());
            assert_eq!(got.feature_token_counts, expected.feature_token_counts);
            assert_eq!(got.item_sizes, expected.item_sizes);
            for (got, want) in got
                .encoder_input_flat()
                .iter()
                .zip(expected.encoder_input_flat().iter())
            {
                assert_eq!(got.to_bits(), want.to_bits());
            }
            assert_eq!(got.model_specific.len(), expected.model_specific.len());
            for (key, want) in &expected.model_specific {
                match (&got.model_specific[key], want) {
                    (
                        llm_multimodal::ModelSpecificValue::IntTensor { data: a, shape: sa },
                        llm_multimodal::ModelSpecificValue::IntTensor { data: b, shape: sb },
                    ) => {
                        assert_eq!(a, b);
                        assert_eq!(sa, sb);
                    }
                    _ => panic!("unexpected MiniMax metadata {key}"),
                }
            }
        }
    }
    // An invalid new image must still fail, even when other items are cached.
    let invalid = Arc::new(ImageFrame::new(
        image::DynamicImage::new_rgb8(0, 1),
        Default::default(),
        Default::default(),
        ImageSource::InlineBytes,
        "invalid".into(),
    ));
    let bad_media = MediaBatch::Images(vec![frames[0].clone(), invalid.clone()]);
    assert!(processor
        .preprocess(
            &[
                frames[0].image().unwrap().clone(),
                invalid.image().unwrap().clone()
            ],
            &config.preprocessor_config
        )
        .is_err());
    assert!(invoke(&bad_media).await.is_err());
    assert!(invoke(&MediaBatch::Images(vec![frames[0].clone()]))
        .await
        .is_ok());
}
