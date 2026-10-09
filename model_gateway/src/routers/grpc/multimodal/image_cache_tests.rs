use std::sync::{Condvar, Mutex};

use llm_multimodal::{ImageSource, Tokenizer, TransformError, VisionPreProcessor};
use ndarray::Array2;

use super::*;
use crate::routers::grpc::multimodal::{
    config::MultimodalConfigRegistry, settings::MultimodalSettings,
};

#[derive(Default)]
struct Calls {
    batches: Mutex<Vec<Vec<u32>>>,
    second_done: (Mutex<bool>, Condvar),
}

struct Processor {
    calls: Arc<Calls>,
    independent: bool,
    reorder: bool,
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
        if self.reorder && ids == [1] {
            let (lock, ready) = &self.calls.second_done;
            let (_guard, timeout) = ready
                .wait_timeout_while(
                    lock.lock().unwrap(),
                    std::time::Duration::from_secs(5),
                    |done| !*done,
                )
                .unwrap();
            assert!(
                !timeout.timed_out(),
                "second miss must complete before first"
            );
        }
        self.calls.batches.lock().unwrap().push(ids.clone());
        if self.reorder && ids == [2] {
            let (lock, ready) = &self.calls.second_done;
            *lock.lock().unwrap() = true;
            ready.notify_all();
        }
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

fn setup(
    independent: bool,
    reorder: bool,
) -> (MultimodalComponents, Arc<Calls>, MultimodalModelConfig) {
    let calls = Arc::new(Calls::default());
    let mut registry = VisionProcessorRegistry::new();
    registry.register(
        "qwen2-vl",
        Box::new(Processor {
            calls: calls.clone(),
            independent,
            reorder,
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
async fn mixed_hits_and_out_of_order_misses_preserve_order() {
    let (components, calls, config) = setup(true, true);
    run(&components, &config, &[3], "tok").await;
    calls.batches.lock().unwrap().clear();
    let result = run(&components, &config, &[1, 3, 2, 3], "tok").await;
    assert_eq!(*calls.batches.lock().unwrap(), vec![vec![2], vec![1]]);
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
        let (mut components, calls, config) = setup(true, true);
        components.pixel_cache = Some(Arc::new(PixelCache::new(budget)));
        let result = run(&components, &config, &[1, 2, 1, 2, 1], "tok").await;
        assert_eq!(*calls.batches.lock().unwrap(), vec![vec![2], vec![1]]);
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
    let (components, calls, mut config) = setup(true, false);
    run(&components, &config, &[3, 4], "tok").await;
    config.preprocessor_config.rescale_factor = Some(2.0);
    calls.batches.lock().unwrap().clear();
    let result = run(&components, &config, &[3, 4], "tok").await;
    assert_eq!(result.encoder_input.flat_f32().into_owned(), [6.0, 8.0]);
    assert_eq!(calls.batches.lock().unwrap().len(), 2);
    config.config["revision"] = serde_json::json!(2);
    calls.batches.lock().unwrap().clear();
    run(&components, &config, &[3, 4], "tok").await;
    assert_eq!(calls.batches.lock().unwrap().len(), 2);
    calls.batches.lock().unwrap().clear();
    run(&components, &config, &[3, 4], "other-tokenizer").await;
    assert_eq!(calls.batches.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn large_batches_unconfirmed_processors_and_disabled_cache_use_batch_path() {
    for (independent, count, cached) in [
        (true, 31, true),
        (true, 32, true),
        (false, 2, true),
        (true, 2, false),
    ] {
        let (mut components, calls, config) = setup(independent, false);
        if !cached {
            components.pixel_cache = None;
        }
        let ids: Vec<_> = (1..=count).collect();
        run(&components, &config, &ids, "tok").await;
        let batches = calls.batches.lock().unwrap();
        if count == 31 {
            assert_eq!(batches.len(), 31);
            assert!(batches.iter().all(|batch| batch.len() == 1));
        } else {
            assert_eq!(*batches, vec![ids]);
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
            let (components, _calls, config) = setup(true, false);
            run(&components, &config, &[1, 2], "tok").await;
            assert_eq!(lookups(&handle, "qwen2-vl"), Some((0.0, 2.0)));

            run(&components, &config, &[1, 2, 1], "tok").await;
            assert_eq!(lookups(&handle, "qwen2-vl"), Some((2.0, 2.0)));

            run(&components, &config, &[3], "tok").await;
            assert_eq!(lookups(&handle, "qwen2-vl"), Some((2.0, 3.0)));
        });
    });
}
