use std::sync::{Condvar, Mutex};

use llm_multimodal::{
    types::{ImageDetail, ImageSize},
    ImageSource, Tokenizer, TransformError, TransparencyPolicy, VisionPreProcessor,
};
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
    run_for(components, config, ids, tokenizer_id, None).await
}

async fn run_for(
    components: &MultimodalComponents,
    config: &MultimodalModelConfig,
    ids: &[u32],
    tokenizer_id: &str,
    image_wire_dtype: Option<EncoderDtype>,
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
        image_wire_dtype,
        TransparencyPolicy::KeepAlpha,
    )
    .await
    .unwrap()
}

/// A result preprocessed for one wire dtype is served again for that wire
/// only: the neutral form and every other dtype miss, as their bytes differ.
#[tokio::test]
async fn a_wire_dtype_keys_the_cache_apart_from_the_other_forms() {
    let (components, calls, config) = setup(true, false);
    let misses = || calls.batches.lock().unwrap().len();
    run_for(
        &components,
        &config,
        &[1, 2],
        "tok",
        Some(EncoderDtype::BFloat16),
    )
    .await;
    assert_eq!(misses(), 2);
    run_for(
        &components,
        &config,
        &[1, 2],
        "tok",
        Some(EncoderDtype::BFloat16),
    )
    .await;
    assert_eq!(misses(), 2, "the same wire is a hit");
    run_for(&components, &config, &[1, 2], "tok", None).await;
    assert_eq!(misses(), 4, "the neutral form is not the bfloat16 entry");
    run_for(
        &components,
        &config,
        &[1, 2],
        "tok",
        Some(EncoderDtype::Float32),
    )
    .await;
    assert_eq!(misses(), 6, "another wire is not the bfloat16 entry either");
    run(&components, &config, &[1, 2], "tok").await;
    assert_eq!(misses(), 6, "the neutral entries are hits again");
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

/// A processor that records what the stage hands it: whether the image still
/// carries an alpha channel, and the RGB of its first pixel.
struct PixelProbe {
    seen: Seen,
}

type Seen = Arc<Mutex<Vec<(bool, [u8; 3])>>>;

impl VisionPreProcessor for PixelProbe {
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
        "pixel-probe"
    }
    fn preprocess(
        &self,
        images: &[image::DynamicImage],
        _: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let mut seen = self.seen.lock().unwrap();
        for image in images {
            seen.push((image.color().has_alpha(), image.to_rgb8().get_pixel(0, 0).0));
        }
        Ok(PreprocessedEncoderInputs::new(
            Array2::from_shape_vec((images.len(), 1), vec![0.0; images.len()]).unwrap(),
            vec![1; images.len()],
            vec![(2, 1); images.len()],
        ))
    }
}

/// A 2x1 RGBA PNG: a fully transparent black pixel, then an opaque one.
fn transparent_png() -> bytes::Bytes {
    let rgba = image::RgbaImage::from_raw(2, 1, vec![0, 0, 0, 0, 10, 20, 30, 255]).unwrap();
    let mut out = Vec::new();
    image::DynamicImage::ImageRgba8(rgba)
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .unwrap();
    bytes::Bytes::from(out)
}

fn probe_setup() -> (MultimodalComponents, MultimodalModelConfig, Seen) {
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let mut registry = VisionProcessorRegistry::new();
    registry.register("qwen2-vl", Box::new(PixelProbe { seen: seen.clone() }));
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
        MultimodalModelConfig {
            config: serde_json::json!({"model_type": "qwen2_vl"}),
            preprocessor_config: PreProcessorConfig::default(),
            video_preprocessor_config: None,
        },
        seen,
    )
}

/// `count` copies of the transparent PNG, decoded under `policy`, through the
/// stage (one takes the per-image cached path, two the whole-batch path).
async fn run_probe(
    components: &MultimodalComponents,
    config: &MultimodalModelConfig,
    count: usize,
    policy: TransparencyPolicy,
) {
    let media = MediaBatch::Images(
        (0..count)
            .map(|_| {
                Arc::new(
                    ImageFrame::encoded(
                        transparent_png(),
                        ImageDetail::Auto,
                        ImageSource::InlineBytes,
                        "probe".to_string(),
                        ImageSize::new(2, 1),
                        None,
                    )
                    .with_transparency(policy),
                )
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
        "tok",
        config,
        None,
        policy,
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn the_stage_hands_over_the_frame_as_its_policy_decoded_it() {
    for count in [1, 2] {
        for (policy, expected) in [
            (
                TransparencyPolicy::CompositeOverWhite,
                (false, [255, 255, 255]),
            ),
            (TransparencyPolicy::DropAlpha, (false, [0, 0, 0])),
            (TransparencyPolicy::KeepAlpha, (true, [0, 0, 0])),
        ] {
            let (components, config, seen) = probe_setup();
            run_probe(&components, &config, count, policy).await;
            assert_eq!(
                *seen.lock().unwrap(),
                vec![expected; count],
                "{policy:?}, {count} image(s)"
            );
        }
    }
}

#[tokio::test]
async fn the_policy_is_part_of_the_pixel_cache_key() {
    let (components, config, seen) = probe_setup();
    run_probe(
        &components,
        &config,
        1,
        TransparencyPolicy::CompositeOverWhite,
    )
    .await;
    run_probe(&components, &config, 1, TransparencyPolicy::DropAlpha).await;
    run_probe(
        &components,
        &config,
        1,
        TransparencyPolicy::CompositeOverWhite,
    )
    .await;
    assert_eq!(
        *seen.lock().unwrap(),
        vec![(false, [255, 255, 255]), (false, [0, 0, 0])],
        "the same image under another policy misses the cache; the first policy hits it again"
    );
}
