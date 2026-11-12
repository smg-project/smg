use std::sync::{Condvar, Mutex};

use llm_multimodal::{
    types::{ImageDetail, ImageSize},
    ImageSource, Tokenizer, TransformError, TransparencyPolicy, VisionPreProcessor,
};
use ndarray::Array2;

use super::*;
use crate::routers::grpc::multimodal::{
    config::MultimodalConfigRegistry, encoder_budget::ItemOverEncoderBudget,
    settings::MultimodalSettings,
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
    /// The config's `rescale_factor`, as a processor built from it retains it.
    scale: f32,
    /// Whether the size-only count is the width, the count the preprocessing
    /// produces, rather than one.
    size_counted: bool,
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
    fn calculate_num_tokens(&self, width: u32, _: u32) -> Result<usize, TransformError> {
        Ok(if self.size_counted { width as usize } else { 1 })
    }
    fn model_name(&self) -> &'static str {
        "cache-test"
    }
    fn preprocess(
        &self,
        images: &[&image::DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let ids: Vec<_> = images
            .iter()
            .copied()
            .map(image::DynamicImage::width)
            .collect();
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
                ids.iter().map(|&id| id as f32 * self.scale).collect(),
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

/// A Qwen2-VL model config with `preprocessor_config` and the processor its
/// spec would build from it.
fn model_config(
    calls: &Arc<Calls>,
    independent: bool,
    reorder: bool,
    preprocessor_config: PreProcessorConfig,
) -> MultimodalModelConfig {
    let processor = Processor {
        calls: calls.clone(),
        independent,
        reorder,
        scale: preprocessor_config.rescale_factor.unwrap_or(1.0) as f32,
        size_counted: false,
    };
    config_with(processor, preprocessor_config)
}

/// A Qwen2-VL model config whose processor counts an image's feature tokens
/// from its size as its preprocessing produces them (one per pixel of
/// width), so a size alone can put an image over a budget.
fn size_counted_config(calls: &Arc<Calls>) -> MultimodalModelConfig {
    let processor = Processor {
        calls: calls.clone(),
        independent: true,
        reorder: false,
        scale: 1.0,
        size_counted: true,
    };
    config_with(processor, PreProcessorConfig::default())
}

/// A Qwen2-VL model config with `preprocessor_config` and `processor` in the
/// place of the one its spec would build.
fn config_with(
    processor: Processor,
    preprocessor_config: PreProcessorConfig,
) -> MultimodalModelConfig {
    MultimodalModelConfig::new(
        serde_json::json!({"model_type": "qwen2_vl"}),
        preprocessor_config,
        None,
    )
    .with_vision_processors(
        "qwen_vl",
        VisionProcessors::of("qwen_vl", Arc::new(processor)),
    )
}

fn setup(
    independent: bool,
    reorder: bool,
) -> (MultimodalComponents, Arc<Calls>, MultimodalModelConfig) {
    let calls = Arc::new(Calls::default());
    let mut components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .unwrap();
    components.pixel_cache = Some(Arc::new(PixelCache::new(1024 * 1024)));
    let config = model_config(&calls, independent, reorder, PreProcessorConfig::default());
    (components, calls, config)
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
    run_budgeted(
        components,
        config,
        ids,
        tokenizer_id,
        image_wire_dtype,
        None,
    )
    .await
    .unwrap()
}

/// `ids` as images of that width and a height of one, through the stage
/// under `encoder_budget`.
async fn run_budgeted(
    components: &MultimodalComponents,
    config: &MultimodalModelConfig,
    ids: &[u32],
    tokenizer_id: &str,
    image_wire_dtype: Option<EncoderDtype>,
    encoder_budget: Option<usize>,
) -> Result<PreprocessedEncoderInputs> {
    let metadata = ModelMetadata {
        model_id: "qwen2-vl",
        tokenizer: &NoTokenizer,
        config: &config.config,
    };
    let spec = components.model_registry.lookup(&metadata).unwrap();
    let ctx = context(
        components,
        config,
        spec,
        metadata.model_id,
        tokenizer_id,
        image_wire_dtype,
        encoder_budget,
    );
    preprocess_modality(&images_of_width(ids), &ctx).await
}

/// One `width` x 1 frame per entry of `widths`, from pixels.
fn images_of_width(widths: &[u32]) -> MediaBatch {
    MediaBatch::Images(
        widths
            .iter()
            .map(|&width| {
                Arc::new(ImageFrame::new(
                    image::DynamicImage::new_rgb8(width, 1),
                    Default::default(),
                    Default::default(),
                    ImageSource::InlineBytes,
                    width.to_string(),
                ))
            })
            .collect(),
    )
}

/// The stage's context for `config` and `spec`, under `encoder_budget`.
fn context<'a>(
    components: &'a MultimodalComponents,
    config: &'a MultimodalModelConfig,
    spec: &'a dyn ModelProcessorSpec,
    model_id: &'a str,
    tokenizer_id: &'a str,
    image_wire_dtype: Option<EncoderDtype>,
    encoder_budget: Option<usize>,
) -> PlanContext<'a> {
    PlanContext {
        components,
        model_id,
        model_type: config
            .config
            .get("model_type")
            .and_then(serde_json::Value::as_str),
        spec,
        tokenizer_id,
        model_config: config,
        vision_processors: config.vision_processors(spec),
        image_wire_dtype,
        transparency: TransparencyPolicy::KeepAlpha,
        encoder_budget,
    }
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
    let (components, calls, config) = setup(true, false);
    run(&components, &config, &[3, 4], "tok").await;
    let mut config = model_config(
        &calls,
        true,
        false,
        PreProcessorConfig {
            rescale_factor: Some(2.0),
            ..PreProcessorConfig::default()
        },
    );
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

/// An image whose header size alone puts it over the engine's budget is
/// refused before the processor is asked for anything, with the item's
/// place, its count and the budget, on the pixel cache's per-image path and
/// on the whole-batch path alike; at the budget the same images are
/// preprocessed.
#[tokio::test]
async fn an_image_over_the_budget_on_its_size_is_refused_before_any_preprocessing() {
    for cached in [true, false] {
        let (mut components, calls, _) = setup(true, false);
        if !cached {
            components.pixel_cache = None;
        }
        let config = size_counted_config(&calls);
        let error = run_budgeted(&components, &config, &[2, 3], "tok", None, Some(2))
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<ItemOverEncoderBudget>(),
            Some(&ItemOverEncoderBudget {
                modality: Modality::Image,
                position: 2,
                count: 2,
                feature_tokens: 3,
                sampled_frames: None,
                budget: 2,
            }),
            "cached {cached}: {error:#}"
        );
        assert!(
            calls.batches.lock().unwrap().is_empty(),
            "cached {cached}: nothing is preprocessed before the refusal"
        );
        let result = run_budgeted(&components, &config, &[2, 3], "tok", None, Some(3))
            .await
            .unwrap();
        assert_eq!(result.feature_token_counts, [2, 3]);
        assert_eq!(
            calls.batches.lock().unwrap().len(),
            if cached { 2 } else { 1 },
            "cached {cached}"
        );
    }
}

/// The size-only check and the check on the produced counts are one rule
/// applied twice: a processor whose size-only count is under the budget and
/// whose produced count is over passes the first, is preprocessed, and is
/// refused by the second on the produced count.
#[tokio::test]
async fn the_produced_count_is_held_to_the_budget_after_the_size_was() {
    let (components, calls, config) = setup(true, false);
    let metadata = ModelMetadata {
        model_id: "qwen2-vl",
        tokenizer: &NoTokenizer,
        config: &config.config,
    };
    let spec = components.model_registry.lookup(&metadata).unwrap();
    let ctx = context(&components, &config, spec, "qwen2-vl", "tok", None, Some(1));
    let error = prepare_parts(
        vec![images_of_width(&[2])],
        &ctx,
        &metadata,
        &llm_tokenizer::MockTokenizer::new(),
    )
    .await
    .err()
    .expect("the refusal on the produced count");
    assert_eq!(
        error.downcast_ref::<ItemOverEncoderBudget>(),
        Some(&ItemOverEncoderBudget {
            modality: Modality::Image,
            position: 1,
            count: 1,
            feature_tokens: 2,
            sampled_frames: None,
            budget: 1,
        }),
        "{error:#}"
    );
    assert_eq!(*calls.batches.lock().unwrap(), vec![vec![2]]);
}

/// Every image family the registry builds from a model type alone is held to
/// the budget on its header size: one token over is refused before any
/// decode (the frame's bytes are empty, so a decode would fail rather than
/// count), and at the budget a small image goes through the family's own
/// preprocessing, which produces the count its size gave. The two families
/// whose encode takes the source grid under `do_resize: false` are counted
/// in that mode too, on sizes their encode accepts as they are (multiples of
/// Gemma 4's pooled patch of 48 and of GLM's factor of 28).
#[tokio::test]
async fn every_image_family_is_held_to_the_budget_on_its_header_size() {
    let defaults = PreProcessorConfig::default;
    let no_resize = || PreProcessorConfig {
        do_resize: Some(false),
        ..PreProcessorConfig::default()
    };
    // (model type, preprocessor config, a header size, a small size)
    let families = [
        ("qwen2_vl", defaults(), (1600, 1200), (64, 64)),
        ("qwen3_vl", defaults(), (1600, 1200), (64, 64)),
        ("step3_vl", defaults(), (1600, 1200), (64, 64)),
        ("deepseek_v41", defaults(), (1600, 1200), (64, 64)),
        ("llava", defaults(), (1600, 1200), (64, 64)),
        ("llava_next", defaults(), (1600, 1200), (64, 64)),
        ("phi3_v", defaults(), (1600, 1200), (64, 64)),
        ("llama4", defaults(), (1600, 1200), (64, 64)),
        ("kimi_k25", defaults(), (1600, 1200), (64, 64)),
        ("kimi_k3", defaults(), (1600, 1200), (64, 64)),
        ("inkling", defaults(), (1600, 1200), (64, 64)),
        ("muse_spark", defaults(), (1600, 1200), (64, 64)),
        ("gemma4", defaults(), (1600, 1200), (64, 64)),
        ("glm53_flash", defaults(), (1600, 1200), (64, 64)),
        ("gemma4", no_resize(), (960, 480), (96, 48)),
        ("glm53_flash", no_resize(), (1120, 560), (56, 56)),
    ];
    for (model_type, preprocessor_config, header, small) in families {
        let mode = if preprocessor_config.do_resize == Some(false) {
            " (do_resize: false)"
        } else {
            ""
        };
        let (components, _, _) = setup(true, false);
        let config = MultimodalModelConfig::new(
            serde_json::json!({"model_type": model_type}),
            preprocessor_config,
            None,
        );
        let metadata = ModelMetadata {
            model_id: model_type,
            tokenizer: &NoTokenizer,
            config: &config.config,
        };
        let spec = components
            .model_registry
            .lookup(&metadata)
            .unwrap_or_else(|| panic!("{model_type}{mode}: no spec"));
        let processors = config.vision_processors(spec);
        let processor = processors
            .image()
            .unwrap_or_else(|error| panic!("{model_type}{mode}: {error:#}"));
        let ctx = |budget| context(&components, &config, spec, model_type, "tok", None, budget);

        let (width, height) = header;
        let count = processor.calculate_num_tokens(width, height).unwrap();
        let header_only = MediaBatch::Images(vec![Arc::new(ImageFrame::encoded(
            Default::default(),
            Default::default(),
            ImageSource::InlineBytes,
            format!("{model_type}-header"),
            ImageSize::new(width, height),
            None,
        ))]);
        let error = preprocess_modality(&header_only, &ctx(Some(count - 1)))
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<ItemOverEncoderBudget>(),
            Some(&ItemOverEncoderBudget {
                modality: Modality::Image,
                position: 1,
                count: 1,
                feature_tokens: count,
                sampled_frames: None,
                budget: count - 1,
            }),
            "{model_type}{mode}: {error:#}"
        );

        let (width, height) = small;
        let count = processor.calculate_num_tokens(width, height).unwrap();
        let small = MediaBatch::Images(vec![Arc::new(ImageFrame::new(
            image::DynamicImage::new_rgb8(width, height),
            Default::default(),
            Default::default(),
            ImageSource::InlineBytes,
            format!("{model_type}{mode}-small"),
        ))]);
        let result = preprocess_modality(&small, &ctx(Some(count)))
            .await
            .unwrap_or_else(|error| panic!("{model_type}{mode}: {error:#}"));
        assert_eq!(
            result.feature_token_counts,
            [count],
            "{model_type}{mode}: the produced count is the size-only count"
        );
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
    fn calculate_num_tokens(&self, _: u32, _: u32) -> Result<usize, TransformError> {
        Ok(1)
    }
    fn model_name(&self) -> &'static str {
        "pixel-probe"
    }
    fn preprocess(
        &self,
        images: &[&image::DynamicImage],
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
    let processor = Arc::new(PixelProbe { seen: seen.clone() });
    let mut components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .unwrap();
    components.pixel_cache = Some(Arc::new(PixelCache::new(1024 * 1024)));
    (
        components,
        MultimodalModelConfig::new(
            serde_json::json!({"model_type": "qwen2_vl"}),
            PreProcessorConfig::default(),
            None,
        )
        .with_vision_processors("qwen_vl", VisionProcessors::of("qwen_vl", processor)),
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
        &PlanContext {
            components,
            model_id: metadata.model_id,
            model_type: Some("qwen2_vl"),
            spec,
            tokenizer_id: "tok",
            model_config: config,
            vision_processors: config.vision_processors(spec),
            image_wire_dtype: None,
            transparency: policy,
            encoder_budget: None,
        },
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

/// Fill and reuse a 32-image batch, including the path that waits for a replay.
#[tokio::test]
async fn bf16_batch_cache_reuses_large_batches_across_image_families() {
    let frame = |id| {
        Arc::new(ImageFrame::new(
            image::RgbImage::from_pixel(112, 112, image::Rgb([id, 21, 203])).into(),
            Default::default(),
            Default::default(),
            ImageSource::InlineBytes,
            id.to_string(),
        ))
    };
    let cold = (0..32).map(frame).collect::<Vec<_>>();
    let appended = frame(32);
    for model_type in [
        "qwen3_vl",
        "gemma4",
        "glm53_flash",
        "qwen2_vl",
        "qwen2_5_vl",
        "muse_spark",
    ] {
        let config = MultimodalModelConfig::new(
            serde_json::json!({"model_type": model_type, "image_token_id": 100}),
            PreProcessorConfig::default(),
            None,
        );
        let mut components = MultimodalComponents::new(
            Arc::new(MultimodalConfigRegistry::new()),
            None,
            None,
            &MultimodalSettings::default(),
        )
        .unwrap();
        components.pixel_cache = Some(Arc::new(PixelCache::new(512 * 1024 * 1024)));
        let spec = components
            .model_registry
            .lookup(&ModelMetadata {
                model_id: model_type,
                tokenizer: &NoTokenizer,
                config: &config.config,
            })
            .unwrap();
        let ctx = PlanContext {
            components: &components,
            model_id: model_type,
            model_type: Some(model_type),
            spec,
            tokenizer_id: model_type,
            model_config: &config,
            vision_processors: config.vision_processors(spec),
            image_wire_dtype: Some(EncoderDtype::BFloat16),
            transparency: TransparencyPolicy::KeepAlpha,
            encoder_budget: None,
        };
        let first = MediaBatch::Images(cold.clone());
        let uncached = preprocess_batch(&first, &ctx).await.unwrap();
        // A newly cacheable pooled batch can defer filling until it recurs.
        for _ in 0..2 {
            let result = preprocess_modality(&first, &ctx).await.unwrap();
            assert_eq!(result.encoder_input.shape(), uncached.encoder_input.shape());
            assert_eq!(
                result.encoder_input.write_as(EncoderDtype::BFloat16),
                uncached.encoder_input.write_as(EncoderDtype::BFloat16),
            );
        }

        // Invalid encoded bytes make an accidental cache miss visible.
        let mut cached = cold
            .iter()
            .rev()
            .map(|image| {
                Arc::new(ImageFrame::encoded(
                    bytes::Bytes::from_static(b"not an image"),
                    ImageDetail::Auto,
                    ImageSource::InlineBytes,
                    image.hash.clone(),
                    ImageSize::new(112, 112),
                    None,
                ))
            })
            .collect::<Vec<_>>();
        cached.push(Arc::clone(&appended));
        let expected = cold
            .iter()
            .rev()
            .chain(std::iter::once(&appended))
            .cloned()
            .collect();
        let result = preprocess_modality(&MediaBatch::Images(cached), &ctx)
            .await
            .unwrap();
        let expected = preprocess_batch(&MediaBatch::Images(expected), &ctx)
            .await
            .unwrap();
        assert_eq!(result.encoder_input.shape(), expected.encoder_input.shape());
        assert_eq!(
            result.encoder_input.write_as(EncoderDtype::BFloat16),
            expected.encoder_input.write_as(EncoderDtype::BFloat16),
        );
        assert_eq!(result.feature_token_counts, expected.feature_token_counts);
        assert_eq!(result.item_sizes, expected.item_sizes);
        assert_eq!(result.model_specific.len(), expected.model_specific.len());
        for (key, value) in expected.model_specific {
            assert_eq!(
                format!("{:?}", result.model_specific[&key]),
                format!("{value:?}")
            );
        }
    }
}

/// Keep newly computed images when an overlapping request cannot replace its old backing.
#[tokio::test]
async fn bf16_recompute_caches_misses_without_replacing_a_partial_backing() {
    for model in [
        "qwen3_vl",
        "glm53_flash",
        "qwen2_vl",
        "qwen2_5_vl",
        "muse_spark",
    ] {
        let (mut components, _, _) = setup(true, false);
        let cache = Arc::new(PixelCache::new(512 * 1024 * 1024));
        components.pixel_cache = Some(cache.clone());
        let config = MultimodalModelConfig::new(
            serde_json::json!({"model_type": model, "image_token_id": 100}),
            PreProcessorConfig::default(),
            None,
        );
        let spec = components
            .model_registry
            .lookup(&ModelMetadata {
                model_id: model,
                tokenizer: &NoTokenizer,
                config: &config.config,
            })
            .unwrap();
        let ctx = context(
            &components,
            &config,
            spec,
            model,
            model,
            Some(EncoderDtype::BFloat16),
            None,
        );
        let fingerprint = image_cache_fingerprint(&ctx).unwrap();
        let key = |id: u8| PixelCacheKey {
            image_hash: id.to_string(),
            config_fingerprint: fingerprint,
        };
        let batch = |ids: [u8; 3]| {
            MediaBatch::Images(
                ids.into_iter()
                    .map(|id| {
                        Arc::new(ImageFrame::new(
                            image::RgbImage::from_pixel(640, 640, image::Rgb([id, 21, 203])).into(),
                            Default::default(),
                            Default::default(),
                            ImageSource::InlineBytes,
                            id.to_string(),
                        ))
                    })
                    .collect(),
            )
        };
        preprocess_modality(&batch([0, 1, 2]), &ctx).await.unwrap();
        let prefix = cache.get(&key(0)).unwrap();
        assert!(prefix
            .preprocessed
            .encoder_input
            .shared_allocation()
            .is_some());

        // One hit and two misses above the recompute threshold. The other
        // session still owns two slices of the original allocation.
        preprocess_modality(&batch([0, 3, 4]), &ctx).await.unwrap();
        assert!(Arc::ptr_eq(&prefix, &cache.get(&key(0)).unwrap()));
        for id in [3, 4] {
            let entry = cache.get(&key(id)).expect("recomputed miss must be cached");
            assert!(entry
                .preprocessed
                .encoder_input
                .shared_allocation()
                .is_none());
        }
    }
}

/// Reuse previously cached images across turns and preserve wire bytes and order.
#[tokio::test]
async fn minimax_batch_cache_reuses_single_and_batch_misses_in_request_order() {
    let config = MultimodalModelConfig::new(
        serde_json::json!({"model_type": "minimax_m3_vl"}),
        PreProcessorConfig::default(),
        None,
    );
    let image = |id, width, height| {
        Arc::new(ImageFrame::new(
            image::RgbImage::from_pixel(width, height, image::Rgb([id, 21, 203])).into(),
            Default::default(),
            Default::default(),
            ImageSource::InlineBytes,
            id.to_string(),
        ))
    };
    let cached_only = |image: &ImageFrame| {
        // This frame can only succeed if preprocessing does not decode it.
        Arc::new(ImageFrame::encoded(
            bytes::Bytes::from_static(b"not an image"),
            ImageDetail::Auto,
            ImageSource::InlineBytes,
            image.hash.clone(),
            ImageSize::new(112, 112),
            None,
        ))
    };
    let first = image(1, 112, 112);
    let second = image(2, 140, 112);
    let third = image(3, 112, 168);
    let first_hit = cached_only(&first);
    let second_hit = cached_only(&second);
    let third_hit = cached_only(&third);
    let fourth = image(4, 168, 140);
    let fifth = image(5, 140, 168);
    let sixth = image(6, 168, 112);
    let seventh = image(7, 112, 140);
    let sixth_hit = cached_only(&sixth);
    let seventh_hit = cached_only(&seventh);

    let mut components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .unwrap();
    components.pixel_cache = Some(Arc::new(PixelCache::new(16 * 1024 * 1024)));
    let spec = components
        .model_registry
        .lookup(&ModelMetadata {
            model_id: "MiniMax-M3",
            tokenizer: &NoTokenizer,
            config: &config.config,
        })
        .unwrap();
    let ctx = PlanContext {
        components: &components,
        model_id: "MiniMax-M3",
        model_type: Some("minimax_m3_vl"),
        spec,
        tokenizer_id: "m3",
        model_config: &config,
        vision_processors: config.vision_processors(spec),
        image_wire_dtype: Some(EncoderDtype::BFloat16),
        transparency: TransparencyPolicy::KeepAlpha,
        encoder_budget: None,
    };
    let batch = |images: &[&Arc<ImageFrame>]| {
        MediaBatch::Images(images.iter().map(|image| Arc::clone(image)).collect())
    };
    let turns = [
        (vec![&first, &fifth], vec![&first, &fifth]),
        (
            vec![&first_hit, &sixth, &seventh],
            vec![&first, &sixth, &seventh],
        ),
        (vec![&sixth_hit, &seventh_hit], vec![&sixth, &seventh]),
        (
            vec![&second, &first_hit, &second],
            vec![&second, &first, &second],
        ),
        (
            vec![&third, &second_hit, &fourth, &first_hit],
            vec![&third, &second, &fourth, &first],
        ),
        (vec![&third_hit, &first_hit], vec![&third, &first]),
    ];
    for (request, reference) in turns {
        let expected = preprocess_batch(&batch(&reference), &ctx).await.unwrap();
        let actual = preprocess_modality(&batch(&request), &ctx).await.unwrap();
        assert_eq!(actual.encoder_input_shape(), expected.encoder_input_shape());
        assert_eq!(
            actual.encoder_input.write_as(EncoderDtype::BFloat16),
            expected.encoder_input.write_as(EncoderDtype::BFloat16)
        );
        assert_eq!(actual.feature_token_counts, expected.feature_token_counts);
        assert_eq!(actual.item_sizes, expected.item_sizes);
        assert_eq!(actual.model_specific.len(), expected.model_specific.len());
        for (key, value) in expected.model_specific {
            assert_eq!(
                format!("{:?}", actual.model_specific[&key]),
                format!("{value:?}")
            );
        }
    }

    let mut small_components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .unwrap();
    small_components.pixel_cache = Some(Arc::new(PixelCache::new(200 * 1024)));
    let small = PlanContext {
        components: &small_components,
        ..ctx
    };
    preprocess_modality(&batch(&[&first]), &small)
        .await
        .unwrap();
    // A single image fits and is cached. The distinct three-image
    // working set exceeds the budget.
    preprocess_modality(&batch(&[&first_hit]), &small)
        .await
        .unwrap();
    assert!(
        preprocess_modality(&batch(&[&first_hit, &second, &third]), &small)
            .await
            .is_err()
    );
    let oversized = batch(&[&first, &second, &third]);
    let expected = preprocess_batch(&oversized, &small).await.unwrap();
    let actual = preprocess_modality(&oversized, &small).await.unwrap();
    assert_eq!(
        actual.encoder_input.write_as(EncoderDtype::BFloat16),
        expected.encoder_input.write_as(EncoderDtype::BFloat16)
    );

    // Repeated images still share request-local work when the unique
    // working set is too large to admit new entries.
    let request = batch(&[&first_hit, &third, &second, &third]);
    let reference = batch(&[&first, &third, &second, &third]);
    let expected = preprocess_batch(&reference, &small).await.unwrap();
    let actual = preprocess_modality(&request, &small).await.unwrap();
    assert_eq!(actual.feature_token_counts, expected.feature_token_counts);
    assert_eq!(
        actual.encoder_input.write_as(EncoderDtype::BFloat16),
        expected.encoder_input.write_as(EncoderDtype::BFloat16)
    );
    assert!(preprocess_modality(&batch(&[&second_hit]), &small)
        .await
        .is_err());
}

/// Mixed hits, reordered duplicates and concurrent misses match an uncached batch.
/// Tiny budgets bypass storage without changing the result.
#[tokio::test]
async fn independent_bf16_images_match_uncached_with_mixed_hits_and_budget_bypass() {
    let frames: Vec<_> = [(112, 56), (84, 280), (56, 112), (224, 112)]
        .into_iter()
        .enumerate()
        .map(|(id, (w, h))| {
            Arc::new(ImageFrame::new(
                image::RgbImage::from_fn(w, h, |x, y| {
                    image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8])
                })
                .into(),
                Default::default(),
                Default::default(),
                ImageSource::InlineBytes,
                id.to_string(),
            ))
        })
        .collect();
    for model_type in ["glm53_flash", "qwen2_vl", "qwen2_5_vl", "muse_spark"] {
        let config = MultimodalModelConfig::new(
            serde_json::json!({"model_type":model_type, "image_token_id":100}),
            PreProcessorConfig::default(),
            None,
        );
        for budget in [512 * 1024 * 1024, 4 * 1024 * 1024, 1, 0] {
            let mut components = MultimodalComponents::new(
                Arc::new(MultimodalConfigRegistry::new()),
                None,
                None,
                &MultimodalSettings::default(),
            )
            .unwrap();
            components.pixel_cache = (budget > 0).then(|| Arc::new(PixelCache::new(budget)));
            let spec = components
                .model_registry
                .lookup(&ModelMetadata {
                    model_id: model_type,
                    tokenizer: &NoTokenizer,
                    config: &config.config,
                })
                .unwrap();
            let ctx = PlanContext {
                components: &components,
                model_id: model_type,
                model_type: Some(model_type),
                spec,
                tokenizer_id: "glm-cache",
                model_config: &config,
                vision_processors: config.vision_processors(spec),
                image_wire_dtype: Some(EncoderDtype::BFloat16),
                transparency: TransparencyPolicy::KeepAlpha,
                encoder_budget: None,
            };
            preprocess_modality(&MediaBatch::Images(frames[..3].to_vec()), &ctx)
                .await
                .unwrap();
            let order = [2, 0, 1, 2, 3];
            let expected = preprocess_batch(
                &MediaBatch::Images(order.iter().map(|&i| frames[i].clone()).collect()),
                &ctx,
            )
            .await
            .unwrap();
            let media = MediaBatch::Images(
                order
                    .iter()
                    .map(|&i| {
                        if i == 3 || budget < 512 * 1024 * 1024 {
                            return frames[i].clone();
                        }
                        let size = frames[i].size();
                        Arc::new(ImageFrame::encoded(
                            bytes::Bytes::from_static(b"cached image must not be decoded"),
                            ImageDetail::Auto,
                            ImageSource::InlineBytes,
                            frames[i].hash.clone(),
                            size,
                            None,
                        ))
                    })
                    .collect(),
            );
            let (a, b) = tokio::join!(
                preprocess_modality(&media, &ctx),
                preprocess_modality(&media, &ctx)
            );
            for result in [a, b] {
                let actual = result
                    .unwrap_or_else(|error| panic!("{model_type}, cache budget {budget}: {error}"));
                assert_eq!(actual.encoder_input.shape(), expected.encoder_input.shape());
                assert_eq!(
                    actual.encoder_input.write_as(EncoderDtype::BFloat16),
                    expected.encoder_input.write_as(EncoderDtype::BFloat16)
                );
                assert_eq!(actual.feature_token_counts, expected.feature_token_counts);
                assert_eq!(actual.item_sizes, expected.item_sizes);
                for (key, value) in &expected.model_specific {
                    assert_eq!(
                        format!("{:?}", actual.model_specific[key]),
                        format!("{value:?}")
                    );
                }
            }
        }
    }
}
