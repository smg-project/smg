//! Multimodal model configuration: the shared config-file registry and the
//! per-router component bundle (media connector + model registry).

use std::{collections::HashMap, fmt, path::Path, sync::Arc};

use anyhow::{Context, Result};
use dashmap::DashMap;
use llm_multimodal::{
    registry::{ModelRegistryError, RegistryResult},
    AudioPreProcessor, EncoderDtype, MediaConnector, MediaConnectorConfig, Modality,
    ModelProcessorSpec, ModelRegistry, PreProcessorConfig, TransparencyPolicy, VisionPreProcessor,
};
use openai_protocol::worker::MmProcessingMode;
use parking_lot::RwLock;
use smg_worker::{connection::http_client::build_with, identity::basic::WorkerType};
use tracing::{debug, warn};

use super::{
    inflight::MultimodalInflight,
    pixel_cache::{pixel_cache_with_budget, PixelCache},
    plan::DeploymentFault,
    process::preprocessor_config_for,
    settings::MultimodalSettings,
};
use crate::observability::metrics::Metrics;

/// Cached model configuration files loaded from the tokenizer directory.
#[derive(Debug)]
pub(crate) struct MultimodalModelConfig {
    /// Model config.json (HuggingFace format)
    pub config: serde_json::Value,
    /// Preprocessor config (preprocessor_config.json)
    pub preprocessor_config: PreProcessorConfig,
    /// Video-specific preprocessor config, when provided by the model repo.
    pub video_preprocessor_config: Option<PreProcessorConfig>,
    /// The vision processors built from these configs, by the name of the
    /// spec that built them (see [`Self::vision_processors`]).
    vision_processors: DashMap<&'static str, Arc<VisionProcessors>>,
    audio_processors: DashMap<&'static str, AudioProcessor>,
    /// The pixel cache's fingerprints of these configs, one per tokenizer,
    /// model, wire dtype and transparency policy a request named (see
    /// [`Self::image_fingerprint`]).
    image_fingerprints: RwLock<Vec<ImageFingerprint>>,
}

/// One pixel-cache fingerprint of a loaded config, with the request
/// parameters it was hashed for.
#[derive(Debug)]
struct ImageFingerprint {
    tokenizer_id: Box<str>,
    model_id: Box<str>,
    image_wire_dtype: Option<EncoderDtype>,
    transparency: TransparencyPolicy,
    value: u64,
}

impl ImageFingerprint {
    fn is_for(
        &self,
        tokenizer_id: &str,
        model_id: &str,
        image_wire_dtype: Option<EncoderDtype>,
        transparency: TransparencyPolicy,
    ) -> bool {
        self.image_wire_dtype == image_wire_dtype
            && self.transparency == transparency
            && &*self.tokenizer_id == tokenizer_id
            && &*self.model_id == model_id
    }
}

impl MultimodalModelConfig {
    pub(crate) fn new(
        config: serde_json::Value,
        preprocessor_config: PreProcessorConfig,
        video_preprocessor_config: Option<PreProcessorConfig>,
    ) -> Self {
        Self {
            config,
            preprocessor_config,
            video_preprocessor_config,
            vision_processors: DashMap::new(),
            audio_processors: DashMap::new(),
            image_fingerprints: RwLock::new(Vec::new()),
        }
    }

    /// The pixel cache's fingerprint of these configs for `model_id` behind
    /// `tokenizer_id`, served over `image_wire_dtype` under `transparency`:
    /// `hash` runs for the first request with that combination and every
    /// later one reads its value back, as nothing the fingerprint covers
    /// changes while a config is loaded (a reload is a new config with no
    /// entries). A deployment names one combination, so the read is a
    /// comparison or two under the read lock, no allocation.
    pub(crate) fn image_fingerprint(
        &self,
        tokenizer_id: &str,
        model_id: &str,
        image_wire_dtype: Option<EncoderDtype>,
        transparency: TransparencyPolicy,
        hash: impl FnOnce() -> Result<u64>,
    ) -> Result<u64> {
        if let Some(entry) = self
            .image_fingerprints
            .read()
            .iter()
            .find(|entry| entry.is_for(tokenizer_id, model_id, image_wire_dtype, transparency))
        {
            return Ok(entry.value);
        }
        let value = hash()?;
        let mut entries = self.image_fingerprints.write();
        if !entries
            .iter()
            .any(|entry| entry.is_for(tokenizer_id, model_id, image_wire_dtype, transparency))
        {
            entries.push(ImageFingerprint {
                tokenizer_id: tokenizer_id.into(),
                model_id: model_id.into(),
                image_wire_dtype,
                transparency,
                value,
            });
        }
        Ok(value)
    }

    /// The vision processors `spec` builds from these configs: built by the
    /// first request that needs them and shared by every later one, so each
    /// processor resolves the configs once per loaded model rather than once
    /// per request.
    pub(crate) fn vision_processors(&self, spec: &dyn ModelProcessorSpec) -> Arc<VisionProcessors> {
        self.vision_processors
            .entry(spec.name())
            .or_insert_with(|| Arc::new(VisionProcessors::build(spec, self)))
            .clone()
    }

    /// Builds `spec`'s audio processor once per config, caching errors too.
    pub(crate) fn audio_processor(
        &self,
        spec: &dyn ModelProcessorSpec,
    ) -> Result<Arc<dyn AudioPreProcessor>> {
        // `DashMap::entry` takes a shard write lock even on hits.
        if let Some(cached) = self.audio_processors.get(spec.name()) {
            return cached.processor();
        }
        self.audio_processors
            .entry(spec.name())
            .or_insert_with(|| AudioProcessor::build(spec, self))
            .processor()
    }

    /// These configs with `processors` standing for what `spec` would build.
    #[cfg(test)]
    pub(crate) fn with_vision_processors(
        self,
        spec: &'static str,
        processors: VisionProcessors,
    ) -> Self {
        self.vision_processors.insert(spec, Arc::new(processors));
        self
    }
}

/// The vision processors one spec builds from a model's configs: the image
/// processor from the image preprocessor config, the video processor from the
/// video one, or the error the spec reported building either.
pub(crate) struct VisionProcessors {
    spec: &'static str,
    image: RegistryResult<Arc<dyn VisionPreProcessor>>,
    video: RegistryResult<Arc<dyn VisionPreProcessor>>,
}

impl VisionProcessors {
    /// Build processors that consume the geometry selected during media fetching.
    fn build(spec: &dyn ModelProcessorSpec, model_config: &MultimodalModelConfig) -> Self {
        let build = |modality| {
            let mut config = preprocessor_config_for(model_config, modality).clone();
            if spec.resizes_source_images()
                && modality == Modality::Image
                && !config
                    .get_extra::<bool>(llm_multimodal::minimax_resize::SGLANG_RESIZE_KEY)
                    .unwrap_or(false)
            {
                if config.max_pixels.is_some_and(|pixels| {
                    pixels != llm_multimodal::minimax_resize::IMAGE_MAX_TOTAL_PIXELS as usize
                }) || config.min_pixels.is_some_and(|pixels| pixels != 28 * 28)
                {
                    warn!(max_pixels = ?config.max_pixels, min_pixels = ?config.min_pixels,
                        "MiniMax M3 source image sizing overrides legacy area limits: use max_long_side_pixel for image size and encoder_budget for an engine token limit");
                }
                config.extra.insert(
                    llm_multimodal::minimax_resize::RESIZED_KEY.into(),
                    serde_json::json!(true),
                );
            }
            spec.vision_processor(&model_config.config, &config, modality)
                .map(Arc::from)
        };
        Self {
            spec: spec.name(),
            image: build(Modality::Image),
            video: build(Modality::Video),
        }
    }

    /// One processor standing for both of a spec's, for tests that observe
    /// its calls.
    #[cfg(test)]
    pub(crate) fn of(spec: &'static str, processor: Arc<dyn VisionPreProcessor>) -> Self {
        Self {
            spec,
            image: Ok(processor.clone()),
            video: Ok(processor),
        }
    }

    /// The image processor, or why the spec has none.
    pub(crate) fn image(&self) -> Result<&Arc<dyn VisionPreProcessor>> {
        self.processor(&self.image, Modality::Image)
    }

    /// The video processor, or why the spec has none.
    pub(crate) fn video(&self) -> Result<&Arc<dyn VisionPreProcessor>> {
        self.processor(&self.video, Modality::Video)
    }

    fn processor<'a>(
        &self,
        built: &'a RegistryResult<Arc<dyn VisionPreProcessor>>,
        modality: Modality,
    ) -> Result<&'a Arc<dyn VisionPreProcessor>> {
        built.as_ref().map_err(|error| {
            anyhow::anyhow!(
                "The {modality} processor of model spec {} cannot be built: {error}",
                self.spec
            )
        })
    }
}

impl fmt::Debug for VisionProcessors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn name(
            built: &RegistryResult<Arc<dyn VisionPreProcessor>>,
        ) -> Result<&'static str, &ModelRegistryError> {
            built.as_ref().map(|processor| processor.model_name())
        }
        f.debug_struct("VisionProcessors")
            .field("spec", &self.spec)
            .field("image", &name(&self.image))
            .field("video", &name(&self.video))
            .finish()
    }
}

struct AudioProcessor {
    spec: &'static str,
    built: RegistryResult<Option<Arc<dyn AudioPreProcessor>>>,
}

impl AudioProcessor {
    fn build(spec: &dyn ModelProcessorSpec, model_config: &MultimodalModelConfig) -> Self {
        let config = preprocessor_config_for(model_config, Modality::Audio);
        Self {
            spec: spec.name(),
            built: spec
                .audio_processor(&model_config.config, config)
                .map(|processor| processor.map(Arc::from)),
        }
    }

    fn processor(&self) -> Result<Arc<dyn AudioPreProcessor>> {
        match &self.built {
            Ok(Some(processor)) => Ok(Arc::clone(processor)),
            Ok(None) => Err(anyhow::anyhow!(
                "No audio processor registered for model spec: {}",
                self.spec
            )),
            Err(error) => Err(anyhow::anyhow!(
                "The audio processor of model spec {} cannot be built: {error}",
                self.spec
            )),
        }
    }
}

impl fmt::Debug for AudioProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioProcessor")
            .field("spec", &self.spec)
            .field("built", &self.built.as_ref().map(Option::is_some))
            .finish()
    }
}

/// Shared cache of multimodal model configuration files keyed by tokenizer UUID.
///
/// Sources of data:
/// 1. Preloaded from `GetTokenizer` bundles during tokenizer registration.
/// 2. Lazy-loaded from local disk / HF on first multimodal request.
pub struct MultimodalConfigRegistry {
    configs: DashMap<String, Arc<MultimodalModelConfig>>,
}

impl MultimodalConfigRegistry {
    pub(crate) fn new() -> Self {
        Self {
            configs: DashMap::new(),
        }
    }

    pub(crate) fn get(&self, tokenizer_id: &str) -> Option<Arc<MultimodalModelConfig>> {
        self.configs.get(tokenizer_id).map(|r| r.clone())
    }

    pub(crate) fn insert(&self, tokenizer_id: String, config: Arc<MultimodalModelConfig>) {
        self.configs.insert(tokenizer_id, config);
    }

    /// Drop the cached config for a tokenizer. Called when a tokenizer is
    /// removed so stale entries don't accumulate across re-registrations
    /// (tokenizer IDs are regenerated on each registration via `Uuid::now_v7`).
    pub(crate) fn remove(&self, tokenizer_id: &str) -> Option<Arc<MultimodalModelConfig>> {
        self.configs.remove(tokenizer_id).map(|(_, v)| v)
    }

    /// Return a cached config if present; otherwise load from `tokenizer_source`
    /// (local dir or HF cache/download via `llm_multimodal::hub`), cache under
    /// `tokenizer_id`, and return it.
    pub(crate) async fn get_or_load(
        &self,
        tokenizer_id: &str,
        tokenizer_source: &str,
    ) -> Result<Arc<MultimodalModelConfig>> {
        if let Some(cached) = self.get(tokenizer_id) {
            debug!(%tokenizer_id, "multimodal config cache hit");
            return Ok(cached);
        }

        debug!(
            %tokenizer_id,
            %tokenizer_source,
            "multimodal config cache miss, loading"
        );

        // The load runs once per tokenizer and is the largest future this
        // path may await; boxed, so the per-request futures that may reach a
        // miss carry a pointer to it, not its state (a jemalloc size class per
        // request on every path, two on Messages, when the loader grows).
        let base_dir = Box::pin(llm_multimodal::hub::resolve_model_config_dir(
            tokenizer_source,
        ))
        .await
        .map_err(|error| {
            // This sentence is the client's: the source's path stays in the
            // cause under it, which only the log writes.
            let sentence = if llm_multimodal::hub::LocalSourceWithoutConfig::is(&error) {
                "Failed to resolve model config directory: the model's local source has no config.json"
                    .to_string()
            } else {
                format!("Failed to resolve model config directory for '{tokenizer_source}'")
            };
            error.context(sentence)
        })?;

        // From here the configuration is one this gateway holds, in the
        // model's directory or in its hub cache: a failure to read or parse
        // it is the deployment's own whatever the request holds, and carries
        // the mark the preparation's stage answers by. Failures to resolve
        // the directory left above, unmarked.
        let config_path = base_dir.join("config.json");
        let config: serde_json::Value = std::fs::read_to_string(&config_path)
            .with_context(|| {
                DeploymentFault(format!(
                    "Failed to read config.json at {}",
                    config_path.display()
                ))
            })
            .and_then(|s| {
                serde_json::from_str(&s).with_context(|| {
                    DeploymentFault(format!(
                        "Failed to parse config.json at {}",
                        config_path.display()
                    ))
                })
            })?;

        // preprocessor_config.json is optional — each vision processor supplies
        // its own model-specific defaults, so missing/unparsable files fall
        // back to `PreProcessorConfig::default()`. This matches the bundle
        // preload path in `try_load_multimodal_config`.
        let preprocessor_config = load_image_preprocessor_config(&base_dir).unwrap_or_else(|| {
            debug!(
                path = %base_dir.display(),
                "No image preprocessor config found; using PreProcessorConfig defaults"
            );
            PreProcessorConfig::default()
        });
        let video_preprocessor_config = load_video_preprocessor_config(&base_dir);

        let model_config = Arc::new(MultimodalModelConfig::new(
            config,
            preprocessor_config,
            video_preprocessor_config,
        ));

        self.configs
            .insert(tokenizer_id.to_string(), model_config.clone());

        debug!(%tokenizer_id, "multimodal config loaded and cached");
        Ok(model_config)
    }
}

impl Default for MultimodalConfigRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn load_preprocessor_config_file(
    path: &Path,
    label: &str,
) -> Option<PreProcessorConfig> {
    if !path.exists() {
        return None;
    }

    match std::fs::read_to_string(path) {
        Ok(config_str) => match PreProcessorConfig::from_json(&config_str) {
            Ok(config) => Some(config),
            Err(e) => {
                warn!(
                    path = %path.display(),
                    error = %e,
                    "Failed to parse {label}"
                );
                None
            }
        },
        Err(e) => {
            warn!(
                path = %path.display(),
                error = %e,
                "Failed to read {label}"
            );
            None
        }
    }
}

pub(crate) fn load_video_preprocessor_config(base_dir: &Path) -> Option<PreProcessorConfig> {
    let video_path = base_dir.join("video_preprocessor_config.json");
    if let Some(config) =
        load_preprocessor_config_file(&video_path, "video_preprocessor_config.json")
    {
        return Some(config);
    }

    let processor_path = base_dir.join("processor_config.json");
    if !processor_path.exists() {
        return None;
    }

    let processor_config = match std::fs::read_to_string(&processor_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    {
        Some(config) => config,
        None => {
            warn!(
                path = %processor_path.display(),
                "Failed to load processor_config.json for video_processor"
            );
            return None;
        }
    };

    let video_processor = processor_config.get("video_processor")?;
    match PreProcessorConfig::from_value(video_processor.clone()) {
        Ok(config) => Some(config),
        Err(error) => {
            warn!(
                path = %processor_path.display(),
                error = %error,
                "Failed to parse video_processor from processor_config.json"
            );
            None
        }
    }
}

pub(crate) fn load_image_preprocessor_config(base_dir: &Path) -> Option<PreProcessorConfig> {
    // Native policy files are authoritative. A malformed file must not select
    // an unrelated generic policy. The native processor rejects empty defaults.
    let native_path = base_dir.join("image_preprocessing.json");
    if native_path.exists() {
        return load_preprocessor_config_file(&native_path, "image_preprocessing.json");
    }
    with_feature_extractor_section(load_image_processor_section(base_dir), base_dir)
}

/// The image processor's config: the flat `preprocessor_config.json`, else
/// the `image_processor` section of `processor_config.json`.
fn load_image_processor_section(base_dir: &Path) -> Option<PreProcessorConfig> {
    let image_path = base_dir.join("preprocessor_config.json");
    if let Some(config) = load_preprocessor_config_file(&image_path, "preprocessor_config.json") {
        return Some(config);
    }

    let processor_path = base_dir.join("processor_config.json");
    if !processor_path.exists() {
        return None;
    }
    let processor_config = match std::fs::read_to_string(&processor_path)
        .ok()
        .and_then(|config| serde_json::from_str::<serde_json::Value>(&config).ok())
    {
        Some(config) => config,
        None => {
            // Symmetric with load_video_preprocessor_config: a present but
            // unreadable file silently degrading to defaults is the worst
            // outcome — wrong normalization with nothing in the logs.
            warn!(
                path = %processor_path.display(),
                "Failed to read or parse processor_config.json for the image fallback"
            );
            return None;
        }
    };
    let image_processor = processor_config.get("image_processor")?;
    PreProcessorConfig::from_value(image_processor.clone())
        .inspect_err(|error| {
            warn!(
                path = %processor_path.display(),
                error = %error,
                "Failed to parse image_processor from processor_config.json"
            );
        })
        .ok()
}

/// A `processor_config.json` that nests the audio feature extractor next to
/// the image and video processors keeps the extractor's parameters
/// (`sampling_rate`, `samples_per_token`, ...) in its `feature_extractor`
/// section. Audio processors read them from the preprocessor config's
/// extras, where a flat `preprocessor_config.json` would have put them, so
/// the section's entries are merged in under the keys the image config does
/// not set; a model with the section and no image config gets the defaults
/// plus the section.
fn with_feature_extractor_section(
    config: Option<PreProcessorConfig>,
    base_dir: &Path,
) -> Option<PreProcessorConfig> {
    let processor_path = base_dir.join("processor_config.json");
    let section = std::fs::read_to_string(&processor_path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(
            |processor_config| match processor_config.get("feature_extractor") {
                Some(serde_json::Value::Object(section)) => Some(section.clone()),
                _ => None,
            },
        );
    let Some(section) = section else {
        return config;
    };
    let mut config = config.unwrap_or_default();
    for (key, value) in section {
        config.extra.entry(key).or_insert(value);
    }
    Some(config)
}

/// Shared multimodal components injected at router creation time.
pub(crate) struct MultimodalComponents {
    pub media_connector: Arc<MediaConnector>,
    pub model_registry: Arc<ModelRegistry>,
    /// Shared reference to the app-level multimodal config cache.
    pub config_registry: Arc<MultimodalConfigRegistry>,
    /// Optional host-DRAM cache of preprocessed per-image encoder inputs.
    pub pixel_cache: Option<Arc<PixelCache>>,
    /// Router-configured per-modality media-count limits replacing spec limits.
    pub modality_limit_overrides: HashMap<Modality, usize>,
    /// Where media is fetched and preprocessed for vLLM gRPC workers.
    pub processing: MmProcessingMode,
    /// Worker admission must cover sibling fetches and blocking decoders on error.
    pub drain_fetches_on_error: bool,
    /// Cap on preprocessed media bytes in flight; `None` leaves it unbounded.
    pub inflight: Option<Arc<MultimodalInflight>>,
    /// The engine's own video frame budget, for the specs that sample a clip
    /// the way its loader does (the worker-side pipeline knows it; `None`
    /// keeps each spec's own constant).
    pub video_frame_budget: Option<usize>,
    /// The role whose workers take a request's pixels in the router's mode
    /// (`Mode::pixels_go_to`); the regular workers unless the router says
    /// otherwise.
    pub pixels_go_to: WorkerType,
}

impl MultimodalComponents {
    /// Create multimodal components with default registries and a reference
    /// to the shared `MultimodalConfigRegistry` owned by `AppContext`.
    /// `settings` is the resolved flag > env > default bundle.
    pub fn new(
        config_registry: Arc<MultimodalConfigRegistry>,
        image_limit_override: Option<usize>,
        max_inflight_bytes: Option<usize>,
        settings: &MultimodalSettings,
    ) -> Result<Self> {
        // The router fetches client-supplied media URLs itself here: the
        // connector's host policy (the allow-list and the address policy)
        // comes from the same resolved settings as the placement mode.
        let connector_config = MediaConnectorConfig {
            allowed_domains: settings.allowed_media_domains.value.clone(),
            allow_private_hosts: settings.allow_private_media_hosts.value,
            fetch_timeout: std::time::Duration::from_millis(settings.fetch_timeout_ms.value),
            video_fetch_timeout: std::time::Duration::from_millis(
                settings.video_fetch_timeout_ms.value,
            ),
            ..MediaConnectorConfig::default()
        };
        // The line shows the allow-list as the connector enforces it (trimmed,
        // a list with no names being none), not as it was spelled.
        tracing::info!(
            allowed_media_domains =
                ?llm_multimodal::effective_allowed_domains(connector_config.allowed_domains.as_deref()),
            allow_private_media_hosts = connector_config.allow_private_hosts,
            allow_private_media_hosts_source = settings.allow_private_media_hosts.source.as_str(),
            "media fetch policy for router-side processing"
        );
        let media_connector = build_with(
            || crate::server::http_client_builder().timeout(std::time::Duration::from_secs(30)),
            false,
            "media HTTP client",
            |client| MediaConnector::new(client, connector_config.clone()),
        )
        .map_err(anyhow::Error::msg)?;

        super::report_jpeg_decoder();
        super::enforce_jpeg_decoder_policy(settings.turbojpeg.value)?;
        tracing::info!(
            policy = %settings.turbojpeg.value,
            source = settings.turbojpeg.source.as_str(),
            "TurboJPEG policy"
        );
        let processing = settings.processing.value;
        tracing::info!(
            mode = %processing,
            source = settings.processing.source.as_str(),
            "multimodal processing mode"
        );
        // The switch is the way back to the two-pass write: say where it
        // stands and who set it, so a rollback that did not take is seen.
        tracing::info!(
            on = settings.one_pass_wire.value,
            source = settings.one_pass_wire.source.as_str(),
            "multimodal one-pass wire write"
        );

        Metrics::init_mm_inflight_series();
        let inflight = max_inflight_bytes
            .map(|bytes| {
                let inflight = MultimodalInflight::new(bytes);
                // Zero is not how an operator asks for no limit: it would
                // refuse every request carrying media. Leaving the setting
                // off is, so a budget too small to admit anything is more
                // likely a mistake than an intent to serve nothing.
                anyhow::ensure!(
                    inflight.budget_bytes() > 0,
                    "multimodal_max_inflight_bytes is {bytes}, too little to admit any request \
                     carrying media; leave it unset to hold an unbounded amount"
                );
                Ok(Arc::new(inflight))
            })
            .transpose()?;

        Ok(Self {
            media_connector: Arc::new(media_connector),
            model_registry: Arc::new(ModelRegistry::default()),
            config_registry,
            pixel_cache: pixel_cache_with_budget(settings.pixel_cache_mb.value),
            modality_limit_overrides: image_limit_override
                .map(|limit| HashMap::from([(Modality::Image, limit)]))
                .unwrap_or_default(),
            processing,
            drain_fetches_on_error: false,
            inflight,
            video_frame_budget: None,
            pixels_go_to: WorkerType::Regular,
        })
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/multimodal/config/tests.rs"]
pub(crate) mod tests;
