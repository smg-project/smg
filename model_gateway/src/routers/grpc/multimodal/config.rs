//! Multimodal model configuration: the shared config-file registry and the
//! per-router component bundle (media connector + model registry).

use std::{collections::HashMap, fmt, path::Path, sync::Arc};

use anyhow::{Context, Result};
use dashmap::DashMap;
use llm_multimodal::{
    registry::{ModelRegistryError, RegistryResult},
    MediaConnector, MediaConnectorConfig, Modality, ModelProcessorSpec, ModelRegistry,
    PreProcessorConfig, VisionPreProcessor,
};
use openai_protocol::worker::MmProcessingMode;
use tracing::{debug, warn};

use super::{
    inflight::MultimodalInflight,
    pixel_cache::{pixel_cache_with_budget, PixelCache},
    process::preprocessor_config_for,
    settings::MultimodalSettings,
};
use crate::{observability::metrics::Metrics, worker::http_client::build_with};

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
        }
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
            if spec.name() == "minimax_m3"
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
        .with_context(|| {
            format!("Failed to resolve model config directory for '{tokenizer_source}'")
        })?;

        let config_path = base_dir.join("config.json");
        let config: serde_json::Value = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config.json at {}", config_path.display()))
            .and_then(|s| {
                serde_json::from_str(&s).with_context(|| {
                    format!("Failed to parse config.json at {}", config_path.display())
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
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicU16, Ordering},
    };

    use tempfile::TempDir;

    use super::*;

    #[tokio::test]
    async fn registry_get_or_load_reads_from_local_dir_and_caches() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"phi3_v","image_token_index":32044}"#,
        )
        .unwrap();
        fs::write(
            tmp.path().join("preprocessor_config.json"),
            r#"{"image_processor_type":"Phi3VImageProcessor"}"#,
        )
        .unwrap();
        let source = tmp.path().to_string_lossy().into_owned();

        let reg = MultimodalConfigRegistry::new();
        let first = reg.get_or_load("tok-uuid-2", &source).await.unwrap();
        assert_eq!(first.config["model_type"].as_str(), Some("phi3_v"));

        let second = reg.get_or_load("tok-uuid-2", &source).await.unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "second call must hit cache and return same Arc"
        );
    }

    #[tokio::test]
    async fn registry_get_or_load_falls_back_when_preprocessor_config_missing() {
        // Mirrors the bundle-preload behavior in try_load_multimodal_config:
        // a local dir without preprocessor_config.json must still load and
        // cache an entry using PreProcessorConfig::default().
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        let source = tmp.path().to_string_lossy().into_owned();

        let reg = MultimodalConfigRegistry::new();
        let loaded = reg
            .get_or_load("tok-uuid-nopp", &source)
            .await
            .expect("must fall back to default preprocessor_config");
        assert_eq!(loaded.config["model_type"].as_str(), Some("llama"));
        assert!(reg.get("tok-uuid-nopp").is_some());
    }

    /// A Hub that answers `status` to every file request, so a checkpoint's
    /// processor-config fetches fail or are refused without a network: the
    /// Hub client asks for a file its cache lacks with one GET and reads the
    /// status before anything else.
    async fn hub_answering(status: Arc<AtomicU16>) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        #[expect(
            clippy::disallowed_methods,
            reason = "the test's Hub lives as long as the test's runtime, which ends it"
        )]
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let status = status.load(Ordering::SeqCst);
                #[expect(
                    clippy::disallowed_methods,
                    reason = "one answer per connection, ended with the test's runtime"
                )]
                tokio::spawn(async move {
                    let mut request = [0u8; 4096];
                    let _ = socket.read(&mut request).await;
                    let reason = if status == 404 {
                        "Not Found"
                    } else {
                        "Service Unavailable"
                    };
                    let answer = format!(
                        "HTTP/1.1 {status} {reason}\r\ncontent-length: 0\r\n\
                         connection: close\r\n\r\n"
                    );
                    let _ = socket.write_all(answer.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        endpoint
    }

    /// A local Hub cache under `home` holding `files` of the checkpoint `repo`
    /// at one commit, laid out as the Hub client keeps it (a cached file is
    /// served with no request). Returns the snapshot directory the files sit in.
    fn hub_cache(home: &Path, repo: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
        let commit = "0".repeat(40);
        let repo_dir = home
            .join("hub")
            .join(format!("models--{}", repo.replace('/', "--")));
        let snapshot = repo_dir.join("snapshots").join(&commit);
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir_all(repo_dir.join("refs")).unwrap();
        fs::write(repo_dir.join("refs").join("main"), commit).unwrap();
        for (name, text) in files {
            fs::write(snapshot.join(name), text).unwrap();
        }
        snapshot
    }

    /// `HF_HOME` and `HF_ENDPOINT` pointed at a test's cache and Hub, and put
    /// back when the test ends, so a failing assertion leaves no other test on
    /// this Hub. The variables are process-wide, so the tests that set them
    /// take turns: the guard holds the turn until the variables are restored.
    struct HubEnv {
        previous: Vec<(&'static str, Option<std::ffi::OsString>)>,
        _turn: tokio::sync::MutexGuard<'static, ()>,
    }

    static HUB_ENV_TURN: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    impl HubEnv {
        async fn set(home: &Path, endpoint: &str) -> Self {
            let turn = HUB_ENV_TURN.lock().await;
            let previous = ["HF_HOME", "HF_ENDPOINT"]
                .into_iter()
                .map(|name| (name, std::env::var_os(name)))
                .collect();
            std::env::set_var("HF_HOME", home);
            std::env::set_var("HF_ENDPOINT", endpoint);
            Self {
                previous,
                _turn: turn,
            }
        }
    }

    impl Drop for HubEnv {
        fn drop(&mut self) {
            // The variables go back before the turn is released (the fields
            // drop after this body).
            for (name, value) in self.previous.drain(..) {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    /// A processor config the Hub fails to serve is not guessed: the request
    /// fails and nothing is cached, so the next request reads the file once
    /// it is there, while the Hub's 404 (a checkpoint without the file) keeps
    /// the default, cached as before. No network: the Hub is a loopback
    /// listener and each checkpoint's `config.json` sits in a local Hub cache.
    #[tokio::test]
    async fn registry_get_or_load_does_not_cache_a_default_after_a_failed_fetch() {
        let home = TempDir::new().unwrap();
        let status = Arc::new(AtomicU16::new(503));
        let endpoint = hub_answering(status.clone()).await;
        let _env = HubEnv::set(home.path(), &endpoint).await;
        let config_json = r#"{"model_type":"llama"}"#;
        let reg = MultimodalConfigRegistry::new();

        let snapshot = hub_cache(home.path(), "cell/failing", &[("config.json", config_json)]);
        let failed = reg
            .get_or_load("tok-uuid-failed", "cell/failing")
            .await
            .expect_err("a 503 on preprocessor_config.json fails the request");
        assert!(
            llm_multimodal::hub::HubFetchError::is(&failed),
            "the failure is the typed one the gateway answers as a 502: {failed:#}"
        );
        let failed = format!("{failed:#}");
        assert!(
            failed.contains("preprocessor_config.json") && failed.contains("503"),
            "the error names the file and the Hub's answer: {failed}"
        );
        assert!(
            reg.get("tok-uuid-failed").is_none(),
            "nothing is cached after a failed fetch"
        );

        // The Hub is back and the checkpoint turns out to ship the file (it is
        // in the cache now, the other two answer 404): the next request reads
        // it instead of a guessed default, and keeps it.
        fs::write(
            snapshot.join("preprocessor_config.json"),
            r#"{"size":{"shortest_edge":448}}"#,
        )
        .unwrap();
        status.store(404, Ordering::SeqCst);
        let loaded = reg
            .get_or_load("tok-uuid-failed", "cell/failing")
            .await
            .expect("the file is read on the next request");
        assert_eq!(
            loaded.preprocessor_config.get_shortest_edge(),
            Some(448),
            "the checkpoint's values, not the default"
        );
        let again = reg
            .get_or_load("tok-uuid-failed", "cell/failing")
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&loaded, &again), "cached once it was read");

        // A checkpoint without the file: the Hub's 404 keeps the default, cached.
        hub_cache(home.path(), "cell/absent", &[("config.json", config_json)]);
        let absent = reg
            .get_or_load("tok-uuid-absent", "cell/absent")
            .await
            .expect("a 404 is a checkpoint without the file");
        assert_eq!(absent.preprocessor_config.get_shortest_edge(), None);
        assert!(
            reg.get("tok-uuid-absent").is_some(),
            "an absent file is cached as the default, as before"
        );

        // config.json itself, on a cold checkpoint: a 503 is the same typed
        // failure, nothing cached; a 404 is a checkpoint without the file, the
        // request's error rather than the Hub's.
        status.store(503, Ordering::SeqCst);
        let failed = reg
            .get_or_load("tok-uuid-cold", "cell/cold")
            .await
            .expect_err("a 503 on config.json fails the request");
        assert!(
            llm_multimodal::hub::HubFetchError::is(&failed),
            "config.json's 503 is the typed failure: {failed:#}"
        );
        let failed = format!("{failed:#}");
        assert!(
            failed.contains("download config.json for") && failed.contains("503"),
            "the error names config.json and the Hub's answer: {failed}"
        );
        assert!(reg.get("tok-uuid-cold").is_none(), "nothing is cached");
        status.store(404, Ordering::SeqCst);
        let missing = reg
            .get_or_load("tok-uuid-missing", "cell/missing")
            .await
            .expect_err("a checkpoint without config.json fails the request");
        assert!(
            !llm_multimodal::hub::HubFetchError::is(&missing),
            "config.json's 404 is not a failure of the Hub: {missing:#}"
        );
        assert!(
            format!("{missing:#}").contains("config.json"),
            "the error names the file: {missing:#}"
        );
        assert!(reg.get("tok-uuid-missing").is_none(), "nothing is cached");

        // Without a token the Hub answers a repository that does not exist
        // with a 401: the same plain error, not a failure of the Hub.
        status.store(401, Ordering::SeqCst);
        let refused = reg
            .get_or_load("tok-uuid-refused", "cell/refused")
            .await
            .expect_err("a 401 on config.json fails the request");
        assert!(
            !llm_multimodal::hub::HubFetchError::is(&refused),
            "config.json's 401 is not a failure of the Hub: {refused:#}"
        );
        assert!(reg.get("tok-uuid-refused").is_none(), "nothing is cached");

        // A processor config of a gated checkpoint whose config.json is a
        // cache hit, without access: the same plain error, nothing cached,
        // and no default in place of a file that may exist.
        hub_cache(home.path(), "cell/gated", &[("config.json", config_json)]);
        let gated = reg
            .get_or_load("tok-uuid-gated", "cell/gated")
            .await
            .expect_err("a 401 on a processor config fails the request");
        assert!(
            !llm_multimodal::hub::HubFetchError::is(&gated),
            "a processor config's 401 is not a failure of the Hub: {gated:#}"
        );
        let gated = format!("{gated:#}");
        assert!(
            gated.contains("preprocessor_config.json") && gated.contains("401"),
            "the error names the file and the Hub's answer: {gated}"
        );
        assert!(reg.get("tok-uuid-gated").is_none(), "nothing is cached");
    }

    /// A Hub that takes the connection and never answers holds a fetch only
    /// for its bound, a processor config's on a warm checkpoint and
    /// `config.json`'s on a cold one: the request fails within it as the typed
    /// error naming the file and the elapsed bound, and nothing is cached. The
    /// clock is paused, so the bound elapses as soon as the runtime idles on
    /// the answer that never comes.
    #[tokio::test(start_paused = true)]
    async fn registry_get_or_load_gives_up_on_a_hub_that_never_answers() {
        let home = TempDir::new().unwrap();
        // Bound and never accepted: the connection completes into the backlog
        // and the request waits for an answer that never comes.
        let silent = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", silent.local_addr().unwrap());
        let _env = HubEnv::set(home.path(), &endpoint).await;
        hub_cache(
            home.path(),
            "cell/silent",
            &[("config.json", r#"{"model_type":"llama"}"#)],
        );
        let reg = MultimodalConfigRegistry::new();

        let failed = tokio::time::timeout(
            std::time::Duration::from_secs(300),
            reg.get_or_load("tok-uuid-silent", "cell/silent"),
        )
        .await
        .expect("the fetch's own bound ends the wait, long before this one")
        .expect_err("a fetch with no answer fails within its bound");
        assert!(
            llm_multimodal::hub::HubFetchError::is(&failed),
            "{failed:#}"
        );
        let failed = format!("{failed:#}");
        assert!(
            failed.contains("preprocessor_config.json") && failed.contains("no answer within"),
            "the error names the file and the elapsed bound: {failed}"
        );
        assert!(
            reg.get("tok-uuid-silent").is_none(),
            "nothing is cached after a fetch that ran out of its bound"
        );

        // A cold checkpoint: `config.json` itself is the first fetch, and the
        // bound covers it the same way.
        let failed = tokio::time::timeout(
            std::time::Duration::from_secs(300),
            reg.get_or_load("tok-uuid-silent-cold", "cell/silent-cold"),
        )
        .await
        .expect("the fetch's own bound ends the wait, long before this one")
        .expect_err("a fetch with no answer fails within its bound");
        assert!(
            llm_multimodal::hub::HubFetchError::is(&failed),
            "{failed:#}"
        );
        let failed = format!("{failed:#}");
        assert!(
            failed.contains("download config.json for") && failed.contains("no answer within"),
            "the error names config.json and the elapsed bound: {failed}"
        );
        assert!(
            reg.get("tok-uuid-silent-cold").is_none(),
            "nothing is cached after a fetch that ran out of its bound"
        );
        drop(silent);
    }

    #[test]
    fn load_video_preprocessor_config_ignores_missing_video_processor_key() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("processor_config.json"),
            r#"{"image_processor":{"image_processor_type":"Qwen3VLImageProcessor"}}"#,
        )
        .unwrap();

        assert!(load_video_preprocessor_config(tmp.path()).is_none());
    }

    #[test]
    fn load_video_preprocessor_config_reads_video_processor_key() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("processor_config.json"),
            r#"{"video_processor":{"image_processor_type":"Qwen3VLVideoProcessor","do_resize":true}}"#,
        )
        .unwrap();

        let config =
            load_video_preprocessor_config(tmp.path()).expect("video_processor should parse");
        assert_eq!(
            config.image_processor_type.as_deref(),
            Some("Qwen3VLVideoProcessor")
        );
        assert_eq!(config.do_resize, Some(true));
    }

    #[test]
    fn load_image_preprocessor_config_merges_the_feature_extractor_section() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("processor_config.json"),
            r#"{"feature_extractor":{"feature_extractor_type":"AudioFeatureExtractor","sampling_rate":24000,"samples_per_token":1920,"padding_value":0.0},
                "image_processor":{"image_processor_type":"NestedImageProcessor","do_resize":true}}"#,
        )
        .unwrap();

        let nested = load_image_preprocessor_config(tmp.path())
            .expect("the image section with the extractor section merged in");
        assert_eq!(
            nested.image_processor_type.as_deref(),
            Some("NestedImageProcessor")
        );
        assert_eq!(nested.do_resize, Some(true));
        assert_eq!(nested.get_extra::<usize>("sampling_rate"), Some(24_000));
        assert_eq!(nested.get_extra::<usize>("samples_per_token"), Some(1_920));
        assert_eq!(
            nested
                .get_extra::<String>("feature_extractor_type")
                .as_deref(),
            Some("AudioFeatureExtractor")
        );

        // The image config's own keys win over the extractor section's.
        fs::write(
            tmp.path().join("preprocessor_config.json"),
            r#"{"image_processor_type":"LegacyImageProcessor","sampling_rate":16000}"#,
        )
        .unwrap();
        let legacy = load_image_preprocessor_config(tmp.path()).expect("legacy image config");
        assert_eq!(
            legacy.image_processor_type.as_deref(),
            Some("LegacyImageProcessor")
        );
        assert_eq!(legacy.get_extra::<usize>("sampling_rate"), Some(16_000));
        assert_eq!(legacy.get_extra::<usize>("samples_per_token"), Some(1_920));

        // An extractor section without any image config: defaults plus the section.
        fs::remove_file(tmp.path().join("preprocessor_config.json")).unwrap();
        fs::write(
            tmp.path().join("processor_config.json"),
            r#"{"feature_extractor":{"sampling_rate":24000}}"#,
        )
        .unwrap();
        let audio_only = load_image_preprocessor_config(tmp.path())
            .expect("the extractor section alone is a config");
        assert_eq!(audio_only.image_processor_type, None);
        assert_eq!(audio_only.get_extra::<usize>("sampling_rate"), Some(24_000));

        // No section: nothing is merged and a missing image config stays missing.
        fs::write(
            tmp.path().join("processor_config.json"),
            r#"{"video_processor":{"image_processor_type":"VideoProcessor"}}"#,
        )
        .unwrap();
        assert!(load_image_preprocessor_config(tmp.path()).is_none());
    }

    #[test]
    fn load_image_preprocessor_config_preserves_legacy_precedence_and_falls_back() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("processor_config.json"),
            r#"{"image_processor":{"image_processor_type":"NestedImageProcessor"}}"#,
        )
        .unwrap();
        let legacy_path = tmp.path().join("preprocessor_config.json");
        fs::write(
            &legacy_path,
            r#"{"image_processor_type":"LegacyImageProcessor"}"#,
        )
        .unwrap();

        let legacy =
            load_image_preprocessor_config(tmp.path()).expect("legacy image config should parse");
        assert_eq!(
            legacy.image_processor_type.as_deref(),
            Some("LegacyImageProcessor")
        );

        fs::write(&legacy_path, "{malformed-json").unwrap();
        let nested = load_image_preprocessor_config(tmp.path())
            .expect("malformed legacy config should fall back to nested image_processor");
        assert_eq!(
            nested.image_processor_type.as_deref(),
            Some("NestedImageProcessor")
        );

        fs::remove_file(legacy_path).unwrap();
        let nested_without_legacy = load_image_preprocessor_config(tmp.path())
            .expect("missing legacy config should fall back to nested image_processor");
        assert_eq!(
            nested_without_legacy.image_processor_type.as_deref(),
            Some("NestedImageProcessor")
        );
    }

    #[tokio::test]
    async fn registry_remove_drops_cached_entry() {
        let reg = MultimodalConfigRegistry::new();
        let cfg = Arc::new(MultimodalModelConfig::new(
            serde_json::json!({"model_type":"phi3_v"}),
            PreProcessorConfig::from_json(r#"{"image_processor_type":"Phi3VImageProcessor"}"#)
                .unwrap(),
            None,
        ));
        reg.insert("tok-uuid-rm".to_string(), cfg.clone());
        assert!(reg.get("tok-uuid-rm").is_some());

        let removed = reg.remove("tok-uuid-rm").expect("remove returns the entry");
        assert!(Arc::ptr_eq(&removed, &cfg));
        assert!(reg.get("tok-uuid-rm").is_none());
        assert!(reg.remove("tok-uuid-rm").is_none());
    }

    #[tokio::test]
    async fn registry_get_or_load_hits_preloaded_entry_without_touching_source() {
        // Regression test for the IGW bug: preload populates the registry
        // under the tokenizer UUID; `get_or_load` must return it without
        // consulting `tokenizer_source` (which in IGW points to an
        // unreachable worker-only path).
        let reg = MultimodalConfigRegistry::new();
        let cfg = Arc::new(MultimodalModelConfig::new(
            serde_json::json!({"model_type":"phi3_v"}),
            PreProcessorConfig::from_json(r#"{"image_processor_type":"Phi3VImageProcessor"}"#)
                .unwrap(),
            None,
        ));
        reg.insert("tok-uuid-3".to_string(), cfg.clone());

        let bad_source = "/nonexistent/worker-only/path-that-would-fail";
        let got = reg
            .get_or_load("tok-uuid-3", bad_source)
            .await
            .expect("preloaded entry must be returned without touching source");
        assert!(Arc::ptr_eq(&got, &cfg));
    }

    /// The resolved settings, not the environment, decide the placement mode
    /// and the pixel cache the components come up with.
    #[test]
    fn components_take_placement_and_pixel_cache_from_the_settings() {
        use super::super::settings::{Setting, SettingSource};

        let settings = MultimodalSettings {
            processing: Setting {
                value: MmProcessingMode::Worker,
                source: SettingSource::Flag,
            },
            ..MultimodalSettings::default()
        };
        let components = MultimodalComponents::new(
            Arc::new(MultimodalConfigRegistry::new()),
            None,
            None,
            &settings,
        )
        .unwrap();
        assert_eq!(components.processing, MmProcessingMode::Worker);
        assert!(
            components.pixel_cache.is_none(),
            "0 MiB keeps the cache off"
        );
    }

    /// The shared router connector applies configured budgets to real stalled
    /// image, audio and video downloads, without needing any engine or decoder.
    #[tokio::test]
    async fn components_enforce_media_fetch_timeouts() {
        use std::time::Duration;

        use llm_multimodal::{ImageFetchConfig, MediaFault, MediaSource, VideoFetchConfig};

        use crate::config::RouterConfig;

        // Leave connections queued without an HTTP response. The listener lives
        // until all fetches finish, so connection refusal cannot pass this test.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/media", listener.local_addr().unwrap());
        let settings = MultimodalSettings::resolve_with(
            &RouterConfig {
                mm_fetch_timeout_ms: Some(50),
                mm_video_fetch_timeout_ms: Some(100),
                mm_allow_private_media_hosts: true,
                ..RouterConfig::default()
            },
            |_| Ok(None),
        )
        .unwrap();
        let components = MultimodalComponents::new(
            Arc::new(MultimodalConfigRegistry::new()),
            None,
            None,
            &settings,
        )
        .unwrap();
        let connector = &components.media_connector;
        for modality in ["image", "audio", "video"] {
            let source = MediaSource::Url(url.clone());
            let fetch = async {
                match modality {
                    "image" => connector
                        .fetch_image(source, ImageFetchConfig::default())
                        .await
                        .map(|_| ()),
                    "audio" => connector.fetch_audio(source).await.map(|_| ()),
                    _ => connector
                        .fetch_video(source, VideoFetchConfig::default())
                        .await
                        .map(|_| ()),
                }
            };
            let error = tokio::time::timeout(Duration::from_secs(2), fetch)
                .await
                .expect("configured budget must expire before the default")
                .unwrap_err();
            assert_eq!(error.fault(), MediaFault::Transient, "{modality}: {error}");
            assert!(
                error.to_string().contains("timed out"),
                "{modality}: {error}"
            );
        }
    }

    /// Under `require` the components come up exactly when the library did:
    /// a host without it is refused with the library and the gauge named, a
    /// host with it starts as before.
    #[test]
    fn a_required_turbojpeg_is_refused_exactly_when_the_library_is_missing() {
        use llm_multimodal::jpeg_turbo::{turbojpeg_available, TurboJpegPolicy};

        use super::super::settings::{Setting, SettingSource};

        let settings = MultimodalSettings {
            turbojpeg: Setting {
                value: TurboJpegPolicy::Require,
                source: SettingSource::Flag,
            },
            ..MultimodalSettings::default()
        };
        let result = MultimodalComponents::new(
            Arc::new(MultimodalConfigRegistry::new()),
            None,
            None,
            &settings,
        );
        if turbojpeg_available() {
            assert!(result.is_ok(), "{:#}", result.err().unwrap());
        } else {
            let error = format!("{:#}", result.err().expect("refused without the library"));
            assert!(error.contains("libturbojpeg.so.0"), "{error}");
            assert!(error.contains("smg_mm_turbojpeg_available"), "{error}");
        }
    }

    fn components(max_inflight_bytes: Option<usize>) -> Result<MultimodalComponents> {
        MultimodalComponents::new(
            Arc::new(MultimodalConfigRegistry::new()),
            None,
            max_inflight_bytes,
            &MultimodalSettings::default(),
        )
    }

    /// A budget too small to admit anything would turn every request carrying
    /// media away. Read as "no limit" it would do the opposite instead, so it
    /// is refused and the operator is told which one to ask for.
    #[test]
    fn a_budget_that_admits_nothing_stops_startup() {
        assert!(components(None).unwrap().inflight.is_none());
        assert!(components(Some(0)).is_err());
        assert!(components(Some(1)).is_err());

        let sized = components(Some(8192)).unwrap();
        let inflight = sized.inflight.expect("a usable budget is kept");
        assert_eq!(inflight.budget_bytes(), 8192);
    }
}
