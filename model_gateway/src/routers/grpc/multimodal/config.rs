//! Multimodal model configuration: the shared config-file registry and the
//! per-router component bundle (media connector + processor/model registries).

use std::{
    collections::HashMap,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use dashmap::DashMap;
use llm_multimodal::{
    MediaConnector, MediaConnectorConfig, Modality, ModelRegistry, PreProcessorConfig,
    VisionProcessorRegistry,
};
use openai_protocol::worker::MmProcessingMode;
use tokio::sync::{Mutex, OwnedMutexGuard};
use tracing::{debug, warn};

use super::{
    inflight::MultimodalInflight,
    pixel_cache::{pixel_cache_with_budget, PixelCache},
    settings::MultimodalSettings,
};

/// Cached model configuration files loaded from the tokenizer directory.
#[derive(Debug, Clone)]
pub(crate) struct MultimodalModelConfig {
    /// Model config.json (HuggingFace format)
    pub config: serde_json::Value,
    /// Preprocessor config (preprocessor_config.json)
    pub preprocessor_config: PreProcessorConfig,
    /// Video-specific preprocessor config, when provided by the model repo.
    pub video_preprocessor_config: Option<PreProcessorConfig>,
}

/// Shared cache of multimodal model configuration files keyed by tokenizer UUID.
///
/// Sources of data:
/// 1. Preloaded from `GetTokenizer` bundles during tokenizer registration.
/// 2. Lazy-loaded from local disk / HF on first multimodal request.
pub struct MultimodalConfigRegistry {
    configs: DashMap<String, Arc<MultimodalModelConfig>>,
    /// Tokenizers whose last load failed: when, and the formatted cause. A
    /// load is a directory probe plus file reads, or a HuggingFace download
    /// attempt; without this a model that has no config would pay for one on
    /// every request that asks (the media-part order, then the placeholder
    /// tokens). The cause travels with the hold so every caller in the window
    /// still sees why, not just the first one.
    failed_loads: DashMap<String, (Instant, Arc<str>)>,
    /// How long a failed load is held before it is tried again.
    failure_ttl: Duration,
    /// One load at a time per tokenizer: a burst of requests arriving before
    /// the first load finishes queues on this lock and then reuses that
    /// load's result, success or recorded failure, instead of each probing
    /// the directory or asking HuggingFace. Entries are dropped by the last
    /// holder (see `LoadSlot`).
    loading: DashMap<String, Arc<Mutex<()>>>,
}

/// A caller's hold on the per-tokenizer load lock. On drop, the lock entry
/// is removed from the map when nobody else holds it: a waiter still holding
/// the `Arc` keeps the entry, so every concurrent caller serializes on the
/// same lock, and a later burst creates a fresh one.
struct LoadSlot<'a> {
    locks: &'a DashMap<String, Arc<Mutex<()>>>,
    key: &'a str,
    guard: Option<OwnedMutexGuard<()>>,
}

impl<'a> LoadSlot<'a> {
    async fn acquire(locks: &'a DashMap<String, Arc<Mutex<()>>>, key: &'a str) -> Self {
        let lock = locks
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let guard = lock.lock_owned().await;
        Self {
            locks,
            key,
            guard: Some(guard),
        }
    }
}

impl Drop for LoadSlot<'_> {
    fn drop(&mut self) {
        // Release our hold (and its reference) first; then the map's own
        // reference is the only one left exactly when no other caller is
        // queued on this lock. `remove_if` runs under the shard lock, so a
        // newcomer's `entry()` either joins the entry or runs after the removal.
        drop(self.guard.take());
        self.locks
            .remove_if(self.key, |_, lock| Arc::strong_count(lock) == 1);
    }
}

/// How long `get_or_load` reports a failed load instead of retrying it: long
/// enough to stop per-request probing, short enough that a transient failure
/// (a HuggingFace timeout, say) clears on its own.
const LOAD_FAILURE_TTL: Duration = Duration::from_secs(5);

impl MultimodalConfigRegistry {
    pub(crate) fn new() -> Self {
        Self::with_failure_ttl(LOAD_FAILURE_TTL)
    }

    fn with_failure_ttl(failure_ttl: Duration) -> Self {
        Self {
            configs: DashMap::new(),
            failed_loads: DashMap::new(),
            failure_ttl,
            loading: DashMap::new(),
        }
    }

    pub(crate) fn get(&self, tokenizer_id: &str) -> Option<Arc<MultimodalModelConfig>> {
        self.configs.get(tokenizer_id).map(|r| r.clone())
    }

    pub(crate) fn insert(&self, tokenizer_id: String, config: Arc<MultimodalModelConfig>) {
        self.failed_loads.remove(&tokenizer_id);
        self.configs.insert(tokenizer_id, config);
    }

    /// Drop the cached config for a tokenizer. Called when a tokenizer is
    /// removed so stale entries don't accumulate across re-registrations
    /// (tokenizer IDs are regenerated on each registration via `Uuid::now_v7`).
    pub(crate) fn remove(&self, tokenizer_id: &str) -> Option<Arc<MultimodalModelConfig>> {
        self.failed_loads.remove(tokenizer_id);
        self.configs.remove(tokenizer_id).map(|(_, v)| v)
    }

    /// Return a cached config if present; otherwise load from `tokenizer_source`
    /// (local dir or HF cache/download via `llm_multimodal::hub`), cache under
    /// `tokenizer_id`, and return it.
    ///
    /// A load that fails is not retried for [`LOAD_FAILURE_TTL`]; until then
    /// this returns an error carrying the earlier failure's cause. Concurrent
    /// callers for the same tokenizer wait for one load and share its result.
    pub(crate) async fn get_or_load(
        &self,
        tokenizer_id: &str,
        tokenizer_source: &str,
    ) -> Result<Arc<MultimodalModelConfig>> {
        if let Some(cached) = self.get(tokenizer_id) {
            debug!(%tokenizer_id, "multimodal config cache hit");
            return Ok(cached);
        }
        self.held_failure(tokenizer_id)?;

        let _loading = LoadSlot::acquire(&self.loading, tokenizer_id).await;

        // Whoever held the lock before us may have loaded, or failed and
        // recorded it, in the meantime.
        if let Some(cached) = self.get(tokenizer_id) {
            debug!(%tokenizer_id, "multimodal config loaded by a concurrent request");
            return Ok(cached);
        }
        self.held_failure(tokenizer_id)?;

        debug!(
            %tokenizer_id,
            %tokenizer_source,
            "multimodal config cache miss, loading"
        );

        let model_config = match load_model_config(tokenizer_source).await {
            Ok(config) => config,
            Err(error) => {
                // The first caller may swallow this error (the media-part
                // order resolution falls back silently), so the cause is
                // logged here, once per hold, as well as kept for later callers.
                let cause: Arc<str> = Arc::from(format!("{error:#}"));
                warn!(
                    %tokenizer_id,
                    %tokenizer_source,
                    %cause,
                    hold = ?self.failure_ttl,
                    "multimodal config load failed; not retried until the hold expires"
                );
                self.failed_loads
                    .insert(tokenizer_id.to_string(), (Instant::now(), cause));
                return Err(error);
            }
        };

        self.insert(tokenizer_id.to_string(), model_config.clone());

        debug!(%tokenizer_id, "multimodal config loaded and cached");
        Ok(model_config)
    }

    /// The error for a failure still within its hold, if any.
    fn held_failure(&self, tokenizer_id: &str) -> Result<()> {
        let Some((failed_at, cause)) = self
            .failed_loads
            .get(tokenizer_id)
            .map(|entry| entry.value().clone())
        else {
            return Ok(());
        };
        let since = failed_at.elapsed();
        if since < self.failure_ttl {
            debug!(%tokenizer_id, ?since, %cause, "multimodal config load failed recently; not retried");
            anyhow::bail!(
                "multimodal config for tokenizer '{tokenizer_id}' failed to load {since:?} ago: \
                 {cause}; not retried for {:?}",
                self.failure_ttl
            );
        }
        Ok(())
    }
}

/// Load a model's multimodal configuration files from `tokenizer_source`
/// (a local directory, or a HuggingFace repo resolved through the hub cache).
async fn load_model_config(tokenizer_source: &str) -> Result<Arc<MultimodalModelConfig>> {
    let base_dir = llm_multimodal::hub::resolve_model_config_dir(tokenizer_source)
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

    Ok(Arc::new(MultimodalModelConfig {
        config,
        preprocessor_config,
        video_preprocessor_config,
    }))
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

/// Shared multimodal components injected at router creation time.
pub(crate) struct MultimodalComponents {
    pub media_connector: Arc<MediaConnector>,
    pub vision_processor_registry: Arc<VisionProcessorRegistry>,
    pub model_registry: Arc<ModelRegistry>,
    /// Shared reference to the app-level multimodal config cache.
    pub config_registry: Arc<MultimodalConfigRegistry>,
    /// Optional host-DRAM cache of preprocessed per-image encoder inputs.
    pub pixel_cache: Option<Arc<PixelCache>>,
    /// Router-configured per-modality media-count limits replacing spec limits.
    pub modality_limit_overrides: HashMap<Modality, usize>,
    /// Where media is fetched and preprocessed for vLLM gRPC workers.
    pub processing: MmProcessingMode,
    /// Cap on preprocessed media bytes in flight; `None` leaves it unbounded.
    pub inflight: Option<Arc<MultimodalInflight>>,
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
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create reqwest client")?;
        let media_connector = MediaConnector::new(client, MediaConnectorConfig::default())
            .context("Failed to create MediaConnector")?;

        super::report_jpeg_decoder();
        let processing = settings.processing.value;
        tracing::info!(
            mode = %processing,
            source = settings.processing.source.as_str(),
            "multimodal processing mode"
        );

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
            vision_processor_registry: Arc::new(VisionProcessorRegistry::with_defaults()),
            model_registry: Arc::new(ModelRegistry::default()),
            config_registry,
            pixel_cache: pixel_cache_with_budget(settings.pixel_cache_mb.value),
            modality_limit_overrides: image_limit_override
                .map(|limit| HashMap::from([(Modality::Image, limit)]))
                .unwrap_or_default(),
            processing,
            inflight,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

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
    async fn registry_get_or_load_does_not_retry_a_failed_load_within_the_ttl() {
        let tmp = TempDir::new().unwrap();
        let config_path = tmp.path().join("config.json");
        fs::write(&config_path, "{not json").unwrap();
        let source = tmp.path().to_string_lossy().into_owned();

        let reg = MultimodalConfigRegistry::new();
        let first = reg.get_or_load("tok-bad", &source).await.unwrap_err();
        assert!(
            first.to_string().contains("Failed to parse config.json"),
            "first failure must be the load error, got: {first:#}"
        );

        // Even once the file is fixed, the recent failure is reported
        // without another load until the TTL passes.
        fs::write(&config_path, r#"{"model_type":"phi3_v"}"#).unwrap();
        let second = reg.get_or_load("tok-bad", &source).await.unwrap_err();
        assert!(
            second.to_string().contains("not retried"),
            "second call must report the held failure, got: {second:#}"
        );
        assert!(reg.get("tok-bad").is_none());

        // A fresh registration (insert or remove) forgets the failure.
        reg.remove("tok-bad");
        let loaded = reg.get_or_load("tok-bad", &source).await.unwrap();
        assert_eq!(loaded.config["model_type"].as_str(), Some("phi3_v"));
    }

    #[tokio::test]
    async fn registry_held_failure_names_the_original_cause() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("config.json"), "{not json").unwrap();
        let source = tmp.path().to_string_lossy().into_owned();
        let reg = MultimodalConfigRegistry::new();
        let first = reg.get_or_load("tok-cause", &source).await.unwrap_err();
        let first_text = format!("{first:#}");
        assert!(
            first_text.contains("Failed to parse config.json"),
            "first failure must be the load error, got: {first_text}"
        );

        // Every caller within the hold sees that same cause, not just "held".
        let held = reg.get_or_load("tok-cause", &source).await.unwrap_err();
        let held_text = held.to_string();
        assert!(
            held_text.contains("not retried") && held_text.contains(&first.to_string()),
            "held failure must carry the original cause, got: {held_text}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_requests_share_one_failed_load() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("config.json"), "{not json").unwrap();
        let source = tmp.path().to_string_lossy().into_owned();
        let reg = Arc::new(MultimodalConfigRegistry::new());

        let mut tasks = tokio::task::JoinSet::new();
        for _ in 0..16 {
            let reg = Arc::clone(&reg);
            let source = source.clone();
            tasks.spawn(async move { reg.get_or_load("tok-burst", &source).await });
        }
        let mut raw = 0;
        let mut held = 0;
        while let Some(result) = tasks.join_next().await {
            let error = result.unwrap().unwrap_err().to_string();
            // A caller that ran the load gets the load error itself; one that
            // waited on it gets the hold, which quotes that same error.
            assert!(error.contains("Failed to parse config.json"), "{error}");
            if error.contains("not retried") {
                held += 1;
            } else {
                raw += 1;
            }
        }
        assert_eq!((raw, held), (1, 15), "exactly one caller must have loaded");
        assert!(
            reg.loading.is_empty(),
            "the load lock is released once the burst is over"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_requests_share_one_loaded_config() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("config.json"), r#"{"model_type":"phi3_v"}"#).unwrap();
        let source = tmp.path().to_string_lossy().into_owned();
        let reg = Arc::new(MultimodalConfigRegistry::new());

        let mut tasks = tokio::task::JoinSet::new();
        for _ in 0..16 {
            let reg = Arc::clone(&reg);
            let source = source.clone();
            tasks.spawn(async move { reg.get_or_load("tok-shared", &source).await });
        }
        let mut configs = Vec::new();
        while let Some(result) = tasks.join_next().await {
            configs.push(result.unwrap().unwrap());
        }
        assert_eq!(configs.len(), 16);
        // One load: every caller holds the very allocation that load cached.
        let cached = reg.get("tok-shared").unwrap();
        assert!(configs.iter().all(|config| Arc::ptr_eq(config, &cached)));
        assert_eq!(cached.config["model_type"].as_str(), Some("phi3_v"));
        assert!(reg.loading.is_empty());
    }

    #[tokio::test]
    async fn registry_get_or_load_retries_a_failed_load_after_the_ttl() {
        let tmp = TempDir::new().unwrap();
        let config_path = tmp.path().join("config.json");
        fs::write(&config_path, "{not json").unwrap();
        let source = tmp.path().to_string_lossy().into_owned();

        let reg = MultimodalConfigRegistry::with_failure_ttl(Duration::from_millis(20));
        reg.get_or_load("tok-ttl", &source).await.unwrap_err();
        fs::write(&config_path, r#"{"model_type":"phi3_v"}"#).unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let loaded = reg.get_or_load("tok-ttl", &source).await.unwrap();
        assert_eq!(loaded.config["model_type"].as_str(), Some("phi3_v"));
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
        let cfg = Arc::new(MultimodalModelConfig {
            config: serde_json::json!({"model_type":"phi3_v"}),
            preprocessor_config: PreProcessorConfig::from_json(
                r#"{"image_processor_type":"Phi3VImageProcessor"}"#,
            )
            .unwrap(),
            video_preprocessor_config: None,
        });
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
        let cfg = Arc::new(MultimodalModelConfig {
            config: serde_json::json!({"model_type":"phi3_v"}),
            preprocessor_config: PreProcessorConfig::from_json(
                r#"{"image_processor_type":"Phi3VImageProcessor"}"#,
            )
            .unwrap(),
            video_preprocessor_config: None,
        });
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
