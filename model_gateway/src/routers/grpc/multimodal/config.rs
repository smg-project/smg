//! Multimodal model configuration: the shared config-file registry and the
//! per-router component bundle (media connector + processor/model registries).

use std::{path::Path, sync::Arc};

use anyhow::{Context, Result};
use dashmap::DashMap;
use llm_multimodal::{
    MediaConnector, MediaConnectorConfig, ModelRegistry, PreProcessorConfig,
    VisionProcessorRegistry,
};
use tracing::{debug, warn};

use super::pixel_cache::{pixel_cache_from_env, PixelCache};

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

        let model_config = Arc::new(MultimodalModelConfig {
            config,
            preprocessor_config,
            video_preprocessor_config,
        });

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
}

impl MultimodalComponents {
    /// Create multimodal components with default registries and a reference
    /// to the shared `MultimodalConfigRegistry` owned by `AppContext`.
    pub fn new(config_registry: Arc<MultimodalConfigRegistry>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create reqwest client")?;
        let media_connector = MediaConnector::new(client, MediaConnectorConfig::default())
            .context("Failed to create MediaConnector")?;

        Ok(Self {
            media_connector: Arc::new(media_connector),
            vision_processor_registry: Arc::new(VisionProcessorRegistry::with_defaults()),
            model_registry: Arc::new(ModelRegistry::default()),
            config_registry,
            pixel_cache: pixel_cache_from_env(),
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
}
