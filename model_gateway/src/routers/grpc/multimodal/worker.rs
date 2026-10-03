//! The Router's media pipeline run worker-side. A vLLM gRPC servicer that
//! processes `media_refs` itself (`--mm-processor smg`) fetches, decodes,
//! preprocesses and expands the placeholders with the code the Router runs
//! for `--mm-processing router`, so the engine receives what the Router
//! would have sent, without the Router carrying the pixels.
//!
//! Only the pipeline is here; the Router's own concerns (worker selection,
//! shared-memory transport, metrics) stay with the Router. The servicer's
//! lifecycle owner implements its `MediaProcessor` over [`WorkerMediaPipeline`].

use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::Context;
use llm_multimodal::{
    configure_parallelism, vision::PreProcessorConfig, MediaConnector, MediaConnectorConfig,
    MediaConnectorError, MediaContentPart, Modality, ModelMetadata, ModelRegistry, MultiModalError,
    Parallelism, VisionProcessorRegistry, POOL_THREADS_ENV,
};
use llm_tokenizer::TokenizerTrait;
use openai_protocol::worker::MmProcessingMode;
use smg_grpc_client::vllm_proto as vllm;
use tracing::warn;

use super::{
    assemble::{assemble_vllm_batches_with, VllmAssembly},
    config::{MultimodalComponents, MultimodalConfigRegistry, MultimodalModelConfig},
    mm_settings,
    pixel_cache::pixel_cache_with_budget,
    plan::{prepare_placeholder_tokens, validate_rendered_media_anchors, MediaPlan},
    process::process_multimodal_plan,
    RegistryTokenizer,
};
use crate::routers::grpc::proto_wrapper::vllm_mm_identity;

/// A worker serves one model: the config registry's one key.
const CONFIG_KEY: &str = "worker";

/// The URL schemes the pipeline fetches: what the Router forwards by reference.
pub const SCHEMES: &str = "http,https,data";

/// How pixels are written for the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// Rescaled and normalized here, written in the engine's dtype: for an
    /// engine that applies no normalization of its own.
    Normalized,
    /// The image's own `0..=255` bytes (`uint8`): for an engine that rescales
    /// and normalizes on device (vLLM's `mm_device_do_normalize`), which
    /// would otherwise normalize twice.
    RawU8,
}

/// What a worker's pipeline is built from.
#[derive(Debug, Clone)]
pub struct WorkerMediaSettings {
    /// Directory (or Hub id) holding the model's `config.json` and
    /// preprocessor configs.
    pub model_dir: String,
    /// The id the model specs match on: the served model path or Hub id.
    pub model_id: String,
    pub pixel_format: PixelFormat,
    /// The engine's dtype (`bfloat16`, `float16`, `float32`); normalized
    /// pixels are written in it directly, so nothing downstream casts them.
    pub encoder_dtype: String,
    /// The engine's `mm_processor_kwargs`, laid over the preprocessor
    /// configs as its own processor would take them. A key the config has
    /// no field for is refused at construction.
    pub processor_kwargs: serde_json::Map<String, serde_json::Value>,
    /// Media items a request may carry per modality; `None` keeps the model
    /// spec's limits.
    pub max_items: Option<usize>,
    /// Cap on an inline (`data:`) item's decoded size; fetched items are
    /// capped by the connector's own limits.
    pub max_item_bytes: Option<usize>,
    pub allowed_domains: Option<Vec<String>>,
    pub fetch_timeout: Duration,
}

/// One `media_refs` item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerMediaItem {
    pub modality: Modality,
    pub url: String,
}

/// What the pipeline produced: the prompt with its anchors expanded, the
/// request's multimodal batches (the first is `mm_inputs`, the rest
/// `extra_mm_inputs`), and, when asked, the identity a PD prefill leg returns
/// so the decode leg is served without pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerMediaOutput {
    pub prompt_token_ids: Vec<u32>,
    pub batches: Vec<vllm::MultimodalInputs>,
    pub identity: Option<vllm::MediaIdentity>,
}

/// Why a request was refused or failed, by whose fault.
#[derive(Debug, thiserror::Error)]
pub enum WorkerMediaError {
    /// The caller's: a bad reference, a prompt the anchors do not fit, more
    /// media than the model takes.
    #[error("{0}")]
    Invalid(String),
    /// Not this worker's fault right now: a fetch that timed out or could not
    /// connect.
    #[error("{0}")]
    Unavailable(String),
    /// A pipeline failure.
    #[error("{0}")]
    Internal(String),
}

/// The pipeline for one served model.
pub struct WorkerMediaPipeline {
    components: MultimodalComponents,
    model_id: String,
    model_dir: String,
    tokenizer: Arc<dyn TokenizerTrait>,
    assembly: VllmAssembly,
    spec_name: &'static str,
    pixel_format: PixelFormat,
    max_item_bytes: Option<usize>,
}

impl WorkerMediaPipeline {
    /// Load the model's configs, resolve its spec and check the pixel format
    /// against it. Fails for a model the pipeline does not support, and for
    /// `RawU8` on a processor that produces normalized floats, not pixels.
    pub async fn new(
        settings: WorkerMediaSettings,
        tokenizer: Arc<dyn TokenizerTrait>,
    ) -> anyhow::Result<Self> {
        // The servicer runs requests concurrently already, so each request's
        // preprocessing stays on its own thread unless an operator sized a pool.
        let parallelism = if std::env::var_os(POOL_THREADS_ENV).is_some() {
            Parallelism::default_pool()
        } else {
            Parallelism::Inline
        };
        if let Err(in_force) = configure_parallelism(parallelism) {
            if in_force != parallelism {
                warn!(
                    ?in_force,
                    ?parallelism,
                    "multimodal preprocessing parallelism was configured before this pipeline"
                );
            }
        }
        super::report_jpeg_decoder();
        let config_registry = Arc::new(MultimodalConfigRegistry::new());
        let loaded = config_registry
            .get_or_load(CONFIG_KEY, &settings.model_dir)
            .await?;
        let loaded = match apply_processor_kwargs(&loaded, &settings.processor_kwargs)? {
            Some(overridden) => {
                let overridden = Arc::new(overridden);
                config_registry.insert(CONFIG_KEY.to_string(), overridden.clone());
                overridden
            }
            None => loaded,
        };
        let model_registry = Arc::new(ModelRegistry::default());
        let spec_name = {
            let adapter = RegistryTokenizer(tokenizer.as_ref());
            let metadata = ModelMetadata {
                model_id: &settings.model_id,
                tokenizer: &adapter,
                config: &loaded.config,
            };
            model_registry
                .lookup(&metadata)
                .map(|spec| spec.name())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "multimodal processing is not supported for model {}",
                        settings.model_id
                    )
                })?
        };
        let vision_processor_registry = Arc::new(VisionProcessorRegistry::with_defaults());
        if settings.pixel_format == PixelFormat::RawU8 {
            let model_type = loaded.config.get("model_type").and_then(|v| v.as_str());
            let emits_pixel_bytes = vision_processor_registry
                .find(&settings.model_id, model_type)
                .is_some_and(|processor| processor.emits_pixel_bytes());
            anyhow::ensure!(
                emits_pixel_bytes,
                "the {spec_name} processor produces normalized floats, not the raw pixels an \
                 engine that normalizes on device takes; start the engine with \
                 --mm-device-do-normalize=false or use --mm-processor inprocess"
            );
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("building the media HTTP client")?;
        let connector = MediaConnector::new(
            client,
            MediaConnectorConfig {
                allowed_domains: settings.allowed_domains.clone(),
                allowed_local_media_path: None,
                fetch_timeout: settings.fetch_timeout,
            },
        )
        .context("building the media connector")?;
        let components = MultimodalComponents {
            media_connector: Arc::new(connector),
            vision_processor_registry,
            model_registry,
            config_registry,
            pixel_cache: pixel_cache_with_budget(mm_settings().pixel_cache_mb.value),
            modality_limit_overrides: settings
                .max_items
                .map(|limit| HashMap::from([(Modality::Image, limit), (Modality::Video, limit)]))
                .unwrap_or_default(),
            processing: MmProcessingMode::Worker,
            inflight: None,
        };
        Ok(Self {
            components,
            model_id: settings.model_id,
            model_dir: settings.model_dir,
            tokenizer,
            assembly: VllmAssembly {
                encoder_dtype: settings.encoder_dtype,
                device_normalizes: settings.pixel_format == PixelFormat::RawU8,
                // In-process: the batches never leave this process as protos.
                shm_enabled: false,
                shm_min_bytes: usize::MAX,
            },
            spec_name,
            pixel_format: settings.pixel_format,
            max_item_bytes: settings.max_item_bytes,
        })
    }

    /// The model spec the pipeline resolved (`qwen3_vl`, `llama4`, ...).
    pub fn spec_name(&self) -> &'static str {
        self.spec_name
    }

    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    /// Process one request's references against its prompt, which carries
    /// one un-expanded anchor token per item as the Router sent it.
    pub async fn process(
        &self,
        prompt_token_ids: Vec<u32>,
        items: &[WorkerMediaItem],
        want_identity: bool,
    ) -> Result<WorkerMediaOutput, WorkerMediaError> {
        if items.is_empty() {
            return Err(WorkerMediaError::Invalid(
                "media_refs is set but carries no items".to_string(),
            ));
        }
        let parts = items
            .iter()
            .enumerate()
            .map(|(index, item)| self.part(index, item))
            .collect::<Result<Vec<_>, _>>()?;
        let plan = MediaPlan::new(parts);
        let tokenizer = self.tokenizer.as_ref();
        let placeholders = prepare_placeholder_tokens(
            &plan,
            &self.model_id,
            tokenizer,
            &self.components,
            CONFIG_KEY,
            &self.model_dir,
        )
        .await
        .map_err(|error| WorkerMediaError::Invalid(format!("{error:#}")))?;
        validate_rendered_media_anchors(&plan, &placeholders, tokenizer, &prompt_token_ids)
            .map_err(|error| WorkerMediaError::Invalid(format!("{error:#}")))?;
        let output = process_multimodal_plan(
            plan,
            &self.model_id,
            tokenizer,
            prompt_token_ids,
            &self.components,
            CONFIG_KEY,
            &self.model_dir,
        )
        .await
        .map_err(classify)?;
        let data = assemble_vllm_batches_with(output.intermediate, &self.assembly)
            .map_err(|error| WorkerMediaError::Internal(format!("{error:#}")))?;
        let (first, extras) = data.into_protos();
        let identity = want_identity.then(|| vllm::MediaIdentity {
            prompt_token_ids: output.expanded_token_ids.clone(),
            mm_inputs: vllm_mm_identity(&first),
            extra_mm_inputs: extras.iter().filter_map(vllm_mm_identity).collect(),
        });
        let mut batches = Vec::with_capacity(1 + extras.len());
        batches.push(first);
        batches.extend(extras);
        Ok(WorkerMediaOutput {
            prompt_token_ids: output.expanded_token_ids,
            batches,
            identity,
        })
    }

    /// The content part for one item, with the checks the Python servicer
    /// makes before fetching: a scheme this worker fetches, an inline payload
    /// under the cap, a modality the pipeline takes.
    fn part(
        &self,
        index: usize,
        item: &WorkerMediaItem,
    ) -> Result<MediaContentPart, WorkerMediaError> {
        let scheme = item
            .url
            .split_once(':')
            .map(|(scheme, _)| scheme.to_ascii_lowercase())
            .unwrap_or_default();
        if !SCHEMES.split(',').any(|allowed| allowed == scheme) {
            return Err(WorkerMediaError::Invalid(format!(
                "media_refs[{index}]: unsupported url scheme {scheme:?}; this worker fetches {SCHEMES}"
            )));
        }
        if let (Some(cap), Some(size)) = (self.max_item_bytes, data_url_payload_bytes(&item.url)) {
            if size > cap {
                return Err(WorkerMediaError::Invalid(format!(
                    "media_refs[{index}]: inline payload is {size} bytes, above the {cap}-byte cap \
                     (--mm-max-item-bytes)"
                )));
            }
        }
        Ok(match item.modality {
            Modality::Image => MediaContentPart::ImageUrl {
                url: item.url.clone(),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            },
            Modality::Video => MediaContentPart::VideoUrl {
                url: item.url.clone(),
                uuid: None,
                fps: None,
                max_long_side_pixel: None,
            },
            other => {
                return Err(WorkerMediaError::Invalid(format!(
                    "media_refs[{index}]: unsupported modality {other}"
                )))
            }
        })
    }
}

/// The decoded size of a `data:` URL's payload, as the Python servicer
/// estimates it (`data_url_payload_bytes`); `None` for any other URL.
fn data_url_payload_bytes(url: &str) -> Option<usize> {
    let rest = url.strip_prefix("data:")?;
    let (header, payload) = rest.split_once(',')?;
    if header.to_ascii_lowercase().contains(";base64") {
        let trimmed = payload.trim_end_matches('=');
        Some(trimmed.len() * 3 / 4)
    } else {
        Some(payload.len())
    }
}

/// The model's configs with `kwargs` laid over each preprocessor config, or
/// `None` when there are none. A key the config has no field for is refused:
/// the engine's own processor would have honoured it, this pipeline cannot.
fn apply_processor_kwargs(
    loaded: &MultimodalModelConfig,
    kwargs: &serde_json::Map<String, serde_json::Value>,
) -> anyhow::Result<Option<MultimodalModelConfig>> {
    if kwargs.is_empty() {
        return Ok(None);
    }
    let known = match serde_json::to_value(PreProcessorConfig::default()) {
        Ok(serde_json::Value::Object(fields)) => fields,
        _ => serde_json::Map::new(),
    };
    let mut unknown: Vec<&str> = kwargs
        .keys()
        .map(String::as_str)
        .filter(|key| !known.contains_key(*key))
        .collect();
    unknown.sort_unstable();
    anyhow::ensure!(
        unknown.is_empty(),
        "the smg media processor does not take mm_processor_kwargs {unknown:?}; drop them or \
         use --mm-processor inprocess"
    );
    let overlay = |config: &PreProcessorConfig| -> anyhow::Result<PreProcessorConfig> {
        let mut value =
            serde_json::to_value(config).context("serializing the preprocessor config")?;
        if let serde_json::Value::Object(fields) = &mut value {
            for (key, kwarg) in kwargs {
                fields.insert(key.clone(), kwarg.clone());
            }
        }
        serde_json::from_value(value)
            .context("mm_processor_kwargs do not fit the preprocessor config")
    };
    let mut overridden = loaded.clone();
    overridden.preprocessor_config = overlay(&loaded.preprocessor_config)?;
    overridden.video_preprocessor_config = loaded
        .video_preprocessor_config
        .as_ref()
        .map(overlay)
        .transpose()?;
    Ok(Some(overridden))
}

/// Whose fault a pipeline failure is, from the typed error in its chain: a
/// fetch that timed out or could not connect is the network's (retryable),
/// a bad reference or undecodable media the caller's, anything else ours.
fn classify(error: anyhow::Error) -> WorkerMediaError {
    let message = format!("{error:#}");
    for cause in error.chain() {
        if let Some(fetch) = cause.downcast_ref::<MediaConnectorError>() {
            return classify_fetch(fetch, message);
        }
        if let Some(media) = cause.downcast_ref::<MultiModalError>() {
            return match media {
                MultiModalError::Media(fetch) => classify_fetch(fetch, message),
                MultiModalError::Join(_) => WorkerMediaError::Internal(message),
                MultiModalError::UnsupportedContent(_) | MultiModalError::Validation(_) => {
                    WorkerMediaError::Invalid(message)
                }
            };
        }
    }
    WorkerMediaError::Internal(message)
}

fn classify_fetch(error: &MediaConnectorError, message: String) -> WorkerMediaError {
    match error {
        MediaConnectorError::Timeout(_) => WorkerMediaError::Unavailable(message),
        MediaConnectorError::Http(http) if http.is_timeout() || http.is_connect() => {
            WorkerMediaError::Unavailable(message)
        }
        MediaConnectorError::Blocking(_) => WorkerMediaError::Internal(message),
        _ => WorkerMediaError::Invalid(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_url_payload_size_is_estimated_like_the_python_servicer() {
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64,AAAA"),
            Some(3)
        );
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64,AAA="),
            Some(2)
        );
        assert_eq!(data_url_payload_bytes("data:text/plain,hello"), Some(5));
        assert_eq!(data_url_payload_bytes("https://example.com/x.png"), None);
    }

    #[test]
    fn fetch_failures_are_classified_by_fault() {
        let timeout = anyhow::Error::new(MultiModalError::Media(MediaConnectorError::Timeout(
            Duration::from_secs(1),
        )))
        .context("Failed to finalize multimodal tracker");
        assert!(matches!(
            classify(timeout),
            WorkerMediaError::Unavailable(_)
        ));
        let scheme = anyhow::Error::new(MediaConnectorError::UnsupportedScheme("ftp".to_string()));
        assert!(matches!(classify(scheme), WorkerMediaError::Invalid(_)));
        let other = anyhow::anyhow!("preprocess failed");
        assert!(matches!(classify(other), WorkerMediaError::Internal(_)));
    }
}

#[cfg(test)]
mod processor_kwargs_tests {
    use super::*;

    fn kwargs(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        match value {
            serde_json::Value::Object(map) => map,
            other => panic!("not an object: {other}"),
        }
    }

    #[test]
    fn processor_kwargs_lay_over_every_preprocessor_config() {
        let loaded = MultimodalModelConfig {
            config: serde_json::json!({"model_type": "qwen3_vl"}),
            preprocessor_config: PreProcessorConfig {
                max_pixels: Some(10),
                ..Default::default()
            },
            video_preprocessor_config: Some(PreProcessorConfig::default()),
        };
        assert!(apply_processor_kwargs(&loaded, &serde_json::Map::new())
            .unwrap()
            .is_none());

        let overridden = apply_processor_kwargs(
            &loaded,
            &kwargs(serde_json::json!({"max_pixels": 20, "min_pixels": 4})),
        )
        .unwrap()
        .unwrap();
        assert_eq!(overridden.preprocessor_config.max_pixels, Some(20));
        assert_eq!(overridden.preprocessor_config.min_pixels, Some(4));
        assert_eq!(
            overridden.video_preprocessor_config.unwrap().max_pixels,
            Some(20)
        );
    }

    /// A knob the engine's processor takes and this pipeline has no field
    /// for would silently change the pixels; it is refused by name.
    #[test]
    fn unknown_processor_kwargs_are_refused_by_name() {
        let loaded = MultimodalModelConfig {
            config: serde_json::json!({"model_type": "qwen3_vl"}),
            preprocessor_config: PreProcessorConfig::default(),
            video_preprocessor_config: None,
        };
        let error = apply_processor_kwargs(
            &loaded,
            &kwargs(serde_json::json!({"fps": 2, "max_pixels": 1, "num_frames": 8})),
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains(r#"["fps", "num_frames"]"#), "{message}");
        assert!(message.contains("--mm-processor inprocess"), "{message}");
    }
}
