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
    configure_parallelism, registry::modality_limit_override, vision::PreProcessorConfig,
    FrameSampling, MediaConnector, MediaConnectorConfig, MediaConnectorError, MediaContentPart,
    Modality, ModelMetadata, ModelRegistry, MultiModalError, Parallelism, VisionProcessorRegistry,
    POOL_THREADS_ENV,
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
use crate::{routers::grpc::proto_wrapper::vllm_mm_identity, worker::http_client::build_client};

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
    /// The name the model is served under, for messages an operator reads;
    /// `model_id` is often a loader's cache directory.
    pub served_model_name: Option<String>,
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
    /// The engine's own per-prompt media limits (vLLM's
    /// `--limit-mm-per-prompt`) by modality: the ceiling the pipeline's own
    /// limits tighten but never loosen. Empty when they are not known.
    pub engine_item_limits: HashMap<Modality, usize>,
    /// Cap on an inline (`data:`) item's decoded size; fetched items are
    /// capped by the connector's own limits.
    pub max_item_bytes: Option<usize>,
    pub allowed_domains: Option<Vec<String>>,
    pub fetch_timeout: Duration,
    /// The engine's own video frame budget: its `--media-io-kwargs`
    /// `video.num_frames` when set (`0` for every frame), else its loader's
    /// default; `None` when unknown. A spec that samples the way the engine's
    /// loader does takes it in place of its own constant.
    pub video_frame_budget: Option<usize>,
    /// A sampling rule of the engine's loader that the pipeline cannot
    /// follow, as the launcher names it (`--media-io-kwargs video.fps=2`,
    /// `VLLM_VIDEO_LOADER_BACKEND=opencv_dynamic`); `None` when the loader
    /// samples as the pipeline expects. Refused at construction for a spec
    /// that samples the way the loader does (see `loader_rule_conflict`).
    pub video_loader_rule: Option<String>,
}

/// Whether the engine's loader rule the launcher reported (`video.fps`, a
/// loader other than the default) conflicts with the spec: a spec that samples
/// the way the loader does would then sample a clip differently from the
/// engine's own server, so it is refused with the rule named; the rate-based
/// samplers never followed the loader and keep their rule, so the engine
/// starts as before.
fn loader_rule_conflict(
    rule: Option<&str>,
    spec_name: &str,
    sampling: FrameSampling,
) -> Option<String> {
    let rule = rule?;
    matches!(sampling, FrameSampling::UpTo { .. }).then(|| {
        format!(
            "{rule} changes how the engine's loader samples a video, and the {spec_name} \
             pipeline samples the way the loader does; drop it or use --mm-processor \
             inprocess or redis"
        )
    })
}

/// What set a modality's effective per-request item limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItemLimitSource {
    /// The engine's `--limit-mm-per-prompt`.
    Engine,
    /// The servicer's `--mm-max-items`.
    Flag,
    /// `SMG_<MODALITY>_MAX_COUNT` in the environment.
    Env,
    /// The model spec's declared limit.
    Spec,
}

impl ItemLimitSource {
    fn describe(self, modality: Modality) -> String {
        match self {
            Self::Engine => "the engine's --limit-mm-per-prompt".to_string(),
            Self::Flag => "--mm-max-items".to_string(),
            Self::Env => format!(
                "SMG_{}_MAX_COUNT",
                modality.to_string().to_ascii_uppercase()
            ),
            Self::Spec => "the model spec".to_string(),
        }
    }
}

/// Whether this pipeline can be handed a video at all: the spec declares the
/// modality (a text-and-image derivative does not) and the effective limit
/// leaves room for one (`--limit-mm-per-prompt '{"video": 0}'`, image-only
/// serving, does not). Such a pipeline never samples a clip, so its engine's
/// loader rule cannot matter and a fleet-wide loader setting must not keep it
/// from starting.
fn serves_video(item_limits: &HashMap<Modality, ItemLimit>) -> bool {
    item_limits
        .get(&Modality::Video)
        .is_some_and(|limit| limit.limit > 0)
}

/// A modality's effective per-request item limit and what set it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ItemLimit {
    limit: usize,
    source: ItemLimitSource,
}

/// The per-request item limit of every modality the spec declares: the
/// pipeline's own limit (`--mm-max-items`, else the `SMG_*_MAX_COUNT`
/// environment override, else the spec's), never above the engine's
/// `--limit-mm-per-prompt` where that is known. The engine sizes its encoder
/// budget by its limit and its own server refuses above it, so this worker
/// must not accept more on its behalf.
fn effective_item_limits(
    spec_limits: &HashMap<Modality, usize>,
    engine_limits: &HashMap<Modality, usize>,
    max_items: Option<usize>,
    env_override: impl Fn(Modality) -> Option<usize>,
) -> HashMap<Modality, ItemLimit> {
    spec_limits
        .iter()
        .map(|(&modality, &spec_limit)| {
            let own = match (max_items, env_override(modality)) {
                (Some(limit), _) => ItemLimit {
                    limit,
                    source: ItemLimitSource::Flag,
                },
                (None, Some(limit)) => ItemLimit {
                    limit,
                    source: ItemLimitSource::Env,
                },
                (None, None) => ItemLimit {
                    limit: spec_limit,
                    source: ItemLimitSource::Spec,
                },
            };
            let effective = match engine_limits.get(&modality) {
                Some(&limit) if limit <= own.limit => ItemLimit {
                    limit,
                    source: ItemLimitSource::Engine,
                },
                _ => own,
            };
            (modality, effective)
        })
        .collect()
}

/// Refuse a request carrying more items of a modality than this worker
/// takes, before any fetch, naming the limit and what set it. A modality
/// without a limit here is left to the spec's own validation.
fn check_item_counts(
    items: &[WorkerMediaItem],
    limits: &HashMap<Modality, ItemLimit>,
) -> Result<(), WorkerMediaError> {
    let mut counts: Vec<(Modality, usize)> = Vec::new();
    for item in items {
        match counts
            .iter_mut()
            .find(|(modality, _)| *modality == item.modality)
        {
            Some((_, count)) => *count += 1,
            None => counts.push((item.modality, 1)),
        }
    }
    for (modality, count) in counts {
        let Some(limit) = limits.get(&modality) else {
            continue;
        };
        if count > limit.limit {
            return Err(WorkerMediaError::Invalid(format!(
                "media_refs carries {count} {modality} items, above this worker's limit of {} ({})",
                limit.limit,
                limit.source.describe(modality)
            )));
        }
    }
    Ok(())
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

/// Why the pipeline refuses a model it has no spec for, in the terms an
/// operator acts on: the model type and architectures from its config and
/// the served name first, then the families the pipeline supports and the
/// modes that would start the deployment; the loader's path (often a cache
/// directory under a streaming loader) last.
fn unsupported_model_message(
    settings: &WorkerMediaSettings,
    config: &serde_json::Value,
    supported: &[&str],
) -> String {
    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let architectures: Vec<&str> = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .map(|values| values.iter().filter_map(|value| value.as_str()).collect())
        .unwrap_or_default();
    let mut facts = Vec::new();
    if !architectures.is_empty() {
        facts.push(format!("architectures [{}]", architectures.join(", ")));
    }
    if let Some(served) = settings
        .served_model_name
        .as_deref()
        .filter(|served| *served != settings.model_id)
    {
        facts.push(format!("served as {served:?}"));
    }
    let facts = if facts.is_empty() {
        String::new()
    } else {
        format!(" ({})", facts.join(", "))
    };
    let mut supported: Vec<&str> = supported.to_vec();
    supported.sort_unstable();
    format!(
        "multimodal processing is not supported for model_type {model_type:?}{facts}; the smg \
         pipeline supports: {}; use --mm-processor inprocess or redis to process media with \
         the engine's own processors, or off (model path: {})",
        supported.join(", "),
        settings.model_id
    )
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
    item_limits: HashMap<Modality, ItemLimit>,
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
        let (spec_name, spec_limits, video_sampling) = {
            let adapter = RegistryTokenizer(tokenizer.as_ref());
            let metadata = ModelMetadata {
                model_id: &settings.model_id,
                tokenizer: &adapter,
                config: &loaded.config,
            };
            let spec = model_registry.lookup(&metadata).ok_or_else(|| {
                anyhow::anyhow!(unsupported_model_message(
                    &settings,
                    &loaded.config,
                    &model_registry.spec_names()
                ))
            })?;
            let limits = spec.modality_limits(&metadata).map_err(|error| {
                anyhow::anyhow!("reading the {} spec's media limits: {error}", spec.name())
            })?;
            (spec.name(), limits, spec.video_frame_sampling())
        };
        let item_limits = effective_item_limits(
            &spec_limits,
            &settings.engine_item_limits,
            settings.max_items,
            modality_limit_override,
        );
        if serves_video(&item_limits) {
            if let Some(message) = loader_rule_conflict(
                settings.video_loader_rule.as_deref(),
                spec_name,
                video_sampling,
            ) {
                anyhow::bail!(message);
            }
        }
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
        let client = build_client(
            || reqwest::Client::builder().timeout(Duration::from_secs(30)),
            false,
            "media HTTP client",
        )
        .map_err(anyhow::Error::msg)?;
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
            // The same numbers for the spec's validation of the plan, so
            // nothing downstream loosens what the engine takes.
            modality_limit_overrides: item_limits
                .iter()
                .map(|(&modality, limit)| (modality, limit.limit))
                .collect(),
            processing: MmProcessingMode::Worker,
            inflight: None,
            video_frame_budget: settings.video_frame_budget,
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
            item_limits,
        })
    }

    /// The model spec the pipeline resolved (`qwen3_vl`, `llama4`, ...).
    pub fn spec_name(&self) -> &'static str {
        self.spec_name
    }

    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    /// The per-request item limits in force and what set each, for the
    /// startup log: `image=8 (the engine's --limit-mm-per-prompt), ...`.
    pub fn item_limits_summary(&self) -> String {
        let mut limits: Vec<String> = self
            .item_limits
            .iter()
            .map(|(modality, limit)| {
                format!(
                    "{modality}={} ({})",
                    limit.limit,
                    limit.source.describe(*modality)
                )
            })
            .collect();
        limits.sort();
        limits.join(", ")
    }

    /// The engine's video frame budget the pipeline samples under, if known.
    pub fn video_frame_budget(&self) -> Option<usize> {
        self.components.video_frame_budget
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
        check_item_counts(items, &self.item_limits)?;
        let parts = items
            .iter()
            .enumerate()
            .map(|(index, item)| self.part(index, item))
            .collect::<Result<Vec<_>, _>>()?;
        let plan = MediaPlan::new(parts);
        let tokenizer = self.tokenizer.as_ref();
        // This worker holds requests to its own engine's limits through the
        // components' limit overrides; there is no fleet to consult here.
        let placeholders = prepare_placeholder_tokens(
            &plan,
            &self.model_id,
            tokenizer,
            &self.components,
            CONFIG_KEY,
            &self.model_dir,
            &HashMap::new(),
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
    use llm_tokenizer::MockTokenizer;

    use super::*;

    fn limits(pairs: &[(Modality, usize)]) -> HashMap<Modality, usize> {
        pairs.iter().copied().collect()
    }

    fn image_items(count: usize) -> Vec<WorkerMediaItem> {
        vec![
            WorkerMediaItem {
                modality: Modality::Image,
                url: "data:image/png;base64,AAAA".to_string(),
            };
            count
        ]
    }

    /// The engine's `--limit-mm-per-prompt` caps the pipeline's own limits
    /// (`--mm-max-items`, the environment override, the spec's), which keep
    /// their precedence below it; without an engine limit nothing changes.
    #[test]
    fn engine_limits_cap_the_pipelines_own_limits() {
        use ItemLimitSource::{Engine, Env, Flag, Spec};
        let spec = limits(&[(Modality::Image, 128), (Modality::Video, 8)]);
        let at = |limit, source| ItemLimit { limit, source };

        let engine = limits(&[(Modality::Image, 1), (Modality::Video, 2)]);
        let effective = effective_item_limits(&spec, &engine, None, |_| None);
        assert_eq!(effective[&Modality::Image], at(1, Engine));
        assert_eq!(effective[&Modality::Video], at(2, Engine));

        let engine = limits(&[(Modality::Image, 8)]);
        let effective = effective_item_limits(&spec, &engine, Some(3), |_| None);
        assert_eq!(effective[&Modality::Image], at(3, Flag));
        assert_eq!(effective[&Modality::Video], at(3, Flag));
        let effective = effective_item_limits(&spec, &engine, Some(16), |_| None);
        assert_eq!(effective[&Modality::Image], at(8, Engine));
        assert_eq!(effective[&Modality::Video], at(16, Flag));

        let env = |modality| (modality == Modality::Image).then_some(5);
        let effective = effective_item_limits(&spec, &engine, None, env);
        assert_eq!(effective[&Modality::Image], at(5, Env));
        assert_eq!(effective[&Modality::Video], at(8, Spec));
        let effective = effective_item_limits(&spec, &limits(&[(Modality::Image, 4)]), None, env);
        assert_eq!(effective[&Modality::Image], at(4, Engine));

        let effective = effective_item_limits(&spec, &HashMap::new(), None, |_| None);
        assert_eq!(effective[&Modality::Image], at(128, Spec));
        assert_eq!(effective[&Modality::Video], at(8, Spec));
    }

    #[test]
    fn over_limit_requests_are_refused_naming_the_limit() {
        let limits = HashMap::from([(
            Modality::Image,
            ItemLimit {
                limit: 1,
                source: ItemLimitSource::Engine,
            },
        )]);
        let error = check_item_counts(&image_items(9), &limits).unwrap_err();
        assert!(matches!(error, WorkerMediaError::Invalid(_)));
        let message = error.to_string();
        assert!(
            message.contains("9 image items")
                && message.contains("limit of 1")
                && message.contains("--limit-mm-per-prompt"),
            "{message}"
        );
        assert!(check_item_counts(&image_items(1), &limits).is_ok());
        // A modality without a limit here is the spec's to validate.
        let videos = vec![
            WorkerMediaItem {
                modality: Modality::Video,
                url: "data:video/mp4;base64,AAAA".to_string(),
            };
            3
        ];
        assert!(check_item_counts(&videos, &limits).is_ok());
    }

    /// End to end on a pipeline: nine images against an engine limit of one
    /// are refused before any fetch, with the limit in the message; one image
    /// goes on into the pipeline.
    #[tokio::test]
    async fn pipeline_refuses_more_images_than_the_engine_takes() {
        let dir = std::env::temp_dir().join(format!(
            "smg-worker-media-engine-limits-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join("config.json");
        std::fs::write(
            &config,
            r#"{"model_type": "llava", "image_token_index": 32000}"#,
        )
        .unwrap();
        let settings = WorkerMediaSettings {
            model_dir: dir.display().to_string(),
            served_model_name: None,
            model_id: dir.display().to_string(),
            pixel_format: PixelFormat::Normalized,
            encoder_dtype: "float32".to_string(),
            processor_kwargs: serde_json::Map::new(),
            max_items: None,
            engine_item_limits: HashMap::from([(Modality::Image, 1)]),
            max_item_bytes: None,
            allowed_domains: None,
            fetch_timeout: Duration::from_secs(1),
            video_frame_budget: None,
            video_loader_rule: None,
        };
        let pipeline = WorkerMediaPipeline::new(settings, Arc::new(MockTokenizer::new()))
            .await
            .unwrap();
        assert_eq!(
            pipeline.item_limits_summary(),
            "image=1 (the engine's --limit-mm-per-prompt)"
        );

        let error = pipeline
            .process(vec![1, 2, 3], &image_items(9), false)
            .await
            .unwrap_err();
        assert!(
            matches!(
                &error,
                WorkerMediaError::Invalid(message)
                    if message.contains("9 image items") && message.contains("limit of 1")
            ),
            "{error}"
        );
        // Within the limit the request reaches the pipeline proper (which the
        // mock tokenizer, lacking the placeholder token, then refuses).
        let error = pipeline
            .process(vec![1, 2, 3], &image_items(1), false)
            .await
            .unwrap_err();
        assert!(!error.to_string().contains("limit of"), "{error}");
        std::fs::remove_file(config).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

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

#[cfg(test)]
mod unsupported_model_tests {
    use llm_tokenizer::MockTokenizer;

    use super::*;

    /// A family without a spec is refused in the terms an operator acts on:
    /// the model type and architectures, the served name, the families the
    /// pipeline supports and the modes that would start the deployment; the
    /// loader's directory (a cache path under a streaming loader) comes last.
    #[tokio::test]
    async fn a_family_without_a_spec_is_refused_by_model_type_with_the_supported_families() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "quux", "architectures": ["QuuxForConditionalGeneration"]}"#,
        )
        .unwrap();
        let path = dir.path().to_str().unwrap().to_string();
        let settings = WorkerMediaSettings {
            model_dir: path.clone(),
            model_id: path.clone(),
            served_model_name: Some("m6".to_string()),
            pixel_format: PixelFormat::Normalized,
            encoder_dtype: "bfloat16".to_string(),
            processor_kwargs: serde_json::Map::new(),
            max_items: None,
            engine_item_limits: HashMap::new(),
            max_item_bytes: None,
            allowed_domains: None,
            fetch_timeout: Duration::from_secs(1),
            video_frame_budget: None,
            video_loader_rule: None,
        };
        let error = WorkerMediaPipeline::new(settings, Arc::new(MockTokenizer::default()))
            .await
            .err()
            .expect("no spec for this family");
        let message = format!("{error:#}");
        assert!(message.contains(r#"model_type "quux""#), "{message}");
        assert!(
            message.contains("QuuxForConditionalGeneration"),
            "{message}"
        );
        assert!(message.contains(r#"served as "m6""#), "{message}");
        for family in ["qwen3_vl", "kimi_k3", "llama4"] {
            assert!(message.contains(family), "{family} missing: {message}");
        }
        assert!(message.contains("--mm-processor inprocess"), "{message}");
        assert!(
            message.find("model_type").unwrap() < message.find(&path).unwrap(),
            "the path is the last detail: {message}"
        );
    }

    /// A served name that is the path itself is not repeated.
    #[test]
    fn the_served_name_is_omitted_when_it_is_the_path() {
        let settings = WorkerMediaSettings {
            model_dir: "/models/x".to_string(),
            model_id: "/models/x".to_string(),
            served_model_name: Some("/models/x".to_string()),
            pixel_format: PixelFormat::Normalized,
            encoder_dtype: "bfloat16".to_string(),
            processor_kwargs: serde_json::Map::new(),
            max_items: None,
            engine_item_limits: HashMap::new(),
            max_item_bytes: None,
            allowed_domains: None,
            fetch_timeout: Duration::from_secs(1),
            video_frame_budget: None,
            video_loader_rule: None,
        };
        let message = unsupported_model_message(
            &settings,
            &serde_json::json!({"model_type": "gemma4"}),
            &["qwen_vl", "llava"],
        );
        assert_eq!(
            message,
            "multimodal processing is not supported for model_type \"gemma4\"; the smg pipeline \
             supports: llava, qwen_vl; use --mm-processor inprocess or redis to process media \
             with the engine's own processors, or off (model path: /models/x)"
        );
    }

    /// A loader rule the pipeline cannot follow refuses only a spec that
    /// samples the way the loader does; the rate-based samplers never did.
    #[test]
    fn a_loader_rule_refuses_only_a_spec_that_samples_like_the_loader() {
        let loader = FrameSampling::UpTo { max_frames: 32 };
        let message = loader_rule_conflict(Some("--media-io-kwargs video.fps=2"), "gemma4", loader)
            .expect("the loader-style spec is refused");
        assert!(
            message.starts_with("--media-io-kwargs video.fps=2 changes"),
            "{message}"
        );
        assert!(
            message.contains("gemma4 pipeline samples the way the loader does"),
            "{message}"
        );
        assert!(message.contains("--mm-processor inprocess"), "{message}");
        for sampling in [FrameSampling::Even, FrameSampling::Interval] {
            assert_eq!(
                loader_rule_conflict(
                    Some("VLLM_VIDEO_LOADER_BACKEND=opencv_dynamic"),
                    "qwen3_vl",
                    sampling
                ),
                None,
                "{sampling:?} never followed the loader"
            );
        }
        assert_eq!(loader_rule_conflict(None, "gemma4", loader), None);
    }

    /// The loader rule matters only to a pipeline that can be handed a video:
    /// a spec without the modality or an engine limit of 0 never samples one.
    #[test]
    fn the_loader_rule_is_moot_without_video() {
        let at = |limit| {
            HashMap::from([(
                Modality::Video,
                ItemLimit {
                    limit,
                    source: ItemLimitSource::Spec,
                },
            )])
        };
        assert!(!serves_video(&HashMap::new()), "no video in the spec");
        assert!(!serves_video(&at(0)), "an engine limit of 0");
        assert!(serves_video(&at(1)));
    }
}
