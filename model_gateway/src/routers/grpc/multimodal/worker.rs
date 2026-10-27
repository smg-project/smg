//! The Router's media pipeline run worker-side. A Rust gRPC servicer that
//! processes media references itself fetches, decodes,
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
    configure_parallelism, data_url::data_url_payload_bytes, jpeg_turbo::TurboJpegPolicy,
    registry::modality_limit_override, vision::PreProcessorConfig, FrameSampling, MediaConnector,
    MediaConnectorConfig, MediaConnectorError, MediaContentPart, MediaErrorCode, MediaFault,
    Modality, ModelMetadata, ModelRegistry, MultiModalError, Parallelism, POOL_THREADS_ENV,
};
use llm_tokenizer::TokenizerTrait;
use openai_protocol::worker::MmProcessingMode;
use smg_grpc_client::{sglang_proto as sglang, vllm_proto as vllm, WorkerMediaFault};
use tracing::{info, warn};

use super::{
    assemble::{
        assemble_sglang_with_dtype, assemble_vllm_batches_with, into_single_batch, VllmAssembly,
    },
    capability::runtime_reads_format,
    config::{MultimodalComponents, MultimodalConfigRegistry, MultimodalModelConfig},
    encoder_budget::ItemOverEncoderBudget,
    media_verdict, mm_settings,
    pixel_cache::pixel_cache_with_budget,
    plan::{prepare_placeholder_tokens, validate_rendered_media_anchors, MediaPlan},
    process::process_multimodal_plan,
    RegistryTokenizer,
};
use crate::{
    routers::grpc::proto_wrapper::vllm_mm_identity,
    worker::{http_client::build_with, RuntimeType},
};

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
    /// Budget for one image or audio fetch by URL, connect to last byte.
    pub fetch_timeout: Duration,
    /// Budget for one video fetch by URL, connect to last byte.
    pub video_fetch_timeout: Duration,
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
    /// The engine's encoder budget: the most feature tokens it can place for
    /// one media item (vLLM's encoder-cache budget, the "budget of N tokens"
    /// of its start-up line); `None` when the launcher could not tell. An
    /// item above it is refused before it reaches the engine, which would
    /// never schedule it (see `super::encoder_budget`).
    pub encoder_budget: Option<usize>,
    /// What the pipeline does when libjpeg-turbo cannot be loaded, as the
    /// launcher resolved it: `Require` refuses to construct the pipeline, so
    /// the servicer does not start, rather than serve the fallback decoder's
    /// pixels. `None` leaves it to `SMG_MM_TURBOJPEG`, read as the router
    /// reads it, strictly (an unreadable value refuses the pipeline, never
    /// passes as the default), for a launcher without the setting or one
    /// that was not given it.
    pub turbojpeg: Option<TurboJpegPolicy>,
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
/// `extra_mm_inputs`), and, when asked, the identity a PD prefill side returns
/// so the decode side is served without pixels.
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
    /// The media the request names failed, by whose fault (see
    /// [`WorkerMediaFault`]): the request's own (a host that cannot be
    /// resolved or refuses the connection, its 4xx, a payload that is not the
    /// media it claims to be or is over a size cap) or its host's for now (a
    /// fetch that ran out of its budget, a 5xx, a broken transfer). Neither
    /// is this worker's; the Router answers the request's own a 400 and its
    /// host's a 502, under `code`.
    #[error("{message}")]
    Media {
        fault: WorkerMediaFault,
        /// How the public API names the failure (see [`MediaErrorCode`]).
        code: MediaErrorCode,
        message: String,
    },
    /// This worker's own: a pipeline failure, a decode task that died, a
    /// scratch file or decoder process that failed on this side (see
    /// [`MediaFault::Internal`]).
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
    sglang_identity_key: Option<String>,
    pixel_format: PixelFormat,
    max_item_bytes: Option<usize>,
    item_limits: HashMap<Modality, ItemLimit>,
    encoder_budget: Option<usize>,
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
        // The launcher's policy, else `SMG_MM_TURBOJPEG` read strictly: the
        // lazily resolved process settings would read an unreadable value as
        // the default, the one misreading this policy must not have.
        let turbojpeg = match settings.turbojpeg {
            Some(policy) => policy,
            None => super::settings::turbojpeg_policy_from_env()?,
        };
        super::enforce_jpeg_decoder_policy(turbojpeg)?;
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
        let (spec_name, spec_limits, video_sampling, vision_processors) = {
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
            anyhow::ensure!(
                runtime_reads_format(RuntimeType::Vllm, spec.encoder_input_format()),
                "the {} spec preprocesses media into a format vLLM does not read; serve this \
                 model's media with --mm-processor inprocess",
                spec.name()
            );
            let limits = spec.modality_limits(&metadata).map_err(|error| {
                anyhow::anyhow!("reading the {} spec's media limits: {error}", spec.name())
            })?;
            (
                spec.name(),
                limits,
                spec.video_frame_sampling(),
                loaded.vision_processors(spec),
            )
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
        // The processors of the modalities the pipeline serves are built here,
        // so a preprocessor config they cannot serve stops the engine's start
        // rather than failing its requests.
        if item_limits
            .get(&Modality::Image)
            .is_some_and(|limit| limit.limit > 0)
        {
            vision_processors.image()?;
        }
        if serves_video(&item_limits) {
            vision_processors.video()?;
        }
        if settings.pixel_format == PixelFormat::RawU8 {
            let emits_pixel_bytes = vision_processors
                .image()
                .is_ok_and(|processor| processor.emits_pixel_bytes());
            anyhow::ensure!(
                emits_pixel_bytes,
                "the {spec_name} processor produces normalized floats, not the raw pixels an \
                 engine that normalizes on device takes; start the engine with \
                 --mm-device-do-normalize=false or use --mm-processor inprocess"
            );
        }
        // The worker fetches the client-supplied media URLs the router
        // forwards: the engine's own allow-list bounds the hosts by name, and
        // the address policy refuses private origins unless the worker's
        // environment allows them (a worker has no flags of its own).
        let connector_config = MediaConnectorConfig {
            allowed_domains: settings.allowed_domains.clone(),
            allow_private_hosts: mm_settings().allow_private_media_hosts.value,
            allowed_local_media_path: None,
            fetch_timeout: settings.fetch_timeout,
            video_fetch_timeout: settings.video_fetch_timeout,
        };
        // The line shows the allow-list as the connector enforces it (trimmed,
        // a list with no names being none), not as it was spelled.
        info!(
            allowed_media_domains =
                ?llm_multimodal::effective_allowed_domains(connector_config.allowed_domains.as_deref()),
            allow_private_media_hosts = connector_config.allow_private_hosts,
            "media fetch policy for worker-side processing"
        );
        let connector = build_with(
            || crate::clients::http_client_builder().timeout(Duration::from_secs(30)),
            false,
            "media HTTP client",
            |client| MediaConnector::new(client, connector_config.clone()),
        )
        .map_err(anyhow::Error::msg)
        .context("building the media connector")?;
        let components = MultimodalComponents {
            media_connector: Arc::new(connector),
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
        // Only processors with a pixel-less Decode metadata contract advertise reuse.
        let sglang_identity_key = if matches!(spec_name, "glm53_flash" | "deepseek_v41") {
            let mut hash = blake3::Hasher::new();
            hash.update(b"sglang-media-identity-v1");
            let mut config = serde_json::json!({
                "image": loaded.preprocessor_config,
                "video": loaded.video_preprocessor_config,
                "dtype": settings.encoder_dtype,
                "format": format!("{:?}", settings.pixel_format),
                "frames": settings.video_frame_budget,
                "model": loaded.config,
                "jpeg": format!("{turbojpeg:?}"),
                "max_items": settings.max_items,
                "max_item_bytes": settings.max_item_bytes,
                "allowed_domains": settings.allowed_domains,
                "allow_private_hosts": mm_settings().allow_private_media_hosts.value,
                "effective_limits": item_limits.iter().map(|(m, n)| (format!("{m:?}"), n.limit)).collect::<std::collections::BTreeMap<_, _>>(),
                "encoder_budget": settings.encoder_budget,
                "limits": settings.engine_item_limits.iter().map(|(m, n)| (format!("{m:?}"), *n)).collect::<std::collections::BTreeMap<_, _>>(),
            });
            config.sort_all_objects();
            hash.update(config.to_string().as_bytes());
            let mut complete = true;
            for name in ["tokenizer.json", "tokenizer_config.json"] {
                match tokio::fs::read(std::path::Path::new(&settings.model_dir).join(name)).await {
                    Ok(bytes) => {
                        hash.update(&(bytes.len() as u64).to_le_bytes());
                        hash.update(&bytes);
                    }
                    Err(_) => {
                        complete = false;
                    }
                }
            }
            complete.then(|| hash.finalize().to_hex().to_string())
        } else {
            None
        };
        Ok(Self {
            sglang_identity_key,
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
            encoder_budget: settings.encoder_budget,
        })
    }

    /// Fingerprint for the supported pixel-less SGLang Decode contract.
    pub fn sglang_identity_key(&self) -> Option<&str> {
        self.sglang_identity_key.as_deref()
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

    /// The engine's encoder budget the pipeline holds each item to, if known.
    pub fn encoder_budget(&self) -> Option<usize> {
        self.encoder_budget
    }

    /// Process one request's references against its prompt, which carries
    /// one un-expanded anchor token per item as the Router sent it.
    pub async fn process(
        &self,
        prompt_token_ids: Vec<u32>,
        items: &[WorkerMediaItem],
        want_identity: bool,
    ) -> Result<WorkerMediaOutput, WorkerMediaError> {
        let output = self
            .process_inputs(prompt_token_ids, items, RuntimeType::Vllm)
            .await?;
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

    /// Run the shared fetch, sampling and token expansion pipeline, then
    /// encode SGLang's processor output rather than vLLM's feature contract.
    pub async fn process_sglang(
        &self,
        prompt_token_ids: Vec<u32>,
        items: &[WorkerMediaItem],
    ) -> Result<(Vec<u32>, sglang::MultimodalInputs), WorkerMediaError> {
        let output = self
            .process_inputs(prompt_token_ids, items, RuntimeType::Sglang)
            .await?;
        let dtype = self.assembly.encoder_dtype.clone();
        let span = tracing::Span::current();
        let inputs = tokio::task::spawn_blocking(move || {
            let _entered = span.enter();
            assemble_sglang_with_dtype(into_single_batch(output.intermediate, "SGLang")?, &dtype)
                .map(|data| data.into_proto())
        })
        .await
        .map_err(|error| WorkerMediaError::Internal(format!("media assembly task: {error}")))?
        .map_err(|error| WorkerMediaError::Invalid(format!("{error:#}")))?;
        Ok((output.expanded_token_ids, inputs))
    }

    /// Fetch and expand media using the destination engine's format and alpha policy.
    async fn process_inputs(
        &self,
        prompt_token_ids: Vec<u32>,
        items: &[WorkerMediaItem],
        runtime: RuntimeType,
    ) -> Result<super::MultimodalOutput, WorkerMediaError> {
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
            |format| runtime_reads_format(runtime, format),
        )
        .await
        .map_err(|error| WorkerMediaError::Invalid(format!("{error:#}")))?;
        validate_rendered_media_anchors(&plan, &placeholders, tokenizer, &prompt_token_ids)
            .map_err(|error| WorkerMediaError::Invalid(format!("{error:#}")))?;
        // Follow the destination engine's image loader. SGLang drops alpha,
        // while vLLM composites over white; model overrides still apply.
        let output = process_multimodal_plan(
            plan,
            &self.model_id,
            tokenizer,
            prompt_token_ids,
            &self.components,
            CONFIG_KEY,
            &self.model_dir,
            if runtime == RuntimeType::Vllm {
                self.assembly.image_wire_dtype()
            } else {
                None
            },
            super::transparency::runtime_transparency_policy(runtime),
            self.encoder_budget,
        )
        .await
        .map_err(classify)?;
        Ok(output)
    }

    /// The content part for one item, with the checks the Python servicer
    /// makes before fetching: a scheme this worker fetches, an inline payload
    /// under the cap, a modality the pipeline takes. The cap's message names
    /// the worker as the flag's owner: the router relays the refusal in its
    /// 400, and its own per-item input bound (`SMG_*_MAX_INPUT_BYTES`, 256 MiB
    /// by default) is checked earlier with a message of its own, so a reader
    /// of either knows which launcher to set.
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
                     (the worker's --mm-max-item-bytes)"
                )));
            }
        }
        Ok(match item.modality {
            Modality::Image => MediaContentPart::ImageUrl {
                url: item.url.clone(),
                detail: None,
                max_long_side_pixel: None,
                uuid: None,
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
    Ok(Some(MultimodalModelConfig::new(
        loaded.config.clone(),
        overlay(&loaded.preprocessor_config)?,
        loaded
            .video_preprocessor_config
            .as_ref()
            .map(overlay)
            .transpose()?,
    )))
}

/// Whose fault a pipeline failure is, from the typed error in its chain: a
/// fetch or decode failure is the media's, by the connector's own reading of
/// it (the request's, or its host's for now) and under the connector's code
/// (the verdict the router-side pipeline answers directly, see
/// `multimodal::processing_failed_response`); a bad part or a prompt the
/// anchors do not fit or an item above the engine's encoder budget the
/// caller's; anything else ours.
fn classify(error: anyhow::Error) -> WorkerMediaError {
    let message = format!("{error:#}");
    if error
        .chain()
        .any(|cause| cause.downcast_ref::<ItemOverEncoderBudget>().is_some())
    {
        return WorkerMediaError::Invalid(message);
    }
    if let Some(fetch) = media_verdict(&error) {
        return classify_fetch(fetch, message);
    }
    match error
        .chain()
        .find_map(|cause| cause.downcast_ref::<MultiModalError>())
    {
        Some(MultiModalError::UnsupportedContent(_) | MultiModalError::Validation(_)) => {
            WorkerMediaError::Invalid(message)
        }
        // A tracker task that died, or nothing typed at all.
        Some(MultiModalError::Join(_) | MultiModalError::Media(_)) | None => {
            WorkerMediaError::Internal(message)
        }
    }
}

fn classify_fetch(error: &MediaConnectorError, message: String) -> WorkerMediaError {
    let fault = match error.fault() {
        MediaFault::Client => WorkerMediaFault::Client,
        MediaFault::Transient => WorkerMediaFault::Transient,
        MediaFault::Internal => return WorkerMediaError::Internal(message),
    };
    WorkerMediaError::Media {
        fault,
        code: error.code(),
        message,
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

    /// The settings every pipeline test starts from: a model at `model_dir`
    /// served under its own path, normalized pixels in bfloat16, one-second
    /// fetch timeouts, and no limit, budget or override of any kind; a test
    /// names the fields it is about.
    pub(super) fn pipeline_settings(model_dir: &str) -> WorkerMediaSettings {
        WorkerMediaSettings {
            model_dir: model_dir.to_string(),
            model_id: model_dir.to_string(),
            served_model_name: None,
            pixel_format: PixelFormat::Normalized,
            encoder_dtype: "bfloat16".to_string(),
            processor_kwargs: serde_json::Map::new(),
            max_items: None,
            engine_item_limits: HashMap::new(),
            max_item_bytes: None,
            allowed_domains: None,
            fetch_timeout: Duration::from_secs(1),
            video_fetch_timeout: Duration::from_secs(1),
            video_frame_budget: None,
            video_loader_rule: None,
            encoder_budget: None,
            turbojpeg: None,
        }
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
            encoder_dtype: "float32".to_string(),
            engine_item_limits: limits(&[(Modality::Image, 1)]),
            encoder_budget: Some(192_000),
            ..pipeline_settings(&dir.display().to_string())
        };
        let pipeline = WorkerMediaPipeline::new(settings, Arc::new(MockTokenizer::new()))
            .await
            .unwrap();
        assert_eq!(
            pipeline.item_limits_summary(),
            "image=1 (the engine's --limit-mm-per-prompt)"
        );
        // The engine's encoder budget reaches the pipeline as given.
        assert_eq!(pipeline.encoder_budget(), Some(192_000));

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

    /// End to end on a pipeline, against the engine's encoder budget: an
    /// image the llava processor turns into 576 feature tokens is refused
    /// against a budget of 575, one token under it, with both numbers in the
    /// message, and admitted through the whole pipeline against a budget of
    /// 576, exactly at it (the admitted path is the one it was).
    #[tokio::test]
    async fn pipeline_refuses_an_item_above_the_engines_encoder_budget() {
        const PIXEL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
        // The processor registry resolves the family from the model id.
        let dir = tempfile::Builder::new()
            .prefix("llava-1.5-")
            .tempdir()
            .unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "llava", "image_token_index": 32000}"#,
        )
        .unwrap();
        let path = dir.path().to_str().unwrap().to_string();
        let settings = |encoder_budget| WorkerMediaSettings {
            encoder_dtype: "float32".to_string(),
            encoder_budget,
            ..pipeline_settings(&path)
        };
        let tokenizer = || Arc::new(MockTokenizer::new().with_tokens(&[("<image>", 32000)]));
        let items = vec![WorkerMediaItem {
            modality: Modality::Image,
            url: PIXEL.to_string(),
        }];

        let pipeline = WorkerMediaPipeline::new(settings(Some(575)), tokenizer())
            .await
            .unwrap();
        let error = pipeline
            .process(vec![1, 32000, 2], &items, false)
            .await
            .unwrap_err();
        assert!(
            matches!(
                &error,
                WorkerMediaError::Invalid(message)
                    if message.contains("image item 1 of 1 is 576 feature tokens")
                        && message.contains("575-token encoder budget")
            ),
            "{error}"
        );

        let pipeline = WorkerMediaPipeline::new(settings(Some(576)), tokenizer())
            .await
            .unwrap();
        let output = pipeline
            .process(vec![1, 32000, 2], &items, false)
            .await
            .unwrap();
        assert_eq!(output.prompt_token_ids.len(), 2 + 576);
        assert_eq!(
            output
                .prompt_token_ids
                .iter()
                .filter(|&&id| id == 32000)
                .count(),
            576
        );
        assert_eq!(output.batches.len(), 1);
        let (tokens, sglang) = pipeline
            .process_sglang(vec![1, 32000, 2], &items)
            .await
            .unwrap();
        assert_eq!(tokens, output.prompt_token_ids);
        assert_eq!(sglang.modalities, ["image"]);
        assert_eq!(sglang.im_token_id, Some(32000));
        assert!(
            sglang.image_data.is_empty(),
            "do not duplicate source media beside tensors"
        );
        assert_eq!(
            sglang.mm_placeholders,
            [sglang::PlaceholderRange {
                offset: 1,
                length: 576
            }]
        );
        assert_eq!(
            sglang.pixel_values.as_ref().unwrap().shape,
            output.batches[0].pixel_values.as_ref().unwrap().shape
        );
        // PIXEL stores half-transparent blue. SGLang drops alpha, so its
        // tensor must equal opaque blue, while vLLM composites it over white.
        let mut opaque = items.clone();
        opaque[0].url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC".into();
        let (_, opaque_sglang) = pipeline
            .process_sglang(vec![1, 32000, 2], &opaque)
            .await
            .unwrap();
        let pixels = &sglang.pixel_values.as_ref().unwrap().data;
        assert_eq!(pixels, &opaque_sglang.pixel_values.unwrap().data);
        let Some(vllm::tensor_data::Payload::Inline(vllm_pixels)) =
            &output.batches[0].pixel_values.as_ref().unwrap().payload
        else {
            panic!("expected inline image tensor")
        };
        assert_ne!(pixels, vllm_pixels);
    }

    /// Build an isolated processor registry with explicit model tokens.
    async fn vision_test_pipeline(
        config: &str,
        tokens: &[(&str, u32)],
    ) -> (tempfile::TempDir, WorkerMediaPipeline) {
        let dir = tempfile::Builder::new()
            .prefix("vision-pipeline-")
            .tempdir()
            .unwrap();
        std::fs::write(dir.path().join("config.json"), config).unwrap();
        let path = dir.path().to_str().unwrap().to_string();
        let settings = WorkerMediaSettings {
            model_dir: path.clone(),
            model_id: path,
            served_model_name: None,
            pixel_format: PixelFormat::Normalized,
            encoder_dtype: "float32".into(),
            processor_kwargs: serde_json::Map::new(),
            max_items: None,
            engine_item_limits: HashMap::new(),
            max_item_bytes: None,
            allowed_domains: None,
            fetch_timeout: Duration::from_secs(1),
            video_fetch_timeout: Duration::from_secs(1),
            video_frame_budget: None,
            video_loader_rule: None,
            encoder_budget: None,
            turbojpeg: None,
        };
        let tokenizer = Arc::new(MockTokenizer::new().with_tokens(tokens));
        let pipeline = WorkerMediaPipeline::new(settings, tokenizer).await.unwrap();
        (dir, pipeline)
    }

    #[tokio::test]
    /// Compare BF16 pixels with an independent nearest-even reference and unchanged metadata.
    async fn sglang_worker_bfloat16_rounds_once_and_preserves_metadata() {
        use base64::{engine::general_purpose::STANDARD, Engine};
        let (_dir, mut pipeline) = vision_test_pipeline(
            r#"{"model_type":"glm53_flash","image_token_id":32000}"#,
            &[
                ("<|image|>", 32000),
                ("<|begin_of_image|>", 32001),
                ("<|end_of_image|>", 32002),
            ],
        )
        .await;
        let image = image::RgbImage::from_fn(56, 56, |x, y| {
            image::Rgb([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8])
        });
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        let items = [WorkerMediaItem {
            modality: Modality::Image,
            url: format!(
                "data:image/png;base64,{}",
                STANDARD.encode(bytes.into_inner())
            ),
        }];
        let (float_tokens, float) = pipeline
            .process_sglang(vec![1, 32000, 2], &items)
            .await
            .unwrap();
        pipeline.assembly.encoder_dtype = "bfloat16".into();
        let (bf_tokens, bf) = pipeline
            .process_sglang(vec![1, 32000, 2], &items)
            .await
            .unwrap();
        assert_eq!(float_tokens, bf_tokens);
        assert_eq!(float.model_specific_tensors, bf.model_specific_tensors);
        assert_eq!(float.mm_placeholders, bf.mm_placeholders);
        let float = float.pixel_values.unwrap();
        let bf = bf.pixel_values.unwrap();
        assert_eq!(float.dtype, "float32");
        assert_eq!(bf.dtype, "bfloat16");
        assert_eq!(float.shape, bf.shape);
        assert_eq!(float.data.len(), bf.data.len() * 2);
        // Independent IEEE round-to-nearest-even reference for these finite values.
        for (source, actual) in float
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .zip(bf.data.as_chunks::<2>().0)
        {
            let bits = u32::from_le_bytes(*source);
            let expected = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
            assert_eq!(u16::from_le_bytes(*actual), expected);
        }
        pipeline.assembly.encoder_dtype = "int8".into();
        assert!(pipeline
            .process_sglang(vec![1, 32000, 2], &items)
            .await
            .unwrap_err()
            .to_string()
            .contains("media dtype"));
    }

    #[tokio::test]
    #[ignore = "requires ffmpeg and ffprobe; run explicitly for SGLang video validation"]
    /// Check real video sampling, timestamps and temporal patch offsets across engines.
    async fn sglang_glm_video_preserves_tensors_and_temporal_patch_offsets() {
        use base64::{engine::general_purpose::STANDARD, Engine};
        let (_dir, pipeline) = vision_test_pipeline(
            r#"{"model_type":"glm53_flash","image_token_id":32000}"#,
            &[
                ("<|image|>", 32000),
                ("<|begin_of_image|>", 32001),
                ("<|end_of_image|>", 32002),
                ("<|video|>", 32003),
                ("0.0", 32004),
                ("seconds", 32005),
            ],
        )
        .await;
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../crates/multimodal/tests/fixtures/videos/testsrc_128x96_8fps_1s.mp4"
        ));
        let items = [WorkerMediaItem {
            modality: Modality::Video,
            url: format!("data:video/mp4;base64,{}", STANDARD.encode(bytes)),
        }];
        let (tokens, sglang) = pipeline
            .process_sglang(vec![1, 32003, 2], &items)
            .await
            .unwrap();
        let vllm = pipeline
            .process(vec![1, 32003, 2], &items, false)
            .await
            .unwrap();
        assert_eq!(tokens, vllm.prompt_token_ids);
        assert_eq!(sglang.modalities, ["video"]);
        assert_eq!(sglang.im_token_id, Some(32000));
        assert!(!tokens.contains(&32003));
        assert!(
            tokens.windows(2).any(|pair| pair == [32004, 32005]),
            "keep the frame timestamp after its vision group"
        );
        assert!(sglang.video_data.is_empty());
        let grid = &sglang.model_specific_tensors["video_grid_thw"];
        let dims: Vec<i64> = grid
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect();
        let expected = (dims[0] * dims[1] * dims[2] / 4) as u32;
        assert_eq!(
            sglang.mm_placeholders.iter().map(|p| p.length).sum::<u32>(),
            expected
        );
        assert_eq!(sglang.mm_placeholders.len(), dims[0] as usize);
        for p in &sglang.mm_placeholders {
            assert!(tokens[p.offset as usize..(p.offset + p.length) as usize]
                .iter()
                .all(|&t| t == 32000));
        }
        let Some(vllm::tensor_data::Payload::Inline(pixels)) = vllm.batches[0]
            .pixel_values
            .as_ref()
            .unwrap()
            .payload
            .as_ref()
        else {
            panic!("worker pipeline must return inline tensors")
        };
        assert_eq!(&sglang.pixel_values.as_ref().unwrap().data, pixels);
    }

    #[tokio::test]
    async fn pipeline_caps_an_inline_payload_whatever_the_scheme_case() {
        let dir = std::env::temp_dir().join(format!(
            "smg-worker-media-item-bytes-{}",
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
            encoder_dtype: "float32".to_string(),
            max_item_bytes: Some(4),
            ..pipeline_settings(&dir.display().to_string())
        };
        let pipeline = WorkerMediaPipeline::new(settings, Arc::new(MockTokenizer::new()))
            .await
            .unwrap();

        for url in [
            "data:image/png;base64,AAAAAAAA",
            "DATA:image/png;base64,AAAAAAAA",
        ] {
            let items = vec![WorkerMediaItem {
                modality: Modality::Image,
                url: url.to_string(),
            }];
            let error = pipeline
                .process(vec![1, 2, 3], &items, false)
                .await
                .unwrap_err();
            assert!(
                matches!(
                    &error,
                    WorkerMediaError::Invalid(message)
                        if message.contains("inline payload is 6 bytes")
                            && message.contains("(the worker's --mm-max-item-bytes)")
                ),
                "{url}: {error}"
            );
        }
        std::fs::remove_file(config).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn data_url_payload_size_is_estimated_like_the_python_servicer() {
        // The shared estimator counts base64 padding, as the servicer does.
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64,AAAA"),
            Some(3)
        );
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64,AAA="),
            Some(3)
        );
        assert_eq!(data_url_payload_bytes("DATA:text/plain,hello"), Some(5));
        assert_eq!(data_url_payload_bytes("https://example.com/x.png"), None);
    }

    /// A fetch that ran out of its budget is the media host's fault for now,
    /// a reference the request got wrong the request's, neither the worker's,
    /// both fetch failures to the client; a clip over the decoded-size cap is
    /// the request's own under the size code; a scratch file that failed on
    /// this side and a decoder this side cannot start are the worker's own,
    /// as is a failure with no connector error in its chain.
    #[test]
    fn fetch_failures_are_classified_by_fault() {
        let timeout = anyhow::Error::new(MultiModalError::Media(MediaConnectorError::Timeout {
            url: "https://media.example/clip.mp4".to_string(),
            budget: Duration::from_secs(1),
        }))
        .context("Failed to finalize multimodal tracker");
        assert!(matches!(
            classify(timeout),
            WorkerMediaError::Media {
                fault: WorkerMediaFault::Transient,
                code: MediaErrorCode::FetchFailed,
                ..
            }
        ));
        let scheme = anyhow::Error::new(MediaConnectorError::UnsupportedScheme("ftp".to_string()));
        assert!(matches!(
            classify(scheme),
            WorkerMediaError::Media {
                fault: WorkerMediaFault::Client,
                code: MediaErrorCode::FetchFailed,
                ..
            }
        ));
        let over_cap = anyhow::Error::new(MultiModalError::Media(
            MediaConnectorError::DecodedTooLarge {
                media: "video RGB",
                bytes: 2_208_384_000,
                limit: 1_073_741_824,
                knob: "SMG_VIDEO_MAX_DECODED_BYTES",
            },
        ))
        .context("Failed to finalize multimodal tracker");
        assert!(matches!(
            classify(over_cap),
            WorkerMediaError::Media {
                fault: WorkerMediaFault::Client,
                code: MediaErrorCode::TooLarge,
                ..
            }
        ));
        let part = anyhow::Error::new(MultiModalError::UnsupportedContent("image_embeds"));
        assert!(matches!(classify(part), WorkerMediaError::Invalid(_)));
        let scratch = anyhow::Error::new(MultiModalError::Media(MediaConnectorError::DecodeIo(
            std::io::Error::other("no space left on device"),
        )))
        .context("Failed to finalize multimodal tracker");
        assert!(matches!(classify(scratch), WorkerMediaError::Internal(_)));
        let decoder = anyhow::Error::new(MultiModalError::Media(
            MediaConnectorError::decoder_unavailable("ffmpeg executable not found".to_string()),
        ))
        .context("Failed to finalize multimodal tracker");
        assert!(matches!(classify(decoder), WorkerMediaError::Internal(_)));
        let other = anyhow::anyhow!("preprocess failed");
        assert!(matches!(classify(other), WorkerMediaError::Internal(_)));
        // An item above the engine's encoder budget is the caller's, with the
        // item's size and the budget in the message, at any depth of the chain.
        let over_budget = anyhow::Error::new(ItemOverEncoderBudget {
            modality: Modality::Video,
            position: 1,
            count: 1,
            feature_tokens: 192_001,
            sampled_frames: Some(900),
            budget: 192_000,
        })
        .context("Failed to process the plan");
        assert!(matches!(
            classify(over_budget),
            WorkerMediaError::Invalid(message)
                if message.contains("192001 feature tokens (900 sampled frames)")
                    && message.contains("192000-token encoder budget")
        ));
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
        let loaded = MultimodalModelConfig::new(
            serde_json::json!({"model_type": "qwen3_vl"}),
            PreProcessorConfig {
                max_pixels: Some(10),
                ..Default::default()
            },
            Some(PreProcessorConfig::default()),
        );
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
        let loaded = MultimodalModelConfig::new(
            serde_json::json!({"model_type": "qwen3_vl"}),
            PreProcessorConfig::default(),
            None,
        );
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

    use super::{tests::pipeline_settings, *};

    /// A spec whose encoder inputs vLLM does not read never preprocesses for
    /// this vLLM worker: the pipeline refuses to start.
    #[tokio::test]
    async fn a_spec_vllm_cannot_read_is_refused_at_startup() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "NemotronH_Nano_Omni_Reasoning_V3"}"#,
        )
        .unwrap();
        let path = dir.path().to_str().unwrap().to_string();
        let settings = pipeline_settings(&path);
        let error = WorkerMediaPipeline::new(settings, Arc::new(MockTokenizer::default()))
            .await
            .err()
            .expect("a spec vLLM cannot read");
        let message = format!("{error:#}");
        assert!(message.contains("format vLLM does not read"), "{message}");
        assert!(message.contains("--mm-processor inprocess"), "{message}");
    }

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
            served_model_name: Some("m6".to_string()),
            ..pipeline_settings(&path)
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
            served_model_name: Some("/models/x".to_string()),
            ..pipeline_settings("/models/x")
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
