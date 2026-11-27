//! Canonical multimodal request planning shared by protocol adapters, rendering,
//! and preprocessing.

use std::{collections::HashMap, fmt};

use anyhow::Result;
use llm_multimodal::{
    registry::{modality_limit_override, ModelRegistryError},
    EncoderInputFormat, FetchSourceKind, MediaContentPart, MediaPartOrder, Modality, ModelMetadata,
};
use llm_tokenizer::{traits::PromptEncoding, TokenizerTrait};
use openai_protocol::worker::MmProcessingMode;

use super::{
    config::MultimodalComponents,
    item_limits::{check_item_counts, effective_item_limits, limit_numbers, EngineItemLimit},
    RegistryTokenizer,
};

/// Ordered media extracted from an API request.
///
/// This is the single hand-off between protocol-specific parsing and the shared
/// multimodal pipeline. Text remains in the message representation used by the
/// chat template; media is kept here in authored order for fetching and count
/// validation.
#[derive(Debug, Clone, Default)]
pub(crate) struct MediaPlan {
    parts: Vec<MediaContentPart>,
    modalities: Vec<Modality>,
    counts: HashMap<Modality, usize>,
    /// The rate the request states for sampling its video clips: the default
    /// of this plan's clips in the gateway's own pipeline, ahead of the
    /// model's. It is no part's own rate, so it decides nothing about where
    /// the media is processed.
    request_video_fps: Option<f64>,
}

impl MediaPlan {
    pub(crate) fn new(parts: impl IntoIterator<Item = MediaContentPart>) -> Self {
        let mut plan = Self::default();
        for part in parts {
            let modality = match &part {
                MediaContentPart::ImageUrl { .. }
                | MediaContentPart::ImageBase64 { .. }
                | MediaContentPart::ImageData { .. } => Some(Modality::Image),
                MediaContentPart::ImageEmbeds { .. } => Some(Modality::ImageEmbeds),
                MediaContentPart::AudioUrl { .. }
                | MediaContentPart::AudioBase64 { .. }
                | MediaContentPart::AudioData { .. } => Some(Modality::Audio),
                MediaContentPart::VideoUrl { .. } | MediaContentPart::VideoData { .. } => {
                    Some(Modality::Video)
                }
                MediaContentPart::Text { .. } => None,
            };

            let Some(modality) = modality else {
                continue;
            };
            if !plan.modalities.contains(&modality) {
                plan.modalities.push(modality);
            }
            *plan.counts.entry(modality).or_default() += 1;
            plan.parts.push(part);
        }
        plan
    }

    /// Carry the rate the request states for sampling its video clips;
    /// `rate` is read only for a plan that holds a video part.
    pub(crate) fn with_request_video_fps(mut self, rate: impl FnOnce() -> Option<f64>) -> Self {
        if self.modalities.contains(&Modality::Video) {
            self.request_video_fps = rate();
        }
        self
    }

    /// The rate the request states for sampling its video clips, if it
    /// states one in range and the plan holds a clip.
    pub(crate) fn request_video_fps(&self) -> Option<f64> {
        self.request_video_fps
    }

    /// The plan's items of `modality` per source (`url`, `data_url`,
    /// `inline`): what the request carries, before any limit is applied.
    pub(crate) fn source_counts(&self, modality: Modality) -> [(&'static str, usize); 3] {
        let mut counts = [
            (FetchSourceKind::Url.as_str(), 0),
            (FetchSourceKind::DataUrl.as_str(), 0),
            (FetchSourceKind::InlineBytes.as_str(), 0),
        ];
        for part in &self.parts {
            let (of, source) = match part {
                MediaContentPart::ImageUrl { url, .. } => (Modality::Image, url_source(url)),
                MediaContentPart::ImageBase64 { .. } => (Modality::Image, BASE64_SOURCE),
                MediaContentPart::ImageData { .. } => (Modality::Image, 2),
                MediaContentPart::ImageEmbeds { .. } => (Modality::ImageEmbeds, 2),
                MediaContentPart::AudioUrl { url, .. } => (Modality::Audio, url_source(url)),
                MediaContentPart::AudioBase64 { .. } => (Modality::Audio, BASE64_SOURCE),
                MediaContentPart::AudioData { .. } => (Modality::Audio, 2),
                MediaContentPart::VideoUrl { url, .. } => (Modality::Video, url_source(url)),
                MediaContentPart::VideoData { .. } => (Modality::Video, 2),
                MediaContentPart::Text { .. } => continue,
            };
            if of == modality {
                counts[source].1 += 1;
            }
        }
        counts
    }

    /// The source class of the plan's media, read over every modality:
    /// `none` for a plan without media, `inline` when every item is a
    /// `data:` URL or carried bytes, `remote` when every item is a fetched
    /// URL, `hybrid` for a mix. The class is the request's, not one
    /// modality's: inline images beside a fetched clip are `hybrid`.
    pub(crate) fn source_class(&self) -> &'static str {
        let (mut inline, mut remote) = (0, 0);
        for &modality in self.modalities() {
            // The sources in the order `source_counts` writes them.
            let [(_, url), (_, data_url), (_, bytes)] = self.source_counts(modality);
            remote += url;
            inline += data_url + bytes;
        }
        match (inline > 0, remote > 0) {
            (false, false) => "none",
            (true, false) => "inline",
            (false, true) => "remote",
            (true, true) => "hybrid",
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    pub(crate) fn modalities(&self) -> &[Modality] {
        &self.modalities
    }

    pub(crate) fn count(&self, modality: Modality) -> usize {
        self.counts.get(&modality).copied().unwrap_or_default()
    }

    pub(crate) fn into_parts(self) -> Vec<MediaContentPart> {
        self.parts
    }

    pub(crate) fn parts(&self) -> &[MediaContentPart] {
        &self.parts
    }

    /// Whether every part is a plain image/audio/video URL without per-item
    /// processing hints, i.e. something a worker can fetch and process itself.
    pub(crate) fn is_forwardable(&self) -> bool {
        self.parts.iter().all(|part| {
            matches!(
                part,
                MediaContentPart::ImageUrl {
                    max_long_side_pixel: None,
                    ..
                } | MediaContentPart::ImageBase64 {
                    max_long_side_pixel: None,
                    ..
                } | MediaContentPart::AudioUrl { .. }
                    | MediaContentPart::AudioBase64 { .. }
                    | MediaContentPart::VideoUrl {
                        fps: None,
                        max_long_side_pixel: None,
                        ..
                    }
            )
        })
    }
}

/// Model-specific structural anchor strings keyed by modality.
///
/// String-format templates need the actual anchor string, while OpenAI-format
/// templates receive canonical `image` / `audio` / `video` parts. Keeping the
/// mapping typed prevents the former `image_placeholder` argument from being
/// accidentally reused for every modality.
#[derive(Debug, Clone, Default)]
pub(crate) struct PlaceholderTokens {
    tokens: HashMap<Modality, String>,
    /// Modalities whose anchor a vLLM worker can expand itself.
    worker_expandable: HashMap<Modality, bool>,
    /// Multi-token text the template writes in place of an anchor, with that anchor.
    rendered_markers: Vec<(String, String)>,
    /// The model matched no registry spec: the router knows no anchors and
    /// no limits for it, so it writes and checks none; its media can only go
    /// to a worker that processes media itself, whose engine owns the
    /// placeholders, their expansion and the per-prompt limits, as on the
    /// engine's own server.
    worker_only: bool,
}

impl PlaceholderTokens {
    /// For a model without a registry spec (see `worker_only`).
    pub(crate) fn worker_only() -> Self {
        Self {
            worker_only: true,
            ..Self::default()
        }
    }

    pub(crate) fn is_worker_only(&self) -> bool {
        self.worker_only
    }

    pub(crate) fn insert(&mut self, modality: Modality, token: String) {
        self.tokens.insert(modality, token);
    }

    pub(crate) fn get(&self, modality: Modality) -> Option<&str> {
        self.tokens.get(&modality).map(String::as_str)
    }

    pub(crate) fn add_rendered_marker(&mut self, marker: String, anchor: String) {
        self.rendered_markers.push((marker, anchor));
    }

    pub(crate) fn set_worker_expandable(&mut self, modality: Modality, expandable: bool) {
        self.worker_expandable.insert(modality, expandable);
    }

    /// Whether a vLLM worker can expand the rendered anchor for `modality`
    /// itself: the spec's opt-in, or every modality of a model without a
    /// spec, whose template renders the engine's own placeholders.
    pub(crate) fn worker_expandable(&self, modality: Modality) -> bool {
        self.worker_only
            || self
                .worker_expandable
                .get(&modality)
                .copied()
                .unwrap_or(false)
    }
}

/// Validate a multimodal request against the model spec and resolve the
/// structural anchors for its active modalities in one config/spec lookup.
///
/// `engine_limits` are the per-request media limits the model's workers
/// advertise for their engine ([`super::engine_item_limits`]): the request is
/// held to them here, before any fetch, as the engine's own front end would
/// (it never sees the precomputed inputs the router sends).
///
/// A model without a spec is refused only where the router would process
/// the media itself (`--mm-processing router`); otherwise its request is
/// marked worker-only and goes on to a worker that processes media itself
/// (see [`PlaceholderTokens::worker_only`]). A spec whose encoder input
/// format the engines that would receive the request do not read
/// (`reads_format` false) counts as no spec.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is an independent input of the one lookup"
)]
pub(crate) async fn prepare_placeholder_tokens(
    plan: &MediaPlan,
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
    engine_limits: &HashMap<Modality, EngineItemLimit>,
    reads_format: impl Fn(EncoderInputFormat) -> bool,
) -> Result<PlaceholderTokens> {
    prepare_placeholder_tokens_with_env(
        plan,
        model_id,
        tokenizer,
        components,
        tokenizer_id,
        tokenizer_source,
        engine_limits,
        reads_format,
        modality_limit_override,
    )
    .await
}

/// A failure of the placeholder preparation that the request itself causes
/// and its client can fix: more items than a limit takes, a modality or an
/// image override the model's spec does not take, media for a model with no
/// media spec on the router. The stage that finds it in a failure's chain
/// writes one WARN and marks its answer; every other failure of the
/// preparation keeps the stage's ERROR line, the ones a site marks as the
/// deployment's own ([`DeploymentFault`]) among them.
#[derive(Debug)]
pub(crate) struct RequestRefusal(String);

impl RequestRefusal {
    /// Whether `error` is this refusal, at any depth of its chain.
    pub(crate) fn is(error: &anyhow::Error) -> bool {
        error.chain().any(|cause| cause.is::<Self>())
    }

    /// `message` as the failure for a registry `error`: this refusal for a
    /// variant the request brings (a modality the spec does not take, more
    /// items than its limit), a deployment fault for every other variant: one
    /// about the model's files, or one no request can bring (a modality
    /// counted twice, when the plan counts each modality once).
    fn of_registry(error: &ModelRegistryError, message: String) -> anyhow::Error {
        if matches!(
            error,
            ModelRegistryError::UnsupportedModality { .. }
                | ModelRegistryError::ModalityLimitExceeded { .. }
        ) {
            Self(message).into()
        } else {
            DeploymentFault(message).into()
        }
    }
}

impl fmt::Display for RequestRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RequestRefusal {}

/// A failure of the placeholder preparation that its site knows as the
/// deployment's own, whatever the request holds: a `config.json` this
/// gateway holds and cannot read or parse, a spec's media limits or its
/// placeholder token that the model's files do not resolve, a vocabulary
/// without the token. The stage that finds it in a failure answers in the
/// server's class ([`super::preparation_fault_response`]). A failure with
/// neither this mark nor [`RequestRefusal`] keeps the 400 it had. That
/// includes unclassified failures while resolving the model's configuration
/// directory, such as an absent or inaccessible configuration and local
/// client or cache setup failures.
#[derive(Debug)]
pub(crate) struct DeploymentFault(pub(crate) String);

impl DeploymentFault {
    /// Whether `error` carries this mark: as its root, or as the context a
    /// site put on a cause it keeps.
    pub(crate) fn is(error: &anyhow::Error) -> bool {
        error.downcast_ref::<Self>().is_some()
    }
}

impl fmt::Display for DeploymentFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DeploymentFault {}

/// `prepare_placeholder_tokens` with the deployment-wide limit override (the
/// `SMG_*_MAX_COUNT` variables, read once per process) passed in: a test
/// passes `|_| None` and holds the spec default under test whatever the shell
/// it runs in exports.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is an independent input of the one lookup"
)]
async fn prepare_placeholder_tokens_with_env(
    plan: &MediaPlan,
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
    engine_limits: &HashMap<Modality, EngineItemLimit>,
    reads_format: impl Fn(EncoderInputFormat) -> bool,
    env_override: impl Fn(Modality) -> Option<usize>,
) -> Result<PlaceholderTokens> {
    anyhow::ensure!(!plan.is_empty(), "multimodal media plan is empty");
    let model_config = components
        .config_registry
        .get_or_load(tokenizer_id, tokenizer_source)
        .await?;
    let registry_tokenizer = RegistryTokenizer(tokenizer);
    let metadata = ModelMetadata {
        model_id,
        tokenizer: &registry_tokenizer,
        config: &model_config.config,
    };
    let Some(spec) = components
        .model_registry
        .lookup(&metadata)
        .filter(|spec| reads_format(spec.encoder_input_format()))
    else {
        anyhow::ensure!(
            components.processing != MmProcessingMode::Router,
            RequestRefusal(format!(
                "multimodal not supported for model: {model_id} on the router (no media spec \
                 for its family); a vLLM worker with --mm-processor inprocess|redis processes \
                 its media itself under --mm-processing worker"
            ))
        );
        return Ok(PlaceholderTokens::worker_only());
    };
    // Reject request overrides before the media connector can resize pixels.
    if spec.name() == "inkling2" {
        anyhow::ensure!(
            !plan.parts.iter().any(|part| matches!(
                part,
                MediaContentPart::ImageUrl {
                    max_long_side_pixel: Some(_),
                    ..
                }
            )),
            RequestRefusal(
                "Inkling2 requires original image dimensions. max_long_side_pixel is unsupported"
                    .to_string()
            )
        );
    }
    let spec_limits = spec.modality_limits(&metadata).map_err(|error| {
        DeploymentFault(format!(
            "reading the {} spec's media limits: {error}",
            spec.name()
        ))
    })?;
    // The router's own limits, capped by the engine's where its workers
    // advertise one; the refusal names the limit and what set it.
    let limits = effective_item_limits(
        &spec_limits,
        engine_limits,
        &components.modality_limit_overrides,
        env_override,
    );
    check_item_counts(plan, &limits).map_err(RequestRefusal)?;
    let requested = plan
        .modalities()
        .iter()
        .map(|&modality| (modality, plan.count(modality)))
        .collect::<Vec<_>>();
    // The same numbers for the spec's own validation, so nothing downstream
    // loosens what the engine takes.
    spec.validate_media_request_with_limits(&metadata, &requested, &limit_numbers(&limits))
        .map_err(|error| {
            RequestRefusal::of_registry(
                &error,
                format!("invalid media request for model {}: {error}", spec.name()),
            )
        })?;
    let mut placeholders = PlaceholderTokens::default();
    for &modality in plan.modalities() {
        let token = spec
            .placeholder_token_for(&metadata, modality)
            // The validation above has accepted the modality, so a failure
            // here is about the spec or the model's files, never the request.
            .map_err(|error| {
                DeploymentFault(format!(
                    "model {} supports {modality} but its placeholder token could not be resolved: {error}",
                    spec.name()
                ))
            })?;
        anyhow::ensure!(
            tokenizer.token_to_id(&token).is_some(),
            DeploymentFault(format!(
                "{modality} placeholder token '{token}' is missing from the tokenizer vocabulary"
            ))
        );
        if let Some(marker) = spec.rendered_marker_for(modality) {
            placeholders.add_rendered_marker(marker.to_string(), token.clone());
        }
        placeholders.insert(modality, token);
        placeholders.set_worker_expandable(modality, spec.worker_expandable(modality));
    }

    Ok(placeholders)
}

/// Resolve the media-part ordering for a model from the same registry that owns
/// its placeholder/prompt logic: where media goes relative to text when the
/// template takes one string per message (a template that takes parts places
/// them itself, in the request's order). Falls back to the engine's media-first
/// placement when the model matches no spec.
pub(crate) async fn resolve_media_part_order(
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
) -> MediaPartOrder {
    let model_config = match components
        .config_registry
        .get_or_load(tokenizer_id, tokenizer_source)
        .await
    {
        Ok(config) => config,
        Err(_) => return MediaPartOrder::MediaFirst,
    };
    let registry_tokenizer = RegistryTokenizer(tokenizer);
    let metadata = ModelMetadata {
        model_id,
        tokenizer: &registry_tokenizer,
        config: &model_config.config,
    };
    components
        .model_registry
        .lookup(&metadata)
        .map(|spec| spec.media_part_order())
        .unwrap_or(MediaPartOrder::MediaFirst)
}

/// The prompt text to tokenize: a text-rendered prompt gets each template
/// marker replaced by its anchor token (see
/// `ModelProcessorSpec::rendered_marker_for`); other prompts pass through.
pub(crate) fn anchor_rendered_markers(
    placeholders: Option<&PlaceholderTokens>,
    encoding: &PromptEncoding,
    text: String,
) -> String {
    match (placeholders, encoding) {
        (Some(placeholders), PromptEncoding::FromText) => placeholders
            .rendered_markers
            .iter()
            .fold(text, |text, (marker, anchor)| text.replace(marker, anchor)),
        _ => text,
    }
}

/// Verify a rendered/tokenized multimodal prompt contains exactly one
/// structural anchor for every planned media item before any media is fetched
/// or preprocessed.
///
/// This catches stale/custom templates, adapter omissions, and literal internal
/// anchors in user text at the cheapest point in the pipeline.
pub(crate) fn validate_rendered_media_anchors(
    plan: &MediaPlan,
    placeholders: &PlaceholderTokens,
    tokenizer: &dyn TokenizerTrait,
    token_ids: &[u32],
) -> Result<()> {
    if placeholders.is_worker_only() {
        // No spec, no anchors of the router's own: the worker's engine checks
        // its placeholders against the media it fetches, as its own server does.
        return Ok(());
    }
    for &modality in plan.modalities() {
        let token = placeholders
            .get(modality)
            .ok_or_else(|| anyhow::anyhow!("missing resolved {modality} placeholder token"))?;
        let token_id = tokenizer.token_to_id(token).ok_or_else(|| {
            anyhow::anyhow!(
                "{modality} placeholder token '{token}' is missing from the tokenizer vocabulary"
            )
        })?;
        let expected = plan.count(modality);
        let actual = token_ids
            .iter()
            .filter(|&&candidate| candidate == token_id)
            .count();
        anyhow::ensure!(
            actual == expected,
            "rendered {modality} anchor count mismatch: expected {expected}, found {actual}; verify the chat template contract and escape literal media anchors in user text"
        );
    }
    Ok(())
}

/// The source index of a URL part in [`MediaPlan::source_counts`], read as
/// the tracker reads the URL at the fetch: a `data:` URL is 1, any other URL
/// 0.
fn url_source(url: &str) -> usize {
    usize::from(FetchSourceKind::of_url(url) == FetchSourceKind::DataUrl)
}

/// The source index of a base64 part: the data URL it stands for.
const BASE64_SOURCE: usize = 1;

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/multimodal/plan/tests.rs"]
pub(crate) mod tests;
