//! Media-reference forwarding: decide per request whether the router
//! preprocesses media itself or forwards references to a vLLM gRPC worker that
//! advertises worker-side processing, and build the wire payload.

use std::collections::HashSet;

use llm_multimodal::{media, MediaContentPart, Modality};
use openai_protocol::worker::MmProcessingMode;
use smg_grpc_client::{common_proto as common, vllm_proto as vllm};
use tracing::{debug, warn};

use super::{
    capability::runtime_supports_modality,
    config::MultimodalComponents,
    plan::{MediaPlan, PlaceholderTokens},
};
use crate::{
    observability::metrics::Metrics,
    routers::grpc::context::WorkerSelection,
    worker::{ConnectionMode, RuntimeType, Worker, WorkerRegistry},
};

/// Worker label carrying the advertised backend (`GetServerInfo.mm_processor`).
pub(crate) const MM_PROCESSOR_LABEL: &str = "mm_processor";
/// Worker label carrying accepted URL schemes (`GetServerInfo.mm_media_ref_schemes`).
pub(crate) const MM_MEDIA_REF_SCHEMES_LABEL: &str = "mm_media_ref_schemes";

/// Where one request's media is fetched and preprocessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MmProcessing {
    Router,
    Worker,
}

impl MmProcessing {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Router => "router",
            Self::Worker => "worker",
        }
    }
}

/// Why a request cannot be forwarded as media references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MmRefsError {
    /// A part carries per-item processing hints the worker cannot honor.
    HintUnsupported,
    /// The model's placeholder anchor is not one the worker can expand.
    ModelNotOptedIn(Modality),
    /// vLLM does not accept this modality at all.
    ModalityUnsupported(Modality),
    /// A part kind with no reference form (inline bytes, embeddings, audio).
    UnsupportedPart(&'static str),
    /// An inline `data:` payload above the router's byte cap.
    RefTooLarge {
        index: usize,
        bytes: usize,
        limit: usize,
    },
    /// The selected worker does not fetch this URL scheme.
    SchemeNotAccepted { scheme: String, accepted: String },
    /// EPD encode workers never process references.
    EncodeNotSupported,
    /// A selected leg does not advertise worker-side processing.
    WorkerNotCapable,
}

impl MmRefsError {
    /// Stable error code for the HTTP response body.
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::HintUnsupported => "multimodal_hint_unsupported_in_worker_mode",
            Self::ModelNotOptedIn(_) => "multimodal_worker_processing_unsupported_model",
            Self::ModalityUnsupported(_) | Self::UnsupportedPart(_) => "multimodal_not_supported",
            Self::RefTooLarge { .. } => "media_ref_too_large",
            Self::SchemeNotAccepted { .. } => "media_ref_scheme_not_accepted",
            Self::EncodeNotSupported | Self::WorkerNotCapable => "multimodal_not_supported",
        }
    }
}

impl std::fmt::Display for MmRefsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HintUnsupported => f.write_str(
                "per-item media hints (max_long_side_pixel, fps) cannot be forwarded to a worker",
            ),
            Self::ModelNotOptedIn(modality) => write!(
                f,
                "this model's {modality} placeholder is not expandable by a vLLM worker"
            ),
            Self::ModalityUnsupported(modality) => {
                write!(f, "vLLM workers do not accept {modality} inputs")
            }
            Self::UnsupportedPart(kind) => {
                write!(f, "{kind} cannot be forwarded as a media reference")
            }
            Self::RefTooLarge {
                index,
                bytes,
                limit,
            } => write!(
                f,
                "media item {index}: inline payload is {bytes} bytes, above the {limit}-byte cap"
            ),
            Self::SchemeNotAccepted { scheme, accepted } => write!(
                f,
                "the selected worker does not fetch '{scheme}' URLs (accepts: {accepted}); \
                 file:// needs --allowed-local-media-path on the worker"
            ),
            Self::EncodeNotSupported => {
                f.write_str("media references cannot be routed through encode workers")
            }
            Self::WorkerNotCapable => f.write_str(
                "the selected worker does not advertise worker-side multimodal processing",
            ),
        }
    }
}

/// Whether `worker` can fetch and process media references itself.
pub(crate) fn worker_accepts_media_refs(worker: &dyn Worker) -> bool {
    let spec = &worker.metadata().spec;
    spec.runtime_type == RuntimeType::Vllm
        && *worker.connection_mode() == ConnectionMode::Grpc
        && spec
            .labels
            .get(MM_PROCESSOR_LABEL)
            .is_some_and(|value| !value.is_empty())
}

/// URL schemes `worker` advertised; empty when it advertised none.
pub(crate) fn worker_media_ref_schemes(worker: &dyn Worker) -> HashSet<String> {
    worker
        .metadata()
        .spec
        .labels
        .get(MM_MEDIA_REF_SCHEMES_LABEL)
        .map(|value| {
            value
                .split(',')
                .map(|scheme| scheme.trim().to_ascii_lowercase())
                .filter(|scheme| !scheme.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn ensure_worker_expandable(
    plan: &MediaPlan,
    placeholders: &PlaceholderTokens,
) -> Result<(), MmRefsError> {
    for &modality in plan.modalities() {
        if !runtime_supports_modality(RuntimeType::Vllm, modality) {
            return Err(MmRefsError::ModalityUnsupported(modality));
        }
        if !placeholders.worker_expandable(modality) {
            return Err(MmRefsError::ModelNotOptedIn(modality));
        }
    }
    Ok(())
}

/// Decide where this request's media is processed. Runs in preparation, before
/// worker selection, so `auto` consults every registered worker of the model.
pub(crate) fn resolve_mm_processing(
    components: &MultimodalComponents,
    registry: &WorkerRegistry,
    model_id: &str,
    plan: &MediaPlan,
    placeholders: &PlaceholderTokens,
) -> Result<MmProcessing, MmRefsError> {
    let (resolved, reason) = match components.processing {
        MmProcessingMode::Router => (MmProcessing::Router, "config"),
        MmProcessingMode::Worker => {
            if !plan.is_forwardable() {
                return Err(MmRefsError::HintUnsupported);
            }
            ensure_worker_expandable(plan, placeholders)?;
            (MmProcessing::Worker, "config")
        }
        MmProcessingMode::Auto => {
            if !plan.is_forwardable() {
                (MmProcessing::Router, "plan_not_forwardable")
            } else if ensure_worker_expandable(plan, placeholders).is_err() {
                (MmProcessing::Router, "model_not_opted_in")
            } else {
                // Every registered worker, healthy or not: a health flap must
                // not flip a model between modes request to request.
                let workers = registry.get_by_model(model_id);
                if workers.is_empty() {
                    (MmProcessing::Router, "auto_none")
                } else if workers
                    .iter()
                    .all(|worker| worker_accepts_media_refs(worker.as_ref()))
                {
                    (MmProcessing::Worker, "auto_uniform")
                } else {
                    (MmProcessing::Router, "auto_mixed")
                }
            }
        }
    };

    Metrics::record_mm_processing(model_id, resolved.as_str(), reason);
    log_transition(components, model_id, resolved, reason);
    Ok(resolved)
}

/// Warn once per (model, resolution) change so a fleet silently staying on
/// the router path is visible; bounded by the number of served models.
fn log_transition(
    components: &MultimodalComponents,
    model_id: &str,
    resolved: MmProcessing,
    reason: &'static str,
) {
    let Ok(mut log) = components.mm_mode_log.lock() else {
        return;
    };
    let previous = log.insert(model_id.to_string(), (resolved, reason));
    if previous == Some((resolved, reason)) {
        return;
    }
    match resolved {
        MmProcessing::Worker => debug!(
            model = %model_id,
            reason,
            "multimodal media is forwarded to workers for processing"
        ),
        MmProcessing::Router if reason == "config" => {}
        MmProcessing::Router => warn!(
            model = %model_id,
            reason,
            "multimodal media stays on the router path"
        ),
    }
}

/// Post-selection check: every leg must accept references and the primary
/// worker must fetch every URL scheme in the plan.
pub(crate) fn ensure_selection_supports_media_refs(
    workers: &WorkerSelection,
    plan: &MediaPlan,
) -> Result<(), MmRefsError> {
    let legs: Vec<&dyn Worker> = match workers {
        WorkerSelection::Single { worker } => vec![worker.as_ref()],
        WorkerSelection::Disaggregated {
            encode_assignments: Some(_),
            ..
        } => return Err(MmRefsError::EncodeNotSupported),
        WorkerSelection::Disaggregated {
            prefill, decode, ..
        } => vec![prefill.as_ref(), decode.as_ref()],
    };
    if !legs.iter().all(|leg| worker_accepts_media_refs(*leg)) {
        return Err(MmRefsError::WorkerNotCapable);
    }
    let Some(primary) = legs.first() else {
        return Err(MmRefsError::WorkerNotCapable);
    };
    let accepted = worker_media_ref_schemes(*primary);
    if accepted.is_empty() {
        return Ok(());
    }
    for part in plan.parts() {
        let Some(url) = part_url(part) else {
            continue;
        };
        let scheme = url_scheme(url);
        if !accepted.contains(&scheme) {
            let mut accepted: Vec<_> = accepted.into_iter().collect();
            accepted.sort();
            return Err(MmRefsError::SchemeNotAccepted {
                scheme,
                accepted: accepted.join(","),
            });
        }
    }
    Ok(())
}

fn part_url(part: &MediaContentPart) -> Option<&str> {
    match part {
        MediaContentPart::ImageUrl { url, .. }
        | MediaContentPart::VideoUrl { url, .. }
        | MediaContentPart::AudioUrl { url, .. } => Some(url),
        _ => None,
    }
}

/// Lower-cased URL scheme; `data` for data URLs, empty for bare paths.
fn url_scheme(url: &str) -> String {
    url.split_once(':')
        .map(|(scheme, _)| scheme)
        .filter(|scheme| {
            !scheme.is_empty()
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        })
        .map(str::to_ascii_lowercase)
        .unwrap_or_default()
}

/// Approximate decoded byte size of a `data:` URL payload; `None` otherwise.
fn data_url_payload_bytes(url: &str) -> Option<usize> {
    let rest = url
        .get(..5)
        .filter(|prefix| prefix.eq_ignore_ascii_case("data:"))
        .map(|_| &url[5..])?;
    let (header, payload) = rest.split_once(',').unwrap_or((rest, ""));
    Some(if header.to_ascii_lowercase().contains(";base64") {
        payload.len() * 3 / 4
    } else {
        payload.len()
    })
}

/// Build the wire payload from the plan, in authored (prompt) order.
pub(crate) fn assemble_media_refs(plan: MediaPlan) -> Result<vllm::MediaRefs, MmRefsError> {
    let mut items = Vec::with_capacity(plan.parts().len());
    for (index, part) in plan.into_parts().into_iter().enumerate() {
        let (modality, url, limit) = match part {
            MediaContentPart::ImageUrl {
                url,
                max_long_side_pixel: None,
                ..
            } => (common::Modality::Image, url, media::image_max_input_bytes()),
            MediaContentPart::VideoUrl {
                url,
                fps: None,
                max_long_side_pixel: None,
                ..
            } => (common::Modality::Video, url, media::video_max_input_bytes()),
            MediaContentPart::ImageUrl { .. } | MediaContentPart::VideoUrl { .. } => {
                return Err(MmRefsError::HintUnsupported)
            }
            MediaContentPart::ImageData { .. } => {
                return Err(MmRefsError::UnsupportedPart("inline image bytes"))
            }
            MediaContentPart::VideoData { .. } => {
                return Err(MmRefsError::UnsupportedPart("inline video bytes"))
            }
            MediaContentPart::AudioUrl { .. } | MediaContentPart::AudioData { .. } => {
                return Err(MmRefsError::UnsupportedPart("audio"))
            }
            MediaContentPart::ImageEmbeds { .. } => {
                return Err(MmRefsError::UnsupportedPart("image embeddings"))
            }
            MediaContentPart::Text { .. } => return Err(MmRefsError::UnsupportedPart("text")),
        };
        if let Some(bytes) = data_url_payload_bytes(&url) {
            if bytes > limit {
                return Err(MmRefsError::RefTooLarge {
                    index,
                    bytes,
                    limit,
                });
            }
        }
        items.push(vllm::MediaRef {
            modality: modality as i32,
            url,
        });
    }
    Ok(vllm::MediaRefs { items })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        routers::grpc::multimodal::config::MultimodalConfigRegistry,
        worker::{BasicWorkerBuilder, ModelCard, WorkerType},
    };

    const MODEL: &str = "qwen3-vl";

    fn image_url(url: &str) -> MediaContentPart {
        MediaContentPart::ImageUrl {
            url: url.to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }
    }

    fn plan(parts: Vec<MediaContentPart>) -> MediaPlan {
        MediaPlan::new(parts)
    }

    fn expandable_placeholders() -> PlaceholderTokens {
        let mut placeholders = PlaceholderTokens::default();
        placeholders.insert(Modality::Image, "<|image_pad|>".to_string());
        placeholders.set_worker_expandable(Modality::Image, true);
        placeholders
    }

    fn components(mode: MmProcessingMode) -> MultimodalComponents {
        let mut components =
            MultimodalComponents::new(Arc::new(MultimodalConfigRegistry::new()), None)
                .expect("components");
        components.processing = mode;
        components
    }

    fn worker(
        url: &str,
        runtime: RuntimeType,
        connection: ConnectionMode,
        labels: &[(&str, &str)],
    ) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new(MODEL))
            .worker_type(WorkerType::Regular)
            .runtime_type(runtime)
            .connection_mode(connection)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            });
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        Arc::new(builder.build())
    }

    fn capable(url: &str) -> Arc<dyn Worker> {
        worker(
            url,
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[
                (MM_PROCESSOR_LABEL, "inprocess"),
                (MM_MEDIA_REF_SCHEMES_LABEL, "http,https,data"),
            ],
        )
    }

    fn registry_with(workers: Vec<Arc<dyn Worker>>) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for worker in workers {
            registry.register(worker).expect("register");
        }
        registry
    }

    type AcceptCase = (
        RuntimeType,
        ConnectionMode,
        &'static [(&'static str, &'static str)],
        bool,
    );

    #[test]
    fn accepts_only_labeled_vllm_grpc_workers() {
        let cases: [AcceptCase; 5] = [
            (
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[(MM_PROCESSOR_LABEL, "inprocess")],
                true,
            ),
            (
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[(MM_PROCESSOR_LABEL, "")],
                false,
            ),
            (RuntimeType::Vllm, ConnectionMode::Grpc, &[], false),
            (
                RuntimeType::Vllm,
                ConnectionMode::Zmq,
                &[(MM_PROCESSOR_LABEL, "inprocess")],
                false,
            ),
            (
                RuntimeType::Sglang,
                ConnectionMode::Grpc,
                &[(MM_PROCESSOR_LABEL, "inprocess")],
                false,
            ),
        ];
        for (i, (runtime, connection, labels, expected)) in cases.into_iter().enumerate() {
            let scheme = if connection == ConnectionMode::Zmq {
                format!("ipc:///tmp/smg-refs-{i}.ipc")
            } else {
                format!("grpc://127.0.0.1:{}", 9100 + i)
            };
            let w = worker(&scheme, runtime, connection, labels);
            assert_eq!(worker_accepts_media_refs(w.as_ref()), expected, "case {i}");
        }
    }

    #[test]
    fn schemes_label_is_parsed_case_insensitively() {
        let w = worker(
            "grpc://127.0.0.1:9200",
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[(MM_MEDIA_REF_SCHEMES_LABEL, "HTTP, https ,data,")],
        );
        let schemes = worker_media_ref_schemes(w.as_ref());
        assert_eq!(schemes.len(), 3);
        assert!(schemes.contains("http") && schemes.contains("https") && schemes.contains("data"));
    }

    #[test]
    fn router_mode_never_forwards() {
        let registry = registry_with(vec![capable("grpc://127.0.0.1:9300")]);
        let resolved = resolve_mm_processing(
            &components(MmProcessingMode::Router),
            &registry,
            MODEL,
            &plan(vec![image_url("https://a/1.png")]),
            &expandable_placeholders(),
        )
        .expect("router mode resolves");
        assert_eq!(resolved, MmProcessing::Router);
    }

    #[test]
    fn auto_forwards_only_for_a_uniform_capable_fleet() {
        let placeholders = expandable_placeholders();
        let plan_ok = plan(vec![image_url("https://a/1.png")]);
        let components = components(MmProcessingMode::Auto);

        let uniform = registry_with(vec![
            capable("grpc://127.0.0.1:9400"),
            capable("grpc://127.0.0.1:9401"),
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &uniform, MODEL, &plan_ok, &placeholders)
                .expect("resolves"),
            MmProcessing::Worker
        );

        let mixed = registry_with(vec![
            capable("grpc://127.0.0.1:9402"),
            worker(
                "grpc://127.0.0.1:9403",
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[],
            ),
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &mixed, MODEL, &plan_ok, &placeholders)
                .expect("resolves"),
            MmProcessing::Router
        );

        let empty = registry_with(vec![]);
        assert_eq!(
            resolve_mm_processing(&components, &empty, MODEL, &plan_ok, &placeholders)
                .expect("resolves"),
            MmProcessing::Router
        );
    }

    #[test]
    fn auto_falls_back_for_hints_and_unopted_models() {
        let components = components(MmProcessingMode::Auto);
        let registry = registry_with(vec![capable("grpc://127.0.0.1:9500")]);

        let hinted = plan(vec![MediaContentPart::ImageUrl {
            url: "https://a/1.png".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: Some(512),
        }]);
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &hinted,
                &expandable_placeholders()
            )
            .expect("resolves"),
            MmProcessing::Router
        );

        let mut unopted = PlaceholderTokens::default();
        unopted.insert(Modality::Image, "<|image|>".to_string());
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![image_url("https://a/1.png")]),
                &unopted
            )
            .expect("resolves"),
            MmProcessing::Router
        );
    }

    #[test]
    fn worker_mode_is_strict() {
        let components = components(MmProcessingMode::Worker);
        // No registered workers: the decision is still Worker; selection enforces.
        let registry = registry_with(vec![]);
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![image_url("https://a/1.png")]),
                &expandable_placeholders()
            )
            .expect("resolves"),
            MmProcessing::Worker
        );

        let mut unopted = PlaceholderTokens::default();
        unopted.insert(Modality::Image, "<|image|>".to_string());
        let err = resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &plan(vec![image_url("https://a/1.png")]),
            &unopted,
        )
        .expect_err("unopted model is refused");
        assert_eq!(err.code(), "multimodal_worker_processing_unsupported_model");

        let hinted = plan(vec![MediaContentPart::VideoUrl {
            url: "https://a/c.mp4".to_string(),
            uuid: None,
            fps: Some(2.0),
            max_long_side_pixel: None,
        }]);
        let err = resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &hinted,
            &expandable_placeholders(),
        )
        .expect_err("hints are refused");
        assert_eq!(err.code(), "multimodal_hint_unsupported_in_worker_mode");
    }

    #[test]
    fn selection_check_requires_every_leg_and_scheme() {
        let plan_https = plan(vec![image_url("https://a/1.png")]);
        let single = WorkerSelection::Single {
            worker: capable("grpc://127.0.0.1:9600"),
        };
        ensure_selection_supports_media_refs(&single, &plan_https).expect("capable single");

        let plain = WorkerSelection::Single {
            worker: worker(
                "grpc://127.0.0.1:9601",
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[],
            ),
        };
        assert_eq!(
            ensure_selection_supports_media_refs(&plain, &plan_https),
            Err(MmRefsError::WorkerNotCapable)
        );

        let file_plan = plan(vec![image_url("file:///media/1.png")]);
        let err = ensure_selection_supports_media_refs(&single, &file_plan)
            .expect_err("file scheme not advertised");
        assert_eq!(err.code(), "media_ref_scheme_not_accepted");

        let pd = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: capable("grpc://127.0.0.1:9602"),
            decode: worker(
                "grpc://127.0.0.1:9603",
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[],
            ),
            runtime_type: RuntimeType::Vllm,
        };
        assert_eq!(
            ensure_selection_supports_media_refs(&pd, &plan_https),
            Err(MmRefsError::WorkerNotCapable)
        );

        let epd = WorkerSelection::Disaggregated {
            encode_assignments: Some(vec![]),
            prefill: capable("grpc://127.0.0.1:9604"),
            decode: capable("grpc://127.0.0.1:9605"),
            runtime_type: RuntimeType::Vllm,
        };
        assert_eq!(
            ensure_selection_supports_media_refs(&epd, &plan_https),
            Err(MmRefsError::EncodeNotSupported)
        );
    }

    #[test]
    fn assemble_keeps_prompt_order_and_maps_modalities() {
        let refs = assemble_media_refs(plan(vec![
            image_url("https://a/1.png"),
            MediaContentPart::VideoUrl {
                url: "data:video/mp4;base64,AAAA".to_string(),
                uuid: None,
                fps: None,
                max_long_side_pixel: None,
            },
            image_url("https://a/2.png"),
        ]))
        .expect("assembles");
        let items: Vec<(i32, &str)> = refs
            .items
            .iter()
            .map(|item| (item.modality, item.url.as_str()))
            .collect();
        assert_eq!(
            items,
            vec![
                (common::Modality::Image as i32, "https://a/1.png"),
                (common::Modality::Video as i32, "data:video/mp4;base64,AAAA"),
                (common::Modality::Image as i32, "https://a/2.png"),
            ]
        );
    }

    #[test]
    fn assemble_rejects_unsupported_parts_and_oversized_data_urls() {
        let err = assemble_media_refs(plan(vec![MediaContentPart::AudioUrl {
            url: "https://a/x.wav".to_string(),
            uuid: None,
        }]))
        .expect_err("audio is not forwardable");
        assert_eq!(err, MmRefsError::UnsupportedPart("audio"));

        let oversized = format!(
            "data:image/png;base64,{}",
            "A".repeat(media::image_max_input_bytes() * 4 / 3 + 8)
        );
        let err =
            assemble_media_refs(plan(vec![image_url(&oversized)])).expect_err("oversized data url");
        assert_eq!(err.code(), "media_ref_too_large");
    }

    #[test]
    fn url_scheme_and_data_url_size() {
        assert_eq!(url_scheme("HTTPS://a/1.png"), "https");
        assert_eq!(url_scheme("data:image/png;base64,AAAA"), "data");
        assert_eq!(url_scheme("/local/path.png"), "");
        assert_eq!(data_url_payload_bytes("https://a/1.png"), None);
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64,AAAAAAAA"),
            Some(6)
        );
        assert_eq!(data_url_payload_bytes("DATA:text/plain,hello"), Some(5));
    }
}
