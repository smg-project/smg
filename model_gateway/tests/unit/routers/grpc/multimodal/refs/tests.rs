use std::sync::Arc;

use openai_protocol::model_card::ModelCard;
use smg_worker::{identity::basic::WorkerType, test_support::no_health_check};

use super::*;
#[cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]
use crate::routers::grpc::testing::bytes_allocated_during;
use crate::routers::{
    common::snapshot::test_support::registry_with,
    grpc::{
        backend_client::BasicWorkerBuilder,
        multimodal::{config::MultimodalConfigRegistry, settings::MultimodalSettings},
    },
};

const MODEL: &str = "qwen3-vl";

fn image_url(url: &str) -> MediaContentPart {
    MediaContentPart::ImageUrl {
        url: url.to_string().into(),
        detail: None,
        uuid: None,
        max_long_side_pixel: None,
    }
}

fn plan(parts: Vec<MediaContentPart>) -> MediaPlan {
    MediaPlan::new(parts)
}

fn base64_image(data: &str) -> MediaContentPart {
    MediaContentPart::ImageBase64 {
        media_type: "image/png".to_string(),
        data: Arc::new(data.to_string()),
        detail: None,
        uuid: None,
        max_long_side_pixel: None,
    }
}

fn base64_audio(data: &str) -> MediaContentPart {
    MediaContentPart::AudioBase64 {
        media_type: "audio/wav".to_string(),
        data: Arc::new(data.to_string()),
        uuid: None,
    }
}

fn expandable_placeholders() -> PlaceholderTokens {
    let mut placeholders = PlaceholderTokens::default();
    placeholders.insert(Modality::Image, "<|image_pad|>".to_string());
    placeholders.set_worker_expandable(Modality::Image, true);
    placeholders
}

fn components(mode: MmProcessingMode) -> MultimodalComponents {
    let mut components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .expect("components");
    components.processing = mode;
    assert!(!components.drain_fetches_on_error);
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
        .health_config(no_health_check());
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

type AcceptCase = (
    RuntimeType,
    ConnectionMode,
    &'static [(&'static str, &'static str)],
    bool,
);

#[test]
/// Forward references only to gRPC workers advertising a supported processor.
fn accepts_only_supported_labeled_grpc_workers() {
    let cases: [AcceptCase; 8] = [
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
        (
            RuntimeType::Sglang,
            ConnectionMode::Grpc,
            &[(MM_PROCESSOR_LABEL, "smg")],
            true,
        ),
        (RuntimeType::Sglang, ConnectionMode::Grpc, &[], false),
        (
            RuntimeType::Sglang,
            ConnectionMode::Zmq,
            &[(MM_PROCESSOR_LABEL, "smg")],
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

/// An inline clip or image whose data URL header the request got wrong
/// is refused at the door as the invalid URL it would make, echoed by a
/// bounded prefix: a media type that is not one type/subtype pair (the
/// chat door writes `audio/<format>` from `input_audio.format` as sent, so
/// a MIME type there doubles the header and an empty format breaks it),
/// or a payload that itself opens with a data URL header. A restricted-name
/// subtype of any case passes; the payload's symbols stay unread.
#[test]
fn an_inline_payload_whose_header_the_request_got_wrong_is_refused_at_the_door() {
    fn clip(media_type: &str, data: &str) -> MediaContentPart {
        MediaContentPart::AudioBase64 {
            media_type: media_type.to_string(),
            data: data.to_string().into(),
            uuid: None,
        }
    }
    fn image(media_type: &str, data: &str) -> MediaContentPart {
        MediaContentPart::ImageBase64 {
            media_type: media_type.to_string(),
            data: data.to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }
    }
    let refused = |part: MediaContentPart, url: &str| {
        assert_eq!(
            check_forwarded_references(&plan(vec![part])),
            Err(MmRefsError::InvalidReference {
                index: 0,
                cause: format!("invalid media URL: {url}"),
            }),
            "{url}"
        );
    };
    // A MIME type as the format: the header the door would build doubles it.
    refused(
        clip("audio/audio/wav", "UklG"),
        "data:audio/audio/wav;base64,UklG",
    );
    // An empty format: no subtype; a subtype opening with a separator.
    refused(clip("audio/", "UklG"), "data:audio/;base64,UklG");
    refused(clip("audio/+x", "UklG"), "data:audio/+x;base64,UklG");
    // No slash (the Messages door copies `source.media_type` as sent), a
    // parameter, a trailing space: not one type/subtype pair.
    refused(image("jpeg", "AAAA"), "data:jpeg;base64,AAAA");
    refused(
        clip("audio/wav;rate=16000", "UklG"),
        "data:audio/wav;rate=16000;base64,UklG",
    );
    refused(image("image/png ", "AAAA"), "data:image/png ;base64,AAAA");
    // A data URL sent as the payload (read trimmed): wrapped a second time.
    refused(
        clip("audio/wav", " data:audio/wav;base64,UklG"),
        "data:audio/wav;base64, data:audio/wav;base64,UklG",
    );
    refused(
        image("image/png", "DATA:image/png;base64,AAAA"),
        "data:image/png;base64,DATA:image/png;base64,AAAA",
    );
    // The echo is bounded.
    let long = "A".repeat(200);
    let Err(MmRefsError::InvalidReference { cause, .. }) =
        check_forwarded_references(&plan(vec![clip("audio/x/y", &long)]))
    else {
        panic!("a doubled header is refused");
    };
    assert_eq!(
        cause.len(),
        "invalid media URL: ".len() + URL_ECHO_CHARS + 3
    );
    assert!(cause.ends_with("..."));
    // Well-formed headers pass whatever the payload's symbols or the subtype's case.
    for part in [
        clip("audio/wav", "UklG"),
        clip("audio/WAV", "UklG"),
        clip("audio/x-wav", "UklG"),
        clip("audio/xyz", "not base64 at all"),
        image("image/svg+xml", "PHN2Zz4="),
    ] {
        check_forwarded_references(&plan(vec![part])).expect("a well-formed header passes");
    }
}

/// A reference no engine serves is refused at the door, before a worker
/// is selected, in the media connector's own words under the router-side
/// pipeline's code for the same media: an empty url, a data URL whose
/// header names no base64 payload, an empty payload (a URL part's or a
/// base64 source's); a well-formed reference still forwards untouched, a
/// data URL with parameters before the marker among them.
#[test]
fn a_reference_no_engine_serves_is_refused_at_the_door() {
    let components = components(MmProcessingMode::Worker);
    let registry = registry_with(vec![]);
    let worker_only = PlaceholderTokens::worker_only();
    let long = format!(" data:image/png;base64,{}", "A".repeat(200));
    let long_echo = format!("invalid media URL: {}...", &long[..64]);
    for (part, cause) in [
        (image_url(""), "invalid media URL: "),
        (image_url("   "), "invalid media URL:    "),
        (
            image_url(" data:image/png;base64,AQ=="),
            "invalid media URL:  data:image/png;base64,AQ==",
        ),
        (image_url(&long), long_echo.as_str()),
        (
            image_url("\u{1}data:image/png;base64,AQ=="),
            "invalid media URL: \u{1}data:image/png;base64,AQ==",
        ),
        (
            image_url("\u{7f}data:image/png;base64,AQ=="),
            "invalid media URL: \u{7f}data:image/png;base64,AQ==",
        ),
        (
            image_url("\u{9b}https://a/1.png"),
            "invalid media URL: \u{9b}https://a/1.png",
        ),
        (
            image_url("data:image/png,%89PNG%0D%0A"),
            "data URL parse error: only base64 encoded data URLs are supported",
        ),
        (
            image_url("data:image/png;base64;charset=x,AQ=="),
            "data URL parse error: only base64 encoded data URLs are supported",
        ),
        (
            image_url("data:image/png;base64,"),
            "data URL parse error: empty payload",
        ),
        (base64_image(""), "data URL parse error: empty payload"),
    ] {
        let error = resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &plan(vec![part]),
            &worker_only,
        )
        .expect_err(cause);
        assert_eq!(error.code(), "invalid_multimodal_request", "{cause}");
        assert_eq!(error.to_string(), format!("media item 0: {cause}"));
    }
    for part in [
        image_url("data:image/png;base64,AQ=="),
        image_url("data:image/png;charset=x;base64,AQ=="),
        image_url("data:image/png;charset=x;name=a.png;base64,AQ=="),
        image_url("https://a/1.png"),
        base64_image("AQ=="),
    ] {
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![part]),
                &worker_only
            )
            .expect("a well-formed reference forwards"),
            MmProcessing::Worker
        );
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

/// A model without a registry spec: worker placement forwards it, auto
/// forwards it when every worker of the model processes media itself and
/// refuses it otherwise, router placement refuses it; the refusal names
/// the router's missing spec, not the model.
#[test]
fn a_model_without_a_spec_goes_to_a_capable_worker_or_is_refused() {
    let worker_only = PlaceholderTokens::worker_only();
    let plan_ok = plan(vec![image_url("https://a/1.png")]);
    assert!(worker_only.worker_expandable(Modality::Image));

    assert_eq!(
        resolve_mm_processing(
            &components(MmProcessingMode::Worker),
            &registry_with(vec![]),
            MODEL,
            &plan_ok,
            &worker_only
        )
        .expect("worker placement forwards"),
        MmProcessing::Worker
    );
    let auto = components(MmProcessingMode::Auto);
    assert_eq!(
        resolve_mm_processing(
            &auto,
            &registry_with(vec![capable("grpc://127.0.0.1:9450")]),
            MODEL,
            &plan_ok,
            &worker_only
        )
        .expect("a uniform capable fleet forwards"),
        MmProcessing::Worker
    );
    for (registry, placement) in [
        (registry_with(vec![]), &auto),
        (
            registry_with(vec![capable("grpc://127.0.0.1:9451")]),
            &components(MmProcessingMode::Router),
        ),
    ] {
        let err = resolve_mm_processing(placement, &registry, MODEL, &plan_ok, &worker_only)
            .expect_err("nothing processes the media");
        assert_eq!(err, MmRefsError::RouterSpecMissing);
        assert_eq!(err.code(), "multimodal_not_supported");
        assert!(err.to_string().contains("no media spec"), "{err}");
    }
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

/// An encode-prefill-decode pool resolves to router-side processing under
/// `auto`, whatever its workers advertise: references never travel on
/// that pipeline, so a pool whose workers all accept them would otherwise
/// resolve to a mode selection refuses, with a 400 naming the mode the
/// router already runs.
#[test]
fn auto_processes_in_the_router_for_an_encode_prefill_decode_pool() {
    let placeholders = expandable_placeholders();
    let plan_ok = plan(vec![image_url("https://a/1.png")]);
    let components = components(MmProcessingMode::Auto);
    let encode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:9410")
            .model(ModelCard::new(MODEL))
            .worker_type(WorkerType::Encode)
            .runtime_type(RuntimeType::Vllm)
            .connection_mode(ConnectionMode::Grpc)
            .label(MM_PROCESSOR_LABEL, "inprocess")
            .label(MM_MEDIA_REF_SCHEMES_LABEL, "http,https,data")
            .health_config(no_health_check())
            .build(),
    );
    let epd = registry_with(vec![
        capable("grpc://127.0.0.1:9411"),
        capable("grpc://127.0.0.1:9412"),
        encode,
    ]);
    assert_eq!(
        resolve_mm_processing(&components, &epd, MODEL, &plan_ok, &placeholders).expect("resolves"),
        MmProcessing::Router
    );
}

#[test]
fn auto_falls_back_for_hints_and_unopted_models() {
    let components = components(MmProcessingMode::Auto);
    let registry = registry_with(vec![capable("grpc://127.0.0.1:9500")]);

    let hinted = plan(vec![MediaContentPart::ImageUrl {
        url: "https://a/1.png".to_string().into(),
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

/// Under worker-side placement a model with workers and none that
/// processes media references is refused in preparation, as the fleet
/// verdict it is: a terminal 503 under the code selection has always
/// answered the condition with, whose message names the flag and the
/// fix. One capable worker lifts it (selection picks it); an empty pool
/// is left to selection; a hinted request the router can serve keeps its
/// router fallback; a fault of the request keeps its 400.
#[test]
fn worker_mode_refuses_a_fleet_that_processes_no_media_references() {
    use crate::routers::common::retry::is_retryable_response;

    let components = components(MmProcessingMode::Worker);
    let plan_ok = plan(vec![image_url("https://a/1.png")]);
    let placeholders = expandable_placeholders();
    let plain = |url: &str| worker(url, RuntimeType::Vllm, ConnectionMode::Grpc, &[]);

    let none_capable = registry_with(vec![
        plain("grpc://127.0.0.1:9800"),
        plain("grpc://127.0.0.1:9801"),
    ]);
    let err = resolve_mm_processing(&components, &none_capable, MODEL, &plan_ok, &placeholders)
        .expect_err("no worker processes media references");
    assert_eq!(
        err,
        MmRefsError::NoCapableWorker {
            model: MODEL.to_string(),
            workers: 2,
        }
    );
    assert_eq!(err.code(), "no_media_ref_capable_worker");
    let message = err.to_string();
    assert!(
        message.contains("--mm-processing worker") && message.contains("2 registered worker"),
        "{message}"
    );
    let response = err.into_response();
    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response.headers()[error::HEADER_X_SMG_ERROR_CODE],
        "no_media_ref_capable_worker"
    );
    assert!(
        !is_retryable_response(&response),
        "a capability verdict is terminal for the retry layer"
    );
    assert!(
        placement::refusal_reason(&response).is_some(),
        "the refusal's line names its reason"
    );

    // An engine of another family never processes references either.
    let other_engine = registry_with(vec![worker(
        "grpc://127.0.0.1:9802",
        RuntimeType::TokenSpeed,
        ConnectionMode::Grpc,
        &[],
    )]);
    assert!(matches!(
        resolve_mm_processing(&components, &other_engine, MODEL, &plan_ok, &placeholders),
        Err(MmRefsError::NoCapableWorker { workers: 1, .. })
    ));

    // A pool with an encode worker runs the encode-prefill-decode
    // pipeline, on which media references never travel whatever the
    // workers advertise: left to selection, whose refusal is the
    // request's own 400 and names the fix that applies there.
    let encode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:9805")
            .model(ModelCard::new(MODEL))
            .worker_type(WorkerType::Encode)
            .runtime_type(RuntimeType::Vllm)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .build(),
    );
    let epd = registry_with(vec![plain("grpc://127.0.0.1:9806"), encode]);
    assert_eq!(
        resolve_mm_processing(&components, &epd, MODEL, &plan_ok, &placeholders)
            .expect("an encode-prefill-decode pool is selection's to refuse"),
        MmProcessing::Worker
    );

    let mixed = registry_with(vec![
        plain("grpc://127.0.0.1:9803"),
        capable("grpc://127.0.0.1:9804"),
    ]);
    assert_eq!(
        resolve_mm_processing(&components, &mixed, MODEL, &plan_ok, &placeholders)
            .expect("one capable worker is enough: selection picks it"),
        MmProcessing::Worker
    );

    let hinted = plan(vec![MediaContentPart::ImageUrl {
        url: "https://a/1.png".to_string().into(),
        detail: None,
        uuid: None,
        max_long_side_pixel: Some(512),
    }]);
    assert_eq!(
        resolve_mm_processing(&components, &none_capable, MODEL, &hinted, &placeholders)
            .expect("the router applies the hint itself"),
        MmProcessing::Router
    );

    let mut unopted = PlaceholderTokens::default();
    unopted.insert(Modality::Image, "<|image|>".to_string());
    let fault = resolve_mm_processing(&components, &mixed, MODEL, &plan_ok, &unopted)
        .expect_err("an unopted model is the request's fault");
    assert_eq!(
        fault.code(),
        "multimodal_worker_processing_unsupported_model"
    );
    assert_eq!(
        fault.into_response().status(),
        http::StatusCode::BAD_REQUEST
    );
}

/// What readiness names: under worker-side placement, a model with a
/// worker that reports a vision encoder and no worker that processes
/// media references. A pool without the label (unknown) or with a
/// `false` (a text-only engine) is not it, one capable worker takes the
/// model off the list, a model with an encode worker is never on it (nor
/// is its registration warned about), and the other placements name
/// nothing.
#[test]
fn models_without_media_processing_names_vision_pools_with_no_capable_worker() {
    fn typed_worker_of(
        model: &str,
        url: &str,
        labels: &[(&str, &str)],
        worker_type: WorkerType,
    ) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model))
            .worker_type(worker_type)
            .runtime_type(RuntimeType::Vllm)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check());
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        Arc::new(builder.build())
    }
    fn worker_of(model: &str, url: &str, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
        typed_worker_of(model, url, labels, WorkerType::Regular)
    }
    let vision = [(SUPPORTS_VISION_LABEL, "true")];
    let vision_capable = [
        (SUPPORTS_VISION_LABEL, "true"),
        (MM_PROCESSOR_LABEL, "inprocess"),
    ];
    let text_only = [(SUPPORTS_VISION_LABEL, "false")];

    let workers: Vec<Arc<dyn Worker>> = vec![
        // Two vision workers, neither processes references: held.
        worker_of("vl-b", "grpc://127.0.0.1:9900", &vision),
        worker_of("vl-b", "grpc://127.0.0.1:9901", &vision),
        // A vision worker without, beside one with: the model is served.
        worker_of("vl-a", "grpc://127.0.0.1:9902", &vision),
        worker_of("vl-a", "grpc://127.0.0.1:9903", &vision_capable),
        // Text-only and unlabeled pools say nothing about media.
        worker_of("text", "grpc://127.0.0.1:9904", &text_only),
        worker_of("legacy", "grpc://127.0.0.1:9905", &[]),
        // An encode-prefill-decode pool: references never travel on it,
        // so its vision sides without a processor are not what readiness
        // waits on.
        typed_worker_of(
            "vl-epd",
            "grpc://127.0.0.1:9906",
            &vision,
            WorkerType::Prefill,
        ),
        typed_worker_of(
            "vl-epd",
            "grpc://127.0.0.1:9907",
            &vision,
            WorkerType::Decode,
        ),
        typed_worker_of("vl-epd", "grpc://127.0.0.1:9908", &[], WorkerType::Encode),
    ];
    assert_eq!(
        models_without_media_processing(MmProcessingMode::Worker, &workers),
        vec!["vl-b".to_string()]
    );
    // The registration warning follows the same line: vl-b's worker is
    // said, the encode-prefill-decode pool's vision side is not, and a
    // capable worker beside a plain one is not either; the plain one
    // beside it is said, with the pool serving the media (the line's
    // consequence reads the pool, `pool_processes_media_refs`).
    let pool_of = |model: &str| {
        workers
            .iter()
            .filter(move |worker| worker.model_id() == model)
            .collect::<Vec<_>>()
    };
    assert!(registration_lacks_media_processing(
        workers[0].as_ref(),
        pool_of("vl-b").into_iter()
    ));
    assert!(!registration_lacks_media_processing(
        workers[6].as_ref(),
        pool_of("vl-epd").into_iter()
    ));
    assert!(!registration_lacks_media_processing(
        workers[3].as_ref(),
        pool_of("vl-a").into_iter()
    ));
    assert!(registration_lacks_media_processing(
        workers[2].as_ref(),
        pool_of("vl-a").into_iter()
    ));
    assert!(pool_processes_media_refs(pool_of("vl-a").into_iter()));
    assert!(!pool_processes_media_refs(pool_of("vl-b").into_iter()));
    assert!(pool_has_encode_worker(pool_of("vl-epd").into_iter()));
    assert!(!pool_has_encode_worker(pool_of("vl-a").into_iter()));
    for placement in [MmProcessingMode::Auto, MmProcessingMode::Router] {
        assert!(models_without_media_processing(placement, &workers).is_empty());
    }
    assert!(models_without_media_processing(MmProcessingMode::Worker, &[]).is_empty());

    // The predicate behind it, per worker.
    assert!(vision_worker_without_media_processing(workers[0].as_ref()));
    assert!(!vision_worker_without_media_processing(workers[3].as_ref()));
    assert!(!vision_worker_without_media_processing(workers[4].as_ref()));
    assert!(!vision_worker_without_media_processing(workers[5].as_ref()));
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

    let inline = plan(vec![MediaContentPart::ImageData {
        data: vec![0, 1, 2],
        mime_type: None,
        uuid: None,
        detail: None,
    }]);
    let err = resolve_mm_processing(
        &components,
        &registry,
        MODEL,
        &inline,
        &expandable_placeholders(),
    )
    .expect_err("inline bytes are refused by name");
    assert_eq!(err, MmRefsError::UnsupportedPart("inline image bytes"));
}

fn hinted_image(url: &str, max_long_side_pixel: u32) -> MediaContentPart {
    MediaContentPart::ImageUrl {
        url: url.to_string().into(),
        detail: None,
        uuid: None,
        max_long_side_pixel: Some(max_long_side_pixel),
    }
}

fn hinted_video(url: &str, fps: Option<f64>, max_long_side_pixel: Option<u32>) -> MediaContentPart {
    MediaContentPart::VideoUrl {
        url: url.to_string().into(),
        uuid: None,
        fps,
        max_long_side_pixel,
    }
}

#[test]
fn worker_mode_preprocesses_hinted_items_itself() {
    let components = components(MmProcessingMode::Worker);
    let registry = registry_with(vec![capable("grpc://127.0.0.1:9700")]);
    let placeholders = expandable_placeholders();

    // The hints are the router pipeline's knobs: a request carrying one
    // is preprocessed here, whichever item carries it.
    for (name, hinted) in [
        (
            "image long side",
            plan(vec![hinted_image("https://a/1.png", 504)]),
        ),
        (
            "video fps",
            plan(vec![hinted_video("https://a/c.mp4", Some(1.0), None)]),
        ),
        (
            "video long side",
            plan(vec![hinted_video("https://a/c.mp4", None, Some(504))]),
        ),
        (
            "one hinted item among plain ones",
            plan(vec![
                image_url("https://a/1.png"),
                hinted_video("https://a/c.mp4", Some(2.0), None),
            ]),
        ),
    ] {
        assert_eq!(
            resolve_mm_processing(&components, &registry, MODEL, &hinted, &placeholders)
                .unwrap_or_else(|err| panic!("{name}: {err}")),
            MmProcessing::Router,
            "{name}"
        );
    }

    // Plain items still forward.
    assert_eq!(
        resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &plan(vec![image_url("https://a/1.png")]),
            &placeholders
        )
        .expect("plain items forward"),
        MmProcessing::Worker
    );

    // The router expands the placeholders itself, so the model's worker
    // expansion is not consulted for a hinted request.
    let mut unopted = PlaceholderTokens::default();
    unopted.insert(Modality::Image, "<|image|>".to_string());
    assert_eq!(
        resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &plan(vec![hinted_image("https://a/1.png", 252)]),
            &unopted
        )
        .expect("the router processes it"),
        MmProcessing::Router
    );

    // A part with no reference form is still refused by name, before or
    // after the hinted one.
    let inline = MediaContentPart::ImageData {
        data: vec![0, 1, 2],
        mime_type: None,
        uuid: None,
        detail: None,
    };
    for parts in [
        vec![hinted_image("https://a/1.png", 504), inline.clone()],
        vec![inline.clone(), hinted_image("https://a/1.png", 504)],
    ] {
        assert_eq!(
            resolve_mm_processing(&components, &registry, MODEL, &plan(parts), &placeholders)
                .expect_err("inline bytes are refused"),
            MmRefsError::UnsupportedPart("inline image bytes")
        );
    }

    // Without a spec the router cannot apply the hints: the request is
    // refused for the hint, not for the missing spec.
    let err = resolve_mm_processing(
        &components,
        &registry,
        MODEL,
        &plan(vec![hinted_image("https://a/1.png", 504)]),
        &PlaceholderTokens::worker_only(),
    )
    .expect_err("no pipeline applies the hint");
    assert_eq!(err.code(), "multimodal_hint_unsupported_in_worker_mode");
    assert!(
        err.to_string().contains("cannot apply them itself"),
        "{err}"
    );

    // vLLM also accepts the router's audio tensors. A hinted request
    // carrying audio is processed locally so its image hint takes effect;
    // the same audio without a hint still forwards.
    let audio = MediaContentPart::AudioUrl {
        url: "https://a/x.wav".to_string().into(),
        uuid: None,
    };
    let mut with_audio = expandable_placeholders();
    with_audio.insert(Modality::Audio, "<|audio_pad|>".to_string());
    with_audio.set_worker_expandable(Modality::Audio, true);
    assert_eq!(
        resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &plan(vec![hinted_image("https://a/1.png", 504), audio.clone()]),
            &with_audio,
        )
        .expect("the router can apply the hint and serve audio"),
        MmProcessing::Router
    );
    assert_eq!(
        resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &plan(vec![audio]),
            &with_audio
        )
        .expect("plain audio forwards"),
        MmProcessing::Worker
    );
}

/// The scheme refusal's remedy follows the refused scheme: a file URL
/// names the worker's local-media flag, a data URL (an inline image the
/// worker's labels leave out) gets the accepted list alone, and the
/// sentence's head is the same for both.
#[test]
fn the_scheme_refusal_names_the_file_remedy_for_a_file_url_alone() {
    let file = MmRefsError::SchemeNotAccepted {
        scheme: "file".to_string(),
        accepted: "data,http,https".to_string(),
    }
    .to_string();
    assert_eq!(
        file,
        "the selected worker does not fetch 'file' URLs (accepts: data,http,https); \
             file:// needs --allowed-local-media-path on the worker"
    );
    let data = MmRefsError::SchemeNotAccepted {
        scheme: "data".to_string(),
        accepted: "http,https".to_string(),
    }
    .to_string();
    assert_eq!(
        data,
        "the selected worker does not fetch 'data' URLs (accepts: http,https)"
    );
    assert!(!data.contains("allowed-local-media-path"));
}

/// The empty scheme of a bare path or a port-less host written without a
/// scheme (what the scheme read gives them; a host with a port reads as a
/// scheme of its own) is refused like any scheme the worker's labels leave
/// out: the accepted list alone, no local-media remedy, since a scheme-less
/// string is not a file URL.
#[test]
fn the_scheme_refusal_gives_an_empty_scheme_the_accepted_list_alone() {
    assert_eq!(url_scheme("images/a.png"), "");
    assert_eq!(url_scheme("example.com/a.png"), "");
    assert_eq!(url_scheme("localhost:8000/a.png"), "localhost");
    let bare = MmRefsError::SchemeNotAccepted {
        scheme: String::new(),
        accepted: "http,https".to_string(),
    }
    .to_string();
    assert_eq!(
        bare,
        "the selected worker does not fetch '' URLs (accepts: http,https)"
    );
    assert!(!bare.contains("allowed-local-media-path"));
}

#[test]
fn selection_check_requires_every_worker_and_scheme() {
    let plan_https = plan(vec![image_url("https://a/1.png")]);
    let single = WorkerSelection::Single {
        worker: capable("grpc://127.0.0.1:9600").into(),
    };
    ensure_selection_supports_media_refs(&single, &plan_https, PipelinePin::Any)
        .expect("capable single");

    let plain = WorkerSelection::Single {
        worker: worker(
            "grpc://127.0.0.1:9601",
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[],
        )
        .into(),
    };
    assert_eq!(
        ensure_selection_supports_media_refs(&plain, &plan_https, PipelinePin::Any),
        Err(MmRefsError::WorkerNotCapable)
    );

    let file_plan = plan(vec![image_url("file:///media/1.png")]);
    let err = ensure_selection_supports_media_refs(&single, &file_plan, PipelinePin::Any)
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
        )
        .into(),
        runtime_type: RuntimeType::Vllm,
    };
    assert_eq!(
        ensure_selection_supports_media_refs(&pd, &plan_https, PipelinePin::Any),
        Err(MmRefsError::WorkerNotCapable)
    );

    // Both workers fetch: the accepted schemes are the intersection of what
    // each advertising worker fetches.
    let file_prefill = worker(
        "grpc://127.0.0.1:9604",
        RuntimeType::Vllm,
        ConnectionMode::Grpc,
        &[
            (MM_PROCESSOR_LABEL, "inprocess"),
            (MM_MEDIA_REF_SCHEMES_LABEL, "http,https,data,file"),
        ],
    );
    let mixed = WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill: file_prefill,
        decode: capable("grpc://127.0.0.1:9605").into(),
        runtime_type: RuntimeType::Vllm,
    };
    ensure_selection_supports_media_refs(&mixed, &plan_https, PipelinePin::Any)
        .expect("https on both workers");
    assert_eq!(
        ensure_selection_supports_media_refs(&mixed, &file_plan, PipelinePin::Any),
        Err(MmRefsError::SchemeNotAccepted {
            scheme: "file".to_string(),
            accepted: "data,http,https".to_string(),
        })
    );

    // A worker that advertises no schemes leaves the other worker's set in force.
    let silent_decode = worker(
        "grpc://127.0.0.1:9606",
        RuntimeType::Vllm,
        ConnectionMode::Grpc,
        &[(MM_PROCESSOR_LABEL, "inprocess")],
    );
    let half = WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill: capable("grpc://127.0.0.1:9607"),
        decode: silent_decode.into(),
        runtime_type: RuntimeType::Vllm,
    };
    assert_eq!(
        ensure_selection_supports_media_refs(&half, &file_plan, PipelinePin::Any)
            .expect_err("prefill's set still applies")
            .code(),
        "media_ref_scheme_not_accepted"
    );
}

/// A worker running the gateway's own pipeline, which expands any
/// spec's anchor itself.
fn pipeline_worker(url: &str) -> Arc<dyn Worker> {
    worker(
        url,
        RuntimeType::Vllm,
        ConnectionMode::Grpc,
        &[
            (MM_PROCESSOR_LABEL, GATEWAY_PIPELINE_PROCESSOR),
            (MM_MEDIA_REF_SCHEMES_LABEL, "http,https,data"),
        ],
    )
}

/// A spec'd model whose image anchor the engine's own processor does not
/// expand.
fn unopted_placeholders() -> PlaceholderTokens {
    let mut placeholders = PlaceholderTokens::default();
    placeholders.insert(Modality::Image, "<|media_pad|>".to_string());
    placeholders
}

/// Only a spec'd model with a modality the engine's processor does not
/// expand needs the pipeline pin: an opted-in modality and a model
/// without a spec (whose template renders the engine's own placeholders)
/// go to any worker that processes references.
#[test]
fn only_unopted_modalities_of_a_specd_model_need_the_pipeline() {
    let image = plan(vec![image_url("https://a/1.png")]);
    assert!(plan_needs_gateway_pipeline(&image, &unopted_placeholders()));
    assert!(!plan_needs_gateway_pipeline(
        &image,
        &expandable_placeholders()
    ));
    assert!(!plan_needs_gateway_pipeline(
        &image,
        &PlaceholderTokens::worker_only()
    ));
    assert!(worker_runs_gateway_pipeline(
        pipeline_worker("grpc://127.0.0.1:9700").as_ref()
    ));
    assert!(!worker_runs_gateway_pipeline(
        capable("grpc://127.0.0.1:9701").as_ref()
    ));

    // The pin selection reads: the pipeline alone for an unopted anchor,
    // never the pipeline for audio, either otherwise.
    let audio = plan(vec![MediaContentPart::AudioUrl {
        url: "https://a/x.wav".to_string().into(),
        uuid: None,
    }]);
    let mut audio_opted = PlaceholderTokens::default();
    audio_opted.insert(Modality::Audio, "<|audio_pad|>".to_string());
    audio_opted.set_worker_expandable(Modality::Audio, true);
    assert_eq!(
        media_refs_pipeline_pin(&image, &unopted_placeholders()),
        PipelinePin::Only
    );
    assert_eq!(
        media_refs_pipeline_pin(&audio, &audio_opted),
        PipelinePin::Never
    );
    assert_eq!(
        media_refs_pipeline_pin(&image, &expandable_placeholders()),
        PipelinePin::Any
    );
    assert_eq!(
        media_refs_pipeline_pin(&image, &PlaceholderTokens::worker_only()),
        PipelinePin::Any
    );
}

/// Strict worker mode forwards a model the engine's processor cannot
/// expand when every media-processing worker of the model runs the
/// gateway's own pipeline; a fleet mixing in the engine's processors, or
/// one with no such worker yet, keeps the refusal that names the opt-in.
#[test]
fn worker_mode_forwards_unopted_models_to_a_pipeline_fleet() {
    let components = components(MmProcessingMode::Worker);
    let image = plan(vec![image_url("https://a/1.png")]);
    let unopted = unopted_placeholders();

    let pipeline_fleet = registry_with(vec![
        pipeline_worker("grpc://127.0.0.1:9710"),
        pipeline_worker("grpc://127.0.0.1:9711"),
    ]);
    assert_eq!(
        resolve_mm_processing(&components, &pipeline_fleet, MODEL, &image, &unopted)
            .expect("forwarded to the pipeline workers"),
        MmProcessing::Worker
    );

    let mixed = registry_with(vec![
        pipeline_worker("grpc://127.0.0.1:9712"),
        capable("grpc://127.0.0.1:9713"),
    ]);
    let err = resolve_mm_processing(&components, &mixed, MODEL, &image, &unopted)
        .expect_err("a worker on the engine's processor keeps the opt-in");
    assert_eq!(err, MmRefsError::ModelNotOptedIn(Modality::Image));
    assert!(
        err.to_string().contains(GATEWAY_PIPELINE_PROCESSOR),
        "{err}"
    );

    let empty = registry_with(vec![]);
    assert_eq!(
        resolve_mm_processing(&components, &empty, MODEL, &image, &unopted).expect_err("no fleet"),
        MmRefsError::ModelNotOptedIn(Modality::Image)
    );

    // The pipeline takes no audio, so an unopted audio anchor is refused
    // here, with the opt-in named, rather than forwarded to a refusal.
    let audio = plan(vec![MediaContentPart::AudioUrl {
        url: "https://a/x.wav".to_string().into(),
        uuid: None,
    }]);
    let mut unopted_audio = PlaceholderTokens::default();
    unopted_audio.insert(Modality::Audio, "<|audio_pad|>".to_string());
    assert_eq!(
        resolve_mm_processing(&components, &pipeline_fleet, MODEL, &audio, &unopted_audio)
            .expect_err("the pipeline expands no audio anchor"),
        MmRefsError::ModelNotOptedIn(Modality::Audio)
    );

    // An opted-in model needs no pipeline worker, as before.
    assert_eq!(
        resolve_mm_processing(
            &components,
            &mixed,
            MODEL,
            &image,
            &expandable_placeholders()
        )
        .expect("opted-in model forwards to any capable worker"),
        MmProcessing::Worker
    );
}

/// `auto` forwards an unopted model only to a fleet made entirely of
/// pipeline workers; one worker on the engine's processor sends the
/// model back to the gateway's own pipeline, as before.
#[test]
fn auto_forwards_unopted_models_only_to_a_uniform_pipeline_fleet() {
    let components = components(MmProcessingMode::Auto);
    let image = plan(vec![image_url("https://a/1.png")]);
    let unopted = unopted_placeholders();

    let pipeline_fleet = registry_with(vec![
        pipeline_worker("grpc://127.0.0.1:9720"),
        pipeline_worker("grpc://127.0.0.1:9721"),
    ]);
    assert_eq!(
        resolve_mm_processing(&components, &pipeline_fleet, MODEL, &image, &unopted)
            .expect("resolves"),
        MmProcessing::Worker
    );

    let mixed = registry_with(vec![
        pipeline_worker("grpc://127.0.0.1:9722"),
        capable("grpc://127.0.0.1:9723"),
    ]);
    assert_eq!(
        resolve_mm_processing(&components, &mixed, MODEL, &image, &unopted).expect("resolves"),
        MmProcessing::Router
    );
}

/// Selection pins an unopted model's references to pipeline workers on
/// every worker, and keeps audio off a pipeline worker, which takes none.
#[test]
fn selection_check_pins_unopted_models_to_pipeline_workers_and_audio_off_them() {
    let image = plan(vec![image_url("https://a/1.png")]);
    let audio = plan(vec![MediaContentPart::AudioUrl {
        url: "https://a/x.wav".to_string().into(),
        uuid: None,
    }]);
    let pipeline_single = WorkerSelection::Single {
        worker: pipeline_worker("grpc://127.0.0.1:9730").into(),
    };
    let engine_single = WorkerSelection::Single {
        worker: capable("grpc://127.0.0.1:9731").into(),
    };
    ensure_selection_supports_media_refs(&pipeline_single, &image, PipelinePin::Only)
        .expect("the pipeline worker takes the unopted model");
    assert_eq!(
        ensure_selection_supports_media_refs(&engine_single, &image, PipelinePin::Only),
        Err(MmRefsError::PipelineWorkerRequired)
    );
    ensure_selection_supports_media_refs(&engine_single, &image, PipelinePin::Any)
        .expect("an opted-in model goes to the engine's processor");

    let half = WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill: pipeline_worker("grpc://127.0.0.1:9732"),
        decode: capable("grpc://127.0.0.1:9733").into(),
        runtime_type: RuntimeType::Vllm,
    };
    assert_eq!(
        ensure_selection_supports_media_refs(&half, &image, PipelinePin::Only),
        Err(MmRefsError::PipelineWorkerRequired)
    );

    assert_eq!(
        ensure_selection_supports_media_refs(&pipeline_single, &audio, PipelinePin::Any),
        Err(MmRefsError::PipelineTakesNoAudio)
    );
    ensure_selection_supports_media_refs(&engine_single, &audio, PipelinePin::Any)
        .expect("the engine's processor takes audio");
    assert_eq!(
        MmRefsError::PipelineTakesNoAudio.code(),
        "multimodal_not_supported"
    );
    assert_eq!(
        MmRefsError::PipelineWorkerRequired.code(),
        "multimodal_worker_processing_unsupported_model"
    );
}

#[test]
fn assemble_keeps_prompt_order_and_maps_modalities() {
    let refs = assemble_media_refs(&plan(vec![
        image_url("https://a/1.png"),
        MediaContentPart::VideoUrl {
            url: "data:video/mp4;base64,AAAA".to_string().into(),
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
fn assemble_builds_the_data_url_of_a_base64_part() {
    let refs = assemble_media_refs(&plan(vec![base64_image("AAAA"), base64_audio("BBBB")]))
        .expect("assembles");
    let urls: Vec<&str> = refs.items.iter().map(|item| item.url.as_str()).collect();
    assert_eq!(
        urls,
        vec!["data:image/png;base64,AAAA", "data:audio/wav;base64,BBBB"]
    );
}

/// Hint detection classifies the parts and builds no URL: the data URL of
/// a base64 part is built once, at assembly, so detection allocates nothing
/// however large the payloads are.
#[cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]
#[test]
fn hint_detection_builds_no_data_url() {
    let plan = plan(vec![
        base64_image(&"A".repeat(4 << 20)),
        base64_audio(&"B".repeat(1 << 20)),
    ]);
    // Warm the one-time caches on the path (the env-read byte caps) so
    // the counter measures the per-request cost alone.
    assert_eq!(plan_carries_hints(&plan), Ok(false));
    let bytes = bytes_allocated_during(|| {
        assert_eq!(plan_carries_hints(&plan), Ok(false));
    });
    assert_eq!(bytes, 0, "detection allocated {bytes} bytes");
}

#[test]
fn audio_refs_require_an_expandable_model_and_stay_in_prompt_order() {
    let plan = plan(vec![
        MediaContentPart::AudioUrl {
            url: "data:audio/wav;base64,AAAA".to_string().into(),
            uuid: None,
        },
        image_url("https://a/1.png"),
    ]);
    assert!(plan.is_forwardable());
    let mut placeholders = expandable_placeholders();
    assert_eq!(
        ensure_worker_expandable(&plan, &placeholders, false),
        Err(MmRefsError::ModelNotOptedIn(Modality::Audio))
    );
    placeholders.set_worker_expandable(Modality::Audio, true);
    ensure_worker_expandable(&plan, &placeholders, false).unwrap();
    let refs = assemble_media_refs(&plan).unwrap();
    assert_eq!(refs.items[0].modality, common::Modality::Audio as i32);
    assert_eq!(refs.items[0].url, "data:audio/wav;base64,AAAA");
    assert_eq!(refs.items[1].modality, common::Modality::Image as i32);
}

#[test]
fn assemble_rejects_unsupported_parts_and_oversized_data_urls() {
    let err = assemble_media_refs(&plan(vec![MediaContentPart::AudioData {
        data: vec![0; 4],
        mime_type: Some("audio/wav".to_string()),
        uuid: None,
    }]))
    .expect_err("raw audio bytes have no reference form");
    assert_eq!(err, MmRefsError::UnsupportedPart("inline audio bytes"));

    let oversized = format!(
        "data:image/png;base64,{}",
        "A".repeat(media::image_max_input_bytes() * 4 / 3 + 8)
    );
    let err =
        assemble_media_refs(&plan(vec![image_url(&oversized)])).expect_err("oversized data url");
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
