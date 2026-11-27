use std::time::Duration;

use axum::body::to_bytes;
use http::StatusCode;
use llm_multimodal::{hub::HubFetchError, MediaConnectorError, Modality, MultiModalError};

use super::{
    config_unavailable_response, encoder_budget::ItemOverEncoderBudget, invalid_multimodal_request,
    processing_failed_message, processing_failed_response, record_decoder_availability,
    report_jpeg_decoder, Response,
};
use crate::{middleware::test_support::json_of, routers::common::retry::NonRetryable};

async fn status_and_error(response: Response) -> (u16, serde_json::Value) {
    let status = response.status().as_u16();
    (status, json_of(response).await)
}

/// A configuration file the model's hub failed to serve is a 502 under its
/// own code with a generic message: the hub's answer and its address stay
/// out of the body, and the response carries no non-retryable mark, since
/// the next request resolves the checkpoint again. Any other failure is
/// not this answer.
#[tokio::test]
async fn a_processor_config_the_hub_failed_to_serve_answers_a_502_without_the_cause() {
    let failed = anyhow::Error::from(HubFetchError {
        model: "org/name".to_string(),
        file: "preprocessor_config.json",
        cause: "request error: HTTP status server error (503 Service Unavailable) for url \
                    (http://hub.internal/org/name/resolve/main/preprocessor_config.json)"
            .to_string(),
    })
    .context("Failed to resolve model config directory for 'org/name'");
    let response = processing_failed_response(&failed);
    assert!(
        response.extensions().get::<NonRetryable>().is_none(),
        "a hub failure is retried by the next request"
    );
    let (status, json) = status_and_error(response).await;
    assert_eq!(status, 502);
    assert_eq!(json["error"]["code"], "multimodal_config_unavailable");
    let message = json["error"]["message"].as_str().unwrap();
    assert!(
        !message.contains("hub.internal") && !message.contains("503"),
        "the cause stays out of the body: {message}"
    );
    assert!(
        message.contains("configuration could not be fetched"),
        "the body says what was not fetched: {message}"
    );

    assert!(config_unavailable_response(&anyhow::anyhow!("decode failed")).is_none());
}

/// An item the engine could never place answers under a code of its own,
/// a 400 like every other processing failure, with the item's size and
/// the budget in the message; a failure of another kind keeps its code.
#[tokio::test]
async fn an_item_over_the_engines_budget_answers_under_its_own_code() {
    let over = anyhow::Error::new(ItemOverEncoderBudget {
        modality: Modality::Video,
        position: 1,
        count: 1,
        feature_tokens: 192_001,
        sampled_frames: Some(900),
        budget: 192_000,
    })
    .context("Failed to process the plan");
    let (status, json) = status_and_error(processing_failed_response(&over)).await;
    assert_eq!(status, 400);
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "multimodal_item_too_large");
    let message = json["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("192001 feature tokens (900 sampled frames)")
            && message.contains("192000-token encoder budget"),
        "{message}"
    );

    let (status, json) = status_and_error(processing_failed_response(&anyhow::anyhow!(
        "decode failed"
    )))
    .await;
    assert_eq!(status, 400);
    assert_eq!(json["error"]["code"], "multimodal_processing_failed");
}

/// A fetch that ran out of its budget is reported with its URL and the
/// budget, under the route's sentence, without the "failed to finalize"
/// context frame around it.
#[test]
fn a_processing_failure_names_its_cause() {
    let error = anyhow::Error::new(MultiModalError::Media(MediaConnectorError::Timeout {
        url: "https://media.example/clip.mp4".to_string(),
        budget: Duration::from_secs(30),
    }))
    .context("Failed to finalize multimodal tracker");
    assert_eq!(
        processing_failed_message(&error),
        "Multimodal processing failed: media fetch of https://media.example/clip.mp4 timed \
             out after 30s"
    );
}

/// The message is the route's sentence and the pipeline's verdict on the
/// request: the gateway's stage frames around the verdict are the log's
/// (the chain keeps them), and the verdict's own source is not repeated
/// after it. The tracker's own typed error under the frames reads its text
/// alone; a failure with neither keeps its whole chain, whose frames say
/// what failed.
#[test]
fn a_processing_failures_message_leaves_the_stage_frames_to_the_log() {
    let undecodable =
        image::ImageError::Unsupported(image::error::UnsupportedError::from_format_and_kind(
            image::error::ImageFormatHint::Unknown,
            image::error::UnsupportedErrorKind::Format(image::error::ImageFormatHint::Unknown),
        ));
    let decoder_said = undecodable.to_string();
    let error = anyhow::Error::new(MultiModalError::Media(MediaConnectorError::Image(
        undecodable,
    )))
    .context("Failed to finalize multimodal tracker")
    .context("Failed to process the plan");
    assert_eq!(
        processing_failed_message(&error),
        format!("Multimodal processing failed: image decode error: {decoder_said}")
    );
    assert!(
        format!("{error:#}").contains("Failed to finalize multimodal tracker: "),
        "the chain keeps the frames for the log: {error:#}"
    );

    let validation = anyhow::Error::new(MultiModalError::Validation(
        "placeholder 3 has no part".to_string(),
    ))
    .context("Failed to finalize multimodal tracker");
    assert_eq!(
        processing_failed_message(&validation),
        "Multimodal processing failed: tracker validation error: placeholder 3 has no part"
    );

    let config_read = anyhow::anyhow!("No such file or directory (os error 2)")
        .context("Failed to read config.json at /models/m/config.json");
    assert_eq!(
        processing_failed_message(&config_read),
        "Multimodal processing failed: Failed to read config.json at /models/m/config.json: \
             No such file or directory (os error 2)"
    );
    let ranges = anyhow::anyhow!("feature range 0 has zero length")
        .context("Invalid explicit feature ranges in image replacement 2");
    assert_eq!(
        processing_failed_message(&ranges),
        "Multimodal processing failed: Invalid explicit feature ranges in image replacement 2: \
             feature range 0 has zero length"
    );
}

/// The status, the `x-smg-error-code` header and the JSON body of an answer.
async fn answer(response: Response) -> (StatusCode, String, serde_json::Value) {
    let status = response.status();
    let header = response
        .headers()
        .get("x-smg-error-code")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("the answer's body");
    let body = serde_json::from_slice(&bytes).expect("a JSON envelope");
    (status, header, body)
}

/// Media that could not be prepared is the request's to fix: a 400 under
/// the media request code, with the cause after the route's lead.
#[tokio::test]
async fn invalid_multimodal_input_maps_to_bad_request() {
    let (status, code, body) =
        answer(invalid_multimodal_request("unsupported audio modality")).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code, "invalid_multimodal_request");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert_eq!(
        body["error"]["message"],
        "Invalid multimodal request: unsupported audio modality"
    );
}

/// A fetch that ran out of its budget and an origin that answered 5xx are
/// the media host's trouble for now, a 502 `server_error` under
/// `media_fetch_failed`; a host that cannot be resolved is the request's
/// own input, a 400 `invalid_request_error` under the same code; a
/// malformed data URL, a parameter out of range and a host the policy
/// refuses are the request's own as sent, 400s under
/// `invalid_multimodal_request`; a clip
/// over the decoded-size cap and an item over the input cap are 400s
/// under `media_too_large`; a scratch file that failed on this side and
/// a preprocessing task that died are 500 `server_error`s; a failure with
/// no connector verdict keeps the pipeline's own 400 code. The message is
/// the route's sentence and the cause in every case.
#[tokio::test]
async fn a_pipeline_failure_is_answered_by_the_connectors_verdict() {
    let finalize = |media: MediaConnectorError| {
        anyhow::Error::new(MultiModalError::Media(media))
            .context("Failed to finalize multimodal tracker")
    };
    let origin_answered = |status: u16| {
        let response = reqwest::Response::from(
            http::Response::builder()
                .status(status)
                .body("")
                .expect("a response"),
        );
        finalize(MediaConnectorError::Http(
            response
                .error_for_status()
                .expect_err("a 5xx is the fetch's error"),
        ))
    };
    let died = tokio::task::spawn_blocking(|| panic!("the preprocessor died"))
        .await
        .expect_err("the task panicked");
    let died = anyhow::Error::new(died).context("Preprocessing task panicked");
    let cases = [
        (
            finalize(MediaConnectorError::Timeout {
                url: "https://media.example/video_251mb.mp4".to_string(),
                budget: Duration::from_secs(30),
            }),
            StatusCode::BAD_GATEWAY,
            "media_fetch_failed",
            "server_error",
        ),
        (
            origin_answered(503),
            StatusCode::BAD_GATEWAY,
            "media_fetch_failed",
            "server_error",
        ),
        (
            origin_answered(404),
            StatusCode::BAD_REQUEST,
            "media_fetch_failed",
            "invalid_request_error",
        ),
        (
            finalize(MediaConnectorError::UnresolvableHost {
                host: "media.invalid".to_string(),
                source: std::io::Error::from(std::io::ErrorKind::NotFound),
            }),
            StatusCode::BAD_REQUEST,
            "media_fetch_failed",
            "invalid_request_error",
        ),
        (
            finalize(MediaConnectorError::DataUrl(
                "missing comma in data url".to_string(),
            )),
            StatusCode::BAD_REQUEST,
            "invalid_multimodal_request",
            "invalid_request_error",
        ),
        (
            finalize(MediaConnectorError::InvalidMaxLongSidePixel {
                value: 0,
                factor: 28,
            }),
            StatusCode::BAD_REQUEST,
            "invalid_multimodal_request",
            "invalid_request_error",
        ),
        (
            finalize(MediaConnectorError::DisallowedAddress {
                host: "127.0.0.1".to_string(),
                address: std::net::IpAddr::from([127, 0, 0, 1]),
                class: "loopback",
            }),
            StatusCode::BAD_REQUEST,
            "invalid_multimodal_request",
            "invalid_request_error",
        ),
        (
            finalize(MediaConnectorError::DecodedTooLarge {
                media: "video RGB",
                bytes: 2_208_384_000,
                limit: 1_073_741_824,
                knob: "SMG_VIDEO_MAX_DECODED_BYTES",
            }),
            StatusCode::BAD_REQUEST,
            "media_too_large",
            "invalid_request_error",
        ),
        (
            finalize(MediaConnectorError::PayloadTooLarge {
                media: "video",
                limit: 268_435_456,
            }),
            StatusCode::BAD_REQUEST,
            "media_too_large",
            "invalid_request_error",
        ),
        (
            finalize(MediaConnectorError::DecodeIo(std::io::Error::other(
                "No space left on device (os error 28)",
            ))),
            StatusCode::INTERNAL_SERVER_ERROR,
            "multimodal_processing_failed",
            "server_error",
        ),
        (
            died,
            StatusCode::INTERNAL_SERVER_ERROR,
            "multimodal_processing_failed",
            "server_error",
        ),
        (
            anyhow::Error::new(MultiModalError::UnsupportedContent("image_embeds"))
                .context("Failed to push content part"),
            StatusCode::BAD_REQUEST,
            "multimodal_processing_failed",
            "invalid_request_error",
        ),
        (
            anyhow::anyhow!("Multimodal not supported for model: m"),
            StatusCode::BAD_REQUEST,
            "multimodal_processing_failed",
            "invalid_request_error",
        ),
    ];
    for (error, status, code, kind) in cases {
        let message = processing_failed_message(&error);
        let (answered, header, body) = answer(processing_failed_response(&error)).await;
        assert_eq!(answered, status, "{message}: {body}");
        assert_eq!(header, code, "{message}");
        assert_eq!(body["error"]["code"], code, "{message}: {body}");
        assert_eq!(body["error"]["type"], kind, "{message}: {body}");
        assert_eq!(body["error"]["message"], message, "{body}");
    }
}

/// A video part on a process without a video decoder is the media
/// stage's own refusal: a 400 `invalid_multimodal_request` whose message
/// names what is missing, never a server error.
#[tokio::test]
async fn a_video_part_without_a_decoder_is_the_media_stages_refusal() {
    let refused = anyhow::Error::new(super::process::VideoDecoderUnavailable {
        why: "ffmpeg executable not found; install ffmpeg to decode video_url inputs",
    });
    let (status, header, body) = answer(processing_failed_response(&refused)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(header, "invalid_multimodal_request");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert_eq!(
        body["error"]["message"],
        "Invalid multimodal request: video input cannot be decoded by this process: ffmpeg \
             executable not found; install ffmpeg to decode video_url inputs",
        "{body}"
    );
}

/// The media host's trouble is answered as the host's (a 502) and is
/// terminal for the retry layer: another attempt, on this worker or
/// another, would fetch from the same host; the request's own fault stays
/// a 400, terminal by its status.
#[test]
fn a_media_hosts_failure_is_terminal_for_the_retry_layer() {
    use crate::routers::common::retry::is_retryable_response;

    let timed_out = anyhow::Error::new(MultiModalError::Media(MediaConnectorError::Timeout {
        url: "https://media.example/clip.mp4".to_string(),
        budget: Duration::from_secs(30),
    }))
    .context("Failed to finalize multimodal tracker");
    let response = processing_failed_response(&timed_out);
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(
        !is_retryable_response(&response),
        "a media host's failure is not replayed"
    );

    let wrong = anyhow::Error::new(MultiModalError::Media(
        MediaConnectorError::UnsupportedScheme("ftp".to_string()),
    ))
    .context("Failed to finalize multimodal tracker");
    let response = processing_failed_response(&wrong);
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(!is_retryable_response(&response));
}

/// The decoder availability gauges land in the recorder in force when the
/// report runs: the gateway installs its recorder first (`server.rs`),
/// so `smg_mm_turbojpeg_available` and `smg_mm_webp_available` are on
/// /metrics under every policy; a report made before the recorder would
/// set them on the no-op one, once, and they would never appear.
#[test]
fn the_decoder_availability_gauges_land_in_the_installed_recorder() {
    use metrics_exporter_prometheus::PrometheusBuilder;

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let (available, webp) = ::metrics::with_local_recorder(&recorder, record_decoder_availability);
    let rendered = handle.render();
    let reads = |name: &str, value: bool| {
        rendered
            .lines()
            .any(|line| line.starts_with(name) && line.ends_with(if value { " 1" } else { " 0" }))
    };
    assert!(reads("smg_mm_turbojpeg_available", available), "{rendered}");
    assert!(reads("smg_mm_webp_available", webp), "{rendered}");
}

/// The report sets the gauges on every call, the log lines once: a report
/// made before the gateway's recorder exists (a caller building the
/// components early) does not leave the start-up's own report with
/// nothing to set.
#[test]
fn a_second_report_still_sets_the_gauges_on_the_recorder_in_force() {
    use metrics_exporter_prometheus::PrometheusBuilder;

    // The first report, on whatever recorder is in force here: the once
    // block is consumed by it (or was, by another test of this process).
    report_jpeg_decoder();

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    ::metrics::with_local_recorder(&recorder, report_jpeg_decoder);
    let rendered = handle.render();
    for gauge in ["smg_mm_turbojpeg_available", "smg_mm_webp_available"] {
        assert!(
            rendered
                .lines()
                .any(|line| line.starts_with(gauge)
                    && (line.ends_with(" 1") || line.ends_with(" 0"))),
            "{gauge} is set by the second report:\n{rendered}"
        );
    }
}

/// The verdict the processing arm writes its WARN on: the connector's
/// client verdict (also under a context), an item over the encoder's
/// budget and a video part without a decoder are the request's own
/// media; a failure with no verdict in its chain is not.
#[test]
fn only_an_explicit_verdict_makes_a_failure_the_requests_media() {
    let refused = anyhow::Error::new(MediaConnectorError::InvalidUrl("x".to_string()))
        .context("Failed to finalize multimodal tracker");
    assert!(super::is_request_media_fault(&refused));
    assert!(super::is_request_media_fault(&anyhow::Error::new(
        super::process::VideoDecoderUnavailable { why: "no decoder" }
    )));
    assert!(super::is_request_media_fault(&anyhow::Error::new(
        ItemOverEncoderBudget {
            modality: Modality::Image,
            position: 1,
            count: 1,
            feature_tokens: 2,
            sampled_frames: None,
            budget: 1,
        }
    )));
    assert!(!super::is_request_media_fault(&anyhow::anyhow!(
        "Image preprocessing failed"
    )));
}
