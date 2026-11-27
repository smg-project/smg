use std::fmt::Write as _;

use smg_worker::identity::basic::{
    ConnectionMode, RuntimeType, Worker, WorkerType, MOONCAKE_CONNECTOR, NIXL_CONNECTOR,
};

use super::*;
use crate::{
    middleware::test_support::text_of,
    routers::{
        common::{attempt_ledger::FailedSide, retry::is_retryable_response},
        grpc::{
            backend_client::BasicWorkerBuilder,
            common::stages::request_execution::tests::{
                answers_after, fails_now, failure, media_refs_request, never_answers,
                tokenspeed_pair, vllm_pd_mm_request,
            },
            proto_wrapper::FanoutStream,
        },
    },
};

fn three_second_decode_bound() -> DecodeBound {
    DecodeBound {
        deadline: Instant::now() + Duration::from_secs(3),
        bound: Duration::from_secs(3),
    }
}

/// A sequential decode call that has returned no stream by its bound is
/// given up on at the bound, not at the request's timeout: the verdict
/// names the silence, and the attempt answers a 504 under its own code
/// that the retry layer may take to another decode worker, this one
/// excluded.
#[tokio::test(start_paused = true)]
async fn a_sequential_decode_dispatch_past_its_bound_is_given_up_on_at_the_bound() {
    let workers = tokenspeed_pair();
    let decode = Arc::clone(workers.decode_worker().expect("a pair"));
    let started = tokio::time::Instant::now();
    let result = dispatch_decode_within(
        Some(three_second_decode_bound()),
        never_answers(),
        Some(Arc::clone(&decode)),
        metrics_labels::CONNECTION_GRPC,
        "chatcmpl-decode-silent".to_string(),
        256,
    )
    .await;
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_secs(3) && elapsed < Duration::from_secs(4),
        "given up on at the bound, not later: {elapsed:?}"
    );
    let status = failure(result);
    assert_eq!(status.code(), tonic::Code::DeadlineExceeded);
    assert!(is_decode_side_silence(&status), "{status:?}");
    assert!(!is_prefill_silence(&status), "{status:?}");

    let ledger = AttemptLedger::default();
    let response =
        sequential_decode_failure(&status, &workers, &ledger, metrics_labels::CONNECTION_GRPC);
    assert_eq!(response.status(), http::StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(
        response
            .headers()
            .get(error::HEADER_X_SMG_ERROR_CODE)
            .and_then(|value| value.to_str().ok()),
        Some("decode_worker_silent")
    );
    assert!(
        response.extensions().get::<FailedSide>().is_some(),
        "the retry window closes on the decode worker alone"
    );
    assert!(
        is_retryable_response(&response),
        "another decode worker may take the request"
    );
}

/// A decode call that answers, or fails on its own, within its bound
/// passes through as it is; an engine's own deadline is not the verdict;
/// without a bound the call is awaited as before.
#[tokio::test(start_paused = true)]
async fn a_decode_that_answers_within_its_bound_passes_through() {
    let workers = tokenspeed_pair();
    let decode = Arc::clone(workers.decode_worker().expect("a pair"));
    let status = failure(
        dispatch_decode_within(
            Some(three_second_decode_bound()),
            answers_after(Duration::from_secs(1)),
            Some(decode),
            metrics_labels::CONNECTION_GRPC,
            "chatcmpl-decode-late".to_string(),
            256,
        )
        .await,
    );
    assert_eq!(status.message(), "late side");
    assert!(!is_decode_side_silence(&status));
    assert!(!is_decode_side_silence(&tonic::Status::deadline_exceeded(
        "bootstrap timeout"
    )));
    let status = failure(
        dispatch_decode_within(
            None,
            fails_now("refused"),
            None,
            metrics_labels::CONNECTION_GRPC,
            "chatcmpl-decode-unbounded".to_string(),
            256,
        )
        .await,
    );
    assert_eq!(status.message(), "refused");
}

/// A decode stream that answers (here: ends) within its bound is left
/// for its reader as it was, and what came is named.
#[tokio::test]
async fn a_decode_stream_that_answers_within_its_bound_is_left_for_its_reader() {
    let workers = tokenspeed_pair();
    let decode = Arc::clone(workers.decode_worker().expect("a pair"));
    let mut stream =
        ProtoStream::Fanout(FanoutStream::new(Vec::new())).tracked_decode_side(Arc::clone(&decode));
    let first = first_decode_response_within(
        &mut stream,
        Some(three_second_decode_bound()),
        Some(&decode),
        metrics_labels::CONNECTION_GRPC,
        "chatcmpl-decode-ended",
        256,
    )
    .await
    .expect("ended within the bound");
    assert_eq!(first, FirstItem::End);
    assert!(stream.next().await.is_none(), "the end is replayed");
}

/// The decode side's first-token bound is sized from the prefill worker
/// only when the decode side pulls its prompt KV from it: a handoff on a
/// request the KV parameters were relayed for. A fan-out the legacy
/// Mooncake arm hands off in name only, and any side without a handoff,
/// recompute the prompt on the decode worker, whose rate sizes the bound.
#[test]
fn the_decode_bound_follows_the_worker_that_prefills_the_prompt() {
    let prefill: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://prefill:30000")
            .worker_type(WorkerType::Prefill)
            .connection_mode(ConnectionMode::Grpc)
            .build(),
    );
    let decode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode:30000")
            .worker_type(WorkerType::Decode)
            .connection_mode(ConnectionMode::Grpc)
            .build(),
    );
    let selection = WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill,
        decode: decode.into(),
        runtime_type: RuntimeType::Vllm,
    };
    let prefilling = |handed_off, relay_kv_params| {
        worker_prefilling_before_decode(&selection, handed_off, relay_kv_params)
            .map(|worker| worker.url().to_string())
    };
    assert_eq!(
        prefilling(true, true).as_deref(),
        Some("grpc://prefill:30000")
    );
    assert_eq!(
        prefilling(true, false).as_deref(),
        Some("grpc://decode:30000")
    );
    assert_eq!(
        prefilling(false, true).as_deref(),
        Some("grpc://decode:30000")
    );
    assert_eq!(
        prefilling(false, false).as_deref(),
        Some("grpc://decode:30000")
    );
}

#[test]
fn mooncake_prefill_params_carry_transfer_id() {
    let value: serde_json::Value =
        serde_json::from_str(&mooncake_prefill_params("xfer-abc")).unwrap();
    assert_eq!(value["do_remote_decode"], true);
    assert_eq!(value["do_remote_prefill"], false);
    assert_eq!(value["transfer_id"], "xfer-abc");
    assert_eq!(value.as_object().unwrap().len(), 3);
}

#[test]
fn mooncake_decode_params_synthesize_full_handoff() {
    let value: serde_json::Value = serde_json::from_str(&mooncake_decode_params(
        "xfer-abc", "engine-1", "10.0.0.1", 8998,
    ))
    .unwrap();
    assert_eq!(value["do_remote_decode"], false);
    assert_eq!(value["do_remote_prefill"], true);
    assert_eq!(value["transfer_id"], "xfer-abc");
    assert_eq!(value["remote_engine_id"], "engine-1");
    assert_eq!(value["remote_bootstrap_addr"], "http://10.0.0.1:8998");
    assert_eq!(value.as_object().unwrap().len(), 5);
}

#[test]
fn nixl_prefill_kv_params_is_valid_json() {
    let value: serde_json::Value = serde_json::from_str(NIXL_PREFILL_KV_PARAMS).unwrap();
    assert_eq!(value["do_remote_decode"], true);
    assert_eq!(value["do_remote_prefill"], false);
    assert_eq!(value.as_object().unwrap().len(), 2);
}

fn identity() -> vllm::MediaIdentity {
    vllm::MediaIdentity {
        prompt_token_ids: vec![7, 8, 100, 100, 100, 9],
        mm_inputs: Some(vllm::MultimodalInputs {
            mm_hashes: vec!["h1".to_string(), "h2".to_string()],
            mm_placeholders: vec![
                vllm::PlaceholderRange {
                    offset: 2,
                    length: 2,
                },
                vllm::PlaceholderRange {
                    offset: 4,
                    length: 1,
                },
            ],
            model_specific_tensors: std::collections::HashMap::from([(
                "image_grid_thw".to_string(),
                vllm::TensorData {
                    shape: vec![2, 3],
                    dtype: "int64".to_string(),
                    payload: Some(vllm::tensor_data::Payload::Inline(vec![0; 48].into())),
                },
            )]),
            batched_keys: vec!["image_grid_thw".to_string()],
            modality: smg_grpc_client::common_proto::Modality::Image as i32,
            ..Default::default()
        }),
        extra_mm_inputs: vec![],
    }
}

/// Without a KV handoff decode recomputes the prompt locally, so it needs
/// the media itself and the identity must not replace its references:
/// n>1 never relays, and a NIXL prefill that returned no params does not
/// hand off either, even when it did return an identity.
#[test]
fn n_greater_than_one_keeps_media_refs_on_decode() {
    let mut decode = media_refs_request("n2", 2).clone_without_mm_pixels();
    apply_prefill_media_identity(&mut decode, false, false, false, Some(&identity()));
    assert!(decode.has_vllm_media_refs());
    let ProtoGenerateRequest::Vllm(request) = &decode else {
        panic!("expected vLLM request");
    };
    assert!(request.mm_inputs.is_none());
}

/// Legacy Mooncake injects its host and port into every decode side, n>1
/// included, so `handed_off` alone does not say that decode will pull
/// its prompt KV from prefill: an n>1 side recomputes locally and keeps
/// its references whatever the prefill returned.
#[test]
fn n_greater_than_one_keeps_media_refs_despite_a_legacy_mooncake_handoff() {
    let mut decode = media_refs_request("n2-mooncake", 2).clone_without_mm_pixels();
    apply_prefill_media_identity(&mut decode, true, false, false, Some(&identity()));
    assert!(decode.has_vllm_media_refs());
    let ProtoGenerateRequest::Vllm(request) = &decode else {
        panic!("expected vLLM request");
    };
    assert!(request.mm_inputs.is_none());
}

#[test]
fn an_identity_without_a_kv_handoff_keeps_the_references() {
    let mut decode = media_refs_request("no-kv", 1).clone_without_mm_pixels();
    apply_prefill_media_identity(&mut decode, false, true, true, Some(&identity()));
    assert!(decode.has_vllm_media_refs());
    let ProtoGenerateRequest::Vllm(request) = &decode else {
        panic!("expected vLLM request");
    };
    assert!(request.mm_inputs.is_none());
}

#[test]
fn decode_side_from_media_identity_has_no_refs_and_no_pixels() {
    let mut decode = media_refs_request("relay", 1).clone_without_mm_pixels();
    apply_prefill_media_identity(&mut decode, true, true, true, Some(&identity()));
    assert!(!decode.has_vllm_media_refs());
    let ProtoGenerateRequest::Vllm(request) = &decode else {
        panic!("expected vLLM request");
    };
    assert!(request.media_refs.is_none());
    let mm = request.mm_inputs.as_ref().expect("identity mm_inputs");
    assert!(mm.pixel_values.is_none());
    assert_eq!(mm.mm_hashes.len(), 2);
    assert_eq!(mm.mm_placeholders.len(), 2);
    assert!(mm.model_specific_tensors.contains_key("image_grid_thw"));
    let Some(vllm::generate_request::Input::Tokenized(tokenized)) = &request.input else {
        panic!("expected tokenized input");
    };
    assert_eq!(tokenized.input_ids, vec![7, 8, 100, 100, 100, 9]);
    assert_eq!(tokenized.original_text, "describe <|image|>");
}

/// An identity the side cannot take, empty ids or a side that is not
/// tokenized, leaves the references in place: better a reprocessed
/// decode than an empty prompt.
#[test]
fn an_unusable_identity_leaves_the_decode_side_untouched() {
    let mut empty = identity();
    empty.prompt_token_ids.clear();
    let mut decode = media_refs_request("empty", 1).clone_without_mm_pixels();
    apply_prefill_media_identity(&mut decode, true, true, true, Some(&empty));
    assert!(decode.has_vllm_media_refs());
    let ProtoGenerateRequest::Vllm(request) = &decode else {
        panic!("expected vLLM request");
    };
    assert!(request.mm_inputs.is_none());

    let mut decode = media_refs_request("text", 1).clone_without_mm_pixels();
    if let ProtoGenerateRequest::Vllm(request) = &mut decode {
        request.input = Some(vllm::generate_request::Input::Text(
            "describe <|image|>".to_string(),
        ));
    }
    apply_prefill_media_identity(&mut decode, true, true, true, Some(&identity()));
    assert!(decode.has_vllm_media_refs());
    let ProtoGenerateRequest::Vllm(request) = &decode else {
        panic!("expected vLLM request");
    };
    assert!(request.mm_inputs.is_none());
    assert!(matches!(
        request.input,
        Some(vllm::generate_request::Input::Text(_))
    ));
}

/// An older servicer returns no identity: the decode side keeps its
/// references and reprocesses, as before; the same when no identity was
/// asked for (a prefill request without KV params).
#[test]
fn missing_identity_leaves_the_decode_side_untouched() {
    for solicited in [true, false] {
        let mut decode = media_refs_request("old", 1).clone_without_mm_pixels();
        apply_prefill_media_identity(&mut decode, true, true, solicited, None);
        assert!(decode.has_vllm_media_refs(), "solicited={solicited}");
    }
}

#[test]
fn sequential_pd_decode_form_picks_the_side_for_the_worker() {
    let mm_request = vllm_pd_mm_request();

    // Full-vision decode worker: today's forms, untouched.
    assert_eq!(
        sequential_pd_decode_form(&mm_request, false, true).expect("vision decode"),
        SequentialPdDecodeForm::IdentityOnly
    );
    assert_eq!(
        sequential_pd_decode_form(&mm_request, false, false).expect("vision decode n>1"),
        SequentialPdDecodeForm::Full
    );

    // Language-model-only decode worker, no grids (MiniMax-style): the
    // fully stripped side. A text request takes the same side harmlessly.
    let mut stripable = vllm_pd_mm_request();
    let ProtoGenerateRequest::Vllm(req) = &mut stripable else {
        panic!("expected vLLM request");
    };
    req.media_refs = None;
    req.mm_inputs
        .as_mut()
        .unwrap()
        .model_specific_tensors
        .clear();
    assert_eq!(
        sequential_pd_decode_form(&stripable, true, true).expect("stripped side"),
        SequentialPdDecodeForm::TextPlusHashes
    );
    let text = ProtoGenerateRequest::Vllm(Box::default());
    assert_eq!(
        sequential_pd_decode_form(&text, true, true).expect("text request"),
        SequentialPdDecodeForm::TextPlusHashes
    );
    assert_eq!(
        sequential_pd_decode_form(&text, true, false).expect("text n>1"),
        SequentialPdDecodeForm::TextPlusHashes
    );
}

/// The refusals of a pool with no vision-capable decode worker (worker
/// selection pairs one when the model has it): each names its code and
/// says that none is paired.
#[tokio::test]
async fn sequential_pd_decode_form_rejects_incompatible_pairings() {
    async fn refusal(response: Response) -> String {
        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
        let text = text_of(response).await;
        assert!(
            text.contains("no decode worker with a vision encoder is paired"),
            "{text}"
        );
        text
    }

    // M-RoPE grids + language-model-only decode: cannot strip (positions
    // would be wrong), cannot send (encoder-cache budget 0).
    let gridded = vllm_pd_mm_request();
    let response = sequential_pd_decode_form(&gridded, true, true)
        .expect_err("mRoPE grids need a vision-capable decode worker");
    assert!(refusal(response)
        .await
        .contains("pd_decode_language_model_only_mrope"));

    // Media references + language-model-only decode: the side would have
    // to expand the anchors itself.
    let mut refs_only = vllm_pd_mm_request();
    let ProtoGenerateRequest::Vllm(req) = &mut refs_only else {
        panic!("expected vLLM request");
    };
    req.mm_inputs = None;
    req.extra_mm_inputs.clear();
    let response = sequential_pd_decode_form(&refs_only, true, true)
        .expect_err("media refs need a processing decode worker");
    assert!(refusal(response)
        .await
        .contains("pd_decode_language_model_only_media_refs"));

    // n>1 + mm + language-model-only decode: no KV handoff, no local
    // recompute either.
    let mut fanout = vllm_pd_mm_request();
    let ProtoGenerateRequest::Vllm(req) = &mut fanout else {
        panic!("expected vLLM request");
    };
    req.media_refs = None;
    req.mm_inputs
        .as_mut()
        .unwrap()
        .model_specific_tensors
        .clear();
    let response = sequential_pd_decode_form(&fanout, true, false)
        .expect_err("n>1 cannot recompute on a pixel-less decode worker");
    assert!(refusal(response)
        .await
        .contains("pd_decode_language_model_only_n_samples"));
}

#[test]
fn kv_transfer_params_json_request_roundtrip() {
    let mut request = ProtoGenerateRequest::Vllm(Box::default());
    request.set_kv_transfer_params_json(NIXL_PREFILL_KV_PARAMS.to_string());
    let ProtoGenerateRequest::Vllm(req) = request else {
        panic!("expected vLLM request");
    };
    assert_eq!(
        req.kv_transfer_params_json.as_deref(),
        Some(NIXL_PREFILL_KV_PARAMS)
    );
}

/// A streaming text request of `n` samples, the client's side of every
/// recorded pair below.
fn recorded_request(n: u32) -> ProtoGenerateRequest {
    ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
        request_id: "chatcmpl-recorded-pair".to_string(),
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                original_text: "the pair on the wire".to_string(),
                input_ids: vec![11, 12, 13, 14, 15],
            },
        )),
        sampling_params: Some(vllm::SamplingParams {
            temperature: Some(0.7),
            top_p: 0.9,
            max_tokens: Some(48),
            n,
            seed: Some(7),
            ..Default::default()
        }),
        stream: true,
        ..Default::default()
    }))
}

/// A prefill/decode pair whose prefill worker reports `connector`, and the
/// engine id a worker reports when it has one.
fn connector_pair(connector: &str, engine_id: Option<&str>) -> WorkerSelection {
    let mut prefill = BasicWorkerBuilder::new("grpc://prefill:30000")
        .worker_type(WorkerType::Prefill)
        .runtime_type(RuntimeType::Vllm)
        .connection_mode(ConnectionMode::Grpc)
        .kv_connector(connector)
        .bootstrap_port(Some(8998));
    if let Some(engine_id) = engine_id {
        prefill = prefill.kv_engine_id(engine_id);
    }
    let prefill: Arc<dyn Worker> = Arc::new(prefill.build());
    let decode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode:30000")
            .worker_type(WorkerType::Decode)
            .runtime_type(RuntimeType::Vllm)
            .connection_mode(ConnectionMode::Grpc)
            .build(),
    );
    WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill,
        decode: decode.into(),
        runtime_type: RuntimeType::Vllm,
    }
}

/// The check of a plan's refusals answers as the plan's own decision
/// does: the same refusal, by its code, for every pairing and request
/// the decision refuses, and none where the decision makes a plan.
#[test]
fn the_check_of_a_plans_refusals_answers_as_the_plan_does() {
    use smg_worker::{identity::basic::MORIIO_CONNECTOR, labels::SUPPORTS_VISION_LABEL};

    let side = |url: &str, worker_type, connector: Option<&str>, vision: Option<&str>| {
        let mut worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .runtime_type(RuntimeType::Vllm)
            .connection_mode(ConnectionMode::Grpc);
        if let Some(connector) = connector {
            worker = worker.kv_connector(connector);
        }
        if let Some(vision) = vision {
            worker = worker.label(SUPPORTS_VISION_LABEL, vision);
        }
        let worker: Arc<dyn Worker> = Arc::new(worker.build());
        worker
    };
    let pair = |prefill: Option<&str>, decode: Option<&str>, vision: Option<&str>| {
        WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: side("grpc://prefill:30000", WorkerType::Prefill, prefill, None),
            decode: side("grpc://decode:30000", WorkerType::Decode, decode, vision).into(),
            runtime_type: RuntimeType::Vllm,
        }
    };
    let code = |refused: Response| {
        refused
            .headers()
            .get(error::HEADER_X_SMG_ERROR_CODE)
            .and_then(|code| code.to_str().ok())
            .map(str::to_owned)
    };

    // A multimodal request in the three forms a language-model-only
    // decode is refused, as the decode form's own case builds them.
    let gridded = vllm_pd_mm_request();
    let mut refs_only = vllm_pd_mm_request();
    let ProtoGenerateRequest::Vllm(req) = &mut refs_only else {
        panic!("expected vLLM request");
    };
    req.mm_inputs = None;
    req.extra_mm_inputs.clear();
    // Pixels alone in one sample, which such a decode takes: the same
    // pixels in two samples are the third form it is refused.
    let mut pixels = vllm_pd_mm_request();
    let ProtoGenerateRequest::Vllm(req) = &mut pixels else {
        panic!("expected vLLM request");
    };
    req.media_refs = None;
    req.mm_inputs
        .as_mut()
        .unwrap()
        .model_specific_tensors
        .clear();
    req.sampling_params.get_or_insert_with(Default::default).n = 1;
    let mut fanout = pixels.clone();
    let ProtoGenerateRequest::Vllm(req) = &mut fanout else {
        panic!("expected vLLM request");
    };
    req.sampling_params.get_or_insert_with(Default::default).n = 2;
    let text = recorded_request(1);

    let nixl = Some(NIXL_CONNECTOR);
    let moriio = Some(MORIIO_CONNECTOR);
    let rows = [
        ("a NIXL pair", pair(nixl, None, None), &text, None),
        (
            "a MoRI-IO decode",
            pair(nixl, moriio, None),
            &text,
            Some("moriio_grpc_pd_unsupported"),
        ),
        (
            "a MoRI-IO prefill",
            pair(moriio, None, None),
            &text,
            Some("moriio_grpc_pd_unsupported"),
        ),
        (
            "grids for a language-model-only decode",
            pair(nixl, None, Some("false")),
            &gridded,
            Some("pd_decode_language_model_only_mrope"),
        ),
        (
            "media references for a language-model-only decode",
            pair(nixl, None, Some("false")),
            &refs_only,
            Some("pd_decode_language_model_only_media_refs"),
        ),
        (
            "samples of a multimodal prompt for a language-model-only decode",
            pair(nixl, None, Some("false")),
            &fanout,
            Some("pd_decode_language_model_only_n_samples"),
        ),
        (
            "one sample of a multimodal prompt for a language-model-only decode",
            pair(nixl, None, Some("false")),
            &pixels,
            None,
        ),
        (
            "text for a language-model-only decode",
            pair(nixl, None, Some("false")),
            &text,
            None,
        ),
        (
            "grids for a decode with the vision encoder",
            pair(nixl, None, Some("true")),
            &gridded,
            None,
        ),
    ];
    for (name, workers, request, refused) in rows {
        // A refusal is read as one whatever it bears: a refusal with no
        // code is not a row's `None`.
        let refused = refused.map(|code| Some(code.to_owned()));
        let checked = SequentialPlan::refuses(request, &workers).err().map(code);
        let decided = SequentialPlan::decide(request, &workers).err().map(code);
        assert_eq!(checked, refused, "{name}: the check");
        assert_eq!(decided, refused, "{name}: the plan's decision");
    }
}

/// The parameters a prefill engine returns under a NIXL handoff.
const NIXL_RETURNED: &str = r#"{"do_remote_prefill":true,"remote_block_ids":[3,4,5],"remote_engine_id":"prefill-engine","remote_host":"10.0.0.2","remote_port":5600}"#;

/// The bytes a request is sent as, in hex.
fn wire(request: &ProtoGenerateRequest) -> String {
    let ProtoGenerateRequest::Vllm(request) = request else {
        panic!("the sequential pair rides one wire");
    };
    prost::Message::encode_to_vec(request.as_ref())
        .into_iter()
        .fold(String::new(), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// The pair the plan sends for `request`: the prefill side as built, the
/// decode side once `prefill_kv_params` (what the prefill returned) is
/// relayed onto it.
fn sent_pair(
    plan: &SequentialPlan,
    request: ProtoGenerateRequest,
    workers: &WorkerSelection,
    prefill_kv_params: Option<&str>,
) -> (String, String) {
    let (prefill, mut decode) = plan.requests(request, workers);
    relay_to_decode(
        plan,
        workers,
        &mut decode,
        prefill_kv_params.map(str::to_owned),
        None,
    )
    .expect("a handoff the plan can send");
    (wire(&prefill), wire(&decode))
}

/// The prefill side of a NIXL handoff: one token, no stream, tagged for the
/// remote decode; its decode side carries what the prefill returned.
const NIXL_PREFILL: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22130d3333333f156666663f400170017801c0010742337b22646f5f72656d6f74655f6465636f6465223a747275652c22646f5f72656d6f74655f70726566696c6c223a66616c73657d";
const NIXL_DECODE: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22110d3333333f156666663f40307801c0010728014285017b22646f5f72656d6f74655f70726566696c6c223a747275652c2272656d6f74655f626c6f636b5f696473223a5b332c342c355d2c2272656d6f74655f656e67696e655f6964223a2270726566696c6c2d656e67696e65222c2272656d6f74655f686f7374223a2231302e302e302e32222c2272656d6f74655f706f7274223a353630307d";
/// A minted Mooncake handoff: the prefill side pinned under the minted id,
/// the decode side's parameters synthesized from the prefill worker.
const MOONCAKE_MINTED_PREFILL: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22130d3333333f156666663f400170017801c0010742517b22646f5f72656d6f74655f6465636f6465223a747275652c22646f5f72656d6f74655f70726566696c6c223a66616c73652c227472616e736665725f6964223a22786665722d7265636f72646564227d";
const MOONCAKE_MINTED_DECODE: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22110d3333333f156666663f40307801c001072801429d017b22646f5f72656d6f74655f6465636f6465223a66616c73652c22646f5f72656d6f74655f70726566696c6c223a747275652c227472616e736665725f6964223a22786665722d7265636f72646564222c2272656d6f74655f656e67696e655f6964223a22656e67696e652d61222c2272656d6f74655f626f6f7473747261705f61646472223a22687474703a2f2f70726566696c6c3a38393938227d";
/// The legacy Mooncake handoff of a worker without an engine id: the typed
/// host and port on the decode side, nothing on the prefill side.
const MOONCAKE_LEGACY_PREFILL: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22130d3333333f156666663f400170017801c00107";
const MOONCAKE_LEGACY_DECODE: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22110d3333333f156666663f40307801c001072801320c0a0770726566696c6c10a646";
/// An n>1 request under NIXL: no handoff on either side, the decode side
/// recomputes the prompt for its samples.
const N2_PREFILL: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22130d3333333f156666663f400170017801c00107";
const N2_DECODE: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22110d3333333f156666663f40307802c001072801";

/// Each connector form's pair goes on the wire byte for byte as the
/// dispatch stage sent it before this module (the constants above were
/// recorded there, from these inputs).
#[test]
fn a_nixl_pair_is_sent_as_recorded() {
    let workers = connector_pair(NIXL_CONNECTOR, None);
    let plan = SequentialPlan::decide(&recorded_request(1), &workers).expect("a NIXL pair");
    assert_eq!(plan.mode, KvConnectorMode::Nixl);
    assert_eq!(
        sent_pair(&plan, recorded_request(1), &workers, Some(NIXL_RETURNED)),
        (NIXL_PREFILL.to_string(), NIXL_DECODE.to_string())
    );
}

#[test]
fn a_minted_mooncake_pair_is_sent_as_recorded() {
    let workers = connector_pair(MOONCAKE_CONNECTOR, Some("engine-a"));
    let mut plan = SequentialPlan::decide(&recorded_request(1), &workers).expect("a Mooncake pair");
    assert!(
        matches!(
            plan.mode,
            KvConnectorMode::Mooncake {
                engine_id: Some(_),
                ..
            }
        ),
        "{:?}",
        plan.mode
    );
    assert!(plan.mooncake_transfer_id.is_some(), "a minted id");
    // The id is minted fresh per request; the record pins the form under one.
    plan.mooncake_transfer_id = Some("xfer-recorded".to_string());
    assert_eq!(
        sent_pair(&plan, recorded_request(1), &workers, None),
        (
            MOONCAKE_MINTED_PREFILL.to_string(),
            MOONCAKE_MINTED_DECODE.to_string()
        )
    );
}

#[test]
fn a_legacy_mooncake_pair_is_sent_as_recorded() {
    let workers = connector_pair(MOONCAKE_CONNECTOR, None);
    let plan = SequentialPlan::decide(&recorded_request(1), &workers).expect("a Mooncake pair");
    assert!(
        matches!(
            plan.mode,
            KvConnectorMode::Mooncake {
                engine_id: None,
                ..
            }
        ),
        "{:?}",
        plan.mode
    );
    assert_eq!(
        plan.mooncake_transfer_id, None,
        "nothing minted without an engine id"
    );
    assert_eq!(
        sent_pair(&plan, recorded_request(1), &workers, None),
        (
            MOONCAKE_LEGACY_PREFILL.to_string(),
            MOONCAKE_LEGACY_DECODE.to_string()
        )
    );
}

#[test]
fn an_n_greater_than_one_pair_is_sent_as_recorded() {
    let workers = connector_pair(NIXL_CONNECTOR, None);
    let plan = SequentialPlan::decide(&recorded_request(2), &workers).expect("a NIXL pair");
    assert!(!plan.relay_kv_params, "n>1 relays no handoff");
    assert_eq!(
        sent_pair(&plan, recorded_request(2), &workers, None),
        (N2_PREFILL.to_string(), N2_DECODE.to_string())
    );
}
