use super::*;

/// The encoder's copy of the prompt carries its own id, no stream, no KV
/// transfer, one token of sampling, and one minted transfer id per media
/// hash when the consumer has a push destination; a pull connector's copy
/// carries an empty object.
#[test]
fn the_encoders_copy_of_the_prompt_mints_one_transfer_id_per_media_hash() {
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;
    let original = vllm::GenerateRequest {
        request_id: "req-1".into(),
        stream: true,
        kv_transfer_params_json: Some("{}".into()),
        mm_inputs: Some(vllm::MultimodalInputs {
            mm_hashes: vec!["a".into(), "b".into()],
            ..Default::default()
        }),
        ..Default::default()
    };
    let encoder: Arc<dyn Worker> =
        Arc::new(BasicWorkerBuilder::new("grpc://127.0.0.1:9000").build());
    let (request, params) =
        encoder_request(&original, encoder.as_ref(), Some("tcp://127.0.0.1:9001"));
    assert!(request.request_id.starts_with("req-1-encode-"));
    assert!(!request.stream);
    assert!(request.kv_transfer_params.is_none() && request.kv_transfer_params_json.is_none());
    let sampling = request.sampling_params.expect("sampling params");
    assert_eq!(
        (sampling.max_tokens, sampling.n, sampling.ignore_eos),
        (Some(1), 1, true)
    );
    assert_eq!(params["consumer_zmq"], "tcp://127.0.0.1:9001");
    assert_eq!(params["ec_items"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        request.ec_transfer_params_json.as_deref(),
        Some(params.to_string().as_str())
    );
    let (request, params) = encoder_request(&original, encoder.as_ref(), None);
    assert_eq!(params, json!({}));
    assert_eq!(request.ec_transfer_params_json.as_deref(), Some("{}"));
}

/// A duplicate image retains two distinct PUSH reservations.
fn completion(handles: &str) -> vllm::GenerateComplete {
    vllm::GenerateComplete {
        media_identity: Some(vllm::MediaIdentity {
            prompt_token_ids: vec![1, 2, 3],
            mm_inputs: Some(vllm::MultimodalInputs {
                mm_hashes: vec!["h".into(), "h".into()],
                mm_placeholders: vec![
                    vllm::PlaceholderRange {
                        offset: 1,
                        length: 1,
                    },
                    vllm::PlaceholderRange {
                        offset: 2,
                        length: 1,
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        }),
        ec_transfer_params_json: Some(handles.into()),
        ..Default::default()
    }
}

/// A request still carrying media must remain intact on failed validation.
fn request() -> ProtoGenerateRequest {
    ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                input_ids: vec![42],
                ..Default::default()
            },
        )),
        media_refs: Some(vllm::MediaRefs::default()),
        ..Default::default()
    }))
}

/// An external-DP servicer's only endpoint can occupy any global rank slot.
#[test]
fn managed_rank_address() {
    let addrs = vec![String::new(), "tcp://prefill:9002".into()];
    assert_eq!(consumer_address(&addrs, None), Some("tcp://prefill:9002"));
    assert_eq!(
        consumer_address(&addrs, Some(1)),
        Some("tcp://prefill:9002")
    );
    assert_eq!(consumer_address(&addrs, Some(0)), None);
    assert_eq!(consumer_address(&["a".into(), "b".into()], None), None);
}

/// Missing PUSH discovery must fail before the encoder is contacted.
#[test]
fn mooncake_requires_a_destination() {
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;
    for connector in ["ECCPUConnector", "ECMooncakeConnector"] {
        let worker = BasicWorkerBuilder::new("grpc://127.0.0.1:9000")
            .label("ec_connector", connector)
            .build();
        assert_eq!(
            push_destination(&worker).is_err(),
            connector == "ECMooncakeConnector"
        );
    }
    for addrs in ["[]", "invalid", r#"["tcp://unused:9001"]"#] {
        let worker = BasicWorkerBuilder::new("grpc://127.0.0.1:9000")
            .label("ec_connector", "ECCPUConnector")
            .label("ec_zmq_addrs", addrs)
            .build();
        assert!(push_destination(&worker).unwrap().is_none());
    }
    let worker = BasicWorkerBuilder::new("grpc://127.0.0.1:9000")
        .label("ec_connector", "ECMooncakeConnector")
        .label("ec_zmq_addrs", r#"["tcp://127.0.0.1:9001"]"#)
        .build();
    assert_eq!(
        push_destination(&worker).unwrap().as_deref(),
        Some("tcp://127.0.0.1:9001")
    );
}

/// Malformed PULL handles must not strip the original media.
#[test]
fn pull_handles_are_validated_before_rewriting() {
    for handles in [
        "[]",
        "bad",
        "{}",
        r#"{"other":{"peer_host":"e","peer_port":1}}"#,
        r#"{"h":{}}"#,
        r#"{"h":{"peer_host":"e"}}"#,
        r#"{"h":{"peer_host":"e","peer_port":0}}"#,
    ] {
        let mut request = request();
        assert!(apply_completion(
            &mut request,
            completion(handles),
            &mut json!({}),
            "ECCPUConnector"
        )
        .is_err());
        assert!(request.has_vllm_media_refs());
    }
    let mut request = request();
    let handles = r#"{"h":{"peer_host":"host","peer_port":9000,"size_bytes":128}}"#;
    apply_completion(
        &mut request,
        completion(handles),
        &mut json!({}),
        "ECCPUConnector",
    )
    .unwrap();
    assert!(!request.has_vllm_media_refs());
    let ProtoGenerateRequest::Vllm(request) = request else {
        panic!("expected a vLLM request")
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&request.ec_transfer_params_json.unwrap())
            .unwrap(),
        serde_json::from_str::<serde_json::Value>(handles).unwrap()
    );
}

/// PUSH reconstructs per-occurrence hashes without collapsing duplicate IDs.
#[test]
fn push_preserves_duplicate_ids_and_rejects_bad_metadata() {
    let mut params = json!({"consumer_zmq":"tcp://p:9000", "ec_items":[{"transfer_id":"one"},{"transfer_id":"two"}]});
    let mut request = request();
    let mut bad = completion(r#"{"h":{"transfer_id":"two"}}"#);
    bad.media_identity
        .as_mut()
        .unwrap()
        .prompt_token_ids
        .clear();
    assert!(apply_completion(&mut request, bad, &mut params, "ECMooncakeConnector").is_err());
    assert!(request.has_vllm_media_refs());
    apply_completion(
        &mut request,
        completion(r#"{"h":{"transfer_id":"two"}}"#),
        &mut params,
        "ECMooncakeConnector",
    )
    .unwrap();
    let ProtoGenerateRequest::Vllm(request) = request else {
        panic!("expected a vLLM request")
    };
    let handles: serde_json::Value =
        serde_json::from_str(&request.ec_transfer_params_json.unwrap()).unwrap();
    assert_eq!(
        handles["ec_items"],
        json!([{"transfer_id":"one","mm_hash":"h"},{"transfer_id":"two","mm_hash":"h"}])
    );
}

/// The encoder's port refuses the dial under an encode call: the client's
/// answer is that failure in the dial's own words, without the encoder's
/// address.
#[cfg(unix)]
#[tokio::test]
async fn a_refused_dial_to_the_encoder_is_answered_without_its_address() {
    use smg_grpc_client::tokenspeed_proto as ts;
    use smg_worker::test_support::RefusedAddress;

    // An encoder the gateway reached once and that is gone, on a port
    // that stays taken after it: the port refuses from then on.
    let address = RefusedAddress::reserve_shared();
    let listener = address.listen_shared();
    let url = format!("http://127.0.0.1:{}", address.port());
    let channel = smg_grpc_client::connect_channel(&url)
        .await
        .expect("the encoder listens");
    let (connection, _) = listener.accept().await.expect("the gateway's connection");
    drop(connection);
    drop(listener);
    // The channel's status for a failed dial is the same under any call
    // of any client. A first call may still go out on the connection
    // that was closed; the call after it dials.
    let mut client = ts::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel);
    let refused = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        for _ in 0..8 {
            let status = client
                .health_check(tonic::Request::new(ts::HealthCheckRequest {}))
                .await
                .expect_err("the encoder is gone");
            if smg_grpc_client::dial_failure_without_peer(&status).is_some() {
                return Some(status);
            }
        }
        None
    })
    .await
    .expect("every call ends");
    let status = refused.expect("a call dials and is refused");
    assert!(status.message().contains("127.0.0.1"), "{status:?}");

    let response = encode_start_failure(&status);
    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("drain body");
    let error: serde_json::Value = serde_json::from_slice(&body).expect("a JSON body");
    assert_eq!(error["error"]["code"], "ec_encode_failed", "{error}");
    let message = error["error"]["message"].as_str().expect("a message");
    assert!(message.contains("refused"), "{message}");
    assert!(
        !message.contains("127.0.0.1") && !message.contains(&address.port().to_string()),
        "the answer names the encoder's address: {message}"
    );
}

/// Cleanup speaks the real JSON/REQ protocol and releases all TP shards.
#[tokio::test]
async fn dropped_push_releases_every_shard() {
    use zeromq::{Socket, SocketRecv, SocketSend};
    for reject_first in [false, true] {
        let mut first = zeromq::RepSocket::new();
        let addr = first.bind("tcp://127.0.0.1:0").await.unwrap().to_string();
        let mut second = zeromq::RepSocket::new();
        let other = second.bind("tcp://127.0.0.1:0").await.unwrap().to_string();
        let ports: Vec<u16> = [&addr, &other]
            .iter()
            .map(|a| a.rsplit_once(':').unwrap().1.parse().unwrap())
            .collect();
        let serve = async {
            let message = first.recv().await.unwrap();
            let message: serde_json::Value =
                serde_json::from_slice(message.get(0).unwrap()).unwrap();
            assert_eq!(message["op"], "peers");
            first
                .send(
                    json!({"ok":true,"result":{"ports":ports}})
                        .to_string()
                        .into(),
                )
                .await
                .unwrap();
            for (shard, socket) in [&mut first, &mut second].into_iter().enumerate() {
                for id in ["one", "two"] {
                    let message = socket.recv().await.unwrap();
                    let message: serde_json::Value =
                        serde_json::from_slice(message.get(0).unwrap()).unwrap();
                    assert_eq!(
                        message,
                        json!({"op":"cancel","transfer_id":id,"reservation_id":""})
                    );
                    socket
                    .send(
                        json!({"ok": !(reject_first && shard == 0 && id == "one"),"result":{"cancelled":true}})
                            .to_string()
                            .into(),
                    )
                    .await
                    .unwrap();
                }
            }
        };
        let cleanup = PushCleanup::new(
            &json!({"consumer_zmq":addr,"ec_items":[{"transfer_id":"one"},{"transfer_id":"two"}]}),
        );
        drop(cleanup);
        tokio::time::timeout(std::time::Duration::from_secs(5), serve)
            .await
            .unwrap();
    }
}
