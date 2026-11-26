use serde_json::json;

use super::*;

fn test_config() -> EncoderConfig {
    EncoderConfig {
        max_active: 4,
        max_cpu: 2,
        execution_capacity: 1,
        max_images: 4,
        max_control_bytes: 1024,
        max_queued_control_bytes: 0,
        control_listen: String::new(),
        native_commands: String::new(),
        native_events: String::new(),
        model_path: "model".into(),
        tokenizer_path: "tok".into(),
        served_model_name: "model".into(),
        model_type: "qwen3_vl".into(),
        mm_item_limits: String::new(),
        capabilities: Capabilities {
            role: "encoder".into(),
            model_revision: "rev1".into(),
            processor_contract: "smg-image-grid-v1".into(),
            embedding_dtype: "bfloat16".into(),
            kv_backend: "mooncake".into(),
            embedding_backend: "mooncake".into(),
            topology_contract: "single-host-tp1".into(),
            receive_endpoint: "http://127.0.0.1:9000".into(),
            tp_size: 1,
            ..Default::default()
        },
    }
}

fn encode_request(id: &str) -> EncodeRequest {
    EncodeRequest {
        req_id: id.into(),
        mm_items: vec!["data:image/png;base64,AA==".into()],
        num_parts: 1,
        part_idx: 0,
        prefill_host: "127.0.0.1".into(),
        embedding_port: vec![],
        item_options: vec![],
        modality: String::new(),
    }
}

async fn recv_cmd(pull: &mut PullSocket) -> Value {
    let message = tokio::time::timeout(Duration::from_secs(2), pull.recv())
        .await
        .expect("timed out waiting for encoder IPC command")
        .unwrap();
    serde_json::from_slice(&message.into_vec()[0]).unwrap()
}

struct EncoderHarness {
    _dir: tempfile::TempDir,
    runtime: Arc<EncoderRuntime>,
    pull: Option<PullSocket>,
    push: PushSocket,
}

impl EncoderHarness {
    async fn start_with_config(
        config: EncoderConfig,
        connect_pull: bool,
        cancel_grace: Duration,
        processor: Option<EncoderProcessor>,
    ) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let commands = format!("ipc://{}/commands.sock", dir.path().display());
        let events = format!("ipc://{}/events.sock", dir.path().display());
        let link = NativeEncoderLink::bind(&commands, &events, 32)
            .await
            .unwrap();
        let pull = if connect_pull {
            let mut pull = PullSocket::new();
            pull.connect(&commands).await.unwrap();
            Some(pull)
        } else {
            None
        };
        let mut push = PushSocket::new();
        push.connect(&events).await.unwrap();
        let processor = processor.unwrap_or_else(|| {
            Arc::new(|_, _| {
                Box::pin(async {
                    Ok(EncoderArtifact {
                        bytes: Bytes::from_static(b"pixels"),
                        embedding_tokens: 4,
                    })
                })
            })
        });
        let runtime =
            EncoderRuntime::with_cancel_grace(config, link, processor, cancel_grace).unwrap();
        assert!(!runtime.ready());
        Self {
            _dir: dir,
            runtime,
            pull,
            push,
        }
    }

    async fn start(
        connect_pull: bool,
        cancel_grace: Duration,
        processor: Option<EncoderProcessor>,
    ) -> Self {
        Self::start_with_config(test_config(), connect_pull, cancel_grace, processor).await
    }

    async fn send_reply(&mut self, value: Value) {
        let bytes = serde_json::to_vec(&value).unwrap();
        self.push
            .send(ZmqMessage::from(Bytes::from(bytes)))
            .await
            .unwrap();
    }

    async fn wait_for(&self, mut pred: impl FnMut() -> bool) {
        for _ in 0..50 {
            if pred() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn mark_ready(&mut self) {
        self.send_reply(json!({"ready": true})).await;
        self.wait_for(|| self.runtime.ready()).await;
        assert!(self.runtime.ready());
    }
}

#[tokio::test]
async fn cancel_releases_queued_immediately_and_inflight_only_after_receipt() {
    let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, None).await;
    let early = h
        .runtime
        .encode(encode_request("early-req"), Duration::from_secs(5))
        .await
        .unwrap_err();
    assert_eq!(early.code(), tonic::Code::Unavailable);
    assert!(h.runtime.fatal_error().is_none());
    h.mark_ready().await;

    let h1 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("req-1"), Duration::from_secs(5))
                .await
        }
    });
    let exec1 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(exec1["operation"], "execute");

    // Queued request cancelled before execution releases immediately and notifies Python.
    let h2 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("req-2"), Duration::from_secs(5))
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(h.runtime.loads(), (1, 1));
    h2.abort();
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(h.runtime.loads(), (1, 0));
    let forget2 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(forget2["operation"], "forget");
    assert_eq!(forget2["rid"], "req-2");

    // Cancelling in-flight req-1 sends cancel wakeup but retains slot until released=true.
    h1.abort();
    let cancel1 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(cancel1["operation"], "cancel");
    assert_eq!(cancel1["rid"], "req-1");
    assert_eq!(h.runtime.loads().0, 1);

    // Unconfirmed release drains runtime and retains the owner.
    h.send_reply(json!({"sequence": exec1["sequence"], "result": {"released": false}}))
        .await;
    h.wait_for(|| h.runtime.fatal_error().is_some()).await;
    assert!(!h.runtime.ready());
    assert!(h.runtime.fatal_error().is_some());
    assert_eq!(h.runtime.loads().0, 1);
}

#[tokio::test]
async fn unsent_command_releases_slot_without_quarantine() {
    let mut h = EncoderHarness::start(false, CANCEL_RECEIPT_GRACE, None).await;
    h.mark_ready().await;

    let err = h
        .runtime
        .encode(encode_request("unsent-req"), Duration::from_secs(5))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::Unavailable);
    assert!(
        h.runtime.ready(),
        "unsent command must not quarantine encoder"
    );
    assert!(h.runtime.fatal_error().is_none());
    assert_eq!(h.runtime.loads(), (0, 0));
}

#[tokio::test]
async fn cancel_receipt_timeout_quarantines_and_recovers_on_late_release() {
    let mut h = EncoderHarness::start(true, Duration::from_millis(80), None).await;
    h.mark_ready().await;

    let handle = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("stuck-req"), Duration::from_secs(5))
                .await
        }
    });
    let exec = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(exec["operation"], "execute");

    handle.abort();
    let cancel = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(cancel["operation"], "cancel");

    h.wait_for(|| !h.runtime.ready()).await;
    assert!(!h.runtime.ready());
    assert!(h.runtime.fatal_error().is_none());
    assert_eq!(h.runtime.loads().0, 1);

    let rejected = h
        .runtime
        .encode(encode_request("while-quarantined"), Duration::from_secs(5))
        .await
        .unwrap_err();
    assert_eq!(rejected.code(), tonic::Code::Unavailable);

    // Late release clears quarantine and returns the slot.
    h.send_reply(
        json!({"sequence": exec["sequence"], "result": {"released": true, "error": "cancelled"}}),
    )
    .await;
    h.wait_for(|| h.runtime.ready() && h.runtime.loads().0 == 0)
        .await;
    assert!(h.runtime.ready());
    assert!(h.runtime.fatal_error().is_none());
    assert_eq!(h.runtime.loads(), (0, 0));
}

#[tokio::test]
async fn unconfirmed_release_receipt_marks_fatal_and_retains_slot() {
    let mut h = EncoderHarness::start(true, Duration::from_millis(80), None).await;
    h.mark_ready().await;

    let handle = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("unconfirmed-req"), Duration::from_secs(5))
                .await
        }
    });
    let exec = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(exec["operation"], "execute");

    handle.abort();
    let cancel = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(cancel["operation"], "cancel");

    h.wait_for(|| !h.runtime.ready()).await;
    assert!(!h.runtime.ready());
    assert!(h.runtime.fatal_error().is_none());

    h.send_reply(
        json!({"sequence": exec["sequence"], "result": {"released": false, "error": "busy"}}),
    )
    .await;
    h.wait_for(|| h.runtime.fatal_error().is_some()).await;
    assert!(!h.runtime.ready());
    assert!(h.runtime.fatal_error().is_some());
    assert_eq!(h.runtime.loads().0, 1);
}

#[tokio::test]
async fn forget_unowned_protects_live_owner_and_notifies_while_draining() {
    let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, None).await;
    h.mark_ready().await;

    let _live = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("req-live"), Duration::from_secs(5))
                .await
        }
    });
    let pull = h.pull.as_mut().unwrap();
    let exec = recv_cmd(pull).await;
    assert_eq!(exec["operation"], "execute");
    assert_eq!(exec["rid"], "req-live");

    let mut malformed = encode_request("req-live");
    malformed.mm_items.clear();
    let err = h
        .runtime
        .encode(malformed, Duration::from_secs(5))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    assert!(
        tokio::time::timeout(Duration::from_millis(80), pull.recv())
            .await
            .is_err(),
        "malformed duplicate rid must not emit forget while owner is live"
    );

    h.runtime.drain();
    let err = h
        .runtime
        .encode(encode_request("req-draining"), Duration::from_secs(5))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::Unavailable);
    let mut saw_forget = false;
    for _ in 0..2 {
        let cmd = recv_cmd(pull).await;
        if cmd["operation"] == "forget" && cmd["rid"] == "req-draining" {
            saw_forget = true;
            break;
        }
    }
    assert!(saw_forget, "expected forget notification while draining");
}

#[tokio::test]
async fn item_options_forward_to_processor_and_reject_mismatches() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_proc = seen.clone();
    let processor: EncoderProcessor = Arc::new(move |modality, items| {
        seen_proc.lock().unwrap().push((modality, items));
        Box::pin(async {
            Ok(EncoderArtifact {
                bytes: Bytes::from_static(b"pixels"),
                embedding_tokens: 4,
            })
        })
    });
    let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, Some(processor)).await;
    h.mark_ready().await;

    for (rid, mutate) in [
        (
            "bad-len",
            Box::new(|r: &mut EncodeRequest| {
                r.item_options = vec![ep::ImageOption::default(); 2];
            }) as Box<dyn FnOnce(&mut EncodeRequest)>,
        ),
        (
            "bad-detail",
            Box::new(|r: &mut EncodeRequest| {
                r.item_options.push(ep::ImageOption {
                    detail: "bogus".into(),
                });
            }),
        ),
        (
            "bad-modality",
            Box::new(|r: &mut EncodeRequest| r.modality = "audio".into()),
        ),
        (
            "video-with-opts",
            Box::new(|r: &mut EncodeRequest| {
                r.modality = "video".into();
                r.item_options.push(ep::ImageOption::default());
            }),
        ),
    ] {
        let mut req = encode_request(rid);
        mutate(&mut req);
        assert_eq!(
            h.runtime
                .encode(req, Duration::from_secs(5))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument,
            "{rid}"
        );
        let forget = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(forget["operation"], "forget");
        assert_eq!(forget["rid"], rid);
    }
    assert!(seen.lock().unwrap().is_empty());

    let req = EncodeRequest {
        req_id: "opts-ok".into(),
        mm_items: vec![
            "https://example.test/a.png".into(),
            "https://example.test/b.png".into(),
        ],
        num_parts: 2,
        part_idx: 0,
        prefill_host: "127.0.0.1".into(),
        embedding_port: vec![],
        item_options: vec![
            ep::ImageOption {
                detail: "high".into(),
            },
            ep::ImageOption {
                detail: "auto".into(),
            },
        ],
        modality: "image".into(),
    };
    let task = tokio::spawn({
        let rt = h.runtime.clone();
        async move { rt.encode(req, Duration::from_secs(5)).await }
    });
    let exec = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(
        (
            &exec["operation"],
            &exec["modality"],
            &exec["num_parts"],
            &exec["part_idx"]
        ),
        (&json!("execute"), &json!("image"), &json!(2), &json!(0))
    );
    h.send_reply(json!({"sequence": exec["sequence"], "result": {"released": true}}))
        .await;
    task.await.unwrap().unwrap();

    let video_req = EncodeRequest {
        req_id: "video-ok".into(),
        mm_items: vec!["https://example.test/clip.mp4".into()],
        num_parts: 2,
        part_idx: 1,
        prefill_host: "127.0.0.1".into(),
        embedding_port: vec![],
        item_options: vec![],
        modality: "video".into(),
    };
    let video_task = tokio::spawn({
        let rt = h.runtime.clone();
        async move { rt.encode(video_req, Duration::from_secs(5)).await }
    });
    let video_exec = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(
        (
            &video_exec["operation"],
            &video_exec["modality"],
            &video_exec["num_parts"],
            &video_exec["part_idx"],
            &video_exec["item_count"]
        ),
        (
            &json!("execute"),
            &json!("video"),
            &json!(2),
            &json!(1),
            &json!(1)
        )
    );
    h.send_reply(json!({"sequence": video_exec["sequence"], "result": {"released": true}}))
        .await;
    video_task.await.unwrap().unwrap();

    assert_eq!(
        seen.lock().unwrap().as_slice(),
        &[
            (
                EncoderModality::Image,
                vec![
                    EncoderMediaItem {
                        url: "https://example.test/a.png".into(),
                        detail: Some("high".into()),
                    },
                    EncoderMediaItem {
                        url: "https://example.test/b.png".into(),
                        detail: Some("auto".into()),
                    },
                ],
            ),
            (
                EncoderModality::Video,
                vec![EncoderMediaItem {
                    url: "https://example.test/clip.mp4".into(),
                    detail: None,
                }],
            ),
        ]
    );
}

#[tokio::test]
async fn encoder_enforces_shared_queued_control_byte_budget() {
    let mut config = test_config();
    config.max_active = 8;
    config.execution_capacity = 1;
    config.max_control_bytes = 2048;
    let large_req = |id: &str| EncodeRequest {
        req_id: id.into(),
        mm_items: vec![format!("data:image/png;base64,{}", "A".repeat(400))],
        num_parts: 1,
        part_idx: 0,
        prefill_host: "127.0.0.1".into(),
        embedding_port: vec![],
        item_options: vec![],
        modality: "image".into(),
    };
    let req_len = large_req("r1").encoded_len();
    // Allow exactly 1 large request queued while another holds the single execution slot.
    config.max_queued_control_bytes = req_len + 16;

    let mut h = EncoderHarness::start_with_config(config, true, Duration::from_secs(1), None).await;
    h.mark_ready().await;
    // r1 acquires the execution permit and releases its queued-control reservation.
    let r1 = {
        let rt = h.runtime.clone();
        tokio::spawn(async move { rt.encode(large_req("r1"), Duration::from_secs(5)).await })
    };
    let r1_cmd = recv_cmd(h.pull.as_mut().unwrap()).await;

    // r2 queues behind r1, consuming almost all of max_queued_control_bytes.
    let r2 = {
        let rt = h.runtime.clone();
        tokio::spawn(async move { rt.encode(large_req("r2"), Duration::from_secs(5)).await })
    };
    tokio::time::sleep(Duration::from_millis(30)).await;

    // r3 exceeds the shared queued-control byte budget even though max_active=8.
    let err = h
        .runtime
        .encode(large_req("r3"), Duration::from_secs(5))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::ResourceExhausted);
    assert!(err.message().contains("queued control"));

    // Releasing r1 promotes r2 into execution (freeing r2's queued reservation), so r3 now admits.
    h.send_reply(json!({"sequence": r1_cmd["sequence"], "result": {"released": true}}))
        .await;
    r1.await.unwrap().unwrap();
    let mut r2_cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
    while r2_cmd["operation"] == "forget" {
        r2_cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
    }
    assert_eq!(r2_cmd["rid"], "r2");

    let r3 = {
        let rt = h.runtime.clone();
        tokio::spawn(async move { rt.encode(large_req("r3"), Duration::from_secs(5)).await })
    };
    h.send_reply(json!({"sequence": r2_cmd["sequence"], "result": {"released": true}}))
        .await;
    r2.await.unwrap().unwrap();
    let r3_cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
    h.send_reply(json!({"sequence": r3_cmd["sequence"], "result": {"released": true}}))
        .await;
    r3.await.unwrap().unwrap();
}

#[tokio::test]
async fn late_not_sent_after_cancel_timeout_clears_quarantine_without_draining() {
    // Without a connected pull socket, commands.send retries ReturnToSender for ~95ms
    // before emitting NativeCallError::NotSent. Using a shorter cancel_grace forces
    // the quarantine path first, then resolves late with NotSent.
    let mut h = EncoderHarness::start(false, Duration::from_millis(20), None).await;
    h.mark_ready().await;

    let handle = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("late-not-sent"), Duration::from_secs(5))
                .await
        }
    });
    h.wait_for(|| h.runtime.loads().0 == 1).await;
    tokio::time::sleep(Duration::from_millis(15)).await;
    handle.abort();

    h.wait_for(|| !h.runtime.ready()).await;
    assert!(h.runtime.fatal_error().is_none());

    // Once NotSent arrives, quarantine clears without setting draining and the slot returns.
    h.wait_for(|| h.runtime.ready() && h.runtime.loads() == (0, 0))
        .await;
    assert!(h.runtime.ready());
    assert!(h.runtime.fatal_error().is_none());
    assert_eq!(h.runtime.loads(), (0, 0));
}
