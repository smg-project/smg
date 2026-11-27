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
        let channel_cap = config.max_active.saturating_mul(16).clamp(64, 65536);
        let link = NativeEncoderLink::bind(&commands, &events, channel_cap)
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
                Box::pin(async { Ok(EncoderArtifact::legacy(Bytes::from_static(b"pixels"), 4)) })
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
        Box::pin(async { Ok(EncoderArtifact::legacy(Bytes::from_static(b"pixels"), 4)) })
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
async fn compact_artifact_and_pre_decode_cache_hit_skips_processor() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_proc = calls.clone();
    let processor: EncoderProcessor = Arc::new(move |_modality, items| {
        calls_proc.fetch_add(1, Ordering::SeqCst);
        let count = items.len();
        Box::pin(async move {
            let hashes = (0..count).map(|i| format!("{i:064x}")).collect();
            Ok(EncoderArtifact {
                payload: EncoderPayload::Compact {
                    batches: vec![NativeBatchMeta {
                        pixel_dtype: "bfloat16".into(),
                        pixel_shape: vec![4 * count as u32, 1],
                        pixel_offset: 0,
                        pixel_nbytes: 8 * count,
                        grid_thw: vec![[1, 2, 2]; count],
                        placeholder_lengths: vec![4; count],
                        second_per_grid_ts: None,
                        second_per_grid_shape: Vec::new(),
                        item_hashes: hashes,
                    }],
                    shm_name: None,
                    shm_nbytes: 0,
                    shm_guard: None,
                    inline_pixels: Some(Bytes::from(vec![0u8; 8 * count])),
                },
                embedding_tokens: 4 * count as u64,
            })
        })
    });
    let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, Some(processor)).await;
    h.mark_ready().await;

    // 1. Cold request populates PreDecodeCache via cache_resident feedback.
    let t1 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("cold-1"), Duration::from_secs(5))
                .await
        }
    });
    let cmd1 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(cmd1["artifact_v2"]["kind"], "compact");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let resident_key = "11".repeat(32);
    h.send_reply(json!({
        "sequence": cmd1["sequence"],
        "result": {"released": true, "cache_resident": [resident_key]}
    }))
    .await;
    t1.await.unwrap().unwrap();

    // 2. Identical data: image hits PreDecodeCache without invoking processor.
    let t2 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("warm-2"), Duration::from_secs(5))
                .await
        }
    });
    let cmd2 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(cmd2["artifact_v2"]["kind"], "cache_hit");
    assert_eq!(cmd2["artifact_v2"]["item_keys"], json!([resident_key]));
    assert_eq!(cmd2["artifact_v2"]["grid_thw"], json!([[1, 2, 2]]));
    assert_eq!(cmd2["artifact_v2"]["token_counts"], json!([4]));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    h.send_reply(json!({"sequence": cmd2["sequence"], "result": {"released": true}}))
        .await;
    t2.await.unwrap().unwrap();

    // 3. Different detail option must NOT collide with cached source key.
    let mut diff_opt = encode_request("diff-opt-3");
    diff_opt.item_options.push(ep::ImageOption {
        detail: "high".into(),
    });
    let t3 = tokio::spawn({
        let rt = h.runtime.clone();
        async move { rt.encode(diff_opt, Duration::from_secs(5)).await }
    });
    let cmd3 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(cmd3["artifact_v2"]["kind"], "compact");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    h.send_reply(json!({"sequence": cmd3["sequence"], "result": {"released": true}}))
        .await;
    t3.await.unwrap().unwrap();

    // 4. Remote HTTP URLs are never cached in PreDecodeCache.
    let mut http_req = encode_request("http-4");
    http_req.mm_items = vec!["https://example.test/img.png".into()];
    let http_req4 = http_req.clone();
    let t4 = tokio::spawn({
        let rt = h.runtime.clone();
        async move { rt.encode(http_req4, Duration::from_secs(5)).await }
    });
    let cmd4 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(cmd4["artifact_v2"]["kind"], "compact");
    h.send_reply(json!({
        "sequence": cmd4["sequence"],
        "result": {"released": true, "cache_resident": ["22".repeat(32)]}
    }))
    .await;
    t4.await.unwrap().unwrap();

    http_req.req_id = "http-5".into();
    let t5 = tokio::spawn({
        let rt = h.runtime.clone();
        async move { rt.encode(http_req, Duration::from_secs(5)).await }
    });
    let cmd5 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(cmd5["artifact_v2"]["kind"], "compact");
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    h.send_reply(json!({"sequence": cmd5["sequence"], "result": {"released": true}}))
        .await;
    t5.await.unwrap().unwrap();
}

#[tokio::test]
async fn pre_decode_cache_eviction_and_race_miss_fallback_to_full_decode() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_proc = calls.clone();
    let processor: EncoderProcessor = Arc::new(move |_, _| {
        calls_proc.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Ok(EncoderArtifact {
                payload: EncoderPayload::Compact {
                    batches: vec![NativeBatchMeta {
                        pixel_dtype: "bfloat16".into(),
                        pixel_shape: vec![4, 1],
                        pixel_offset: 0,
                        pixel_nbytes: 8,
                        grid_thw: vec![[1, 2, 2]],
                        placeholder_lengths: vec![4],
                        second_per_grid_ts: None,
                        second_per_grid_shape: Vec::new(),
                        item_hashes: vec!["aa".repeat(32)],
                    }],
                    shm_name: None,
                    shm_nbytes: 0,
                    shm_guard: None,
                    inline_pixels: Some(Bytes::from_static(&[0u8; 8])),
                },
                embedding_tokens: 4,
            })
        })
    });
    let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, Some(processor)).await;
    h.mark_ready().await;

    let resident_key = "ab".repeat(32);
    // Populate PreDecodeCache.
    let t1 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("seed"), Duration::from_secs(5))
                .await
        }
    });
    let cmd1 = recv_cmd(h.pull.as_mut().unwrap()).await;
    h.send_reply(json!({
        "sequence": cmd1["sequence"],
        "result": {"released": true, "cache_resident": [resident_key]}
    }))
    .await;
    t1.await.unwrap().unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Simulate LRU eviction race on Python: Python returns cache_miss=true; Rust falls back
    // to full decode under the exact same rid and completes cleanly.
    let t2 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("race-req"), Duration::from_secs(5))
                .await
        }
    });
    let probe = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(probe["artifact_v2"]["kind"], "cache_hit");
    assert_eq!(probe["rid"], "race-req");
    h.send_reply(json!({
        "sequence": probe["sequence"],
        "result": {"released": true, "cache_miss": true, "cache_evicted": [resident_key]}
    }))
    .await;

    let fallback = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(fallback["artifact_v2"]["kind"], "compact");
    assert_eq!(fallback["rid"], "race-req");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    h.send_reply(json!({
        "sequence": fallback["sequence"],
        "result": {"released": true, "cache_resident": [resident_key]}
    }))
    .await;
    t2.await.unwrap().unwrap();

    // Explicit cache_evicted feedback clears PreDecodeCache before the next request.
    let t3 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("evict-trig"), Duration::from_secs(5))
                .await
        }
    });
    let hit3 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(hit3["artifact_v2"]["kind"], "cache_hit");
    h.send_reply(json!({
        "sequence": hit3["sequence"],
        "result": {"released": true, "cache_evicted": [resident_key]}
    }))
    .await;
    t3.await.unwrap().unwrap();

    let t4 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("post-evict"), Duration::from_secs(5))
                .await
        }
    });
    let miss4 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(miss4["artifact_v2"]["kind"], "compact");
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    h.send_reply(json!({"sequence": miss4["sequence"], "result": {"released": true}}))
        .await;
    t4.await.unwrap().unwrap();
}

#[tokio::test]
async fn shm_guard_unlinks_after_release_and_on_unsent_error() {
    let dir = tempfile::tempdir().unwrap();
    let path1 = dir.path().join("smg-epd-test-1");
    std::fs::write(&path1, b"pixel-bytes").unwrap();
    let path1_proc = path1.clone();
    let processor: EncoderProcessor = Arc::new(move |_, _| {
        let p = path1_proc.clone();
        Box::pin(async move {
            Ok(EncoderArtifact {
                payload: EncoderPayload::Compact {
                    batches: vec![NativeBatchMeta {
                        pixel_dtype: "bfloat16".into(),
                        pixel_shape: vec![4, 1],
                        pixel_offset: 0,
                        pixel_nbytes: 8,
                        grid_thw: vec![[1, 2, 2]],
                        placeholder_lengths: vec![4],
                        second_per_grid_ts: None,
                        second_per_grid_shape: Vec::new(),
                        item_hashes: vec!["cc".repeat(32)],
                    }],
                    shm_name: Some("smg-epd-test-1".into()),
                    shm_nbytes: 8,
                    shm_guard: Some(ShmGuard::new(p)),
                    inline_pixels: None,
                },
                embedding_tokens: 4,
            })
        })
    });
    let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, Some(processor)).await;
    h.mark_ready().await;

    let task = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("shm-ok"), Duration::from_secs(5))
                .await
        }
    });
    let cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(cmd["artifact_v2"]["shm_name"], "smg-epd-test-1");
    assert!(path1.exists(), "SHM file must stay alive while in-flight");
    h.send_reply(json!({"sequence": cmd["sequence"], "result": {"released": true}}))
        .await;
    task.await.unwrap().unwrap();
    assert!(
        !path1.exists(),
        "ShmGuard must unlink SHM file after release"
    );
}

#[test]
fn pre_decode_cache_compacts_evicted_order_entries_under_churn() {
    let mut cache = PreDecodeCache::new(4);
    for i in 0u32..50 {
        let mut src = [0u8; 32];
        src[..4].copy_from_slice(&i.to_le_bytes());
        let key = format!("{i:064x}");
        cache.insert(
            src,
            CachedSourceMeta {
                item_key_hex: key.clone(),
                grid_thw: [1, 2, 2],
                token_count: 4,
            },
        );
        cache.evict_item_keys([key.as_str()]);
    }
    assert!(cache.by_source.is_empty());
    assert!(
        cache.order.len() <= cache.max_entries * 2,
        "order queue must stay bounded under LRU churn (len={})",
        cache.order.len()
    );
}

#[tokio::test]
async fn burst_128_queues_behind_execution_capacity_without_resource_exhausted() {
    let mut cfg = test_config();
    cfg.max_active = 256;
    cfg.execution_capacity = 8;
    cfg.max_cpu = 8;
    let mut h = EncoderHarness::start_with_config(cfg, true, CANCEL_RECEIPT_GRACE, None).await;
    h.mark_ready().await;

    let mut tasks = Vec::with_capacity(128);
    for idx in 0..128 {
        let rt = h.runtime.clone();
        tasks.push(tokio::spawn(async move {
            rt.encode(
                encode_request(&format!("burst-{idx}")),
                Duration::from_secs(10),
            )
            .await
        }));
    }

    // All 128 requests must be admitted (running <= 8, waiting >= 120), never RESOURCE_EXHAUSTED.
    h.wait_for(|| {
        let (running, waiting) = h.runtime.loads();
        running + waiting == 128
    })
    .await;
    let (running, waiting) = h.runtime.loads();
    assert_eq!(running + waiting, 128);
    assert!(
        running <= 8,
        "running {running} exceeded execution_capacity=8"
    );
    assert!(
        waiting >= 120,
        "expected queued waiters, got waiting={waiting}"
    );

    // Drain all 128 executes in waves of up to execution_capacity, verifying bounded concurrency.
    let mut completed = 0usize;
    let mut in_flight: Vec<u64> = Vec::new();
    let mut peak_inflight = 0usize;
    while completed < 128 {
        while in_flight.len() < 8 && completed + in_flight.len() < 128 {
            let cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
            assert_eq!(cmd["operation"], "execute");
            in_flight.push(cmd["sequence"].as_u64().unwrap());
            peak_inflight = peak_inflight.max(in_flight.len());
        }
        let (r, _) = h.runtime.loads();
        assert!(r <= 8, "running {r} exceeded execution_capacity=8");
        for seq in in_flight.drain(..) {
            h.send_reply(json!({"sequence": seq, "result": {"released": true}}))
                .await;
            completed += 1;
        }
    }
    assert_eq!(peak_inflight, 8);

    for task in tasks {
        task.await.unwrap().expect("burst request must succeed");
    }
    assert_eq!(h.runtime.loads(), (0, 0));
}

#[tokio::test]
async fn queue_enforces_hard_bound_and_releases_on_cancel_or_deadline() {
    let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, None).await;
    h.mark_ready().await;

    // Fill max_active = 4 (1 running + 3 queued).
    let h1 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("q-1"), Duration::from_secs(5))
                .await
        }
    });
    let exec1 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(exec1["operation"], "execute");

    let h2 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("q-2"), Duration::from_secs(5))
                .await
        }
    });
    let h3 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("q-3-short"), Duration::from_millis(60))
                .await
        }
    });
    let h4 = tokio::spawn({
        let rt = h.runtime.clone();
        async move {
            rt.encode(encode_request("q-4"), Duration::from_secs(5))
                .await
        }
    });
    h.wait_for(|| h.runtime.loads() == (1, 3)).await;
    assert_eq!(h.runtime.loads(), (1, 3));

    // 5th request exceeds hard max_active=4 -> ResourceExhausted.
    let overflow = h
        .runtime
        .encode(encode_request("q-5-full"), Duration::from_secs(5))
        .await
        .unwrap_err();
    assert_eq!(overflow.code(), tonic::Code::ResourceExhausted);
    let forget_full = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(forget_full["operation"], "forget");
    assert_eq!(forget_full["rid"], "q-5-full");

    // Cancel q-2 and let q-3-short expire while queued; both release immediately without running.
    h2.abort();
    let err3 = h3.await.unwrap().unwrap_err();
    assert_eq!(err3.code(), tonic::Code::DeadlineExceeded);
    h.wait_for(|| h.runtime.loads() == (1, 1)).await;
    assert_eq!(h.runtime.loads(), (1, 1));

    let mut forgets = Vec::new();
    for _ in 0..2 {
        let cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(cmd["operation"], "forget");
        forgets.push(cmd["rid"].as_str().unwrap().to_owned());
    }
    forgets.sort();
    assert_eq!(forgets, vec!["q-2", "q-3-short"]);

    // Releasing q-1 lets live q-4 proceed directly (never stolen by cancelled/expired waiters).
    h.send_reply(json!({"sequence": exec1["sequence"], "result": {"released": true}}))
        .await;
    h1.await.unwrap().unwrap();
    let exec4 = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(exec4["operation"], "execute");
    assert_eq!(exec4["rid"], "q-4");
    h.send_reply(json!({"sequence": exec4["sequence"], "result": {"released": true}}))
        .await;
    h4.await.unwrap().unwrap();
    assert_eq!(h.runtime.loads(), (0, 0));
}

#[tokio::test]
async fn admit_accepts_multi_encoder_part_counts_up_to_max_images() {
    let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, None).await;
    h.mark_ready().await;

    let mut req = encode_request("part-3-of-4");
    req.num_parts = 4;
    req.part_idx = 3;
    let task = tokio::spawn({
        let rt = h.runtime.clone();
        async move { rt.encode(req, Duration::from_secs(5)).await }
    });
    let exec = recv_cmd(h.pull.as_mut().unwrap()).await;
    assert_eq!(
        (&exec["num_parts"], &exec["part_idx"]),
        (&json!(4), &json!(3))
    );
    h.send_reply(json!({"sequence": exec["sequence"], "result": {"released": true}}))
        .await;
    task.await.unwrap().unwrap();

    // Exceeding max_images (4 in test_config) or part_idx >= num_parts is still rejected.
    for (rid, num_parts, part_idx) in [("zero", 0, 0), ("oob-idx", 4, 4), ("too-many", 5, 0)] {
        let mut bad = encode_request(rid);
        bad.num_parts = num_parts;
        bad.part_idx = part_idx;
        assert_eq!(
            h.runtime
                .encode(bad, Duration::from_secs(5))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument,
            "{rid}"
        );
    }
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
async fn utf8_multibyte_url_does_not_panic_or_leak_admission_slot() {
    let item = EncoderMediaItem {
        url: "ééé".into(),
        detail: None,
    };
    assert_eq!(compute_source_key(EncoderModality::Image, &item), None);

    let processor: EncoderProcessor =
        Arc::new(|_, _| Box::pin(async { Err(Status::invalid_argument("malformed media url")) }));
    let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, Some(processor)).await;
    h.mark_ready().await;

    let mut req = encode_request("utf8-bad");
    req.mm_items = vec!["ééé".into()];
    let err = h
        .runtime
        .encode(req, Duration::from_secs(5))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    assert_eq!(h.runtime.loads(), (0, 0));
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

#[test]
fn pre_decode_cache_generation_prevents_stale_eviction_after_reinsert() {
    let mut cache = PreDecodeCache::new(2);
    let k_a = [1u8; 32];
    let k_b = [2u8; 32];
    let k_c = [3u8; 32];
    let meta = |hex: &str| CachedSourceMeta {
        item_key_hex: hex.into(),
        grid_thw: [1, 2, 2],
        token_count: 1,
    };
    cache.insert(k_a, meta("aa"));
    cache.evict_item_keys(["aa"]);
    cache.insert(k_b, meta("bb"));
    cache.insert(k_a, meta("aa"));
    // At capacity 2 with order [(a,0-stale), (b,1), (a,2)], inserting c must skip
    // the stale (a,0) entry, evict b, and keep the freshly re-inserted a.
    cache.insert(k_c, meta("cc"));
    assert!(cache.lookup_all(&[Some(k_a)]).is_some());
    assert!(cache.lookup_all(&[Some(k_b)]).is_none());
    assert!(cache.lookup_all(&[Some(k_c)]).is_some());
}
