//! vLLM's encoder completes before prefill consumes its metadata-only prompt.

use axum::response::Response;
use serde_json::json;
use smg_grpc_client::vllm_proto as vllm;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        context::WorkerSelection,
        multimodal::worker_supports_ec_transfer,
        proto_wrapper::{ProtoGenerateRequest, ProtoGenerateResponse},
        utils::tonic_ext::TonicStatusExt,
    },
};

/// Encode one prompt, validating every handoff before removing its media.
pub(super) async fn encode(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
) -> Result<PushCleanup, Response> {
    if request.sampling_n() != 1 {
        return Err(error::bad_request(
            "ec_multiple_samples",
            "vLLM EPD currently requires n=1",
        ));
    }
    let WorkerSelection::Disaggregated {
        encode_assignments: Some(assignments),
        prefill,
        ..
    } = workers
    else {
        return Err(error::service_unavailable(
            "ec_encoder_unavailable",
            "No vLLM encoder selected",
        ));
    };
    let [assignment] = assignments.as_slice() else {
        return Err(error::internal_error(
            "ec_encoder_assignment",
            "vLLM requires one encoder per prompt",
        ));
    };
    if !worker_supports_ec_transfer(prefill.as_ref()) {
        return Err(error::service_unavailable(
            "ec_consumer_unsupported",
            "Prefill worker does not support EC transfer",
        ));
    }
    let ProtoGenerateRequest::Vllm(original) = request else {
        return Err(error::internal_error(
            "ec_request_type",
            "Expected a vLLM request",
        ));
    };
    let mut encoder = original.clone();
    encoder.request_id = format!("{}-encode-{}", encoder.request_id, Uuid::now_v7());
    encoder.stream = false;
    encoder.kv_transfer_params = None;
    encoder.kv_transfer_params_json = None;
    encoder.data_parallel_rank = assignment.worker.dp_rank().map(|rank| rank as i32);
    encoder.sampling_params = Some(vllm::SamplingParams {
        max_tokens: Some(1),
        n: 1,
        ignore_eos: true,
        ..Default::default()
    });
    let mut params = json!({});
    // Mooncake PUSH needs the exact selected DP replica, not the HTTP endpoint.
    // PULL connectors leave this list empty and return their handles from E.
    if let Some(addrs) = prefill.metadata().spec.labels.get("ec_zmq_addrs") {
        let addrs: Vec<String> = serde_json::from_str(addrs)
            .map_err(|e| error::internal_error("ec_consumer_address", e.to_string()))?;
        let addr = consumer_address(&addrs, prefill.dp_rank()).ok_or_else(|| {
            error::service_unavailable(
                "ec_consumer_address",
                "No EC destination for the selected DP rank",
            )
        })?;
        params["consumer_zmq"] = json!(addr);
        let count = encoder.media_refs.as_ref().map_or_else(
            || {
                encoder
                    .mm_inputs
                    .iter()
                    .chain(&encoder.extra_mm_inputs)
                    .map(|mm| mm.mm_hashes.len())
                    .sum()
            },
            |refs| refs.items.len(),
        );
        params["ec_items"] = (0..count)
            .map(|_| json!({"transfer_id": Uuid::now_v7().to_string()}))
            .collect();
    }
    encoder.ec_transfer_params_json = Some(params.to_string());
    let cleanup = PushCleanup::new(&params);
    let backend = assignment
        .worker
        .get_backend_client()
        .await
        .map_err(|e| error::service_unavailable("ec_encoder_connection", e.to_string()))?
        .ok_or_else(|| {
            error::service_unavailable("ec_encoder_connection", "Encoder is not connected")
        })?;
    let mut client = backend.as_ref().clone();
    let mut stream = client
        .generate(ProtoGenerateRequest::Vllm(encoder))
        .await
        .map_err(|e| e.to_http_error("ec_encode_failed", e.message().to_string()))?;
    while let Some(reply) = stream.next().await {
        let reply =
            reply.map_err(|e| e.to_http_error("ec_encode_failed", e.message().to_string()))?;
        let ProtoGenerateResponse::Vllm(reply) = reply else {
            return Err(error::internal_error(
                "ec_encode_response",
                "Expected a vLLM encoder reply",
            ));
        };
        let Some(vllm::generate_response::Response::Complete(complete)) = reply.response else {
            continue;
        };
        if complete.finish_reason == "error" {
            return Err(error::bad_gateway("ec_encode_failed", "Encoder failed"));
        }
        let connector = assignment
            .worker
            .metadata()
            .spec
            .labels
            .get("ec_connector")
            .map(String::as_str)
            .unwrap_or("");
        apply_completion(request, complete, &mut params, connector)?;
        stream.mark_completed();
        return Ok(cleanup);
    }
    Err(error::bad_gateway(
        "ec_encode_incomplete",
        "Encoder closed without a completion",
    ))
}

/// Resolve global rank slots, allowing an unranked worker only one destination.
fn consumer_address(addrs: &[String], rank: Option<usize>) -> Option<&str> {
    if let Some(rank) = rank {
        return addrs
            .get(rank)
            .map(String::as_str)
            .filter(|addr| !addr.is_empty());
    }
    let mut managed = addrs.iter().filter(|addr| !addr.is_empty());
    let addr = managed.next()?;
    managed.next().is_none().then_some(addr.as_str())
}

/// Validate the completion atomically before replacing the request's pixels or refs.
fn apply_completion(
    request: &mut ProtoGenerateRequest,
    complete: vllm::GenerateComplete,
    params: &mut serde_json::Value,
    connector: &str,
) -> Result<(), Response> {
    let identity = complete.media_identity.ok_or_else(|| {
        error::bad_gateway("ec_metadata_missing", "Encoder returned no media identity")
    })?;
    let handles = complete.ec_transfer_params_json.ok_or_else(|| {
        error::bad_gateway(
            "ec_handles_missing",
            "Encoder returned no EC transfer handles",
        )
    })?;
    let mut handles: serde_json::Value = serde_json::from_str(&handles)
        .map_err(|e| error::bad_gateway("ec_handles_invalid", e.to_string()))?;
    if !handles.is_object() {
        return Err(error::bad_gateway(
            "ec_handles_invalid",
            "Expected an EC handle object",
        ));
    }
    for hash in identity
        .mm_inputs
        .iter()
        .chain(&identity.extra_mm_inputs)
        .flat_map(|mm| &mm.mm_hashes)
    {
        let handle = handles.get(hash).filter(|handle| {
            handle.is_object() && handle.as_object().is_some_and(|h| !h.is_empty())
        });
        let valid = handle.is_some_and(|handle| {
            connector != "ECCPUConnector"
                || (handle["peer_host"]
                    .as_str()
                    .is_some_and(|host| !host.is_empty())
                    && handle["peer_port"]
                        .as_u64()
                        .is_some_and(|port| (1..=65535).contains(&port)))
        });
        if !valid {
            return Err(error::bad_gateway(
                "ec_handles_invalid",
                format!("Missing or invalid EC handle for {hash}"),
            ));
        }
    }
    if let Some(items) = params
        .get_mut("ec_items")
        .and_then(serde_json::Value::as_array_mut)
    {
        let mut features: Vec<_> = identity
            .mm_inputs
            .as_ref()
            .into_iter()
            .chain(identity.extra_mm_inputs.iter())
            .flat_map(|mm| {
                mm.mm_placeholders
                    .iter()
                    .zip(&mm.mm_hashes)
                    .map(|(range, hash)| (range.offset, hash))
            })
            .collect();
        features.sort_by_key(|(offset, _)| *offset);
        if features.len() != items.len() {
            return Err(error::bad_gateway(
                "ec_item_count",
                "Encoder media count changed",
            ));
        }
        for (item, (_, hash)) in items.iter_mut().zip(features) {
            item["mm_hash"] = json!(hash);
        }
        // A hash-keyed response collapses duplicates; retain every push ID.
        handles["ec_items"] = json!(items);
    }
    if !request.apply_media_identity(&identity) {
        return Err(error::bad_gateway(
            "ec_metadata_invalid",
            "Encoder returned an invalid media identity",
        ));
    }
    if let ProtoGenerateRequest::Vllm(request) = request {
        request.ec_transfer_params_json = Some(handles.to_string());
    }
    Ok(())
}

/// Own unconsumed PUSH ids until a validated handoff can reach prefill.
pub(super) struct PushCleanup(Option<(String, Vec<String>)>);

impl PushCleanup {
    /// Prefill accepted the handoff; its engine now owns the reservations.
    pub(super) fn disarm(&mut self) {
        self.0 = None;
    }

    /// PULL has no destination reservation to release.
    fn new(params: &serde_json::Value) -> Self {
        Self(params["consumer_zmq"].as_str().map(|addr| {
            (
                addr.to_owned(),
                params["ec_items"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|item| item["transfer_id"].as_str().map(str::to_owned))
                    .collect(),
            )
        }))
    }
}

impl Drop for PushCleanup {
    /// Cancellation and malformed completions both release every TP shard.
    #[expect(
        clippy::disallowed_methods,
        reason = "Bounded best-effort EC cleanup must outlive a cancelled request, like abort-on-drop."
    )]
    fn drop(&mut self) {
        let Some((addr, ids)) = self.0.take() else {
            return;
        };
        tokio::spawn(async move {
            match tokio::time::timeout(std::time::Duration::from_secs(2), release_push(&addr, &ids))
                .await
            {
                Ok(Ok(())) => {}
                result => tracing::warn!(?result, "Failed to release unused EC transfers"),
            }
        });
    }
}

/// Use Mooncake's cancellation protocol without abandoning an active remote write.
async fn release_push(addr: &str, ids: &[String]) -> anyhow::Result<()> {
    let peers = control_request(addr, json!({"op": "peers"}), &mut None).await?;
    let ports = peers["ports"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Invalid EC peers"))?;
    anyhow::ensure!(!ports.is_empty(), "Empty EC peers");
    let (host, _) = addr
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("Invalid EC address"))?;
    // Independent shards progress even when one is unreachable. Each shard
    // reuses its REQ connection, replacing it after a failed exchange.
    let failures = futures::future::join_all(ports.iter().map(|port| async move {
        let Some(port) = port.as_u64().filter(|port| (1..=65535).contains(port)) else {
            return vec![format!("Invalid EC peer port: {port}")];
        };
        let shard = format!("{host}:{port}");
        let mut socket = None;
        let mut failures = Vec::new();
        for id in ids {
            let result = tokio::time::timeout(
                std::time::Duration::from_millis(250),
                control_request(
                    &shard,
                    json!({"op": "cancel", "transfer_id": id, "reservation_id": ""}),
                    &mut socket,
                ),
            )
            .await;
            if !matches!(result, Ok(Ok(_))) {
                failures.push(format!("{shard}/{id}: {result:?}"));
                socket = None;
            }
        }
        failures
    }))
    .await
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    anyhow::ensure!(failures.is_empty(), "EC cleanup failed: {failures:?}");
    Ok(())
}

/// Exchange one JSON control frame; the caller bounds the complete cleanup.
async fn control_request(
    addr: &str,
    payload: serde_json::Value,
    connection: &mut Option<zeromq::ReqSocket>,
) -> anyhow::Result<serde_json::Value> {
    use zeromq::{Socket, SocketRecv, SocketSend};
    if connection.is_none() {
        let mut socket = zeromq::ReqSocket::new();
        socket.connect(addr).await?;
        *connection = Some(socket);
    }
    let socket = connection
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("EC connection missing"))?;
    socket.send(payload.to_string().into()).await?;
    let reply = socket.recv().await?;
    let reply: serde_json::Value = serde_json::from_slice(
        reply
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("Empty EC response"))?,
    )?;
    anyhow::ensure!(reply["ok"] == true, "EC control rejected request: {reply}");
    Ok(reply["result"].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
