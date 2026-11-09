//! vLLM's encoder completes before prefill consumes its metadata-only prompt.

use axum::response::Response;
use serde_json::json;
use smg_grpc_client::vllm_proto as vllm;
use uuid::Uuid;

use crate::{
    routers::{
        error,
        grpc::{
            context::WorkerSelection,
            multimodal::worker_supports_ec_transfer,
            proto_wrapper::{ProtoGenerateRequest, ProtoGenerateResponse},
            utils::tonic_ext::TonicStatusExt,
        },
    },
    worker::Worker,
};

/// Encode one prompt, validating every handoff before removing its media.
pub(super) async fn encode(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
    image_anchor: Option<u32>,
) -> Result<Vec<PushCleanup>, Response> {
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
    let Some(assignment) = assignments.first() else {
        return Err(error::internal_error(
            "ec_encoder_assignment",
            "No encoder assignment",
        ));
    };
    if !worker_supports_ec_transfer(prefill.as_ref()) {
        return Err(error::service_unavailable(
            "ec_consumer_unsupported",
            "Prefill worker does not support EC transfer",
        ));
    }
    if let Some(mut groups) =
        image_anchor.and_then(|anchor| images::split(request, assignments, anchor))
    {
        let cleanups =
            futures::future::try_join_all(groups.iter_mut().map(|group| {
                encode_one(&mut group.request, group.worker.as_ref(), prefill.as_ref())
            }))
            .await?;
        if let Some(merged) = images::merge(request, &groups, image_anchor) {
            *request = merged;
            return Ok(cleanups);
        }
        // A processor may update tokens outside the image anchor. Preserve its
        // full-prompt behavior rather than guessing how to splice those edits.
        tracing::debug!("Encoder prompt updates require single-worker encoding");
        drop(cleanups);
    }
    encode_one(request, assignment.worker.as_ref(), prefill.as_ref())
        .await
        .map(|cleanup| vec![cleanup])
}

async fn encode_one(
    request: &mut ProtoGenerateRequest,
    worker: &dyn Worker,
    prefill: &dyn Worker,
) -> Result<PushCleanup, Response> {
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
    encoder.data_parallel_rank = worker.dp_rank().map(|rank| rank as i32);
    encoder.sampling_params = Some(vllm::SamplingParams {
        max_tokens: Some(1),
        n: 1,
        ignore_eos: true,
        ..Default::default()
    });
    let mut params = json!({});
    // Mooncake PUSH needs the exact selected DP replica, not the HTTP endpoint.
    // PULL connectors leave this list empty and return their handles from E.
    if let Some(addr) = push_destination(prefill)? {
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
    let backend = worker
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
        let connector = worker
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

/// A Mooncake consumer must advertise a destination before E reserves buffers.
fn push_destination(prefill: &dyn Worker) -> Result<Option<String>, Response> {
    let labels = &prefill.metadata().spec.labels;
    let required = labels
        .get("ec_connector")
        .is_some_and(|name| name == "ECMooncakeConnector");
    if !required {
        return Ok(None);
    }
    let addrs = labels.get("ec_zmq_addrs").ok_or_else(|| {
        error::service_unavailable(
            "ec_consumer_address",
            "Mooncake consumer has no EC destination",
        )
    })?;
    let addrs: Vec<String> = serde_json::from_str(addrs)
        .map_err(|e| error::internal_error("ec_consumer_address", e.to_string()))?;
    consumer_address(&addrs, prefill.dp_rank())
        .map(|addr| Some(addr.to_owned()))
        .ok_or_else(|| {
            error::service_unavailable(
                "ec_consumer_address",
                "No EC destination for the selected DP rank",
            )
        })
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

mod images {
    //! Image-only fan-out, retaining the original prompt outside image anchors.

    use std::sync::Arc;

    use smg_grpc_client::{common_proto::Modality, vllm_proto as vllm};

    use crate::{
        routers::grpc::{context::EncodeWorkerAssignment, proto_wrapper::ProtoGenerateRequest},
        worker::Worker,
    };

    pub(super) struct ImageGroup {
        pub worker: Arc<dyn Worker>,
        indices: Vec<usize>,
        pub request: ProtoGenerateRequest,
    }

    fn tokens(request: &vllm::GenerateRequest) -> Option<&[u32]> {
        match request.input.as_ref()? {
            vllm::generate_request::Input::Tokenized(input) => Some(&input.input_ids),
            vllm::generate_request::Input::Text(_) => None,
        }
    }

    /// Batch images assigned to the same E; leave single-E and mixed media alone.
    pub(super) fn split(
        request: &ProtoGenerateRequest,
        assignments: &[EncodeWorkerAssignment],
        anchor: u32,
    ) -> Option<Vec<ImageGroup>> {
        let ProtoGenerateRequest::Vllm(original) = request else {
            return None;
        };
        let refs = original.media_refs.as_ref()?;
        let input = tokens(original)?;
        if refs.items.len() != assignments.len()
            || refs
                .items
                .iter()
                .any(|item| item.modality != Modality::Image as i32)
            || input.iter().filter(|&&token| token == anchor).count() != refs.items.len()
        {
            return None;
        }
        let mut groups: Vec<ImageGroup> = Vec::new();
        for (index, assignment) in assignments.iter().enumerate() {
            if assignment.item_index != index {
                return None;
            }
            let group = groups.iter().position(|group| {
                group.worker.url() == assignment.worker.url()
                    && group.worker.dp_rank() == assignment.worker.dp_rank()
            });
            if let Some(group) = group {
                groups[group].indices.push(index);
            } else {
                groups.push(ImageGroup {
                    worker: assignment.worker.clone(),
                    indices: vec![index],
                    request: request.clone(),
                });
            }
        }
        if groups.len() < 2 {
            return None;
        }
        for group in &mut groups {
            let ProtoGenerateRequest::Vllm(child) = &mut group.request else {
                return None;
            };
            let vllm::generate_request::Input::Tokenized(input) = child.input.as_mut()? else {
                return None;
            };
            input.original_text.clear();
            let mut index = 0;
            input.input_ids.retain(|&token| {
                if token != anchor {
                    return true;
                }
                let keep = group.indices.binary_search(&index).is_ok();
                index += 1;
                keep
            });
            let mut index = 0;
            child.media_refs.as_mut()?.items.retain(|_| {
                let keep = group.indices.binary_search(&index).is_ok();
                index += 1;
                keep
            });
        }
        Some(groups)
    }

    /// Merge only updates confined to each image anchor. A processor with other
    /// prompt edits falls back to the existing full-prompt encode path.
    pub(super) fn merge(
        request: &ProtoGenerateRequest,
        groups: &[ImageGroup],
        anchor: Option<u32>,
    ) -> Option<ProtoGenerateRequest> {
        let anchor = anchor?;
        let ProtoGenerateRequest::Vllm(original) = request else {
            return None;
        };
        let original_tokens = tokens(original)?;
        let count = original.media_refs.as_ref()?.items.len();
        let mut images = vec![None; count];
        let mut handles = serde_json::Map::new();
        let mut transfer_ids = vec![None; count];
        for group in groups {
            let ProtoGenerateRequest::Vllm(child) = &group.request else {
                return None;
            };
            let output = tokens(child)?;
            let mm = child.mm_inputs.as_ref()?;
            if !child.extra_mm_inputs.is_empty()
                || mm.pixel_values.is_some()
                || !mm.flat_keys.is_empty()
                || mm.mm_hashes.len() != group.indices.len()
                || mm.mm_placeholders.len() != group.indices.len()
            {
                return None;
            }
            let mut cursor = 0;
            let mut index = 0;
            for &token in original_tokens {
                if token != anchor {
                    if output.get(cursor) != Some(&token) {
                        return None;
                    }
                    cursor += 1;
                    continue;
                }
                if let Ok(local) = group.indices.binary_search(&index) {
                    let range = &mm.mm_placeholders[local];
                    let end = cursor.checked_add(range.length as usize)?;
                    if range.offset as usize != cursor
                        || range.length == 0
                        || images[index].is_some()
                    {
                        return None;
                    }
                    images[index] = Some((mm, local, output.get(cursor..end)?));
                    cursor = end;
                }
                index += 1;
            }
            if cursor != output.len() {
                return None;
            }
            let mut params: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(child.ec_transfer_params_json.as_ref()?).ok()?;
            if let Some(items) = params.remove("ec_items") {
                let items = items.as_array()?;
                if items.len() != group.indices.len() {
                    return None;
                }
                for (&index, item) in group.indices.iter().zip(items) {
                    transfer_ids[index] = Some(item.clone());
                }
            }
            handles.extend(params);
        }
        let images = images.into_iter().collect::<Option<Vec<_>>>()?;
        let first = images.first()?.0;
        if images.iter().any(|&(batch, _, _)| {
            batch.im_token_id != first.im_token_id
                || batch.modality != first.modality
                || batch.batched_keys != first.batched_keys
                || batch.keep_on_cpu_keys != first.keep_on_cpu_keys
                || batch.model_specific_tensors.len() != first.model_specific_tensors.len()
        }) {
            return None;
        }
        let mut mm = first.clone();
        mm.mm_hashes.clear();
        mm.mm_placeholders.clear();
        mm.model_specific_tensors.clear();
        for (key, template) in &first.model_specific_tensors {
            if !first.batched_keys.contains(key) {
                return None;
            }
            let mut data = Vec::new();
            for &(batch, local, _) in &images {
                let tensor = batch.model_specific_tensors.get(key)?;
                let Some(vllm::tensor_data::Payload::Inline(bytes)) = &tensor.payload else {
                    return None;
                };
                let rows = batch.mm_hashes.len();
                if tensor.dtype != template.dtype
                    || tensor.shape.first().copied()? as usize != rows
                    || tensor.shape.get(1..)? != template.shape.get(1..)?
                    || bytes.len() % rows != 0
                {
                    return None;
                }
                let stride = bytes.len() / rows;
                data.extend_from_slice(bytes.get(local * stride..(local + 1) * stride)?);
            }
            let mut tensor = template.clone();
            tensor.shape[0] = count.try_into().ok()?;
            tensor.payload = Some(vllm::tensor_data::Payload::Inline(data.into()));
            mm.model_specific_tensors.insert(key.clone(), tensor);
        }
        let mut expanded = Vec::new();
        let mut index = 0;
        for &token in original_tokens {
            if token != anchor {
                expanded.push(token);
                continue;
            }
            let (batch, local, image_tokens) = images[index];
            mm.mm_hashes.push(batch.mm_hashes[local].clone());
            mm.mm_placeholders.push(vllm::PlaceholderRange {
                offset: expanded.len().try_into().ok()?,
                length: image_tokens.len().try_into().ok()?,
            });
            expanded.extend_from_slice(image_tokens);
            index += 1;
        }
        if transfer_ids.iter().any(Option::is_some) {
            handles.insert(
                "ec_items".into(),
                transfer_ids.into_iter().collect::<Option<Vec<_>>>()?.into(),
            );
        }
        let mut merged = request.clone();
        if !merged.apply_media_identity(&vllm::MediaIdentity {
            prompt_token_ids: expanded,
            mm_inputs: Some(mm),
            extra_mm_inputs: Vec::new(),
        }) {
            return None;
        }
        let ProtoGenerateRequest::Vllm(result) = &mut merged else {
            return None;
        };
        result.ec_transfer_params_json = Some(serde_json::Value::Object(handles).to_string());
        Some(merged)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::worker::BasicWorkerBuilder;

        fn fixture() -> (ProtoGenerateRequest, Vec<EncodeWorkerAssignment>) {
            let request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
                input: Some(vllm::generate_request::Input::Tokenized(
                    vllm::TokenizedInput {
                        input_ids: vec![11, 9, 12, 9, 13, 9, 14],
                        original_text: "original".into(),
                    },
                )),
                media_refs: Some(vllm::MediaRefs {
                    items: ["a", "b", "a"]
                        .into_iter()
                        .map(|url| vllm::MediaRef {
                            modality: Modality::Image as i32,
                            url: url.into(),
                        })
                        .collect(),
                }),
                ..Default::default()
            }));
            let workers: Vec<Arc<dyn Worker>> = ["grpc://e0:9000", "grpc://e1:9000"]
                .into_iter()
                .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
                .collect();
            let assignments = [0, 1, 1]
                .into_iter()
                .enumerate()
                .map(|(item_index, worker)| EncodeWorkerAssignment {
                    item_index,
                    worker: workers[worker].clone(),
                })
                .collect();
            (request, assignments)
        }

        /// Complete each E group with different image lengths and per-occurrence PUSH ids.
        fn complete(groups: &mut [ImageGroup]) {
            for group in groups {
                let ProtoGenerateRequest::Vllm(child) = &group.request else {
                    panic!()
                };
                let refs = &child.media_refs.as_ref().unwrap().items;
                let mut mm = vllm::MultimodalInputs {
                    modality: Modality::Image as i32,
                    batched_keys: vec!["image_grid_thw".into()],
                    ..Default::default()
                };
                let mut expanded = Vec::new();
                let mut local = 0;
                let mut grid = Vec::new();
                let mut handles = serde_json::Map::new();
                for &token in tokens(child).unwrap() {
                    if token != 9 {
                        expanded.push(token);
                        continue;
                    }
                    let hash = &refs[local].url;
                    let length = if hash == "a" { 2 } else { 3 };
                    mm.mm_hashes.push(hash.clone());
                    mm.mm_placeholders.push(vllm::PlaceholderRange {
                        offset: expanded.len() as u32,
                        length,
                    });
                    expanded.extend(std::iter::repeat_n(9, length as usize));
                    grid.extend(
                        [1_i64, length as i64, 4]
                            .into_iter()
                            .flat_map(i64::to_le_bytes),
                    );
                    handles.insert(hash.clone(), serde_json::json!({"size_bytes": 128}));
                    local += 1;
                }
                mm.model_specific_tensors.insert(
                    "image_grid_thw".into(),
                    vllm::TensorData {
                        shape: vec![local as u32, 3],
                        dtype: "int64".into(),
                        payload: Some(vllm::tensor_data::Payload::Inline(grid.into())),
                    },
                );
                let mut params = serde_json::json!({"ec_items": group.indices.iter().map(|i| {
                serde_json::json!({"transfer_id": format!("t{i}")})
            }).collect::<Vec<_>>()});
                super::super::apply_completion(
                    &mut group.request,
                    vllm::GenerateComplete {
                        media_identity: Some(vllm::MediaIdentity {
                            prompt_token_ids: expanded,
                            mm_inputs: Some(mm),
                            extra_mm_inputs: vec![],
                        }),
                        ec_transfer_params_json: Some(
                            serde_json::Value::Object(handles).to_string(),
                        ),
                        ..Default::default()
                    },
                    &mut params,
                    "ECMooncakeConnector",
                )
                .unwrap();
            }
        }

        #[test]
        fn grouped_images_restore_text_order_grids_and_duplicate_transfers() {
            let (request, assignments) = fixture();
            let mut groups = split(&request, &assignments, 9).unwrap();
            assert_eq!(
                groups.iter().map(|g| g.indices.clone()).collect::<Vec<_>>(),
                vec![vec![0], vec![1, 2]]
            );
            complete(&mut groups);
            groups.reverse(); // Encoder completion order must not affect prompt order.
            let ProtoGenerateRequest::Vllm(merged) = merge(&request, &groups, Some(9)).unwrap()
            else {
                panic!()
            };
            assert_eq!(
                tokens(&merged).unwrap(),
                &[11, 9, 9, 12, 9, 9, 9, 13, 9, 9, 14]
            );
            assert!(merged.media_refs.is_none());
            let mm = merged.mm_inputs.unwrap();
            assert_eq!(mm.mm_hashes, ["a", "b", "a"]);
            assert_eq!(
                mm.mm_placeholders
                    .iter()
                    .map(|p| (p.offset, p.length))
                    .collect::<Vec<_>>(),
                [(1, 2), (4, 3), (8, 2)]
            );
            let grid = &mm.model_specific_tensors["image_grid_thw"];
            assert_eq!(grid.shape, [3, 3]);
            let Some(vllm::tensor_data::Payload::Inline(bytes)) = &grid.payload else {
                panic!()
            };
            let values: Vec<_> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .copied()
                .map(i64::from_le_bytes)
                .collect();
            assert_eq!(values, [1, 2, 4, 1, 3, 4, 1, 2, 4]);
            let handles: serde_json::Value =
                serde_json::from_str(merged.ec_transfer_params_json.as_ref().unwrap()).unwrap();
            assert_eq!(
                handles["ec_items"],
                serde_json::json!([
                    {"mm_hash":"a", "transfer_id":"t0"},
                    {"mm_hash":"b", "transfer_id":"t1"},
                    {"mm_hash":"a", "transfer_id":"t2"},
                ])
            );
        }

        #[test]
        fn pull_handles_from_different_encoders_are_merged_without_push_ids() {
            let (request, assignments) = fixture();
            let mut groups = split(&request, &assignments, 9).unwrap();
            complete(&mut groups);
            for group in &mut groups {
                let ProtoGenerateRequest::Vllm(child) = &mut group.request else {
                    panic!()
                };
                let mut handles: serde_json::Value =
                    serde_json::from_str(child.ec_transfer_params_json.as_ref().unwrap()).unwrap();
                handles.as_object_mut().unwrap().remove("ec_items");
                child.ec_transfer_params_json = Some(handles.to_string());
            }
            let ProtoGenerateRequest::Vllm(merged) = merge(&request, &groups, Some(9)).unwrap()
            else {
                panic!()
            };
            let handles: serde_json::Value =
                serde_json::from_str(merged.ec_transfer_params_json.as_ref().unwrap()).unwrap();
            assert!(handles.get("ec_items").is_none());
            assert_eq!(handles["a"]["size_bytes"], 128);
            assert_eq!(handles["b"]["size_bytes"], 128);
        }

        #[test]
        fn incompatible_prompt_edits_leave_the_original_request_intact() {
            let (request, assignments) = fixture();
            let mut groups = split(&request, &assignments, 9).unwrap();
            complete(&mut groups);
            let ProtoGenerateRequest::Vllm(child) = &mut groups[0].request else {
                panic!()
            };
            let Some(vllm::generate_request::Input::Tokenized(input)) = child.input.as_mut() else {
                panic!()
            };
            input.input_ids[0] = 77;
            assert!(merge(&request, &groups, Some(9)).is_none());
            assert!(request.has_vllm_media_refs());
        }

        #[test]
        fn single_encoder_and_mixed_modalities_keep_the_whole_prompt() {
            let (request, mut assignments) = fixture();
            assignments[0].worker = assignments[1].worker.clone();
            assert!(split(&request, &assignments, 9).is_none());
            let (mut request, assignments) = fixture();
            let ProtoGenerateRequest::Vllm(original) = &mut request else {
                panic!()
            };
            original.media_refs.as_mut().unwrap().items[1].modality = Modality::Video as i32;
            assert!(split(&request, &assignments, 9).is_none());
        }
    }
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

    /// Missing PUSH discovery must fail before the encoder is contacted.
    #[test]
    fn mooncake_requires_a_destination() {
        use crate::worker::BasicWorkerBuilder;
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
