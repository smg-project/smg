//! The TokenSpeed scheduler service of the mock gRPC worker: this proto's
//! `Generate` read into the core's input and the core's frames shaped into
//! its chunks and completions, with the worker's `weight_version` stamped on
//! each as the engine stamps them; its info, loads and KV-event subscription.
//! `--capture` records this wire's requests (the capture file's keys are its
//! field names).
//! An encode worker also serves the engine's encoder service beside it
//! ([`TokenSpeedEncoder`]): its `Encode` takes a request's one media item
//! with its rendezvous room, and its `Generate` is refused as the wrong pool.
//! A prefill worker checks the rooms a request's `encode_bootstrap_info`
//! hands it against the request's media items.
//! The admin fan-out's flush and profile RPCs are answered by the core, the
//! same on every wire.

use std::{collections::BTreeMap, pin::Pin};

use enc::token_speed_encoder_server::{TokenSpeedEncoder, TokenSpeedEncoderServer};
use engine_zmq_adapter::{tensor_dtype_width, wire_tensor_tokenspeed};
use futures::{Stream, StreamExt};
use smg_grpc_client::{
    common_proto as common, tokenspeed_encoder_proto as enc,
    tokenspeed_scheduler::tokenspeed_proto as ts,
};
use tonic::{Request, Response, Status};
use ts::{
    generate_response::Response as GenResp,
    token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
};

use super::{Frame, GenerateInput, LogProbs, MockScheduler, Rendezvous, TokenizerStream};
use crate::{config::Role, drain::InFlight, engine::KvEventStream, script::MatchedStop};

pub(super) type Server = TokenSpeedSchedulerServer<MockScheduler>;
/// The encoder service an encode worker mounts beside [`Server`].
pub(super) type EncoderServer = TokenSpeedEncoderServer<MockScheduler>;
type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;

#[tonic::async_trait]
impl TokenSpeedScheduler for MockScheduler {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = KvEventStream;
    type GetTokenizerStream = TokenizerStream;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        // An encode worker on this wire takes its requests on the encoder
        // service: a `Generate` here went to the wrong pool.
        if self.cfg.role == Some(Role::Encode) {
            return Err(Status::failed_precondition(
                "mock-worker: an encode worker on this wire takes Encode, not Generate",
            ));
        }
        let req = request.into_inner();
        // Record before anything else: the line is in the file before the
        // stream is returned, so before the gateway can read any frame.
        if let Some(capture) = &self.capture {
            if let Err(e) = capture.record(&req) {
                tracing::error!("capturing request {} failed: {e}", req.request_id);
                return Err(Status::internal(format!("mock-worker capture failed: {e}")));
            }
        }
        check_encode_bootstrap(&req)?;
        check_mm_inputs(&req, self.cfg.role)?;
        let tokenized = req.tokenized.unwrap_or_default();
        let params = req.sampling_params.unwrap_or_default();
        let rendezvous = req
            .kv_bootstrap_info
            .as_ref()
            .map_or(Rendezvous::None, |info| Rendezvous::Named {
                host: info.bootstrap_host.clone(),
                port: info.bootstrap_port,
                room: info.bootstrap_room,
            });
        let frames = self
            .generate_frames(GenerateInput {
                request_id: req.request_id.clone(),
                prompt_ids: tokenized.input_ids,
                rendered: tokenized.original_text,
                choices: usize::try_from(params.n).unwrap_or(usize::MAX),
                max_new: params.max_new_tokens,
                stream: req.stream,
                logprobs: req
                    .return_logprob
                    .then(|| usize::try_from(req.top_logprobs_num).unwrap_or(0)),
                // This proto carries no scores row for requested token ids.
                score_ids: Vec::new(),
                pin: req.data_parallel_rank,
                rendezvous,
            })
            .await?;
        let request_id = req.request_id;
        let weight_version = self.cfg.weight_version.clone();
        Ok(Response::new(Box::pin(frames.map(move |item| {
            item.map(|frame| response(&request_id, weight_version.clone(), frame))
        }))))
    }

    async fn health_check(
        &self,
        _request: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        let (healthy, message) = self.health();
        Ok(Response::new(ts::HealthCheckResponse {
            healthy,
            message: message.to_string(),
        }))
    }

    async fn abort(
        &self,
        request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        self.abort_everywhere(&request.into_inner().request_id);
        Ok(Response::new(ts::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        self.overrides.model_info().refusal("model info")?;
        Ok(Response::new(ts::GetModelInfoResponse {
            model_path: self.cfg.model_id.clone(),
            tokenizer_path: self.cfg.tokenizer_path.clone(),
            served_model_name: self.cfg.model_id.clone(),
            model_type: "mock".to_string(),
            architectures: vec!["MockForCausalLM".to_string()],
            max_context_length: self.cfg.context_length.min(i32::MAX as u32) as i32,
            max_req_input_len: self.cfg.context_length.min(i32::MAX as u32) as i32,
            vocab_size: 32000,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            bos_token_id: 1,
            weight_version: "mock".to_string(),
            default_sampling_params_json: self.cfg.default_sampling_params_json.clone(),
            supports_vision: self.overrides.info_bool("supports_vision").unwrap_or(false),
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        self.overrides.server_info().refusal("server info")?;
        Ok(Response::new(ts::GetServerInfoResponse {
            server_args: self.server_args(),
            scheduler_info: None,
            active_requests: 0,
            is_paused: false,
            uptime_seconds: 0.0,
            max_total_num_tokens: 1_000_000,
            tokenspeed_version: "mock".to_string(),
            start_time: None,
            constrained_decoding: String::new(),
        }))
    }

    async fn get_loads(
        &self,
        request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        let loads: Vec<ts::SchedulerLoad> = self
            .load_rows(request.into_inner().dp_rank)
            .into_iter()
            .map(|(dp_rank, s)| ts::SchedulerLoad {
                active_token_usage: None,
                dp_rank,
                num_running_reqs: s.num_running_reqs,
                num_waiting_reqs: s.num_waiting_reqs,
                num_waiting_uncached_tokens: s.num_waiting_uncached_tokens,
                num_total_reqs: s.num_running_reqs + s.num_waiting_reqs,
                num_used_tokens: s.num_used_tokens,
                max_total_num_tokens: s.max_total_num_tokens,
                max_running_requests: s.max_running_requests,
                token_usage: s.token_usage,
                gen_throughput: s.gen_throughput,
                prefill_throughput: s.prefill_throughput,
                cache_hit_rate: s.cache_hit_rate,
                utilization: s.token_usage,
                memory: None,
                queues: None,
                num_resident_tokens: None,
            })
            .collect();
        Ok(Response::new(ts::GetLoadsResponse {
            timestamp: String::new(),
            version: "mock".to_string(),
            dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
            loads,
            aggregate: None,
        }))
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Ok(Response::new(self.kv_events(request.into_inner())?))
    }

    async fn flush_cache(
        &self,
        _request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Ok(Response::new(self.flush_cache_response()))
    }

    async fn start_profile(
        &self,
        request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Ok(Response::new(
            self.start_profile_response(request.into_inner()),
        ))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Ok(Response::new(self.stop_profile_response()))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        // The mock has no tokenizer artifacts to serve; the gateway's
        // remote-tokenizer fallback treats Unimplemented as "not supported".
        Err(Status::unimplemented("mock-worker"))
    }
}

/// The rendezvous a prefill side receives for an encode-prefill-decode
/// request: one entry per media item, each naming the item, the encode
/// worker's bootstrap server and the room the embedding travels under. The
/// engine's prefill waits on that room and fails without it, so a request
/// whose entries do not cover each of its items once is refused; a request
/// without entries has no encoder (a prefill alone processes its own media).
/// An info with entries and no media is refused by the first arm too: with
/// no item, its first entry names an item outside the covered range.
fn check_encode_bootstrap(req: &ts::GenerateRequest) -> Result<(), Status> {
    let Some(info) = &req.encode_bootstrap_info else {
        return Ok(());
    };
    let items = req.mm_inputs.as_ref().map_or(0, |mm| mm.items.len());
    let mut covered = vec![false; items];
    for entry in &info.items {
        let slot = usize::try_from(entry.item_index)
            .ok()
            .and_then(|index| covered.get_mut(index))
            .ok_or_else(|| {
                Status::invalid_argument(format!(
                    "mock-worker: encode bootstrap info names item {} of {items}",
                    entry.item_index
                ))
            })?;
        if std::mem::replace(slot, true) {
            return Err(Status::invalid_argument(format!(
                "mock-worker: encode bootstrap info names item {} twice",
                entry.item_index
            )));
        }
        if entry.bootstrap_room == 0 || entry.bootstrap_host.is_empty() || entry.bootstrap_port <= 0
        {
            return Err(Status::invalid_argument(format!(
                "mock-worker: encode bootstrap info of item {} names no rendezvous \
                 (host {:?}, port {}, room {})",
                entry.item_index, entry.bootstrap_host, entry.bootstrap_port, entry.bootstrap_room
            )));
        }
    }
    if let Some(missing) = covered.iter().position(|seen| !seen) {
        return Err(Status::invalid_argument(format!(
            "mock-worker: encode bootstrap info covers {} of {items} media items \
             (item {missing} has no room)",
            info.items.len()
        )));
    }
    Ok(())
}

/// A request's media items against the shape the engine requires before it
/// places an embedding; the gateway builds them per wire, so a wrong item is
/// a gateway regression the canned answer would otherwise hide. The list
/// holds at least one item (the adapter refuses the empty container). Each
/// item: a specified modality, a content hash (the engine's pad value derives
/// from it), its tensors read as the wire reads them, placeholder spans
/// inside the prompt, and, when the item names its placeholder token id, that
/// the prompt's tokens of that id are exactly the items' spans. The encoder
/// input is the prefill and regular roles' to require, inline or through a
/// room of the request's `encode_bootstrap_info` (an encode worker ran the
/// tower); the decode role takes the gateway's pixel-free clone of the
/// request, every item's encoder input taken out and the rooms dropped, the
/// metadata kept so decode pads the placeholders as prefill did.
fn check_mm_inputs(req: &ts::GenerateRequest, role: Option<Role>) -> Result<(), Status> {
    let Some(mm) = &req.mm_inputs else {
        return Ok(());
    };
    if mm.items.is_empty() {
        return Err(Status::invalid_argument(
            "mock-worker: mm_inputs carries no items",
        ));
    }
    let prompt_ids = req
        .tokenized
        .as_ref()
        .map_or(&[][..], |tokenized| tokenized.input_ids.as_slice());
    let roomed: Vec<u32> = req
        .encode_bootstrap_info
        .as_ref()
        .map(|info| info.items.iter().map(|entry| entry.item_index).collect())
        .unwrap_or_default();
    let encoder_input = if role == Some(Role::Decode) {
        EncoderInput::Detached
    } else {
        EncoderInput::Required { roomed: &roomed }
    };
    check_media_items(
        &mm.items,
        Some(prompt_ids),
        encoder_input,
        TensorRole::SchedulerFeature,
    )
}

/// Whether an item carries its feature, read as the engine reads a tensor (an
/// inline payload whose bytes match the dtype and shape, an SHM handle with
/// the servicer's checks, a remote handle as it came): a malformed one is
/// refused on every role, as the engine refuses it. `feature` is what the
/// RPC's servicer makes of the feature.
fn check_encoder_input(
    item: &ts::MultimodalItem,
    what: &str,
    feature: TensorRole,
) -> Result<bool, Status> {
    let Some(tensor) = &item.encoder_input else {
        return Ok(false);
    };
    read_tensor(tensor, feature, what, "encoder_input")?;
    Ok(true)
}

/// What a tensor is to the engine servicer, which decides its checks. The
/// scheduler reads a request's feature through `_feature_from_proto`
/// (tokenspeed/servicer.py, a dtype rule of its own) and pulls a remote one
/// through `feature_from_remote`; the encoder servicer reads its feature
/// through the generic `_tensor_from_proto` (tokenspeed/encoder_servicer.py)
/// and pulls a remote one the same way; a side tensor goes through the
/// generic reader on both servicers, whose `tensor_payload_bytes` (mm_shm.py)
/// refuses a remote payload.
#[derive(Clone, Copy)]
enum TensorRole {
    /// The feature of a `Generate` request's item.
    SchedulerFeature,
    /// The feature of an `Encode` request's item.
    EncoderFeature,
    /// A side tensor of an item (`model_specific_tensors`), on either RPC.
    Side,
}

/// An item's tensor read as the engine reads it: an inline payload through
/// the wire's own reader, so the mock refuses the tensors the engine's adapter
/// refuses; an SHM handle with the servicer's checks (the gateway hands the
/// gRPC wire SHM handles under its shm transport); a remote handle on a
/// feature as it came, since both servicers pull it over their remote
/// transport (`feature_from_remote` in tokenspeed/servicer.py and in
/// tokenspeed/encoder_servicer.py), which the mock cannot reach; a remote
/// handle on a side tensor refused, as the generic reader's
/// `tensor_payload_bytes` (mm_shm.py) refuses it on both. `name` is the field
/// in the refusal.
fn read_tensor(
    tensor: &ts::TensorData,
    role: TensorRole,
    what: &str,
    name: &str,
) -> Result<(), Status> {
    let read = match &tensor.payload {
        Some(ts::tensor_data::Payload::Shm(handle)) => read_shm_handle(tensor, role, handle),
        Some(ts::tensor_data::Payload::Remote(_)) => match role {
            TensorRole::SchedulerFeature | TensorRole::EncoderFeature => Ok(()),
            TensorRole::Side => Err("TensorData.remote payload is not read for a side tensor \
                                     (the servicer pulls a remote handle for the feature alone)"
                .to_string()),
        },
        _ => wire_tensor_tokenspeed(tensor.clone()).map(|_| ()),
    };
    read.map_err(|error| Status::invalid_argument(format!("mock-worker: {what}'s {name}: {error}")))
}

/// The dtypes the scheduler's feature path takes for an SHM handle
/// (`_torch_dtype_from_proto` in tokenspeed/servicer.py); any other is
/// refused there before the handle is read.
const SHM_FEATURE_DTYPES: [&str; 3] = ["bfloat16", "float16", "float32"];

/// The engine servicer's checks on an SHM handle, the segment itself never
/// opened. On the scheduler's feature the dtype is one of
/// [`SHM_FEATURE_DTYPES`]; the encoder's feature and a side tensor keep their
/// wire dtype, one the wire does not know passing as the inline reader passes
/// it. On every role the byte length is the dtype's width over the shape
/// (`_feature_from_proto` and `_tensor_from_proto` in tokenspeed/servicer.py)
/// and the name is a bare segment name once a leading slash is stripped, no
/// slash, dot entry or NUL inside (`validated_shm_name` in mm_shm.py). The
/// offset is free: the servicer reads a handle with one through its generic
/// path.
fn read_shm_handle(
    tensor: &ts::TensorData,
    role: TensorRole,
    handle: &common::ShmHandle,
) -> Result<(), String> {
    if matches!(role, TensorRole::SchedulerFeature)
        && !SHM_FEATURE_DTYPES.contains(&tensor.dtype.as_str())
    {
        return Err(format!(
            "unsupported TensorData dtype for an SHM feature: {:?} (one of {SHM_FEATURE_DTYPES:?})",
            tensor.dtype
        ));
    }
    if let Some(width) = tensor_dtype_width(&tensor.dtype) {
        let expected = tensor
            .shape
            .iter()
            .try_fold(width as u64, |bytes, &dim| {
                bytes.checked_mul(u64::from(dim))
            })
            .ok_or_else(|| "multimodal tensor shape overflows".to_string())?;
        if handle.nbytes != expected {
            return Err(format!(
                "TensorData.shm byte length {} does not match dtype {:?} shape {:?} (expected {expected})",
                handle.nbytes, tensor.dtype, tensor.shape
            ));
        }
    }
    let name = handle.name.trim_start_matches('/');
    if name.is_empty() || name.contains('/') || name == "." || name == ".." || name.contains('\0') {
        return Err(format!(
            "TensorData.shm name {:?} is not a bare segment name",
            handle.name
        ));
    }
    Ok(())
}

/// What a role asks of a media item's encoder input.
#[derive(Clone, Copy)]
enum EncoderInput<'a> {
    /// The role runs the tower on the item: the feature rides inline unless
    /// one of the request's encode rooms (by item index) hands it over.
    Required { roomed: &'a [u32] },
    /// The gateway's decode side is the request's pixel-free clone: no feature
    /// and no rooms, the item's metadata alone.
    Detached,
}

/// The item checks shared by `Generate` (with its prompt and the role's
/// expectation of the encoder input) and `Encode` (no prompt, and the feature
/// required on every item: the encoder runs the tower on it). `feature` is
/// what the RPC's servicer makes of the encoder input: the scheduler's
/// feature for `Generate`, the encoder's for `Encode`.
fn check_media_items(
    items: &[ts::MultimodalItem],
    prompt_ids: Option<&[u32]>,
    encoder_input: EncoderInput<'_>,
    feature: TensorRole,
) -> Result<(), Status> {
    let mut covered: BTreeMap<u32, usize> = BTreeMap::new();
    let mut claimed = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let what = format!("media item {index}");
        if item.modality() == common::Modality::Unspecified {
            return Err(Status::invalid_argument(format!(
                "mock-worker: {what} carries an unspecified modality"
            )));
        }
        if item.content_hash.is_empty() {
            return Err(Status::invalid_argument(format!(
                "mock-worker: {what} carries no content hash (the engine's pad value derives \
                 from it)"
            )));
        }
        let has_feature = check_encoder_input(item, &what, feature)?;
        if let EncoderInput::Required { roomed } = encoder_input {
            let has_room = roomed.contains(&u32::try_from(index).unwrap_or(u32::MAX));
            if !has_feature && !has_room {
                return Err(Status::invalid_argument(format!(
                    "mock-worker: {what} carries no encoder input and no encode room hands it over"
                )));
            }
        }
        for (key, tensor) in &item.model_specific_tensors {
            read_tensor(tensor, TensorRole::Side, &what, key)?;
        }
        let spans = item
            .placeholders
            .iter()
            .map(|placeholder| (placeholder.offset, placeholder.length));
        let length = super::check_placeholder_spans(
            prompt_ids,
            spans,
            item.placeholder_token_id,
            &mut claimed,
            &what,
        )?;
        if let Some(id) = item.placeholder_token_id {
            *covered.entry(id).or_default() += length;
        }
    }
    if let Some(prompt_ids) = prompt_ids {
        for (id, length) in covered {
            super::check_placeholder_tokens(prompt_ids, id, length)?;
        }
    }
    Ok(())
}

/// The engine's encoder service, served by an encode worker beside the
/// scheduler: the gateway hands it a request's media with one rendezvous
/// room per item and reads its acceptance. The embedding would travel to the
/// prefill side out of band; the mock moves nothing and counts the request
/// in the admin API's `encodes`.
#[tonic::async_trait]
impl TokenSpeedEncoder for MockScheduler {
    async fn encode(
        &self,
        request: Request<enc::EncodeRequest>,
    ) -> Result<Response<enc::EncodeResponse>, Status> {
        let req = request.into_inner();
        // The request's own checks come first, as the prefill role checks its
        // input before the hooks: a malformed request spends no armed fault.
        // The engine's encoder takes exactly one media item under one room per
        // request, named by a request id (its adapter refuses the rest), so a
        // gateway that batched two items into one `Encode` or dropped the id
        // would pass a looser mock and fail the worker.
        if req.request_id.is_empty() {
            return Err(Status::invalid_argument(
                "mock-worker: an encode request names its request id, got an empty one",
            ));
        }
        let items = req.mm_inputs.as_ref().map_or(0, |mm| mm.items.len());
        if items != 1 || req.items.len() != 1 {
            return Err(Status::invalid_argument(format!(
                "mock-worker: an encode request carries exactly one media item in one room, \
                 got {items} item(s) in {} room(s)",
                req.items.len()
            )));
        }
        if req.items.iter().any(|item| item.bootstrap_room == 0) {
            return Err(Status::invalid_argument(
                "mock-worker: an encode request's room is never 0",
            ));
        }
        if let Some(mm) = &req.mm_inputs {
            check_media_items(
                &mm.items,
                None,
                EncoderInput::Required { roomed: &[] },
                TensorRole::EncoderFeature,
            )?;
        }
        // The realistic engine's fault and admission hooks apply as to the
        // prefill role, and the request counts in flight until its answer; an
        // armed `finish=error` is the engine refusing the request.
        let _in_flight = InFlight::enter_on(&self.in_flight);
        let finish_error = self.admit_without_tokens(&req.request_id, None).await?;
        self.ops.record_encode();
        let rooms: Vec<i64> = req.items.iter().map(|item| item.bootstrap_room).collect();
        tracing::info!(
            "mock-worker {}: encode request {} of {items} item(s) in rooms {rooms:?} {}",
            self.name,
            req.request_id,
            if finish_error {
                "refused by the armed fault"
            } else {
                "accepted"
            }
        );
        Ok(Response::new(enc::EncodeResponse {
            accepted: !finish_error,
        }))
    }
}

/// A frame in this wire's shape, under the request's id, the worker's
/// `weight_version` on it (`None` leaves the field unset, like an engine
/// that predates it).
fn response(
    request_id: &str,
    weight_version: Option<String>,
    frame: Frame,
) -> ts::GenerateResponse {
    let response = match frame {
        Frame::Chunk(chunk) => GenResp::Chunk(ts::GenerateStreamChunk {
            token_ids: chunk.token_ids,
            prompt_tokens: chunk.prompt_tokens,
            completion_tokens: chunk.completion_tokens,
            cached_tokens: chunk.cached_tokens,
            output_logprobs: chunk.logprobs.map(logprobs),
            index: chunk.index,
            weight_version,
        }),
        Frame::Complete(complete) => GenResp::Complete(ts::GenerateComplete {
            output_ids: complete.output_ids,
            finish_reason: complete.finish_reason,
            prompt_tokens: complete.prompt_tokens,
            completion_tokens: complete.completion_tokens,
            cached_tokens: complete.cached_tokens,
            output_logprobs: complete.logprobs.map(logprobs),
            matched_stop: complete.matched_stop.map(|stop| match stop {
                MatchedStop::Token(id) => ts::generate_complete::MatchedStop::MatchedTokenId(id),
                MatchedStop::Text(text) => ts::generate_complete::MatchedStop::MatchedStopStr(text),
            }),
            index: complete.index,
            weight_version,
            ..Default::default()
        }),
    };
    ts::GenerateResponse {
        request_id: request_id.to_string(),
        response: Some(response),
    }
}

/// The frame's logprobs in this wire's shape, one top entry per position.
fn logprobs(logprobs: LogProbs) -> ts::OutputLogProbs {
    ts::OutputLogProbs {
        token_logprobs: logprobs.token_logprobs,
        token_ids: logprobs.token_ids,
        top_logprobs: logprobs
            .top
            .into_iter()
            .map(|position| ts::TopLogProbs {
                values: position.values,
                token_ids: position.ids,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use tonic::Code;

    use super::*;
    use crate::config::{Config, SchedulerWire};

    fn service(port: u16) -> (MockScheduler, tokio::sync::watch::Sender<bool>) {
        MockScheduler::for_tests(
            Config {
                grpc_wire: SchedulerWire::TokenSpeed,
                ..Config::default()
            },
            port,
        )
    }

    fn request(id: &str, input_ids: &[u32]) -> ts::GenerateRequest {
        ts::GenerateRequest {
            request_id: id.to_string(),
            tokenized: Some(ts::TokenizedInput {
                input_ids: input_ids.to_vec(),
                ..Default::default()
            }),
            sampling_params: Some(ts::SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        }
    }

    /// An image item as the gateway builds it: placeholder token 9 under one
    /// span, an eight-byte content hash, an inline feature.
    fn image_item(offset: u32, length: u32) -> ts::MultimodalItem {
        ts::MultimodalItem {
            modality: common::Modality::Image as i32,
            content_hash: vec![1, 2, 3, 4, 5, 6, 7, 8],
            encoder_input: Some(ts::TensorData {
                shape: vec![1, 4],
                dtype: "float32".to_string(),
                payload: Some(ts::tensor_data::Payload::Inline(vec![0; 16].into())),
            }),
            placeholders: vec![ts::PlaceholderRange { offset, length }],
            placeholder_token_id: Some(9),
            ..Default::default()
        }
    }

    fn with_items(
        mut req: ts::GenerateRequest,
        items: Vec<ts::MultimodalItem>,
    ) -> ts::GenerateRequest {
        req.mm_inputs = Some(ts::MultimodalInputs { items });
        req
    }

    async fn refused(service: &MockScheduler, req: ts::GenerateRequest) -> Status {
        match service.generate(Request::new(req)).await {
            Err(status) => status,
            Ok(_) => panic!("the malformed media item was accepted"),
        }
    }

    /// The prompt `[1, 9, 9, 2]`: one image under tokens 1..=2.
    const PROMPT: [u32; 4] = [1, 9, 9, 2];

    #[tokio::test]
    async fn a_well_formed_media_item_is_accepted() {
        let (service, _stop) = service(65_340);
        let mut stream = service
            .generate(Request::new(with_items(
                request("ts-mm-ok", &PROMPT),
                vec![image_item(1, 2)],
            )))
            .await
            .expect("a well-formed item is accepted")
            .into_inner();
        assert!(stream.next().await.is_some(), "the answer streams");
    }

    #[tokio::test]
    async fn a_media_item_without_a_content_hash_is_refused() {
        let (service, _stop) = service(65_341);
        let mut item = image_item(1, 2);
        item.content_hash.clear();
        let status = refused(
            &service,
            with_items(request("ts-mm-hash", &PROMPT), vec![item]),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("media item 0") && status.message().contains("content hash"),
            "{status:?}"
        );
    }

    #[tokio::test]
    async fn a_media_item_without_placeholders_is_refused() {
        let (service, _stop) = service(65_342);
        let mut item = image_item(1, 2);
        item.placeholders.clear();
        let status = refused(
            &service,
            with_items(request("ts-mm-spans", &PROMPT), vec![item]),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(status.message().contains("no placeholders"), "{status:?}");
        let mut zero = image_item(1, 2);
        zero.placeholders[0].length = 0;
        let status = refused(
            &service,
            with_items(request("ts-mm-zero", &PROMPT), vec![zero]),
        )
        .await;
        assert!(status.message().contains("length 0"), "{status:?}");
    }

    #[tokio::test]
    async fn a_placeholder_outside_the_prompt_is_refused() {
        let (service, _stop) = service(65_343);
        let status = refused(
            &service,
            with_items(request("ts-mm-out", &PROMPT), vec![image_item(3, 2)]),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("outside the prompt of 4 tokens"),
            "{status:?}"
        );
    }

    #[tokio::test]
    async fn a_placeholder_that_does_not_cover_its_token_is_refused() {
        let (service, _stop) = service(65_344);
        // The span sits on token 1, not on the placeholder token 9.
        let status = refused(
            &service,
            with_items(request("ts-mm-cover", &PROMPT), vec![image_item(0, 2)]),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("not its placeholder token 9"),
            "{status:?}"
        );
        // One placeholder token the spans leave out: an item the gateway dropped.
        let status = refused(
            &service,
            with_items(
                request("ts-mm-stray", &[1, 9, 9, 9]),
                vec![image_item(1, 2)],
            ),
        )
        .await;
        assert!(
            status.message().contains("holds 3 placeholder token(s) 9")
                && status.message().contains("cover 2"),
            "{status:?}"
        );
    }

    #[tokio::test]
    async fn overlapping_placeholders_across_items_are_refused() {
        let (service, _stop) = service(65_348);
        let status = refused(
            &service,
            with_items(
                request("ts-mm-overlap", &PROMPT),
                vec![image_item(1, 1), image_item(1, 1)],
            ),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("media item 1")
                && status
                    .message()
                    .contains("overlaps an earlier placeholder at token 1"),
            "{status:?}"
        );
    }

    #[tokio::test]
    async fn a_media_item_with_an_unspecified_modality_is_refused() {
        let (service, _stop) = service(65_345);
        let mut item = image_item(1, 2);
        item.modality = common::Modality::Unspecified as i32;
        let status = refused(
            &service,
            with_items(request("ts-mm-mod", &PROMPT), vec![item]),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("unspecified modality"),
            "{status:?}"
        );
    }

    /// Without an encoder input the item is refused, as the engine refuses a
    /// tower run on nothing; a prefill whose `encode_bootstrap_info` hands the
    /// item a room takes it without one (the encode worker ran the tower).
    #[tokio::test]
    async fn a_media_item_without_an_encoder_input_needs_an_encode_room() {
        let (service, _stop) = service(65_346);
        let mut item = image_item(1, 2);
        item.encoder_input = None;
        let status = refused(
            &service,
            with_items(request("ts-mm-feat", &PROMPT), vec![item.clone()]),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(status.message().contains("no encoder input"), "{status:?}");
        let (prefill, _stop) = MockScheduler::for_tests(
            Config {
                role: Some(Role::Prefill),
                grpc_wire: SchedulerWire::TokenSpeed,
                bootstrap_base_port: Some(8998),
                ..Config::default()
            },
            65_347,
        );
        let mut req = with_items(request("ts-mm-room", &PROMPT), vec![item]);
        req.kv_bootstrap_info = Some(ts::KvBootstrapInfo {
            bootstrap_host: "127.0.0.1".to_string(),
            bootstrap_port: 8998,
            bootstrap_room: 7,
        });
        req.encode_bootstrap_info = Some(ts::EncodeBootstrapInfo {
            items: vec![ts::EncodeItemBootstrapInfo {
                item_index: 0,
                bootstrap_host: "127.0.0.1".to_string(),
                bootstrap_port: 8999,
                bootstrap_room: 11,
            }],
        });
        prefill
            .generate(Request::new(req))
            .await
            .expect("the roomed item is accepted without its feature");
    }

    /// The gateway's decode side is the request's pixel-free clone (every
    /// item's encoder input taken out, the encode rooms dropped, the
    /// metadata kept): a decode worker takes it, and a prefill worker
    /// refuses the same form as the engine refuses a tower run on nothing.
    #[tokio::test]
    async fn a_decode_worker_takes_the_pixel_free_clone_a_prefill_worker_refuses_it() {
        let mut item = image_item(1, 2);
        item.encoder_input = None;
        let clone = |id: &str| {
            let mut req = with_items(request(id, &PROMPT), vec![item.clone()]);
            req.kv_bootstrap_info = Some(ts::KvBootstrapInfo {
                bootstrap_host: "127.0.0.1".to_string(),
                bootstrap_port: 8996,
                bootstrap_room: 7,
            });
            req
        };
        let (decode, _stop) = MockScheduler::for_tests(
            Config {
                role: Some(Role::Decode),
                grpc_wire: SchedulerWire::TokenSpeed,
                ..Config::default()
            },
            65_356,
        );
        let mut stream = decode
            .generate(Request::new(clone("ts-mm-decode")))
            .await
            .expect("the decode side takes the pixel-free clone")
            .into_inner();
        assert!(stream.next().await.is_some(), "the answer streams");
        let (prefill, _stop) = MockScheduler::for_tests(
            Config {
                role: Some(Role::Prefill),
                grpc_wire: SchedulerWire::TokenSpeed,
                bootstrap_base_port: Some(8996),
                ..Config::default()
            },
            65_357,
        );
        let status = refused(&prefill, clone("ts-mm-prefill")).await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(status.message().contains("no encoder input"), "{status:?}");
    }

    /// The adapter refuses the empty container before any translation.
    #[tokio::test]
    async fn an_empty_items_list_is_refused() {
        let (service, _stop) = service(65_366);
        let status = refused(&service, with_items(request("ts-mm-none", &PROMPT), vec![])).await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(status.message().contains("no items"), "{status:?}");
    }

    /// An item's tensors are read as the engine reads them: an inline payload
    /// whose bytes match the dtype and shape, an SHM handle with that byte
    /// length, a bare name and, for the scheduler's feature, one of the
    /// servicer's three dtypes (an offset is free), or a remote handle as it
    /// came, the encoder input and the model-specific tensors alike.
    #[tokio::test]
    async fn a_tensor_is_read_as_the_wire_reads_it() {
        let (service, _stop) = service(65_367);
        let mut short = image_item(1, 2);
        short.encoder_input.as_mut().expect("the feature").payload =
            Some(ts::tensor_data::Payload::Inline(vec![0; 15].into()));
        let status = refused(
            &service,
            with_items(request("ts-mm-bytes", &PROMPT), vec![short]),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("media item 0's encoder_input")
                && status.message().contains("byte length 15"),
            "{status:?}"
        );
        let shm = |name: &str, offset: u64, nbytes: u64, dtype: &str| {
            let mut item = image_item(1, 2);
            let feature = item.encoder_input.as_mut().expect("the feature");
            feature.dtype = dtype.to_string();
            feature.payload = Some(ts::tensor_data::Payload::Shm(common::ShmHandle {
                name: name.to_string(),
                offset,
                nbytes,
                ..Default::default()
            }));
            item
        };
        for handle in [
            shm("smg-mm-1", 0, 16, "float32"),
            shm("smg-mm-1", 8, 16, "float32"),
        ] {
            let mut stream = service
                .generate(Request::new(with_items(
                    request("ts-mm-shm", &PROMPT),
                    vec![handle],
                )))
                .await
                .expect("an SHM handle the servicer would accept is accepted")
                .into_inner();
            assert!(stream.next().await.is_some(), "the answer streams");
        }
        let mut remote = image_item(1, 2);
        remote.encoder_input.as_mut().expect("the feature").payload = Some(
            ts::tensor_data::Payload::Remote(common::RemoteTensorHandle {
                transport: "nixl".to_string(),
                nbytes: 16,
                ..Default::default()
            }),
        );
        let mut stream = service
            .generate(Request::new(with_items(
                request("ts-mm-remote", &PROMPT),
                vec![remote],
            )))
            .await
            .expect("a remote handle passes as the servicer pulls it")
            .into_inner();
        assert!(stream.next().await.is_some(), "the answer streams");
        for (handle, refusal) in [
            (shm("smg-mm-1", 0, 15, "float32"), "byte length 15"),
            (shm("../smg-mm-1", 0, 16, "float32"), "name"),
            (shm("smg-mm-1", 0, 32, "int64"), "dtype"),
        ] {
            let status = refused(
                &service,
                with_items(request("ts-mm-shm", &PROMPT), vec![handle]),
            )
            .await;
            assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
            assert!(
                status.message().contains("media item 0's encoder_input")
                    && status.message().contains(refusal),
                "{status:?}"
            );
        }
        let mut unfilled = image_item(1, 2);
        unfilled
            .encoder_input
            .as_mut()
            .expect("the feature")
            .payload = None;
        let status = refused(
            &service,
            with_items(request("ts-mm-payload", &PROMPT), vec![unfilled]),
        )
        .await;
        assert!(status.message().contains("no payload"), "{status:?}");
        let mut grid = image_item(1, 2);
        grid.model_specific_tensors.insert(
            "image_grid_thw".to_string(),
            ts::TensorData {
                shape: vec![1, 3],
                dtype: "int64".to_string(),
                payload: Some(ts::tensor_data::Payload::Inline(vec![0; 8].into())),
            },
        );
        let status = refused(
            &service,
            with_items(request("ts-mm-grid", &PROMPT), vec![grid]),
        )
        .await;
        assert!(
            status.message().contains("image_grid_thw")
                && status.message().contains("byte length 8"),
            "{status:?}"
        );
        let mut remote_grid = image_item(1, 2);
        remote_grid.model_specific_tensors.insert(
            "image_grid_thw".to_string(),
            ts::TensorData {
                shape: vec![1, 3],
                dtype: "int64".to_string(),
                payload: Some(ts::tensor_data::Payload::Remote(
                    common::RemoteTensorHandle {
                        transport: "nixl".to_string(),
                        nbytes: 24,
                        ..Default::default()
                    },
                )),
            },
        );
        let status = refused(
            &service,
            with_items(request("ts-mm-remote-grid", &PROMPT), vec![remote_grid]),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("image_grid_thw")
                && status.message().contains("remote payload"),
            "{status:?}"
        );
    }
}
