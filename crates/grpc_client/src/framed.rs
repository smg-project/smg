//! A `GenerateRequest` of the TokenSpeed, vLLM or SGLang wire sent as HTTP/2
//! data frames: the message's small fields in one scratch frame, every
//! inline tensor payload of at least `FRAME_MIN_BYTES` as a frame of its
//! own, shared with the request and never copied. prost's one-buffer
//! encoding copies a large image tensor into a buffer of the whole
//! message's size on every call; above the allocator's oversize threshold
//! that copy is a fresh mapping and its page faults, paid per request.
//!
//! The body is the generated encoding byte for byte. prost writes a
//! message's fields in field-number order (a oneof at the lowest number
//! among its arms), so each message on the path to a tensor is written in
//! three parts: its fields before the path's field, encoded by prost from a
//! copy that holds only those; the path's field key and length, written
//! here with the length prost computes for the sub-message; and its fields
//! after the path's field, encoded by prost from a copy that holds only
//! those. The TokenSpeed path mirrors four field numbers of the scheduler
//! proto: `mm_inputs` 9 of `GenerateRequest`, `items` 1 of
//! `MultimodalInputs`, `encoder_input` 3 of `MultimodalItem`, `inline` 3 of
//! `TensorData` (its payload oneof is the message's last field, so a
//! tensor's other fields all come before it); only the item's primary
//! encoder input rides as a frame. The vLLM path mirrors `mm_inputs` 7 and
//! every element of `extra_mm_inputs` 11 of `GenerateRequest`,
//! `pixel_values` 2 and the value 2 of every `model_specific_tensors` 3
//! entry of `MultimodalInputs` (an entry written as prost writes a map
//! entry: the key unless it is empty, the value unless it is the default,
//! in the map's own order), `inline` 1 of `TensorData` (its payload oneof
//! is the message's first field, so the payload leads the shape and dtype).
//! The SGLang path mirrors `mm_inputs` 3 and every element of
//! `extra_mm_inputs` 21 of its `GenerateRequest`, `pixel_values` 4 and the
//! value of every `model_specific_tensors` 9 entry of its `MultimodalInputs`,
//! and `data` 1 of its `TensorData` (a plain bytes field and the message's
//! first, so the payload leads there too); its raw media lists are owned
//! byte vectors and stay in the scratch.
//! Every message on any path is taken apart by an exhaustive pattern, so
//! a field added to any of them stops this module from compiling until the
//! layout names it; the tests compare the joined frames with
//! `encode_to_vec` over a matrix of requests and decode them back. A
//! request without a large inline payload is one frame of the plain
//! encoding, written once behind a reserved prefix with the length taken
//! from the buffer, as the crate's codec writes it. The request is
//! consumed: its fields move into the partial
//! messages prost encodes, so the caller keeps no tensor once the frames
//! exist; the length prefix is checked against the four-byte gRPC limit as
//! tonic's encoder checks it. Nothing in this module is `unsafe`, and
//! nothing in it is public: the hand-mirrored layouts stay inside the
//! crate.

use std::future::poll_fn;

use bytes::{BufMut, Bytes, BytesMut};
use http_body_util::StreamBody;
use prost::{
    encoding::{encode_key, encode_varint, message, string, WireType},
    Message,
};
use tonic::{
    client::GrpcService,
    codec::{BufferSettings, Streaming},
    codegen::http::{self, header::HeaderValue, Method, Request as HttpRequest, Version},
    Status,
};
use tonic_prost::ProstCodec;

use crate::{
    channel::Channel,
    codec::{encode_prompt_carrier, reserves_for, EAGER_BUFFER_BYTES},
    sglang_proto as sg,
    tokenspeed_scheduler::tokenspeed_proto as ts,
    vllm_proto as vllm,
};

/// Payloads shorter than this are copied into the scratch like any field:
/// a frame per tiny payload would cost more than its copy.
const FRAME_MIN_BYTES: usize = 64 * 1024;

/// The gRPC message prefix: the compression flag and the four-byte length.
const PREFIX_BYTES: usize = 5;

/// The frames of a message being written: a scratch for the small fields,
/// cut into a frame whenever a payload frame follows.
struct Frames {
    scratch: BytesMut,
    frames: Vec<Bytes>,
}

impl Frames {
    /// A builder for a message of `len` encoded bytes, `framed_bytes` of
    /// them riding as payload frames, its gRPC prefix written: the scratch
    /// holds every byte that is not a payload frame without growing.
    fn for_message(len: usize, framed_bytes: usize) -> Result<Self, Status> {
        let prefix = prefix_of(len)?;
        let mut out = Self {
            scratch: BytesMut::with_capacity(PREFIX_BYTES + len - framed_bytes),
            frames: Vec::with_capacity(8),
        };
        out.scratch.put_u8(0);
        out.scratch.put_u32(prefix);
        Ok(out)
    }

    /// The one frame of a message none of whose payloads rides as a frame
    /// of its own: the prefix reserved, the message written once, the
    /// length taken from the buffer, as the crate's codec writes a plain
    /// request. A text-only request that carries a prompt goes through the
    /// codec's one-pass writer, as the codec writes it, with this frame as
    /// the buffer: the writer reserves the frame's length for one walk of
    /// the request's other fields. A message the writer's entry refuses is
    /// prost's one write, with no walk over its length before it unless it
    /// carries media: then the buffer is reserved to that length first (the
    /// codec's own rule, [`reserves_for`]); else it starts at tonic's eager
    /// size, so a plain request grows it no more than the codec's buffer
    /// grows. The message is consumed: the writer takes its request id and
    /// its prompt out of it.
    fn whole<M: Message + 'static>(mut message: M) -> Result<Vec<Bytes>, Status> {
        let mut scratch = if reserves_for(&message) {
            BytesMut::with_capacity(PREFIX_BYTES + message.encoded_len())
        } else {
            BytesMut::with_capacity(EAGER_BUFFER_BYTES)
        };
        scratch.put_bytes(0, PREFIX_BYTES);
        if !encode_prompt_carrier(&mut message, &mut scratch) {
            message.encode_raw(&mut scratch);
        }
        let prefix = prefix_of(scratch.len() - PREFIX_BYTES)?;
        scratch[1..PREFIX_BYTES].copy_from_slice(&prefix.to_be_bytes());
        Ok(vec![scratch.freeze()])
    }

    /// The scratch so far becomes a frame.
    fn flush(&mut self) {
        if !self.scratch.is_empty() {
            self.frames.push(self.scratch.split().freeze());
        }
    }

    /// `payload` as a frame of its own, after the scratch so far.
    fn payload(&mut self, payload: Bytes) {
        self.flush();
        self.frames.push(payload);
    }

    /// The key of a length-delimited field and the length that follows it.
    fn length_delimited(&mut self, tag: u32, len: usize) {
        encode_key(tag, WireType::LengthDelimited, &mut self.scratch);
        encode_varint(len as u64, &mut self.scratch);
    }

    fn finish(mut self) -> Vec<Bytes> {
        self.flush();
        self.frames
    }
}

/// The four-byte length of the gRPC prefix, or the refusal tonic's encoder
/// gives a message longer than it can carry.
fn prefix_of(len: usize) -> Result<u32, Status> {
    u32::try_from(len).map_err(|_| {
        Status::resource_exhausted(format!(
            "Cannot return body with more than 4GB of data: {len} bytes"
        ))
    })
}

/// One entry of a `map<string, M>` field `tag` as prost writes it: the
/// entry's key and length, then the key as field 1 unless it is empty and
/// the value as field 2 unless it is the default, the value through
/// `value_frames` so a large payload still takes its own frame.
fn map_entry_frames<M: Message + Default + PartialEq>(
    tag: u32,
    key: String,
    value: M,
    out: &mut Frames,
    value_frames: fn(u32, M, &mut Frames),
) {
    let key_len = if key.is_empty() {
        0
    } else {
        string::encoded_len(1, &key)
    };
    let value_len = if value == M::default() {
        0
    } else {
        message::encoded_len(2, &value)
    };
    out.length_delimited(tag, key_len + value_len);
    if key_len != 0 {
        string::encode(1, &key, &mut out.scratch);
    }
    if value_len != 0 {
        value_frames(2, value, out);
    }
}

/// The body frames of the gRPC message for a TokenSpeed `GenerateRequest`:
/// the 5-byte length prefix and the message, every inline encoder input of at
/// least `FRAME_MIN_BYTES` as a frame of its own. The joined frames are the
/// message's generated encoding byte for byte. The request is consumed; a
/// message longer than the four-byte prefix can carry is refused as tonic's
/// encoder refuses it.
pub(crate) fn tokenspeed_frames(req: ts::GenerateRequest) -> Result<Vec<Bytes>, Status> {
    let framed_bytes: usize = req
        .mm_inputs
        .as_ref()
        .map_or(0, |mm| mm.items.iter().filter_map(framed_payload_len).sum());
    if framed_bytes == 0 {
        return Frames::whole(req);
    }
    let mut out = Frames::for_message(req.encoded_len(), framed_bytes)?;
    let mm_len = req.mm_inputs.as_ref().map_or(0, Message::encoded_len);
    // Every field named, so a field added to the message is a compile error
    // here until the layout places it before or after the media inputs.
    let ts::GenerateRequest {
        request_id,
        tokenized,
        sampling_params,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        token_ids_logprob,
        stream,
        mm_inputs,
        encode_bootstrap_info,
        kv_bootstrap_info,
        data_parallel_rank,
    } = req;
    // Fields 1-8, in prost's order, moved.
    ts::GenerateRequest {
        request_id,
        tokenized,
        sampling_params,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        token_ids_logprob,
        stream,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
    // Field 9: the media inputs hold the items alone.
    out.length_delimited(9, mm_len);
    let items = match mm_inputs {
        Some(ts::MultimodalInputs { items }) => items,
        None => Vec::new(),
    };
    for item in items {
        item_frames(item, &mut out);
    }
    // Fields 10-12.
    ts::GenerateRequest {
        encode_bootstrap_info,
        kv_bootstrap_info,
        data_parallel_rank,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
    Ok(out.finish())
}

/// One item of the media inputs: its fields before the encoder input, the
/// encoder input with the payload as a frame of its own, its fields after.
/// An item without a large inline payload is written whole, in order. The
/// item is consumed either way.
fn item_frames(item: ts::MultimodalItem, out: &mut Frames) {
    let item_len = item.encoded_len();
    let tensor_len = item.encoder_input.as_ref().map_or(0, Message::encoded_len);
    let ts::MultimodalItem {
        modality,
        content_hash,
        encoder_input,
        model_specific_tensors,
        placeholders,
        placeholder_token_id,
    } = item;
    let (tensor, payload) = match encoder_input {
        Some(ts::TensorData {
            shape,
            dtype,
            payload: Some(ts::tensor_data::Payload::Inline(bytes)),
        }) if frames_its_own(&bytes) => (
            ts::TensorData {
                shape,
                dtype,
                payload: None,
            },
            bytes,
        ),
        encoder_input => {
            let item = ts::MultimodalItem {
                modality,
                content_hash,
                encoder_input,
                model_specific_tensors,
                placeholders,
                placeholder_token_id,
            };
            out.length_delimited(1, item_len);
            item.encode_raw(&mut out.scratch);
            return;
        }
    };
    out.length_delimited(1, item_len);
    // Fields 1-2.
    ts::MultimodalItem {
        modality,
        content_hash,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
    // Field 3: the tensor's fields before its payload, then the payload as
    // the next frame; the payload oneof is the tensor's last field.
    out.length_delimited(3, tensor_len);
    tensor.encode_raw(&mut out.scratch);
    out.length_delimited(3, payload.len());
    out.payload(payload);
    // Fields 4-6.
    ts::MultimodalItem {
        model_specific_tensors,
        placeholders,
        placeholder_token_id,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
}

/// Whether an inline payload rides as a frame of its own.
fn frames_its_own(bytes: &Bytes) -> bool {
    bytes.len() >= FRAME_MIN_BYTES
}

/// The length of the item's inline encoder payload when it rides as a frame
/// of its own.
fn framed_payload_len(item: &ts::MultimodalItem) -> Option<usize> {
    match item.encoder_input.as_ref()?.payload.as_ref()? {
        ts::tensor_data::Payload::Inline(bytes) if frames_its_own(bytes) => Some(bytes.len()),
        _ => None,
    }
}

/// The body frames of the gRPC message for a vLLM `GenerateRequest`: the
/// 5-byte length prefix and the message, every inline tensor payload of at
/// least `FRAME_MIN_BYTES` as a frame of its own, in the media inputs and in
/// every extra media input alike. The joined frames are the message's
/// generated encoding byte for byte. The request is consumed; a message
/// longer than the four-byte prefix can carry is refused as tonic's encoder
/// refuses it.
pub(crate) fn vllm_frames(req: vllm::GenerateRequest) -> Result<Vec<Bytes>, Status> {
    let framed_bytes = req.mm_inputs.as_ref().map_or(0, framed_inputs_len)
        + req
            .extra_mm_inputs
            .iter()
            .map(framed_inputs_len)
            .sum::<usize>();
    if framed_bytes == 0 {
        return Frames::whole(req);
    }
    let mut out = Frames::for_message(req.encoded_len(), framed_bytes)?;
    // Every field named, so a field added to the message is a compile error
    // here until the layout places it before, between or after the two
    // media paths.
    let vllm::GenerateRequest {
        request_id,
        input,
        sampling_params,
        stream,
        kv_transfer_params,
        mm_inputs,
        kv_transfer_params_json,
        data_parallel_rank,
        media_refs,
        extra_mm_inputs,
        ec_transfer_params_json,
        chat_template_kwargs_json,
    } = req;
    // Fields 1-6, in prost's order, moved.
    vllm::GenerateRequest {
        request_id,
        sampling_params,
        stream,
        kv_transfer_params,
        input,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
    // Field 7.
    if let Some(inputs) = mm_inputs {
        out.length_delimited(7, inputs.encoded_len());
        inputs_frames(inputs, &mut out);
    }
    // Fields 8-10.
    vllm::GenerateRequest {
        kv_transfer_params_json,
        data_parallel_rank,
        media_refs,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
    // Field 11, every element in order.
    for inputs in extra_mm_inputs {
        out.length_delimited(11, inputs.encoded_len());
        inputs_frames(inputs, &mut out);
    }
    // Fields 12-13.
    vllm::GenerateRequest {
        ec_transfer_params_json,
        chat_template_kwargs_json,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
    Ok(out.finish())
}

/// One media input of a vLLM request, its field key and length already
/// written: the pixel values and the model-specific tensors with every
/// large inline payload as a frame of its own, then its fields after them.
/// A media input without a large inline payload is written whole, in order.
/// The input is consumed either way.
fn inputs_frames(inputs: vllm::MultimodalInputs, out: &mut Frames) {
    if framed_inputs_len(&inputs) == 0 {
        inputs.encode_raw(&mut out.scratch);
        return;
    }
    let vllm::MultimodalInputs {
        pixel_values,
        model_specific_tensors,
        im_token_id,
        mm_placeholders,
        mm_hashes,
        batched_keys,
        flat_keys,
        keep_on_cpu_keys,
        modality,
        encoder_input_key,
    } = inputs;
    // Field 2.
    if let Some(tensor) = pixel_values {
        tensor_frames(2, tensor, out);
    }
    // Field 3: the map's entries, in the map's own order.
    for (key, tensor) in model_specific_tensors {
        map_entry_frames(3, key, tensor, out, tensor_frames);
    }
    // Fields 4-11.
    vllm::MultimodalInputs {
        im_token_id,
        mm_placeholders,
        mm_hashes,
        batched_keys,
        flat_keys,
        keep_on_cpu_keys,
        modality,
        encoder_input_key,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
}

/// A vLLM tensor as field `tag` of the message being written: an inline
/// payload of at least `FRAME_MIN_BYTES` as the next frame, led by its own
/// key and length and followed by the tensor's shape and dtype (the payload
/// oneof is the tensor's first field); any other tensor whole, in order.
fn tensor_frames(tag: u32, tensor: vllm::TensorData, out: &mut Frames) {
    out.length_delimited(tag, tensor.encoded_len());
    match tensor {
        vllm::TensorData {
            shape,
            dtype,
            payload: Some(vllm::tensor_data::Payload::Inline(bytes)),
        } if frames_its_own(&bytes) => {
            out.length_delimited(1, bytes.len());
            out.payload(bytes);
            vllm::TensorData {
                shape,
                dtype,
                payload: None,
            }
            .encode_raw(&mut out.scratch);
        }
        tensor => tensor.encode_raw(&mut out.scratch),
    }
}

/// The bytes of a vLLM media input's inline payloads that ride as frames of
/// their own: the pixel values and every model-specific tensor.
fn framed_inputs_len(inputs: &vllm::MultimodalInputs) -> usize {
    inputs.pixel_values.as_ref().map_or(0, framed_tensor_len)
        + inputs
            .model_specific_tensors
            .values()
            .map(framed_tensor_len)
            .sum::<usize>()
}

/// The length of a vLLM tensor's inline payload when it rides as a frame of
/// its own, else 0.
fn framed_tensor_len(tensor: &vllm::TensorData) -> usize {
    match &tensor.payload {
        Some(vllm::tensor_data::Payload::Inline(bytes)) if frames_its_own(bytes) => bytes.len(),
        _ => 0,
    }
}

/// The body frames of the gRPC message for an SGLang `GenerateRequest`: the
/// 5-byte length prefix and the message, every inline tensor payload of at
/// least `FRAME_MIN_BYTES` as a frame of its own, in the media inputs and in
/// every extra media input alike. The joined frames are the message's
/// generated encoding byte for byte. The request is consumed; a message
/// longer than the four-byte prefix can carry is refused as tonic's encoder
/// refuses it.
pub(crate) fn sglang_frames(req: sg::GenerateRequest) -> Result<Vec<Bytes>, Status> {
    let framed_bytes = req.mm_inputs.as_deref().map_or(0, sglang_framed_inputs_len)
        + req
            .extra_mm_inputs
            .iter()
            .map(sglang_framed_inputs_len)
            .sum::<usize>();
    if framed_bytes == 0 {
        return Frames::whole(req);
    }
    let mut out = Frames::for_message(req.encoded_len(), framed_bytes)?;
    // Every field named, so a field added to the message is a compile error
    // here until the layout places it before, between or after the two
    // media paths.
    let sg::GenerateRequest {
        request_id,
        tokenized,
        mm_inputs,
        sampling_params,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        token_ids_logprob,
        return_hidden_states,
        disaggregated_params,
        custom_logit_processor,
        timestamp,
        log_metrics,
        input_embeds,
        lora_id,
        data_parallel_rank,
        stream,
        require_reasoning,
        return_media_identity,
        media_identity,
        extra_mm_inputs,
    } = req;
    // Fields 1-2, moved.
    sg::GenerateRequest {
        request_id,
        tokenized,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
    // Field 3, out of its box.
    if let Some(inputs) = mm_inputs {
        out.length_delimited(3, inputs.encoded_len());
        sglang_inputs_frames(*inputs, &mut out);
    }
    // Fields 4-20.
    sg::GenerateRequest {
        sampling_params,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        token_ids_logprob,
        return_hidden_states,
        disaggregated_params,
        custom_logit_processor,
        timestamp,
        log_metrics,
        input_embeds,
        lora_id,
        data_parallel_rank,
        stream,
        require_reasoning,
        return_media_identity,
        media_identity,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
    // Field 21, every element in order; the message's last field.
    for inputs in extra_mm_inputs {
        out.length_delimited(21, inputs.encoded_len());
        sglang_inputs_frames(inputs, &mut out);
    }
    Ok(out.finish())
}

/// One media input of an SGLang request, its field key and length already
/// written: the media references, the pixel values with a large inline
/// payload as a frame of its own, the raw media lists and modalities, the
/// model-specific tensors the same way, then the fields after them. A media
/// input without a large inline payload is written whole, in order. The
/// input is consumed either way.
fn sglang_inputs_frames(inputs: sg::MultimodalInputs, out: &mut Frames) {
    if sglang_framed_inputs_len(&inputs) == 0 {
        inputs.encode_raw(&mut out.scratch);
        return;
    }
    let sg::MultimodalInputs {
        image_urls,
        video_urls,
        audio_urls,
        pixel_values,
        image_data,
        video_data,
        audio_data,
        modalities,
        model_specific_tensors,
        im_token_id,
        mm_placeholders,
        feature_hash,
    } = inputs;
    // Fields 1-3.
    sg::MultimodalInputs {
        image_urls,
        video_urls,
        audio_urls,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
    // Field 4.
    if let Some(tensor) = pixel_values {
        sglang_tensor_frames(4, tensor, out);
    }
    // Fields 5-8: the raw media lists are owned byte vectors, not shared
    // buffers; they stay in the scratch.
    sg::MultimodalInputs {
        image_data,
        video_data,
        audio_data,
        modalities,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
    // Field 9: the map's entries, in the map's own order.
    for (key, tensor) in model_specific_tensors {
        map_entry_frames(9, key, tensor, out, sglang_tensor_frames);
    }
    // Fields 10-12.
    sg::MultimodalInputs {
        im_token_id,
        mm_placeholders,
        feature_hash,
        ..Default::default()
    }
    .encode_raw(&mut out.scratch);
}

/// An SGLang tensor as field `tag` of the message being written: a payload
/// of at least `FRAME_MIN_BYTES` as the next frame, led by its own key and
/// length and followed by the tensor's shape and dtype (the payload is the
/// tensor's first field, a plain bytes field); any other tensor whole, in
/// order.
fn sglang_tensor_frames(tag: u32, tensor: sg::TensorData, out: &mut Frames) {
    out.length_delimited(tag, tensor.encoded_len());
    if !frames_its_own(&tensor.data) {
        tensor.encode_raw(&mut out.scratch);
        return;
    }
    let sg::TensorData { data, shape, dtype } = tensor;
    out.length_delimited(1, data.len());
    out.payload(data);
    // An empty bytes field is not written, so the copy holds the two fields
    // after the payload alone.
    sg::TensorData {
        data: Bytes::new(),
        shape,
        dtype,
    }
    .encode_raw(&mut out.scratch);
}

/// The bytes of an SGLang media input's inline payloads that ride as frames
/// of their own: the pixel values and every model-specific tensor.
fn sglang_framed_inputs_len(inputs: &sg::MultimodalInputs) -> usize {
    inputs
        .pixel_values
        .as_ref()
        .map_or(0, sglang_framed_tensor_len)
        + inputs
            .model_specific_tensors
            .values()
            .map(sglang_framed_tensor_len)
            .sum::<usize>()
}

/// The length of an SGLang tensor's payload when it rides as a frame of its
/// own, else 0.
fn sglang_framed_tensor_len(tensor: &sg::TensorData) -> usize {
    if frames_its_own(&tensor.data) {
        tensor.data.len()
    } else {
        0
    }
}

/// The unary-streaming call with `frames` as the request body over the
/// client's own channel: the request headers (the trace metadata among them)
/// and the gRPC content type, the response read as `R` messages. Transport
/// errors map through tonic's `Status::from_error`, as the generated client
/// maps them.
pub(crate) async fn call_with_frames<R>(
    channel: &Channel,
    path: &'static str,
    headers: http::HeaderMap,
    frames: Vec<Bytes>,
) -> Result<Streaming<R>, Status>
where
    R: Message + Default + Send + 'static,
{
    let body = tonic::body::Body::new(StreamBody::new(futures::stream::iter(
        frames
            .into_iter()
            .map(|frame| Ok::<_, Status>(http_body::Frame::data(frame))),
    )));
    let mut request = HttpRequest::builder()
        .method(Method::POST)
        .uri(path)
        .version(Version::HTTP_2)
        .body(body)
        .map_err(|error| Status::internal(format!("framed request: {error}")))?;
    let request_headers = request.headers_mut();
    *request_headers = headers;
    request_headers.insert(http::header::TE, HeaderValue::from_static("trailers"));
    request_headers.insert(
        http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/grpc"),
    );
    let mut channel = channel.clone();
    poll_fn(|cx| GrpcService::<tonic::body::Body>::poll_ready(&mut channel, cx))
        .await
        .map_err(|error| Status::unknown(format!("Service was not ready: {error}")))?;
    let response = GrpcService::<tonic::body::Body>::call(&mut channel, request)
        .await
        .map_err(Status::from_error)?;
    response_stream(response)
}

/// The response of a framed call as a stream of `R`, read as tonic's own
/// client reads it: a status in the headers (a trailers-only response) is the
/// error it names, or an empty stream when it is OK; otherwise the body
/// streams its messages and ends on its trailers.
fn response_stream<B, R>(response: http::Response<B>) -> Result<Streaming<R>, Status>
where
    B: http_body::Body + Send + 'static,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    R: Message + Default + Send + 'static,
{
    let decoder = ProstCodec::<ts::GenerateRequest, R>::raw_decoder(BufferSettings::default());
    if let Some(status) = Status::from_header_map(response.headers()) {
        if status.code() != tonic::Code::Ok {
            return Err(status);
        }
        return Ok(Streaming::new_empty(decoder, response.into_body()));
    }
    let status_code = response.status();
    Ok(Streaming::new_response(
        decoder,
        response.into_body(),
        status_code,
        None,
        None,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{codec::scratch_capacity, common_proto::ShmHandle};

    /// The 1080p float32 class and the 4k float32 class of the bed's corpus.
    const CLASS_1080P: usize = 1920 * 1080 * 3 * 4;
    const CLASS_4K: usize = 4096 * 2304 * 3 * 4;

    fn inline_item(fill: u8, len: usize) -> ts::MultimodalItem {
        ts::MultimodalItem {
            content_hash: vec![fill; 4],
            encoder_input: Some(ts::TensorData {
                shape: vec![1, (len / 4) as u32],
                dtype: "float32".into(),
                payload: Some(ts::tensor_data::Payload::Inline(Bytes::from(vec![
                    fill;
                    len
                ]))),
            }),
            placeholder_token_id: Some(7),
            ..Default::default()
        }
    }

    fn shm_item() -> ts::MultimodalItem {
        ts::MultimodalItem {
            content_hash: vec![9; 4],
            encoder_input: Some(ts::TensorData {
                shape: vec![3, 224, 224],
                dtype: "bfloat16".into(),
                payload: Some(ts::tensor_data::Payload::Shm(ShmHandle {
                    name: "/smg-encoder-input".into(),
                    offset: 4096,
                    nbytes: 301_056,
                    owner_id: "gateway".into(),
                })),
            }),
            ..Default::default()
        }
    }

    fn request(items: Vec<ts::MultimodalItem>) -> ts::GenerateRequest {
        ts::GenerateRequest {
            request_id: "r".into(),
            mm_inputs: Some(ts::MultimodalInputs { items }),
            ..Default::default()
        }
    }

    /// Every field of every message on the path set, the fields after the
    /// path's field included: the proof that the frames keep prost's order.
    fn request_with_every_field() -> ts::GenerateRequest {
        let mut item = inline_item(1, FRAME_MIN_BYTES + 3);
        item.modality = 1;
        item.model_specific_tensors = HashMap::from([(
            "image_grid_thw".to_string(),
            ts::TensorData {
                shape: vec![1, 3],
                dtype: "int64".into(),
                payload: Some(ts::tensor_data::Payload::Inline(Bytes::from(vec![0u8; 24]))),
            },
        )]);
        item.placeholders = vec![ts::PlaceholderRange {
            offset: 3,
            length: 64,
        }];
        ts::GenerateRequest {
            request_id: "every-field".into(),
            tokenized: Some(ts::TokenizedInput::default()),
            sampling_params: Some(ts::SamplingParams::default()),
            return_logprob: true,
            logprob_start_len: Some(2),
            top_logprobs_num: 5,
            token_ids_logprob: vec![11, 12],
            stream: true,
            mm_inputs: Some(ts::MultimodalInputs {
                items: vec![item, shm_item()],
            }),
            encode_bootstrap_info: Some(ts::EncodeBootstrapInfo::default()),
            kv_bootstrap_info: Some(ts::KvBootstrapInfo {
                bootstrap_host: "decode-0".into(),
                bootstrap_port: 8998,
                bootstrap_room: 42,
            }),
            data_parallel_rank: Some(1),
        }
    }

    fn joined(frames: &[Bytes]) -> (u32, Vec<u8>) {
        let all: Vec<u8> = frames.iter().flat_map(|f| f.iter().copied()).collect();
        assert_eq!(all[0], 0, "no compression flag");
        let len = u32::from_be_bytes([all[1], all[2], all[3], all[4]]);
        (len, all[5..].to_vec())
    }

    /// The frames of `req`, which stays with the caller for the comparison.
    fn frames_of(req: &ts::GenerateRequest) -> Vec<Bytes> {
        tokenspeed_frames(req.clone()).expect("a request the prefix can carry")
    }

    /// The inline payload of the request's first item.
    fn first_payload(req: &ts::GenerateRequest) -> &Bytes {
        match &req.mm_inputs.as_ref().unwrap().items[0]
            .encoder_input
            .as_ref()
            .unwrap()
            .payload
        {
            Some(ts::tensor_data::Payload::Inline(bytes)) => bytes,
            _ => panic!("an inline payload"),
        }
    }

    /// The joined frames are the generated encoding and decode to the request.
    fn assert_identical<M>(req: &M, frames: &[Bytes])
    where
        M: Message + Default + PartialEq + std::fmt::Debug,
    {
        let (len, body) = joined(frames);
        assert_eq!(len as usize, req.encoded_len(), "the length prefix");
        assert!(
            body == req.encode_to_vec(),
            "the body is not prost's encoding byte for byte"
        );
        let decoded = M::decode(body.as_slice()).expect("the body decodes");
        assert_eq!(&decoded, req);
    }

    #[test]
    fn a_request_without_media_is_one_frame_of_the_plain_encoding() {
        let req = ts::GenerateRequest {
            request_id: "text".into(),
            stream: true,
            ..Default::default()
        };
        let frames = frames_of(&req);
        assert_eq!(frames.len(), 1);
        assert_identical(&req, &frames);
    }

    #[test]
    fn payloads_below_the_frame_size_stay_inline_in_one_frame() {
        let req = request(vec![
            inline_item(1, 10),
            inline_item(2, FRAME_MIN_BYTES - 1),
        ]);
        let frames = frames_of(&req);
        assert_eq!(frames.len(), 1);
        assert_identical(&req, &frames);
    }

    #[test]
    fn one_large_part_rides_as_its_own_frame_and_shares_its_memory() {
        let req = request(vec![inline_item(1, CLASS_1080P)]);
        let frames = frames_of(&req);
        assert_eq!(frames.len(), 3, "prefix and head, the payload, the tail");
        assert_eq!(
            frames[1].as_ptr(),
            first_payload(&req).as_ptr(),
            "the payload frame is the payload's memory"
        );
        assert_eq!(frames[1].len(), CLASS_1080P);
        assert_identical(&req, &frames);
    }

    #[test]
    fn many_parts_keep_their_order_with_small_and_shared_memory_parts_between() {
        let req = request(vec![
            inline_item(1, FRAME_MIN_BYTES + 3),
            inline_item(2, 10),
            shm_item(),
            inline_item(3, FRAME_MIN_BYTES),
        ]);
        let frames = frames_of(&req);
        assert_eq!(
            frames.len(),
            5,
            "prefix and head, payload, the middle, payload, tail"
        );
        assert_identical(&req, &frames);
    }

    #[test]
    fn every_field_set_keeps_prosts_field_order_around_the_media_path() {
        let req = request_with_every_field();
        let frames = frames_of(&req);
        assert_eq!(frames.len(), 3);
        assert_identical(&req, &frames);
    }

    #[test]
    fn a_plain_request_is_one_frame_of_exactly_its_length() {
        let req = request_with_every_field();
        let plain = ts::GenerateRequest {
            mm_inputs: None,
            ..req
        };
        let frames = frames_of(&plain);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), PREFIX_BYTES + plain.encoded_len());
        assert_identical(&plain, &frames);
    }

    /// A tokenized prompt: ids across the varint widths and the rendered text.
    fn prompt() -> ts::TokenizedInput {
        ts::TokenizedInput {
            input_ids: (0..1_114u32)
                .map(|i| i.wrapping_mul(2_654_435_761) % 150_000 + 10)
                .collect(),
            original_text: "a rendered conversation ".repeat(175),
        }
    }

    /// A text-only request that carries a prompt is one frame of prost's
    /// bytes, written by the codec's one-pass writer: on a fresh thread the
    /// writer's scratch holds its packed ids afterwards.
    #[test]
    fn a_text_only_prompt_is_one_frame_through_the_one_pass_writer() {
        std::thread::spawn(|| {
            let req = ts::GenerateRequest {
                request_id: "chatcmpl-0123456789abcdef".into(),
                tokenized: Some(prompt()),
                sampling_params: Some(ts::SamplingParams::default()),
                stream: true,
                ..Default::default()
            };
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].len(), PREFIX_BYTES + req.encoded_len());
            assert_identical(&req, &frames);
            let ids = scratch_capacity();
            assert!(ids > 0, "the one-pass writer packed the ids: {ids}");
        })
        .join()
        .expect("the text-only pin holds");
    }

    /// A request with media never reaches the one-pass writer, whatever
    /// carries the media: a payload that rides as a frame of its own, a small
    /// payload inline in the one frame, a shared-memory handle. Each is
    /// prost's bytes and the thread's scratch stays empty: the writer's entry
    /// refused the request.
    #[test]
    fn a_request_with_media_never_reaches_the_one_pass_writer() {
        std::thread::spawn(|| {
            for items in [
                vec![inline_item(1, CLASS_1080P)],
                vec![inline_item(2, FRAME_MIN_BYTES - 1)],
                vec![shm_item()],
            ] {
                let req = ts::GenerateRequest {
                    tokenized: Some(prompt()),
                    ..request(items)
                };
                assert_identical(&req, &frames_of(&req));
            }
            assert_eq!(scratch_capacity(), 0);
        })
        .join()
        .expect("the media pin holds");
    }

    fn trailers_only(status: &str, message: Option<&str>) -> http::Response<tonic::body::Body> {
        let mut response = http::Response::builder()
            .status(200)
            .header("content-type", "application/grpc")
            .header("grpc-status", status);
        if let Some(message) = message {
            response = response.header("grpc-message", message);
        }
        response.body(tonic::body::Body::empty()).unwrap()
    }

    #[tokio::test]
    async fn a_trailers_only_ok_response_is_an_empty_stream() {
        let mut stream = response_stream::<_, ts::GenerateResponse>(trailers_only("0", None))
            .expect("an OK status in the headers is not an error");
        assert!(
            stream.message().await.expect("a clean end").is_none(),
            "no message, no error"
        );
    }

    #[tokio::test]
    async fn a_trailers_only_error_is_the_status_it_names() {
        let error =
            response_stream::<_, ts::GenerateResponse>(trailers_only("14", Some("engine away")))
                .expect_err("the status in the headers");
        assert_eq!(error.code(), tonic::Code::Unavailable);
        assert_eq!(error.message(), "engine away");
    }

    #[test]
    fn the_largest_tensor_class_is_one_frame_of_its_own() {
        let req = request(vec![inline_item(5, CLASS_4K)]);
        let frames = frames_of(&req);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[1].len(), CLASS_4K);
        assert_identical(&req, &frames);
    }

    /// The vLLM wire: the same matrix over its message, whose tensors sit in
    /// `pixel_values`, in a map and in extra media inputs, and whose inline
    /// payload is the tensor's first field.
    mod vllm_wire {
        use std::collections::HashMap;

        use bytes::Bytes;
        use prost::Message;

        use super::{assert_identical, joined, CLASS_1080P, CLASS_4K};
        use crate::{
            codec::scratch_capacity,
            common_proto::{RemoteTensorHandle, ShmHandle},
            framed::{vllm_frames, FRAME_MIN_BYTES, PREFIX_BYTES},
            vllm_proto as vllm,
        };

        fn inline_tensor(fill: u8, len: usize) -> vllm::TensorData {
            vllm::TensorData {
                shape: vec![1, (len / 4) as u32],
                dtype: "float32".into(),
                payload: Some(vllm::tensor_data::Payload::Inline(Bytes::from(vec![
                    fill;
                    len
                ]))),
            }
        }

        fn shm_tensor() -> vllm::TensorData {
            vllm::TensorData {
                shape: vec![3, 224, 224],
                dtype: "bfloat16".into(),
                payload: Some(vllm::tensor_data::Payload::Shm(ShmHandle {
                    name: "/smg-pixel-values".into(),
                    offset: 4096,
                    nbytes: 301_056,
                    owner_id: "gateway".into(),
                })),
            }
        }

        fn remote_tensor() -> vllm::TensorData {
            vllm::TensorData {
                shape: vec![1, 1176],
                dtype: "float32".into(),
                payload: Some(vllm::tensor_data::Payload::Remote(RemoteTensorHandle {
                    transport: "nixl".into(),
                    descriptor: vec![7; 16],
                    nbytes: 4704,
                })),
            }
        }

        /// Media inputs with `pixel_values` and the named model-specific
        /// tensors, nothing else.
        fn inputs(
            pixel_values: Option<vllm::TensorData>,
            tensors: impl IntoIterator<Item = (&'static str, vllm::TensorData)>,
        ) -> vllm::MultimodalInputs {
            vllm::MultimodalInputs {
                pixel_values,
                model_specific_tensors: tensors
                    .into_iter()
                    .map(|(key, tensor)| (key.to_string(), tensor))
                    .collect(),
                ..Default::default()
            }
        }

        fn request(mm_inputs: vllm::MultimodalInputs) -> vllm::GenerateRequest {
            vllm::GenerateRequest {
                request_id: "r".into(),
                mm_inputs: Some(mm_inputs),
                ..Default::default()
            }
        }

        /// Every field of the media inputs set: a large and a small inline
        /// tensor, a shared-memory one, an empty key and a default value in
        /// the map, every list, the flat keys.
        fn inputs_with_every_field(fill: u8) -> vllm::MultimodalInputs {
            vllm::MultimodalInputs {
                pixel_values: Some(inline_tensor(fill, FRAME_MIN_BYTES + 3)),
                model_specific_tensors: HashMap::from([
                    ("image_grid_thw".to_string(), inline_tensor(fill, 24)),
                    (String::new(), inline_tensor(fill, FRAME_MIN_BYTES)),
                    ("empty".to_string(), vllm::TensorData::default()),
                    ("shm".to_string(), shm_tensor()),
                ]),
                im_token_id: Some(151_655),
                mm_placeholders: vec![vllm::PlaceholderRange {
                    offset: 3,
                    length: 64,
                }],
                mm_hashes: vec!["h1".into()],
                batched_keys: vec!["pixel_values".into()],
                flat_keys: HashMap::from([("image_grid_thw".to_string(), "x".to_string())]),
                keep_on_cpu_keys: vec!["image_grid_thw".into()],
                modality: 1,
                encoder_input_key: Some("pixel_values".into()),
            }
        }

        /// Every field of the request set, the fields between and after the
        /// two media paths included: the proof that the frames keep prost's
        /// order around both.
        fn request_with_every_field() -> vllm::GenerateRequest {
            vllm::GenerateRequest {
                request_id: "every-field".into(),
                sampling_params: Some(vllm::SamplingParams::default()),
                stream: true,
                kv_transfer_params: Some(vllm::KvTransferParams {
                    remote_host: "decode-0".into(),
                    remote_port: 8998,
                }),
                mm_inputs: Some(inputs_with_every_field(1)),
                extra_mm_inputs: vec![
                    inputs_with_every_field(2),
                    vllm::MultimodalInputs::default(),
                ],
                ec_transfer_params_json: Some("{\"ec\":1}".into()),
                kv_transfer_params_json: Some("{}".into()),
                data_parallel_rank: Some(1),
                media_refs: Some(vllm::MediaRefs::default()),
                chat_template_kwargs_json: Some("{}".into()),
                input: Some(vllm::generate_request::Input::Tokenized(
                    vllm::TokenizedInput {
                        original_text: "hi".into(),
                        input_ids: vec![1, 2],
                    },
                )),
            }
        }

        /// The frames of `req`, which stays with the caller for the comparison.
        fn frames_of(req: &vllm::GenerateRequest) -> Vec<Bytes> {
            vllm_frames(req.clone()).expect("a request the prefix can carry")
        }

        fn payload_of(tensor: &vllm::TensorData) -> &Bytes {
            match &tensor.payload {
                Some(vllm::tensor_data::Payload::Inline(bytes)) => bytes,
                _ => panic!("an inline payload"),
            }
        }

        #[test]
        fn a_request_without_media_is_one_frame_of_the_plain_encoding() {
            let req = vllm::GenerateRequest {
                request_id: "text".into(),
                stream: true,
                ..Default::default()
            };
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 1);
            assert_identical(&req, &frames);
        }

        #[test]
        fn payloads_below_the_frame_size_stay_inline_in_one_frame() {
            let mut req = request(inputs(
                Some(inline_tensor(1, 10)),
                [("image_grid_thw", inline_tensor(2, FRAME_MIN_BYTES - 1))],
            ));
            req.extra_mm_inputs = vec![inputs(Some(inline_tensor(3, 16)), [])];
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].len(), PREFIX_BYTES + req.encoded_len());
            assert_identical(&req, &frames);
        }

        #[test]
        fn large_pixel_values_ride_as_their_own_frame_led_by_their_key() {
            let req = request(inputs(Some(inline_tensor(1, CLASS_1080P)), []));
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 3, "prefix and head, the payload, the tail");
            let pixel_values = req
                .mm_inputs
                .as_ref()
                .unwrap()
                .pixel_values
                .as_ref()
                .unwrap();
            assert_eq!(
                frames[1].as_ptr(),
                payload_of(pixel_values).as_ptr(),
                "the payload frame is the payload's memory"
            );
            assert_eq!(frames[1].len(), CLASS_1080P);
            assert_eq!(
                frames[2][0], 0x12,
                "the tensor's shape (field 2) follows its payload (field 1)"
            );
            assert_identical(&req, &frames);
        }

        #[test]
        fn map_entries_keep_the_maps_order_around_an_empty_key_and_a_default_value() {
            let req = request(inputs(
                None,
                [
                    ("", inline_tensor(1, FRAME_MIN_BYTES + 1)),
                    ("empty", vllm::TensorData::default()),
                    ("image_grid_thw", inline_tensor(2, 24)),
                    ("second_image", inline_tensor(3, FRAME_MIN_BYTES)),
                ],
            ));
            let frames = frames_of(&req);
            assert_eq!(
                frames.len(),
                5,
                "prefix and head, a payload, the entries between, a payload, the tail"
            );
            assert_identical(&req, &frames);
        }

        #[test]
        fn extra_media_inputs_follow_the_fields_between_with_frames_of_their_own() {
            let req = vllm::GenerateRequest {
                request_id: "extra".into(),
                mm_inputs: Some(vllm::MultimodalInputs::default()),
                extra_mm_inputs: vec![
                    inputs(
                        Some(inline_tensor(1, 16)),
                        [("image_grid_thw", inline_tensor(1, 24))],
                    ),
                    inputs(
                        None,
                        [("video_grid_thw", inline_tensor(2, FRAME_MIN_BYTES + 5))],
                    ),
                    inputs(Some(inline_tensor(3, CLASS_1080P)), []),
                ],
                ec_transfer_params_json: Some("{\"ec\":1}".into()),
                kv_transfer_params_json: Some("{}".into()),
                data_parallel_rank: Some(2),
                media_refs: Some(vllm::MediaRefs::default()),
                chat_template_kwargs_json: Some("{}".into()),
                ..Default::default()
            };
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 5);
            assert_identical(&req, &frames);
        }

        #[test]
        fn shared_memory_and_remote_payloads_stay_in_the_scratch() {
            let req = request(inputs(
                Some(shm_tensor()),
                [
                    ("remote", remote_tensor()),
                    ("pixel_values_2", inline_tensor(4, FRAME_MIN_BYTES)),
                ],
            ));
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 3);
            assert_identical(&req, &frames);
        }

        #[test]
        fn every_field_set_keeps_prosts_field_order_around_both_media_paths() {
            let req = request_with_every_field();
            let frames = frames_of(&req);
            // Two large inline tensors in each of the two media inputs that
            // hold them: prefix and head, then a payload and the bytes after
            // it, four times.
            assert_eq!(frames.len(), 9);
            assert_identical(&req, &frames);
        }

        #[test]
        fn a_plain_request_is_one_frame_of_exactly_its_length() {
            let plain = vllm::GenerateRequest {
                mm_inputs: None,
                extra_mm_inputs: Vec::new(),
                ..request_with_every_field()
            };
            let frames = frames_of(&plain);
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].len(), PREFIX_BYTES + plain.encoded_len());
            assert_identical(&plain, &frames);
        }

        /// A tokenized prompt: ids across the varint widths and the rendered
        /// text.
        fn prompt() -> vllm::generate_request::Input {
            vllm::generate_request::Input::Tokenized(vllm::TokenizedInput {
                original_text: "a rendered conversation ".repeat(175),
                input_ids: (0..1_114u32)
                    .map(|i| i.wrapping_mul(2_654_435_761) % 150_000 + 10)
                    .collect(),
            })
        }

        /// A text-only request that carries a prompt is one frame of prost's
        /// bytes, written by the codec's one-pass writer: on a fresh thread
        /// the writer's scratch holds its packed ids afterwards.
        #[test]
        fn a_text_only_prompt_is_one_frame_through_the_one_pass_writer() {
            std::thread::spawn(|| {
                let req = vllm::GenerateRequest {
                    request_id: "chatcmpl-0123456789abcdef".into(),
                    input: Some(prompt()),
                    sampling_params: Some(vllm::SamplingParams::default()),
                    stream: true,
                    ..Default::default()
                };
                let frames = frames_of(&req);
                assert_eq!(frames.len(), 1);
                assert_eq!(frames[0].len(), PREFIX_BYTES + req.encoded_len());
                assert_identical(&req, &frames);
                let ids = scratch_capacity();
                assert!(ids > 0, "the one-pass writer packed the ids: {ids}");
            })
            .join()
            .expect("the text-only pin holds");
        }

        /// A request with media never reaches the one-pass writer, whatever
        /// carries the media: a payload that rides as a frame of its own, a
        /// small payload inline in the one frame, an extra media input, a
        /// media reference. Each is prost's bytes and the thread's scratch
        /// stays empty: the writer's entry refused the request.
        #[test]
        fn a_request_with_media_never_reaches_the_one_pass_writer() {
            std::thread::spawn(|| {
                let extra = vllm::GenerateRequest {
                    extra_mm_inputs: vec![inputs(Some(inline_tensor(3, CLASS_1080P)), [])],
                    ..Default::default()
                };
                let reference = vllm::GenerateRequest {
                    media_refs: Some(vllm::MediaRefs {
                        items: vec![vllm::MediaRef {
                            modality: 1,
                            url: "data:image/png;base64,iVBORw0KGgo=".into(),
                        }],
                    }),
                    ..Default::default()
                };
                for media in [
                    request(inputs(Some(inline_tensor(1, CLASS_1080P)), [])),
                    request(inputs(Some(inline_tensor(2, FRAME_MIN_BYTES - 1)), [])),
                    extra,
                    reference,
                ] {
                    let req = vllm::GenerateRequest {
                        input: Some(prompt()),
                        ..media
                    };
                    assert_identical(&req, &frames_of(&req));
                }
                assert_eq!(scratch_capacity(), 0);
            })
            .join()
            .expect("the media pin holds");
        }

        #[test]
        fn the_largest_tensor_class_is_one_frame_of_its_own() {
            let req = request(inputs(Some(inline_tensor(5, CLASS_4K)), []));
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 3);
            assert_eq!(frames[1].len(), CLASS_4K);
            assert_identical(&req, &frames);
        }

        #[test]
        fn the_joined_frames_start_with_the_prefix_of_the_whole_message() {
            let req = request_with_every_field();
            let (len, body) = joined(&frames_of(&req));
            assert_eq!(len as usize, body.len());
        }
    }

    /// The SGLang wire: the same matrix over its message, whose tensors sit
    /// in `pixel_values`, in a map and in extra media inputs, whose payload
    /// is a plain bytes field and the tensor's first, and whose raw media
    /// lists stay in the scratch.
    mod sglang_wire {
        use std::collections::HashMap;

        use bytes::Bytes;
        use prost::Message;

        use super::{assert_identical, joined, CLASS_1080P, CLASS_4K};
        use crate::{
            codec::scratch_capacity,
            framed::{sglang_frames, FRAME_MIN_BYTES, PREFIX_BYTES},
            sglang_proto as sg,
        };

        fn inline_tensor(fill: u8, len: usize) -> sg::TensorData {
            sg::TensorData {
                data: Bytes::from(vec![fill; len]),
                shape: vec![1, (len / 4) as u32],
                dtype: "float32".into(),
            }
        }

        /// Media inputs with `pixel_values` and the named model-specific
        /// tensors, nothing else.
        fn inputs(
            pixel_values: Option<sg::TensorData>,
            tensors: impl IntoIterator<Item = (&'static str, sg::TensorData)>,
        ) -> sg::MultimodalInputs {
            sg::MultimodalInputs {
                pixel_values,
                model_specific_tensors: tensors
                    .into_iter()
                    .map(|(key, tensor)| (key.to_string(), tensor))
                    .collect(),
                ..Default::default()
            }
        }

        fn request(mm_inputs: sg::MultimodalInputs) -> sg::GenerateRequest {
            sg::GenerateRequest {
                request_id: "r".into(),
                mm_inputs: Some(Box::new(mm_inputs)),
                ..Default::default()
            }
        }

        /// Every field of the media inputs set: the three reference lists, a
        /// large and a small inline tensor, the three raw media lists, an
        /// empty key and a default value in the map, the placeholders.
        fn inputs_with_every_field(fill: u8) -> sg::MultimodalInputs {
            sg::MultimodalInputs {
                image_urls: vec!["image-1".into()],
                video_urls: vec!["video-1".into()],
                audio_urls: vec!["audio-1".into()],
                pixel_values: Some(inline_tensor(fill, FRAME_MIN_BYTES + 3)),
                image_data: vec![vec![fill; 3]],
                video_data: vec![vec![fill; 2]],
                audio_data: vec![vec![fill; 1]],
                modalities: vec!["image".into()],
                model_specific_tensors: HashMap::from([
                    ("image_grid_thw".to_string(), inline_tensor(fill, 24)),
                    (String::new(), inline_tensor(fill, FRAME_MIN_BYTES)),
                    ("empty".to_string(), sg::TensorData::default()),
                ]),
                im_token_id: Some(151_655),
                mm_placeholders: vec![sg::PlaceholderRange {
                    offset: 3,
                    length: 64,
                }],
                feature_hash: Some(7),
            }
        }

        /// Every field of the request set, the seventeen fields between the
        /// two media paths included: the proof that the frames keep prost's
        /// order around both.
        fn request_with_every_field() -> sg::GenerateRequest {
            sg::GenerateRequest {
                request_id: "every-field".into(),
                tokenized: Some(sg::TokenizedInput {
                    original_text: "hi".into(),
                    input_ids: vec![1, 2],
                }),
                mm_inputs: Some(Box::new(inputs_with_every_field(1))),
                sampling_params: Some(sg::SamplingParams::default()),
                return_logprob: true,
                logprob_start_len: 2,
                top_logprobs_num: 5,
                token_ids_logprob: vec![11, 12],
                return_hidden_states: true,
                disaggregated_params: Some(sg::DisaggregatedParams {
                    bootstrap_host: "decode-0".into(),
                    bootstrap_port: 8998,
                    bootstrap_room: 42,
                }),
                custom_logit_processor: "processor".into(),
                timestamp: Some(prost_types::Timestamp {
                    seconds: 1,
                    nanos: 2,
                }),
                log_metrics: true,
                input_embeds: vec![0.5, 1.5],
                lora_id: "lora".into(),
                data_parallel_rank: Some(1),
                stream: true,
                require_reasoning: true,
                return_media_identity: true,
                media_identity: Some(Box::new(sg::MediaIdentity {
                    version: 1,
                    ..Default::default()
                })),
                extra_mm_inputs: vec![inputs_with_every_field(2), sg::MultimodalInputs::default()],
            }
        }

        /// The frames of `req`, which stays with the caller for the comparison.
        fn frames_of(req: &sg::GenerateRequest) -> Vec<Bytes> {
            sglang_frames(req.clone()).expect("a request the prefix can carry")
        }

        #[test]
        fn a_request_without_media_is_one_frame_of_the_plain_encoding() {
            let req = sg::GenerateRequest {
                request_id: "text".into(),
                stream: true,
                ..Default::default()
            };
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 1);
            assert_identical(&req, &frames);
        }

        #[test]
        fn payloads_below_the_frame_size_stay_inline_in_one_frame() {
            let mut req = request(inputs(
                Some(inline_tensor(1, 10)),
                [("image_grid_thw", inline_tensor(2, FRAME_MIN_BYTES - 1))],
            ));
            req.extra_mm_inputs = vec![inputs(Some(inline_tensor(3, 16)), [])];
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].len(), PREFIX_BYTES + req.encoded_len());
            assert_identical(&req, &frames);
        }

        #[test]
        fn large_pixel_values_ride_as_their_own_frame_led_by_their_key() {
            let req = request(inputs(Some(inline_tensor(1, CLASS_1080P)), []));
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 3, "prefix and head, the payload, the tail");
            let pixel_values = req
                .mm_inputs
                .as_ref()
                .unwrap()
                .pixel_values
                .as_ref()
                .unwrap();
            assert_eq!(
                frames[1].as_ptr(),
                pixel_values.data.as_ptr(),
                "the payload frame is the payload's memory"
            );
            assert_eq!(frames[1].len(), CLASS_1080P);
            assert_eq!(
                frames[2][0], 0x12,
                "the tensor's shape (field 2) follows its payload (field 1)"
            );
            assert_identical(&req, &frames);
        }

        #[test]
        fn map_entries_keep_the_maps_order_around_an_empty_key_and_a_default_value() {
            let req = request(inputs(
                None,
                [
                    ("", inline_tensor(1, FRAME_MIN_BYTES + 1)),
                    ("empty", sg::TensorData::default()),
                    ("image_grid_thw", inline_tensor(2, 24)),
                    ("second_image", inline_tensor(3, FRAME_MIN_BYTES)),
                ],
            ));
            let frames = frames_of(&req);
            assert_eq!(
                frames.len(),
                5,
                "prefix and head, a payload, the entries between, a payload, the tail"
            );
            assert_identical(&req, &frames);
        }

        #[test]
        fn extra_media_inputs_follow_the_fields_between_with_frames_of_their_own() {
            let req = sg::GenerateRequest {
                request_id: "extra".into(),
                mm_inputs: Some(Box::default()),
                sampling_params: Some(sg::SamplingParams::default()),
                disaggregated_params: Some(sg::DisaggregatedParams::default()),
                timestamp: Some(prost_types::Timestamp {
                    seconds: 3,
                    nanos: 4,
                }),
                lora_id: "lora".into(),
                data_parallel_rank: Some(2),
                media_identity: Some(Box::default()),
                extra_mm_inputs: vec![
                    inputs(
                        Some(inline_tensor(1, 16)),
                        [("image_grid_thw", inline_tensor(1, 24))],
                    ),
                    inputs(
                        None,
                        [("video_grid_thw", inline_tensor(2, FRAME_MIN_BYTES + 5))],
                    ),
                    inputs(Some(inline_tensor(3, CLASS_1080P)), []),
                ],
                ..Default::default()
            };
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 5);
            assert_identical(&req, &frames);
        }

        #[test]
        fn raw_media_lists_stay_in_the_scratch_beside_a_framed_tensor() {
            let mut media = inputs(Some(inline_tensor(1, FRAME_MIN_BYTES)), []);
            media.image_data = vec![vec![9; 100 * 1024]];
            media.modalities = vec!["image".into()];
            let req = request(media);
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 3);
            assert!(
                frames[2].len() > 100 * 1024,
                "the raw image bytes are in the tail, not a frame"
            );
            assert_identical(&req, &frames);
        }

        #[test]
        fn every_field_set_keeps_prosts_field_order_around_both_media_paths() {
            let req = request_with_every_field();
            let frames = frames_of(&req);
            // Two large inline tensors in each of the two media inputs that
            // hold them: prefix and head, then a payload and the bytes after
            // it, four times.
            assert_eq!(frames.len(), 9);
            assert_identical(&req, &frames);
        }

        #[test]
        fn a_plain_request_is_one_frame_of_exactly_its_length() {
            let plain = sg::GenerateRequest {
                mm_inputs: None,
                extra_mm_inputs: Vec::new(),
                ..request_with_every_field()
            };
            let frames = frames_of(&plain);
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].len(), PREFIX_BYTES + plain.encoded_len());
            assert_identical(&plain, &frames);
        }

        /// A tokenized prompt: ids across the varint widths and the rendered
        /// text.
        fn prompt() -> sg::TokenizedInput {
            sg::TokenizedInput {
                original_text: "a rendered conversation ".repeat(175),
                input_ids: (0..1_114u32)
                    .map(|i| i.wrapping_mul(2_654_435_761) % 150_000 + 10)
                    .collect(),
            }
        }

        /// A text-only request that carries a prompt is one frame of prost's
        /// bytes, written by the codec's one-pass writer: on a fresh thread
        /// the writer's scratch holds its packed ids afterwards.
        #[test]
        fn a_text_only_prompt_is_one_frame_through_the_one_pass_writer() {
            std::thread::spawn(|| {
                let req = sg::GenerateRequest {
                    request_id: "chatcmpl-0123456789abcdef".into(),
                    tokenized: Some(prompt()),
                    sampling_params: Some(sg::SamplingParams::default()),
                    stream: true,
                    ..Default::default()
                };
                let frames = frames_of(&req);
                assert_eq!(frames.len(), 1);
                assert_eq!(frames[0].len(), PREFIX_BYTES + req.encoded_len());
                assert_identical(&req, &frames);
                let ids = scratch_capacity();
                assert!(ids > 0, "the one-pass writer packed the ids: {ids}");
            })
            .join()
            .expect("the text-only pin holds");
        }

        /// A request with media never reaches the one-pass writer, whatever
        /// carries the media: a payload that rides as a frame of its own, a
        /// small payload inline in the one frame, an extra media input of
        /// either size, raw media bytes, a media reference. Each is prost's
        /// bytes and the thread's scratch stays empty: the writer's entry
        /// refused the request.
        #[test]
        fn a_request_with_media_never_reaches_the_one_pass_writer() {
            std::thread::spawn(|| {
                let extra_framed = sg::GenerateRequest {
                    extra_mm_inputs: vec![inputs(Some(inline_tensor(3, CLASS_1080P)), [])],
                    ..Default::default()
                };
                let extra_inline = sg::GenerateRequest {
                    extra_mm_inputs: vec![inputs(Some(inline_tensor(4, 16)), [])],
                    ..Default::default()
                };
                let mut raw = inputs(None, []);
                raw.image_data = vec![vec![9; 1024]];
                let mut reference = inputs(None, []);
                reference.image_urls = vec!["data:image/png;base64,iVBORw0KGgo=".into()];
                for media in [
                    request(inputs(Some(inline_tensor(1, CLASS_1080P)), [])),
                    request(inputs(Some(inline_tensor(2, FRAME_MIN_BYTES - 1)), [])),
                    extra_framed,
                    extra_inline,
                    request(raw),
                    request(reference),
                ] {
                    let req = sg::GenerateRequest {
                        tokenized: Some(prompt()),
                        ..media
                    };
                    assert_identical(&req, &frames_of(&req));
                }
                assert_eq!(scratch_capacity(), 0);
            })
            .join()
            .expect("the media pin holds");
        }

        #[test]
        fn the_largest_tensor_class_is_one_frame_of_its_own() {
            let req = request(inputs(Some(inline_tensor(5, CLASS_4K)), []));
            let frames = frames_of(&req);
            assert_eq!(frames.len(), 3);
            assert_eq!(frames[1].len(), CLASS_4K);
            assert_identical(&req, &frames);
        }

        #[test]
        fn the_joined_frames_start_with_the_prefix_of_the_whole_message() {
            let req = request_with_every_field();
            let (len, body) = joined(&frames_of(&req));
            assert_eq!(len as usize, body.len());
        }
    }
}
