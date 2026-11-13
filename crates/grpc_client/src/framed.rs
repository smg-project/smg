//! A TokenSpeed `GenerateRequest` sent as HTTP/2 data frames: the message's
//! small fields in one scratch frame, every inline encoder input of at least
//! `FRAME_MIN_BYTES` as a frame of its own, shared with the request and
//! never copied. prost's one-buffer encoding copies a large image tensor
//! into a buffer of the whole message's size on every call; above the
//! allocator's oversize threshold that copy is a fresh mapping and its page
//! faults, paid per request.
//!
//! The body is the generated encoding byte for byte. prost writes a
//! message's fields in field-number order, so each message on the path to
//! the tensor is written in three parts: its fields before the path's
//! field, encoded by prost from a copy that holds only those; the path's
//! field key and length, written here with the length prost computes for
//! the sub-message; and its fields after the path's field, encoded by prost
//! from a copy that holds only those. The path mirrors four field numbers of
//! the TokenSpeed scheduler proto: `mm_inputs` 9 of `GenerateRequest`,
//! `items` 1 of `MultimodalInputs`, `encoder_input` 3 of `MultimodalItem`,
//! `inline` 3 of `TensorData` (its payload oneof is the message's last
//! field, so a tensor's other fields all come before it). Every message on
//! the path is taken apart by an exhaustive pattern, so a field added to any
//! of them stops this module from compiling until the layout names it; the
//! tests compare the joined frames with `encode_to_vec` over a matrix of
//! requests and decode them back. Only the item's primary encoder input
//! rides as a frame; a request without one is one frame of the plain
//! encoding. The request is consumed: its fields move into the partial
//! messages prost encodes, so the caller keeps no tensor once the frames
//! exist; the length prefix is checked against the four-byte gRPC limit as
//! tonic's encoder checks it. Nothing in this module is `unsafe`, and
//! nothing in it is public: the hand-mirrored layout stays inside the crate.

use std::future::poll_fn;

use bytes::{BufMut, Bytes, BytesMut};
use http_body_util::StreamBody;
use prost::{
    encoding::{encode_key, encode_varint, WireType},
    Message,
};
use tonic::{
    client::GrpcService,
    codec::{BufferSettings, Streaming},
    codegen::http::{self, header::HeaderValue, Method, Request as HttpRequest, Version},
    Status,
};
use tonic_prost::ProstCodec;

use crate::tokenspeed_scheduler::tokenspeed_proto as ts;

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
    /// A builder whose scratch holds `scratch_bytes` without growing: the
    /// bytes of the message that do not ride as payload frames.
    fn new(scratch_bytes: usize) -> Self {
        Self {
            scratch: BytesMut::with_capacity(scratch_bytes),
            frames: Vec::with_capacity(8),
        }
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

/// The body frames of the gRPC message for a TokenSpeed `GenerateRequest`:
/// the 5-byte length prefix and the message, every inline encoder input of at
/// least `FRAME_MIN_BYTES` as a frame of its own. The joined frames are the
/// message's generated encoding byte for byte. The request is consumed; a
/// message longer than the four-byte prefix can carry is refused as tonic's
/// encoder refuses it.
pub(crate) fn tokenspeed_frames(req: ts::GenerateRequest) -> Result<Vec<Bytes>, Status> {
    let len = req.encoded_len();
    let prefix = u32::try_from(len).map_err(|_| {
        Status::resource_exhausted(format!(
            "Cannot return body with more than 4GB of data: {len} bytes"
        ))
    })?;
    let framed_bytes: usize = req
        .mm_inputs
        .as_ref()
        .map_or(0, |mm| mm.items.iter().filter_map(framed_payload_len).sum());
    // The scratch takes every byte that is not a payload frame, exactly.
    let mut out = Frames::new(PREFIX_BYTES + len - framed_bytes);
    out.scratch.put_u8(0);
    out.scratch.put_u32(prefix);
    if framed_bytes == 0 {
        req.encode_raw(&mut out.scratch);
        return Ok(out.finish());
    }
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

/// The unary-streaming call with `frames` as the request body over the
/// client's own channel: the request headers (the trace metadata among them)
/// and the gRPC content type, the response read as `R` messages. Transport
/// errors map through tonic's `Status::from_error`, as the generated client
/// maps them.
pub(crate) async fn call_with_frames<R>(
    channel: &crate::channel::Channel,
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
    use crate::common_proto::ShmHandle;

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
    fn assert_identical(req: &ts::GenerateRequest, frames: &[Bytes]) {
        let (len, body) = joined(frames);
        assert_eq!(len as usize, req.encoded_len(), "the length prefix");
        assert!(
            body == req.encode_to_vec(),
            "the body is not prost's encoding byte for byte"
        );
        let decoded = ts::GenerateRequest::decode(body.as_slice()).expect("the body decodes");
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
}
