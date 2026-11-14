//! The gRPC codec of the engine clients: prost's, with two writes of its own
//! for the request shapes the gateway sends most.
//!
//! A message that carries inline tensors or media references has the encode
//! buffer reserved to its encoded length before prost writes it, so a
//! multi-megabyte payload lands once in one allocation instead of being
//! copied again at every doubling of tonic's growing `BytesMut`. The length
//! walk is a few fields' worth on such a message, next to its payload.
//!
//! A text-only request that carries a prompt is written in one pass. prost
//! walks a prompt's packed ids three times per request (the prompt message's
//! length for its prefix, the packed field's length for its prefix, then the
//! varints a byte at a time), about nine instructions per id per walk. Here
//! the request id, the prompt text and the ids are taken out of the owned
//! request. The ids get one length walk and one tight pass into a per-thread
//! scratch. The writer puts the request id and the prompt's text and ids in
//! prost's field order into a buffer reserved to the frame's length (one
//! walk of the few other fields gives it), the text copied once, straight
//! from the request; prost writes the prompt message's other fields (none
//! on most wires) and then the request's behind them, straight into the
//! buffer. The frame is byte for byte prost's. The scratch holds the packed
//! ids alone and keeps the size of the largest prompt it packed, up to a
//! cap; neither the text nor any other field (a structured-output
//! constraint may be many kilobytes) passes through it. A
//! request with media never reaches the writer: its one entry refuses it by
//! the request's own fields, never by a size. A generate call that goes out
//! as body frames (the crate's `framed` module) does not pass through this
//! codec; its request that is one frame is written by the same rule, a
//! text-only prompt by this same writer into the frame itself.
//!
//! Every other message is reserved and written as before, a few fields'
//! worth to walk. A generate request without a tokenized prompt (a text
//! input) and an embedding request are prost's `encode_raw` without the
//! reservation: `encode` would only walk the message's length for a
//! capacity check that cannot fail against a buffer that grows on demand.
//! An embedding request carries the same prompt message; the writer has no
//! arm for it, so its ids keep prost's three walks. Decoding is prost's own
//! decoder, unchanged. tonic builds the encode buffer itself; the tests
//! below drive the codec through tonic's own request body, as a client call
//! does.

use std::{any::Any, cell::RefCell, marker::PhantomData};

use prost::{
    bytes::{BufMut, BytesMut},
    encoding::{encode_key, encode_varint, encoded_len_varint, key_len, WireType},
    Message,
};
use tonic::{
    codec::{BufferSettings, Codec, EncodeBuf, Encoder},
    Status,
};
use tonic_prost::{ProstCodec, ProstDecoder};

use crate::{mlx_proto, sglang_proto, tokenspeed_proto, trtllm_proto, vllm_proto};

/// tonic's eager encode buffer (`BufferSettings::default()`, whose size is
/// private): the capacity the codec's unreserved write starts from, and the
/// framed single-frame path with it, so a plain request grows neither.
pub(crate) const EAGER_BUFFER_BYTES: usize = 8 * 1024;

/// A [`Codec`] for `T` requests and `U` responses: prost's bytes, the encode
/// buffer reserved to the encoded length of a message carrying inline tensors
/// before prost writes it, and a text-only request that carries a prompt
/// written by the one-pass writer, which reserves its frame's length.
#[derive(Debug, Clone)]
pub struct SizedProstCodec<T, U> {
    _pd: PhantomData<(T, U)>,
}

impl<T, U> Default for SizedProstCodec<T, U> {
    fn default() -> Self {
        Self { _pd: PhantomData }
    }
}

impl<T, U> Codec for SizedProstCodec<T, U>
where
    T: Message + Send + 'static,
    U: Message + Default + Send + 'static,
{
    type Encode = T;
    type Decode = U;
    type Encoder = SizedProstEncoder<T>;
    type Decoder = ProstDecoder<U>;

    fn encoder(&mut self) -> Self::Encoder {
        SizedProstEncoder { _pd: PhantomData }
    }

    fn decoder(&mut self) -> Self::Decoder {
        ProstCodec::<T, U>::raw_decoder(BufferSettings::default())
    }
}

/// prost's encoding: written by prost into a buffer reserved to the message's
/// length first when the message carries inline tensors; by the one-pass
/// writer, which reserves the frame for one walk of the request's other
/// fields, when it is a text-only request that carries a prompt; by prost
/// into the growing buffer otherwise.
#[derive(Debug, Clone)]
pub struct SizedProstEncoder<T> {
    _pd: PhantomData<T>,
}

impl<T: Message + 'static> Encoder for SizedProstEncoder<T> {
    type Item = T;
    type Error = Status;

    fn encode(&mut self, mut item: Self::Item, buf: &mut EncodeBuf<'_>) -> Result<(), Self::Error> {
        if reserves_for(&item) {
            buf.reserve(item.encoded_len());
        }
        // A text-only request that carries a prompt is the one-pass writer's;
        // its entry refuses every other message, which is prost's write. The
        // buffer is `BytesMut`-backed and grows on demand, so the write
        // cannot run out of room: prost's `encode` would only walk the
        // message's length to tell us so.
        if !encode_prompt_carrier(&mut item, buf) {
            item.encode_raw(buf);
        }
        Ok(())
    }

    fn buffer_settings(&self) -> BufferSettings {
        BufferSettings::default()
    }
}

/// A buffer a request is written into: tonic's encode buffer on a codec
/// call, the frame itself on a call sent as body frames.
pub(crate) trait ReserveBuf: BufMut {
    /// Room for `additional` more bytes, taken in one growth.
    fn reserve(&mut self, additional: usize);
}

impl ReserveBuf for EncodeBuf<'_> {
    fn reserve(&mut self, additional: usize) {
        EncodeBuf::reserve(self, additional);
    }
}

impl ReserveBuf for BytesMut {
    fn reserve(&mut self, additional: usize) {
        BytesMut::reserve(self, additional);
    }
}

thread_local! {
    /// The one-pass writer's scratch: the ids' varint bytes, kept per thread at
    /// the size of the largest prompt it packed, up to [`SCRATCH_CAP`]. Nothing
    /// else of a request enters it: the writer copies the prompt text from the
    /// request into the frame, and prost writes every other field there.
    static PACKED: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// The capacity a thread keeps of the scratch between writes. Grown past it
/// by one outsized prompt, the scratch is released after that write, so a
/// thread holds at most this much for the requests that follow and nothing
/// of the outsized one.
const SCRATCH_CAP: usize = 1 << 20;

/// The scratch's capacity on this thread: the tests here and the framed
/// path's read from it whether a request went through the one-pass writer.
#[cfg(test)]
pub(crate) fn scratch_capacity() -> usize {
    PACKED.with(|packed| packed.borrow().capacity())
}

/// prost's encoded length of a string field: nothing when empty.
fn string_field_len(tag: u32, value: &str) -> usize {
    if value.is_empty() {
        0
    } else {
        key_len(tag) + encoded_len_varint(value.len() as u64) + value.len()
    }
}

/// The ids' packed bytes: one walk for the length, one pass for the bytes.
fn pack_ids(ids: &[u32], packed: &mut Vec<u8>) {
    let len: usize = ids
        .iter()
        .map(|&id| encoded_len_varint(u64::from(id)))
        .sum();
    packed.clear();
    packed.resize(len, 0);
    let mut at = 0;
    for &id in ids {
        let mut value = id;
        while value >= 0x80 {
            packed[at] = (value as u8) | 0x80;
            value >>= 7;
            at += 1;
        }
        packed[at] = value as u8;
        at += 1;
    }
    debug_assert_eq!(at, len);
}

/// A request's head, taken out of it for the one-pass write: its request
/// id, the prompt message without its text and its ids, and the two with
/// their field tags. On every wire the message's other fields have higher
/// tags than both; the byte-identity cases pin it.
struct Prompt<'a, P> {
    request_id: &'a str,
    message: &'a P,
    text_tag: u32,
    text: &'a str,
    ids_tag: u32,
    ids: &'a [u32],
}

/// A string field as prost writes it: nothing when it is empty.
fn put_string(tag: u32, value: &str, buf: &mut impl BufMut) {
    if !value.is_empty() {
        encode_key(tag, WireType::LengthDelimited, buf);
        encode_varint(value.len() as u64, buf);
        buf.put_slice(value.as_bytes());
    }
}

/// The packed ids as prost writes the field: nothing when there are none.
fn put_ids(tag: u32, packed: &[u8], buf: &mut impl BufMut) {
    if !packed.is_empty() {
        encode_key(tag, WireType::LengthDelimited, buf);
        encode_varint(packed.len() as u64, buf);
        buf.put_slice(packed);
    }
}

/// Write `request`, its request id and its prompt message taken out, with
/// both put back where prost writes them: the request id is field
/// `REQUEST_ID_TAG`, the prompt message field `PROMPT_TAG` behind it, and
/// every other field of the request has a higher tag, so prost writes that
/// remainder behind them, straight into the buffer. Inside the prompt
/// message the text and the ids come in tag order, then the message's other
/// fields, which prost writes in place as well. The bytes are prost's own;
/// the ids cost one length walk and one write instead of three walks and a
/// byte at a time; the text and every other field are each copied once, into
/// the frame. The scratch keeps its size for the next request, up to
/// [`SCRATCH_CAP`].
fn encode_prompt_request<R: Message, P: Message, B: ReserveBuf>(
    request: &R,
    prompt: Prompt<'_, P>,
    buf: &mut B,
) {
    let id_field = string_field_len(REQUEST_ID_TAG, prompt.request_id);
    let text_field = string_field_len(prompt.text_tag, prompt.text);
    let prompt_rest = prompt.message.encoded_len();
    PACKED.with(|packed| {
        let mut packed = packed.borrow_mut();
        pack_ids(prompt.ids, &mut packed);
        let ids_field = if packed.is_empty() {
            0
        } else {
            key_len(prompt.ids_tag) + encoded_len_varint(packed.len() as u64) + packed.len()
        };
        let prompt_len = text_field + ids_field + prompt_rest;
        let prompt_field = key_len(PROMPT_TAG) + encoded_len_varint(prompt_len as u64) + prompt_len;
        // The frame's length, for one walk of the few other fields of the
        // request and of its prompt message: reserved, the buffer grows at
        // most once under the puts.
        buf.reserve(id_field + prompt_field + request.encoded_len());
        put_string(REQUEST_ID_TAG, prompt.request_id, buf);
        encode_key(PROMPT_TAG, WireType::LengthDelimited, buf);
        encode_varint(prompt_len as u64, buf);
        if prompt.text_tag < prompt.ids_tag {
            put_string(prompt.text_tag, prompt.text, buf);
            put_ids(prompt.ids_tag, &packed, buf);
        } else {
            put_ids(prompt.ids_tag, &packed, buf);
            put_string(prompt.text_tag, prompt.text, buf);
        }
        if packed.capacity() > SCRATCH_CAP {
            *packed = Vec::new();
        }
    });
    // The prompt message's other fields, then the request's, as prost orders
    // them: each has a higher tag than what was put by hand before it.
    prompt.message.encode_raw(buf);
    request.encode_raw(buf);
}

/// The request id's tag on every wire.
const REQUEST_ID_TAG: u32 = 1;

/// The prompt message's tag on every wire (`tokenized = 2`, behind
/// `request_id = 1`).
const PROMPT_TAG: u32 = 2;

/// Write a text-only request that carries a prompt through
/// [`encode_prompt_request`]; `false` for any other message, left as it was.
/// On `true` the item is written and is left without its request id and its
/// prompt, which the writer took: it must not be written again. The one
/// entry to the one-pass writer, for its two callers: the codec's encoder
/// with tonic's buffer, and the framed path's single-frame write with the
/// frame itself. A message with inline tensors or media references is
/// refused here, by its fields ([`reserves_for`]) and never by a size, so no
/// caller can send one through: prost writes that message once into a
/// reserved buffer, or its large payloads ride as frames of their own, and
/// nothing of it reaches the scratch. Each wire names its own prompt
/// message and the tags of its text and its ids; the request id is taken
/// only once the prompt is known to be tokenized.
pub(crate) fn encode_prompt_carrier<T: 'static, B: ReserveBuf>(item: &mut T, buf: &mut B) -> bool {
    if reserves_for(&*item) {
        return false;
    }
    let item: &mut dyn Any = item;
    if let Some(request) = item.downcast_mut::<vllm_proto::GenerateRequest>() {
        use vllm_proto::generate_request::Input;
        let mut prompt = match request.input.take() {
            Some(Input::Tokenized(prompt)) => prompt,
            other => {
                request.input = other;
                return false;
            }
        };
        let ids = std::mem::take(&mut prompt.input_ids);
        let text = std::mem::take(&mut prompt.original_text);
        let request_id = std::mem::take(&mut request.request_id);
        encode_prompt_request(
            request,
            Prompt {
                request_id: &request_id,
                message: &prompt,
                text_tag: 1,
                text: &text,
                ids_tag: 2,
                ids: &ids,
            },
            buf,
        );
        return true;
    }
    if let Some(request) = item.downcast_mut::<mlx_proto::GenerateRequest>() {
        use mlx_proto::generate_request::Input;
        let mut prompt = match request.input.take() {
            Some(Input::Tokenized(prompt)) => prompt,
            other => {
                request.input = other;
                return false;
            }
        };
        let ids = std::mem::take(&mut prompt.input_ids);
        let text = std::mem::take(&mut prompt.original_text);
        let request_id = std::mem::take(&mut request.request_id);
        encode_prompt_request(
            request,
            Prompt {
                request_id: &request_id,
                message: &prompt,
                text_tag: 1,
                text: &text,
                ids_tag: 2,
                ids: &ids,
            },
            buf,
        );
        return true;
    }
    if let Some(request) = item.downcast_mut::<sglang_proto::GenerateRequest>() {
        let Some(mut prompt) = request.tokenized.take() else {
            return false;
        };
        let ids = std::mem::take(&mut prompt.input_ids);
        let text = std::mem::take(&mut prompt.original_text);
        let request_id = std::mem::take(&mut request.request_id);
        encode_prompt_request(
            request,
            Prompt {
                request_id: &request_id,
                message: &prompt,
                text_tag: 1,
                text: &text,
                ids_tag: 2,
                ids: &ids,
            },
            buf,
        );
        return true;
    }
    if let Some(request) = item.downcast_mut::<tokenspeed_proto::GenerateRequest>() {
        let Some(mut prompt) = request.tokenized.take() else {
            return false;
        };
        let ids = std::mem::take(&mut prompt.input_ids);
        let text = std::mem::take(&mut prompt.original_text);
        let request_id = std::mem::take(&mut request.request_id);
        // The ids are the prompt message's first field, the text its second.
        encode_prompt_request(
            request,
            Prompt {
                request_id: &request_id,
                message: &prompt,
                text_tag: 2,
                text: &text,
                ids_tag: 1,
                ids: &ids,
            },
            buf,
        );
        return true;
    }
    if let Some(request) = item.downcast_mut::<trtllm_proto::GenerateRequest>() {
        let Some(mut prompt) = request.tokenized.take() else {
            return false;
        };
        let ids = std::mem::take(&mut prompt.input_token_ids);
        let text = std::mem::take(&mut prompt.original_text);
        let request_id = std::mem::take(&mut request.request_id);
        // The query ids, field 3, stay in the message: prost writes them last.
        encode_prompt_request(
            request,
            Prompt {
                request_id: &request_id,
                message: &prompt,
                text_tag: 1,
                text: &text,
                ids_tag: 2,
                ids: &ids,
            },
            buf,
        );
        return true;
    }
    false
}

/// Whether `item` is written by prost into a buffer reserved to its length.
/// A request that carries a prompt is when it also carries multimodal inputs
/// or media references (a media reference may be a data URL of many
/// megabytes), the shapes whose inline payloads the reservation is for and
/// that the one-pass writer's entry refuses; every other message reserves as
/// before. Each caller is instantiated per message type, so each type check
/// here folds at compile time and a text request pays one field read. The
/// framed path's single-frame write reads it too, for a request none of
/// whose payloads rides as a frame of its own.
pub(crate) fn reserves_for<T: 'static>(item: &T) -> bool {
    let item: &dyn Any = item;
    if let Some(request) = item.downcast_ref::<tokenspeed_proto::GenerateRequest>() {
        return request.mm_inputs.is_some();
    }
    if let Some(request) = item.downcast_ref::<vllm_proto::GenerateRequest>() {
        return request.mm_inputs.is_some()
            || !request.extra_mm_inputs.is_empty()
            || request.media_refs.is_some();
    }
    if let Some(request) = item.downcast_ref::<sglang_proto::GenerateRequest>() {
        return request.mm_inputs.is_some() || !request.extra_mm_inputs.is_empty();
    }
    if let Some(request) = item.downcast_ref::<sglang_proto::EmbedRequest>() {
        return request.mm_inputs.is_some();
    }
    if let Some(request) = item.downcast_ref::<trtllm_proto::GenerateRequest>() {
        return request.multimodal_input.is_some();
    }
    // The prompt-carrying requests whose wire has no multimodal field.
    !(item.is::<vllm_proto::EmbedRequest>()
        || item.is::<trtllm_proto::EmbedRequest>()
        || item.is::<mlx_proto::GenerateRequest>())
}

#[cfg(test)]
mod tests {
    use std::{
        pin::pin,
        task::{Context, Poll},
    };

    use futures::{future::ready, stream};
    use prost::bytes::Bytes;
    use tonic::{codec::EncodeBody, codegen::Body};

    use super::*;
    use crate::{
        mlx_proto, sglang_proto, tokenspeed_encoder_proto, tokenspeed_proto,
        tokenspeed_scheduler::tokenspeed_proto::{tensor_data::Payload, TensorData},
        trtllm_proto, vllm_proto,
    };

    fn tensor() -> TensorData {
        TensorData {
            shape: vec![2, 2],
            dtype: "bfloat16".to_string(),
            payload: Some(Payload::Inline(Bytes::from(vec![1u8, 2, 3, 4, 5, 6, 7, 8]))),
        }
    }

    /// The protobuf bytes of a tensor whose payload is `Bytes`: the packed
    /// shape, the dtype string and the payload as a length-delimited field,
    /// which is what every reader of the `bytes` field expects.
    #[test]
    fn a_bytes_payload_encodes_as_the_bytes_field() {
        let mut expected = vec![0x0a, 2, 2, 2];
        expected.extend_from_slice(&[0x12, 8]);
        expected.extend_from_slice(b"bfloat16");
        expected.extend_from_slice(&[0x1a, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(tensor().encode_to_vec(), expected);
    }

    /// What tonic puts on the wire for one message through `codec`: the data
    /// frames of the client request body it builds, in order.
    fn wire_bytes<C, T>(mut codec: C, message: T) -> Vec<u8>
    where
        C: Codec<Encode = T>,
        T: Send + 'static,
    {
        let source = stream::once(ready(Ok::<T, Status>(message)));
        let mut body = pin!(EncodeBody::new_client(codec.encoder(), source, None, None));
        let mut cx = Context::from_waker(std::task::Waker::noop());
        let mut bytes = Vec::new();
        loop {
            match body.as_mut().poll_frame(&mut cx) {
                Poll::Ready(Some(Ok(frame))) => {
                    if let Ok(data) = frame.into_data() {
                        bytes.extend_from_slice(&data);
                    }
                }
                Poll::Ready(Some(Err(status))) => panic!("the request body failed: {status}"),
                Poll::Ready(None) => return bytes,
                Poll::Pending => panic!("a ready source never pends"),
            }
        }
    }

    /// The gRPC frame of `message` as prost's own encoder writes it: the
    /// five-byte header (no compression, the big-endian length) and the
    /// message bytes.
    fn prost_frame<T: Message>(message: &T) -> Vec<u8> {
        let payload = message.encode_to_vec();
        let mut frame = vec![0u8];
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(&payload);
        frame
    }

    /// The codec's frame for `message` is byte for byte prost's, and byte for
    /// byte the frame tonic's stock prost codec writes.
    fn frames_as_prost_does<T>(message: T)
    where
        T: Message + Clone + Default + Send + 'static,
    {
        let expected = prost_frame(&message);
        assert_eq!(
            wire_bytes(SizedProstCodec::<T, T>::default(), message.clone()),
            expected
        );
        assert_eq!(wire_bytes(ProstCodec::<T, T>::default(), message), expected);
    }

    fn prompt_ids(count: usize) -> Vec<u32> {
        (0..count as u32)
            .map(|i| i.wrapping_mul(2_654_435_761) % 150_000 + 10)
            .collect()
    }

    fn prompt_text(bytes: usize) -> String {
        "a rendered conversation ".repeat(bytes / 24 + 1)[..bytes].to_string()
    }

    fn inline_payload(bytes: usize) -> Bytes {
        Bytes::from(vec![9u8; bytes])
    }

    /// Multimodal inputs on each wire that carries inline tensors: one
    /// float32 tensor of `bytes` bytes as the payload.
    fn tokenspeed_multimodal(bytes: usize) -> tokenspeed_proto::MultimodalInputs {
        tokenspeed_proto::MultimodalInputs {
            items: vec![tokenspeed_proto::MultimodalItem {
                encoder_input: Some(TensorData {
                    shape: vec![1, 1, bytes as u32 / 4],
                    dtype: "float32".to_string(),
                    payload: Some(Payload::Inline(inline_payload(bytes))),
                }),
                ..Default::default()
            }],
        }
    }

    fn vllm_multimodal(bytes: usize) -> vllm_proto::MultimodalInputs {
        vllm_proto::MultimodalInputs {
            pixel_values: Some(vllm_proto::TensorData {
                shape: vec![1, 1, bytes as u32 / 4],
                dtype: "float32".to_string(),
                payload: Some(vllm_proto::tensor_data::Payload::Inline(inline_payload(
                    bytes,
                ))),
            }),
            ..Default::default()
        }
    }

    fn sglang_multimodal(bytes: usize) -> sglang_proto::MultimodalInputs {
        sglang_proto::MultimodalInputs {
            pixel_values: Some(sglang_proto::TensorData {
                data: inline_payload(bytes),
                shape: vec![1, 1, bytes as u32 / 4],
                dtype: "float32".to_string(),
            }),
            ..Default::default()
        }
    }

    /// The reservation follows the message's shape: a request that carries a
    /// prompt reserves only when it also carries multimodal inputs or media
    /// references; every other message reserves.
    #[test]
    fn the_reservation_follows_the_message_shape() {
        assert!(!reserves_for(&tokenspeed_proto::GenerateRequest::default()));
        assert!(!reserves_for(&vllm_proto::GenerateRequest::default()));
        assert!(!reserves_for(&sglang_proto::GenerateRequest::default()));
        assert!(!reserves_for(&trtllm_proto::GenerateRequest::default()));
        assert!(!reserves_for(&mlx_proto::GenerateRequest::default()));
        assert!(!reserves_for(&vllm_proto::EmbedRequest::default()));
        assert!(!reserves_for(&sglang_proto::EmbedRequest::default()));
        assert!(!reserves_for(&trtllm_proto::EmbedRequest::default()));

        assert!(reserves_for(&tokenspeed_proto::GenerateRequest {
            mm_inputs: Some(tokenspeed_multimodal(8)),
            ..Default::default()
        }));
        assert!(reserves_for(&vllm_proto::GenerateRequest {
            mm_inputs: Some(vllm_multimodal(8)),
            ..Default::default()
        }));
        assert!(reserves_for(&vllm_proto::GenerateRequest {
            extra_mm_inputs: vec![vllm_multimodal(8)],
            ..Default::default()
        }));
        assert!(reserves_for(&vllm_proto::GenerateRequest {
            media_refs: Some(vllm_proto::MediaRefs {
                items: vec![vllm_proto::MediaRef {
                    modality: crate::common_proto::Modality::Image as i32,
                    url: "data:image/png;base64,iVBORw0KGgo=".to_string(),
                }],
            }),
            ..Default::default()
        }));
        assert!(reserves_for(&sglang_proto::GenerateRequest {
            mm_inputs: Some(Box::new(sglang_multimodal(8))),
            ..Default::default()
        }));
        assert!(reserves_for(&sglang_proto::GenerateRequest {
            extra_mm_inputs: vec![sglang_multimodal(8)],
            ..Default::default()
        }));
        assert!(reserves_for(&sglang_proto::EmbedRequest {
            mm_inputs: Some(sglang_multimodal(8)),
            ..Default::default()
        }));
        assert!(reserves_for(&trtllm_proto::GenerateRequest {
            multimodal_input: Some(trtllm_proto::MultimodalInput::default()),
            ..Default::default()
        }));

        assert!(reserves_for(
            &tokenspeed_encoder_proto::EncodeRequest::default()
        ));
        assert!(reserves_for(&tensor()));
        assert!(reserves_for(&vllm_proto::HealthCheckRequest::default()));
    }

    /// Every engine's generate request, at the size of a tool-call turn (a
    /// thousand prompt ids, a few kilobytes of rendered text), frames as
    /// prost's bytes behind the five-byte header on all five wires: the
    /// one-pass write changes nothing the engine reads.
    #[test]
    fn a_generate_request_frames_as_prosts_bytes_behind_the_five_byte_header() {
        let request_id = "chatcmpl-0123456789abcdef0123456789abcdef".to_string();
        frames_as_prost_does(tokenspeed_proto::GenerateRequest {
            request_id: request_id.clone(),
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                input_ids: prompt_ids(1_114),
                original_text: prompt_text(4_196),
            }),
            sampling_params: Some(tokenspeed_proto::SamplingParams {
                temperature: Some(0.7),
                top_p: Some(0.9),
                max_new_tokens: Some(64),
                ..Default::default()
            }),
            ..Default::default()
        });
        frames_as_prost_does(vllm_proto::GenerateRequest {
            request_id: request_id.clone(),
            input: Some(vllm_proto::generate_request::Input::Tokenized(
                vllm_proto::TokenizedInput {
                    original_text: prompt_text(4_196),
                    input_ids: prompt_ids(1_114),
                },
            )),
            sampling_params: Some(vllm_proto::SamplingParams {
                temperature: Some(0.7),
                top_p: 0.9,
                max_tokens: Some(64),
                ..Default::default()
            }),
            ..Default::default()
        });
        frames_as_prost_does(mlx_proto::GenerateRequest {
            request_id: request_id.clone(),
            input: Some(mlx_proto::generate_request::Input::Tokenized(
                mlx_proto::TokenizedInput {
                    original_text: prompt_text(4_196),
                    input_ids: prompt_ids(1_114),
                },
            )),
            sampling_params: Some(mlx_proto::SamplingParams {
                temperature: Some(0.7),
                top_p: 0.9,
                ..Default::default()
            }),
            ..Default::default()
        });
        frames_as_prost_does(trtllm_proto::GenerateRequest {
            request_id: request_id.clone(),
            tokenized: Some(trtllm_proto::TokenizedInput {
                original_text: prompt_text(4_196),
                input_token_ids: prompt_ids(1_114),
                ..Default::default()
            }),
            ..Default::default()
        });
        frames_as_prost_does(sglang_proto::GenerateRequest {
            request_id,
            tokenized: Some(sglang_proto::TokenizedInput {
                original_text: prompt_text(4_196),
                input_ids: prompt_ids(1_114),
            }),
            sampling_params: Some(sglang_proto::SamplingParams {
                temperature: 0.7,
                top_p: 0.9,
                max_new_tokens: Some(64),
                ..Default::default()
            }),
            ..Default::default()
        });
    }

    /// A generate request that carries multimodal inputs, with an inline
    /// payload beyond tonic's eager buffer, frames as prost's bytes on every
    /// wire that carries one: prost's reserved write changes nothing the
    /// engine reads.
    #[test]
    fn a_request_with_multimodal_inputs_frames_as_prosts_bytes() {
        let request_id = "chatcmpl-0123456789abcdef0123456789abcdef".to_string();
        frames_as_prost_does(tokenspeed_proto::GenerateRequest {
            request_id: request_id.clone(),
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                input_ids: prompt_ids(77),
                original_text: prompt_text(448),
            }),
            mm_inputs: Some(tokenspeed_multimodal(64 * 1024)),
            ..Default::default()
        });
        frames_as_prost_does(vllm_proto::GenerateRequest {
            request_id: request_id.clone(),
            input: Some(vllm_proto::generate_request::Input::Tokenized(
                vllm_proto::TokenizedInput {
                    original_text: prompt_text(448),
                    input_ids: prompt_ids(77),
                },
            )),
            mm_inputs: Some(vllm_multimodal(64 * 1024)),
            ..Default::default()
        });
        frames_as_prost_does(trtllm_proto::GenerateRequest {
            request_id: request_id.clone(),
            tokenized: Some(trtllm_proto::TokenizedInput {
                original_text: prompt_text(448),
                input_token_ids: prompt_ids(77),
                ..Default::default()
            }),
            multimodal_input: Some(trtllm_proto::MultimodalInput {
                image_data: vec![inline_payload(64 * 1024)],
            }),
            ..Default::default()
        });
        frames_as_prost_does(sglang_proto::GenerateRequest {
            request_id,
            tokenized: Some(sglang_proto::TokenizedInput {
                original_text: prompt_text(448),
                input_ids: prompt_ids(77),
            }),
            mm_inputs: Some(Box::new(sglang_multimodal(64 * 1024))),
            ..Default::default()
        });
    }

    /// A message larger than tonic's eager encode buffer, as ids (a long
    /// prompt, written into the growing buffer) and as one payload (a tensor,
    /// written into the reserved one): both frame the same bytes.
    #[test]
    fn a_message_beyond_the_eager_buffer_frames_the_same_bytes() {
        frames_as_prost_does(tokenspeed_proto::GenerateRequest {
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                input_ids: prompt_ids(20_000),
                original_text: prompt_text(60_000),
            }),
            ..Default::default()
        });
        frames_as_prost_does(TensorData {
            shape: vec![1, 128, 128],
            dtype: "float32".to_string(),
            payload: Some(Payload::Inline(Bytes::from(vec![9u8; 64 * 1024]))),
        });
    }

    /// The ids' varints across every byte width, and an empty id list, frame
    /// as prost's bytes on every wire that carries a prompt, behind a request
    /// id and a prompt text on each.
    #[test]
    fn prompt_ids_of_every_varint_width_frame_as_prosts_bytes() {
        let widths = vec![
            0u32,
            1,
            127,
            128,
            300,
            16_383,
            16_384,
            2_097_151,
            2_097_152,
            u32::MAX,
        ];
        for ids in [widths, Vec::new()] {
            frames_as_prost_does(tokenspeed_proto::GenerateRequest {
                request_id: "r".to_string(),
                tokenized: Some(tokenspeed_proto::TokenizedInput {
                    input_ids: ids.clone(),
                    original_text: "two words".to_string(),
                }),
                ..Default::default()
            });
            frames_as_prost_does(vllm_proto::GenerateRequest {
                request_id: "r".to_string(),
                input: Some(vllm_proto::generate_request::Input::Tokenized(
                    vllm_proto::TokenizedInput {
                        original_text: "two words".to_string(),
                        input_ids: ids.clone(),
                    },
                )),
                ..Default::default()
            });
            frames_as_prost_does(sglang_proto::GenerateRequest {
                request_id: "r".to_string(),
                tokenized: Some(sglang_proto::TokenizedInput {
                    original_text: "two words".to_string(),
                    input_ids: ids.clone(),
                }),
                ..Default::default()
            });
            frames_as_prost_does(trtllm_proto::GenerateRequest {
                request_id: "r".to_string(),
                tokenized: Some(trtllm_proto::TokenizedInput {
                    original_text: "two words".to_string(),
                    input_token_ids: ids.clone(),
                    query_token_ids: vec![7, 300],
                }),
                ..Default::default()
            });
            frames_as_prost_does(mlx_proto::GenerateRequest {
                request_id: "r".to_string(),
                input: Some(mlx_proto::generate_request::Input::Tokenized(
                    mlx_proto::TokenizedInput {
                        original_text: "two words".to_string(),
                        input_ids: ids,
                    },
                )),
                ..Default::default()
            });
        }
    }

    /// A request whose prompt is text, and one without a prompt message,
    /// take prost's path unchanged.
    #[test]
    fn a_request_without_prompt_ids_frames_as_prosts_bytes() {
        frames_as_prost_does(vllm_proto::GenerateRequest {
            request_id: "r".to_string(),
            input: Some(vllm_proto::generate_request::Input::Text(prompt_text(300))),
            ..Default::default()
        });
        frames_as_prost_does(mlx_proto::GenerateRequest {
            input: Some(mlx_proto::generate_request::Input::Text(prompt_text(300))),
            ..Default::default()
        });
        frames_as_prost_does(sglang_proto::GenerateRequest {
            request_id: "r".to_string(),
            ..Default::default()
        });
        frames_as_prost_does(trtllm_proto::GenerateRequest::default());
    }

    /// An empty request id is no field at all, as prost writes it, on every
    /// wire the writer serves, behind it a prompt with a text and ids.
    #[test]
    fn a_prompt_behind_an_empty_request_id_frames_as_prosts_bytes() {
        let ids = vec![1u32, 300];
        frames_as_prost_does(tokenspeed_proto::GenerateRequest {
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                input_ids: ids.clone(),
                original_text: "two words".to_string(),
            }),
            stream: true,
            ..Default::default()
        });
        frames_as_prost_does(vllm_proto::GenerateRequest {
            input: Some(vllm_proto::generate_request::Input::Tokenized(
                vllm_proto::TokenizedInput {
                    original_text: "two words".to_string(),
                    input_ids: ids.clone(),
                },
            )),
            stream: true,
            ..Default::default()
        });
        frames_as_prost_does(sglang_proto::GenerateRequest {
            tokenized: Some(sglang_proto::TokenizedInput {
                original_text: "two words".to_string(),
                input_ids: ids.clone(),
            }),
            stream: true,
            ..Default::default()
        });
        frames_as_prost_does(trtllm_proto::GenerateRequest {
            tokenized: Some(trtllm_proto::TokenizedInput {
                original_text: "two words".to_string(),
                input_token_ids: ids.clone(),
                query_token_ids: vec![7, 300],
            }),
            streaming: true,
            ..Default::default()
        });
        frames_as_prost_does(mlx_proto::GenerateRequest {
            input: Some(mlx_proto::generate_request::Input::Tokenized(
                mlx_proto::TokenizedInput {
                    original_text: "two words".to_string(),
                    input_ids: ids,
                },
            )),
            stream: true,
            ..Default::default()
        });
    }

    /// A text-only request whose other fields outweigh its prompt (a
    /// structured-output constraint beyond the scratch cap behind a short
    /// prompt) frames as prost's bytes, and those fields never pass through
    /// the thread's scratch: prost writes them straight into the frame, so
    /// the scratch holds the short prompt's packed ids and nothing else.
    #[test]
    fn a_large_constraint_behind_a_short_prompt_frames_as_prosts_bytes_outside_the_scratch() {
        std::thread::spawn(|| {
            let schema = r#"{"type":"object","properties":{}}"#.repeat(SCRATCH_CAP / 32 + 1);
            assert!(schema.len() > SCRATCH_CAP);
            frames_as_prost_does(vllm_proto::GenerateRequest {
                request_id: "r".to_string(),
                input: Some(vllm_proto::generate_request::Input::Tokenized(
                    vllm_proto::TokenizedInput {
                        original_text: prompt_text(256),
                        input_ids: prompt_ids(64),
                    },
                )),
                sampling_params: Some(vllm_proto::SamplingParams {
                    constraint: Some(vllm_proto::sampling_params::Constraint::JsonSchema(
                        schema.clone(),
                    )),
                    ..Default::default()
                }),
                stream: true,
                ..Default::default()
            });
            frames_as_prost_does(sglang_proto::GenerateRequest {
                request_id: "r".to_string(),
                tokenized: Some(sglang_proto::TokenizedInput {
                    original_text: prompt_text(256),
                    input_ids: prompt_ids(64),
                }),
                sampling_params: Some(sglang_proto::SamplingParams {
                    constraint: Some(sglang_proto::sampling_params::Constraint::JsonSchema(
                        schema.clone(),
                    )),
                    ..Default::default()
                }),
                stream: true,
                ..Default::default()
            });
            frames_as_prost_does(tokenspeed_proto::GenerateRequest {
                request_id: "r".to_string(),
                tokenized: Some(tokenspeed_proto::TokenizedInput {
                    input_ids: prompt_ids(64),
                    original_text: prompt_text(256),
                }),
                sampling_params: Some(tokenspeed_proto::SamplingParams {
                    constraint: Some(tokenspeed_proto::sampling_params::Constraint::JsonSchema(
                        schema,
                    )),
                    ..Default::default()
                }),
                stream: true,
                ..Default::default()
            });
            let ids = scratch_capacity();
            assert!(
                (1..4096).contains(&ids),
                "the scratch holds the packed ids alone: {ids}"
            );
        })
        .join()
        .expect("the remainder pins hold");
    }

    /// The thread's scratch outlives a request: a second, smaller request
    /// after a large one frames its own bytes.
    #[test]
    fn a_smaller_request_after_a_large_one_frames_its_own_bytes() {
        for count in [5_000usize, 3, 700] {
            frames_as_prost_does(tokenspeed_proto::GenerateRequest {
                request_id: "r".to_string(),
                tokenized: Some(tokenspeed_proto::TokenizedInput {
                    input_ids: prompt_ids(count),
                    original_text: prompt_text(count * 3),
                }),
                ..Default::default()
            });
        }
    }

    /// A request with multimodal inputs is prost's one reserved write on
    /// every wire that carries them: the frame is prost's and the thread's
    /// scratch stays empty, so a payload is never copied through it or kept
    /// by it.
    #[test]
    fn a_multimodal_request_never_reaches_the_scratch() {
        std::thread::spawn(|| {
            let request_id = "chatcmpl-0123456789abcdef0123456789abcdef".to_string();
            frames_as_prost_does(vllm_proto::GenerateRequest {
                request_id: request_id.clone(),
                input: Some(vllm_proto::generate_request::Input::Tokenized(
                    vllm_proto::TokenizedInput {
                        original_text: prompt_text(448),
                        input_ids: prompt_ids(77),
                    },
                )),
                mm_inputs: Some(vllm_multimodal(SCRATCH_CAP + 1)),
                ..Default::default()
            });
            frames_as_prost_does(sglang_proto::GenerateRequest {
                request_id: request_id.clone(),
                tokenized: Some(sglang_proto::TokenizedInput {
                    original_text: prompt_text(448),
                    input_ids: prompt_ids(77),
                }),
                mm_inputs: Some(Box::new(sglang_multimodal(SCRATCH_CAP + 1))),
                ..Default::default()
            });
            frames_as_prost_does(tokenspeed_proto::GenerateRequest {
                request_id: request_id.clone(),
                tokenized: Some(tokenspeed_proto::TokenizedInput {
                    input_ids: prompt_ids(77),
                    original_text: prompt_text(448),
                }),
                mm_inputs: Some(tokenspeed_multimodal(SCRATCH_CAP + 1)),
                ..Default::default()
            });
            frames_as_prost_does(trtllm_proto::GenerateRequest {
                request_id,
                tokenized: Some(trtllm_proto::TokenizedInput {
                    original_text: prompt_text(448),
                    input_token_ids: prompt_ids(77),
                    ..Default::default()
                }),
                multimodal_input: Some(trtllm_proto::MultimodalInput {
                    image_data: vec![inline_payload(SCRATCH_CAP + 1)],
                }),
                ..Default::default()
            });
            assert_eq!(scratch_capacity(), 0);
        })
        .join()
        .expect("the multimodal pins hold");
    }

    /// A text-only prompt beyond the scratch cap frames as prost's bytes
    /// through the one-pass writer, and the thread does not keep it: the
    /// outsized ids are released after the write, and the next request, a few
    /// ids behind a text beyond the cap, leaves the scratch at the size of its
    /// ids, because the text never enters it.
    #[test]
    fn a_prompt_beyond_the_scratch_cap_frames_as_prosts_bytes_and_is_not_kept() {
        std::thread::spawn(|| {
            frames_as_prost_does(tokenspeed_proto::GenerateRequest {
                request_id: "r".to_string(),
                tokenized: Some(tokenspeed_proto::TokenizedInput {
                    input_ids: prompt_ids(400_000),
                    original_text: prompt_text(SCRATCH_CAP + 1),
                }),
                ..Default::default()
            });
            assert_eq!(scratch_capacity(), 0, "the outsized ids are released");
            frames_as_prost_does(tokenspeed_proto::GenerateRequest {
                request_id: "r".to_string(),
                tokenized: Some(tokenspeed_proto::TokenizedInput {
                    input_ids: prompt_ids(77),
                    original_text: prompt_text(SCRATCH_CAP + 1),
                }),
                ..Default::default()
            });
            let ids = scratch_capacity();
            assert!(
                (1..4096).contains(&ids),
                "the text never enters the scratch, which holds the ids alone: {ids}"
            );
        })
        .join()
        .expect("the cap pins hold");
    }
}
