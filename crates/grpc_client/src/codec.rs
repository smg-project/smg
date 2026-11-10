//! The gRPC codec of the engine clients: prost's, with the encode buffer
//! reserved once for a message that carries inline tensors, and every
//! message written in one pass.
//!
//! tonic's prost encoder writes a message into a growing `BytesMut`, so a
//! multi-megabyte pixel tensor is copied again at every doubling of the
//! buffer, into fresh pages each time. Here a request that carries multimodal
//! inputs or media references has the buffer reserved to its encoded length
//! before the write, so the payload is written once into one allocation. A
//! request that carries a prompt and neither is written without the
//! reservation: the
//! length walk it would take costs about nine instructions per prompt id,
//! more than the doublings it spares (a copy at a fraction of an instruction
//! per byte, and none at all under tonic's eager buffer). Every other message
//! reserves as before; its length is a few fields' worth to walk. The write
//! is prost's `encode_raw`, the writer behind its `encode` and
//! `encode_to_vec`: `encode` walks every field's length for a capacity check
//! (as long a walk as the reservation's on a request of a thousand prompt
//! ids, with no byte written), and the check cannot fail against a buffer
//! that grows on demand, so it is not paid. Decoding is prost's own decoder,
//! unchanged. tonic builds the encode buffer itself; the tests below drive
//! the codec through tonic's own request body, as a client call does.

use std::{any::Any, marker::PhantomData};

use prost::Message;
use tonic::{
    codec::{BufferSettings, Codec, EncodeBuf, Encoder},
    Status,
};
use tonic_prost::{ProstCodec, ProstDecoder};

use crate::{mlx_proto, sglang_proto, tokenspeed_proto, trtllm_proto, vllm_proto};

/// A [`Codec`] for `T` requests and `U` responses that reserves the encoded
/// length of a message carrying inline tensors before writing it.
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

/// prost's encoding, into a buffer reserved to the message's length first
/// when the message carries inline tensors.
#[derive(Debug, Clone)]
pub struct SizedProstEncoder<T> {
    _pd: PhantomData<T>,
}

impl<T: Message + 'static> Encoder for SizedProstEncoder<T> {
    type Item = T;
    type Error = Status;

    fn encode(&mut self, item: Self::Item, buf: &mut EncodeBuf<'_>) -> Result<(), Self::Error> {
        if reserves_for(&item) {
            buf.reserve(item.encoded_len());
        }
        // The buffer is `BytesMut`-backed and grows on demand, so the write
        // cannot run out of room: prost's `encode` would only walk the
        // message's length to tell us so.
        item.encode_raw(buf);
        Ok(())
    }

    fn buffer_settings(&self) -> BufferSettings {
        BufferSettings::default()
    }
}

/// Whether the encode buffer is reserved to `item`'s length before the
/// write. A request that carries a prompt reserves only when it also carries
/// multimodal inputs or media references (a media reference may be a data URL
/// of many megabytes), the shapes whose inline payloads the reservation is
/// for; every other message reserves as before. The encoder is instantiated
/// per message type, so each type check here folds at compile time and a
/// text request pays one field read.
fn reserves_for<T: 'static>(item: &T) -> bool {
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
    /// prost's bytes behind the five-byte header: the one-pass write without
    /// a reservation changes nothing the engine reads.
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
    /// tensor beyond tonic's eager buffer, frames as prost's bytes on every
    /// wire that carries one: the reserved write changes nothing the engine
    /// reads.
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
}
