//! The gRPC codec of the engine clients: prost's, with the encode buffer
//! sized once and the message walked once more than it is written, not twice.
//!
//! tonic's prost encoder writes a message into a growing `BytesMut`, so a
//! multi-megabyte pixel tensor is copied again at every doubling of the
//! buffer, into fresh pages each time. Here the buffer is reserved to the
//! message's encoded length before the write, so the payload is written
//! once into one allocation. The write is prost's `encode_raw`, the writer
//! behind its `encode` and `encode_to_vec`: `encode` walks every field's
//! length a second time for a capacity check (as long a walk as the first on
//! a request of a thousand prompt ids, with no byte written), and the check
//! cannot fail against a buffer that grows on demand, so it is not paid.
//! Decoding is prost's own decoder, unchanged. tonic builds the encode
//! buffer itself; the tests below drive the codec through tonic's own
//! request body, as a client call does.

use std::marker::PhantomData;

use prost::Message;
use tonic::{
    codec::{BufferSettings, Codec, EncodeBuf, Encoder},
    Status,
};
use tonic_prost::{ProstCodec, ProstDecoder};

/// A [`Codec`] for `T` requests and `U` responses that reserves each encoded
/// message's length before writing it.
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

/// prost's encoding into a buffer reserved to the message's length first.
#[derive(Debug, Clone)]
pub struct SizedProstEncoder<T> {
    _pd: PhantomData<T>,
}

impl<T: Message> Encoder for SizedProstEncoder<T> {
    type Item = T;
    type Error = Status;

    fn encode(&mut self, item: Self::Item, buf: &mut EncodeBuf<'_>) -> Result<(), Self::Error> {
        buf.reserve(item.encoded_len());
        // The buffer is `BytesMut`-backed and grows on demand, so the write
        // cannot run out of room: prost's `encode` would only repeat the
        // length walk above to tell us so.
        item.encode_raw(buf);
        Ok(())
    }

    fn buffer_settings(&self) -> BufferSettings {
        BufferSettings::default()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        pin::pin,
        task::{Context, Poll},
    };

    use futures::{future::ready, stream};
    use hyper::body::Body;
    use prost::bytes::Bytes;
    use tonic::codec::EncodeBody;

    use super::*;
    use crate::{
        sglang_proto, tokenspeed_proto,
        tokenspeed_scheduler::tokenspeed_proto::{tensor_data::Payload, TensorData},
        vllm_proto,
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

    /// Every engine's generate request, at the size of a tool-call turn (a
    /// thousand prompt ids, a few kilobytes of rendered text), frames as
    /// prost's bytes behind the five-byte header: the one-pass write changes
    /// nothing the engine reads.
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

    /// A message larger than tonic's eager encode buffer, as ids (a long
    /// prompt) and as one payload (a tensor): the reservation path frames
    /// the same bytes.
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
