//! The gRPC codec of the engine clients: prost's, with the encode buffer
//! sized once.
//!
//! tonic's prost encoder writes a message into a growing `BytesMut`, so a
//! multi-megabyte pixel tensor is copied again at every doubling of the
//! buffer, into fresh pages each time. Here the buffer is reserved to the
//! message's encoded length before the write (prost computes that length for
//! its capacity check as well), so the payload is written once into one
//! allocation.
//! Decoding is prost's own decoder, unchanged. tonic builds the encode
//! buffer itself, so the reservation is exercised by every client call of
//! the gateway's gRPC tests rather than by a unit test here.

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
        // `encode` walks the field lengths once more for its capacity check
        // (no payload bytes touched); it is the public API, where
        // `encode_raw` is prost's own. The buffer was reserved to the length
        // it checks, so the error is unreachable; it is still a status, not
        // a panic, in a request path.
        item.encode(buf)
            .map_err(|error| Status::internal(format!("message encoding failed: {error}")))
    }

    fn buffer_settings(&self) -> BufferSettings {
        BufferSettings::default()
    }
}

#[cfg(test)]
mod tests {
    use prost::bytes::Bytes;

    use super::*;
    use crate::tokenspeed_scheduler::tokenspeed_proto::{tensor_data::Payload, TensorData};

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
}
