//! The gRPC codec of the engine wire: prost's, with the encode buffer
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
//!
//! The buffer settings size the two buffers tonic allocates eagerly for
//! every call: an encode buffer per outgoing stream and a decode buffer per
//! incoming stream, each of the settings' `buffer_size` up front. The
//! default is 8 KiB on both, which is spent whether a message uses it or
//! not. The encoder here reserves each message's own length before it
//! writes, so the eager encode buffer only has to hold a message that fits
//! it without a second allocation.
//!
//! The codec serves both ends of the wire, where the roles are reversed: a
//! generated client encodes the request stream and decodes the response
//! stream, a generated server (an engine servicer, the mock worker) encodes
//! the response stream and decodes the request stream. The eager encode
//! size is therefore a parameter of the codec, fixed when the code is
//! generated (`build.rs`): the clients' codec ([`ClientCodec`]) starts a
//! request stream at the size a text request fits, the servers' keeps
//! tonic's default for a response stream, whose frames a server packs from
//! every chunk that is ready. The decode buffer keeps its default at both
//! ends: a streamed response's frames carry several messages when the
//! sender had several ready, so that buffer grows past its start once per
//! stream already (see the constants). The yield threshold keeps tonic's
//! default too: it decides when a run of encoded messages is flushed as a
//! frame, not how much is reserved.

use std::marker::PhantomData;

use prost::Message;
use tonic::{
    codec::{BufferSettings, Codec, EncodeBuf, Encoder},
    Status,
};
use tonic_prost::{ProstCodec, ProstDecoder};

/// The eager encode buffer of a client's request stream: a text request
/// with its token ids and sampling parameters encodes to under 4 KiB (a
/// packed token id is two or three bytes, so about 1.3K to 2K ids with the
/// parameters beside them) and is written into this one allocation. A
/// request between 4 and 8 KiB (2K to 4K ids, or a large structured-output
/// constraint) grows the buffer once, to the 8 KiB it started with before;
/// a larger message (a request carrying an inline tensor) is reserved to
/// its exact length once, as before.
pub const CLIENT_ENCODE_BUFFER_BYTES: usize = 4 * 1024;

/// The eager encode buffer of a server's response stream: tonic's default.
/// A server packs every chunk that is ready into one frame, and that end of
/// the wire is not what the gateway's measurements read, so it is left as
/// it was.
pub const SERVER_ENCODE_BUFFER_BYTES: usize = 8 * 1024;

/// The eager decode buffer per incoming stream keeps tonic's default. A
/// streamed response does not arrive one message at a time: the sender
/// encodes every message that is ready into one frame, so under load a
/// frame carries several chunks and the buffer grows past its start once
/// per stream already; a smaller start only adds growth steps to that
/// (measured: 2 KiB added two allocations per streamed request).
const DECODE_BUFFER_BYTES: usize = 8 * 1024;

/// tonic's default: the encoded bytes buffered before a frame is yielded.
const YIELD_THRESHOLD_BYTES: usize = 32 * 1024;

/// A [`Codec`] for `T` outgoing and `U` incoming messages that reserves each
/// encoded message's length before writing it, with the eager encode buffer
/// of `ENCODE_BUFFER_BYTES`. The generated code names one of the two roles
/// below; the size is a parameter rather than a default so that neither
/// role is the implicit one.
#[derive(Debug, Clone)]
pub struct SizedProstCodec<T, U, const ENCODE_BUFFER_BYTES: usize> {
    _pd: PhantomData<(T, U)>,
}

/// The generated clients' codec: a request stream's eager encode buffer is
/// sized to a text request.
pub type ClientCodec<T, U> = SizedProstCodec<T, U, CLIENT_ENCODE_BUFFER_BYTES>;

/// The generated servers' codec: a response stream's eager encode buffer
/// keeps tonic's default.
pub type ServerCodec<T, U> = SizedProstCodec<T, U, SERVER_ENCODE_BUFFER_BYTES>;

impl<T, U, const ENCODE_BUFFER_BYTES: usize> Default
    for SizedProstCodec<T, U, ENCODE_BUFFER_BYTES>
{
    fn default() -> Self {
        Self { _pd: PhantomData }
    }
}

impl<T, U, const ENCODE_BUFFER_BYTES: usize> Codec for SizedProstCodec<T, U, ENCODE_BUFFER_BYTES>
where
    T: Message + Send + 'static,
    U: Message + Default + Send + 'static,
{
    type Encode = T;
    type Decode = U;
    type Encoder = SizedProstEncoder<T, ENCODE_BUFFER_BYTES>;
    type Decoder = ProstDecoder<U>;

    fn encoder(&mut self) -> Self::Encoder {
        SizedProstEncoder { _pd: PhantomData }
    }

    fn decoder(&mut self) -> Self::Decoder {
        ProstCodec::<T, U>::raw_decoder(BufferSettings::new(
            DECODE_BUFFER_BYTES,
            YIELD_THRESHOLD_BYTES,
        ))
    }
}

/// prost's encoding into a buffer reserved to the message's length first,
/// started at `ENCODE_BUFFER_BYTES` for every stream.
#[derive(Debug, Clone)]
pub struct SizedProstEncoder<T, const ENCODE_BUFFER_BYTES: usize> {
    _pd: PhantomData<T>,
}

impl<T: Message, const ENCODE_BUFFER_BYTES: usize> Encoder
    for SizedProstEncoder<T, ENCODE_BUFFER_BYTES>
{
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
        BufferSettings::new(ENCODE_BUFFER_BYTES, YIELD_THRESHOLD_BYTES)
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
