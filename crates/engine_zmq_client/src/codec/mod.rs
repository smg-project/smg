// Adapted from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/mod.rs.

//! Engine-agnostic wire primitives: msgpack (de)serialization, numpy dtype
//! handling, and the zero-copy tensor aux-frame codec.

use std::{
    any::type_name,
    fmt,
    io::{self, Cursor, Write},
    marker::PhantomData,
};

use rmpv::Value;
use serde::{
    de::{value::SeqAccessDeserializer, DeserializeSeed, IgnoredAny, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};

use crate::error::{Error, Result};

pub mod dtype;
pub mod frame_str;
pub mod tensor;

pub use frame_str::FrameStr;

/// Dynamic msgpack value used for schema positions that are preserved but not
/// yet strongly typed.
pub type OpaqueValue = Value;

/// Decode adapter for msgspec `array_like` structs. Upstream grows those
/// structs by appending fields, so a newer engine sends a longer positional
/// array than this client knows about — and `rmp-serde` rejects an array with
/// unconsumed elements, which would fail the whole message. This wrapper
/// consumes the elements `T` declares and discards the trailing remainder.
#[derive(Debug, Clone, PartialEq)]
pub struct TrailingTolerant<T>(pub T);

impl<'de, T> Deserialize<'de> for TrailingTolerant<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TupleVisitor<T>(PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for TupleVisitor<T> {
            type Value = TrailingTolerant<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "a positional array for `{}`", type_name::<T>())
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let value = T::deserialize(SeqAccessDeserializer::new(&mut seq))?;
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(TrailingTolerant(value))
            }
        }

        deserializer.deserialize_seq(TupleVisitor(PhantomData))
    }
}

/// Deserialize a sequence of `array_like` structs, tolerating trailing wire
/// fields in each element. Use as `#[serde(deserialize_with = ...)]`.
pub fn deserialize_tolerant_seq<'de, D, T>(deserializer: D) -> std::result::Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let items = Vec::<TrailingTolerant<T>>::deserialize(deserializer)?;
    Ok(items.into_iter().map(|item| item.0).collect())
}

/// Encode a Rust value into msgpack using named fields (map encoding for
/// structs; positional arrays come from `serde_tuple`-derived types). The
/// buffer grows from empty; a value whose size is known up front goes
/// through [`encode_msgpack_with_capacity`].
pub fn encode_msgpack<T>(value: &T) -> Result<Vec<u8>>
where
    T: Serialize + fmt::Debug,
{
    encode_msgpack_with_capacity(value, 0)
}

/// [`encode_msgpack`] into a buffer of `capacity` bytes, reserved before the
/// write: a value that fits costs one allocation, where a buffer grown from
/// empty costs one per doubling and asks for a third more bytes than it
/// keeps. A value past the capacity still encodes; its buffer grows as before.
pub fn encode_msgpack_with_capacity<T>(value: &T, capacity: usize) -> Result<Vec<u8>>
where
    T: Serialize + fmt::Debug,
{
    let encode_error = |message: String| Error::Encode {
        target_type: type_name::<T>(),
        message,
    };
    let mut buffer = FallibleBuffer(Vec::new());
    buffer
        .0
        .try_reserve_exact(capacity)
        .map_err(|error| encode_error(format!("failed to reserve {capacity} bytes: {error}")))?;
    rmp_serde::encode::write_named(&mut buffer, value)
        .map_err(|error| encode_error(format!("failed to encode value `{value:?}`: {error}")))?;
    Ok(buffer.0)
}

/// The encoder's buffer: a `Vec<u8>` whose growth fails as an I/O error the
/// encode reports instead of aborting the process. The same writer as the
/// one `rmp_serde::to_vec_named` uses, which is private and starts from an
/// empty vector, so a reserved buffer needs its own.
struct FallibleBuffer(Vec<u8>);

impl Write for FallibleBuffer {
    // Inlined as the encoder's own writer is: a message is written in many
    // small pieces, and a call per piece shows in the instructions.
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.write_all(bytes)?;
        Ok(bytes.len())
    }

    #[inline]
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.0
            .try_reserve(bytes.len())
            .map_err(|_| io::ErrorKind::OutOfMemory)?;
        self.0.extend_from_slice(bytes);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Room for an add-request's fields beside its prompt token ids: the request
/// id, the sampling parameters, the flags, a structured-output grammar or a
/// trace context of ordinary size.
const ADD_REQUEST_ROOM: usize = 1024;

/// The buffer an add-request of `prompt_tokens` ids is encoded into in one
/// write: msgpack spells a token id in at most five bytes (`uint32`), and the
/// rest of the request fits `ADD_REQUEST_ROOM`. The frame bytes do not
/// depend on it; a request past the bound grows its buffer as it used to.
pub fn add_request_capacity(prompt_tokens: usize) -> usize {
    prompt_tokens
        .saturating_mul(5)
        .saturating_add(ADD_REQUEST_ROOM)
}

fn decode_value_preview(bytes: &[u8]) -> String {
    match decode_value(bytes) {
        Ok(value) => format!("{value}"),
        Err(error) => format!("<value decode failed: {error}>"),
    }
}

/// Decode a msgpack payload with a [`DeserializeSeed`]: a decoder that needs
/// a fact known only at run time (the positions an engine declared for the
/// optional columns of its output batch), with the diagnostics of
/// [`decode_msgpack`].
pub fn decode_msgpack_seed<'de, S>(bytes: &'de [u8], seed: S) -> Result<S::Value>
where
    S: DeserializeSeed<'de>,
{
    let mut deserializer = rmp_serde::Deserializer::from_read_ref(bytes);
    seed.deserialize(&mut deserializer)
        .map_err(|error| Error::Decode {
            target_type: type_name::<S::Value>(),
            message: format!("{error}; value fallback: {}", decode_value_preview(bytes)),
        })
}

/// Decode a msgpack payload into a strongly typed value, attaching a dynamic
/// value preview to decode failures for diagnostics.
pub fn decode_msgpack<T>(bytes: &[u8]) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    rmp_serde::from_slice(bytes).map_err(|error| Error::Decode {
        target_type: type_name::<T>(),
        message: format!("{error}; value fallback: {}", decode_value_preview(bytes)),
    })
}

/// Decode a msgpack payload into a dynamic value for diagnostics and tests.
pub fn decode_value(bytes: &[u8]) -> Result<Value> {
    Ok(rmpv::decode::read_value(&mut Cursor::new(bytes))?)
}

/// The bytes of a hex pin captured from an engine's own encoder; spaces and
/// line breaks between the digits are ignored (tests only).
#[cfg(test)]
pub(crate) fn hex(text: &str) -> Vec<u8> {
    let digits: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    (0..digits.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&digits[i..i + 2], 16).expect("a hex digit pair"))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn decode_msgpack_includes_type_name_and_value_fallback() {
        let payload = rmp_serde::to_vec_named(&BTreeMap::from([("status", "READY")])).unwrap();
        let error = decode_msgpack::<u64>(&payload).unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("u64"), "missing type name: {rendered}");
        assert!(
            rendered.contains("READY"),
            "missing value fallback: {rendered}"
        );
    }

    #[test]
    fn roundtrip_named_map_struct() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Cfg {
            status: String,
            headless: bool,
        }
        let cfg = Cfg {
            status: "HELLO".into(),
            headless: true,
        };
        let bytes = encode_msgpack(&cfg).unwrap();
        assert_eq!(decode_msgpack::<Cfg>(&bytes).unwrap(), cfg);
    }
}
