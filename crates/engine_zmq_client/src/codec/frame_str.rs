//! A string an engine message carries, kept as a view of the message's frame.

use std::{fmt, ops::Deref};

use bytes::Bytes;
use serde::{de::Visitor, Deserialize, Deserializer, Serialize, Serializer};

use super::tensor::frame_view_bytes;

/// A string decoded from an engine message: a view of the frame it was
/// decoded within ([`super::tensor::decode_within_frame`]), so the frame's
/// buffer is shared and nothing is copied; a copy when the decode ran outside
/// a frame or the string was built from a Rust string. Every constructor
/// takes a `str`, so the bytes are UTF-8 by construction, and [`Self::as_str`]
/// re-reads them instead of trusting them unchecked.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct FrameStr(Bytes);

impl FrameStr {
    /// The string.
    pub fn as_str(&self) -> &str {
        // UTF-8 by construction: every constructor takes a `str`. The check
        // reads a request id's few dozen bytes, which costs less than the
        // string the view replaces; the fallback is unreachable and named so.
        let text = std::str::from_utf8(&self.0);
        debug_assert!(text.is_ok(), "a FrameStr is built from a str");
        text.unwrap_or_default()
    }

    /// The string's length in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the string is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Deref for FrameStr {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for FrameStr {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Debug for FrameStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl fmt::Display for FrameStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for FrameStr {
    fn from(text: &str) -> Self {
        Self(Bytes::copy_from_slice(text.as_bytes()))
    }
}

impl From<String> for FrameStr {
    fn from(text: String) -> Self {
        Self(Bytes::from(text.into_bytes()))
    }
}

impl From<&FrameStr> for String {
    fn from(text: &FrameStr) -> Self {
        text.as_str().to_string()
    }
}

impl PartialEq<str> for FrameStr {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for FrameStr {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for FrameStr {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Serialize for FrameStr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for FrameStr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FrameStrVisitor;

        impl<'de> Visitor<'de> for FrameStrVisitor {
            type Value = FrameStr;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string, as text or as UTF-8 bytes")
            }

            // A bytes-typed id is read as a `String` would read it: checked
            // as UTF-8, then kept the same way as text.
            fn visit_borrowed_bytes<E: serde::de::Error>(
                self,
                bytes: &'de [u8],
            ) -> Result<FrameStr, E> {
                let text = std::str::from_utf8(bytes).map_err(E::custom)?;
                self.visit_borrowed_str(text)
            }

            fn visit_bytes<E: serde::de::Error>(self, bytes: &[u8]) -> Result<FrameStr, E> {
                std::str::from_utf8(bytes)
                    .map(FrameStr::from)
                    .map_err(E::custom)
            }

            fn visit_borrowed_str<E: serde::de::Error>(
                self,
                text: &'de str,
            ) -> Result<FrameStr, E> {
                Ok(FrameStr(frame_view_bytes(text.as_bytes())))
            }

            fn visit_str<E: serde::de::Error>(self, text: &str) -> Result<FrameStr, E> {
                Ok(FrameStr::from(text))
            }

            fn visit_string<E: serde::de::Error>(self, text: String) -> Result<FrameStr, E> {
                Ok(FrameStr::from(text))
            }
        }

        deserializer.deserialize_str(FrameStrVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{
        decode_msgpack, encode_msgpack,
        tensor::{decode_within_frame, lies_within},
    };

    /// Decoded within its frame, a string is a view of the frame's buffer;
    /// decoded from a plain slice, it is a copy. Both read the same text.
    #[test]
    fn a_string_decoded_within_its_frame_is_a_view_of_it() {
        let frame = Bytes::from(encode_msgpack(&"req-0123456789abcdef").unwrap());
        let within: FrameStr =
            decode_within_frame(&frame, || decode_msgpack(&frame)).expect("decode");
        let outside: FrameStr = decode_msgpack(&frame).expect("decode");
        assert_eq!(within, "req-0123456789abcdef");
        assert_eq!(outside, within);
        let frame_range = frame.as_ptr_range();
        let view_range = within.0.as_ptr_range();
        assert!(frame_range.start <= view_range.start && view_range.end <= frame_range.end);
        let copy_range = outside.0.as_ptr_range();
        assert!(copy_range.start < frame_range.start || copy_range.start >= frame_range.end);
    }

    /// A bytes-typed id reads as text, as a `String` field reads one, and
    /// bytes that are not UTF-8 are refused.
    #[test]
    fn a_bytes_typed_id_decodes_as_text_and_refuses_other_bytes() {
        // msgpack bin8 of "req-bin", the shape a `String` field also accepts.
        let frame = Bytes::from_static(b"\xc4\x07req-bin");
        let id: FrameStr = decode_within_frame(&frame, || decode_msgpack(&frame)).expect("decode");
        assert_eq!(id, "req-bin");
        assert!(lies_within(&frame, id.0.as_ref()));
        let broken = Bytes::from_static(b"\xc4\x02\xff\xfe");
        assert!(decode_msgpack::<FrameStr>(&broken).is_err());
    }

    /// The string round-trips through the encoder as a plain msgpack string
    /// and reads as one in every comparison and format.
    #[test]
    fn a_frame_string_encodes_and_compares_as_a_string() {
        let text = FrameStr::from("req-1");
        assert_eq!(
            encode_msgpack(&text).unwrap(),
            encode_msgpack(&"req-1").unwrap()
        );
        assert_eq!(text, *"req-1");
        assert_eq!(text, "req-1".to_string());
        assert_eq!(format!("{text}"), "req-1");
        assert_eq!(format!("{text:?}"), "\"req-1\"");
        assert_eq!(String::from(&text), "req-1");
        assert_eq!(text.len(), 5);
        assert!(!text.is_empty());
        assert!(FrameStr::default().is_empty());
        assert_eq!(FrameStr::from("a".to_string()), "a");
    }
}
