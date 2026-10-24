// The control reply shared by the slim wires: a tagged positional msgpack
// array `[tag, call_id, success, message, engine_index]`. **Field order is
// the wire contract** — do not reorder.

use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{
    codec::OpaqueValue,
    protocol::positional::{drain_trailing, expect_tag, next_field},
};

/// The msgspec tag for [`ControlReplySlim`] (element 0 on the wire).
pub const CONTROL_REPLY_SLIM_TAG: &str = "ControlReplySlim";

/// The answer to a control call (`ControlReplySlim`) on a slim wire: the
/// scheduler's `success`/`message` for the call id SMG issued. The SGLang
/// plugin sends it; a TokenSpeed scheduler of the control-wire revision
/// sends the same message and declares it in its handshake.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ControlReplySlim {
    pub call_id: i64,
    pub success: bool,
    pub message: Option<String>,
    pub engine_index: u32,
}

impl Serialize for ControlReplySlim {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(5)?;
        tuple.serialize_element(CONTROL_REPLY_SLIM_TAG)?;
        tuple.serialize_element(&self.call_id)?;
        tuple.serialize_element(&self.success)?;
        tuple.serialize_element(&self.message)?;
        tuple.serialize_element(&self.engine_index)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for ControlReplySlim {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ReplyVisitor;

        impl<'de> Visitor<'de> for ReplyVisitor {
            type Value = ControlReplySlim;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged ControlReplySlim positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                expect_tag(&mut seq, CONTROL_REPLY_SLIM_TAG)?;
                read_control_reply_after_tag(&mut seq)
            }
        }

        deserializer.deserialize_seq(ReplyVisitor)
    }
}

/// The reply's fields after the tag; shared by the struct's own decoder and
/// the wire enums'.
pub(crate) fn read_control_reply_after_tag<'de, A: SeqAccess<'de>>(
    seq: &mut A,
) -> Result<ControlReplySlim, A::Error> {
    let reply = ControlReplySlim {
        call_id: next_field(seq, "call_id")?,
        success: next_field(seq, "success")?,
        message: next_field(seq, "message")?,
        engine_index: seq.next_element::<u32>()?.unwrap_or(0),
    };
    drain_trailing(seq)?;
    Ok(reply)
}

impl ControlReplySlim {
    /// The reply as a utility outcome: a map of `success` and `message`, so
    /// a refused flush (success false) is a reply, not a transport failure.
    pub fn into_outcome(self) -> OpaqueValue {
        OpaqueValue::Map(vec![
            (
                OpaqueValue::from("success"),
                OpaqueValue::Boolean(self.success),
            ),
            (
                OpaqueValue::from("message"),
                self.message.map_or(OpaqueValue::Nil, OpaqueValue::from),
            ),
        ])
    }
}
