//! Decoding helpers for the dialects whose messages are tagged positional
//! msgpack arrays (`msgspec.Struct(array_like=True)` on the engine side):
//! read a required element, check the class-name tag at element 0, and skip
//! the trailing fields a newer engine may append.

use serde::de::{Deserialize, Error as _, IgnoredAny, SeqAccess};

/// Read the next positional element, failing loudly when the array is shorter
/// than the modeled prefix (every modeled field is required on decode).
pub(crate) fn next_field<'de, A, T>(seq: &mut A, name: &'static str) -> Result<T, A::Error>
where
    A: SeqAccess<'de>,
    T: Deserialize<'de>,
{
    seq.next_element::<T>()?
        .ok_or_else(|| A::Error::custom(format!("missing positional field `{name}`")))
}

/// Validate the msgspec tag string at element 0. A wrong tag means the payload
/// is a different message type — fail loudly instead of misreading fields.
pub(crate) fn expect_tag<'de, A>(seq: &mut A, expected: &'static str) -> Result<(), A::Error>
where
    A: SeqAccess<'de>,
{
    let tag: String = next_field(seq, "_tag")?;
    if tag != expected {
        return Err(A::Error::custom(format!(
            "wrong msgspec tag: expected `{expected}`, got `{tag}`"
        )));
    }
    Ok(())
}

/// Drain positional elements beyond the modeled prefix. TokenSpeed appends new
/// fields at the end of its structs, so unknown trailing elements are skipped
/// rather than treated as a decode error.
pub(crate) fn drain_trailing<'de, A>(seq: &mut A) -> Result<(), A::Error>
where
    A: SeqAccess<'de>,
{
    while seq.next_element::<IgnoredAny>()?.is_some() {}
    Ok(())
}
