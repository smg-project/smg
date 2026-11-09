//! Decoding helpers for the dialects whose messages are tagged positional
//! msgpack arrays (`msgspec.Struct(array_like=True)` on the engine side):
//! read a required element, check the class-name tag at element 0, and skip
//! the trailing fields a newer engine may append.

use std::collections::BTreeMap;

use serde::de::{Deserialize, Error as _, IgnoredAny, SeqAccess};

use crate::error;

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

/// Compare the field order this side encodes for `struct_name` (`sent`, the
/// prefix its encoder fills) with the order the engine declared for it at
/// the handshake. A struct the engine did not declare passes: there is
/// nothing to compare against. One it declared passes only when every sent
/// field sits at the same index on the engine's side; the error names the
/// struct and the first field that differs, or the engine's shorter array,
/// so a mismatch is read once at connect time rather than off every request
/// the engine refuses.
pub(crate) fn check_prefix(
    struct_name: &str,
    sent: &[&str],
    declared: &BTreeMap<String, Vec<String>>,
) -> error::Result<()> {
    let Some(engine) = declared.get(struct_name) else {
        return Ok(());
    };
    for (index, field) in sent.iter().enumerate() {
        let message = match engine.get(index) {
            Some(theirs) if theirs == field => continue,
            Some(theirs) => format!(
                "`{struct_name}` field {index} is `{field}` on this side and `{theirs}` on the \
                 engine's (the engine declares {} fields, this side fills {})",
                engine.len(),
                sent.len()
            ),
            None => format!(
                "`{struct_name}` has {} fields on the engine's side, this side fills {} (field \
                 {index}, `{field}`, has no slot there)",
                engine.len(),
                sent.len()
            ),
        };
        return Err(error::Error::WireLayoutMismatch { message });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared(fields: &[&str]) -> BTreeMap<String, Vec<String>> {
        BTreeMap::from([(
            "S".to_string(),
            fields.iter().map(|field| field.to_string()).collect(),
        )])
    }

    #[test]
    fn a_prefix_the_engine_declares_in_the_same_order_passes() {
        assert!(check_prefix("S", &["a", "b"], &declared(&["a", "b", "c"])).is_ok());
        assert!(
            check_prefix("S", &["a", "b"], &BTreeMap::new()).is_ok(),
            "nothing declared: nothing to compare against"
        );
        assert!(
            check_prefix("T", &["x"], &declared(&["a"])).is_ok(),
            "another struct declared"
        );
    }

    #[test]
    fn a_field_that_moved_or_a_shorter_array_is_named() {
        let moved = check_prefix("S", &["a", "b", "c"], &declared(&["a", "x", "b", "c"]))
            .unwrap_err()
            .to_string();
        assert!(
            moved.contains("`S` field 1 is `b` on this side and `x` on the engine's"),
            "{moved}"
        );
        assert!(
            moved.contains("declares 4 fields, this side fills 3"),
            "{moved}"
        );
        let short = check_prefix("S", &["a", "b", "c"], &declared(&["a", "b"]))
            .unwrap_err()
            .to_string();
        assert!(
            short.contains(
                "`S` has 2 fields on the engine's side, this side fills 3 (field 2, `c`, has no \
                 slot there)"
            ),
            "{short}"
        );
    }
}
