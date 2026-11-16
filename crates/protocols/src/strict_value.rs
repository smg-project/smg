//! A JSON value read with every repeated key refused.
//!
//! [`Value`]'s own reader keeps the last of a repeated key at every depth, so a
//! hand-written reader that buffers a value before its typed read hides the
//! repeat the typed read would refuse. Such a reader buffers through this one.
//! It refuses every repeat, also where the typed read would keep the last value
//! (a map, a free-form value) or pass over the key (one it does not know). A
//! reader with one member whose repeats it must keep names that key to
//! [`strict_entries`]: the member's value is read by [`Value`]'s own reader.
//! The refusal is in serde's words for a duplicate field; its place is the
//! deserializer's to name, as it is for the derived readers.

use std::fmt;

use serde::de::{DeserializeSeed, Deserializer, Error, MapAccess, SeqAccess, Visitor};
use serde_json::{map::Entry, Map, Number, Value};

/// The entries of an object, each value read by [`StrictValue`] but the value
/// of the key `plain` names, which [`Value`]'s own reader reads. A repeated
/// key is refused when it is met, before its value is read. One map slot per
/// key, taken before the value is read and filled after: the read costs what
/// the plain value read costs.
pub(crate) fn strict_entries<'de, A: MapAccess<'de>>(
    entries: &mut A,
    plain: Option<&str>,
) -> Result<Map<String, Value>, A::Error> {
    let mut fields = Map::new();
    while let Some(key) = entries.next_key::<String>()? {
        let slot = match fields.entry(key) {
            Entry::Occupied(slot) => {
                return Err(Error::custom(format!("duplicate field `{}`", slot.key())));
            }
            Entry::Vacant(slot) => slot,
        };
        let value = if plain == Some(slot.key().as_str()) {
            entries.next_value()?
        } else {
            entries.next_value_seed(StrictValue)?
        };
        slot.insert(value);
    }
    Ok(fields)
}

/// Builds what [`Value`]'s own reader builds (the same scalars, the same
/// ordered map) and refuses a key repeated in an object at any depth.
pub(crate) struct StrictValue;

impl<'de> DeserializeSeed<'de> for StrictValue {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for StrictValue {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("any valid JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Value, E> {
        Ok(Number::from_f64(value).map_or(Value::Null, Value::Number))
    }

    fn visit_str<E>(self, value: &str) -> Result<Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Value, E> {
        Ok(Value::String(value))
    }

    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        deserializer.deserialize_any(self)
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut elements: A) -> Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = elements.next_element_seed(StrictValue)? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut entries: A) -> Result<Value, A::Error> {
        strict_entries(&mut entries, None).map(Value::Object)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The strict reader builds every accepted value as the plain value
    /// reader does: the same scalars, the same key order, nested as deep as
    /// the input goes.
    #[test]
    fn the_strict_value_reader_builds_an_accepted_value_as_the_plain_reader_does() {
        let text = r#"{"z":1,"a":[true,null,-2,3.5,1e3,"s",{"y":{"x":[[]]},"b":{}}],"m":18446744073709551615}"#;
        let plain: Value = serde_json::from_str(text).expect("the plain reader accepts it");
        let mut deserializer = serde_json::Deserializer::from_str(text);
        let strict = DeserializeSeed::deserialize(StrictValue, &mut deserializer)
            .expect("the strict reader accepts it");
        assert_eq!(strict, plain);
        assert_eq!(strict.to_string(), plain.to_string());
    }
}
