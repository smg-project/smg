//! A JSON value read with every repeated key refused.
//!
//! [`Value`]'s own reader keeps the last of a repeated key at every depth, so a
//! hand-written reader that buffers a value before its typed read hides the
//! repeat the typed read would refuse. Such a reader buffers through this one.
//! It refuses every repeat, also where the typed read would keep the last value
//! (a map, a free-form value) or pass over the key (one it does not know): a
//! reader that must keep those reads them itself.
//! The refusal is in serde's words for a duplicate field; its place is the
//! deserializer's to name, as it is for the derived readers.

use std::{collections::HashSet, fmt};

use serde::de::{DeserializeSeed, Deserializer, Error, MapAccess, SeqAccess, Visitor};
use serde_json::{map::Entry, Map, Number, Value};

/// The entries of an object, each value read by [`StrictValue`]: the read
/// costs what the plain value read costs.
pub(crate) fn strict_entries<'de, A: MapAccess<'de>>(
    entries: &mut A,
) -> Result<Map<String, Value>, A::Error> {
    entries_with(entries, |_, entries| entries.next_value_seed(StrictValue))
}

/// The entries of an object, each value read by `read`, which is given the
/// value's key. A repeated key is refused when it is met, before its value is
/// read. One map slot per key, taken before the value is read and filled
/// after.
pub(crate) fn entries_with<'de, A, F>(
    entries: &mut A,
    mut read: F,
) -> Result<Map<String, Value>, A::Error>
where
    A: MapAccess<'de>,
    F: FnMut(&str, &mut A) -> Result<Value, A::Error>,
{
    let mut fields = Map::new();
    while let Some(key) = entries.next_key::<String>()? {
        let slot = match fields.entry(key) {
            Entry::Occupied(slot) => {
                return Err(Error::custom(format!("duplicate field `{}`", slot.key())));
            }
            Entry::Vacant(slot) => slot,
        };
        let value = read(slot.key().as_str(), entries)?;
        slot.insert(value);
    }
    Ok(fields)
}

/// The entries of an object in arrival order, each value read by
/// [`StrictValue`], for a reader that walks the few keys of one object: no
/// map is built. A repeated key is refused when it is met, as
/// [`strict_entries`] refuses it. A walk over the keys met so far tells a
/// repeat while they are few; past [`WALKED_KEYS`] a hashed set does, so no
/// object makes the check quadratic. The entries start with room for eight,
/// which holds most such objects in one allocation; the set is built only
/// for an object that needs it.
pub(crate) fn strict_fields<'de, A: MapAccess<'de>>(
    entries: &mut A,
) -> Result<Vec<(String, Value)>, A::Error> {
    let mut fields: Vec<(String, Value)> = Vec::with_capacity(8);
    let mut hashed: Option<HashSet<String>> = None;
    while let Some(key) = entries.next_key::<String>()? {
        let repeated = if fields.len() < WALKED_KEYS {
            fields.iter().any(|(known, _)| *known == key)
        } else {
            let hashed = hashed
                .get_or_insert_with(|| fields.iter().map(|(known, _)| known.clone()).collect());
            !hashed.insert(key.clone())
        };
        if repeated {
            return Err(Error::custom(format!("duplicate field `{key}`")));
        }
        fields.push((key, entries.next_value_seed(StrictValue)?));
    }
    Ok(fields)
}

/// More keys than any object read by [`strict_fields`] has by its shape.
const WALKED_KEYS: usize = 16;

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
        strict_entries(&mut entries).map(Value::Object)
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

    /// The ordered reader keeps arrival order and refuses a repeated key:
    /// among few keys (told by the walk), among many (by the hashed set)
    /// and below the object's own keys.
    #[test]
    fn the_ordered_reader_keeps_arrival_order_and_refuses_a_repeated_key() {
        struct Fields;

        impl<'de> Visitor<'de> for Fields {
            type Value = Vec<(String, Value)>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("an object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut entries: A) -> Result<Self::Value, A::Error> {
                strict_fields(&mut entries)
            }
        }

        let read = |text: &str| {
            let mut deserializer = serde_json::Deserializer::from_str(text);
            deserializer.deserialize_map(Fields)
        };
        let last = 2 * WALKED_KEYS - 1;
        let many: String = (0..=last)
            .map(|key| format!(r#""k{key}":{key},"#))
            .collect();
        let fields = read(&format!(r#"{{{many}"z":{{"a":1}}}}"#)).expect("no key is repeated");
        let names: Vec<&str> = fields.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names.len(), last + 2);
        assert_eq!((names[0], names[last + 1]), ("k0", "z"));
        assert_eq!(names[last], format!("k{last}"));
        for (text, key) in [
            (r#"{"a":1,"b":2,"a":3}"#.to_string(), "a".to_string()),
            (format!(r#"{{{many}"k1":0}}"#), "k1".to_string()),
            (format!(r#"{{{many}"k{last}":0}}"#), format!("k{last}")),
            (r#"{"a":{"b":1,"b":2}}"#.to_string(), "b".to_string()),
        ] {
            let error = read(&text).expect_err("a repeated key");
            assert!(
                error
                    .to_string()
                    .starts_with(&format!("duplicate field `{key}`")),
                "{error}"
            );
        }
    }
}
