//! Outbound proxy body construction for typed requests.
//!
//! Serializes the typed request straight to bytes and edits the top-level
//! object as borrowed [`RawValue`] slices, so token-heavy payloads
//! (`input_ids`, messages) are never materialized as a `serde_json::Value`
//! tree. The DP-aware rank and the PD router's bootstrap, rank and KV-handoff
//! fields are inserted the same way (see [`RawBody`]). Only a body the raw
//! editor cannot parse (in practice: a non-object) takes the `Value` path,
//! whose bytes every raw edit reproduces.

use std::borrow::Cow;

use serde::{
    de::{self, MapAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_json::value::{to_raw_value, RawValue};
use smg_worker::identity::{basic::Worker, error::WorkerError};

use super::thinking_budget::{self, BudgetWire};
use crate::routers::common::{
    request_to_value, serialize_json_sized,
    sglang_fields::{
        is_stripped_sglang_default, may_carry_stripped_sglang_default, strip_default_sglang_fields,
        SGLANG_FIELDS,
    },
};

#[derive(Debug)]
pub(crate) enum RequestBodyError {
    ThinkingBudget(String),
    Serialize(serde_json::Error),
    Prepare(WorkerError),
}

/// The field the built-in [`Worker::prepare_request`] adds for a DP-aware
/// worker.
const DATA_PARALLEL_RANK: &str = "data_parallel_rank";

/// Serialize a typed request into the exact bytes the `Value`-mediated
/// pipeline (`to_value` → model rewrite → `prepare_request` → strip →
/// `to_vec`) produces.
pub(crate) fn serialize_request_body<T: Serialize>(
    typed_req: &T,
    canonical_model: Option<&str>,
    worker: &dyn Worker,
    raw_len: Option<usize>,
) -> Result<Vec<u8>, RequestBodyError> {
    serialize_request_body_with_edits(typed_req, canonical_model, worker, raw_len, true, None)
}

/// Apply the selected worker's wire edits to the same raw body used for model
/// aliases and DP rank. Decisions and SystemOne preserve their backend fields.
pub(super) fn serialize_request_body_with_edits<T: Serialize>(
    typed_req: &T,
    canonical_model: Option<&str>,
    worker: &dyn Worker,
    raw_len: Option<usize>,
    strip_defaults: bool,
    budget: Option<i64>,
) -> Result<Vec<u8>, RequestBodyError> {
    let edit = budget
        .map(|budget| {
            (
                thinking_budget::wire(worker.metadata().spec.runtime_type),
                budget,
            )
        })
        .filter(|(wire, _)| *wire != BudgetWire::Passthrough);
    let value_fallback =
        || value_request_body(typed_req, canonical_model, worker, raw_len, strip_defaults);
    let mutates = worker.mutates_request();
    let rank = if mutates && worker.uses_builtin_prepare_request() {
        worker.dp_rank()
    } else {
        None
    };
    // A custom hook owns the model rewrite and default stripping in the Value
    // path. Apply the budget to its output, in the same order as for built-ins.
    let (bytes, canonical_model, strip_defaults) = if mutates && rank.is_none() {
        let bytes = value_fallback()?;
        if edit.is_none() {
            return Ok(bytes);
        }
        (bytes, None, false)
    } else {
        let bytes =
            serialize_json_sized(typed_req, raw_len).map_err(RequestBodyError::Serialize)?;
        (bytes, canonical_model, strip_defaults)
    };
    if edit.is_none()
        && canonical_model.is_none()
        && rank.is_none()
        && bytes.first() == Some(&b'{')
        && !(strip_defaults && may_carry_stripped_sglang_default(&bytes))
    {
        return Ok(bytes);
    }

    let mut body = match RawBody::parse(&bytes) {
        Ok(body) => body,
        Err(e) if edit.is_some() => return Err(RequestBodyError::Serialize(e)),
        Err(_) => return value_fallback(),
    };
    let mut extra = 0;
    if let Some(model) = canonical_model {
        let model = to_raw_value(model).map_err(RequestBodyError::Serialize)?;
        extra += model.get().len();
        body.set_model(model);
    }
    if let Some(rank) = rank {
        let rank = to_raw_value(&rank).map_err(RequestBodyError::Serialize)?;
        extra += DATA_PARALLEL_RANK.len() + rank.get().len() + 4;
        body.insert(DATA_PARALLEL_RANK, rank);
    }
    if strip_defaults {
        body.strip_default_sglang_fields();
    }
    if let Some((wire, budget)) = edit {
        thinking_budget::translate(&mut body, wire, budget)
            .map_err(RequestBodyError::ThinkingBudget)?;
    }
    if body.mutated() {
        let mut out = Vec::with_capacity(bytes.len() + extra);
        serde_json::to_writer(&mut out, &body).map_err(RequestBodyError::Serialize)?;
        Ok(out)
    } else {
        Ok(bytes)
    }
}

/// The `Value` pipeline, kept for bodies [`RawBody`] cannot parse and for a
/// worker whose `prepare_request` is not the built-in DP-rank edit.
fn value_request_body<T: Serialize>(
    typed_req: &T,
    canonical_model: Option<&str>,
    worker: &dyn Worker,
    raw_len: Option<usize>,
    strip_defaults: bool,
) -> Result<Vec<u8>, RequestBodyError> {
    let mut json_val = request_to_value(typed_req, raw_len).map_err(RequestBodyError::Serialize)?;
    if let Some(canonical_model) = canonical_model {
        super::set_request_model(&mut json_val, canonical_model);
    }
    let mut json_val = worker
        .prepare_request(json_val)
        .map_err(RequestBodyError::Prepare)?;
    if strip_defaults {
        strip_default_sglang_fields(&mut json_val);
    }
    serialize_json_sized(&json_val, raw_len).map_err(RequestBodyError::Serialize)
}

/// Top-level fields of a serialized JSON object; values stay borrowed raw
/// JSON until an edit replaces one.
///
/// Every edit does what the same edit does to a `serde_json::Map` under
/// `preserve_order`: [`Self::insert`] replaces in place or appends,
/// [`Self::remove`] swap-removes (the last field moves into the freed slot).
/// A body built here is therefore byte-identical to one built through a
/// `Value`, which the tests use as the oracle. Cloning copies the pointers
/// to the borrowed names and slices, not the payload; only an owned entry
/// (a value an edit already replaced, such as the model rewrite, or a name
/// the parser had to unescape) is copied, which is what the PD router's two
/// sides need.
#[derive(Clone)]
pub(crate) struct RawBody<'a> {
    fields: Vec<(Cow<'a, str>, Cow<'a, RawValue>)>,
    mutated: bool,
}

impl<'a> RawBody<'a> {
    /// Parse the top level of a JSON object; anything else is an error.
    pub(crate) fn parse(bytes: &'a [u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }

    /// Whether an edit changed the body since it was parsed. A body that
    /// did not change can go out as the bytes it was parsed from.
    pub(crate) fn mutated(&self) -> bool {
        self.mutated
    }

    /// The raw JSON of a top-level field.
    pub(crate) fn get(&self, name: &str) -> Option<&RawValue> {
        self.fields
            .iter()
            .find(|(field, _)| field == name)
            .map(|(_, value)| value.as_ref())
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.fields.iter().any(|(field, _)| field == name)
    }

    /// Mirrors [`super::set_request_model`]: only replaces an existing
    /// field. A body that already names the canonical model is left as it
    /// was parsed.
    pub(crate) fn set_model(&mut self, canonical_model: Box<RawValue>) {
        if let Some((_, value)) = self.fields.iter_mut().find(|(name, _)| name == "model") {
            if value.get() != canonical_model.get() {
                *value = Cow::Owned(canonical_model);
                self.mutated = true;
            }
        }
    }

    /// `serde_json::Map::insert` under `preserve_order`: an existing field
    /// keeps its position, a new one goes last. A value equal to the one
    /// present changes nothing, so the body stays unmutated. `name` outlives
    /// the body (a constant, as every caller's is) and is kept as it is.
    pub(crate) fn insert(&mut self, name: &'a str, value: Box<RawValue>) {
        match self.fields.iter_mut().find(|(field, _)| field == name) {
            Some((_, slot)) => {
                if slot.get() == value.get() {
                    return;
                }
                *slot = Cow::Owned(value);
            }
            None => self.fields.push((Cow::Borrowed(name), Cow::Owned(value))),
        }
        self.mutated = true;
    }

    /// `serde_json::Map::remove` under `preserve_order`: the last field moves
    /// into the removed one's slot. Returns whether the field was present.
    pub(crate) fn remove(&mut self, name: &str) -> bool {
        match self.fields.iter().position(|(field, _)| field == name) {
            Some(index) => {
                self.fields.swap_remove(index);
                self.mutated = true;
                true
            }
            None => false,
        }
    }

    /// Mirrors [`strip_default_sglang_fields`], including the swap-remove
    /// ordering of `serde_json::Map::remove` under `preserve_order`.
    pub(crate) fn strip_default_sglang_fields(&mut self) {
        for field in SGLANG_FIELDS {
            let index = self.fields.iter().position(|(name, value)| {
                name == field && is_stripped_sglang_default(field, value.get())
            });
            if let Some(index) = index {
                self.fields.swap_remove(index);
                self.mutated = true;
            }
        }
    }

    /// The wire bytes, into a buffer pre-sized from the incoming request
    /// length (see [`serialize_json_sized`]).
    pub(crate) fn to_bytes(&self, raw_len: Option<usize>) -> serde_json::Result<Vec<u8>> {
        serialize_json_sized(self, raw_len)
    }
}

/// A top-level field's name: borrowed from the body's bytes where the
/// parser lends them (a name without an escape, as every field a typed
/// request declares is), owned where it had to unescape one (a client's key
/// kept in a flattened `other` map can carry an escape).
struct FieldName<'a>(Cow<'a, str>);

impl<'de> Deserialize<'de> for FieldName<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FieldNameVisitor;

        impl<'de> Visitor<'de> for FieldNameVisitor {
            type Value = FieldName<'de>;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a field name")
            }

            fn visit_borrowed_str<E: de::Error>(self, name: &'de str) -> Result<Self::Value, E> {
                Ok(FieldName(Cow::Borrowed(name)))
            }

            fn visit_str<E: de::Error>(self, name: &str) -> Result<Self::Value, E> {
                Ok(FieldName(Cow::Owned(name.to_owned())))
            }

            fn visit_string<E: de::Error>(self, name: String) -> Result<Self::Value, E> {
                Ok(FieldName(Cow::Owned(name)))
            }
        }

        deserializer.deserialize_str(FieldNameVisitor)
    }
}

impl<'de> Deserialize<'de> for RawBody<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RawBodyVisitor;

        impl<'de> Visitor<'de> for RawBodyVisitor {
            type Value = RawBody<'de>;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut fields = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some((FieldName(name), value)) =
                    map.next_entry::<FieldName<'de>, &'de RawValue>()?
                {
                    fields.push((name, Cow::Borrowed(value)));
                }
                Ok(RawBody {
                    fields,
                    mutated: false,
                })
            }
        }

        deserializer.deserialize_map(RawBodyVisitor)
    }
}

impl Serialize for RawBody<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.fields.len()))?;
        for (name, value) in &self.fields {
            map.serialize_entry(name, value.as_ref())?;
        }
        map.end()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/routers/http/request_body/tests.rs"]
mod tests;
