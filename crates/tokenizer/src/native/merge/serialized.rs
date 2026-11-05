//! The merges of a BPE model, read through its `Serialize` impl and nothing
//! else of it. The model's merge map is private; its serializer writes the
//! merges in rank order, but it also writes the vocabulary, by walking every
//! id up to the largest and warning about each id without a token, which a
//! vocabulary with gaps would log at every load. This serializer takes the
//! `merges` field alone and skips the rest.

use std::fmt;

use serde::{
    ser::{Impossible, SerializeStruct},
    Serialize, Serializer,
};
use tokenizers::models::bpe::BPE;

/// The model's `merges` field as JSON (`[[a, b], ...]` in rank order).
pub(super) fn merges_json(model: &BPE) -> Result<String, String> {
    merges_of(model)
}

/// The `merges` field of a struct that serializes one, as JSON; the values
/// of its other fields are never serialized.
fn merges_of(model: &impl Serialize) -> Result<String, String> {
    match model.serialize(MergesOnly) {
        Ok(Some(merges)) => Ok(merges),
        Ok(None) => Err("the model's serialized form has no merges field".to_owned()),
        Err(NotAModel(reason)) => Err(reason),
    }
}

/// A serializer that accepts one struct and keeps its `merges` field.
struct MergesOnly;

/// The struct being serialized: the `merges` field once it has been seen.
struct Fields(Option<String>);

/// Anything but a struct, or a field that does not serialize to JSON.
#[derive(Debug)]
struct NotAModel(String);

impl fmt::Display for NotAModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NotAModel {}

impl serde::ser::Error for NotAModel {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Self(msg.to_string())
    }
}

impl SerializeStruct for Fields {
    type Ok = Option<String>;
    type Error = NotAModel;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), NotAModel> {
        if key == "merges" {
            self.0 = Some(serde_json::to_string(value).map_err(serde::ser::Error::custom)?);
        }
        Ok(())
    }

    fn end(self) -> Result<Option<String>, NotAModel> {
        Ok(self.0)
    }
}

/// Every serializer method but `serialize_struct` refuses: the model is a
/// struct, and nothing else reaches this serializer.
macro_rules! refuse {
    ($($method:ident($($arg:ty),*)),* $(,)?) => {
        $(fn $method(self, $(_: $arg),*) -> Result<Option<String>, NotAModel> {
            Err(NotAModel(concat!("the model is not a struct (", stringify!($method), ")").to_owned()))
        })*
    };
}

impl Serializer for MergesOnly {
    type Ok = Option<String>;
    type Error = NotAModel;
    type SerializeSeq = Impossible<Option<String>, NotAModel>;
    type SerializeTuple = Impossible<Option<String>, NotAModel>;
    type SerializeTupleStruct = Impossible<Option<String>, NotAModel>;
    type SerializeTupleVariant = Impossible<Option<String>, NotAModel>;
    type SerializeMap = Impossible<Option<String>, NotAModel>;
    type SerializeStruct = Fields;
    type SerializeStructVariant = Impossible<Option<String>, NotAModel>;

    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Fields, NotAModel> {
        Ok(Fields(None))
    }

    refuse!(
        serialize_bool(bool),
        serialize_i8(i8),
        serialize_i16(i16),
        serialize_i32(i32),
        serialize_i64(i64),
        serialize_u8(u8),
        serialize_u16(u16),
        serialize_u32(u32),
        serialize_u64(u64),
        serialize_f32(f32),
        serialize_f64(f64),
        serialize_char(char),
        serialize_str(&str),
        serialize_bytes(&[u8]),
        serialize_none(),
        serialize_unit(),
        serialize_unit_struct(&'static str),
        serialize_unit_variant(&'static str, u32, &'static str),
    );

    fn serialize_some<T: ?Sized + Serialize>(self, _: &T) -> Result<Option<String>, NotAModel> {
        Err(NotAModel(
            "the model is not a struct (serialize_some)".to_owned(),
        ))
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        _: &T,
    ) -> Result<Option<String>, NotAModel> {
        Err(NotAModel(
            "the model is not a struct (serialize_newtype_struct)".to_owned(),
        ))
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: &T,
    ) -> Result<Option<String>, NotAModel> {
        Err(NotAModel(
            "the model is not a struct (serialize_newtype_variant)".to_owned(),
        ))
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, NotAModel> {
        Err(NotAModel(
            "the model is not a struct (serialize_seq)".to_owned(),
        ))
    }

    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, NotAModel> {
        Err(NotAModel(
            "the model is not a struct (serialize_tuple)".to_owned(),
        ))
    }

    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, NotAModel> {
        Err(NotAModel(
            "the model is not a struct (serialize_tuple_struct)".to_owned(),
        ))
    }

    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, NotAModel> {
        Err(NotAModel(
            "the model is not a struct (serialize_tuple_variant)".to_owned(),
        ))
    }

    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, NotAModel> {
        Err(NotAModel(
            "the model is not a struct (serialize_map)".to_owned(),
        ))
    }

    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, NotAModel> {
        Err(NotAModel(
            "the model is not a struct (serialize_struct_variant)".to_owned(),
        ))
    }
}

#[cfg(test)]
mod tests;
