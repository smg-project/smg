//! How a shaped call's values read: the template's parser for a value's text, and the reader that
//! types it, behind one hook.
//!
//! A response template says how a tagged value's text reads before it is typed (its
//! `value_parser`: the text as it is or stripped, JSON, a number, a boolean), and the library
//! that reads the template types the result by the parameter's schema. Neither is the crate's to
//! decide: a table names its reader ([`Format::reads_values_with`]), and every shaped call of the
//! table reads its values through it. [`Inferred`], the default, reads them as the tagged tables
//! do; a crate that implements a template library's own semantics gives its reader instead, so
//! that the compiled table and that library's reader type a value with one implementation.
//!
//! The reader is asked four things: the JSON for a tagged value's text ([`ValueRead::read`]);
//! whether a member of JSON arguments may stream as the model's bytes before it is whole
//! ([`ValueRead::streams`]), and, when it may not, the JSON written for it once it is
//! ([`ValueRead::member`]); and what a key written twice means ([`ValueRead::duplicates`]).
//!
//! [`Format::reads_values_with`]: crate::Format::reads_values_with

use std::{fmt, sync::Arc};

use serde_json::Value;

use super::Dialect;
use crate::tagged::value::{json, kind_of};

/// How a tagged value's text reads before it is typed: a template's `value_parser`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValueParser {
    /// No parser: the text as it is.
    Raw,
    /// `text`: the text, stripped of Python's whitespace at both ends when `strip`.
    Text { strip: bool },
    /// `json`: JSON in `dialect`; with `allow_non_json`, text that is not JSON reads as `text`
    /// does, stripped when `strip`.
    Json {
        dialect: Dialect,
        allow_non_json: bool,
        strip: bool,
    },
    /// `int`: Python's `int` of the text.
    Int,
    /// `float`: Python's `float` of the text.
    Float,
    /// `bool`: whether the text, stripped when `strip`, is `true` or `1` in any case.
    Bool { strip: bool },
}

impl ValueParser {
    /// How a member of JSON arguments reads: as plain JSON.
    pub fn json() -> Self {
        Self::Json {
            dialect: Dialect::default(),
            allow_non_json: false,
            strip: true,
        }
    }
}

/// A parameter's schema as the request's tool declares it: the property's own schema, and the
/// tool's whole `parameters`, where a `$ref` in it points.
#[derive(Clone, Copy, Debug)]
pub struct Schema<'a> {
    /// The tool's `parameters`.
    pub root: &'a Value,
    /// The parameter's own schema, `parameters.properties[key]`.
    pub property: &'a Value,
}

/// What a key written twice in one call means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Duplicates {
    /// The later value is written as a second member, which a JSON reader takes as the last value
    /// at the first one's place, as a Python dict does.
    Last,
    /// The first value stands; a later one is returned as `Malformed`.
    First,
}

/// How a table's shaped calls read their values. A table names its reader and shares it with every
/// engine that reads the table; the crate holds no template library's semantics, only this hook.
pub trait ValueRead: Send + Sync + fmt::Debug {
    /// The JSON for one tagged value's `text`, read by `parser` and typed by its parameter's
    /// `schema` (`None` when the request's tools declare no such parameter). The result is one JSON
    /// value.
    fn read(&self, parser: &ValueParser, text: &str, schema: Option<Schema<'_>>) -> String;

    /// Whether a value of `schema`, read by `parser`, may be sent before it is whole. A string
    /// member of JSON arguments' top level ([`Arguments::Json`](super::Arguments::Json)) is asked
    /// with [`ValueParser::json`]: one that may not is held from its opening quote until it closes,
    /// and written as [`member`](Self::member) says. A tagged value is written whole, when its
    /// parameter ends, whatever this says.
    fn streams(&self, parser: &ValueParser, schema: Option<Schema<'_>>) -> bool;

    /// The JSON written for a string member of JSON arguments that was held until it closed, `raw`
    /// its JSON text, typed by its `schema`; `None` writes the model's bytes.
    fn member(&self, raw: &str, schema: Option<Schema<'_>>) -> Option<String>;

    /// What a key written twice in one call means.
    fn duplicates(&self) -> Duplicates;
}

/// The values of a shaped call as the tagged tables read theirs, the default for every table: a
/// value's text is a declared string as it is, a declared integer when it is one, and otherwise
/// the JSON or Python literal it spells, or the string; the template's parser is not applied. A
/// member of JSON arguments is the model's bytes, sent as they arrive, and a key written twice is
/// written again.
#[derive(Clone, Copy, Debug, Default)]
pub struct Inferred;

impl ValueRead for Inferred {
    fn read(&self, _parser: &ValueParser, text: &str, schema: Option<Schema<'_>>) -> String {
        json(
            text,
            schema.and_then(|schema| kind_of(schema.root, schema.property)),
        )
    }

    fn streams(&self, _parser: &ValueParser, _schema: Option<Schema<'_>>) -> bool {
        true
    }

    fn member(&self, _raw: &str, _schema: Option<Schema<'_>>) -> Option<String> {
        None
    }

    fn duplicates(&self) -> Duplicates {
        Duplicates::Last
    }
}

/// The reader a table names for its values: the default, or one reader the table was given, which
/// the engines that read the table share. Two tables name the same reader when it is one value.
#[derive(Clone)]
pub(crate) enum Values {
    /// [`Inferred`], the default.
    Inferred,
    /// A reader the table was given.
    Reader(Arc<dyn ValueRead>),
}

impl Values {
    /// The reader itself.
    pub(crate) fn reader(&self) -> &dyn ValueRead {
        match self {
            Self::Inferred => &Inferred,
            Self::Reader(reader) => reader.as_ref(),
        }
    }
}

impl PartialEq for Values {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Inferred, Self::Inferred) => true,
            (Self::Reader(this), Self::Reader(that)) => Arc::ptr_eq(this, that),
            _ => false,
        }
    }
}

impl Eq for Values {}

impl fmt::Debug for Values {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.reader().fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::tagged::Declared;

    fn declared() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: value!({
                    "type": "object",
                    "properties": {
                        "s": {"type": "string"},
                        "n": {"type": "integer"},
                        "o": {"type": "object"},
                        "r": {"$ref": "#/$defs/Count"},
                    },
                    "$defs": {"Count": {"type": "integer"}},
                }),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    #[test]
    fn inferred_reads_a_value_as_the_tagged_tables_do_whatever_the_parser() {
        let declared = declared();
        let schema = |key: &str| {
            declared
                .parameter("f", key)
                .map(|(root, property)| Schema { root, property })
        };
        for key in ["s", "n", "o", "r", "x"] {
            for text in ["5", " 5 ", "True", "{'a': 1}", "Paris", "007", ""] {
                let tagged = json(text, declared.kind("f", key));
                for parser in [
                    ValueParser::Raw,
                    ValueParser::Text { strip: true },
                    ValueParser::json(),
                    ValueParser::Int,
                ] {
                    assert_eq!(
                        Inferred.read(&parser, text, schema(key)),
                        tagged,
                        "{key} {text:?} {parser:?}"
                    );
                }
            }
        }
        assert_eq!(
            Inferred.read(&ValueParser::Raw, "007", schema("r")),
            "7",
            "a reference is followed into the tool's parameters"
        );
        assert!(Inferred.streams(&ValueParser::json(), schema("n")));
        assert_eq!(Inferred.member("\"5\"", schema("n")), None);
        assert_eq!(Inferred.duplicates(), Duplicates::Last);
    }

    #[derive(Debug)]
    struct Other;

    impl ValueRead for Other {
        fn read(&self, _parser: &ValueParser, text: &str, _schema: Option<Schema<'_>>) -> String {
            text.to_string()
        }

        fn streams(&self, _parser: &ValueParser, _schema: Option<Schema<'_>>) -> bool {
            false
        }

        fn member(&self, _raw: &str, _schema: Option<Schema<'_>>) -> Option<String> {
            None
        }

        fn duplicates(&self) -> Duplicates {
            Duplicates::First
        }
    }

    #[test]
    fn a_reader_is_the_same_reader_when_it_is_one_value() {
        let other: Arc<dyn ValueRead> = Arc::new(Other);
        assert_eq!(Values::Inferred, Values::Inferred);
        assert_eq!(Values::Reader(other.clone()), Values::Reader(other.clone()));
        assert_ne!(
            Values::Reader(Arc::new(Other)),
            Values::Reader(other.clone())
        );
        assert_ne!(Values::Reader(other.clone()), Values::Inferred);
        assert_eq!(format!("{:?}", Values::Reader(other)), "Other");
    }
}
