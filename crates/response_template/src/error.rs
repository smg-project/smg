//! Errors. Loading fails where transformers raises (`Invalid`) or on a
//! construct this crate does not reproduce (`Unsupported`); parsing fails with
//! the class of the exception transformers raises.

use std::fmt;

/// Why [`load_response_template`](crate::load_response_template) refused a template.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum LoadError {
    /// transformers rejects this template too.
    #[error("{scope}: {message}")]
    Invalid { scope: String, message: String },
    /// This crate would not parse this template exactly as transformers does
    /// (which may accept it, or reject it for another reason), so it refuses it.
    #[error("{scope}: unsupported {feature}")]
    Unsupported { scope: String, feature: Unsupported },
}

/// What a template uses that this crate does not reproduce.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Unsupported {
    /// A regex construct outside the ported subset (listed in the README).
    Regex(String),
    /// A start anchor or tag pattern that can match the empty string, or a
    /// field whose open and close can both match it (transformers can loop
    /// forever there).
    EmptyMatch,
    /// `content_args` that transformers only reads while parsing and that
    /// fail there or depend on Python typing.
    ContentArgs(&'static str),
    /// A value transformers stores in two places and then changes in place: a
    /// list default of a repeated field, or one value used twice by a transform.
    SharedValue,
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Regex(what) => write!(f, "regex: {what}"),
            Self::EmptyMatch => f.write_str("empty match"),
            Self::ContentArgs(what) => write!(f, "content_args: {what}"),
            Self::SharedValue => f.write_str("shared value"),
        }
    }
}

/// The class of the Python exception transformers raises while parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PyErrorKind {
    Value,
    Key,
    Type,
    Attribute,
    /// JSON nested 512 deep. transformers raises `RecursionError` only near
    /// Python's C recursion limit, about 10,000 deep (see the README).
    Recursion,
}

impl fmt::Display for PyErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Value => "ValueError",
            Self::Key => "KeyError",
            Self::Type => "TypeError",
            Self::Attribute => "AttributeError",
            Self::Recursion => "RecursionError",
        })
    }
}

/// Why [`ResponseParser`](crate::ResponseParser) or
/// [`parse_response`](crate::parse_response) failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ParseError {
    /// transformers raises this exception while closing a region of `field`.
    #[error("field '{field}': {kind}: {message}")]
    Content {
        field: String,
        kind: PyErrorKind,
        message: String,
    },
    /// transformers: "Required response_template fields missing from parsed output".
    #[error("required response_template fields missing from parsed output: {0:?}")]
    MissingRequired(Vec<String>),
    /// transformers returns a value JSON cannot hold (NaN, an infinity, an
    /// integer outside i64 and u64, a string with a lone surrogate). Raised by
    /// the call whose events or message would carry it, and by every later
    /// `finalize` while the message holds it.
    #[error("value not representable in JSON: {0}")]
    Unrepresentable(&'static str),
}

/// A Python exception raised inside the port, before it is tied to a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PyError(pub(crate) PyErrorKind, pub(crate) String);

impl PyError {
    pub(crate) fn value(message: impl Into<String>) -> Self {
        Self(PyErrorKind::Value, message.into())
    }

    pub(crate) fn in_field(self, field: &str) -> ParseError {
        ParseError::Content {
            field: field.to_owned(),
            kind: self.0,
            message: self.1,
        }
    }
}

pub(crate) type PyResult<T> = Result<T, PyError>;

/// A value Python holds and JSON cannot, found while computing a value: the
/// value carries a placeholder of the same Python type and this says why.
pub(crate) type Taint = Option<&'static str>;
