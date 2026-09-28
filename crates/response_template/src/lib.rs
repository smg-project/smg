//! Tokenizer-declared response templates.
//!
//! A checkpoint can describe its output format in the `response_template`
//! object of `tokenizer_config.json`: regex openers and literal closes for a
//! `thinking`, a `content` and a repeated `tool_calls` field. This crate
//! validates that object and provides the matching primitives that the
//! template reasoning parser and the template tool parser share.

mod compiled;
mod error;
mod scan;
mod schema;

pub use compiled::{CompiledTemplate, Field, Tag};
pub use error::TemplateError;
pub use scan::{consume, find_close, partial_close_len, Opener};
pub use schema::{
    ClosePattern, ContentArgs, FieldTemplate, ResponseTemplate, Transform, ValueParser,
    ValueParserArgs,
};
