//! The grammar spelled for the grammar backend a worker's engine runs: the one place a [`Grammar`]
//! becomes the text a request carries. xgrammar takes the structural tag the tree serializes to
//! ([`Grammar::payload_text`]), in the vocabulary the worker's compiler reads
//! ([`Grammar::in_older_vocabulary`] for an older one); outlines takes a JSON schema alone, as its
//! text. The other backends the engines run take other spellings of the same tree, each added
//! beside this one; a backend with no spelling for a grammar refuses it by name here, before the
//! request leaves, instead of sending a text its compiler would reject with its own words after
//! the prompt ran.

use std::fmt;

use openai_protocol::worker::TagVocabulary;

use super::Grammar;

mod lark;

pub use self::lark::{AddedTokens, Inexpressible, Lark};

/// The grammar backend a worker's engine compiles constraints on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GrammarBackend {
    /// Structural tags, JSON schemas, regular expressions and EBNF: the tables' tags as derived.
    #[default]
    Xgrammar,
    /// Lark grammars, JSON schemas and regular expressions.
    Llguidance,
    /// JSON schemas and regular expressions.
    Outlines,
}

impl GrammarBackend {
    /// The backend's name, as the engines spell it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Xgrammar => "xgrammar",
            Self::Llguidance => "llguidance",
            Self::Outlines => "outlines",
        }
    }
}

impl fmt::Display for GrammarBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A grammar spelled for one backend: the constraint the request carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Spelling {
    /// The structural tag, written straight from the tree.
    StructuralTag(String),
    /// A JSON schema alone, written as the text the engines' JSON-schema constraint takes: the
    /// outlines backend's one spelling, for an answer the engine starts at the schema.
    JsonSchema(String),
}

/// A grammar the backend has no spelling for. The request is refused with the backend's name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unspelled {
    /// The backend without a spelling.
    pub backend: GrammarBackend,
}

impl fmt::Display for Unspelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the {} grammar backend has no spelling of the grammar",
            self.backend
        )
    }
}

impl std::error::Error for Unspelled {}

impl Grammar {
    /// The grammar spelled for `backend`, in the `vocabulary` its compiler reads: for xgrammar
    /// the structural tag [`Grammar::payload_text`] writes, byte for byte, spelled without the
    /// newer nodes for an older compiler ([`Grammar::in_older_vocabulary`]); for outlines a JSON
    /// schema alone, as its text ([`Grammar::json_schema_alone`]). A backend with no spelling of
    /// the tree refuses it by name.
    pub fn spelled(
        self,
        backend: GrammarBackend,
        vocabulary: TagVocabulary,
    ) -> Result<Spelling, Unspelled> {
        match backend {
            GrammarBackend::Xgrammar => Ok(Spelling::StructuralTag(match vocabulary {
                TagVocabulary::Current => self.payload_text(),
                TagVocabulary::Older => self.in_older_vocabulary().payload_text(),
            })),
            GrammarBackend::Outlines => self
                .json_schema_alone()
                .map(Spelling::JsonSchema)
                .ok_or(Unspelled { backend }),
            GrammarBackend::Llguidance => Err(Unspelled { backend }),
        }
    }

    /// The text of a grammar that is one JSON schema alone, in the plain JSON style and with no
    /// excluded words: the one tree the outlines backend has a spelling of, since it builds a
    /// regular expression from a schema and nothing wider (no text around the schema, no tag, no
    /// words held out of its strings). Its whitespace bound is not written: that backend keeps
    /// its own whitespace pattern, as an older xgrammar does.
    pub fn json_schema_alone(&self) -> Option<String> {
        match self {
            Self::JsonSchema {
                schema,
                style,
                excludes,
                ..
            } if style.as_deref().is_none_or(|style| style == "json") && excludes.is_empty() => {
                Some(schema.to_string())
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
