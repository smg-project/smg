//! A model output format: the definition from which a parser and its grammar are built.
//!
//! The full definition (terminals, states, transitions, argument syntax) arrives with the engine.
//! This type exists now so the public surface is complete; its fields are private so it can grow.

/// The definition of one format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Format {
    name: String,
}

impl Format {
    /// A format known only by name; everything else follows.
    pub fn named(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// The format's name, as used in configuration and fixtures.
    pub fn name(&self) -> &str {
        &self.name
    }
}
