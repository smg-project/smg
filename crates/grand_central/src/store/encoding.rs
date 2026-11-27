//! What a worker's registration states of the encode handoff: its end of
//! it ([`EncodeOffer`]), with the form the end takes ([`EncodeHandoff`]).
//! The store's worker record holds the offer
//! ([`WorkerFacts::encode`](super::WorkerFacts::encode)). The rule that
//! reads offers into encode classes is the engine's
//! ([`EncodeClasses`](crate::engine::EncodeClasses)).
//!
//! A connector is free text. The store's door normalizes it once a write
//! ([`EncodeOffer::normalized`]), so a blank one states no connector.

use super::pairing::normalize;

/// The form a worker's registration states for the encode handoff.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncodeHandoff {
    /// No form is stated: the worker has no end of an encode handoff.
    #[default]
    None,
    /// The encoder pushes an encoded item to the prefill over a connector.
    Push,
    /// The encoder and the prefill meet at a rendezvous.
    Rendezvous,
}

/// A worker's end of the encode handoff, as its registration states it. The
/// default states nothing.
///
/// The record gains fields: code outside this crate starts from the default
/// and assigns what it knows, and a pattern of it ends with `..`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct EncodeOffer {
    pub handoff: EncodeHandoff,
    /// The worker states the producer role: an encoder's, under a push.
    pub producer: bool,
    /// The worker states the consumer role: a prefill's, under a push.
    pub consumer: bool,
    /// The connector the role is stated on.
    pub connector: Option<String>,
    /// The worker, an encoder, processes an item's references itself.
    pub references: bool,
}

impl EncodeOffer {
    /// The offer as the rule reads it: a blank connector dropped and a name
    /// read past the blanks around it, by the one function that normalizes
    /// a pairing descriptor's values. The store's door makes a worker's
    /// offer so, once a write.
    #[must_use]
    pub fn normalized(mut self) -> Self {
        normalize(&mut self.connector);
        self
    }
}
