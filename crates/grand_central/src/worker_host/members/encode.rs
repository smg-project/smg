//! The columns an encode request's decision reads beside a generation's
//! encoder list and its prefill list.

use super::Members;
use crate::engine::EncoderFacts;

/// What an encode request's decision reads beside a generation's encoders
/// and beside its prefills, each by a worker's place in its list AS LENT:
/// the two columns a host hands with those lists
/// ([`EncodeLists::facts`](crate::engine::EncodeLists::facts),
/// [`EncodeLists::prefill_classes`](crate::engine::EncodeLists::prefill_classes)).
///
/// Each column holds one entry a worker of its list, so the engine takes
/// both as they are. The default is the columns of two empty lists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EncodeColumns {
    /// What the encode side reads of each encoder, in the encoder list's
    /// order.
    facts: Box<[EncoderFacts]>,
    /// The encode class each prefill names, in the prefill list's order.
    prefill_classes: Box<[u32]>,
}

impl EncodeColumns {
    /// The columns of an encoder list and of a prefill list: one entry a
    /// worker of each, in its list's order.
    pub(in crate::worker_host) fn new(
        facts: Box<[EncoderFacts]>,
        prefill_classes: Box<[u32]>,
    ) -> Self {
        Self {
            facts,
            prefill_classes,
        }
    }

    /// What the encode side reads of each encoder beside its record: its
    /// encode class, whether its registration states a push and whether it
    /// takes an item's references. One entry an encoder, by its place in
    /// the list lent.
    #[must_use]
    pub fn facts(&self) -> &[EncoderFacts] {
        &self.facts
    }

    /// The encode class each prefill names: `0` for a prefill no encoder
    /// hands to. One entry a prefill, by its place in the list lent.
    #[must_use]
    pub fn prefill_classes(&self) -> &[u32] {
        &self.prefill_classes
    }
}

impl Members {
    /// The encode columns of the generation's encoder and prefill lists.
    /// `derive` derives them where none stand. Asks that meet there wait
    /// for the one that derives, and read the same columns.
    pub(in crate::worker_host) fn encode_columns(
        &self,
        derive: impl FnOnce() -> EncodeColumns,
    ) -> &EncodeColumns {
        self.encode.get_or_init(derive)
    }
}
