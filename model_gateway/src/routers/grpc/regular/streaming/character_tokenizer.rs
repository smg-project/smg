//! The character tokenizer of the streaming tests and the stream benchmark.

use std::sync::atomic::{AtomicUsize, Ordering};

use llm_tokenizer::{
    traits::{Encoding, Tokenizer},
    SpecialTokens,
};

/// A tokenizer whose tokens are characters (`id = char as u32`): the
/// streaming tests' and the stream benchmark's, so an engine stream scripted
/// as text decodes with no model's vocabulary behind it.
#[derive(Default)]
pub(crate) struct CharacterTokenizer {
    special_tokens: SpecialTokens,
    /// How many times the eager decode step ran (a test reads it).
    pub(crate) eager_steps: AtomicUsize,
    /// The ids `eos_token_ids` reports (none by default).
    eos: Vec<u32>,
}

impl CharacterTokenizer {
    /// A tokenizer whose EOS ids are `eos`: the gateway's stop decoder takes
    /// them as stops, as it does a model's. The tests' constructor: the
    /// benchmark decodes with no stop, so the integration build leaves it out.
    #[cfg(test)]
    pub(crate) fn with_eos(eos: Vec<u32>) -> Self {
        Self {
            eos,
            ..Self::default()
        }
    }
}

impl llm_tokenizer::traits::Encoder for CharacterTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
        Ok(Encoding::Plain(text.chars().map(u32::from).collect()))
    }

    fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }
}

impl llm_tokenizer::traits::Decoder for CharacterTokenizer {
    fn decode_step_eager(
        &self,
        token_id: u32,
        ids: &mut Vec<u32>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> anyhow::Result<Option<String>> {
        self.eager_steps.fetch_add(1, Ordering::Relaxed);
        self.decode_step(token_id, ids, prefix, prefix_index, skip_special_tokens)
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
        ids.iter()
            .map(|id| char::from_u32(*id).ok_or_else(|| anyhow::anyhow!("invalid character token")))
            .collect()
    }
}

impl Tokenizer for CharacterTokenizer {
    fn vocab_size(&self) -> usize {
        128
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        token.chars().next().map(u32::from)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        char::from_u32(id).map(|character| character.to_string())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
