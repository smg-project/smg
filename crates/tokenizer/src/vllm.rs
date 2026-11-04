//! The opt-in `vllm-tokenizer` backend (feature `vllm`): the vLLM Rust
//! frontend's Hugging Face tokenizer serves a tokenizer's flat encodes and
//! batch decodes in place of `tokenizers` and the direct encoders. Model
//! loading and configuration, the vocabulary maps, chat templates, the
//! renderers with encoders of their own and the streaming step stay in the
//! wrapper, so there is one source of semantics (the shape of the v1
//! opt-in, see `v1`).
//!
//! The backend reads `tokenizer.json` alone, so the load hands it the file
//! only where the wrapper's pipeline is the file's: a tokenizer built in
//! memory, a vocabulary read from `vocab.json` and `merges.txt`, the Llama
//! class's pipeline installed over a legacy file, a post-processor the
//! configuration prescribes, or added tokens settled beyond the file keep
//! the established paths, with the reason logged once at load. So does a
//! file `fastokens` does not read: the frontend's own fallback for those is
//! the `tokenizers` crate with its default per-thread word cache, which the
//! wrapper bounds on its own pipeline and could not on the frontend's.

use anyhow::{Error, Result};
use vllm_tokenizer::{HuggingFaceTokenizer as VllmHuggingFaceTokenizer, Tokenizer as _};

use crate::{
    encode_path::{Declined, VllmSource},
    traits::{Encoding, TokenIdType},
};

/// The backend over one `tokenizer.json`, read by `fastokens`.
pub(crate) struct VllmBackend {
    tokenizer: VllmHuggingFaceTokenizer,
}

impl VllmBackend {
    /// The backend over the file `source` names, or why it declines: the
    /// load's reason for naming no file, or a file `fastokens` does not read.
    pub(crate) fn load(source: VllmSource<'_>) -> Result<Self, Declined> {
        let path = source?;
        let tokenizer = VllmHuggingFaceTokenizer::new_fastokens(path).map_err(|error| {
            Declined::from(format!(
                "vllm-tokenizer's fastokens backend did not load the file: {error}"
            ))
        })?;
        // The backend builds its pre-tokenizer's matcher on the first encode
        // (some 30 ms on a Qwen3 vocabulary): paid here, not by the first
        // request. A space is a token of every byte-level vocabulary.
        tokenizer.encode(" ", false).map_err(|error| {
            Declined::from(format!(
                "vllm-tokenizer's fastokens backend did not encode a space: {error}"
            ))
        })?;
        Ok(Self { tokenizer })
    }

    pub(crate) fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        self.tokenizer
            .encode(input, add_special_tokens)
            .map(Encoding::Plain)
            .map_err(|error| Error::msg(format!("Encoding failed: {error}")))
    }

    /// Input by input on the calling thread, as the direct encoders' batches
    /// run: the gateway fans a batch out over `encode` itself.
    pub(crate) fn encode_batch(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        inputs
            .iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect()
    }

    pub(crate) fn decode(
        &self,
        token_ids: &[TokenIdType],
        skip_special_tokens: bool,
    ) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| Error::msg(format!("Decoding failed: {error}")))
    }
}
