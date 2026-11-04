//! The `vllm-tokenizer` backends: the vLLM Rust frontend's Hugging Face
//! tokenizer serves a tokenizer's flat encodes and batch decodes in place of
//! `tokenizers` and the direct encoders. Model
//! loading and configuration, the vocabulary maps, chat templates, the
//! renderers with encoders of their own and the streaming step stay in the
//! wrapper, so there is one source of semantics.
//!
//! The frontend's tiktoken backend (`riptoken`) serves the tiktoken wrapper
//! the same way ([`VllmTiktokenBackend`]): the wrapper keeps the Kimi
//! family's piece cut, its special-token and vocabulary tables, chat
//! templates, renderers and the per-stream decoder, and hands the backend a
//! file it reads as the wrapper does.
//!
//! The backend reads `tokenizer.json` alone, so the load hands it the file
//! only where the wrapper's pipeline is the file's: a tokenizer built in
//! memory, a vocabulary read from `vocab.json` and `merges.txt`, the Llama
//! class's pipeline installed over a legacy file, or added tokens settled
//! beyond the file keep the established paths, with the reason logged once
//! at load. The BOS and EOS the configuration prescribes on request, where
//! the file's post-processor does not frame an encode the same way, reach
//! the backend as a [`Frame`] of ids it puts around its bare encode. So does a
//! file `fastokens` does not read: the frontend's own fallback for those is
//! the `tokenizers` crate with its default per-thread word cache, which the
//! wrapper bounds on its own pipeline and could not on the frontend's.

use std::path::Path;

use anyhow::{Error, Result};
use vllm_tokenizer::{
    HuggingFaceTokenizer as VllmHuggingFaceTokenizer, TiktokenTokenizer as VllmTiktokenTokenizer,
    Tokenizer as _,
};

use crate::{
    encode_path::{Declined, VllmSource},
    traits::{Encoding, TokenIdType},
};

/// The ids a request for special tokens gets around a bare encode: the BOS
/// the configuration prescribes before it, the EOS after it, where the
/// file's own post-processor frames an encode otherwise (the rule is on
/// `HuggingFaceTokenizer::from_file`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Frame {
    pub(crate) prefix: Vec<TokenIdType>,
    pub(crate) suffix: Vec<TokenIdType>,
}

impl Frame {
    /// `ids` with the frame around it.
    fn around(&self, ids: Vec<TokenIdType>) -> Vec<TokenIdType> {
        if self.prefix.is_empty() && self.suffix.is_empty() {
            return ids;
        }
        let mut framed = Vec::with_capacity(self.prefix.len() + ids.len() + self.suffix.len());
        framed.extend_from_slice(&self.prefix);
        framed.extend(ids);
        framed.extend_from_slice(&self.suffix);
        framed
    }
}

/// The backend over one `tokenizer.json`, read by `fastokens`.
pub(crate) struct VllmBackend {
    tokenizer: VllmHuggingFaceTokenizer,
    /// The configured frame, when the file's post-processor is not it; `None`
    /// leaves a request for special tokens to the file's post-processor.
    frame: Option<Frame>,
}

impl VllmBackend {
    /// The backend over the file `source` names, with `frame` around the ids
    /// of a request for special tokens, or why it declines: the load's
    /// reason for naming no file, or a file `fastokens` does not read.
    pub(crate) fn load(source: VllmSource<'_>, frame: Option<Frame>) -> Result<Self, Declined> {
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
        Ok(Self { tokenizer, frame })
    }

    /// The ids of `input`; on a request for special tokens the file's
    /// post-processor frames them, or the configured [`Frame`] where the
    /// load handed one over.
    pub(crate) fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        let framed_by_file = add_special_tokens && self.frame.is_none();
        let ids = self
            .tokenizer
            .encode(input, framed_by_file)
            .map_err(|error| Error::msg(format!("Encoding failed: {error}")))?;
        Ok(Encoding::Plain(match &self.frame {
            Some(frame) if add_special_tokens => frame.around(ids),
            _ => ids,
        }))
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

/// The frontend's tiktoken backend over one BPE file (`tiktoken.model`,
/// `*.tiktoken`), read by `riptoken`.
pub(crate) struct VllmTiktokenBackend {
    tokenizer: VllmTiktokenTokenizer,
}

impl VllmTiktokenBackend {
    /// The backend over the BPE file at `path`, or why it declines: a file
    /// `riptoken` does not read, or one the frontend reads otherwise than the
    /// wrapper. The wrapper applies the Kimi family's pre-tokenizer pattern
    /// where `kimi` says so (the configuration names the family's tokenizer
    /// class, or `config.json` its model type); the frontend reads the model
    /// type alone ([`frontend_kimi_pattern`]). The wrapper's ids end at
    /// `ids_end` (its largest rank or added id, plus one, the Kimi family's
    /// reserved slots included); the frontend registers a reserved special
    /// token for every id from the ranks' end to the configuration's
    /// `vocab_size`, so where the two ends differ a spelling or an id reads
    /// differently.
    pub(crate) fn load(
        path: &Path,
        ids_end: usize,
        kimi: bool,
        frontend_kimi: bool,
    ) -> Result<Self, Declined> {
        if kimi != frontend_kimi {
            return Err(Declined::from(if kimi {
                "the Kimi family's pattern applies, which the frontend reads from config.json's \
                 model_type alone"
            } else {
                "the frontend would apply the Kimi family's pattern (config.json's model_type) \
                 where the cl100k pattern applies"
            }));
        }
        let tokenizer = VllmTiktokenTokenizer::new_riptoken(path).map_err(|error| {
            Declined::from(format!(
                "vllm-tokenizer's riptoken backend did not load the file: {error}"
            ))
        })?;
        let frontend_ids_end = tokenizer.vocab_size();
        if frontend_ids_end != ids_end {
            return Err(Declined::from(format!(
                "the frontend's ids end at {frontend_ids_end} where the file's and the \
                 configuration's end at {ids_end}"
            )));
        }
        Ok(Self { tokenizer })
    }

    /// The ids of `text`, its special tokens recognized (tiktoken has no
    /// `add_special_tokens`: the wrapper's engine recognizes them too).
    pub(crate) fn encode(&self, text: &str) -> Result<Vec<TokenIdType>> {
        self.tokenizer
            .encode(text, false)
            .map_err(|error| Error::msg(format!("tiktoken encode failed: {error}")))
    }

    /// The text of `token_ids`, every id kept (the wrapper strips what
    /// `skip_special_tokens` strips before, and rejects an id the vocabulary
    /// lacks, which the backend would skip).
    pub(crate) fn decode(&self, token_ids: &[TokenIdType]) -> Result<String> {
        self.tokenizer
            .decode(token_ids, false)
            .map_err(|error| Error::msg(format!("tiktoken decode failed: {error}")))
    }
}

/// Whether the frontend applies the Kimi family's pre-tokenizer pattern to
/// the tiktoken file in `dir`: its rule reads `config.json`'s `model_type`
/// (or `text_config.model_type`) alone, as the Dynamo reference does.
pub(crate) fn frontend_kimi_pattern(dir: &Path) -> bool {
    fn model_type(config: &serde_json::Value) -> Option<&str> {
        config
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .or_else(|| config.get("text_config").and_then(model_type))
    }
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
        .is_some_and(|config| {
            matches!(
                model_type(&config),
                Some("kimi" | "kimi_k2" | "kimi_k25" | "kimi_k3" | "deepseek_v3")
            )
        })
}
