//! Optional v1 encode acceleration. Keep model loading/configuration, decoding
//! and streaming in the existing wrapper so there is one source of semantics.

use anyhow::{anyhow, Result};
use serde::Serialize;
use tokenizers::{models::ModelWrapper, Model};
use tokenizers_v1::{
    models::bpe::PipelineBPE,
    pipeline::{
        EncodeOptions, Inputs, PipelineModel, PipelineNormalizer, PipelinePostProcessor,
        PipelinePreTokenizer, PipelineTokenizer,
    },
    pre_tokenizers::split::Search,
    SplitDelimiterBehavior,
};

use crate::{
    encode_path::Declined,
    traits::{Encoder, Encoding},
    word_cache::{ModelView, TokenizerOf},
};

pub(crate) struct V1Encoder {
    pipeline: PipelineTokenizer,
    encode_special_tokens: bool,
}

// rc.2's reader can accept a Split without a usable regex engine. Check the
// constructed pipeline, including its fused DeepSeek sequence, before selecting it.
fn supported_splits(pre: &PipelinePreTokenizer) -> bool {
    match pre {
        PipelinePreTokenizer::Split(split) => {
            !matches!(split.search, Search::Unavailable)
                || (split.grammar().is_some()
                    && split.behavior == SplitDelimiterBehavior::Isolated
                    && !split.invert)
        }
        PipelinePreTokenizer::Sequence(sequence) => {
            sequence.is_deepseek() || sequence.pre_tokenizers().iter().all(supported_splits)
        }
        _ => true,
    }
}

// The RC normalizer enum has no Clone implementation; isolate this adapter here
// until the upstream reader exposes the BPE cache limit.
fn clone_normalizer(n: &PipelineNormalizer) -> PipelineNormalizer {
    match n {
        PipelineNormalizer::Metaspace(n) => PipelineNormalizer::Metaspace(n.clone()),
        PipelineNormalizer::Replace(n) => PipelineNormalizer::Replace(n.clone()),
        PipelineNormalizer::Prepend(n) => PipelineNormalizer::Prepend(n.clone()),
        PipelineNormalizer::Strip(n) => PipelineNormalizer::Strip(*n),
        PipelineNormalizer::Lowercase(n) => PipelineNormalizer::Lowercase(*n),
        PipelineNormalizer::ByteLevel(n) => PipelineNormalizer::ByteLevel(n.clone()),
        PipelineNormalizer::Bert(n) => PipelineNormalizer::Bert(*n),
        PipelineNormalizer::StripAccents(n) => PipelineNormalizer::StripAccents(*n),
        PipelineNormalizer::NFC(n) => PipelineNormalizer::NFC(*n),
        PipelineNormalizer::NFD(n) => PipelineNormalizer::NFD(*n),
        PipelineNormalizer::NFKC(n) => PipelineNormalizer::NFKC(*n),
        PipelineNormalizer::NFKD(n) => PipelineNormalizer::NFKD(*n),
        PipelineNormalizer::Nmt(n) => PipelineNormalizer::Nmt(*n),
        PipelineNormalizer::Precompiled(n) => PipelineNormalizer::Precompiled(n.clone()),
    }
}

impl V1Encoder {
    /// The pipeline for `tokenizer`, or why it declines: a configuration
    /// outside what the RC is trusted with here.
    pub(crate) fn from_tokenizer<M: Model + ModelView + Serialize>(
        tokenizer: &TokenizerOf<M>,
        cache_capacity: usize,
    ) -> Result<Self, Declined> {
        // Limit the RC to deterministic, unpadded/untruncated BPE. Other models
        // keep the established implementation, including its rich Encoding metadata.
        let ModelWrapper::BPE(bpe) = tokenizer.get_model().wrapper() else {
            return Err(Declined::from("the model is not BPE"));
        };
        if bpe.dropout.is_some_and(|p| p > 0.0) {
            return Err(Declined::from("BPE dropout is set"));
        }
        if tokenizer.get_padding().is_some() {
            return Err(Declined::from("padding is configured"));
        }
        if tokenizer.get_truncation().is_some() {
            return Err(Declined::from("truncation is configured"));
        }
        // Serialize AFTER config.json/tokenizer_config.json overrides, including
        // BOS/EOS processing. Loading the original file here would lose them.
        let build = || -> Result<PipelineTokenizer> {
            let json = tokenizer
                .to_string(false)
                .map_err(|e| anyhow!(e.to_string()))?;
            let canonical = tokenizers_v1::canonicalize_str(&json)?;
            let pipeline =
                tokenizers_v1::from_json(&canonical).map_err(|e| anyhow!(e.to_string()))?;
            anyhow::ensure!(
                supported_splits(pipeline.get_pre_tokenizer()),
                "Split has no usable regex backend"
            );
            // The rc.2 JSON reader ignores BPE cache_capacity. Rebuild just the
            // model with the wrapper's bound, then reuse the parsed components.
            let PipelineModel::BPE(model) = pipeline.get_model() else {
                return Err(anyhow!("expected a BPE pipeline"));
            };
            let mut config = model.to_config().map_err(|e| anyhow!(e.to_string()))?;
            config.cache_capacity = cache_capacity;
            let model = PipelineBPE::from_config(config).map_err(|e| anyhow!(e.to_string()))?;
            Ok(PipelineTokenizer::from_parts(
                pipeline.get_added_vocabulary().clone(),
                pipeline
                    .get_normalizers()
                    .iter()
                    .map(clone_normalizer)
                    .collect(),
                pipeline.get_pre_tokenizer().clone(),
                PipelineModel::BPE(model),
                PipelinePostProcessor {
                    single: pipeline.get_post_processor().single.clone(),
                    pair: pipeline.get_post_processor().pair.clone(),
                },
                pipeline.get_decoder().cloned(),
                pipeline.get_role_to_token().clone(),
                pipeline.get_padding().cloned(),
                pipeline.get_truncation().cloned(),
            ))
        };
        match build() {
            Ok(pipeline) => Ok(Self {
                pipeline,
                encode_special_tokens: tokenizer.get_encode_special_tokens(),
            }),
            Err(error) => Err(Declined::from(error.to_string())),
        }
    }

    fn run(&self, inputs: impl Into<Inputs>, add_special_tokens: bool) -> Result<Vec<Encoding>> {
        let options = EncodeOptions {
            add_special_tokens,
            encode_special_tokens: self.encode_special_tokens,
            ..Default::default()
        };
        self.pipeline
            .encode(inputs, &options)
            .wait()
            .map_err(|e| anyhow!("tokenizers v1 encoding failed: {e}"))
            .map(|encodings| {
                encodings
                    .into_iter()
                    .map(|e| Encoding::Plain(e.ids().iter().map(|token| token.id()).collect()))
                    .collect()
            })
    }
}

impl Encoder for V1Encoder {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        // The borrowed serial API avoids copying input and assembling a batch
        // handle. Small inputs cannot benefit from internal splitting; longer
        // inputs use the general API when internal parallelism is requested.
        // CachedTokenizer already fans out on Rayon workers; keep those calls
        // serial instead of nesting the v1 pool inside the outer batch.
        if input.len() < 16 * 1024
            || rayon::current_thread_index().is_some()
            || !tokenizers_v1::parallelism::get_parallelism()
            || tokenizers_v1::parallelism::num_threads() == 1
        {
            let options = EncodeOptions {
                add_special_tokens,
                encode_special_tokens: self.encode_special_tokens,
                ..Default::default()
            };
            let mut ids = Vec::with_capacity(input.len());
            self.pipeline
                .encode_into(input, &options, &mut ids)
                .map_err(|e| anyhow!("tokenizers v1 encoding failed: {e}"))?;
            return Ok(Encoding::Plain(
                ids.into_iter().map(|token| token.id()).collect(),
            ));
        }
        self.run(input, add_special_tokens)?
            .pop()
            .ok_or_else(|| anyhow!("tokenizers v1 returned no encoding for one input"))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        self.run(inputs, add_special_tokens)
    }
}
