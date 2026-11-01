//! Optional v1 encode acceleration. Keep model loading/configuration, decoding
//! and streaming in the existing wrapper so there is one source of semantics.

use anyhow::{anyhow, Result};
use serde::Serialize;
use serde_json::Value;
use tokenizers::{models::ModelWrapper, pre_tokenizers::PreTokenizerWrapper, Model};
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

/// Whether `pre` is a `Metaspace` pre-tokenizer, alone or in a sequence.
fn has_metaspace(pre: Option<&PreTokenizerWrapper>) -> bool {
    match pre {
        Some(PreTokenizerWrapper::Metaspace(_)) => true,
        Some(PreTokenizerWrapper::Sequence(sequence)) => sequence
            .as_ref()
            .iter()
            .any(|step| matches!(step, PreTokenizerWrapper::Metaspace(_))),
        _ => false,
    }
}

/// Whether a match of `after` (an added token matched after normalization)
/// could begin before a match of `raw` (one matched on the raw text) and
/// reach into it: `after` holds `raw`, or ends with a prefix of it. Cutting
/// the two kinds in one pass would then take `after` where the reference's
/// raw pass, which runs first, takes `raw`.
fn reaches_into(after: &str, raw: &str) -> bool {
    let (after, raw) = (after.as_bytes(), raw.as_bytes());
    after.windows(raw.len()).any(|window| window == raw)
        || (1..after.len()).any(|at| raw.starts_with(&after[at..]))
}

/// Whether the pipeline is handed the added tokens matched after
/// normalization as raw ones.
///
/// The reader lowers a `Metaspace` pre-tokenizer into the last normalizer
/// (the pipeline has no `Metaspace` pre-tokenizer), and the pipeline cuts the
/// raw added tokens before the normalizers run and the others after them;
/// the reference library cuts both kinds first and runs the pre-tokenizer on
/// each split that remains, with the split's offset. The two agree until such
/// a token meets the word marker the scheme prepends: a marker token that
/// opens the text gets a word-marker piece in front of it (`first`), the
/// split after one loses its word marker (`always`), and a token with a
/// space in it is never found (the spaces of the text are the marker by
/// then). With no normalizer the text a token is matched on after
/// normalization is the raw text, so the raw pass finds the same tokens, and
/// cuts them before the lowered pre-tokenizer runs: the reference's order.
///
/// Declines the shapes where the pass order decides a match: a token matched
/// after normalization that could reach into a raw one; a `single_word`
/// token, whose boundary one pass judges against the token after it over the
/// whole text where the reference's second pass judges it at the end of a
/// piece (and after a boundary miss one pass resumes one byte on, where the
/// reference skips the match); a strip that would take whitespace held by
/// the neighbouring token's content (a raw `lstrip` token beside a token
/// matched after normalization that ends in whitespace, or a `rstrip` token
/// matched after normalization beside a raw token that starts with it: the
/// reference's raw pass has cut the raw token first, one pass gives the
/// whitespace to whichever match comes first); `encode_special_tokens`, under
/// which the reference's raw pass leaves a special token's text to its second
/// pass where one pass has moved on.
fn added_tokens_cut_raw<M: Model>(tokenizer: &TokenizerOf<M>) -> Result<bool, Declined> {
    if tokenizer.get_normalizer().is_some() || !has_metaspace(tokenizer.get_pre_tokenizer()) {
        return Ok(false);
    }
    let added = tokenizer.get_added_vocabulary().get_added_tokens_decoder();
    let after_normalization: Vec<&str> = added
        .values()
        .filter(|token| token.normalized)
        .map(|token| token.content.as_str())
        .collect();
    if after_normalization.is_empty() {
        return Ok(false);
    }
    if tokenizer.get_encode_special_tokens() {
        return Err(Declined::from(
            "encode_special_tokens is set, with added tokens matched after normalization under a \
             Metaspace pre-tokenizer (the reader lowers Metaspace into a normalizer, so the two \
             kinds would be cut in one pass, which moves past a special token the reference leaves \
             as text)",
        ));
    }
    if let Some(token) = added.values().find(|token| token.single_word) {
        return Err(Declined::from(format!(
            "the added token {:?} has single_word, with added tokens matched after normalization \
             under a Metaspace pre-tokenizer (the reader lowers Metaspace into a normalizer, so the \
             two kinds would be cut in one pass, which judges a word boundary against the token \
             after it where the reference judges it at the end of a piece)",
            token.content
        )));
    }
    let raw: Vec<&str> = added
        .values()
        .filter(|token| !token.normalized)
        .map(|token| token.content.as_str())
        .collect();
    let whitespace = |c: char| c.is_whitespace();
    let lstrip_raw = added
        .values()
        .find(|token| !token.normalized && token.lstrip);
    let rstrip_after = added
        .values()
        .find(|token| token.normalized && token.rstrip);
    let strip_and_neighbour = lstrip_raw
        .zip(
            after_normalization
                .iter()
                .find(|content| content.ends_with(whitespace)),
        )
        .or_else(|| rstrip_after.zip(raw.iter().find(|content| content.starts_with(whitespace))));
    if let Some((strip, neighbour)) = strip_and_neighbour {
        return Err(Declined::from(format!(
            "the added token {:?}'s strip could take whitespace held by the content of \
             {neighbour:?}, with added tokens matched after normalization under a Metaspace \
             pre-tokenizer (the reader lowers Metaspace into a normalizer, so the two kinds would \
             be cut in one pass, which gives the whitespace to whichever match comes first where \
             the reference cuts the raw token first)",
            strip.content
        )));
    }
    if let Some((after, raw)) = after_normalization
        .iter()
        .flat_map(|after| raw.iter().map(move |raw| (*after, *raw)))
        .find(|(after, raw)| reaches_into(after, raw))
    {
        return Err(Declined::from(format!(
            "the added token {after:?}, matched after normalization, could reach into the raw \
             one {raw:?} under a Metaspace pre-tokenizer (the reader lowers Metaspace into a \
             normalizer, so the two kinds would be cut in one pass)"
        )));
    }
    tracing::debug!(
        tokens = after_normalization.len(),
        "the added tokens matched after normalization are handed to the v1 pipeline as raw ones: \
         no normalizer, and a Metaspace pre-tokenizer the reader lowers into one"
    );
    Ok(true)
}

/// The serialized tokenizer with every added token marked as matched raw.
fn with_added_tokens_raw(json: &str) -> Result<String> {
    let mut tokenizer: Value = serde_json::from_str(json)?;
    let tokens = tokenizer
        .get_mut("added_tokens")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| anyhow!("the serialized tokenizer has no added_tokens array"))?;
    for token in tokens.iter_mut() {
        token
            .as_object_mut()
            .ok_or_else(|| anyhow!("an added token is not an object"))?
            .insert("normalized".to_owned(), Value::Bool(false));
    }
    Ok(tokenizer.to_string())
}

// The RC reader (rc.3 as rc.2) can accept a Split without a usable regex engine.
// Check the constructed pipeline, including its fused DeepSeek sequence, before
// selecting it.
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
        PipelineNormalizer::StripAccents(n) => PipelineNormalizer::StripAccents(n.clone()),
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
        let cut_raw = added_tokens_cut_raw(tokenizer)?;
        // Serialize AFTER config.json/tokenizer_config.json overrides, including
        // BOS/EOS processing. Loading the original file here would lose them.
        let build = || -> Result<PipelineTokenizer> {
            let json = tokenizer
                .to_string(false)
                .map_err(|e| anyhow!(e.to_string()))?;
            let json = if cut_raw {
                with_added_tokens_raw(&json)?
            } else {
                json
            };
            let canonical = tokenizers_v1::canonicalize_str(&json)?;
            let pipeline =
                tokenizers_v1::from_json(&canonical).map_err(|e| anyhow!(e.to_string()))?;
            anyhow::ensure!(
                supported_splits(pipeline.get_pre_tokenizer()),
                "Split has no usable regex backend"
            );
            // A tokenizer.json carries no BPE cache bound, so the RC reader builds
            // the model without one. Rebuild just the model with the wrapper's
            // bound, then reuse the parsed components.
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
