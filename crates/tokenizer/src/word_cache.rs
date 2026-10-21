//! The bound on the `tokenizers` BPE word cache, applied without a copy of
//! the model.
//!
//! The BPE model of the `tokenizers` crate remembers the words it merged in
//! a cache per thread, up to a capacity the model carries; the wrapper sets
//! that capacity ([`HuggingFaceTokenizer::BPE_WORDS_CACHED_PER_THREAD`], or
//! none behind the direct encode path). A built tokenizer hands out its
//! model by shared reference only, so a bound set on one means a copy of
//! the model (vocabulary, reverse vocabulary and merge table, a few hundred
//! thousand entries for a large vocabulary) installed in place of the
//! original. That is how the bound was first applied, and it cost the
//! decode path: under the system allocator a full decode of a byte-level
//! vocabulary measured almost twice as slow with the copy as without it,
//! and an encode on the `tokenizers` path a quarter slower (the gateway's
//! own allocator hid it). The loaders read a file into a tokenizer whose
//! model is a [`LateBoundModel`] instead: the file's model as it is, with
//! the capacity beside it, decided once the encode path is known and
//! applied as the model moves into the built tokenizer. One model, no copy.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tokenizers::{
    decoders::DecoderWrapper,
    models::{ModelWrapper, TrainerWrapper},
    normalizers::NormalizerWrapper,
    pre_tokenizers::PreTokenizerWrapper,
    processors::PostProcessorWrapper,
    Model, Result as TokenizersResult, Token, TokenizerImpl,
};

use crate::huggingface::HuggingFaceTokenizer;

/// A tokenizer of the crate's components over any model type: a built
/// [`tokenizers::Tokenizer`] derefs to the one over [`ModelWrapper`], a
/// [`LoadingTokenizer`] carries a [`LateBoundModel`]. What the load reads
/// of a tokenizer before its encode path is chosen takes either.
pub(crate) type TokenizerOf<M> =
    TokenizerImpl<M, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>;

/// A tokenizer being loaded: the file's model, its word-cache bound still
/// open. [`tokenizers::Tokenizer::from`] builds the tokenizer from it, the
/// model moving in with the bound applied.
pub(crate) type LoadingTokenizer = TokenizerOf<LateBoundModel>;

/// The `tokenizers` model behind a tokenizer's model type, built or loading.
pub(crate) trait ModelView {
    fn wrapper(&self) -> &ModelWrapper;
}

impl ModelView for ModelWrapper {
    fn wrapper(&self) -> &ModelWrapper {
        self
    }
}

/// A model read from a file, with the words its BPE may cache per thread
/// decided later: the load reads the tokenizer, chooses its encode path and
/// sets the bound through a shared reference (the only kind a tokenizer
/// hands out of its model); the conversion into the built tokenizer applies
/// it to the model as it moves, in place.
pub(crate) struct LateBoundModel {
    model: ModelWrapper,
    /// Words the BPE model caches per thread once built: the general bound
    /// until the load decides otherwise.
    words: AtomicUsize,
}

impl LateBoundModel {
    /// Sets the words the BPE model caches per thread once built (other
    /// model kinds have no such cache).
    pub(crate) fn bound(&self, words: usize) {
        self.words.store(words, Ordering::Relaxed);
    }
}

impl ModelView for LateBoundModel {
    fn wrapper(&self) -> &ModelWrapper {
        &self.model
    }
}

impl From<ModelWrapper> for LateBoundModel {
    fn from(model: ModelWrapper) -> Self {
        Self {
            model,
            words: AtomicUsize::new(HuggingFaceTokenizer::BPE_WORDS_CACHED_PER_THREAD),
        }
    }
}

impl From<LateBoundModel> for ModelWrapper {
    /// The model moves into the built tokenizer with its bound applied.
    fn from(late: LateBoundModel) -> Self {
        let LateBoundModel { mut model, words } = late;
        if let ModelWrapper::BPE(bpe) = &mut model {
            bpe.resize_cache(words.into_inner());
        }
        model
    }
}

impl<'de> Deserialize<'de> for LateBoundModel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ModelWrapper::deserialize(deserializer).map(Self::from)
    }
}

impl Serialize for LateBoundModel {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.model.serialize(serializer)
    }
}

impl Model for LateBoundModel {
    type Trainer = TrainerWrapper;

    fn tokenize(&self, sequence: &str) -> TokenizersResult<Vec<Token>> {
        self.model.tokenize(sequence)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.id_to_token(id)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.model.get_vocab()
    }

    fn get_vocab_size(&self) -> usize {
        self.model.get_vocab_size()
    }

    fn save(&self, folder: &Path, prefix: Option<&str>) -> TokenizersResult<Vec<PathBuf>> {
        self.model.save(folder, prefix)
    }

    fn get_trainer(&self) -> TrainerWrapper {
        self.model.get_trainer()
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::{
        models::bpe::{Vocab, BPE},
        pre_tokenizers::whitespace::Whitespace,
        AddedToken, Tokenizer,
    };

    use super::*;

    /// A BPE over three letters and one merge.
    fn letters() -> BPE {
        let mut vocab = Vocab::default();
        for (id, token) in ["a", "b", "c", "ab"].into_iter().enumerate() {
            vocab.insert(token.to_string(), id as u32);
        }
        BPE::builder()
            .vocab_and_merges(vocab, vec![("a".to_string(), "b".to_string())])
            .build()
            .expect("a BPE over three letters")
    }

    #[test]
    fn the_loading_model_answers_as_the_model_it_carries() {
        let model = ModelWrapper::BPE(letters());
        let late = LateBoundModel::from(ModelWrapper::BPE(letters()));
        assert_eq!(late.get_vocab_size(), model.get_vocab_size());
        assert_eq!(late.get_vocab(), model.get_vocab());
        assert_eq!(late.token_to_id("ab"), model.token_to_id("ab"));
        assert_eq!(late.id_to_token(2), model.id_to_token(2));
        let ids = |tokens: Vec<Token>| tokens.into_iter().map(|t| t.id).collect::<Vec<_>>();
        assert_eq!(
            ids(late.tokenize("abc").expect("a tokenize")),
            ids(model.tokenize("abc").expect("a tokenize"))
        );
        assert_eq!(
            serde_json::to_string(&late).expect("the loading model serializes"),
            serde_json::to_string(&model).expect("the model serializes")
        );
    }

    #[test]
    fn a_file_read_into_a_loading_tokenizer_builds_the_tokenizer_the_file_describes() {
        let mut built = Tokenizer::new(letters());
        built.with_pre_tokenizer(Some(Whitespace));
        built
            .add_special_tokens([AddedToken::from("<s>", true)])
            .expect("the special token");
        let json = built.to_string(false).expect("the tokenizer serializes");
        let loading: LoadingTokenizer =
            serde_json::from_str(&json).expect("the loading tokenizer reads the same file");
        loading.get_model().bound(2);
        let rebuilt = Tokenizer::from(loading);
        assert_eq!(rebuilt.get_vocab(true), built.get_vocab(true));
        let encode = |tokenizer: &Tokenizer| {
            tokenizer
                .encode("ab c <s>", true)
                .expect("an encode")
                .get_ids()
                .to_vec()
        };
        assert_eq!(encode(&rebuilt), encode(&built));
        assert_eq!(
            rebuilt
                .to_string(false)
                .expect("the rebuilt tokenizer serializes"),
            json
        );
    }
}
