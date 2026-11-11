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
    borrow::Cow,
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use serde::{
    de::{self, DeserializeSeed, Error as _, IgnoredAny, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_json::Value;
use tokenizers::{
    decoders::DecoderWrapper,
    models::{
        bpe::{Vocab, BPE},
        ModelWrapper, TrainerWrapper,
    },
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

    /// What the one-pass read of the file saw of a BPE section, while the
    /// model is loading; nothing once it is built.
    fn read(&self) -> Option<&ReadBpe> {
        None
    }
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
    /// What the one-pass read of a BPE section saw, for the loader's own
    /// tables; `None` for a model built in memory or read by the crate.
    read: Option<ReadBpe>,
}

/// What the one-pass read of a BPE section saw that the loader's own tables
/// are built from, so none of it is read back from the built model: the
/// merges as ids (left, right, product; the rank is the index), when every
/// merge names tokens of the vocabulary, the vocabulary's largest id, and
/// whether its ids are exactly the range `0..n` of its `n` entries.
/// Dropped with the loading model when it moves into the built tokenizer.
pub(crate) struct ReadBpe {
    pub(crate) pairs: Option<Box<[[u32; 3]]>>,
    pub(crate) max_id: Option<u32>,
    /// The ids are `0..n` with no gap and no id twice: the tables may
    /// walk the entries by id and find every one of them.
    pub(crate) dense: bool,
    /// The vocabulary's text by id when the ids are dense (see
    /// [`Listing`]), shared with the tables built after the model moved
    /// into the built tokenizer.
    pub(crate) tokens: Option<Arc<Listing>>,
}

/// The vocabulary's text, listed once by the read: every token's bytes in
/// one buffer and, by id, where each token lies in it. The passes that walk
/// the vocabulary (the seed, the preformed analysis, the character symbols,
/// the decode tables) borrow their tokens from it instead of cloning each
/// out of the model (`id_to_token`), one `String` per id per pass.
pub(crate) struct Listing {
    text: String,
    spans: Vec<(u32, u32)>,
}

impl Listing {
    /// The token of `id` (`None` past the end of the vocabulary).
    pub(crate) fn get(&self, id: u32) -> Option<&str> {
        let &(start, end) = self.spans.get(id as usize)?;
        Some(&self.text[start as usize..end as usize])
    }

    /// How many ids the listing covers (`0..len`).
    pub(crate) fn len(&self) -> u32 {
        self.spans.len() as u32
    }

    /// The listing of a dense vocabulary (every id below its length, none
    /// twice), or `None` for a sparse one (an id past the vocabulary's size,
    /// two tokens on one id) or one too large for the spans' width.
    pub(crate) fn of(vocab: &Vocab) -> Option<Self> {
        let n = vocab.len();
        let bytes = vocab.keys().map(String::len).sum::<usize>();
        // Under the marker, so no span's start can read as an unwritten slot.
        if bytes >= u32::MAX as usize || u32::try_from(n).is_err() {
            return None;
        }
        let mut text = String::with_capacity(bytes);
        let mut spans = vec![(u32::MAX, u32::MAX); n];
        for (token, &id) in vocab {
            let start = text.len() as u32;
            text.push_str(token);
            let slot = spans.get_mut(id as usize)?;
            if slot.0 != u32::MAX {
                return None;
            }
            *slot = (start, text.len() as u32);
        }
        Some(Self { text, spans })
    }
}

impl ReadBpe {
    fn of(vocab: &Vocab, merges: &[(String, String)]) -> Self {
        let max_id = vocab.values().copied().max();
        let n = vocab.len();
        let mut seen = vec![0u64; n.div_ceil(64)];
        let dense = vocab.values().all(|&id| {
            let id = id as usize;
            id < n && {
                let (word, bit) = (id / 64, 1u64 << (id % 64));
                let fresh = seen[word] & bit == 0;
                seen[word] |= bit;
                fresh
            }
        });
        let tokens = dense.then(|| Listing::of(vocab)).flatten().map(Arc::new);
        let mut token = String::new();
        let mut pairs = Vec::with_capacity(merges.len());
        for (left, right) in merges {
            token.clear();
            token.push_str(left);
            token.push_str(right);
            match (
                vocab.get(left.as_str()),
                vocab.get(right.as_str()),
                vocab.get(token.as_str()),
            ) {
                (Some(&a), Some(&b), Some(&product)) => pairs.push([a, b, product]),
                // A merge naming a token outside the vocabulary: the table
                // built from the model declines it with the reason.
                _ => {
                    return Self {
                        pairs: None,
                        max_id,
                        dense,
                        tokens,
                    }
                }
            }
        }
        Self {
            pairs: Some(pairs.into_boxed_slice()),
            max_id,
            dense,
            tokens,
        }
    }
}

/// The token of `id` as the decode sees it: an added token first (its own
/// map, as `id_to_token` looks), then the model's: borrowed from the read's
/// listing when there is one, cloned out of the model otherwise.
pub(crate) fn token_of<'a>(
    tokenizer: &TokenizerOf<ModelWrapper>,
    listing: Option<&'a Listing>,
    id: u32,
) -> Option<Cow<'a, str>> {
    if let Some(added) = tokenizer.get_added_vocabulary().simple_id_to_token(id) {
        return Some(Cow::Owned(added));
    }
    match listing {
        Some(listing) if id < listing.len() => listing.get(id).map(Cow::Borrowed),
        _ => tokenizer.id_to_token(id).map(Cow::Owned),
    }
}

/// The model's vocabulary as `(id, token)` entries, one at a time: borrowed
/// from the read's listing when it has one (a dense vocabulary, `Listing`);
/// else by id when the read says the ids are exactly the range `0..n` of the
/// vocabulary's `n` entries (one lookup per id, each token used and dropped
/// before the next, no clone of the map); otherwise a walk of the map, as
/// for a model without a read. A vocabulary numbered sparsely, or with two
/// tokens on one id, is never listed and never starts the walk by id.
pub(crate) fn entries_of<'a>(
    bpe: &'a BPE,
    read: Option<&'a ReadBpe>,
) -> Box<dyn Iterator<Item = (u32, Cow<'a, str>)> + 'a> {
    if let Some(tokens) = read.and_then(|read| read.tokens.as_deref()) {
        return Box::new(
            (0..tokens.len())
                .filter_map(move |id| tokens.get(id).map(|token| (id, Cow::Borrowed(token)))),
        );
    }
    match read.filter(|read| read.dense).and_then(|read| read.max_id) {
        Some(max_id) => Box::new(
            (0..=max_id)
                .filter_map(move |id| bpe.id_to_token(id).map(|token| (id, Cow::Owned(token)))),
        ),
        None => Box::new(
            bpe.get_vocab()
                .into_iter()
                .map(|(token, id)| (id, Cow::Owned(token))),
        ),
    }
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

    fn read(&self) -> Option<&ReadBpe> {
        self.read.as_ref()
    }
}

impl From<ModelWrapper> for LateBoundModel {
    fn from(model: ModelWrapper) -> Self {
        Self {
            model,
            words: AtomicUsize::new(HuggingFaceTokenizer::BPE_WORDS_CACHED_PER_THREAD),
            read: None,
        }
    }
}

impl From<LateBoundModel> for ModelWrapper {
    /// The model moves into the built tokenizer with its bound applied.
    fn from(late: LateBoundModel) -> Self {
        let LateBoundModel {
            mut model,
            words,
            read: _,
        } = late;
        if let ModelWrapper::BPE(bpe) = &mut model {
            bpe.resize_cache(words.into_inner());
        }
        model
    }
}

impl<'de> Deserialize<'de> for LateBoundModel {
    /// The file's model section, read in one pass (see [`ModelSection`]).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(ModelSection)
    }
}

/// The model section of a tokenizer file, read in one pass.
///
/// The `tokenizers` crate reads the section through an untagged helper:
/// serde buffers the whole section (the vocabulary map, the merge list) as
/// its generic content, the helper copies that into a `serde_json::Value`
/// tree and builds the model's tables from the tree, with the merge list
/// buffered once more on the way. Four copies of a 150k-entry vocabulary
/// are alive at once, 60-80 MB of transients for an 11 MB file, a third of
/// the load's time; and they are freed into the allocator's dirty-page
/// cache right before the model's own tables and the loader's are placed,
/// which then sit on recycled pages with their size-class tails resident
/// for the life of the process.
///
/// A BPE section, whose `type` key comes first in every file the crate
/// writes, streams here: the vocabulary into its hash map, the merges into
/// their list, both handed to the crate's own builder, so the model is the
/// one the crate builds from the same file. Any other section (another
/// model kind, a file without the `type` key or with it later) goes to the
/// crate's own reader, which streams it as before with the one entry
/// already read put back in front.
struct ModelSection;

impl<'de> Visitor<'de> for ModelSection {
    type Value = LateBoundModel;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a tokenizer model section")
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<LateBoundModel, A::Error> {
        let mut replay = Replay {
            key: None,
            value: None,
            rest: map,
        };
        if let Some(key) = replay.rest.next_key::<String>()? {
            let value = if key == "type" {
                let kind: String = replay.rest.next_value()?;
                if kind == "BPE" {
                    return bpe_section(replay.rest);
                }
                Value::String(kind)
            } else {
                replay.rest.next_value()?
            };
            replay.key = Some(key);
            replay.value = Some(value);
        }
        ModelWrapper::deserialize(de::value::MapAccessDeserializer::new(replay))
            .map(LateBoundModel::from)
    }
}

/// The section's entries with the one already read put back in front: the
/// crate's reader streams the section as it would have from the start.
struct Replay<A> {
    key: Option<String>,
    value: Option<Value>,
    rest: A,
}

impl<'de, A: MapAccess<'de>> MapAccess<'de> for Replay<A> {
    type Error = A::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, A::Error> {
        match self.key.take() {
            Some(key) => seed
                .deserialize(de::value::StringDeserializer::<A::Error>::new(key))
                .map(Some),
            None => self.rest.next_key_seed(seed),
        }
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, A::Error> {
        match self.value.take() {
            Some(value) => seed.deserialize(value).map_err(A::Error::custom),
            None => self.rest.next_value_seed(seed),
        }
    }
}

/// The rest of a BPE section after its `type`, fed to the crate's builder
/// field by field: the fields, their defaults and the skipping of unknown
/// ones are the crate's own (`tokenizers::models::bpe::serialization`). What
/// the loader's own tables need of the section rides on the loading model
/// (`ReadBpe`), read here once.
fn bpe_section<'de, A: MapAccess<'de>>(mut map: A) -> Result<LateBoundModel, A::Error> {
    let mut builder = BPE::builder();
    let mut vocab: Option<Vocab> = None;
    let mut merges: Option<Vec<(String, String)>> = None;
    while let Some(key) = map.next_key::<String>()? {
        match key.as_str() {
            "dropout" => {
                if let Some(dropout) = map.next_value::<Option<f32>>()? {
                    builder = builder.dropout(dropout);
                }
            }
            "unk_token" => {
                if let Some(token) = map.next_value::<Option<String>>()? {
                    builder = builder.unk_token(token);
                }
            }
            "continuing_subword_prefix" => {
                if let Some(prefix) = map.next_value::<Option<String>>()? {
                    builder = builder.continuing_subword_prefix(prefix);
                }
            }
            "end_of_word_suffix" => {
                if let Some(suffix) = map.next_value::<Option<String>>()? {
                    builder = builder.end_of_word_suffix(suffix);
                }
            }
            "fuse_unk" => {
                if let Some(fuse) = map.next_value::<Option<bool>>()? {
                    builder = builder.fuse_unk(fuse);
                }
            }
            "byte_fallback" => {
                if let Some(fallback) = map.next_value::<Option<bool>>()? {
                    builder = builder.byte_fallback(fallback);
                }
            }
            "ignore_merges" => {
                if let Some(ignore) = map.next_value::<Option<bool>>()? {
                    builder = builder.ignore_merges(ignore);
                }
            }
            "vocab" => vocab = Some(map.next_value()?),
            // The merge list is about the vocabulary's size, which the files
            // write first: one allocation for the list, no growth copies.
            "merges" => {
                let about = vocab.as_ref().map_or(0, |vocab| vocab.len());
                merges = Some(map.next_value_seed(MergeList { about })?);
            }
            "type" => {
                let kind: String = map.next_value()?;
                if kind != "BPE" {
                    return Err(A::Error::invalid_value(de::Unexpected::Str(&kind), &"BPE"));
                }
            }
            _ => {
                map.next_value::<IgnoredAny>()?;
            }
        }
    }
    let (Some(vocab), Some(merges)) = (vocab, merges) else {
        return Err(A::Error::custom("Missing vocab/merges"));
    };
    let read = ReadBpe::of(&vocab, &merges);
    let model = builder
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(A::Error::custom)?;
    Ok(LateBoundModel {
        model: ModelWrapper::BPE(model),
        words: AtomicUsize::new(HuggingFaceTokenizer::BPE_WORDS_CACHED_PER_THREAD),
        read: Some(read),
    })
}

/// The merge list of a BPE section: each merge a pair `["a", "b"]`, or in
/// the older files one line `"a b"` (a `#version` line among them is
/// skipped, as the crate's reader of that form skips it).
struct MergeList {
    /// How many merges to expect when the list gives no count.
    about: usize,
}

impl<'de> DeserializeSeed<'de> for MergeList {
    type Value = Vec<(String, String)>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for MergeList {
    type Value = Vec<(String, String)>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a list of merges")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut merges = Vec::with_capacity(seq.size_hint().unwrap_or(self.about));
        while let Some(Merge(merge)) = seq.next_element()? {
            merges.extend(merge);
        }
        Ok(merges)
    }
}

/// One merge as the file spells it; `None` for a `#version` line.
struct Merge(Option<(String, String)>);

impl<'de> Deserialize<'de> for Merge {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(MergeSpelling)
    }
}

struct MergeSpelling;

impl<'de> Visitor<'de> for MergeSpelling {
    type Value = Merge;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a merge: a pair of pieces, or one string with a space between them")
    }

    fn visit_str<E: de::Error>(self, line: &str) -> Result<Merge, E> {
        if line.starts_with("#version") {
            return Ok(Merge(None));
        }
        let mut pieces = line.split(' ');
        match (pieces.next(), pieces.next(), pieces.next()) {
            (Some(a), Some(b), None) => Ok(Merge(Some((a.to_string(), b.to_string())))),
            _ => Err(E::invalid_value(de::Unexpected::Str(line), &self)),
        }
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Merge, A::Error> {
        let a: String = seq
            .next_element()?
            .ok_or_else(|| A::Error::invalid_length(0, &self))?;
        let b: String = seq
            .next_element()?
            .ok_or_else(|| A::Error::invalid_length(1, &self))?;
        if seq.next_element::<IgnoredAny>()?.is_some() {
            return Err(A::Error::invalid_length(3, &self));
        }
        Ok(Merge(Some((a, b))))
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
    use serde_json::Map;
    use tokenizers::{
        models::{
            bpe::{Vocab, BPE},
            unigram::Unigram,
            wordlevel::WordLevel,
            wordpiece::WordPiece,
        },
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

    /// The model as the file spells it, through the crate's own reader.
    fn crate_reading(json: &str) -> String {
        let model: ModelWrapper = serde_json::from_str(json).expect("the crate reads the section");
        serde_json::to_string(&model).expect("the model serializes")
    }

    /// The model as the file spells it, through the loading model's reader.
    fn loading_reading(json: &str) -> String {
        let late: LateBoundModel =
            serde_json::from_str(json).expect("the loading model reads the section");
        serde_json::to_string(&late).expect("the model serializes")
    }

    /// A BPE with every field off its default: what the one-pass read must
    /// carry through to the builder.
    fn full_bpe_json() -> String {
        let mut vocab = Vocab::default();
        for (id, token) in ["<unk>", "a", "_b</w>", "ab</w>"].into_iter().enumerate() {
            vocab.insert(token.to_string(), id as u32);
        }
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![("a".to_string(), "_b</w>".to_string())])
            .dropout(0.25)
            .unk_token("<unk>".to_string())
            .continuing_subword_prefix("_".to_string())
            .end_of_word_suffix("</w>".to_string())
            .fuse_unk(true)
            .byte_fallback(true)
            .ignore_merges(true)
            .build()
            .expect("a BPE with every field set");
        serde_json::to_string(&ModelWrapper::BPE(bpe)).expect("the model serializes")
    }

    /// The fields `bpe_section` reads are the crate's: a `tokenizers` bump
    /// that adds or drops a BPE field fails here, where the one-pass read
    /// would otherwise skip the new field quietly.
    #[test]
    fn the_one_pass_reader_knows_every_field_the_crate_writes() {
        let section: Map<String, Value> =
            serde_json::from_str(&full_bpe_json()).expect("the section is a map");
        let keys: Vec<&str> = section.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "type",
                "dropout",
                "unk_token",
                "continuing_subword_prefix",
                "end_of_word_suffix",
                "fuse_unk",
                "byte_fallback",
                "ignore_merges",
                "vocab",
                "merges",
            ],
            "the crate's BPE gained or lost a field: teach bpe_section to read it"
        );
    }

    #[test]
    fn a_bpe_section_streams_into_the_model_the_crate_builds() {
        let json = full_bpe_json();
        assert!(json.starts_with(r#"{"type":"BPE","dropout":0.25,"unk_token":"<unk>""#));
        assert_eq!(loading_reading(&json), json);
        assert_eq!(crate_reading(&json), json);
        let plain = serde_json::to_string(&ModelWrapper::BPE(letters())).expect("serializes");
        assert_eq!(loading_reading(&plain), plain);
    }

    #[test]
    fn the_older_merge_spelling_and_an_unknown_field_read_as_the_crate_reads_them() {
        let json = full_bpe_json();
        let older = json.replace(
            r#""merges":[["a","_b</w>"]]"#,
            r##""merges":["#version: 0.2","a _b</w>"]"##,
        );
        assert_ne!(
            older, json,
            "the merge list was rewritten in the older spelling"
        );
        assert_eq!(loading_reading(&older), json);
        assert_eq!(crate_reading(&older), json);
        let unknown = json.replace(
            r#""dropout":0.25,"#,
            r#""dropout":0.25,"future":{"x":[1,2]},"#,
        );
        assert_ne!(unknown, json);
        assert_eq!(loading_reading(&unknown), json);
        assert_eq!(crate_reading(&unknown), json);
    }

    #[test]
    fn a_merge_that_is_not_two_pieces_is_refused() {
        let json = full_bpe_json();
        for merges in [r#"["a b c"]"#, r#"[["a","_b</w>","c"]]"#, r#"[["a"]]"#] {
            let broken = json.replace(r#"[["a","_b</w>"]]"#, merges);
            assert_ne!(broken, json);
            assert!(
                serde_json::from_str::<LateBoundModel>(&broken).is_err(),
                "{merges} read as a merge list"
            );
        }
    }

    #[test]
    fn another_model_kind_or_a_type_key_later_takes_the_crate_reader() {
        let mut words = Vocab::default();
        for (id, token) in ["[UNK]", "a", "b"].into_iter().enumerate() {
            words.insert(token.to_string(), id as u32);
        }
        let word_level = WordLevel::builder()
            .vocab(words.clone())
            .unk_token("[UNK]".to_string())
            .build()
            .expect("a word-level model");
        let word_piece = WordPiece::builder()
            .vocab(words)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("a word-piece model");
        let unigram = Unigram::from(
            vec![("<unk>".to_string(), 0.0), ("a".to_string(), -1.0)],
            Some(0),
            false,
        )
        .expect("a unigram model");
        for model in [
            ModelWrapper::WordLevel(word_level),
            ModelWrapper::WordPiece(word_piece),
            ModelWrapper::Unigram(unigram),
        ] {
            let json = serde_json::to_string(&model).expect("the model serializes");
            assert_eq!(loading_reading(&json), json);
        }
        // The `type` key after the fields, and a file without it (the
        // crate's older files): the crate's reader decides, as before.
        for json in [full_bpe_json(), {
            let mut words = Vocab::default();
            words.insert("[UNK]".to_string(), 0);
            let model = WordPiece::builder()
                .vocab(words)
                .unk_token("[UNK]".to_string())
                .build()
                .expect("a word-piece model");
            serde_json::to_string(&ModelWrapper::WordPiece(model)).expect("serializes")
        }] {
            let mut section: Map<String, Value> =
                serde_json::from_str(&json).expect("the section is a map");
            let kind = section
                .shift_remove("type")
                .expect("the section names its type");
            let without_type = serde_json::to_string(&section).expect("serializes");
            section.insert("type".to_string(), kind);
            let type_last = serde_json::to_string(&section).expect("serializes");
            assert!(
                type_last.ends_with(r#""type":"BPE"}"#)
                    || type_last.ends_with(r#""type":"WordPiece"}"#)
            );
            assert_eq!(loading_reading(&type_last), json);
            assert_eq!(loading_reading(&without_type), crate_reading(&without_type));
        }
    }
}
