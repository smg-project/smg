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
        bpe::{BpeBuilder, Vocab, BPE},
        unigram::Unigram,
        wordpiece::WordPiece,
        ModelWrapper, TrainerWrapper,
    },
    normalizers::NormalizerWrapper,
    pre_tokenizers::PreTokenizerWrapper,
    processors::PostProcessorWrapper,
    Model, Result as TokenizersResult, Token, TokenizerImpl,
};

use crate::{
    byte_level::{TextIndex, TextStore},
    huggingface::HuggingFaceTokenizer,
    unigram_scores,
};

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

    /// What the one-pass read of the file saw of a BPE, WordPiece or
    /// Unigram section, while the model is loading; nothing once it is
    /// built.
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
    /// What the one-pass read of a BPE, WordPiece or Unigram section saw,
    /// for the loader's own tables; `None` for a model built in memory or
    /// read by the crate.
    read: Option<ReadBpe>,
    /// Whether the crate's reader made a Unigram model of the section: the
    /// scores are then that reader's, and the loader reads the section once
    /// more for them ([`unigram_scores::of_file`]).
    rescore: bool,
}

/// What the one-pass read of a BPE section saw that the loader's own tables
/// are built from, so none of it is read back from the built model: the
/// merges as ids (left, right, product; the rank is the index), when every
/// merge names tokens of the vocabulary, the vocabulary's largest id, and
/// whether its ids are exactly the range `0..n` of its `n` entries. A
/// WordPiece section leaves the same read without merges (`pairs` is
/// `None`), and so does a Unigram section, where the id of a piece is its
/// place in the section's list.
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
    /// The ids by their text over `tokens`, where the section's model was
    /// not built ([`listed_section`]): the loading model then answers from
    /// the read, and the model it carries is empty.
    pub(crate) index: Option<Arc<TextIndex>>,
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

    /// How many tokens have at most `most` bytes or are taken by `longer`;
    /// one of at most `most` bytes is counted from its span alone.
    pub(crate) fn count(&self, most: usize, longer: impl Fn(&str) -> bool) -> usize {
        self.spans
            .iter()
            .filter(|&&(start, end)| {
                (end - start) as usize <= most || longer(&self.text[start as usize..end as usize])
            })
            .count()
    }

    /// How many ids the listing covers (`0..len`).
    pub(crate) fn len(&self) -> u32 {
        self.spans.len() as u32
    }

    /// How many bytes the text of its tokens is, all together.
    pub(crate) fn bytes(&self) -> usize {
        self.text.len()
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

impl TextStore for Listing {
    fn ids(&self) -> u32 {
        self.len()
    }

    /// As bytes: a lookup compares bytes and needs no character boundary
    /// checked.
    fn bytes_of(&self, id: u32) -> Option<&[u8]> {
        let &(start, end) = self.spans.get(id as usize)?;
        self.text.as_bytes().get(start as usize..end as usize)
    }
}

impl ReadBpe {
    /// What the vocabulary alone tells: the read of a section that has no
    /// merges, and the start of the read of one that has.
    fn without_merges(vocab: &Vocab) -> Self {
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
        Self {
            pairs: None,
            max_id,
            dense,
            tokens,
            index: None,
        }
    }

    /// The read of a Unigram section: the id of a piece is its place in
    /// the section's list, so the ids are dense however the section is
    /// written, and the largest is the count of pieces less one. The
    /// pieces are not listed a second time: the tables read each from
    /// the model by its id, and a copy of them stays resident after the
    /// load.
    fn of_pieces(unigram: &Unigram) -> Self {
        Self {
            pairs: None,
            max_id: u32::try_from(unigram.get_vocab_size())
                .ok()
                .and_then(|pieces| pieces.checked_sub(1)),
            dense: true,
            tokens: None,
            index: None,
        }
    }

    fn of(vocab: &Vocab, merges: &[(String, String)]) -> Self {
        let mut read = Self::without_merges(vocab);
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
                _ => return read,
            }
        }
        read.pairs = Some(pairs.into_boxed_slice());
        read
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
    /// The model [`unigram_scores::section`] read, with the read of its
    /// pieces ([`ReadBpe::of_pieces`]): the loader's tables take the
    /// largest id from it, as they do after a BPE or WordPiece section,
    /// and clone no vocabulary to find it.
    fn of_unigram(model: ModelWrapper) -> Self {
        let read = match &model {
            ModelWrapper::Unigram(unigram) => Some(ReadBpe::of_pieces(unigram)),
            // That reader builds no other kind.
            _ => None,
        };
        Self {
            read,
            ..Self::from(model)
        }
    }

    /// Sets the words the BPE model caches per thread once built (other
    /// model kinds have no such cache).
    pub(crate) fn bound(&self, words: usize) {
        self.words.store(words, Ordering::Relaxed);
    }

    /// Whether the crate's reader made a Unigram model of the section, so
    /// that the loader reads the section once more for its scores.
    pub(crate) fn rescore(&self) -> bool {
        self.rescore
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
            rescore: false,
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
            rescore: _,
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
/// one the crate builds from the same file. A Unigram section is read by
/// [`unigram_scores::section`], for the scores the reference reads, in any
/// order of its keys (`type`, `unk_id`, `vocab`, `byte_fallback`): the
/// entries are read until one tells the kind, which is the value of `type`
/// or a vocabulary that is a list of pieces with their scores; it leaves
/// the read of its pieces ([`ReadBpe::of_pieces`]). Any other entry is held
/// until then, so a `type` behind it still tells the kind. The merges of a
/// BPE section, and the vocabulary of a section that has shown a key of
/// another kind (`unk_token`, `dropout` and the like) and not yet its
/// `type`, hand the section to the crate's own reader, that value unread,
/// so a file of another kind, written without `type` or with its keys
/// sorted, loads at the cost it had. A WordPiece section streams as well,
/// with `type` or, as the older files are, without it
/// ([`wordpiece_section`]). Any other section (another model kind, a BPE
/// section with `type` later) goes to the crate's own reader as well, which
/// streams it as before with the entries already read put back in front.
/// When that reader makes a Unigram model of a section, the loader reads
/// the section once more for its scores ([`unigram_scores::of_file`]).
struct ModelSection;

impl<'de> Visitor<'de> for ModelSection {
    type Value = LateBoundModel;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a tokenizer model section")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<LateBoundModel, A::Error> {
        // The entries read before one tells the kind, and the key that
        // ended the reading with its value still in the section.
        let mut read = Vec::new();
        let mut pending = None;
        // Whether a key of another kind of section has been read.
        let mut another = false;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => {
                    let kind: String = map.next_value()?;
                    if kind == "BPE" && read.is_empty() {
                        return bpe_section(map);
                    }
                    if kind == "Unigram" {
                        return unigram_scores::section(None, Replay::new(read, None, map))
                            .map(LateBoundModel::of_unigram);
                    }
                    let word_piece = kind == "WordPiece" && read.is_empty();
                    read.push((key, Value::String(kind)));
                    if word_piece {
                        return wordpiece_section(read, map);
                    }
                    break;
                }
                // The merges of a BPE section, wherever they stand, and the
                // vocabulary behind a key of another kind: the crate's
                // reader takes the section from here, with this value unread.
                "merges" => {
                    pending = Some(key);
                    break;
                }
                "vocab" if another => {
                    pending = Some(key);
                    break;
                }
                // A list of pieces with their scores is a Unigram section's.
                "vocab" => match map.next_value_seed(unigram_scores::VocabShape)? {
                    unigram_scores::Vocab::Scored(vocab) => {
                        return unigram_scores::section(Some(vocab), Replay::new(read, None, map))
                            .map(LateBoundModel::of_unigram);
                    }
                    unigram_scores::Vocab::Other(vocab) => {
                        read.push((key, vocab));
                        break;
                    }
                },
                // A WordPiece section opens with one of these when `type`
                // is not its first key, and so may a BPE or WordLevel one.
                "unk_token" | "continuing_subword_prefix" | "max_input_chars_per_word"
                    if read.is_empty() =>
                {
                    read.push((key, map.next_value()?));
                    return wordpiece_section(read, map);
                }
                // A key of another kind of section, which no Unigram section
                // has. It is held like the rest, so a `type` behind it still
                // tells the kind; the section's vocabulary is not read here.
                "dropout"
                | "unk_token"
                | "continuing_subword_prefix"
                | "end_of_word_suffix"
                | "fuse_unk"
                | "ignore_merges"
                | "max_input_chars_per_word" => {
                    another = true;
                    read.push((key, map.next_value()?));
                }
                // No other entry tells the kind: `unk_id`, which a section
                // of another kind may carry (the crate's reader skips it
                // there), `byte_fallback`, which a BPE section has too, or a
                // key no section has. Each is held, so where it stands does
                // not decide which reader takes the section.
                _ => read.push((key, map.next_value()?)),
            }
        }
        crate_section(read, pending, map)
    }
}

/// The section through the crate's own reader: the entries already read in
/// front, then the key whose value is still in the section, then the rest
/// (see [`Replay`]). A Unigram model it makes is marked, so the loader
/// reads the section once more for its scores.
fn crate_section<'de, A: MapAccess<'de>>(
    read: Vec<(String, Value)>,
    key: Option<String>,
    rest: A,
) -> Result<LateBoundModel, A::Error> {
    let model = ModelWrapper::deserialize(de::value::MapAccessDeserializer::new(Replay::new(
        read, key, rest,
    )))?;
    // A Unigram model from the crate's reader has that reader's scores.
    let rescore = matches!(model, ModelWrapper::Unigram(_));
    Ok(LateBoundModel {
        rescore,
        ..LateBoundModel::from(model)
    })
}

/// The section's entries for a reader that takes it from its start: the
/// entries already read, in the file's order, then the key read whose value
/// is still in the section, then the rest of the section.
struct Replay<A> {
    read: std::vec::IntoIter<(String, Value)>,
    /// The value of the entry of `read` whose key was just given out.
    value: Option<Value>,
    key: Option<String>,
    rest: A,
}

impl<A> Replay<A> {
    fn new(read: Vec<(String, Value)>, key: Option<String>, rest: A) -> Self {
        Self {
            read: read.into_iter(),
            value: None,
            key,
            rest,
        }
    }
}

impl<'de, A: MapAccess<'de>> MapAccess<'de> for Replay<A> {
    type Error = A::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, A::Error> {
        let key = match self.read.next() {
            Some((key, value)) => {
                self.value = Some(value);
                key
            }
            None => match self.key.take() {
                Some(key) => key,
                None => return self.rest.next_key_seed(seed),
            },
        };
        seed.deserialize(de::value::StringDeserializer::<A::Error>::new(key))
            .map(Some)
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
    let mut prefixed = false;
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
                    prefixed = !prefix.is_empty();
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
                    // The rest of a section that says so before its vocabulary
                    // is read by the listing reader (the opt-in pipeline writes
                    // the model out again, and keeps this reader).
                    if fallback
                        && vocab.is_none()
                        && merges.is_none()
                        && !cfg!(feature = "tokenizers-v1")
                    {
                        match listed_entry(map, builder, prefixed)? {
                            Listed::Model(model) => return Ok(model),
                            Listed::Back(settings, map_vocab, texts) => {
                                builder = settings;
                                vocab = Some(map_vocab);
                                merges = Some(texts);
                                break;
                            }
                        }
                    }
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
        rescore: false,
    })
}

/// The rest of a BPE section that said `byte_fallback` before its
/// vocabulary: the vocabulary is listed, not mapped, its merges are resolved
/// as they are read, and no model is built of them here. The loader builds
/// one only where a path runs it (`HuggingFaceTokenizer::from_file`).
#[inline(never)]
fn listed_section<'de, A: MapAccess<'de>>(
    mut map: A,
    mut builder: BpeBuilder,
    mut prefixed: bool,
) -> Result<Listed, A::Error> {
    let mut vocab: Option<VocabRead> = None;
    let mut merges: Option<MergesRead> = None;
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
                    prefixed = !prefix.is_empty();
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
            "vocab" => {
                // Merges resolved in a vocabulary that another now replaces
                // go back to their texts: the builder resolves them again.
                if let (Some(VocabRead::Listed(listing, _)), Some(MergesRead::Ids(pairs))) =
                    (&vocab, &merges)
                {
                    merges = Some(MergesRead::Texts(texts_of(listing, pairs)));
                }
                vocab = Some(map.next_value_seed(ListedVocab)?);
            }
            // The merge list is about the vocabulary's size, which the files
            // write first: one allocation for the list, no growth copies.
            "merges" => {
                let listed = match &vocab {
                    Some(VocabRead::Listed(listing, index)) => Some((&**listing, &**index)),
                    _ => None,
                };
                merges = Some(map.next_value_seed(MergeIds { listed })?);
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
    match (vocab, merges) {
        // The model stays empty, with the section's settings; the read
        // answers for it while the load runs.
        (VocabRead::Listed(listing, index), MergesRead::Ids(pairs)) if !prefixed => {
            let model = builder
                .cache_capacity(0)
                .build()
                .map_err(A::Error::custom)?;
            Ok(Listed::Model(LateBoundModel {
                model: ModelWrapper::BPE(model),
                words: AtomicUsize::new(HuggingFaceTokenizer::BPE_WORDS_CACHED_PER_THREAD),
                read: Some(ReadBpe {
                    pairs: Some(pairs.into_boxed_slice()),
                    max_id: listing.len().checked_sub(1),
                    dense: true,
                    tokens: Some(listing),
                    index: Some(index),
                }),
                rescore: false,
            }))
        }
        // Anything else goes back to `bpe_section` as the map and the texts
        // it reads itself.
        (vocab, merges) => {
            let merges = match (&vocab, merges) {
                (_, MergesRead::Texts(texts)) => texts,
                (VocabRead::Listed(listing, _), MergesRead::Ids(pairs)) => {
                    texts_of(listing, &pairs)
                }
                // Never: ids are resolved in a listing that is still the
                // vocabulary.
                (VocabRead::Map(_), MergesRead::Ids(_)) => {
                    return Err(A::Error::custom(
                        "merges resolved in a vocabulary that is no longer listed",
                    ))
                }
            };
            let vocab = match vocab {
                VocabRead::Map(vocab) => vocab,
                VocabRead::Listed(listing, _) => mapped(&listing),
            };
            Ok(Listed::Back(builder, vocab, merges))
        }
    }
}

/// What the listing reader made of a section.
enum Listed {
    /// The loading model of a listed section.
    Model(LateBoundModel),
    /// A section that is not listed: its settings, its vocabulary as a map
    /// and its merges as texts, which [`bpe_section`] builds.
    Back(BpeBuilder, Vocab, Vec<(String, String)>),
}

/// The way from [`bpe_section`] into the listing reader. Cold and never
/// inlined, so that the loops of that reader compile as they do without the
/// call: with the listing reader inlined among them, every other BPE file
/// paid seven instructions an entry of its vocabulary. The listing reader
/// itself reads a whole vocabulary and is not cold.
#[cold]
#[inline(never)]
fn listed_entry<'de, A: MapAccess<'de>>(
    map: A,
    builder: BpeBuilder,
    prefixed: bool,
) -> Result<Listed, A::Error> {
    listed_section(map, builder, prefixed)
}

/// The rest of a section that opened as a WordPiece section does: with
/// `type` saying so, or with one of the settings such a section has.
///
/// It is read here when it is the plain section: `unk_token`,
/// `continuing_subword_prefix`, `max_input_chars_per_word` and `vocab` once
/// each, of the type the crate's builder takes, `type` at most once and
/// saying WordPiece, and no other key. From such a section the crate's
/// reader builds a WordPiece model of those four values, with `type` or
/// without it, and so does this one, through the same builder. Any other
/// section goes to the crate's reader with its entries in the file's order,
/// and that reader decides as it did before: the same model, a model of
/// another kind, or its error.
///
/// The vocabulary streams into its map once the section has shown `type`
/// or `max_input_chars_per_word`, which no other kind of section has.
/// Before either, the section may still be a BPE or WordLevel one, and goes
/// to the crate's reader with its vocabulary unread, at the cost it had.
fn wordpiece_section<'de, A: MapAccess<'de>>(
    mut read: Vec<(String, Value)>,
    mut map: A,
) -> Result<LateBoundModel, A::Error> {
    // The vocabulary read as ids, and its place among the entries read.
    let mut vocab: Option<(usize, Vocab)> = None;
    let mut pending = None;
    while let Some(key) = map.next_key::<String>()? {
        let seen = |name: &str| read.iter().any(|(key, _)| key == name);
        match key.as_str() {
            "type" | "unk_token" | "continuing_subword_prefix" | "max_input_chars_per_word"
                if !seen(&key) =>
            {
                let value: Value = map.next_value()?;
                let other_kind = key == "type" && value != "WordPiece";
                read.push((key, value));
                if other_kind {
                    break;
                }
            }
            "vocab" if vocab.is_none() && (seen("type") || seen("max_input_chars_per_word")) => {
                match map.next_value_seed(IdVocabShape)? {
                    IdVocab::Ids(ids) => vocab = Some((read.len(), ids)),
                    IdVocab::Other(other) => {
                        read.push((key, other));
                        break;
                    }
                }
            }
            // A key twice, a key of another kind, or the vocabulary before
            // the section has shown its kind.
            _ => {
                pending = Some(key);
                break;
            }
        }
    }
    let settings = match pending {
        None => wordpiece_settings(&read),
        Some(_) => None,
    };
    match (settings, vocab) {
        (Some((unk_token, prefix, max_chars)), Some((_, vocab))) => {
            let read = ReadBpe::without_merges(&vocab);
            let model = WordPiece::builder()
                .vocab(vocab)
                .unk_token(unk_token)
                .continuing_subword_prefix(prefix)
                .max_input_chars_per_word(max_chars)
                .build()
                .map_err(A::Error::custom)?;
            Ok(LateBoundModel {
                model: ModelWrapper::WordPiece(model),
                words: AtomicUsize::new(HuggingFaceTokenizer::BPE_WORDS_CACHED_PER_THREAD),
                read: Some(read),
                rescore: false,
            })
        }
        (_, vocab) => {
            if let Some((at, ids)) = vocab {
                let ids = ids
                    .into_iter()
                    .map(|(token, id)| (token, Value::from(id)))
                    .collect();
                read.insert(at, ("vocab".to_string(), Value::Object(ids)));
            }
            crate_section(read, pending, map)
        }
    }
}

/// The three settings of a WordPiece section among the entries read, each
/// of the type the crate's builder takes: the unknown token, the prefix of
/// a continuing piece, the most characters a word may have. `None` when one
/// is missing or of another type, or when `type` names another kind.
fn wordpiece_settings(read: &[(String, Value)]) -> Option<(String, String, usize)> {
    let field = |name: &str| {
        read.iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    };
    if field("type").is_some_and(|kind| kind != "WordPiece") {
        return None;
    }
    Some((
        field("unk_token")?.as_str()?.to_string(),
        field("continuing_subword_prefix")?.as_str()?.to_string(),
        usize::try_from(field("max_input_chars_per_word")?.as_u64()?).ok()?,
    ))
}

/// The vocabulary of a WordPiece section as [`IdVocabShape`] reads it.
enum IdVocab {
    /// A map whose every value is an id: a whole number that fits a `u32`,
    /// which is what the crate's reader takes.
    Ids(Vocab),
    /// A map with any other value in it, or a list: the JSON it is, kept
    /// for the crate's reader.
    Other(Value),
}

/// Reads a `vocab` entry by its shape (see [`IdVocab`]). A vocabulary that
/// is neither a map nor a list is refused, as every kind of model refuses
/// it.
struct IdVocabShape;

impl<'de> DeserializeSeed<'de> for IdVocabShape {
    type Value = IdVocab;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<IdVocab, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for IdVocabShape {
    type Value = IdVocab;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a vocabulary")
    }

    fn visit_seq<S: SeqAccess<'de>>(self, seq: S) -> Result<IdVocab, S::Error> {
        Value::deserialize(de::value::SeqAccessDeserializer::new(seq)).map(IdVocab::Other)
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<IdVocab, M::Error> {
        let mut ids = Vocab::default();
        while let Some(token) = map.next_key::<String>()? {
            let value: Value = map.next_value()?;
            match value.as_u64().and_then(|id| u32::try_from(id).ok()) {
                Some(id) => {
                    ids.insert(token, id);
                }
                // Not an id: the whole vocabulary as JSON, a token read
                // twice keeping its later value as in the crate's reader.
                None => {
                    let mut other: serde_json::Map<String, Value> = ids
                        .into_iter()
                        .map(|(token, id)| (token, Value::from(id)))
                        .collect();
                    other.insert(token, value);
                    while let Some((token, value)) = map.next_entry()? {
                        other.insert(token, value);
                    }
                    return Ok(IdVocab::Other(Value::Object(other)));
                }
            }
        }
        Ok(IdVocab::Ids(ids))
    }
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

/// The vocabulary of a BPE section as [`ListedVocab`] reads it.
enum VocabRead {
    /// Every token's id is its place in the section and no text comes
    /// twice: the tokens listed, and the ids by their text.
    Listed(Arc<Listing>, Arc<TextIndex>),
    /// Any other vocabulary: the map the crate's builder takes.
    Map(Vocab),
}

/// The merges of a BPE section as [`MergeIds`] reads them.
enum MergesRead {
    /// Every merge names tokens of the vocabulary: left, right, product.
    Ids(Vec<[u32; 3]>),
    /// The merges as the crate's builder takes them.
    Texts(Vec<(String, String)>),
}

/// The listing as the map the crate's builder takes.
fn mapped(listing: &Listing) -> Vocab {
    (0..listing.len())
        .filter_map(|id| Some((listing.get(id)?.to_owned(), id)))
        .collect()
}

/// Resolved merges as the texts the crate's builder takes.
fn texts_of(listing: &Listing, pairs: &[[u32; 3]]) -> Vec<(String, String)> {
    let text = |id: u32| listing.get(id).unwrap_or_default().to_owned();
    pairs.iter().map(|&[a, b, _]| (text(a), text(b))).collect()
}

/// A string of the file appended to a buffer, no allocation a string.
struct Appended<'a>(&'a mut String);

impl<'de> DeserializeSeed<'de> for Appended<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl Visitor<'_> for Appended<'_> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a string")
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<(), E> {
        self.0.push_str(text);
        Ok(())
    }
}

/// Reads a `vocab` entry of a BPE section into a [`Listing`] while its ids
/// come as `0..n` in the file's order, with the ids by their text; a
/// vocabulary that is not so written, that spells one text twice, or whose
/// text is longer than a span's 32 bits address, is the map the crate's
/// builder takes.
struct ListedVocab;

impl<'de> DeserializeSeed<'de> for ListedVocab {
    type Value = VocabRead;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<VocabRead, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for ListedVocab {
    type Value = VocabRead;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a map")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<VocabRead, M::Error> {
        let mut text = String::new();
        let mut spans: Vec<(u32, u32)> = Vec::new();
        loop {
            let start = text.len();
            if map.next_key_seed(Appended(&mut text))?.is_none() {
                break;
            }
            let id: u32 = map.next_value()?;
            let end = u32::try_from(text.len())
                .ok()
                .filter(|_| id as usize == spans.len());
            let Some(end) = end else {
                let mut vocab: Vocab = spans
                    .iter()
                    .zip(0u32..)
                    .map(|(&(start, end), id)| (text[start as usize..end as usize].to_owned(), id))
                    .collect();
                vocab.insert(text[start..].to_owned(), id);
                // The rest by the seed of this reader: a second caller of
                // the map read of `bpe_section` cost that reader its
                // inlining there.
                let mut token = String::new();
                while map.next_key_seed(Appended(&mut token))?.is_some() {
                    vocab.insert(std::mem::take(&mut token), map.next_value()?);
                }
                return Ok(VocabRead::Map(vocab));
            };
            spans.push((start as u32, end));
        }
        text.shrink_to_fit();
        spans.shrink_to_fit();
        let listing = Listing { text, spans };
        let (index, twice) = TextIndex::build(&listing);
        if twice {
            return Ok(VocabRead::Map(mapped(&listing)));
        }
        Ok(VocabRead::Listed(Arc::new(listing), Arc::new(index)))
    }
}

/// The merge list of a BPE section whose vocabulary is listed,
/// each merge resolved in it as it is read, no string kept; from the first
/// merge that names a text outside the vocabulary on, the texts, for the
/// crate's builder to refuse.
struct MergeIds<'a> {
    /// `None`: no listed vocabulary to resolve in, the texts from the start.
    listed: Option<(&'a Listing, &'a TextIndex)>,
}

impl<'de> DeserializeSeed<'de> for MergeIds<'_> {
    type Value = MergesRead;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<MergesRead, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for MergeIds<'_> {
    type Value = MergesRead;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a list of merges")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<MergesRead, A::Error> {
        let about = self.listed.map_or(0, |(listing, _)| listing.len() as usize);
        let mut pairs = Vec::with_capacity(seq.size_hint().unwrap_or(about));
        // The merges as texts, from the first that does not resolve on.
        let mut texts: Option<Vec<(String, String)>> = self.listed.is_none().then(Vec::new);
        let mut both = String::new();
        loop {
            both.clear();
            let Some(cut) = seq.next_element_seed(MergeText(&mut both))? else {
                break;
            };
            // A `#version` line.
            let Some(cut) = cut else { continue };
            let (left, right) = both.split_at(cut);
            if let (None, Some((listing, index))) = (&texts, self.listed) {
                let id = |text: &str| index.get(listing, text.as_bytes());
                match (id(left), id(right), id(&both)) {
                    (Some(a), Some(b), Some(product)) => {
                        pairs.push([a, b, product]);
                        continue;
                    }
                    _ => texts = Some(texts_of(listing, &pairs)),
                }
            }
            if let Some(texts) = &mut texts {
                texts.push((left.to_owned(), right.to_owned()));
            }
        }
        Ok(match texts {
            Some(texts) => MergesRead::Texts(texts),
            None => MergesRead::Ids(pairs),
        })
    }
}

/// One merge as the file spells it ([`MergeSpelling`]), its two
/// pieces appended to the buffer: where the first ends, or `None` for a
/// `#version` line.
struct MergeText<'a>(&'a mut String);

impl<'de> DeserializeSeed<'de> for MergeText<'_> {
    type Value = Option<usize>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for MergeText<'_> {
    type Value = Option<usize>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a merge: a pair of pieces, or one string with a space between them")
    }

    fn visit_str<E: de::Error>(self, line: &str) -> Result<Option<usize>, E> {
        if line.starts_with("#version") {
            return Ok(None);
        }
        let mut pieces = line.split(' ');
        match (pieces.next(), pieces.next(), pieces.next()) {
            (Some(a), Some(b), None) => {
                self.0.push_str(a);
                self.0.push_str(b);
                Ok(Some(a.len()))
            }
            _ => Err(E::invalid_value(de::Unexpected::Str(line), &self)),
        }
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Option<usize>, A::Error> {
        if seq.next_element_seed(Appended(&mut *self.0))?.is_none() {
            return Err(A::Error::invalid_length(0, &self));
        }
        let cut = self.0.len();
        if seq.next_element_seed(Appended(&mut *self.0))?.is_none() {
            return Err(A::Error::invalid_length(1, &self));
        }
        if seq.next_element::<IgnoredAny>()?.is_some() {
            return Err(A::Error::invalid_length(3, &self));
        }
        Ok(Some(cut))
    }
}

impl LateBoundModel {
    /// The read that answers for the model where the section's model was
    /// not built.
    fn listed(&self) -> Option<(&Listing, &TextIndex)> {
        let read = self.read.as_ref()?;
        Some((read.tokens.as_deref()?, read.index.as_deref()?))
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
        match self.listed() {
            Some((listing, index)) => index.get(listing, token.as_bytes()),
            None => self.model.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match self.listed() {
            Some((listing, _)) => listing.get(id).map(str::to_owned),
            None => self.model.id_to_token(id),
        }
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        match self.listed() {
            Some((listing, _)) => (0..listing.len())
                .filter_map(|id| Some((listing.get(id)?.to_owned(), id)))
                .collect(),
            None => self.model.get_vocab(),
        }
    }

    fn get_vocab_size(&self) -> usize {
        match self.listed() {
            Some((listing, _)) => listing.len() as usize,
            None => self.model.get_vocab_size(),
        }
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

    /// The entries read before the kind is known go back in front, in the
    /// file's order, for the crate's reader: a BPE section with its keys
    /// sorted (`byte_fallback`, then a key no Unigram section has),
    /// `byte_fallback` then `type`, a map vocabulary before `type`, and
    /// `byte_fallback` then a map vocabulary, where two entries are held
    /// and the model needs both.
    #[test]
    fn the_entries_read_before_the_kind_is_known_reach_the_crate_reader() {
        let fields: std::collections::BTreeMap<String, Value> =
            serde_json::from_str(&full_bpe_json()).expect("a BPE section");
        let section = |first: &[&str]| {
            let entry = |key: &str| format!("{}:{}", Value::from(key), fields[key]);
            let rest = fields
                .keys()
                .filter(|key| !first.contains(&key.as_str()))
                .map(|key| entry(key));
            let entries: Vec<String> = first.iter().map(|key| entry(key)).chain(rest).collect();
            format!("{{{}}}", entries.join(","))
        };
        for json in [
            section(&[]),
            section(&["byte_fallback", "type"]),
            section(&["vocab", "byte_fallback"]),
            section(&["byte_fallback", "vocab"]),
        ] {
            assert_eq!(loading_reading(&json), crate_reading(&json), "{json}");
        }
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
    fn a_section_of_another_kind_that_carries_unk_id_takes_the_crate_reader() {
        // Only a Unigram section has `unk_id`, but the crate's reader skips
        // a key it does not know: the key does not claim the section.
        let json = r#"{"unk_id":0,"type":"BPE","vocab":{"a":0},"merges":[]}"#;
        assert_eq!(loading_reading(json), crate_reading(json));
    }

    #[test]
    fn only_a_unigram_model_of_the_crate_reader_asks_for_a_second_read() {
        // A Unigram section the section's own reader takes, named and
        // unnamed; a BPE section in one pass and through the crate's reader.
        let sorted: std::collections::BTreeMap<String, Value> =
            serde_json::from_str(&full_bpe_json()).expect("a BPE section");
        for json in [
            r#"{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["a",-1.0]]}"#.to_string(),
            r#"{"unk_id":0,"vocab":[["<unk>",0.0],["a",-1.0]]}"#.to_string(),
            full_bpe_json(),
            serde_json::to_string(&sorted).expect("serializes"),
        ] {
            let late: LateBoundModel = serde_json::from_str(&json).expect("the section reads");
            assert!(!late.rescore(), "{json}");
        }
    }

    #[test]
    fn a_key_of_another_kind_hands_the_section_to_the_crate_reader() {
        // Each is a key of a BPE, WordPiece or WordLevel section, and no
        // Unigram section has it. The section names no kind, so it is the
        // crate's when its vocabulary comes. The scores are the tie of
        // `unigram_scores`, which the crate's reader and the section's own
        // read one bit apart: the reading is the crate's.
        for entry in [
            r#""dropout":null"#,
            r#""unk_token":"<unk>""#,
            r#""continuing_subword_prefix":"""#,
            r#""end_of_word_suffix":"""#,
            r#""fuse_unk":false"#,
            r#""ignore_merges":false"#,
            r#""merges":[]"#,
            r#""max_input_chars_per_word":100"#,
        ] {
            let json = format!(
                r#"{{{entry},"unk_id":0,"vocab":[["<unk>",0.0],["a",-1.1695698339729451],["aa",-2.3391396679458905]]}}"#
            );
            assert_eq!(loading_reading(&json), crate_reading(&json), "{json}");
            // The loader reads the scores of such a model once more.
            let late: LateBoundModel = serde_json::from_str(&json).expect("the section reads");
            assert!(late.rescore(), "{json}");
        }
    }

    #[test]
    fn another_model_kind_or_a_type_key_later_reads_as_the_crate_reads_it() {
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
        // crate's older files): the model the crate's reader builds (a BPE
        // section through that reader, a WordPiece section in one pass).
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

    /// A WordPiece section with every setting off its default, as the
    /// entries the crate writes: `type`, the three settings, the vocabulary.
    fn word_piece_entries() -> Vec<String> {
        let mut vocab = Vocab::default();
        for (id, token) in ["<unk>", "a", "_b", "ab"].into_iter().enumerate() {
            vocab.insert(token.to_string(), id as u32);
        }
        let model = WordPiece::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .continuing_subword_prefix("_".to_string())
            .max_input_chars_per_word(7)
            .build()
            .expect("a WordPiece with every setting set");
        let json = serde_json::to_string(&ModelWrapper::WordPiece(model)).expect("serializes");
        let section: Map<String, Value> =
            serde_json::from_str(&json).expect("the section is a map");
        section
            .iter()
            .map(|(key, value)| format!("{}:{value}", Value::from(key.as_str())))
            .collect()
    }

    /// The entries as one section.
    fn section_of(entries: &[String]) -> String {
        format!("{{{}}}", entries.join(","))
    }

    /// Every order of the entries.
    fn orders(entries: &[String]) -> Vec<Vec<String>> {
        if entries.len() < 2 {
            return vec![entries.to_vec()];
        }
        let mut all = Vec::new();
        for first in 0..entries.len() {
            let mut rest = entries.to_vec();
            let head = rest.remove(first);
            for mut order in orders(&rest) {
                order.insert(0, head.clone());
                all.push(order);
            }
        }
        all
    }

    /// What a reader makes of a section: the model as it serializes, or the
    /// text of its error without the place in the file.
    fn outcome<M: de::DeserializeOwned + Serialize>(json: &str) -> Result<String, String> {
        match serde_json::from_str::<M>(json) {
            Ok(model) => Ok(serde_json::to_string(&model).expect("the model serializes")),
            Err(error) => {
                let text = error.to_string();
                let place = text.find(" at line ").unwrap_or(text.len());
                Err(text[..place].to_string())
            }
        }
    }

    /// Whether the loading model's reader read the section itself, in one
    /// pass: it then leaves a read, where the crate's reader leaves none.
    fn in_one_pass(json: &str) -> bool {
        serde_json::from_str::<LateBoundModel>(json).is_ok_and(|late| late.read().is_some())
    }

    /// The fields `wordpiece_section` reads are the crate's: a `tokenizers`
    /// bump that adds a WordPiece field fails here, where every file the
    /// crate then writes would otherwise take the slower reader quietly.
    #[test]
    fn the_one_pass_reader_knows_every_field_of_a_word_piece_section() {
        let section: Map<String, Value> =
            serde_json::from_str(&section_of(&word_piece_entries())).expect("the section is a map");
        let keys: Vec<&str> = section.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "type",
                "unk_token",
                "continuing_subword_prefix",
                "max_input_chars_per_word",
                "vocab",
            ],
            "the crate's WordPiece gained or lost a field: teach wordpiece_section to read it"
        );
    }

    #[test]
    fn a_word_piece_section_reads_as_the_crate_reads_it_in_any_order_of_its_keys() {
        let entries = word_piece_entries();
        let json = section_of(&entries);
        // With `type` and, as the older files are, without it.
        for entries in [&entries[..], &entries[1..]] {
            for order in orders(entries) {
                let text = section_of(&order);
                assert_eq!(outcome::<ModelWrapper>(&text).as_ref(), Ok(&json), "{text}");
                assert_eq!(
                    outcome::<LateBoundModel>(&text).as_ref(),
                    Ok(&json),
                    "{text}"
                );
                // In one pass when the section shows its kind before its
                // vocabulary; through the crate's reader otherwise.
                let at = |key: &str| order.iter().position(|entry| entry.starts_with(key));
                let kind = at(r#""type""#).into_iter().chain(at(r#""max_input"#)).min();
                assert_eq!(in_one_pass(&text), kind < at(r#""vocab""#), "{text}");
            }
        }
    }

    #[test]
    fn a_word_piece_section_with_a_key_of_no_kind_takes_the_crate_reader() {
        let entries = word_piece_entries();
        let json = section_of(&entries);
        for entries in [&entries[..], &entries[1..]] {
            for at in 0..=entries.len() {
                let mut with = entries.to_vec();
                with.insert(at, r#""future":{"x":[1,2]}"#.to_string());
                let text = section_of(&with);
                assert_eq!(outcome::<ModelWrapper>(&text).as_ref(), Ok(&json), "{text}");
                assert_eq!(
                    outcome::<LateBoundModel>(&text).as_ref(),
                    Ok(&json),
                    "{text}"
                );
                assert!(!in_one_pass(&text), "{text}");
            }
        }
    }

    #[test]
    fn a_section_of_another_kind_that_opens_like_a_word_piece_section_takes_the_crate_reader() {
        let mut words = Vocab::default();
        for (id, token) in ["[UNK]", "a", "b"].into_iter().enumerate() {
            words.insert(token.to_string(), id as u32);
        }
        let word_level = WordLevel::builder()
            .vocab(words)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("a word-level model");
        let word_level =
            serde_json::to_string(&ModelWrapper::WordLevel(word_level)).expect("serializes");
        let bpe = full_bpe_json();
        // The section with one of its keys moved to the front, with `type`
        // after it or without `type`.
        let opened_with = |json: &str, first: &str, with_type: bool| {
            let fields: Map<String, Value> =
                serde_json::from_str(json).expect("the section is a map");
            let entry = |key: &str| format!("{}:{}", Value::from(key), fields[key]);
            let rest = fields
                .keys()
                .filter(|key| *key != first && (with_type || *key != "type"))
                .map(|key| entry(key));
            section_of(
                &std::iter::once(entry(first))
                    .chain(rest)
                    .collect::<Vec<_>>(),
            )
        };
        for (json, first) in [
            (&bpe, "unk_token"),
            (&bpe, "continuing_subword_prefix"),
            (&word_level, "unk_token"),
        ] {
            for with_type in [true, false] {
                let text = opened_with(json, first, with_type);
                assert_eq!(outcome::<ModelWrapper>(&text).as_ref(), Ok(json), "{text}");
                assert_eq!(
                    outcome::<LateBoundModel>(&text).as_ref(),
                    Ok(json),
                    "{text}"
                );
                assert!(!in_one_pass(&text), "{text}");
            }
        }
    }

    #[test]
    fn a_word_piece_field_of_another_type_reads_as_the_crate_reads_it() {
        let entries = word_piece_entries();
        for (at, other) in [
            (1, r#""unk_token":null"#),
            (1, r#""unk_token":7"#),
            (2, r#""continuing_subword_prefix":["_"]"#),
            (3, r#""max_input_chars_per_word":"7""#),
            (3, r#""max_input_chars_per_word":-7"#),
            (3, r#""max_input_chars_per_word":7.5"#),
            (4, r#""vocab":{"<unk>":0,"a":1.5,"_b":2,"ab":3}"#),
            (4, r#""vocab":{"<unk>":0,"a":-1}"#),
            (4, r#""vocab":{"<unk>":0,"a":4294967296}"#),
            (4, r#""vocab":{"<unk>":0,"a":"1"}"#),
            (4, r#""vocab":[["<unk>",0.0],["a",-1.5]]"#),
        ] {
            let mut with = entries.clone();
            with[at] = other.to_string();
            // With `type` the crate's reader has no model for it; without
            // `type` it may build another kind from the fields that fit one.
            assert!(
                outcome::<ModelWrapper>(&section_of(&with)).is_err(),
                "{other}"
            );
            for entries in [&with[..], &with[1..]] {
                let text = section_of(entries);
                assert_eq!(
                    outcome::<LateBoundModel>(&text),
                    outcome::<ModelWrapper>(&text),
                    "{text}"
                );
                assert!(!in_one_pass(&text), "{text}");
            }
        }
        // A vocabulary that is neither a map nor a list: no model of any kind.
        for other in [r#""vocab":null"#, r#""vocab":"a""#, r#""vocab":7"#] {
            let mut with = entries.clone();
            with[4] = other.to_string();
            for entries in [&with[..], &with[1..]] {
                let text = section_of(entries);
                assert!(outcome::<ModelWrapper>(&text).is_err(), "{text}");
                assert!(outcome::<LateBoundModel>(&text).is_err(), "{text}");
            }
        }
    }

    #[test]
    fn a_key_twice_in_a_word_piece_section_reads_as_the_crate_reads_it() {
        let entries = word_piece_entries();
        let json = section_of(&entries);
        // The crate's reader keeps the later of two settings or of two
        // vocabularies, and has no model for `type` twice.
        for again in [
            r#""type":"WordPiece""#,
            r#""unk_token":"a""#,
            r#""continuing_subword_prefix":"a""#,
            r#""max_input_chars_per_word":1"#,
            r#""vocab":{"a":0}"#,
        ] {
            for at in [0, entries.len()] {
                let mut with = entries.clone();
                with.insert(at, again.to_string());
                let text = section_of(&with);
                assert_eq!(
                    outcome::<LateBoundModel>(&text),
                    outcome::<ModelWrapper>(&text),
                    "{text}"
                );
                assert!(!in_one_pass(&text), "{text}");
            }
        }
        // A token twice in the vocabulary keeps its later id, in one pass.
        let mut with = entries;
        with[4] = r#""vocab":{"<unk>":0,"a":9,"a":1,"_b":2,"ab":3}"#.to_string();
        let text = section_of(&with);
        assert_eq!(outcome::<ModelWrapper>(&text).as_ref(), Ok(&json));
        assert_eq!(outcome::<LateBoundModel>(&text).as_ref(), Ok(&json));
        assert!(in_one_pass(&text));
    }

    #[test]
    fn a_word_piece_section_with_a_field_missing_reads_as_the_crate_reads_it() {
        let entries = word_piece_entries();
        for missing in 1..entries.len() {
            let mut without = entries.clone();
            without.remove(missing);
            // With `type` the crate's reader names the missing field;
            // without `type` it may build another kind from the rest.
            assert!(outcome::<ModelWrapper>(&section_of(&without)).is_err());
            for entries in [&without[..], &without[1..]] {
                let text = section_of(entries);
                assert_eq!(
                    outcome::<LateBoundModel>(&text),
                    outcome::<ModelWrapper>(&text),
                    "{text}"
                );
                assert!(!in_one_pass(&text), "{text}");
            }
        }
    }

    #[test]
    fn a_word_piece_section_read_in_one_pass_reports_its_vocabulary() {
        let entries = word_piece_entries();
        let late: LateBoundModel =
            serde_json::from_str(&section_of(&entries)).expect("a WordPiece section");
        let read = late.read().expect("a WordPiece section leaves a read");
        assert!(read.pairs.is_none(), "a WordPiece section has no merges");
        assert_eq!((read.max_id, read.dense), (Some(3), true));
        let tokens = read.tokens.as_deref().expect("dense ids are listed");
        assert_eq!(tokens.len(), 4);
        for id in 0..5 {
            assert_eq!(
                tokens.get(id).map(str::to_string),
                late.id_to_token(id),
                "id {id}"
            );
        }
        // A gap in the ids: the largest id, and no listing.
        let mut sparse = entries;
        sparse[4] = r#""vocab":{"<unk>":0,"a":1,"_b":2,"ab":7}"#.to_string();
        let late: LateBoundModel =
            serde_json::from_str(&section_of(&sparse)).expect("a WordPiece section");
        let read = late.read().expect("a WordPiece section leaves a read");
        assert_eq!((read.max_id, read.dense), (Some(7), false));
        assert!(read.tokens.is_none());
    }

    /// A Unigram section as the entries the crate writes: `type`, the
    /// unknown piece, the pieces with their scores (one piece twice), the
    /// byte fallback.
    fn unigram_entries() -> Vec<String> {
        let pieces = [("<unk>", 0.0), ("a", -1.0), ("ab", -2.5), ("a", -3.0)];
        let model = Unigram::from(
            pieces
                .into_iter()
                .map(|(piece, score)| (piece.to_string(), score))
                .collect(),
            Some(0),
            false,
        )
        .expect("a Unigram model");
        let json = serde_json::to_string(&ModelWrapper::Unigram(model)).expect("serializes");
        let section: Map<String, Value> =
            serde_json::from_str(&json).expect("the section is a map");
        section
            .iter()
            .map(|(key, value)| format!("{}:{value}", Value::from(key.as_str())))
            .collect()
    }

    #[test]
    fn a_unigram_section_reports_its_largest_id_in_any_order_of_its_keys() {
        let entries = unigram_entries();
        assert_eq!(entries.len(), 4, "type, unk_id, vocab, byte_fallback");
        // With `type` and without it: without `type`, the list of scored
        // pieces tells the kind, and `unk_id` is held until then.
        for entries in [&entries[..], &entries[1..]] {
            for order in orders(entries) {
                let text = section_of(&order);
                let late: LateBoundModel = serde_json::from_str(&text).expect("a Unigram section");
                let read = late.read().unwrap_or_else(|| panic!("no read of {text}"));
                // The report is the range of the ids: four pieces, the
                // two equal ones each on an id of its own, so 0 to 3.
                assert_eq!((read.max_id, read.dense), (Some(3), true), "{text}");
                // It carries the ids alone. The pieces stay with the
                // model, which the tables ask by id.
                assert!(read.pairs.is_none() && read.tokens.is_none(), "{text}");
                let pieces: Vec<_> = (0..5).filter_map(|id| late.id_to_token(id)).collect();
                assert_eq!(pieces, ["<unk>", "a", "ab", "a"], "{text}");
            }
        }
    }

    #[test]
    fn a_unigram_section_without_pieces_reports_an_empty_range_of_ids() {
        // Without an unknown piece: the crate's builder refuses an empty
        // list that names one.
        let late: LateBoundModel =
            serde_json::from_str(r#"{"type":"Unigram","unk_id":null,"vocab":[]}"#)
                .expect("a Unigram section without pieces");
        let read = late.read().expect("a Unigram section leaves a read");
        // The empty range: dense, and no id is its largest.
        assert_eq!((read.max_id, read.dense), (None, true));
    }

    #[test]
    fn a_unigram_section_reports_past_a_held_key_and_not_through_the_word_piece_reader() {
        let entries = unigram_entries();
        let json = section_of(&entries);
        // A key no section has: in front it is held until an entry tells
        // the kind, behind that entry this reader skips it as the crate's
        // does. The model is the crate's and the section reports, wherever
        // the key stands.
        for at in 0..=entries.len() {
            let mut with = entries.clone();
            with.insert(at, r#""future":{"x":[1,2]}"#.to_string());
            let text = section_of(&with);
            assert_eq!(outcome::<ModelWrapper>(&text).as_ref(), Ok(&json), "{text}");
            assert_eq!(
                outcome::<LateBoundModel>(&text).as_ref(),
                Ok(&json),
                "{text}"
            );
            assert!(in_one_pass(&text), "{text}");
        }
        // A key of another kind of section in front of `type`. `dropout` is
        // held and `type` still tells the kind: the section reports. A key
        // a WordPiece section opens with sends the section through that
        // reader to the crate's: the model is the crate's, with no report,
        // and the loader reads its scores once more.
        for (key, reports) in [
            (r#""dropout":null"#, true),
            (r#""unk_token":"<unk>""#, false),
        ] {
            let mut with = entries.clone();
            with.insert(0, key.to_string());
            let text = section_of(&with);
            assert_eq!(outcome::<ModelWrapper>(&text).as_ref(), Ok(&json), "{text}");
            assert_eq!(
                outcome::<LateBoundModel>(&text).as_ref(),
                Ok(&json),
                "{text}"
            );
            assert_eq!(in_one_pass(&text), reports, "{text}");
        }
    }
}
