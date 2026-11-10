//! The engines' KV-cache event wire format and its normalization into the
//! proto the gateway indexes.
//!
//! Both vLLM (`vllm/distributed/kv_events.py`) and SGLang
//! (`sglang/srt/disaggregation/kv_events.py`) publish one msgspec batch per
//! scheduler step: a positional array `[ts, events, dp_rank]` whose events
//! are tagged maps (`{"type": "BlockStored", ...}`, defaulted fields
//! omitted). Older publishers emit events as positional arrays with the tag
//! first; both layouts decode here.
//!
//! Normalization applies, per stream, what a per-worker prefix index needs
//! and nothing else:
//! - only local blocks (`locality` absent or `LOCAL`); remote ones describe a
//!   shared pool and are dropped;
//! - only blocks the engine manages itself (`ownership` other than a
//!   residency agent);
//! - the storage tier from `medium`, unknown media dropped;
//! - the KV cache groups by their kind: the main-attention group's stores
//!   carry their tokens and feed the prefix index; the other groups' events
//!   on a rank that has a main group (hybrid models publish one group per
//!   attention kind under the same hashes) are forwarded with their hashes
//!   and group fields and no tokens, what a consumer needs to bound a prefix
//!   hit by the blocks a sliding-window or state group retains; a rank that
//!   publishes only sliding-window groups forwards their stores, the hashes
//!   aligned to the tail of the tokens, as its only signal (see
//!   [`Normalizer`]);
//! - whole blocks only (hash count × block size == token count, or the
//!   tail of a longer span), no self-referencing hash chains, no offload
//!   placeholders (a chunk key with no tokens);
//! - speculative-decoding bigram pages folded to their tokens, so an Eagle
//!   engine's blocks hash like a plain engine's;
//! - the cache namespace (LoRA name, cache salt) carried on every store and
//!   inherited down the parent chain when a child store omits it.
//!
//! Stores and removals are forwarded one for one. vLLM keeps up to two
//! physical copies of one hash and removes them one at a time, so a removal
//! can arrive while another copy is still cached: every copy's store and
//! every removal go through, and the gateway counts copies per worker, tier
//! and hash (capped), as the relay's own live-block record and the hash
//! check's parent memory do.
//!
//! Hash identity: an integer hash is used as is (vLLM sends the low 64 bits
//! of the digest as an unsigned integer, SGLang the high 64 bits as a signed
//! one; both are 64-bit patterns the proto carries as `int64`); a raw digest
//! folds to its last eight bytes big-endian, the integer vLLM would have sent
//! for it.

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use serde::{
    de::{self, IgnoredAny, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer,
};
use smg_grpc_client::common_proto::{
    self as common, kv_block_extra_key, kv_cache_event, KvCacheLocality, KvCacheTier,
};
use tracing::{debug, warn};

use crate::{
    engine_hash::{self, Digest32, EngineHash, VllmExtraKey},
    kv_state::COPIES_CAP,
};

/// `int.from_bytes(bytes, "big")` kept to 64 bits: the whole value for the
/// publisher's eight-byte sequence frame, the low 64 bits of a longer hash.
pub(crate) fn low64_big_endian(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0, |value, &byte| (value << 8) | u64::from(byte))
}

/// A publisher batch: msgspec `array_like`, `[ts, events, dp_rank]`, the
/// rank named `data_parallel_rank` by vLLM and `attn_dp_rank` by SGLang and
/// omittable by both; later fields are tolerated by the caller's codec.
#[derive(Deserialize)]
pub struct WireBatch {
    pub ts: f64,
    pub events: Vec<WireEvent>,
    #[serde(default)]
    pub dp_rank: Option<i32>,
}

/// A block hash as the proto's signed 64-bit identity: sha256 bytes keep
/// their low 64 bits read big-endian; an int is already 64 bits wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockHash(pub i64);

impl<'de> Deserialize<'de> for BlockHash {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct HashVisitor;

        impl Visitor<'_> for HashVisitor {
            type Value = BlockHash;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a block hash as bytes or an integer")
            }

            fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<Self::Value, E> {
                Ok(BlockHash(low64_big_endian(bytes) as i64))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(BlockHash(value as i64))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(BlockHash(value))
            }
        }

        deserializer.deserialize_any(HashVisitor)
    }
}

/// The fields a store and a removal share beyond hashes and tokens.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EventTail {
    pub medium: Option<String>,
    pub group_idx: Option<u32>,
    pub kv_cache_spec_kind: Option<String>,
    pub kv_cache_spec_sliding_window: Option<u32>,
    pub locality: Option<String>,
    pub ownership: Option<String>,
    pub session_id: Option<String>,
    /// The cache groups the event is about, bit i for the engine's group i:
    /// on a store the groups that hold the boundary after it, on a removal
    /// the groups that dropped it. Written by a publisher that reports a
    /// boundary once for every group (after the reference layout's last
    /// slot); absent from one that reports each group on its own.
    pub group_mask: Option<u64>,
}

/// Token ids of a store: plain ids, or the (token, next token) bigrams a
/// speculative-decoding (Eagle) publisher emits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireTokens {
    Ids(Vec<u32>),
    Bigrams(Vec<(u32, u32)>),
}

/// One entry of vLLM's untagged per-block `extra_keys`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtraKey {
    Text(String),
    Number(i64),
    Blob(Vec<u8>),
    Multimodal {
        identifier: String,
        offset: i64,
    },
    /// An item of a shape this relay does not model; kept as a marker so the
    /// per-block key count stays truthful.
    Opaque,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WireStored {
    pub block_hashes: Vec<BlockHash>,
    pub parent_block_hash: Option<BlockHash>,
    pub token_ids: WireTokens,
    pub block_size: i64,
    pub lora_id: Option<i64>,
    pub lora_name: Option<String>,
    pub cache_salt: Option<String>,
    /// One entry per block, `None` for a block without extra keys.
    pub extra_keys: Option<Vec<Option<Vec<ExtraKey>>>>,
    pub tail: EventTail,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WireRemoved {
    pub block_hashes: Vec<BlockHash>,
    pub tail: EventTail,
}

#[derive(Clone, Debug, PartialEq)]
pub enum WireEvent {
    BlockStored(WireStored),
    BlockRemoved(WireRemoved),
    AllBlocksCleared {
        ownership: Option<String>,
    },
    /// An event type this relay does not convert (one a newer engine added):
    /// skipped on its own so the batch's other events still go through.
    Unknown,
    /// A known event whose named field is missing or of a shape this relay
    /// cannot read: skipped on its own, the rest of the batch still goes
    /// through.
    Malformed(&'static str),
}

// ---------------------------------------------------------------------------
// Decoding: tagged maps and tag-first arrays
// ---------------------------------------------------------------------------

/// A loosely typed scalar for the optional tail slots of the array layout and
/// for `extra_keys` items.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Loose {
    Nil,
    Unsigned(u64),
    Signed(i64),
    Text(String),
    Bytes(Vec<u8>),
    Seq(Vec<Loose>),
    Other,
}

impl<'de> Deserialize<'de> for Loose {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LooseVisitor;

        impl<'de> Visitor<'de> for LooseVisitor {
            type Value = Loose;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a scalar, bytes, string, array or nil")
            }

            fn visit_unit<E: de::Error>(self) -> Result<Loose, E> {
                Ok(Loose::Nil)
            }

            fn visit_none<E: de::Error>(self) -> Result<Loose, E> {
                Ok(Loose::Nil)
            }

            fn visit_some<D2: Deserializer<'de>>(
                self,
                deserializer: D2,
            ) -> Result<Loose, D2::Error> {
                Loose::deserialize(deserializer)
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Loose, E> {
                Ok(Loose::Unsigned(u64::from(value)))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Loose, E> {
                Ok(Loose::Unsigned(value))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Loose, E> {
                Ok(if value >= 0 {
                    Loose::Unsigned(value as u64)
                } else {
                    Loose::Signed(value)
                })
            }

            fn visit_f64<E: de::Error>(self, _value: f64) -> Result<Loose, E> {
                Ok(Loose::Other)
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Loose, E> {
                Ok(Loose::Text(value.to_owned()))
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<Loose, E> {
                Ok(Loose::Text(value))
            }

            fn visit_bytes<E: de::Error>(self, value: &[u8]) -> Result<Loose, E> {
                Ok(Loose::Bytes(value.to_vec()))
            }

            fn visit_byte_buf<E: de::Error>(self, value: Vec<u8>) -> Result<Loose, E> {
                Ok(Loose::Bytes(value))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Loose, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element::<Loose>()? {
                    items.push(item);
                }
                Ok(Loose::Seq(items))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Loose, A::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(Loose::Other)
            }
        }

        deserializer.deserialize_any(LooseVisitor)
    }
}

impl Loose {
    fn text(self) -> Option<String> {
        match self {
            Loose::Text(text) => Some(text),
            _ => None,
        }
    }

    fn unsigned32(self) -> Option<u32> {
        match self {
            Loose::Unsigned(value) => u32::try_from(value).ok(),
            _ => None,
        }
    }

    fn unsigned64(self) -> Option<u64> {
        match self {
            Loose::Unsigned(value) => Some(value),
            _ => None,
        }
    }

    fn signed(self) -> Option<i64> {
        match self {
            Loose::Unsigned(value) => i64::try_from(value).ok(),
            Loose::Signed(value) => Some(value),
            _ => None,
        }
    }

    fn block_hash(&self) -> Option<BlockHash> {
        match self {
            Loose::Unsigned(value) => Some(BlockHash(*value as i64)),
            Loose::Signed(value) => Some(BlockHash(*value)),
            Loose::Bytes(bytes) => Some(BlockHash(low64_big_endian(bytes) as i64)),
            _ => None,
        }
    }

    fn block_hashes(self) -> Option<Vec<BlockHash>> {
        match self {
            Loose::Seq(items) => items.iter().map(Loose::block_hash).collect(),
            _ => None,
        }
    }

    fn tokens(self) -> Option<WireTokens> {
        let Loose::Seq(items) = self else {
            return None;
        };
        let mut ids = Vec::with_capacity(items.len());
        let mut pairs = Vec::new();
        for item in items {
            match item {
                Loose::Unsigned(value) => ids.push(u32::try_from(value).ok()?),
                Loose::Seq(pair) => match pair.as_slice() {
                    [Loose::Unsigned(first), Loose::Unsigned(second)] => {
                        pairs.push((u32::try_from(*first).ok()?, u32::try_from(*second).ok()?));
                    }
                    _ => return None,
                },
                _ => return None,
            }
        }
        if pairs.is_empty() {
            Some(WireTokens::Ids(ids))
        } else if ids.is_empty() {
            Some(WireTokens::Bigrams(pairs))
        } else {
            None
        }
    }

    fn extra_key(self) -> ExtraKey {
        match self {
            Loose::Text(text) => ExtraKey::Text(text),
            Loose::Unsigned(value) => {
                i64::try_from(value).map_or(ExtraKey::Opaque, ExtraKey::Number)
            }
            Loose::Signed(value) => ExtraKey::Number(value),
            Loose::Bytes(bytes) => ExtraKey::Blob(bytes),
            Loose::Seq(items) => match items.as_slice() {
                [Loose::Text(identifier), Loose::Unsigned(offset)] => ExtraKey::Multimodal {
                    identifier: identifier.clone(),
                    offset: i64::try_from(*offset).unwrap_or(i64::MAX),
                },
                [Loose::Text(identifier), Loose::Signed(offset)] => ExtraKey::Multimodal {
                    identifier: identifier.clone(),
                    offset: *offset,
                },
                _ => ExtraKey::Opaque,
            },
            Loose::Nil | Loose::Other => ExtraKey::Opaque,
        }
    }

    /// `extra_keys`: one list (or nil) per block.
    fn extra_keys(self) -> Option<Vec<Option<Vec<ExtraKey>>>> {
        match self {
            Loose::Seq(per_block) => Some(
                per_block
                    .into_iter()
                    .map(|keys| match keys {
                        Loose::Seq(items) => {
                            Some(items.into_iter().map(Loose::extra_key).collect())
                        }
                        _ => None,
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    /// The array layout's seventh store slot: vLLM's `lora_name`.
    fn namespace_slot(self) -> (Option<String>, Option<String>) {
        match self {
            Loose::Text(name) => (Some(name), None),
            _ => (None, None),
        }
    }
}

/// Everything a map-layout event may carry, collected before dispatch on
/// `type` so key order does not matter.
#[derive(Default)]
struct Fields {
    event_type: Option<String>,
    block_hashes: Option<Loose>,
    parent_block_hash: Option<Loose>,
    token_ids: Option<Loose>,
    block_size: Option<Loose>,
    lora_id: Option<Loose>,
    lora_name: Option<Loose>,
    cache_salt: Option<Loose>,
    extra_keys: Option<Loose>,
    tail: [Option<Loose>; 8],
}

const TAIL_KEYS: [&str; 8] = [
    "medium",
    "group_idx",
    "kv_cache_spec_kind",
    "kv_cache_spec_sliding_window",
    "locality",
    "ownership",
    "session_id",
    "group_mask",
];

fn tail_from(slots: [Option<Loose>; 8]) -> EventTail {
    let [medium, group_idx, kind, sliding, locality, ownership, session_id, group_mask] = slots;
    EventTail {
        medium: medium.and_then(Loose::text),
        group_idx: group_idx.and_then(Loose::unsigned32),
        kv_cache_spec_kind: kind.and_then(Loose::text),
        kv_cache_spec_sliding_window: sliding.and_then(Loose::unsigned32),
        locality: locality.and_then(Loose::text),
        ownership: ownership.and_then(Loose::text),
        session_id: session_id.and_then(Loose::text),
        group_mask: group_mask.and_then(Loose::unsigned64),
    }
}

impl Fields {
    fn into_event(self) -> WireEvent {
        let Some(event_type) = self.event_type else {
            return WireEvent::Malformed("type");
        };
        match event_type.as_str() {
            "BlockStored" => {
                let Some(block_hashes) = self.block_hashes.and_then(Loose::block_hashes) else {
                    return WireEvent::Malformed("block_hashes");
                };
                let Some(token_ids) = self.token_ids.and_then(Loose::tokens) else {
                    return WireEvent::Malformed("token_ids");
                };
                let Some(block_size) = self.block_size.and_then(Loose::signed) else {
                    return WireEvent::Malformed("block_size");
                };
                WireEvent::BlockStored(WireStored {
                    block_hashes,
                    parent_block_hash: self.parent_block_hash.and_then(|hash| hash.block_hash()),
                    token_ids,
                    block_size,
                    lora_id: self.lora_id.and_then(Loose::signed),
                    lora_name: self.lora_name.and_then(Loose::text),
                    cache_salt: self.cache_salt.and_then(Loose::text),
                    extra_keys: self.extra_keys.and_then(Loose::extra_keys),
                    tail: tail_from(self.tail),
                })
            }
            "BlockRemoved" => match self.block_hashes.and_then(Loose::block_hashes) {
                Some(block_hashes) => WireEvent::BlockRemoved(WireRemoved {
                    block_hashes,
                    tail: tail_from(self.tail),
                }),
                None => WireEvent::Malformed("block_hashes"),
            },
            "AllBlocksCleared" => {
                let [_, _, _, _, _, ownership, _, _] = self.tail;
                WireEvent::AllBlocksCleared {
                    ownership: ownership.and_then(Loose::text),
                }
            }
            _ => WireEvent::Unknown,
        }
    }
}

impl<'de> Deserialize<'de> for WireEvent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EventVisitor;

        impl<'de> Visitor<'de> for EventVisitor {
            type Value = WireEvent;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a KV cache event as a tagged map or a tag-first array")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<WireEvent, A::Error> {
                let mut fields = Fields::default();
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "type" => fields.event_type = map.next_value::<Loose>()?.text(),
                        "block_hashes" => fields.block_hashes = Some(map.next_value()?),
                        "parent_block_hash" => fields.parent_block_hash = Some(map.next_value()?),
                        "token_ids" => fields.token_ids = Some(map.next_value()?),
                        "block_size" => fields.block_size = Some(map.next_value()?),
                        "lora_id" => fields.lora_id = Some(map.next_value()?),
                        "lora_name" => fields.lora_name = Some(map.next_value()?),
                        "cache_salt" => fields.cache_salt = Some(map.next_value()?),
                        "extra_keys" => fields.extra_keys = Some(map.next_value()?),
                        other => match TAIL_KEYS.iter().position(|name| *name == other) {
                            Some(slot) => fields.tail[slot] = Some(map.next_value()?),
                            None => {
                                map.next_value::<IgnoredAny>()?;
                            }
                        },
                    }
                }
                Ok(fields.into_event())
            }

            /// The tag-first array layout, slots in the order vLLM's msgspec
            /// structs declare their fields: a store is `[tag, block_hashes,
            /// parent, token_ids, block_size, lora_id, medium, lora_name,
            /// extra_keys, group_idx, kind, sliding_window, locality,
            /// ownership, session_id]`, a removal `[tag, block_hashes,
            /// medium, group_idx, locality, ownership]`, a clear `[tag]`;
            /// trailing defaults are omitted. A publisher that reports a
            /// boundary once for every group appends `group_mask` after the
            /// layout's last slot of a store and of a removal. SGLang's legacy arrays put
            /// `cache_salt` where vLLM has `lora_name`; the two are not
            /// distinguishable there, and SGLang has published maps since it
            /// added the field.
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<WireEvent, A::Error> {
                let tag: String = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &"an event tag"))?;
                let mut slots = Vec::new();
                while let Some(slot) = seq.next_element::<Loose>()? {
                    slots.push(slot);
                }
                let mut slots = slots.into_iter();
                let mut next = || slots.next().unwrap_or(Loose::Nil);
                let mut fields = Fields {
                    event_type: Some(tag.clone()),
                    ..Fields::default()
                };
                match tag.as_str() {
                    "BlockStored" => {
                        fields.block_hashes = Some(next());
                        fields.parent_block_hash = Some(next());
                        fields.token_ids = Some(next());
                        fields.block_size = Some(next());
                        fields.lora_id = Some(next());
                        fields.tail[0] = Some(next());
                        let (lora_name, cache_salt) = next().namespace_slot();
                        fields.lora_name = lora_name.map(Loose::Text);
                        fields.cache_salt = cache_salt.map(Loose::Text);
                        fields.extra_keys = Some(next());
                        for slot in 1..=7 {
                            fields.tail[slot] = Some(next());
                        }
                    }
                    "BlockRemoved" => {
                        fields.block_hashes = Some(next());
                        for slot in [0, 1, 4, 5, 7] {
                            fields.tail[slot] = Some(next());
                        }
                    }
                    "AllBlocksCleared" => {
                        fields.tail[5] = Some(next());
                    }
                    _ => {}
                }
                Ok(fields.into_event())
            }
        }

        deserializer.deserialize_any(EventVisitor)
    }
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Why the relay dropped an event instead of forwarding it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DropReason {
    UnknownType,
    UnsupportedOwnership,
    NonLocalLocality,
    UnknownMedium,
    UnalignedBlocks,
    SelfReferencingHashes,
    /// A store with nothing to index: an offload chunk placeholder (a chunk
    /// key with no tokens) or an empty hash list.
    Placeholder,
    /// A known event with a field missing or unreadable.
    Malformed,
}

impl DropReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnknownType => "unknown_type",
            Self::UnsupportedOwnership => "unsupported_ownership",
            Self::NonLocalLocality => "non_local_locality",
            Self::UnknownMedium => "unknown_medium",
            Self::UnalignedBlocks => "unaligned_blocks",
            Self::SelfReferencingHashes => "self_referencing_hashes",
            Self::Placeholder => "placeholder",
            Self::Malformed => "malformed",
        }
    }
}

/// What one stream has forwarded and dropped, by reason.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub forwarded_stored: u64,
    pub forwarded_removed: u64,
    pub forwarded_cleared: u64,
    /// Forwarded stores whose every hash this stream had already seen on that
    /// rank and tier: vLLM's second physical copy, or a replayed batch.
    pub duplicate_stores: u64,
    /// Forwarded stores whose tokens arrived as speculative-decoding bigrams.
    pub bigram_stores: u64,
    /// Blocks rehashed the worker's way under the opt-in engine-hash check.
    pub hash_checked: u64,
    /// Checked blocks whose published hash differs from the recomputed one.
    pub hash_mismatch: u64,
    /// Blocks the check could not rehash: an unknown parent, a LoRA block
    /// (the adapter path is not in the event), or a key shape it does not
    /// model.
    pub hash_unverifiable: u64,
    /// Forwarded stores of a sliding-window group on a rank that showed no
    /// main-attention group (a window-only model): the rank's only signal,
    /// see the group policy on [`Normalizer`].
    pub window_only_stores: u64,
    /// Forwarded stores that named fewer blocks than their tokens spanned,
    /// the hashes aligned to the tail of the tokens.
    pub tail_aligned_stores: u64,
    /// Forwarded stores and removals of a cache group other than main
    /// attention on a rank that has a main group (a hybrid model's
    /// sliding-window and state groups): hashes and group fields, no
    /// tokens, see the group policy on [`Normalizer`].
    pub group_stores: u64,
    pub group_removals: u64,
    pub dropped: HashMap<DropReason, u64>,
}

impl Counts {
    pub fn dropped(&self, reason: DropReason) -> u64 {
        self.dropped.get(&reason).copied().unwrap_or(0)
    }

    /// `other` added to this tally: several streams' counters summed.
    pub fn add(&mut self, other: &Counts) {
        self.forwarded_stored += other.forwarded_stored;
        self.forwarded_removed += other.forwarded_removed;
        self.forwarded_cleared += other.forwarded_cleared;
        self.duplicate_stores += other.duplicate_stores;
        self.bigram_stores += other.bigram_stores;
        self.hash_checked += other.hash_checked;
        self.hash_mismatch += other.hash_mismatch;
        self.hash_unverifiable += other.hash_unverifiable;
        self.window_only_stores += other.window_only_stores;
        self.tail_aligned_stores += other.tail_aligned_stores;
        self.group_stores += other.group_stores;
        self.group_removals += other.group_removals;
        for (reason, count) in &other.dropped {
            *self.dropped.entry(*reason).or_insert(0) += count;
        }
    }
}

/// The cache namespace block hashes were computed under.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Namespace {
    pub lora_name: Option<String>,
    pub cache_salt: Option<String>,
}

impl Namespace {
    fn is_empty(&self) -> bool {
        self.lora_name.is_none() && self.cache_salt.is_none()
    }
}

/// What this stream remembers about a stored engine hash, kept until the
/// last of its physical copies is removed: the engines keep up to two
/// copies of a block and remove them one at a time, and the gateway counts
/// them the same way, so a child stored after one copy went still has its
/// parent here.
#[derive(Clone, Debug, Default)]
struct BlockRecord {
    /// The namespace it was stored under, for children that omit theirs.
    namespace: Option<Namespace>,
    /// Physical copies stored and not yet removed, capped as the gateway
    /// caps them.
    copies: u32,
    /// Its recomputed full digest when the hash check ran: what a child
    /// chains on.
    digest: Option<Digest32>,
}

#[derive(Default)]
struct RankState {
    /// Per tier: the engine hashes this stream has seen stored and not yet
    /// removed, with what it remembers about each.
    tiers: HashMap<i32, HashMap<i64, BlockRecord>>,
    /// KV cache groups seen on stores: whether each is a main-attention group.
    groups: HashMap<u32, bool>,
    /// Hashes forwarded from a sliding-window group while the rank had no
    /// main-attention group: the record holds them as main-lane blocks, so
    /// their removals reach it once the rank has a main group too.
    window_forwarded: HashSet<i64>,
}

impl RankState {
    fn main_seen(&self) -> bool {
        self.groups.values().any(|&main| main)
    }
}

/// Which lane of the stream an admitted event belongs to: the group policy's
/// answer for it (see [`Normalizer`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    /// The main-attention group, or an event with no group at all: whole
    /// blocks with their tokens, what the prefix index is built from.
    Main,
    /// A sliding-window group's event on a rank without a main-attention
    /// group, forwarded as the rank's only signal.
    WindowOnly,
    /// A cache group other than main attention on a rank that has a main
    /// group: forwarded with its hashes and group fields, no tokens.
    Group,
}

/// What the shared gates let through, and on which lane.
struct Admitted {
    tier: KvCacheTier,
    locality: KvCacheLocality,
    lane: Lane,
}

/// Per-stream normalization state (one engine endpoint, all its DP ranks).
///
/// KV-cache groups: an event names its group (`group_idx`) and, on stores,
/// the group's attention kind. A model with full attention alongside sliding
/// windows or state-space layers (vLLM's hybrid KV-cache manager) publishes
/// a block under the same hash once per group that keeps it: the
/// full-attention group holds the whole prefix, which is what a prefix hit
/// starts from, while a window group names only the window's blocks against
/// the whole computed span and a state group the one block ending at a
/// boundary, and each removes its own as it moves on. The engine's own hit
/// is bounded by every group: the prefix it reuses ends where the window
/// group's tail or the state group's boundary block is gone, however long
/// the full-attention chain runs. So on a rank with a main-attention group
/// the main group's events are forwarded whole and every other group's
/// events, stores and removals alike, are forwarded on their own lane
/// (`Lane::Group`): the hashes, the parent and the group fields, no
/// tokens (a sparse group's token span is the step's whole range, not a
/// token list per hash, and a consumer keys the group's blocks by their
/// hash). The gateway keeps those memberships apart from its prefix index,
/// so a window group's removal never takes a full-attention copy out of its
/// count. A rank that shows only sliding-window groups (a window-only model)
/// would otherwise give the gateway nothing: its window stores are forwarded
/// with their group identity, the hashes aligned to the tail of the tokens,
/// and counted ([`Counts::window_only_stores`]). The groups are learned for
/// the rank's life: an `AllBlocksCleared` empties the block records, not
/// them, so the stores that share a batch with the clear (an engine that
/// resets its prefix cache and stores the next prompt in one step) are
/// judged like every other step's.
#[derive(Default)]
pub struct Normalizer {
    ranks: HashMap<i32, RankState>,
    counts: Counts,
    /// The engine hash to recompute per store, when verification is on.
    hash_check: Option<EngineHash>,
}

/// The environment variable that turns the engine-hash check on for every
/// relay in the process: `sglang` or `vllm-sha256-cbor`.
pub const HASH_CHECK_ENV: &str = "SMG_KV_EVENT_HASH_CHECK";

/// Cache-group kinds whose blocks hold the main attention KV: the lane the
/// prefix index is built from. The gateway keeps the same list.
pub(crate) const MAIN_ATTENTION_KINDS: [&str; 3] =
    ["full_attention", "mla_attention", "sink_full_attention"];

/// The tier an engine medium names: the device when the medium is absent,
/// `None` for a medium this relay does not know.
pub fn tier_of(medium: Option<&str>) -> Option<KvCacheTier> {
    let Some(medium) = medium else {
        return Some(KvCacheTier::Device);
    };
    let upper = medium.to_ascii_uppercase();
    match upper.as_str() {
        "GPU" | "DEVICE" => Some(KvCacheTier::Device),
        "CPU" | "CPU_PINNED" | "CPU_TIER1" => Some(KvCacheTier::Host),
        "CPU_TIER2" | "DISK" | "NVME" | "STORAGE" => Some(KvCacheTier::Disk),
        "EXTERNAL" | "NETWORK" | "REMOTE" | "SHARED" => Some(KvCacheTier::External),
        _ => None,
    }
}

/// `KvBlock.cache_level` for a tier: `None` on the device (older consumers
/// read an absent level as the device), the tier's rank otherwise.
pub(crate) fn cache_level_of(tier: KvCacheTier) -> Option<i32> {
    match tier {
        KvCacheTier::Unspecified | KvCacheTier::Device => None,
        KvCacheTier::Host => Some(1),
        KvCacheTier::Disk => Some(2),
        KvCacheTier::External => Some(3),
    }
}

fn locality_of(locality: Option<&str>) -> Result<KvCacheLocality, ()> {
    match locality.map(str::to_ascii_uppercase).as_deref() {
        None | Some("LOCAL") => Ok(KvCacheLocality::Local),
        Some("REMOTE") => Err(()),
        Some(_) => Err(()),
    }
}

fn is_residency_agent(ownership: Option<&str>) -> bool {
    ownership.is_some_and(|owner| owner.eq_ignore_ascii_case("kvcr"))
}

fn extra_key_proto(key: ExtraKey) -> Option<common::KvBlockExtraKey> {
    let key = match key {
        ExtraKey::Text(text) => kv_block_extra_key::Key::Text(text),
        ExtraKey::Number(number) => kv_block_extra_key::Key::Number(number),
        ExtraKey::Blob(blob) => kv_block_extra_key::Key::Blob(blob),
        ExtraKey::Multimodal { identifier, offset } => {
            kv_block_extra_key::Key::Multimodal(common::KvMultimodalKey { identifier, offset })
        }
        ExtraKey::Opaque => return None,
    };
    Some(common::KvBlockExtraKey { key: Some(key) })
}

/// vLLM's cache salt rides inside `extra_keys`: the first text item of the
/// first block's keys that is not the LoRA name.
fn salt_from_extra_keys(
    extra_keys: Option<&[Option<Vec<ExtraKey>>]>,
    lora_name: Option<&str>,
) -> Option<String> {
    let first = extra_keys?.first()?.as_ref()?;
    first.iter().find_map(|key| match key {
        ExtraKey::Text(text) if Some(text.as_str()) != lora_name && !text.is_empty() => {
            Some(text.clone())
        }
        _ => None,
    })
}

impl Normalizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// A normalizer that also rehashes every verifiable store with `check`
    /// and counts mismatches; nothing is dropped for it. See
    /// [`crate::engine_hash`].
    pub fn with_hash_check(check: EngineHash) -> Self {
        Self {
            hash_check: Some(check),
            ..Self::default()
        }
    }

    /// [`Self::new`], or [`Self::with_hash_check`] when [`HASH_CHECK_ENV`]
    /// names an algorithm; an unknown value is logged and the check stays off.
    pub fn from_env() -> Self {
        match std::env::var(HASH_CHECK_ENV) {
            Ok(value) if !value.trim().is_empty() => match EngineHash::parse(&value) {
                Some(check) => Self::with_hash_check(check),
                None => {
                    warn!(%value, "{HASH_CHECK_ENV} names no known engine hash; check off");
                    Self::new()
                }
            },
            _ => Self::new(),
        }
    }

    pub fn counts(&self) -> &Counts {
        &self.counts
    }

    pub fn hash_check(&self) -> Option<EngineHash> {
        self.hash_check
    }

    fn drop(&mut self, reason: DropReason, event_id: u64) -> DropReason {
        let count = self.counts.dropped.entry(reason).or_insert(0);
        *count += 1;
        if *count <= 3 {
            debug!(event_id, reason = reason.as_str(), "KV event not forwarded");
        }
        reason
    }

    /// A whole publisher batch as its proto, `event_id` advancing once per
    /// event whether or not it is forwarded, so ids stay monotonic.
    pub fn normalize_batch(
        &mut self,
        batch: WireBatch,
        sequence_number: u64,
        event_id: &mut u64,
    ) -> common::KvEventBatch {
        // Learn the batch's groups before normalizing any of its events: the
        // engine lists a sliding-window group's store before the
        // full-attention group's in the same step, and what to do with the
        // window group depends on whether the rank has a main group at all.
        let rank = batch.dp_rank.unwrap_or(-1);
        for event in &batch.events {
            if let WireEvent::BlockStored(stored) = event {
                if let (Some(group), Some(kind)) = (
                    stored.tail.group_idx,
                    stored.tail.kv_cache_spec_kind.as_deref(),
                ) {
                    self.ranks
                        .entry(rank)
                        .or_default()
                        .groups
                        .insert(group, MAIN_ATTENTION_KINDS.contains(&kind));
                }
            }
        }
        let mut events = Vec::with_capacity(batch.events.len());
        for event in batch.events {
            *event_id += 1;
            if let Ok(converted) = self.normalize(event, batch.dp_rank, *event_id) {
                events.push(converted);
            }
        }
        common::KvEventBatch {
            sequence_number,
            timestamp: batch.ts,
            events,
            dp_rank: batch.dp_rank,
            snapshot: None,
            load: None,
        }
    }

    /// One event as the proto the gateway should index, or why not.
    pub fn normalize(
        &mut self,
        event: WireEvent,
        dp_rank: Option<i32>,
        event_id: u64,
    ) -> Result<common::KvCacheEvent, DropReason> {
        let rank = dp_rank.unwrap_or(-1);
        let data = match event {
            WireEvent::Unknown => return Err(self.drop(DropReason::UnknownType, event_id)),
            WireEvent::Malformed(field) => {
                debug!(event_id, field, "KV event field unreadable");
                return Err(self.drop(DropReason::Malformed, event_id));
            }
            WireEvent::AllBlocksCleared { ownership } => {
                if is_residency_agent(ownership.as_deref()) {
                    return Err(self.drop(DropReason::UnsupportedOwnership, event_id));
                }
                // The cache is empty: the block records go, and the hashes
                // forwarded while the rank was window-only with them. The
                // groups stay: they describe the model, not the cache, and
                // the stores that share this batch with the clear (an engine
                // resets its prefix cache and stores the next prompt in one
                // step) are judged under them, so the window groups' copies
                // of the prompt are dropped as on every other step instead
                // of forwarded as window-only ahead of the main group's.
                if let Some(state) = self.ranks.get_mut(&rank) {
                    state.tiers.clear();
                    state.window_forwarded.clear();
                }
                self.counts.forwarded_cleared += 1;
                kv_cache_event::Data::Cleared(common::KvCacheCleared { ownership })
            }
            WireEvent::BlockStored(stored) => self.normalize_stored(stored, rank, event_id)?,
            WireEvent::BlockRemoved(removed) => self.normalize_removed(removed, rank, event_id)?,
        };
        Ok(common::KvCacheEvent {
            event_id,
            data: Some(data),
        })
    }

    /// The shared gates: ownership, locality, medium, and the lane the
    /// event's cache group puts it on.
    fn admit(
        &mut self,
        tail: &EventTail,
        rank: i32,
        event_id: u64,
        learn_group: bool,
    ) -> Result<Admitted, DropReason> {
        if is_residency_agent(tail.ownership.as_deref()) {
            return Err(self.drop(DropReason::UnsupportedOwnership, event_id));
        }
        let locality = locality_of(tail.locality.as_deref())
            .map_err(|()| self.drop(DropReason::NonLocalLocality, event_id))?;
        let tier = tier_of(tail.medium.as_deref())
            .ok_or_else(|| self.drop(DropReason::UnknownMedium, event_id))?;
        let mut lane = Lane::Main;
        if let Some(group) = tail.group_idx {
            let state = self.ranks.entry(rank).or_default();
            let main = match tail.kv_cache_spec_kind.as_deref() {
                Some(kind) => {
                    let main = MAIN_ATTENTION_KINDS.contains(&kind);
                    if learn_group {
                        state.groups.insert(group, main);
                    }
                    main
                }
                // A kind-less event on a group we learned follows the group;
                // an unknown group is treated as main, as legacy publishers
                // with a single group are.
                None => state.groups.get(&group).copied().unwrap_or(true),
            };
            if !main {
                lane = if state.main_seen() {
                    Lane::Group
                } else {
                    Lane::WindowOnly
                };
            }
        }
        Ok(Admitted {
            tier,
            locality,
            lane,
        })
    }

    /// The forwarded store once a lane has shaped its blocks: the event's
    /// tail and the namespace the lane settled on go through as they are,
    /// and the store counts as forwarded.
    fn forward_stored(
        &mut self,
        blocks: Vec<common::KvBlock>,
        parent: Option<BlockHash>,
        tier: KvCacheTier,
        locality: KvCacheLocality,
        tail: EventTail,
        namespace: Namespace,
    ) -> kv_cache_event::Data {
        self.counts.forwarded_stored += 1;
        let EventTail {
            medium,
            group_idx,
            kv_cache_spec_kind,
            kv_cache_spec_sliding_window,
            ownership,
            session_id,
            group_mask,
            ..
        } = tail;
        kv_cache_event::Data::Stored(common::KvBlocksStored {
            blocks,
            parent_block_hash: parent.map(|hash| hash.0),
            tier: Some(tier as i32),
            medium,
            group_idx,
            kv_cache_spec_kind,
            kv_cache_spec_sliding_window,
            locality: Some(locality as i32),
            ownership,
            session_id,
            lora_name: namespace.lora_name,
            cache_salt: namespace.cache_salt,
            group_mask,
        })
    }

    /// A non-main group's store on a rank with a main group: its hashes,
    /// parent and group fields go through, its tokens do not. The engine
    /// writes such a store with the step's whole token span and only the
    /// hashes the group retains, so the tokens are not a per-block list, and
    /// the consumer keys the group's blocks by hash. The relay's own record
    /// of the main lane (namespaces, copies, the hash check) is not touched.
    fn forward_group_stored(
        &mut self,
        stored: WireStored,
        tier: KvCacheTier,
        locality: KvCacheLocality,
        event_id: u64,
    ) -> Result<kv_cache_event::Data, DropReason> {
        if stored.block_hashes.is_empty() {
            return Err(self.drop(DropReason::Placeholder, event_id));
        }
        {
            let mut seen = HashSet::with_capacity(stored.block_hashes.len() + 1);
            if let Some(parent) = stored.parent_block_hash {
                seen.insert(parent.0);
            }
            if stored.block_hashes.iter().any(|hash| !seen.insert(hash.0)) {
                return Err(self.drop(DropReason::SelfReferencingHashes, event_id));
            }
        }
        let cache_level = cache_level_of(tier);
        let block_size = i32::try_from(stored.block_size).unwrap_or(i32::MAX).max(0);
        let blocks = stored
            .block_hashes
            .iter()
            .map(|hash| common::KvBlock {
                block_hash: hash.0,
                token_ids: Vec::new(),
                block_size,
                lora_id: stored.lora_id,
                cache_level,
                extra_keys: Vec::new(),
            })
            .collect();
        self.counts.group_stores += 1;
        let namespace = Namespace {
            lora_name: stored.lora_name.filter(|name| !name.is_empty()),
            cache_salt: stored.cache_salt.filter(|salt| !salt.is_empty()),
        };
        Ok(self.forward_stored(
            blocks,
            stored.parent_block_hash,
            tier,
            locality,
            stored.tail,
            namespace,
        ))
    }

    fn normalize_stored(
        &mut self,
        stored: WireStored,
        rank: i32,
        event_id: u64,
    ) -> Result<kv_cache_event::Data, DropReason> {
        let Admitted {
            tier,
            locality,
            lane,
        } = self.admit(&stored.tail, rank, event_id, true)?;
        if lane == Lane::Group {
            return self.forward_group_stored(stored, tier, locality, event_id);
        }
        let window_only = lane == Lane::WindowOnly;
        // A bigram page lists (token, next token) per position; its tokens
        // are the first elements and the page grid is unchanged. The pairs
        // stay around for the engine-hash check, which hashes both words.
        let (mut token_ids, mut bigram_words): (Vec<u32>, Option<Vec<u32>>) = match stored.token_ids
        {
            WireTokens::Ids(ids) => (ids, None),
            WireTokens::Bigrams(pairs) => (
                pairs.iter().map(|&(token, _)| token).collect(),
                Some(
                    pairs
                        .iter()
                        .flat_map(|&(token, next)| [token, next])
                        .collect(),
                ),
            ),
        };
        let bigrams = bigram_words.is_some();
        if stored.block_hashes.is_empty() || token_ids.is_empty() {
            return Err(self.drop(DropReason::Placeholder, event_id));
        }
        // The hashes name block_size tokens each, normally the whole span. A
        // sliding-window group's store spans the whole range the step
        // computed while naming only the window's blocks, the newest ones,
        // so fewer hashes than blocks take the tail of the tokens, never the
        // head; a span that is not whole blocks is unreadable.
        let width = usize::try_from(stored.block_size)
            .ok()
            .filter(|&width| width > 0 && i32::try_from(width).is_ok());
        let Some(width) = width else {
            return Err(self.drop(DropReason::UnalignedBlocks, event_id));
        };
        let span = token_ids.len();
        let tail_aligned = match stored.block_hashes.len().checked_mul(width) {
            Some(need) if need == span => false,
            Some(need) if need < span && span.is_multiple_of(width) => {
                let start = span - need;
                token_ids = token_ids.split_off(start);
                bigram_words = bigram_words.map(|mut words| words.split_off(start * 2));
                true
            }
            _ => return Err(self.drop(DropReason::UnalignedBlocks, event_id)),
        };
        {
            let mut seen = HashSet::with_capacity(stored.block_hashes.len() + 1);
            if let Some(parent) = stored.parent_block_hash {
                seen.insert(parent.0);
            }
            if stored.block_hashes.iter().any(|hash| !seen.insert(hash.0)) {
                return Err(self.drop(DropReason::SelfReferencingHashes, event_id));
            }
        }

        // Namespace: what the event says, else what the parent was stored under.
        let lora_name = stored.lora_name.filter(|name| !name.is_empty());
        let cache_salt = stored
            .cache_salt
            .filter(|salt| !salt.is_empty())
            .or_else(|| salt_from_extra_keys(stored.extra_keys.as_deref(), lora_name.as_deref()));
        let mut namespace = Namespace {
            lora_name,
            cache_salt,
        };
        let blocks_state = self
            .ranks
            .entry(rank)
            .or_default()
            .tiers
            .entry(tier as i32)
            .or_default();
        // vLLM names the salt on block 0 only and SGLang repeats it; a chain
        // hashed under a namespace stays in it, so a child fills what it
        // omits from its parent.
        if namespace.lora_name.is_none() || namespace.cache_salt.is_none() {
            if let Some(parent) = stored
                .parent_block_hash
                .and_then(|parent| blocks_state.get(&parent.0))
                .and_then(|record| record.namespace.clone())
            {
                if namespace.lora_name.is_none() {
                    namespace.lora_name = parent.lora_name;
                }
                if namespace.cache_salt.is_none() {
                    namespace.cache_salt = parent.cache_salt;
                }
            }
        }
        let stored_namespace = (!namespace.is_empty()).then(|| namespace.clone());
        let mut all_seen = true;
        for hash in &stored.block_hashes {
            match blocks_state.get_mut(&hash.0) {
                // A second physical copy: the record stays, its digest with
                // it, with one more copy counted.
                Some(record) => {
                    record.copies = record.copies.saturating_add(1).min(COPIES_CAP);
                    record.namespace.clone_from(&stored_namespace);
                }
                None => {
                    all_seen = false;
                    blocks_state.insert(
                        hash.0,
                        BlockRecord {
                            namespace: stored_namespace.clone(),
                            copies: 1,
                            digest: None,
                        },
                    );
                }
            }
        }
        if all_seen {
            self.counts.duplicate_stores += 1;
        }
        if bigrams {
            self.counts.bigram_stores += 1;
        }
        if tail_aligned {
            self.counts.tail_aligned_stores += 1;
        }
        if let Some(check) = self.hash_check {
            verify_hashes(
                check,
                blocks_state,
                &mut self.counts,
                HashInput {
                    hashes: &stored.block_hashes,
                    parent: stored.parent_block_hash,
                    tokens: &token_ids,
                    width,
                    bigram_words: bigram_words.as_deref(),
                    extra_keys: stored.extra_keys.as_deref(),
                    lora: stored.lora_id.is_some() || namespace.lora_name.is_some(),
                    cache_salt: namespace.cache_salt.as_deref(),
                },
            );
        }

        if window_only {
            self.counts.window_only_stores += 1;
            self.ranks
                .entry(rank)
                .or_default()
                .window_forwarded
                .extend(stored.block_hashes.iter().map(|hash| hash.0));
        }

        let cache_level = cache_level_of(tier);
        let mut extra_keys = stored.extra_keys.unwrap_or_default().into_iter();
        let blocks = stored
            .block_hashes
            .iter()
            .zip(token_ids.chunks_exact(width))
            .map(|(hash, tokens)| common::KvBlock {
                block_hash: hash.0,
                token_ids: tokens.to_vec(),
                block_size: i32::try_from(width).unwrap_or(i32::MAX),
                lora_id: stored.lora_id,
                cache_level,
                extra_keys: extra_keys
                    .next()
                    .flatten()
                    .map(|keys| keys.into_iter().filter_map(extra_key_proto).collect())
                    .unwrap_or_default(),
            })
            .collect();
        Ok(self.forward_stored(
            blocks,
            stored.parent_block_hash,
            tier,
            locality,
            stored.tail,
            namespace,
        ))
    }

    fn normalize_removed(
        &mut self,
        removed: WireRemoved,
        rank: i32,
        event_id: u64,
    ) -> Result<kv_cache_event::Data, DropReason> {
        let Admitted {
            tier,
            locality,
            lane,
        } = self.admit(&removed.tail, rank, event_id, false)?;
        let block_hashes: Vec<i64> = removed.block_hashes.iter().map(|hash| hash.0).collect();
        if let Some(state) = self.ranks.get_mut(&rank) {
            // The relay's record follows the main lane. A window-only rank's
            // removal names blocks forwarded as its main signal; once the rank
            // has a main group, a group's removal touches the record only for
            // the blocks forwarded while it had none.
            let recorded: Vec<i64> = match lane {
                Lane::Main => block_hashes.clone(),
                Lane::WindowOnly => {
                    for hash in &block_hashes {
                        state.window_forwarded.remove(hash);
                    }
                    block_hashes.clone()
                }
                Lane::Group => block_hashes
                    .iter()
                    .copied()
                    .filter(|hash| state.window_forwarded.remove(hash))
                    .collect(),
            };
            if let Some(blocks_state) = state.tiers.get_mut(&(tier as i32)) {
                // One physical copy goes; the record goes with the last.
                for hash in &recorded {
                    if let Some(record) = blocks_state.get_mut(hash) {
                        record.copies = record.copies.saturating_sub(1);
                        if record.copies == 0 {
                            blocks_state.remove(hash);
                        }
                    }
                }
            }
        }
        if lane == Lane::Group {
            self.counts.group_removals += 1;
        }
        self.counts.forwarded_removed += 1;
        let EventTail {
            medium,
            group_idx,
            ownership,
            group_mask,
            ..
        } = removed.tail;
        Ok(kv_cache_event::Data::Removed(common::KvBlocksRemoved {
            block_hashes,
            cache_level: cache_level_of(tier),
            tier: Some(tier as i32),
            medium,
            group_idx,
            locality: Some(locality as i32),
            ownership,
            group_mask,
        }))
    }
}

/// The engine-hash check's view of one admitted store.
struct HashInput<'a> {
    hashes: &'a [BlockHash],
    parent: Option<BlockHash>,
    tokens: &'a [u32],
    width: usize,
    /// Both words of every bigram, when the page came as bigrams.
    bigram_words: Option<&'a [u32]>,
    extra_keys: Option<&'a [Option<Vec<ExtraKey>>]>,
    /// The store belongs to a LoRA request (vLLM folds the adapter path into
    /// the hash and does not publish it).
    lora: bool,
    cache_salt: Option<&'a str>,
}

/// Rehash a store's blocks the way `check` says the worker did and count the
/// outcome; what is forwarded never changes. Digests are kept on the records
/// so children can chain on them.
fn verify_hashes(
    check: EngineHash,
    records: &mut HashMap<i64, BlockRecord>,
    counts: &mut Counts,
    input: HashInput<'_>,
) {
    let blocks = input.hashes.len() as u64;
    let mut prior: Option<Digest32> = match input.parent {
        Some(parent) => match records.get(&parent.0).and_then(|record| record.digest) {
            Some(digest) => Some(digest),
            None => {
                counts.hash_unverifiable += blocks;
                return;
            }
        },
        None => match check {
            EngineHash::Sglang => input.cache_salt.map(engine_hash::sglang_salt_seed),
            // `vllm_block` applies NONE_HASH itself.
            EngineHash::VllmSha256Cbor => None,
        },
    };
    if check == EngineHash::VllmSha256Cbor && (input.lora || input.bigram_words.is_some()) {
        counts.hash_unverifiable += blocks;
        return;
    }
    for (index, hash) in input.hashes.iter().enumerate() {
        let tokens = &input.tokens[index * input.width..(index + 1) * input.width];
        let digest = match check {
            EngineHash::Sglang => {
                let words = match input.bigram_words {
                    Some(words) => &words[index * 2 * input.width..(index + 1) * 2 * input.width],
                    None => tokens,
                };
                engine_hash::sglang_page(prior.as_ref(), words)
            }
            EngineHash::VllmSha256Cbor => {
                let keys = input
                    .extra_keys
                    .and_then(|keys| keys.get(index))
                    .and_then(Option::as_deref);
                let Ok(keys) = vllm_keys(keys, index) else {
                    counts.hash_unverifiable += blocks - index as u64;
                    return;
                };
                engine_hash::vllm_block(prior.as_ref(), tokens, keys.as_deref())
            }
        };
        let expected = match check {
            EngineHash::Sglang => engine_hash::sglang_event_int(&digest),
            EngineHash::VllmSha256Cbor => engine_hash::vllm_event_int(&digest),
        };
        counts.hash_checked += 1;
        if expected != hash.0 {
            counts.hash_mismatch += 1;
        }
        if let Some(record) = records.get_mut(&hash.0) {
            record.digest = Some(digest);
        }
        prior = Some(digest);
    }
}

/// vLLM's untagged event keys as the tagged keys inside the hash: block 0's
/// text is the cache salt (a LoRA request was excluded before), a pair is a
/// multimodal item, bytes are a prompt-embeddings digest. `Err` for a shape
/// the hash input cannot be rebuilt from.
fn vllm_keys(keys: Option<&[ExtraKey]>, index: usize) -> Result<Option<Vec<VllmExtraKey>>, ()> {
    let Some(keys) = keys.filter(|keys| !keys.is_empty()) else {
        return Ok(None);
    };
    keys.iter()
        .map(|key| match key {
            ExtraKey::Text(text) if index == 0 => Ok(VllmExtraKey::CacheSalt(text.clone())),
            ExtraKey::Multimodal { identifier, offset } => Ok(VllmExtraKey::Mm {
                identifier: identifier.clone(),
                offset: *offset,
            }),
            ExtraKey::Blob(blob) => Ok(VllmExtraKey::PromptEmbeds(blob.clone())),
            ExtraKey::Text(_) | ExtraKey::Number(_) | ExtraKey::Opaque => Err(()),
        })
        .collect::<Result<Vec<_>, ()>>()
        .map(Some)
}

#[cfg(test)]
mod tests;
