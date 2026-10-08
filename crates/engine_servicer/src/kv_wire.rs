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
//! - the main-attention KV cache group where a rank has one (hybrid models
//!   publish one group per attention kind under the same hashes, and the
//!   sliding-window and state-space groups' events, stores and removals, are
//!   dropped); a rank that publishes only sliding-window groups forwards
//!   their stores, the hashes aligned to the tail of the tokens, as its only
//!   signal (see [`Normalizer`]);
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
    tail: [Option<Loose>; 7],
}

const TAIL_KEYS: [&str; 7] = [
    "medium",
    "group_idx",
    "kv_cache_spec_kind",
    "kv_cache_spec_sliding_window",
    "locality",
    "ownership",
    "session_id",
];

fn tail_from(slots: [Option<Loose>; 7]) -> EventTail {
    let [medium, group_idx, kind, sliding, locality, ownership, session_id] = slots;
    EventTail {
        medium: medium.and_then(Loose::text),
        group_idx: group_idx.and_then(Loose::unsigned32),
        kv_cache_spec_kind: kind.and_then(Loose::text),
        kv_cache_spec_sliding_window: sliding.and_then(Loose::unsigned32),
        locality: locality.and_then(Loose::text),
        ownership: ownership.and_then(Loose::text),
        session_id: session_id.and_then(Loose::text),
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
                let [_, _, _, _, _, ownership, _] = self.tail;
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
            /// trailing defaults are omitted. SGLang's legacy arrays put
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
                        for slot in 1..=6 {
                            fields.tail[slot] = Some(next());
                        }
                    }
                    "BlockRemoved" => {
                        fields.block_hashes = Some(next());
                        for slot in [0, 1, 4, 5] {
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
    NonMainAttentionGroup,
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
            Self::NonMainAttentionGroup => "non_main_attention_group",
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
    pub dropped: HashMap<DropReason, u64>,
}

impl Counts {
    pub fn dropped(&self, reason: DropReason) -> u64 {
        self.dropped.get(&reason).copied().unwrap_or(0)
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
    /// main-attention group, so their removals still go through once it has.
    window_forwarded: HashSet<i64>,
}

impl RankState {
    fn main_seen(&self) -> bool {
        self.groups.values().any(|&main| main)
    }
}

/// What the shared gates let through, and under which group policy.
struct Admitted {
    tier: KvCacheTier,
    locality: KvCacheLocality,
    /// A sliding-window group's event on a rank without a main-attention
    /// group, forwarded as the rank's only signal.
    window_only: bool,
}

/// Per-stream normalization state (one engine endpoint, all its DP ranks).
///
/// KV-cache groups: an event names its group (`group_idx`) and, on stores,
/// the group's attention kind. A model with full attention alongside sliding
/// windows (vLLM's hybrid KV-cache manager) publishes every block twice under
/// the same hash, once per group: the full-attention group holds the whole
/// prefix, which is what a prefix hit needs, while the window group names
/// only the window's blocks against the whole computed span and removes them
/// as the window moves on. So on a rank with a main-attention group every
/// non-main event, store and removal alike, is dropped and counted
/// ([`DropReason::NonMainAttentionGroup`]); forwarding the window group's
/// removals would take the full group's copies out of the gateway's count.
/// A rank that shows only sliding-window groups (a window-only model) would
/// otherwise give the gateway nothing: its window stores are forwarded with
/// their group identity, the hashes aligned to the tail of the tokens, and
/// counted ([`Counts::window_only_stores`]).
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

const MAIN_ATTENTION_KINDS: [&str; 3] = ["full_attention", "mla_attention", "sink_full_attention"];

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
                self.ranks.remove(&rank);
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

    /// The shared gates: ownership, locality, medium, cache group. `hashes`
    /// are the event's, for the one non-main event a rank with a main group
    /// still forwards: the removal of blocks it forwarded while window-only.
    fn admit(
        &mut self,
        tail: &EventTail,
        rank: i32,
        event_id: u64,
        learn_group: bool,
        hashes: &[BlockHash],
    ) -> Result<Admitted, DropReason> {
        if is_residency_agent(tail.ownership.as_deref()) {
            return Err(self.drop(DropReason::UnsupportedOwnership, event_id));
        }
        let locality = locality_of(tail.locality.as_deref())
            .map_err(|()| self.drop(DropReason::NonLocalLocality, event_id))?;
        let tier = tier_of(tail.medium.as_deref())
            .ok_or_else(|| self.drop(DropReason::UnknownMedium, event_id))?;
        let mut window_only = false;
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
                let forwarded_before = !learn_group
                    && hashes
                        .iter()
                        .any(|hash| state.window_forwarded.contains(&hash.0));
                if state.main_seen() && !forwarded_before {
                    return Err(self.drop(DropReason::NonMainAttentionGroup, event_id));
                }
                window_only = true;
            }
        }
        Ok(Admitted {
            tier,
            locality,
            window_only,
        })
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
            window_only,
        } = self.admit(&stored.tail, rank, event_id, true, &stored.block_hashes)?;
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
        self.counts.forwarded_stored += 1;
        let EventTail {
            medium,
            group_idx,
            kv_cache_spec_kind,
            kv_cache_spec_sliding_window,
            ownership,
            session_id,
            ..
        } = stored.tail;
        Ok(kv_cache_event::Data::Stored(common::KvBlocksStored {
            blocks,
            parent_block_hash: stored.parent_block_hash.map(|hash| hash.0),
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
        }))
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
            window_only,
        } = self.admit(&removed.tail, rank, event_id, false, &removed.block_hashes)?;
        let mut block_hashes: Vec<i64> = removed.block_hashes.iter().map(|hash| hash.0).collect();
        if let Some(state) = self.ranks.get_mut(&rank) {
            if window_only {
                // Once the rank has a main group, a window group's removal
                // reaches only the blocks forwarded while it had none.
                if state.main_seen() {
                    block_hashes.retain(|hash| state.window_forwarded.contains(hash));
                }
                for hash in &block_hashes {
                    state.window_forwarded.remove(hash);
                }
            }
            if let Some(blocks_state) = state.tiers.get_mut(&(tier as i32)) {
                // One physical copy goes; the record goes with the last.
                for hash in &block_hashes {
                    if let Some(record) = blocks_state.get_mut(hash) {
                        record.copies = record.copies.saturating_sub(1);
                        if record.copies == 0 {
                            blocks_state.remove(hash);
                        }
                    }
                }
            }
        }
        self.counts.forwarded_removed += 1;
        let EventTail {
            medium,
            group_idx,
            ownership,
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
mod shapes_tests;

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;

    /// A publisher batch from JSON: objects become msgpack maps (the engines'
    /// tagged-map events), arrays become arrays (the batch envelope and the
    /// legacy event layout).
    fn batch_from(value: Value) -> WireBatch {
        let bytes = rmp_serde::to_vec_named(&value).expect("encodes");
        rmp_serde::from_slice(&bytes).expect("decodes")
    }

    fn normalize_all(batches: Vec<Value>) -> (Vec<common::KvEventBatch>, Normalizer) {
        let mut normalizer = Normalizer::new();
        let mut event_id = 0;
        let out = batches
            .into_iter()
            .enumerate()
            .map(|(seq, value)| normalize(&mut normalizer, value, seq as u64, &mut event_id))
            .collect();
        (out, normalizer)
    }

    fn normalize(
        normalizer: &mut Normalizer,
        value: Value,
        seq: u64,
        event_id: &mut u64,
    ) -> common::KvEventBatch {
        normalizer.normalize_batch(batch_from(value), seq, event_id)
    }

    fn one(events: Vec<Value>) -> Value {
        json!([1700000000.5, events, 0])
    }

    fn store(hashes: &[i64], parent: Option<i64>, tokens: &[u32]) -> Value {
        json!({
            "type": "BlockStored",
            "block_hashes": hashes,
            "parent_block_hash": parent,
            "token_ids": tokens,
            "block_size": 4,
            "lora_id": null,
            "medium": "GPU",
            "lora_name": null,
            "group_idx": 0,
            "kv_cache_spec_kind": "full_attention",
        })
    }

    fn remove(hashes: &[i64]) -> Value {
        json!({"type": "BlockRemoved", "block_hashes": hashes, "medium": "GPU", "group_idx": 0})
    }

    fn with(mut value: Value, key: &str, item: Value) -> Value {
        value[key] = item;
        value
    }

    fn stored(event: &common::KvCacheEvent) -> &common::KvBlocksStored {
        match event.data {
            Some(kv_cache_event::Data::Stored(ref stored)) => stored,
            ref other => panic!("not a store: {other:?}"),
        }
    }

    fn removed(event: &common::KvCacheEvent) -> &common::KvBlocksRemoved {
        match event.data {
            Some(kv_cache_event::Data::Removed(ref removed)) => removed,
            ref other => panic!("not a removal: {other:?}"),
        }
    }

    #[test]
    fn hash_identity_folds_like_vllm() {
        let mut digest = [0u8; 32];
        digest[23] = 0xaa;
        digest[24..].copy_from_slice(&0x8000_0000_0000_0001u64.to_be_bytes());
        assert_eq!(low64_big_endian(&digest), 0x8000_0000_0000_0001);
        assert_eq!(low64_big_endian(&[0, 0, 0, 0, 0, 0, 0, 7]), 7);

        let (batches, _) = normalize_all(vec![one(vec![store(
            &[i64::MIN + 1, -3],
            None,
            &[1, 2, 3, 4, 5, 6, 7, 8],
        )])]);
        let blocks = &stored(&batches[0].events[0]).blocks;
        assert_eq!(
            blocks[0].block_hash,
            i64::MIN + 1,
            "a u64 above i64::MAX keeps its bits"
        );
        assert_eq!(
            blocks[1].block_hash, -3,
            "SGLang's signed form passes through"
        );
    }

    #[test]
    fn unknown_and_malformed_events_cost_themselves_not_the_batch() {
        let (batches, normalizer) = normalize_all(vec![one(vec![
            json!({"type": "BlockMigrated", "block_hashes": [1], "destination": "peer"}),
            json!({"type": "BlockStored", "block_hashes": "nope", "token_ids": [1], "block_size": 1}),
            json!({"block_hashes": [1]}),
            json!(["BlockStored", "nope"]),
            remove(&[9]),
        ])]);
        assert_eq!(batches[0].events.len(), 1);
        assert_eq!(removed(&batches[0].events[0]).block_hashes, vec![9]);
        assert_eq!(
            batches[0].events[0].event_id, 5,
            "ids advance for dropped events too"
        );
        let counts = normalizer.counts();
        assert_eq!(counts.dropped(DropReason::UnknownType), 1);
        assert_eq!(counts.dropped(DropReason::Malformed), 3);
        assert_eq!(counts.forwarded_removed, 1);
    }

    #[test]
    fn residency_agent_events_are_dropped() {
        let (batches, normalizer) = normalize_all(vec![one(vec![
            with(store(&[1], None, &[1, 2, 3, 4]), "ownership", json!("kvcr")),
            with(remove(&[1]), "ownership", json!("KVCR")),
            json!({"type": "AllBlocksCleared", "ownership": "kvcr"}),
            json!({"type": "AllBlocksCleared"}),
        ])]);
        assert_eq!(batches[0].events.len(), 1);
        assert!(matches!(
            batches[0].events[0].data,
            Some(kv_cache_event::Data::Cleared(_))
        ));
        assert_eq!(
            normalizer
                .counts()
                .dropped(DropReason::UnsupportedOwnership),
            3
        );
        assert_eq!(normalizer.counts().forwarded_cleared, 1);
    }

    #[test]
    fn remote_and_unknown_localities_are_dropped() {
        let (batches, normalizer) = normalize_all(vec![one(vec![
            with(
                store(&[1], None, &[1, 2, 3, 4]),
                "locality",
                json!("REMOTE"),
            ),
            with(remove(&[1]), "locality", json!("elsewhere")),
            with(store(&[2], None, &[1, 2, 3, 4]), "locality", json!("local")),
        ])]);
        assert_eq!(batches[0].events.len(), 1);
        assert_eq!(
            stored(&batches[0].events[0]).locality,
            Some(KvCacheLocality::Local as i32)
        );
        assert_eq!(normalizer.counts().dropped(DropReason::NonLocalLocality), 2);
    }

    #[test]
    fn media_map_to_tiers_and_cache_levels() {
        let table = [
            (None, KvCacheTier::Device, None),
            (Some("GPU"), KvCacheTier::Device, None),
            (Some("device"), KvCacheTier::Device, None),
            (Some("CPU"), KvCacheTier::Host, Some(1)),
            (Some("CPU_PINNED"), KvCacheTier::Host, Some(1)),
            (Some("CPU_TIER1"), KvCacheTier::Host, Some(1)),
            (Some("CPU_TIER2"), KvCacheTier::Disk, Some(2)),
            (Some("DISK"), KvCacheTier::Disk, Some(2)),
            (Some("NVME"), KvCacheTier::Disk, Some(2)),
            (Some("STORAGE"), KvCacheTier::Disk, Some(2)),
            (Some("EXTERNAL"), KvCacheTier::External, Some(3)),
            (Some("NETWORK"), KvCacheTier::External, Some(3)),
            (Some("REMOTE"), KvCacheTier::External, Some(3)),
            (Some("SHARED"), KvCacheTier::External, Some(3)),
        ];
        for (medium, tier, level) in table {
            assert_eq!(tier_of(medium), Some(tier), "{medium:?}");
            assert_eq!(cache_level_of(tier), level, "{medium:?}");
        }
        assert_eq!(tier_of(Some("MARS")), None);

        let (batches, normalizer) = normalize_all(vec![one(vec![
            with(
                store(&[1], None, &[1, 2, 3, 4]),
                "medium",
                json!("CPU_PINNED"),
            ),
            with(remove(&[1]), "medium", json!("STORAGE")),
            with(store(&[2], None, &[1, 2, 3, 4]), "medium", json!("MARS")),
            with(store(&[3], None, &[1, 2, 3, 4]), "medium", Value::Null),
        ])]);
        let events = &batches[0].events;
        assert_eq!(events.len(), 3);
        let host = stored(&events[0]);
        assert_eq!(host.tier, Some(KvCacheTier::Host as i32));
        assert_eq!(host.medium.as_deref(), Some("CPU_PINNED"));
        assert_eq!(host.blocks[0].cache_level, Some(1));
        let disk = removed(&events[1]);
        assert_eq!(disk.tier, Some(KvCacheTier::Disk as i32));
        assert_eq!(disk.cache_level, Some(2));
        let device = stored(&events[2]);
        assert_eq!(device.tier, Some(KvCacheTier::Device as i32));
        assert_eq!(device.medium, None);
        assert_eq!(device.blocks[0].cache_level, None);
        assert_eq!(normalizer.counts().dropped(DropReason::UnknownMedium), 1);
    }

    #[test]
    fn non_main_attention_groups_are_dropped_and_remembered() {
        let sliding = |hash: i64| {
            let event = with(store(&[hash], None, &[1, 2, 3, 4]), "group_idx", json!(1));
            let event = with(event, "kv_cache_spec_kind", json!("sliding_window"));
            with(event, "kv_cache_spec_sliding_window", json!(128))
        };
        let (batches, normalizer) = normalize_all(vec![
            one(vec![
                sliding(1),
                store(&[2], None, &[1, 2, 3, 4]),
                with(
                    with(
                        store(&[3], None, &[1, 2, 3, 4]),
                        "kv_cache_spec_kind",
                        json!("mla_attention"),
                    ),
                    "group_idx",
                    json!(2),
                ),
                with(
                    with(
                        store(&[4], None, &[1, 2, 3, 4]),
                        "kv_cache_spec_kind",
                        json!("sink_full_attention"),
                    ),
                    "group_idx",
                    json!(3),
                ),
                with(
                    with(
                        store(&[5], None, &[1, 2, 3, 4]),
                        "kv_cache_spec_kind",
                        json!("mamba"),
                    ),
                    "group_idx",
                    json!(4),
                ),
            ]),
            one(vec![
                // Removals carry no kind: the learned groups decide.
                with(remove(&[1]), "group_idx", json!(1)),
                with(remove(&[2]), "group_idx", json!(0)),
                // An unlearned group without a kind counts as main.
                with(remove(&[7]), "group_idx", json!(9)),
                // A kind-less store on a learned non-main group is dropped too.
                with(
                    with(store(&[8], None, &[1, 2, 3, 4]), "group_idx", json!(4)),
                    "kv_cache_spec_kind",
                    Value::Null,
                ),
            ]),
        ]);
        assert_eq!(batches[0].events.len(), 3);
        assert_eq!(batches[1].events.len(), 2);
        assert_eq!(removed(&batches[1].events[0]).block_hashes, vec![2]);
        assert_eq!(removed(&batches[1].events[1]).block_hashes, vec![7]);
        assert_eq!(
            normalizer
                .counts()
                .dropped(DropReason::NonMainAttentionGroup),
            4
        );
        assert_eq!(
            stored(&batches[0].events[1]).kv_cache_spec_kind.as_deref(),
            Some("mla_attention")
        );
    }

    #[test]
    fn placeholders_unaligned_and_self_referencing_stores_are_dropped() {
        let (batches, normalizer) = normalize_all(vec![one(vec![
            // vLLM's CPU offload placeholder: a chunk key, no tokens, block_size 0.
            with(
                with(store(&[1], None, &[]), "block_size", json!(0)),
                "medium",
                json!("CPU"),
            ),
            store(&[], None, &[1, 2, 3, 4]),
            store(&[2], None, &[1, 2, 3, 4, 5, 6]),
            with(store(&[3], None, &[1, 2, 3, 4]), "block_size", json!(0)),
            store(&[4], Some(4), &[1, 2, 3, 4]),
            store(&[5, 5], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
            store(&[6], None, &[1, 2, 3, 4]),
        ])]);
        assert_eq!(batches[0].events.len(), 1);
        assert_eq!(stored(&batches[0].events[0]).blocks[0].block_hash, 6);
        let counts = normalizer.counts();
        assert_eq!(counts.dropped(DropReason::Placeholder), 2);
        assert_eq!(counts.dropped(DropReason::UnalignedBlocks), 2);
        assert_eq!(counts.dropped(DropReason::SelfReferencingHashes), 2);
    }

    #[test]
    fn bigram_pages_fold_to_their_tokens() {
        let (batches, normalizer) = normalize_all(vec![one(vec![with(
            store(&[1], None, &[]),
            "token_ids",
            json!([[1, 2], [2, 3], [3, 4], [4, 5]]),
        )])]);
        let block = &stored(&batches[0].events[0]).blocks[0];
        assert_eq!(block.token_ids, vec![1, 2, 3, 4]);
        assert_eq!(block.block_size, 4);
        assert_eq!(normalizer.counts().bigram_stores, 1);
        assert_eq!(normalizer.counts().forwarded_stored, 1);
    }

    #[test]
    fn stores_and_removals_are_forwarded_one_for_one() {
        let (batches, normalizer) = normalize_all(vec![
            one(vec![
                store(&[1, 2], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
                store(&[1, 2], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
                store(&[2, 3], Some(1), &[5, 6, 7, 8, 9, 10, 11, 12]),
            ]),
            one(vec![
                remove(&[1]),
                remove(&[1]),
                remove(&[1, 2]),
                store(&[1], None, &[1, 2, 3, 4]),
            ]),
        ]);
        assert_eq!(batches[0].events.len(), 3);
        assert_eq!(batches[1].events.len(), 4);
        for event in &batches[1].events[..2] {
            assert_eq!(removed(event).block_hashes, vec![1]);
        }
        assert_eq!(removed(&batches[1].events[2]).block_hashes, vec![1, 2]);
        let counts = normalizer.counts();
        assert_eq!(counts.forwarded_stored, 4);
        assert_eq!(counts.forwarded_removed, 3);
        assert_eq!(
            counts.duplicate_stores, 1,
            "only the exact resend; a re-store after removal is new"
        );
        assert!(counts.dropped.is_empty());
    }

    #[test]
    fn namespaces_come_from_the_event_its_extra_keys_or_its_parent() {
        // (A prompt-embeddings digest is msgpack bin, which JSON cannot
        // express; the generated fixtures cover it.)
        let lora = |value: Value| with(value, "lora_name", json!("adapter"));
        let (batches, _) = normalize_all(vec![
            one(vec![
                // vLLM: the salt rides in block 0's extra keys, after the LoRA
                // name and the multimodal (identifier, offset) pairs.
                with(
                    lora(store(&[1], None, &[1, 2, 3, 4])),
                    "extra_keys",
                    json!([["adapter", ["mm-abc", 0], "salt-1"]]),
                ),
                with(
                    lora(store(&[2], Some(1), &[5, 6, 7, 8])),
                    "extra_keys",
                    json!([["adapter"]]),
                ),
                store(&[3], Some(2), &[9, 10, 11, 12]),
                // SGLang: the salt is a field; empty strings count as absent.
                with(
                    store(&[4], None, &[1, 2, 3, 4]),
                    "cache_salt",
                    json!("tenant-a"),
                ),
                with(
                    with(store(&[5], None, &[1, 2, 3, 4]), "cache_salt", json!("")),
                    "lora_name",
                    json!(""),
                ),
            ]),
            // Another rank does not inherit from this one.
            json!([1700000001.0, [store(&[6], Some(3), &[13, 14, 15, 16])], 1]),
        ]);
        let events = &batches[0].events;
        let first = stored(&events[0]);
        assert_eq!(first.lora_name.as_deref(), Some("adapter"));
        assert_eq!(first.cache_salt.as_deref(), Some("salt-1"));
        let keys: Vec<_> = first.blocks[0]
            .extra_keys
            .iter()
            .map(|key| key.key.clone().expect("a key"))
            .collect();
        assert_eq!(
            keys,
            vec![
                kv_block_extra_key::Key::Text("adapter".into()),
                kv_block_extra_key::Key::Multimodal(common::KvMultimodalKey {
                    identifier: "mm-abc".into(),
                    offset: 0,
                }),
                kv_block_extra_key::Key::Text("salt-1".into()),
            ]
        );
        let child = stored(&events[1]);
        assert_eq!(child.lora_name.as_deref(), Some("adapter"));
        assert_eq!(
            child.cache_salt.as_deref(),
            Some("salt-1"),
            "inherited from block 0"
        );
        let grandchild = stored(&events[2]);
        assert_eq!(grandchild.lora_name.as_deref(), Some("adapter"));
        assert_eq!(grandchild.cache_salt.as_deref(), Some("salt-1"));
        assert_eq!(stored(&events[3]).cache_salt.as_deref(), Some("tenant-a"));
        assert_eq!(stored(&events[3]).lora_name, None);
        assert_eq!(stored(&events[4]).cache_salt, None);
        assert_eq!(stored(&events[4]).lora_name, None);
        let other_rank = stored(&batches[1].events[0]);
        assert_eq!(other_rank.lora_name, None);
        assert_eq!(other_rank.cache_salt, None);
    }

    #[test]
    fn a_clear_resets_its_rank_only() {
        let (batches, normalizer) = normalize_all(vec![
            one(vec![store(&[1], None, &[1, 2, 3, 4])]),
            json!([1700000001.0, [store(&[1], None, &[1, 2, 3, 4])], 1]),
            one(vec![
                json!({"type": "AllBlocksCleared"}),
                store(&[1], None, &[1, 2, 3, 4]),
            ]),
            json!([1700000003.0, [store(&[1], None, &[1, 2, 3, 4])], 1]),
        ]);
        assert!(matches!(
            batches[2].events[0].data,
            Some(kv_cache_event::Data::Cleared(_))
        ));
        assert_eq!(batches[2].events.len(), 2);
        assert_eq!(normalizer.counts().forwarded_cleared, 1);
        assert_eq!(
            normalizer.counts().duplicate_stores,
            1,
            "rank 1 kept its seen set"
        );
        assert_eq!(batches[3].dp_rank, Some(1));
    }

    #[test]
    fn array_and_map_layouts_decode_alike() {
        let map = one(vec![
            with(
                with(
                    store(&[1, 2], Some(7), &[1, 2, 3, 4, 5, 6, 7, 8]),
                    "session_id",
                    json!("req-1"),
                ),
                "lora_name",
                json!("adapter"),
            ),
            with(remove(&[1]), "locality", json!("LOCAL")),
            json!({"type": "AllBlocksCleared"}),
        ]);
        let array = json!([
            1700000000.5,
            [
                [
                    "BlockStored",
                    [1, 2],
                    7,
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    4,
                    null,
                    "GPU",
                    "adapter",
                    null,
                    0,
                    "full_attention",
                    null,
                    null,
                    null,
                    "req-1"
                ],
                ["BlockRemoved", [1], "GPU", 0, "LOCAL"],
                ["AllBlocksCleared"],
            ],
            0
        ]);
        let (from_map, _) = normalize_all(vec![map]);
        let (from_array, _) = normalize_all(vec![array]);
        assert_eq!(from_map, from_array);
        let first = stored(&from_map[0].events[0]);
        assert_eq!(first.session_id.as_deref(), Some("req-1"));
        assert_eq!(first.lora_name.as_deref(), Some("adapter"));
        assert_eq!(first.parent_block_hash, Some(7));
        assert_eq!(first.group_idx, Some(0));
        assert_eq!(
            removed(&from_map[0].events[1]).locality,
            Some(KvCacheLocality::Local as i32)
        );
    }

    #[test]
    fn legacy_arrays_keep_their_trailing_slots_in_order() {
        // ownership is the removal's sixth slot and must gate the event even
        // when locality (the fifth) is set.
        let (batches, normalizer) = normalize_all(vec![json!([
            1700000000.5,
            [
                ["BlockRemoved", [1], "STORAGE", 0, "LOCAL", "kvcr"],
                ["BlockRemoved", [2], "STORAGE", 0, "REMOTE"],
                ["BlockRemoved", [3], "GPU"],
            ],
            0
        ])]);
        assert_eq!(batches[0].events.len(), 1);
        assert_eq!(removed(&batches[0].events[0]).block_hashes, vec![3]);
        assert_eq!(
            normalizer
                .counts()
                .dropped(DropReason::UnsupportedOwnership),
            1
        );
        assert_eq!(normalizer.counts().dropped(DropReason::NonLocalLocality), 1);
    }

    #[test]
    fn hash_check_verifies_sglang_chains_and_counts_mismatches() {
        use crate::engine_hash::{sglang_chain, sglang_salt_seed};

        let chain = sglang_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None);
        let (first, second) = (chain[0].1, chain[1].1);
        assert_eq!(first, -3488128144981237669);
        let mut normalizer = Normalizer::with_hash_check(EngineHash::Sglang);
        let mut event_id = 0;
        let batch = normalize(
            &mut normalizer,
            one(vec![
                store(&[first], None, &[1, 2, 3, 4]),
                store(&[second], Some(first), &[5, 6, 7, 8]),
                // Tampered: still forwarded, counted.
                store(&[second + 1], Some(first), &[5, 6, 7, 8]),
                // A parent this stream never saw: nothing to chain on.
                store(&[99], Some(12345), &[9, 10, 11, 12]),
            ]),
            1,
            &mut event_id,
        );
        assert_eq!(batch.events.len(), 4, "a mismatch never drops");
        let counts = normalizer.counts();
        assert_eq!(
            (
                counts.hash_checked,
                counts.hash_mismatch,
                counts.hash_unverifiable
            ),
            (3, 1, 1)
        );

        // A salted request seeds its chain; a coalesced two-page store
        // chains page to page.
        let seed = sglang_salt_seed("tenant-a");
        let salted = sglang_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, Some(&seed));
        let mut normalizer = Normalizer::with_hash_check(EngineHash::Sglang);
        normalize(
            &mut normalizer,
            one(vec![with(
                store(&[salted[0].1, salted[1].1], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
                "cache_salt",
                json!("tenant-a"),
            )]),
            1,
            &mut 0,
        );
        let counts = normalizer.counts();
        assert_eq!((counts.hash_checked, counts.hash_mismatch), (2, 0));

        // An Eagle bigram page hashes both words of every pair.
        let mut normalizer = Normalizer::with_hash_check(EngineHash::Sglang);
        normalize(
            &mut normalizer,
            one(vec![with(
                store(&[-638950109823820341], None, &[]),
                "token_ids",
                json!([[1, 2], [2, 3], [3, 4], [4, 5]]),
            )]),
            1,
            &mut 0,
        );
        let counts = normalizer.counts();
        assert_eq!((counts.hash_checked, counts.hash_mismatch), (1, 0));
    }

    #[test]
    fn hash_check_verifies_vllm_sha256_cbor_chains() {
        use crate::engine_hash::{vllm_block, vllm_event_int};

        // Vectors A and B from the reference run.
        let (a, b) = (-8885242862429187823i64, -3153830497298837583i64);
        let mut normalizer = Normalizer::with_hash_check(EngineHash::VllmSha256Cbor);
        let batch = normalize(
            &mut normalizer,
            one(vec![
                store(&[a, b], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
                // A LoRA block: the adapter path is not in the event.
                with(
                    store(&[7], Some(b), &[9, 10, 11, 12]),
                    "lora_name",
                    json!("adapter"),
                ),
                // Unaligned: dropped before the check runs.
                store(&[8], Some(b), &[9, 10, 11, 12, 13]),
            ]),
            1,
            &mut 0,
        );
        assert_eq!(batch.events.len(), 2);
        let counts = normalizer.counts();
        assert_eq!(
            (
                counts.hash_checked,
                counts.hash_mismatch,
                counts.hash_unverifiable
            ),
            (2, 0, 1)
        );

        // Multimodal, salt and prompt-embeddings keys rebuild the tagged hash
        // input (vectors E, F, G); the events carry the bytes JSON cannot.
        let embeds: Vec<u8> = (0..32).collect();
        let e = vllm_block(
            None,
            &[1, 2, 3, 4],
            Some(&[
                VllmExtraKey::Mm {
                    identifier: "mm-abc".into(),
                    offset: 0,
                },
                VllmExtraKey::CacheSalt("salt-1".into()),
                VllmExtraKey::PromptEmbeds(embeds.clone()),
            ]),
        );
        let f = vllm_block(
            Some(&e),
            &[5, 6, 7, 8],
            Some(&[VllmExtraKey::Mm {
                identifier: "mm-abc".into(),
                offset: -4,
            }]),
        );
        let g = vllm_block(Some(&f), &[9, 10, 11, 12], None);
        let stored = |hashes: Vec<i64>, parent: Option<i64>, tokens: Vec<u32>, keys| {
            WireEvent::BlockStored(WireStored {
                block_hashes: hashes.into_iter().map(BlockHash).collect(),
                parent_block_hash: parent.map(BlockHash),
                token_ids: WireTokens::Ids(tokens),
                block_size: 4,
                lora_id: None,
                lora_name: None,
                cache_salt: None,
                extra_keys: keys,
                tail: EventTail {
                    medium: Some("GPU".into()),
                    group_idx: Some(0),
                    kv_cache_spec_kind: Some("full_attention".into()),
                    ..EventTail::default()
                },
            })
        };
        let mut normalizer = Normalizer::with_hash_check(EngineHash::VllmSha256Cbor);
        let events = [
            stored(
                vec![vllm_event_int(&e)],
                None,
                vec![1, 2, 3, 4],
                Some(vec![Some(vec![
                    ExtraKey::Multimodal {
                        identifier: "mm-abc".into(),
                        offset: 0,
                    },
                    ExtraKey::Text("salt-1".into()),
                    ExtraKey::Blob(embeds),
                ])]),
            ),
            stored(
                vec![vllm_event_int(&f)],
                Some(vllm_event_int(&e)),
                vec![5, 6, 7, 8],
                Some(vec![Some(vec![ExtraKey::Multimodal {
                    identifier: "mm-abc".into(),
                    offset: -4,
                }])]),
            ),
            stored(
                vec![vllm_event_int(&g)],
                Some(vllm_event_int(&f)),
                vec![9, 10, 11, 12],
                None,
            ),
            // A key shape the hash input cannot be rebuilt from.
            stored(
                vec![5],
                Some(vllm_event_int(&g)),
                vec![13, 14, 15, 16],
                Some(vec![Some(vec![ExtraKey::Number(3)])]),
            ),
        ];
        for (index, event) in events.into_iter().enumerate() {
            assert!(normalizer
                .normalize(event, Some(0), index as u64 + 1)
                .is_ok());
        }
        let counts = normalizer.counts();
        assert_eq!(
            (
                counts.hash_checked,
                counts.hash_mismatch,
                counts.hash_unverifiable
            ),
            (3, 0, 1)
        );
        assert_eq!(stored_salt(&normalizer), Some("salt-1".to_string()));
    }

    fn stored_salt(normalizer: &Normalizer) -> Option<String> {
        normalizer
            .ranks
            .get(&0)
            .and_then(|rank| rank.tiers.get(&(KvCacheTier::Device as i32)))
            .and_then(|tier| tier.values().find_map(|record| record.namespace.clone()))
            .and_then(|namespace| namespace.cache_salt)
    }

    #[test]
    fn hash_check_is_off_unless_asked() {
        assert_eq!(Normalizer::new().hash_check(), None);
        assert_eq!(
            Normalizer::with_hash_check(EngineHash::Sglang).hash_check(),
            Some(EngineHash::Sglang)
        );
        // Without the check, a wrong hash is nobody's business here.
        let (_, normalizer) = normalize_all(vec![one(vec![store(&[1], None, &[1, 2, 3, 4])])]);
        assert_eq!(normalizer.counts().hash_checked, 0);
    }

    /// A sliding-window group's store: the window's blocks against the whole
    /// span the step computed.
    fn window(hashes: &[i64], parent: Option<i64>, tokens: &[u32]) -> Value {
        let event = with(store(hashes, parent, tokens), "group_idx", json!(1));
        let event = with(event, "kv_cache_spec_kind", json!("sliding_window"));
        with(event, "kv_cache_spec_sliding_window", json!(128))
    }

    fn remove_in(hashes: &[i64], group: u32) -> Value {
        with(remove(hashes), "group_idx", json!(group))
    }

    #[test]
    fn two_groups_with_identical_hashes_index_the_full_attention_group_once() {
        // vLLM's hybrid manager publishes every block in both groups under
        // the same hash, the window group's store first in the step and its
        // removals as the window moves on; only the full-attention group's
        // events are forwarded, stores and removals alike, so the gateway
        // counts one copy per physical block.
        let (batches, normalizer) = normalize_all(vec![
            one(vec![
                window(&[2], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
                store(&[1, 2], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
            ]),
            one(vec![
                remove_in(&[2], 1),
                store(&[1, 2], None, &[1, 2, 3, 4, 5, 6, 7, 8]), // the second copy
                remove_in(&[1], 0),
                remove_in(&[1], 0),
            ]),
        ]);
        assert_eq!(batches[0].events.len(), 1);
        let full = stored(&batches[0].events[0]);
        assert_eq!(
            full.blocks.iter().map(|b| b.block_hash).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(full.group_idx, Some(0));
        assert_eq!(batches[1].events.len(), 3);
        assert_eq!(removed(&batches[1].events[1]).block_hashes, vec![1]);
        assert_eq!(removed(&batches[1].events[2]).block_hashes, vec![1]);
        let counts = normalizer.counts();
        assert_eq!(counts.dropped(DropReason::NonMainAttentionGroup), 2);
        assert_eq!((counts.forwarded_stored, counts.forwarded_removed), (2, 2));
        assert_eq!(
            (
                counts.duplicate_stores,
                counts.window_only_stores,
                counts.tail_aligned_stores
            ),
            (1, 0, 0)
        );
    }

    #[test]
    fn a_window_groups_hashes_take_the_tail_of_its_tokens() {
        // A rank with only a sliding-window group: a store spans the whole
        // range the step computed and names the window's last blocks, so the
        // hashes get the tail of the tokens, and the group rides along.
        let span: Vec<u32> = (1..=12).collect();
        let (batches, normalizer) = normalize_all(vec![one(vec![
            window(&[5], None, &span),
            window(&[6, 7], Some(5), &span),
            // A span that is not whole blocks is unreadable.
            window(&[8], Some(7), &(1..=13).collect::<Vec<u32>>()),
            // The window moved on without caching anything new.
            window(&[], Some(7), &(1..=16).collect::<Vec<u32>>()),
        ])]);
        assert_eq!(batches[0].events.len(), 2);
        let first = stored(&batches[0].events[0]);
        assert_eq!(first.blocks[0].token_ids, vec![9, 10, 11, 12]);
        assert_eq!(first.group_idx, Some(1));
        assert_eq!(first.kv_cache_spec_kind.as_deref(), Some("sliding_window"));
        assert_eq!(first.kv_cache_spec_sliding_window, Some(128));
        let second = stored(&batches[0].events[1]);
        assert_eq!(
            second
                .blocks
                .iter()
                .map(|b| b.token_ids.clone())
                .collect::<Vec<_>>(),
            vec![vec![5, 6, 7, 8], vec![9, 10, 11, 12]]
        );
        assert_eq!(second.parent_block_hash, Some(5));
        let counts = normalizer.counts();
        assert_eq!(
            (counts.window_only_stores, counts.tail_aligned_stores),
            (2, 2)
        );
        assert_eq!(counts.dropped(DropReason::UnalignedBlocks), 1);
        assert_eq!(counts.dropped(DropReason::Placeholder), 1);
    }

    #[test]
    fn a_window_only_rank_is_forwarded_until_a_main_group_appears() {
        let (batches, normalizer) = normalize_all(vec![
            one(vec![
                window(&[1], None, &[1, 2, 3, 4]),
                remove_in(&[1], 1),
                window(&[2], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
            ]),
            one(vec![
                store(&[3], None, &[1, 2, 3, 4]), // a main-attention group shows up
                window(&[4], Some(3), &[1, 2, 3, 4, 5, 6, 7, 8]),
                remove_in(&[2], 1), // forwarded while window-only: still removable
                remove_in(&[9], 1), // never forwarded: dropped
                remove_in(&[2, 9], 1), // nothing left of it
            ]),
        ]);
        assert_eq!(batches[0].events.len(), 3);
        assert_eq!(batches[1].events.len(), 2);
        assert_eq!(removed(&batches[1].events[1]).block_hashes, vec![2]);
        let counts = normalizer.counts();
        assert_eq!(
            (counts.window_only_stores, counts.tail_aligned_stores),
            (2, 1)
        );
        assert_eq!(counts.dropped(DropReason::NonMainAttentionGroup), 3);
    }

    #[test]
    fn one_copy_removals_pass_through_uncollapsed() {
        // vLLM keeps up to two physical blocks per hash and publishes a
        // removal when one copy goes; the gateway counts copies, so both
        // stores and both removals must reach it as they are.
        let (batches, normalizer) = normalize_all(vec![one(vec![
            store(&[1], None, &[1, 2, 3, 4]),
            store(&[1], None, &[1, 2, 3, 4]),
            remove(&[1]),
            remove(&[1]),
        ])]);
        assert_eq!(batches[0].events.len(), 4);
        let counts = normalizer.counts();
        assert_eq!(
            (
                counts.forwarded_stored,
                counts.forwarded_removed,
                counts.duplicate_stores
            ),
            (2, 2, 1)
        );
    }

    /// The engine keeps up to two physical copies of a block and removes
    /// them one at a time; the check's parent memory counts the copies the
    /// way the gateway does, so a child stored after the first copy went
    /// still chains on its parent's digest, and only after the last copy is
    /// the parent gone and the child unverifiable. Both removals are
    /// forwarded either way.
    #[test]
    fn the_hash_checks_parent_memory_counts_physical_copies() {
        let root = engine_hash::sglang_chain(&[1, 2, 3, 4], 4, None)[0].1;
        let child_b = engine_hash::sglang_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None)[1].1;
        let child_c = engine_hash::sglang_chain(&[1, 2, 3, 4, 13, 14, 15, 16], 4, None)[1].1;
        let mut normalizer = Normalizer::with_hash_check(EngineHash::Sglang);
        let mut event_id = 0;
        normalize(
            &mut normalizer,
            one(vec![
                store(&[root], None, &[1, 2, 3, 4]),
                store(&[root], None, &[1, 2, 3, 4]), // the second physical copy
                remove(&[root]),                     // one copy goes
                store(&[child_b], Some(root), &[5, 6, 7, 8]),
            ]),
            1,
            &mut event_id,
        );
        let counts = normalizer.counts();
        assert_eq!(
            (
                counts.hash_checked,
                counts.hash_mismatch,
                counts.hash_unverifiable,
                counts.duplicate_stores
            ),
            (3, 0, 0, 1)
        );
        normalize(
            &mut normalizer,
            one(vec![
                remove(&[root]), // the last copy
                store(&[child_c], Some(root), &[13, 14, 15, 16]),
            ]),
            2,
            &mut event_id,
        );
        let counts = normalizer.counts();
        assert_eq!((counts.hash_checked, counts.hash_unverifiable), (3, 1));
        assert_eq!((counts.forwarded_stored, counts.forwarded_removed), (4, 2));
    }
}
