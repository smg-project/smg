//! The engines' KV-event wire shapes, built in code and normalized: vLLM and
//! SGLang, each in the current tagged-map layout and the legacy tag-first
//! array layout, with every field variant the relay reads (both hash forms,
//! the tiers and their cache levels, parents, LoRA names and cache salts,
//! extra keys, bigram pages, a clear, a second DP rank and every drop rule),
//! the normalizer's output asserted event by event and in its counters, and
//! the normalized streams' round trip into the gateway's reference index.
//!
//! The bytes follow msgspec's encoding of the engines' own structs (vLLM
//! `vllm/distributed/kv_events.py`, SGLang
//! `python/sglang/srt/disaggregation/kv_events.py`): a batch is the array
//! `[ts, events, dp_rank]`; an event is a map with its `type` tag in which,
//! under `omit_defaults`, a field with a default appears only when set, or
//! the legacy array with the tag first and every field in declaration order,
//! nil when unset (`omit_defaults` does not thin an array). The
//! Python servicer's `tests/test_kv_relay.py` builds the same scenarios with
//! msgspec itself and expects the same output, which keeps the two relays
//! in step.

use std::collections::BTreeMap;

use engine_zmq_client::codec::TrailingTolerant;
use kv_index::{
    compute_request_content_hashes,
    salt::{content_hash_with_seed, namespace_seed, namespaced_request_content_hashes},
    ApplyError, ContentHash, ReferenceIndexer, SequenceHash, StoredBlock,
};
use rmpv::Value;
use sha2::{Digest, Sha256};
use smg_grpc_client::common_proto::{
    kv_block_extra_key, kv_cache_event, KvBlocksStored, KvCacheEvent, KvCacheTier, KvEventBatch,
};

use super::{Counts, Normalizer, WireBatch};

const BLOCK_SIZE: u64 = 4;

// ---------------------------------------------------------------------------
// Encoding: msgspec's two layouts
// ---------------------------------------------------------------------------

/// How msgspec lays an event out: a map carrying its `type` tag, or the
/// legacy tag-first array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    Map,
    Array,
}

impl Layout {
    /// An event from its tag, its required fields and its defaulted fields in
    /// declaration order (`None` is a field left at its default). The map
    /// keeps every required field, nil included, and only the set defaulted
    /// ones; the array keeps every slot, nil when unset.
    fn event(
        self,
        tag: &str,
        required: Vec<(&str, Value)>,
        defaulted: Vec<(&str, Option<Value>)>,
    ) -> Value {
        match self {
            Layout::Map => {
                let mut entries = vec![(Value::from("type"), Value::from(tag))];
                entries.extend(
                    required
                        .into_iter()
                        .map(|(key, value)| (Value::from(key), value)),
                );
                entries.extend(
                    defaulted
                        .into_iter()
                        .filter_map(|(key, value)| Some((Value::from(key), value?))),
                );
                Value::Map(entries)
            }
            Layout::Array => {
                let mut items = vec![Value::from(tag)];
                items.extend(required.into_iter().map(|(_, value)| value));
                items.extend(
                    defaulted
                        .into_iter()
                        .map(|(_, value)| value.unwrap_or(Value::Nil)),
                );
                Value::Array(items)
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Layout::Map => "map",
            Layout::Array => "array",
        }
    }
}

/// A block hash as an engine publishes it.
#[derive(Clone, Copy, Debug)]
enum Hash {
    /// vLLM's integer form: the low 64 bits of the digest, unsigned.
    Unsigned(u64),
    /// SGLang's integer form: the high 64 bits of the digest, signed.
    Signed(i64),
    /// vLLM's raw digest (`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`).
    Digest([u8; 32]),
}

impl Hash {
    fn wire(self) -> Value {
        match self {
            Hash::Unsigned(value) => Value::from(value),
            Hash::Signed(value) => Value::from(value),
            Hash::Digest(digest) => Value::from(digest.to_vec()),
        }
    }

    /// The identity the relay forwards: the same 64 bits as an `i64`, a
    /// digest's low 64 bits read big-endian.
    fn forwarded(self) -> i64 {
        match self {
            Hash::Unsigned(value) => value as i64,
            Hash::Signed(value) => value,
            Hash::Digest(digest) => i64::from_be_bytes(low64(&digest)),
        }
    }
}

fn digest(label: &str) -> [u8; 32] {
    Sha256::digest(label.as_bytes()).into()
}

fn low64(digest: &[u8; 32]) -> [u8; 8] {
    let mut low = [0u8; 8];
    low.copy_from_slice(&digest[24..]);
    low
}

fn vllm_hash(label: &str) -> Hash {
    Hash::Unsigned(u64::from_be_bytes(low64(&digest(label))))
}

fn sglang_hash(label: &str) -> Hash {
    let mut high = [0u8; 8];
    high.copy_from_slice(&digest(label)[..8]);
    Hash::Signed(i64::from_be_bytes(high))
}

fn hashes(hashes: &[Hash]) -> Value {
    Value::Array(hashes.iter().map(|hash| hash.wire()).collect())
}

fn ints(tokens: &[u32]) -> Value {
    Value::Array(
        tokens
            .iter()
            .map(|&token| Value::from(u64::from(token)))
            .collect(),
    )
}

/// An Eagle bigram page: `[token, next token]` per position.
fn bigrams(pairs: &[(u32, u32)]) -> Value {
    Value::Array(
        pairs
            .iter()
            .map(|&(token, next)| {
                Value::Array(vec![
                    Value::from(u64::from(token)),
                    Value::from(u64::from(next)),
                ])
            })
            .collect(),
    )
}

fn opt_text(text: Option<&str>) -> Value {
    text.map_or(Value::Nil, Value::from)
}

fn opt_hash(hash: Option<Hash>) -> Value {
    hash.map_or(Value::Nil, Hash::wire)
}

/// vLLM's `extra_keys`: one list per block, nil for a block without any.
fn extra_keys(per_block: &[Option<Vec<Value>>]) -> Value {
    Value::Array(
        per_block
            .iter()
            .map(|keys| {
                keys.as_ref()
                    .map_or(Value::Nil, |items| Value::Array(items.clone()))
            })
            .collect(),
    )
}

/// A publisher batch, `[ts, events, dp_rank]`, as bytes. Both engines'
/// batches are `array_like`; vLLM's omits a rank left at its default and
/// SGLang's always writes the slot, which is the same thing once a rank is
/// set, as it is in every scenario here.
fn batch(ts: f64, rank: i64, events: Vec<Value>) -> Vec<u8> {
    let value = Value::Array(vec![
        Value::from(ts),
        Value::Array(events),
        Value::from(rank),
    ]);
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, &value).expect("msgpack encodes");
    bytes
}

/// vLLM's `BlockStored` with the struct's defaults: `None` in a defaulted
/// field is a field left at its default.
#[derive(Clone)]
struct VllmStored {
    hashes: Vec<Hash>,
    parent: Option<Hash>,
    tokens: Vec<u32>,
    block_size: u64,
    lora_id: Option<i64>,
    medium: Option<&'static str>,
    lora_name: Option<&'static str>,
    extra_keys: Option<Vec<Option<Vec<Value>>>>,
    group_idx: Option<u64>,
    spec_kind: Option<&'static str>,
    sliding_window: Option<u64>,
    locality: Option<&'static str>,
    ownership: Option<&'static str>,
    session_id: Option<&'static str>,
}

impl VllmStored {
    /// A store on the device from the main attention group.
    fn gpu(hashes: Vec<Hash>, parent: Option<Hash>, tokens: Vec<u32>) -> Self {
        Self {
            hashes,
            parent,
            tokens,
            block_size: BLOCK_SIZE,
            lora_id: None,
            medium: Some("GPU"),
            lora_name: None,
            extra_keys: None,
            group_idx: Some(0),
            spec_kind: Some("full_attention"),
            sliding_window: None,
            locality: None,
            ownership: None,
            session_id: None,
        }
    }

    fn wire(&self, layout: Layout) -> Value {
        layout.event(
            "BlockStored",
            vec![
                ("block_hashes", hashes(&self.hashes)),
                ("parent_block_hash", opt_hash(self.parent)),
                ("token_ids", ints(&self.tokens)),
                ("block_size", Value::from(self.block_size)),
                ("lora_id", self.lora_id.map_or(Value::Nil, Value::from)),
                ("medium", opt_text(self.medium)),
                ("lora_name", opt_text(self.lora_name)),
            ],
            vec![
                ("extra_keys", self.extra_keys.as_deref().map(extra_keys)),
                ("group_idx", self.group_idx.map(Value::from)),
                ("kv_cache_spec_kind", self.spec_kind.map(Value::from)),
                (
                    "kv_cache_spec_sliding_window",
                    self.sliding_window.map(Value::from),
                ),
                ("locality", self.locality.map(Value::from)),
                ("ownership", self.ownership.map(Value::from)),
                ("session_id", self.session_id.map(Value::from)),
            ],
        )
    }
}

/// vLLM's `BlockRemoved`: the medium is required, the rest defaulted.
fn vllm_removed(layout: Layout, removed: &[Hash], medium: &str, group_idx: Option<u64>) -> Value {
    layout.event(
        "BlockRemoved",
        vec![
            ("block_hashes", hashes(removed)),
            ("medium", Value::from(medium)),
        ],
        vec![
            ("group_idx", group_idx.map(Value::from)),
            ("locality", None),
            ("ownership", None),
        ],
    )
}

/// SGLang's `BlockStored`: `lora_id` is required (always nil), the medium,
/// salt and session are defaulted.
#[derive(Clone)]
struct SglangStored {
    hashes: Vec<Hash>,
    parent: Option<Hash>,
    tokens: Value,
    medium: Option<&'static str>,
    cache_salt: Option<&'static str>,
    session_id: Option<&'static str>,
}

impl SglangStored {
    fn gpu(hashes: Vec<Hash>, parent: Option<Hash>, tokens: &[u32]) -> Self {
        Self {
            hashes,
            parent,
            tokens: ints(tokens),
            medium: Some("GPU"),
            cache_salt: None,
            session_id: None,
        }
    }

    fn wire(&self, layout: Layout) -> Value {
        layout.event(
            "BlockStored",
            vec![
                ("block_hashes", hashes(&self.hashes)),
                ("parent_block_hash", opt_hash(self.parent)),
                ("token_ids", self.tokens.clone()),
                ("block_size", Value::from(BLOCK_SIZE)),
                ("lora_id", Value::Nil),
            ],
            vec![
                ("medium", self.medium.map(Value::from)),
                ("cache_salt", self.cache_salt.map(Value::from)),
                ("session_id", self.session_id.map(Value::from)),
            ],
        )
    }
}

/// SGLang's `BlockRemoved`: only the hashes are required.
fn sglang_removed(layout: Layout, removed: &[Hash], medium: &str) -> Value {
    layout.event(
        "BlockRemoved",
        vec![("block_hashes", hashes(removed))],
        vec![("medium", Some(Value::from(medium)))],
    )
}

fn cleared(layout: Layout) -> Value {
    layout.event("AllBlocksCleared", Vec::new(), Vec::new())
}

/// An event type the relay does not know (stands in for a future one).
fn migrated(layout: Layout, moved: &[Hash]) -> Value {
    layout.event(
        "BlockMigrated",
        vec![
            ("block_hashes", hashes(moved)),
            ("destination", Value::from("peer")),
        ],
        Vec::new(),
    )
}

/// A store whose hashes are not a list: no struct produces it, so it is
/// written out by hand in each layout.
fn malformed_store(layout: Layout) -> Value {
    match layout {
        Layout::Map => Value::Map(vec![
            (Value::from("type"), Value::from("BlockStored")),
            (Value::from("block_hashes"), Value::from("nope")),
            (Value::from("parent_block_hash"), Value::Nil),
            (Value::from("token_ids"), ints(&[1, 2, 3, 4])),
            (Value::from("block_size"), Value::from(BLOCK_SIZE)),
        ]),
        Layout::Array => Value::Array(vec![
            Value::from("BlockStored"),
            Value::from("nope"),
            Value::Nil,
            ints(&[1, 2, 3, 4]),
            Value::from(BLOCK_SIZE),
        ]),
    }
}

// ---------------------------------------------------------------------------
// Expectations
// ---------------------------------------------------------------------------

/// A forwarded store as the gateway reads it.
#[derive(Debug)]
struct WantStored {
    rank: i32,
    hashes: Vec<i64>,
    parent: Option<i64>,
    tier: KvCacheTier,
    cache_level: Option<i32>,
    tokens: Vec<Vec<u32>>,
    lora_name: Option<&'static str>,
    cache_salt: Option<&'static str>,
    group_idx: Option<u32>,
    session_id: Option<&'static str>,
    /// Per block, when the scenario pins them.
    extra_keys: Option<Vec<Vec<Key>>>,
}

impl WantStored {
    /// A plain device store on rank 0.
    fn device(hashes: Vec<i64>, tokens: Vec<Vec<u32>>) -> Self {
        Self {
            rank: 0,
            hashes,
            parent: None,
            tier: KvCacheTier::Device,
            cache_level: None,
            tokens,
            lora_name: None,
            cache_salt: None,
            group_idx: None,
            session_id: None,
            extra_keys: None,
        }
    }
}

/// One of a block's extra keys, in the shape the relay forwards it.
#[derive(Debug, PartialEq, Eq)]
enum Key {
    Text(&'static str),
    Multimodal(&'static str, i64),
    BlobLen(usize),
}

#[derive(Debug)]
enum Want {
    Stored(WantStored),
    Removed {
        rank: i32,
        hashes: Vec<i64>,
        tier: KvCacheTier,
        cache_level: Option<i32>,
    },
    Cleared {
        rank: i32,
    },
}

fn removed(hashes: Vec<i64>) -> Want {
    Want::Removed {
        rank: 0,
        hashes,
        tier: KvCacheTier::Device,
        cache_level: None,
    }
}

fn removed_on(hashes: Vec<i64>, tier: KvCacheTier, cache_level: i32) -> Want {
    Want::Removed {
        rank: 0,
        hashes,
        tier,
        cache_level: Some(cache_level),
    }
}

#[derive(Debug, Default)]
struct WantCounts {
    stored: u64,
    removed: u64,
    cleared: u64,
    duplicate_stores: u64,
    bigram_stores: u64,
    dropped: BTreeMap<&'static str, u64>,
}

/// One engine's stream in one layout: its batches as the publisher's bytes
/// and what the relay must make of them.
struct Scenario {
    name: String,
    engine: &'static str,
    layout: Layout,
    batches: Vec<Vec<u8>>,
    want: Vec<Want>,
    counts: WantCounts,
}

/// The scenario decoded and normalized the way the relay forwards it.
fn normalized(scenario: &Scenario) -> (Vec<KvEventBatch>, Counts) {
    let mut normalizer = Normalizer::new();
    let mut event_id = 0;
    let batches = scenario
        .batches
        .iter()
        .enumerate()
        .map(|(seq, bytes)| {
            let batch = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(bytes)
                .unwrap_or_else(|error| panic!("{} batch {seq}: {error}", scenario.name))
                .0;
            normalizer.normalize_batch(batch, seq as u64, &mut event_id)
        })
        .collect();
    (batches, normalizer.counts().clone())
}

fn key_shape(key: &kv_block_extra_key::Key) -> Key {
    match key {
        kv_block_extra_key::Key::Text(text) => Key::Text(Box::leak(text.clone().into_boxed_str())),
        kv_block_extra_key::Key::Number(number) => {
            panic!("no scenario forwards a numeric key, got {number}")
        }
        kv_block_extra_key::Key::Blob(blob) => Key::BlobLen(blob.len()),
        kv_block_extra_key::Key::Multimodal(mm) => {
            Key::Multimodal(Box::leak(mm.identifier.clone().into_boxed_str()), mm.offset)
        }
    }
}

/// Every forwarded event against its expectation, then the counters.
fn check(scenario: &Scenario, batches: &[KvEventBatch], counts: &Counts) {
    let name = &scenario.name;
    let forwarded: Vec<(Option<i32>, &KvCacheEvent)> = batches
        .iter()
        .flat_map(|batch| batch.events.iter().map(move |event| (batch.dp_rank, event)))
        .collect();
    assert_eq!(
        forwarded.len(),
        scenario.want.len(),
        "{name}: forwarded event count; got {:#?}",
        forwarded.iter().map(|(_, event)| event).collect::<Vec<_>>()
    );
    for (index, ((rank, event), want)) in forwarded.iter().zip(&scenario.want).enumerate() {
        let at = format!("{name} forwarded event {index} (id {})", event.event_id);
        match (&event.data, want) {
            (Some(kv_cache_event::Data::Stored(stored)), Want::Stored(want)) => {
                assert_eq!(*rank, Some(want.rank), "{at}: dp_rank");
                let got_hashes: Vec<i64> = stored.blocks.iter().map(|b| b.block_hash).collect();
                assert_eq!(got_hashes, want.hashes, "{at}: hashes");
                assert_eq!(stored.parent_block_hash, want.parent, "{at}: parent");
                assert_eq!(stored.tier, Some(want.tier as i32), "{at}: tier");
                let got_tokens: Vec<Vec<u32>> =
                    stored.blocks.iter().map(|b| b.token_ids.clone()).collect();
                assert_eq!(got_tokens, want.tokens, "{at}: tokens");
                for block in &stored.blocks {
                    assert_eq!(block.cache_level, want.cache_level, "{at}: cache_level");
                    assert_eq!(
                        block.block_size as usize,
                        block.token_ids.len(),
                        "{at}: block_size"
                    );
                }
                assert_eq!(
                    stored.lora_name.as_deref(),
                    want.lora_name,
                    "{at}: lora_name"
                );
                assert_eq!(
                    stored.cache_salt.as_deref(),
                    want.cache_salt,
                    "{at}: cache_salt"
                );
                assert_eq!(stored.group_idx, want.group_idx, "{at}: group_idx");
                assert_eq!(
                    stored.session_id.as_deref(),
                    want.session_id,
                    "{at}: session_id"
                );
                if let Some(expected_keys) = &want.extra_keys {
                    let got_keys: Vec<Vec<Key>> = stored
                        .blocks
                        .iter()
                        .map(|block| {
                            block
                                .extra_keys
                                .iter()
                                .map(|key| key_shape(key.key.as_ref().expect("a key")))
                                .collect()
                        })
                        .collect();
                    assert_eq!(&got_keys, expected_keys, "{at}: extra_keys");
                }
            }
            (
                Some(kv_cache_event::Data::Removed(got)),
                Want::Removed {
                    rank: want_rank,
                    hashes,
                    tier,
                    cache_level,
                },
            ) => {
                assert_eq!(*rank, Some(*want_rank), "{at}: dp_rank");
                assert_eq!(&got.block_hashes, hashes, "{at}: hashes");
                assert_eq!(got.tier, Some(*tier as i32), "{at}: tier");
                assert_eq!(got.cache_level, *cache_level, "{at}: cache_level");
            }
            (Some(kv_cache_event::Data::Cleared(_)), Want::Cleared { rank: want_rank }) => {
                assert_eq!(*rank, Some(*want_rank), "{at}: dp_rank");
            }
            (got, want) => panic!("{at}: got {got:?}, wanted {want:?}"),
        }
    }

    let want = &scenario.counts;
    assert_eq!(
        counts.forwarded_stored, want.stored,
        "{name}: forwarded_stored"
    );
    assert_eq!(
        counts.forwarded_removed, want.removed,
        "{name}: forwarded_removed"
    );
    assert_eq!(
        counts.forwarded_cleared, want.cleared,
        "{name}: forwarded_cleared"
    );
    assert_eq!(
        counts.duplicate_stores, want.duplicate_stores,
        "{name}: duplicate_stores"
    );
    assert_eq!(
        counts.bigram_stores, want.bigram_stores,
        "{name}: bigram_stores"
    );
    let dropped: BTreeMap<&'static str, u64> = counts
        .dropped
        .iter()
        .map(|(reason, count)| (reason.as_str(), *count))
        .collect();
    assert_eq!(dropped, want.dropped, "{name}: dropped");
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// vLLM: both hash forms, a sliding-window group, a second physical copy
/// with per-copy removals, the offload tiers and every drop rule, a pool
/// reset, a LoRA request with a multimodal item, a cache salt and a prompt
/// embeddings digest whose child inherits the salt, and a second DP rank.
fn vllm(layout: Layout) -> Scenario {
    let h = |name: &str| vllm_hash(&format!("vllm-{name}"));
    let e = |name: &str| h(name).forwarded();
    let (d1, d2) = (
        Hash::Digest(digest("vllm-digest-1")),
        Hash::Digest(digest("vllm-digest-2")),
    );
    let embeds = digest("vllm-prompt-embeds");
    // The low 64 bits of a digest can exceed i64::MAX; make sure one does.
    assert!(
        "abcdefghijk"
            .chars()
            .any(|name| matches!(h(&name.to_string()), Hash::Unsigned(value) if value >= 1 << 63)),
        "pick labels with a high bit set"
    );
    let gpu = VllmStored::gpu;
    let mut batches = Vec::new();
    let mut want = Vec::new();

    // Batch 0: a plain chain in both hash forms; a sliding-window group store.
    batches.push(batch(
        1_700_000_000.0,
        0,
        vec![
            VllmStored {
                session_id: Some("req-1"),
                ..gpu(vec![h("a"), h("b")], None, (1..=8).collect())
            }
            .wire(layout),
            // Sliding-window group: more tokens than hashes x block size, no
            // hashes at all. Dropped by the group gate.
            VllmStored {
                group_idx: Some(1),
                spec_kind: Some("sliding_window"),
                sliding_window: Some(128),
                ..gpu(Vec::new(), None, (1..=8).collect())
            }
            .wire(layout),
            // The same group's usual shape: token_ids span the whole computed
            // range and block_hashes name only the window's last blocks, so
            // the hashes belong to the tail of the tokens. Dropped whole by
            // the group gate, never sliced from the head.
            VllmStored {
                group_idx: Some(1),
                spec_kind: Some("sliding_window"),
                sliding_window: Some(128),
                ..gpu(vec![h("k")], None, (1..=24).collect())
            }
            .wire(layout),
            // Raw digests, the parent given as an int, no extra keys on
            // either block.
            VllmStored {
                extra_keys: Some(vec![None, None]),
                ..gpu(vec![d1, d2], Some(h("b")), (9..=16).collect())
            }
            .wire(layout),
        ],
    ));
    want.extend([
        Want::Stored(WantStored {
            group_idx: Some(0),
            session_id: Some("req-1"),
            ..WantStored::device(
                vec![e("a"), e("b")],
                vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
            )
        }),
        Want::Stored(WantStored {
            parent: Some(e("b")),
            group_idx: Some(0),
            ..WantStored::device(
                vec![d1.forwarded(), d2.forwarded()],
                vec![vec![9, 10, 11, 12], vec![13, 14, 15, 16]],
            )
        }),
    ]);

    // Batch 1: a second physical copy, per-copy removals, offload tiers and
    // every other drop rule.
    batches.push(batch(
        1_700_000_001.0,
        0,
        vec![
            gpu(vec![h("a"), h("b")], None, (1..=8).collect()).wire(layout), // duplicate copy
            vllm_removed(layout, &[h("a")], "GPU", Some(0)),
            vllm_removed(layout, &[h("a")], "GPU", Some(0)), // the other copy
            // CPU offload placeholder: a chunk key, no tokens, block_size 0.
            VllmStored {
                medium: Some("CPU"),
                block_size: 0,
                spec_kind: None,
                ..gpu(vec![h("c")], None, Vec::new())
            }
            .wire(layout),
            vllm_removed(layout, &[h("c")], "CPU", Some(0)),
            VllmStored {
                medium: Some("STORAGE"),
                locality: Some("REMOTE"),
                ..gpu(vec![h("d")], None, vec![1, 2, 3, 4])
            }
            .wire(layout),
            VllmStored {
                medium: Some("STORAGE"),
                locality: Some("LOCAL"),
                ownership: Some("kvcr"),
                ..gpu(vec![h("d")], None, vec![1, 2, 3, 4])
            }
            .wire(layout),
            VllmStored {
                medium: Some("STORAGE"),
                locality: Some("LOCAL"),
                ..gpu(vec![h("d")], None, vec![1, 2, 3, 4])
            }
            .wire(layout),
            VllmStored {
                medium: Some("MARS"),
                ..gpu(vec![h("f")], None, vec![1, 2, 3, 4])
            }
            .wire(layout),
            gpu(vec![h("g")], None, vec![1, 2, 3, 4, 5, 6]).wire(layout), // unaligned
            gpu(vec![h("i")], Some(h("i")), vec![1, 2, 3, 4]).wire(layout), // parent is itself
            migrated(layout, &[h("j")]),
            malformed_store(layout),
        ],
    ));
    want.extend([
        Want::Stored(WantStored {
            group_idx: Some(0),
            ..WantStored::device(
                vec![e("a"), e("b")],
                vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
            )
        }),
        removed(vec![e("a")]),
        removed(vec![e("a")]),
        removed_on(vec![e("c")], KvCacheTier::Host, 1),
        Want::Stored(WantStored {
            tier: KvCacheTier::Disk,
            cache_level: Some(2),
            group_idx: Some(0),
            ..WantStored::device(vec![e("d")], vec![vec![1, 2, 3, 4]])
        }),
    ]);

    // Batch 2: the pool reset, then the chain again (not a duplicate any more).
    batches.push(batch(
        1_700_000_002.0,
        0,
        vec![
            cleared(layout),
            gpu(vec![h("a"), h("b")], None, (1..=8).collect()).wire(layout),
        ],
    ));
    want.extend([
        Want::Cleared { rank: 0 },
        Want::Stored(WantStored {
            group_idx: Some(0),
            ..WantStored::device(
                vec![e("a"), e("b")],
                vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
            )
        }),
    ]);

    // Batch 3: a LoRA request with a multimodal item, a cache salt and prompt
    // embeddings; the salt rides in block 0's extra keys only and the child
    // inherits it.
    batches.push(batch(
        1_700_000_003.0,
        0,
        vec![
            VllmStored {
                lora_id: Some(7),
                lora_name: Some("adapter"),
                extra_keys: Some(vec![Some(vec![
                    Value::from("adapter"),
                    Value::Array(vec![Value::from("mm-abc"), Value::from(0u64)]),
                    Value::from("salt-1"),
                    Value::from(embeds.to_vec()),
                ])]),
                session_id: Some("req-2"),
                ..gpu(vec![h("e")], None, vec![1, 2, 3, 4])
            }
            .wire(layout),
            VllmStored {
                lora_id: Some(7),
                lora_name: Some("adapter"),
                extra_keys: Some(vec![Some(vec![Value::from("adapter")])]),
                session_id: Some("req-2"),
                ..gpu(vec![h("h")], Some(h("e")), vec![5, 6, 7, 8])
            }
            .wire(layout),
        ],
    ));
    want.extend([
        Want::Stored(WantStored {
            lora_name: Some("adapter"),
            cache_salt: Some("salt-1"),
            group_idx: Some(0),
            session_id: Some("req-2"),
            extra_keys: Some(vec![vec![
                Key::Text("adapter"),
                Key::Multimodal("mm-abc", 0),
                Key::Text("salt-1"),
                Key::BlobLen(32),
            ]]),
            ..WantStored::device(vec![e("e")], vec![vec![1, 2, 3, 4]])
        }),
        Want::Stored(WantStored {
            parent: Some(e("e")),
            lora_name: Some("adapter"),
            cache_salt: Some("salt-1"),
            group_idx: Some(0),
            session_id: Some("req-2"),
            extra_keys: Some(vec![vec![Key::Text("adapter")]]),
            ..WantStored::device(vec![e("h")], vec![vec![5, 6, 7, 8]])
        }),
    ]);

    // Batch 4: another DP rank stores the same hashes; seen-sets are per rank.
    batches.push(batch(
        1_700_000_004.0,
        1,
        vec![gpu(vec![h("a"), h("b")], None, (1..=8).collect()).wire(layout)],
    ));
    want.push(Want::Stored(WantStored {
        rank: 1,
        group_idx: Some(0),
        ..WantStored::device(
            vec![e("a"), e("b")],
            vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
        )
    }));

    Scenario {
        name: format!("vllm-{}", layout.name()),
        engine: "vllm",
        layout,
        batches,
        want,
        counts: WantCounts {
            stored: 8,
            removed: 3,
            cleared: 1,
            duplicate_stores: 1,
            bigram_stores: 0,
            dropped: [
                ("non_main_attention_group", 2),
                ("placeholder", 1),
                ("non_local_locality", 1),
                ("unsupported_ownership", 1),
                ("unknown_medium", 1),
                ("unaligned_blocks", 1),
                ("self_referencing_hashes", 1),
                ("unknown_type", 1),
                ("malformed", 1),
            ]
            .into_iter()
            .collect(),
        },
    }
}

/// SGLang: the startup clear, a chain with a coalesced two-page store,
/// HiCache write-through (store on the host, demote, load back, evict the
/// host copy), a salted chain, an Eagle bigram page, the DISK and EXTERNAL
/// media, an unknown event type and a second attention DP rank.
fn sglang(layout: Layout) -> Scenario {
    let s = |name: &str| sglang_hash(&format!("sglang-{name}"));
    let e = |name: &str| s(name).forwarded();
    assert!(
        "abcdefgh".chars().any(|name| e(&name.to_string()) < 0),
        "pick labels with a negative i64"
    );
    let gpu = SglangStored::gpu;
    // The legacy array has no readable salt slot (it is vLLM's `lora_name`
    // there), so that variant stores the chain unsalted; and it puts
    // `session_id` where vLLM has `extra_keys`, where the relay cannot read
    // it.
    let (salt, session) = match layout {
        Layout::Map => (Some("tenant-a"), Some("req-1")),
        Layout::Array => (None, None),
    };
    let mut batches = Vec::new();
    let mut want = Vec::new();

    // Batch 0: the first batch after startup clears.
    batches.push(batch(1_700_000_000.0, 0, vec![cleared(layout)]));
    want.push(Want::Cleared { rank: 0 });

    // Batch 1: a chain; the second store is coalesced over two pages.
    batches.push(batch(
        1_700_000_001.0,
        0,
        vec![
            SglangStored {
                session_id: Some("req-1"),
                ..gpu(vec![s("a")], None, &[1, 2, 3, 4])
            }
            .wire(layout),
            SglangStored {
                session_id: Some("req-1"),
                ..gpu(
                    vec![s("b"), s("c")],
                    Some(s("a")),
                    &(5..=12).collect::<Vec<_>>(),
                )
            }
            .wire(layout),
        ],
    ));
    want.extend([
        Want::Stored(WantStored {
            session_id: session,
            ..WantStored::device(vec![e("a")], vec![vec![1, 2, 3, 4]])
        }),
        Want::Stored(WantStored {
            parent: Some(e("a")),
            session_id: session,
            ..WantStored::device(
                vec![e("b"), e("c")],
                vec![vec![5, 6, 7, 8], vec![9, 10, 11, 12]],
            )
        }),
    ]);

    // Batch 2: HiCache write-through: back up to host, demote (device copy
    // goes, host stays), load back, evict the host copy.
    batches.push(batch(
        1_700_000_002.0,
        0,
        vec![
            SglangStored {
                medium: Some("CPU_PINNED"),
                ..gpu(vec![s("a")], None, &[1, 2, 3, 4])
            }
            .wire(layout),
            sglang_removed(layout, &[s("a")], "GPU"),
            gpu(vec![s("a")], None, &[1, 2, 3, 4]).wire(layout),
            sglang_removed(layout, &[s("a")], "CPU_PINNED"),
        ],
    ));
    want.extend([
        Want::Stored(WantStored {
            tier: KvCacheTier::Host,
            cache_level: Some(1),
            ..WantStored::device(vec![e("a")], vec![vec![1, 2, 3, 4]])
        }),
        removed(vec![e("a")]),
        Want::Stored(WantStored::device(vec![e("a")], vec![vec![1, 2, 3, 4]])),
        removed_on(vec![e("a")], KvCacheTier::Host, 1),
    ]);

    // Batch 3: a salted request's chain.
    batches.push(batch(
        1_700_000_003.0,
        0,
        vec![
            SglangStored {
                cache_salt: salt,
                ..gpu(vec![s("d")], None, &[1, 2, 3, 4])
            }
            .wire(layout),
            SglangStored {
                cache_salt: salt,
                ..gpu(vec![s("e")], Some(s("d")), &[5, 6, 7, 8])
            }
            .wire(layout),
        ],
    ));
    want.extend([
        Want::Stored(WantStored {
            cache_salt: salt,
            ..WantStored::device(vec![e("d")], vec![vec![1, 2, 3, 4]])
        }),
        Want::Stored(WantStored {
            parent: Some(e("d")),
            cache_salt: salt,
            ..WantStored::device(vec![e("e")], vec![vec![5, 6, 7, 8]])
        }),
    ]);

    // Batch 4: an Eagle bigram page, removed again in the same batch. (Its
    // tokens differ from the plain chain's: two engine hashes with the same
    // tokens at the same position share one index membership per worker.)
    batches.push(batch(
        1_700_000_004.0,
        0,
        vec![
            SglangStored {
                tokens: bigrams(&[(21, 22), (22, 23), (23, 24), (24, 25)]),
                ..gpu(vec![s("f")], None, &[])
            }
            .wire(layout),
            sglang_removed(layout, &[s("f")], "GPU"),
        ],
    ));
    want.extend([
        Want::Stored(WantStored::device(vec![e("f")], vec![vec![21, 22, 23, 24]])),
        removed(vec![e("f")]),
    ]);

    // Batch 5: the tiers the default core never emits but defines; an event
    // type the relay does not know.
    batches.push(batch(
        1_700_000_005.0,
        0,
        vec![
            SglangStored {
                medium: Some("DISK"),
                ..gpu(vec![s("g")], None, &[1, 2, 3, 4])
            }
            .wire(layout),
            SglangStored {
                medium: Some("EXTERNAL"),
                ..gpu(vec![s("h")], None, &[1, 2, 3, 4])
            }
            .wire(layout),
            migrated(layout, &[s("h")]),
        ],
    ));
    want.extend([
        Want::Stored(WantStored {
            tier: KvCacheTier::Disk,
            cache_level: Some(2),
            ..WantStored::device(vec![e("g")], vec![vec![1, 2, 3, 4]])
        }),
        Want::Stored(WantStored {
            tier: KvCacheTier::External,
            cache_level: Some(3),
            ..WantStored::device(vec![e("h")], vec![vec![1, 2, 3, 4]])
        }),
    ]);

    // Batch 6: another attention DP rank; its batch carries its rank.
    batches.push(batch(
        1_700_000_006.0,
        1,
        vec![gpu(vec![s("a")], None, &[1, 2, 3, 4]).wire(layout)],
    ));
    want.push(Want::Stored(WantStored {
        rank: 1,
        ..WantStored::device(vec![e("a")], vec![vec![1, 2, 3, 4]])
    }));

    Scenario {
        name: format!("sglang-{}", layout.name()),
        engine: "sglang",
        layout,
        batches,
        want,
        counts: WantCounts {
            stored: 10,
            removed: 3,
            cleared: 1,
            duplicate_stores: 0,
            bigram_stores: 1,
            dropped: [("unknown_type", 1)].into_iter().collect(),
        },
    }
}

fn scenarios() -> [Scenario; 4] {
    [
        vllm(Layout::Map),
        vllm(Layout::Array),
        sglang(Layout::Map),
        sglang(Layout::Array),
    ]
}

fn run(scenario: &Scenario) {
    let (batches, counts) = normalized(scenario);
    check(scenario, &batches, &counts);
}

#[test]
fn vllm_tagged_maps_normalize_as_expected() {
    run(&vllm(Layout::Map));
}

#[test]
fn vllm_legacy_arrays_normalize_as_expected() {
    run(&vllm(Layout::Array));
}

#[test]
fn sglang_tagged_maps_normalize_as_expected() {
    run(&sglang(Layout::Map));
}

#[test]
fn sglang_legacy_arrays_normalize_as_expected() {
    run(&sglang(Layout::Array));
}

/// The two layouts of one engine carry the same events, so the relay must
/// forward the same stream from either, down to the event ids; the one
/// difference is what the legacy array cannot say (SGLang's salt and
/// session, which the array scenario leaves out on both sides).
#[test]
fn both_layouts_forward_the_same_stream() {
    for engine in [vllm, sglang] {
        let (from_maps, map_counts) = normalized(&engine(Layout::Map));
        let (from_arrays, array_counts) = normalized(&engine(Layout::Array));
        assert_eq!(map_counts, array_counts);
        let strip = |batches: Vec<KvEventBatch>| -> Vec<KvEventBatch> {
            batches
                .into_iter()
                .map(|mut batch| {
                    for event in &mut batch.events {
                        if let Some(kv_cache_event::Data::Stored(stored)) = &mut event.data {
                            stored.session_id = None;
                            stored.cache_salt = None;
                        }
                    }
                    batch
                })
                .collect()
        };
        assert_eq!(strip(from_maps), strip(from_arrays));
    }
}

// ---------------------------------------------------------------------------
// Round trip: wire -> proto -> index
// ---------------------------------------------------------------------------

/// Apply normalized events to the index the way the gateway's monitor does
/// for the device tier: stores hashed under the event's namespace, removals
/// by engine hash, clears per worker, each rank its own worker. Host
/// residency is the monitor's own bookkeeping and has its tests there; here
/// host events are skipped.
fn index(batches: &[KvEventBatch]) -> ReferenceIndexer {
    let mut indexer = ReferenceIndexer::new();
    for batch in batches {
        let worker = u32::try_from(batch.dp_rank.unwrap_or(0)).expect("a rank");
        for event in &batch.events {
            match &event.data {
                Some(kv_cache_event::Data::Stored(stored)) => {
                    apply_stored(&mut indexer, worker, stored);
                }
                Some(kv_cache_event::Data::Removed(removed)) => {
                    if device_tier(removed.tier) {
                        let hashes: Vec<SequenceHash> = removed
                            .block_hashes
                            .iter()
                            .map(|&hash| SequenceHash::from(hash))
                            .collect();
                        indexer.apply_removed(worker, &hashes);
                    }
                }
                Some(kv_cache_event::Data::Cleared(_)) => indexer.apply_cleared(worker),
                None => {}
            }
        }
    }
    indexer
}

fn apply_stored(indexer: &mut ReferenceIndexer, worker: u32, stored: &KvBlocksStored) {
    if !device_tier(stored.tier) {
        return;
    }
    let seed = namespace_seed(stored.lora_name.as_deref(), stored.cache_salt.as_deref());
    let converted: Vec<StoredBlock> = stored
        .blocks
        .iter()
        .map(|block| StoredBlock {
            seq_hash: SequenceHash::from(block.block_hash),
            content_hash: content_hash_with_seed(&block.token_ids, seed),
        })
        .collect();
    let parent = stored.parent_block_hash.map(SequenceHash::from);
    if let Err(ApplyError::WorkerNotTracked | ApplyError::ParentBlockNotFound) =
        indexer.apply_stored(worker, &converted, parent)
    {
        indexer
            .apply_stored(worker, &converted, None)
            .expect("a parentless store applies");
    }
}

fn device_tier(tier: Option<i32>) -> bool {
    matches!(
        KvCacheTier::try_from(tier.unwrap_or_default()),
        Ok(KvCacheTier::Device | KvCacheTier::Unspecified)
    )
}

/// The longest prefix of the request `worker` holds.
fn depth(indexer: &ReferenceIndexer, worker: u32, hashes: &[ContentHash]) -> u32 {
    indexer
        .find_matches(hashes)
        .get(&worker)
        .copied()
        .unwrap_or(0)
}

#[test]
fn the_normalized_streams_round_trip_into_the_reference_index() {
    let tokens: Vec<u32> = (1..=16).collect();
    let plain = |n: usize| compute_request_content_hashes(&tokens[..n], 4);
    for scenario in &scenarios() {
        let (batches, _) = normalized(scenario);
        let indexer = index(&batches);
        let name = &scenario.name;
        match scenario.engine {
            "vllm" => {
                // The chain of two device blocks survives the duplicate copy,
                // its per-copy removals and the reset (it is stored again).
                assert_eq!(depth(&indexer, 0, &plain(8)), 2, "{name}: plain chain");
                // The digest-hashed continuation was cleared and not restored.
                assert_eq!(depth(&indexer, 0, &plain(16)), 2, "{name}: cleared tail");
                // The LoRA + salt chain matches only under its namespace, the
                // child included (it inherited the salt from block 0).
                let salted = namespaced_request_content_hashes(
                    &tokens[..8],
                    4,
                    Some("adapter"),
                    Some("salt-1"),
                );
                assert_eq!(depth(&indexer, 0, &salted), 2, "{name}: salted chain");
                let lora_only =
                    namespaced_request_content_hashes(&tokens[..8], 4, Some("adapter"), None);
                assert_eq!(
                    depth(&indexer, 0, &lora_only),
                    0,
                    "{name}: lora without salt"
                );
                let salt_only =
                    namespaced_request_content_hashes(&tokens[..8], 4, None, Some("salt-1"));
                assert_eq!(
                    depth(&indexer, 0, &salt_only),
                    0,
                    "{name}: salt without lora"
                );
                assert_eq!(depth(&indexer, 1, &plain(8)), 2, "{name}: rank 1");
            }
            "sglang" => {
                // Device chain of three pages; the HiCache demote and host
                // eviction left the load-back copy in place.
                assert_eq!(depth(&indexer, 0, &plain(12)), 3, "{name}: plain chain");
                if scenario.layout == Layout::Map {
                    let salted =
                        namespaced_request_content_hashes(&tokens[..8], 4, None, Some("tenant-a"));
                    assert_eq!(depth(&indexer, 0, &salted), 2, "{name}: salted chain");
                    let other =
                        namespaced_request_content_hashes(&tokens[..8], 4, None, Some("tenant-b"));
                    assert_eq!(depth(&indexer, 0, &other), 0, "{name}: other salt");
                }
                assert_eq!(depth(&indexer, 1, &plain(4)), 1, "{name}: rank 1");
            }
            other => panic!("{name}: unknown engine {other}"),
        }
    }
}
