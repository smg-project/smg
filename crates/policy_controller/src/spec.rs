//! `PolicySpec` (design 4.6): the one versioned document. The view says what
//! the host builds into every decision, the index map what the prefix index
//! keeps per model, profiles name stages with their parameters, plans carry
//! their rules, and bindings match requests to profiles or plans and carry
//! the per-binding rules. Validation answers what the host must feed before
//! any request exists.
//!
//! The document has an `apiVersion` and no `kind`: its forms are a file and
//! the API. An unknown field is refused by name, never ignored. A field this
//! build cannot honour yet (a stage, a plan, a matcher or a binding rule of a
//! later step) is checked in shape like every other field and then refused
//! by name as not in this build, so a document written against the published
//! schema validates the day its stage lands. [`PolicySpec::schema`] is that
//! schema; the field documentation of this module is its descriptions.

use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::{btree_map, BTreeMap, BTreeSet},
    fmt, io,
    marker::PhantomData,
    sync::LazyLock,
};

use rustc_hash::FxHashMap;
use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{
    de::{self, value::SeqAccessDeserializer, MapAccess, SeqAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};
use sha2::{Digest as _, Sha256};

use crate::{
    affinity::{blocks_within, Gates, KvPressure, PrefixAffinity, WarmupSlice},
    bucket::Bucket,
    builtins::{
        FewestSessions, LeastLoad, LeastRequests, ResidentWork, SampledBest, SessionAffinity,
        DEFAULT_MEAN_PREFILL_TOKENS,
    },
    catalog::{Catalog, Found},
    framework::{Binding, Engine, OverloadThresholds, Picker, Plan, Profile, Scorer, Weighted},
    ring::ConsistentHash,
    sessions::SessionRule,
    view::{permille, HashKind, Phase},
};

pub const API_VERSION: &str = "smg.policy/v1";

/// Header names no allow-list may carry (design 4.6). A host adds the tenant
/// header it is configured with through [`PolicySpec::validate_with`].
pub const DENIED_HEADERS: [&str; 7] = [
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
];

/// The most headers `view.headers` may list: a view names an allow-listed
/// header by its position in one byte.
pub const MAX_HEADERS: usize = 256;

/// The highest `view.max_block_hashes`: the hashes a `DecideRequest` at
/// the contract's bound of 64 KiB carries beside an id of one byte, at
/// eight bytes a hash and ten of tags and lengths. A higher cap would
/// bind on no frame the call accepts.
pub const MAX_BLOCK_HASHES: u32 = 8_190;

/// The most label selectors `view.labels` may list: a row holds its labels
/// as one bit a selector, in 64 bits.
pub const MAX_LABELS: usize = 64;

/// The most zones `view.zones` may list. A row's position is wider: the cap
/// is the document's own, so a list never meets the width by accident.
pub const MAX_ZONES: usize = 256;

/// The most engine families `view.engines` may list, capped as the zones
/// are.
pub const MAX_ENGINES: usize = 64;

/// The filters this build carries.
pub const FILTERS: [&str; 5] = [
    "healthy",
    "not-overloaded",
    "has-capacity",
    "has-room",
    "not-tried",
];

/// The scorers of v1. Each has a parameter struct in this module; the ones
/// this build lacks parse and are refused as not in this build.
pub const SCORERS: [&str; 10] = [
    "least-load",
    "least-requests",
    "fewest-sessions",
    "session-affinity",
    "prefix-affinity",
    "kv-pressure",
    "warmup-slice",
    "bucket",
    "consistent-hash",
    "resident-work",
];

/// The pickers of v1; `count-first` and `power-of-two-count-first` compare
/// the requests in flight before the score; `manual` leaves the placement
/// to the binding's sessions rule, by its assignment; `best-with-temperature`
/// takes its temperature and is written in the parameter form, as `best` is
/// when it takes a sample.
pub const PICKERS: [&str; 9] = [
    "best",
    "round-robin",
    "random",
    "power-of-two",
    "count-first",
    "power-of-two-count-first",
    "passthrough",
    "manual",
    "best-with-temperature",
];

/// The stock build's catalog: what [`PolicySpec::validate`] resolves a
/// stage name in.
static STOCK: LazyLock<Catalog> = LazyLock::new(Catalog::builtin);

/// A sampling picker compares its draws by the profile's scorers, so it
/// needs at least one.
const SAMPLING_PICKERS: [&str; 2] = ["power-of-two", "power-of-two-count-first"];

/// The most rows the `best` picker's sample may hold: a row joins the
/// sample after a search of the rows already in it.
const MAX_SAMPLE: u32 = 64;

/// The side names a binding's `profiles` map may use.
const SIDES: [&str; 4] = ["single", "prefill", "decode", "encode"];

/// What the host must feed for the spec's stages to mean anything, per
/// deployment. Read at apply, never per request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Needs {
    /// The request text in the view (a text-window hash form, for a host
    /// without a tokenizer).
    pub text: bool,
    /// The request's prefix block hashes over its token ids. A decision is
    /// handed the hashes as deep as the stages that declare them read
    /// ([`AppliedSpec::block_hashes_read`]), so a stage that reads the list
    /// declares one of the two kinds: under a generation where no stage
    /// does, it is handed none. A plugin is shown the list where its own
    /// row declares one of the two, and none of it otherwise.
    pub token_hashes: bool,
    /// The request's prefix block hashes over windows of its text, which a
    /// host without a tokenizer feeds in their place. A stage that reads a
    /// prefix by either kind sets this beside `text`.
    pub text_hashes: bool,
    /// Load reports per worker.
    pub loads: bool,
    /// KV events into an index.
    pub events: bool,
    /// A consistent-hash ring per model.
    pub ring: bool,
    /// Allow-listed headers in the view.
    pub headers: bool,
    pub media: bool,
    /// The count of session keys with a request in flight, per worker (the
    /// `min_group` assignment and the `fewest-sessions` scorer).
    pub active_sessions: bool,
    /// The request's own key's in-flight count per worker (a sessions rule
    /// with an in-flight cap).
    pub session_in_flight: bool,
    /// The affinity column
    /// ([`RequestFacts::affinity`](crate::RequestFacts::affinity)): one
    /// [`AffinityFacts`](crate::AffinityFacts) a candidate. A stage that
    /// reads any of the five declares this, whatever feeds them: `events`
    /// asks for the index and the hash fields for the request's hashes, and
    /// neither stands for the column. A host fills the column where this is
    /// declared and may hand an empty column elsewhere. Rows it names in
    /// the column's place ([`NamedAffinity`](crate::NamedAffinity)) answer
    /// the crate's own affinity stages and no other reader: a named row
    /// says its two depths and nothing of its age, growth or index size, so
    /// a host fills the column for a profile with a stage that reads those,
    /// a guest's among them. The engine checks the column's shape, never
    /// its presence: an unfilled column reads as nothing held for every
    /// candidate and nothing is refused. What holds a host to the need is
    /// the builder of its view, by a case of its own.
    pub affinity: bool,
    /// The sample of a picker that decides a large pool by a draw of this
    /// many rows and the rows the host names; `0` for none. A host that can
    /// name the request's holders and the pool's warming rows
    /// ([`NamedAffinity`](crate::NamedAffinity)) hands those in the
    /// column's place, each list capped at this many rows: the holders by
    /// depth, whole depth groups from the deepest while they fit and the
    /// group at the cap's edge by a uniform draw from the decision's seed;
    /// the warming rows by that draw over their whole list. A host that
    /// cannot fills the column, and the pool is read whole. The cap of a
    /// decision is its own profile's, which
    /// [`Engine::sample_for`](crate::Engine::sample_for) says: the engine's
    /// summary holds the largest sample of every profile, and a decode
    /// worker's needs the larger of its two profiles'.
    pub sample: u32,
}

impl Needs {
    /// Whether these declare a request's block hashes, in the token form or
    /// the text form. A stage that declares neither reads none of them.
    pub(crate) fn request_hashes(self) -> bool {
        self.token_hashes || self.text_hashes
    }

    /// Both sides' needs: what a host feeding either of two stage sets must
    /// feed.
    pub(crate) fn or(self, other: Self) -> Self {
        Self {
            text: self.text || other.text,
            token_hashes: self.token_hashes || other.token_hashes,
            text_hashes: self.text_hashes || other.text_hashes,
            loads: self.loads || other.loads,
            events: self.events || other.events,
            ring: self.ring || other.ring,
            headers: self.headers || other.headers,
            media: self.media || other.media,
            active_sessions: self.active_sessions || other.active_sessions,
            session_in_flight: self.session_in_flight || other.session_in_flight,
            affinity: self.affinity || other.affinity,
            sample: self.sample.max(other.sample),
        }
    }
}

fn is_default<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

/// What the host builds into every decision's view beyond the facts it
/// derives itself: the request headers a stage may read, the hashing
/// inputs a host keeps until it learns them from its engines, and the
/// lists a worker's row names its labels, its zone and its engine family
/// by.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct ViewSpec {
    /// The request headers a stage may read, lowercase, in the order their
    /// positions take on the wire. A name on the deny-list (`authorization`,
    /// `proxy-authorization`, `cookie`, `set-cookie`, `x-api-key`, `api-key`,
    /// `x-goog-api-key`, the host's tenant header) is refused. Every header a
    /// binding's `sessions`, `canary` or `match.header` names must be listed
    /// here. At most 256: a view names a header by its position in one byte.
    #[schemars(length(max = 256))]
    pub headers: Vec<String>,
    /// The token block size the host hashes prefixes by when no engine has
    /// published one; absent, the host's own default. The block size of
    /// record learned from an engine's events wins at run time.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub block_size: Option<u32>,
    /// Ascending token positions the hash index and the prefix-hash ring
    /// key on; empty for none. A hash index needs at least one.
    pub boundaries: Vec<u32>,
    /// How many prefix block hashes a request carries into the view, 8,190
    /// at most: one frame of the Decide call carries no more.
    #[schemars(range(min = 1, max = 8190))]
    pub max_block_hashes: u32,
    /// The worker labels a stage may read, each a selector of one label
    /// key: `key` (the worker carries the label, whatever its value) or
    /// `key=value` (it carries the label with that value), split at the
    /// first `=`. A worker's row holds one bit for each selector it meets,
    /// in this order. At most 64; empty for none.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(max = 64))]
    pub labels: Vec<String>,
    /// The zones a stage may tell apart. A worker's row names its zone by
    /// its position in this list, from 1; a worker with no zone reads 0,
    /// and so does one whose zone this list does not name. At most 256;
    /// empty for none, where every row reads 0.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(max = 256))]
    pub zones: Vec<String>,
    /// The engine families a stage may tell apart, spelled as a worker's
    /// registration spells its own. A worker's row names its family by its
    /// position in this list, from 1; a worker whose family this list does
    /// not name reads 0. At most 64; empty for none, where every row reads
    /// 0.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(max = 64))]
    pub engines: Vec<String>,
}

impl Default for ViewSpec {
    fn default() -> Self {
        Self {
            headers: Vec::new(),
            block_size: None,
            boundaries: Vec::new(),
            max_block_hashes: 256,
            labels: Vec::new(),
            zones: Vec::new(),
            engines: Vec::new(),
        }
    }
}

/// The under-layer of a model's prefix index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum IndexKind {
    /// Radix trees over the prefixes seen.
    #[default]
    Tree,
    /// A placement map keyed by the request's head at `view.boundaries`,
    /// each placement living `ttl_s`.
    Hash,
}

/// A model's prefix-index settings. They are per model, never per profile,
/// so two profiles bound to one model never carry two settings.
/// Two keys matching one model resolve by specificity, see
/// [`PolicySpec::index_for`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct IndexSpec {
    pub kind: IndexKind,
    /// How long a hash-index placement stays routable, in seconds.
    #[schemars(range(min = 1))]
    pub ttl_s: u32,
    /// The size bound of a tree index: characters on a text host, tokens
    /// otherwise.
    #[schemars(range(min = 1))]
    pub max_size: u64,
    /// The index's sweep cadence, in seconds.
    #[schemars(range(min = 1))]
    pub sweep_s: u32,
}

impl Default for IndexSpec {
    fn default() -> Self {
        Self {
            kind: IndexKind::Tree,
            ttl_s: 180,
            max_size: 67_108_864,
            sweep_s: 120,
        }
    }
}

/// How strictly a prefill pairs with a decode on their KV-transfer
/// descriptor. A descriptor component an engine does not report is unknown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Pairing {
    /// Pair on nothing, as if no descriptor existed.
    Off,
    /// A known difference in runtime, transport or KV layout refuses the
    /// pair; unknown components pair with anything.
    #[default]
    Lenient,
    /// Runtime, transport and KV layout are known on both sides and agree,
    /// and reported engine versions match.
    Strict,
}

/// The `pd-pair` plan's rules.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct PdPairPlanSpec {
    pub pairing: Pairing,
}

/// The plans' rules, by plan name.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct PlansSpec {
    #[serde(rename = "pd-pair")]
    pub pd_pair: PdPairPlanSpec,
}

fn one() -> f32 {
    1.0
}

/// A scorer with no parameters of its own: its weight alone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Weight {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
}

impl Default for Weight {
    fn default() -> Self {
        Self { weight: 1.0 }
    }
}

/// The least-load scorer: the expected wait from token work over the drain
/// rate, with a convex KV-pressure barrier.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LeastLoadParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
    /// The weight of the KV-pressure barrier in the expected wait; at or
    /// above zero (default 0.15).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub kv_pressure_weight: Option<f32>,
    /// The token-work estimate for a request whose count is unknown; above
    /// zero (default 1024).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub mean_prefill_tokens: Option<u32>,
    /// The drain rate assumed when no candidate reports a live one, in
    /// tokens per second; above zero (default 2000).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub default_throughput: Option<f32>,
}

impl Default for LeastLoadParams {
    fn default() -> Self {
        Self {
            weight: 1.0,
            kv_pressure_weight: None,
            mean_prefill_tokens: None,
            default_throughput: None,
        }
    }
}

/// Where prefix affinity reads placements from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum AffinitySource {
    /// The event-fed prefix index.
    #[default]
    Index,
    /// The side index of prefixes sent, for workers without an event
    /// subscription.
    Predictor,
    /// Both, merged by per-worker maximum.
    Merged,
}

/// The spill gate of prefix affinity: when the affine worker is loaded this
/// far past the least-loaded one, the request spills to the latter.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpillParams {
    /// The absolute part of the gate, in requests (default 64).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abs: Option<u32>,
    /// The relative part of the gate; at least 1 (default 1.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1.0))]
    pub rel: Option<f32>,
    /// The gate's floor from the held prefix, in tokens (default 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_margin_tokens: Option<u32>,
    /// Credit a hit with its uncached remainder only (default false).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncached_credit: Option<bool>,
}

/// The prefix-affinity scorer: the matched prefix share per candidate, with
/// the match threshold, the spill gate and the placement source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PrefixAffinityParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
    /// The matched share under which a request is balanced instead of
    /// pinned; between 0 and 1 (default 0.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub threshold: Option<f32>,
    /// The decay of the overlap by the candidate's waiting backlog; at or
    /// above zero (default 0, off).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub decay: Option<f32>,
    /// Where placements come from (default `index`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<AffinitySource>,
    /// How long an optimistic dispatch booking lives before the engine
    /// reports it, in milliseconds (default 0, off).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accounting_ttl_ms: Option<u64>,
    /// The spill gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spill: Option<SpillParams>,
}

impl Default for PrefixAffinityParams {
    fn default() -> Self {
        Self {
            weight: 1.0,
            threshold: None,
            decay: None,
            source: None,
            accounting_ttl_ms: None,
            spill: None,
        }
    }
}

/// The KV-pressure gate: the fleet's KV-usage spread, or the usage the
/// hottest candidate crosses, past which the profile's affinity stages
/// yield to load. The stage orders nothing itself.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KvPressureParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
    /// The usage spread above which load wins; above zero, 1 and above
    /// switches the rule off (default 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub spread: Option<f32>,
    /// The usage the hottest candidate crosses for affinity to yield to
    /// load; above zero, 1 and above switches the rule off (default 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub ceiling: Option<f32>,
}

impl Default for KvPressureParams {
    fn default() -> Self {
        Self {
            weight: 1.0,
            spread: None,
            ceiling: None,
        }
    }
}

/// The warm-up slice: the share of misses offered to workers whose index is
/// still thin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WarmupSliceParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
    /// How long after admission a worker counts as warming, in seconds; 0
    /// switches the age rule off (default 60).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secs: Option<u64>,
    /// The share of misses offered to warming workers; between 0 and 1
    /// (default 0.25).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub share: Option<f32>,
    /// The index growth after which a worker is warm, in blocks (default
    /// 1024).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocks: Option<u64>,
    /// The share of the fleet's median index under which a worker is thin;
    /// between 0 and 1, 0 keeps the age rule alone (default 0.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub thin_ratio: Option<f32>,
    /// One hit in this many is diverted to a thin worker; 0 switches the
    /// diversion off (default 8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub divert_every: Option<u64>,
    /// The index overlap share at or under which a hit counts as a miss
    /// for the slice; between 0 and 1 (default 0.3, the affinity
    /// threshold's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub miss_share: Option<f32>,
}

impl Default for WarmupSliceParams {
    fn default() -> Self {
        Self {
            weight: 1.0,
            secs: None,
            share: None,
            blocks: None,
            thin_ratio: None,
            divert_every: None,
            miss_share: None,
        }
    }
}

/// The bucket scorer: prefill placement by request size, over ranges of
/// sizes. A cut of the ranges falls due each interval and is made, from the
/// tokens booked on the workers, where a worker's share of them moved by
/// more than a factor of two since the last cut or the listing changed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BucketParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
    /// The absolute part of the balance rule, in booked tokens (default 64).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abs_threshold: Option<u32>,
    /// The relative part of the balance rule; at least 1 (default 1.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1.0))]
    pub rel_threshold: Option<f32>,
    /// The bookings' window and the cut's cadence, in seconds; above zero
    /// (default 5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub adjust_interval_s: Option<u32>,
}

impl Default for BucketParams {
    fn default() -> Self {
        Self {
            weight: 1.0,
            abs_threshold: None,
            rel_threshold: None,
            adjust_interval_s: None,
        }
    }
}

/// The bounded-load rule of the consistent-hash scorer: a ring owner whose
/// requests in flight are past both parts of the rule over the candidates'
/// mean, the request being placed counted in the mean, is passed over. A
/// prefix keyed at a boundary is then tried at the next boundary down. A
/// request no owner keeps is scored as one with no key: the profile's next
/// stage places it, and in a profile with no stage after the ring the picker
/// draws among every candidate, the passed owner among them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoundedLoadParams {
    /// The relative part of the rule; at least 1 (default 1.25).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1.0))]
    pub factor: Option<f64>,
    /// The absolute part of the rule, in requests (default 10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abs_margin: Option<u32>,
}

/// The consistent-hash scorer: a per-model ring keyed by the routing key
/// and, under `prefix_tokens`, by the prefix of a request that carries none,
/// with a bounded-load rule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConsistentHashParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
    /// The depth in tokens at which a request with no routing key is keyed
    /// by its prefix: the last whole block at or under the depth, so at
    /// least one block of the view's block size and at most the blocks a
    /// request carries. Under the view's `boundaries` a request that reaches
    /// one is keyed there instead, at the deepest it reaches. The depth is
    /// checked in the view's token blocks and read in the blocks a request
    /// was hashed by. A text request's windows are `block_size` characters,
    /// a quarter of a token block, so it carries a quarter of the checked
    /// depth and a deeper depth keys at its last window. Under a block size
    /// an engine published above the depth a request has no key. Absent, a
    /// request with no routing key has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub prefix_tokens: Option<u32>,
    /// The bounded-load rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounded_load: Option<BoundedLoadParams>,
}

impl Default for ConsistentHashParams {
    fn default() -> Self {
        Self {
            weight: 1.0,
            prefix_tokens: None,
            bounded_load: None,
        }
    }
}

/// The resident-work scorer: the context tokens resident on a candidate,
/// and a price in tokens for every request it holds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResidentWorkParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
    /// What one request a candidate holds adds to its resident tokens, in
    /// context tokens; zero compares the tokens alone.
    pub request_tokens: u32,
}

/// The `best` picker's parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BestParams {
    /// How many rows the picker draws where the host names the request's
    /// holders and the pool's warming rows: a pool of more than twice this
    /// many rows, and of at least 16, is decided by that sample and the
    /// named rows, any other by the whole pool. From 1 to 64.
    #[schemars(range(min = 1, max = 64))]
    pub sample: u32,
}

/// The `best-with-temperature` picker's parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BestWithTemperatureParams {
    /// The softmax temperature over the combined scores; at or above zero,
    /// 0 picks the best.
    #[schemars(range(min = 0.0))]
    pub temperature: f32,
}

/// A scorer with its parameters, keyed by the scorer's name: `{ least-load:
/// { weight: 0.3 } }`. One key per entry; a second key or an unknown
/// parameter is refused by name.
#[derive(Clone, Debug, PartialEq, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ScoreStage {
    LeastLoad(LeastLoadParams),
    LeastRequests(Weight),
    FewestSessions(Weight),
    SessionAffinity(Weight),
    PrefixAffinity(PrefixAffinityParams),
    KvPressure(KvPressureParams),
    WarmupSlice(WarmupSliceParams),
    Bucket(BucketParams),
    ConsistentHash(ConsistentHashParams),
    ResidentWork(ResidentWorkParams),
}

impl ScoreStage {
    /// The scorer's name, as the document spells it.
    pub fn name(&self) -> &'static str {
        match self {
            Self::LeastLoad(_) => "least-load",
            Self::LeastRequests(_) => "least-requests",
            Self::FewestSessions(_) => "fewest-sessions",
            Self::SessionAffinity(_) => "session-affinity",
            Self::PrefixAffinity(_) => "prefix-affinity",
            Self::KvPressure(_) => "kv-pressure",
            Self::WarmupSlice(_) => "warmup-slice",
            Self::Bucket(_) => "bucket",
            Self::ConsistentHash(_) => "consistent-hash",
            Self::ResidentWork(_) => "resident-work",
        }
    }

    /// The scorer's weight in the profile's combined score.
    pub fn weight(&self) -> f32 {
        match self {
            Self::LeastLoad(p) => p.weight,
            Self::LeastRequests(p) | Self::FewestSessions(p) | Self::SessionAffinity(p) => p.weight,
            Self::PrefixAffinity(p) => p.weight,
            Self::KvPressure(p) => p.weight,
            Self::WarmupSlice(p) => p.weight,
            Self::Bucket(p) => p.weight,
            Self::ConsistentHash(p) => p.weight,
            Self::ResidentWork(p) => p.weight,
        }
    }

    /// The stage a bare name stands for, with its parameters at their
    /// defaults; `None` for a name outside the catalog or one whose
    /// parameters have no default (`resident-work`).
    fn by_name(name: &str) -> Option<Self> {
        Some(match name {
            "least-load" => Self::LeastLoad(LeastLoadParams::default()),
            "least-requests" => Self::LeastRequests(Weight::default()),
            "fewest-sessions" => Self::FewestSessions(Weight::default()),
            "session-affinity" => Self::SessionAffinity(Weight::default()),
            "prefix-affinity" => Self::PrefixAffinity(PrefixAffinityParams::default()),
            "kv-pressure" => Self::KvPressure(KvPressureParams::default()),
            "warmup-slice" => Self::WarmupSlice(WarmupSliceParams::default()),
            "bucket" => Self::Bucket(BucketParams::default()),
            "consistent-hash" => Self::ConsistentHash(ConsistentHashParams::default()),
            _ => return None,
        })
    }
}

/// A scorer reference: a bare name (a built-in at its defaults, or a scorer
/// the catalog adds at weight one), or the name mapped to its weight and
/// parameters ([`ScoreStage`]).
#[derive(Clone, Debug, PartialEq)]
pub enum ScoreSpec {
    Name(String),
    Params(ScoreStage),
    /// A name that is no built-in's, mapped to what the document wrote under
    /// it: a scorer the catalog adds, with its weight (`{ name: { weight }
    /// }`). Kept as read: the validator asks the catalog for the name first
    /// and reads the keys of a name it holds. A key written twice is the
    /// parser's refusal, as under a built-in.
    Added {
        name: String,
        written: serde_json::Value,
    },
}

impl ScoreSpec {
    /// The scorer's name as written.
    pub fn name(&self) -> &str {
        match self {
            Self::Name(name) | Self::Added { name, .. } => name,
            Self::Params(stage) => stage.name(),
        }
    }
}

impl From<&str> for ScoreSpec {
    fn from(name: &str) -> Self {
        Self::Name(name.to_string())
    }
}

impl From<ScoreStage> for ScoreSpec {
    fn from(stage: ScoreStage) -> Self {
        Self::Params(stage)
    }
}

impl Serialize for ScoreSpec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let stage = match self {
            Self::Name(name) => return serializer.serialize_str(name),
            Self::Added { name, written } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(name, written)?;
                return map.end();
            }
            Self::Params(stage) => stage,
        };
        // One key, the stage's name, to its parameters: the document form in
        // JSON and in YAML alike (a derived enum would render as a YAML tag).
        let mut map = serializer.serialize_map(Some(1))?;
        match stage {
            ScoreStage::LeastLoad(p) => map.serialize_entry(stage.name(), p)?,
            ScoreStage::LeastRequests(p)
            | ScoreStage::FewestSessions(p)
            | ScoreStage::SessionAffinity(p) => {
                map.serialize_entry(stage.name(), p)?;
            }
            ScoreStage::PrefixAffinity(p) => map.serialize_entry(stage.name(), p)?,
            ScoreStage::KvPressure(p) => map.serialize_entry(stage.name(), p)?,
            ScoreStage::WarmupSlice(p) => map.serialize_entry(stage.name(), p)?,
            ScoreStage::Bucket(p) => map.serialize_entry(stage.name(), p)?,
            ScoreStage::ConsistentHash(p) => map.serialize_entry(stage.name(), p)?,
            ScoreStage::ResidentWork(p) => map.serialize_entry(stage.name(), p)?,
        }
        map.end()
    }
}

/// The one scorer of a score entry's map, read by its key; a built-in's key
/// chooses the parameter struct, so the struct's own message names an
/// unknown or mistyped parameter. Any other key is kept with what is
/// written under it ([`ScoreSpec::Added`]): only the validator knows the
/// names a catalog adds.
fn scorer_entry<'de, A: MapAccess<'de>>(mut map: A) -> Result<ScoreSpec, A::Error> {
    let Some(key) = map.next_key::<String>()? else {
        return Err(de::Error::custom("a score entry names one scorer"));
    };
    let stage = match key.as_str() {
        "least-load" => ScoreStage::LeastLoad(map.next_value()?),
        "least-requests" => ScoreStage::LeastRequests(map.next_value()?),
        "fewest-sessions" => ScoreStage::FewestSessions(map.next_value()?),
        "session-affinity" => ScoreStage::SessionAffinity(map.next_value()?),
        "prefix-affinity" => ScoreStage::PrefixAffinity(map.next_value()?),
        "kv-pressure" => ScoreStage::KvPressure(map.next_value()?),
        "warmup-slice" => ScoreStage::WarmupSlice(map.next_value()?),
        "bucket" => ScoreStage::Bucket(map.next_value()?),
        "consistent-hash" => ScoreStage::ConsistentHash(map.next_value()?),
        "resident-work" => ScoreStage::ResidentWork(map.next_value()?),
        _ => {
            let Written(written) = map.next_value()?;
            return single_key(map, ScoreSpec::Added { name: key, written });
        }
    };
    single_key(map, ScoreSpec::Params(stage))
}

/// `entry`, once its map holds no second key.
fn single_key<'de, A: MapAccess<'de>>(mut map: A, entry: ScoreSpec) -> Result<ScoreSpec, A::Error> {
    if map.next_key::<de::IgnoredAny>()?.is_some() {
        return Err(de::Error::custom(
            "a score entry maps a single key to its parameters; write one entry per scorer",
        ));
    }
    Ok(entry)
}

/// One of the document's maps, each key once: a key written twice is refused
/// by name, in whatever form the document came. The map of the standard
/// library keeps the last value and says nothing, so a JSON document that
/// wrote one profile's name twice ran the second.
fn keyed_once<'de, D, V>(deserializer: D) -> Result<BTreeMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct Once<V>(PhantomData<V>);

    impl<'de, V: Deserialize<'de>> Visitor<'de> for Once<V> {
        type Value = BTreeMap<String, V>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a map")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut read = BTreeMap::new();
            while let Some(key) = map.next_key::<String>()? {
                match read.entry(key) {
                    btree_map::Entry::Vacant(place) => {
                        place.insert(map.next_value()?);
                    }
                    btree_map::Entry::Occupied(first) => {
                        return Err(de::Error::custom(format_args!(
                            "the key `{}` is written twice",
                            first.key()
                        )));
                    }
                }
            }
            Ok(read)
        }
    }

    deserializer.deserialize_map(Once(PhantomData))
}

/// What a document wrote under a scorer's name that is no built-in's, kept
/// as read for the validator. A key written twice is refused here, in the
/// words a built-in's parameters refuse it: the map of `serde_json` keeps
/// the last value and says nothing. Only the entry's own keys are asked:
/// the validator takes the weight alone, a number.
struct Written(serde_json::Value);

impl<'de> Deserialize<'de> for Written {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct WrittenVisitor;

        impl<'de> Visitor<'de> for WrittenVisitor {
            type Value = Written;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a scorer's parameters")
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Written, E> {
                Ok(Written(value.into()))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Written, E> {
                Ok(Written(value.into()))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Written, E> {
                Ok(Written(value.into()))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Written, E> {
                Ok(Written(value.into()))
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Written, E> {
                Ok(Written(value.into()))
            }

            fn visit_unit<E: de::Error>(self) -> Result<Written, E> {
                Ok(Written(serde_json::Value::Null))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Written, A::Error> {
                Deserialize::deserialize(SeqAccessDeserializer::new(seq)).map(Written)
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Written, A::Error> {
                let mut keys = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if keys.contains_key(&key) {
                        return Err(de::Error::custom(format_args!("duplicate field `{key}`")));
                    }
                    keys.insert(key, map.next_value()?);
                }
                Ok(Written(keys.into()))
            }
        }

        deserializer.deserialize_any(WrittenVisitor)
    }
}

impl<'de> Deserialize<'de> for ScoreSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ScoreVisitor;

        impl<'de> Visitor<'de> for ScoreVisitor {
            type Value = ScoreSpec;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a scorer name, or one scorer name mapped to its parameters")
            }

            fn visit_str<E: de::Error>(self, name: &str) -> Result<ScoreSpec, E> {
                Ok(ScoreSpec::Name(name.to_string()))
            }

            // The map is read by its one key, so the message names a
            // built-in's wrong parameter, never "no variant matched".
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<ScoreSpec, A::Error> {
                scorer_entry(map)
            }
        }

        deserializer.deserialize_any(ScoreVisitor)
    }
}

impl JsonSchema for ScoreSpec {
    fn schema_name() -> Cow<'static, str> {
        "ScoreSpec".into()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        // The bare form stands for the stage at its defaults, so a scorer
        // whose parameters have none is written in the parameter form only.
        let names: Vec<&str> = SCORERS
            .iter()
            .copied()
            .filter(|name| *name != "resident-work")
            .collect();
        json_schema!({
            "description": "A scorer: its bare name (the stage at its defaults), or the name mapped to its weight and parameters.",
            "anyOf": [
                { "type": "string", "enum": names },
                generator.subschema_for::<ScoreStage>()
            ]
        })
    }
}

/// A picker with its parameters, keyed by the picker's name.
#[derive(Clone, Debug, PartialEq, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum PickStage {
    Best(BestParams),
    BestWithTemperature(BestWithTemperatureParams),
}

impl PickStage {
    /// The picker's name, as the document spells it.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Best(_) => "best",
            Self::BestWithTemperature(_) => "best-with-temperature",
        }
    }
}

/// A picker reference: a bare name, or the name mapped to its parameters
/// ([`PickStage`]).
#[derive(Clone, Debug, PartialEq)]
pub enum PickSpec {
    Name(String),
    Params(PickStage),
}

impl PickSpec {
    /// The picker's name as written.
    pub fn name(&self) -> &str {
        match self {
            Self::Name(name) => name,
            Self::Params(stage) => stage.name(),
        }
    }
}

impl From<&str> for PickSpec {
    fn from(name: &str) -> Self {
        Self::Name(name.to_string())
    }
}

impl From<String> for PickSpec {
    fn from(name: String) -> Self {
        Self::Name(name)
    }
}

impl From<PickStage> for PickSpec {
    fn from(stage: PickStage) -> Self {
        Self::Params(stage)
    }
}

impl Serialize for PickSpec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Name(name) => serializer.serialize_str(name),
            Self::Params(PickStage::Best(p)) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("best", p)?;
                map.end()
            }
            Self::Params(PickStage::BestWithTemperature(p)) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("best-with-temperature", p)?;
                map.end()
            }
        }
    }
}

/// The one picker of a pick entry's map, read by its key.
fn picker_entry<'de, A: MapAccess<'de>>(mut map: A) -> Result<PickStage, A::Error> {
    const WITH_PARAMETERS: [&str; 2] = ["best", "best-with-temperature"];
    let Some(key) = map.next_key::<String>()? else {
        return Err(de::Error::custom("a pick entry names one picker"));
    };
    let stage = match key.as_str() {
        "best" => PickStage::Best(map.next_value()?),
        "best-with-temperature" => PickStage::BestWithTemperature(map.next_value()?),
        other => return Err(de::Error::unknown_variant(other, &WITH_PARAMETERS)),
    };
    if map.next_key::<de::IgnoredAny>()?.is_some() {
        return Err(de::Error::custom(
            "a pick entry maps a single key to its parameters",
        ));
    }
    Ok(stage)
}

impl<'de> Deserialize<'de> for PickSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PickVisitor;

        impl<'de> Visitor<'de> for PickVisitor {
            type Value = PickSpec;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a picker name, or one picker name mapped to its parameters")
            }

            fn visit_str<E: de::Error>(self, name: &str) -> Result<PickSpec, E> {
                Ok(PickSpec::Name(name.to_string()))
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<PickSpec, A::Error> {
                picker_entry(map).map(PickSpec::Params)
            }
        }

        deserializer.deserialize_any(PickVisitor)
    }
}

impl JsonSchema for PickSpec {
    fn schema_name() -> Cow<'static, str> {
        "PickSpec".into()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        let names: Vec<&str> = PICKERS
            .iter()
            .copied()
            .filter(|name| *name != "best-with-temperature")
            .collect();
        json_schema!({
            "description": "A picker: its bare name, or the name mapped to its parameters. `power-of-two` and `power-of-two-count-first` compare two workers drawn from those the filters left, `count-first` the fewest in flight among them: the built-in scorers score the workers compared and no other. Between two workers a normalized score is all or nothing, so several weighted scorers under a two-draw picker vote by weight where over the pool they would blend: name one scorer there, or the `best` picker over the pool.",
            "anyOf": [
                { "type": "string", "enum": names },
                generator.subschema_for::<PickStage>()
            ]
        })
    }
}

/// One profile: an ordered set of filters, weighted scorers and exactly one
/// picker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProfileSpec {
    /// The hard filters, in order. Which workers the hard filters keep does
    /// not turn on their order; a shed's reason does: a shed names the
    /// first filter, in this order, that left nobody.
    #[serde(default)]
    pub filter: Vec<String>,
    /// The scorers, each a name or a name with its weight and parameters.
    #[serde(default)]
    pub score: Vec<ScoreSpec>,
    /// The picker.
    pub pick: PickSpec,
}

/// A closed range with one or both bounds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(rename = "Bounds_for_{T}")]
pub struct Bounds<T> {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<T>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<T>,
}

/// A request header among `view.headers`: present, or holding one of the
/// values.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HeaderMatch {
    /// The header's name, lowercase, listed in `view.headers`.
    pub name: String,
    /// The values that match; empty matches the header's presence.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<String>,
}

/// What a binding matches on. Bindings are ordered; the first match wins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct Match {
    /// The model id after aliasing: an exact id, `prefix*`, `*suffix` or
    /// `*`.
    pub model: String,
    /// The phases the binding covers; empty covers every phase. A plan with
    /// several sides names every side's phase or none.
    pub phase: Vec<Phase>,
    /// The tenant the request resolved to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// The priority band the request carries, as a closed range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<Bounds<i32>>,
    /// A request header, present or holding one of the values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<HeaderMatch>,
    /// The request's prompt token count, as a closed range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<Bounds<u32>>,
    /// Registration labels the binding is restricted to.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
}

impl Default for Match {
    fn default() -> Self {
        Self {
            model: "*".to_string(),
            phase: Vec::new(),
            tenant: None,
            priority: None,
            header: None,
            prompt_tokens: None,
            labels: Vec::new(),
        }
    }
}

/// The queued requests, summed over an engine's ranks, at or above which
/// its rows are vetoed where a binding's overload block names no count.
pub const OVERLOAD_WAITING_REQUESTS: u32 = 8;

/// The mean KV usage over an engine's ranks at or above which its rows are
/// vetoed where a binding's overload block names no usage.
pub const OVERLOAD_TOKEN_USAGE: f32 = 0.8;

/// The overload veto's defaults for the workers of the pools decided under
/// the binding, as the binding's `overload` field says: evaluated per load
/// report by the host and latched, never per request; a worker's own
/// registration thresholds resolve over these.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct OverloadSpec {
    /// Whether the host's thresholds veto at all (a worker's own block still
    /// does).
    pub enabled: bool,
    /// Queued requests, summed over an engine's ranks, at or above which a
    /// worker is vetoed; at least 1 (default 8).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub waiting_requests: Option<u32>,
    /// Mean KV usage over an engine's ranks at or above which a worker is
    /// vetoed; above 0 and at most 1 (default 0.8).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(extend("exclusiveMinimum" = 0), range(max = 1.0))]
    pub token_usage: Option<f32>,
    /// Shed with a 503 when every usable worker is vetoed, instead of
    /// falling back to the least-vetoed one. The same word decides what a
    /// session key does under consistent hashing while its own worker is
    /// vetoed: set, the key moves to the ring's next worker; unset, the
    /// key stays on its worker over the veto.
    pub shed_when_all_vetoed: bool,
}

impl Default for OverloadSpec {
    fn default() -> Self {
        Self {
            enabled: true,
            waiting_requests: None,
            token_usage: None,
            shed_when_all_vetoed: false,
        }
    }
}

/// How a session key seen for the first time picks its worker. Under the
/// `manual` picker the same assignment places a request with no key, which
/// is not pinned, and a `manual` profile bound with no sessions rule places
/// every request by `min_load`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Assignment {
    /// A random worker.
    Random,
    /// The worker with the fewest running requests.
    MinLoad,
    /// The worker with the fewest active keys.
    MinGroup,
    /// The profile's own pick, then pinned. The `manual` picker, having no
    /// profile to delegate to, takes `min_load`.
    #[default]
    Delegate,
}

/// Which of a key's two pins leads once both are routable again: the first
/// worker the key was placed on (the manual policy's order) or the latest
/// one it was dispatched to (the sticky override's under `delegate`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Recovery {
    #[default]
    Original,
    Latest,
}

/// Sticky sessions on the binding: a keyed request stays on the worker its
/// key was first placed on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionsSpec {
    /// The header whose value is the session key; listed in `view.headers`.
    pub sticky_header: String,
    /// The headers looked up in order for the key, the sticky header among
    /// them; empty means the sticky header alone. Each listed in
    /// `view.headers`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headers: Vec<String>,
    /// How long an idle key stays pinned, in seconds; above zero (default
    /// 14400), `4294967295` for ever.
    #[serde(default = "default_ttl_s")]
    #[schemars(range(min = 1))]
    pub ttl_s: u32,
    /// The sweep cadence of idle keys, in seconds; above zero (default 120),
    /// `4294967295` for never, allowed under a `ttl_s` for ever alone.
    #[serde(default = "default_sweep_s")]
    #[schemars(range(min = 1))]
    pub sweep_s: u32,
    /// How a first-seen key picks its worker (default `delegate`); under
    /// the `manual` picker, a request with no key as well.
    #[serde(default)]
    pub assignment: Assignment,
    /// How many requests of one key a worker carries before the next steps
    /// aside; above zero (default 2), `65535` for no cap.
    #[serde(default = "default_inflight_cap")]
    #[schemars(range(min = 1))]
    pub inflight_cap: u16,
    /// Issue a session id to a request that carries none (default false;
    /// not in this build: set, it is refused by name).
    #[serde(default)]
    pub issue_session_id: bool,
    /// Which pin leads once both of a key's pins are routable (default
    /// `original`).
    #[serde(default)]
    pub recovery: Recovery,
}

fn default_ttl_s() -> u32 {
    14_400
}

fn default_sweep_s() -> u32 {
    120
}

fn default_inflight_cap() -> u16 {
    2
}

/// A second profile deciding a share of the binding's requests, the
/// deciding profile recorded on the lease. Exactly one of `percent`,
/// `tenants` and `header` chooses the share.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanarySpec {
    /// The profile deciding the canary's share.
    pub profile: String,
    /// A share of requests, 0 to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1, max = 100))]
    pub percent: Option<u8>,
    /// The tenants whose requests the canary decides.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tenants: Vec<String>,
    /// A request header that selects the canary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<HeaderMatch>,
}

/// Whether a request may name its profile through the override header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Override {
    /// The override header is honoured.
    Allowed,
    /// The override header is ignored.
    #[default]
    Locked,
}

/// Fairness admission on the binding; reserved in v1 and off.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct AdmissionSpec {
    pub enabled: bool,
}

/// The plan a binding runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum PlanName {
    /// One profile over one pool.
    #[default]
    Single,
    /// The prefill profile, then the decode profile over the chosen
    /// prefill's partners; both sides named.
    PdPair,
    /// The encode profile once per encode item, then the pair; all three
    /// sides named.
    Epd,
}

impl PlanName {
    /// The plan's name, as the document spells it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::PdPair => "pd-pair",
            Self::Epd => "epd",
        }
    }

    /// The sides the plan's `profiles` map names, in the plan's order.
    fn sides(self) -> &'static [&'static str] {
        match self {
            Self::Single => &["single"],
            Self::PdPair => &["prefill", "decode"],
            Self::Epd => &["encode", "prefill", "decode"],
        }
    }

    /// The phases a multi-sided plan's phase list names, all or none.
    fn phases(self) -> &'static [(Phase, &'static str)] {
        match self {
            Self::Single => &[],
            Self::PdPair => &[(Phase::Prefill, "prefill"), (Phase::Decode, "decode")],
            Self::Epd => &[
                (Phase::Encode, "encode"),
                (Phase::Prefill, "prefill"),
                (Phase::Decode, "decode"),
            ],
        }
    }
}

/// A binding: the requests it matches, the profile or plan they resolve to,
/// and the rules around the decision.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct BindingSpec {
    #[serde(rename = "match")]
    pub matcher: Match,
    /// The profile of a single decision.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// `single` (the default), `pd-pair` or `epd`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<PlanName>,
    /// The profiles per side: `single`, or every side of the plan
    /// (`prefill` and `decode`; `encode` too under `epd`), each named.
    #[serde(
        deserialize_with = "keyed_once",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub profiles: BTreeMap<String, String>,
    /// The overload veto's defaults for the workers of the pools decided
    /// under the binding: the pools whose model and phase it is the first
    /// to match. A decode pool of a model whose prefill lookup resolves to
    /// a pair binding (`pd-pair` or `epd`) is decided under that binding,
    /// so its block judges the pool: a decode-phase binding's block is not
    /// read for it, and a pair binding with its block switched off leaves
    /// the pool unjudged. Where a binding of another plan takes the
    /// model's prefill lookup first, no pair decision runs for the model
    /// and its decode pool is the decode phase's own.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overload: Option<OverloadSpec>,
    /// Sticky sessions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sessions: Option<SessionsSpec>,
    /// A profile evaluated after the live one on the same view, never
    /// dispatched, counted as disagreement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow: Option<String>,
    /// A second profile deciding a share of the requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canary: Option<CanarySpec>,
    /// Whether a request may name its profile (default `locked`).
    #[serde(rename = "override", skip_serializing_if = "is_default")]
    pub profile_override: Override,
    /// Fairness admission; reserved in v1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub admission: Option<AdmissionSpec>,
}

/// The policy specification, version `smg.policy/v1`: what the host builds
/// into the view, the per-model index settings, the profiles with their
/// stages, the plans' rules and the ordered bindings. An unknown field at
/// any level is refused by name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PolicySpec {
    /// `smg.policy/v1`.
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    /// The document's generation: written by the operator, or by the apply
    /// that persisted it.
    #[serde(default)]
    pub generation: u64,
    /// A `kind` line is read only to refuse it by name: the document has
    /// no kind (its forms are a file and the API). Never in the schema, and
    /// private so that no code writes one: a document is built by parsing,
    /// by a preset or by [`PolicySpec::new`].
    #[serde(default, skip_serializing)]
    #[schemars(skip)]
    kind: Option<String>,
    /// What the host builds into every decision's view.
    #[serde(default, skip_serializing_if = "is_default")]
    pub view: ViewSpec,
    /// The prefix-index settings per model glob. Where two keys match one
    /// model the most specific wins: an exact id, then the longer of a
    /// `prefix*` and a `*suffix`, then `*`; a `prefix*` and a `*suffix` of
    /// one length are refused as a pair.
    #[serde(
        default,
        deserialize_with = "keyed_once",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub index: BTreeMap<String, IndexSpec>,
    /// The profiles by name.
    #[serde(deserialize_with = "keyed_once")]
    pub profiles: BTreeMap<String, ProfileSpec>,
    /// The plans' rules.
    #[serde(default, skip_serializing_if = "is_default")]
    pub plans: PlansSpec,
    /// The bindings in order; the first match wins. Empty binds the first
    /// profile to everything.
    #[serde(default)]
    pub bindings: Vec<BindingSpec>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SpecError {
    #[error("the document carries kind {0:?}: a policy spec has no kind field; its forms are a file and the API, never a custom resource")]
    Kind(String),
    #[error("apiVersion {0:?} is not {API_VERSION}")]
    ApiVersion(String),
    #[error("profile {profile:?}: unknown filter {name:?}")]
    UnknownFilter { profile: String, name: String },
    #[error("profile {profile:?}: unknown scorer {name:?}")]
    UnknownScorer { profile: String, name: String },
    #[error("profile {profile:?}: unknown picker {name:?}")]
    UnknownPicker { profile: String, name: String },
    #[error("profile {profile:?}: picker {picker:?} needs a scorer to compare by")]
    PickerNeedsScorer { profile: String, picker: String },
    #[error("profile {profile:?}: picker {picker:?} leaves the placement to the binding's sessions rule and runs no scorer; remove `score`")]
    PickerRunsNoScorer { profile: String, picker: String },
    #[error("binding {index}: unknown profile {name:?}")]
    UnknownProfile { index: usize, name: String },
    #[error("binding {index} names no profile")]
    NoProfile { index: usize },
    #[error("binding {index}: the {plan} plan names no {side} profile")]
    SideUnnamed {
        index: usize,
        plan: &'static str,
        side: &'static str,
    },
    #[error("binding {index}: the {plan} plan takes its profiles per side, not one profile")]
    ProfileUnderPlan { index: usize, plan: &'static str },
    #[error("binding {index}: names its profile twice, as profile and under profiles")]
    ProfileTwice { index: usize },
    #[error("the document names {profiles} profiles and no bindings: with more than one profile the bindings name which serves what")]
    NoBindings { profiles: usize },
    #[error(
        "index keys {first:?} and {second:?} match the same models and neither is more specific"
    )]
    IndexOverlap { first: String, second: String },
    #[error("the document names {count} profiles; at most {max}")]
    TooManyProfiles { count: usize, max: u16 },
    #[error("view.headers names {count} headers; at most {max}")]
    TooManyHeaders { count: usize, max: usize },
    #[error("{path} lists {count} entries; at most {max}")]
    TooManyListed {
        path: &'static str,
        count: usize,
        max: usize,
    },
    #[error("binding {index}: profiles names an unknown side {side:?}")]
    UnknownSide { index: usize, side: String },
    #[error("binding {index}: the {plan} plan has no {side:?} side")]
    SideNotInPlan {
        index: usize,
        plan: &'static str,
        side: String,
    },
    #[error(
        "binding {index}: the {plan} plan's phase list leaves out {side}; name every side or none"
    )]
    PlanPhases {
        index: usize,
        plan: &'static str,
        side: &'static str,
    },
    #[error("binding {index}: model {model:?} is not an id, `prefix*`, `*suffix` or `*`")]
    BadModelGlob { index: usize, model: String },
    #[error("{path} = {value} is not {reason}")]
    BadField {
        path: String,
        value: String,
        reason: &'static str,
    },
    #[error("{path}: {what} is not in this build")]
    NotInBuild { path: String, what: &'static str },
    #[error("{path}: header {name:?} is on the deny-list")]
    DeniedHeader { path: String, name: String },
    #[error("{path}: header {name:?} is not in view.headers")]
    HeaderNotAllowed { path: String, name: String },
    #[error("{path}: {name:?} appears twice")]
    Duplicate { path: String, name: String },
    #[error("{path}: unknown profile {name:?}")]
    UnknownProfileAt { path: String, name: String },
    #[error("{path}: names {count} of percent, tenants and header; name exactly one")]
    CanaryForm { path: String, count: usize },
    #[error("index {model:?}: a hash index needs view.boundaries")]
    HashIndexNeedsBoundaries { model: String },
    #[error("{path}: {reason}")]
    Reserved { path: String, reason: &'static str },
    #[error("binding {index}: sessions {name} is not {reason}")]
    BadSessions {
        index: usize,
        name: &'static str,
        reason: &'static str,
    },
    #[error("no profiles")]
    Empty,
    #[error("unknown preset {0:?}")]
    UnknownPreset(String),
    #[error("the document has no canonical form to digest: {0}")]
    NoCanonicalForm(String),
}

impl SpecError {
    /// The field of the document the error is about, as a path from the
    /// document's root in the document's own names (`profiles.default.pick`,
    /// `bindings[2].match.model`): what a `SpecIssue` carries. It names the
    /// field to edit: an unknown stage is at the list that names it, a
    /// binding's error at the binding's field, and a path the validator
    /// built is handed back as it stands. Empty where the error is about no
    /// one field of a document.
    pub fn field_path(&self) -> String {
        match self {
            Self::Kind(_) => "kind".to_owned(),
            Self::ApiVersion(_) => "apiVersion".to_owned(),
            Self::UnknownFilter { profile, .. } => format!("profiles.{profile}.filter"),
            Self::UnknownScorer { profile, .. } | Self::PickerRunsNoScorer { profile, .. } => {
                format!("profiles.{profile}.score")
            }
            Self::UnknownPicker { profile, .. } | Self::PickerNeedsScorer { profile, .. } => {
                format!("profiles.{profile}.pick")
            }
            Self::UnknownProfile { index, .. } => format!("bindings[{index}]"),
            Self::NoProfile { index }
            | Self::ProfileUnderPlan { index, .. }
            | Self::ProfileTwice { index } => format!("bindings[{index}].profile"),
            Self::SideUnnamed { index, side, .. } => format!("bindings[{index}].profiles.{side}"),
            Self::UnknownSide { index, side } | Self::SideNotInPlan { index, side, .. } => {
                format!("bindings[{index}].profiles.{side}")
            }
            Self::PlanPhases { index, .. } => format!("bindings[{index}].match.phase"),
            Self::BadModelGlob { index, .. } => format!("bindings[{index}].match.model"),
            // The one sessions refusal that is about the binding's plan.
            Self::BadSessions {
                index,
                name: "plan",
                ..
            } => format!("bindings[{index}].plan"),
            Self::BadSessions { index, name, .. } => format!("bindings[{index}].sessions.{name}"),
            Self::NoBindings { .. } => "bindings".to_owned(),
            Self::IndexOverlap { first, .. } => format!("index.{first}"),
            Self::HashIndexNeedsBoundaries { model } => format!("index.{model}.kind"),
            Self::TooManyProfiles { .. } | Self::Empty => "profiles".to_owned(),
            Self::TooManyHeaders { .. } => "view.headers".to_owned(),
            Self::TooManyListed { path, .. } => (*path).to_owned(),
            Self::BadField { path, .. }
            | Self::NotInBuild { path, .. }
            | Self::DeniedHeader { path, .. }
            | Self::HeaderNotAllowed { path, .. }
            | Self::Duplicate { path, .. }
            | Self::UnknownProfileAt { path, .. }
            | Self::CanaryForm { path, .. }
            | Self::Reserved { path, .. } => path.clone(),
            Self::UnknownPreset(_) | Self::NoCanonicalForm(_) => String::new(),
        }
    }
}

/// A field path under construction, for the messages.
struct At(String);

impl At {
    fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    fn field(&self, name: &str) -> String {
        format!("{}.{name}", self.0)
    }

    fn sub(&self, name: &str) -> Self {
        Self(self.field(name))
    }

    fn bad(&self, name: &str, value: impl fmt::Display, reason: &'static str) -> SpecError {
        SpecError::BadField {
            path: self.field(name),
            value: value.to_string(),
            reason,
        }
    }

    /// A written number passes `ok`, or is refused for `reason`.
    fn number<T: Copy + fmt::Display>(
        &self,
        name: &str,
        value: Option<T>,
        ok: fn(T) -> bool,
        reason: &'static str,
    ) -> Result<(), SpecError> {
        match value {
            Some(v) if !ok(v) => Err(self.bad(name, v, reason)),
            _ => Ok(()),
        }
    }

    /// A written count is above zero.
    fn nonzero<T>(&self, name: &str, value: Option<T>) -> Result<(), SpecError>
    where
        T: Copy + Default + PartialEq + fmt::Display,
    {
        match value {
            Some(v) if v == T::default() => Err(self.bad(name, v, "above zero")),
            _ => Ok(()),
        }
    }
}

fn above_zero(v: f32) -> bool {
    v.is_finite() && v > 0.0
}

fn at_or_above_zero(v: f32) -> bool {
    v.is_finite() && v >= 0.0
}

fn unit_interval(v: f32) -> bool {
    v.is_finite() && (0.0..=1.0).contains(&v)
}

fn at_least_one<T: Into<f64>>(v: T) -> bool {
    let v: f64 = v.into();
    v.is_finite() && v >= 1.0
}

fn unit_fraction(v: f32) -> bool {
    v.is_finite() && v > 0.0 && v <= 1.0
}

/// The matcher understands an id, `prefix*`, `*suffix` and `*`; any other
/// star would match only its literal self.
fn glob_ok(model: &str) -> bool {
    let stars = model.matches('*').count();
    !model.is_empty()
        && (stars == 0 || (stars == 1 && (model.starts_with('*') || model.ends_with('*'))))
}

/// A lowercase header name outside the deny-lists.
fn check_header_name(name: &str, path: String, denied: &[&str]) -> Result<(), SpecError> {
    let token =
        |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit() || b"!#$%&'*+-.^_`|~".contains(&b);
    if name.is_empty() || !name.bytes().all(token) {
        return Err(SpecError::BadField {
            path,
            value: name.to_string(),
            reason: "a lowercase header name",
        });
    }
    if DENIED_HEADERS.contains(&name) || denied.iter().any(|d| d.eq_ignore_ascii_case(name)) {
        return Err(SpecError::DeniedHeader {
            path,
            name: name.to_string(),
        });
    }
    Ok(())
}

/// Each name once.
fn check_unique<'a>(
    names: impl IntoIterator<Item = &'a String>,
    path: &str,
) -> Result<(), SpecError> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name.as_str()) {
            return Err(SpecError::Duplicate {
                path: path.to_string(),
                name: name.clone(),
            });
        }
    }
    Ok(())
}

fn check_view(view: &ViewSpec, denied: &[&str]) -> Result<(), SpecError> {
    let at = At::new("view");
    if view.headers.len() > MAX_HEADERS {
        return Err(SpecError::TooManyHeaders {
            count: view.headers.len(),
            max: MAX_HEADERS,
        });
    }
    for (i, header) in view.headers.iter().enumerate() {
        check_header_name(header, format!("view.headers[{i}]"), denied)?;
    }
    check_unique(&view.headers, "view.headers")?;
    at.nonzero("block_size", view.block_size)?;
    if view.boundaries.first() == Some(&0) {
        return Err(SpecError::BadField {
            path: "view.boundaries[0]".to_string(),
            value: "0".to_string(),
            reason: "above zero",
        });
    }
    for (i, pair) in view.boundaries.windows(2).enumerate() {
        if let [low, high] = pair {
            if low >= high {
                return Err(SpecError::BadField {
                    path: format!("view.boundaries[{}]", i + 1),
                    value: high.to_string(),
                    reason: "above the boundary before it",
                });
            }
        }
    }
    at.nonzero("max_block_hashes", Some(view.max_block_hashes))?;
    if view.max_block_hashes > MAX_BLOCK_HASHES {
        return Err(at.bad(
            "max_block_hashes",
            view.max_block_hashes,
            "at most 8190, the hashes one Decide frame carries",
        ));
    }
    check_list(&view.labels, "view.labels", MAX_LABELS)?;
    for (i, label) in view.labels.iter().enumerate() {
        let (key, value) = selector(label);
        if key.is_empty() || value == Some("") {
            return Err(SpecError::BadField {
                path: format!("view.labels[{i}]"),
                value: format!("{label:?}"),
                reason: "a label key or `key=value` with no empty side",
            });
        }
    }
    check_list(&view.zones, "view.zones", MAX_ZONES)?;
    check_list(&view.engines, "view.engines", MAX_ENGINES)
}

/// A label selector's two sides: the key, and the value after the first
/// `=` when it names one.
fn selector(label: &str) -> (&str, Option<&str>) {
    label
        .split_once('=')
        .map_or((label, None), |(key, value)| (key, Some(value)))
}

/// A list the view declares: at most `max` entries, none empty, each once.
fn check_list(list: &[String], path: &'static str, max: usize) -> Result<(), SpecError> {
    if list.len() > max {
        return Err(SpecError::TooManyListed {
            path,
            count: list.len(),
            max,
        });
    }
    if let Some(i) = list.iter().position(String::is_empty) {
        return Err(SpecError::BadField {
            path: format!("{path}[{i}]"),
            value: "\"\"".to_string(),
            reason: "a name",
        });
    }
    check_unique(list, path)
}

fn check_index(index: &BTreeMap<String, IndexSpec>, view: &ViewSpec) -> Result<(), SpecError> {
    for (model, spec) in index {
        // The key's path is built at a refusal alone: a key that passes
        // formats nothing.
        let at = || At::new(format!("index.{model}"));
        if !glob_ok(model) {
            return Err(SpecError::BadField {
                path: at().0,
                value: model.clone(),
                reason: "an id, `prefix*`, `*suffix` or `*`",
            });
        }
        let counts = [
            ("ttl_s", u64::from(spec.ttl_s)),
            ("max_size", spec.max_size),
            ("sweep_s", u64::from(spec.sweep_s)),
        ];
        if let Some((name, zero)) = counts.into_iter().find(|(_, count)| *count == 0) {
            return Err(at().bad(name, zero, "above zero"));
        }
        if spec.kind == IndexKind::Hash && view.boundaries.is_empty() {
            return Err(SpecError::HashIndexNeedsBoundaries {
                model: model.clone(),
            });
        }
    }
    match overlap(index.keys()) {
        Some((first, second)) => Err(SpecError::IndexOverlap {
            first: first.clone(),
            second: second.clone(),
        }),
        None => Ok(()),
    }
}

/// The first two keys that overlap, in the keys' order: the first key with
/// a partner after it, and the first such partner. A `prefix*` and a
/// `*suffix` of one literal length overlap: they match a common model and
/// neither is the more specific, so the pair is refused.
///
/// Each key is read once, whatever their number. Only the first `prefix*`
/// and the first `*suffix` of a literal length can be that pair, so those
/// are what the pass keeps.
fn overlap<K: AsRef<str>>(keys: impl IntoIterator<Item = K>) -> Option<(K, K)> {
    // By literal length: the first key of each side, with its place.
    let mut firsts: BTreeMap<usize, [Option<(usize, K)>; 2]> = BTreeMap::new();
    for (place, key) in keys.into_iter().enumerate() {
        let Some((which, literal)) = side(key.as_ref()) else {
            continue;
        };
        let first = &mut firsts.entry(literal).or_insert([None, None])[which];
        if first.is_none() {
            *first = Some((place, key));
        }
    }
    firsts
        .into_values()
        .filter_map(|sides| match sides {
            [Some(prefix), Some(suffix)] if prefix.0 < suffix.0 => Some((prefix, suffix)),
            [Some(prefix), Some(suffix)] => Some((suffix, prefix)),
            _ => None,
        })
        .min_by_key(|(first, _)| first.0)
        .map(|((_, first), (_, second))| (first, second))
}

/// The side of a model a glob holds, 0 for `prefix*` and 1 for `*suffix`,
/// with its literal length. An id and `*` hold none.
fn side(glob: &str) -> Option<(usize, usize)> {
    match (glob.strip_suffix('*'), glob.strip_prefix('*')) {
        _ if glob == "*" => None,
        (Some(prefix), _) => Some((0, prefix.len())),
        (_, Some(suffix)) => Some((1, suffix.len())),
        _ => None,
    }
}

/// Whether a model glob (`*`, `prefix*`, `*suffix` or an id) matches a model.
fn glob_matches(pattern: &str, model: &str) -> bool {
    match (pattern.strip_suffix('*'), pattern.strip_prefix('*')) {
        _ if pattern == "*" => true,
        (Some(prefix), _) => model.starts_with(prefix),
        (_, Some(suffix)) => model.ends_with(suffix),
        _ => pattern == model,
    }
}

/// How specific a glob is: an id, then a pattern by its literal length, then
/// `*`.
fn specificity(pattern: &str) -> (u8, usize) {
    if pattern == "*" {
        (0, 0)
    } else if pattern.contains('*') {
        (1, pattern.len() - 1)
    } else {
        (2, pattern.len())
    }
}

/// A stage's parameters, refused by name: the weight is finite and above
/// zero; each parameter is in its range whether or not this build carries
/// the stage, so a document is well-formed before it is current.
fn check_stage(stage: &ScoreStage, at: &At) -> Result<(), SpecError> {
    if !above_zero(stage.weight()) {
        return Err(at.bad("weight", stage.weight(), "a finite number above zero"));
    }
    match stage {
        ScoreStage::LeastLoad(p) => {
            at.number(
                "kv_pressure_weight",
                p.kv_pressure_weight,
                at_or_above_zero,
                "a finite number at or above zero",
            )?;
            at.nonzero("mean_prefill_tokens", p.mean_prefill_tokens)?;
            at.number(
                "default_throughput",
                p.default_throughput,
                above_zero,
                "a finite number above zero",
            )
        }
        // The resident-work price has no bound: zero compares the tokens
        // alone.
        ScoreStage::LeastRequests(_)
        | ScoreStage::FewestSessions(_)
        | ScoreStage::SessionAffinity(_)
        | ScoreStage::ResidentWork(_) => Ok(()),
        ScoreStage::PrefixAffinity(p) => {
            at.number("threshold", p.threshold, unit_interval, "between 0 and 1")?;
            at.number(
                "decay",
                p.decay,
                at_or_above_zero,
                "a finite number at or above zero",
            )?;
            match &p.spill {
                Some(spill) => at.sub("spill").number(
                    "rel",
                    spill.rel,
                    at_least_one,
                    "a finite number at least 1",
                ),
                None => Ok(()),
            }
        }
        ScoreStage::KvPressure(p) => {
            at.number("spread", p.spread, above_zero, "a finite number above zero")?;
            at.number(
                "ceiling",
                p.ceiling,
                above_zero,
                "a finite number above zero",
            )
        }
        ScoreStage::WarmupSlice(p) => {
            at.number("share", p.share, unit_interval, "between 0 and 1")?;
            at.number("thin_ratio", p.thin_ratio, unit_interval, "between 0 and 1")?;
            at.number("miss_share", p.miss_share, unit_interval, "between 0 and 1")
        }
        ScoreStage::Bucket(p) => {
            at.number(
                "rel_threshold",
                p.rel_threshold,
                at_least_one,
                "a finite number at least 1",
            )?;
            at.nonzero("adjust_interval_s", p.adjust_interval_s)
        }
        ScoreStage::ConsistentHash(p) => {
            at.nonzero("prefix_tokens", p.prefix_tokens)?;
            match &p.bounded_load {
                Some(bounded) => at.sub("bounded_load").number(
                    "factor",
                    bounded.factor,
                    at_least_one,
                    "a finite number at least 1",
                ),
                None => Ok(()),
            }
        }
    }
}

/// A stage of this build, or the name of what the build lacks.
enum Resolved<T> {
    Built(T),
    NotInBuild(&'static str),
}

/// The first thing this build lacks on the validator's walk (profiles in
/// name order, then bindings in their order), kept until the whole
/// document's shape has been read: a shape error outranks it.
#[derive(Default)]
struct Deferred(Option<SpecError>);

impl Deferred {
    fn note(&mut self, path: String, what: &'static str) {
        if self.0.is_none() {
            self.0 = Some(SpecError::NotInBuild { path, what });
        }
    }
}

/// What a prefix depth breaks against the view that carries a request's
/// block hashes: `under` when it holds no whole block of the view's block
/// size, `over` when a request carries no block that deep.
fn depth_edge(
    depth: u32,
    gates: Gates,
    view: &ViewSpec,
    under: &'static str,
    over: &'static str,
) -> Option<&'static str> {
    match depth / gates.block_size {
        0 => Some(under),
        blocks if blocks > view.max_block_hashes => Some(over),
        _ => None,
    }
}

/// A consistent-hash stage's prefix depths against the view. A key is the
/// last whole block at or under its depth, so `prefix_tokens` and each of
/// the view's boundaries hold at least one block and no more blocks than a
/// request carries.
fn check_prefix_depth(
    stage: &ScoreStage,
    gates: Gates,
    view: &ViewSpec,
    at: &At,
) -> Result<(), SpecError> {
    let ScoreStage::ConsistentHash(ConsistentHashParams {
        prefix_tokens: Some(tokens),
        ..
    }) = stage
    else {
        return Ok(());
    };
    if let Some(reason) = depth_edge(
        *tokens,
        gates,
        view,
        "at least one block of view.block_size",
        "at most view.max_block_hashes blocks of view.block_size",
    ) {
        return Err(at.bad("prefix_tokens", tokens, reason));
    }
    for (i, &boundary) in view.boundaries.iter().enumerate() {
        if let Some(reason) = depth_edge(
            boundary,
            gates,
            view,
            "at least one block of view.block_size under a consistent-hash prefix key",
            "at most view.max_block_hashes blocks of view.block_size under a consistent-hash prefix key",
        ) {
            return Err(SpecError::BadField {
                path: format!("view.boundaries[{i}]"),
                value: boundary.to_string(),
                reason,
            });
        }
    }
    Ok(())
}

fn built_scorer(scorer: impl Scorer + 'static) -> Resolved<Box<dyn Scorer>> {
    Resolved::Built(Box::new(scorer))
}

/// The gates a profile's affinity stages share: the spill gate of its
/// `prefix-affinity` entry, the fleet gate of its `kv-pressure` entry and
/// the view's block size.
fn gates_of(p: &ProfileSpec, view: &ViewSpec) -> Gates {
    let mut affinity = None;
    let mut kv = None;
    for entry in &p.score {
        match entry {
            ScoreSpec::Params(ScoreStage::PrefixAffinity(params)) => affinity = Some(params),
            ScoreSpec::Params(ScoreStage::KvPressure(params)) => kv = Some(params),
            _ => {}
        }
    }
    Gates::of(affinity, kv, view.block_size)
}

/// The stage's scorer. `boundaries` are the view's, where a prefix ring keys.
fn scorer_of(stage: &ScoreStage, gates: Gates, boundaries: &[u32]) -> Resolved<Box<dyn Scorer>> {
    match stage {
        ScoreStage::LeastLoad(p) => built_scorer(LeastLoad::new(
            p.kv_pressure_weight.unwrap_or(0.15),
            p.mean_prefill_tokens.unwrap_or(DEFAULT_MEAN_PREFILL_TOKENS),
            p.default_throughput.unwrap_or(2000.0),
        )),
        ScoreStage::LeastRequests(_) => built_scorer(LeastRequests),
        ScoreStage::FewestSessions(_) => built_scorer(FewestSessions),
        ScoreStage::SessionAffinity(_) => built_scorer(SessionAffinity),
        ScoreStage::PrefixAffinity(p) if p.decay.is_some_and(|decay| decay > 0.0) => {
            Resolved::NotInBuild("the prefix-affinity decay")
        }
        ScoreStage::PrefixAffinity(p) if p.accounting_ttl_ms.is_some_and(|ttl| ttl > 0) => {
            Resolved::NotInBuild("the prefix-affinity accounting")
        }
        ScoreStage::PrefixAffinity(p)
            if p.spill
                .as_ref()
                .is_some_and(|spill| spill.uncached_credit == Some(true)) =>
        {
            Resolved::NotInBuild("the prefix-affinity uncached credit")
        }
        ScoreStage::PrefixAffinity(p) => built_scorer(PrefixAffinity::new(p, gates)),
        ScoreStage::KvPressure(_) => built_scorer(KvPressure),
        ScoreStage::WarmupSlice(p) => built_scorer(WarmupSlice::new(p, gates)),
        ScoreStage::Bucket(p) => built_scorer(Bucket::new(p)),
        ScoreStage::ConsistentHash(p) => {
            let ring = ConsistentHash::new(p.bounded_load.as_ref());
            built_scorer(match p.prefix_tokens {
                Some(tokens) => ring.keyed_by_prefix(tokens, gates, boundaries),
                None => ring,
            })
        }
        ScoreStage::ResidentWork(p) => built_scorer(ResidentWork::new(p.request_tokens)),
    }
}

fn picker_of(
    pick: &PickSpec,
    profile: &str,
    at: &At,
    catalog: &Catalog,
) -> Result<Resolved<Box<dyn Picker>>, SpecError> {
    Ok(match pick {
        PickSpec::Name(name) => match catalog.find_picker(name) {
            Some(Found::Built(picker)) => Resolved::Built(picker),
            // The one picker with parameters, named without them.
            Some(Found::Typed) => {
                return Err(at.bad(
                    "pick",
                    name,
                    "a picker with its temperature: { best-with-temperature: { temperature } }",
                ))
            }
            None => {
                return Err(SpecError::UnknownPicker {
                    profile: profile.to_string(),
                    name: name.clone(),
                })
            }
        },
        PickSpec::Params(PickStage::Best(p)) => {
            if !(1..=MAX_SAMPLE).contains(&p.sample) {
                return Err(at.sub("pick").sub("best").bad(
                    "sample",
                    p.sample,
                    "a whole number from 1 to 64",
                ));
            }
            Resolved::Built(Box::new(SampledBest::new(p.sample)))
        }
        PickSpec::Params(PickStage::BestWithTemperature(p)) => {
            if !at_or_above_zero(p.temperature) {
                return Err(at.sub("pick").sub("best-with-temperature").bad(
                    "temperature",
                    p.temperature,
                    "a finite number at or above zero",
                ));
            }
            Resolved::NotInBuild("the best-with-temperature picker")
        }
    })
}

/// What a score entry stands for.
enum Scored {
    /// A built-in, by its parameters: the written ones, or a bare name's
    /// defaults.
    Typed(ScoreStage),
    /// A scorer the catalog built, with its weight.
    Built(Weighted),
}

/// The stage a score entry stands for. The catalog is asked for the name
/// first: a name it does not hold is refused as unknown whatever is written
/// under it, so a misspelt built-in reads its name and not a key it carries.
fn scored(
    entry: &ScoreSpec,
    profile: &str,
    at: &At,
    catalog: &Catalog,
) -> Result<Scored, SpecError> {
    let name = entry.name();
    let Some(found) = catalog.find_scorer(name) else {
        return Err(SpecError::UnknownScorer {
            profile: profile.to_string(),
            name: name.to_string(),
        });
    };
    match (entry, found) {
        (ScoreSpec::Params(stage), _) => Ok(Scored::Typed(stage.clone())),
        (ScoreSpec::Name(_), Found::Typed) => match ScoreStage::by_name(name) {
            Some(stage) => Ok(Scored::Typed(stage)),
            // The one scorer whose parameters have no default.
            None => Err(SpecError::BadField {
                path: at.0.clone(),
                value: name.to_string(),
                reason: "a scorer with its request_tokens: { resident-work: { request_tokens } }",
            }),
        },
        (ScoreSpec::Name(_), Found::Built(scorer)) => Ok(Scored::Built(Weighted {
            scorer,
            weight: 1.0,
        })),
        (ScoreSpec::Added { written, .. }, Found::Built(scorer)) => Ok(Scored::Built(Weighted {
            scorer,
            weight: added_weight(written, &at.sub(name))?,
        })),
        // Only a document built in code: the parser reads a built-in's key
        // into its parameter struct.
        (ScoreSpec::Added { .. }, Found::Typed) => Err(SpecError::BadField {
            path: at.0.clone(),
            value: name.to_string(),
            reason: "a scorer added in code: a built-in's parameters are typed",
        }),
    }
}

/// The weight a document gives a scorer added in code: `{ weight }` and no
/// other key, each refused by name; one when the map is empty.
fn added_weight(written: &serde_json::Value, at: &At) -> Result<f32, SpecError> {
    let Some(keys) = written.as_object() else {
        return Err(SpecError::BadField {
            path: at.0.clone(),
            value: written.to_string(),
            reason: "a map of the scorer's weight: { weight }",
        });
    };
    let mut weight = 1.0;
    for (key, value) in keys {
        if key != "weight" {
            return Err(at.bad(
                key,
                value,
                "a parameter of a scorer added in code, which takes its weight alone",
            ));
        }
        // A number past an `f32` narrows to infinity and is refused below.
        weight = value.as_f64().map_or(f32::NAN, |number| number as f32);
        if !above_zero(weight) {
            return Err(at.bad("weight", value, "a finite number above zero"));
        }
    }
    Ok(weight)
}

/// How deep the stages of one generation read a request's block hashes, in
/// the depths' own order: a deeper read holds every shallower one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
enum HashDepth {
    /// No stage reads them.
    #[default]
    Unread,
    /// The deepest prefix a stage keys at, in tokens.
    Tokens(u32),
    /// A stage reads the list whole, as far as the view's cap carries it.
    Whole,
}

impl HashDepth {
    /// What a stage that declares `needs` reads when it names no depth of
    /// its own: the list whole where it declares the request's hashes.
    fn of(needs: Needs) -> Self {
        if needs.request_hashes() {
            Self::Whole
        } else {
            Self::Unread
        }
    }

    /// What a built scorer reads: a consistent-hash stage keyed by a prefix
    /// reads as deep as its `prefix_tokens` or the deepest of the view's
    /// `boundaries`, where a request that reaches one is keyed, and any
    /// other stage as its `needs` declare.
    fn of_scorer(stage: &ScoreStage, boundaries: &[u32], needs: Needs) -> Self {
        match stage {
            ScoreStage::ConsistentHash(ConsistentHashParams {
                prefix_tokens: Some(tokens),
                ..
            }) => {
                let deepest = boundaries.iter().copied().max().unwrap_or(0);
                Self::Tokens(deepest.max(*tokens))
            }
            _ => Self::of(needs),
        }
    }

    /// This depth or `other`, the deeper.
    fn deepen(&mut self, other: Self) {
        *self = (*self).max(other);
    }
}

/// One profile's stages, built when this build carries every one of them,
/// and whether one of them keeps a session key on its worker over the
/// overload veto: the ring stage in its plain form, not in its prefix form.
/// `depth` is deepened to what the built stages read of a request's block
/// hashes.
fn check_profile(
    name: &str,
    p: &ProfileSpec,
    view: &ViewSpec,
    catalog: &Catalog,
    deferred: &mut Deferred,
    depth: &mut HashDepth,
) -> Result<Option<(Profile, bool)>, SpecError> {
    let gates = gates_of(p, view);
    let at = At::new(format!("profiles.{name}"));
    let mut filters = Vec::with_capacity(p.filter.len());
    for f in &p.filter {
        filters.push(
            catalog
                .find_filter(f)
                .ok_or_else(|| SpecError::UnknownFilter {
                    profile: name.to_string(),
                    name: f.clone(),
                })?,
        );
    }
    let mut scorers = Vec::with_capacity(p.score.len());
    let mut complete = true;
    let mut by_key = false;
    for (i, entry) in p.score.iter().enumerate() {
        let entry_at = At::new(format!("{}.score[{i}]", at.0));
        let stage = match scored(entry, name, &entry_at, catalog)? {
            Scored::Typed(stage) => stage,
            Scored::Built(weighted) => {
                depth.deepen(HashDepth::of(weighted.scorer.needs()));
                scorers.push(weighted);
                continue;
            }
        };
        let stage_at = entry_at.sub(stage.name());
        check_stage(&stage, &stage_at)?;
        check_prefix_depth(&stage, gates, view, &stage_at)?;
        match scorer_of(&stage, gates, &view.boundaries) {
            Resolved::Built(scorer) => {
                let read = HashDepth::of_scorer(&stage, &view.boundaries, scorer.needs());
                depth.deepen(read);
                // The ring's prefix form takes no mark, though a request's
                // key leads its prefix there: the gateway's prefix hash
                // moves a key off a worker over its thresholds.
                by_key |=
                    matches!(&stage, ScoreStage::ConsistentHash(p) if p.prefix_tokens.is_none());
                scorers.push(Weighted {
                    scorer,
                    weight: stage.weight(),
                });
            }
            Resolved::NotInBuild(what) => {
                complete = false;
                deferred.note(stage_at.0, what);
            }
        }
    }
    let picker = match picker_of(&p.pick, name, &at, catalog)? {
        Resolved::Built(picker) => Some(picker),
        Resolved::NotInBuild(what) => {
            deferred.note(at.field("pick"), what);
            None
        }
    };
    // A host that makes some lifecycle calls only holds no stage that needs
    // them: it would decide on state nothing keeps. Every entry of a
    // complete profile is built, so a scorer stands at its entry's place.
    if complete && !catalog.makes_lifecycle_calls() {
        let mut entries = p.score.iter().zip(&scorers).enumerate();
        let scorer = entries.find_map(|(i, (entry, built))| {
            let needs = built.scorer.needs_lifecycle();
            needs.then(|| format!("{}.score[{i}].{}", at.0, entry.name()))
        });
        let needs = picker
            .as_ref()
            .is_some_and(|picker| picker.needs_lifecycle());
        if let Some(path) = scorer.or_else(|| needs.then(|| at.field("pick"))) {
            complete = false;
            deferred.note(path, "the lifecycle this stage needs");
        }
    }
    if p.score.is_empty() && SAMPLING_PICKERS.contains(&p.pick.name()) {
        return Err(SpecError::PickerNeedsScorer {
            profile: name.to_string(),
            picker: p.pick.name().to_string(),
        });
    }
    // A picker that leaves the placement to the binding's rule: the engine
    // never runs such a profile's scorers, so a document that names one
    // names a stage that never runs.
    if !p.score.is_empty() && picker.as_ref().is_some_and(|picker| picker.by_assignment()) {
        return Err(SpecError::PickerRunsNoScorer {
            profile: name.to_string(),
            picker: p.pick.name().to_string(),
        });
    }
    Ok(match picker {
        Some(picker) if complete => {
            // A filter and a picker name no depth of their own.
            let unkeyed = filters.iter().map(|filter| filter.needs());
            depth.deepen(HashDepth::of(unkeyed.fold(picker.needs(), Needs::or)));
            Some((
                Profile {
                    name: name.to_string(),
                    filters,
                    scorers,
                    picker,
                },
                by_key,
            ))
        }
        _ => None,
    })
}

/// What a binding's checks read beside the binding.
struct BindingContext<'a> {
    index: usize,
    view: &'a ViewSpec,
    denied: &'a [&'a str],
    /// Profile name to its position in the document's map.
    profiles: &'a BTreeMap<&'a str, u16>,
    /// The profiles whose built picker leaves the placement to a sessions
    /// rule ([`Picker::by_assignment`]), by name.
    by_assignment: &'a [&'a str],
}

fn check_bounds<T>(bounds: &Bounds<T>, at: &At) -> Result<(), SpecError>
where
    T: Copy + PartialOrd + fmt::Display,
{
    match (bounds.min, bounds.max) {
        (None, None) => Err(SpecError::BadField {
            path: at.0.clone(),
            value: "{}".to_string(),
            reason: "a range with min, max or both",
        }),
        (Some(min), Some(max)) if min > max => Err(at.bad("min", min, "at or below max")),
        _ => Ok(()),
    }
}

/// A header match names an allow-listed header.
fn check_header_match(
    header: &HeaderMatch,
    cx: &BindingContext<'_>,
    at: &At,
) -> Result<(), SpecError> {
    check_header_name(&header.name, at.field("name"), cx.denied)?;
    if !cx.view.headers.contains(&header.name) {
        return Err(SpecError::HeaderNotAllowed {
            path: at.field("name"),
            name: header.name.clone(),
        });
    }
    Ok(())
}

fn check_matcher(
    m: &Match,
    cx: &BindingContext<'_>,
    at: &At,
    deferred: &mut Deferred,
) -> Result<(), SpecError> {
    if !glob_ok(&m.model) {
        return Err(SpecError::BadModelGlob {
            index: cx.index,
            model: m.model.clone(),
        });
    }
    let at = at.sub("match");
    if let Some(tenant) = &m.tenant {
        if tenant.is_empty() {
            return Err(at.bad("tenant", "\"\"", "a tenant id"));
        }
        deferred.note(at.field("tenant"), "the tenant matcher");
    }
    if let Some(priority) = &m.priority {
        check_bounds(priority, &at.sub("priority"))?;
        deferred.note(at.field("priority"), "the priority matcher");
    }
    if let Some(header) = &m.header {
        check_header_match(header, cx, &at.sub("header"))?;
        deferred.note(at.field("header"), "the header matcher");
    }
    if let Some(prompt_tokens) = &m.prompt_tokens {
        check_bounds(prompt_tokens, &at.sub("prompt_tokens"))?;
        deferred.note(at.field("prompt_tokens"), "the prompt-tokens matcher");
    }
    if !m.labels.is_empty() {
        check_unique(&m.labels, &at.field("labels"))?;
        deferred.note(at.field("labels"), "the label matcher");
    }
    Ok(())
}

/// The thresholds a binding hands its pools: its block's values, each else
/// the default, and none when the block is switched off. A binding with no
/// block has the default block. The usage goes to permille by the rounding
/// a report's usage takes ([`permille`]), so one rank's usage at or above
/// the threshold in the document's unit is never read as under it. A usage
/// that would round to no permille is held as one: at none, every engine
/// with a row on file would read over it.
fn overload_thresholds(block: Option<&OverloadSpec>) -> Option<OverloadThresholds> {
    let (enabled, waiting, usage) = block.map_or((true, None, None), |block| {
        (block.enabled, block.waiting_requests, block.token_usage)
    });
    enabled.then(|| OverloadThresholds {
        waiting_requests: waiting.unwrap_or(OVERLOAD_WAITING_REQUESTS),
        kv_usage_permille: permille(f64::from(usage.unwrap_or(OVERLOAD_TOKEN_USAGE))).max(1),
    })
}

fn check_overload(overload: &OverloadSpec, at: &At) -> Result<(), SpecError> {
    at.nonzero("waiting_requests", overload.waiting_requests)?;
    at.number(
        "token_usage",
        overload.token_usage,
        unit_fraction,
        "a fraction above zero and at most 1",
    )
}

fn check_sessions(
    sessions: &SessionsSpec,
    cx: &BindingContext<'_>,
    at: &At,
) -> Result<(), SpecError> {
    check_header_name(
        &sessions.sticky_header,
        at.field("sticky_header"),
        cx.denied,
    )?;
    for (i, header) in sessions.headers.iter().enumerate() {
        check_header_name(header, format!("{}.headers[{i}]", at.0), cx.denied)?;
    }
    check_unique(&sessions.headers, &at.field("headers"))?;
    if !sessions.headers.is_empty() && !sessions.headers.contains(&sessions.sticky_header) {
        return Err(at.bad(
            "sticky_header",
            &sessions.sticky_header,
            "one of sessions.headers",
        ));
    }
    if !cx.view.headers.contains(&sessions.sticky_header) {
        return Err(SpecError::HeaderNotAllowed {
            path: at.field("sticky_header"),
            name: sessions.sticky_header.clone(),
        });
    }
    for header in &sessions.headers {
        if !cx.view.headers.contains(header) {
            return Err(SpecError::HeaderNotAllowed {
                path: at.field("headers"),
                name: header.clone(),
            });
        }
    }
    at.nonzero("ttl_s", Some(sessions.ttl_s))?;
    at.nonzero("sweep_s", Some(sessions.sweep_s))?;
    at.nonzero("inflight_cap", Some(sessions.inflight_cap))
}

fn check_canary(canary: &CanarySpec, cx: &BindingContext<'_>, at: &At) -> Result<(), SpecError> {
    if !cx.profiles.contains_key(canary.profile.as_str()) {
        return Err(SpecError::UnknownProfileAt {
            path: at.field("profile"),
            name: canary.profile.clone(),
        });
    }
    let forms = usize::from(canary.percent.is_some())
        + usize::from(!canary.tenants.is_empty())
        + usize::from(canary.header.is_some());
    if forms != 1 {
        return Err(SpecError::CanaryForm {
            path: at.0.clone(),
            count: forms,
        });
    }
    if let Some(percent) = canary.percent.filter(|p| *p == 0 || *p > 100) {
        return Err(at.bad("percent", percent, "between 1 and 100"));
    }
    check_unique(&canary.tenants, &at.field("tenants"))?;
    match &canary.header {
        Some(header) => check_header_match(header, cx, &at.sub("header")),
        None => Ok(()),
    }
}

/// The binding's rules around the decision, in shape; the ones this build
/// does not evaluate are noted as deferred.
fn check_binding_rules(
    b: &BindingSpec,
    cx: &BindingContext<'_>,
    at: &At,
    deferred: &mut Deferred,
) -> Result<(), SpecError> {
    if let Some(overload) = &b.overload {
        check_overload(overload, &at.sub("overload"))?;
    }
    if let Some(sessions) = &b.sessions {
        let at = at.sub("sessions");
        check_sessions(sessions, cx, &at)?;
        if sessions.issue_session_id {
            deferred.note(at.field("issue_session_id"), "issuing a session id");
        }
    }
    if let Some(shadow) = &b.shadow {
        if !cx.profiles.contains_key(shadow.as_str()) {
            return Err(SpecError::UnknownProfileAt {
                path: at.field("shadow"),
                name: shadow.clone(),
            });
        }
        let live = b
            .profile
            .as_deref()
            .or_else(|| b.profiles.get("single").map(String::as_str));
        if live == Some(shadow.as_str()) {
            return Err(at.bad("shadow", shadow, "a profile other than the live one"));
        }
        deferred.note(at.field("shadow"), "the shadow binding");
    }
    if let Some(canary) = &b.canary {
        check_canary(canary, cx, &at.sub("canary"))?;
        deferred.note(at.field("canary"), "the canary binding");
    }
    if b.profile_override == Override::Allowed {
        deferred.note(at.field("override"), "the profile override header");
    }
    if b.admission.as_ref().is_some_and(|a| a.enabled) {
        return Err(SpecError::Reserved {
            path: at.sub("admission").field("enabled"),
            reason: "admission is reserved in v1 and stays off",
        });
    }
    Ok(())
}

/// The binding's plan and sides, resolved to profile positions.
fn resolve_plan(b: &BindingSpec, cx: &BindingContext<'_>) -> Result<Option<Binding>, SpecError> {
    let index = cx.index;
    let plan = b.plan.unwrap_or_default();
    for key in b.profiles.keys() {
        if !SIDES.contains(&key.as_str()) {
            return Err(SpecError::UnknownSide {
                index,
                side: key.clone(),
            });
        }
        if !plan.sides().contains(&key.as_str()) {
            return Err(SpecError::SideNotInPlan {
                index,
                plan: plan.name(),
                side: key.clone(),
            });
        }
    }
    let resolve = |name: &str| {
        cx.profiles
            .get(name)
            .copied()
            .ok_or_else(|| SpecError::UnknownProfile {
                index,
                name: name.to_string(),
            })
    };
    let side = |name: &str| b.profiles.get(name).map(String::as_str);
    let binding = |plan: Plan, profile: u16, decode_profile: Option<u16>| Binding {
        model: b.matcher.model.clone(),
        phases: b.matcher.phase.clone(),
        plan,
        profile,
        decode_profile,
        sessions: None,
        overload: overload_thresholds(b.overload.as_ref()),
    };
    if plan == PlanName::Single {
        if b.profile.is_some() && !b.profiles.is_empty() {
            return Err(SpecError::ProfileTwice { index });
        }
        let name = b
            .profile
            .as_deref()
            .or_else(|| side("single"))
            .ok_or(SpecError::NoProfile { index })?;
        return Ok(Some(binding(Plan::Single, resolve(name)?, None)));
    }
    if b.profile.is_some() {
        return Err(SpecError::ProfileUnderPlan {
            index,
            plan: plan.name(),
        });
    }
    let mut resolved = Vec::with_capacity(plan.sides().len());
    for name in plan.sides() {
        let profile = side(name).ok_or_else(|| SpecError::SideUnnamed {
            index,
            plan: plan.name(),
            side: name,
        })?;
        resolved.push(resolve(profile)?);
    }
    // A plan resolves from one side's lookup and serves every side, so its
    // phase list names them all or none.
    if !b.matcher.phase.is_empty() {
        for (phase, name) in plan.phases() {
            if !b.matcher.phase.contains(phase) {
                return Err(SpecError::PlanPhases {
                    index,
                    plan: plan.name(),
                    side: name,
                });
            }
        }
    }
    Ok(match (plan, resolved.as_slice()) {
        (PlanName::PdPair, &[prefill, decode]) => {
            Some(binding(Plan::PdPair, prefill, Some(decode)))
        }
        (PlanName::Epd, &[encode, prefill, decode]) => {
            Some(binding(Plan::Epd { encode }, prefill, Some(decode)))
        }
        _ => None,
    })
}

fn check_binding(
    b: &BindingSpec,
    cx: &BindingContext<'_>,
    needs: &mut Needs,
    deferred: &mut Deferred,
) -> Result<Option<Binding>, SpecError> {
    let at = At::new(format!("bindings[{}]", cx.index));
    check_matcher(&b.matcher, cx, &at, deferred)?;
    check_binding_rules(b, cx, &at, deferred)?;
    let mut built = resolve_plan(b, cx)?;
    // A picker that leaves the placement to a sessions rule, and a rule runs
    // under the single plan alone: on a side of a pair or of `epd` nothing
    // would place its requests. The built picker answers, as it answers the
    // engine.
    if b.plan.unwrap_or_default() != PlanName::Single {
        for (side, name) in &b.profiles {
            if cx.by_assignment.contains(&name.as_str()) {
                deferred.note(
                    at.sub("profiles").field(side),
                    "a picker that places by the sessions rule outside the single plan",
                );
            }
        }
    }
    if let (Some(binding), Some(sessions)) = (built.as_mut(), &b.sessions) {
        let rule = session_rule(sessions, cx.index, binding.plan)?;
        needs.active_sessions |= rule.assignment == Assignment::MinGroup;
        needs.session_in_flight |= rule.inflight_cap > 0;
        binding.sessions = Some(rule);
    }
    Ok(built)
}

/// The sessions rule as the engine runs it: the schema's largest cap is no
/// cap, its largest idle time or sweep is for ever (a sweep for ever under a
/// finite idle time is refused, as nothing would forget the keys; an idle
/// time for ever sweeps nothing), and the rule runs under the `single` plan
/// alone.
fn session_rule(
    sessions: &SessionsSpec,
    index: usize,
    plan: Plan,
) -> Result<SessionRule, SpecError> {
    if plan != Plan::Single {
        return Err(SpecError::BadSessions {
            index,
            name: "plan",
            reason: "sessions run under the single plan",
        });
    }
    let forever = |secs: u32| secs == u32::MAX;
    if forever(sessions.sweep_s) && !forever(sessions.ttl_s) {
        return Err(SpecError::BadSessions {
            index,
            name: "sweep_s",
            reason: "finite while ttl_s is",
        });
    }
    Ok(SessionRule {
        assignment: sessions.assignment,
        inflight_cap: if sessions.inflight_cap == u16::MAX {
            0
        } else {
            sessions.inflight_cap
        },
        ttl_micros: (!forever(sessions.ttl_s)).then(|| u64::from(sessions.ttl_s) * 1_000_000),
        sweep_micros: if forever(sessions.ttl_s) {
            0
        } else {
            u64::from(sessions.sweep_s) * 1_000_000
        },
        recovery: sessions.recovery,
    })
}

/// The ten preset names of today, as whole profiles.
pub const PRESETS: [&str; 10] = [
    "random",
    "round_robin",
    "passthrough",
    "power_of_two",
    "least_load",
    "consistent_hashing",
    "prefix_hash",
    "manual",
    "cache_aware",
    "bucket",
];

/// The header the `manual` preset names as its session key, the gateway's
/// routing-key header; the host feeds the key it resolved under it.
pub const MANUAL_SESSION_HEADER: &str = "x-smg-routing-key";

fn standard_filters() -> Vec<String> {
    vec![
        "healthy".into(),
        "not-overloaded".into(),
        "not-tried".into(),
    ]
}

/// The load metric the `power_of_two` preset compares its candidates by:
/// one form of the preset each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerOfTwoMetric {
    /// Two draws, the lower expected wait: the preset under its bare name.
    ExpectedWait,
    /// Two draws, the fewer requests in flight, the expected wait between
    /// equal counts.
    Requests,
    /// The fewest requests in flight over every candidate, the expected wait
    /// between equal counts.
    LeastRequests,
    /// The least resident work over every candidate, each request priced at
    /// `request_tokens` context tokens.
    ResidentTokens { request_tokens: u32 },
}

impl PolicySpec {
    /// The `power_of_two` preset in the form its load metric names, bound to
    /// every model and phase: the count forms compare the requests in flight
    /// before the score through the count-first pickers, and the resident
    /// form scores every candidate and takes the best, with no draw of two.
    pub fn power_of_two(metric: PowerOfTwoMetric) -> Result<Self, SpecError> {
        const NAME: &str = "power_of_two";
        let mut spec = Self::preset(NAME)?;
        let profile = spec.profiles.get_mut(NAME).ok_or(SpecError::Empty)?;
        match metric {
            PowerOfTwoMetric::ExpectedWait => {}
            PowerOfTwoMetric::Requests => profile.pick = "power-of-two-count-first".into(),
            PowerOfTwoMetric::LeastRequests => profile.pick = "count-first".into(),
            PowerOfTwoMetric::ResidentTokens { request_tokens } => {
                profile.score = vec![ScoreStage::ResidentWork(ResidentWorkParams {
                    weight: 1.0,
                    request_tokens,
                })
                .into()];
                profile.pick = "best".into();
            }
        }
        Ok(spec)
    }

    /// A spec of one preset profile bound to every model and phase.
    pub fn preset(name: &str) -> Result<Self, SpecError> {
        let profile = match name {
            "random" => ProfileSpec {
                filter: standard_filters(),
                score: vec![],
                pick: "random".into(),
            },
            "round_robin" => ProfileSpec {
                filter: standard_filters(),
                score: vec![],
                pick: "round-robin".into(),
            },
            "passthrough" => ProfileSpec {
                filter: standard_filters(),
                score: vec![],
                pick: "passthrough".into(),
            },
            // The expected-wait form: two draws compared by least-load.
            "power_of_two" => ProfileSpec {
                filter: standard_filters(),
                score: vec!["least-load".into()],
                pick: "power-of-two".into(),
            },
            "least_load" => ProfileSpec {
                filter: standard_filters(),
                score: vec!["least-load".into()],
                pick: "best".into(),
            },
            // The ring owner of the decision's key (an encode item's content
            // hash, a request's routing key), the ring's next members as the
            // alternates; a keyless decision is the picker's draw.
            "consistent_hashing" => ProfileSpec {
                filter: standard_filters(),
                score: vec!["consistent-hash".into()],
                pick: "best".into(),
            },
            // The ring owner of the request's key: its routing key, else its
            // prefix at 256 tokens. An owner past the bounded-load rule is
            // passed over and a request with no key has no owner: the
            // fewest requests in flight take both. The ring's weight is by
            // arithmetic: an owner leads the ring's next member by a
            // quarter of the stage's score and the count ranges over one
            // whole score, so any weight past four keeps a kept owner above
            // the least loaded worker. The pick names no sample: the owner
            // must be among the rows scored, so the pool is read whole.
            "prefix_hash" => ProfileSpec {
                filter: standard_filters(),
                score: vec![
                    ScoreStage::ConsistentHash(ConsistentHashParams {
                        weight: 8.0,
                        prefix_tokens: Some(256),
                        bounded_load: Some(BoundedLoadParams {
                            factor: Some(1.25),
                            abs_margin: Some(10),
                        }),
                    })
                    .into(),
                    "least-requests".into(),
                ],
                pick: "best".into(),
            },
            // Sticky sessions over a random assignment: a key stays with the
            // first worker it was placed on, no in-flight cap, the first
            // worker back in the lead when it recovers; the idle TTL of four
            // hours swept each minute, as the keyed policy of today. The
            // picker leaves the placement to the rule, so another assignment
            // is the rule's one field and no other profile.
            "manual" => ProfileSpec {
                filter: standard_filters(),
                score: vec![],
                pick: "manual".into(),
            },
            // Affinity first, the expected wait within the group: the slice
            // and the diversion over everything when due, the holder tiers
            // over the load, the fleet gate configuring both. The weights
            // are ranks: each stage's values are 0, 0.5 or 1, so a stage
            // outweighs everything after it.
            "cache_aware" => ProfileSpec {
                filter: standard_filters(),
                score: vec![
                    ScoreStage::WarmupSlice(WarmupSliceParams {
                        weight: 8.0,
                        ..WarmupSliceParams::default()
                    })
                    .into(),
                    ScoreStage::PrefixAffinity(PrefixAffinityParams {
                        weight: 4.0,
                        ..PrefixAffinityParams::default()
                    })
                    .into(),
                    "kv-pressure".into(),
                    "least-load".into(),
                ],
                // A large pool is decided by a draw of sixteen rows and the
                // rows the host names: its holders and its warming rows.
                pick: PickStage::Best(BestParams { sample: 16 }).into(),
            },
            // Prefill placement by request size: the worker whose range of
            // sizes holds the prompt, the least booked survivor when the
            // balance rule spills, each at the stage's defaults; any other
            // phase is the picker's draw.
            "bucket" => ProfileSpec {
                filter: standard_filters(),
                score: vec!["bucket".into()],
                pick: "best".into(),
            },
            other => return Err(SpecError::UnknownPreset(other.to_string())),
        };
        let manual = name == "manual";
        let sessions = manual.then(|| SessionsSpec {
            sticky_header: MANUAL_SESSION_HEADER.into(),
            headers: vec![],
            ttl_s: 14_400,
            sweep_s: 60,
            assignment: Assignment::Random,
            inflight_cap: u16::MAX,
            issue_session_id: false,
            recovery: Recovery::Original,
        });
        let view = ViewSpec {
            headers: if manual {
                vec![MANUAL_SESSION_HEADER.into()]
            } else {
                vec![]
            },
            ..ViewSpec::default()
        };
        let mut profiles = BTreeMap::new();
        profiles.insert(name.to_string(), profile);
        Ok(Self {
            api_version: API_VERSION.to_string(),
            generation: 1,
            kind: None,
            view,
            index: BTreeMap::new(),
            profiles,
            plans: PlansSpec::default(),
            bindings: vec![BindingSpec {
                profile: Some(name.to_string()),
                sessions,
                ..BindingSpec::default()
            }],
        })
    }

    /// A pair spec: `prefill` and `decode` presets under the `pd-pair` plan,
    /// with the capacity and room filters in front of each side.
    pub fn pair_presets(prefill: &str, decode: &str) -> Result<Self, SpecError> {
        let mut spec = Self::preset(prefill)?;
        let decode_spec = Self::preset(decode)?;
        let mut prefill_profile = spec.profiles.remove(prefill).ok_or(SpecError::Empty)?;
        prefill_profile.filter.insert(2, "has-capacity".into());
        let mut decode_profile = decode_spec
            .profiles
            .get(decode)
            .cloned()
            .ok_or(SpecError::Empty)?;
        decode_profile.filter.insert(2, "has-room".into());
        spec.profiles.clear();
        spec.profiles.insert("prefill".into(), prefill_profile);
        spec.profiles.insert("decode".into(), decode_profile);
        let mut profiles = BTreeMap::new();
        profiles.insert("prefill".to_string(), "prefill".to_string());
        profiles.insert("decode".to_string(), "decode".to_string());
        spec.bindings = vec![BindingSpec {
            plan: Some(PlanName::PdPair),
            profiles,
            ..BindingSpec::default()
        }];
        Ok(spec)
    }

    /// An encode-prefill-decode spec: the `encode` preset over the encode
    /// pool and the pair presets of [`PolicySpec::pair_presets`], under the
    /// `epd` plan.
    pub fn epd_presets(encode: &str, prefill: &str, decode: &str) -> Result<Self, SpecError> {
        let mut spec = Self::pair_presets(prefill, decode)?;
        let encode_profile = Self::preset(encode)?
            .profiles
            .remove(encode)
            .ok_or(SpecError::Empty)?;
        spec.profiles.insert("encode".into(), encode_profile);
        let binding = spec.bindings.first_mut().ok_or(SpecError::Empty)?;
        binding.plan = Some(PlanName::Epd);
        binding
            .profiles
            .insert("encode".to_string(), "encode".to_string());
        Ok(spec)
    }

    /// Validate the document and build the engine; the needs summary comes
    /// with it. The built-in header deny-list applies; a host with a tenant
    /// header of its own uses [`PolicySpec::validate_with`].
    pub fn validate(&self) -> Result<Engine, SpecError> {
        self.validate_with(&[])
    }

    /// [`PolicySpec::validate`] with the host's own denied header names
    /// (the configured tenant header) added to the built-in deny-list.
    pub fn validate_with(&self, denied_headers: &[&str]) -> Result<Engine, SpecError> {
        self.validate_in(&STOCK, denied_headers)
    }

    /// [`PolicySpec::validate_with`] over the stages `catalog` holds: the
    /// stock build's and the ones a host added in code, each named in the
    /// document as a built-in is. Every stage name is resolved here, once;
    /// a factory builds its stage for this engine alone.
    ///
    /// Every field is checked in shape first; a field this build cannot
    /// honour (a stage, a plan, a matcher or a binding rule of a later step)
    /// is then refused by name as [`SpecError::NotInBuild`], and so is a
    /// stage that needs the lifecycle's calls
    /// ([`crate::Lifecycle::needs_lifecycle`]) under the standalone
    /// controller, which makes the membership call alone.
    pub fn validate_in(
        &self,
        catalog: &Catalog,
        denied_headers: &[&str],
    ) -> Result<Engine, SpecError> {
        let (engine, _) = self.built(catalog, denied_headers)?;
        Ok(engine)
    }

    /// [`PolicySpec::validate_in`], and beside the engine how deep its
    /// stages read a request's block hashes.
    fn built(
        &self,
        catalog: &Catalog,
        denied_headers: &[&str],
    ) -> Result<(Engine, HashDepth), SpecError> {
        if let Some(kind) = &self.kind {
            return Err(SpecError::Kind(kind.clone()));
        }
        if self.api_version != API_VERSION {
            return Err(SpecError::ApiVersion(self.api_version.clone()));
        }
        if self.profiles.is_empty() {
            return Err(SpecError::Empty);
        }
        if self.profiles.len() > usize::from(u16::MAX) {
            return Err(SpecError::TooManyProfiles {
                count: self.profiles.len(),
                max: u16::MAX,
            });
        }
        if self.bindings.is_empty() && self.profiles.len() > 1 {
            return Err(SpecError::NoBindings {
                profiles: self.profiles.len(),
            });
        }
        check_view(&self.view, denied_headers)?;
        check_index(&self.index, &self.view)?;
        let mut deferred = Deferred::default();
        let mut depth = HashDepth::default();
        // What no stage declares: the sessions rules' counts and the view's
        // headers. The engine folds every profile's stages in.
        let mut needs = Needs::default();
        let mut profiles = Vec::with_capacity(self.profiles.len());
        // Noted by name: the built vector lines up with the document's
        // positions only once nothing is deferred.
        let mut by_assignment = Vec::new();
        // The built profiles a stage of which places a request by its
        // session key, by their place in `profiles`.
        let mut keyed = Vec::new();
        for (name, profile) in &self.profiles {
            let built = check_profile(
                name,
                profile,
                &self.view,
                catalog,
                &mut deferred,
                &mut depth,
            )?;
            if let Some((built, by_key)) = built {
                if built.picker.by_assignment() {
                    by_assignment.push(name.as_str());
                }
                if by_key {
                    keyed.push(profiles.len());
                }
                profiles.push(built);
            }
        }
        // A binding addresses its profile by position in the built vector,
        // which holds every document profile once the deferred check below
        // has passed; the bound above keeps the position in a u16.
        let mut positions: BTreeMap<&str, u16> = BTreeMap::new();
        for (i, name) in self.profiles.keys().enumerate() {
            let position = u16::try_from(i).map_err(|_| SpecError::TooManyProfiles {
                count: self.profiles.len(),
                max: u16::MAX,
            })?;
            positions.insert(name.as_str(), position);
        }
        let mut bindings = Vec::with_capacity(self.bindings.len());
        // The built bindings whose document sheds a pool the overload veto
        // empties, by their place in `bindings`.
        let mut sheds = Vec::new();
        for (index, binding) in self.bindings.iter().enumerate() {
            let cx = BindingContext {
                index,
                view: &self.view,
                denied: denied_headers,
                profiles: &positions,
                by_assignment: &by_assignment,
            };
            if let Some(built) = check_binding(binding, &cx, &mut needs, &mut deferred)? {
                if binding
                    .overload
                    .as_ref()
                    .is_some_and(|overload| overload.shed_when_all_vetoed)
                {
                    sheds.push(bindings.len());
                }
                bindings.push(built);
            }
        }
        if let Some(missing) = deferred.0 {
            return Err(missing);
        }
        debug_assert_eq!(
            profiles.len(),
            self.profiles.len(),
            "every profile is built once the deferred check has passed"
        );
        if self.bindings.is_empty() {
            // One profile, refused above when there are more.
            bindings.push(Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                sessions: None,
                overload: overload_thresholds(None),
            });
        }
        needs.headers |= !self.view.headers.is_empty();
        let mut engine = Engine::new(profiles, bindings, self.generation, needs);
        for at in sheds {
            engine.shed_when_all_vetoed(at);
        }
        for at in keyed {
            engine.places_by_key(at);
        }
        Ok((engine, depth))
    }

    /// A document of this version with the given profiles and nothing else
    /// written: one profile binds to everything, more need bindings.
    #[must_use]
    pub fn new(profiles: BTreeMap<String, ProfileSpec>) -> Self {
        Self {
            api_version: API_VERSION.to_string(),
            generation: 0,
            kind: None,
            view: ViewSpec::default(),
            index: BTreeMap::new(),
            profiles,
            plans: PlansSpec::default(),
            bindings: Vec::new(),
        }
    }

    /// The index settings for a model: the most specific matching key, if
    /// any (an exact id, then the longer `prefix*` or `*suffix`, then `*`).
    #[must_use]
    pub fn index_for(&self, model: &str) -> Option<(&str, &IndexSpec)> {
        self.index
            .iter()
            .filter(|(pattern, _)| glob_matches(pattern, model))
            .max_by_key(|(pattern, _)| specificity(pattern))
            .map(|(pattern, spec)| (pattern.as_str(), spec))
    }

    /// The JSON schema of the document, generated from these types: the
    /// published `policyspec.schema.json`.
    pub fn schema() -> Schema {
        schemars::schema_for!(PolicySpec)
    }

    /// The schema as the committed artifact's text: pretty JSON with one
    /// trailing newline.
    pub fn schema_json() -> Result<String, serde_json::Error> {
        let mut text = serde_json::to_string_pretty(&Self::schema())?;
        text.push('\n');
        Ok(text)
    }

    /// The document's digest ([`Digest`]): a change of the file's form or of
    /// the `generation` line leaves it as it was, and any other change to a
    /// document the validator admits moves it. A `kind` line is in no
    /// canonical form and so in no digest: the validator refuses it, and
    /// equal digests say nothing of a document that was not validated.
    pub fn digest(&self) -> Result<Digest, SpecError> {
        let mut hash = Sha256::new();
        serde_json::to_writer(HashWriter(&mut hash), &self.lineless())
            .map_err(|error| SpecError::NoCanonicalForm(error.to_string()))?;
        Ok(Digest(hash.finalize().into()))
    }

    /// The document's canonical form, borrowed: a digest copies none of the
    /// document. Every field is named here, so a field the type gains does
    /// not compile until this form takes it or leaves it out by name.
    fn lineless(&self) -> Lineless<'_> {
        let Self {
            api_version,
            generation: _,
            kind: _,
            view,
            index,
            profiles,
            plans,
            bindings,
        } = self;
        Lineless {
            api_version,
            generation: 0,
            view,
            index,
            profiles,
            plans,
            bindings,
        }
    }
}

/// A document's canonical form ([`Digest`]) by reference: [`PolicySpec`]'s
/// fields in its order and under its attributes, the `generation` line
/// zeroed and no `kind` line. It serializes as the document itself does
/// under a zeroed line, byte for byte.
#[derive(Serialize)]
struct Lineless<'a> {
    #[serde(rename = "apiVersion")]
    api_version: &'a str,
    generation: u64,
    #[serde(skip_serializing_if = "is_default::<ViewSpec>")]
    view: &'a ViewSpec,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    index: &'a BTreeMap<String, IndexSpec>,
    profiles: &'a BTreeMap<String, ProfileSpec>,
    #[serde(skip_serializing_if = "is_default::<PlansSpec>")]
    plans: &'a PlansSpec,
    bindings: &'a [BindingSpec],
}

/// A document's digest: SHA-256 over its canonical bytes, which are the
/// document's JSON as [`PolicySpec`] serializes (its maps ordered, its fields
/// in the type's order) with the `generation` line zeroed. Two files that
/// parse into one document share it whatever their form (YAML or JSON, the
/// order of their keys, their comments), and a document applied again under
/// a higher number keeps it. It is written `sha256:` and 64 hex digits.
///
/// The canonical bytes are this build's: two builds whose schema differs may
/// print two digests for one file.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest([u8; 32]);

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("sha256:")?;
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// What a serializer writes, fed to a hash: the canonical bytes are never
/// held whole.
struct HashWriter<'a>(&'a mut Sha256);

impl io::Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// One generation as a store applies it: the engine validated from a
/// document, with the generation's number, its needs and its view's cap on
/// a request's block hashes.
///
/// One invariant: **a decision reads its rows, its engine and its number
/// in one load.** A store publishes every world under one of these
/// ([`World::spec`](crate::store::World::spec)), so no reader pairs one
/// generation's engine with rows published under another. The number is the
/// engine's own, the document's `generation` line: there is no second
/// holder.
///
/// A document becomes one by validation alone ([`AppliedSpec::validate`]):
/// one that fails validation never does. The other constructor validates
/// nothing: [`AppliedSpec::unbound`] is what a store holds before any
/// document is applied, and the only generation 0 a store ever holds. A
/// document that names no generation validates to number 0 too, and the
/// store's apply refuses it as not above the standing one.
///
/// It carries the document's digest beside the number ([`Digest`]): the
/// number says which generation a replica runs, the digest which document,
/// and one load reads the pair.
///
/// It also holds what a store derives under, beside the engine: the
/// document's pairing mode ([`AppliedSpec::pairing`]). And it says how deep
/// a host hashes a request for it ([`AppliedSpec::block_hashes_read`]).
#[derive(Debug)]
pub struct AppliedSpec {
    engine: Engine,
    /// The digest of the document the engine was validated from; none for
    /// the unbound start, which has no document.
    digest: Option<Digest>,
    /// How deep the engine's stages read a request's block hashes.
    depth: HashDepth,
    /// The document's `view.max_block_hashes`: no read is past it.
    max_block_hashes: u32,
    /// The view's block size: a request that names none is counted by it.
    block_size: u32,
    pairing: Pairing,
    lists: ViewLists,
}

impl AppliedSpec {
    /// The generation `document` validates into ([`PolicySpec::validate`]).
    pub fn validate(document: &PolicySpec) -> Result<Self, SpecError> {
        Self::validate_with(document, &[])
    }

    /// [`AppliedSpec::validate`] with the host's own denied header names,
    /// as [`PolicySpec::validate_with`] takes them.
    pub fn validate_with(
        document: &PolicySpec,
        denied_headers: &[&str],
    ) -> Result<Self, SpecError> {
        let built = document.built(&STOCK, denied_headers)?;
        Ok(Self::of(document, document.digest()?, built))
    }

    /// [`AppliedSpec::validate`] over the stages `catalog` holds, under a
    /// digest its caller took already: the controller's door reads the
    /// document's digest against the standing one and hands it on, so an
    /// apply digests its document once.
    #[cfg(feature = "controller")]
    pub(crate) fn validate_digested(
        document: &PolicySpec,
        digest: Digest,
        catalog: &Catalog,
    ) -> Result<Self, SpecError> {
        let built = document.built(catalog, &[])?;
        Ok(Self::of(document, digest, built))
    }

    /// The generation of `document` under its `digest`: the engine it
    /// validated into, with the depth of that engine's stages.
    fn of(document: &PolicySpec, digest: Digest, (engine, depth): (Engine, HashDepth)) -> Self {
        Self {
            engine,
            digest: Some(digest),
            depth,
            max_block_hashes: document.view.max_block_hashes,
            block_size: Gates::of(None, None, document.view.block_size).block_size,
            pairing: document.plans.pd_pair.pairing,
            lists: ViewLists::of(&document.view),
        }
    }

    /// The generation a store holds before any document is applied: number
    /// 0, an engine with no profile and no binding, which declines every
    /// decision by name as an engine declines a model no binding covers,
    /// and the pairing mode a document that names none has.
    #[must_use]
    pub fn unbound() -> Self {
        Self {
            engine: Engine::new(Vec::new(), Vec::new(), 0, Needs::default()),
            digest: None,
            depth: HashDepth::Unread,
            max_block_hashes: ViewSpec::default().max_block_hashes,
            block_size: Gates::default().block_size,
            pairing: Pairing::default(),
            lists: ViewLists::default(),
        }
    }

    /// The generation's number: the engine's own.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.engine.generation()
    }

    /// The digest of the document this generation was validated from
    /// ([`PolicySpec::digest`]); `None` for the unbound start, which has no
    /// document.
    #[must_use]
    pub fn digest(&self) -> Option<Digest> {
        self.digest
    }

    /// The engine a decision under this generation runs.
    #[must_use]
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// The most prefix block hashes of one request a decision under this
    /// generation reads: the document's `view.max_block_hashes`. What a
    /// decision reads of a request is [`Self::block_hashes_read`], this
    /// many at most.
    #[must_use]
    pub fn max_block_hashes(&self) -> usize {
        usize::try_from(self.max_block_hashes).unwrap_or(usize::MAX)
    }

    /// How many of a request's prefix block hashes a decision under this
    /// generation reads, for a request counted in `kind` by blocks of
    /// `block_size`, `0` for the view's size. The generation's depth is in
    /// tokens, the deepest `prefix_tokens` of its consistent-hash stages,
    /// and the blocks are the request's own, counted as the key counts
    /// them. A stage that declares the request's hashes and names no depth
    /// reads to the cap ([`Self::max_block_hashes`]), which no depth is
    /// past, and a generation with no such stage reads none. A host hashes
    /// that many blocks of a request and no more, and hands the engine the
    /// first that many of a longer list.
    #[must_use]
    pub fn block_hashes_read(&self, kind: HashKind, block_size: u32) -> usize {
        let blocks = match self.depth {
            HashDepth::Unread => 0,
            HashDepth::Tokens(tokens) => {
                let block_size = if block_size > 0 {
                    block_size
                } else {
                    self.block_size
                };
                blocks_within(tokens, kind, block_size).min(self.max_block_hashes)
            }
            HashDepth::Whole => self.max_block_hashes,
        };
        usize::try_from(blocks).unwrap_or(usize::MAX)
    }

    /// What the host must feed under this generation
    /// ([`Engine::needs`]).
    #[must_use]
    pub fn needs(&self) -> Needs {
        self.engine.needs()
    }

    /// The mode a store derives its pair classes under while this
    /// generation stands: the document's `plans.pd-pair.pairing`.
    #[must_use]
    pub fn pairing(&self) -> Pairing {
        self.pairing
    }

    /// The lists this generation declares, as a store reads them when it
    /// writes a worker's row.
    #[must_use]
    pub fn lists(&self) -> &ViewLists {
        &self.lists
    }
}

/// The lists a generation declares, as a store reads them when it writes a
/// worker's row ([`Candidate`](crate::view::Candidate)): the row names its
/// engine family and its zone by a position from 1, and holds one bit for
/// each label selector the worker meets. Built once, at validation.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ViewLists {
    /// The label selectors in their keys' order, each with its bit: a
    /// worker's labels are in their keys' order too, so one pass over the
    /// two finds every selector the worker meets.
    labels: Vec<LabelSelector>,
    zones: FxHashMap<Box<str>, u16>,
    engines: FxHashMap<Box<str>, u16>,
}

/// One label selector: its key, the value it asks for when it names one,
/// and its bit, which is its place in the document's list.
#[derive(Debug, PartialEq, Eq)]
struct LabelSelector {
    key: Box<str>,
    value: Option<Box<str>>,
    bit: u8,
}

/// What one worker's registration reads as under a generation's lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Listed {
    /// The position of the worker's engine family, from 1; `0` when the
    /// list does not name it.
    pub engine_kind: u16,
    /// The position of the worker's zone, from 1; `0` for a worker with no
    /// zone and for a zone the list does not name.
    pub zone: u16,
    /// One bit for each label selector the worker meets.
    pub labels: u64,
    /// The worker names an engine family, and the list holds entries and
    /// not that one.
    pub engine_outside: bool,
    /// The worker names a zone, and the list holds entries and not that
    /// one.
    pub zone_outside: bool,
}

impl ViewLists {
    fn of(view: &ViewSpec) -> Self {
        let positions = |list: &[String]| {
            let named = list.iter().zip(1u16..);
            named.map(|(name, at)| (name.as_str().into(), at)).collect()
        };
        let sides = |(label, bit): (&String, u8)| {
            let (key, value) = selector(label);
            LabelSelector {
                key: key.into(),
                value: value.map(Into::into),
                bit,
            }
        };
        let mut labels: Vec<LabelSelector> = view.labels.iter().zip(0u8..).map(sides).collect();
        // The sort is stable: selectors of one key keep the list's order.
        labels.sort_by(|a, b| a.key.cmp(&b.key));
        Self {
            labels,
            zones: positions(&view.zones),
            engines: positions(&view.engines),
        }
    }

    /// Whether the generation declares no list: every row reads 0 in the
    /// three fields, and a store looks nothing up.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty() && self.zones.is_empty() && self.engines.is_empty()
    }

    /// A worker's three fields under these lists, from the engine family,
    /// the zone and the labels of its registration.
    #[must_use]
    pub fn listed(
        &self,
        engine: &str,
        zone: Option<&str>,
        labels: &BTreeMap<String, String>,
    ) -> Listed {
        let engine_at = self.engines.get(engine).copied();
        let zone = zone.filter(|zone| !zone.is_empty());
        let zone_at = zone.and_then(|zone| self.zones.get(zone).copied());
        Listed {
            engine_kind: engine_at.unwrap_or(0),
            zone: zone_at.unwrap_or(0),
            labels: self.met(labels),
            engine_outside: engine_at.is_none() && !engine.is_empty() && !self.engines.is_empty(),
            zone_outside: zone_at.is_none() && zone.is_some() && !self.zones.is_empty(),
        }
    }

    /// The bits of the selectors a worker's labels meet, in one pass over
    /// the two, each in its keys' order.
    fn met(&self, labels: &BTreeMap<String, String>) -> u64 {
        let mut bits = 0;
        let mut wanted = self.labels.iter().peekable();
        'held: for (key, held) in labels {
            while let Some(want) = wanted.peek() {
                match (*want.key).cmp(key.as_str()) {
                    Ordering::Less => {}
                    Ordering::Greater => continue 'held,
                    Ordering::Equal => {
                        let value = want.value.as_deref();
                        let met = value.is_none_or(|value| value == held.as_str());
                        bits |= u64::from(met) << want.bit;
                    }
                }
                wanted.next();
            }
            break;
        }
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Candidate, Constraints, Filter, Lifecycle, RequestFacts, Targets, TieBand};

    #[test]
    fn an_applied_generation_reads_block_hashes_as_deep_as_its_stages_do() {
        use HashKind::{Text, Token};
        let preset = |name: &str| {
            let mut document = PolicySpec::preset(name).expect("a preset of this build");
            document.generation = 1;
            document
        };
        let applied = |document: &PolicySpec| {
            AppliedSpec::validate(document).expect("the document validates")
        };
        // No stage of the random preset reads the hashes, and the unbound
        // start has no stage.
        let random = applied(&preset("random"));
        assert_eq!(random.block_hashes_read(Token, 0), 0);
        assert_eq!(random.block_hashes_read(Text, 0), 0);
        assert_eq!(AppliedSpec::unbound().block_hashes_read(Token, 0), 0);
        assert_eq!(AppliedSpec::unbound().max_block_hashes(), 256);
        // A prefix key 256 tokens deep, in the request's own blocks: 16 of
        // the view's 16 tokens, 32 of a request counted by 8 and 8 of one
        // counted by 32, and of a text its 1,024 characters in blocks of
        // the size it was counted by. The cap stays the document's.
        let mut keyed = preset("prefix_hash");
        let ring = applied(&keyed);
        assert_eq!(ring.block_hashes_read(Token, 0), 16);
        assert_eq!(ring.block_hashes_read(Token, 8), 32);
        assert_eq!(ring.block_hashes_read(Token, 32), 8);
        assert_eq!(ring.block_hashes_read(Text, 0), 64);
        assert_eq!(ring.block_hashes_read(Text, 64), 16);
        assert_eq!(ring.max_block_hashes(), 256);
        // A request that names no block size is counted by the view's, and
        // no depth is past the view's cap.
        keyed.view.block_size = Some(32);
        keyed.view.max_block_hashes = 24;
        let ring = applied(&keyed);
        assert_eq!(ring.max_block_hashes(), 24);
        assert_eq!(ring.block_hashes_read(Token, 0), 8);
        assert_eq!(ring.block_hashes_read(Text, 0), 24);
        // The controller's door builds a generation under a digest it took
        // already: that builder keeps the depth too.
        #[cfg(feature = "controller")]
        {
            let digest = keyed.digest().expect("the document digests");
            let applied = AppliedSpec::validate_digested(&keyed, digest, &Catalog::builtin())
                .expect("the preset validates");
            assert_eq!(applied.block_hashes_read(Token, 0), 8);
        }
        // A stage with no depth of its own reads to the view's cap.
        let mut held = preset("cache_aware");
        assert_eq!(applied(&held).block_hashes_read(Token, 0), 256);
        held.view.max_block_hashes = 128;
        assert_eq!(applied(&held).block_hashes_read(Text, 4), 128);
        // A generation reads as deep as the deepest of its profiles.
        let two = doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":[{"consistent-hash":{"prefix_tokens":256}}],"pick":"best"},"q":{"score":[{"consistent-hash":{"prefix_tokens":512}}],"pick":"best"}},"bindings":[{"match":{"model":"a*"},"profiles":{"single":"p"}},{"match":{"model":"*"},"profiles":{"single":"q"}}]}"#,
        );
        assert_eq!(applied(&two).block_hashes_read(Token, 0), 32);
        // Under a view's boundaries a prefix key is read at the deepest one
        // a request reaches, so the generation reads as deep as that one,
        // and no shallower than the key's own depth.
        let mut levelled = preset("prefix_hash");
        levelled.view.boundaries = vec![64, 1024];
        assert_eq!(applied(&levelled).block_hashes_read(Token, 0), 64);
        levelled.view.boundaries = vec![64];
        assert_eq!(applied(&levelled).block_hashes_read(Token, 0), 16);
    }

    /// A guest stage of each kind, under the name it is registered by, that
    /// declares the request's token hashes and names no depth.
    #[derive(Debug)]
    struct ReadsHashes(&'static str);

    impl ReadsHashes {
        fn declared() -> Needs {
            Needs {
                token_hashes: true,
                ..Needs::default()
            }
        }
    }

    impl Lifecycle for ReadsHashes {}

    impl Filter for ReadsHashes {
        fn name(&self) -> &'static str {
            self.0
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn needs(&self) -> Needs {
            Self::declared()
        }
    }

    impl Scorer for ReadsHashes {
        fn name(&self) -> &'static str {
            self.0
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(0.0);
            TieBand::NORMALIZED
        }

        fn needs(&self) -> Needs {
            Self::declared()
        }
    }

    impl Picker for ReadsHashes {
        fn name(&self) -> &'static str {
            self.0
        }

        fn pick(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            _: &[f32],
            _: TieBand,
            _: &mut Targets,
        ) {
        }

        fn needs(&self) -> Needs {
            Self::declared()
        }
    }

    #[test]
    fn a_stage_that_names_no_depth_reads_the_block_hashes_to_the_cap() {
        let mut catalog = Catalog::builtin();
        catalog
            .filter("hashing-filter", || Box::new(ReadsHashes("hashing-filter")))
            .and_then(|catalog| {
                catalog.scorer("hashing-scorer", || Box::new(ReadsHashes("hashing-scorer")))
            })
            .and_then(|catalog| {
                catalog.picker("hashing-picker", || Box::new(ReadsHashes("hashing-picker")))
            })
            .expect("three free names");
        let depth = |profile: &str| {
            let document = doc(&format!(
                r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{profile}}}}}"#
            ));
            let (_, depth) = document
                .built(&catalog, &[])
                .expect("the document validates");
            depth
        };
        assert_eq!(depth(r#"{"pick":"random"}"#), HashDepth::Unread);
        // A guest of each kind declares the hashes and names no depth.
        for profile in [
            r#"{"filter":["hashing-filter"],"pick":"random"}"#,
            r#"{"score":["hashing-scorer"],"pick":"best"}"#,
            r#"{"pick":"hashing-picker"}"#,
        ] {
            assert_eq!(depth(profile), HashDepth::Whole, "{profile}");
        }
        // A prefix key reads to its own depth, and beside a stage that
        // reads the list whole the generation reads it whole.
        let key = r#"{"consistent-hash":{"prefix_tokens":256}}"#;
        assert_eq!(
            depth(&format!(r#"{{"score":[{key}],"pick":"best"}}"#)),
            HashDepth::Tokens(256)
        );
        assert_eq!(
            depth(&format!(
                r#"{{"score":[{key},"hashing-scorer"],"pick":"best"}}"#
            )),
            HashDepth::Whole
        );
    }

    fn doc(text: &str) -> PolicySpec {
        serde_json::from_str(text).expect("a well-formed document")
    }

    fn parse_error(text: &str) -> String {
        serde_json::from_str::<PolicySpec>(text)
            .expect_err("a refused document")
            .to_string()
    }

    fn spec_with_score(score: &str) -> PolicySpec {
        doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"filter":["healthy"],"score":[{score}],"pick":"best"}}}}}}"#
        ))
    }

    fn spec_with_pick(pick: &str) -> PolicySpec {
        doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"score":["least-load"],"pick":{pick}}}}}}}"#
        ))
    }

    /// Two profiles and an allow-listed header, for the binding tests.
    fn spec_with_binding(binding: &str) -> PolicySpec {
        doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","view":{{"headers":["x-session-id"]}},"profiles":{{"p":{{"pick":"best"}},"q":{{"pick":"random"}}}},"bindings":[{binding}]}}"#
        ))
    }

    fn spec_with_view(view: &str) -> PolicySpec {
        doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","view":{view},"profiles":{{"p":{{"pick":"best"}}}}}}"#
        ))
    }

    fn not_in_build(spec: &PolicySpec) -> (String, &'static str) {
        match spec.validate() {
            Err(SpecError::NotInBuild { path, what }) => (path, what),
            other => panic!("validated as {other:?}"),
        }
    }

    fn bad_field(spec: &PolicySpec) -> String {
        match spec.validate() {
            Err(SpecError::BadField { path, .. }) => path,
            other => panic!("validated as {other:?}"),
        }
    }

    #[test]
    fn every_preset_validates_and_declares_its_needs() {
        for name in PRESETS {
            let spec = PolicySpec::preset(name).expect(name);
            let engine = spec.validate().expect(name);
            let load_aware = ["least_load", "power_of_two", "cache_aware"].contains(&name);
            assert_eq!(engine.needs().loads, load_aware, "{name}");
            let ring = ["consistent_hashing", "prefix_hash"].contains(&name);
            assert_eq!(engine.needs().ring, ring, "{name}");
            // A request's block hashes are fed where a stage reads them: the
            // affinity stage and the ring's prefix form.
            let hashes = ["cache_aware", "prefix_hash"].contains(&name);
            assert_eq!(engine.needs().token_hashes, hashes, "{name}");
            assert_eq!(engine.needs().events, name == "cache_aware", "{name}");
            // The affinity column is fed where an affinity stage reads it
            // and nowhere else: a preset without one declares no column.
            assert_eq!(engine.needs().affinity, name == "cache_aware", "{name}");
            assert_eq!(engine.needs().headers, name == "manual", "{name}");
            assert!(!engine.needs().active_sessions, "{name}");
            assert!(!engine.needs().session_in_flight, "{name}");
            assert!(engine.profile_for("any", Phase::Single).is_some(), "{name}");
            // One profile bound to every model and phase: the per-phase and
            // per-binding reads are the summary.
            for phase in [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode] {
                assert_eq!(engine.needs_for_phase(phase), engine.needs(), "{name}");
                assert_eq!(engine.needs_for("any", phase), engine.needs(), "{name}");
            }
        }
        assert_eq!(
            PolicySpec::preset("no_such_policy").err(),
            Some(SpecError::UnknownPreset("no_such_policy".into()))
        );
    }

    /// The needs are read per binding and per phase: a pd-pair binding's
    /// prefill side reads what the prefill profile's stages read, its decode
    /// side the decode profile's, and a phase's fold spans every binding
    /// that covers it.
    #[test]
    fn the_needs_are_read_per_binding_and_per_phase() {
        let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "default": { "filter": ["healthy"], "pick": "round-robin" },
            "prefill": { "filter": ["healthy", "has-capacity"], "score": ["least-load"], "pick": "best" },
            "decode": { "filter": ["healthy", "has-room"], "score": ["session-affinity"], "pick": "best" }
          },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
        let engine = doc(text).validate().expect("validates");
        let loads = Needs {
            loads: true,
            ..Needs::default()
        };
        let headers = Needs {
            headers: true,
            ..Needs::default()
        };
        assert_eq!(engine.needs(), loads.or(headers));
        assert_eq!(engine.needs_for("pd/x", Phase::Prefill), loads);
        assert_eq!(engine.needs_for("pd/x", Phase::Decode), headers);
        // The pair binding covers neither phase; the catch-all's profile
        // reads nothing.
        assert_eq!(engine.needs_for("pd/x", Phase::Single), Needs::default());
        assert_eq!(engine.needs_for("other", Phase::Prefill), Needs::default());
        // Per phase: the pair binding and the catch-all, whose empty phase
        // list covers every phase.
        assert_eq!(engine.needs_for_phase(Phase::Prefill), loads);
        assert_eq!(engine.needs_for_phase(Phase::Decode), headers);
        assert_eq!(engine.needs_for_phase(Phase::Single), Needs::default());
        assert_eq!(engine.needs_for_phase(Phase::Encode), Needs::default());

        // The shape the gateway's configuration produces: every model's pair
        // phases bound first, the catch-all behind it. The catch-all decides
        // no pair request, so its stages add nothing to the pair phases.
        let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "default": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" },
            "prefill": { "filter": ["healthy", "has-capacity"], "score": ["session-affinity"], "pick": "best" },
            "decode": { "filter": ["healthy", "has-room"], "pick": "round-robin" }
          },
          "bindings": [
            { "match": { "model": "*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
        let engine = doc(text).validate().expect("validates");
        assert_eq!(engine.needs_for_phase(Phase::Prefill), headers);
        assert_eq!(engine.needs_for_phase(Phase::Decode), Needs::default());
        assert_eq!(engine.needs_for_phase(Phase::Single), loads);
        assert_eq!(engine.needs_for_phase(Phase::Encode), loads);

        // A decode-phase binding before the pair binding decides the decode
        // requests that come alone; a pair's decode side is decided by the
        // binding its prefill lookup resolves to, with that pair's decode
        // profile. The decode needs carry both. The pair's decode profile
        // is not the first profile in name order, the order the engine
        // holds them in: a decode slot that read the first profile would
        // find the decode-only one and lose the pair's needs.
        let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "decode-only": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" },
            "prefill": { "filter": ["healthy"], "pick": "round-robin" },
            "pair-decode": { "filter": ["healthy"], "score": ["session-affinity"], "pick": "best" },
            "default": { "filter": ["healthy"], "pick": "round-robin" }
          },
          "bindings": [
            { "match": { "model": "*", "phase": ["decode"] }, "profile": "decode-only" },
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "pair-decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
        let engine = doc(text).validate().expect("validates");
        assert_eq!(engine.needs_for("pd/x", Phase::Decode), loads.or(headers));
        assert_eq!(engine.needs_for("other", Phase::Decode), loads);
        assert_eq!(engine.needs_for("pd/x", Phase::Prefill), Needs::default());
        assert_eq!(engine.needs_for_phase(Phase::Decode), loads.or(headers));
        assert_eq!(engine.needs_for_phase(Phase::Prefill), Needs::default());
    }

    /// The needs rows are what a host must feed: a host that feeds, for a
    /// model and a phase, every row whose glob matches the model feeds what
    /// the decision of that model and phase reads, with no copy of the
    /// bindings' order or of the pair's rule. A pair's decode profile is a
    /// decode row of the pair binding's glob, two bindings of one glob write
    /// one row a phase, and nothing to feed is no row.
    #[test]
    fn the_needs_rows_feed_what_every_decision_reads() {
        /// What a host feeds for `model` and `phase`: the union of the rows
        /// whose glob matches.
        fn fed(engine: &Engine, model: &str, phase: Phase) -> Needs {
            engine
                .needs_rows()
                .into_iter()
                .filter(|(glob, at, _)| {
                    let matches = match (glob.strip_prefix('*'), glob.strip_suffix('*')) {
                        _ if *glob == "*" => true,
                        (Some(suffix), _) => model.ends_with(suffix),
                        (_, Some(prefix)) => model.starts_with(prefix),
                        _ => *glob == model,
                    };
                    matches && *at == phase
                })
                .fold(Needs::default(), |fed, (_, _, needs)| fed.or(needs))
        }
        let loads = Needs {
            loads: true,
            ..Needs::default()
        };
        let headers = Needs {
            headers: true,
            ..Needs::default()
        };
        let phases = [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode];

        // A decode-phase binding before a pair binding, and a catch-all that
        // reads nothing: the pair's decode profile is a row of the pair's
        // glob, and the catch-all writes none.
        let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "decode-only": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" },
            "prefill": { "filter": ["healthy"], "pick": "round-robin" },
            "pair-decode": { "filter": ["healthy"], "score": ["session-affinity"], "pick": "best" },
            "default": { "filter": ["healthy"], "pick": "round-robin" }
          },
          "bindings": [
            { "match": { "model": "*", "phase": ["decode"] }, "profile": "decode-only" },
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "pair-decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
        let engine = doc(text).validate().expect("validates");
        assert_eq!(
            engine.needs_rows(),
            [
                ("*", Phase::Decode, loads),
                ("pd/*", Phase::Decode, headers)
            ]
        );
        for model in ["pd/x", "other"] {
            for phase in phases {
                assert_eq!(
                    fed(&engine, model, phase),
                    engine.needs_for(model, phase),
                    "{model} {phase:?}"
                );
            }
        }

        // Two bindings of one glob write one row a phase. The first decides
        // the single phase and reads the headers there: the union feeds the
        // loads over its word, and never under it.
        let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "keyed": { "filter": ["healthy"], "score": ["session-affinity"], "pick": "best" },
            "loaded": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" }
          },
          "bindings": [
            { "match": { "model": "m", "phase": ["single"] }, "profile": "keyed" },
            { "match": { "model": "m" }, "profile": "loaded" }
          ]
        }"#;
        let engine = doc(text).validate().expect("validates");
        assert_eq!(engine.needs_for("m", Phase::Single), headers);
        assert_eq!(
            engine.needs_rows(),
            phases.map(|phase| match phase {
                Phase::Single => ("m", phase, headers.or(loads)),
                _ => ("m", phase, loads),
            })
        );
    }

    #[test]
    fn a_document_round_trips_and_names_its_mistakes() {
        let text = r#"{
          "apiVersion": "smg.policy/v1",
          "generation": 3,
          "profiles": {
            "default": { "filter": ["healthy", "not-overloaded"], "score": [{"least-load": {"weight": 0.3}}, "session-affinity"], "pick": "best" },
            "prefill": { "filter": ["healthy", "has-capacity"], "pick": "round-robin" },
            "decode": { "filter": ["healthy", "has-room"], "score": ["least-requests"], "pick": "best" }
          },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
        let spec = doc(text);
        let engine = spec.validate().expect("validates");
        assert_eq!(engine.generation(), 3);
        assert!(engine.needs().loads && engine.needs().headers);
        let (_, prefill) = engine
            .profile_for("pd/x", Phase::Prefill)
            .expect("pd binding");
        assert_eq!(prefill.name, "prefill");
        let (_, decode) = engine
            .profile_for("pd/x", Phase::Decode)
            .expect("pd binding");
        assert_eq!(decode.name, "decode");
        let (_, single) = engine.profile_for("chat", Phase::Single).expect("default");
        assert_eq!(single.name, "default");
        let again: PolicySpec =
            serde_json::from_str(&serde_json::to_string(&spec).expect("serializes"))
                .expect("round trip");
        assert_eq!(again, spec);

        let mut bad = spec.clone();
        bad.profiles.get_mut("default").expect("default").pick = "best-of-all".into();
        assert_eq!(
            bad.validate().err(),
            Some(SpecError::UnknownPicker {
                profile: "default".into(),
                name: "best-of-all".into()
            })
        );
        let mut unknown_filter = spec.clone();
        unknown_filter
            .profiles
            .get_mut("default")
            .expect("default")
            .filter
            .push("wire-compatible".into());
        // The catalog names no filter of a gateway's wires: the deciders
        // read them before any profile runs, and a profile that lists one
        // is refused by its name.
        assert_eq!(
            unknown_filter.validate().err(),
            Some(SpecError::UnknownFilter {
                profile: "default".into(),
                name: "wire-compatible".into()
            })
        );
        let mut unbound = spec.clone();
        unbound.bindings[1].profile = Some("missing".into());
        assert_eq!(
            unbound.validate().err(),
            Some(SpecError::UnknownProfile {
                index: 1,
                name: "missing".into()
            })
        );
        let mut wrong = spec;
        wrong.api_version = "smg.policy/v0".into();
        assert!(matches!(wrong.validate(), Err(SpecError::ApiVersion(_))));
    }

    #[test]
    fn a_kind_line_is_refused_naming_the_decision_and_an_unknown_field_by_name() {
        let with_kind = doc(r#"{ "apiVersion": "smg.policy/v1", "kind": "PolicySpec",
          "profiles": { "p": { "pick": "random" } } }"#);
        let err = with_kind.validate().expect_err("a kind line is refused");
        assert_eq!(err, SpecError::Kind("PolicySpec".into()));
        assert!(err.to_string().contains("no kind field"), "{err}");
        let serialized = serde_json::to_string(&with_kind).expect("serializes");
        assert!(!serialized.contains("kind"), "{serialized}");

        let err = parse_error(
            r#"{ "apiVersion": "smg.policy/v1", "plugins": [],
          "profiles": { "p": { "pick": "random" } } }"#,
        );
        assert!(err.contains("unknown field `plugins`"), "{err}");
        let err = parse_error(
            r#"{ "apiVersion": "smg.policy/v1",
          "profiles": { "p": { "pick": "random" } },
          "bindings": [ { "profile": "p", "fallback": "p" } ] }"#,
        );
        assert!(err.contains("unknown field `fallback`"), "{err}");
        let err = parse_error(
            r#"{ "apiVersion": "smg.policy/v1",
          "profiles": { "p": { "pick": "random" } },
          "bindings": [ { "profile": "p", "sessions": { "ttl": 600 } } ] }"#,
        );
        assert!(err.contains("unknown field `ttl`"), "{err}");
    }

    /// A key written twice in one of the document's maps is refused by name
    /// where it is read: the profiles, the index and a binding's profiles.
    /// A JSON document ran the last of the two before.
    #[test]
    fn a_map_key_written_twice_is_refused_by_name() {
        let documents = [
            (
                r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"random"},"p":{"pick":"round-robin"}}}"#,
                "p",
            ),
            (
                r#"{"apiVersion":"smg.policy/v1","index":{"a*":{},"a*":{"ttl_s":1}},"profiles":{"p":{"pick":"random"}}}"#,
                "a*",
            ),
            (
                r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"random"}},"bindings":[{"match":{"model":"*"},"profiles":{"single":"p","single":"p"}}]}"#,
                "single",
            ),
        ];
        for (document, key) in documents {
            let refused = parse_error(document);
            assert!(
                refused.contains(&format!("the key `{key}` is written twice")),
                "{refused}"
            );
        }
        // Each key once reads as before.
        let read = doc(
            r#"{"apiVersion":"smg.policy/v1","index":{"a*":{}},"profiles":{"p":{"pick":"random"},"q":{"pick":"round-robin"}},"bindings":[{"match":{"model":"*"},"profiles":{"single":"p"}}]}"#,
        );
        assert_eq!(read.profiles.len(), 2);
        assert_eq!(read.index.len(), 1);
        assert_eq!(read.bindings[0].profiles["single"], "p");
    }

    /// The YAML form of the three maps is refused in the same words: its
    /// reader hands a key written twice to the map's own read, which kept
    /// the last of the two before.
    #[test]
    fn a_map_key_written_twice_in_yaml_is_refused_by_name() {
        let documents = [
            (
                "apiVersion: smg.policy/v1\nprofiles:\n  p: {pick: random}\n  p: {pick: round-robin}\n",
                "p",
            ),
            (
                "apiVersion: smg.policy/v1\nindex:\n  \"a*\": {}\n  \"a*\": {ttl_s: 1}\nprofiles:\n  p: {pick: random}\n",
                "a*",
            ),
            (
                "apiVersion: smg.policy/v1\nprofiles:\n  p: {pick: random}\nbindings:\n  - match: {model: \"*\"}\n    profiles: {single: p, single: p}\n",
                "single",
            ),
        ];
        for (document, key) in documents {
            let refused = serde_yaml::from_str::<PolicySpec>(document)
                .expect_err("a key written twice")
                .to_string();
            assert!(
                refused.contains(&format!("the key `{key}` is written twice")),
                "{refused}"
            );
        }
    }

    #[test]
    fn the_pair_presets_put_the_admission_filters_in_front_of_each_side() {
        let spec = PolicySpec::pair_presets("least_load", "power_of_two").expect("pair");
        assert!(spec.profiles["prefill"]
            .filter
            .contains(&"has-capacity".to_string()));
        assert!(spec.profiles["decode"]
            .filter
            .contains(&"has-room".to_string()));
        let engine = spec.validate().expect("validates");
        let binding = engine.binding("m", Phase::Prefill).expect("bound");
        assert_eq!(binding.plan, Plan::PdPair);
        assert!(binding.decode_profile.is_some());
    }

    #[test]
    fn a_plan_names_every_side_and_refuses_a_side_outside_it() {
        let mut spec = PolicySpec::pair_presets("least_load", "least_load").expect("pair");
        spec.bindings[0].profiles.remove("decode");
        assert_eq!(
            spec.validate().err(),
            Some(SpecError::SideUnnamed {
                index: 0,
                plan: "pd-pair",
                side: "decode"
            })
        );
        let mut one_profile = PolicySpec::pair_presets("least_load", "least_load").expect("pair");
        one_profile.bindings[0].profile = Some("prefill".into());
        assert_eq!(
            one_profile.validate().err(),
            Some(SpecError::ProfileUnderPlan {
                index: 0,
                plan: "pd-pair"
            })
        );
        let mut encode_under_pair =
            PolicySpec::pair_presets("least_load", "least_load").expect("pair");
        encode_under_pair.bindings[0]
            .profiles
            .insert("encode".into(), "prefill".into());
        assert_eq!(
            encode_under_pair.validate().err(),
            Some(SpecError::SideNotInPlan {
                index: 0,
                plan: "pd-pair",
                side: "encode".into()
            })
        );
        let mut odd_side = PolicySpec::pair_presets("least_load", "least_load").expect("pair");
        odd_side.bindings[0]
            .profiles
            .insert("tail".into(), "prefill".into());
        assert_eq!(
            odd_side.validate().err(),
            Some(SpecError::UnknownSide {
                index: 0,
                side: "tail".into()
            })
        );
        // A side under the single plan was read and ignored before; it is
        // refused now.
        let under_single = spec_with_binding(r#"{"profile":"p","profiles":{"decode":"q"}}"#);
        assert_eq!(
            under_single.validate().err(),
            Some(SpecError::SideNotInPlan {
                index: 0,
                plan: "single",
                side: "decode".into()
            })
        );
        let twice = spec_with_binding(r#"{"profile":"p","profiles":{"single":"q"}}"#);
        assert_eq!(
            twice.validate().err(),
            Some(SpecError::ProfileTwice { index: 0 })
        );
        let via_single_side = spec_with_binding(r#"{"profiles":{"single":"q"}}"#);
        let mut two_profiles_no_bindings = spec_with_binding(r#"{"profile":"p"}"#);
        two_profiles_no_bindings.bindings.clear();
        assert_eq!(
            two_profiles_no_bindings.validate().err(),
            Some(SpecError::NoBindings { profiles: 2 })
        );
        let mut with_deferred_stage = two_profiles_no_bindings.clone();
        with_deferred_stage
            .profiles
            .get_mut("q")
            .expect("profile q")
            .score
            .push(ScoreSpec::from("kv-pressure"));
        assert_eq!(
            with_deferred_stage.validate().err(),
            Some(SpecError::NoBindings { profiles: 2 }),
            "a shape refusal outranks what the build lacks"
        );
        two_profiles_no_bindings.profiles.remove("q");
        assert!(two_profiles_no_bindings.validate().is_ok());
        let engine = via_single_side
            .validate()
            .expect("the single side names the profile");
        assert_eq!(
            engine
                .profile_for("m", Phase::Single)
                .expect("bound")
                .1
                .name,
            "q"
        );
        let err = parse_error(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"best"}},"bindings":[{"plan":"triple","profile":"p"}]}"#,
        );
        assert!(err.contains("unknown variant `triple`"), "{err}");
    }

    /// The document's `bounded_load` reaches the scorer: a decision through
    /// the validated engine passes a loaded owner on, where the plain ring
    /// would keep it.
    #[test]
    fn the_document_s_bounded_load_reaches_the_consistent_hash_scorer() {
        use crate::{
            framework::Scratch,
            view::{Candidate, Constraints, Health, Identity, PoolFacts, RequestFacts, WorkerId},
        };
        let engine = doc(
            r#"{"apiVersion": "smg.policy/v1", "profiles": {"p": {"filter": ["healthy"],
                "score": [{"consistent-hash": {"bounded_load": {"factor": 1.0, "abs_margin": 0}}}],
                "pick": "best"}}}"#,
        )
        .validate()
        .expect("a bounded consistent-hash profile validates");
        let pool: Vec<Candidate> = (0..4u32)
            .map(|i| Candidate {
                position: i,
                id: WorkerId(100 + u64::from(i)),
                health: Health {
                    healthy: true,
                    ..Health::default()
                },
                ..Candidate::default()
            })
            .collect();
        let facts = RequestFacts {
            model: "m",
            session_key: Identity::hashed(5),
            seed: 1,
            ..RequestFacts::default()
        };
        let pool_facts = PoolFacts {
            pool_size: 4,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let mut scratch = Scratch::new();
        let first = |decision: &crate::Decision| {
            decision
                .pick()
                .expect("a pick")
                .targets
                .as_slice()
                .first()
                .expect("a target")
                .position
        };
        let plain = engine.decide(&facts, &pool_facts, &pool, &cx, &mut scratch);
        let owner = first(&plain);
        let mut loaded = pool.clone();
        loaded[owner as usize].in_flight = 8;
        let bounded = engine.decide(&facts, &pool_facts, &loaded, &cx, &mut scratch);
        assert_ne!(
            first(&bounded),
            owner,
            "the document's rule passes the loaded owner on"
        );
    }

    /// The `epd` plan binds its three sides: the encode phase resolves to
    /// the encode profile, the pair phases to theirs, and a single request
    /// to none; a side left unnamed or a phase list leaving one out is
    /// refused by name.
    #[test]
    fn the_epd_plan_binds_its_three_sides() {
        let epd = spec_with_binding(
            r#"{"plan":"epd","profiles":{"encode":"q","prefill":"p","decode":"p"}}"#,
        );
        let engine = epd.validate().expect("the epd plan is in this build");
        let binding = engine.binding("m", Phase::Encode).expect("bound");
        assert_eq!(binding.plan, Plan::Epd { encode: 1 });
        assert_eq!((binding.profile, binding.decode_profile), (0, Some(0)));
        let side = |phase: Phase| engine.profile_for("m", phase).map(|(_, p)| p.name.clone());
        assert_eq!(side(Phase::Encode), Some("q".to_string()));
        assert_eq!(side(Phase::Prefill), Some("p".to_string()));
        assert_eq!(side(Phase::Decode), Some("p".to_string()));
        assert_eq!(side(Phase::Single), None, "no side of the plan");
        let presets = PolicySpec::epd_presets("consistent_hashing", "least_load", "round_robin")
            .expect("the epd presets");
        assert_eq!(presets.bindings[0].plan, Some(PlanName::Epd));
        assert_eq!(presets.profiles["encode"].score.len(), 1);
        assert!(presets.profiles["prefill"]
            .filter
            .contains(&"has-capacity".to_string()));
        let engine = presets.validate().expect("validates");
        assert!(engine.needs().ring && engine.needs().loads);
        assert!(matches!(
            engine.binding("m", Phase::Prefill).map(|b| b.plan),
            Some(Plan::Epd { .. })
        ));
        let no_encode =
            spec_with_binding(r#"{"plan":"epd","profiles":{"prefill":"p","decode":"p"}}"#);
        assert_eq!(
            no_encode.validate().err(),
            Some(SpecError::SideUnnamed {
                index: 0,
                plan: "epd",
                side: "encode"
            })
        );
        let two_phases = spec_with_binding(
            r#"{"match":{"phase":["prefill","decode"]},"plan":"epd","profiles":{"encode":"q","prefill":"p","decode":"p"}}"#,
        );
        assert_eq!(
            two_phases.validate().err(),
            Some(SpecError::PlanPhases {
                index: 0,
                plan: "epd",
                side: "encode"
            })
        );
    }

    #[test]
    fn a_sampling_picker_without_a_scorer_is_refused() {
        let mut spec = PolicySpec::preset("power_of_two").expect("preset");
        spec.profiles
            .get_mut("power_of_two")
            .expect("profile")
            .score
            .clear();
        assert_eq!(
            spec.validate().err(),
            Some(SpecError::PickerNeedsScorer {
                profile: "power_of_two".into(),
                picker: "power-of-two".into()
            })
        );
        let count_first_form = doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"power-of-two-count-first"}}}"#,
        );
        assert_eq!(
            count_first_form.validate().err(),
            Some(SpecError::PickerNeedsScorer {
                profile: "p".into(),
                picker: "power-of-two-count-first".into()
            })
        );
        let requests_form = doc(r#"{ "apiVersion": "smg.policy/v1",
          "profiles": { "p2": { "score": ["least-requests"], "pick": "power-of-two" } } }"#);
        let engine = requests_form
            .validate()
            .expect("the requests form validates");
        assert!(!engine.needs().loads);
    }

    #[test]
    fn a_spec_without_bindings_binds_its_first_profile_to_everything() {
        let spec = doc(r#"{ "apiVersion": "smg.policy/v1",
          "profiles": { "only": { "filter": ["healthy"], "pick": "passthrough" } } }"#);
        let engine = spec.validate().expect("validates");
        let (_, profile) = engine
            .profile_for("anything", Phase::Decode)
            .expect("bound");
        assert_eq!(profile.name, "only");
        assert_eq!(
            PolicySpec {
                profiles: BTreeMap::new(),
                ..spec
            }
            .validate()
            .err(),
            Some(SpecError::Empty)
        );
    }

    #[test]
    fn a_weight_or_parameter_out_of_range_is_refused_by_its_path() {
        let cases = [
            (r#"{"least-load":{"weight":-1}}"#, "weight"),
            (r#"{"least-load":{"weight":0}}"#, "weight"),
            (
                r#"{"least-load":{"kv_pressure_weight":-0.5}}"#,
                "kv_pressure_weight",
            ),
            (
                r#"{"least-load":{"mean_prefill_tokens":0}}"#,
                "mean_prefill_tokens",
            ),
            (
                r#"{"least-load":{"default_throughput":0}}"#,
                "default_throughput",
            ),
        ];
        for (score, name) in cases {
            assert_eq!(
                bad_field(&spec_with_score(score)),
                format!("profiles.p.score[0].least-load.{name}"),
                "{score}"
            );
        }
    }

    #[test]
    fn a_misspelt_or_unknown_scorer_is_refused_by_name() {
        // A key that is no built-in's parses, since a catalog may add the
        // name; the validator refuses it by name, whatever is written under
        // it.
        match spec_with_score(r#"{"least_load":{"kv_pressure_weight":5}}"#).validate() {
            Err(SpecError::UnknownScorer { name, .. }) => assert_eq!(name, "least_load"),
            other => panic!("validated as {other:?}"),
        }
        // With parameters, a built-in's one-key map goes to its parameter
        // struct whole, so the message names the key instead of "no variant
        // matched".
        let err = parse_error(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":[{"least-requests":{"kv_pressure_weight":5}}],"pick":"best"}}}"#,
        );
        assert!(err.contains("unknown field `kv_pressure_weight`"), "{err}");
        let err = parse_error(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":[{"least-load":{"weight":1},"least-requests":{}}],"pick":"best"}}}"#,
        );
        assert!(err.contains("single key"), "{err}");
        match spec_with_score(r#""least_load""#).validate() {
            Err(SpecError::UnknownScorer { name, .. }) => assert_eq!(name, "least_load"),
            other => panic!("validated as {other:?}"),
        }
    }

    #[test]
    fn a_model_glob_is_a_single_leading_or_trailing_star() {
        for model in ["llama*-chat", "*-7b*", "", "**"] {
            let spec = spec_with_binding(&format!(
                r#"{{"match":{{"model":{model:?}}},"profile":"p"}}"#
            ));
            assert!(
                matches!(
                    spec.validate(),
                    Err(SpecError::BadModelGlob { index: 0, .. })
                ),
                "{model:?}"
            );
        }
        for model in ["*", "llama*", "*-chat", "llama-3"] {
            let spec = spec_with_binding(&format!(
                r#"{{"match":{{"model":{model:?}}},"profile":"p"}}"#
            ));
            assert!(spec.validate().is_ok(), "{model:?}");
        }
    }

    #[test]
    fn a_pair_binding_names_both_phases_or_none() {
        let spec = spec_with_binding(
            r#"{"match":{"phase":["prefill"]},"plan":"pd-pair","profiles":{"prefill":"p","decode":"p"}}"#,
        );
        assert_eq!(
            spec.validate().err(),
            Some(SpecError::PlanPhases {
                index: 0,
                plan: "pd-pair",
                side: "decode"
            })
        );
        let both = spec_with_binding(
            r#"{"match":{"phase":["prefill","decode"]},"plan":"pd-pair","profiles":{"prefill":"p","decode":"p"}}"#,
        );
        assert!(both.validate().is_ok());
    }

    #[test]
    fn the_view_refuses_denied_duplicate_and_malformed_headers() {
        assert_eq!(
            bad_field(&spec_with_view(r#"{"headers":["Authorization"]}"#)),
            "view.headers[0]"
        );
        assert_eq!(
            spec_with_view(r#"{"headers":["x-a","cookie"]}"#)
                .validate()
                .err(),
            Some(SpecError::DeniedHeader {
                path: "view.headers[1]".into(),
                name: "cookie".into()
            })
        );
        assert_eq!(
            spec_with_view(r#"{"headers":["x-a","x-a"]}"#)
                .validate()
                .err(),
            Some(SpecError::Duplicate {
                path: "view.headers".into(),
                name: "x-a".into()
            })
        );
        let tenant = spec_with_view(r#"{"headers":["x-smg-tenant-id"]}"#);
        assert!(tenant.validate().is_ok());
        assert_eq!(
            tenant.validate_with(&["X-SMG-Tenant-Id"]).err(),
            Some(SpecError::DeniedHeader {
                path: "view.headers[0]".into(),
                name: "x-smg-tenant-id".into()
            })
        );
        let engine = spec_with_view(r#"{"headers":["x-session-id"]}"#)
            .validate()
            .expect("an allow-listed header");
        assert!(engine.needs().headers);
    }

    #[test]
    fn a_generation_is_refused_for_a_header_the_host_denies() {
        let tenant = spec_with_view(r#"{"headers":["x-smg-tenant-id"]}"#);
        assert!(AppliedSpec::validate(&tenant).is_ok());
        assert_eq!(
            AppliedSpec::validate_with(&tenant, &["X-SMG-Tenant-Id"]).err(),
            Some(SpecError::DeniedHeader {
                path: "view.headers[0]".into(),
                name: "x-smg-tenant-id".into()
            })
        );
    }

    #[test]
    fn a_generation_declares_the_needs_of_its_engine() {
        for name in PRESETS {
            let document = PolicySpec::preset(name).expect(name);
            let applied = AppliedSpec::validate(&document).expect(name);
            assert_eq!(applied.needs(), applied.engine().needs(), "{name}");
        }
        let cache_aware = PolicySpec::preset("cache_aware").expect("a preset of this build");
        let needs = AppliedSpec::validate(&cache_aware)
            .expect("a preset validates")
            .needs();
        assert!(needs.loads && needs.events && needs.affinity);
    }

    #[test]
    fn the_view_lists_at_most_256_headers() {
        let with_headers = |count: usize| {
            let names: Vec<String> = (0..count).map(|i| format!("\"x-h{i}\"")).collect();
            spec_with_view(&format!(r#"{{"headers":[{}]}}"#, names.join(",")))
        };
        let engine = with_headers(MAX_HEADERS)
            .validate()
            .expect("256 headers fill the byte of positions");
        assert!(engine.needs().headers);
        assert_eq!(
            with_headers(MAX_HEADERS + 1).validate().err(),
            Some(SpecError::TooManyHeaders {
                count: 257,
                max: 256
            })
        );
    }

    #[test]
    fn the_view_checks_block_size_boundaries_and_the_hash_cap() {
        assert_eq!(
            bad_field(&spec_with_view(r#"{"block_size":0}"#)),
            "view.block_size"
        );
        assert_eq!(
            bad_field(&spec_with_view(r#"{"boundaries":[0,16]}"#)),
            "view.boundaries[0]"
        );
        for boundaries in ["[256,128]", "[256,256]"] {
            assert_eq!(
                bad_field(&spec_with_view(&format!(
                    r#"{{"boundaries":{boundaries}}}"#
                ))),
                "view.boundaries[1]",
                "{boundaries}"
            );
        }
        assert_eq!(
            bad_field(&spec_with_view(r#"{"max_block_hashes":0}"#)),
            "view.max_block_hashes"
        );
        // One frame of the Decide call carries 8,190 hashes: that cap is
        // taken, and one above it is refused by the field's name.
        let cap = |hashes: u32| spec_with_view(&format!(r#"{{"max_block_hashes":{hashes}}}"#));
        assert!(cap(MAX_BLOCK_HASHES).validate().is_ok());
        let refused = cap(MAX_BLOCK_HASHES + 1)
            .validate()
            .expect_err("a cap one frame cannot carry");
        assert_eq!(
            refused.to_string(),
            "view.max_block_hashes = 8191 is not at most 8190, the hashes one Decide frame carries"
        );
        let spec = spec_with_view(r#"{"block_size":128,"boundaries":[256,1024,4096]}"#);
        assert!(spec.validate().is_ok());
        assert_eq!(spec.view.max_block_hashes, 256);
    }

    #[test]
    fn the_view_lists_are_capped_named_and_listed_once() {
        let list = |count: usize| {
            let names: Vec<String> = (0..count).map(|n| format!("\"n{n}\"")).collect();
            format!("[{}]", names.join(","))
        };
        let caps = [
            ("labels", MAX_LABELS),
            ("zones", MAX_ZONES),
            ("engines", MAX_ENGINES),
        ];
        for (field, max) in caps {
            let at_cap = spec_with_view(&format!(r#"{{"{field}":{}}}"#, list(max)));
            assert!(at_cap.validate().is_ok(), "{field}");
            let over = spec_with_view(&format!(r#"{{"{field}":{}}}"#, list(max + 1)));
            assert!(
                matches!(
                    over.validate(),
                    Err(SpecError::TooManyListed { path, count, max: cap })
                        if path == format!("view.{field}") && count == max + 1 && cap == max
                ),
                "{field}"
            );
            let unnamed = spec_with_view(&format!(r#"{{"{field}":["a",""]}}"#));
            assert_eq!(bad_field(&unnamed), format!("view.{field}[1]"));
            let twice = spec_with_view(&format!(r#"{{"{field}":["a","b","a"]}}"#));
            assert!(
                matches!(
                    twice.validate(),
                    Err(SpecError::Duplicate { path, name })
                        if path == format!("view.{field}") && name == "a"
                ),
                "{field}"
            );
        }
        // A label is a key, or a key and a value split at the first `=`:
        // neither side is empty, and the value may hold a `=` of its own.
        for label in ["=gold", "tier="] {
            let empty = spec_with_view(&format!(r#"{{"labels":["tier","{label}"]}}"#));
            assert_eq!(bad_field(&empty), "view.labels[1]", "{label}");
        }
        let spec = spec_with_view(r#"{"labels":["tier","tier=gold","note=a=b"]}"#);
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn a_view_that_declares_no_list_is_written_as_before() {
        let written = serde_json::to_string(&ViewSpec::default()).expect("the view serializes");
        for field in ["labels", "zones", "engines"] {
            assert!(!written.contains(field), "{written}");
        }
        let spec =
            spec_with_view(r#"{"labels":["tier=gold"],"zones":["east"],"engines":["alpha"]}"#);
        let written = serde_json::to_string(&spec.view).expect("the view serializes");
        let read: ViewSpec = serde_json::from_str(&written).expect("the view reads back");
        assert_eq!(read, spec.view);
        assert_eq!(
            (read.labels.len(), read.zones.len(), read.engines.len()),
            (1, 1, 1)
        );
    }

    #[test]
    fn the_lists_read_a_worker_by_position_and_by_bit() {
        let labels = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
            let owned = |(key, value): &(&str, &str)| ((*key).to_string(), (*value).to_string());
            pairs.iter().map(owned).collect()
        };
        let spec = spec_with_view(
            r#"{"labels":["tier","tier=gold","note=a=b"],"zones":["east","west"],"engines":["alpha","beta"]}"#,
        );
        let applied = AppliedSpec::validate(&spec).expect("the document validates");
        let lists = applied.lists();
        assert!(!lists.is_empty());
        let held = labels(&[("tier", "gold"), ("note", "a=b")]);
        let gold = lists.listed("beta", Some("west"), &held);
        assert_eq!((gold.engine_kind, gold.zone, gold.labels), (2, 2, 0b111));
        assert!(!gold.engine_outside && !gold.zone_outside);
        let held = labels(&[("tier", "silver"), ("note", "a")]);
        let silver = lists.listed("alpha", Some("east"), &held);
        assert_eq!(
            (silver.engine_kind, silver.zone, silver.labels),
            (1, 1, 0b001)
        );
        // A bit is the selector's place in the list, whatever the keys'
        // order: two selectors of one key stand apart here.
        let mixed = spec_with_view(r#"{"labels":["z","b=1","m","b","a=2","q=x"]}"#);
        let mixed = AppliedSpec::validate(&mixed).expect("the document validates");
        let held = labels(&[("a", "2"), ("b", "1"), ("c", "9"), ("m", ""), ("q", "y")]);
        assert_eq!(mixed.lists().listed("", None, &held).labels, 0b01_1110);
        // A name the list does not hold reads 0 and says so. No name reads
        // 0 and is not outside.
        let stray = lists.listed("gamma", Some("north"), &labels(&[]));
        assert_eq!((stray.engine_kind, stray.zone, stray.labels), (0, 0, 0));
        assert!(stray.engine_outside && stray.zone_outside);
        for zone in [None, Some("")] {
            assert_eq!(lists.listed("", zone, &labels(&[])), Listed::default());
        }
        // No list: every worker reads 0 and none is outside.
        let none = AppliedSpec::validate(&spec_with_view("{}")).expect("the document validates");
        assert!(none.lists().is_empty() && AppliedSpec::unbound().lists().is_empty());
        let read = none.lists().listed("beta", Some("west"), &held);
        assert_eq!(read, Listed::default());
    }

    #[test]
    fn the_index_map_is_per_model_glob_and_a_hash_index_needs_boundaries() {
        let with_index = |index: &str, view: &str| {
            doc(&format!(
                r#"{{"apiVersion":"smg.policy/v1","view":{view},"index":{index},"profiles":{{"p":{{"pick":"best"}}}}}}"#
            ))
        };
        assert_eq!(bad_field(&with_index(r#"{"a*b":{}}"#, "{}")), "index.a*b");
        // Each count is above zero, refused by its own name, the first
        // written in the type's order where two are zero.
        for count in ["ttl_s", "max_size", "sweep_s"] {
            let index = format!(r#"{{"*":{{"{count}":0}}}}"#);
            assert_eq!(
                bad_field(&with_index(&index, "{}")),
                format!("index.*.{count}")
            );
        }
        assert_eq!(
            bad_field(&with_index(r#"{"*":{"sweep_s":0,"ttl_s":0}}"#, "{}")),
            "index.*.ttl_s"
        );
        assert_eq!(
            with_index(r#"{"*":{"kind":"hash"}}"#, "{}")
                .validate()
                .err(),
            Some(SpecError::HashIndexNeedsBoundaries { model: "*".into() })
        );
        let hashed = with_index(
            r#"{"*":{"kind":"hash","ttl_s":300},"chat-*":{"max_size":1024}}"#,
            r#"{"boundaries":[512,2048]}"#,
        );
        assert!(hashed.validate().is_ok());
        assert_eq!(hashed.index["chat-*"].sweep_s, 120);
    }

    /// A zero count of an index key is refused whole: by its path, its
    /// value and its reason.
    #[test]
    fn a_zero_count_of_an_index_key_is_refused_by_path_value_and_reason() {
        for count in ["ttl_s", "max_size", "sweep_s"] {
            let spec = doc(&format!(
                r#"{{"apiVersion":"smg.policy/v1","index":{{"*":{{"{count}":0}}}},"profiles":{{"p":{{"pick":"best"}}}}}}"#
            ));
            assert_eq!(
                spec.validate().err(),
                Some(SpecError::BadField {
                    path: format!("index.*.{count}"),
                    value: "0".to_owned(),
                    reason: "above zero",
                })
            );
        }
    }

    #[test]
    fn the_power_of_two_preset_has_four_forms_by_its_load_metric() {
        let form = |metric| {
            let spec = PolicySpec::power_of_two(metric).expect("a preset");
            let engine = spec.validate().expect("validates");
            assert!(engine.needs().loads, "{metric:?}");
            let profile = spec.profiles["power_of_two"].clone();
            (profile.score, profile.pick.name().to_string())
        };
        let least_load: Vec<ScoreSpec> = vec!["least-load".into()];
        assert_eq!(
            form(PowerOfTwoMetric::ExpectedWait),
            (least_load.clone(), "power-of-two".to_string())
        );
        assert_eq!(
            form(PowerOfTwoMetric::Requests),
            (least_load.clone(), "power-of-two-count-first".to_string())
        );
        assert_eq!(
            form(PowerOfTwoMetric::LeastRequests),
            (least_load, "count-first".to_string())
        );
        let resident: Vec<ScoreSpec> = vec![ScoreStage::ResidentWork(ResidentWorkParams {
            weight: 1.0,
            request_tokens: 2048,
        })
        .into()];
        assert_eq!(
            form(PowerOfTwoMetric::ResidentTokens {
                request_tokens: 2048
            }),
            (resident, "best".to_string())
        );
        // A request priced at nothing is a form too: the tokens alone.
        PolicySpec::power_of_two(PowerOfTwoMetric::ResidentTokens { request_tokens: 0 })
            .expect("a preset")
            .validate()
            .expect("the tokens alone");
    }

    #[test]
    fn every_catalog_scorer_parses_and_the_ones_this_build_lacks_are_refused_by_name() {
        let in_build = [
            "least-load",
            "least-requests",
            "fewest-sessions",
            "session-affinity",
            "prefix-affinity",
            "consistent-hash",
            "kv-pressure",
            "warmup-slice",
            "bucket",
        ];
        for name in SCORERS {
            let bare = spec_with_score(&format!("{name:?}"));
            if in_build.contains(&name) {
                assert!(bare.validate().is_ok(), "{name}");
            } else if name == "resident-work" {
                assert_eq!(bad_field(&bare), "profiles.p.score[0]");
            } else {
                assert_eq!(
                    not_in_build(&bare).0,
                    format!("profiles.p.score[0].{name}"),
                    "{name}"
                );
            }
        }
        // The affinity stages and the bucket stage build with their
        // parameters; the forms a later step carries are refused by name.
        for score in [
            r#"{"kv-pressure":{"spread":0.3,"ceiling":0.9}}"#,
            r#"{"warmup-slice":{"secs":0,"miss_share":0.5}}"#,
            r#"{"prefix-affinity":{"threshold":0.5,"source":"merged","spill":{"abs":8,"rel":2.0}}}"#,
            r#"{"bucket":{"abs_threshold":32,"rel_threshold":1.0001,"adjust_interval_s":5}}"#,
        ] {
            assert!(spec_with_score(score).validate().is_ok(), "{score}");
        }
        let with_params = [
            (r#"{"prefix-affinity":{"decay":0.9}}"#, "prefix-affinity"),
            (
                r#"{"prefix-affinity":{"accounting_ttl_ms":200}}"#,
                "prefix-affinity",
            ),
            (
                r#"{"prefix-affinity":{"spill":{"uncached_credit":true}}}"#,
                "prefix-affinity",
            ),
        ];
        for (score, name) in with_params {
            let (path, what) = not_in_build(&spec_with_score(score));
            assert_eq!(path, format!("profiles.p.score[0].{name}"), "{score}");
            assert!(what.contains(name), "{what}");
        }
        // The resident-work scorer is in the build with its parameter, and
        // reads the load reports.
        let resident = spec_with_score(r#"{"resident-work":{"request_tokens":512}}"#);
        assert!(resident.validate().expect("validates").needs().loads);
        // The consistent-hash scorer is in the build with its bounded-load
        // rule and with its prefix form, which reads the request's block
        // hashes; without the form it asks for none.
        let bounded = spec_with_score(r#"{"consistent-hash":{"bounded_load":{"factor":1.5}}}"#);
        let needs = bounded.validate().expect("validates").needs();
        assert!(needs.ring && !needs.token_hashes && !needs.text_hashes);
        let prefix = spec_with_score(
            r#"{"consistent-hash":{"prefix_tokens":512,"bounded_load":{"factor":1.5,"abs_margin":4}}}"#,
        );
        let needs = prefix.validate().expect("validates").needs();
        assert!(needs.ring && needs.token_hashes && needs.text_hashes);
        // Under a view that names boundaries the prefix ring keys at the
        // deepest one a request reaches, and asks for the same.
        let levelled = doc(
            r#"{"apiVersion":"smg.policy/v1","view":{"boundaries":[256,1024]},"profiles":{"p":{"score":[{"consistent-hash":{"prefix_tokens":512}}],"pick":"best"}}}"#,
        );
        let needs = levelled.validate().expect("validates").needs();
        assert!(needs.ring && needs.token_hashes && needs.text_hashes);
        // A ring with no prefix form reads no boundary and asks for no
        // hashes there.
        let plain = doc(
            r#"{"apiVersion":"smg.policy/v1","view":{"boundaries":[256,1024]},"profiles":{"p":{"score":[{"consistent-hash":{"bounded_load":{"factor":1.5}}}],"pick":"best"}}}"#,
        );
        let needs = plain.validate().expect("validates").needs();
        assert!(needs.ring && !needs.token_hashes && !needs.text_hashes);
    }

    /// The prefix depth is held to the view that carries the hashes: at
    /// least one block of the view's block size, at most the blocks a
    /// request carries.
    #[test]
    fn a_prefix_depth_the_view_cannot_carry_is_refused() {
        let spec = |view: &str, tokens: u32| {
            doc(&format!(
                r#"{{"apiVersion":"smg.policy/v1","view":{view},"profiles":{{"p":{{"score":[{{"consistent-hash":{{"prefix_tokens":{tokens}}}}}],"pick":"best"}}}}}}"#
            ))
        };
        let refused = |view: &str, tokens: u32| match spec(view, tokens).validate() {
            Err(SpecError::BadField { path, reason, .. }) => {
                assert_eq!(
                    path, "profiles.p.score[0].consistent-hash.prefix_tokens",
                    "{view} {tokens}"
                );
                reason
            }
            other => panic!("{view} {tokens}: validated as {other:?}"),
        };
        let under = "at least one block of view.block_size";
        let over = "at most view.max_block_hashes blocks of view.block_size";
        // The default view: blocks of 16 tokens, 256 of them a request.
        for tokens in [16, 256, 4096, 4111] {
            assert!(spec("{}", tokens).validate().is_ok(), "{tokens}");
        }
        assert_eq!(refused("{}", 15), under);
        assert_eq!(refused("{}", 4112), over);
        // The view's own block size and its cap move both edges.
        let view = r#"{"block_size":64,"max_block_hashes":4}"#;
        for tokens in [64, 256, 319] {
            assert!(spec(view, tokens).validate().is_ok(), "{tokens}");
        }
        assert_eq!(refused(view, 63), under);
        assert_eq!(refused(view, 320), over);
        // Under a prefix key each of the view's boundaries is held to the
        // same two edges.
        let boundary = |boundaries: &str| {
            let view = format!(r#"{{"boundaries":{boundaries}}}"#);
            match spec(&view, 256).validate() {
                Err(SpecError::BadField { path, reason, .. }) => (path, reason),
                other => panic!("{boundaries}: validated as {other:?}"),
            }
        };
        assert!(spec(r#"{"boundaries":[16,4096,4111]}"#, 256)
            .validate()
            .is_ok());
        assert_eq!(
            boundary("[8,64]"),
            (
                "view.boundaries[0]".to_string(),
                "at least one block of view.block_size under a consistent-hash prefix key"
            )
        );
        assert_eq!(
            boundary("[64,4112]"),
            (
                "view.boundaries[1]".to_string(),
                "at most view.max_block_hashes blocks of view.block_size under a consistent-hash prefix key"
            )
        );
        // A view that no prefix key reads keeps such boundaries.
        assert!(spec_with_view(r#"{"boundaries":[8,4112]}"#)
            .validate()
            .is_ok());
    }

    #[test]
    fn stage_parameters_are_range_checked_before_the_build_check() {
        let cases = [
            (r#"{"kv-pressure":{"spread":0}}"#, "kv-pressure.spread"),
            (r#"{"warmup-slice":{"share":2}}"#, "warmup-slice.share"),
            (
                r#"{"bucket":{"rel_threshold":0.5}}"#,
                "bucket.rel_threshold",
            ),
            (
                r#"{"consistent-hash":{"prefix_tokens":0}}"#,
                "consistent-hash.prefix_tokens",
            ),
            (
                r#"{"consistent-hash":{"bounded_load":{"factor":0.9}}}"#,
                "consistent-hash.bounded_load.factor",
            ),
            (
                r#"{"prefix-affinity":{"threshold":1.5}}"#,
                "prefix-affinity.threshold",
            ),
            (
                r#"{"prefix-affinity":{"spill":{"rel":0.5}}}"#,
                "prefix-affinity.spill.rel",
            ),
        ];
        for (score, path) in cases {
            assert_eq!(
                bad_field(&spec_with_score(score)),
                format!("profiles.p.score[0].{path}"),
                "{score}"
            );
        }
    }

    /// One profile for every assignment: the `manual` preset's picker
    /// leaves the placement to the binding's rule, the rule names the
    /// classic default, and each other assignment is that one field and
    /// validates. The recorded rows hold where each one places. The built
    /// engine answers each rule with the assignment the document names:
    /// under this picker it places a `delegate` by the fewest in flight,
    /// and a reader of the binding still reads `delegate`.
    #[test]
    fn the_manual_preset_is_one_profile_for_every_assignment() {
        let mut spec = PolicySpec::preset("manual").expect("the preset");
        let pick = spec.profiles.get("manual").map(|p| p.pick.name());
        assert_eq!(pick, Some("manual"));
        let default = spec
            .bindings
            .first()
            .and_then(|b| b.sessions.as_ref())
            .map(|rule| rule.assignment);
        assert_eq!(default, Some(Assignment::Random), "the classic default");
        for assignment in [
            Assignment::Random,
            Assignment::MinLoad,
            Assignment::MinGroup,
            Assignment::Delegate,
        ] {
            let rule = spec
                .bindings
                .first_mut()
                .and_then(|b| b.sessions.as_mut())
                .expect("the preset's sessions rule");
            rule.assignment = assignment;
            let engine = spec.validate().expect("every assignment validates");
            assert_eq!(
                engine.needs().active_sessions,
                assignment == Assignment::MinGroup,
                "{assignment:?}"
            );
            let read_back = engine
                .binding("m", Phase::Single)
                .and_then(|binding| binding.sessions)
                .map(|rule| rule.assignment);
            assert_eq!(read_back, Some(assignment), "read back as declared");
        }
    }

    /// The `manual` preset's whole rule, as the gateway's keyed policy runs
    /// today: the routing-key header as the session key and the view's one
    /// header, no header list of the rule's own, four hours of idle time
    /// swept each minute, a random assignment, no in-flight cap, no session
    /// id issued, the first worker back in the lead.
    #[test]
    fn the_manual_preset_states_its_whole_rule() {
        let spec = PolicySpec::preset("manual").expect("the preset");
        let rule = spec
            .bindings
            .first()
            .and_then(|b| b.sessions.as_ref())
            .expect("the preset's sessions rule");
        assert_eq!(
            *rule,
            SessionsSpec {
                sticky_header: "x-smg-routing-key".into(),
                headers: vec![],
                ttl_s: 14_400,
                sweep_s: 60,
                assignment: Assignment::Random,
                inflight_cap: u16::MAX,
                issue_session_id: false,
                recovery: Recovery::Original,
            }
        );
        assert_eq!(spec.view.headers, ["x-smg-routing-key"]);
    }

    /// The manual picker places by a sessions rule, and a rule runs under
    /// the single plan alone: on a side of a pair or of `epd` the profile is
    /// refused, with the side that names it. The built picker answers, so
    /// any picker that leaves the placement to the rule is refused the same
    /// way.
    #[test]
    fn the_manual_picker_outside_the_single_plan_is_not_in_this_build() {
        let spec = doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"q":{"pick":"random"},"sticky":{"pick":"manual"}},"bindings":[{"plan":"pd-pair","profiles":{"prefill":"q","decode":"sticky"}}]}"#,
        );
        assert_eq!(
            not_in_build(&spec),
            (
                "bindings[0].profiles.decode".to_string(),
                "a picker that places by the sessions rule outside the single plan"
            )
        );
        let epd = doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"q":{"pick":"random"},"sticky":{"pick":"manual"}},"bindings":[{"plan":"epd","profiles":{"encode":"sticky","prefill":"q","decode":"q"}}]}"#,
        );
        assert_eq!(
            not_in_build(&epd),
            (
                "bindings[0].profiles.encode".to_string(),
                "a picker that places by the sessions rule outside the single plan"
            )
        );
        // Under the single plan the same profile validates, here with no
        // rule; the preset is the profile with one.
        let single = doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"sticky":{"pick":"manual"}},"bindings":[{"profile":"sticky"}]}"#,
        );
        assert!(single.validate().is_ok());
    }

    #[test]
    fn the_pickers_of_v1_resolve_or_are_refused_by_name() {
        assert_eq!(
            bad_field(&spec_with_pick(r#""best-with-temperature""#)),
            "profiles.p.pick"
        );
        assert_eq!(
            not_in_build(&spec_with_pick(
                r#"{"best-with-temperature":{"temperature":0.05}}"#
            )),
            (
                "profiles.p.pick".to_string(),
                "the best-with-temperature picker"
            )
        );
        assert_eq!(
            bad_field(&spec_with_pick(
                r#"{"best-with-temperature":{"temperature":-1}}"#
            )),
            "profiles.p.pick.best-with-temperature.temperature"
        );
        let err = parse_error(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":{"best-with-temperature":{"heat":1}}}}}"#,
        );
        assert!(err.contains("unknown field `heat`"), "{err}");
        for name in [
            "best",
            "round-robin",
            "random",
            "power-of-two",
            "count-first",
            "power-of-two-count-first",
            "passthrough",
        ] {
            assert!(
                spec_with_pick(&format!("{name:?}")).validate().is_ok(),
                "{name}"
            );
        }
        // The manual picker leaves the placement to the binding's rule: no
        // scorer of its profile would ever run, so one is refused by name.
        match spec_with_pick(r#""manual""#).validate() {
            Err(SpecError::PickerRunsNoScorer { profile, picker }) => {
                assert_eq!((profile.as_str(), picker.as_str()), ("p", "manual"));
            }
            other => panic!("a scorer beside the manual picker validated as {other:?}"),
        }
        let alone = doc(r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"manual"}}}"#);
        assert!(alone.validate().is_ok());
    }

    #[test]
    fn the_binding_rules_are_checked_in_shape_then_by_the_build() {
        let overload = spec_with_binding(
            r#"{"profile":"p","overload":{"waiting_requests":32,"token_usage":0.9,"shed_when_all_vetoed":true}}"#,
        );
        let thresholds = |spec: &PolicySpec| {
            spec.validate()
                .expect("the binding validates")
                .binding("m", Phase::Single)
                .and_then(|binding| binding.overload)
        };
        assert_eq!(
            thresholds(&overload),
            Some(OverloadThresholds {
                waiting_requests: 32,
                kv_usage_permille: 900,
            }),
            "the block's two values, the usage in permille"
        );
        // A value the block does not name is the default, and so is a
        // binding with no block and the binding of a document that
        // writes none.
        let defaults = OverloadThresholds {
            waiting_requests: OVERLOAD_WAITING_REQUESTS,
            kv_usage_permille: 800,
        };
        assert_eq!(
            thresholds(&spec_with_binding(
                r#"{"profile":"p","overload":{"waiting_requests":3}}"#
            )),
            Some(OverloadThresholds {
                waiting_requests: 3,
                ..defaults
            })
        );
        assert_eq!(
            thresholds(&spec_with_binding(
                r#"{"profile":"p","overload":{"token_usage":0.25}}"#
            )),
            Some(OverloadThresholds {
                kv_usage_permille: 250,
                ..defaults
            })
        );
        // A usage that would round to no permille is held as one: at none,
        // every engine with a row on file would read over it.
        assert_eq!(
            thresholds(&spec_with_binding(
                r#"{"profile":"p","overload":{"token_usage":0.0004}}"#
            )),
            Some(OverloadThresholds {
                kv_usage_permille: 1,
                ..defaults
            })
        );
        // The usage is rounded to its permille, not cut.
        assert_eq!(
            thresholds(&spec_with_binding(
                r#"{"profile":"p","overload":{"token_usage":0.2496}}"#
            )),
            Some(OverloadThresholds {
                kv_usage_permille: 250,
                ..defaults
            })
        );
        assert_eq!(
            thresholds(&spec_with_binding(r#"{"profile":"p"}"#)),
            Some(defaults)
        );
        assert_eq!(
            thresholds(&doc(
                r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["kv-pressure"],"pick":"best"}}}"#
            )),
            Some(defaults)
        );
        assert_eq!(
            bad_field(&spec_with_binding(
                r#"{"profile":"p","overload":{"waiting_requests":0}}"#
            )),
            "bindings[0].overload.waiting_requests"
        );
        for token_usage in ["1.5", "0"] {
            assert_eq!(
                bad_field(&spec_with_binding(&format!(
                    r#"{{"profile":"p","overload":{{"token_usage":{token_usage}}}}}"#
                ))),
                "bindings[0].overload.token_usage",
                "{token_usage}"
            );
        }
        // Off is the block's switch alone: the binding hands no threshold,
        // whatever values the block still names.
        for off in [
            r#"{"profile":"p","overload":{"enabled":false}}"#,
            r#"{"profile":"p","overload":{"enabled":false,"waiting_requests":3,"token_usage":0.5}}"#,
        ] {
            assert_eq!(thresholds(&spec_with_binding(off)), None, "{off}");
        }

        let sessions = spec_with_binding(
            r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","ttl_s":600,"assignment":"min_load"}}"#,
        );
        let built = sessions
            .validate()
            .expect("the session table is in this build");
        assert!(built.needs().headers && !built.needs().active_sessions);
        assert!(built.needs().session_in_flight, "the default cap of two");
        assert_eq!(
            built.needs_for_phase(Phase::Single),
            built.needs(),
            "the rule's needs are the binding's phase needs"
        );
        assert_eq!(
            spec_with_binding(r#"{"profile":"p","sessions":{"sticky_header":"x-other"}}"#)
                .validate()
                .err(),
            Some(SpecError::HeaderNotAllowed {
                path: "bindings[0].sessions.sticky_header".into(),
                name: "x-other".into()
            })
        );
        assert_eq!(
            bad_field(&spec_with_binding(
                r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","headers":["x-session-id"],"ttl_s":0}}"#
            )),
            "bindings[0].sessions.ttl_s"
        );
        assert_eq!(
            bad_field(&spec_with_binding(
                r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","headers":["x-request-id"]}}"#
            )),
            "bindings[0].sessions.sticky_header"
        );
        assert_eq!(
            bad_field(&spec_with_binding(
                r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","inflight_cap":0}}"#
            )),
            "bindings[0].sessions.inflight_cap"
        );
        // A sweep for ever under a finite idle time would never forget a key.
        assert_eq!(
            spec_with_binding(
                r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","ttl_s":600,"sweep_s":4294967295}}"#
            )
            .validate()
            .err(),
            Some(SpecError::BadSessions {
                index: 0,
                name: "sweep_s",
                reason: "finite while ttl_s is",
            })
        );
        // An idle time for ever keeps every key and sweeps nothing, whatever
        // the sweep says; a finite pair maps to the rule in microseconds.
        let rule = |sessions: &str| {
            spec_with_binding(&format!(
                r#"{{"profile":"p","sessions":{{"sticky_header":"x-session-id",{sessions}}}}}"#
            ))
            .validate()
            .expect("a sessions rule")
            .binding("m", Phase::Single)
            .and_then(|binding| binding.sessions)
            .map(|rule| (rule.ttl_micros, rule.sweep_micros))
        };
        assert_eq!(
            rule(r#""ttl_s":4294967295,"sweep_s":4294967295"#),
            Some((None, 0))
        );
        assert_eq!(rule(r#""ttl_s":4294967295,"sweep_s":60"#), Some((None, 0)));
        assert_eq!(
            rule(r#""ttl_s":600,"sweep_s":60"#),
            Some((Some(600_000_000), 60_000_000))
        );
        assert_eq!(
            not_in_build(&spec_with_binding(
                r#"{"profile":"p","sessions":{"sticky_header":"x-session-id","issue_session_id":true}}"#
            )),
            (
                "bindings[0].sessions.issue_session_id".to_string(),
                "issuing a session id"
            )
        );
        let err = parse_error(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"best"}},"bindings":[{"profile":"p","sessions":{"ttl_s":1}}]}"#,
        );
        assert!(err.contains("missing field `sticky_header`"), "{err}");

        assert_eq!(
            spec_with_binding(r#"{"profile":"p","shadow":"ghost"}"#)
                .validate()
                .err(),
            Some(SpecError::UnknownProfileAt {
                path: "bindings[0].shadow".into(),
                name: "ghost".into()
            })
        );
        assert_eq!(
            bad_field(&spec_with_binding(r#"{"profile":"p","shadow":"p"}"#)),
            "bindings[0].shadow"
        );
        assert_eq!(
            not_in_build(&spec_with_binding(r#"{"profile":"p","shadow":"q"}"#)),
            ("bindings[0].shadow".to_string(), "the shadow binding")
        );

        assert_eq!(
            not_in_build(&spec_with_binding(
                r#"{"profile":"p","canary":{"profile":"q","percent":5}}"#
            )),
            ("bindings[0].canary".to_string(), "the canary binding")
        );
        assert_eq!(
            spec_with_binding(
                r#"{"profile":"p","canary":{"profile":"q","percent":5,"tenants":["acme"]}}"#
            )
            .validate()
            .err(),
            Some(SpecError::CanaryForm {
                path: "bindings[0].canary".into(),
                count: 2
            })
        );
        assert_eq!(
            spec_with_binding(r#"{"profile":"p","canary":{"profile":"q"}}"#)
                .validate()
                .err(),
            Some(SpecError::CanaryForm {
                path: "bindings[0].canary".into(),
                count: 0
            })
        );
        assert_eq!(
            bad_field(&spec_with_binding(
                r#"{"profile":"p","canary":{"profile":"q","percent":101}}"#
            )),
            "bindings[0].canary.percent"
        );
        assert_eq!(
            bad_field(&spec_with_binding(
                r#"{"profile":"p","canary":{"profile":"q","percent":0}}"#
            )),
            "bindings[0].canary.percent"
        );
        assert_eq!(
            not_in_build(&spec_with_binding(
                r#"{"profile":"p","canary":{"profile":"q","header":{"name":"x-session-id","values":["beta"]}}}"#
            ))
            .1,
            "the canary binding"
        );

        assert_eq!(
            not_in_build(&spec_with_binding(
                r#"{"profile":"p","override":"allowed"}"#
            )),
            (
                "bindings[0].override".to_string(),
                "the profile override header"
            )
        );
        assert!(spec_with_binding(r#"{"profile":"p","override":"locked"}"#)
            .validate()
            .is_ok());
        assert_eq!(
            spec_with_binding(r#"{"profile":"p","admission":{"enabled":true}}"#)
                .validate()
                .err(),
            Some(SpecError::Reserved {
                path: "bindings[0].admission.enabled".into(),
                reason: "admission is reserved in v1 and stays off"
            })
        );
        assert!(
            spec_with_binding(r#"{"profile":"p","admission":{"enabled":false}}"#)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn the_matchers_are_checked_in_shape_and_refused_as_not_in_this_build() {
        let matching = |m: &str| spec_with_binding(&format!(r#"{{"match":{m},"profile":"p"}}"#));
        assert_eq!(
            bad_field(&matching(r#"{"tenant":""}"#)),
            "bindings[0].match.tenant"
        );
        assert_eq!(
            not_in_build(&matching(r#"{"tenant":"acme"}"#)),
            ("bindings[0].match.tenant".to_string(), "the tenant matcher")
        );
        assert_eq!(
            bad_field(&matching(r#"{"priority":{}}"#)),
            "bindings[0].match.priority"
        );
        assert_eq!(
            bad_field(&matching(r#"{"priority":{"min":5,"max":1}}"#)),
            "bindings[0].match.priority.min"
        );
        assert_eq!(
            not_in_build(&matching(r#"{"priority":{"max":-1}}"#)).1,
            "the priority matcher"
        );
        assert_eq!(
            not_in_build(&matching(r#"{"header":{"name":"x-session-id"}}"#)).1,
            "the header matcher"
        );
        assert_eq!(
            matching(r#"{"header":{"name":"x-other","values":["a"]}}"#)
                .validate()
                .err(),
            Some(SpecError::HeaderNotAllowed {
                path: "bindings[0].match.header.name".into(),
                name: "x-other".into()
            })
        );
        assert_eq!(
            not_in_build(&matching(r#"{"prompt_tokens":{"min":1024}}"#)).1,
            "the prompt-tokens matcher"
        );
        assert_eq!(
            matching(r#"{"labels":["large","large"]}"#).validate().err(),
            Some(SpecError::Duplicate {
                path: "bindings[0].match.labels".into(),
                name: "large".into()
            })
        );
        assert_eq!(
            not_in_build(&matching(r#"{"labels":["large"]}"#)).1,
            "the label matcher"
        );
    }

    #[test]
    fn a_shape_error_outranks_what_the_build_lacks() {
        let spec = doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["kv-pressure"],"pick":"best"}},"bindings":[{"profile":"p","overload":{"token_usage":2}}]}"#,
        );
        assert_eq!(bad_field(&spec), "bindings[0].overload.token_usage");
    }

    #[test]
    fn the_plans_section_carries_the_pairing_mode() {
        let spec = doc(
            r#"{"apiVersion":"smg.policy/v1","plans":{"pd-pair":{"pairing":"strict"}},"profiles":{"p":{"pick":"best"}}}"#,
        );
        assert_eq!(spec.plans.pd_pair.pairing, Pairing::Strict);
        assert!(spec.validate().is_ok());
        let err = parse_error(
            r#"{"apiVersion":"smg.policy/v1","plans":{"epd":{}},"profiles":{"p":{"pick":"best"}}}"#,
        );
        assert!(err.contains("unknown field `epd`"), "{err}");
    }

    #[test]
    fn a_document_with_every_section_round_trips_and_writes_the_stage_forms() {
        let text = r#"{
          "apiVersion": "smg.policy/v1",
          "generation": 7,
          "view": { "headers": ["x-session-id", "x-tenant-tier"], "block_size": 16, "boundaries": [256, 1024], "max_block_hashes": 128 },
          "index": { "*": { "kind": "hash", "ttl_s": 180, "max_size": 67108864, "sweep_s": 120 } },
          "profiles": {
            "cache_aware": {
              "filter": ["healthy", "not-overloaded", "not-tried"],
              "score": [
                { "prefix-affinity": { "weight": 0.7, "threshold": 0.3, "decay": 0.9, "source": "merged", "accounting_ttl_ms": 200, "spill": { "abs": 64, "rel": 1.5, "prefix_margin_tokens": 0, "uncached_credit": false } } },
                { "kv-pressure": { "spread": 0.3, "ceiling": 0.9 } },
                { "least-load": { "weight": 0.3, "kv_pressure_weight": 0.15 } },
                { "warmup-slice": { "secs": 60, "share": 0.25, "blocks": 1024, "thin_ratio": 0.5, "divert_every": 8 } },
                { "bucket": { "abs_threshold": 64, "rel_threshold": 1.5, "adjust_interval_s": 5 } },
                { "consistent-hash": { "prefix_tokens": 256, "bounded_load": { "factor": 1.25, "abs_margin": 10 } } },
                { "resident-work": { "request_tokens": 512 } },
                "least-requests",
                "session-affinity"
              ],
              "pick": { "best-with-temperature": { "temperature": 0.05 } }
            },
            "least_load": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" }
          },
          "plans": { "pd-pair": { "pairing": "lenient" } },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"], "tenant": "acme", "priority": { "max": -1 }, "header": { "name": "x-tenant-tier", "values": ["gold"] }, "prompt_tokens": { "min": 1, "max": 4096 }, "labels": ["large"] },
              "plan": "pd-pair", "profiles": { "prefill": "least_load", "decode": "least_load" } },
            { "match": { "model": "*" }, "profile": "cache_aware", "shadow": "least_load",
              "canary": { "profile": "least_load", "percent": 5 }, "override": "allowed",
              "overload": { "enabled": true, "waiting_requests": 64, "token_usage": 0.9, "shed_when_all_vetoed": false },
              "sessions": { "sticky_header": "x-session-id", "headers": ["x-session-id"], "ttl_s": 600, "sweep_s": 120, "assignment": "delegate", "inflight_cap": 2, "issue_session_id": false },
              "admission": { "enabled": false } }
          ]
        }"#;
        let spec = doc(text);
        let written = serde_json::to_string(&spec).expect("serializes");
        assert_eq!(doc(&written), spec);
        assert!(
            written.contains(r#"{"prefix-affinity":{"weight":0.7"#),
            "{written}"
        );
        assert!(
            written.contains(r#""least-requests","session-affinity"]"#),
            "{written}"
        );
        assert!(written.contains(r#""pick":{"best-with-temperature":{"temperature":0.05}}"#));
        assert!(written.contains(r#""override":"allowed""#));
        // The shape is whole; this build lacks the stages, in document order.
        assert_eq!(
            not_in_build(&spec),
            (
                "profiles.cache_aware.score[0].prefix-affinity".to_string(),
                "the prefix-affinity decay"
            )
        );
        let yaml = serde_yaml::to_string(&spec).expect("yaml");
        let from_yaml: PolicySpec = serde_yaml::from_str(&yaml).expect("yaml round trip");
        assert_eq!(from_yaml, spec);
    }

    #[test]
    fn the_schema_is_generated_from_the_types() {
        let text = PolicySpec::schema_json().expect("the schema serializes");
        assert!(text.ends_with("}\n"));
        let value: serde_json::Value = serde_json::from_str(&text).expect("JSON");
        assert_eq!(value["title"], "PolicySpec");
        assert!(value["properties"]["bindings"].is_object());
        assert!(value["$defs"]["ScoreStage"]["oneOf"].is_array());
    }

    #[test]
    fn the_index_map_resolves_by_specificity_and_refuses_an_ambiguous_pair() {
        let spec = doc(
            r#"{"apiVersion":"smg.policy/v1","index":{"*":{},"chat-*":{},"chat-7b":{},"*-7b":{}},
                "profiles":{"p":{"pick":"round-robin"}}}"#,
        );
        spec.validate().expect("distinct keys");
        let key = |model: &str| spec.index_for(model).map(|(key, _)| key);
        assert_eq!(key("chat-7b"), Some("chat-7b"));
        assert_eq!(key("chat-13b"), Some("chat-*"));
        assert_eq!(key("llama-7b"), Some("*-7b"));
        assert_eq!(key("other"), Some("*"));
        let mut without_default = spec.clone();
        without_default.index.remove("*");
        assert_eq!(without_default.index_for("other"), None);
        let ambiguous = doc(
            r#"{"apiVersion":"smg.policy/v1","index":{"ab*":{},"*cd":{}},
                "profiles":{"p":{"pick":"round-robin"}}}"#,
        );
        assert_eq!(
            ambiguous.validate().err(),
            Some(SpecError::IndexOverlap {
                first: "*cd".into(),
                second: "ab*".into()
            })
        );
    }

    /// An index key, with a count of the times it is read.
    struct Read<'a>(&'a str, &'a std::cell::Cell<usize>);

    impl AsRef<str> for Read<'_> {
        fn as_ref(&self) -> &str {
            self.1.set(self.1.get() + 1);
            self.0
        }
    }

    /// The overlap pass reads each index key once, whatever their number: a
    /// thousand keys are a thousand reads and eight thousand are eight
    /// thousand. A pass over every pair reads them by the square.
    #[test]
    fn the_overlap_pass_reads_each_index_key_once() {
        for entries in [1_000, 8_000] {
            // A prefix and a suffix by turns, of two literal lengths: no two
            // overlap, so no pass ends early.
            let keys: Vec<String> = (0..entries)
                .map(|n| {
                    if n % 2 == 0 {
                        format!("p{n:05}*")
                    } else {
                        format!("*{n:05}ss")
                    }
                })
                .collect();
            let reads = std::cell::Cell::new(0);
            let found = overlap(keys.iter().map(|key| Read(key, &reads)));
            assert!(found.is_none(), "{entries} keys of no overlap");
            assert_eq!(reads.get(), entries, "{entries} keys");
        }
    }

    /// The pass over every pair, as the check ran before: the measure of
    /// the pair the overlap pass names.
    fn pairwise<'a>(keys: &[&'a str]) -> Option<(&'a str, &'a str)> {
        let ambiguous = |a: &str, b: &str| matches!((side(a), side(b)), (Some((sa, la)), Some((sb, lb))) if sa != sb && la == lb);
        keys.iter().enumerate().find_map(|(place, first)| {
            let second = keys[place + 1..]
                .iter()
                .find(|second| ambiguous(first, second))?;
            Some((*first, *second))
        })
    }

    /// The next of a draw's numbers, under `bound`.
    fn below(state: &mut u64, bound: u64) -> usize {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from((*state >> 33) % bound).expect("under the bound")
    }

    /// The pair refused is the pair a pass over every pair names: the first
    /// key with a later partner, and its first partner. Four thousand key
    /// sets by draw, of ids, prefixes, suffixes and `*`, in the map's order.
    #[test]
    fn the_overlap_named_is_the_pair_a_pass_over_every_pair_names() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let (mut overlapping, mut distinct) = (0, 0);
        for _ in 0..4_000 {
            let mut keys = BTreeSet::new();
            for _ in 0..below(&mut state, 12) {
                let literal: String = (0..below(&mut state, 4))
                    .map(|_| ["a", "b"][below(&mut state, 2)])
                    .collect();
                keys.insert(match below(&mut state, 4) {
                    0 => format!("{literal}*"),
                    1 => format!("*{literal}"),
                    2 => "*".to_owned(),
                    _ => format!("{literal}x"),
                });
            }
            let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
            let named = overlap(keys.iter().copied());
            assert_eq!(named, pairwise(&keys), "{keys:?}");
            if named.is_some() {
                overlapping += 1;
            } else {
                distinct += 1;
            }
        }
        assert!(
            overlapping > 500 && distinct > 500,
            "{overlapping} sets overlap, {distinct} do not"
        );
    }

    #[test]
    fn more_profiles_than_a_binding_can_address_are_refused() {
        let mut spec = spec_with_binding(r#"{"profile":"p"}"#);
        let template = spec.profiles["p"].clone();
        for i in 0..=u16::MAX {
            spec.profiles.insert(format!("p{i}"), template.clone());
        }
        assert_eq!(
            spec.validate().err(),
            Some(SpecError::TooManyProfiles {
                count: spec.profiles.len(),
                max: u16::MAX
            })
        );
    }

    /// Every refusal names the field it is about, in the document's own
    /// names; a path the validator built is handed back as it stands.
    #[test]
    fn every_refusal_names_its_field() {
        let word = || "x".to_owned();
        let profile = || "default".to_owned();
        let built = || "a.b[0]".to_owned();
        let rows = [
            (SpecError::Kind(word()), "kind"),
            (SpecError::ApiVersion(word()), "apiVersion"),
            (
                SpecError::UnknownFilter {
                    profile: profile(),
                    name: word(),
                },
                "profiles.default.filter",
            ),
            (
                SpecError::UnknownScorer {
                    profile: profile(),
                    name: word(),
                },
                "profiles.default.score",
            ),
            (
                SpecError::PickerRunsNoScorer {
                    profile: profile(),
                    picker: word(),
                },
                "profiles.default.score",
            ),
            (
                SpecError::UnknownPicker {
                    profile: profile(),
                    name: word(),
                },
                "profiles.default.pick",
            ),
            (
                SpecError::PickerNeedsScorer {
                    profile: profile(),
                    picker: word(),
                },
                "profiles.default.pick",
            ),
            (
                SpecError::UnknownProfile {
                    index: 2,
                    name: word(),
                },
                "bindings[2]",
            ),
            (SpecError::NoProfile { index: 2 }, "bindings[2].profile"),
            (
                SpecError::ProfileUnderPlan {
                    index: 2,
                    plan: "pd-pair",
                },
                "bindings[2].profile",
            ),
            (SpecError::ProfileTwice { index: 2 }, "bindings[2].profile"),
            (
                SpecError::SideUnnamed {
                    index: 2,
                    plan: "pd-pair",
                    side: "decode",
                },
                "bindings[2].profiles.decode",
            ),
            (
                SpecError::UnknownSide {
                    index: 2,
                    side: "left".to_owned(),
                },
                "bindings[2].profiles.left",
            ),
            (
                SpecError::SideNotInPlan {
                    index: 2,
                    plan: "single",
                    side: "decode".to_owned(),
                },
                "bindings[2].profiles.decode",
            ),
            (
                SpecError::PlanPhases {
                    index: 2,
                    plan: "pd-pair",
                    side: "decode",
                },
                "bindings[2].match.phase",
            ),
            (
                SpecError::BadModelGlob {
                    index: 2,
                    model: word(),
                },
                "bindings[2].match.model",
            ),
            (
                SpecError::BadSessions {
                    index: 2,
                    name: "plan",
                    reason: "so",
                },
                "bindings[2].plan",
            ),
            (
                SpecError::BadSessions {
                    index: 2,
                    name: "sweep_s",
                    reason: "so",
                },
                "bindings[2].sessions.sweep_s",
            ),
            (SpecError::NoBindings { profiles: 2 }, "bindings"),
            (
                SpecError::IndexOverlap {
                    first: "a*".to_owned(),
                    second: "*b".to_owned(),
                },
                "index.a*",
            ),
            (
                SpecError::HashIndexNeedsBoundaries {
                    model: "a*".to_owned(),
                },
                "index.a*.kind",
            ),
            (
                SpecError::TooManyProfiles {
                    count: 70_000,
                    max: u16::MAX,
                },
                "profiles",
            ),
            (SpecError::Empty, "profiles"),
            (
                SpecError::TooManyHeaders {
                    count: 300,
                    max: 256,
                },
                "view.headers",
            ),
            (
                SpecError::TooManyListed {
                    path: "view.zones",
                    count: 300,
                    max: 256,
                },
                "view.zones",
            ),
            (
                SpecError::BadField {
                    path: built(),
                    value: word(),
                    reason: "so",
                },
                "a.b[0]",
            ),
            (
                SpecError::NotInBuild {
                    path: built(),
                    what: "it",
                },
                "a.b[0]",
            ),
            (
                SpecError::DeniedHeader {
                    path: built(),
                    name: word(),
                },
                "a.b[0]",
            ),
            (
                SpecError::HeaderNotAllowed {
                    path: built(),
                    name: word(),
                },
                "a.b[0]",
            ),
            (
                SpecError::Duplicate {
                    path: built(),
                    name: word(),
                },
                "a.b[0]",
            ),
            (
                SpecError::UnknownProfileAt {
                    path: built(),
                    name: word(),
                },
                "a.b[0]",
            ),
            (
                SpecError::CanaryForm {
                    path: built(),
                    count: 2,
                },
                "a.b[0]",
            ),
            (
                SpecError::Reserved {
                    path: built(),
                    reason: "so",
                },
                "a.b[0]",
            ),
            (SpecError::UnknownPreset(word()), ""),
            (SpecError::NoCanonicalForm(word()), ""),
        ];
        for (error, path) in rows {
            assert_eq!(error.field_path(), path, "{error}");
        }
    }

    /// The path is the validator's own finding's: a document refused for a
    /// stage, a weight, a binding or its version reads the field to edit.
    #[test]
    fn a_refused_document_reads_the_field_to_edit() {
        let refused = |document: &str| {
            let document: PolicySpec = serde_yaml::from_str(document).expect("a document");
            document.validate().expect_err("refused").field_path()
        };
        let profile =
            |stages: &str| format!("apiVersion: smg.policy/v1\nprofiles:\n  default:\n{stages}");
        assert_eq!(
            refused(&profile("    filter: [nowhere]\n    pick: random\n")),
            "profiles.default.filter"
        );
        assert_eq!(
            refused(&profile("    pick: nothing\n")),
            "profiles.default.pick"
        );
        let weight = refused(&profile(
            "    score: [{ least-load: { weight: -1.0 } }]\n    pick: best\n",
        ));
        assert!(weight.starts_with("profiles.default.score[0]"), "{weight}");
        let bound = |binding: &str| {
            refused(&format!(
                "{}bindings:\n  - {binding}\n",
                profile("    pick: random\n")
            ))
        };
        assert_eq!(
            bound("{ match: { model: \"*\" }, profile: nowhere }"),
            "bindings[0]"
        );
        assert_eq!(
            bound("{ match: { model: \"a*b*\" }, profile: default }"),
            "bindings[0].match.model"
        );
        assert_eq!(
            refused("apiVersion: smg.policy/v9\nprofiles:\n  default:\n    pick: random\n"),
            "apiVersion"
        );
    }

    #[test]
    fn a_document_built_in_code_is_this_version_and_validates() {
        let mut profiles = spec_with_binding(r#"{"profile":"p"}"#).profiles;
        profiles.remove("q");
        let spec = PolicySpec::new(profiles);
        assert_eq!(spec.api_version, API_VERSION);
        spec.validate().expect("one profile binds to everything");
        let text = serde_json::to_string(&spec).expect("serializes");
        assert!(!text.contains("kind"), "{text}");
    }

    /// The spill parameters of a prefix-affinity entry reach the gates the
    /// profile's stages share: under a relative threshold of four a holder
    /// three times as loaded as the mean keeps its request; under the
    /// default (1.5, plus the margin) it spills.
    #[test]
    fn the_spill_parameters_of_the_spec_reach_the_profiles_gates() {
        use crate::{
            AffinityFacts, Candidate, Constraints, Health, PoolFacts, RequestFacts, Scratch,
            WorkerId,
        };
        let hashes = [0u64; 16];
        // The first worker holds the whole request; the column says so.
        let held = [
            AffinityFacts {
                overlap_blocks: 16,
                ..AffinityFacts::default()
            },
            AffinityFacts::default(),
            AffinityFacts::default(),
        ];
        let facts = RequestFacts {
            model: "m",
            block_hashes: &hashes,
            prompt_tokens: 256,
            seed: 7,
            affinity: &held,
            ..RequestFacts::default()
        };
        let row = |position: u32, in_flight: u32| Candidate {
            position,
            id: WorkerId(100 + u64::from(position)),
            health: Health {
                healthy: true,
                ..Health::default()
            },
            in_flight,
            ..Candidate::default()
        };
        let pool = [row(0, 20), row(1, 0), row(2, 0)];
        let leader = |score: &str| {
            let engine = spec_with_score(score).validate().expect("a valid profile");
            let decision = engine.decide(
                &facts,
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut Scratch::new(),
            );
            decision.pick().expect("a pick").targets.as_slice()[0].position
        };
        assert_eq!(
            leader(r#"{"prefix-affinity":{"spill":{"rel":4.0}}}"#),
            0,
            "a holder under the configured gate keeps its request"
        );
        assert_ne!(
            leader(r#"{"prefix-affinity":{}}"#),
            0,
            "under the default gate the loaded holder spills"
        );
    }

    /// The digest names the document, not the file: the form (YAML or JSON,
    /// the order of the keys, a comment) and the `generation` line leave it
    /// as it was, and any other field moves it. Its input is the document's
    /// JSON with the line zeroed, under SHA-256.
    #[test]
    fn a_digest_holds_across_the_form_and_the_line_and_moves_with_any_other_field() {
        let written = doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"filter":["healthy"],"score":["least-load"],"pick":"best"}}}"#,
        );
        let digest = written.digest().expect("a digest");

        let numbered = PolicySpec {
            generation: 7,
            ..written.clone()
        };
        assert_eq!(numbered.digest(), Ok(digest), "the line is not in it");
        let yaml: PolicySpec = serde_yaml::from_str(
            "profiles:\n  p:\n    pick: best   # a comment\n    score: [least-load]\n    filter: [healthy]\ngeneration: 3\napiVersion: smg.policy/v1\n",
        )
        .expect("the same document in YAML, its keys in another order");
        assert_eq!(yaml.digest(), Ok(digest), "the form is not in it");

        let mut moved = written.clone();
        moved
            .profiles
            .get_mut("p")
            .expect("the profile")
            .filter
            .push("not-tried".to_owned());
        assert_ne!(moved.digest(), Ok(digest), "a filter more is in it");

        // The one line a document can write that no canonical form carries:
        // the validator refuses it, and the digest does not read it.
        let kinded: PolicySpec = serde_yaml::from_str(
            "kind: PolicySpec\napiVersion: smg.policy/v1\nprofiles:\n  p:\n    filter: [healthy]\n    score: [least-load]\n    pick: best\n",
        )
        .expect("a kind line parses");
        assert_eq!(kinded.digest(), Ok(digest), "a kind line is not in it");
        assert_eq!(
            kinded.validate().err(),
            Some(SpecError::Kind("PolicySpec".into()))
        );

        let canonical = serde_json::to_string(&PolicySpec {
            generation: 0,
            ..numbered
        })
        .expect("serializes");
        assert!(
            canonical.starts_with(r#"{"apiVersion":"smg.policy/v1","generation":0,"#),
            "{canonical}"
        );
        let hex: String = Sha256::digest(canonical.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(digest.to_string(), format!("sha256:{hex}"));
    }

    /// The digest's "any", over the type's fields: a change to each field
    /// the schema names moves the digest, the `generation` line alone
    /// excepted. The names are the schema's, so a field added to the type
    /// joins the case, and fails it until a change to it is shown here.
    #[test]
    fn a_digest_moves_with_every_field_the_schema_names_but_the_line() {
        let base = PolicySpec::preset("least_load").expect("a preset");
        let digest = base.digest().expect("a digest");

        let mut bindings = base.bindings.clone();
        bindings.push(BindingSpec::default());
        let mut index = base.index.clone();
        index.insert("another*".to_owned(), IndexSpec::default());
        let mut profiles = base.profiles.clone();
        let profile = profiles.values().next().expect("a profile").clone();
        profiles.insert("another".to_owned(), profile);
        let plans = PlansSpec {
            pd_pair: PdPairPlanSpec {
                pairing: Pairing::Off,
            },
        };
        let view = ViewSpec {
            headers: vec!["x-team".to_owned()],
            ..base.view.clone()
        };
        let changed = [
            (
                "apiVersion",
                PolicySpec {
                    api_version: "smg.policy/v2".to_owned(),
                    ..base.clone()
                },
            ),
            (
                "bindings",
                PolicySpec {
                    bindings,
                    ..base.clone()
                },
            ),
            (
                "generation",
                PolicySpec {
                    generation: base.generation + 9,
                    ..base.clone()
                },
            ),
            (
                "index",
                PolicySpec {
                    index,
                    ..base.clone()
                },
            ),
            (
                "plans",
                PolicySpec {
                    plans,
                    ..base.clone()
                },
            ),
            (
                "profiles",
                PolicySpec {
                    profiles,
                    ..base.clone()
                },
            ),
            (
                "view",
                PolicySpec {
                    view,
                    ..base.clone()
                },
            ),
        ];

        let schema = serde_json::to_value(PolicySpec::schema()).expect("the schema is JSON");
        let named: BTreeSet<&str> = schema["properties"]
            .as_object()
            .expect("the type's fields")
            .keys()
            .map(String::as_str)
            .collect();
        let shown: BTreeSet<&str> = changed.iter().map(|(field, _)| *field).collect();
        assert_eq!(named, shown, "one change a field the schema names");
        for (field, document) in &changed {
            assert_ne!(document, &base, "{field} is changed");
            let moved = document.digest().expect("a digest") != digest;
            assert_eq!(moved, *field != "generation", "{field}");
        }
    }

    /// The digest copies none of the document: what it hashes is, byte for
    /// byte, the JSON the document itself writes under a zeroed line. Held
    /// for every preset, under a line and under none, for a document that
    /// writes every field the schema names and for one that writes the
    /// least: a field left out where it is the default is left out of both.
    #[test]
    fn a_digest_hashes_what_the_document_itself_writes_under_a_zeroed_line() {
        let base = PolicySpec::preset("least_load").expect("a preset");
        let every_field = PolicySpec {
            view: ViewSpec {
                headers: vec!["x-team".to_owned()],
                ..base.view.clone()
            },
            index: BTreeMap::from([("another*".to_owned(), IndexSpec::default())]),
            plans: PlansSpec {
                pd_pair: PdPairPlanSpec {
                    pairing: Pairing::Off,
                },
            },
            bindings: vec![BindingSpec::default()],
            ..base
        };
        let written = serde_json::to_value(&every_field).expect("the document is JSON");
        let schema = serde_json::to_value(PolicySpec::schema()).expect("the schema is JSON");
        let fields = |of: &serde_json::Value| -> BTreeSet<String> {
            let fields = of.as_object().expect("a map of fields");
            fields.keys().cloned().collect()
        };
        assert_eq!(
            fields(&written),
            fields(&schema["properties"]),
            "the document writes every field the schema names"
        );
        let least = doc(r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"best"}}}"#);
        let mut documents = vec![every_field, least];
        for name in PRESETS {
            let preset = PolicySpec::preset(name).expect(name);
            documents.push(PolicySpec {
                generation: 7,
                ..preset.clone()
            });
            documents.push(preset);
        }
        for document in documents {
            let borrowed = serde_json::to_string(&document.lineless()).expect("serializes");
            let owned = serde_json::to_string(&PolicySpec {
                generation: 0,
                ..document
            })
            .expect("serializes");
            assert_eq!(borrowed, owned);
        }
    }

    /// A validated generation carries its document's digest; the unbound
    /// start has no document and no digest.
    #[test]
    fn an_applied_generation_carries_its_documents_digest_and_the_unbound_start_none() {
        let document = PolicySpec::preset("least_load").expect("a preset");
        let applied = AppliedSpec::validate(&document).expect("a preset validates");
        assert_eq!(applied.digest(), Some(document.digest().expect("a digest")));
        assert_eq!(AppliedSpec::unbound().digest(), None);
    }

    /// A digest handed on is the generation's, as given: the door's apply
    /// digests its document once.
    #[cfg(feature = "controller")]
    #[test]
    fn a_digest_handed_on_is_carried_as_given() {
        let document = PolicySpec::preset("least_load").expect("a preset");
        let handed = PolicySpec::preset("random")
            .expect("a preset")
            .digest()
            .expect("a digest");
        assert_ne!(document.digest(), Ok(handed));
        let applied = AppliedSpec::validate_digested(&document, handed, &Catalog::builtin())
            .expect("a preset validates");
        assert_eq!(applied.digest(), Some(handed));
    }

    /// A digest handed on moves nothing else of the generation: its pairing
    /// mode is the document's, as where the generation digests the document
    /// itself.
    #[cfg(feature = "controller")]
    #[test]
    fn a_generation_under_a_handed_digest_takes_its_documents_pairing_mode() {
        let mut document = PolicySpec::preset("least_load").expect("a preset");
        document.plans.pd_pair.pairing = Pairing::Strict;
        let digest = document.digest().expect("a digest");
        let handed = AppliedSpec::validate_digested(&document, digest, &Catalog::builtin())
            .expect("a preset validates");
        assert_eq!(handed.pairing(), Pairing::Strict);
        let own = AppliedSpec::validate(&document).expect("a preset validates");
        assert_eq!(own.pairing(), Pairing::Strict);
    }

    /// `best` takes a sample in the parameter form and stays the unit by its
    /// name: the sample is a whole number from 1 to 64 and the profile's
    /// need, and the cache-aware preset states sixteen.
    #[test]
    fn the_best_picker_takes_a_sample_in_the_parameter_form() {
        let sampled = spec_with_pick(r#"{"best":{"sample":16}}"#);
        let engine = sampled.validate().expect("validates");
        assert_eq!(engine.needs().sample, 16);
        let written = serde_json::to_string(&sampled).expect("serializes");
        assert!(
            written.contains(r#""pick":{"best":{"sample":16}}"#),
            "{written}"
        );
        assert_eq!(doc(&written), sampled);
        // By its name it is the unit: the whole pool, and no sample needed.
        let unit = spec_with_pick(r#""best""#);
        assert_eq!(unit.validate().expect("validates").needs().sample, 0);
        for sample in [0, 65] {
            let pick = format!(r#"{{"best":{{"sample":{sample}}}}}"#);
            assert_eq!(
                bad_field(&spec_with_pick(&pick)),
                "profiles.p.pick.best.sample"
            );
        }
        let err = parse_error(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":{"best":{"draws":4}}}}}"#,
        );
        assert!(err.contains("unknown field `draws`"), "{err}");
        let preset = PolicySpec::preset("cache_aware").expect("a preset");
        assert_eq!(preset.validate().expect("validates").needs().sample, 16);
    }

    /// The guard of the count's one reader, over the catalog's own lists: a
    /// profile of each stock stage decides alike over rows that carry their
    /// requests in flight and over rows that read otherwise with the counts
    /// in a table lent beside them. A stage that reads a row's own field
    /// reads another count beside lent rows, and parts here.
    #[test]
    fn every_catalog_stage_reads_a_lent_count_as_a_rows_own() {
        use crate::{counts::tests::Lent, view::Identity, Health, PoolFacts, Scratch, WorkerId};
        let sessions = r#""score":["fewest-sessions"]"#;
        let scored = |pick: &str| format!(r#"{{{sessions},"pick":{pick}}}"#);
        let alone = |pick: &str| format!(r#"{{"filter":["healthy"],"pick":{pick}}}"#);
        // The forms that take parameters, then each other name of the
        // lists in the forms it validates in. A name this build lacks is
        // refused by name in each of its forms and is no reader.
        let bounded = r#"{"consistent-hash":{"bounded_load":{"factor":1.0,"abs_margin":0}}}"#;
        let tempered = r#"{"best-with-temperature":{"temperature":1.0}}"#;
        let mut profiles: Vec<Vec<String>> = vec![
            vec![r#"{"score":[{"resident-work":{"request_tokens":512}}],"pick":"best"}"#.into()],
            vec![format!(r#"{{"score":[{bounded}],"pick":"best"}}"#)],
            vec![scored(tempered), alone(tempered)],
        ];
        let apart = ["resident-work", "best-with-temperature"];

        for name in FILTERS {
            profiles.push(vec![format!(
                r#"{{"filter":["{name}"],{sessions},"pick":"best"}}"#
            )]);
        }
        for name in SCORERS.iter().filter(|name| !apart.contains(name)) {
            profiles.push(vec![format!(r#"{{"score":["{name}"],"pick":"best"}}"#)]);
        }
        for name in PICKERS.iter().filter(|name| !apart.contains(name)) {
            let pick = format!("{name:?}");
            profiles.push(vec![
                scored(&pick),
                alone(&pick),
                format!(r#"{{"pick":{pick}}}"#),
            ]);
        }
        let document = |profile: &String| {
            doc(&format!(
                r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{profile}}}}}"#
            ))
        };
        let lacked =
            |form: &String| matches!(document(form).validate(), Err(SpecError::NotInBuild { .. }));
        profiles.retain(|forms| !forms.iter().all(lacked));
        let build = |forms: &[String]| -> Engine {
            let built = forms.iter().find_map(|form| document(form).validate().ok());
            built.unwrap_or_else(|| panic!("no form validates: {forms:?}"))
        };
        // Counts that part the rows, the same rows reading otherwise with
        // the counts in a lent table, and rows that read none.
        let counts = [5, 1, 3, 0, 4, 2, 6, 1];
        let row = |(position, in_flight): (u32, u32)| Candidate {
            position,
            id: WorkerId(100 + u64::from(position)),
            in_flight,
            health: Health {
                healthy: true,
                ..Health::default()
            },
            ..Candidate::default()
        };
        let rows = |held: [u32; 8]| -> Vec<Candidate> { (0..).zip(held).map(row).collect() };
        let (own, mut other, bare) = (rows(counts), rows(counts), rows([0; 8]));
        let lent = Lent::take(&mut other);
        for (row, held) in other.iter_mut().zip([1, 6, 2, 4, 0, 3, 1, 5]) {
            row.in_flight = held;
        }
        let pool = PoolFacts {
            pool_size: 8,
            generation: 1,
            ..PoolFacts::default()
        };
        // One engine a feed: a stage that keeps a rotation starts each
        // feed's decisions at the same place.
        let run = |forms: &[String], rows: &[Candidate], beside: bool| {
            let engine = build(forms);
            let picks = |seed: u64| {
                let facts = RequestFacts {
                    model: "m",
                    session_key: Identity::hashed(seed),
                    seed,
                    ..RequestFacts::default()
                };
                let facts = if beside { lent.beside(&facts) } else { facts };
                let cx = Constraints::default();
                let decision = engine.decide(&facts, &pool, rows, &cx, &mut Scratch::new());
                let picked = decision.pick();
                picked.map(|pick| {
                    pick.targets
                        .as_slice()
                        .iter()
                        .map(|t| t.id)
                        .collect::<Vec<_>>()
                })
            };
            (0..32).map(picks).collect::<Vec<_>>()
        };
        for forms in &profiles {
            let written = run(forms, &own, false);
            let read = run(forms, &other, true);
            assert_eq!(
                written, read,
                "a lent count is not read as a row's own: {forms:?}"
            );
            // The counts part the rows: the fewest requests answer by them.
            let moved = written != run(forms, &bare, false);
            assert!(moved || !forms[0].contains("least-requests"), "{forms:?}");
        }
    }
}
