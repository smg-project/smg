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
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{
    de::{self, MapAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{
    builtins::{
        Best, CountFirst, FewestSessions, HasCapacity, HasRoom, Healthy, LeastLoad, LeastRequests,
        NotOverloaded, NotTried, Passthrough, PowerOfTwo, PowerOfTwoCountFirst, PrefixAffinity,
        Random, RoundRobin, SessionAffinity,
    },
    framework::{Binding, Engine, Filter, Picker, Plan, Profile, Scorer, Weighted},
    ring::ConsistentHash,
    sessions::SessionRule,
    view::Phase,
};

pub const API_VERSION: &str = "smg.policy/v1";

/// Header names no allow-list may carry (design 4.6). A host adds the tenant
/// header it is configured with through [`PolicySpec::validate_with`].
pub const DENIED_HEADERS: [&str; 5] = [
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
];

/// The most headers `view.headers` may list: a view names an allow-listed
/// header by its position in one byte.
pub const MAX_HEADERS: usize = 256;

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
/// the requests in flight before the score; `best-with-temperature` takes
/// its temperature and is written in the parameter form.
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

/// A sampling picker compares its draws by the profile's scorers, so it
/// needs at least one.
const SAMPLING_PICKERS: [&str; 2] = ["power-of-two", "power-of-two-count-first"];

/// The side names a binding's `profiles` map may use.
const SIDES: [&str; 4] = ["single", "prefill", "decode", "encode"];

/// What the host must feed for the spec's stages to mean anything, per
/// deployment. Read at apply, never per request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Needs {
    /// The request text in the view (a text-window hash form, for a host
    /// without a tokenizer).
    pub text: bool,
    pub token_hashes: bool,
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
}

impl Needs {
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
        }
    }
}

fn is_default<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

/// What the host builds into every decision's view beyond the facts it
/// derives itself: the request headers a stage may read, and the hashing
/// inputs a host keeps until it learns them from its engines.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct ViewSpec {
    /// The request headers a stage may read, lowercase, in the order their
    /// positions take on the wire. A name on the deny-list (`authorization`,
    /// `proxy-authorization`, `cookie`, `set-cookie`, `x-api-key`, the host's
    /// tenant header) is refused. Every header a binding's `sessions`,
    /// `canary` or `match.header` names must be listed here. At most 256: a
    /// view names a header by its position in one byte.
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
    /// How many prefix block hashes a request carries into the view.
    #[schemars(range(min = 1))]
    pub max_block_hashes: u32,
}

impl Default for ViewSpec {
    fn default() -> Self {
        Self {
            headers: Vec::new(),
            block_size: None,
            boundaries: Vec::new(),
            max_block_hashes: 256,
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

impl PrefixAffinityParams {
    /// Whether any parameter beyond the weight is written.
    fn has_parameters(&self) -> bool {
        self.threshold.is_some()
            || self.decay.is_some()
            || self.source.is_some()
            || self.accounting_ttl_ms.is_some()
            || self.spill.is_some()
    }
}

/// The KV-pressure scorer: the fleet's KV-usage spread past which affinity
/// yields to load, and the per-candidate ceiling above which the hottest
/// candidate is de-ranked.
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
    /// The usage ceiling above which a candidate is de-ranked; above zero,
    /// 1 and above switches the rule off (default 1).
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
        }
    }
}

/// The bucket scorer: prefill placement by request-size buckets whose
/// boundaries a periodic task adjusts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BucketParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
    /// The absolute part of the balance rule, in requests (default 64).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abs_threshold: Option<u32>,
    /// The relative part of the balance rule; at least 1 (default 1.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1.0))]
    pub rel_threshold: Option<f32>,
    /// The boundary task's cadence, in seconds; above zero (default 5).
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

/// The bounded-load rule of the consistent-hash scorer: a ring owner loaded
/// this far past the mean passes the request on.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoundedLoadParams {
    /// The relative part of the rule; at least 1 (default 1.25).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1.0))]
    pub factor: Option<f32>,
    /// The absolute part of the rule, in requests (default 10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abs_margin: Option<u32>,
}

/// The consistent-hash scorer: a per-model ring keyed by the routing key or
/// by the request's prefix, with a bounded-load rule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConsistentHashParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
    /// How many prefix tokens hash when the ring keys by prefix; above zero
    /// (default 256).
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

/// The resident-work scorer: the tokens resident on a candidate, with the
/// request's own token count added for the comparison.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResidentWorkParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
    /// The request's token count added to each candidate's resident
    /// tokens; above zero.
    #[schemars(range(min = 1))]
    pub request_tokens: u32,
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

/// A scorer reference: a bare name, or the name mapped to its weight and
/// parameters ([`ScoreStage`]).
#[derive(Clone, Debug, PartialEq)]
pub enum ScoreSpec {
    Name(String),
    Params(ScoreStage),
}

impl ScoreSpec {
    /// The scorer's name as written.
    pub fn name(&self) -> &str {
        match self {
            Self::Name(name) => name,
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
        let Self::Params(stage) = self else {
            return serializer.serialize_str(self.name());
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

/// The one scorer of a score entry's map, read by its key; the key chooses
/// the parameter struct, so the struct's own message names an unknown or
/// mistyped parameter.
fn scorer_entry<'de, A: MapAccess<'de>>(mut map: A) -> Result<ScoreStage, A::Error> {
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
        other => return Err(de::Error::unknown_variant(other, &SCORERS)),
    };
    if map.next_key::<de::IgnoredAny>()?.is_some() {
        return Err(de::Error::custom(
            "a score entry maps a single key to its parameters; write one entry per scorer",
        ));
    }
    Ok(stage)
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

            // The map is read by its one key, so the message names an
            // unknown scorer or a wrong parameter, never "no variant matched".
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<ScoreSpec, A::Error> {
                scorer_entry(map).map(ScoreSpec::Params)
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
    BestWithTemperature(BestWithTemperatureParams),
}

impl PickStage {
    /// The picker's name, as the document spells it.
    pub fn name(&self) -> &'static str {
        match self {
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
    const WITH_PARAMETERS: [&str; 1] = ["best-with-temperature"];
    let Some(key) = map.next_key::<String>()? else {
        return Err(de::Error::custom("a pick entry names one picker"));
    };
    let stage = match key.as_str() {
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
            "description": "A picker: its bare name, or the name mapped to its parameters.",
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
    /// The hard filters, in order.
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

/// The overload veto's defaults for the binding's workers: evaluated per
/// load report by the host and latched, never per request; a worker's own
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
    /// falling back to the least-vetoed one.
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

/// How a session key seen for the first time picks its worker.
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
    /// How a first-seen key picks its worker (default `delegate`).
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
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub profiles: BTreeMap<String, String>,
    /// The overload veto's defaults for the binding's workers.
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
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub index: BTreeMap<String, IndexSpec>,
    /// The profiles by name.
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
    fn number(
        &self,
        name: &str,
        value: Option<f32>,
        ok: fn(f32) -> bool,
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

fn at_least_one(v: f32) -> bool {
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
    at.nonzero("max_block_hashes", Some(view.max_block_hashes))
}

fn check_index(index: &BTreeMap<String, IndexSpec>, view: &ViewSpec) -> Result<(), SpecError> {
    for (model, spec) in index {
        let at = At::new(format!("index.{model}"));
        if !glob_ok(model) {
            return Err(SpecError::BadField {
                path: at.0,
                value: model.clone(),
                reason: "an id, `prefix*`, `*suffix` or `*`",
            });
        }
        at.nonzero("ttl_s", Some(spec.ttl_s))?;
        at.nonzero("max_size", Some(spec.max_size))?;
        at.nonzero("sweep_s", Some(spec.sweep_s))?;
        if spec.kind == IndexKind::Hash && view.boundaries.is_empty() {
            return Err(SpecError::HashIndexNeedsBoundaries {
                model: model.clone(),
            });
        }
    }
    let keys: Vec<&str> = index.keys().map(String::as_str).collect();
    for (i, first) in keys.iter().enumerate() {
        if let Some(second) = keys[i + 1..].iter().find(|second| ambiguous(first, second)) {
            return Err(SpecError::IndexOverlap {
                first: (*first).to_string(),
                second: (*second).to_string(),
            });
        }
    }
    Ok(())
}

/// A `prefix*` and a `*suffix` of one literal length: they match a common
/// model and neither is the more specific, so the pair is refused.
fn ambiguous(a: &str, b: &str) -> bool {
    let shape = |g: &str| match (g.strip_suffix('*'), g.strip_prefix('*')) {
        _ if g == "*" => None,
        (Some(prefix), _) => Some((true, prefix.len())),
        (_, Some(suffix)) => Some((false, suffix.len())),
        _ => None,
    };
    matches!((shape(a), shape(b)), (Some((pa, la)), Some((pb, lb))) if pa != pb && la == lb)
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
        ScoreStage::LeastRequests(_)
        | ScoreStage::FewestSessions(_)
        | ScoreStage::SessionAffinity(_) => Ok(()),
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
            at.number("thin_ratio", p.thin_ratio, unit_interval, "between 0 and 1")
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
        ScoreStage::ResidentWork(p) => at.nonzero("request_tokens", Some(p.request_tokens)),
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

fn built_scorer(scorer: impl Scorer + 'static) -> Resolved<Box<dyn Scorer>> {
    Resolved::Built(Box::new(scorer))
}

fn scorer_of(stage: &ScoreStage) -> Resolved<Box<dyn Scorer>> {
    match stage {
        ScoreStage::LeastLoad(p) => built_scorer(LeastLoad::new(
            p.kv_pressure_weight.unwrap_or(0.15),
            p.mean_prefill_tokens.unwrap_or(1024),
            p.default_throughput.unwrap_or(2000.0),
        )),
        ScoreStage::LeastRequests(_) => built_scorer(LeastRequests),
        ScoreStage::FewestSessions(_) => built_scorer(FewestSessions),
        ScoreStage::SessionAffinity(_) => built_scorer(SessionAffinity),
        ScoreStage::PrefixAffinity(p) if p.has_parameters() => {
            Resolved::NotInBuild("the prefix-affinity parameters")
        }
        ScoreStage::PrefixAffinity(_) => built_scorer(PrefixAffinity),
        ScoreStage::KvPressure(_) => Resolved::NotInBuild("the kv-pressure scorer"),
        ScoreStage::WarmupSlice(_) => Resolved::NotInBuild("the warmup-slice scorer"),
        ScoreStage::Bucket(_) => Resolved::NotInBuild("the bucket scorer"),
        ScoreStage::ConsistentHash(p) if p.prefix_tokens.is_some() => {
            Resolved::NotInBuild("the consistent-hash prefix key")
        }
        ScoreStage::ConsistentHash(p) => built_scorer(ConsistentHash::new(p.bounded_load.as_ref())),
        ScoreStage::ResidentWork(_) => Resolved::NotInBuild("the resident-work scorer"),
    }
}

fn built_picker(picker: impl Picker + 'static) -> Resolved<Box<dyn Picker>> {
    Resolved::Built(Box::new(picker))
}

fn picker_of(
    pick: &PickSpec,
    profile: &str,
    at: &At,
) -> Result<Resolved<Box<dyn Picker>>, SpecError> {
    Ok(match pick {
        PickSpec::Name(name) => match name.as_str() {
            "best" => built_picker(Best),
            "round-robin" => built_picker(RoundRobin::new()),
            "random" => built_picker(Random),
            "power-of-two" => built_picker(PowerOfTwo),
            "count-first" => built_picker(CountFirst),
            "power-of-two-count-first" => built_picker(PowerOfTwoCountFirst),
            "passthrough" => built_picker(Passthrough),
            "manual" => Resolved::NotInBuild("the manual picker"),
            "best-with-temperature" => {
                return Err(at.bad(
                    "pick",
                    name,
                    "a picker with its temperature: { best-with-temperature: { temperature } }",
                ))
            }
            _ => {
                return Err(SpecError::UnknownPicker {
                    profile: profile.to_string(),
                    name: name.clone(),
                })
            }
        },
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

fn filter_by_name(name: &str) -> Option<Box<dyn Filter>> {
    Some(match name {
        "healthy" => Box::new(Healthy),
        "not-overloaded" => Box::new(NotOverloaded),
        "has-capacity" => Box::new(HasCapacity),
        "has-room" => Box::new(HasRoom),
        "not-tried" => Box::new(NotTried),
        _ => return None,
    })
}

/// The stage a score entry stands for: a bare catalog name at its defaults,
/// or the written parameters.
fn stage_of(entry: &ScoreSpec, profile: &str, at: &At) -> Result<ScoreStage, SpecError> {
    match entry {
        ScoreSpec::Params(stage) => Ok(stage.clone()),
        ScoreSpec::Name(name) if name == "resident-work" => Err(SpecError::BadField {
            path: at.0.clone(),
            value: name.clone(),
            reason: "a scorer with its request_tokens: { resident-work: { request_tokens } }",
        }),
        ScoreSpec::Name(name) => {
            ScoreStage::by_name(name).ok_or_else(|| SpecError::UnknownScorer {
                profile: profile.to_string(),
                name: name.clone(),
            })
        }
    }
}

/// One profile's stages, built when this build carries every one of them.
fn check_profile(
    name: &str,
    p: &ProfileSpec,
    needs: &mut Needs,
    deferred: &mut Deferred,
) -> Result<Option<Profile>, SpecError> {
    let at = At::new(format!("profiles.{name}"));
    let mut filters = Vec::with_capacity(p.filter.len());
    for f in &p.filter {
        filters.push(filter_by_name(f).ok_or_else(|| SpecError::UnknownFilter {
            profile: name.to_string(),
            name: f.clone(),
        })?);
    }
    let mut scorers = Vec::with_capacity(p.score.len());
    let mut complete = true;
    for (i, entry) in p.score.iter().enumerate() {
        let entry_at = At::new(format!("{}.score[{i}]", at.0));
        let stage = stage_of(entry, name, &entry_at)?;
        let stage_at = entry_at.sub(stage.name());
        check_stage(&stage, &stage_at)?;
        match scorer_of(&stage) {
            Resolved::Built(scorer) => {
                *needs = needs.or(scorer.needs());
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
    let picker = match picker_of(&p.pick, name, &at)? {
        Resolved::Built(picker) => Some(picker),
        Resolved::NotInBuild(what) => {
            deferred.note(at.field("pick"), what);
            None
        }
    };
    if p.score.is_empty() && SAMPLING_PICKERS.contains(&p.pick.name()) {
        return Err(SpecError::PickerNeedsScorer {
            profile: name.to_string(),
            picker: p.pick.name().to_string(),
        });
    }
    Ok(match picker {
        Some(picker) if complete => Some(Profile {
            name: name.to_string(),
            filters,
            scorers,
            picker,
        }),
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

/// The ten preset names of today, as whole profiles. The ones whose stages
/// this build does not carry yet answer `UnknownPreset`.
pub const PRESETS: [&str; 7] = [
    "random",
    "round_robin",
    "passthrough",
    "power_of_two",
    "least_load",
    "consistent_hashing",
    "manual",
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

impl PolicySpec {
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
            // Sticky sessions over a random assignment: a key stays with the
            // first worker it was placed on, no in-flight cap, the first
            // worker back in the lead when it recovers; the idle TTL of four
            // hours swept each minute, as the keyed policy of today.
            "manual" => ProfileSpec {
                filter: standard_filters(),
                score: vec![],
                pick: "random".into(),
            },
            other => return Err(SpecError::UnknownPreset(other.to_string())),
        };
        let manual = name == "manual";
        let sessions = manual.then(|| SessionsSpec {
            sticky_header: MANUAL_SESSION_HEADER.into(),
            headers: vec![],
            ttl_s: 14_400,
            sweep_s: 60,
            assignment: Assignment::Delegate,
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
    ///
    /// Every field is checked in shape first; a field this build cannot
    /// honour (a stage, a plan, a matcher or a binding rule of a later step)
    /// is then refused by name as [`SpecError::NotInBuild`].
    pub fn validate_with(&self, denied_headers: &[&str]) -> Result<Engine, SpecError> {
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
        let mut needs = Needs::default();
        let mut profiles = Vec::with_capacity(self.profiles.len());
        for (name, profile) in &self.profiles {
            if let Some(built) = check_profile(name, profile, &mut needs, &mut deferred)? {
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
        for (index, binding) in self.bindings.iter().enumerate() {
            let cx = BindingContext {
                index,
                view: &self.view,
                denied: denied_headers,
                profiles: &positions,
            };
            if let Some(built) = check_binding(binding, &cx, &mut needs, &mut deferred)? {
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
            });
        }
        needs.headers |= !self.view.headers.is_empty();
        Ok(Engine::new(profiles, bindings, self.generation, needs))
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let load_aware = name == "least_load" || name == "power_of_two";
            assert_eq!(engine.needs().loads, load_aware, "{name}");
            assert_eq!(engine.needs().ring, name == "consistent_hashing", "{name}");
            assert!(!engine.needs().events, "{name}");
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
            PolicySpec::preset("cache_aware").err(),
            Some(SpecError::UnknownPreset("cache_aware".into()))
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
        // profile. The decode needs carry both.
        let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "decode-only": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" },
            "prefill": { "filter": ["healthy"], "pick": "round-robin" },
            "decode": { "filter": ["healthy"], "score": ["session-affinity"], "pick": "best" },
            "default": { "filter": ["healthy"], "pick": "round-robin" }
          },
          "bindings": [
            { "match": { "model": "*", "phase": ["decode"] }, "profile": "decode-only" },
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
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
        assert!(matches!(
            unknown_filter.validate(),
            Err(SpecError::UnknownFilter { .. })
        ));
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
        // With parameters, the one-key map goes to the stage enum whole, so
        // the message names the key instead of "no variant matched".
        let err = parse_error(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":[{"least_load":{"kv_pressure_weight":5}}],"pick":"best"}}}"#,
        );
        assert!(err.contains("unknown variant `least_load`"), "{err}");
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
        let spec = spec_with_view(r#"{"block_size":128,"boundaries":[256,1024,4096]}"#);
        assert!(spec.validate().is_ok());
        assert_eq!(spec.view.max_block_hashes, 256);
    }

    #[test]
    fn the_index_map_is_per_model_glob_and_a_hash_index_needs_boundaries() {
        let with_index = |index: &str, view: &str| {
            doc(&format!(
                r#"{{"apiVersion":"smg.policy/v1","view":{view},"index":{index},"profiles":{{"p":{{"pick":"best"}}}}}}"#
            ))
        };
        assert_eq!(bad_field(&with_index(r#"{"a*b":{}}"#, "{}")), "index.a*b");
        assert_eq!(
            bad_field(&with_index(r#"{"*":{"ttl_s":0}}"#, "{}")),
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

    #[test]
    fn every_catalog_scorer_parses_and_the_ones_this_build_lacks_are_refused_by_name() {
        let in_build = [
            "least-load",
            "least-requests",
            "fewest-sessions",
            "session-affinity",
            "prefix-affinity",
            "consistent-hash",
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
        let with_params = [
            (
                r#"{"kv-pressure":{"spread":0.3,"ceiling":0.9}}"#,
                "kv-pressure",
            ),
            (r#"{"warmup-slice":{"secs":0}}"#, "warmup-slice"),
            (r#"{"bucket":{"abs_threshold":32}}"#, "bucket"),
            (
                r#"{"consistent-hash":{"prefix_tokens":512,"bounded_load":{"factor":1.5,"abs_margin":4}}}"#,
                "consistent-hash",
            ),
            (
                r#"{"resident-work":{"request_tokens":512}}"#,
                "resident-work",
            ),
            (
                r#"{"prefix-affinity":{"threshold":0.5}}"#,
                "prefix-affinity",
            ),
        ];
        for (score, name) in with_params {
            let (path, what) = not_in_build(&spec_with_score(score));
            assert_eq!(path, format!("profiles.p.score[0].{name}"), "{score}");
            assert!(what.contains(name), "{what}");
        }
        // The consistent-hash scorer is in the build with its bounded-load
        // rule; the prefix key form waits for the block hashes in the view.
        let bounded = spec_with_score(r#"{"consistent-hash":{"bounded_load":{"factor":1.5}}}"#);
        assert!(bounded.validate().expect("validates").needs().ring);
        assert_eq!(
            not_in_build(&spec_with_score(
                r#"{"consistent-hash":{"prefix_tokens":512}}"#
            ))
            .1,
            "the consistent-hash prefix key"
        );
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
                r#"{"resident-work":{"request_tokens":0}}"#,
                "resident-work.request_tokens",
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

    #[test]
    fn the_pickers_of_v1_resolve_or_are_refused_by_name() {
        assert_eq!(
            not_in_build(&spec_with_pick(r#""manual""#)),
            ("profiles.p.pick".to_string(), "the manual picker")
        );
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
    }

    #[test]
    fn the_binding_rules_are_checked_in_shape_then_by_the_build() {
        let overload = spec_with_binding(
            r#"{"profile":"p","overload":{"waiting_requests":32,"token_usage":0.9,"shed_when_all_vetoed":true}}"#,
        );
        assert!(overload.validate().is_ok());
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
        let off = spec_with_binding(r#"{"profile":"p","overload":{"enabled":false}}"#);
        assert!(off.validate().is_ok());

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
                "the prefix-affinity parameters"
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
}
