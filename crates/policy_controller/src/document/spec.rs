//! `PolicySpec` (design 4.6): the one versioned document. The view says what
//! the host builds into every decision, the index map what the prefix index
//! keeps per model, profiles name stages with their parameters, plans carry
//! their rules, bindings match requests to profiles or plans, and explain
//! sets the decision sampling rate. Validation answers what the host must
//! feed before any request exists.
//!
//! The document has an `apiVersion` and no `kind`: its forms are a file and
//! the API. An unknown field is refused by name, never ignored. The one
//! exception is the parameters of a scorer or filter a host registers, which
//! its factory's own type reads ([`crate::stages::StageBuild::params`]). A field this
//! build cannot honour yet (a stage, a plan, a matcher or a binding rule of a
//! later step) is checked in shape like every other field and then refused
//! by name as not in this build, so a document written against the published
//! schema validates the day its stage lands. [`PolicySpec::schema`] is that
//! schema; the field documentation of this module is its descriptions.

#[cfg(feature = "controller")]
use std::sync::atomic::{self, AtomicBool};
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::{btree_map, BTreeMap, BTreeSet},
    fmt, io,
    marker::PhantomData,
    sync::{Arc, LazyLock},
};

use rustc_hash::FxHashMap;
use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};
use sha2::{Digest as _, Sha256};

#[cfg(feature = "in-process")]
use crate::engine::sticky_sessions::SessionTable;
use crate::{
    document::routing_key::KeySources,
    engine::{
        framework::{
            Binding, Engine, Filter, OverloadThresholds, Picker, Plan, Profile, Scorer,
            WeightedScorer,
        },
        sticky_sessions::SessionRule,
        view::{blocks_within, threshold_permille, HashKind, Phase, MAX_LOAD_REPORT_AGE_MS},
    },
    load::{
        load_report_reader::LoadReportReader,
        signals::{column_named, SignalColumn},
    },
    policies::{
        self,
        bucket::Bucket,
        cache_aware::{Gates, KvPressure, PrefixAffinity, WarmupSlice},
        consistent_hashing::ConsistentHash,
        least_load::LeastLoad,
    },
    stages::{
        cost_terms::{signal_named, CostTerm, SIGNAL_FAMILY},
        filters::{has_room_rule, HasRoomRule, HasRoomSignals},
        pickers::SampledBest,
        registry::{
            profile_sent, FilterFactory, Found, PickerFactory, SentSlot, StageBuild, StageRegistry,
        },
        scorers::{
            with_load_report_age_limit, FewestSessions, LeastCost, LeastCostSignals,
            LeastCostWithDispatches, LeastRequests, ResidentWork, SessionAffinity,
            DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT,
        },
    },
    tracking::dispatch_credit::{DispatchCredit, SentSettings},
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

/// Header names no routing key source may carry: the credentials of
/// [`DENIED_HEADERS`]. `cookie` may be a source, and stays off every
/// allow-list: a key read from it leaves as its fingerprint alone
/// ([`DerivedKey`](crate::document::DerivedKey)). A host's tenant header is refused
/// here as there.
pub const DENIED_KEY_HEADERS: [&str; 6] = [
    "authorization",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
];

/// The most bytes of a value a routing key source hands where its entry
/// names no `max_bytes`: the gateway's cap on a routing key header.
pub const KEY_BYTES: u32 = 128;

/// The most bytes an entry's `max_bytes` may ask for.
pub const MAX_KEY_BYTES: u32 = 8192;

/// The most sources one `routing_key` block lists, its client sources not
/// counted.
pub const MAX_KEY_SOURCES: usize = 8;

/// The most client sources one `routing_key` block lists.
pub const MAX_CLIENT_SOURCES: usize = 4;

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
pub const SCORERS: &[&str] = &[
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
    LeastCost::NAME,
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
static STOCK: LazyLock<StageRegistry> = LazyLock::new(StageRegistry::builtin);

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
///
/// The record gains flags. Code outside this crate writes its needs as one
/// chain from [`Needs::NONE`], a call a flag
/// (`Needs::NONE.with_loads().with_prompt_size()`), or assigns a flag that
/// follows a value (`needs.signal_cells = keeps`). It reads a field, and
/// matches the record with `..`: a literal of it does not build there.
///
/// ```
/// use smg_policy_controller::document::Needs;
/// let needs = Needs::NONE.with_loads().with_prompt_size();
/// assert!(needs.loads && needs.prompt_size && !needs.events);
/// ```
///
/// ```compile_fail
/// use smg_policy_controller::document::Needs;
/// let _ = Needs { loads: true, ..Needs::NONE };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Needs {
    /// The request's text itself, for a controller that hashes it into
    /// windows for a foreign host. No stage of this build declares it: a
    /// host that holds the text hashes its windows itself
    /// ([`Self::text_hashes`]).
    pub text: bool,
    /// The request's prefix block hashes over its token ids. A decision is
    /// handed the hashes as deep as the stages that declare them read
    /// ([`AppliedSpec::block_hashes_read`]), so a stage that reads the list
    /// declares one of the two kinds: under a generation where no stage
    /// does, it is handed none. A plugin is shown the list where its own
    /// row declares one of the two, and none of it otherwise.
    pub token_hashes: bool,
    /// The request's prefix block hashes over fixed windows of its text's
    /// bytes, which a host with no token ids feeds in the token hashes'
    /// place. A stage that reads a prefix by either kind sets this beside
    /// `token_hashes`.
    pub text_hashes: bool,
    /// The request's size
    /// ([`RequestFacts::prompt_tokens`](crate::engine::RequestFacts::prompt_tokens)).
    /// A stage that places by it declares this, and so does a stage that
    /// prices what was sent to a worker by its size, as a `sent.tokens`
    /// term does: a send of text alone is counted by its text's bytes. A
    /// host that reads every request's prompt counts it and states it
    /// always. A host that can send a request on without reading its
    /// prompt reads the prompt where this is declared, and may state zero
    /// where it is not: a stage that read the size undeclared there would
    /// place every request as an empty one, with no error. Where a request
    /// states its prompt's token ids ahead of its body, their count stands
    /// for the size, up to the host's cap on such a list. A plugin is
    /// shown the size where its own row declares this, and zero otherwise,
    /// whatever the host read.
    #[serde(default)]
    pub prompt_size: bool,
    /// Load reports per worker.
    pub loads: bool,
    /// KV events into an index. A stage that declares this beside
    /// `token_hashes` reads the request's holders from the index, and a
    /// host names the index's holders ahead of the record's for its
    /// profile. A stage that reads the events for another fact of a row,
    /// as a warm-up does, declares no hashes beside it.
    pub events: bool,
    /// The stage reads the prefixes sent to each worker.
    #[serde(default)]
    pub sent_prefixes: bool,
    /// A consistent-hash ring per model. A host that keeps one lends it with
    /// a request ([`RequestFacts::ring`](crate::engine::RequestFacts::ring)).
    pub ring: bool,
    /// Allow-listed headers in the view.
    pub headers: bool,
    pub media: bool,
    /// The count of session keys with a request in flight, per worker (the
    /// `min_group` assignment and the `fewest-sessions` scorer).
    pub active_sessions: bool,
    /// The request's own key's in-flight count per worker (a sessions rule
    /// with an in-flight cap). A host that writes it says so on the request
    /// ([`RequestFacts::session_counted`](crate::engine::RequestFacts::session_counted)).
    pub session_in_flight: bool,
    /// The affinity column
    /// ([`RequestFacts::affinity`](crate::engine::RequestFacts::affinity)): one
    /// [`AffinityFacts`](crate::engine::AffinityFacts) a candidate. A stage that
    /// reads any of the five declares this, whatever feeds them: `events`
    /// asks for the index and the hash fields for the request's hashes, and
    /// neither stands for the column. A host fills the column where this is
    /// declared and may hand an empty column elsewhere. Rows it names in
    /// the column's place ([`AffinityRows`](crate::engine::AffinityRows))
    /// answer the crate's own affinity stages and no other reader: a named row
    /// says its two depths and nothing of its age, growth or index size, so a
    /// host fills the column for a profile with a stage that reads those, a
    /// guest's among them. The engine checks the column's shape, never its
    /// presence: an unfilled column reads as nothing held for every candidate
    /// and nothing is refused. What holds a host to the need is the builder of
    /// its view, by a case of its own.
    pub affinity: bool,
    /// The sample of a picker that decides a large pool by a draw of this
    /// many rows and the rows the host names; `0` for none. A host that can
    /// name the request's holders and the pool's warming rows
    /// ([`AffinityRows`](crate::engine::AffinityRows)) hands those in the
    /// column's place, each list capped at this many rows: the holders by
    /// depth, whole depth groups from the deepest while they fit and the
    /// group at the cap's edge by a uniform draw from the decision's seed;
    /// the warming rows by that draw over their whole list. A host that
    /// cannot fills the column, and the pool is read whole. The cap of a
    /// decision is its own profile's, which
    /// [`Engine::sample_for`](crate::engine::Engine::sample_for) says: the engine's
    /// summary holds the largest sample of every profile, and a decode
    /// worker's needs the larger of its two profiles'.
    pub sample: u32,
    /// The signal cells beside the rows
    /// ([`RequestFacts::signal_cells`](crate::engine::RequestFacts::signal_cells)).
    /// A stage that keeps a signal's column declares this, and no other
    /// stage does: the validator holds every stage a catalog adds to both.
    /// One a factory builds under a build context resolves its column
    /// there
    /// ([`StageBuild::signal_column`](crate::stages::StageBuild::signal_column)).
    /// One built with no context resolves none, so it declares none.
    /// A host hands the cells where this is declared and none elsewhere.
    pub signal_cells: bool,
}

/// Writes one call a flag of [`Needs`]: the needs with that flag declared.
/// The flags are listed once, below.
macro_rules! declares {
    ($($call:ident => $flag:ident),+ $(,)?) => {
        impl Needs {
            $(
                #[doc = concat!("These needs with [`Self::", stringify!($flag), "`] declared.")]
                #[must_use]
                pub const fn $call(mut self) -> Self {
                    self.$flag = true;
                    self
                }
            )+
        }

        // Never run: the pattern names every field, so a flag the list
        // lacks stops the build and no flag comes without its call.
        const _: () = {
            let Needs { $($flag: _,)+ sample: _ } = Needs::NONE;
        };
    };
}

declares! {
    with_text => text,
    with_token_hashes => token_hashes,
    with_text_hashes => text_hashes,
    with_prompt_size => prompt_size,
    with_loads => loads,
    with_events => events,
    with_sent_prefixes => sent_prefixes,
    with_ring => ring,
    with_headers => headers,
    with_media => media,
    with_active_sessions => active_sessions,
    with_session_in_flight => session_in_flight,
    with_affinity => affinity,
    with_signal_cells => signal_cells,
}

impl Needs {
    /// No need: what a stage that reads nothing the host feeds declares,
    /// and where every other stage's chain starts. The default, as a
    /// constant.
    pub const NONE: Self = Self {
        text: false,
        token_hashes: false,
        text_hashes: false,
        prompt_size: false,
        loads: false,
        events: false,
        sent_prefixes: false,
        ring: false,
        headers: false,
        media: false,
        active_sessions: false,
        session_in_flight: false,
        affinity: false,
        sample: 0,
        signal_cells: false,
    };

    /// These needs with a sample of `rows` rows ([`Self::sample`]).
    #[must_use]
    pub const fn with_sample(mut self, rows: u32) -> Self {
        self.sample = rows;
        self
    }

    /// Whether these declare a request's block hashes, in the token form or
    /// the text form. A stage that declares neither reads none of them.
    pub(crate) fn request_hashes(self) -> bool {
        self.token_hashes || self.text_hashes
    }

    /// Whether a host that feeds `fed` feeds everything these declare. The
    /// sample is no feed: a host that names no rows fills the column.
    #[must_use]
    pub fn within(self, fed: Self) -> bool {
        let declared = [
            (self.text, fed.text),
            (self.token_hashes, fed.token_hashes),
            (self.text_hashes, fed.text_hashes),
            (self.prompt_size, fed.prompt_size),
            (self.loads, fed.loads),
            (self.events, fed.events),
            (self.sent_prefixes, fed.sent_prefixes),
            (self.ring, fed.ring),
            (self.headers, fed.headers),
            (self.media, fed.media),
            (self.active_sessions, fed.active_sessions),
            (self.session_in_flight, fed.session_in_flight),
            (self.affinity, fed.affinity),
            (self.signal_cells, fed.signal_cells),
        ];
        declared.iter().all(|&(need, feed)| feed || !need)
    }

    /// Both sides' needs: what a host feeding either of two stage sets must
    /// feed.
    pub(crate) fn or(self, other: Self) -> Self {
        Self {
            text: self.text || other.text,
            token_hashes: self.token_hashes || other.token_hashes,
            text_hashes: self.text_hashes || other.text_hashes,
            prompt_size: self.prompt_size || other.prompt_size,
            loads: self.loads || other.loads,
            events: self.events || other.events,
            sent_prefixes: self.sent_prefixes || other.sent_prefixes,
            ring: self.ring || other.ring,
            headers: self.headers || other.headers,
            media: self.media || other.media,
            active_sessions: self.active_sessions || other.active_sessions,
            session_in_flight: self.session_in_flight || other.session_in_flight,
            affinity: self.affinity || other.affinity,
            sample: self.sample.max(other.sample),
            signal_cells: self.signal_cells || other.signal_cells,
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
    /// With this limit in milliseconds set, `least-load`, `resident-work`,
    /// `least-cost` and the KV-pressure gates of `prefix-affinity` and
    /// `warmup-slice` treat older reports as missing. A report exactly at
    /// the limit still counts. A host-registered stage reads a report
    /// under the limit through its build context's `load_report_reader()`,
    /// and one that reads the row's report itself sees it at any age.
    /// Overload verdicts can still reflect older reports, and plugins
    /// interpret report age themselves. Omit it to use reports of any age.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1, max = MAX_LOAD_REPORT_AGE_MS))]
    pub max_report_age_ms: Option<u64>,
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
            max_report_age_ms: None,
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
    /// otherwise. The crate's own store reads it for a pool whose profile
    /// reads sent prefixes: the model's record is bound to `max_size` over
    /// the block size placements, never under one, one a block a worker
    /// was sent. A lowered bound is met by sweeps at the record's next
    /// writes, and a pool whose lanes are all heard writes none.
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

/// When a pair's decode side is picked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum DecodePick {
    /// With the prefill, in the pair's one decision.
    #[default]
    WithPrefill,
    /// After the prefill has run, on a route that sends the decode then.
    /// The pair's first decision books the prefill alone and names the
    /// decode's targets. The host asks the decode again when the prefill
    /// has answered, and that decision reads the rows of its own instant.
    /// A route that sends both sides at once keeps the first decision's
    /// decode.
    AfterPrefill,
}

/// The `pd-pair` plan's rules.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct PdPairPlanSpec {
    pub pairing: Pairing,
    /// When a pair's decode side is picked: `with-prefill` (the default)
    /// or `after-prefill`.
    pub decode: DecodePick,
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
    /// The index of cache events alone.
    Index,
    /// The record of the prefixes sent alone.
    Predictor,
    /// Each worker by what it has: its cache events where it publishes
    /// them, and the prefixes sent to it where it does not.
    #[default]
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
    /// Where placements come from (default `merged`).
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
    /// by its prefix through the last whole block at or under the depth, so at
    /// least one block of the view's block size and at most the blocks a
    /// request carries. Under the view's `boundaries` a request that reaches
    /// one is keyed there instead, at the deepest it reaches. The depth is
    /// checked in the view's token blocks and read in the blocks a request
    /// was hashed by. A host that hashes a text request by windows of the
    /// block size times four bytes gives it the checked depth in windows,
    /// as a token request has it in blocks. Under a block size
    /// an engine published above the depth a request has no key. Where
    /// the request's caller lends its ring no block is read: the request
    /// is keyed at the token, at each boundary it reaches or else at the
    /// depth, a shorter request whole, and a text at four characters a
    /// token. A routing key the request's own id handed (a `rid` source)
    /// leads no prompt under this form: it names the session alone, and
    /// the request is keyed by its prefix as one with no routing key. That
    /// holds in a gateway's own process: no frame states where a key came
    /// from, so under a controller's address the id's key places as any
    /// other. Absent, a request with no routing key has none.
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

/// The largest magnitude of a cost rule's parameter. The signal sink holds
/// a value it takes to the same bound.
pub(crate) const COST_PARAMETER_LIMIT: f64 = 1e15;
const RELATIVE_MARGIN_LIMIT: f64 = 1e6;

/// The least-cost scorer: a sum of a row's fields times their coefficients,
/// in the document's units. Lower cost scores higher.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LeastCostParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0, "maximum" = COST_PARAMETER_LIMIT))]
    pub weight: f32,
    /// Field names and their coefficients. At least one field is required.
    /// A name written `signal.<name>` reads a signal the document declares.
    #[serde(deserialize_with = "keyed_once")]
    #[schemars(extend(
        "minProperties" = 1,
        "propertyNames" = {
            "anyOf": [
                { "enum": CostTerm::NAMES.map(|(name, _)| name) },
                { "pattern": signal_pattern() }
            ]
        },
        "additionalProperties" = {
            "type": "number",
            "format": "double",
            "minimum": -COST_PARAMETER_LIMIT,
            "maximum": COST_PARAMETER_LIMIT,
            "not": { "const": 0 }
        }
    ))]
    pub terms: BTreeMap<String, f64>,
    /// How far above the least cost a row still ties for the top score: the
    /// map of its two parts, or a bare number, the absolute margin.
    #[serde(default, deserialize_with = "absolute_or_parts")]
    #[schemars(with = "AbsoluteOrParts")]
    pub margin: CostMargin,
    /// What a row costs when a named value is missing. Required for terms
    /// that read a report or a signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unreported: Option<Unreported>,
}

/// How to cost a row that lacks a value a term reads.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(
    rename_all = "kebab-case",
    deny_unknown_fields,
    from = "UnreportedEntry",
    into = "UnreportedEntry"
)]
#[schemars(!from, !into)]
#[non_exhaustive]
pub enum Unreported {
    /// Start at the least cost among the reported rows handed to the scorer,
    /// or zero if none report, and add `request_cost` for each request in flight.
    OverLeast {
        #[schemars(extend("minimum" = 0, "maximum" = COST_PARAMETER_LIMIT))]
        request_cost: f64,
    },
    /// Sum the row's available values. A missing value adds nothing.
    MissingAsZero,
}

// Use a string or a one-key map in both document formats.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
enum UnreportedEntry {
    MissingAsZero,
    #[serde(untagged)]
    OverLeast {
        #[serde(rename = "over-least")]
        over_least: RequestCost,
    },
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestCost {
    request_cost: f64,
}

impl From<UnreportedEntry> for Unreported {
    fn from(entry: UnreportedEntry) -> Self {
        match entry {
            UnreportedEntry::MissingAsZero => Self::MissingAsZero,
            UnreportedEntry::OverLeast { over_least } => Self::OverLeast {
                request_cost: over_least.request_cost,
            },
        }
    }
}

impl From<Unreported> for UnreportedEntry {
    fn from(unreported: Unreported) -> Self {
        match unreported {
            Unreported::MissingAsZero => Self::MissingAsZero,
            Unreported::OverLeast { request_cost } => Self::OverLeast {
                over_least: RequestCost { request_cost },
            },
        }
    }
}

/// The sum of the absolute and relative margins. Both default to zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct CostMargin {
    /// In cost units.
    #[schemars(extend("minimum" = 0, "maximum" = COST_PARAMETER_LIMIT))]
    pub absolute: f64,
    /// A share of the magnitude of the least cost.
    #[schemars(extend("minimum" = 0, "maximum" = RELATIVE_MARGIN_LIMIT))]
    pub relative: f64,
}

/// Reads a margin as a document writes it. A bare number is the absolute
/// margin, and a map is read part by part.
fn absolute_or_parts<'de, D: Deserializer<'de>>(deserializer: D) -> Result<CostMargin, D::Error> {
    struct MarginVisitor;

    impl<'de> Visitor<'de> for MarginVisitor {
        type Value = CostMargin;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a number, the absolute margin, or a map of `absolute` and `relative`")
        }

        fn visit_f64<E: de::Error>(self, absolute: f64) -> Result<CostMargin, E> {
            Ok(CostMargin {
                absolute,
                ..CostMargin::default()
            })
        }

        fn visit_i64<E: de::Error>(self, absolute: i64) -> Result<CostMargin, E> {
            self.visit_f64(absolute as f64)
        }

        fn visit_u64<E: de::Error>(self, absolute: u64) -> Result<CostMargin, E> {
            self.visit_f64(absolute as f64)
        }

        fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<CostMargin, A::Error> {
            CostMargin::deserialize(de::value::MapAccessDeserializer::new(map))
        }
    }

    deserializer.deserialize_any(MarginVisitor)
}

/// The schema of a margin's two forms in a document. It holds no value.
enum AbsoluteOrParts {}

impl JsonSchema for AbsoluteOrParts {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        "AbsoluteOrParts".into()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "anyOf": [
                {
                    "description": "The absolute margin, in cost units.",
                    "type": "number",
                    "format": "double",
                    "minimum": 0,
                    "maximum": COST_PARAMETER_LIMIT
                },
                generator.subschema_for::<CostMargin>()
            ]
        })
    }
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
#[non_exhaustive]
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
    LeastCost(LeastCostParams),
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
            Self::LeastCost(_) => LeastCost::NAME,
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
            Self::LeastCost(p) => p.weight,
        }
    }

    /// The stage a bare name stands for, with its parameters at their
    /// defaults; `None` for a name outside the catalog or one whose
    /// parameters have no default.
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

/// A filter reference: a bare name, a built-in mapped to its parameters
/// ([`FilterStage`]), or a name the catalog adds mapped to the parameters
/// its factory reads. A name YAML reads as no string (`true`, `7`, `~`) is
/// written in quotes.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum FilterSpec {
    Name(String),
    Params(FilterStage),
    /// A name that is no built-in's with parameters, mapped to what the
    /// document wrote under it. The map is kept as read so the validator
    /// can ask the catalog for the name first. Parsing refuses a repeated
    /// key.
    Added {
        name: String,
        written: serde_json::Value,
    },
}

impl FilterSpec {
    /// The filter's name as written.
    pub fn name(&self) -> &str {
        match self {
            Self::Name(name) | Self::Added { name, .. } => name,
            Self::Params(stage) => stage.name(),
        }
    }
}

impl From<FilterStage> for FilterSpec {
    fn from(stage: FilterStage) -> Self {
        Self::Params(stage)
    }
}

/// A built-in filter with its parameters, keyed by the filter's name:
/// `{ has-room: { used, capacity } }`. One key per entry.
#[derive(Clone, Debug, PartialEq, JsonSchema)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum FilterStage {
    HasRoom(HasRoomParams),
}

impl FilterStage {
    /// The filter's name, as the document spells it.
    pub fn name(&self) -> &'static str {
        match self {
            Self::HasRoom(_) => HasRoomRule::NAME,
        }
    }
}

/// `has-room` by a rule: a worker has room where what it holds is at most
/// its capacity less the headroom, `used <= (1 - headroom) * capacity`.
/// The filter stays soft: where no worker has room, every worker stands.
/// Written bare, `has-room` reads the bit its host sets. A null under the
/// name, the slip of a bare name written with a colon, is refused with
/// both forms named, whichever reader takes the document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[serde(expecting = "a rule with `used` and `capacity`, or the bare name `has-room`")]
pub struct HasRoomParams {
    /// What a worker holds: field names and their coefficients, summed. At
    /// least one is required. A name written `signal.<name>` reads a signal
    /// the document declares. `request.max_output_tokens` reads the most
    /// tokens the request's answer may hold, so a coefficient above zero
    /// reserves room for the answer: a request that states no limit
    /// reserves none. Under a capacity that is a number, a field or a
    /// signal of the worker is required: the request's own values alone
    /// read nothing of a worker there. A worker that lacks a named value
    /// has no room.
    #[serde(deserialize_with = "keyed_once")]
    #[schemars(extend(
        "minProperties" = 1,
        "propertyNames" = {
            "anyOf": [
                { "enum": room_fields() },
                { "pattern": signal_pattern() }
            ]
        },
        "additionalProperties" = {
            "type": "number",
            "format": "double",
            "minimum": -COST_PARAMETER_LIMIT,
            "maximum": COST_PARAMETER_LIMIT,
            "not": { "const": 0 }
        }
    ))]
    pub used: BTreeMap<String, f64>,
    /// What a worker can hold, in the units of `used`: a number, or a
    /// signal the document declares. A worker whose capacity reads zero or
    /// less has no room.
    pub capacity: Capacity,
    /// The share of the capacity kept free: from 0 to under 1.
    #[serde(default)]
    #[schemars(extend("minimum" = 0, "exclusiveMaximum" = 1))]
    pub headroom: f64,
}

/// The schema's pattern of a signal's name as a sum writes it: the
/// family's name before the signal names' own pattern
/// ([`SIGNAL_NAME_PATTERN`]), which is written once.
fn signal_pattern() -> String {
    format!(
        "^{}{}",
        SIGNAL_FAMILY.replace('.', r"\."),
        SIGNAL_NAME_PATTERN.trim_start_matches('^')
    )
}

/// The names `has-room` may sum: every term's, the request's size and its
/// output limit.
fn room_fields() -> Vec<&'static str> {
    let mut names: Vec<_> = CostTerm::NAMES.iter().map(|&(name, _)| name).collect();
    names.extend([HasRoomRule::REQUEST_TOKENS, HasRoomRule::OUTPUT_TOKENS]);
    names.sort_unstable();
    names
}

/// What a worker can hold: a number above zero in the units of `used`, or
/// the name of a signal that holds one, written `signal.<name>`. No field
/// of this build holds a capacity, so any other name is read and refused.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Capacity {
    Fixed(f64),
    /// A name as the document writes it: a declared signal's.
    Field(String),
}

impl Serialize for Capacity {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Fixed(capacity) => serializer.serialize_f64(*capacity),
            Self::Field(name) => serializer.serialize_str(name),
        }
    }
}

impl<'de> Deserialize<'de> for Capacity {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct CapacityVisitor;

        impl Visitor<'_> for CapacityVisitor {
            type Value = Capacity;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a number, or a declared signal written signal.<name>")
            }

            fn visit_f64<E: de::Error>(self, capacity: f64) -> Result<Capacity, E> {
                Ok(Capacity::Fixed(capacity))
            }

            fn visit_i64<E: de::Error>(self, capacity: i64) -> Result<Capacity, E> {
                Ok(Capacity::Fixed(capacity as f64))
            }

            fn visit_u64<E: de::Error>(self, capacity: u64) -> Result<Capacity, E> {
                Ok(Capacity::Fixed(capacity as f64))
            }

            fn visit_str<E: de::Error>(self, name: &str) -> Result<Capacity, E> {
                Ok(Capacity::Field(name.to_string()))
            }
        }

        deserializer.deserialize_any(CapacityVisitor)
    }
}

impl JsonSchema for Capacity {
    fn schema_name() -> Cow<'static, str> {
        "Capacity".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "description": "What a worker can hold: a number above zero in the units of `used`, or a signal the document declares, written `signal.<name>`.",
            "anyOf": [
                { "type": "number", "format": "double", "exclusiveMinimum": 0 },
                { "type": "string", "pattern": signal_pattern() }
            ]
        })
    }
}

impl From<&str> for FilterSpec {
    fn from(name: &str) -> Self {
        Self::Name(name.to_string())
    }
}

impl From<String> for FilterSpec {
    fn from(name: String) -> Self {
        Self::Name(name)
    }
}

/// A scorer reference: a bare name (a built-in at its defaults, or a scorer
/// the catalog adds at weight one), or the name mapped to its weight and
/// parameters ([`ScoreStage`]).
#[derive(Clone, Debug, PartialEq)]
pub enum ScoreSpec {
    Name(String),
    Params(ScoreStage),
    /// A name that is no built-in's, mapped to its weight and any parameters
    /// its factory reads. The map is kept as read so the validator can ask
    /// the catalog for the name first. Parsing refuses a repeated key.
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

/// What a document wrote under a registered stage's name, written with
/// the keys of every map in order: the canonical bytes, and so the digest,
/// are the same whatever order a file wrote the keys in. The validator
/// hands a factory its parameters in that order too.
struct Ordered<'a>(&'a serde_json::Value);

impl Serialize for Ordered<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            serde_json::Value::Object(written) => {
                let mut keys: Vec<&String> = written.keys().collect();
                keys.sort_unstable();
                let mut map = serializer.serialize_map(Some(keys.len()))?;
                for key in keys {
                    map.serialize_entry(key, &Ordered(&written[key.as_str()]))?;
                }
                map.end()
            }
            serde_json::Value::Array(values) => serializer.collect_seq(values.iter().map(Ordered)),
            value => value.serialize(serializer),
        }
    }
}

impl Serialize for FilterSpec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Name(name) => serializer.serialize_str(name),
            // One key, the stage's name, to its parameters, as a score
            // entry is written.
            Self::Params(stage) => {
                let mut map = serializer.serialize_map(Some(1))?;
                match stage {
                    FilterStage::HasRoom(p) => map.serialize_entry(stage.name(), p)?,
                }
                map.end()
            }
            Self::Added { name, written } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(name, &Ordered(written))?;
                map.end()
            }
        }
    }
}

impl Serialize for ScoreSpec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let stage = match self {
            Self::Name(name) => return serializer.serialize_str(name),
            Self::Added { name, written } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(name, &Ordered(written))?;
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
            ScoreStage::LeastCost(p) => map.serialize_entry(stage.name(), p)?,
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
        LeastCost::NAME => ScoreStage::LeastCost(map.next_value()?),
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

/// What a document wrote under the name of a stage the catalog adds, kept
/// as read for the validator. Read maps and lists recursively so a key
/// written twice is refused even inside a parameter.
struct Written(serde_json::Value);

impl<'de> Deserialize<'de> for Written {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct WrittenVisitor;

        impl<'de> Visitor<'de> for WrittenVisitor {
            type Value = Written;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a stage's parameters")
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

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Written, A::Error> {
                let mut values = Vec::new();
                while let Some(Written(value)) = seq.next_element()? {
                    values.push(value);
                }
                Ok(Written(serde_json::Value::Array(values)))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Written, A::Error> {
                let mut keys = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if keys.contains_key(&key) {
                        return Err(de::Error::custom(format_args!("duplicate field `{key}`")));
                    }
                    let Written(value) = map.next_value()?;
                    keys.insert(key, value);
                }
                Ok(Written(keys.into()))
            }
        }

        deserializer.deserialize_any(WrittenVisitor)
    }
}

/// What a document wrote under `has-room`: the rule. A null there is the
/// slip of a bare name written with a colon, and it is refused here with
/// both forms named, ahead of the rule's own read: a reader that takes a
/// null for an empty map would name the part the rule lacks. The refusal
/// is made inside the value's read, so each reader places it at the
/// rule's path.
struct RoomRule(HasRoomParams);

impl<'de> Deserialize<'de> for RoomRule {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct RuleVisitor;

        impl<'de> Visitor<'de> for RuleVisitor {
            type Value = RoomRule;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a rule with `used` and `capacity`, or the bare name `has-room`")
            }

            // A null, in the words of a reader that hands the rule one.
            fn visit_unit<E: de::Error>(self) -> Result<RoomRule, E> {
                let expected: &dyn de::Expected = &self;
                Err(E::custom(format_args!(
                    "invalid type: null, expected {expected}"
                )))
            }

            fn visit_map<A: MapAccess<'de>>(self, rule: A) -> Result<RoomRule, A::Error> {
                HasRoomParams::deserialize(de::value::MapAccessDeserializer::new(rule))
                    .map(RoomRule)
            }

            // The rule's own read takes its parts as a list too.
            fn visit_seq<A: SeqAccess<'de>>(self, rule: A) -> Result<RoomRule, A::Error> {
                HasRoomParams::deserialize(de::value::SeqAccessDeserializer::new(rule))
                    .map(RoomRule)
            }
        }

        // Asked by what stands there: a reader places a refusal made in
        // this read at the rule's own path, and one made behind an
        // option's read at the entry's.
        deserializer.deserialize_any(RuleVisitor)
    }
}

impl<'de> Deserialize<'de> for FilterSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FilterVisitor;

        impl<'de> Visitor<'de> for FilterVisitor {
            type Value = FilterSpec;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a filter name, or one filter name mapped to its parameters")
            }

            fn visit_str<E: de::Error>(self, name: &str) -> Result<FilterSpec, E> {
                Ok(FilterSpec::Name(name.to_string()))
            }

            // A built-in's key chooses its parameter struct, so the
            // struct's own message names an unknown or mistyped parameter.
            // What stands under any other name is kept as read: only the
            // validator knows the names a catalog adds.
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<FilterSpec, A::Error> {
                let Some(name) = map.next_key::<String>()? else {
                    return Err(de::Error::custom("a filter entry names one filter"));
                };
                let entry = if name == HasRoomRule::NAME {
                    let RoomRule(rule) = map.next_value()?;
                    FilterSpec::Params(FilterStage::HasRoom(rule))
                } else {
                    let Written(written) = map.next_value()?;
                    FilterSpec::Added { name, written }
                };
                if map.next_key::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom(
                        "a filter entry maps a single key to its parameters; write one entry per filter",
                    ));
                }
                Ok(entry)
            }
        }

        deserializer.deserialize_any(FilterVisitor)
    }
}

impl JsonSchema for FilterSpec {
    fn schema_name() -> Cow<'static, str> {
        "FilterSpec".into()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        // A name is open: a host adds names. The stock filters stand as
        // the open string's examples, listed and not enforced.
        json_schema!({
            "description": "A filter: its bare name, a built-in mapped to its parameters, or the name of a filter the host adds mapped to its parameters.",
            "anyOf": [
                { "type": "string", "examples": FILTERS },
                generator.subschema_for::<FilterStage>(),
                {
                    "type": "object",
                    "minProperties": 1,
                    "maxProperties": 1,
                    "propertyNames": { "not": { "const": HasRoomRule::NAME } },
                    "additionalProperties": { "type": "object" }
                }
            ]
        })
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
        // The schema cannot know a host's catalog, so a name is open in
        // both forms: the validator refuses an unknown one by its path. The
        // bare form stands for a stock scorer at its defaults, so one whose
        // parameters have none is written in the parameter form only: the
        // stock scorers that stand bare are the open string's examples,
        // listed and not enforced, and the description names them too.
        let names: Vec<&str> = SCORERS
            .iter()
            .copied()
            .filter(|name| ScoreStage::by_name(name).is_some())
            .collect();
        let description = format!(
            "A scorer: its bare name (a stock scorer at its defaults, or a scorer the host adds at weight one), a stock scorer mapped to its weight and parameters, or the name of a scorer the host adds mapped to its weight and parameters. The stock scorers a document names bare: {}.",
            names.join(", ")
        );
        json_schema!({
            "description": description,
            "anyOf": [
                { "type": "string", "examples": names },
                generator.subschema_for::<ScoreStage>(),
                {
                    "type": "object",
                    "minProperties": 1,
                    "maxProperties": 1,
                    "propertyNames": { "not": { "enum": SCORERS } },
                    "additionalProperties": { "type": "object" }
                }
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

/// A picker reference: a bare name, a built-in mapped to its parameters
/// ([`PickStage`]), or a name the catalog adds mapped to the parameters
/// its factory reads.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum PickSpec {
    Name(String),
    Params(PickStage),
    /// A name that is no built-in's with parameters, mapped to what the
    /// document wrote under it. The map is kept as read so the validator
    /// can ask the catalog for the name first. Parsing refuses a repeated
    /// key.
    Added {
        name: String,
        written: serde_json::Value,
    },
}

impl PickSpec {
    /// The picker's name as written.
    pub fn name(&self) -> &str {
        match self {
            Self::Name(name) | Self::Added { name, .. } => name,
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
            Self::Added { name, written } => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(name, &Ordered(written))?;
                map.end()
            }
        }
    }
}

/// The built-in pickers a document writes with parameters: the keys the
/// reader reads into [`PickStage`].
const TYPED_PICKERS: [&str; 2] = ["best", "best-with-temperature"];

/// The one picker of a pick entry's map, read by its key. A built-in's
/// key chooses its parameter struct, so the struct's own message names an
/// unknown or mistyped parameter. What stands under any other name is
/// kept as read: only the validator knows the names a catalog adds.
fn picker_entry<'de, A: MapAccess<'de>>(mut map: A) -> Result<PickSpec, A::Error> {
    let Some(key) = map.next_key::<String>()? else {
        return Err(de::Error::custom("a pick entry names one picker"));
    };
    let entry = match key.as_str() {
        "best" => PickSpec::Params(PickStage::Best(map.next_value()?)),
        "best-with-temperature" => {
            PickSpec::Params(PickStage::BestWithTemperature(map.next_value()?))
        }
        _ => {
            let Written(written) = map.next_value()?;
            PickSpec::Added { name: key, written }
        }
    };
    if map.next_key::<de::IgnoredAny>()?.is_some() {
        return Err(de::Error::custom(
            "a pick entry maps a single key to its parameters",
        ));
    }
    Ok(entry)
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
                picker_entry(map)
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
        // A name is open in both forms, as a scorer's is: the stock pickers
        // a document names bare are the open string's examples, and the
        // description names them too.
        let description = format!(
            "{} The stock pickers a document names bare: {}.",
            "A picker: its bare name, a built-in mapped to its parameters, or the name of a picker the host adds mapped to its parameters. `power-of-two` and `power-of-two-count-first` compare two workers drawn from those the filters left, `count-first` the fewest in flight among them: the built-in scorers score the workers compared and no other. Between two workers a normalized score is all or nothing, so several weighted scorers under a two-draw picker vote by weight where over the pool they would blend: name one scorer there, or the `best` picker over the pool.",
            names.join(", ")
        );
        json_schema!({
            "description": description,
            "anyOf": [
                { "type": "string", "examples": names },
                generator.subschema_for::<PickStage>(),
                {
                    "type": "object",
                    "minProperties": 1,
                    "maxProperties": 1,
                    "propertyNames": { "not": { "enum": TYPED_PICKERS } },
                    "additionalProperties": { "type": "object" }
                }
            ]
        })
    }
}

/// One profile: an ordered set of filters, weighted scorers and exactly one
/// picker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProfileSpec {
    /// The hard filters, in order: each a name, or the name of a filter the
    /// catalog adds mapped to its parameters. Which workers the hard
    /// filters keep does not turn on their order; a shed's reason does: a
    /// shed names the first filter, in this order, that left nobody.
    #[serde(default)]
    pub filter: Vec<FilterSpec>,
    /// The scorers, each a name or a name with its weight and parameters.
    #[serde(default)]
    pub score: Vec<ScoreSpec>,
    /// The picker: a name, a built-in mapped to its parameters, or the name
    /// of a picker the catalog adds mapped to its parameters.
    pub pick: PickSpec,
    /// The number of milliseconds a send counts for. An unset limit keeps
    /// sends from expiring by age. A profile whose stages do not read its
    /// count of sends is refused if it sets a limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub sent_ttl_ms: Option<u32>,
    /// The least number of milliseconds between a send and the sample of a
    /// load report that ends it: a report sampled less than that long after
    /// a send leaves the send counted. Unset, a report sampled at or after
    /// a send ends it. A written zero is refused, as a zero limit is. A
    /// profile whose stages do not read its count of sends is refused if it
    /// sets a delay. The limit ends a send whatever the delay. The
    /// profiles of a document that set the same limit and the same delay
    /// read one count, under the name of the first of them that reads it,
    /// in the order of their names. A new document's count takes over the
    /// sends the standing count under the same name holds and shares them:
    /// a read under the shorter of the two delays ends such a send for
    /// both. A count under a name with no standing count starts empty: a
    /// document that drops that first profile, or groups it apart from the
    /// others, starts the count of the others empty, once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub sent_report_delay_ms: Option<u32>,
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
/// a binding whose overload block is on vetoes the engine's rows, where
/// the worker's registration states no count: the crate's default. No
/// binding states one.
pub const OVERLOAD_WAITING_REQUESTS: u32 = 8;

/// The mean active KV usage over an engine's ranks at or above which a
/// binding whose overload block is on vetoes the engine's rows, where the
/// worker's registration states no usage: the crate's default. No binding
/// states one.
pub const OVERLOAD_TOKEN_USAGE: f32 = 0.8;

/// What a binding says of the overload veto on the workers of the pools
/// decided under it, as its `overload` field: whether the default
/// thresholds veto at all, and what a decision does where every worker is
/// vetoed. The veto is evaluated per load report by the host and latched,
/// never per request. The block names no threshold: a worker's
/// registration states its own two, over its program's default where the
/// program has one. The store's judge reads the switch, for each binding's
/// pools. A gateway that runs the document in its own process reads the
/// switch once, where it builds its workers' default thresholds, and its
/// routes read one shed word: it refuses a document whose bindings differ
/// on either word. A controller that judges on its worker objects does not
/// read the switch yet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct OverloadSpec {
    /// Whether the host's thresholds veto at all (a worker's own block still
    /// does).
    pub enabled: bool,
    /// A number here is refused: a binding names no count of queued
    /// requests. A worker's registration states the count at or above which
    /// it is vetoed, or its program's default does. The field stands so
    /// that a document which names a number is refused by its path. Its
    /// schema is `null` alone, as a writer that spells an absent field
    /// writes it: an editor refuses the number on shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(with = "()")]
    pub waiting_requests: Option<u32>,
    /// A number here is refused: a binding names no KV usage. A worker's
    /// registration states the usage at or above which it is vetoed, or its
    /// program's default does. The field stands so that a document which
    /// names a number is refused by its path, and its schema is `null`
    /// alone.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(with = "()")]
    pub token_usage: Option<f32>,
    /// Shed with a 503 when every usable worker is vetoed, instead of
    /// falling back to the least-vetoed one. The same word decides what a
    /// session key does while its own worker is vetoed, where the ring
    /// alone places the request: under the single plan, in a profile whose
    /// only scorer is `consistent-hash` with no `prefix_tokens` and no
    /// `bounded_load`, picked by `best` with no parameters, under no
    /// sessions rule or one that delegates. A rule with an in-flight cap
    /// keeps the key only under a host that counts the key's requests, as
    /// the controller's own service does. A rule with
    /// `inflight_cap: 65535` has no cap. Set, the key moves on: under a
    /// sessions rule to its next routable pin, and to the ring's next
    /// worker where no pin stands or no rule does. Unset, the key stays
    /// on its worker over the veto. Everywhere else a keyed request
    /// keeps the veto. The word is the overload veto's alone: a worker
    /// the liveness check doubts still takes a request no other worker
    /// can, unless it is over its thresholds too.
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

/// Whether a single request a sessions rule placed alone is the profile's
/// own: a pin of the request's key followed, or the rule's own assignment.
/// No stage of the profile placed such a request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RulePlaced {
    /// The profile's own: its stages hear the request's issue and its
    /// dispatch, it counts among the profile's sends, and its prompt is
    /// recorded as behind any pick.
    #[default]
    Told,
    /// Kept from the profile, as a gateway's sticky map keeps a request it
    /// places from the gateway's policy.
    Untold,
}

/// Sticky sessions on the binding: a keyed request stays on the worker its
/// key was first placed on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionsSpec {
    /// The header whose value is the session key; listed in `view.headers`.
    /// Left out where the binding names its key's sources under
    /// `routing_key`: a binding names them once.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub sticky_header: String,
    /// The headers looked up in order for the key, the sticky header among
    /// them; empty means the sticky header alone. Each listed in
    /// `view.headers`. Left out under `routing_key` too.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headers: Vec<String>,
    /// How long an idle key stays pinned, in seconds, before the rule's
    /// next sweep forgets it; above zero (default 14400), `4294967295` for
    /// ever.
    #[serde(default = "default_ttl_s")]
    #[schemars(range(min = 1))]
    pub ttl_s: u32,
    /// The sweep cadence of idle keys, in seconds: a key idle for `ttl_s`
    /// is forgotten at the next sweep, and one used again ahead of it stays
    /// pinned. Above zero (default 120), `4294967295` for never, allowed
    /// under a `ttl_s` for ever alone.
    #[serde(default = "default_sweep_s")]
    #[schemars(range(min = 1))]
    pub sweep_s: u32,
    /// How a first-seen key picks its worker (default `delegate`); under
    /// the `manual` picker, a request with no key as well.
    #[serde(default)]
    pub assignment: Assignment,
    /// How many requests of one key a worker carries before the next is
    /// placed again; above zero (default 2), `65535` for no cap. At the cap
    /// the stages place the request over every usable worker, and the pin
    /// moves only to another worker under the cap.
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
    /// Whether a single request this rule placed alone is the profile's
    /// own (default `told`). Under `untold` no stage hears that request's
    /// issue or its dispatch, it is none of the profile's sends, and
    /// neither the engine's record of sent prefixes nor a lent token tree
    /// takes its prompt. The session table still hears its dispatch,
    /// and the stages its completion and its release. A pair's sides are
    /// told under either word. A controller keeps the word on the lease it
    /// files, and tells such a request off its file by it all the same.
    #[serde(default, skip_serializing_if = "is_default")]
    pub rule_placed: RulePlaced,
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

/// Where a binding's requests take their routing key from: the sources a
/// host reads in order, and the first that holds a value is the key. A ring
/// places by the key and a sessions rule pins it. In a gateway's own
/// process a ring's prefix form places by a header's key alone. No value
/// enters a view, so the list is not held to `view.headers`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RoutingKeySpec {
    /// The sources, read in order: one to eight, each named once, with up
    /// to four client sources and one target source beside them.
    #[schemars(length(min = 1, max = 13))]
    pub sources: Vec<KeySourceSpec>,
    /// Take the turn and retry suffixes off a key, so every turn of a
    /// conversation shares one (`conv_t2_r1` keys as `conv`); default
    /// false.
    #[serde(default, skip_serializing_if = "is_default")]
    pub strip_lineage: bool,
}

/// One source of a routing key: exactly one of `header` and `rid`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
pub struct KeySourceSpec {
    /// A request header, lowercase. A credential header is refused, as on
    /// `view.headers`; `cookie` is not. A key from a header `view.headers`
    /// does not list leaves as its fingerprint alone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    /// The request's own id, the `rid` of its body: `true` names it. Under
    /// a ring's prefix form (`prefix_tokens`) in a gateway's own process
    /// its key names the session alone: a sessions rule pins it, and the
    /// ring places the request by its prompt. The plain ring places by it,
    /// and so does a served controller's prefix ring.
    #[serde(skip_serializing_if = "is_default")]
    pub rid: bool,
    /// The most bytes of a value this source hands, 8192 at most (default
    /// 128). A longer value is passed over and the next source is read. A
    /// header is measured as sent, and the request's own id after the
    /// block's `strip_lineage` took its suffixes off.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1, max = 8192))]
    pub max_bytes: Option<u32>,
    /// A source that tells one client from another, not one session: a
    /// header a proxy or a browser sets. Its key places a request that
    /// carries no session key on a ring. It lifts no overload veto, pins
    /// no session and is counted under no key. Of a binding's client
    /// sources, the first whose header a request carries is the one read,
    /// and a value it cannot hand ends the read. The value is read whole
    /// and leaves as its fingerprint alone, so `rid` and `max_bytes` are
    /// refused beside it. A header may be named both ways: capped as a
    /// session's source, and whole as a client's where that read handed no
    /// key. Default false.
    #[serde(skip_serializing_if = "is_default")]
    pub client: bool,
    /// A source that names the one worker the request goes to, and hands
    /// no key. The header's value is a number that counts, from 0 and in
    /// the order of the request's list of workers, the workers that can
    /// take the request when it arrives: so a number names another worker
    /// while one ahead of it is down. A request that carries it goes to
    /// that worker and to no other. Where the value is no number, or
    /// fewer workers can take the request, it is shed under
    /// `target-worker`. Where no worker at all can take it, it is placed
    /// as a request that names none and carries no key. A header that is
    /// absent, empty or not text names no worker. One entry a block,
    /// beside the eight and never without one of them: a header alone,
    /// so `rid`, `max_bytes` and `client` are refused beside it. It is
    /// read for a single request a program places in process, and for no
    /// pair. Default false.
    #[serde(skip_serializing_if = "is_default")]
    pub target: bool,
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

/// A second profile for a binding of one profile, or a profile per side.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(untagged)]
#[non_exhaustive]
pub enum ShadowSpec {
    /// The shadow of a binding that names one live profile.
    Profile(String),
    /// A pair's shadow profiles by side: `prefill`, `decode` or both.
    Sides(BTreeMap<String, String>),
}

/// Read by form, so a value that is neither is refused by what the field
/// takes, and a side written twice by its key.
impl<'de> Deserialize<'de> for ShadowSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ShadowVisitor;

        impl<'de> Visitor<'de> for ShadowVisitor {
            type Value = ShadowSpec;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a profile name, or a map of sides to profile names")
            }

            fn visit_str<E: de::Error>(self, name: &str) -> Result<ShadowSpec, E> {
                Ok(ShadowSpec::Profile(name.to_string()))
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<ShadowSpec, A::Error> {
                keyed_once(de::value::MapAccessDeserializer::new(map)).map(ShadowSpec::Sides)
            }
        }

        deserializer.deserialize_any(ShadowVisitor)
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
    /// What the binding says of the overload veto for the workers of the
    /// pools decided under it: the pools whose model and phase it is the first
    /// to match. A decode pool of a model whose prefill lookup resolves to
    /// a pair binding (`pd-pair` or `epd`) is decided under that binding,
    /// so its block judges the pool: a decode-phase binding's block is not
    /// read for it, and a pair binding with its block switched off leaves
    /// the pool unjudged. Where a binding of another plan takes the
    /// model's prefill lookup first, no pair decision runs for the model
    /// and its decode pool is the decode phase's own.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overload: Option<OverloadSpec>,
    /// Where the binding's requests take their routing key from. Absent, a
    /// sessions rule's own headers are the sources, and a binding with no
    /// rule names none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_key: Option<RoutingKeySpec>,
    /// Sticky sessions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sessions: Option<SessionsSpec>,
    /// A second profile for comparison on the same view. Only the live
    /// profile's decision serves the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow: Option<ShadowSpec>,
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

/// Which decisions a decider keeps for explain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExplainSpec {
    /// Keep every Nth pick or shed per decider under this generation. Zero
    /// turns sampling off and one keeps every pick and shed.
    #[serde(default)]
    pub sample: u32,
}

/// One declared signal: a named value of a worker's report that the
/// document's stages read beside the rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SignalSpec {
    /// The unit a source states for its value to be taken: 1 to 31 ASCII
    /// characters with no space. It is compared byte for byte with the
    /// unit a report states.
    #[schemars(length(min = 1, max = 31))]
    #[schemars(extend("pattern" = SIGNAL_UNIT_PATTERN))]
    pub unit: String,
    /// A value older than this reads as no value: 1 to 3,599,999, the
    /// bound of the view's `max_report_age_ms`. A store's clock keeps one
    /// hour, and a sample older than that stands at the clock's origin:
    /// under a longer limit it would read as fresh. Absent: a value does
    /// not end by its age.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1, max = MAX_LOAD_REPORT_AGE_MS))]
    pub max_age_ms: Option<u32>,
    /// How one row's value is made where its report holds several ranks.
    /// Absent: the sum.
    #[serde(default, skip_serializing_if = "is_default")]
    pub across_ranks: AcrossRanks,
}

/// How one row's value is made where its report holds several ranks: a
/// worker of several data-parallel ranks that stands as one row. Where
/// each rank is a row of its own, nothing is reduced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum AcrossRanks {
    /// The ranks' values added: an amount each rank holds a part of.
    #[default]
    Sum,
    /// The largest of the ranks' values.
    Max,
    /// The smallest of the ranks' values.
    Min,
    /// The mean of the ranks' values.
    Mean,
}

/// The most signals one document declares.
pub const MAX_SIGNALS: usize = 64;

/// The most bytes a signal's name takes.
const MAX_SIGNAL_NAME: usize = 63;

/// A signal's name as the schema states it: what [`check_signals`] takes.
const SIGNAL_NAME_PATTERN: &str = "^[a-z][a-z0-9_]{0,62}$";

/// The most bytes a signal's unit takes.
const MAX_SIGNAL_UNIT: usize = 31;

/// A unit's characters as the schema states them: visible ASCII, so no
/// space. One at either end of a unit would not be seen in a document and
/// would match no report. A unit's bytes are then its characters, so the
/// schema's length is the one [`check_signals`] takes.
const SIGNAL_UNIT_PATTERN: &str = "^[!-~]*$";

/// The policy specification, version `smg.policy/v1`: what the host builds
/// into the view, the per-model index settings, the signals its stages
/// read, the profiles with their stages, the plans' rules and the ordered
/// bindings. An unknown field at any level is refused by name.
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
    /// The signals the document's stages read, by name: a lower-case
    /// letter, then lower-case letters, digits and underscores, at most 63
    /// bytes. At most 64. A signal's column is its place among the names,
    /// in name order.
    #[serde(
        default,
        deserialize_with = "keyed_once",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    #[schemars(extend(
        "maxProperties" = MAX_SIGNALS,
        "propertyNames" = { "pattern": SIGNAL_NAME_PATTERN }
    ))]
    pub signals: BTreeMap<String, SignalSpec>,
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
    /// The sampling rate for explain.
    #[serde(default, skip_serializing_if = "is_default")]
    pub explain: ExplainSpec,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
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
    /// A factory's refusal or a stage whose name differs from its registration.
    #[error("{path}: {reason}")]
    BadStage { path: String, reason: String },
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
            | Self::BadStage { path, .. }
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

/// The matcher understands an id, `prefix*`, `*suffix` and `*`; any other
/// star would match only its literal self.
fn glob_ok(model: &str) -> bool {
    let stars = model.matches('*').count();
    !model.is_empty()
        && (stars == 0 || (stars == 1 && (model.starts_with('*') || model.ends_with('*'))))
}

/// A lowercase header name outside the deny-lists.
fn check_header_name(name: &str, path: String, denied: &[&str]) -> Result<(), SpecError> {
    check_header_outside(name, path, &DENIED_HEADERS, denied)
}

/// A lowercase header name outside `stock` and the host's own `denied`.
fn check_header_outside(
    name: &str,
    path: String,
    stock: &[&str],
    denied: &[&str],
) -> Result<(), SpecError> {
    let token =
        |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit() || b"!#$%&'*+-.^_`|~".contains(&b);
    if name.is_empty() || !name.bytes().all(token) {
        return Err(SpecError::BadField {
            path,
            value: name.to_string(),
            reason: "a lowercase header name",
        });
    }
    if stock.contains(&name) || denied.iter().any(|d| d.eq_ignore_ascii_case(name)) {
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
    at.number(
        "max_report_age_ms",
        view.max_report_age_ms,
        |ms| (1..=u64::from(MAX_LOAD_REPORT_AGE_MS)).contains(&ms),
        "a whole number from 1 to 3599999",
    )?;
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

/// The `signals` section's checks, each refused by its path: at most
/// [`MAX_SIGNALS`] names, each a lower-case identifier of at most 63
/// bytes, a unit of 1 to 31 ASCII characters with no space, and a maximum
/// age of 1 to 3,599,999 where one is written. A signal no stage reads is allowed: a host's own
/// stage may read it.
fn check_signals(signals: &BTreeMap<String, SignalSpec>) -> Result<(), SpecError> {
    if signals.len() > MAX_SIGNALS {
        return Err(SpecError::TooManyListed {
            path: "signals",
            count: signals.len(),
            max: MAX_SIGNALS,
        });
    }
    for (name, spec) in signals {
        let mut bytes = name.bytes();
        let named = name.len() <= MAX_SIGNAL_NAME
            && bytes.next().is_some_and(|first| first.is_ascii_lowercase())
            && bytes.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
        if !named {
            return Err(SpecError::BadField {
                path: "signals".to_owned(),
                value: name.clone(),
                reason: "a lower-case name of letters, digits and underscores, at most 63 bytes",
            });
        }
        // The key's path is built at a refusal alone.
        let at = || At::new(format!("signals.{name}"));
        let visible = spec.unit.bytes().all(|b| b.is_ascii_graphic());
        if spec.unit.is_empty() || spec.unit.len() > MAX_SIGNAL_UNIT || !visible {
            return Err(at().bad(
                "unit",
                format!("{:?}", spec.unit),
                "a unit of 1 to 31 ASCII characters with no space",
            ));
        }
        match spec.max_age_ms {
            Some(0) => return Err(at().bad("max_age_ms", 0, "above zero")),
            Some(ms) if ms > MAX_LOAD_REPORT_AGE_MS => {
                return Err(at().bad("max_age_ms", ms, "a whole number from 1 to 3599999"));
            }
            _ => {}
        }
    }
    Ok(())
}

/// The document's signals, resolved once: one column a name, in name
/// order, so a name's place in the list is its column's number.
fn resolve_signals(signals: &BTreeMap<String, SignalSpec>) -> Arc<[SignalColumn]> {
    signals
        .iter()
        .zip(0u16..)
        .map(|((name, spec), column)| {
            SignalColumn::new(name, &spec.unit, column, spec.max_age_ms).across(spec.across_ranks)
        })
        .collect()
}

/// What a term's written name reads: a field of this build, or a signal
/// the document declares.
enum Named {
    Field(CostTerm),
    Signal(SignalColumn),
}

/// The term `name` in the map `terms` of the stage at `at`. A name of the
/// signal family (`signal.<name>`) reads a signal the document declares,
/// and any other name a field of this build. A name that reads neither is
/// refused at the map, by the name as written.
fn named_term(
    signals: &[SignalColumn],
    at: &At,
    terms: &str,
    name: &str,
) -> Result<Named, SpecError> {
    match signal_named(name) {
        Some(signal) => column_named(signals, signal)
            .map(|column| Named::Signal(column.clone()))
            .ok_or_else(|| at.bad(terms, name, "a signal the document declares")),
        None => CostTerm::named(name)
            .map(Named::Field)
            .ok_or_else(|| at.bad(terms, name, "a field this build reads")),
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
        ScoreStage::LeastCost(_) => Ok(()),
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

/// Check the cost parameters while resolving the field names into readers
/// and the signals' names into their columns.
fn least_cost_of(
    p: &LeastCostParams,
    signals: &[SignalColumn],
    sent: impl FnOnce() -> Arc<DispatchCredit>,
    max_age_ms: Option<u64>,
    at: &At,
) -> Result<Box<dyn Scorer>, SpecError> {
    const COST_RANGE: &str = "a finite number from 0 to 1e15";
    if f64::from(p.weight) > COST_PARAMETER_LIMIT {
        return Err(at.bad(
            "weight",
            p.weight,
            "a finite number above zero and at most 1e15",
        ));
    }
    if p.terms.is_empty() {
        return Err(at.bad("terms", "{}", "at least one field with its coefficient"));
    }
    let mut terms = Vec::new();
    let mut signal_terms = Vec::new();
    for (name, &coefficient) in &p.terms {
        let term = named_term(signals, at, "terms", name)?;
        if coefficient == 0.0
            || !(-COST_PARAMETER_LIMIT..=COST_PARAMETER_LIMIT).contains(&coefficient)
        {
            return Err(at.sub("terms").bad(
                name,
                coefficient,
                "a finite number other than zero, of magnitude at most 1e15",
            ));
        }
        match term {
            Named::Field(term) => terms.push((term, coefficient)),
            Named::Signal(signal) => signal_terms.push((signal, coefficient)),
        }
    }
    if !(0.0..=COST_PARAMETER_LIMIT).contains(&p.margin.absolute) {
        return Err(at
            .sub("margin")
            .bad("absolute", p.margin.absolute, COST_RANGE));
    }
    if !(0.0..=RELATIVE_MARGIN_LIMIT).contains(&p.margin.relative) {
        return Err(at.sub("margin").bad(
            "relative",
            p.margin.relative,
            "a finite number from 0 to 1e6",
        ));
    }
    if let Some(Unreported::OverLeast { request_cost }) = p.unreported {
        if !(0.0..=COST_PARAMETER_LIMIT).contains(&request_cost) {
            return Err(at.sub("unreported.over-least").bad(
                "request_cost",
                request_cost,
                COST_RANGE,
            ));
        }
    }
    let reads_sent = terms.iter().any(|&(term, _)| term.reads_sent());
    let scorer = LeastCost::new(terms.into_boxed_slice(), p.margin, p.unreported);
    // A row with no fresh value for a named signal is unreported for the
    // scorer, as a row with no report is: the document says what such a
    // row costs.
    if (scorer.needs().loads || !signal_terms.is_empty()) && p.unreported.is_none() {
        return Err(at.bad(
            "unreported", "null",
            "what a row with no report, or with no fresh value of a signal, costs, which a term that reads a report or a signal needs: over-least or missing-as-zero",
        ));
    }
    // Only a term that reads sends makes the profile's count, and only a
    // term that reads a signal the scorer that adds one.
    Ok(if !signal_terms.is_empty() {
        let scorer = LeastCostSignals::new(
            scorer,
            reads_sent.then(sent),
            signal_terms.into_boxed_slice(),
        );
        with_load_report_age_limit(scorer, max_age_ms)
    } else if reads_sent {
        with_load_report_age_limit(LeastCostWithDispatches::new(scorer, sent()), max_age_ms)
    } else {
        with_load_report_age_limit(scorer, max_age_ms)
    })
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

/// A consistent-hash stage's prefix depths against the view. A key ends
/// with the last whole block at or under its depth, so `prefix_tokens` and
/// each of the view's boundaries hold at least one block and no more blocks
/// than a request carries.
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
/// the view's block size. Of two entries of a kind the last one's stands
/// here; a `prefix-affinity` scorer spills by its own entry's gate.
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

/// The stage's scorer, with its profile's gates and the view's settings.
/// `sent` hands out the profile's count of sends.
fn scorer_of(
    stage: &ScoreStage,
    gates: Gates,
    view: &ViewSpec,
    signals: &[SignalColumn],
    sent: impl FnOnce() -> Arc<DispatchCredit>,
    at: &At,
) -> Result<Resolved<Box<dyn Scorer>>, SpecError> {
    let max_age = view.max_report_age_ms;
    Ok(match stage {
        ScoreStage::LeastLoad(p) => Resolved::Built(with_load_report_age_limit(
            LeastLoad::of_profile(
                p.kv_pressure_weight.unwrap_or(DEFAULT_KV_PRESSURE_WEIGHT),
                p.mean_prefill_tokens.unwrap_or(DEFAULT_MEAN_PREFILL_TOKENS),
                p.default_throughput.unwrap_or(DEFAULT_THROUGHPUT),
                sent(),
            ),
            max_age,
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
        ScoreStage::PrefixAffinity(p) => {
            // The entry spills by the gate it writes itself. The KV gate
            // and the block size stay the profile's.
            let own = Gates::of(Some(p), None, None);
            let gates = Gates {
                rel: own.rel,
                abs: own.abs,
                prefix_margin_tokens: own.prefix_margin_tokens,
                ..gates
            };
            Resolved::Built(with_load_report_age_limit(
                PrefixAffinity::new(p, gates),
                max_age,
            ))
        }
        ScoreStage::KvPressure(_) => built_scorer(KvPressure),
        ScoreStage::WarmupSlice(p) => Resolved::Built(with_load_report_age_limit(
            WarmupSlice::new(p, gates),
            max_age,
        )),
        ScoreStage::Bucket(p) => built_scorer(Bucket::new(p)),
        ScoreStage::ConsistentHash(p) => {
            let ring = ConsistentHash::new(p.bounded_load.as_ref());
            built_scorer(match p.prefix_tokens {
                Some(tokens) => ring.keyed_by_prefix(tokens, gates, &view.boundaries),
                None => ring,
            })
        }
        ScoreStage::ResidentWork(p) => Resolved::Built(with_load_report_age_limit(
            ResidentWork::new(p.request_tokens),
            max_age,
        )),
        ScoreStage::LeastCost(p) => Resolved::Built(least_cost_of(p, signals, sent, max_age, at)?),
    })
}

/// The profile's picker. The catalog is asked for a name first, as it is
/// for a filter: a picker it builds from the document's map is handed what
/// stands under the name, a picker registered with no parameter is written
/// by its name alone, and a name it does not hold is refused as unknown
/// whatever is written under it. `sent` hands a factory that asks for it
/// the profile's count of sends.
fn picker_of(
    pick: &PickSpec,
    profile: &str,
    at: &At,
    from: StageInputs<'_>,
    settings: SentSettings,
    sent: &mut SentSlot,
) -> Result<Resolved<Box<dyn Picker>>, SpecError> {
    let named = pick.name();
    let path = || format!("{}.pick.{named}", at.0);
    // The catalog's factory builds its picker from the document's map. A
    // picker has no weight: the map is the factory's whole.
    let made = |build: &PickerFactory,
                mut params: serde_json::Map<String, serde_json::Value>,
                sent: &mut SentSlot| {
        in_key_order(&mut params);
        let picker = StageBuild::built(
            &params,
            profile,
            settings,
            sent,
            from.reports,
            from.signals,
            build,
            Picker::needs,
        )
        .map_err(|error| error.at(&path()))?;
        let answers = picker.name();
        if answers != named {
            return Err(SpecError::BadStage {
                path: path(),
                reason: format!("picker registered as {named:?} answers the name {answers:?}"),
            });
        }
        Ok(Resolved::Built(picker))
    };
    let unknown = || SpecError::UnknownPicker {
        profile: profile.to_string(),
        name: named.to_string(),
    };
    match pick {
        PickSpec::Name(name) => match from.catalog.find_picker(name) {
            Some(Found::Built(picker)) => {
                // A picker the catalog built with no context.
                no_cells_without_a_context(
                    picker.needs(),
                    "register it by picker_with to resolve one, or leave the need unset",
                    path,
                )?;
                Ok(Resolved::Built(picker))
            }
            Some(Found::With(build)) => made(build, serde_json::Map::new(), sent),
            // The one picker with parameters, named without them.
            Some(Found::Typed) => Err(at.bad(
                "pick",
                name,
                "a picker with its temperature: { best-with-temperature: { temperature } }",
            )),
            None => Err(unknown()),
        },
        PickSpec::Params(PickStage::Best(p)) => {
            if !(1..=MAX_SAMPLE).contains(&p.sample) {
                return Err(at.sub("pick").sub("best").bad(
                    "sample",
                    p.sample,
                    "a whole number from 1 to 64",
                ));
            }
            Ok(Resolved::Built(Box::new(SampledBest::new(p.sample))))
        }
        PickSpec::Params(PickStage::BestWithTemperature(p)) => {
            if !at_or_above_zero(p.temperature) {
                return Err(at.sub("pick").sub("best-with-temperature").bad(
                    "temperature",
                    p.temperature,
                    "a finite number at or above zero",
                ));
            }
            Ok(Resolved::NotInBuild("the best-with-temperature picker"))
        }
        PickSpec::Added { name, written } => match from.catalog.find_picker(name) {
            // Only a document built in code: the reader reads a typed
            // built-in's key into its parameter struct.
            _ if TYPED_PICKERS.contains(&name.as_str()) => Err(SpecError::BadField {
                path: path(),
                value: name.clone(),
                reason: "a picker added in code: a built-in's parameters are typed",
            }),
            Some(Found::With(build)) => {
                let params = written.as_object().ok_or_else(|| SpecError::BadField {
                    path: path(),
                    value: written.to_string(),
                    reason: "a map of the picker's parameters",
                })?;
                made(build, params.clone(), sent)
            }
            // A picker registered with no parameter is written by its name
            // alone: whatever stands under it is refused.
            Some(Found::Built(_)) => Err(SpecError::BadField {
                path: path(),
                value: written.to_string(),
                reason: "left out: this picker takes no parameter, write its bare name",
            }),
            // The catalog holds no picker by its type alone under a name the
            // reader kept as read.
            None | Some(Found::Typed) => Err(unknown()),
        },
    }
}

/// Holds a stage the catalog built with no context to the signal cells,
/// as [`StageBuild::built`] holds one a factory built under a context. It
/// resolved no signal, so its needs declare no cells: a host would hand
/// cells nothing reads. `remedy` is what the stage's author can do, and
/// `path` is the stage's own, made where a refusal names it.
fn no_cells_without_a_context(
    needs: Needs,
    remedy: &'static str,
    path: impl FnOnce() -> String,
) -> Result<(), SpecError> {
    StageBuild::held_to_cells(false, needs, remedy).map_err(|error| error.at(&path()))
}

/// What a document's stages are built from: the catalog that makes them,
/// the view's reading of a report, which a factory's build context hands
/// out, and the document's signals, resolved, which it answers from.
#[derive(Clone, Copy)]
struct StageInputs<'a> {
    catalog: &'a StageRegistry,
    reports: LoadReportReader,
    signals: &'a [SignalColumn],
}

/// What a filter entry stands for.
enum Filtered<'a> {
    /// A built-in, by the parameters the document wrote.
    Typed(&'a FilterStage),
    /// A filter the catalog built.
    Built(Box<dyn Filter>),
}

/// What the entry at `index` of a profile's list stands for. The catalog
/// is asked for the name first: a name it does not hold is refused as
/// unknown whatever is written under it. `at` is the profile's path: the
/// entry's own is written where a refusal names it, so an entry that
/// validates builds none.
fn filtered<'a>(
    entry: &'a FilterSpec,
    profile: &str,
    at: &At,
    index: usize,
    from: StageInputs<'_>,
    settings: SentSettings,
    sent: &mut SentSlot,
) -> Result<Filtered<'a>, SpecError> {
    let name = entry.name();
    let path = || format!("{}.filter[{index}].{name}", at.0);
    // The catalog's factory builds its filter from the document's map.
    // A filter has no weight: the map is the factory's whole.
    let made = |build: &FilterFactory,
                mut params: serde_json::Map<String, serde_json::Value>,
                sent: &mut SentSlot| {
        in_key_order(&mut params);
        let filter = StageBuild::built(
            &params,
            profile,
            settings,
            sent,
            from.reports,
            from.signals,
            build,
            Filter::needs,
        )
        .map_err(|error| error.at(&path()))?;
        let answers = filter.name();
        if answers != name {
            return Err(SpecError::BadStage {
                path: path(),
                reason: format!("filter registered as {name:?} answers the name {answers:?}"),
            });
        }
        Ok(Filtered::Built(filter))
    };
    match (from.catalog.find_filter(name), entry) {
        // Only a document built in code: the parser reads the built-in's
        // key into its parameter struct. The catalog holds the name for
        // its bare form, and that arm's refusal would name the bare form.
        (_, FilterSpec::Added { .. }) if name == HasRoomRule::NAME => Err(SpecError::BadField {
            path: format!("{}.filter[{index}]", at.0),
            value: name.to_string(),
            reason: "a filter added in code: a built-in's parameters are typed",
        }),
        // A built-in's parameters are the validator's to check, under a
        // name the catalog holds.
        (Some(Found::Built(_) | Found::With(_)), FilterSpec::Params(stage)) => {
            Ok(Filtered::Typed(stage))
        }
        (Some(Found::Built(filter)), FilterSpec::Name(_)) => {
            no_cells_without_a_context(
                filter.needs(),
                "register it by filter_with to resolve one, or leave the need unset",
                path,
            )?;
            Ok(Filtered::Built(filter))
        }
        // A filter registered with no parameter is written by its name
        // alone: whatever stands under it is refused.
        (Some(Found::Built(_)), FilterSpec::Added { written, .. }) => Err(SpecError::BadField {
            path: path(),
            value: written.to_string(),
            reason: "left out: this filter takes no parameter, write its bare name",
        }),
        (Some(Found::With(build)), FilterSpec::Name(_)) => {
            made(build, serde_json::Map::new(), sent)
        }
        (Some(Found::With(build)), FilterSpec::Added { written, .. }) => {
            let params = written.as_object().ok_or_else(|| SpecError::BadField {
                path: path(),
                value: written.to_string(),
                reason: "a map of the filter's parameters",
            })?;
            made(build, params.clone(), sent)
        }
        // The catalog holds no filter by its type alone.
        (None | Some(Found::Typed), _) => Err(SpecError::UnknownFilter {
            profile: profile.to_string(),
            name: name.to_string(),
        }),
    }
}

/// The stage's filter, with the view's settings. `sent` hands out the
/// profile's count of sends.
fn filter_of(
    stage: &FilterStage,
    view: &ViewSpec,
    signals: &[SignalColumn],
    sent: impl FnOnce() -> Arc<DispatchCredit>,
    at: &At,
) -> Result<Box<dyn Filter>, SpecError> {
    match stage {
        FilterStage::HasRoom(p) => has_room_of(p, signals, sent, view.max_report_age_ms, at),
    }
}

/// Check the rule's parameters while resolving the field names into
/// readers and the signals' names into their columns. Only a term that
/// reads sends makes the profile's count.
fn has_room_of(
    p: &HasRoomParams,
    signals: &[SignalColumn],
    sent: impl FnOnce() -> Arc<DispatchCredit>,
    max_age_ms: Option<u64>,
    at: &At,
) -> Result<Box<dyn Filter>, SpecError> {
    // The first name, in name order, which is the map's own: a rule that
    // comes to read nothing of a worker is refused by it.
    let Some(first) = p.used.keys().next().map(String::as_str) else {
        return Err(at.bad("used", "{}", "at least one field with its coefficient"));
    };
    let mut terms = Vec::new();
    let mut signal_terms = Vec::new();
    let mut request_tokens = 0.0;
    let mut output_tokens = 0.0;
    for (name, &coefficient) in &p.used {
        // The request's own values are read off the request: no term of a
        // worker stands for one.
        let term = if name == HasRoomRule::REQUEST_TOKENS || name == HasRoomRule::OUTPUT_TOKENS {
            None
        } else {
            Some(named_term(signals, at, "used", name)?)
        };
        if coefficient == 0.0
            || !(-COST_PARAMETER_LIMIT..=COST_PARAMETER_LIMIT).contains(&coefficient)
        {
            return Err(at.sub("used").bad(
                name,
                coefficient,
                "a finite number other than zero, of magnitude at most 1e15",
            ));
        }
        match term {
            Some(Named::Field(term)) => terms.push((term, coefficient)),
            Some(Named::Signal(signal)) => signal_terms.push((signal, coefficient)),
            None if name == HasRoomRule::OUTPUT_TOKENS => output_tokens = coefficient,
            None => request_tokens = coefficient,
        }
    }
    // Under a signal's capacity the rule's own bound is that of a capacity
    // of one: a row's value scales it.
    let (capacity, capacity_signal) = match &p.capacity {
        Capacity::Fixed(capacity) if capacity.is_finite() && *capacity > 0.0 => (*capacity, None),
        Capacity::Fixed(capacity) => {
            return Err(at.bad("capacity", capacity, "a finite number above zero"));
        }
        // A name is a declared signal's: no field of this build holds a
        // capacity.
        Capacity::Field(name) => match signal_named(name) {
            Some(signal) => {
                let column = column_named(signals, signal)
                    .ok_or_else(|| at.bad("capacity", name, "a signal the document declares"))?;
                (1.0, Some(column.clone()))
            }
            None => {
                return Err(at.bad("capacity", name, "a declared signal, written signal.<name>"))
            }
        },
    };
    // Under a capacity that is a number, the request's own values alone are
    // one answer for every row: such a rule reads nothing of a worker, and
    // it keeps every row or steps aside. A signal in the sum reads a worker,
    // and so does a capacity read of each worker, so both are settled
    // first. Every name of such a rule is the request's, since any other
    // became a term or was refused: the refusal names the first of them.
    if terms.is_empty() && signal_terms.is_empty() && capacity_signal.is_none() {
        return Err(at.bad(
            "used",
            first,
            "at least one field or signal of the worker beside the request's own values",
        ));
    }
    if !(0.0..1.0).contains(&p.headroom) {
        return Err(at.bad("headroom", p.headroom, "a number from 0 to under 1"));
    }
    let reads_sent = terms.iter().any(|&(term, _)| term.reads_sent());
    let rule = HasRoomRule::new(
        &terms,
        request_tokens,
        output_tokens,
        capacity,
        p.headroom,
        reads_sent.then(sent),
    );
    // The signal form is a type of its own: a rule that names no signal
    // is built as it is with no signal declared.
    Ok(if signal_terms.is_empty() && capacity_signal.is_none() {
        has_room_rule(rule, max_age_ms)
    } else {
        Box::new(HasRoomSignals::new(
            rule,
            signal_terms.into_boxed_slice(),
            capacity_signal,
            max_age_ms,
        ))
    })
}

/// What a score entry stands for.
enum Scored {
    /// A built-in, by its parameters: the written ones, or a bare name's
    /// defaults.
    Typed(ScoreStage),
    /// A scorer the catalog built, with its weight.
    Built(WeightedScorer),
}

/// The stage a score entry stands for. The catalog is asked for the name
/// first: a name it does not hold is refused as unknown whatever is written
/// under it, so a misspelt built-in reads its name and not a key it carries.
fn scored(
    entry: &ScoreSpec,
    profile: &str,
    at: &At,
    from: StageInputs<'_>,
    settings: SentSettings,
    sent: &mut SentSlot,
) -> Result<Scored, SpecError> {
    let name = entry.name();
    let Some(found) = from.catalog.find_scorer(name) else {
        return Err(SpecError::UnknownScorer {
            profile: profile.to_string(),
            name: name.to_string(),
        });
    };
    // A scorer the catalog built with no context, bare or under a weight.
    if let Found::Built(scorer) = &found {
        no_cells_without_a_context(
            scorer.needs(),
            "register it by scorer_with to resolve one, or leave the need unset",
            || at.field(name),
        )?;
    }
    match (entry, found) {
        (ScoreSpec::Params(stage), _) => Ok(Scored::Typed(stage.clone())),
        (ScoreSpec::Name(_), Found::Typed) => match ScoreStage::by_name(name) {
            Some(stage) => Ok(Scored::Typed(stage)),
            None => Err(SpecError::BadField {
                path: at.0.clone(),
                value: name.to_string(),
                reason: match name {
                    "resident-work" => {
                        "a scorer with its request_tokens: { resident-work: { request_tokens } }"
                    }
                    LeastCost::NAME => "a scorer with its terms: { least-cost: { terms } }",
                    _ => "a scorer with its required parameters",
                },
            }),
        },
        (ScoreSpec::Name(_), Found::Built(scorer)) => Ok(Scored::Built(WeightedScorer {
            scorer,
            weight: 1.0,
        })),
        (ScoreSpec::Added { written, .. }, Found::Built(scorer)) => {
            Ok(Scored::Built(WeightedScorer {
                scorer,
                weight: added_weight(written, &at.sub(name))?,
            }))
        }
        (entry, Found::With(build)) => {
            let at = at.sub(name);
            let mut params = if let ScoreSpec::Added { written, .. } = entry {
                added_parameters(written, &at)?.clone()
            } else {
                serde_json::Map::new()
            };
            let weight = match params.shift_remove("weight") {
                Some(value) => check_added_weight(&value, &at)?,
                None => 1.0,
            };
            in_key_order(&mut params);
            let scorer = StageBuild::built(
                &params,
                profile,
                settings,
                sent,
                from.reports,
                from.signals,
                build,
                Scorer::needs,
            )
            .map_err(|error| error.at(&at.0))?;
            let answers = scorer.name();
            if answers != name {
                return Err(SpecError::BadStage {
                    path: at.0,
                    reason: format!("scorer registered as {name:?} answers the name {answers:?}"),
                });
            }
            Ok(Scored::Built(WeightedScorer { scorer, weight }))
        }
        // Only a document built in code: the parser reads a built-in's key
        // into its parameter struct.
        (ScoreSpec::Added { .. }, Found::Typed) => Err(SpecError::BadField {
            path: at.0.clone(),
            value: name.to_string(),
            reason: "a scorer added in code: a built-in's parameters are typed",
        }),
    }
}

/// One order of a stage's parameters for its factory, the canonical bytes
/// and the digest: the keys of every map in order, whatever order the
/// document wrote them in.
fn in_key_order(params: &mut serde_json::Map<String, serde_json::Value>) {
    params.sort_keys();
    params
        .values_mut()
        .for_each(serde_json::Value::sort_all_objects);
}

fn added_parameters<'a>(
    written: &'a serde_json::Value,
    at: &At,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, SpecError> {
    written.as_object().ok_or_else(|| SpecError::BadField {
        path: at.0.clone(),
        value: written.to_string(),
        reason: "a map of the scorer's weight: { weight }",
    })
}

/// The weight of a scorer whose factory takes no parameters. Any other key
/// is refused by name. An empty map weighs one.
fn added_weight(written: &serde_json::Value, at: &At) -> Result<f32, SpecError> {
    let keys = added_parameters(written, at)?;
    let mut weight = 1.0;
    for (key, value) in keys {
        if key != "weight" {
            return Err(at.bad(
                key,
                value,
                "a parameter of a scorer added in code, which takes its weight alone",
            ));
        }
        weight = check_added_weight(value, at)?;
    }
    Ok(weight)
}

fn check_added_weight(value: &serde_json::Value, at: &At) -> Result<f32, SpecError> {
    // A number past an `f32` narrows to infinity and is refused below.
    let weight = value.as_f64().map_or(f32::NAN, |number| number as f32);
    if !above_zero(weight) {
        return Err(at.bad("weight", value, "a finite number above zero"));
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
/// and whether the profile keeps a session key on its worker over the
/// overload veto: the plain ring alone decides its pick. `depth` is
/// deepened to what the built stages read of a request's block hashes.
fn check_profile(
    name: &str,
    p: &ProfileSpec,
    view: &ViewSpec,
    from: StageInputs<'_>,
    deferred: &mut Deferred,
    depth: &mut HashDepth,
    sent_credits: &mut Vec<Arc<DispatchCredit>>,
) -> Result<Option<(Profile, bool)>, SpecError> {
    let catalog = from.catalog;
    let gates = gates_of(p, view);
    let at = At::new(format!("profiles.{name}"));
    at.nonzero("sent_ttl_ms", p.sent_ttl_ms)?;
    at.nonzero("sent_report_delay_ms", p.sent_report_delay_ms)?;
    let settings = SentSettings {
        ttl_ms: p.sent_ttl_ms,
        report_delay_ms: p.sent_report_delay_ms,
    };
    // The profile's count of sends, made where a stage's factory first
    // asks for it: a filter's may, ahead of any scorer's. An earlier
    // profile's count under the same two settings is the one handed.
    let mut sent = SentSlot {
        asked: None,
        shared: sent_credits
            .iter()
            .find(|count| count.settings == settings)
            .cloned(),
    };
    let mut filters = Vec::with_capacity(p.filter.len());
    for (index, entry) in p.filter.iter().enumerate() {
        let filter = match filtered(entry, name, &at, index, from, settings, &mut sent)? {
            Filtered::Built(filter) => filter,
            Filtered::Typed(stage) => {
                let stage_at = At::new(format!("{}.filter[{index}].{}", at.0, stage.name()));
                let count = || profile_sent(&mut sent, name, settings);
                filter_of(stage, view, from.signals, count, &stage_at)?
            }
        };
        filters.push(filter);
    }
    let mut scorers = Vec::with_capacity(p.score.len());
    let mut complete = true;
    let mut plain_ring = false;
    for (i, entry) in p.score.iter().enumerate() {
        let entry_at = At::new(format!("{}.score[{i}]", at.0));
        let stage = match scored(entry, name, &entry_at, from, settings, &mut sent)? {
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
        let count = || profile_sent(&mut sent, name, settings);
        match scorer_of(&stage, gates, view, from.signals, count, &stage_at)? {
            Resolved::Built(scorer) => {
                let read = HashDepth::of_scorer(&stage, &view.boundaries, scorer.needs());
                depth.deepen(read);
                plain_ring |= matches!(
                    &stage,
                    ScoreStage::ConsistentHash(ring)
                        if ring.prefix_tokens.is_none() && ring.bounded_load.is_none()
                );
                scorers.push(WeightedScorer {
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
    let picker = match picker_of(&p.pick, name, &at, from, settings, &mut sent)? {
        Resolved::Built(picker) => Some(picker),
        Resolved::NotInBuild(what) => {
            deferred.note(at.field("pick"), what);
            None
        }
    };
    if let Some(ms) = p.sent_ttl_ms.filter(|_| sent.asked.is_none()) {
        return Err(at.bad("sent_ttl_ms", ms, "a limit any stage of this profile reads"));
    }
    if let Some(ms) = p.sent_report_delay_ms.filter(|_| sent.asked.is_none()) {
        return Err(at.bad(
            "sent_report_delay_ms",
            ms,
            "a delay any stage of this profile reads",
        ));
    }
    // The scores combine by each weight's share of the weights' sum: a sum
    // that is not finite, or whose reciprocal is not, gives shares that
    // lose the order of every row. A stage this build does not carry adds
    // no weight, so a deferred profile's sum is over part of its list. An
    // infinite part is an infinite whole, every weight being above zero,
    // and a part's reciprocal says nothing of the whole's.
    let total: f32 = scorers.iter().map(|weighted| weighted.weight).sum();
    let no_shares = !total.is_finite() || (complete && !total.recip().is_finite());
    if !scorers.is_empty() && no_shares {
        return Err(at.bad(
            "score",
            total,
            "weights whose sum, and one over that sum, are finite numbers",
        ));
    }
    // The key's mark: the plain ring alone decides the profile's pick. It
    // is the only scorer, with no bound, under `best` named bare, so the
    // key's owner holds the best score and the picker takes it. The engine
    // lifts the overload veto for the whole run of a marked profile, and
    // only in this shape does that run keep the key on its owner. A bound
    // passes an owner over and ranks no row, another scorer can outweigh
    // the owner, and a picker that draws, turns or samples does not follow
    // the ring: each keeps the veto for a keyed request. The prefix form
    // takes no mark either, though a request's key leads its prefix there:
    // the gateway's prefix hash moves a key off a worker over its
    // thresholds.
    let by_key = plain_ring
        && p.score.len() == 1
        && matches!(&p.pick, PickSpec::Name(picker) if picker == "best");
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
            // A shared count stands in the list under the profile that
            // made it.
            if sent.shared.is_none() {
                sent_credits.extend(sent.asked);
            }
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
    live_profiles: &'a BTreeSet<&'a str>,
    scoped_stages: &'a BTreeMap<&'a str, &'static str>,
    /// The profiles whose built picker leaves the placement to a sessions
    /// rule ([`Picker::by_assignment`]), by name.
    by_assignment: &'a [&'a str],
    /// Whether the host makes the lifecycle's calls
    /// ([`StageRegistry::makes_lifecycle_calls`]).
    lifecycle: bool,
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

/// The thresholds a binding hands its pools: the crate's two defaults, and
/// none when the block is switched off. A binding with no block has the
/// default block. A block names no number of its own ([`check_overload`]):
/// a worker's registration states its two over these. The usage goes to
/// permille by the one conversion of a threshold ([`threshold_permille`]):
/// the rounding a report's usage takes, so one rank's usage at or above the
/// threshold as a ratio is never read as under it.
fn overload_thresholds(block: Option<&OverloadSpec>) -> Option<OverloadThresholds> {
    let enabled = block.is_none_or(|block| block.enabled);
    enabled.then(|| OverloadThresholds {
        waiting_requests: OVERLOAD_WAITING_REQUESTS,
        kv_usage_permille: threshold_permille(f64::from(OVERLOAD_TOKEN_USAGE)),
    })
}

/// A binding's overload block names no number, with its switch on or off:
/// a block that names one is refused by the number's path. One set of
/// thresholds stands for each worker, its registration's own over its
/// program's default, and a worker is judged by that set alone.
fn check_overload(overload: &OverloadSpec, at: &At) -> Result<(), SpecError> {
    const STATED: &str =
        "left out: a worker's registration states this threshold, or its program's default does";
    if let Some(waiting) = overload.waiting_requests {
        return Err(at.bad("waiting_requests", waiting, STATED));
    }
    match overload.token_usage {
        Some(usage) => Err(at.bad("token_usage", usage, STATED)),
        None => Ok(()),
    }
}

fn check_sessions(
    sessions: &SessionsSpec,
    by_routing_key: bool,
    cx: &BindingContext<'_>,
    at: &At,
) -> Result<(), SpecError> {
    if by_routing_key {
        // A binding names its key's sources once: under `routing_key` the
        // rule's own spelling is refused by name.
        for (name, written) in [
            ("sticky_header", !sessions.sticky_header.is_empty()),
            ("headers", !sessions.headers.is_empty()),
        ] {
            if written {
                return Err(SpecError::BadSessions {
                    index: cx.index,
                    name,
                    reason: "left out while the binding names routing_key",
                });
            }
        }
    } else if sessions.sticky_header.is_empty() {
        return Err(SpecError::BadSessions {
            index: cx.index,
            name: "sticky_header",
            reason: "named while the binding names no routing_key",
        });
    } else {
        check_session_headers(sessions, cx, at)?;
    }
    at.nonzero("ttl_s", Some(sessions.ttl_s))?;
    at.nonzero("sweep_s", Some(sessions.sweep_s))?;
    at.nonzero("inflight_cap", Some(sessions.inflight_cap))
}

/// The headers a sessions rule names itself: each a header name on the
/// view's allow-list, the sticky header among the listed ones.
fn check_session_headers(
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
    Ok(())
}

/// A binding's routing key sources: one to eight entries, with up to four
/// client entries and one target entry beside them, never those with none
/// of the eight, each a header outside the credential names or the
/// request's own id, each named once as a session's source, once as a
/// client's and once as the target's, with a byte cap inside the schema's.
fn check_routing_key(
    block: &RoutingKeySpec,
    cx: &BindingContext<'_>,
    at: &At,
) -> Result<(), SpecError> {
    let targets = block.sources.iter().filter(|source| source.target).count();
    let clients = block
        .sources
        .iter()
        .filter(|source| source.client && !source.target);
    let clients = clients.count();
    let listed = block.sources.len() - clients - targets;
    if block.sources.is_empty() || listed > MAX_KEY_SOURCES {
        return Err(at.bad(
            "sources",
            format_args!("{listed} entries"),
            "a list of one to eight sources",
        ));
    }
    if clients > MAX_CLIENT_SOURCES {
        return Err(at.bad(
            "sources",
            format_args!("{clients} client entries"),
            "a list of four client sources at most",
        ));
    }
    if targets > 1 {
        return Err(at.bad(
            "sources",
            format_args!("{targets} target entries"),
            "a list of one target source at most",
        ));
    }
    let mut seen = BTreeSet::new();
    for (i, source) in block.sources.iter().enumerate() {
        let at = At::new(format!("{}.sources[{i}]", at.0));
        let name = match (&source.header, source.rid) {
            (Some(name), false) => {
                check_header_outside(name, at.field("header"), &DENIED_KEY_HEADERS, cx.denied)?;
                name.as_str()
            }
            (None, true) => "rid",
            (written, _) => {
                return Err(SpecError::BadField {
                    path: at.0,
                    value: written.as_ref().map_or("neither", |_| "both").to_string(),
                    reason: "one of header and rid",
                });
            }
        };
        if source.client && (source.rid || source.max_bytes.is_some()) {
            return Err(SpecError::BadField {
                path: at.field("client"),
                value: "true".to_string(),
                reason: "a header read whole, with no rid and no max_bytes beside it",
            });
        }
        if source.target && (source.rid || source.max_bytes.is_some() || source.client) {
            return Err(SpecError::BadField {
                path: at.field("target"),
                value: "true".to_string(),
                reason: "a header alone, with no rid, no max_bytes and no client beside it",
            });
        }
        if !seen.insert((source.rid, source.client, source.target, name)) {
            return Err(SpecError::Duplicate {
                path: at.0,
                name: name.to_string(),
            });
        }
        at.number(
            "max_bytes",
            source.max_bytes,
            |bytes| (1..=MAX_KEY_BYTES).contains(&bytes),
            "between 1 and 8192",
        )?;
    }
    if listed == 0 {
        // Neither hands a session's key: a sessions rule beside the block
        // would have no source of one.
        let alone = match targets {
            0 => "client entries alone",
            _ => "a target entry and no source of a key",
        };
        return Err(at.bad("sources", alone, "a list of one to eight sources"));
    }
    Ok(())
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

/// The first stage that needs to open a decision scope, in stage order.
fn scoped_stage(profile: &Profile) -> Option<&'static str> {
    profile
        .filters
        .iter()
        .find_map(|stage| stage.scope().map(|_| stage.name()))
        .or_else(|| {
            profile
                .scorers
                .iter()
                .find_map(|stage| stage.scorer.scope().map(|_| stage.scorer.name()))
        })
        .or_else(|| profile.picker.scope().map(|_| profile.picker.name()))
}

fn resolve_shadow_profile(
    name: &str,
    at: At,
    cx: &BindingContext<'_>,
    deferred: &mut Deferred,
) -> Result<u16, SpecError> {
    let Some(&position) = cx.profiles.get(name) else {
        return Err(SpecError::UnknownProfileAt {
            path: at.0,
            name: name.to_owned(),
        });
    };
    // Stages may change their state when asked to decide. A live profile
    // must have its own stages.
    if cx.live_profiles.contains(name) {
        return Err(SpecError::BadField {
            path: at.0,
            value: name.to_owned(),
            reason: "a profile unused by live decisions",
        });
    }
    if let Some(&stage) = cx.scoped_stages.get(name) {
        deferred.note(at.0, stage);
    }
    Ok(position)
}

/// The binding's shadow profiles by position: a single's, or the prefill
/// side's of a pair, then the decode side's of a pair. A pair names its
/// shadows by side, and a side the map leaves out is not shadowed. The
/// encode side takes none.
fn check_shadow(
    shadow: &ShadowSpec,
    plan: PlanName,
    cx: &BindingContext<'_>,
    at: &At,
    deferred: &mut Deferred,
) -> Result<[Option<u16>; 2], SpecError> {
    let shadow_at = at.sub("shadow");
    match shadow {
        ShadowSpec::Profile(name) if plan == PlanName::Single => {
            let single = resolve_shadow_profile(name, shadow_at, cx, deferred)?;
            Ok([Some(single), None])
        }
        ShadowSpec::Profile(name) => {
            Err(at.bad("shadow", name, "a map of shadow profiles by side"))
        }
        // A binding of one profile names its shadow by a profile's
        // name, and a map that names no side names no shadow: either
        // loads as a document that shadows nothing, where its writer
        // meant one.
        ShadowSpec::Sides(_) if plan == PlanName::Single => {
            Err(at.bad("shadow", "a map", "one shadow profile's name"))
        }
        ShadowSpec::Sides(sides) if sides.is_empty() => {
            let sides = "a profile on the prefill or the decode side of a pair";
            Err(at.bad("shadow", "{}", sides))
        }
        ShadowSpec::Sides(sides) => {
            let mut by_side = [None; 2];
            for (side, name) in sides {
                let place = match side.as_str() {
                    "prefill" => 0,
                    "decode" => 1,
                    _ => {
                        let sides = "a profile on the prefill or the decode side of a pair";
                        return Err(shadow_at.bad(side, name, sides));
                    }
                };
                let profile = resolve_shadow_profile(name, shadow_at.sub(side), cx, deferred)?;
                by_side[place] = Some(profile);
            }
            Ok(by_side)
        }
    }
}

/// The binding's rules around the decision, in shape; the ones this build
/// does not evaluate are noted as deferred. Returns the shadows' positions
/// ([`check_shadow`]).
fn check_binding_rules(
    b: &BindingSpec,
    cx: &BindingContext<'_>,
    at: &At,
    deferred: &mut Deferred,
) -> Result<[Option<u16>; 2], SpecError> {
    if let Some(overload) = &b.overload {
        check_overload(overload, &at.sub("overload"))?;
    }
    if let Some(block) = &b.routing_key {
        check_routing_key(block, cx, &at.sub("routing_key"))?;
    }
    if let Some(sessions) = &b.sessions {
        let at = at.sub("sessions");
        check_sessions(sessions, b.routing_key.is_some(), cx, &at)?;
        if sessions.issue_session_id {
            deferred.note(at.field("issue_session_id"), "issuing a session id");
        }
    }
    let shadow = match &b.shadow {
        Some(shadow) => check_shadow(shadow, b.plan.unwrap_or_default(), cx, at, deferred)?,
        None => [None; 2],
    };
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
    Ok(shadow)
}

/// The binding's plan and sides, resolved to profile positions.
fn resolve_plan(
    b: &BindingSpec,
    cx: &BindingContext<'_>,
    [shadow, decode_shadow]: [Option<u16>; 2],
) -> Result<Option<Binding>, SpecError> {
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
        shadow,
        decode_shadow,
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
    let shadow = check_binding_rules(b, cx, &at, deferred)?;
    let mut built = resolve_plan(b, cx, shadow)?;
    // A picker that leaves the placement to a sessions rule, and a rule runs
    // under the single plan and on the two sides of a pair: on a side of
    // `epd` nothing would place its requests. The built picker answers, as
    // it answers the engine.
    if b.plan.unwrap_or_default() == PlanName::Epd {
        for (side, name) in &b.profiles {
            if cx.by_assignment.contains(&name.as_str()) {
                deferred.note(
                    at.sub("profiles").field(side),
                    "a picker that places by the sessions rule under the epd plan",
                );
            }
        }
    }
    // The store's file of leases keeps no session verdict of a pair's
    // decode side: that side's lease states the prefill side's
    // (`store::DecodeSide::lease`). A controller that serves from the file
    // would settle a decode side's claim by the prefill side's word, so a
    // rule under the pair plan is not in its build. The catalog's word is
    // how the validator knows such a controller today: it makes some
    // lifecycle calls only. The change that makes the file keep that
    // verdict takes this refusal out.
    if b.sessions.is_some() && b.plan.unwrap_or_default() == PlanName::PdPair && !cx.lifecycle {
        deferred.note(
            at.field("sessions"),
            "the session verdict a pair's decode side keeps of its own",
        );
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
/// time for ever sweeps nothing). The rule runs under the `single` plan, and
/// on the two sides of a `pd-pair` plan where it states no cap and tells
/// its stages: a pair's side reads no count of a key's requests in flight,
/// and a pair's pick has one word for whether its stages hear of it, not
/// one a side.
fn session_rule(
    sessions: &SessionsSpec,
    index: usize,
    plan: Plan,
) -> Result<SessionRule, SpecError> {
    let refused = match plan {
        Plan::Single => None,
        Plan::PdPair if sessions.inflight_cap != u16::MAX => {
            Some(("inflight_cap", "65535 (no cap) under the pd-pair plan"))
        }
        Plan::PdPair if sessions.rule_placed != RulePlaced::Told => {
            Some(("rule_placed", "`told` under the pd-pair plan"))
        }
        Plan::PdPair => None,
        Plan::Epd { .. } => Some(("plan", "`single` or `pd-pair`")),
    };
    if let Some((name, reason)) = refused {
        return Err(SpecError::BadSessions {
            index,
            name,
            reason,
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
        rule_placed: sessions.rule_placed,
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

/// The routing key sources of a preset whose ring places by the key, as
/// the gateway's policy of that name reads them at its defaults: the
/// routing key header, and for the hash ring, behind it, the two headers
/// that tell one client from another, as client sources. `None` for every
/// other preset: `manual` names its one header in its sessions rule, and no
/// other preset's stage reads a key.
fn preset_routing_key(name: &str) -> Option<RoutingKeySpec> {
    let header = |name: &str, client| KeySourceSpec {
        header: Some(name.to_string()),
        client,
        ..KeySourceSpec::default()
    };
    let sources = match name {
        "prefix_hash" => policies::prefix_hash::key_sources(header),
        "consistent_hashing" => policies::consistent_hashing::key_sources(header),
        _ => return None,
    };
    Some(RoutingKeySpec {
        sources,
        strip_lineage: false,
    })
}

impl PolicySpec {
    /// A spec of one preset profile bound to every model and phase.
    pub fn preset(name: &str) -> Result<Self, SpecError> {
        let profile = match name {
            "random" => policies::random::profile(),
            "round_robin" => policies::round_robin::profile(),
            "passthrough" => policies::passthrough::profile(),
            "power_of_two" => policies::power_of_two::profile(),
            "least_load" => policies::least_load::profile(),
            "consistent_hashing" => policies::consistent_hashing::profile(),
            "prefix_hash" => policies::prefix_hash::profile(),
            "manual" => policies::manual::profile(),
            "cache_aware" => policies::cache_aware::profile(),
            "bucket" => policies::bucket::profile(),
            other => return Err(SpecError::UnknownPreset(other.to_string())),
        };
        let manual = name == "manual";
        let sessions = manual.then(policies::manual::sessions);
        let view = ViewSpec {
            headers: if manual {
                policies::manual::view_headers()
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
            signals: BTreeMap::new(),
            profiles,
            plans: PlansSpec::default(),
            bindings: vec![BindingSpec {
                profile: Some(name.to_string()),
                routing_key: preset_routing_key(name),
                sessions,
                ..BindingSpec::default()
            }],
            explain: ExplainSpec::default(),
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
            // One request, one key: the sides' session sources are the one
            // header, and the hash ring's side adds its client sources.
            routing_key: [prefill, decode]
                .into_iter()
                .filter_map(preset_routing_key)
                .max_by_key(|key| key.sources.len()),
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
    /// ([`crate::stages::Lifecycle::needs_lifecycle`]) under the standalone
    /// controller, which makes the membership call alone.
    pub fn validate_in(
        &self,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<Engine, SpecError> {
        let (engine, _, _) = self.built(catalog, denied_headers)?;
        Ok(engine)
    }

    /// [`PolicySpec::validate_in`], and beside the engine how deep its
    /// stages read a request's block hashes and the document's signals,
    /// resolved.
    fn built(
        &self,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<(Engine, HashDepth, Arc<[SignalColumn]>), SpecError> {
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
        check_signals(&self.signals)?;
        let signal_columns = resolve_signals(&self.signals);
        let mut deferred = Deferred::default();
        let mut depth = HashDepth::default();
        // What no stage declares: the sessions rules' counts and the view's
        // headers. The engine folds every profile's stages in.
        let mut needs = Needs::default();
        let mut profiles = Vec::with_capacity(self.profiles.len());
        let has_shadows = self.bindings.iter().any(|binding| binding.shadow.is_some());
        let mut scoped_stages = BTreeMap::new();
        let mut sent_credits = Vec::new();
        // Noted by name: the built vector lines up with the document's
        // positions only once nothing is deferred.
        let mut by_assignment = Vec::new();
        // The built profiles the plain ring alone places by a request's
        // session key, by their place in `profiles`.
        let mut keyed = Vec::new();
        // A registered stage reads a report through the view's reading:
        // the limit the built-in scorers are wrapped with. It resolves a
        // signal's column from the document's, resolved once.
        let from = StageInputs {
            catalog,
            reports: LoadReportReader::new(self.view.max_report_age_ms),
            signals: &signal_columns,
        };
        for (name, profile) in &self.profiles {
            let built = check_profile(
                name,
                profile,
                &self.view,
                from,
                &mut deferred,
                &mut depth,
                &mut sent_credits,
            )?;
            if let Some((built, by_key)) = built {
                if has_shadows {
                    if let Some(stage) = scoped_stage(&built) {
                        scoped_stages.insert(name.as_str(), stage);
                    }
                }
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
        let mut live_profiles = BTreeSet::new();
        if has_shadows {
            live_profiles.extend(self.bindings.iter().flat_map(|binding| {
                binding
                    .profile
                    .iter()
                    .chain(binding.profiles.values())
                    .chain(binding.canary.iter().map(|canary| &canary.profile))
                    .map(String::as_str)
            }));
        }
        // The built bindings whose document sheds a pool the overload veto
        // empties, by their place in `bindings`.
        let mut sheds = Vec::new();
        for (index, binding) in self.bindings.iter().enumerate() {
            let cx = BindingContext {
                index,
                view: &self.view,
                denied: denied_headers,
                profiles: &positions,
                live_profiles: &live_profiles,
                scoped_stages: &scoped_stages,
                by_assignment: &by_assignment,
                lifecycle: catalog.makes_lifecycle_calls(),
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
        // A binding's routing key sources are kept by its place in the
        // document and found by its place in the engine
        // ([`AppliedSpec::key_sources_for`]): the two are one place.
        debug_assert_eq!(
            bindings.len(),
            self.bindings.len(),
            "every binding is built once the deferred check has passed"
        );
        if self.bindings.is_empty() {
            // One profile, refused above when there are more.
            bindings.push(Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: overload_thresholds(None),
            });
        }
        needs.headers |= !self.view.headers.is_empty();
        let mut engine = Engine::new(profiles, bindings, self.generation, needs);
        // In the order of the profiles' names, as the map walks them: a
        // later engine finds a count by the name of the profile that made
        // it.
        debug_assert!(sent_credits
            .windows(2)
            .all(|pair| pair[0].profile < pair[1].profile));
        engine.sent_credits = sent_credits;
        engine.set_explain_sample(self.explain.sample);
        engine.limit_report_age(self.view.max_report_age_ms);
        for at in sheds {
            engine.shed_when_all_vetoed(at);
        }
        for at in keyed {
            engine.places_by_key(at);
        }
        Ok((engine, depth, signal_columns))
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
            signals: BTreeMap::new(),
            profiles,
            plans: PlansSpec::default(),
            bindings: Vec::new(),
            explain: ExplainSpec::default(),
        }
    }

    /// The index settings for a model: the most specific matching key, if
    /// any (an exact id, then the longer `prefix*` or `*suffix`, then `*`).
    #[must_use]
    pub fn index_for(&self, model: &str) -> Option<(&str, &IndexSpec)> {
        index_matching(&self.index, model)
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
            signals,
            profiles,
            plans,
            bindings,
            explain,
        } = self;
        Lineless {
            api_version,
            generation: 0,
            view,
            index,
            signals,
            profiles,
            plans,
            bindings,
            explain,
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
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    signals: &'a BTreeMap<String, SignalSpec>,
    profiles: &'a BTreeMap<String, ProfileSpec>,
    #[serde(skip_serializing_if = "is_default::<PlansSpec>")]
    plans: &'a PlansSpec,
    bindings: &'a [BindingSpec],
    #[serde(skip_serializing_if = "is_default::<ExplainSpec>")]
    explain: &'a ExplainSpec,
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

fn index_matching<'a>(
    index: &'a BTreeMap<String, IndexSpec>,
    model: &str,
) -> Option<(&'a str, &'a IndexSpec)> {
    index
        .iter()
        .filter(|(pattern, _)| glob_matches(pattern, model))
        .max_by_key(|(pattern, _)| specificity(pattern))
        .map(|(pattern, spec)| (pattern.as_str(), spec))
}

/// One generation as a store applies it: the engine validated from a
/// document, with the generation's number, its needs and its view's cap on
/// a request's block hashes.
///
/// One invariant: **a decision reads its gateway's vetoes, its engine and
/// its number in one load.** A store publishes every world under one of these
/// ([`StoreSnapshot::spec`](crate::store::StoreSnapshot::spec)), so no reader
/// pairs one generation's engine with rows published under another. The number
/// is the engine's own, the document's `generation` line: there is no second
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
/// document's pairing mode ([`AppliedSpec::pairing`]). It says when a host
/// asks a pair's decode ([`AppliedSpec::decode_pick`]). And it says how deep
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
    /// The index settings a model takes at a membership write or apply.
    index: BTreeMap<String, IndexSpec>,
    pairing: Pairing,
    /// The document's `plans.pd-pair.decode`.
    decode_pick: DecodePick,
    lists: ViewLists,
    /// The document's `view.headers`, in their order
    /// ([`AppliedSpec::view_headers`]).
    view_headers: Box<[Box<str>]>,
    /// Each binding's routing key sources, by the binding's place in the
    /// document; none where a binding names none
    /// ([`AppliedSpec::key_sources_for`]).
    key_sources: Box<[Option<KeySources>]>,
    /// Whether a binding names a target source
    /// ([`AppliedSpec::names_target`]).
    names_target: bool,
    /// The document's signals, resolved, in column order
    /// ([`AppliedSpec::signal_columns`]).
    signal_columns: Arc<[SignalColumn]>,
    /// A start's document waits for the first lead of this store's
    /// replica ([`AppliedSpec::mark_document_awaited`]): set on the unbound start
    /// alone, and gone with it at the first publication.
    #[cfg(feature = "controller")]
    awaited: AtomicBool,
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
        Self::validate_in(document, &STOCK, denied_headers)
    }

    /// [`AppliedSpec::validate_with`] over the stages `catalog` holds,
    /// including stages the host registers.
    ///
    /// The caller assigns the document's generation number.
    /// [`MemoryStore::apply`](crate::store::MemoryStore::apply) accepts a
    /// generation whose number is above the standing one.
    pub fn validate_in(
        document: &PolicySpec,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<Self, SpecError> {
        let built = document.built(catalog, denied_headers)?;
        Ok(Self::of(document, document.digest()?, built))
    }

    /// [`AppliedSpec::validate`] over the stages `catalog` holds, under a
    /// digest its caller took already: the operator door reads the
    /// document's digest against the standing one and hands it on, so an
    /// apply digests its document once.
    #[cfg(any(test, feature = "operator"))]
    pub(crate) fn validate_digested(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<Self, SpecError> {
        let built = document.built(catalog, denied_headers)?;
        Ok(Self::of(document, digest, built))
    }

    /// The generation of `document` under its `digest`: the engine it
    /// validated into, with the depth of that engine's stages.
    fn of(
        document: &PolicySpec,
        digest: Digest,
        (engine, depth, signal_columns): (Engine, HashDepth, Arc<[SignalColumn]>),
    ) -> Self {
        let sources = |binding| KeySources::of_binding(binding, &document.view);
        let key_sources: Box<[_]> = document.bindings.iter().map(sources).collect();
        Self {
            engine,
            digest: Some(digest),
            depth,
            signal_columns,
            max_block_hashes: document.view.max_block_hashes,
            block_size: Gates::of(None, None, document.view.block_size).block_size,
            index: document.index.clone(),
            pairing: document.plans.pd_pair.pairing,
            decode_pick: document.plans.pd_pair.decode,
            lists: ViewLists::of(&document.view),
            view_headers: document
                .view
                .headers
                .iter()
                .map(|name| name.as_str().into())
                .collect(),
            names_target: key_sources.iter().flatten().any(KeySources::names_target),
            key_sources,
            #[cfg(feature = "controller")]
            awaited: AtomicBool::new(false),
        }
    }

    /// The generation a store holds before any document is applied: number
    /// 0, an engine with no profile and no binding, which declines every
    /// decision by name as an engine declines a model no binding covers,
    /// and the pair plan's rules a document that names none has.
    #[must_use]
    pub fn unbound() -> Self {
        Self {
            engine: Engine::new(Vec::new(), Vec::new(), 0, Needs::default()),
            digest: None,
            depth: HashDepth::Unread,
            signal_columns: Arc::default(),
            max_block_hashes: ViewSpec::default().max_block_hashes,
            block_size: Gates::default().block_size,
            index: BTreeMap::new(),
            pairing: Pairing::default(),
            decode_pick: DecodePick::default(),
            lists: ViewLists::default(),
            view_headers: Box::default(),
            key_sources: Box::default(),
            names_target: false,
            #[cfg(feature = "controller")]
            awaited: AtomicBool::new(false),
        }
    }

    /// The document's signals, resolved, in column order: a signal's place
    /// here is its column's number. What a stage's build answered from
    /// ([`StageBuild::signal_column`](crate::stages::StageBuild::signal_column)),
    /// and the one reading of the names for a store that keeps the cells.
    /// Empty for a document that declares none and for the unbound start.
    pub fn signal_columns(&self) -> &[SignalColumn] {
        &self.signal_columns
    }

    /// [`Self::signal_columns`] as the list is shared: what the store's
    /// signal sink hands a report's conversion.
    pub(crate) fn shared_signal_columns(&self) -> Arc<[SignalColumn]> {
        Arc::clone(&self.signal_columns)
    }

    /// Marks the unbound start of a replica whose start's document waits
    /// for its first lead. Under the mark a request is no decision, and
    /// its answer is `Busy` with no reason: the unbound start of a store
    /// with no document sheds it by the binding, which would say here that
    /// a document refuses what no document has read. The mark is no
    /// generation: it takes no number and is told to no standby, and it
    /// goes with the unbound start at the first publication.
    #[cfg(feature = "controller")]
    pub(crate) fn mark_document_awaited(&self) {
        self.awaited.store(true, atomic::Ordering::Relaxed);
    }

    /// Whether a start's document waits for this generation's place
    /// ([`AppliedSpec::mark_document_awaited`]).
    #[cfg(feature = "controller")]
    pub(crate) fn awaits_document(&self) -> bool {
        self.awaited.load(atomic::Ordering::Relaxed)
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

    /// The document's engine reads and writes `table` from here on
    /// ([`Engine::share_sessions`]): the session table of an engine object
    /// that outlives the document.
    #[cfg(feature = "in-process")]
    pub(crate) fn share_sessions(&mut self, table: Arc<SessionTable>) {
        self.engine.share_sessions(table);
    }

    /// The document's block size, in tokens: the size a request's blocks
    /// are counted by where neither the request nor a record says one.
    #[must_use]
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// The most specific matching index settings, or the default.
    pub(crate) fn index_for(&self, model: &str) -> IndexSpec {
        index_matching(&self.index, model)
            .map(|(_, spec)| spec.clone())
            .unwrap_or_default()
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
    /// past, and a generation with no such stage reads none. A kind no
    /// stage of the generation declares is read by none
    /// ([`Needs::token_hashes`], [`Needs::text_hashes`]): `0`, whatever the
    /// other kind's depth. A host hashes that many blocks of a request and
    /// no more, and hands the engine the first that many of a longer list.
    #[must_use]
    pub fn block_hashes_read(&self, kind: HashKind, block_size: u32) -> usize {
        let needs = self.needs();
        let declared = match kind {
            HashKind::Token => needs.token_hashes,
            HashKind::Text => needs.text_hashes,
        };
        if !declared {
            return 0;
        }
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

    /// The sources a request of `phase` for `model` takes its routing key
    /// from: those of the first binding that matches it, a pair request's
    /// by its prefill lookup. The host reads the request by them
    /// ([`KeySources::derive`]) and hands the engine the one key
    /// ([`RequestFacts::session_key`](crate::engine::RequestFacts::session_key)).
    /// `None` where no binding matches or the binding names no source: the
    /// request carries no key.
    #[must_use]
    pub fn key_sources_for(&self, model: &str, phase: Phase) -> Option<&KeySources> {
        let at = self.engine.binding_position(model, phase)?;
        self.key_sources.get(usize::try_from(at).ok()?)?.as_ref()
    }

    /// Whether a binding of this generation takes a routing key from the
    /// request's own id ([`KeySources::reads_rid`]). A host that learns
    /// which binding a request meets from its body alone, where the model
    /// stands, keeps every request's body readable while this generation
    /// stands. A binding no request reaches counts too.
    #[must_use]
    pub fn reads_request_id(&self) -> bool {
        let mut sources = self.key_sources.iter().flatten();
        sources.any(KeySources::reads_rid)
    }

    /// Whether a binding of this generation names a target source
    /// ([`KeySourceSpec::target`]), decided when the document was applied.
    /// A request under a document that names none pays this one test, and
    /// no header of it is read for a target.
    #[must_use]
    pub fn names_target(&self) -> bool {
        self.names_target
    }

    /// The mode a store derives its pair classes under while this
    /// generation stands: the document's `plans.pd-pair.pairing`.
    #[must_use]
    pub fn pairing(&self) -> Pairing {
        self.pairing
    }

    /// When a pair's decode side is picked while this generation stands:
    /// the document's `plans.pd-pair.decode`. A host whose route sends a
    /// pair's decode behind its prefill states
    /// [`Constraints::decode_later`](crate::engine::view::Constraints::decode_later)
    /// under [`DecodePick::AfterPrefill`], and asks the decode again when
    /// the prefill has answered.
    #[must_use]
    pub fn decode_pick(&self) -> DecodePick {
        self.decode_pick
    }

    /// The lists this generation declares, as a store reads them when it
    /// writes a worker's row.
    #[must_use]
    pub fn lists(&self) -> &ViewLists {
        &self.lists
    }

    /// The request headers a stage may read (`view.headers`), in the
    /// document's order. A host states each one a request carries as a
    /// [`Header`](crate::engine::view::Header), which names its header by
    /// its place here.
    pub fn view_headers(&self) -> impl Iterator<Item = &str> {
        self.view_headers.iter().map(|name| &**name)
    }
}

/// The lists a generation declares, as a store reads them when it writes a
/// worker's row ([`Candidate`](crate::engine::view::Candidate)): the row names its
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
pub struct ViewListPositions {
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
    pub fn positions(
        &self,
        engine: &str,
        zone: Option<&str>,
        labels: &BTreeMap<String, String>,
    ) -> ViewListPositions {
        let engine_at = self.engines.get(engine).copied();
        let zone = zone.filter(|zone| !zone.is_empty());
        let zone_at = zone.and_then(|zone| self.zones.get(zone).copied());
        ViewListPositions {
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
    use crate::{
        engine::{Candidate, Constraints, Filter, PoolFacts, RequestFacts, Targets, TieBand},
        stages::Lifecycle,
    };

    #[test]
    fn a_parameter_key_written_twice_is_refused_inside_maps_and_lists() {
        let score_yaml = |score: &str| {
            format!("apiVersion: {API_VERSION}\nprofiles:\n  default:\n    score: {score}\n    pick: best\n")
        };
        for score in [
            "[{ queue-price: { costs: { unit: 17, unit: 31 } } }]",
            "[{ queue-price: { costs: [[{ unit: 17, unit: 31 }]] } }]",
        ] {
            let error =
                serde_yaml::from_str::<PolicySpec>(&score_yaml(score)).expect_err("duplicate key");
            assert!(
                error.to_string().contains("duplicate field `unit`"),
                "{error}"
            );
        }
        let json = format!(
            r#"{{"apiVersion":"{API_VERSION}","profiles":{{"default":{{"score":[{{"queue-price":{{"costs":[{{"unit":17,"unit":31}}]}}}}],"pick":"best"}}}}}}"#
        );
        let error = serde_json::from_str::<PolicySpec>(&json).expect_err("duplicate key");
        assert!(
            error.to_string().contains("duplicate field `unit`"),
            "{error}"
        );

        let read: PolicySpec = serde_yaml::from_str(&score_yaml(
            "[{ queue-price: { costs: [[{ unit: 17 }]], weight: 3.5 } }]",
        ))
        .expect("a well-formed document");
        let value = serde_json::to_value(&read).expect("serializes");
        assert_eq!(
            value["profiles"]["default"]["score"],
            serde_json::json!([{ "queue-price": { "costs": [[{ "unit": 17 }]], "weight": 3.5 } }])
        );
        let again: PolicySpec = serde_json::from_value(value).expect("parses again");
        assert_eq!(
            read.digest().expect("digests"),
            again.digest().expect("digests")
        );
    }

    /// The order a file writes a registered scorer's parameter keys in is
    /// in no digest, at any depth: two files that write one map in two
    /// orders are one document. A value that differs is another.
    #[test]
    fn the_order_of_a_scorers_parameter_keys_is_in_no_digest() {
        let document = |score: &str| -> PolicySpec {
            serde_yaml::from_str(&format!(
                "apiVersion: {API_VERSION}\nprofiles:\n  default:\n    score: {score}\n    pick: best\n"
            ))
            .expect("a well-formed document")
        };
        let one = document(
            "[{ queue-price: { weight: 2, per_request: 500, costs: [{ unit: 17, flat: 3 }] } }]",
        );
        let other = document(
            "[{ queue-price: { costs: [{ flat: 3, unit: 17 }], per_request: 500, weight: 2 } }]",
        );
        assert_eq!(
            one.digest().expect("digests"),
            other.digest().expect("digests"),
            "two orders of one map"
        );
        assert_eq!(
            serde_json::to_string(&one).expect("serializes"),
            serde_json::to_string(&other).expect("serializes")
        );
        let moved = document(
            "[{ queue-price: { weight: 2, per_request: 501, costs: [{ unit: 17, flat: 3 }] } }]",
        );
        assert_ne!(
            one.digest().expect("digests"),
            moved.digest().expect("digests")
        );
    }

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
        // The source each prefix stage a document states with parameters
        // reads, as the presets state theirs.
        let read_from = |document: &mut PolicySpec, source: AffinitySource| {
            let stages = document.profiles.values_mut();
            for stage in stages.flat_map(|profile| &mut profile.score) {
                if let ScoreSpec::Params(ScoreStage::PrefixAffinity(params)) = stage {
                    params.source = Some(source);
                }
            }
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
            let applied =
                AppliedSpec::validate_digested(&keyed, digest, &StageRegistry::builtin(), &[])
                    .expect("the preset validates");
            assert_eq!(applied.block_hashes_read(Token, 0), 8);
        }
        // A stage with no depth of its own reads to the view's cap.
        let mut held = preset("cache_aware");
        assert_eq!(applied(&held).block_hashes_read(Token, 0), 256);
        held.view.max_block_hashes = 128;
        // A kind no stage declares is read by none. The prefix stage
        // declares text hashes where its source reads the record of sent
        // prefixes, and none where it reads cache events alone.
        read_from(&mut held, AffinitySource::Index);
        assert_eq!(applied(&held).block_hashes_read(Token, 0), 128);
        assert_eq!(applied(&held).block_hashes_read(Text, 4), 0);
        read_from(&mut held, AffinitySource::Merged);
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
        let mut catalog = StageRegistry::builtin();
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
            let (_, depth, _) = document
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

    #[test]
    fn shadow_absent_keeps_every_preset_digest() {
        let expected = [
            (
                "random",
                "sha256:12f28a338634ac96fb8d5883f9e354341329ffb8cf23417c4b6ace700d03d3ec",
            ),
            (
                "round_robin",
                "sha256:62e5652e4940b185ed50b247c68b27c4a3f7ac43683f66b97404d449669720fc",
            ),
            (
                "passthrough",
                "sha256:3defa376775031aa71f5d87b1dd2c7dbdaf7eaf42f7f0f0616ac345cdddd6f1d",
            ),
            (
                "power_of_two",
                "sha256:4c4b153064585f5e23a4b62e1fd4cf2d6b07c0dea6340c24834260d2e4281f66",
            ),
            (
                "least_load",
                "sha256:80d7fcef3d2df7ee25aa79c1ccfa784cad0dbc91785194d48e108e660e8cda88",
            ),
            (
                "consistent_hashing",
                "sha256:c9bb5fdc96f68381ac99700d79a793499cdf915ab2ce590ffb82942b398e6745",
            ),
            (
                "prefix_hash",
                "sha256:f5bc21e35bb782d5ee2a559ece6b18c0905cebcd20c354b294ecf1c1e1c5d871",
            ),
            (
                "manual",
                "sha256:d12dd3db8aa8ece3fbb62bd354c3db14d97d92e57079dc191ac3546ccd04593e",
            ),
            (
                "cache_aware",
                "sha256:864958791685f8894b437e698f0bcee6ad48a81b56d248b230c93dd75b9b0513",
            ),
            (
                "bucket",
                "sha256:168a1e9a62410e6bba35226e96c40715392d7fe5e73b88cf242e2c48a0686f3c",
            ),
        ];
        assert_eq!(expected.map(|(name, _)| name), PRESETS);
        for (name, digest) in expected {
            let spec = PolicySpec::preset(name).expect("a preset");
            assert_eq!(
                spec.digest().expect("a digest").to_string(),
                digest,
                "{name}"
            );
        }
    }

    #[test]
    fn shadow_round_trips_as_a_profile_name() {
        let spec = spec_with_binding(r#"{"profile":"p","shadow":"q"}"#);
        let json = serde_json::to_value(&spec).expect("writes JSON");
        assert_eq!(json["bindings"][0]["shadow"], "q");
        assert_eq!(
            serde_json::from_value::<PolicySpec>(json).expect("reads JSON"),
            spec
        );
        let yaml = serde_yaml::to_string(&spec).expect("writes YAML");
        assert_eq!(
            serde_yaml::from_str::<PolicySpec>(&yaml).expect("reads YAML"),
            spec
        );
        let engine = spec.validate().expect("a separate shadow profile is valid");
        let binding = engine.binding("m", Phase::Single).expect("the binding");
        assert_eq!(binding.shadow, Some(1));
        assert_eq!(binding.decode_shadow, None);
        assert_eq!(engine.profiles()[1].name, "q");
    }

    #[test]
    fn shadow_refuses_an_unknown_profile_at_the_field() {
        let spec = spec_with_binding(r#"{"profile":"p","shadow":"absent"}"#);
        assert_eq!(
            spec.validate().expect_err("the profile is absent"),
            SpecError::UnknownProfileAt {
                path: "bindings[0].shadow".into(),
                name: "absent".into(),
            }
        );
    }

    #[test]
    fn shadow_refuses_a_profile_named_live_anywhere() {
        for (bindings, name) in [
            (r#"{"profile":"p","shadow":"p"}"#, "p"),
            (r#"{"profiles":{"single":"p"},"shadow":"p"}"#, "p"),
            (r#"{"profile":"p","shadow":"q"},{"profile":"q"}"#, "q"),
            (
                r#"{"profile":"p","shadow":"q"},{"plan":"pd-pair","profiles":{"prefill":"p","decode":"q"}}"#,
                "q",
            ),
            (
                r#"{"profile":"p","shadow":"q"},{"profile":"p","canary":{"profile":"q","percent":19}}"#,
                "q",
            ),
        ] {
            let error = spec_with_binding(bindings)
                .validate()
                .expect_err("the shadow is also named live");
            assert!(
                matches!(
                    error,
                    SpecError::BadField { ref path, ref value, .. }
                        if path == "bindings[0].shadow" && value == name
                ),
                "{error} for {bindings}"
            );
        }
    }

    #[test]
    fn shadow_on_a_pair_refuses_a_single_name_at_the_field() {
        let spec = spec_with_binding(
            r#"{"plan":"pd-pair","profiles":{"prefill":"p","decode":"p"},"shadow":"q"}"#,
        );
        let error = spec.validate().expect_err("a pair names its sides");
        assert!(
            matches!(
                error,
                SpecError::BadField { ref path, .. } if path == "bindings[0].shadow"
            ),
            "{error}"
        );
    }

    /// A pair names its shadows by side: the map's keys are `prefill` and
    /// `decode`, and a side it leaves out is not shadowed. Each value is a
    /// profile no binding decides by live. Any other key, and a map on a
    /// binding of one profile, are refused by path.
    #[test]
    fn a_pairs_shadow_map_names_a_profile_a_side() {
        let pair = |shadow: &str| {
            spec_with_binding(&format!(
                r#"{{"plan":"pd-pair","profiles":{{"prefill":"p","decode":"p"}},"shadow":{shadow}}}"#,
            ))
        };
        let sides = |shadow: &str| {
            let engine = pair(shadow).validate().expect("a pair's shadow map");
            let binding = engine.binding("m", Phase::Prefill);
            let binding = binding.expect("the pair's binding");
            (binding.shadow, binding.decode_shadow)
        };
        // `q` is the document's second profile.
        assert_eq!(sides(r#"{"decode":"q"}"#), (None, Some(1)));
        assert_eq!(sides(r#"{"prefill":"q"}"#), (Some(1), None));
        assert_eq!(sides(r#"{"prefill":"q","decode":"q"}"#), (Some(1), Some(1)));
        assert_eq!(
            bad_field(&pair(r#"{"encode":"q"}"#)),
            "bindings[0].shadow.encode"
        );
        // A binding of one profile takes no map, and no plan an empty
        // one: either would load as a document that shadows nothing.
        let single = spec_with_binding(r#"{"profile":"p","shadow":{"decode":"q"}}"#);
        assert_eq!(bad_field(&single), "bindings[0].shadow");
        let empty = spec_with_binding(r#"{"profile":"p","shadow":{}}"#);
        assert_eq!(bad_field(&empty), "bindings[0].shadow");
        assert_eq!(bad_field(&pair("{}")), "bindings[0].shadow");
        assert_eq!(
            pair(r#"{"decode":"absent"}"#)
                .validate()
                .expect_err("unknown profile"),
            SpecError::UnknownProfileAt {
                path: "bindings[0].shadow.decode".into(),
                name: "absent".into(),
            }
        );
        assert_eq!(
            bad_field(&pair(r#"{"decode":"p"}"#)),
            "bindings[0].shadow.decode"
        );
        let doubled = r#"{"profile":"p","shadow":{"single":"q","single":"q"}}"#;
        assert!(serde_json::from_str::<BindingSpec>(doubled).is_err());
    }

    /// A bad `shadow` value is refused by what is wrong with it: a side
    /// written twice by its key, and a value of neither form by the forms
    /// the field takes.
    #[test]
    fn shadow_refuses_a_bad_value_by_its_key_or_its_form() {
        let doubled = r#"{"profile":"p","shadow":{"single":"q","single":"q"}}"#;
        let refusal = serde_json::from_str::<BindingSpec>(doubled)
            .expect_err("a side written twice")
            .to_string();
        assert!(
            refusal.contains("the key `single` is written twice"),
            "{refusal}"
        );
        let number = r#"{"profile":"p","shadow":5}"#;
        let refusal = serde_json::from_str::<BindingSpec>(number)
            .expect_err("a number in the field")
            .to_string();
        assert!(
            refusal.contains("expected a profile name, or a map of sides to profile names"),
            "{refusal}"
        );
    }

    #[test]
    fn shadow_refuses_each_stage_that_opens_a_plugin_scope_by_name() {
        #[derive(Debug)]
        struct Scoped;

        impl crate::engine::DecisionScope for Scoped {
            fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}
            fn close(&self) {}
        }

        impl Lifecycle for Scoped {}

        impl Filter for Scoped {
            fn name(&self) -> &'static str {
                "scoped-filter"
            }
            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }
            fn scope(&self) -> Option<&dyn crate::engine::DecisionScope> {
                Some(self)
            }
        }

        impl Scorer for Scoped {
            fn name(&self) -> &'static str {
                "scoped-scorer"
            }
            fn score(
                &self,
                _: &RequestFacts<'_>,
                _: &[Candidate],
                _: &[u32],
                _: &mut [f32],
            ) -> TieBand {
                TieBand::NORMALIZED
            }
            fn scope(&self) -> Option<&dyn crate::engine::DecisionScope> {
                Some(self)
            }
        }

        impl Picker for Scoped {
            fn name(&self) -> &'static str {
                "scoped-picker"
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
            fn scope(&self) -> Option<&dyn crate::engine::DecisionScope> {
                Some(self)
            }
        }

        let mut catalog = StageRegistry::builtin();
        catalog
            .filter("scoped-filter", || Box::new(Scoped))
            .and_then(|catalog| catalog.scorer("scoped-scorer", || Box::new(Scoped)))
            .and_then(|catalog| catalog.picker("scoped-picker", || Box::new(Scoped)))
            .expect("the stage names are available");
        for (profile, name) in [
            (
                r#"{"filter":["scoped-filter"],"pick":"best"}"#,
                "scoped-filter",
            ),
            (
                r#"{"score":["scoped-scorer"],"pick":"best"}"#,
                "scoped-scorer",
            ),
            (r#"{"pick":"scoped-picker"}"#, "scoped-picker"),
        ] {
            let mut spec = spec_with_binding(r#"{"profile":"p","shadow":"q"}"#);
            spec.profiles.insert(
                "q".into(),
                serde_json::from_str(profile).expect("a profile"),
            );
            assert_eq!(
                spec.validate_in(&catalog, &[])
                    .expect_err("the stage needs a scope"),
                SpecError::NotInBuild {
                    path: "bindings[0].shadow".into(),
                    what: name,
                }
            );
        }
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
    fn explain_sample_is_an_unsigned_count_that_defaults_to_off() {
        let base = PolicySpec::preset("random").expect("a preset");
        let serde_json::Value::Object(mut json) = serde_json::to_value(&base).expect("a document")
        else {
            panic!("a document is an object");
        };
        assert!(!json.contains_key("explain"));
        for sample in [0, 1, 3, u32::MAX] {
            json.insert("explain".into(), serde_json::json!({ "sample": sample }));
            let read = serde_json::from_value::<PolicySpec>(json.clone().into())
                .expect("an unsigned count");
            assert_eq!(read.explain.sample, sample);
            assert_eq!(
                read.validate().expect("a valid document").explain_sample(),
                sample
            );
            assert_eq!(read.digest().ok() == base.digest().ok(), sample == 0);
        }
        for invalid in [
            serde_json::json!({ "sample": -1 }),
            serde_json::json!({ "sample": 1.5 }),
            serde_json::json!({ "sample": "1" }),
            serde_json::json!({ "sample": u64::from(u32::MAX) + 1 }),
            serde_json::json!({ "slots": 1_024 }),
        ] {
            json.insert("explain".into(), invalid);
            assert!(serde_json::from_value::<PolicySpec>(json.clone().into()).is_err());
        }
        json.insert("explain".into(), serde_json::json!({}));
        let empty = serde_json::from_value::<PolicySpec>(json.into()).expect("an empty section");
        assert_eq!(empty, base);
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
            assert_eq!(engine.needs().text_hashes, hashes, "{name}");
            assert!(!engine.needs().text, "{name}");
            // The request's size is counted where a stage places by it or
            // takes a share of it: a host that sends a request on unread
            // reads it for that stage.
            let sized = ["bucket", "cache_aware"].contains(&name);
            assert_eq!(engine.needs().prompt_size, sized, "{name}");
            // The affinity stage's source is left out: it reads each worker
            // by what it has, its cache events or the prefixes sent to it.
            assert_eq!(engine.needs().events, name == "cache_aware", "{name}");
            assert_eq!(
                engine.needs().sent_prefixes,
                name == "cache_aware",
                "{name}"
            );
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

        // An encode-phase binding before an `epd` binding decides the
        // encode items that come alone. The encode side of an encode
        // request decided in one call is decided by the binding its
        // prefill lookup resolves to, with that pair's encode profile. The
        // encode needs carry both, for the model and for the phase.
        let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "encode-only": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" },
            "side": { "filter": ["healthy"], "pick": "round-robin" },
            "pair-encode": { "filter": ["healthy"], "score": ["session-affinity"], "pick": "best" },
            "default": { "filter": ["healthy"], "pick": "round-robin" }
          },
          "bindings": [
            { "match": { "model": "*", "phase": ["encode"] }, "profile": "encode-only" },
            { "match": { "model": "epd/*" }, "plan": "epd", "profiles": { "encode": "pair-encode", "prefill": "side", "decode": "side" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
        let engine = doc(text).validate().expect("validates");
        assert_eq!(engine.needs_for("epd/x", Phase::Encode), loads.or(headers));
        assert_eq!(engine.needs_for("other", Phase::Encode), loads);
        assert_eq!(engine.needs_for("epd/x", Phase::Prefill), Needs::default());
        assert_eq!(engine.needs_for_phase(Phase::Encode), loads.or(headers));
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
            .contains(&"has-capacity".into()));
        assert!(spec.profiles["decode"].filter.contains(&"has-room".into()));
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
        use crate::engine::{
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
        let first = |decision: &crate::engine::Decision| {
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
            .contains(&"has-capacity".into()));
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

    /// The scores combine by each weight's share of the weights' sum. A
    /// profile whose sum is not finite, or has no finite reciprocal, is
    /// refused at its score list: its shares would lose the order of every
    /// row. A weight under the normal range beside one that is not passes.
    #[test]
    fn weights_whose_sum_gives_no_finite_share_are_refused_at_the_score_list() {
        for score in [
            r#"{"least-load":{"weight":1e-40}}"#,
            r#"{"least-load":{"weight":3e38}},{"least-requests":{"weight":3e38}}"#,
        ] {
            assert_eq!(
                bad_field(&spec_with_score(score)),
                "profiles.p.score",
                "{score}"
            );
        }
        spec_with_score(r#"{"least-load":{"weight":1e-40}},"least-requests""#)
            .validate()
            .expect("a sum in the normal range");
    }

    /// A profile with a stage this build does not carry is judged by the
    /// weights the build does carry only where a part decides for the
    /// whole. A part with no finite reciprocal defers with the profile:
    /// the stage left out may bring the sum into range. An infinite part
    /// is refused: no weight left out can bring it back.
    #[test]
    fn a_deferred_profile_is_refused_for_an_infinite_part_of_its_weights_alone() {
        let deferred = r#"{"prefix-affinity":{"decay":0.9}}"#;
        let small = format!(r#"{{"least-load":{{"weight":1e-40}}}},{deferred}"#);
        let (path, what) = not_in_build(&spec_with_score(&small));
        assert_eq!(path, "profiles.p.score[1].prefix-affinity");
        assert!(what.contains("prefix-affinity"), "{what}");
        let large = format!(
            r#"{{"least-load":{{"weight":3e38}}}},{{"least-requests":{{"weight":3e38}}}},{deferred}"#
        );
        assert_eq!(bad_field(&spec_with_score(&large)), "profiles.p.score");
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
        let gold = lists.positions("beta", Some("west"), &held);
        assert_eq!((gold.engine_kind, gold.zone, gold.labels), (2, 2, 0b111));
        assert!(!gold.engine_outside && !gold.zone_outside);
        let held = labels(&[("tier", "silver"), ("note", "a")]);
        let silver = lists.positions("alpha", Some("east"), &held);
        assert_eq!(
            (silver.engine_kind, silver.zone, silver.labels),
            (1, 1, 0b001)
        );
        // A bit is the selector's place in the list, whatever the keys'
        // order: two selectors of one key stand apart here.
        let mixed = spec_with_view(r#"{"labels":["z","b=1","m","b","a=2","q=x"]}"#);
        let mixed = AppliedSpec::validate(&mixed).expect("the document validates");
        let held = labels(&[("a", "2"), ("b", "1"), ("c", "9"), ("m", ""), ("q", "y")]);
        assert_eq!(mixed.lists().positions("", None, &held).labels, 0b01_1110);
        // A name the list does not hold reads 0 and says so. No name reads
        // 0 and is not outside.
        let stray = lists.positions("gamma", Some("north"), &labels(&[]));
        assert_eq!((stray.engine_kind, stray.zone, stray.labels), (0, 0, 0));
        assert!(stray.engine_outside && stray.zone_outside);
        for zone in [None, Some("")] {
            assert_eq!(
                lists.positions("", zone, &labels(&[])),
                ViewListPositions::default()
            );
        }
        // No list: every worker reads 0 and none is outside.
        let none = AppliedSpec::validate(&spec_with_view("{}")).expect("the document validates");
        assert!(none.lists().is_empty() && AppliedSpec::unbound().lists().is_empty());
        let read = none.lists().positions("beta", Some("west"), &held);
        assert_eq!(read, ViewListPositions::default());
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

    /// A document that names no source reads each worker by what it has:
    /// its stage asks a host for cache events and for the prefixes sent.
    #[test]
    fn a_source_left_out_reads_cache_events_and_sent_prefixes() {
        let needs = |score: &str| spec_with_score(score).validate().expect(score).needs();
        let left_out = needs(r#"{"prefix-affinity":{}}"#);
        assert_eq!(
            left_out,
            needs(r#"{"prefix-affinity":{"source":"merged"}}"#)
        );
        assert!(left_out.events && left_out.sent_prefixes);
        let index = needs(r#"{"prefix-affinity":{"source":"index"}}"#);
        assert!(index.events && !index.sent_prefixes);
    }

    #[test]
    fn least_cost_reads_the_compared_rows_whole_behind_a_rank() {
        let spec = spec_with_score(
            r#"{"least-cost":{"terms":{"load.running":7},"unreported":"missing-as-zero"}}"#,
        );
        let catalog = StageRegistry::builtin();
        let entry = &spec.profiles["p"].score[0];
        let from = StageInputs {
            catalog: &catalog,
            reports: LoadReportReader::new(None),
            signals: &[],
        };
        let Scored::Typed(stage) = scored(
            entry,
            "p",
            &At::new("profiles.p.score[0]"),
            from,
            SentSettings::default(),
            &mut SentSlot::default(),
        )
        .expect("a built-in scorer") else {
            panic!("a typed scorer");
        };
        let Resolved::Built(scorer) = scorer_of(
            &stage,
            Gates::of(None, None, None),
            &spec.view,
            &[],
            || panic!("no term reads sends"),
            &At::new("profiles.p.score[0].least-cost"),
        )
        .expect("valid parameters") else {
            panic!("least-cost is built");
        };
        assert_eq!(scorer.name(), "least-cost");
        assert!(scorer.row_wise());
        assert!(!scorer.front_wise());
    }

    /// Only a `sent.` term makes the profile's count of sends, and a time
    /// limit on the sends needs one where `least-cost` is the only scorer.
    #[test]
    fn least_cost_keeps_a_count_of_sends_only_for_a_sent_term() {
        for (term, limit, want) in [
            ("candidate.in_flight", None, Ok(0)),
            ("sent.tokens", None, Ok(1)),
            ("sent.requests", Some(43), Ok(1)),
            (
                "candidate.in_flight",
                Some(43),
                Err("profiles.p.sent_ttl_ms"),
            ),
        ] {
            let mut spec =
                spec_with_score(&format!(r#"{{"least-cost":{{"terms":{{"{term}":7}}}}}}"#));
            spec.profiles.get_mut("p").expect("the profile").sent_ttl_ms = limit;
            let counts = match spec.validate() {
                Ok(engine) => Ok(engine.sent_credits.len()),
                Err(SpecError::BadField { path, .. }) => Err(path),
                Err(other) => panic!("{term} under {limit:?}: {other:?}"),
            };
            assert_eq!(
                counts,
                want.map_err(str::to_owned),
                "{term} under {limit:?}"
            );
        }
    }

    /// Profiles under one time limit and one report delay read one count
    /// of sends. A profile under another limit or another delay keeps a
    /// count of its own.
    #[test]
    fn profiles_under_one_sent_setting_read_one_count_of_sends() {
        for (sent, want) in [
            ((None, None), 1),
            ((Some(43), None), 2),
            ((None, Some(5)), 2),
        ] {
            let mut spec = doc(
                r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["least-load"],"pick":"best"},"q":{"score":["least-load"],"pick":"best"}},"bindings":[{"match":{"model":"m"},"profile":"p"},{"match":{"model":"*"},"profile":"q"}]}"#,
            );
            let q = spec.profiles.get_mut("q").expect("the profile");
            (q.sent_ttl_ms, q.sent_report_delay_ms) = sent;
            let engine = spec.validate().expect("a valid document");
            assert_eq!(engine.sent_credits.len(), want, "q under {sent:?}");
            // Both scorers hold a count of the list, the ones the engine
            // feeds: the list holds each count once beside its readers.
            let readers = engine
                .sent_credits
                .iter()
                .map(|count| Arc::strong_count(count) - 1);
            assert_eq!(readers.sum::<usize>(), 2, "q under {sent:?}");
        }
    }

    /// A later document takes over a shared count by the name of the
    /// profile that made it: the same profiles again keep its sends, and
    /// where that profile is gone the count of the one that stays starts
    /// empty.
    #[test]
    fn a_later_document_takes_a_shared_count_by_the_name_of_its_maker() {
        let both = r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["least-load"],"pick":"best"},"q":{"score":["least-load"],"pick":"best"}},"bindings":[{"match":{"model":"m"},"profile":"p"},{"match":{"model":"*"},"profile":"q"}]}"#;
        let q_alone = r#"{"apiVersion":"smg.policy/v1","profiles":{"q":{"score":["least-load"],"pick":"best"}},"bindings":[{"match":{"model":"*"},"profile":"q"}]}"#;
        let worker = crate::engine::WorkerId(7);
        let target = crate::engine::Target {
            id: worker,
            ..Default::default()
        };
        // The later document, and the sends its one count holds.
        for (later, want) in [(both, 1), (q_alone, 0)] {
            let standing = doc(both).validate().expect("a valid document");
            standing.on_dispatched(
                &crate::tracking::Lease::default(),
                &target,
                crate::tracking::Side::Single,
                0,
            );
            let next = doc(later).validate().expect("a valid document");
            next.succeed(&standing);
            let held: Vec<_> = next
                .sent_credits
                .iter()
                .map(|count| count.credits.held(worker))
                .collect();
            assert_eq!(held, [want], "{later}");
        }
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
        for &name in SCORERS {
            let bare = spec_with_score(&format!("{name:?}"));
            if in_build.contains(&name) {
                assert!(bare.validate().is_ok(), "{name}");
            } else if matches!(name, "resident-work" | "least-cost") {
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
                rule_placed: RulePlaced::Told,
            }
        );
        assert_eq!(spec.view.headers, ["x-smg-routing-key"]);
    }

    /// The manual picker places by a sessions rule, and a rule runs under
    /// the single plan and on the two sides of a pair: on a side of `epd`
    /// the profile is refused, with the side that names it. The built
    /// picker answers, so any picker that leaves the placement to the rule
    /// is refused the same way.
    #[test]
    fn the_manual_picker_under_the_epd_plan_is_not_in_this_build() {
        let epd = doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"q":{"pick":"random"},"sticky":{"pick":"manual"}},"bindings":[{"plan":"epd","profiles":{"encode":"sticky","prefill":"q","decode":"q"}}]}"#,
        );
        assert_eq!(
            not_in_build(&epd),
            (
                "bindings[0].profiles.encode".to_string(),
                "a picker that places by the sessions rule under the epd plan"
            )
        );
        // On a side of a pair the same profile validates, and so it does
        // under the single plan, here with no rule; the preset is the
        // profile with one.
        let pair = doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"q":{"pick":"random"},"sticky":{"pick":"manual"}},"bindings":[{"plan":"pd-pair","profiles":{"prefill":"q","decode":"sticky"}}]}"#,
        );
        assert!(pair.validate().is_ok());
        let single = doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"sticky":{"pick":"manual"}},"bindings":[{"profile":"sticky"}]}"#,
        );
        assert!(single.validate().is_ok());
    }

    /// A host that makes some lifecycle calls only keeps one session
    /// verdict for the two sides of a pair's lease. A sessions rule under
    /// the pair plan is not in its build, by the rule's place. The same
    /// rule under the single plan is, and so is a pair under no rule.
    #[test]
    fn a_pairs_sessions_rule_is_not_in_the_build_of_a_host_with_one_verdict_a_lease() {
        let (whole, some) = (
            StageRegistry::builtin(),
            StageRegistry::builtin().without_lifecycle(),
        );
        let rule = r#"{"sticky_header":"x-session-id","inflight_cap":65535}"#;
        let pair = format!(
            r#"{{"plan":"pd-pair","profiles":{{"prefill":"q","decode":"q"}},"sessions":{rule}}}"#
        );
        let pair = spec_with_binding(&pair);
        assert!(pair.validate_in(&whole, &[]).is_ok());
        assert_eq!(
            pair.validate_in(&some, &[]).err(),
            Some(SpecError::NotInBuild {
                path: "bindings[0].sessions".to_string(),
                what: "the session verdict a pair's decode side keeps of its own",
            })
        );
        let single = spec_with_binding(&format!(r#"{{"profile":"q","sessions":{rule}}}"#));
        assert!(single.validate_in(&some, &[]).is_ok());
        let unruled =
            spec_with_binding(r#"{"plan":"pd-pair","profiles":{"prefill":"q","decode":"q"}}"#);
        assert!(unruled.validate_in(&some, &[]).is_ok());
    }

    /// A sessions rule runs under the single plan, and on the two sides of
    /// a pair where it states no cap and tells its stages. A cap under the
    /// pair is refused by its own name, the schema's default cap as one
    /// that is written. A rule that keeps what it places from the profile
    /// is refused by that field's name, and a rule under `epd` by the
    /// plan's.
    #[test]
    fn a_sessions_rule_runs_under_a_pair_with_no_cap_and_not_under_epd() {
        let under = |plan: &str, profiles: &str, sessions: &str| {
            spec_with_binding(&format!(
                r#"{{"plan":"{plan}","profiles":{profiles},"sessions":{sessions}}}"#
            ))
            .validate()
            .err()
        };
        let pair = r#"{"prefill":"q","decode":"q"}"#;
        let uncapped = r#"{"sticky_header":"x-session-id","inflight_cap":65535}"#;
        assert_eq!(under("pd-pair", pair, uncapped), None);
        for capped in [
            r#"{"sticky_header":"x-session-id"}"#,
            r#"{"sticky_header":"x-session-id","inflight_cap":4}"#,
        ] {
            assert_eq!(
                under("pd-pair", pair, capped),
                Some(SpecError::BadSessions {
                    index: 0,
                    name: "inflight_cap",
                    reason: "65535 (no cap) under the pd-pair plan",
                }),
                "{capped}"
            );
        }
        let untold =
            r#"{"sticky_header":"x-session-id","inflight_cap":65535,"rule_placed":"untold"}"#;
        assert_eq!(
            under("pd-pair", pair, untold),
            Some(SpecError::BadSessions {
                index: 0,
                name: "rule_placed",
                reason: "`told` under the pd-pair plan",
            })
        );
        // The single plan takes each of the three.
        for rule in [uncapped, untold, r#"{"sticky_header":"x-session-id"}"#] {
            let single = spec_with_binding(&format!(r#"{{"profile":"q","sessions":{rule}}}"#));
            assert_eq!(single.validate().err(), None, "{rule}");
        }
        let epd = r#"{"encode":"q","prefill":"q","decode":"q"}"#;
        let refused = under("epd", epd, uncapped);
        assert_eq!(
            refused,
            Some(SpecError::BadSessions {
                index: 0,
                name: "plan",
                reason: "`single` or `pd-pair`",
            })
        );
        assert_eq!(
            refused.map(|err| err.to_string()),
            Some("binding 0: sessions plan is not `single` or `pd-pair`".to_string())
        );
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
        let thresholds = |spec: &PolicySpec| {
            spec.validate()
                .expect("the binding validates")
                .binding("m", Phase::Single)
                .and_then(|binding| binding.overload)
        };
        // A block names no number. Switched on, a binding hands its pools
        // the crate's two, the usage in permille, whatever its shed word
        // says. So does a binding with no block, and the binding of a
        // document that writes none.
        let defaults = OverloadThresholds {
            waiting_requests: OVERLOAD_WAITING_REQUESTS,
            kv_usage_permille: 800,
        };
        for on in [
            r#"{"profile":"p"}"#,
            r#"{"profile":"p","overload":{}}"#,
            r#"{"profile":"p","overload":{"enabled":true,"shed_when_all_vetoed":true}}"#,
            r#"{"profile":"p","overload":{"waiting_requests":null,"token_usage":null}}"#,
        ] {
            assert_eq!(thresholds(&spec_with_binding(on)), Some(defaults), "{on}");
        }
        assert_eq!(
            thresholds(&doc(
                r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"score":["kv-pressure"],"pick":"best"}}}"#
            )),
            Some(defaults)
        );
        // Off is the block's switch alone: the binding hands no threshold.
        let off = r#"{"profile":"p","overload":{"enabled":false,"shed_when_all_vetoed":true}}"#;
        assert_eq!(thresholds(&spec_with_binding(off)), None);

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
        // A rule names its key's header, or the binding names `routing_key`.
        assert_eq!(
            spec_with_binding(r#"{"profile":"p","sessions":{"ttl_s":1}}"#)
                .validate()
                .err(),
            Some(SpecError::BadSessions {
                index: 0,
                name: "sticky_header",
                reason: "named while the binding names no routing_key",
            })
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

    /// A binding's `routing_key` block names one to eight sources, each a
    /// header outside the credential names or the request's own id, each
    /// once, with a byte cap inside the schema's. Beside it a sessions rule
    /// names no header of its own.
    #[test]
    fn a_routing_key_block_is_checked_in_shape() {
        let block = |block: &str| {
            spec_with_binding(&format!(r#"{{"profile":"p","routing_key":{block}}}"#)).validate()
        };
        // Every form of a source, `cookie` among the headers: no value of
        // these enters a view, so none is held to `view.headers`.
        assert!(block(
            r#"{"sources":[{"header":"x-smg-routing-key"},{"rid":true},{"header":"x-forwarded-for","max_bytes":8192},{"header":"cookie"}],"strip_lineage":true}"#
        )
        .is_ok());
        // A header named `rid` and the request's own id are two sources.
        assert!(block(r#"{"sources":[{"header":"rid"},{"rid":true}]}"#).is_ok());
        let refused = |text: &str| block(text).expect_err(text).to_string();
        let nine: Vec<String> = (0..9).map(|i| format!(r#"{{"header":"x-{i}"}}"#)).collect();
        let nine = format!(r#"{{"sources":[{}]}}"#, nine.join(","));
        // Client sources are counted apart: four stand beside eight.
        let beside_eight = |clients: usize| {
            let listed: Vec<String> = (0..8)
                .map(|i| format!(r#"{{"header":"x-{i}"}}"#))
                .chain((0..clients).map(|i| format!(r#"{{"header":"x-c{i}","client":true}}"#)))
                .collect();
            format!(r#"{{"sources":[{}]}}"#, listed.join(","))
        };
        assert!(block(&beside_eight(4)).is_ok());
        let five = beside_eight(5);
        for (text, message) in [
            (
                r#"{"sources":[]}"#,
                "bindings[0].routing_key.sources = 0 entries is not a list of one to eight sources",
            ),
            (
                nine.as_str(),
                "bindings[0].routing_key.sources = 9 entries is not a list of one to eight sources",
            ),
            (
                five.as_str(),
                "bindings[0].routing_key.sources = 5 client entries is not a list of four client sources at most",
            ),
            (
                r#"{"sources":[{}]}"#,
                "bindings[0].routing_key.sources[0] = neither is not one of header and rid",
            ),
            (
                r#"{"sources":[{"header":"x-a","rid":true}]}"#,
                "bindings[0].routing_key.sources[0] = both is not one of header and rid",
            ),
            (
                r#"{"sources":[{"header":"X-A"}]}"#,
                "bindings[0].routing_key.sources[0].header = X-A is not a lowercase header name",
            ),
            (
                r#"{"sources":[{"header":"authorization"}]}"#,
                "bindings[0].routing_key.sources[0].header: header \"authorization\" is on the deny-list",
            ),
            (
                r#"{"sources":[{"header":"x-a"},{"header":"x-a"}]}"#,
                "bindings[0].routing_key.sources[1]: \"x-a\" appears twice",
            ),
            (
                r#"{"sources":[{"rid":true},{"rid":true}]}"#,
                "bindings[0].routing_key.sources[1]: \"rid\" appears twice",
            ),
            (
                r#"{"sources":[{"rid":true,"max_bytes":0}]}"#,
                "bindings[0].routing_key.sources[0].max_bytes = 0 is not between 1 and 8192",
            ),
            (
                r#"{"sources":[{"rid":true,"max_bytes":8193}]}"#,
                "bindings[0].routing_key.sources[0].max_bytes = 8193 is not between 1 and 8192",
            ),
        ] {
            assert_eq!(refused(text), message);
        }
        // A host's own denied name, its tenant header, is refused as a
        // credential is.
        let tenant = spec_with_binding(
            r#"{"profile":"p","routing_key":{"sources":[{"header":"x-tenant-id"}]}}"#,
        );
        assert!(tenant.validate().is_ok());
        assert_eq!(
            tenant.validate_with(&["x-tenant-id"]).err(),
            Some(SpecError::DeniedHeader {
                path: "bindings[0].routing_key.sources[0].header".to_string(),
                name: "x-tenant-id".to_string(),
            })
        );
        // A client source is a header read whole.
        let client = |source: &str| {
            spec_with_binding(&format!(
                r#"{{"profile":"p","routing_key":{{"sources":[{source}]}}}}"#
            ))
            .validate()
            .err()
        };
        assert_eq!(
            client(r#"{"header":"x-a"},{"header":"cookie","client":true}"#),
            None
        );
        // Client sources alone are no block: a sessions rule beside the
        // block would have no source of a session's key.
        assert_eq!(
            client(r#"{"header":"cookie","client":true}"#).map(|refused| refused.to_string()),
            Some(
                "bindings[0].routing_key.sources = client entries alone is not a list of one to eight sources"
                    .to_string()
            )
        );
        // A header is named once each way.
        assert_eq!(
            client(r#"{"header":"x-forwarded-for"},{"header":"x-forwarded-for","client":true}"#),
            None
        );
        assert!(matches!(
            client(r#"{"header":"cookie","client":true},{"header":"cookie","client":true}"#),
            Some(SpecError::Duplicate { .. })
        ));
        for refused in [
            r#"{"rid":true,"client":true}"#,
            r#"{"header":"cookie","max_bytes":64,"client":true}"#,
        ] {
            assert!(
                matches!(
                    client(refused),
                    Some(SpecError::BadField { path, .. }) if path.ends_with("sources[0].client")
                ),
                "{refused}"
            );
        }
        // A target source stands beside the eight and the four. It hands
        // no key, so it is no block alone, nor with client sources only: a
        // sessions rule beside the block would have no source of a
        // session's key.
        let with_target =
            |block: String| block.replace("]}", r#",{"header":"x-target","target":true}]}"#);
        assert!(block(&with_target(beside_eight(4))).is_ok());
        for alone in [
            r#"{"header":"x-target","target":true}"#,
            r#"{"header":"cookie","client":true},{"header":"x-target","target":true}"#,
        ] {
            assert_eq!(
                client(alone).map(|refused| refused.to_string()),
                Some(
                    "bindings[0].routing_key.sources = a target entry and no source of a key is not a list of one to eight sources"
                        .to_string()
                ),
                "{alone}"
            );
        }
        // A header is named once as the target's too, and a block names
        // one target.
        assert_eq!(
            client(r#"{"header":"x-a"},{"header":"x-a","target":true}"#),
            None
        );
        assert_eq!(
            client(r#"{"header":"x-a","target":true},{"header":"x-b","target":true}"#)
                .map(|refused| refused.to_string()),
            Some(
                "bindings[0].routing_key.sources = 2 target entries is not a list of one target source at most"
                    .to_string()
            )
        );
        // A target source is a header alone.
        for refused in [
            r#"{"rid":true,"target":true}"#,
            r#"{"header":"x-target","max_bytes":64,"target":true}"#,
            r#"{"header":"x-target","client":true,"target":true}"#,
        ] {
            assert!(
                matches!(
                    client(refused),
                    Some(SpecError::BadField { path, .. }) if path.ends_with("sources[0].target")
                ),
                "{refused}"
            );
        }
        // One spelling or the other beside a sessions rule.
        let beside = |sessions: &str| {
            spec_with_binding(&format!(
                r#"{{"profile":"p","routing_key":{{"sources":[{{"rid":true}}]}},"sessions":{sessions}}}"#
            ))
            .validate()
            .err()
        };
        assert_eq!(beside(r#"{"ttl_s":600}"#), None);
        for (sessions, name) in [
            (r#"{"sticky_header":"x-session-id"}"#, "sticky_header"),
            (r#"{"headers":["x-session-id"]}"#, "headers"),
        ] {
            assert_eq!(
                beside(sessions),
                Some(SpecError::BadSessions {
                    index: 0,
                    name,
                    reason: "left out while the binding names routing_key",
                })
            );
        }
    }

    /// The applied document answers a request's key sources from whichever
    /// spelling its binding writes: the `routing_key` block, or a sessions
    /// rule's own headers. A binding with neither names none.
    #[test]
    fn the_applied_document_answers_a_bindings_key_sources() {
        let applied = AppliedSpec::validate(&doc(
            r#"{"apiVersion":"smg.policy/v1","view":{"headers":["x-session-id","x-tenant-session"]},"profiles":{"p":{"pick":"best"}},"bindings":[
                {"match":{"model":"chat/*"},"profile":"p","routing_key":{"sources":[{"header":"x-smg-routing-key"},{"rid":true},{"header":"cookie"}]}},
                {"match":{"model":"code/*"},"profile":"p","sessions":{"sticky_header":"x-session-id","headers":["x-tenant-session","x-session-id"]}},
                {"profile":"p"}]}"#,
        ))
        .expect("the document validates");
        let read = |model: &str| {
            applied
                .key_sources_for(model, Phase::Single)
                .map(|sources| (sources.headers().collect::<Vec<_>>(), sources.reads_rid()))
        };
        assert_eq!(
            read("chat/a"),
            Some((vec!["x-smg-routing-key", "cookie"], true))
        );
        assert_eq!(
            read("code/a"),
            Some((vec!["x-tenant-session", "x-session-id"], false))
        );
        assert_eq!(read("other"), None, "a binding with no source");
        // A rule's own header is on the view's allow-list: its key leaves in
        // the clear.
        let rule = applied
            .key_sources_for("code/a", Phase::Single)
            .expect("the rule's headers");
        let derived = rule
            .derive(|name| (name == "x-session-id").then_some(&b"s-1"[..]), None)
            .expect("a key");
        assert_eq!(derived.key, crate::engine::Identity::plain("s-1"));
        assert!(AppliedSpec::unbound()
            .key_sources_for("chat/a", Phase::Single)
            .is_none());
        // One binding of the three keys by the request's own id, so the
        // document does. The same document without that source does not,
        // and neither does one that binds nothing.
        assert!(applied.reads_request_id());
        let headers_alone = AppliedSpec::validate(&doc(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"best"}},"bindings":[
                {"match":{"model":"chat/*"},"profile":"p","routing_key":{"sources":[{"header":"x-smg-routing-key"},{"header":"cookie"}]}},
                {"profile":"p"}]}"#,
        ))
        .expect("the document validates");
        assert!(!headers_alone.reads_request_id());
        assert!(!AppliedSpec::unbound().reads_request_id());
    }

    /// Whether a binding names a target source is decided when the document
    /// is applied. One binding of several is enough, and its sources alone
    /// say so. A document that names none says so, as the generation no
    /// document was applied to does.
    #[test]
    fn the_applied_document_says_whether_a_binding_names_a_target() {
        let applied = |second: &str| {
            AppliedSpec::validate(&doc(&format!(
                r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"pick":"best"}}}},"bindings":[
                {{"match":{{"model":"chat/*"}},"profile":"p","routing_key":{{"sources":[{{"header":"x-smg-routing-key"}}]}}}},
                {{"profile":"p"{second}}}]}}"#
            )))
            .expect("the document validates")
        };
        assert!(!applied("").names_target());
        let named = applied(
            r#","routing_key":{"sources":[{"header":"x-b"},{"header":"x-smg-target-worker","target":true}]}"#,
        );
        assert!(named.names_target());
        let sources = |model| named.key_sources_for(model, Phase::Single);
        let first = sources("chat/a").expect("the first binding's sources");
        assert!(!first.names_target());
        let second = sources("other").expect("the second binding's sources");
        assert!(second.names_target());
        assert_eq!(
            second.headers().collect::<Vec<_>>(),
            ["x-b"],
            "no key's source"
        );
        assert!(!AppliedSpec::unbound().names_target());
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

    /// A binding's overload block names no number. Either one is refused by
    /// its path, with the block switched on, switched off or left as it
    /// stands, at the crate's own number too, and the refusal says where a
    /// threshold is stated and states the number as it was written. A block
    /// that names both is refused by the count, switched off too.
    #[test]
    fn a_number_in_a_bindings_overload_block_is_refused_by_its_path() {
        const STATED: &str =
            "left out: a worker's registration states this threshold, or its program's default does";
        let refused = |block: &str| {
            let binding = format!(r#"{{"profile":"p","overload":{block}}}"#);
            match spec_with_binding(&binding).validate().err() {
                Some(SpecError::BadField {
                    path,
                    value,
                    reason,
                }) => (path, value, reason),
                other => panic!("{block} validated as {other:?}"),
            }
        };
        for (block, field, value) in [
            (r#"{"waiting_requests":8}"#, "waiting_requests", "8"),
            (r#"{"waiting_requests":0}"#, "waiting_requests", "0"),
            (r#"{"waiting_requests":300}"#, "waiting_requests", "300"),
            (
                r#"{"enabled":true,"waiting_requests":32}"#,
                "waiting_requests",
                "32",
            ),
            (
                r#"{"enabled":false,"waiting_requests":3}"#,
                "waiting_requests",
                "3",
            ),
            (r#"{"token_usage":0.8}"#, "token_usage", "0.8"),
            (r#"{"token_usage":1.5}"#, "token_usage", "1.5"),
            (r#"{"token_usage":0}"#, "token_usage", "0"),
            (r#"{"token_usage":-0.5}"#, "token_usage", "-0.5"),
            (
                r#"{"enabled":true,"token_usage":0.9}"#,
                "token_usage",
                "0.9",
            ),
            (
                r#"{"enabled":false,"token_usage":0.5}"#,
                "token_usage",
                "0.5",
            ),
            (
                r#"{"waiting_requests":32,"token_usage":0.9,"shed_when_all_vetoed":true}"#,
                "waiting_requests",
                "32",
            ),
            (
                r#"{"enabled":false,"waiting_requests":32,"token_usage":0.9}"#,
                "waiting_requests",
                "32",
            ),
        ] {
            assert_eq!(
                refused(block),
                (
                    format!("bindings[0].overload.{field}"),
                    value.to_owned(),
                    STATED
                ),
                "{block}"
            );
        }
        // The refusal as a document's writer reads it.
        let named = spec_with_binding(r#"{"profile":"p","overload":{"token_usage":0.9}}"#);
        assert_eq!(
            named.validate().err().map(|refusal| refusal.to_string()),
            Some(format!(
                "bindings[0].overload.token_usage = 0.9 is not {STATED}"
            ))
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
    fn the_plans_section_says_when_a_pairs_decode_is_picked() {
        let document = |plans: &str| {
            doc(&format!(
                r#"{{"apiVersion":"smg.policy/v1"{plans},"profiles":{{"p":{{"pick":"best"}}}}}}"#
            ))
        };
        let applied =
            |spec: &PolicySpec| AppliedSpec::validate(spec).map(|applied| applied.decode_pick());
        // A document that names no rule, and one that names the other rule
        // of the plan alone, pick the decode with the prefill.
        for plans in ["", r#","plans":{"pd-pair":{"pairing":"strict"}}"#] {
            let spec = document(plans);
            assert_eq!(
                spec.plans.pd_pair.decode,
                DecodePick::WithPrefill,
                "{plans}"
            );
            assert_eq!(
                applied(&spec).ok(),
                Some(DecodePick::WithPrefill),
                "{plans}"
            );
        }
        let spec = document(r#","plans":{"pd-pair":{"decode":"with-prefill"}}"#);
        assert_eq!(applied(&spec).ok(), Some(DecodePick::WithPrefill));
        let spec = document(r#","plans":{"pd-pair":{"decode":"after-prefill"}}"#);
        assert_eq!(spec.plans.pd_pair.decode, DecodePick::AfterPrefill);
        assert_eq!(spec.plans.pd_pair.pairing, Pairing::default());
        assert_eq!(applied(&spec).ok(), Some(DecodePick::AfterPrefill));
        // The document written back names the rule, and reads back the same.
        let written = serde_json::to_string(&spec).expect("the document writes");
        assert!(written.contains(r#""decode":"after-prefill""#), "{written}");
        assert_eq!(doc(&written), spec);
        // A start with no document picks with the prefill.
        assert_eq!(
            AppliedSpec::unbound().decode_pick(),
            DecodePick::WithPrefill
        );
        let err = parse_error(
            r#"{"apiVersion":"smg.policy/v1","plans":{"pd-pair":{"decode":"later"}},"profiles":{"p":{"pick":"best"}}}"#,
        );
        assert!(err.contains("unknown variant `later`"), "{err}");
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
            "least_load": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" },
            "spare": { "pick": "random" }
          },
          "plans": { "pd-pair": { "pairing": "lenient" } },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"], "tenant": "acme", "priority": { "max": -1 }, "header": { "name": "x-tenant-tier", "values": ["gold"] }, "prompt_tokens": { "min": 1, "max": 4096 }, "labels": ["large"] },
              "plan": "pd-pair", "profiles": { "prefill": "least_load", "decode": "least_load" } },
            { "match": { "model": "*" }, "profile": "cache_aware", "shadow": "spare",
              "canary": { "profile": "least_load", "percent": 5 }, "override": "allowed",
              "overload": { "enabled": true, "shed_when_all_vetoed": false },
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

    /// A filter entry is written as it was read: a bare name as the string
    /// it always was, a name with parameters as its one-key map, the keys
    /// of every map in order. The order a file wrote the keys in is in no
    /// digest.
    #[test]
    fn a_filter_entry_round_trips_a_bare_name_as_a_string_and_a_map_as_a_map() {
        let document = |entry: &str| {
            doc(&format!(
                r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"filter":["healthy",{entry},"not-tried"],"pick":"random"}}}}}}"#
            ))
        };
        let spec = document(r#"{"min-free":{"slots":2,"by":{"zone":3,"class":"a"}}}"#);
        assert_eq!(
            spec.profiles["p"].filter,
            [
                FilterSpec::Name("healthy".into()),
                FilterSpec::Added {
                    name: "min-free".into(),
                    written: serde_json::json!({ "slots": 2, "by": { "zone": 3, "class": "a" } }),
                },
                "not-tried".into(),
            ]
        );
        let written = serde_json::to_string(&spec).expect("serializes");
        assert!(
            written.contains(
                r#""filter":["healthy",{"min-free":{"by":{"class":"a","zone":3},"slots":2}},"not-tried"]"#
            ),
            "{written}"
        );
        assert_eq!(doc(&written), spec);
        let yaml = serde_yaml::to_string(&spec).expect("yaml");
        let from_yaml: PolicySpec = serde_yaml::from_str(&yaml).expect("yaml round trip");
        assert_eq!(from_yaml, spec);
        // Another order of the same keys is the same document, and another
        // value is another.
        let reordered = document(r#"{"min-free":{"by":{"class":"a","zone":3},"slots":2}}"#);
        assert_eq!(reordered.digest(), spec.digest());
        assert_eq!(
            serde_json::to_string(&reordered).expect("serializes"),
            written
        );
        let moved = document(r#"{"min-free":{"slots":3,"by":{"zone":3,"class":"a"}}}"#);
        assert_ne!(moved.digest(), spec.digest());
        // An entry names one filter.
        for (entry, message) in [
            (r#"{"min-free":{"slots":2},"healthy":{}}"#, "single key"),
            ("{}", "names one filter"),
            (
                "7",
                "a filter name, or one filter name mapped to its parameters",
            ),
        ] {
            let err = parse_error(&format!(
                r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"filter":[{entry}],"pick":"random"}}}}}}"#
            ));
            assert!(err.contains(message), "{entry}: {err}");
        }
    }

    /// YAML reads a bare `true`, a number or `~` as no string. A filter of
    /// such a name is written in quotes, as a scorer of such a name is.
    #[test]
    fn a_filter_name_yaml_reads_as_no_string_is_written_in_quotes() {
        let document = |entry: &str| {
            format!(
                "apiVersion: smg.policy/v1\nprofiles: {{p: {{filter: [{entry}], pick: random}}}}\n"
            )
        };
        for bare in ["true", "7", "~"] {
            let refused = serde_yaml::from_str::<PolicySpec>(&document(bare))
                .expect_err("no string")
                .to_string();
            assert!(
                refused.contains("a filter name, or one filter name mapped to its parameters"),
                "{bare}: {refused}"
            );
        }
        let quoted: PolicySpec = serde_yaml::from_str(&document(r#""true""#)).expect("a name");
        assert_eq!(
            quoted.profiles["p"].filter,
            [FilterSpec::Name("true".into())]
        );
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
    /// default (1.5, plus the margin) it spills. Of two entries each spills
    /// by its own gate.
    #[test]
    fn the_spill_parameters_of_the_spec_reach_the_profiles_gates() {
        use crate::engine::{
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
        // Two entries, each under its own gate: the heavier one keeps the
        // holder whichever is written first.
        let heavy = r#"{"prefix-affinity":{"weight":4.0,"spill":{"rel":4.0}}}"#;
        let light = r#"{"prefix-affinity":{"weight":1.0}}"#;
        for score in [format!("{light},{heavy}"), format!("{heavy},{light}")] {
            assert_eq!(leader(&score), 0, "{score}");
        }
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
            .push("not-tried".into());
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
        let held = SignalSpec {
            unit: "tokens".to_owned(),
            max_age_ms: None,
            across_ranks: AcrossRanks::default(),
        };
        let signals = BTreeMap::from([("held".to_owned(), held)]);
        let mut profiles = base.profiles.clone();
        let profile = profiles.values().next().expect("a profile").clone();
        profiles.insert("another".to_owned(), profile);
        let plans = PlansSpec {
            pd_pair: PdPairPlanSpec {
                pairing: Pairing::Off,
                ..PdPairPlanSpec::default()
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
                "explain",
                PolicySpec {
                    explain: ExplainSpec { sample: 2 },
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
                "signals",
                PolicySpec {
                    signals,
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
            explain: ExplainSpec { sample: 2 },
            view: ViewSpec {
                headers: vec!["x-team".to_owned()],
                ..base.view.clone()
            },
            index: BTreeMap::from([("another*".to_owned(), IndexSpec::default())]),
            signals: BTreeMap::from([(
                "held".to_owned(),
                SignalSpec {
                    unit: "tokens".to_owned(),
                    max_age_ms: Some(4_000),
                    across_ranks: AcrossRanks::Max,
                },
            )]),
            plans: PlansSpec {
                pd_pair: PdPairPlanSpec {
                    pairing: Pairing::Off,
                    ..PdPairPlanSpec::default()
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

    #[test]
    fn validate_in_builds_the_registered_scorer_for_a_store_decision() {
        use crate::{
            engine::{Health, PoolFacts, Scratch, WorkerId},
            store::MemoryStore,
        };

        const SCORER: &str = "prefer-idle";

        #[derive(Debug)]
        struct PreferIdle;

        impl Lifecycle for PreferIdle {}

        impl Scorer for PreferIdle {
            fn name(&self) -> &'static str {
                SCORER
            }

            fn score(
                &self,
                _: &RequestFacts<'_>,
                candidates: &[Candidate],
                kept: &[u32],
                out: &mut [f32],
            ) -> TieBand {
                for (&index, score) in kept.iter().zip(out) {
                    *score = if candidates[index as usize].in_flight == 0 {
                        1.0
                    } else {
                        0.0
                    };
                }
                TieBand::NORMALIZED
            }
        }

        let mut catalog = StageRegistry::builtin();
        catalog
            .scorer(SCORER, || Box::new(PreferIdle))
            .expect("a free scorer name");
        let mut document = spec_with_score(&format!(r#""{SCORER}""#));
        document.generation = 43;
        let applied = AppliedSpec::validate_in(&document, &catalog, &[])
            .expect("the registered scorer validates");
        let store = MemoryStore::new();
        store.apply(applied).expect("a newer generation");

        store.deciding(|world| {
            let spec = world.spec();
            let pool = PoolFacts {
                pool_size: 2,
                generation: spec.generation(),
                ..PoolFacts::default()
            };
            let mut scratch = Scratch::new();
            for idle in [0, 1] {
                let row = |position| Candidate {
                    position,
                    id: WorkerId(137 + u64::from(position)),
                    health: Health {
                        healthy: true,
                        ..Health::default()
                    },
                    in_flight: u32::from(position != idle),
                    ..Candidate::default()
                };
                let decision = spec.engine().decide(
                    &RequestFacts::default(),
                    &pool,
                    &[row(0), row(1)],
                    &Constraints::default(),
                    &mut scratch,
                );
                let targets = &decision.pick().expect("a pick").targets;
                assert_eq!(targets.first().expect("a target").position, idle);
            }
        });
    }

    #[test]
    fn validate_in_refuses_headers_the_host_denies() {
        let document = spec_with_view(r#"{"headers":["x-private-context"]}"#);
        let catalog = StageRegistry::builtin();
        let denied = ["X-Private-Context"];
        let expected = SpecError::DeniedHeader {
            path: "view.headers[0]".into(),
            name: "x-private-context".into(),
        };
        assert!(AppliedSpec::validate_in(&document, &catalog, &[]).is_ok());
        assert_eq!(
            AppliedSpec::validate_in(&document, &catalog, &denied).expect_err("a denied header"),
            expected
        );
        assert_eq!(
            AppliedSpec::validate_with(&document, &denied).expect_err("a denied header"),
            expected
        );
    }

    #[test]
    fn validate_in_preserves_the_stock_generation_digest_and_depth() {
        let mut document = spec_with_score(r#"{"consistent-hash":{"prefix_tokens":143}}"#);
        document.generation = 47;
        let catalog = StageRegistry::builtin();
        let applied = AppliedSpec::validate_in(&document, &catalog, &[])
            .expect("a stock scorer validates in the stock catalog");
        let stock = AppliedSpec::validate_with(&document, &[]).expect("a stock scorer validates");
        let depth = |spec: &AppliedSpec| spec.block_hashes_read(HashKind::Token, 11);
        assert_eq!(applied.generation(), 47);
        assert_eq!(applied.generation(), stock.generation());
        assert_eq!(applied.digest(), Some(document.digest().expect("a digest")));
        assert_eq!(applied.digest(), stock.digest());
        assert_eq!(depth(&applied), 13);
        assert_eq!(depth(&applied), depth(&stock));
    }

    #[test]
    fn validate_in_leaves_an_unnumbered_document_at_generation_zero() {
        use crate::store::{MemoryStore, StaleGeneration};

        let document = spec_with_view("{}");
        let catalog = StageRegistry::builtin();
        let validate = |document: &PolicySpec| {
            AppliedSpec::validate_in(document, &catalog, &[]).expect("the document validates")
        };
        let store = MemoryStore::new();
        let standing = 71;
        store
            .apply(validate(&PolicySpec {
                generation: standing,
                ..document.clone()
            }))
            .expect("a newer generation");

        let unnumbered = validate(&document);
        assert_eq!(unnumbered.generation(), 0);
        assert_eq!(
            store.apply(unnumbered),
            Err(StaleGeneration {
                standing,
                offered: 0,
            })
        );
        assert_eq!(store.snapshot().spec().generation(), standing);
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
        let applied =
            AppliedSpec::validate_digested(&document, handed, &StageRegistry::builtin(), &[])
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
        let handed =
            AppliedSpec::validate_digested(&document, digest, &StageRegistry::builtin(), &[])
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
    /// reads another count beside lent rows, and parts here. The same
    /// lists hold each stock stage to its word on what the gateways
    /// report of leases
    /// ([`Filter::refuses_by_reports`](crate::engine::Filter::refuses_by_reports)).
    #[test]
    fn every_catalog_stage_reads_a_lent_count_as_a_rows_own() {
        use crate::{
            engine::{view::Identity, Health, PoolFacts, Scratch, WorkerId},
            tracking::in_flight::tests::Lent,
        };
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
            vec![
                r#"{"score":[{"least-cost":{"terms":{"candidate.in_flight":3}}}],"pick":"best"}"#
                    .into(),
            ],
            vec![format!(r#"{{"score":[{bounded}],"pick":"best"}}"#)],
            vec![scored(tempered), alone(tempered)],
        ];
        let apart = ["resident-work", "least-cost", "best-with-temperature"];

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
            // No stock stage takes a silent stage's side: each says it
            // refuses by nothing a gateway reports of leases, so a decision
            // under it applies nothing first.
            let (_, _, applies_first) = build(forms).reads_for("m", Phase::Single);
            assert!(!applies_first, "a stock stage says no line: {forms:?}");
        }
    }
}
