//! `PolicySpec` (design 4.6): the one versioned document. The view says what
//! the host builds into every decision, the index map what the prefix index
//! keeps per model, profiles name stages with their parameters, plans carry
//! their rules, bindings match requests to profiles or plans, and explain
//! sets the decision sampling rate. Validation answers what the host must
//! feed before any request exists.
//!
//! The document has an `apiVersion` and no `kind`: its forms are a file and
//! the API. An unknown field is refused by name, never ignored. The one
//! exception is the parameters of a scorer, a filter or a picker a host
//! registers, which its factory's own type reads
//! ([`crate::stages::StageBuild::params`]). A field this
//! build cannot honour yet (a stage, a plan, a matcher or a binding rule of a
//! later step) is checked in shape like every other field and then refused
//! by name as not in this build, so a document written against the published
//! schema validates the day its stage lands. [`PolicySpec::schema`] is that
//! schema; the field documentation of this module is its descriptions.

use std::{
    borrow::Cow,
    collections::{btree_map, BTreeMap},
    fmt,
    marker::PhantomData,
    sync::LazyLock,
};

use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    ser::SerializeMap,
    Deserialize, Deserializer, Serialize, Serializer,
};

// The crate's own paths through this module still name the type here.
pub(crate) use crate::document::applied::AppliedSpec;
use crate::{
    engine::view::{Phase, MAX_LOAD_REPORT_AGE_MS},
    stages::{
        cost_terms::{CostTerm, SIGNAL_FAMILY},
        filters::HasRoomRule,
        registry::StageRegistry,
        scorers::LeastCost,
    },
};

mod digest;
mod presets;
mod validate;
mod view_lists;

pub use self::{
    digest::Digest,
    presets::{MANUAL_SESSION_HEADER, PRESETS},
    view_lists::{ViewListPositions, ViewLists},
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
pub(super) static STOCK: LazyLock<StageRegistry> = LazyLock::new(StageRegistry::builtin);

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
/// use smg_grand_central::document::Needs;
/// let needs = Needs::NONE.with_loads().with_prompt_size();
/// assert!(needs.loads && needs.prompt_size && !needs.events);
/// ```
///
/// ```compile_fail
/// use smg_grand_central::document::Needs;
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
    /// `token_hashes` reads the request's holders from the index. For its
    /// profile a host names a single request of token ids by the index
    /// alone where the model's index holds a block, and every other
    /// request's holders in the index ahead of the record's. A stage that
    /// reads the events for another fact of a row, as a warm-up does,
    /// declares no hashes beside it.
    pub events: bool,
    /// The stage reads the prefixes sent to each worker. Beside a stage of
    /// its profile that reads a request's holders from the index, it is
    /// named none for a single request of token ids under a model whose
    /// index holds a block: the index names that request alone.
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
#[non_exhaustive]
pub struct ViewSpec {
    /// The request headers a stage may read, lowercase, in the order their
    /// positions take on the wire. A name on the deny-list (`authorization`,
    /// `proxy-authorization`, `cookie`, `set-cookie`, `x-api-key`, `api-key`,
    /// `x-goog-api-key`, the host's tenant header) is refused. Every header a
    /// binding's `sessions`, `canary` or `match.header` names must be listed
    /// here. At most 256: a view names a header by its position in one byte.
    #[schemars(length(max = 256), inner(regex(pattern = HEADER_NAME_PATTERN)))]
    #[schemars(extend("uniqueItems" = true))]
    pub headers: Vec<String>,
    /// The token block size the host hashes prefixes by when no engine has
    /// published one; absent, the host's own default. The block size of
    /// record learned from an engine's events wins at run time.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub block_size: Option<u32>,
    /// Ascending token positions, each above zero, that the hash index and
    /// the prefix-hash ring key on; empty for none. A hash index needs at
    /// least one.
    #[schemars(inner(range(min = 1)))]
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
    #[schemars(length(max = 64), inner(regex(pattern = LABEL_SELECTOR_PATTERN)))]
    #[schemars(extend("uniqueItems" = true))]
    pub labels: Vec<String>,
    /// The zones a stage may tell apart. A worker's row names its zone by
    /// its position in this list, from 1; a worker with no zone reads 0,
    /// and so does one whose zone this list does not name. At most 256;
    /// empty for none, where every row reads 0.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(max = 256), inner(length(min = 1)))]
    #[schemars(extend("uniqueItems" = true))]
    pub zones: Vec<String>,
    /// The engine families a stage may tell apart, spelled as a worker's
    /// registration spells its own. A worker's row names its family by its
    /// position in this list, from 1; a worker whose family this list does
    /// not name reads 0. At most 64; empty for none, where every row reads
    /// 0.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(max = 64), inner(length(min = 1)))]
    #[schemars(extend("uniqueItems" = true))]
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

impl ViewSpec {
    /// Whether this view and `other` are equal but for their header lists.
    /// Two equal views are such a pair.
    #[cfg(feature = "in-process")]
    pub(crate) fn same_but_for_headers(&self, other: &Self) -> bool {
        // Every field stands here by its name, with no `..`. So a field
        // added to the view does not build until this function compares it
        // or names it as left out.
        let Self {
            headers: _,
            block_size,
            boundaries,
            max_block_hashes,
            max_report_age_ms,
            labels,
            zones,
            engines,
        } = self;
        *block_size == other.block_size
            && *boundaries == other.boundaries
            && *max_block_hashes == other.max_block_hashes
            && *max_report_age_ms == other.max_report_age_ms
            && *labels == other.labels
            && *zones == other.zones
            && *engines == other.engines
    }
}

/// The under-layer of a model's prefix index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum IndexKind {
    /// Radix trees over the prefixes seen.
    #[default]
    Tree,
    /// The record of sent prefixes keyed by the request's head at
    /// `view.boundaries` alone, as the classic hash index keys it. The
    /// crate's engine keys the record at the last whole block under each
    /// boundary that holds one, where the request carries a block hash
    /// for each block under the boundary. A request that carries
    /// fewer block hashes than the first such boundary has whole
    /// blocks, or one hashed by its text, has no key: the record
    /// names no worker for it and takes no placement for it. A
    /// boundary of more whole blocks than `view.max_block_hashes`
    /// keys nothing either, for a request that carries no more
    /// hashes than that. A holder's depth is its boundary's, so a
    /// `prefix-affinity` stage counts it only where that share of the
    /// request clears the stage's `threshold`. The settings of the classic
    /// hash index translate to a threshold of zero, which counts a holder
    /// at any boundary; a document written by hand writes `threshold: 0`
    /// for the same. No placement of the engine's record expires: the
    /// index's `max_size` bounds the record instead.
    Hash,
}

/// A model's prefix-index settings. They are per model, never per profile,
/// so two profiles bound to one model never carry two settings.
/// Two keys matching one model resolve by specificity, see
/// [`PolicySpec::index_for`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
pub struct IndexSpec {
    pub kind: IndexKind,
    /// How long a placement of the classic hash index stays routable, in
    /// seconds. The crate's engine does not read it: no placement of its
    /// record expires.
    #[schemars(range(min = 1))]
    pub ttl_s: u32,
    /// The size bound of a tree index, summed over the workers: on a text
    /// host the characters of the nodes each worker stands on plus 1024 for
    /// each such node, otherwise the tokens each worker holds, so a prefix
    /// that several workers hold counts once for each of them.
    /// The crate's engine reads it for a pool whose profile reads sent
    /// prefixes: the model's record is bound to `max_size` over the block
    /// size placements, never under one. A placement is a block a worker
    /// was sent, and under a hash index a boundary a worker was sent. A
    /// lowered bound is met by sweeps, at the record's next writes and at
    /// the engine's upkeep.
    #[schemars(range(min = 1))]
    pub max_size: u64,
    /// The index's sweep cadence, in seconds. At each sweep the gateway trims
    /// each tree of a model, by itself, toward `max_size`, and writes take a
    /// tree past it until the next. At `4294967295` it starts no sweep: a
    /// tree under such an entry is held to no size. The engine's record
    /// keeps its bound at any cadence.
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
#[expect(clippy::exhaustive_enums, reason = "a caller answers every case")]
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
#[non_exhaustive]
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
#[non_exhaustive]
pub struct PdPairPlanSpec {
    pub pairing: Pairing,
    /// When a pair's decode side is picked: `with-prefill` (the default)
    /// or `after-prefill`.
    pub decode: DecodePick,
}

/// The plans' rules, by plan name.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
pub enum AffinitySource {
    /// The index of cache events alone.
    Index,
    /// The record of the prefixes sent alone.
    Predictor,
    /// Each worker by what it has: its cache events where it publishes
    /// them, and the prefixes sent to it where it does not. A request
    /// hashed by its text reads the prefixes sent to every worker: cache
    /// events are token blocks. A single request of token ids under a
    /// model whose index holds a block is read from the index alone.
    #[default]
    Merged,
}

/// The spill gate of prefix affinity: when the affine worker is loaded this
/// far past the least-loaded one, the request spills to the latter.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
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
#[non_exhaustive]
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
/// yield to load. The stage orders nothing itself. A profile has one gate:
/// its `kv-pressure` entries write the same limits. Limits that are off (1
/// and above) count as the same.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
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
/// still thin. A single request warms nobody while the model's index of
/// cache events holds no block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// the request is keyed by its prefix as one with no routing key. A
    /// gateway's frame states to a controller where the key came from.
    /// The frame of a host older than that field states nothing, and
    /// under it the id's key places as any other. Absent, a request with
    /// no routing key has none.
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
#[non_exhaustive]
pub struct ResidentWorkParams {
    /// The scorer's weight in the profile's combined score.
    #[serde(default = "one")]
    #[schemars(extend("exclusiveMinimum" = 0))]
    pub weight: f32,
    /// What one request a candidate holds adds to its resident tokens, in
    /// context tokens; zero compares the tokens alone.
    pub request_tokens: u32,
}

/// The largest magnitude of a cost rule's parameter. A signal's value is
/// held to the same bound where its cell is written.
pub(crate) const COST_PARAMETER_LIMIT: f64 = 1e15;
const RELATIVE_MARGIN_LIMIT: f64 = 1e6;

/// The least-cost scorer: a sum of a row's fields times their coefficients,
/// in the document's units. Lower cost scores higher.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// read one count. A new document's count takes over the sends of the
    /// standing count under the same limit and delay, whichever profiles
    /// read that one, and the two counts share them. Where none stands
    /// under them, it takes a copy of the sends the standing counts hold,
    /// for each rank of a worker those of the count that holds the most,
    /// and counts each under its own limit and delay from there. A send
    /// that every standing count has ended is in no copy. A count starts
    /// empty where no count stands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub sent_report_delay_ms: Option<u32>,
}

impl ProfileSpec {
    /// A profile of `pick` alone: what a document that writes its picker and
    /// nothing else gives. It takes the one field a document must write.
    /// The filters and the scorers start empty and the two limits of a send
    /// unset, each as the field's own read leaves it where a document
    /// leaves it out. It checks nothing: a document's validation does. Code
    /// outside this crate starts here and assigns what it states otherwise:
    /// the record gains fields, so no literal of it builds there, and a
    /// pattern of it ends with `..`.
    #[must_use]
    pub const fn of(pick: PickSpec) -> Self {
        Self {
            filter: Vec::new(),
            score: Vec::new(),
            pick,
            sent_ttl_ms: None,
            sent_report_delay_ms: None,
        }
    }
}

/// A closed range with one or both bounds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(rename = "Bounds_for_{T}")]
#[non_exhaustive]
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
#[non_exhaustive]
pub struct HeaderMatch {
    /// The header's name, lowercase, listed in `view.headers`.
    #[schemars(extend("pattern" = HEADER_NAME_PATTERN))]
    pub name: String,
    /// The values that match; empty matches the header's presence.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<String>,
}

/// What a binding matches on. Bindings are ordered; the first match wins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
pub struct Match {
    /// The model id after aliasing: an exact id, `prefix*`, `*suffix` or
    /// `*`.
    #[schemars(extend("pattern" = MODEL_GLOB_PATTERN))]
    pub model: String,
    /// The phases the binding covers; empty covers every phase. A plan with
    /// several sides names every side's phase or none.
    pub phase: Vec<Phase>,
    /// The tenant the request resolved to.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
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
    #[schemars(extend("uniqueItems" = true))]
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
/// program has one. A gateway that runs a supplied document in its own
/// process reads the switch where it builds its workers' default
/// thresholds, and its routes read one shed word, both from that document:
/// it refuses one whose bindings differ on either word. With no document
/// supplied it reads its own two settings. A controller that judges on its
/// worker objects reads the switch there too, at each document that takes
/// force: its default is off where every binding switches it off. It
/// refuses a document whose bindings differ on the switch, as a gateway
/// does.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
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

/// What a document's bindings say of the overload switch, taken together
/// ([`OverloadSpec::enabled`]). A host builds its workers with one default
/// set of thresholds, so it reads the bindings' words as one
/// ([`AppliedSpec::overload_switch`]). A binding with no block says the
/// block's default, which is on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum OverloadSwitch {
    /// Every binding says on, or the document has no binding.
    #[default]
    On,
    /// Every binding says off.
    Off,
    /// The bindings differ. What becomes of the default then is the rule
    /// of the host that reads the word: a controller refuses the document
    /// where it is handed one, by the first binding that differs.
    Differ,
}

impl OverloadSwitch {
    /// The word `bindings` say, taken together.
    pub(super) fn of(bindings: &[BindingSpec]) -> Self {
        match (Self::differs(bindings), bindings.first().map(Self::says)) {
            (Some(_), _) => Self::Differ,
            (None, Some(false)) => Self::Off,
            (None, _) => Self::On,
        }
    }

    /// Whether `binding` says on, by its block or by the block's default.
    fn says(binding: &BindingSpec) -> bool {
        binding.overload.as_ref().is_none_or(|block| block.enabled)
    }

    /// The first of `bindings` that says another word than the first one
    /// does: its place and its word. None where they say one word. The
    /// one reading of the bindings' words: [`OverloadSwitch::Differ`] is
    /// the word of the bindings this names a place in.
    pub(crate) fn differs(bindings: &[BindingSpec]) -> Option<(usize, bool)> {
        let first = Self::says(bindings.first()?);
        let mut words = bindings.iter().map(Self::says).enumerate();
        words.find(|&(_, says)| says != first)
    }
}

/// How a session key seen for the first time picks its worker. Under the
/// `manual` picker the same assignment places a request with no key, which
/// is not pinned, and a `manual` profile bound with no sessions rule places
/// every request by `min_load`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
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
#[non_exhaustive]
pub enum Recovery {
    #[default]
    Original,
    Latest,
}

/// Whether a request a sessions rule placed alone is the profile's own: a
/// pin of the request's key followed, or the rule's own assignment. No
/// stage of the profile placed such a request. A single request is such a
/// request, and so is a side of a pair, each side by its own decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
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
#[non_exhaustive]
pub struct SessionsSpec {
    /// The header whose value is the session key; listed in `view.headers`.
    /// Left out where the binding names its key's sources under
    /// `routing_key`: a binding names them once.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    #[schemars(extend("pattern" = STICKY_HEADER_PATTERN))]
    pub sticky_header: String,
    /// The headers looked up in order for the key, the sticky header among
    /// them; empty means the sticky header alone. Each listed in
    /// `view.headers`. Left out under `routing_key` too.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(inner(regex(pattern = HEADER_NAME_PATTERN)))]
    #[schemars(extend("uniqueItems" = true))]
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
    /// Whether a request this rule placed alone is the profile's own
    /// (default `told`). Under `untold` no stage hears that request's
    /// issue or its dispatch, it is none of the profile's sends, and
    /// neither the engine's record of sent prefixes nor a lent tree
    /// takes its prompt. The session table still hears its dispatch,
    /// and the stages its completion and its release. Under the `pd-pair`
    /// plan each side of a pair follows the word on its own: a side the
    /// rule placed alone is kept from that side's profile, beside a side
    /// the stages placed, which is told. A controller keeps the word on
    /// the lease it files for a single request, and tells such a request
    /// off its file by it all the same. Its file keeps no word of a pair's
    /// side: a controller takes `told` alone under the `pd-pair` plan.
    #[serde(default, skip_serializing_if = "is_default")]
    pub rule_placed: RulePlaced,
    /// The sides of a pair the rule covers: `prefill` or `decode`, or both,
    /// each named once. Left out, the rule covers both sides. Stated under
    /// the `pd-pair` plan alone.
    ///
    /// A side the rule does not name is its profile's alone. The rule reads,
    /// claims and writes no pin of that side's session. A side whose picker
    /// leaves the placement to the rule is not left out. A gateway leaves a
    /// role out of its sticky override where the role's policy keeps a
    /// routing key's handling to itself, as its `manual` and its
    /// `consistent_hashing` do.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(
        length(min = 1, max = 2),
        extend("uniqueItems" = true),
        extend("items" = { "type": "string", "enum": ["prefill", "decode"] })
    )]
    pub sides: Option<Vec<String>>,
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
#[non_exhaustive]
pub struct RoutingKeySpec {
    /// The sources, read in order: one to eight, each named once, with up
    /// to four client sources and one target source beside them.
    #[schemars(length(min = 1, max = 13))]
    #[schemars(extend("uniqueItems" = true))]
    pub sources: Vec<KeySourceSpec>,
    /// Take the turn and retry suffixes off a key, so every turn of a
    /// conversation shares one (`conv_t2_r1` keys as `conv`); default
    /// false.
    #[serde(default, skip_serializing_if = "is_default")]
    pub strip_lineage: bool,
}

impl RoutingKeySpec {
    /// A routing key read from `sources` alone: what a document that writes
    /// its sources and nothing else gives. It takes the one field a document
    /// must write. `strip_lineage` starts false, as the field's own read
    /// leaves it where a document leaves it out. It checks nothing: a
    /// document's validation does. Code outside this crate starts here and
    /// assigns what it states otherwise: the record gains fields, so no
    /// literal of it builds there, and a pattern of it ends with `..`.
    #[must_use]
    pub const fn of(sources: Vec<KeySourceSpec>) -> Self {
        Self {
            sources,
            strip_lineage: false,
        }
    }
}

/// One source of a routing key: exactly one of `header` and `rid`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
pub struct KeySourceSpec {
    /// A request header, lowercase. A credential header is refused, as on
    /// `view.headers`; `cookie` is not. A key from a header `view.headers`
    /// does not list leaves as its fingerprint alone.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(extend("pattern" = HEADER_NAME_PATTERN))]
    pub header: Option<String>,
    /// The request's own id, the `rid` of its body: `true` names it. Under
    /// a ring's prefix form (`prefix_tokens`) its key names the session
    /// alone: a sessions rule pins it, and the ring places the request by
    /// its prompt. The plain ring places by it, and so does a controller's
    /// prefix ring where its host's frame does not say the id handed it.
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
    /// read for a request a program places in process: a single request,
    /// and a pair under the `pd-pair` plan, on each side whose profile
    /// scores by `consistent-hash` with no `prefix_tokens`. A pair's
    /// prefill is counted among the prefills that have a decode to pair
    /// with. Its decode is counted among the partners of its prefill: the
    /// one the request names, or the one the prefill's own profile picks.
    /// A request under the `epd` plan is not read for it. Default false.
    #[serde(skip_serializing_if = "is_default")]
    pub target: bool,
}

/// A second profile deciding a share of the binding's requests, the
/// deciding profile recorded on the lease. Exactly one of `percent`,
/// `tenants` and `header` chooses the share.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct CanarySpec {
    /// The profile deciding the canary's share.
    pub profile: String,
    /// A share of requests, 0 to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1, max = 100))]
    pub percent: Option<u8>,
    /// The tenants whose requests the canary decides.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(extend("uniqueItems" = true))]
    pub tenants: Vec<String>,
    /// A request header that selects the canary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<HeaderMatch>,
}

/// Whether a request may name its profile through the override header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
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
#[non_exhaustive]
pub struct AdmissionSpec {
    pub enabled: bool,
}

/// The plan a binding runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum PlanName {
    /// One profile over one pool.
    #[default]
    Single,
    /// The prefill profile, then the decode profile over the chosen
    /// prefill's partners; both sides named.
    PdPair,
    /// The pair, then the encode profile over the encoders of the chosen
    /// prefill's class, once per encode item or once for the items taken
    /// as one; all three sides named. A request with items that no encoder
    /// is open to is refused ahead of the pair. The pair of a request with
    /// items is picked among the prefills whose class holds an open encoder
    /// that takes its media. A decode without the vision a request asks of
    /// it is no candidate of the pair either.
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
#[non_exhaustive]
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
#[non_exhaustive]
pub struct ExplainSpec {
    /// Each decider of a controller, and each calling thread in a host's
    /// own process, keeps every Nth pick or shed it counts. The count
    /// starts over under a new generation, and in a controller under a new
    /// lead. In a host's own process it also starts over under another
    /// engine, by the document's number. Zero keeps none. In a host's own
    /// process a request is counted only where an id names it.
    #[serde(default)]
    pub sample: u32,
}

/// One declared signal: a named value of a worker's report that the
/// document's stages read beside the rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
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

/// A signal's name as the schema states it: what `validate::check_signals` takes.
const SIGNAL_NAME_PATTERN: &str = "^[a-z][a-z0-9_]{0,62}$";

/// The most bytes a signal's unit takes.
const MAX_SIGNAL_UNIT: usize = 31;

/// A unit's characters as the schema states them: visible ASCII, so no
/// space. One at either end of a unit would not be seen in a document and
/// would match no report. A unit's bytes are then its characters, so the
/// schema's length is the one `validate::check_signals` takes.
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
    #[schemars(extend("const" = API_VERSION))]
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
    #[schemars(extend("propertyNames" = { "pattern": MODEL_GLOB_PATTERN }))]
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
    /// From 1 to 65535 profiles by name, each name of at least one character.
    #[serde(deserialize_with = "keyed_once")]
    #[schemars(extend(
        "minProperties" = 1,
        "maxProperties" = u16::MAX,
        "propertyNames" = { "minLength": 1 }
    ))]
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
    /// A factory's refusal, a stage whose name differs from its
    /// registration, or a host's stage that the document names under a name
    /// the engine refuses under.
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

/// A model glob as the schema states it: the forms `validate::glob_ok` takes.
const MODEL_GLOB_PATTERN: &str = r"^(\*[^*]*|[^*]+\*?)$";

/// A header name as the schema states it: one or more of the characters
/// `validate::check_header_outside` takes. The deny-lists have no form in a
/// pattern: a name on one passes the schema and validation refuses it.
const HEADER_NAME_PATTERN: &str = r"^[a-z0-9!#$%&'*+.^_`|~-]+$";

/// The sticky header as the schema states it: a header name
/// ([`HEADER_NAME_PATTERN`]), or nothing where the binding names its key's
/// sources under `routing_key`.
const STICKY_HEADER_PATTERN: &str = r"^[a-z0-9!#$%&'*+.^_`|~-]*$";

/// A label selector as the schema states it: `key`, or `key=value` split
/// at the first `=`, with no empty side, as `validate::check_view` takes it.
const LABEL_SELECTOR_PATTERN: &str = r"^[^=]+(=[\s\S]+)?$";

/// A label selector's two sides: the key, and the value after the first
/// `=` when it names one.
fn selector(label: &str) -> (&str, Option<&str>) {
    label
        .split_once('=')
        .map_or((label, None), |(key, value)| (key, Some(value)))
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

/// How deep the stages of one generation read a request's block hashes, in
/// the depths' own order: a deeper read holds every shallower one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum HashDepth {
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

/// The two depths of one generation: how deep its stages read a request's
/// block hashes, and how deep a stage reads them itself. The
/// prefix-affinity stage declares the hashes and reads none of them: the
/// lookup that names a request's holders reads them for it. Of the list
/// the stage reads the length alone, where it sizes the request
/// (`Gates::whole`, `RequestFacts::blocks`), and only under a `max` with
/// the count the prompt's size gives. The list the engine's own call makes
/// of a prompt's token ids is never longer than that count: one hash for
/// each whole block, as deep as the document reads at most, one for a
/// shorter prompt, none for an empty one. So the engine sizes such a
/// request the same with its list and with none. Facts a host states
/// itself can hold more hashes than their size gives: those are sized by
/// the list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct HashDepths {
    /// The deepest read under the generation.
    pub(super) read: HashDepth,
    /// The deepest read a stage makes itself.
    pub(super) by_stage: HashDepth,
}

impl HashDepths {
    /// Deepens the generation's read to `other`, and its stages' own where
    /// the stage reads the hashes `itself`.
    fn deepen(&mut self, other: HashDepth, itself: bool) {
        self.read.deepen(other);
        if itself {
            self.by_stage.deepen(other);
        }
    }
}

impl PolicySpec {
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
}

pub(super) fn index_matching<'a>(
    index: &'a BTreeMap<String, IndexSpec>,
    model: &str,
) -> Option<(&'a str, &'a IndexSpec)> {
    index
        .iter()
        .filter(|(pattern, _)| glob_matches(pattern, model))
        .max_by_key(|(pattern, _)| specificity(pattern))
        .map(|(pattern, spec)| (pattern.as_str(), spec))
}

#[cfg(test)]
#[path = "../../tests/unit/document/spec/tests.rs"]
mod tests;
